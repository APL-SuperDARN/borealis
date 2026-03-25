"""
rx_signal_processing
~~~~~~~~~~~~~~~~~~~~~
This process handles the digital signal processing side of Borealis

:copyright: 2020 SuperDARN Canada
:author: Keith Kotyk
"""

import argparse
import math
import mmap
from multiprocessing import shared_memory
import os
import pickle
import sys
import threading
import time
from dataclasses import dataclass

import numpy as np
import posix_ipc as ipc
import zmq
from scipy.constants import speed_of_light

try:
    import cupy as xp

    mempool = xp.get_default_memory_pool()
except ImportError:
    cupy_available = False
    import numpy as xp
else:
    cupy_available = True


sys.path.append(os.environ["BOREALISPATH"])

from utils.message_formats import (
    ProcessedSequenceMessage,
    DebugDataStage,
    OutputDataset,
    SequenceMetadataMessage,
    RxSamplesMetadata,
)
from utils.signals import DSP


@dataclass
class RxProcessingParameters:
    """
    dataclass to hold parameters used in writing data
    """

    sequence_num: int
    main_beam_angles: list
    intf_beam_angles: list
    mixing_freqs: list
    slice_details: list
    start_sample: int
    end_sample: int
    processed_data: list
    intf_antennas: list
    filter_taps: list
    downsample_rates: list
    acf: bool
    xcf: bool
    acfint: bool
    cfs_scan_flag: bool
    samples_needed: int
    rx_rate: float
    output_sample_rate: float
    cfs_fft_n: int


IMAGING_TARGET_FREQ_KHZ = 12000.0
IMAGING_FREQ_TOLERANCE_KHZ = 250.0
IMAGING_RANGE_RES_KM = 1.0
IMAGING_RANGE_WINDOW_KM = 400.0
IMAGING_MIN_SOURCE_RATE_HZ = 150000.0
IMAGING_EL_BINS_DEG = np.arange(0.0, 46.0, 1.0, dtype=np.float32)


def _need_main_beamforming(rx_params, enable_bfiq: bool) -> bool:
    """Return True when main-array beamforming products are required."""

    return bool(rx_params.acf or rx_params.xcf or enable_bfiq)


def _need_intf_beamforming(rx_params, enable_bfiq: bool) -> bool:
    """Return True when interferometer beamforming products are required."""

    return bool(
        len(rx_params.intf_antennas) > 0
        and (rx_params.acfint or rx_params.xcf or enable_bfiq)
    )


def _compute_imaging_products_for_slice(
    main_samples: np.ndarray,
    intf_samples: np.ndarray,
    main_phases: np.ndarray,
    intf_phases: np.ndarray,
    stage_sample_rate: float,
    range_start_km: float,
    rx_freq_khz: float,
    intf_baseline_m: float,
):
    """
    Build coarse az/el/range and high-range-resolution products for one slice.

    :returns: imaging_cube, hires_power, range_start_km, range_sep_km
    :rtype: tuple[np.ndarray, np.ndarray, float, float]
    """

    num_beams = main_phases.shape[0]
    if num_beams == 0:
        return None, None, None, None

    # [num_beams, num_samples]
    main_beamformed = np.einsum(
        "ba,as->bs", main_phases, main_samples, optimize="greedy"
    )

    num_ranges = int(IMAGING_RANGE_WINDOW_KM / IMAGING_RANGE_RES_KM)
    range_bins_km = (
        np.float32(range_start_km)
        + np.arange(num_ranges, dtype=np.float32) * np.float32(IMAGING_RANGE_RES_KM)
    )

    sample_centers = np.round(
        (2.0 * range_bins_km * 1.0e3 / speed_of_light) * stage_sample_rate
    ).astype(np.int32)

    samples_per_bin = max(
        1,
        int(
            round(
                (2.0 * IMAGING_RANGE_RES_KM * 1.0e3 / speed_of_light)
                * stage_sample_rate
            )
        ),
    )
    half_window = samples_per_bin // 2

    hires_power = np.zeros((num_beams, num_ranges), dtype=np.float32)
    imaging_cube = np.zeros(
        (num_beams, IMAGING_EL_BINS_DEG.size, num_ranges), dtype=np.float32
    )

    intf_beamformed = None
    wavelength_m = None
    if intf_samples is not None and intf_baseline_m > 0.1:
        intf_beamformed = np.einsum(
            "ba,as->bs", intf_phases, intf_samples, optimize="greedy"
        )
        wavelength_m = speed_of_light / (rx_freq_khz * 1.0e3)

    num_samps = main_beamformed.shape[-1]
    for r, center in enumerate(sample_centers):
        start = max(0, int(center) - half_window)
        stop = min(num_samps, int(center) + half_window + 1)
        if stop <= start:
            continue

        main_window = main_beamformed[:, start:stop]
        power = np.mean(np.abs(main_window) ** 2, axis=1).astype(np.float32)
        hires_power[:, r] = power

        if intf_beamformed is None:
            imaging_cube[:, 0, r] = power
            continue

        cross = np.mean(
            intf_beamformed[:, start:stop] * np.conjugate(main_window), axis=1
        )
        phase = np.angle(cross)
        path_diff_m = phase * wavelength_m / (2.0 * np.pi)
        sin_el = np.clip(np.abs(path_diff_m) / intf_baseline_m, 0.0, 1.0)
        el_deg = np.degrees(np.arcsin(sin_el))
        el_idx = np.clip(np.rint(el_deg).astype(np.int32), 0, IMAGING_EL_BINS_DEG.size - 1)
        imaging_cube[np.arange(num_beams), el_idx, r] = power

    return (
        imaging_cube.astype(np.float32),
        hires_power.astype(np.float32),
        np.float32(range_start_km),
        np.float32(IMAGING_RANGE_RES_KM),
    )


def fill_datawrite_message(processed_data, slice_details, data_outputs, cfs_scan_flag):
    """
    Fills the datawrite message with processed data.

    :param      processed_data:  The processed data message
    :type       processed_data:  ProcessedSequenceMessage
    :param      slice_details:   The details for each slice that was processed.
    :type       slice_details:   list
    :param      data_outputs:    The processed data outputs.
    :type       data_outputs:    dict
    """
    if cfs_scan_flag:
        processed_data.cfs_freq = data_outputs["cfs_freq"]

    for slice_num, sd in enumerate(slice_details):
        output_dataset = OutputDataset(
            sd["slice_id"], sd["num_beams"], sd["num_range_gates"], sd["num_lags"]
        )

        def add_array(ndarray):
            """
            Creates shared memory and stores ndarray in it.

            :param      ndarray: array to be created
            :type       ndarray: numpy.ndarray
            :returns:   The shared memory name.
            :rtype:     str
            """
            if ndarray.size != 0:
                shm = shared_memory.SharedMemory(create=True, size=ndarray.nbytes)
                shared_array = np.ndarray(
                    ndarray.shape, dtype=ndarray.dtype, buffer=shm.buf
                )
                shared_array[...] = ndarray[...]
                name = shm.name
                # This closes the current SharedMemory instance, but the memory isn't free until data_write unlinks it.
                shm.close()
                return name

        if cfs_scan_flag:
            cfs_data = data_outputs["cfs_data"]
            output_dataset.cfs_data = cfs_data[slice_num]

            processed_data.output_datasets.append(output_dataset)
            # if a clear frequency search was performed, add cfs data to message
        else:
            if "main_corrs" in data_outputs:
                main_corrs = data_outputs["main_corrs"][sd["slice_num"]]
                output_dataset.main_acf_shm = add_array(main_corrs)

            if "intf_corrs" in data_outputs:
                intf_corrs = data_outputs["intf_corrs"][sd["slice_num"]]
                output_dataset.intf_acf_shm = add_array(intf_corrs)

            if "cross_corrs" in data_outputs:
                cross_corrs = data_outputs["cross_corrs"][sd["slice_num"]]
                output_dataset.xcf_shm = add_array(cross_corrs)

            if "img3d_power" in data_outputs:
                img3d_power = data_outputs["img3d_power"][sd["slice_num"]]
                if img3d_power is not None:
                    output_dataset.img3d_power_shm = add_array(img3d_power)
                    output_dataset.img3d_num_az = img3d_power.shape[0]
                    output_dataset.img3d_num_el = img3d_power.shape[1]
                    output_dataset.img3d_num_ranges = img3d_power.shape[2]

            if "hires_range_power" in data_outputs:
                hires_range_power = data_outputs["hires_range_power"][sd["slice_num"]]
                if hires_range_power is not None:
                    output_dataset.hires_range_power_shm = add_array(hires_range_power)
                    output_dataset.hires_num_az = hires_range_power.shape[0]
                    output_dataset.hires_num_ranges = hires_range_power.shape[1]

                    output_dataset.imaging_source_sample_rate = data_outputs[
                        "imaging_source_sample_rate"
                    ][sd["slice_num"]]
                    output_dataset.imaging_range_start_km = data_outputs[
                        "imaging_range_start_km"
                    ][sd["slice_num"]]
                    output_dataset.imaging_range_sep_km = data_outputs[
                        "imaging_range_sep_km"
                    ][sd["slice_num"]]

            processed_data.output_datasets.append(output_dataset)


def sequence_worker(options, ringbuffer):
    inproc_socket = zmq.Context().instance().socket(zmq.PAIR)
    inproc_socket.connect("inproc://sqn_worker")

    while True:
        rx_params = inproc_socket.recv_pyobj()
        # Wait until kwargs received from main thread
        if cupy_available:
            mempool.free_all_blocks()  # Free all unused gpu memory allocations before processing

        seq_begin_iden = options.dspbegin_to_brian_identity + str(
            rx_params.sequence_num
        )
        seq_end_iden = options.dspend_to_brian_identity + str(rx_params.sequence_num)
        if rx_params.cfs_scan_flag:
            sender_iden = options.dsp_cfs_identity
            recipient_iden = options.radctrl_cfs_identity
        else:
            sender_iden = options.dsp_to_dw_identity + str(rx_params.sequence_num)
            recipient_iden = options.dw_to_dsp_identity
        log.debug(
            "socket identities:",
            sender=sender_iden,
            recip=recipient_iden,
            cfs_flag=rx_params.cfs_scan_flag,
        )
        sequence_worker_sockets = so.create_sockets(
            options.router_address,
            seq_begin_iden,
            seq_end_iden,
            sender_iden,
        )

        dspbegin_to_brian = sequence_worker_sockets[0]
        dspend_to_brian = sequence_worker_sockets[1]
        processed_socket = sequence_worker_sockets[2]

        # Generate a timer dict for a uniform log
        log_dict = {"time_unit": "ms"}
        start_timer = time.perf_counter()

        # Copy samples from ring buffer
        indices = np.arange(
            rx_params.start_sample, rx_params.start_sample + rx_params.samples_needed
        )
        # x.take makes a copy of the array. We want to avoid making a copy using Cupy so that
        # data is moved directly from the ring buffer to the GPU. Simple indexing creates a view
        # of existing data without making a copy.
        if cupy_available:
            if rx_params.end_sample > ringbuffer.shape[1]:
                piece1 = ringbuffer[:, rx_params.start_sample :]
                piece2 = ringbuffer[:, : rx_params.end_sample - ringbuffer.shape[1]]
                tmp1 = xp.array(piece1)
                tmp2 = xp.array(piece2)
                sequence_samples = xp.concatenate((tmp1, tmp2), axis=1)
            else:
                sequence_samples = xp.array(
                    ringbuffer[:, rx_params.start_sample : rx_params.end_sample]
                )
        else:
            sequence_samples = ringbuffer.take(indices, axis=1, mode="wrap")
        log_dict["copy_samples_from_ringbuffer_time"] = (
            time.perf_counter() - start_timer
        ) * 1e3

        # Tell brian DSP is about to begin
        mark_timer = time.perf_counter()
        reply_packet = {"sequence_num": rx_params.sequence_num}
        msg = pickle.dumps(reply_packet, protocol=pickle.HIGHEST_PROTOCOL)
        so.recv_bytes(dspbegin_to_brian, options.brian_to_dspbegin_identity, log)
        so.send_bytes(dspbegin_to_brian, options.brian_to_dspbegin_identity, msg)
        log_dict["dsp_begin_msg_time"] = (time.perf_counter() - mark_timer) * 1e3

        if rx_params.cfs_scan_flag:
            # CFS analysis
            mark_timer = time.perf_counter()
            cfs_processor = DSP(
                rx_params.rx_rate,
                rx_params.filter_taps,
                rx_params.mixing_freqs,
                rx_params.downsample_rates,
                use_shared_mem=False,
            )
            cfs_processor.apply_filters(sequence_samples)
            cfs_processor.move_filter_results()
            total_antennas = int(
                len(options.rx_main_antennas) + len(options.rx_intf_antennas)
            )
            total_phases = np.empty(
                (rx_params.main_beam_angles.shape[0], 1, total_antennas),
                dtype=np.complex64,
            )
            for slice_ind in range(rx_params.main_beam_angles.shape[0]):
                total_phases[slice_ind, 0, :] = np.append(
                    rx_params.main_beam_angles[slice_ind],
                    rx_params.intf_beam_angles[slice_ind],
                )
            cfs_processor.beamform(total_phases)
            cfs_data, cfs_freq = cfs_processor.cfs_freq_analysis(
                rx_params.slice_details[0], rx_params.cfs_fft_n
            )

            del cfs_processor

            log_dict["cfs_dsp_time"] = (time.perf_counter() - mark_timer) * 1e3

        else:
            need_main_beamforming = _need_main_beamforming(
                rx_params, options.enable_bfiq
            )
            need_intf_beamforming = _need_intf_beamforming(
                rx_params, options.enable_bfiq
            )

            # Process main samples
            mark_timer = time.perf_counter()
            main_sequence_samples = sequence_samples[: len(options.rx_main_antennas), :]
            main_sequence_samples_shape = main_sequence_samples.shape
            main_processor = DSP(
                rx_params.rx_rate,
                rx_params.filter_taps,
                rx_params.mixing_freqs,
                rx_params.downsample_rates,
            )
            main_processor.apply_filters(main_sequence_samples)
            main_processor.move_filter_results()
            if need_main_beamforming:
                main_processor.beamform(rx_params.main_beam_angles)
            if rx_params.acf:
                for slice_info in rx_params.slice_details:
                    slice_info["skip"] = not slice_info["acf"]
                main_corrs = DSP.correlations_from_samples(
                    main_processor.beamformed_samples,
                    main_processor.beamformed_samples,
                    rx_params.output_sample_rate,
                    rx_params.slice_details,
                )
            log_dict["main_dsp_time"] = (time.perf_counter() - mark_timer) * 1e3

            # Process intf samples if intf exists
            mark_timer = time.perf_counter()
            intf_sequence_samples_shape = None
            log_dict["intf antennas"] = rx_params.intf_antennas
            if len(rx_params.intf_antennas) > 0:
                intf_sequence_samples = sequence_samples[
                    len(options.rx_main_antennas) :, :
                ]
                intf_sequence_samples_shape = intf_sequence_samples.shape
                intf_processor = DSP(
                    rx_params.rx_rate,
                    rx_params.filter_taps,
                    rx_params.mixing_freqs,
                    rx_params.downsample_rates,
                )
                intf_processor.apply_filters(intf_sequence_samples)
                intf_processor.move_filter_results()
                if need_intf_beamforming:
                    intf_processor.beamform(rx_params.intf_beam_angles)
                if rx_params.acfint:
                    for slice_info in rx_params.slice_details:
                        slice_info["skip"] = not slice_info["acfint"]
                    intf_corrs = DSP.correlations_from_samples(
                        intf_processor.beamformed_samples,
                        intf_processor.beamformed_samples,
                        rx_params.output_sample_rate,
                        rx_params.slice_details,
                    )
                if rx_params.xcf:
                    for slice_info in rx_params.slice_details:
                        slice_info["skip"] = not slice_info["xcf"]
                    cross_corrs = DSP.correlations_from_samples(
                        intf_processor.beamformed_samples,
                        main_processor.beamformed_samples,
                        rx_params.output_sample_rate,
                        rx_params.slice_details,
                    )
            log_dict["intf_dsp_time"] = (time.perf_counter() - mark_timer) * 1e3

        # Tell brian DSP how long it took
        mark_timer = time.perf_counter()
        if rx_params.cfs_scan_flag:
            reply_packet["kerneltime"] = log_dict["cfs_dsp_time"]
        else:
            reply_packet["kerneltime"] = (
                log_dict["main_dsp_time"] + log_dict["intf_dsp_time"]
            )
        msg = pickle.dumps(reply_packet, protocol=pickle.HIGHEST_PROTOCOL)
        so.recv_bytes(dspend_to_brian, options.brian_to_dspend_identity, log)
        so.send_bytes(dspend_to_brian, options.brian_to_dspend_identity, msg)
        log_dict["dsp_end_msg_time"] = (time.perf_counter() - mark_timer) * 1e3

        total_processing_time = (time.perf_counter() - start_timer) * 1e3
        log_dict["total_sequence_process_time"] = total_processing_time
        if rx_params.cfs_scan_flag:
            log.verbose(
                "CFS processing sequence",
                **log_dict,
            )
        else:
            log.verbose(
                "processing sequence",
                sequence_num=rx_params.sequence_num,
                mixing_freqs=rx_params.mixing_freqs,
                mixing_freqs_units="Hz",
                main_beam_angles=rx_params.main_beam_angles.shape,
                intf_beam_angles=rx_params.main_beam_angles.shape,
                main_buffer_shape=main_sequence_samples_shape,
                intf_buffer_shape=intf_sequence_samples_shape,
                **log_dict,
            )

        # Generate a new timer dict for a uniform log
        log_dict = {"time_unit": "ms"}
        start_timer = time.perf_counter()

        # Extract outputs from processing into groups that will be put into message fields.
        data_outputs = {}

        if rx_params.cfs_scan_flag:
            mark_timer = time.perf_counter()
            data_outputs["cfs_data"] = cfs_data
            data_outputs["cfs_freq"] = cfs_freq

            # Fill message with the slice-specific fields
            fill_datawrite_message(
                rx_params.processed_data,
                rx_params.slice_details,
                data_outputs,
                rx_params.cfs_scan_flag,
            )
            log_dict["cfs_to_stage_time"] = (time.perf_counter() - mark_timer) * 1e3

        else:

            def debug_data_in_shm(holder, data_array, array_name):
                """
                Adds an array of antennas data (filter outputs or antennas_iq) into a dictionary
                for later entry in a processed data message.

                :param  holder:     Dictionary to store the shared memory parameters.
                :type   holder:     dict
                :param  data_array: array to hold the data
                :type   data_array: cp.ndarray or np.ndarray
                :param  array_name: 'main' or 'intf'
                :type   array_name: str
                """

                shm = shared_memory.SharedMemory(create=True, size=data_array.nbytes)
                data = np.ndarray(data_array.shape, dtype=np.complex64, buffer=shm.buf)
                if cupy_available:
                    data[...] = xp.asnumpy(data_array)
                else:
                    data[...] = data_array

                if array_name == "main":
                    holder.main_shm = shm.name
                elif array_name == "intf":
                    holder.intf_shm = shm.name
                else:
                    log.error(f"unknown array name {array_name} not in [main, intf]")
                    log.exception(f"unknown array name {array_name} [main, intf]")
                    sys.exit(1)

                holder.num_samps = data_array.shape[-1]
                shm.close()

            # Add the filter stage data if in debug mode
            if __debug__:
                for i, main_data in enumerate(main_processor.filter_outputs[:-1]):
                    stage = DebugDataStage(f"stage_{i}")
                    debug_data_in_shm(stage, main_data, "main")

                    if len(rx_params.intf_antennas) > 0:
                        intf_data = intf_processor.filter_outputs[i]
                        debug_data_in_shm(stage, intf_data, "intf")

                    rx_params.processed_data.debug_data.append(stage)

            # Add antennas_iq data
            stage = DebugDataStage()
            stage.stage_name = "antennas"
            main_shm = main_processor.shared_mem["antennas_iq"]
            stage.main_shm = main_shm.name
            stage.num_samps = main_processor.antennas_iq_samples.shape[-1]
            main_shm.close()
            if len(rx_params.intf_antennas) > 0:
                intf_shm = intf_processor.shared_mem["antennas_iq"]
                stage.intf_shm = intf_shm.name
                intf_shm.close()
            rx_params.processed_data.debug_data.append(stage)
            log_dict["add_antiq_to_stage_time"] = (
                time.perf_counter() - start_timer
            ) * 1e3
            mark_timer = time.perf_counter()

            # Add rawrf data
            if __debug__:
                # np.complex64 in bytes * num_antennas * num_samps
                rawrf_size = (
                    np.dtype(np.complex64).itemsize
                    * ringbuffer.shape[0]
                    * indices.shape[-1]
                )
                rawrf_shm = shared_memory.SharedMemory(create=True, size=rawrf_size)
                rawrf_array = np.ndarray(
                    (ringbuffer.shape[0], indices.shape[-1]),
                    dtype=np.complex64,
                    buffer=rawrf_shm.buf,
                )
                rawrf_array[...] = ringbuffer.take(indices, axis=1, mode="wrap")
                rx_params.processed_data.rawrf_shm = rawrf_shm.name
                rx_params.processed_data.rawrf_num_samps = indices.shape[-1]
                rawrf_shm.close()
                log_dict["put_rawrf_in_shm_time"] = (
                    time.perf_counter() - mark_timer
                ) * 1e3

            # Add bfiq and correlations data
            mark_timer = time.perf_counter()
            if rx_params.acf:
                data_outputs["main_corrs"] = main_corrs

            if options.enable_bfiq:
                beamformed_m = main_processor.beamformed_samples
                rx_params.processed_data.bfiq_main_shm = main_processor.shared_mem[
                    "bfiq"
                ].name
                rx_params.processed_data.max_num_beams = beamformed_m.shape[1]
                rx_params.processed_data.num_samps = beamformed_m.shape[-1]
                main_processor.shared_mem["bfiq"].close()

            if len(rx_params.intf_antennas) > 0:
                if rx_params.xcf:
                    data_outputs["cross_corrs"] = cross_corrs
                if rx_params.acfint:
                    data_outputs["intf_corrs"] = intf_corrs
                if options.enable_bfiq:
                    rx_params.processed_data.bfiq_intf_shm = intf_processor.shared_mem[
                        "bfiq"
                    ].name
                    intf_processor.shared_mem["bfiq"].close()

            has_target_freq = any(
                abs(sd["rx_freq_khz"] - IMAGING_TARGET_FREQ_KHZ)
                <= IMAGING_FREQ_TOLERANCE_KHZ
                for sd in rx_params.slice_details
            )
            stage0_sample_rate = float(
                rx_params.rx_rate / float(rx_params.downsample_rates[0])
            )
            if has_target_freq and stage0_sample_rate >= IMAGING_MIN_SOURCE_RATE_HZ:
                if cupy_available:
                    main_stage0 = xp.asnumpy(main_processor.filter_outputs[0])
                else:
                    main_stage0 = main_processor.filter_outputs[0]

                intf_stage0 = None
                if len(rx_params.intf_antennas) > 0:
                    if cupy_available:
                        intf_stage0 = xp.asnumpy(intf_processor.filter_outputs[0])
                    else:
                        intf_stage0 = intf_processor.filter_outputs[0]

                intf_baseline_m = 0.0
                if len(rx_params.intf_antennas) > 0:
                    main_centroid = np.mean(
                        options.main_antenna_locations[options.rx_main_antennas],
                        axis=0,
                    )
                    intf_indices = np.array(
                        sorted(rx_params.intf_antennas), dtype=np.int32
                    )
                    intf_centroid = np.mean(
                        options.intf_antenna_locations[intf_indices], axis=0
                    )
                    intf_baseline_m = float(np.linalg.norm(intf_centroid - main_centroid))

                img3d_power = [None] * len(rx_params.slice_details)
                hires_range_power = [None] * len(rx_params.slice_details)
                imaging_range_start_km = [None] * len(rx_params.slice_details)
                imaging_range_sep_km = [None] * len(rx_params.slice_details)
                imaging_source_sample_rate = [
                    np.float32(stage0_sample_rate)
                ] * len(rx_params.slice_details)

                for slc, slice_info in enumerate(rx_params.slice_details):
                    if (
                        abs(slice_info["rx_freq_khz"] - IMAGING_TARGET_FREQ_KHZ)
                        > IMAGING_FREQ_TOLERANCE_KHZ
                    ):
                        continue

                    num_beams = int(slice_info["num_beams"])
                    main_phases = rx_params.main_beam_angles[slc, :num_beams, :]
                    intf_phases = rx_params.intf_beam_angles[slc, :num_beams, :]
                    intf_samples = intf_stage0[slc] if intf_stage0 is not None else None

                    products = _compute_imaging_products_for_slice(
                        main_stage0[slc],
                        intf_samples,
                        main_phases,
                        intf_phases,
                        stage0_sample_rate,
                        float(slice_info["first_range"]),
                        float(slice_info["rx_freq_khz"]),
                        intf_baseline_m,
                    )

                    img3d_power[slc] = products[0]
                    hires_range_power[slc] = products[1]
                    imaging_range_start_km[slc] = products[2]
                    imaging_range_sep_km[slc] = products[3]

                if any(x is not None for x in img3d_power):
                    data_outputs["img3d_power"] = img3d_power
                    data_outputs["hires_range_power"] = hires_range_power
                    data_outputs["imaging_source_sample_rate"] = (
                        imaging_source_sample_rate
                    )
                    data_outputs["imaging_range_start_km"] = imaging_range_start_km
                    data_outputs["imaging_range_sep_km"] = imaging_range_sep_km

            # Fill message with the slice-specific fields
            fill_datawrite_message(
                rx_params.processed_data,
                rx_params.slice_details,
                data_outputs,
                rx_params.cfs_scan_flag,
            )

            del main_processor
            if len(rx_params.intf_antennas) > 0:
                del intf_processor

            log_dict["add_bfiq_and_acfs_to_stage_time"] = (
                time.perf_counter() - mark_timer
            ) * 1e3

        log.debug(
            "Sending processed data",
            recieve=recipient_iden,
            sender=processed_socket.get(zmq.IDENTITY),
        )
        so.send_pyobj(
            processed_socket, recipient_iden, rx_params.processed_data, log=log
        )

        log_dict["total_serialize_send_time"] = (
            time.perf_counter() - start_timer
        ) * 1e3
        log.info(
            "done with sequence",
            sequence_num=rx_params.sequence_num,
            processing_time=total_processing_time,
            time_unit="ms",
            slice_ids=[d["slice_id"] for d in rx_params.slice_details],
        )
        log.verbose("sequence timing", sequence_num=rx_params.sequence_num, **log_dict)


def main(enable_bfiq: bool = False):
    options = Options()
    options.enable_bfiq = enable_bfiq

    sockets = so.create_sockets(
        options.router_address,
        options.dsp_to_radctrl_identity,
        options.dsp_to_driver_identity,
    )

    dsp_to_radar_control = sockets[0]
    dsp_to_driver = sockets[1]

    sequence_worker_socket = zmq.Context().instance().socket(zmq.PAIR)
    sequence_worker_socket.bind("inproc://sqn_worker")

    ringbuffer = None

    total_antennas = len(options.rx_main_antennas) + len(options.rx_intf_antennas)

    first_time = True
    while True:
        sqn_meta_message = so.recv_pyobj(
            dsp_to_radar_control,
            options.radctrl_to_dsp_identity,
            log,
            expected_type=SequenceMetadataMessage,
        )

        log.debug("Sending ACK to radctrl")
        so.send_string(
            dsp_to_radar_control, options.radctrl_to_dsp_identity, "Received metadata"
        )
        log.debug("ACK sent")
        # Let radar_control know that the metadata was received

        rx_rate = np.float64(sqn_meta_message.rx_rate)
        output_sample_rate = np.float64(sqn_meta_message.output_sample_rate)
        first_rx_sample_off = sqn_meta_message.offset_to_first_rx_sample
        rx_center_freq = sqn_meta_message.rx_ctr_freq
        cfs_scan_flag = sqn_meta_message.cfs_scan_flag
        cfs_fft_n = sqn_meta_message.cfs_fft_n

        processed_data = ProcessedSequenceMessage()

        processed_data.sequence_num = sqn_meta_message.sequence_num
        processed_data.rx_sample_rate = rx_rate
        processed_data.output_sample_rate = output_sample_rate

        mixing_freqs = []
        main_beam_angles = []
        intf_beam_angles = []
        intf_antennas = set()

        # Parse out details and force the data type so that Cupy can optimize with standardized
        # data types.
        slice_details = []
        for i, chan in enumerate(sqn_meta_message.rx_channels):
            detail = {}

            mixing_freqs.append(chan.rx_freq - rx_center_freq)

            detail["slice_id"] = chan.slice_id
            detail["slice_num"] = i
            detail["first_range"] = np.float32(chan.first_range)
            detail["range_sep"] = np.float32(chan.range_sep)
            detail["tau_spacing"] = np.uint32(chan.tau_spacing)
            detail["num_range_gates"] = np.uint32(chan.num_ranges)
            detail["first_range_off"] = np.uint32(
                round(chan.first_range / chan.range_sep)
            )
            detail["acf"] = chan.acf
            detail["xcf"] = chan.xcf
            detail["acfint"] = chan.acfint

            lag_phase_offsets = []

            lags = []
            for lag in chan.lags:
                lags.append([lag.pulse_1, lag.pulse_2])
                lag_phase_offsets.append(
                    lag.phase_offset_real + 1j * lag.phase_offset_imag
                )

            detail["lag_phase_offsets"] = np.array(
                lag_phase_offsets, dtype=np.complex64
            )

            detail["lags"] = np.array(lags, dtype=np.uint32)
            detail["num_lags"] = len(lags)

            main_beams = chan.beam_phases[:, : len(options.rx_main_antennas)]
            intf_beams = chan.beam_phases[:, len(options.rx_main_antennas) :]

            detail["num_beams"] = main_beams.shape[0]
            detail["rx_freq_khz"] = np.float32(chan.rx_freq)
            detail["pulses"] = chan.pulses

            slice_details.append(detail)
            main_beam_angles.append(main_beams)
            intf_beam_angles.append(intf_beams)

            intf_antennas.update(
                set(chan.rx_intf_antennas)
            )  # Keep track of all intf antennas for the sequence

        # Different slices can have a different amount of beams used. Slices that use fewer beams
        # than the max number of beams are padded with zeros so that matrix calculations can be
        # used. The extra beams that are processed will be not be parsed for data writing.
        max_num_beams = max([x.shape[0] for x in main_beam_angles])
        padded_main_phases = np.zeros(
            (
                len(sqn_meta_message.rx_channels),
                max_num_beams,
                len(options.rx_main_antennas),
            ),
            dtype=np.complex64,
        )
        padded_intf_phases = np.zeros(
            (
                len(sqn_meta_message.rx_channels),
                max_num_beams,
                len(options.rx_intf_antennas),
            ),
            dtype=np.complex64,
        )

        for i, x in enumerate(main_beam_angles):
            padded_main_phases[i, : len(x)] = x
        for i, x in enumerate(intf_beam_angles):
            padded_intf_phases[i, : len(x)] = x

        mixing_freqs = np.array(mixing_freqs, dtype=np.float64)

        # Get meta from driver
        message = "Need data to process"
        so.send_string(dsp_to_driver, options.driver_to_dsp_identity, message)
        log.debug("Requested driver for data")
        reply = so.recv_bytes(dsp_to_driver, options.driver_to_dsp_identity, log)
        log.debug("Received data from driver")

        # parse the message from usrp_driver into a dataclass
        rx_metadata = RxSamplesMetadata.parse(reply.decode("utf-8"))

        if sqn_meta_message.sequence_num != rx_metadata.sequence_num:
            log.error(
                "driver packets != radctrl packets",
                sqn_meta_sqn_num=sqn_meta_message.sequence_num,
                rx_meta_sqn_num=rx_metadata.sequence_num,
            )
            sys.exit(-1)

        # First time configuration
        if first_time:
            shm = ipc.SharedMemory(options.ringbuffer_name)
            mapped_mem = mmap.mmap(shm.fd, shm.size)
            ringbuffer = np.frombuffer(mapped_mem, dtype=np.complex64).reshape(
                total_antennas, -1
            )

            if cupy_available:
                xp.cuda.runtime.hostRegister(ringbuffer.ctypes.data, ringbuffer.size, 0)
                xp.cuda.runtime.setDevice(0)

            seq_thread = threading.Thread(
                target=sequence_worker, args=(options, ringbuffer)
            )
            seq_thread.daemon = True
            seq_thread.start()

            first_time = False

        # Set up the filtering/downsampling strategy
        taps_per_stage = []
        dm_scheme_taps = []
        dm_rates = sqn_meta_message.decimation_scheme.dm_rates
        extra_samples = 0
        for stage in sqn_meta_message.decimation_scheme.stages:
            dm_scheme_taps.append(np.array(stage.filter_taps, dtype=np.complex64))
            taps_per_stage.append(len(stage.filter_taps))
        log.verbose(
            "stage decimation and filter taps",
            decimation_rates=dm_rates,
            filter_taps_per_stage=taps_per_stage,
        )
        dm_rates = np.array(dm_rates, dtype=np.uint32)
        for dm, taps in zip(reversed(dm_rates), reversed(dm_scheme_taps)):
            extra_samples = (extra_samples * dm) + len(taps) // 2
        total_dm_rate = np.prod(dm_rates)

        # Calculate where in the ringbuffer the samples are located.
        samples_needed = rx_metadata.num_rx_samps + 2 * extra_samples
        samples_needed = int(
            math.ceil(float(samples_needed) / float(total_dm_rate)) * total_dm_rate
        )

        sample_time_diff = (
            rx_metadata.sequence_start_time - rx_metadata.initialization_time
        )
        sample_in_time = (
            (sample_time_diff * rx_rate) + first_rx_sample_off - extra_samples
        )
        start_sample = int(math.fmod(sample_in_time, ringbuffer.shape[1]))
        end_sample = start_sample + samples_needed

        processed_data.initialization_time = rx_metadata.initialization_time
        processed_data.sequence_start_time = rx_metadata.sequence_start_time
        processed_data.gps_to_system_time_diff = rx_metadata.gps_to_system_time_diff
        processed_data.agc_status_bank_h = rx_metadata.agc_status_bank_h
        processed_data.lp_status_bank_h = rx_metadata.lp_status_bank_h
        processed_data.agc_status_bank_l = rx_metadata.agc_status_bank_l
        processed_data.lp_status_bank_l = rx_metadata.lp_status_bank_l
        processed_data.gps_locked = rx_metadata.gps_locked

        rx_params = RxProcessingParameters(
            sqn_meta_message.sequence_num,
            padded_main_phases,
            padded_intf_phases,
            mixing_freqs,
            slice_details,
            start_sample,
            end_sample,
            processed_data,
            intf_antennas,
            dm_scheme_taps,
            dm_rates,
            sqn_meta_message.acf,
            sqn_meta_message.xcf,
            sqn_meta_message.acfint,
            cfs_scan_flag,
            samples_needed,
            rx_rate,
            output_sample_rate,
            cfs_fft_n,
        )

        sequence_worker_socket.send_pyobj(rx_params)


def rx_signal_processing_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--enable-bfiq",
        help="Generate beamformed IQ products for downstream consumers.",
        action="store_true",
    )
    return parser


if __name__ == "__main__":
    from utils.options import Options
    from utils import socket_operations as so
    from utils import log_config

    args = rx_signal_processing_parser().parse_args()
    log = log_config.log()
    log.info("RX_SIGNAL_PROCESSING BOOTED")
    if not cupy_available:
        log.warning("cupy not installed")
    try:
        main(enable_bfiq=args.enable_bfiq)
        log.info("RX_SIGNAL_PROCESSING EXITED")
    except Exception as main_exception:
        log.critical("RX_SIGNAL_PROCESSING CRASHED", error=main_exception)
        log.exception("RX_SIGNAL_PROCESSING CRASHED", exception=main_exception)
