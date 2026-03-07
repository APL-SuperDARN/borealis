#!/usr/bin/env python3
"""
Standalone A/B imaging processor for SuperDARN antennas_iq files.

Produces conventional beamforming outputs alongside Capon and/or MUSIC spectra
for direct comparison. This runs outside realtime Borealis and can be used for
rapid experimentation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import h5py
import numpy as np
from scipy.constants import speed_of_light


def _steering_vector(az_deg: float, antenna_x_m: np.ndarray, freq_khz: float) -> np.ndarray:
    k = 2.0 * np.pi * (freq_khz * 1.0e3) / speed_of_light
    return np.exp(1j * k * (-np.sin(np.deg2rad(az_deg))) * antenna_x_m).astype(np.complex64)


def _parse_records(all_records: list[str], record: str | None, records: int) -> list[str]:
    if record:
        if record not in all_records:
            raise ValueError(f"Record {record} not found in file")
        return [record]

    count = min(max(records, 1), len(all_records))
    return all_records[-count:]


def _load_calibration(path: str | None) -> tuple[np.ndarray, np.ndarray] | None:
    if path is None:
        return None
    cal = np.load(path)
    ant_ids = cal["antenna_ids"].astype(np.int32)
    gains = cal["channel_gains"].astype(np.complex64)
    return ant_ids, gains


def _range_grid(start_km: float, stop_km: float, step_km: float) -> np.ndarray:
    n = int(np.floor((stop_km - start_km) / step_km)) + 1
    if n <= 0:
        raise ValueError("Invalid range grid")
    return (start_km + np.arange(n, dtype=np.float32) * step_km).astype(np.float32)


def _nearest_sample_indices(sample_time_us: np.ndarray, ranges_km: np.ndarray) -> np.ndarray:
    two_way_us = (2.0 * ranges_km * 1.0e3 / speed_of_light) * 1.0e6
    idx = np.array([np.argmin(np.abs(sample_time_us - t)) for t in two_way_us], dtype=np.int32)
    return idx


def _extract_main_array_data(record_group: h5py.Group) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ant_iq = record_group["antennas_iq_data"][...].astype(np.complex64)  # [ant, seq, time]
    rx_antennas = record_group["rx_antennas"][...].astype(np.int32)
    rx_main_antennas = record_group["rx_main_antennas"][...].astype(np.int32)
    ant_locs = record_group["antenna_locations"][...].astype(np.float32)

    main_mask = np.isin(rx_antennas, rx_main_antennas)
    if not np.any(main_mask):
        raise RuntimeError("No main-array channels found in record")

    main_data = ant_iq[main_mask, :, :]
    main_ids = rx_antennas[main_mask]
    main_x = ant_locs[main_ids, 0]

    order = np.argsort(main_ids)
    return main_data[order], main_ids[order], main_x[order], rx_main_antennas


def _apply_calibration(
    main_data: np.ndarray,
    main_ids: np.ndarray,
    calibration: tuple[np.ndarray, np.ndarray] | None,
) -> np.ndarray:
    if calibration is None:
        return main_data

    cal_ids, cal_gains = calibration
    gain_map = {int(a): g for a, g in zip(cal_ids.tolist(), cal_gains.tolist())}

    cal_vec = np.ones(main_ids.shape[0], dtype=np.complex64)
    for i, ant in enumerate(main_ids.tolist()):
        if ant in gain_map:
            cal_vec[i] = np.complex64(gain_map[ant])

    return main_data / cal_vec[:, np.newaxis, np.newaxis]


def _covariance_from_snapshots(x: np.ndarray, diag_loading: float) -> np.ndarray:
    # x: [channels, snapshots]
    if x.shape[1] < x.shape[0]:
        # Allow small-snapshot cases, but still produce a robust loaded covariance.
        pass

    r = (x @ x.conj().T) / np.float32(max(x.shape[1], 1))
    load = diag_loading * np.trace(r).real / np.float32(r.shape[0])
    r_loaded = r + np.eye(r.shape[0], dtype=np.complex64) * np.float32(load)
    return r_loaded


def _capon_spectrum(r_loaded: np.ndarray, steering: np.ndarray) -> np.ndarray:
    # steering: [az, channels]
    inv_r = np.linalg.pinv(r_loaded)
    out = np.empty(steering.shape[0], dtype=np.float32)
    for i, a in enumerate(steering):
        denom = np.real(np.conj(a) @ (inv_r @ a))
        out[i] = np.float32(1.0 / max(denom, 1.0e-12))
    return out


def _music_spectrum(r_loaded: np.ndarray, steering: np.ndarray, model_order: int) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(r_loaded)
    n_chan = r_loaded.shape[0]
    k = min(max(model_order, 1), n_chan - 1)
    en = eigvecs[:, : n_chan - k]  # noise subspace (smallest eigenvalues)

    out = np.empty(steering.shape[0], dtype=np.float32)
    for i, a in enumerate(steering):
        denom = np.linalg.norm(en.conj().T @ a) ** 2
        out[i] = np.float32(1.0 / max(denom.real, 1.0e-12))
    return out


def _db(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (10.0 * np.log10(np.maximum(x, np.float32(1.0e-12)))).astype(np.float32)


def run(args: argparse.Namespace) -> Path:
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        output_path = input_path.with_suffix("").with_suffix("")
        output_path = output_path.with_name(output_path.name + ".imaging_ab.h5")

    az_grid = np.arange(args.az_min, args.az_max + 0.5 * args.az_step, args.az_step, dtype=np.float32)
    if az_grid.size < 2:
        raise ValueError("Azimuth grid needs at least 2 bins")

    ranges_km = _range_grid(args.range_start_km, args.range_stop_km, args.range_step_km)
    calibration = _load_calibration(args.calibration)

    with h5py.File(input_path, "r") as src, h5py.File(output_path, "w") as dst:
        all_records = sorted([k for k in src.keys() if k != "metadata"])
        use_records = _parse_records(all_records, args.record, args.records)

        meta = dst.create_group("metadata")
        meta.attrs["source_file"] = str(input_path)
        meta.attrs["method"] = args.method
        meta.attrs["diag_loading"] = args.diag_loading
        meta.attrs["model_order"] = args.model_order
        meta.attrs["window_samples"] = args.window_samples
        meta.attrs["notes"] = "Conventional steering scan + Capon/MUSIC A/B outputs"
        if calibration is not None:
            meta.attrs["calibration_file"] = str(Path(args.calibration).expanduser().resolve())

        for rec_name in use_records:
            rec_src = src[rec_name]
            rec_dst = dst.create_group(rec_name)

            main_data, main_ids, main_x, _ = _extract_main_array_data(rec_src)
            main_data = _apply_calibration(main_data, main_ids, calibration)

            sample_time_us = rec_src["sample_time"][...].astype(np.float32)
            freq_khz = float(rec_src["freq"][()])
            sample_idx = _nearest_sample_indices(sample_time_us, ranges_km)

            steering = np.array(
                [_steering_vector(float(az), main_x, freq_khz) for az in az_grid],
                dtype=np.complex64,
            )

            n_az = az_grid.size
            n_rg = ranges_km.size
            capon = np.zeros((n_az, n_rg), dtype=np.float32)
            music = np.zeros((n_az, n_rg), dtype=np.float32)
            bf_scan = np.zeros((n_az, n_rg), dtype=np.float32)

            sched_weights = None
            sched_az = None
            if "rx_main_excitations" in rec_src:
                sched_weights = rec_src["rx_main_excitations"][...].astype(np.complex64)
                if "beam_azms" in rec_src:
                    sched_az = rec_src["beam_azms"][...].astype(np.float32)
                sched_power = np.zeros((sched_weights.shape[0], n_rg), dtype=np.float32)
            else:
                sched_power = np.zeros((0, n_rg), dtype=np.float32)

            half = max(args.window_samples // 2, 0)
            n_time = main_data.shape[-1]

            for ri, center in enumerate(sample_idx.tolist()):
                lo = max(0, center - half)
                hi = min(n_time, center + half + 1)
                if hi <= lo:
                    continue

                x = main_data[:, :, lo:hi].reshape(main_data.shape[0], -1)
                if x.shape[1] == 0:
                    continue

                r_loaded = _covariance_from_snapshots(x, args.diag_loading)

                for ai in range(n_az):
                    a = steering[ai]
                    y = (np.conj(a) @ x) / np.float32(main_data.shape[0])
                    bf_scan[ai, ri] = np.float32(np.mean(np.abs(y) ** 2))

                if args.method in ("capon", "both"):
                    capon[:, ri] = _capon_spectrum(r_loaded, steering)
                if args.method in ("music", "both"):
                    music[:, ri] = _music_spectrum(r_loaded, steering, args.model_order)

                if sched_weights is not None:
                    for bi in range(sched_weights.shape[0]):
                        w = sched_weights[bi]
                        yb = w @ x
                        sched_power[bi, ri] = np.float32(np.mean(np.abs(yb) ** 2))

            rec_dst.create_dataset("azimuth_grid_deg", data=az_grid)
            rec_dst.create_dataset("range_grid_km", data=ranges_km)
            rec_dst.create_dataset("main_array_antennas", data=main_ids.astype(np.int32))
            rec_dst.create_dataset("main_array_x_m", data=main_x.astype(np.float32))
            rec_dst.create_dataset("conventional_scan_power", data=bf_scan)
            rec_dst.create_dataset("conventional_scan_power_db", data=_db(bf_scan))
            rec_dst.create_dataset("scheduled_beam_power", data=sched_power)
            rec_dst.create_dataset("scheduled_beam_power_db", data=_db(sched_power))
            if sched_az is not None:
                rec_dst.create_dataset("scheduled_beam_azimuth_deg", data=sched_az)

            if args.method in ("capon", "both"):
                rec_dst.create_dataset("capon_power", data=capon)
                rec_dst.create_dataset("capon_power_db", data=_db(capon))
                rec_dst.create_dataset("capon_peak_azimuth_deg", data=az_grid[np.argmax(capon, axis=0)])

            if args.method in ("music", "both"):
                rec_dst.create_dataset("music_power", data=music)
                rec_dst.create_dataset("music_power_db", data=_db(music))
                rec_dst.create_dataset("music_peak_azimuth_deg", data=az_grid[np.argmax(music, axis=0)])

            rec_dst.attrs["source_record"] = rec_name
            rec_dst.attrs["freq_khz"] = freq_khz
            rec_dst.attrs["num_snapshots_per_bin"] = int(main_data.shape[1] * max(1, args.window_samples))

    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="A/B conventional beamforming vs Capon/MUSIC on antennas_iq data")
    p.add_argument("--input", required=True, help="Path to antennas_iq HDF5 file")
    p.add_argument("--output", help="Output HDF5 path (default: <input>.imaging_ab.h5)")
    p.add_argument("--record", help="Specific record group name to process")
    p.add_argument("--records", type=int, default=1, help="Process latest N records if --record is not given")
    p.add_argument("--method", choices=["capon", "music", "both"], default="both")
    p.add_argument("--az-min", type=float, default=-30.0)
    p.add_argument("--az-max", type=float, default=30.0)
    p.add_argument("--az-step", type=float, default=0.25)
    p.add_argument("--range-start-km", type=float, default=180.0)
    p.add_argument("--range-stop-km", type=float, default=580.0)
    p.add_argument("--range-step-km", type=float, default=1.0)
    p.add_argument("--window-samples", type=int, default=3, help="Time-window around each range-bin sample")
    p.add_argument("--diag-loading", type=float, default=1.0e-2)
    p.add_argument("--model-order", type=int, default=1, help="Signal subspace order for MUSIC")
    p.add_argument("--calibration", help="Optional .npz from self_calibrate_array.py")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    out = run(args)
    print(json.dumps({"status": "ok", "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
