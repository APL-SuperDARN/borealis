#!/usr/bin/env python3
"""
Shared helpers for Wallops MCCM qualification and solve scripts.
"""

from __future__ import annotations

import csv
import json
import math
import os
import shlex
from pathlib import Path
from typing import Any

import h5py
import numpy as np

DEFAULT_SCALE_LIST = [0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1]
DEFAULT_PNR_THRESHOLD_DB = 10.0
DEFAULT_MIN_CONSECUTIVE = 3
DEFAULT_RECORDS_PER_STEP = 12
DEFAULT_DIRECT_PATH_TIME_US = 300.0
DEFAULT_NOISE_GUARD_MULTIPLIER = 3.0
DEFAULT_TOP_SEARCH_RESULTS = 5


def resolve_borealis_path(borealis_path: str | Path | None = None) -> Path:
    if borealis_path is not None:
        return Path(borealis_path).expanduser().resolve()
    env_path = os.environ.get("BOREALISPATH")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def resolve_radar_id(radar_id: str | None = None) -> str:
    return radar_id or os.environ.get("RADAR_ID", "wal")


def resolve_config_path(
    config_path: str | Path | None = None,
    radar_id: str | None = None,
    borealis_path: str | Path | None = None,
) -> Path:
    if config_path is not None:
        return Path(config_path).expanduser().resolve()
    base_path = resolve_borealis_path(borealis_path)
    rid = resolve_radar_id(radar_id)
    return (base_path / "config" / rid / f"{rid}_config.ini").resolve()


def load_config(
    config_path: str | Path | None = None,
    radar_id: str | None = None,
    borealis_path: str | Path | None = None,
) -> tuple[dict[str, Any], Path]:
    path = resolve_config_path(config_path, radar_id, borealis_path)
    with path.open("r", encoding="utf-8") as src:
        return json.load(src), path


def parse_int_list(value: str | list[int] | tuple[int, ...] | None) -> list[int]:
    if value is None:
        return []
    if isinstance(value, str):
        if value.strip() == "":
            return []
        return [int(part) for part in value.split(",") if part.strip()]
    return [int(item) for item in value]


def parse_float_list(value: str | list[float] | tuple[float, ...]) -> list[float]:
    if isinstance(value, str):
        return [float(part) for part in value.split(",") if part.strip()]
    return [float(item) for item in value]


def parse_float_grid(value: str | list[float] | tuple[float, ...]) -> list[float]:
    if not isinstance(value, str):
        return [float(item) for item in value]
    text = value.strip()
    if not text:
        return []
    if ":" not in text:
        return parse_float_list(text)
    parts = [part.strip() for part in text.split(":")]
    if len(parts) != 3:
        raise ValueError(
            f"Expected start:stop:step float grid, got '{value}'"
        )
    start, stop, step = (float(part) for part in parts)
    if not math.isfinite(start) or not math.isfinite(stop) or not math.isfinite(step):
        raise ValueError(f"Non-finite value in float grid '{value}'")
    if step <= 0.0:
        raise ValueError(f"Step must be > 0 in float grid '{value}'")
    values: list[float] = []
    current = start
    limit = stop + (step * 0.5)
    while current <= limit:
        values.append(float(round(current, 9)))
        current += step
    return values


def main_locations(config: dict[str, Any]) -> dict[int, np.ndarray]:
    return {
        int(ant): np.asarray(coords, dtype=np.float64)
        for ant, coords in config["antennas"]["main_locations"].items()
    }


def active_main_antennas(config: dict[str, Any]) -> list[int]:
    antennas = set()
    for n200 in config["n200s"]:
        for key in ("rx_channel_0", "rx_channel_1"):
            channel = n200.get(key, "")
            if channel.startswith("m"):
                antennas.add(int(channel[1:]))
    return sorted(antennas)


def active_intf_antennas(config: dict[str, Any]) -> list[int]:
    antennas = set()
    for n200 in config["n200s"]:
        for key in ("rx_channel_0", "rx_channel_1"):
            channel = n200.get(key, "")
            if channel.startswith("i"):
                antennas.add(int(channel[1:]))
    return sorted(antennas)


def active_tx_antennas(config: dict[str, Any]) -> list[int]:
    antennas = set()
    for n200 in config["n200s"]:
        channel = n200.get("tx_channel_0", "")
        if channel.startswith("m"):
            antennas.add(int(channel[1:]))
    return sorted(antennas)


def rank_rx_antennas(
    tx_ant: int,
    rx_antennas: list[int],
    locations: dict[int, np.ndarray],
) -> list[dict[str, float | int]]:
    tx_loc = locations[tx_ant]
    ranked = []
    for rx_ant in rx_antennas:
        if rx_ant == tx_ant:
            continue
        distance_m = float(np.linalg.norm(locations[rx_ant] - tx_loc))
        ranked.append({"rx_ant": int(rx_ant), "distance_m": distance_m})
    return sorted(ranked, key=lambda item: (-item["distance_m"], item["rx_ant"]))


def shlex_join(command: list[str | Path]) -> str:
    return shlex.join([str(part) for part in command])


def format_active_reference_cal_command(
    *,
    borealis_path: str | Path | None,
    freq: int,
    tx_ant: int,
    rx_main_antennas: list[int],
    tx_scale: float,
    intt_ms: int,
    num_ranges: int,
    rx_intf_antennas: list[int] | None = None,
    pulse_scheme: str | None = None,
    pulse_sequence: list[int] | None = None,
    tau_spacing_us: int | None = None,
    pulse_len_us: int | None = None,
    run_mode: str = "release",
    scheduling_mode: str = "special",
    realtime_off: bool = True,
) -> list[str]:
    base_path = resolve_borealis_path(borealis_path)
    kwargs = [
        f"freq={int(freq)}",
        f"tx_ant={int(tx_ant)}",
        f"tx_scale={tx_scale:.6f}",
        f"intt={int(intt_ms)}",
        f"num_ranges={int(num_ranges)}",
        f"rx_main_antennas={','.join(str(ant) for ant in rx_main_antennas)}",
    ]
    if rx_intf_antennas is not None:
        kwargs.append(
            f"rx_intf_antennas={','.join(str(ant) for ant in rx_intf_antennas)}"
        )
    if pulse_scheme:
        kwargs.append(f"pulse_scheme={pulse_scheme}")
    if pulse_sequence:
        kwargs.append(
            f"pulse_sequence={','.join(str(int(pulse)) for pulse in pulse_sequence)}"
        )
    if tau_spacing_us is not None:
        kwargs.append(f"tau_spacing={int(tau_spacing_us)}")
    if pulse_len_us is not None:
        kwargs.append(f"pulse_len={int(pulse_len_us)}")

    command = [
        str(base_path / "scripts" / "steamed_hams.py"),
        "active_reference_cal",
        run_mode,
        scheduling_mode,
    ]
    if realtime_off:
        command.append("--realtime-off")
    command.append("--kwargs")
    command.extend(kwargs)
    return command


def _get_dataset(group: h5py.Group, name: str) -> np.ndarray:
    if name in group:
        return group[name][...]
    if name in group.file:
        return group.file[name][...]
    raise KeyError(f"Dataset '{name}' not found in record '{group.name}'")


def _get_scalar(group: h5py.Group, name: str) -> Any:
    if name in group:
        return group[name][()]
    if name in group.file:
        return group.file[name][()]
    raise KeyError(f"Dataset '{name}' not found in record '{group.name}'")


def _optional_dataset(group: h5py.Group, name: str) -> np.ndarray | None:
    if name in group:
        return group[name][...]
    if name in group.file:
        return group.file[name][...]
    return None


def pick_records(src: h5py.File, record: str | None, records: int) -> list[str]:
    all_records = sorted(
        key for key in src.keys() if isinstance(src[key], h5py.Group) and key != "metadata"
    )
    if not all_records:
        raise RuntimeError("No records in file")

    if record:
        if record not in all_records:
            raise ValueError(f"Record {record} not in file")
        return [record]

    if records <= 0:
        return all_records

    keep = min(records, len(all_records))
    return all_records[-keep:]


def determine_tx_metadata(group: h5py.Group) -> tuple[int, float]:
    tx_antennas = np.asarray(_get_dataset(group, "tx_antennas"), dtype=np.int32).reshape(
        -1
    )
    tx_excitations = np.asarray(
        _get_dataset(group, "tx_excitations"), dtype=np.complex64
    ).reshape(-1)
    if tx_excitations.size == 0:
        raise RuntimeError("tx_excitations dataset is empty")

    magnitudes = np.abs(tx_excitations)
    if tx_antennas.size == 1:
        return int(tx_antennas[0]), float(np.max(magnitudes))

    if tx_excitations.size == tx_antennas.size and tx_antennas.size > 0:
        best_idx = int(np.argmax(magnitudes))
        return int(tx_antennas[best_idx]), float(magnitudes[best_idx])

    best_idx = int(np.argmax(magnitudes))
    return best_idx, float(magnitudes[best_idx])


def _sample_step_us(sample_time_us: np.ndarray) -> float:
    if sample_time_us.size < 2:
        return 1.0
    diffs = np.diff(sample_time_us)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return 1.0
    return float(np.median(diffs))


def _window_bounds(
    sample_time_us: np.ndarray,
    peak_idx: int,
    pulse_window_us: float,
) -> tuple[int, int]:
    sample_step_us = _sample_step_us(sample_time_us)
    window_len = max(1, int(np.ceil(max(pulse_window_us, sample_step_us) / sample_step_us)))
    start = max(0, peak_idx - (window_len // 2))
    stop = min(sample_time_us.size, start + window_len)
    start = max(0, stop - window_len)
    return start, stop


def pulse_offsets_us_for_group(
    group: h5py.Group,
    *,
    pulse_offsets_us: list[float] | np.ndarray | None = None,
) -> np.ndarray:
    if pulse_offsets_us is not None:
        offsets = np.asarray(pulse_offsets_us, dtype=np.float64).reshape(-1)
        if offsets.size == 0:
            return np.asarray([0.0], dtype=np.float64)
        return offsets

    pulse_sequence = _optional_dataset(group, "pulse_sequence")
    if pulse_sequence is None:
        pulse_sequence = _optional_dataset(group, "pulses")
    if pulse_sequence is None:
        return np.asarray([0.0], dtype=np.float64)

    tau_spacing_value = _optional_dataset(group, "tau_spacing")
    if tau_spacing_value is None:
        tx_pulse_len = float(_get_scalar(group, "tx_pulse_len"))
        tau_spacing_us = tx_pulse_len
    else:
        tau_spacing_us = float(np.asarray(tau_spacing_value, dtype=np.float64).reshape(-1)[0])

    sequence = np.asarray(pulse_sequence, dtype=np.float64).reshape(-1)
    if sequence.size == 0:
        return np.asarray([0.0], dtype=np.float64)
    return sequence * tau_spacing_us


def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not spans:
        return []
    ordered = sorted(spans, key=lambda item: (item[0], item[1]))
    merged = [ordered[0]]
    for start, stop in ordered[1:]:
        prev_start, prev_stop = merged[-1]
        if start <= prev_stop:
            merged[-1] = (prev_start, max(prev_stop, stop))
        else:
            merged.append((start, stop))
    return merged


def _aggregate_complex(values: list[complex]) -> complex:
    finite = np.asarray(values, dtype=np.complex128)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.complex128(np.nan + 1j * np.nan)
    amplitudes = np.abs(finite)
    phases = np.angle(finite)
    phase_mean = np.angle(np.mean(np.exp(1j * phases)))
    amplitude_median = np.median(amplitudes)
    return np.complex128(amplitude_median * np.exp(1j * phase_mean))


def max_consecutive_true(flags: list[bool]) -> int:
    best = 0
    current = 0
    for flag in flags:
        if flag:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def measure_record(
    group: h5py.Group,
    rx_ant: int,
    *,
    pnr_threshold_db: float = DEFAULT_PNR_THRESHOLD_DB,
    pulse_window_us: float | None = None,
    direct_path_time_us: float | None = DEFAULT_DIRECT_PATH_TIME_US,
    noise_guard_multiplier: float = DEFAULT_NOISE_GUARD_MULTIPLIER,
    pulse_offsets_us: list[float] | np.ndarray | None = None,
) -> dict[str, Any]:
    tx_ant, tx_scale = determine_tx_metadata(group)

    rx_antennas = np.asarray(_get_dataset(group, "rx_antennas"), dtype=np.int32).reshape(
        -1
    )
    matches = np.where(rx_antennas == int(rx_ant))[0]
    if matches.size == 0:
        raise RuntimeError(f"RX antenna {rx_ant} not present in record {group.name}")

    sample_time_us = np.asarray(_get_dataset(group, "sample_time"), dtype=np.float64).reshape(
        -1
    )
    channel_index = int(matches[0])
    channel_data = np.asarray(group["antennas_iq_data"][channel_index], dtype=np.complex64)
    if channel_data.size == 0:
        raise RuntimeError(f"RX antenna {rx_ant} has no samples in record {group.name}")

    tx_pulse_len_us = float(_get_scalar(group, "tx_pulse_len"))
    window_us = pulse_window_us if pulse_window_us is not None else tx_pulse_len_us

    power_time = np.mean(np.abs(channel_data) ** 2, axis=0)
    if direct_path_time_us is None:
        first_peak_idx = int(np.argmax(power_time))
        base_peak_time_us = float(sample_time_us[first_peak_idx])
    else:
        base_peak_time_us = float(direct_path_time_us)

    offsets_us = pulse_offsets_us_for_group(group, pulse_offsets_us=pulse_offsets_us)
    target_times_us = [base_peak_time_us + float(offset) for offset in offsets_us]
    peak_indices = [
        int(np.argmin(np.abs(sample_time_us - target_time_us)))
        for target_time_us in target_times_us
    ]
    pulse_spans = [
        _window_bounds(sample_time_us, peak_idx, window_us)
        for peak_idx in peak_indices
    ]
    merged_spans = _merge_spans(pulse_spans)

    pulse_span_lengths = [stop - start for start, stop in merged_spans]
    representative_window_len = max(1, int(np.median(pulse_span_lengths)))
    guard_len = max(
        1,
        int(np.ceil(representative_window_len * max(noise_guard_multiplier, 1.0))),
    )
    noise_mask = np.ones(sample_time_us.size, dtype=bool)
    for pulse_start, pulse_stop in merged_spans:
        noise_mask[
            max(0, pulse_start - guard_len) : min(sample_time_us.size, pulse_stop + guard_len)
        ] = False
    if not np.any(noise_mask):
        noise_mask[:] = True
        for pulse_start, pulse_stop in merged_spans:
            noise_mask[pulse_start:pulse_stop] = False
    if not np.any(noise_mask):
        raise RuntimeError(f"Unable to construct off-pulse noise window for record {group.name}")

    pulse_segments = [channel_data[:, pulse_start:pulse_stop].reshape(-1) for pulse_start, pulse_stop in merged_spans]
    pulse_samples = np.concatenate(pulse_segments)
    noise_samples = channel_data[:, noise_mask].reshape(-1)
    pulse_power = float(np.mean(np.abs(pulse_samples) ** 2))
    noise_power = float(np.mean(np.abs(noise_samples) ** 2))
    pulse_excess_power = float(max(pulse_power - noise_power, 0.0))
    if noise_power <= 0.0:
        pnr_db = float("nan")
        pulse_to_noise_linear = float("nan")
    else:
        pulse_to_noise_linear = float(max(pulse_power, 1.0e-30) / noise_power)
        pnr_db = float(10.0 * np.log10(pulse_to_noise_linear))

    pulse_complex_means = [
        np.mean(channel_data[:, pulse_start:pulse_stop].reshape(-1))
        for pulse_start, pulse_stop in merged_spans
    ]
    complex_response = np.complex128(np.mean(np.asarray(pulse_complex_means, dtype=np.complex128)))
    agc_status_word = int(_get_scalar(group, "agc_status_word"))
    accepted = bool(
        agc_status_word == 0
        and np.isfinite(pnr_db)
        and pnr_db >= pnr_threshold_db
        and np.isfinite(complex_response.real)
        and np.isfinite(complex_response.imag)
    )

    return {
        "record": group.name.rsplit("/", 1)[-1],
        "tx_ant": int(tx_ant),
        "tx_scale": float(tx_scale),
        "rx_ant": int(rx_ant),
        "pulse_power": pulse_power,
        "noise_power": noise_power,
        "pulse_excess_power": pulse_excess_power,
        "pulse_to_noise_linear": pulse_to_noise_linear,
        "pnr_db": pnr_db,
        "pnr_threshold_db": float(pnr_threshold_db),
        "peak_sample_mag": float(np.max(np.abs(pulse_samples))),
        "peak_time_us": float(base_peak_time_us),
        "direct_path_time_us": (
            None if direct_path_time_us is None else float(direct_path_time_us)
        ),
        "pulse_window_us": float(window_us),
        "pulse_count": int(offsets_us.size),
        "pulse_offsets_us": [float(offset) for offset in offsets_us.tolist()],
        "pulse_center_times_us": [
            float(sample_time_us[peak_idx]) for peak_idx in peak_indices
        ],
        "noise_guard_multiplier": float(max(noise_guard_multiplier, 1.0)),
        "pulse_window_start_us": float(sample_time_us[merged_spans[0][0]]),
        "pulse_window_stop_us": float(sample_time_us[merged_spans[-1][1] - 1]),
        "agc_status_word": agc_status_word,
        "complex_response_real": float(complex_response.real),
        "complex_response_imag": float(complex_response.imag),
        "accepted": accepted,
    }


def aggregate_measurements(
    input_path: str | Path,
    metrics: list[dict[str, Any]],
    *,
    min_consecutive: int = DEFAULT_MIN_CONSECUTIVE,
) -> dict[str, Any]:
    if not metrics:
        raise RuntimeError("No record metrics were produced")

    tx_ants = {metric["tx_ant"] for metric in metrics}
    rx_ants = {metric["rx_ant"] for metric in metrics}
    if len(tx_ants) != 1:
        raise RuntimeError(f"Inconsistent TX antennas in {input_path}: {sorted(tx_ants)}")
    if len(rx_ants) != 1:
        raise RuntimeError(f"Inconsistent RX antennas in {input_path}: {sorted(rx_ants)}")

    accepted_metrics = [metric for metric in metrics if metric["accepted"]]
    complex_values = [
        complex(metric["complex_response_real"], metric["complex_response_imag"])
        for metric in (accepted_metrics or metrics)
    ]
    complex_response = _aggregate_complex(complex_values)
    accept_flags = [metric["accepted"] for metric in metrics]

    return {
        "input": str(Path(input_path).expanduser().resolve()),
        "tx_ant": int(next(iter(tx_ants))),
        "tx_scale": float(np.median([metric["tx_scale"] for metric in metrics])),
        "rx_ant": int(next(iter(rx_ants))),
        "direct_path_time_us": next(
            (
                float(metric["direct_path_time_us"])
                for metric in metrics
                if metric["direct_path_time_us"] is not None
            ),
            None,
        ),
        "pulse_window_us": float(np.median([metric["pulse_window_us"] for metric in metrics])),
        "pulse_count": int(np.median([metric["pulse_count"] for metric in metrics])),
        "pulse_offsets_us": next(
            (
                metric["pulse_offsets_us"]
                for metric in metrics
                if metric.get("pulse_offsets_us")
            ),
            [0.0],
        ),
        "noise_guard_multiplier": float(np.median([metric["noise_guard_multiplier"] for metric in metrics])),
        "pulse_power": float(np.median([metric["pulse_power"] for metric in metrics])),
        "noise_power": float(np.median([metric["noise_power"] for metric in metrics])),
        "pulse_excess_power": float(np.median([metric["pulse_excess_power"] for metric in metrics])),
        "pulse_to_noise_linear": float(np.median([metric["pulse_to_noise_linear"] for metric in metrics])),
        "pnr_db": float(np.median([metric["pnr_db"] for metric in metrics])),
        "pnr_threshold_db": float(np.median([metric["pnr_threshold_db"] for metric in metrics])),
        "pnr_margin_db": float(
            np.median([metric["pnr_db"] - metric["pnr_threshold_db"] for metric in metrics])
        ),
        "peak_sample_mag": float(np.median([metric["peak_sample_mag"] for metric in metrics])),
        "peak_time_us": float(np.median([metric["peak_time_us"] for metric in metrics])),
        "agc_status_word": int(np.bitwise_or.reduce([metric["agc_status_word"] for metric in metrics])),
        "records_evaluated": len(metrics),
        "accepted_record_count": len(accepted_metrics),
        "accepted_fraction": float(len(accepted_metrics) / len(metrics)),
        "max_consecutive_accepts": max_consecutive_true(accept_flags),
        "detection_pass": bool(max_consecutive_true(accept_flags) >= min_consecutive),
        "complex_response_real": float(complex_response.real),
        "complex_response_imag": float(complex_response.imag),
        "record_metrics": metrics,
    }


def evaluate_file(
    input_path: str | Path,
    *,
    rx_ant: int | None = None,
    record: str | None = None,
    records: int = DEFAULT_RECORDS_PER_STEP,
    pnr_threshold_db: float = DEFAULT_PNR_THRESHOLD_DB,
    min_consecutive: int = DEFAULT_MIN_CONSECUTIVE,
    pulse_window_us: float | None = None,
    direct_path_time_us: float | None = DEFAULT_DIRECT_PATH_TIME_US,
    noise_guard_multiplier: float = DEFAULT_NOISE_GUARD_MULTIPLIER,
    pulse_offsets_us: list[float] | np.ndarray | None = None,
) -> dict[str, Any]:
    path = Path(input_path).expanduser().resolve()
    with h5py.File(path, "r") as src:
        selected = pick_records(src, record, records)
        first_group = src[selected[0]]
        if rx_ant is None:
            rx_main_antennas = np.asarray(
                _get_dataset(first_group, "rx_main_antennas"), dtype=np.int32
            ).reshape(-1)
            if rx_main_antennas.size != 1:
                raise ValueError(
                    f"File {path} contains {rx_main_antennas.size} main RX antennas; supply --rx-ant"
                )
            rx_ant = int(rx_main_antennas[0])

        metrics = [
            measure_record(
                src[record_name],
                int(rx_ant),
                pnr_threshold_db=pnr_threshold_db,
                pulse_window_us=pulse_window_us,
                direct_path_time_us=direct_path_time_us,
                noise_guard_multiplier=noise_guard_multiplier,
                pulse_offsets_us=pulse_offsets_us,
            )
            for record_name in selected
        ]
    return aggregate_measurements(path, metrics, min_consecutive=min_consecutive)


def evaluate_file_all_main_rx(
    input_path: str | Path,
    *,
    record: str | None = None,
    records: int = DEFAULT_RECORDS_PER_STEP,
    pnr_threshold_db: float = DEFAULT_PNR_THRESHOLD_DB,
    min_consecutive: int = DEFAULT_MIN_CONSECUTIVE,
    pulse_window_us: float | None = None,
    direct_path_time_us: float | None = DEFAULT_DIRECT_PATH_TIME_US,
    noise_guard_multiplier: float = DEFAULT_NOISE_GUARD_MULTIPLIER,
    pulse_offsets_us: list[float] | np.ndarray | None = None,
) -> list[dict[str, Any]]:
    path = Path(input_path).expanduser().resolve()
    with h5py.File(path, "r") as src:
        selected = pick_records(src, record, records)
        first_group = src[selected[0]]
        rx_main_antennas = np.asarray(
            _get_dataset(first_group, "rx_main_antennas"), dtype=np.int32
        ).reshape(-1)
        summaries = []
        for rx_ant in rx_main_antennas.tolist():
            metrics = [
                measure_record(
                    src[record_name],
                    int(rx_ant),
                    pnr_threshold_db=pnr_threshold_db,
                    pulse_window_us=pulse_window_us,
                    direct_path_time_us=direct_path_time_us,
                    noise_guard_multiplier=noise_guard_multiplier,
                    pulse_offsets_us=pulse_offsets_us,
                )
                for record_name in selected
            ]
            summaries.append(
                aggregate_measurements(path, metrics, min_consecutive=min_consecutive)
            )
    return summaries


def _summary_score(summary: dict[str, Any]) -> tuple[float, ...]:
    return (
        0.0 if int(summary.get("agc_status_word", 0)) != 0 else 1.0,
        1.0 if bool(summary.get("detection_pass", False)) else 0.0,
        float(summary.get("accepted_record_count", 0)),
        float(summary.get("max_consecutive_accepts", 0)),
        float(summary.get("pnr_db", float("-inf"))),
        float(summary.get("pulse_excess_power", float("-inf"))),
        -abs(float(summary.get("direct_path_time_us") or 0.0) - DEFAULT_DIRECT_PATH_TIME_US),
    )


def evaluate_file_search(
    input_path: str | Path,
    *,
    rx_ant: int | None = None,
    record: str | None = None,
    records: int = DEFAULT_RECORDS_PER_STEP,
    pnr_threshold_db: float = DEFAULT_PNR_THRESHOLD_DB,
    min_consecutive: int = DEFAULT_MIN_CONSECUTIVE,
    direct_path_times_us: list[float] | tuple[float, ...],
    pulse_window_values_us: list[float] | tuple[float, ...],
    noise_guard_multiplier: float = DEFAULT_NOISE_GUARD_MULTIPLIER,
    pulse_offsets_us: list[float] | np.ndarray | None = None,
    top_results: int = DEFAULT_TOP_SEARCH_RESULTS,
) -> dict[str, Any]:
    if not direct_path_times_us:
        raise ValueError("direct_path_times_us must not be empty")
    if not pulse_window_values_us:
        raise ValueError("pulse_window_values_us must not be empty")

    path = Path(input_path).expanduser().resolve()
    with h5py.File(path, "r") as src:
        selected = pick_records(src, record, records)
        first_group = src[selected[0]]
        if rx_ant is None:
            rx_main_antennas = np.asarray(
                _get_dataset(first_group, "rx_main_antennas"), dtype=np.int32
            ).reshape(-1)
            if rx_main_antennas.size != 1:
                raise ValueError(
                    f"File {path} contains {rx_main_antennas.size} main RX antennas; supply --rx-ant"
                )
            rx_ant = int(rx_main_antennas[0])

        search_results: list[dict[str, Any]] = []
        for direct_path_time_us in direct_path_times_us:
            for pulse_window_us in pulse_window_values_us:
                metrics = [
                    measure_record(
                        src[record_name],
                        int(rx_ant),
                        pnr_threshold_db=pnr_threshold_db,
                        pulse_window_us=float(pulse_window_us),
                        direct_path_time_us=float(direct_path_time_us),
                        noise_guard_multiplier=noise_guard_multiplier,
                        pulse_offsets_us=pulse_offsets_us,
                    )
                    for record_name in selected
                ]
                summary = aggregate_measurements(path, metrics, min_consecutive=min_consecutive)
                summary["search_mode"] = "adaptive_grid"
                summary["search_score"] = list(_summary_score(summary))
                search_results.append(summary)

    ranked = sorted(search_results, key=_summary_score, reverse=True)
    best = dict(ranked[0])
    best["search_mode"] = "adaptive_grid"
    best["search_candidate_count"] = len(ranked)
    best["search_direct_path_times_us"] = [float(value) for value in direct_path_times_us]
    best["search_pulse_window_values_us"] = [float(value) for value in pulse_window_values_us]
    best["top_search_results"] = [
        {
            "direct_path_time_us": float(item["direct_path_time_us"]),
            "pulse_window_us": float(item["pulse_window_us"]),
            "pnr_db": float(item["pnr_db"]),
            "accepted_record_count": int(item["accepted_record_count"]),
            "max_consecutive_accepts": int(item["max_consecutive_accepts"]),
            "detection_pass": bool(item["detection_pass"]),
            "pulse_excess_power": float(item["pulse_excess_power"]),
        }
        for item in ranked[: max(1, int(top_results))]
    ]
    return best


def append_csv_row(output_path: str | Path, row: dict[str, Any]) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "input",
        "tx_ant",
        "tx_scale",
        "rx_ant",
        "direct_path_time_us",
        "pulse_window_us",
        "pulse_count",
        "pulse_power",
        "noise_power",
        "pulse_excess_power",
        "pulse_to_noise_linear",
        "pnr_db",
        "pnr_threshold_db",
        "pnr_margin_db",
        "peak_sample_mag",
        "peak_time_us",
        "agc_status_word",
        "accepted_fraction",
        "records_evaluated",
        "accepted_record_count",
        "max_consecutive_accepts",
        "detection_pass",
        "search_mode",
        "search_candidate_count",
        "complex_response_real",
        "complex_response_imag",
    ]
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({name: row.get(name) for name in fieldnames})
    return path
