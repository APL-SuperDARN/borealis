#!/usr/bin/env python3
"""
Shared helpers for Wallops MCCM qualification and solve scripts.
"""

from __future__ import annotations

import csv
import json
import os
import shlex
from pathlib import Path
from typing import Any

import h5py
import numpy as np

DEFAULT_SCALE_LIST = [0.001, 0.002, 0.003, 0.005, 0.007, 0.01]
DEFAULT_PNR_THRESHOLD_DB = 10.0
DEFAULT_MIN_CONSECUTIVE = 3


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

    keep = min(max(records, 1), len(all_records))
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
    peak_idx = int(np.argmax(power_time))
    pulse_start, pulse_stop = _window_bounds(sample_time_us, peak_idx, window_us)

    guard_len = max(1, pulse_stop - pulse_start)
    noise_mask = np.ones(sample_time_us.size, dtype=bool)
    noise_mask[max(0, pulse_start - guard_len) : min(sample_time_us.size, pulse_stop + guard_len)] = False
    if not np.any(noise_mask):
        noise_mask[:] = True
        noise_mask[pulse_start:pulse_stop] = False
    if not np.any(noise_mask):
        raise RuntimeError(f"Unable to construct off-pulse noise window for record {group.name}")

    pulse_samples = channel_data[:, pulse_start:pulse_stop].reshape(-1)
    noise_samples = channel_data[:, noise_mask].reshape(-1)
    pulse_power = float(np.mean(np.abs(pulse_samples) ** 2))
    noise_power = float(np.mean(np.abs(noise_samples) ** 2))
    if noise_power <= 0.0:
        pnr_db = float("nan")
    else:
        pnr_db = float(10.0 * np.log10(max(pulse_power, 1.0e-30) / noise_power))

    complex_response = np.complex128(np.mean(pulse_samples))
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
        "pnr_db": pnr_db,
        "peak_sample_mag": float(np.max(np.abs(pulse_samples))),
        "peak_time_us": float(sample_time_us[peak_idx]),
        "pulse_window_start_us": float(sample_time_us[pulse_start]),
        "pulse_window_stop_us": float(sample_time_us[pulse_stop - 1]),
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
        "pulse_power": float(np.median([metric["pulse_power"] for metric in metrics])),
        "noise_power": float(np.median([metric["noise_power"] for metric in metrics])),
        "pnr_db": float(np.median([metric["pnr_db"] for metric in metrics])),
        "peak_sample_mag": float(np.median([metric["peak_sample_mag"] for metric in metrics])),
        "peak_time_us": float(np.median([metric["peak_time_us"] for metric in metrics])),
        "agc_status_word": int(np.bitwise_or.reduce([metric["agc_status_word"] for metric in metrics])),
        "records_evaluated": len(metrics),
        "accepted_record_count": len(accepted_metrics),
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
    records: int = 5,
    pnr_threshold_db: float = DEFAULT_PNR_THRESHOLD_DB,
    min_consecutive: int = DEFAULT_MIN_CONSECUTIVE,
    pulse_window_us: float | None = None,
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
            )
            for record_name in selected
        ]
    return aggregate_measurements(path, metrics, min_consecutive=min_consecutive)


def evaluate_file_all_main_rx(
    input_path: str | Path,
    *,
    record: str | None = None,
    records: int = 5,
    pnr_threshold_db: float = DEFAULT_PNR_THRESHOLD_DB,
    min_consecutive: int = 1,
    pulse_window_us: float | None = None,
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
                )
                for record_name in selected
            ]
            summaries.append(
                aggregate_measurements(path, metrics, min_consecutive=min_consecutive)
            )
    return summaries


def append_csv_row(output_path: str | Path, row: dict[str, Any]) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "input",
        "tx_ant",
        "tx_scale",
        "rx_ant",
        "pulse_power",
        "noise_power",
        "pnr_db",
        "peak_sample_mag",
        "peak_time_us",
        "agc_status_word",
        "records_evaluated",
        "accepted_record_count",
        "max_consecutive_accepts",
        "detection_pass",
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
