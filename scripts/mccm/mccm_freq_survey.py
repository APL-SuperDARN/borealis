#!/usr/bin/env python3
"""
Choose a fixed MCCM session frequency from a set of RX-only survey captures.

This script can score candidate frequencies in two modes:
1. noise-only: rank by per-channel RX noise from the actual MCCM receive subset
2. combined: blend RX noise with channel-specific installed match quality from
   Wallops Touchstone files for the selected TX and RX subsets
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from mccm_common import active_main_antennas, active_tx_antennas, load_config, parse_int_list, pick_records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pick a fixed MCCM session frequency from RX-only survey files")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input antennas_iq survey files",
    )
    parser.add_argument(
        "--records",
        type=int,
        default=12,
        help="Number of most recent records to analyze per file (<=0 means all)",
    )
    parser.add_argument(
        "--rx-antennas",
        help="Comma-separated explicit main-array RX antenna list to score; defaults to active main antennas from config",
    )
    parser.add_argument(
        "--tx-antennas",
        help="Comma-separated TX main antennas to include in combined match scoring; defaults to active TX antennas from config",
    )
    parser.add_argument(
        "--measurement-sources-csv",
        help="Optional main-array measurement source CSV with source_file paths for Touchstone interpolation",
    )
    parser.add_argument(
        "--noise-weight",
        type=float,
        default=0.6,
        help="Relative weight for the RX-noise score in combined mode",
    )
    parser.add_argument(
        "--tx-match-weight",
        type=float,
        default=0.3,
        help="Relative weight for the TX-path match score in combined mode",
    )
    parser.add_argument(
        "--rx-match-weight",
        type=float,
        default=0.1,
        help="Relative weight for the RX-path match score in combined mode",
    )
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--borealis-path", help="Path to Borealis repo if --config omitted")
    parser.add_argument("--radar-id", default="wal", help="Radar ID if --config omitted")
    parser.add_argument("--summary-json", help="Optional JSON output path")
    parser.add_argument("--summary-csv", help="Optional CSV output path")
    return parser


def parse_s1p(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fmt = None
    unit_scale = 1.0
    rows: list[tuple[float, float, float]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("!"):
            continue
        if line.startswith("#"):
            parts = line.split()
            unit = parts[1].upper()
            fmt = parts[3].upper()
            unit_scale = {
                "HZ": 1.0,
                "KHZ": 1e3,
                "MHZ": 1e6,
                "GHZ": 1e9,
            }[unit]
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        rows.append((float(parts[0]) * unit_scale, float(parts[1]), float(parts[2])))
    if fmt != "DB":
        raise ValueError(f"Unsupported Touchstone format in {path}: {fmt}")
    arr = np.asarray(rows, dtype=float)
    freq_hz = arr[:, 0]
    magnitude = 10 ** (arr[:, 1] / 20.0)
    phase_rad = np.deg2rad(arr[:, 2])
    gamma = np.clip(np.abs(magnitude * np.exp(1j * phase_rad)), 0.0, 0.999999999)
    rl_db = -20.0 * np.log10(np.clip(gamma, 1.0e-15, None))
    vswr = (1.0 + gamma) / (1.0 - gamma)
    return freq_hz, vswr, rl_db


def resolve_source_path(csv_path: Path, raw_path: str) -> Path | None:
    source_path = Path(raw_path).expanduser()
    if source_path.exists():
        return source_path.resolve()
    candidate = csv_path.parent / source_path.name
    if candidate.exists():
        return candidate.resolve()
    return None


def median_noise_by_rx(input_path: Path, *, records: int, rx_antennas: list[int] | None = None) -> dict[str, Any]:
    with h5py.File(input_path, "r") as src:
        selected_records = pick_records(src, None, records)
        first_group = src[selected_records[0]]
        file_rx_antennas = np.asarray(first_group["rx_main_antennas"], dtype=np.int32).reshape(-1).tolist()
        if rx_antennas is None:
            use_antennas = file_rx_antennas
        else:
            use_antennas = [ant for ant in rx_antennas if ant in file_rx_antennas]
        if not use_antennas:
            raise RuntimeError(f"No requested RX antennas present in {input_path}")

        freq_khz = int(np.asarray(first_group["freq"], dtype=np.int32).reshape(-1)[0])
        per_ant_record_noise: dict[int, list[float]] = {int(ant): [] for ant in use_antennas}
        for record_name in selected_records:
            group = src[record_name]
            rx_list = np.asarray(group["rx_antennas"], dtype=np.int32).reshape(-1)
            data = np.asarray(group["antennas_iq_data"], dtype=np.complex64)
            for ant in use_antennas:
                matches = np.where(rx_list == int(ant))[0]
                if matches.size == 0:
                    continue
                channel_data = data[int(matches[0])]
                per_ant_record_noise[int(ant)].append(float(np.mean(np.abs(channel_data.reshape(-1)) ** 2)))

    per_ant_median = {
        int(ant): float(np.median(values))
        for ant, values in per_ant_record_noise.items()
        if values
    }
    if not per_ant_median:
        raise RuntimeError(f"No per-channel noise metrics were produced for {input_path}")

    return {
        "input": str(input_path.resolve()),
        "freq_khz": freq_khz,
        "records_evaluated": len(selected_records),
        "rx_antennas": sorted(per_ant_median),
        "channel_noise_power": per_ant_median,
        "median_noise_power": float(np.median(list(per_ant_median.values()))),
        "max_noise_power": float(np.max(list(per_ant_median.values()))),
    }


def summaries_from_combined_json(input_path: Path, *, rx_antennas: list[int] | None = None) -> list[dict[str, Any]]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    rows = payload.get("frequencies")
    if not isinstance(rows, list):
        raise ValueError(f"JSON input {input_path} does not contain a frequency summary")

    summaries = []
    for row in rows:
        channel_noise_power = {
            int(ant): float(noise)
            for ant, noise in row.get("channel_noise_power", {}).items()
            if rx_antennas is None or int(ant) in rx_antennas
        }
        if not channel_noise_power:
            continue
        summaries.append(
            {
                "input": str(input_path.resolve()),
                "freq_khz": int(row["freq_khz"]),
                "records_evaluated": int(row.get("records_evaluated", 0)),
                "rx_antennas": sorted(channel_noise_power),
                "channel_noise_power": channel_noise_power,
                "median_noise_power": float(np.median(list(channel_noise_power.values()))),
                "max_noise_power": float(np.max(list(channel_noise_power.values()))),
            }
        )
    if not summaries:
        raise RuntimeError(f"No usable frequency rows found in {input_path}")
    return summaries


def build_noise_summary(file_summaries: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[int, dict[int, float]]]:
    per_freq: dict[int, dict[str, Any]] = {}
    all_antennas = sorted(
        {int(ant) for summary in file_summaries for ant in summary["channel_noise_power"].keys()}
    )
    for summary in file_summaries:
        freq = int(summary["freq_khz"])
        bucket = per_freq.setdefault(
            freq,
            {
                "freq_khz": freq,
                "inputs": [],
                "channel_noise_power": {int(ant): [] for ant in all_antennas},
            },
        )
        bucket["inputs"].append(summary["input"])
        for ant, noise in summary["channel_noise_power"].items():
            bucket["channel_noise_power"][int(ant)].append(float(noise))

    for bucket in per_freq.values():
        bucket["channel_noise_power"] = {
            int(ant): float(np.median(values))
            for ant, values in bucket["channel_noise_power"].items()
            if values
        }
        bucket["median_noise_power"] = float(np.median(list(bucket["channel_noise_power"].values())))
        bucket["median_noise_db"] = float(10.0 * np.log10(bucket["median_noise_power"]))

    noise_db_by_ant: dict[int, dict[int, float]] = {}
    for ant in all_antennas:
        freq_values = {
            int(freq): float(10.0 * np.log10(bucket["channel_noise_power"][int(ant)]))
            for freq, bucket in per_freq.items()
            if int(ant) in bucket["channel_noise_power"] and bucket["channel_noise_power"][int(ant)] > 0.0
        }
        if freq_values:
            noise_db_by_ant[int(ant)] = freq_values

    rows = [per_freq[freq] for freq in sorted(per_freq)]
    return rows, noise_db_by_ant


def normalize_penalties(values_by_freq: dict[int, float], *, reverse: bool = False) -> dict[int, float]:
    if not values_by_freq:
        return {}
    freqs = sorted(values_by_freq)
    values = np.asarray([values_by_freq[freq] for freq in freqs], dtype=np.float64)
    if reverse:
        values = -values
    finite = np.isfinite(values)
    if not np.any(finite):
        return {freq: 0.0 for freq in freqs}
    use = values[finite]
    span = float(np.max(use) - np.min(use))
    if span <= 1.0e-12:
        return {freq: 0.0 for freq in freqs}
    baseline = float(np.min(use))
    penalties = {}
    for freq, value in zip(freqs, values):
        if not np.isfinite(value):
            penalties[int(freq)] = 1.0
        else:
            penalties[int(freq)] = float((value - baseline) / span)
    return penalties


def load_measurement_penalties(
    csv_path: Path,
    *,
    candidate_freqs_khz: list[int],
    channels: list[int],
) -> tuple[dict[int, dict[int, float]], dict[int, dict[int, dict[str, float]]], list[int]]:
    candidate_freqs_hz = [float(freq) * 1.0e3 for freq in candidate_freqs_khz]
    selected_channels = {int(channel) for channel in channels}
    raw_metrics: dict[int, dict[int, dict[str, float]]] = {}
    found_channels: list[int] = []
    with csv_path.open("r", encoding="utf-8", newline="") as src:
        reader = csv.DictReader(src)
        for row in reader:
            channel = int(row["channel"])
            if channel not in selected_channels:
                continue
            source_path = resolve_source_path(csv_path, row["source_file"])
            if source_path is None or not source_path.exists():
                continue
            freq_hz, vswr, rl_db = parse_s1p(source_path)
            found_channels.append(channel)
            channel_metrics: dict[int, dict[str, float]] = {}
            for freq_khz, freq_hz_target in zip(candidate_freqs_khz, candidate_freqs_hz):
                channel_metrics[int(freq_khz)] = {
                    "vswr": float(np.interp(freq_hz_target, freq_hz, vswr)),
                    "rl_db": float(np.interp(freq_hz_target, freq_hz, rl_db)),
                }
            raw_metrics[channel] = channel_metrics

    penalties: dict[int, dict[int, float]] = {}
    for channel, channel_metrics in raw_metrics.items():
        vswr_values = {freq: metrics["vswr"] for freq, metrics in channel_metrics.items()}
        rl_values = {freq: metrics["rl_db"] for freq, metrics in channel_metrics.items()}
        vswr_penalty = normalize_penalties(vswr_values)
        rl_penalty = normalize_penalties(rl_values, reverse=True)
        penalties[channel] = {
            freq: float(0.5 * vswr_penalty[freq] + 0.5 * rl_penalty[freq])
            for freq in channel_metrics
        }
    return penalties, raw_metrics, sorted(set(found_channels))


def summarize_penalty_group(
    penalties_by_channel: dict[int, dict[int, float]],
    *,
    freqs: list[int],
) -> tuple[dict[int, float], dict[int, dict[int, float]]]:
    per_freq: dict[int, list[float]] = {int(freq): [] for freq in freqs}
    detail: dict[int, dict[int, float]] = {int(freq): {} for freq in freqs}
    for channel, penalties in penalties_by_channel.items():
        for freq in freqs:
            if int(freq) not in penalties:
                continue
            per_freq[int(freq)].append(float(penalties[int(freq)]))
            detail[int(freq)][int(channel)] = float(penalties[int(freq)])
    summary = {
        int(freq): float(np.median(values)) if values else 0.0
        for freq, values in per_freq.items()
    }
    return summary, detail


def build_frequency_summary(
    file_summaries: list[dict[str, Any]],
    *,
    requested_rx: list[int],
    requested_tx: list[int],
    measurement_sources_csv: Path | None,
    noise_weight: float,
    tx_match_weight: float,
    rx_match_weight: float,
) -> dict[str, Any]:
    rows, noise_db_by_ant = build_noise_summary(file_summaries)
    freqs = [int(row["freq_khz"]) for row in rows]

    noise_penalties_by_ant = {
        int(ant): normalize_penalties(values_by_freq)
        for ant, values_by_freq in noise_db_by_ant.items()
    }
    noise_score, noise_detail = summarize_penalty_group(noise_penalties_by_ant, freqs=freqs)

    tx_match_score = {int(freq): 0.0 for freq in freqs}
    rx_match_score = {int(freq): 0.0 for freq in freqs}
    tx_match_detail = {int(freq): {} for freq in freqs}
    rx_match_detail = {int(freq): {} for freq in freqs}
    measurement_channels_available: list[int] = []
    score_mode = "noise_only"

    if measurement_sources_csv is not None:
        tx_penalties, tx_metrics, tx_found = load_measurement_penalties(
            measurement_sources_csv,
            candidate_freqs_khz=freqs,
            channels=requested_tx,
        )
        rx_penalties, rx_metrics, rx_found = load_measurement_penalties(
            measurement_sources_csv,
            candidate_freqs_khz=freqs,
            channels=requested_rx,
        )
        measurement_channels_available = sorted(set(tx_found) | set(rx_found))
        if tx_penalties or rx_penalties:
            score_mode = "combined"
            tx_match_score, tx_match_detail = summarize_penalty_group(tx_penalties, freqs=freqs)
            rx_match_score, rx_match_detail = summarize_penalty_group(rx_penalties, freqs=freqs)
        else:
            tx_metrics = {}
            rx_metrics = {}
    else:
        tx_metrics = {}
        rx_metrics = {}

    weight_total = float(max(noise_weight + tx_match_weight + rx_match_weight, 1.0e-9))
    norm_noise_weight = float(noise_weight / weight_total)
    norm_tx_weight = float(tx_match_weight / weight_total)
    norm_rx_weight = float(rx_match_weight / weight_total)

    ranked = []
    for row in rows:
        freq = int(row["freq_khz"])
        row["noise_score"] = float(noise_score.get(freq, 0.0))
        row["tx_match_score"] = float(tx_match_score.get(freq, 0.0))
        row["rx_match_score"] = float(rx_match_score.get(freq, 0.0))
        row["combined_score"] = float(
            norm_noise_weight * row["noise_score"]
            + norm_tx_weight * row["tx_match_score"]
            + norm_rx_weight * row["rx_match_score"]
        )
        row["channel_noise_penalty"] = noise_detail.get(freq, {})
        row["tx_match_penalty"] = tx_match_detail.get(freq, {})
        row["rx_match_penalty"] = rx_match_detail.get(freq, {})
        if tx_metrics:
            row["tx_measurements"] = {
                int(channel): tx_metrics[int(channel)][freq]
                for channel in tx_metrics
                if freq in tx_metrics[int(channel)]
            }
        if rx_metrics:
            row["rx_measurements"] = {
                int(channel): rx_metrics[int(channel)][freq]
                for channel in rx_metrics
                if freq in rx_metrics[int(channel)]
            }
        ranked.append(row)

    ranked.sort(
        key=lambda item: (
            item["combined_score"],
            item["noise_score"],
            item["median_noise_db"],
            item["freq_khz"],
        )
    )
    best = ranked[0] if ranked else None
    return {
        "best_frequency_khz": None if best is None else int(best["freq_khz"]),
        "score_mode": score_mode,
        "weights": {
            "noise_weight": norm_noise_weight,
            "tx_match_weight": norm_tx_weight,
            "rx_match_weight": norm_rx_weight,
        },
        "requested_rx_antennas": requested_rx,
        "requested_tx_antennas": requested_tx,
        "measurement_channels_available": measurement_channels_available,
        "frequencies": ranked,
    }


def write_csv(path: Path, summary: dict[str, Any]) -> None:
    rows = summary["frequencies"]
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "freq_khz",
        "combined_score",
        "noise_score",
        "tx_match_score",
        "rx_match_score",
        "median_noise_power",
        "median_noise_db",
        "inputs",
    ]
    with path.open("w", encoding="utf-8", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "freq_khz": row["freq_khz"],
                    "combined_score": row["combined_score"],
                    "noise_score": row["noise_score"],
                    "tx_match_score": row["tx_match_score"],
                    "rx_match_score": row["rx_match_score"],
                    "median_noise_power": row["median_noise_power"],
                    "median_noise_db": row["median_noise_db"],
                    "inputs": ";".join(row["inputs"]),
                }
            )


def main() -> None:
    args = build_parser().parse_args()
    config, _ = load_config(args.config, args.radar_id, args.borealis_path)
    requested_rx = parse_int_list(args.rx_antennas) if args.rx_antennas else active_main_antennas(config)
    requested_tx = parse_int_list(args.tx_antennas) if args.tx_antennas else active_tx_antennas(config)
    measurement_sources_csv = (
        Path(args.measurement_sources_csv).expanduser().resolve()
        if args.measurement_sources_csv
        else None
    )
    if measurement_sources_csv is not None and not measurement_sources_csv.exists():
        raise FileNotFoundError(f"Measurement source CSV not found: {measurement_sources_csv}")

    inputs = [Path(item).expanduser().resolve() for item in args.inputs]
    file_summaries: list[dict[str, Any]] = []
    for path in inputs:
        if path.suffix.lower() == ".json":
            file_summaries.extend(summaries_from_combined_json(path, rx_antennas=requested_rx))
        else:
            file_summaries.append(median_noise_by_rx(path, records=args.records, rx_antennas=requested_rx))
    summary = build_frequency_summary(
        file_summaries,
        requested_rx=requested_rx,
        requested_tx=requested_tx,
        measurement_sources_csv=measurement_sources_csv,
        noise_weight=args.noise_weight,
        tx_match_weight=args.tx_match_weight,
        rx_match_weight=args.rx_match_weight,
    )

    if args.summary_json:
        output_path = Path(args.summary_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.summary_csv:
        write_csv(Path(args.summary_csv).expanduser().resolve(), summary)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
