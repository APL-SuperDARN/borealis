#!/usr/bin/env python3
"""
Aggregate repeated MCCM RX-only survey summaries into long-term channel/frequency trends.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate repeated MCCM noise/frequency survey summaries")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input freq-survey summary JSON files produced by mccm_freq_survey.py",
    )
    parser.add_argument("--summary-json", help="Optional JSON output path")
    parser.add_argument("--summary-csv", help="Optional CSV output path")
    return parser


def load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate(inputs: list[Path]) -> dict[str, Any]:
    per_channel_noise_db: dict[int, list[float]] = {}
    per_freq_scores: dict[int, list[float]] = {}
    best_freq_counts: dict[int, int] = {}
    sessions = []

    for path in inputs:
        summary = load_summary(path)
        sessions.append(
            {
                "input": str(path),
                "best_frequency_khz": summary.get("best_frequency_khz"),
                "score_mode": summary.get("score_mode", "noise_only"),
            }
        )
        best_freq = summary.get("best_frequency_khz")
        if best_freq is not None:
            best_freq_counts[int(best_freq)] = best_freq_counts.get(int(best_freq), 0) + 1
        for freq_row in summary.get("frequencies", []):
            freq = int(freq_row["freq_khz"])
            per_freq_scores.setdefault(freq, []).append(float(freq_row.get("combined_score", freq_row.get("noise_score", 0.0))))
            for channel, noise_power in freq_row.get("channel_noise_power", {}).items():
                if float(noise_power) <= 0.0:
                    continue
                per_channel_noise_db.setdefault(int(channel), []).append(float(10.0 * np.log10(float(noise_power))))

    channel_rows = []
    for channel in sorted(per_channel_noise_db):
        values = np.asarray(per_channel_noise_db[channel], dtype=np.float64)
        channel_rows.append(
            {
                "channel": int(channel),
                "samples": int(values.size),
                "median_noise_db": float(np.median(values)),
                "p10_noise_db": float(np.percentile(values, 10)),
                "p90_noise_db": float(np.percentile(values, 90)),
                "spread_db": float(np.percentile(values, 90) - np.percentile(values, 10)),
            }
        )

    freq_rows = []
    for freq in sorted(per_freq_scores):
        values = np.asarray(per_freq_scores[freq], dtype=np.float64)
        freq_rows.append(
            {
                "freq_khz": int(freq),
                "sessions": int(values.size),
                "median_combined_score": float(np.median(values)),
                "best_count": int(best_freq_counts.get(int(freq), 0)),
            }
        )
    freq_rows.sort(key=lambda row: (row["median_combined_score"], -row["best_count"], row["freq_khz"]))

    return {
        "inputs": [str(path) for path in inputs],
        "session_count": len(inputs),
        "channel_noise_summary": channel_rows,
        "frequency_summary": freq_rows,
        "sessions": sessions,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = build_parser().parse_args()
    inputs = [Path(item).expanduser().resolve() for item in args.inputs]
    summary = aggregate(inputs)

    if args.summary_json:
        json_path = Path(args.summary_json).expanduser().resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.summary_csv:
        csv_path = Path(args.summary_csv).expanduser().resolve()
        write_csv(
            csv_path,
            summary["channel_noise_summary"],
            ["channel", "samples", "median_noise_db", "p10_noise_db", "p90_noise_db", "spread_db"],
        )
        write_csv(
            csv_path.with_name(csv_path.stem + "_freqs.csv"),
            summary["frequency_summary"],
            ["freq_khz", "sessions", "median_combined_score", "best_count"],
        )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
