#!/usr/bin/env python3
"""
Compute MCCM qualification metrics from a single antennas_iq capture.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mccm_common import (
    DEFAULT_DIRECT_PATH_TIME_US,
    DEFAULT_NOISE_GUARD_MULTIPLIER,
    DEFAULT_RECORDS_PER_STEP,
    DEFAULT_MIN_CONSECUTIVE,
    DEFAULT_PNR_THRESHOLD_DB,
    DEFAULT_TOP_SEARCH_RESULTS,
    append_csv_row,
    evaluate_file,
    evaluate_file_search,
    parse_float_grid,
    parse_float_list,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QC one antennas_iq capture for Wallops MCCM qualification")
    parser.add_argument("--input", required=True, help="Path to antennas_iq HDF5 file")
    parser.add_argument("--rx-ant", type=int, help="RX main antenna to evaluate. Required if the file contains more than one main RX antenna")
    parser.add_argument("--record", help="Specific record name")
    parser.add_argument(
        "--records",
        type=int,
        default=DEFAULT_RECORDS_PER_STEP,
        help="Use the latest N records if --record omitted; use 0 to evaluate all available records",
    )
    parser.add_argument("--pnr-threshold-db", type=float, default=DEFAULT_PNR_THRESHOLD_DB, help="Minimum pulse-to-noise ratio for an accepted record")
    parser.add_argument("--min-consecutive", type=int, default=DEFAULT_MIN_CONSECUTIVE, help="Minimum consecutive accepted records for detection_pass")
    parser.add_argument("--pulse-window-us", type=float, help="Override the pulse window length in microseconds")
    parser.add_argument(
        "--noise-guard-multiplier",
        type=float,
        default=DEFAULT_NOISE_GUARD_MULTIPLIER,
        help="Exclude this many pulse-window widths on both sides of the pulse when estimating off-pulse noise",
    )
    parser.add_argument(
        "--direct-path-time-us",
        type=float,
        default=DEFAULT_DIRECT_PATH_TIME_US,
        help="Expected direct-path arrival time in microseconds",
    )
    parser.add_argument(
        "--adaptive-direct-path-us",
        help="Adaptive direct-path center grid in microseconds, as start:stop:step or comma list",
    )
    parser.add_argument(
        "--adaptive-pulse-window-us",
        help="Adaptive pulse-window grid in microseconds, as start:stop:step or comma list",
    )
    parser.add_argument(
        "--pulse-offsets-us",
        help="Optional explicit pulse offsets in microseconds; defaults to file pulse_sequence metadata when present",
    )
    parser.add_argument(
        "--top-search-results",
        type=int,
        default=DEFAULT_TOP_SEARCH_RESULTS,
        help="How many top adaptive-window candidates to preserve in the JSON summary",
    )
    parser.add_argument("--summary-json", help="Optional JSON summary path")
    parser.add_argument("--summary-csv", help="Optional CSV path to append a one-row summary")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pulse_offsets_us = (
        parse_float_list(args.pulse_offsets_us) if args.pulse_offsets_us else None
    )
    if args.adaptive_direct_path_us or args.adaptive_pulse_window_us:
        direct_path_times_us = (
            parse_float_grid(args.adaptive_direct_path_us)
            if args.adaptive_direct_path_us
            else [args.direct_path_time_us]
        )
        pulse_window_values_us = (
            parse_float_grid(args.adaptive_pulse_window_us)
            if args.adaptive_pulse_window_us
            else [args.pulse_window_us or DEFAULT_DIRECT_PATH_TIME_US]
        )
        summary = evaluate_file_search(
            args.input,
            rx_ant=args.rx_ant,
            record=args.record,
            records=args.records,
            pnr_threshold_db=args.pnr_threshold_db,
            min_consecutive=args.min_consecutive,
            direct_path_times_us=direct_path_times_us,
            pulse_window_values_us=pulse_window_values_us,
            noise_guard_multiplier=args.noise_guard_multiplier,
            pulse_offsets_us=pulse_offsets_us,
            top_results=args.top_search_results,
        )
    else:
        summary = evaluate_file(
            args.input,
            rx_ant=args.rx_ant,
            record=args.record,
            records=args.records,
            pnr_threshold_db=args.pnr_threshold_db,
            min_consecutive=args.min_consecutive,
            pulse_window_us=args.pulse_window_us,
            direct_path_time_us=args.direct_path_time_us,
            noise_guard_multiplier=args.noise_guard_multiplier,
            pulse_offsets_us=pulse_offsets_us,
        )

    if args.summary_json:
        json_path = Path(args.summary_json).expanduser().resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.summary_csv:
        append_csv_row(args.summary_csv, summary)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
