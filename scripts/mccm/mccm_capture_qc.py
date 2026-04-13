#!/usr/bin/env python3
"""
Compute MCCM qualification metrics from a single antennas_iq capture.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mccm_common import (
    DEFAULT_MIN_CONSECUTIVE,
    DEFAULT_PNR_THRESHOLD_DB,
    append_csv_row,
    evaluate_file,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QC one antennas_iq capture for Wallops MCCM qualification")
    parser.add_argument("--input", required=True, help="Path to antennas_iq HDF5 file")
    parser.add_argument("--rx-ant", type=int, help="RX main antenna to evaluate. Required if the file contains more than one main RX antenna")
    parser.add_argument("--record", help="Specific record name")
    parser.add_argument("--records", type=int, default=5, help="Use the latest N records if --record omitted")
    parser.add_argument("--pnr-threshold-db", type=float, default=DEFAULT_PNR_THRESHOLD_DB, help="Minimum pulse-to-noise ratio for an accepted record")
    parser.add_argument("--min-consecutive", type=int, default=DEFAULT_MIN_CONSECUTIVE, help="Minimum consecutive accepted records for detection_pass")
    parser.add_argument("--pulse-window-us", type=float, help="Override the pulse window length in microseconds")
    parser.add_argument("--summary-json", help="Optional JSON summary path")
    parser.add_argument("--summary-csv", help="Optional CSV path to append a one-row summary")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = evaluate_file(
        args.input,
        rx_ant=args.rx_ant,
        record=args.record,
        records=args.records,
        pnr_threshold_db=args.pnr_threshold_db,
        min_consecutive=args.min_consecutive,
        pulse_window_us=args.pulse_window_us,
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
