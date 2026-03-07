#!/usr/bin/env python3
"""
Near-realtime runner for A/B imaging products.

Watches antennas_iq records, periodically updates calibration, and appends
per-record conventional/Capon/MUSIC outputs to a rolling HDF5 product.
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"files": {}}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {"files": {}}


def _save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True))


def _sorted_records(input_file: Path) -> list[str]:
    with h5py.File(input_file, "r") as f:
        recs = sorted([k for k in f.keys() if k != "metadata"])
    return recs


def _compute_calibration_metrics(cal_path: Path) -> dict[str, float]:
    d = np.load(cal_path)
    g = d["channel_gains"]
    amps = np.abs(g)
    phases = np.angle(g, deg=True)
    return {
        "channels": int(g.shape[0]),
        "amp_min": float(amps.min()),
        "amp_med": float(np.median(amps)),
        "amp_max": float(amps.max()),
        "amp_spread_max_over_min": float(amps.max() / max(amps.min(), 1.0e-12)),
        "phase_std_deg": float(np.std(phases)),
    }


def _run_cmd(cmd: list[str]) -> None:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            f"Command failed ({p.returncode}): {' '.join(cmd)}\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}"
        )


def _ensure_calibration(args: argparse.Namespace, input_file: Path, cal_path: Path) -> dict[str, float]:
    cmd = [
        str(args.python),
        str(args.selfcal_script),
        "--input",
        str(input_file),
        "--records",
        str(args.cal_records),
        "--array",
        args.cal_array,
        "--mode",
        args.cal_mode,
        "--output",
        str(cal_path),
    ]
    if args.cal_reference_antenna is not None:
        cmd += ["--reference-antenna", str(args.cal_reference_antenna)]
    if args.cal_known_az_deg is not None:
        cmd += ["--known-az-deg", str(args.cal_known_az_deg)]

    _run_cmd(cmd)
    return _compute_calibration_metrics(cal_path)


def _merge_record_from_tmp(tmp_file: Path, out_file: Path, record: str) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(tmp_file, "r") as src, h5py.File(out_file, "a") as dst:
        if "metadata" not in dst:
            dst_meta = dst.create_group("metadata")
            for k, v in src["metadata"].attrs.items():
                dst_meta.attrs[k] = v

        if record in dst:
            del dst[record]

        src.copy(src[record], dst, name=record)


def _process_record(
    args: argparse.Namespace,
    input_file: Path,
    output_file: Path,
    record: str,
    cal_path: Path | None,
) -> None:
    with tempfile.TemporaryDirectory(prefix="imaging_ab_") as td:
        tmp_file = Path(td) / "single_record.h5"

        cmd = [
            str(args.python),
            str(args.imaging_script),
            "--input",
            str(input_file),
            "--record",
            record,
            "--method",
            args.method,
            "--az-min",
            str(args.az_min),
            "--az-max",
            str(args.az_max),
            "--az-step",
            str(args.az_step),
            "--range-start-km",
            str(args.range_start_km),
            "--range-stop-km",
            str(args.range_stop_km),
            "--range-step-km",
            str(args.range_step_km),
            "--window-samples",
            str(args.window_samples),
            "--diag-loading",
            str(args.diag_loading),
            "--model-order",
            str(args.model_order),
            "--output",
            str(tmp_file),
        ]
        if cal_path is not None:
            cmd += ["--calibration", str(cal_path)]

        _run_cmd(cmd)
        _merge_record_from_tmp(tmp_file, output_file, record)


def _select_input_file(args: argparse.Namespace) -> Path:
    if args.input_file:
        p = Path(args.input_file).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(p)
        return p

    candidates = sorted(glob.glob(args.input_glob))
    if not candidates:
        raise FileNotFoundError(f"No files matched --input-glob {args.input_glob}")
    return Path(candidates[-1]).resolve()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Near-realtime A/B imaging runner")

    p.add_argument("--python", default="./borealis_env3.11/bin/python3")
    p.add_argument("--imaging-script", default="./scripts/imaging_ab_compare.py")
    p.add_argument("--selfcal-script", default="./scripts/self_calibrate_array.py")

    p.add_argument("--input-file", help="Single antennas_iq file to watch")
    p.add_argument(
        "--input-glob",
        default="/data/borealis_data/*/*.antennas_iq.h5",
        help="Glob for latest antennas_iq file when --input-file is omitted",
    )
    p.add_argument("--output", required=True, help="Rolling output HDF5")
    p.add_argument("--state", required=True, help="State JSON path")

    p.add_argument("--method", choices=["capon", "music", "both"], default="both")
    p.add_argument("--az-min", type=float, default=-35.0)
    p.add_argument("--az-max", type=float, default=35.0)
    p.add_argument("--az-step", type=float, default=0.5)
    p.add_argument("--range-start-km", type=float, default=180.0)
    p.add_argument("--range-stop-km", type=float, default=400.0)
    p.add_argument("--range-step-km", type=float, default=2.0)
    p.add_argument("--window-samples", type=int, default=5)
    p.add_argument("--diag-loading", type=float, default=1.0e-2)
    p.add_argument("--model-order", type=int, default=1)

    p.add_argument("--use-calibration", action="store_true")
    p.add_argument("--calibration-file", default="./tmp/realtime_selfcal.npz")
    p.add_argument("--cal-records", type=int, default=5)
    p.add_argument("--cal-array", choices=["main", "intf", "all"], default="main")
    p.add_argument("--cal-mode", choices=["crossref", "dominant"], default="dominant")
    p.add_argument("--cal-reference-antenna", type=int)
    p.add_argument("--cal-known-az-deg", type=float)
    p.add_argument("--cal-every-records", type=int, default=20)

    p.add_argument("--poll-seconds", type=float, default=2.0)
    p.add_argument("--max-records-per-cycle", type=int, default=10)
    p.add_argument("--run-once", action="store_true")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    def _abs_no_resolve(p: str) -> Path:
        q = Path(p).expanduser()
        if not q.is_absolute():
            q = Path.cwd() / q
        return q

    args.python = _abs_no_resolve(args.python)
    args.imaging_script = _abs_no_resolve(args.imaging_script)
    args.selfcal_script = _abs_no_resolve(args.selfcal_script)
    output_file = Path(args.output).expanduser().resolve()
    state_path = Path(args.state).expanduser().resolve()
    cal_path = Path(args.calibration_file).expanduser().resolve()

    state = _load_state(state_path)
    if "files" not in state:
        state["files"] = {}

    processed_since_cal = 0

    while True:
        input_file = _select_input_file(args)
        key = str(input_file)
        if key not in state["files"]:
            state["files"][key] = {"processed_records": []}

        processed = set(state["files"][key].get("processed_records", []))
        records = _sorted_records(input_file)
        new_records = [r for r in records if r not in processed]
        if args.max_records_per_cycle > 0:
            new_records = new_records[: args.max_records_per_cycle]

        if new_records:
            cal_metrics = None
            if args.use_calibration and (
                (not cal_path.exists())
                or (processed_since_cal >= args.cal_every_records)
            ):
                cal_metrics = _ensure_calibration(args, input_file, cal_path)
                processed_since_cal = 0

            for rec in new_records:
                _process_record(
                    args,
                    input_file,
                    output_file,
                    rec,
                    cal_path if args.use_calibration else None,
                )
                processed.add(rec)
                processed_since_cal += 1

            state["files"][key]["processed_records"] = sorted(processed)
            state["files"][key]["last_seen_record"] = records[-1]
            state["files"][key]["output_file"] = str(output_file)
            if cal_metrics is not None:
                state["files"][key]["last_calibration_metrics"] = cal_metrics
                state["files"][key]["calibration_file"] = str(cal_path)
            _save_state(state_path, state)

            print(
                json.dumps(
                    {
                        "status": "processed",
                        "input_file": str(input_file),
                        "new_records": new_records,
                        "output": str(output_file),
                        "calibration": str(cal_path) if args.use_calibration else None,
                        "calibration_metrics": cal_metrics,
                    },
                    indent=2,
                )
            )
        else:
            print(
                json.dumps(
                    {
                        "status": "idle",
                        "input_file": str(input_file),
                        "known_records": len(records),
                    }
                )
            )

        if args.run_once:
            return

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
