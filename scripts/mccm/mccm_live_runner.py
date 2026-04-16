#!/usr/bin/env python3
"""
Run live Wallops MCCM collection loops from an interactive terminal.

This script exists because `steamed_hams.py` needs a real TTY. Run this on `wal`
inside an interactive shell or PTY session.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import h5py

from mccm_common import (
    DEFAULT_DIRECT_PATH_TIME_US,
    DEFAULT_MIN_CONSECUTIVE,
    active_tx_antennas,
    active_main_antennas,
    append_csv_row,
    evaluate_file,
    evaluate_file_search,
    load_config,
    main_locations,
    parse_float_grid,
    parse_float_list,
    parse_int_list,
    rank_rx_antennas,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live MCCM runner for Wallops PTY sessions")
    parser.add_argument("--borealis-path", help="Path to Borealis repo")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--radar-id", default="wal", help="Radar ID if --config omitted")
    parser.add_argument("--records", type=int, default=12, help="Records required per capture")
    parser.add_argument("--timeout-sec", type=float, default=180.0, help="Per-capture timeout")
    parser.add_argument("--poll-sec", type=float, default=2.0, help="Polling interval while waiting")
    parser.add_argument(
        "--direct-path-time-us",
        type=float,
        default=300.0,
        help="Fixed direct-path center for QC",
    )
    parser.add_argument(
        "--pnr-threshold-db",
        type=float,
        default=10.0,
        help="Minimum per-record PNR for an accepted record",
    )
    parser.add_argument(
        "--noise-guard-multiplier",
        type=float,
        default=3.0,
        help="Exclude this many pulse-window widths around the pulse when estimating off-pulse noise",
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
        help="Optional explicit pulse offsets in microseconds for QC; defaults to file pulse_sequence metadata when present",
    )
    parser.add_argument(
        "--top-search-results",
        type=int,
        default=5,
        help="How many top adaptive-window candidates to retain in JSON outputs",
    )
    parser.add_argument(
        "--pulse-scheme",
        default="single",
        help="Pulse scheme for active_reference_cal: single, mccm7, mccm8, 7p, 8p, or custom via --pulse-sequence",
    )
    parser.add_argument(
        "--pulse-sequence",
        help="Comma-separated custom pulse sequence in tau-spacing units for active_reference_cal",
    )
    parser.add_argument(
        "--tau-spacing-us",
        type=int,
        help="Tau spacing in microseconds for custom pulse_sequence or to override a named pulse scheme",
    )
    parser.add_argument(
        "--pulse-len-us",
        type=int,
        help="Transmit pulse length in microseconds for active_reference_cal",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    ladder = subparsers.add_parser("ladder", help="Run one-RX-at-a-time MCCM ladder")
    ladder.add_argument("--tx-ant", type=int, required=True)
    ladder.add_argument("--freq", type=int, required=True, help="Fixed session frequency in kHz")
    ladder.add_argument(
        "--scales",
        default="0.010,0.015,0.020,0.030,0.040,0.050,0.070,0.100",
        help="Comma-separated tx_scale ladder",
    )
    ladder.add_argument("--intt", type=int, default=2000, help="Integration time in ms")
    ladder.add_argument("--num-ranges", type=int, default=25)
    ladder.add_argument(
        "--rx-main-antennas",
        help="Comma-separated RX main antennas; defaults to active main antennas excluding tx",
    )
    ladder.add_argument(
        "--rx-order",
        choices=("auto", "given"),
        default="given",
        help="If auto, rank requested RX antennas farthest to nearest from tx",
    )
    ladder.add_argument(
        "--rx-intf-antennas",
        default="",
        help="Comma-separated interferometer antennas to include for every rung",
    )
    ladder.add_argument("--output-dir", required=True, help="Directory for JSON/CSV outputs")
    ladder.add_argument(
        "--stop-on-agc-fault",
        action="store_true",
        help="Abort the ladder if a rung reports agc_status_word != 0",
    )

    tx_sweep = subparsers.add_parser(
        "tx-sweep",
        help="Run a fixed-frequency, fixed-scale MCCM TX sweep against a simultaneous RX subset",
    )
    tx_sweep.add_argument("--freq", type=int, required=True, help="Fixed session frequency in kHz")
    tx_sweep.add_argument("--tx-scale", type=float, required=True, help="Fixed tx_scale for the sweep")
    tx_sweep.add_argument("--intt", type=int, default=2000, help="Integration time in ms")
    tx_sweep.add_argument("--num-ranges", type=int, default=25)
    tx_sweep.add_argument(
        "--tx-main-antennas",
        help="Comma-separated TX main antennas; defaults to active configured TX antennas",
    )
    tx_sweep.add_argument(
        "--rx-main-antennas",
        required=True,
        help="Comma-separated simultaneous RX main antennas for every TX rung",
    )
    tx_sweep.add_argument(
        "--rx-intf-antennas",
        default="",
        help="Comma-separated interferometer antennas to include for every rung",
    )
    tx_sweep.add_argument("--output-dir", required=True, help="Directory for JSON/CSV outputs")
    tx_sweep.add_argument(
        "--stop-on-agc-fault",
        action="store_true",
        help="Abort the sweep if any evaluated RX reports agc_status_word != 0",
    )

    survey = subparsers.add_parser(
        "freq-survey",
        help="Run RX-only narrow-band MCCM frequency survey and choose a session frequency",
    )
    survey.add_argument(
        "--freqs",
        required=True,
        help="Comma-separated candidate frequencies in kHz",
    )
    survey.add_argument("--intt", type=int, default=2000, help="Integration time in ms")
    survey.add_argument("--num-ranges", type=int, default=25)
    survey.add_argument(
        "--rx-main-antennas",
        help="Comma-separated RX main antennas; defaults to active main antennas",
    )
    survey.add_argument(
        "--tx-main-antennas",
        help="Comma-separated TX main antennas used for optional subset-aware frequency scoring; defaults to active TX antennas",
    )
    survey.add_argument(
        "--rx-intf-antennas",
        default="",
        help="Comma-separated interferometer antennas to include in survey captures",
    )
    survey.add_argument("--output-dir", required=True, help="Directory for survey artifacts")
    survey.add_argument(
        "--summary-json",
        help="Optional explicit path for final survey JSON summary; defaults inside output dir",
    )
    survey.add_argument(
        "--summary-csv",
        help="Optional explicit path for final survey CSV summary; defaults inside output dir",
    )
    survey.add_argument(
        "--measurement-sources-csv",
        help="Optional main-array measurement source CSV for combined quietness + match scoring",
    )
    survey.add_argument(
        "--noise-weight",
        type=float,
        default=0.6,
        help="Relative weight for receive-noise score when combined scoring is enabled",
    )
    survey.add_argument(
        "--tx-match-weight",
        type=float,
        default=0.3,
        help="Relative weight for TX-path match score when combined scoring is enabled",
    )
    survey.add_argument(
        "--rx-match-weight",
        type=float,
        default=0.1,
        help="Relative weight for RX-path match score when combined scoring is enabled",
    )

    return parser


def candidate_date_dirs(data_directory: Path, start_time: float) -> list[Path]:
    threshold = datetime.fromtimestamp(start_time) - timedelta(days=1)
    threshold_name = threshold.strftime("%Y%m%d")
    candidates = []
    if not data_directory.exists():
        return candidates
    for child in data_directory.iterdir():
        if child.is_dir() and child.name.isdigit() and len(child.name) == 8 and child.name >= threshold_name:
            candidates.append(child)
    return sorted(candidates)


def latest_antennas_iq_after(data_directory: Path, start_time: float) -> Path | None:
    latest_path: Path | None = None
    latest_mtime = -1.0
    for date_dir in candidate_date_dirs(data_directory, start_time):
        for path in date_dir.glob("*.antennas_iq.h5"):
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            if stat.st_mtime < start_time:
                continue
            if stat.st_mtime > latest_mtime:
                latest_path = path
                latest_mtime = stat.st_mtime
    return latest_path


def count_records(path: Path) -> int:
    with h5py.File(path, "r") as src:
        return len([key for key in src.keys() if isinstance(src[key], h5py.Group) and key != "metadata"])


def wait_for_capture(
    data_directory: Path,
    start_time: float,
    *,
    records: int,
    timeout_sec: float,
    poll_sec: float,
) -> tuple[Path, int]:
    deadline = time.time() + timeout_sec
    latest_path: Path | None = None
    latest_records = 0
    while time.time() < deadline:
        path = latest_antennas_iq_after(data_directory, start_time)
        if path is not None:
            latest_path = path
            try:
                latest_records = count_records(path)
            except OSError:
                latest_records = 0
            if latest_records >= records:
                return path, latest_records
        time.sleep(poll_sec)
    if latest_path is None:
        raise RuntimeError(f"No antennas_iq file appeared in {data_directory} before timeout")
    raise RuntimeError(
        f"Timed out waiting for {records} records; latest file {latest_path} only had {latest_records}"
    )


def stop_borealis_screens() -> None:
    result = subprocess.run(["screen", "-ls"], capture_output=True, text=True, check=False)
    sessions = []
    for line in result.stdout.splitlines():
        if ".borealis" in line:
            sessions.append(line.strip().split()[0])
    for session in sessions:
        subprocess.run(["screen", "-S", session, "-X", "quit"], check=False)
    time.sleep(3.0)


def run_command(command: list[str]) -> None:
    env = os.environ.copy()
    if env.get("TERM", "").strip().lower() in ("", "dumb", "unknown"):
        env["TERM"] = "xterm"
    subprocess.run(command, check=True, env=env)


def steamed_hams_python(base_path: Path) -> Path:
    candidate = base_path / "borealis_env3.11" / "bin" / "python"
    if candidate.exists():
        return candidate
    return Path(sys.executable)


def launch_active_reference(
    *,
    base_path: Path,
    freq: int,
    tx_ant: int,
    tx_scale: float,
    intt: int,
    num_ranges: int,
    rx_main_antennas: list[int],
    rx_intf_antennas: list[int],
    pulse_scheme: str,
    pulse_sequence: str | None,
    tau_spacing_us: int | None,
    pulse_len_us: int | None,
) -> None:
    kwargs = [
        f"freq={int(freq)}",
        f"tx_ant={int(tx_ant)}",
        f"tx_scale={float(tx_scale):.6f}",
        f"intt={int(intt)}",
        f"num_ranges={int(num_ranges)}",
        f"rx_main_antennas={','.join(str(ant) for ant in rx_main_antennas)}",
        f"rx_intf_antennas={','.join(str(ant) for ant in rx_intf_antennas)}",
        f"pulse_scheme={pulse_scheme}",
    ]
    if pulse_sequence:
        kwargs.append(f"pulse_sequence={pulse_sequence}")
    if tau_spacing_us is not None:
        kwargs.append(f"tau_spacing={int(tau_spacing_us)}")
    if pulse_len_us is not None:
        kwargs.append(f"pulse_len={int(pulse_len_us)}")
    command = [
        str(steamed_hams_python(base_path)),
        str(base_path / "scripts" / "steamed_hams.py"),
        "active_reference_cal",
        "release",
        "special",
        "--realtime-off",
        "--kwargs",
        *kwargs,
    ]
    run_command(command)


def launch_noise_survey(
    *,
    base_path: Path,
    freq: int,
    intt: int,
    num_ranges: int,
    rx_main_antennas: list[int],
    rx_intf_antennas: list[int],
) -> None:
    kwargs = [
        f"freq={int(freq)}",
        f"intt={int(intt)}",
        f"num_ranges={int(num_ranges)}",
        f"rx_main_antennas={','.join(str(ant) for ant in rx_main_antennas)}",
        f"rx_intf_antennas={','.join(str(ant) for ant in rx_intf_antennas)}",
    ]
    command = [
        str(steamed_hams_python(base_path)),
        str(base_path / "scripts" / "steamed_hams.py"),
        "mccm_noise_survey",
        "release",
        "special",
        "--realtime-off",
        "--kwargs",
        *kwargs,
    ]
    run_command(command)


def load_summary(summary_path: Path) -> dict[str, Any]:
    return json.loads(summary_path.read_text(encoding="utf-8"))


def ensure_output_dir(path: str | Path) -> Path:
    output_path = Path(path).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def data_directory_from_config(config: dict[str, Any]) -> Path:
    return Path(config["data_directory"]).expanduser().resolve()


def resolve_rx_main_antennas(config: dict[str, Any], value: str | None, *, tx_ant: int | None) -> list[int]:
    if value is not None:
        antennas = parse_int_list(value)
    else:
        antennas = active_main_antennas(config)
    if tx_ant is not None:
        antennas = [ant for ant in antennas if ant != int(tx_ant)]
    return antennas


def resolve_tx_main_antennas(config: dict[str, Any], value: str | None) -> list[int]:
    if value is not None:
        return parse_int_list(value)
    return active_tx_antennas(config)


def evaluate_capture_summary(args: argparse.Namespace, capture_path: Path, rx_ant: int) -> dict[str, Any]:
    pulse_offsets_us = parse_float_list(args.pulse_offsets_us) if args.pulse_offsets_us else None
    if args.adaptive_direct_path_us or args.adaptive_pulse_window_us:
        direct_path_times_us = (
            parse_float_grid(args.adaptive_direct_path_us)
            if args.adaptive_direct_path_us
            else [args.direct_path_time_us]
        )
        pulse_window_values_us = (
            parse_float_grid(args.adaptive_pulse_window_us)
            if args.adaptive_pulse_window_us
            else [DEFAULT_DIRECT_PATH_TIME_US]
        )
        return evaluate_file_search(
            capture_path,
            rx_ant=rx_ant,
            records=args.records,
            pnr_threshold_db=args.pnr_threshold_db,
            min_consecutive=DEFAULT_MIN_CONSECUTIVE,
            direct_path_times_us=direct_path_times_us,
            pulse_window_values_us=pulse_window_values_us,
            noise_guard_multiplier=args.noise_guard_multiplier,
            pulse_offsets_us=pulse_offsets_us,
            top_results=args.top_search_results,
        )
    return evaluate_file(
        capture_path,
        rx_ant=rx_ant,
        records=args.records,
        pnr_threshold_db=args.pnr_threshold_db,
        direct_path_time_us=args.direct_path_time_us,
        noise_guard_multiplier=args.noise_guard_multiplier,
        pulse_offsets_us=pulse_offsets_us,
    )


def combined_intf_rx_ids(config: dict[str, Any], rx_intf_antennas: list[int]) -> list[int]:
    main_count = len(main_locations(config))
    return [main_count + ant for ant in rx_intf_antennas]


def run_ladder(args: argparse.Namespace, config: dict[str, Any], base_path: Path) -> None:
    output_dir = ensure_output_dir(args.output_dir)
    summary_csv = output_dir / "summary.csv"
    if summary_csv.exists():
        summary_csv.unlink()

    rx_main = resolve_rx_main_antennas(config, args.rx_main_antennas, tx_ant=args.tx_ant)
    if args.rx_order == "auto":
        locations = main_locations(config)
        rx_main = [item["rx_ant"] for item in rank_rx_antennas(args.tx_ant, rx_main, locations)]
    rx_intf = parse_int_list(args.rx_intf_antennas)
    rx_targets: list[tuple[str, int, int | None]] = [("main", ant, ant) for ant in rx_main]
    rx_targets.extend(
        ("intf", combined_id, intf_ant)
        for combined_id, intf_ant in zip(combined_intf_rx_ids(config, rx_intf), rx_intf)
    )
    scales = parse_float_list(args.scales)
    data_directory = data_directory_from_config(config)

    manifest = {
        "mode": "ladder",
        "tx_ant": int(args.tx_ant),
        "freq_khz": int(args.freq),
        "scales": scales,
        "records": int(args.records),
        "intt_ms": int(args.intt),
        "num_ranges": int(args.num_ranges),
        "rx_main_antennas": rx_main,
        "rx_intf_antennas": rx_intf,
        "pulse_scheme": args.pulse_scheme,
        "pulse_sequence": args.pulse_sequence,
        "tau_spacing_us": args.tau_spacing_us,
        "pulse_len_us": args.pulse_len_us,
        "adaptive_direct_path_us": args.adaptive_direct_path_us,
        "adaptive_pulse_window_us": args.adaptive_pulse_window_us,
        "pulse_offsets_us": args.pulse_offsets_us,
        "started_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for rx_kind, rx_ant, raw_intf_ant in rx_targets:
        for scale in scales:
            start_time = time.time()
            launch_active_reference(
                base_path=base_path,
                freq=args.freq,
                tx_ant=args.tx_ant,
                tx_scale=scale,
                intt=args.intt,
                num_ranges=args.num_ranges,
                rx_main_antennas=[rx_ant] if rx_kind == "main" else [],
                rx_intf_antennas=[raw_intf_ant] if rx_kind == "intf" and raw_intf_ant is not None else [],
                pulse_scheme=args.pulse_scheme,
                pulse_sequence=args.pulse_sequence,
                tau_spacing_us=args.tau_spacing_us,
                pulse_len_us=args.pulse_len_us,
            )
            try:
                capture_path, records_found = wait_for_capture(
                    data_directory,
                    start_time,
                    records=args.records,
                    timeout_sec=args.timeout_sec,
                    poll_sec=args.poll_sec,
                )
            finally:
                stop_borealis_screens()

            summary_path = output_dir / f"tx{args.tx_ant}_rx{rx_ant}_scale{scale:.3f}.json"
            summary = evaluate_capture_summary(args, capture_path, rx_ant)
            summary["records_found_before_stop"] = int(records_found)
            summary["rx_kind"] = rx_kind
            if raw_intf_ant is not None:
                summary["rx_intf_antenna"] = int(raw_intf_ant)
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            append_csv_row(summary_csv, summary)
            print(
                json.dumps(
                    {
                        "rx_kind": rx_kind,
                        "rx_ant": rx_ant,
                        "tx_scale": scale,
                        "pulse_power": summary["pulse_power"],
                        "noise_power": summary["noise_power"],
                        "pnr_db": summary["pnr_db"],
                        "pnr_margin_db": summary["pnr_margin_db"],
                        "accepted_record_count": summary["accepted_record_count"],
                        "records_evaluated": summary["records_evaluated"],
                        "detection_pass": summary["detection_pass"],
                        "agc_status_word": summary["agc_status_word"],
                    }
                ),
                flush=True,
            )
            if args.stop_on_agc_fault and int(summary["agc_status_word"]) != 0:
                raise RuntimeError(
                    f"Stopping ladder because agc_status_word={summary['agc_status_word']} "
                    f"for rx={rx_ant}, scale={scale}"
                )


def run_freq_survey(args: argparse.Namespace, config: dict[str, Any], base_path: Path) -> None:
    output_dir = ensure_output_dir(args.output_dir)
    data_directory = data_directory_from_config(config)
    rx_main = resolve_rx_main_antennas(config, args.rx_main_antennas, tx_ant=None)
    tx_main = resolve_tx_main_antennas(config, args.tx_main_antennas)
    rx_intf = parse_int_list(args.rx_intf_antennas)
    freqs = [int(freq) for freq in parse_float_list(args.freqs)]

    manifest = {
        "mode": "freq-survey",
        "freqs_khz": freqs,
        "records": int(args.records),
        "intt_ms": int(args.intt),
        "num_ranges": int(args.num_ranges),
        "rx_main_antennas": rx_main,
        "tx_main_antennas": tx_main,
        "rx_intf_antennas": rx_intf,
        "started_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    captured_files: list[Path] = []
    for freq in freqs:
        start_time = time.time()
        launch_noise_survey(
            base_path=base_path,
            freq=freq,
            intt=args.intt,
            num_ranges=args.num_ranges,
            rx_main_antennas=rx_main,
            rx_intf_antennas=rx_intf,
        )
        try:
            capture_path, records_found = wait_for_capture(
                data_directory,
                start_time,
                records=args.records,
                timeout_sec=args.timeout_sec,
                poll_sec=args.poll_sec,
            )
        finally:
            stop_borealis_screens()
        captured_files.append(capture_path)
        metadata = {
            "freq_khz": int(freq),
            "capture_path": str(capture_path),
            "records_found_before_stop": int(records_found),
        }
        (output_dir / f"freq_{freq}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(json.dumps(metadata), flush=True)

    summary_json = (
        Path(args.summary_json).expanduser().resolve()
        if args.summary_json
        else output_dir / "freq_survey_summary.json"
    )
    summary_csv = (
        Path(args.summary_csv).expanduser().resolve()
        if args.summary_csv
        else output_dir / "freq_survey_summary.csv"
    )
    command = [
        str(steamed_hams_python(base_path)),
        str(base_path / "scripts" / "mccm" / "mccm_freq_survey.py"),
        "--records",
        str(args.records),
        "--rx-antennas",
        ",".join(str(ant) for ant in rx_main),
        "--tx-antennas",
        ",".join(str(ant) for ant in tx_main),
        "--summary-json",
        str(summary_json),
        "--summary-csv",
        str(summary_csv),
        "--noise-weight",
        str(args.noise_weight),
        "--tx-match-weight",
        str(args.tx_match_weight),
        "--rx-match-weight",
        str(args.rx_match_weight),
        "--inputs",
        *(str(path) for path in captured_files),
    ]
    if args.measurement_sources_csv:
        command.extend(["--measurement-sources-csv", str(Path(args.measurement_sources_csv).expanduser().resolve())])
    run_command(command)
    summary = load_summary(summary_json)
    print(
        json.dumps(
            {
                "best_frequency_khz": summary["best_frequency_khz"],
                "score_mode": summary.get("score_mode", "noise_only"),
            }
        ),
        flush=True,
    )


def run_tx_sweep(args: argparse.Namespace, config: dict[str, Any], base_path: Path) -> None:
    output_dir = ensure_output_dir(args.output_dir)
    summary_csv = output_dir / "summary.csv"
    if summary_csv.exists():
        summary_csv.unlink()

    tx_main = resolve_tx_main_antennas(config, args.tx_main_antennas)
    rx_main_requested = resolve_rx_main_antennas(config, args.rx_main_antennas, tx_ant=None)
    rx_intf = parse_int_list(args.rx_intf_antennas)
    rx_intf_combined = combined_intf_rx_ids(config, rx_intf)
    data_directory = data_directory_from_config(config)

    manifest = {
        "mode": "tx-sweep",
        "freq_khz": int(args.freq),
        "tx_scale": float(args.tx_scale),
        "records": int(args.records),
        "intt_ms": int(args.intt),
        "num_ranges": int(args.num_ranges),
        "tx_main_antennas": tx_main,
        "rx_main_antennas_requested": rx_main_requested,
        "rx_intf_antennas": rx_intf,
        "rx_intf_combined_ids": rx_intf_combined,
        "pulse_scheme": args.pulse_scheme,
        "pulse_sequence": args.pulse_sequence,
        "tau_spacing_us": args.tau_spacing_us,
        "pulse_len_us": args.pulse_len_us,
        "adaptive_direct_path_us": args.adaptive_direct_path_us,
        "adaptive_pulse_window_us": args.adaptive_pulse_window_us,
        "pulse_offsets_us": args.pulse_offsets_us,
        "started_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for tx_ant in tx_main:
        rx_main = [ant for ant in rx_main_requested if ant != tx_ant]
        if not rx_main:
            continue

        start_time = time.time()
        launch_active_reference(
            base_path=base_path,
            freq=args.freq,
            tx_ant=tx_ant,
            tx_scale=args.tx_scale,
            intt=args.intt,
            num_ranges=args.num_ranges,
            rx_main_antennas=rx_main,
            rx_intf_antennas=rx_intf,
            pulse_scheme=args.pulse_scheme,
            pulse_sequence=args.pulse_sequence,
            tau_spacing_us=args.tau_spacing_us,
            pulse_len_us=args.pulse_len_us,
        )
        try:
            capture_path, records_found = wait_for_capture(
                data_directory,
                start_time,
                records=args.records,
                timeout_sec=args.timeout_sec,
                poll_sec=args.poll_sec,
            )
        finally:
            stop_borealis_screens()

        capture_info = {
            "tx_ant": int(tx_ant),
            "tx_scale": float(args.tx_scale),
            "freq_khz": int(args.freq),
            "capture_path": str(capture_path),
            "records_found_before_stop": int(records_found),
            "rx_main_antennas": rx_main,
            "rx_intf_antennas": rx_intf,
        }
        (output_dir / f"tx{tx_ant}_capture.json").write_text(
            json.dumps(capture_info, indent=2), encoding="utf-8"
        )

        pair_summaries = []
        rx_targets: list[tuple[str, int, int | None]] = [("main", ant, ant) for ant in rx_main]
        rx_targets.extend(
            ("intf", combined_id, intf_ant)
            for combined_id, intf_ant in zip(rx_intf_combined, rx_intf)
        )

        for rx_kind, rx_ant, raw_intf_ant in rx_targets:
            summary = evaluate_capture_summary(args, capture_path, rx_ant)
            summary["records_found_before_stop"] = int(records_found)
            summary["capture_path"] = str(capture_path)
            summary["rx_kind"] = rx_kind
            if raw_intf_ant is not None:
                summary["rx_intf_antenna"] = int(raw_intf_ant)
            summary_path = output_dir / f"tx{tx_ant}_rx{rx_ant}_scale{args.tx_scale:.3f}.json"
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            append_csv_row(summary_csv, summary)
            pair_summaries.append(
                {
                    "rx_kind": rx_kind,
                    "rx_ant": rx_ant,
                    "pulse_power": summary["pulse_power"],
                    "noise_power": summary["noise_power"],
                    "pnr_db": summary["pnr_db"],
                    "pnr_margin_db": summary["pnr_margin_db"],
                    "accepted_record_count": summary["accepted_record_count"],
                    "records_evaluated": summary["records_evaluated"],
                    "detection_pass": summary["detection_pass"],
                    "agc_status_word": summary["agc_status_word"],
                }
            )
            if args.stop_on_agc_fault and int(summary["agc_status_word"]) != 0:
                raise RuntimeError(
                    f"Stopping sweep because agc_status_word={summary['agc_status_word']} "
                    f"for tx={tx_ant}, rx={rx_ant}"
                )

        print(
            json.dumps(
                {
                    "tx_ant": tx_ant,
                    "tx_scale": float(args.tx_scale),
                    "capture_path": str(capture_path),
                    "pair_summaries": pair_summaries,
                }
            ),
            flush=True,
        )


def main() -> None:
    args = build_parser().parse_args()
    config, config_path = load_config(args.config, args.radar_id, args.borealis_path)
    base_path = config_path.parents[2]
    if args.command == "ladder":
        run_ladder(args, config, base_path)
    elif args.command == "freq-survey":
        run_freq_survey(args, config, base_path)
    else:
        run_tx_sweep(args, config, base_path)


if __name__ == "__main__":
    main()
