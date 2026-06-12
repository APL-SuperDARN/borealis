#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fieldfox_installed_common import (
    DEFAULT_FIELDFOX_HOST,
    DEFAULT_TIMEOUT_SEC,
    MeasurementSettings,
    ChannelSpec,
    ROOT,
    capture_paths_for_channel,
    capture_installed_channel,
    configure_installed_measurement,
    ensure_installed_calibration_on_u,
    normalize_radar_id,
    open_fieldfox_with_fallback,
    render_fieldfox_error_log,
    render_quick_summary,
)


def prompt(text: str, default: str) -> str:
    response = input("%s [%s]: " % (text, default)).strip()
    return response or default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture one installed-path FieldFox measurement set for a single main or interferometer channel."
    )
    parser.add_argument("--radar", help="Radar ID such as WAL.")
    parser.add_argument(
        "--kind",
        choices=["antenna", "interferometer"],
        default="antenna",
        help="Channel kind to measure.",
    )
    parser.add_argument("--channel", type=int, help="Channel number to measure.")
    parser.add_argument("--host", default=DEFAULT_FIELDFOX_HOST, help="FieldFox LAN IP or hostname.")
    parser.add_argument("--port", type=int, help="FieldFox SCPI socket port. If omitted, the script tries common FieldFox ports automatically.")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SEC, help="Socket timeout in seconds.")
    parser.add_argument("--root", type=Path, default=ROOT, help="Wallops workspace root.")
    parser.add_argument("--date-tag", help="Date suffix for output files. Defaults to today's YYYYMMDD.")
    parser.add_argument(
        "--distance-stop-m",
        type=float,
        help="Override the DTF stop distance in meters for this one capture.",
    )
    parser.add_argument(
        "--configure-setup",
        action="store_true",
        help="Configure the installed-path CAT/DTF setup before capturing. Use this when the instrument is not already prepared.",
    )
    parser.add_argument(
        "--preset-first",
        action="store_true",
        help="Apply a full instrument preset before configuring setup. This clears the current setup and calibration context.",
    )
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Do not prompt for channel connection confirmation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    radar_id = normalize_radar_id(args.radar or prompt("Radar ID", "WAL"))
    kind = args.kind
    channel = args.channel
    if channel is None:
        channel = int(prompt("%s number" % ("Antenna" if kind == "antenna" else "Interferometer"), "1"))
    spec = ChannelSpec(kind=kind, number=channel)
    settings = MeasurementSettings()
    planned_paths = capture_paths_for_channel(args.root, radar_id, spec, args.date_tag)
    error_log_path = planned_paths.error_log_path()
    instrument_id = "not connected"
    run_status = "FAILED"
    failure_message = None
    client = None
    result = None

    try:
        client = open_fieldfox_with_fallback(args.host, port=args.port, timeout_sec=args.timeout)
        instrument_id = client.idn()
        print("Connected to FieldFox: %s" % instrument_id)

        if args.configure_setup or args.preset_first:
            distance_stop_m = args.distance_stop_m if args.distance_stop_m is not None else spec.distance_stop_m()
            configure_installed_measurement(
                client,
                settings=settings,
                distance_stop_m=distance_stop_m,
                preset_first=args.preset_first,
            )
            print(
                "Installed-path setup applied for %s with DTF stop distance %s m."
                % (spec.label(), ("%0.1f" % distance_stop_m))
            )
            if not args.no_prompt:
                input(
                    "Run the OSL calibration on the FieldFox now, then press Enter when calibration is complete. "
                )
                status = ensure_installed_calibration_on_u(client, prompt_for_retry=True)
                print("Calibration confirmed: RL %s, DTF %s." % (status.rl_state, status.dtf_state))

        if not args.no_prompt:
            input("Connect %s at the control-room reference plane, then press Enter to capture. " % spec.label())

        result = capture_installed_channel(
            client,
            root=args.root,
            radar_id=radar_id,
            spec=spec,
            settings=settings,
            instrument_id=instrument_id,
            date_tag=args.date_tag,
            distance_stop_m=args.distance_stop_m,
            reassert_distance_stop=not (args.configure_setup or args.preset_first),
        )
        run_status = "SUCCESS"
    except KeyboardInterrupt:
        failure_message = "Capture canceled by user."
        print("\nCapture canceled.")
        return 1
    except Exception as exc:
        failure_message = str(exc)
        print("FieldFox capture failed: %s" % exc, file=sys.stderr)
        return 1
    finally:
        if client is not None:
            client.system_errors(context="end-of-run poll")
            error_events = client.error_events()
            client.close()
        else:
            error_events = []
        if result is not None:
            error_log_path = result.paths.error_log_path()
        error_log_path.write_text(
            render_fieldfox_error_log(
                radar_id=radar_id,
                spec=spec,
                instrument_id=instrument_id,
                host=args.host,
                port=args.port,
                status=run_status,
                error_events=error_events,
                failure_message=failure_message,
            )
        )
        print("FieldFox error log: %s" % error_log_path)

    print(render_quick_summary(result, args.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
