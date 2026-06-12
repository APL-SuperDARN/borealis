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
    capture_installed_channel,
    configure_installed_measurement,
    ensure_installed_calibration_on_u,
    normalize_radar_id,
    open_fieldfox_with_fallback,
    render_quick_summary,
    set_distance_stop,
    write_batch_outputs,
)


def prompt(text: str, default: str) -> str:
    response = input("%s [%s]: " % (text, default)).strip()
    return response or default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full installed-path control-room FieldFox acquisition for 16 main and 4 interferometer channels."
    )
    parser.add_argument("--radar", help="Radar ID such as WAL.")
    parser.add_argument("--host", default=DEFAULT_FIELDFOX_HOST, help="FieldFox LAN IP or hostname.")
    parser.add_argument("--port", type=int, help="FieldFox SCPI socket port. If omitted, the script tries common FieldFox ports automatically.")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SEC, help="Socket timeout in seconds.")
    parser.add_argument("--root", type=Path, default=ROOT, help="Wallops workspace root.")
    parser.add_argument("--date-tag", help="Date suffix for output files. Defaults to today's YYYYMMDD.")
    parser.add_argument(
        "--no-preset",
        action="store_true",
        help="Do not send a full preset before configuring the installed-path setup.",
    )
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Do not pause for channel reconnection confirmation prompts.",
    )
    return parser.parse_args()


def channel_sequence() -> list:
    ordered = []
    for number in range(1, 17):
        ordered.append(ChannelSpec(kind="antenna", number=number))
    for number in range(1, 5):
        ordered.append(ChannelSpec(kind="interferometer", number=number))
    return ordered


def print_top_findings(results: list) -> None:
    main_results = sorted(
        [item for item in results if item.spec.kind == "antenna"],
        key=lambda item: item.summary.urgency_score,
        reverse=True,
    )
    int_results = sorted(
        [item for item in results if item.spec.kind == "interferometer"],
        key=lambda item: item.summary.urgency_score,
        reverse=True,
    )
    print("")
    print("Main-array first-look priorities:")
    for item in main_results[:5]:
        print(
            "  - %s: %s, urgency %0.1f, VSWR@14 %0.3f, RL@14 %0.2f dB"
            % (
                item.summary.channel,
                item.summary.verdict,
                item.summary.urgency_score,
                item.summary.vswr_14,
                item.summary.rl_14_db,
            )
        )
    if int_results:
        print("")
        print("Interferometer first-look priorities:")
        for item in int_results[:4]:
            print(
                "  - %s: %s, urgency %0.1f, VSWR@14 %0.3f, RL@14 %0.2f dB"
                % (
                    item.summary.channel,
                    item.summary.verdict,
                    item.summary.urgency_score,
                    item.summary.vswr_14,
                    item.summary.rl_14_db,
                )
            )


def main() -> int:
    args = parse_args()
    radar_id = normalize_radar_id(args.radar or prompt("Radar ID", "WAL"))
    settings = MeasurementSettings()
    results = []
    connected_port = args.port

    try:
        with open_fieldfox_with_fallback(args.host, port=args.port, timeout_sec=args.timeout) as client:
            instrument_id = client.idn()
            connected_port = client.port
            print("Connected to FieldFox: %s" % instrument_id)
            print("Verifying Ethernet control path over %s:%d succeeded." % (args.host, connected_port))

            configure_installed_measurement(
                client,
                settings=settings,
                distance_stop_m=ChannelSpec(kind="antenna", number=1).distance_stop_m(),
                preset_first=not args.no_preset,
            )
            print("")
            print("Installed-path setup is ready:")
            print("  - Mode: CAT/TDR")
            print("  - Trace 1: Return Loss over 8-20 MHz")
            print("  - Trace 2: DTF over the 8-20 MHz frequency span")
            print("  - Sweep points: %d" % settings.points)
            print("  - Velocity factor: %0.2f" % settings.velocity_factor)
            print("")
            input(
                "Run the OSL calibration now at the control-room reference plane. "
                "When correction is on and the FieldFox is ready, press Enter to continue. "
            )
            status = ensure_installed_calibration_on_u(client, prompt_for_retry=True)
            print("Calibration confirmed: RL %s, DTF %s." % (status.rl_state, status.dtf_state))

            for spec in channel_sequence():
                desired_stop_m = spec.distance_stop_m()
                if client.distance_stop_m is None or abs(client.distance_stop_m - desired_stop_m) >= 0.05:
                    recalibration_required = set_distance_stop(
                        client,
                        desired_stop_m,
                        settings=settings,
                        allow_calibration_disable=True,
                    )
                    print("")
                    print(
                        "DTF stop distance updated to %0.1f m for %s."
                        % (desired_stop_m, spec.noun().lower() + " channels")
                    )
                    if recalibration_required:
                        if args.no_prompt:
                            raise RuntimeError(
                                "Changing to the %0.1f m DTF stop distance disabled correction. "
                                "Re-run without --no-prompt so the script can pause for recalibration."
                                % desired_stop_m
                            )
                        input(
                            "Run the OSL calibration again for this DTF stop-distance setup, "
                            "then press Enter to re-check correction. "
                        )
                        status = ensure_installed_calibration_on_u(client, prompt_for_retry=True)
                        print("Calibration confirmed: RL %s, DTF %s." % (status.rl_state, status.dtf_state))
                if not args.no_prompt:
                    input("Connect %s, then press Enter to capture. " % spec.label())
                result = capture_installed_channel(
                    client,
                    root=args.root,
                    radar_id=radar_id,
                    spec=spec,
                    settings=settings,
                    instrument_id=instrument_id,
                    date_tag=args.date_tag,
                    reassert_distance_stop=False,
                )
                results.append(result)
                print("")
                print(render_quick_summary(result, args.root))
                print("")

    except KeyboardInterrupt:
        print("\nBatch run interrupted.")
        if not results:
            return 1
    except Exception as exc:
        print("Batch capture failed: %s" % exc, file=sys.stderr)
        return 1

    if not results:
        print("No channel captures were completed.", file=sys.stderr)
        return 1

    outputs = write_batch_outputs(
        root=args.root,
        radar_id=radar_id,
        results=results,
        instrument_id="Keysight FieldFox via %s:%d" % (args.host, connected_port),
        date_tag=args.date_tag,
    )

    print("Batch summary written to %s" % outputs["report_dir"].relative_to(args.root))
    print_top_findings(results)
    print("")
    print("Generated summary artifacts:")
    for key in (
        "report_txt",
        "main_csv",
        "int_csv",
        "main_vswr_plot",
        "main_rl_plot",
        "main_dtf_plot",
        "main_rank_plot",
        "int_vswr_plot",
        "int_rl_plot",
        "int_dtf_plot",
        "int_rank_plot",
    ):
        path = outputs.get(key)
        if path is not None and path.exists():
            print("  - %s" % path.relative_to(args.root))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
