#!/usr/bin/env python3
"""
Generate the safe farthest-to-nearest Wallops MCCM qualification sequence.

This script is intentionally a dry-run command generator. It emits the ordered
steps and the exact Borealis launch commands to use for each scale/pair test,
but it does not automatically restart or stop Borealis services.
"""

from __future__ import annotations

import argparse
import json

from mccm_common import (
    DEFAULT_SCALE_LIST,
    DEFAULT_RECORDS_PER_STEP,
    active_main_antennas,
    active_tx_antennas,
    format_active_reference_cal_command,
    load_config,
    main_locations,
    parse_float_list,
    parse_int_list,
    rank_rx_antennas,
    shlex_join,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the staged Wallops MCCM qualification plan")
    parser.add_argument("--tx-ant", required=True, type=int, help="Main-array TX antenna to qualify")
    parser.add_argument("--freq", required=True, type=int, help="Transmit frequency in kHz")
    parser.add_argument(
        "--scale-list",
        default=",".join(str(scale) for scale in DEFAULT_SCALE_LIST),
        help="Comma-separated list of tx_scale values",
    )
    parser.add_argument(
        "--rx-order",
        default="auto",
        help="auto for farthest-to-nearest ordering, or a comma-separated explicit RX list",
    )
    parser.add_argument(
        "--active-from-config",
        action="store_true",
        help="Restrict candidate RX antennas to currently active config antennas (default behavior for auto order)",
    )
    parser.add_argument(
        "--records-per-step",
        type=int,
        default=DEFAULT_RECORDS_PER_STEP,
        help="Target number of records to QC per step",
    )
    parser.add_argument("--intt", type=int, default=2000, help="Integration time in milliseconds")
    parser.add_argument("--num-ranges", type=int, default=25, help="Number of ranges for active_reference_cal")
    parser.add_argument(
        "--pulse-scheme",
        default="single",
        help="Pulse scheme for active_reference_cal: single, mccm7, mccm8, 7p, 8p, or custom via --pulse-sequence",
    )
    parser.add_argument(
        "--pulse-sequence",
        help="Comma-separated custom pulse sequence in tau-spacing units for active_reference_cal",
    )
    parser.add_argument("--tau-spacing-us", type=int, help="Tau spacing in microseconds for the pulse scheme")
    parser.add_argument("--pulse-len-us", type=int, help="Transmit pulse length in microseconds")
    parser.add_argument("--run-mode", default="release", choices=["release", "debug", "rawrf", "engdebug"], help="Borealis run mode")
    parser.add_argument(
        "--scheduling-mode",
        default="special",
        choices=["common", "discretionary", "special"],
        help="Borealis scheduling mode",
    )
    parser.add_argument("--config", help="Path to config file. Defaults to <BOREALISPATH>/config/<radar>/<radar>_config.ini")
    parser.add_argument("--borealis-path", help="Path to Borealis repo if --config omitted")
    parser.add_argument("--radar-id", default="wal", help="Radar ID if --config omitted")
    parser.add_argument("--json", action="store_true", help="Emit JSON only")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config, config_path = load_config(args.config, args.radar_id, args.borealis_path)
    active_main = active_main_antennas(config)
    active_tx = active_tx_antennas(config)
    locations = main_locations(config)
    scales = parse_float_list(args.scale_list)

    if args.tx_ant not in locations:
        raise ValueError(f"tx-ant {args.tx_ant} not in main antenna location list")
    if args.tx_ant not in active_tx:
        raise ValueError(
            f"tx-ant {args.tx_ant} not in configured TX antenna list {active_tx}"
        )

    if args.rx_order == "auto":
        candidate_rx = active_main
        ordered_rx = rank_rx_antennas(args.tx_ant, candidate_rx, locations)
    else:
        requested_rx = [ant for ant in parse_int_list(args.rx_order) if ant != args.tx_ant]
        ordered_rx = rank_rx_antennas(args.tx_ant, requested_rx, locations)

    pair_plans = []
    for ordered in ordered_rx:
        rx_ant = int(ordered["rx_ant"])
        scale_steps = []
        for step_index, scale in enumerate(scales, start=1):
            command = format_active_reference_cal_command(
                borealis_path=args.borealis_path,
                freq=args.freq,
                tx_ant=args.tx_ant,
                rx_main_antennas=[rx_ant],
                rx_intf_antennas=[],
                tx_scale=scale,
                intt_ms=args.intt,
                num_ranges=args.num_ranges,
                pulse_scheme=args.pulse_scheme,
                pulse_sequence=parse_int_list(args.pulse_sequence) if args.pulse_sequence else None,
                tau_spacing_us=args.tau_spacing_us,
                pulse_len_us=args.pulse_len_us,
                run_mode=args.run_mode,
                scheduling_mode=args.scheduling_mode,
            )
            scale_steps.append(
                {
                    "scale_step": step_index,
                    "tx_scale": scale,
                    "command": command,
                    "command_str": shlex_join(command),
                }
            )
        pair_plans.append(
            {
                "tx_ant": args.tx_ant,
                "rx_ant": rx_ant,
                "distance_m": ordered["distance_m"],
                "records_per_step": args.records_per_step,
                "pulse_scheme": args.pulse_scheme,
                "pulse_sequence": parse_int_list(args.pulse_sequence) if args.pulse_sequence else None,
                "tau_spacing_us": args.tau_spacing_us,
                "pulse_len_us": args.pulse_len_us,
                "stop_when": {
                    "pnr_db_gte": 10.0,
                    "consecutive_records": 3,
                    "agc_status_word": 0,
                },
                "scale_steps": scale_steps,
            }
        )

    output = {
        "config": str(config_path),
        "tx_ant": args.tx_ant,
        "freq_khz": args.freq,
        "active_main_antennas": active_main,
        "active_tx_antennas": active_tx,
        "scale_list": scales,
        "records_per_step": args.records_per_step,
        "pair_plans": pair_plans,
        "global_stop_on": {
            "agc_fault": True,
            "obvious_saturation_signature": True,
            "remote_only_max_tx_scale": max(scales) if scales else None,
        },
    }

    if args.json:
        print(json.dumps(output, indent=2))
        return

    print(json.dumps(output, indent=2))
    if pair_plans:
        first_pair = pair_plans[0]
        print(
            f"First pair from current config ordering: TX {first_pair['tx_ant']} -> RX {first_pair['rx_ant']}"
        )
    for pair_plan in pair_plans:
        print(
            f"TX {pair_plan['tx_ant']} -> RX {pair_plan['rx_ant']} ({pair_plan['distance_m']:.3f} m)"
        )
        for step in pair_plan["scale_steps"]:
            print(f"  scale={step['tx_scale']:.6f}  {step['command_str']}")


if __name__ == "__main__":
    main()
