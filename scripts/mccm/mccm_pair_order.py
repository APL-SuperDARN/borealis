#!/usr/bin/env python3
"""
Rank candidate MCCM receive antennas by physical separation from each transmitter.
"""

from __future__ import annotations

import argparse
import json

from mccm_common import (
    active_main_antennas,
    active_tx_antennas,
    load_config,
    main_locations,
    rank_rx_antennas,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rank MCCM receive antennas by geometric separation")
    parser.add_argument("--config", help="Path to config file. Defaults to <BOREALISPATH>/config/<radar>/<radar>_config.ini")
    parser.add_argument("--borealis-path", help="Path to Borealis repo if --config omitted")
    parser.add_argument("--radar-id", default="wal", help="Radar ID if --config omitted")
    parser.add_argument("--tx-ant", type=int, help="Single TX antenna to rank against. Defaults to all active TX antennas")
    parser.add_argument("--json", action="store_true", help="Emit JSON only")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config, config_path = load_config(args.config, args.radar_id, args.borealis_path)
    active_main = active_main_antennas(config)
    active_tx = active_tx_antennas(config)
    locations = main_locations(config)

    tx_antennas = [args.tx_ant] if args.tx_ant is not None else active_tx
    rankings = []
    for tx_ant in tx_antennas:
        ordered = rank_rx_antennas(tx_ant, active_main, locations)
        rankings.append(
            {
                "tx_ant": int(tx_ant),
                "rx_order": ordered,
            }
        )

    output = {
        "config": str(config_path),
        "active_main_antennas": active_main,
        "rankings": rankings,
    }

    if args.json:
        print(json.dumps(output, indent=2))
        return

    print(json.dumps(output, indent=2))
    for ranking in rankings:
        print(f"TX {ranking['tx_ant']}:")
        for item in ranking["rx_order"]:
            print(f"  RX {item['rx_ant']:>2}  separation={item['distance_m']:.3f} m")


if __name__ == "__main__":
    main()
