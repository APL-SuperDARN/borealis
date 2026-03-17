#!/usr/bin/env python3
"""Reset N200 USRPs listed in the Borealis radar config using UHD image loader."""

import argparse
import json
import pathlib
import shlex
import subprocess
import sys


def parse_args():
    p = argparse.ArgumentParser(description="Reset USRPs via uhd_image_loader")
    p.add_argument(
        "--config",
        default="/home/radar/borealis/config/wal/wal_config.ini",
        help="Path to radar config JSON file (default: wal)",
    )
    p.add_argument(
        "--timeout-sec",
        type=int,
        default=8,
        help="Per-device timeout for uhd_image_loader",
    )
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue attempting all devices even if one fails",
    )
    return p.parse_args()


def get_ips(config_path):
    cfg = json.loads(config_path.read_text())
    ips = [d.get("addr") for d in cfg.get("n200s", []) if d.get("addr")]
    uniq = []
    seen = set()
    for ip in ips:
        if ip not in seen:
            seen.add(ip)
            uniq.append(ip)
    return uniq


def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = p.stdout.strip()
    return p.returncode, out


def main():
    args = parse_args()
    config = pathlib.Path(args.config)
    if not config.exists():
        print("ERROR: config not found: {}".format(config))
        return 2

    ips = get_ips(config)
    if not ips:
        print("ERROR: no n200 addresses found in config")
        return 2

    print("USRP reset target list:")
    print("  " + " ".join(ips))

    failures = []
    for ip in ips:
        cmd = [
            "timeout",
            "{}s".format(args.timeout_sec),
            "uhd_image_loader",
            "--args=type=usrp2,addr={},reset".format(ip),
            "--no-fw",
            "--no-fpga",
        ]
        rc, out = run(cmd)
        print("{}: rc={} cmd={}".format(ip, rc, shlex.join(cmd)))
        if out:
            print("  " + out.replace("\n", "\n  "))
        if rc != 0:
            failures.append(ip)
            if not args.continue_on_error:
                print("Stopping due to failure (use --continue-on-error to keep going).")
                break

    if failures:
        print("FAILED IPs: " + " ".join(failures))
        return 1

    print("All reset commands completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
