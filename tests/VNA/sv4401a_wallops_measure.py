#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import platform
import re
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

try:
    import serial  # type: ignore
    from serial.tools import list_ports  # type: ignore
except ImportError as exc:
    raise SystemExit("This script requires pyserial. Install with: python3 -m pip install pyserial") from exc

DEFAULT_BAUD = 115200
DEFAULT_TIMEOUT = 2.0
DEFAULT_SWEEP_POINTS = 801
DEFAULT_START_HZ = 8_000_000
DEFAULT_STOP_HZ = 20_000_000
DEFAULT_SWEEP_SETTLE_SEC = 1.0
DEFAULT_OUTPUT_ROOT = Path("wallops_2026_04/sv4401a_runs")
MAIN_CHANNELS = [f"ANT{index}" for index in range(1, 17)]
INTERFEROMETER_CHANNELS = [f"INT{index}" for index in range(1, 5)]


@dataclass
class SweepData:
    frequencies_hz: List[float]
    s11_complex: List[complex]


class SV4401A:
    def __init__(self, port: str, baud: int = DEFAULT_BAUD, timeout: float = DEFAULT_TIMEOUT) -> None:
        self.serial = serial.Serial(port=port, baudrate=baud, timeout=timeout)
        time.sleep(0.2)
        self.serial.reset_input_buffer()
        self.serial.reset_output_buffer()

    def close(self) -> None:
        self.serial.close()

    def command(self, text: str, wait_sec: float = 0.15) -> List[str]:
        self.serial.write((text + "\r").encode("ascii"))
        self.serial.flush()
        time.sleep(wait_sec)
        return self._read_lines()

    def _read_lines(self) -> List[str]:
        timeout = self.serial.timeout or DEFAULT_TIMEOUT
        deadline = time.time() + timeout
        chunks: List[bytes] = []
        while time.time() < deadline:
            waiting = self.serial.in_waiting
            if waiting:
                chunks.append(self.serial.read(waiting))
                deadline = time.time() + timeout
            else:
                time.sleep(0.05)
        blob = b"".join(chunks).decode("utf-8", errors="replace")
        return [line.strip() for line in blob.replace("\r", "\n").split("\n") if line.strip()]

    def info(self) -> str:
        lines = self.command("info")
        return " | ".join(lines) if lines else "(no info response)"

    def version(self) -> str:
        lines = self.command("version")
        return " | ".join(lines) if lines else "(no version response)"

    def sweep_status(self) -> str:
        lines = self.command("sweep")
        return " | ".join(lines) if lines else "(no sweep response)"

    def set_sweep(self, start_hz: int, stop_hz: int, points: int = DEFAULT_SWEEP_POINTS) -> None:
        self.command(f"sweep {start_hz} {stop_hz} {points}")

    def frequencies(self) -> List[float]:
        return _parse_float_lines(self.command("frequencies", wait_sec=0.4))

    def s11_data(self) -> List[complex]:
        lines = self.command("data 0", wait_sec=0.4)
        values: List[complex] = []
        for line in lines:
            numbers = _numbers_from_line(line)
            if len(numbers) >= 2:
                values.append(complex(numbers[0], numbers[1]))
        return values

    def cal(self, step: str) -> List[str]:
        return self.command(f"cal {step}", wait_sec=0.8 if step == "done" else 0.4)


def _numbers_from_line(line: str) -> List[float]:
    found = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
    return [float(item) for item in found]


def _parse_float_lines(lines: Iterable[str]) -> List[float]:
    values: List[float] = []
    for line in lines:
        nums = _numbers_from_line(line)
        if len(nums) == 1:
            values.append(nums[0])
        elif len(nums) > 1:
            values.extend(nums)
    return values


def return_loss_db(gamma_mag: float) -> float:
    if gamma_mag <= 0:
        return float("inf")
    return -20.0 * math.log10(gamma_mag)


def vswr_from_gamma(gamma_mag: float) -> float:
    if gamma_mag >= 1:
        return float("inf")
    return (1 + gamma_mag) / (1 - gamma_mag)


def capture_s11_sweep(device: SV4401A, start_hz: int, stop_hz: int, points: int, settle_sec: float) -> SweepData:
    device.set_sweep(start_hz, stop_hz, points)
    time.sleep(settle_sec)
    frequencies = device.frequencies()
    s11 = device.s11_data()
    if len(frequencies) != len(s11):
        size = min(len(frequencies), len(s11))
        frequencies = frequencies[:size]
        s11 = s11[:size]
    if not frequencies or not s11:
        raise RuntimeError("SV4401A returned no sweep data")
    return SweepData(frequencies_hz=frequencies, s11_complex=s11)


def summarize_sweep(sweep: SweepData) -> dict:
    mags = [abs(value) for value in sweep.s11_complex]
    rl = [return_loss_db(mag) for mag in mags]
    vswr = [vswr_from_gamma(mag) for mag in mags]
    finite_vswr = [value for value in vswr if math.isfinite(value)]
    best_idx = max(range(len(rl)), key=lambda idx: rl[idx])
    return {
        "best_return_loss_db": rl[best_idx],
        "best_return_loss_freq_hz": sweep.frequencies_hz[best_idx],
        "min_vswr": min(finite_vswr) if finite_vswr else None,
        "median_vswr": statistics.median(finite_vswr) if finite_vswr else None,
        "median_return_loss_db": statistics.median(rl),
    }


def write_s11_csv(path: Path, sweep: SweepData) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frequency_hz", "s11_real", "s11_imag", "s11_mag", "return_loss_db", "vswr"])
        for freq, value in zip(sweep.frequencies_hz, sweep.s11_complex):
            mag = abs(value)
            writer.writerow([freq, value.real, value.imag, mag, return_loss_db(mag), vswr_from_gamma(mag)])


def write_touchstone_s1p(path: Path, sweep: SweepData) -> None:
    with path.open("w") as handle:
        handle.write("# Hz S RI R 50\n")
        for freq, value in zip(sweep.frequencies_hz, sweep.s11_complex):
            handle.write(f"{freq:.0f} {value.real:.9e} {value.imag:.9e}\n")


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def normalize_channels(selection: str) -> List[str]:
    lowered = selection.strip().lower()
    if lowered == "main":
        return MAIN_CHANNELS
    if lowered == "interferometer":
        return INTERFEROMETER_CHANNELS
    if lowered == "all":
        return MAIN_CHANNELS + INTERFEROMETER_CHANNELS
    return [item.strip().upper() for item in selection.split(",") if item.strip()]


def choose_default_port() -> str | None:
    ports = list(list_ports.comports())
    if not ports:
        return None
    preferred = []
    for port in ports:
        device = (port.device or "").lower()
        description = (port.description or "").lower()
        if any(token in device for token in ("ttyusb", "ttyacm", "cu.usb", "tty.usb")) or "usb" in description:
            preferred.append(port.device)
    return preferred[0] if preferred else ports[0].device


def print_port_help() -> None:
    system_name = platform.system()
    print("Detected serial ports:")
    ports = list(list_ports.comports())
    if not ports:
        print("  (none found)")
    for port in ports:
        print(f"  {port.device} - {port.description}")
    if system_name == "Darwin":
        print("Typical macOS port names look like /dev/cu.usbmodem* or /dev/cu.usbserial*.")
    elif system_name == "Linux":
        print("Typical Ubuntu port names look like /dev/ttyACM0 or /dev/ttyUSB0.")


def prompt_enter(message: str) -> None:
    input(message + "\nPress Enter when ready: ")


def prompt_text(message: str) -> str:
    return input(message).strip()


def run_guided_calibration(device: SV4401A) -> dict:
    print("\n=== Calibration setup ===")
    print("Set the reference plane at the end of the jumper/adapter stack you will use for the channel measurements.")
    prompt_enter("Disconnect the antenna/channel and connect the calibration OPEN standard at the reference plane.")
    open_response = device.cal("open")
    prompt_enter("Replace OPEN with the SHORT standard at the same reference plane.")
    short_response = device.cal("short")
    prompt_enter("Replace SHORT with the LOAD standard at the same reference plane.")
    load_response = device.cal("load")
    prompt_enter("Leave the setup undisturbed. The script will now finish and enable calibration.")
    done_response = device.cal("done")
    on_response = device.cal("on")
    return {
        "open_response": open_response,
        "short_response": short_response,
        "load_response": load_response,
        "done_response": done_response,
        "on_response": on_response,
    }


def verify_calibration(device: SV4401A, start_hz: int, stop_hz: int, points: int, settle_sec: float) -> dict:
    print("\n=== Calibration check ===")
    print("Best available automatic check: measure the LOAD still connected at the reference plane.")
    prompt_enter("Confirm the 50 ohm LOAD is still connected for the calibration check.")
    sweep = capture_s11_sweep(device, start_hz, stop_hz, points, settle_sec)
    summary = summarize_sweep(sweep)
    passed = False
    best_rl = summary.get("best_return_loss_db")
    min_vswr = summary.get("min_vswr")
    if isinstance(best_rl, (int, float)) and isinstance(min_vswr, (int, float)):
        passed = best_rl >= 20.0 and min_vswr <= 1.25
    print(f"Calibration check summary: best RL={best_rl:.2f} dB, min VSWR={min_vswr:.3f}")
    print("Calibration check result: " + ("PASS" if passed else "CHECK MANUALLY"))
    return {"summary": summary, "pass": passed}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Guided SV4401A Wallops installed-path measurement script.")
    parser.add_argument("--port", help="Serial port. If omitted, the script tries to pick one and shows what it found.")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--channels", default="all", help="all, main, interferometer, or ANT1,ANT2,INT1")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--start-hz", type=int, default=DEFAULT_START_HZ)
    parser.add_argument("--stop-hz", type=int, default=DEFAULT_STOP_HZ)
    parser.add_argument("--points", type=int, default=DEFAULT_SWEEP_POINTS)
    parser.add_argument("--settle-sec", type=float, default=DEFAULT_SWEEP_SETTLE_SEC)
    parser.add_argument("--skip-calibration", action="store_true", help="Skip the guided OSL calibration steps.")
    parser.add_argument("--notes", help="Optional session note.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    port = args.port or choose_default_port()
    print_port_help()
    if not port:
        print("No serial port could be selected automatically. Re-run with --port.", file=sys.stderr)
        return 2
    print(f"\nUsing serial port: {port}")

    channels = normalize_channels(args.channels)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_root = args.output_root / stamp
    session_root.mkdir(parents=True, exist_ok=True)

    device = SV4401A(port, baud=args.baud)
    try:
        instrument_info = device.info()
        instrument_version = device.version()
        sweep_status = device.sweep_status()
        print(f"Connected instrument: {instrument_info}")
        print(f"Firmware: {instrument_version}")
        print(f"Current sweep state: {sweep_status}")

        session_summary: dict = {
            "timestamp": stamp,
            "platform": platform.platform(),
            "serial_port": port,
            "instrument_info": instrument_info,
            "instrument_version": instrument_version,
            "initial_sweep_state": sweep_status,
            "channels": channels,
            "start_hz": args.start_hz,
            "stop_hz": args.stop_hz,
            "points": args.points,
            "notes": args.notes,
            "results": [],
        }

        if not args.skip_calibration:
            calibration = run_guided_calibration(device)
            calibration_check = verify_calibration(device, args.start_hz, args.stop_hz, args.points, args.settle_sec)
            session_summary["calibration"] = calibration
            session_summary["calibration_check"] = calibration_check
            prompt_enter("Disconnect the calibration load. Prepare to connect channel 1.")
        else:
            print("\nSkipping calibration at user request.")

        for channel in channels:
            print(f"\n=== {channel} ===")
            prompt_enter(f"Connect {channel} to the SV4401A at the calibrated reference plane.")
            channel_dir = session_root / channel.lower()
            channel_dir.mkdir(parents=True, exist_ok=True)
            sweep = capture_s11_sweep(device, args.start_hz, args.stop_hz, args.points, args.settle_sec)
            summary = summarize_sweep(sweep)
            write_s11_csv(channel_dir / f"{channel.lower()}_installed_s11.csv", sweep)
            write_touchstone_s1p(channel_dir / f"{channel.lower()}_installed.s1p", sweep)
            save_json(channel_dir / f"{channel.lower()}_installed_summary.json", {"channel": channel, "summary": summary})
            print(
                "Saved measurement for %s: best RL %.2f dB at %.3f MHz, min VSWR %.3f"
                % (
                    channel,
                    summary["best_return_loss_db"],
                    summary["best_return_loss_freq_hz"] / 1e6,
                    summary["min_vswr"],
                )
            )
            note = prompt_text("Optional quick note for this channel (press Enter to skip): ")
            record = {"channel": channel, "summary": summary}
            if note:
                (channel_dir / "note.txt").write_text(note + "\n")
                record["note"] = note
            session_summary["results"].append(record)

        save_json(session_root / "session_summary.json", session_summary)
        print(f"\nSession saved under {session_root}")
        return 0
    finally:
        device.close()


if __name__ == "__main__":
    raise SystemExit(main())
