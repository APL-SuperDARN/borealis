#!/usr/bin/env python3
"""
Self-calibration helper for SuperDARN array channels from antennas_iq captures.

Two calibration modes are implemented:
- crossref: ratio against a chosen reference antenna channel
- dominant: dominant-eigenvector calibration from covariance matrix

The output npz can be used by scripts/imaging_ab_compare.py via --calibration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import os

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py
import numpy as np
from scipy.constants import speed_of_light


def _steering_vector(az_deg: float, antenna_x_m: np.ndarray, freq_khz: float) -> np.ndarray:
    k = 2.0 * np.pi * (freq_khz * 1.0e3) / speed_of_light
    return np.exp(1j * k * (-np.sin(np.deg2rad(az_deg))) * antenna_x_m).astype(np.complex64)


def _pick_records(src: h5py.File, record: str | None, records: int) -> list[str]:
    all_records = sorted([k for k in src.keys() if k != "metadata"])
    if not all_records:
        raise RuntimeError("No records in file")

    if record:
        if record not in all_records:
            raise ValueError(f"Record {record} not in file")
        return [record]

    n = min(max(records, 1), len(all_records))
    return all_records[-n:]


def _extract_array_data(
    group: h5py.Group,
    array: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    ant_iq = group["antennas_iq_data"][...].astype(np.complex64)  # [ant, seq, time]
    rx_antennas = group["rx_antennas"][...].astype(np.int32)
    ant_locs = group["antenna_locations"][...].astype(np.float32)
    freq_khz = float(group["freq"][()])

    if array == "main":
        want = group["rx_main_antennas"][...].astype(np.int32)
    elif array == "intf":
        want = group["rx_intf_antennas"][...].astype(np.int32)
    else:
        want = rx_antennas

    mask = np.isin(rx_antennas, want)
    if not np.any(mask):
        raise RuntimeError(f"No channels found for array={array}")

    data = ant_iq[mask, :, :]
    ids = rx_antennas[mask]
    x_m = ant_locs[ids, 0]

    order = np.argsort(ids)
    return data[order], ids[order], x_m[order], freq_khz


def _estimate_crossref(x: np.ndarray, ref_idx: int) -> np.ndarray:
    # x: [channels, snapshots]
    x_ref = x[ref_idx]
    denom = np.mean(np.abs(x_ref) ** 2)
    if denom <= 1.0e-18:
        raise RuntimeError("Reference channel has near-zero power")

    gains = np.mean(x * np.conjugate(x_ref)[np.newaxis, :], axis=1) / denom
    gains = gains.astype(np.complex64)
    return gains


def _estimate_dominant(x: np.ndarray) -> np.ndarray:
    # x: [channels, snapshots]
    r = (x @ x.conj().T) / np.float32(max(x.shape[1], 1))
    eigvals, eigvecs = np.linalg.eigh(r)
    v = eigvecs[:, np.argmax(eigvals)]
    return v.astype(np.complex64)


def run(args: argparse.Namespace) -> Path:
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        output_path = input_path.with_suffix("").with_suffix("")
        output_path = output_path.with_name(output_path.name + f".{args.array}.selfcal.npz")

    with h5py.File(input_path, "r") as src:
        selected = _pick_records(src, args.record, args.records)

        snaps = []
        array_ids = None
        array_x = None
        freq_vals = []

        for rec in selected:
            d, ids, x_m, freq_khz = _extract_array_data(src[rec], args.array)
            if array_ids is None:
                array_ids = ids
                array_x = x_m
            elif not np.array_equal(ids, array_ids):
                raise RuntimeError("Antenna set changed across selected records")

            snaps.append(d.reshape(d.shape[0], -1))
            freq_vals.append(freq_khz)

    x = np.concatenate(snaps, axis=1)
    if x.shape[1] == 0:
        raise RuntimeError("No snapshots available")

    ref_idx = 0
    ref_ant = int(array_ids[0])
    if args.reference_antenna is not None:
        matches = np.where(array_ids == int(args.reference_antenna))[0]
        if matches.size == 0:
            raise ValueError(
                f"reference antenna {args.reference_antenna} not in selected array {array_ids.tolist()}"
            )
        ref_idx = int(matches[0])
        ref_ant = int(array_ids[ref_idx])

    if args.mode == "crossref":
        gains = _estimate_crossref(x, ref_idx)
    else:
        gains = _estimate_dominant(x)

    # Normalize to chosen reference channel.
    gains = gains / gains[ref_idx]

    freq_khz = float(np.median(np.array(freq_vals, dtype=np.float32)))
    if args.known_az_deg is not None:
        a = _steering_vector(args.known_az_deg, array_x, freq_khz)
        a = a / a[ref_idx]
        gains = gains / a

    amps = np.abs(gains)
    phases_deg = np.angle(gains, deg=True)

    correction = (1.0 / gains).astype(np.complex64)

    np.savez(
        output_path,
        antenna_ids=array_ids.astype(np.int32),
        channel_gains=gains.astype(np.complex64),
        correction=correction,
        array_x_m=array_x.astype(np.float32),
        reference_antenna=np.int32(ref_ant),
        freq_khz=np.float32(freq_khz),
        mode=np.bytes_(args.mode),
        array=np.bytes_(args.array),
        records=np.array(selected, dtype="S64"),
    )

    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "array": args.array,
        "mode": args.mode,
        "reference_antenna": ref_ant,
        "freq_khz": freq_khz,
        "records": selected,
        "channels": [
            {
                "antenna": int(a),
                "amp": float(aa),
                "phase_deg": float(ph),
            }
            for a, aa, ph in zip(array_ids.tolist(), amps.tolist(), phases_deg.tolist())
        ],
    }

    if args.summary_json:
        json_path = Path(args.summary_json).expanduser().resolve()
        json_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Estimate per-channel complex calibration from antennas_iq data")
    p.add_argument("--input", required=True, help="Path to antennas_iq HDF5 file")
    p.add_argument("--output", help="Output npz path (default: <input>.<array>.selfcal.npz)")
    p.add_argument("--record", help="Specific record name")
    p.add_argument("--records", type=int, default=5, help="Use latest N records if --record omitted")
    p.add_argument("--array", choices=["main", "intf", "all"], default="main")
    p.add_argument("--mode", choices=["crossref", "dominant"], default="dominant")
    p.add_argument("--reference-antenna", type=int, help="Antenna ID to normalize against")
    p.add_argument("--known-az-deg", type=float, help="Optional known source azimuth to remove geometric steering")
    p.add_argument("--summary-json", help="Optional JSON summary output path")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
