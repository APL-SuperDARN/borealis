#!/usr/bin/env python3
"""
Solve for a receive-side MCCM correction vector from accepted antennas_iq captures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from mccm_common import (
    DEFAULT_PNR_THRESHOLD_DB,
    _aggregate_complex,
    active_main_antennas,
    evaluate_file,
    evaluate_file_all_main_rx,
    load_config,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Solve receive-side MCCM correction from antennas_iq captures")
    parser.add_argument("--manifest", help="Optional JSON manifest of capture files")
    parser.add_argument("--input-dir", help="Optional directory tree to scan for HDF5 captures")
    parser.add_argument("--glob", default="*.h5", help="Glob used when scanning --input-dir")
    parser.add_argument("--records", type=int, default=5, help="Use the latest N records from each file if --record omitted")
    parser.add_argument("--record", help="Specific record name to evaluate in each file")
    parser.add_argument("--pnr-threshold-db", type=float, default=DEFAULT_PNR_THRESHOLD_DB, help="Minimum per-record PNR for acceptance")
    parser.add_argument("--min-consecutive", type=int, default=1, help="Minimum consecutive accepted records required for a file/pair measurement")
    parser.add_argument("--reference-antenna", type=int, help="Reference RX antenna for normalization")
    parser.add_argument("--pulse-window-us", type=float, help="Override pulse window length in microseconds")
    parser.add_argument("--config", help="Path to config file. Defaults to <BOREALISPATH>/config/<radar>/<radar>_config.ini")
    parser.add_argument("--borealis-path", help="Path to Borealis repo if --config omitted")
    parser.add_argument("--radar-id", default="wal", help="Radar ID if --config omitted")
    parser.add_argument("--output", help="Output npz path. Defaults to ./mccm_rxcal.npz")
    parser.add_argument("--summary-json", help="Optional JSON summary path")
    return parser


def _load_manifest(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    if isinstance(data, dict):
        items = data.get("captures", data.get("accepted_runs", []))
    else:
        items = data
    normalized = []
    for item in items:
        if isinstance(item, str):
            normalized.append({"input": item})
        else:
            normalized.append(
                {
                    "input": item["input"],
                    "rx_ant": item.get("rx_ant"),
                }
            )
    return normalized


def _collect_capture_entries(args: argparse.Namespace) -> list[dict[str, Any]]:
    entries = []
    if args.manifest:
        entries.extend(_load_manifest(args.manifest))
    if args.input_dir:
        input_dir = Path(args.input_dir).expanduser().resolve()
        for path in sorted(input_dir.rglob(args.glob)):
            entries.append({"input": str(path)})
    if not entries:
        raise ValueError("Supply --manifest or --input-dir")

    deduped = []
    seen = set()
    for entry in entries:
        key = (entry["input"], entry.get("rx_ant"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entry)
    return deduped


def _aggregate_pair_measurements(summaries: list[dict[str, Any]]) -> dict[tuple[int, int], dict[str, Any]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for summary in summaries:
        key = (int(summary["tx_ant"]), int(summary["rx_ant"]))
        grouped.setdefault(key, []).append(summary)

    aggregated = {}
    for key, items in grouped.items():
        responses = [
            complex(item["complex_response_real"], item["complex_response_imag"])
            for item in items
        ]
        response = _aggregate_complex(responses)
        aggregated[key] = {
            "tx_ant": key[0],
            "rx_ant": key[1],
            "complex_response": response,
            "count": len(items),
            "inputs": [item["input"] for item in items],
        }
    return aggregated


def _choose_reference(
    antenna_ids: list[int],
    pair_matrix: np.ndarray,
    requested_reference: int | None,
) -> int:
    if requested_reference is not None:
        if requested_reference not in antenna_ids:
            raise ValueError(
                f"reference antenna {requested_reference} not present in antenna_ids {antenna_ids}"
            )
        return requested_reference

    counts = np.sum(np.isfinite(pair_matrix.real), axis=0)
    best_count = int(np.max(counts))
    candidates = [ant for ant, count in zip(antenna_ids, counts.tolist()) if count == best_count]
    return int(min(candidates))


def _safe_inverse(values: np.ndarray) -> np.ndarray:
    out = np.full(values.shape, np.nan + 1j * np.nan, dtype=np.complex64)
    mask = np.isfinite(values.real) & np.isfinite(values.imag) & (np.abs(values) > 1.0e-12)
    out[mask] = (1.0 / values[mask]).astype(np.complex64)
    return out


def main() -> None:
    args = build_parser().parse_args()
    config, _config_path = load_config(args.config, args.radar_id, args.borealis_path)
    capture_entries = _collect_capture_entries(args)

    accepted = []
    rejected = []
    for entry in capture_entries:
        input_path = entry["input"]
        rx_ant = entry.get("rx_ant")
        try:
            if rx_ant is None:
                summaries = evaluate_file_all_main_rx(
                    input_path,
                    record=args.record,
                    records=args.records,
                    pnr_threshold_db=args.pnr_threshold_db,
                    min_consecutive=args.min_consecutive,
                    pulse_window_us=args.pulse_window_us,
                )
            else:
                summaries = [
                    evaluate_file(
                        input_path,
                        rx_ant=int(rx_ant),
                        record=args.record,
                        records=args.records,
                        pnr_threshold_db=args.pnr_threshold_db,
                        min_consecutive=args.min_consecutive,
                        pulse_window_us=args.pulse_window_us,
                    )
                ]
        except Exception as exc:
            rejected.append({"input": str(Path(input_path).expanduser().resolve()), "reason": str(exc)})
            continue

        for summary in summaries:
            if summary["accepted_record_count"] > 0 and summary["max_consecutive_accepts"] >= args.min_consecutive:
                accepted.append(summary)
            else:
                rejected.append(
                    {
                        "input": summary["input"],
                        "rx_ant": summary["rx_ant"],
                        "reason": "No accepted records for this pair",
                    }
                )

    if not accepted:
        raise RuntimeError("No accepted MCCM capture measurements were found")

    pair_measurements = _aggregate_pair_measurements(accepted)
    tx_antenna_ids = sorted({pair[0] for pair in pair_measurements})
    antenna_ids = active_main_antennas(config) or sorted(
        {pair[0] for pair in pair_measurements} | {pair[1] for pair in pair_measurements}
    )

    pair_matrix = np.full(
        (len(tx_antenna_ids), len(antenna_ids)),
        np.nan + 1j * np.nan,
        dtype=np.complex64,
    )
    pair_counts = np.zeros((len(tx_antenna_ids), len(antenna_ids)), dtype=np.int32)
    for (tx_ant, rx_ant), measurement in pair_measurements.items():
        row = tx_antenna_ids.index(tx_ant)
        col = antenna_ids.index(rx_ant)
        pair_matrix[row, col] = np.complex64(measurement["complex_response"])
        pair_counts[row, col] = int(measurement["count"])

    reference_antenna = _choose_reference(
        antenna_ids, pair_matrix, args.reference_antenna
    )
    reference_index = antenna_ids.index(reference_antenna)

    channel_gains = np.full(len(antenna_ids), np.nan + 1j * np.nan, dtype=np.complex64)
    channel_gains[reference_index] = np.complex64(1.0 + 0.0j)
    for col, antenna in enumerate(antenna_ids):
        if col == reference_index:
            continue
        ratios = []
        for row in range(pair_matrix.shape[0]):
            reference_value = pair_matrix[row, reference_index]
            value = pair_matrix[row, col]
            if (
                np.isfinite(reference_value.real)
                and np.isfinite(reference_value.imag)
                and np.isfinite(value.real)
                and np.isfinite(value.imag)
                and np.abs(reference_value) > 1.0e-12
            ):
                ratios.append(value / reference_value)
        if ratios:
            channel_gains[col] = np.complex64(_aggregate_complex(ratios))

    row_scales = np.full(len(tx_antenna_ids), np.nan + 1j * np.nan, dtype=np.complex64)
    for row in range(pair_matrix.shape[0]):
        scales = []
        for col in range(pair_matrix.shape[1]):
            value = pair_matrix[row, col]
            gain = channel_gains[col]
            if (
                np.isfinite(value.real)
                and np.isfinite(value.imag)
                and np.isfinite(gain.real)
                and np.isfinite(gain.imag)
                and np.abs(gain) > 1.0e-12
            ):
                scales.append(value / gain)
        if scales:
            row_scales[row] = np.complex64(_aggregate_complex(scales))

    predicted = row_scales[:, np.newaxis] * channel_gains[np.newaxis, :]
    residual_matrix = pair_matrix - predicted
    valid_mask = np.isfinite(pair_matrix.real) & np.isfinite(pair_matrix.imag)
    normalized_residual = np.full(pair_matrix.shape, np.nan, dtype=np.float32)
    normalized_residual[valid_mask] = (
        np.abs(residual_matrix[valid_mask]) / np.maximum(np.abs(pair_matrix[valid_mask]), 1.0e-12)
    ).astype(np.float32)
    residual_values = normalized_residual[np.isfinite(normalized_residual)]
    residual_rms = float(np.sqrt(np.mean(residual_values ** 2))) if residual_values.size else float("nan")
    residual_median = float(np.median(residual_values)) if residual_values.size else float("nan")

    correction = _safe_inverse(channel_gains)

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        output_path = Path.cwd() / "mccm_rxcal.npz"
        output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_path,
        antenna_ids=np.asarray(antenna_ids, dtype=np.int32),
        tx_antenna_ids=np.asarray(tx_antenna_ids, dtype=np.int32),
        pair_matrix=pair_matrix,
        pair_counts=pair_counts,
        channel_gains=channel_gains,
        correction=correction,
        row_scales=row_scales,
        residual_matrix=residual_matrix.astype(np.complex64),
        residual_rms=np.float32(residual_rms),
        residual_median=np.float32(residual_median),
        reference_antenna=np.int32(reference_antenna),
        accepted_runs=np.asarray([item["input"] for item in accepted], dtype="S512"),
        rejected_runs=np.asarray(
            [
                f"{item['input']}::{item.get('reason', '')}"
                for item in rejected
            ],
            dtype="S512",
        ),
    )

    channel_summary = []
    for antenna, gain, correction_value in zip(antenna_ids, channel_gains, correction):
        channel_summary.append(
            {
                "antenna": int(antenna),
                "gain_amp": float(np.abs(gain)),
                "gain_phase_deg": float(np.angle(gain, deg=True)),
                "correction_amp": float(np.abs(correction_value)),
                "correction_phase_deg": float(np.angle(correction_value, deg=True)),
            }
        )

    summary = {
        "output": str(output_path),
        "reference_antenna": int(reference_antenna),
        "tx_antenna_ids": tx_antenna_ids,
        "antenna_ids": antenna_ids,
        "accepted_measurement_count": len(accepted),
        "rejected_measurement_count": len(rejected),
        "residual_rms": residual_rms,
        "residual_median": residual_median,
        "channels": channel_summary,
        "accepted_runs": [item["input"] for item in accepted],
        "rejected_runs": rejected,
    }

    if args.summary_json:
        summary_path = Path(args.summary_json).expanduser().resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
