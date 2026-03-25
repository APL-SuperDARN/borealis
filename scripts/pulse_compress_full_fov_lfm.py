#!/usr/bin/env python3
"""
Offline pulse compression and diagnostic plots for FullFOV LFM antennas_iq files.

The default workflow is conservative and calibration-free:
- read an antennas_iq HDF5 file written by FullFOV with pulse_waveform=lfm
- matched-filter each sequence with the recorded LFM waveform metadata
- use the last pulse in the 7-pulse FullFOV sequence so the full standard range
  extent is available without overlap from a following pulse
- incoherently average the strongest main-array channels to build stable plots
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


C_MPS = 299_792_458.0
DEFAULT_FULL_FOV_SEQUENCE = np.array([0, 9, 12, 20, 22, 26, 27], dtype=float)


def _decode(value):
    return value.decode() if isinstance(value, (bytes, np.bytes_)) else value


def _pick_records(src: h5py.File, record: str | None, records: int) -> list[str]:
    all_records = sorted(k for k in src.keys() if k != "metadata")
    if not all_records:
        raise RuntimeError("No records found in file")

    if record is not None:
        if record not in all_records:
            raise ValueError(f"Record {record!r} not found in file")
        return [record]

    if records <= 0 or records >= len(all_records):
        return all_records

    return all_records[-records:]


def _sample_spacing_us(group: h5py.Group) -> float:
    sample_time = np.asarray(group["sample_time"][()], dtype=np.float64)
    if sample_time.size > 1:
        return float(sample_time[1] - sample_time[0])
    return float(1.0e6 / float(group["rx_sample_rate"][()]))


def _build_template(
    fs_hz: float,
    pulse_len_us: float,
    bandwidth_hz: float,
    sweep: str,
    ramp_time_us: float,
) -> np.ndarray:
    pulse_len_s = pulse_len_us * 1.0e-6
    n = int(round(pulse_len_s * fs_hz))
    t = np.arange(n, dtype=np.float64) / fs_hz
    phase = np.pi * bandwidth_hz / pulse_len_s * (t - pulse_len_s / 2.0) ** 2
    if sweep == "down":
        phase = -phase

    sig = np.exp(1j * phase).astype(np.complex64)

    nr = int(round(ramp_time_us * 1.0e-6 * fs_hz))
    if nr > 0 and 2 * nr < n:
        win = np.ones(n, dtype=np.float32)
        x = np.linspace(0.0, 1.0, nr, endpoint=False, dtype=np.float32)
        ramp = np.sin(0.5 * np.pi * x) ** 2
        win[:nr] = ramp
        win[-nr:] = ramp[::-1]
        sig *= win

    sig /= np.sqrt(np.vdot(sig, sig).real)
    return sig


def _parse_pulse_sequence(group: h5py.Group, fallback_csv: str) -> np.ndarray:
    if "pulse_sequence" in group:
        return np.asarray(group["pulse_sequence"][()], dtype=float)
    return np.array([float(x.strip()) for x in fallback_csv.split(",") if x.strip()], dtype=float)


def _parse_pulse_index(value: str, num_pulses: int) -> int:
    if value == "last":
        return num_pulses - 1
    idx = int(value)
    if idx < 0 or idx >= num_pulses:
        raise ValueError(f"pulse index {idx} out of bounds for {num_pulses} pulses")
    return idx


def _rank_antennas(
    src: h5py.File,
    records: Iterable[str],
    channel_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    power = np.zeros(channel_indices.size, dtype=np.float64)
    for rec in records:
        data = np.asarray(src[rec]["antennas_iq_data"][channel_indices], dtype=np.complex64)
        power += np.mean(np.abs(data) ** 2, axis=(1, 2))
    power /= max(len(list(records)), 1)
    order = np.argsort(power)[::-1]
    return channel_indices[order], power[order]


def _matched_filter(x: np.ndarray, template: np.ndarray) -> np.ndarray:
    return np.convolve(x, np.conjugate(template[::-1]), mode="full")


def _build_summary(
    args: argparse.Namespace,
    group: h5py.Group,
    records: list[str],
    seq_dt: np.ndarray,
    selected_antennas: np.ndarray,
    selected_powers: np.ndarray,
    template_scores: dict[str, float],
    width_stats_us: dict[str, float],
    range_km: np.ndarray,
    elapsed_s: np.ndarray,
) -> dict:
    return {
        "input": str(Path(args.input).expanduser().resolve()),
        "output_dir": str(Path(args.output_dir).expanduser().resolve()) if args.output_dir else None,
        "record_count": len(records),
        "records": records,
        "experiment_name": _decode(group["experiment_name"][()]) if "experiment_name" in group else None,
        "borealis_git_hash": _decode(group["borealis_git_hash"][()]),
        "tx_pulse_waveform": _decode(group["tx_pulse_waveform"][()]),
        "tx_pulse_waveform_bandwidth_hz": float(group["tx_pulse_waveform_bandwidth"][()]),
        "tx_pulse_waveform_sweep": _decode(group["tx_pulse_waveform_sweep"][()]),
        "rx_sample_rate_hz": float(group["rx_sample_rate"][()]),
        "sample_spacing_us": _sample_spacing_us(group),
        "tau_spacing_us": float(group["tau_spacing"][()]),
        "tx_pulse_len_us": float(group["tx_pulse_len"][()]),
        "first_range_km": float(group["first_range"][()]),
        "range_sep_km": float(group["range_sep"][()]),
        "num_ranges": int(np.asarray(group["range_gates"][()]).size),
        "selected_antennas": selected_antennas.astype(int).tolist(),
        "selected_antenna_mean_power": selected_powers.tolist(),
        "sequence_delta_mean_s": float(np.mean(seq_dt)) if seq_dt.size else None,
        "sequence_delta_std_s": float(np.std(seq_dt)) if seq_dt.size else None,
        "num_sequences_mean": float(np.mean([group.file[r]["num_sequences"][()] for r in records])),
        "num_sequences_std": float(np.std([group.file[r]["num_sequences"][()] for r in records])),
        "matched_filter_scores": template_scores,
        "compressed_width_us": width_stats_us,
        "range_window_km": [float(range_km[0]), float(range_km[-1])],
        "elapsed_time_s": [float(elapsed_s[0]), float(elapsed_s[-1])],
    }


def run(args: argparse.Namespace) -> Path:
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        stem = input_path.name.replace(".antennas_iq.h5", "")
        output_dir = input_path.parent / f"{stem}.lfm_post"
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_path, "r") as src:
        records = _pick_records(src, args.record, args.records)
        first = src[records[0]]

        waveform = _decode(first["tx_pulse_waveform"][()])
        if waveform != "lfm":
            raise ValueError(f"This script expects tx_pulse_waveform=lfm, got {waveform!r}")

        fs_hz = float(first["rx_sample_rate"][()])
        pulse_len_us = float(first["tx_pulse_len"][()])
        bandwidth_hz = float(first["tx_pulse_waveform_bandwidth"][()])
        sweep = _decode(first["tx_pulse_waveform_sweep"][()])
        tau_spacing_us = float(first["tau_spacing"][()])
        pulse_sequence = _parse_pulse_sequence(first, args.pulse_sequence)
        pulse_index = _parse_pulse_index(args.pulse_index, len(pulse_sequence))
        pulse_starts = np.round((pulse_sequence * tau_spacing_us * 1.0e-6) * fs_hz).astype(int)

        template = _build_template(fs_hz, pulse_len_us, bandwidth_hz, sweep, args.ramp_time_us)
        mf_delay = pulse_starts[pulse_index] + template.size - 1

        rx_antennas = np.asarray(first["rx_antennas"][()], dtype=np.int32)
        if args.array == "main" and "rx_main_antennas" in first:
            want = np.asarray(first["rx_main_antennas"][()], dtype=np.int32)
            base_indices = np.where(np.isin(rx_antennas, want))[0]
        else:
            base_indices = np.arange(rx_antennas.size)

        if base_indices.size == 0:
            raise RuntimeError(f"No receive channels found for array={args.array}")

        ranked_indices, ranked_power = _rank_antennas(src, records, base_indices)
        score_ranked_indices, _ = _rank_antennas(src, records[: min(len(records), max(args.score_records, 1))], np.arange(rx_antennas.size))
        score_index = int(score_ranked_indices[0])
        score_antenna = int(rx_antennas[score_index])
        top_n = min(args.top_antennas, ranked_indices.size)
        selected_indices = ranked_indices[:top_n]
        selected_fetch_indices = np.sort(selected_indices)
        selected_fetch_order = np.searchsorted(selected_fetch_indices, selected_indices)
        selected_antennas = rx_antennas[selected_indices]
        selected_power = ranked_power[:top_n]
        strongest_index = int(ranked_indices[0])
        strongest_antenna = int(rx_antennas[strongest_index])

        first_range_km = float(first["first_range"][()])
        default_max_km = first_range_km + float(np.asarray(first["range_gates"][()]).size) * float(first["range_sep"][()])
        range_min_km = args.range_min_km if args.range_min_km is not None else first_range_km
        available_max_lag = int(first["antennas_iq_data"].shape[-1] - pulse_starts[pulse_index] - 1)
        available_max_km = available_max_lag / fs_hz * C_MPS / 2.0 / 1000.0
        range_max_km = args.range_max_km if args.range_max_km is not None else min(default_max_km, available_max_km)

        lag_min = max(0, int(np.floor(range_min_km * 1000.0 * 2.0 / C_MPS * fs_hz)))
        lag_max = min(available_max_lag, int(np.ceil(range_max_km * 1000.0 * 2.0 / C_MPS * fs_hz)))
        lags = np.arange(lag_min, lag_max + 1, dtype=np.int32)
        range_km = lags.astype(np.float64) / fs_hz * C_MPS / 2.0 / 1000.0

        elapsed_s = np.zeros(len(records), dtype=np.float64)
        int_time_s = np.zeros(len(records), dtype=np.float64)
        num_sequences = np.zeros(len(records), dtype=np.int32)
        rti_power = np.zeros((len(records), lags.size), dtype=np.float32)
        example_raw = None
        example_comp = None

        t0 = float(src[records[0]]["sqn_timestamps"][()][0])

        for rec_idx, rec in enumerate(records):
            group = src[rec]
            data = np.asarray(group["antennas_iq_data"][selected_fetch_indices], dtype=np.complex64)[selected_fetch_order]
            acc = np.zeros(lags.size, dtype=np.float64)
            count = 0
            for ant_i in range(data.shape[0]):
                for seq_i in range(data.shape[1]):
                    comp = _matched_filter(data[ant_i, seq_i], template)
                    seg = comp[mf_delay + lags]
                    acc += np.abs(seg) ** 2
                    count += 1
                    if example_comp is None and ant_i == 0 and seq_i == 0:
                        example_comp = 10.0 * np.log10(np.abs(seg) ** 2 + 1.0e-12)
                        raw = np.asarray(data[ant_i, seq_i], dtype=np.complex64)
                        raw_lo = max(0, pulse_starts[pulse_index] - template.size)
                        raw_hi = min(raw.size, pulse_starts[pulse_index] + 4 * template.size)
                        example_raw = {
                            "time_ms": np.arange(raw_lo, raw_hi, dtype=np.float64) / fs_hz * 1.0e3,
                            "amplitude": np.abs(raw[raw_lo:raw_hi]),
                        }

            rti_power[rec_idx] = acc / max(count, 1)
            elapsed_s[rec_idx] = float(group["sqn_timestamps"][()][0]) - t0
            int_time_s[rec_idx] = float(group["int_time"][()])
            num_sequences[rec_idx] = int(group["num_sequences"][()])

        all_ts = np.concatenate([src[r]["sqn_timestamps"][()] for r in records]).astype(np.float64)
        seq_dt = np.diff(all_ts)

        score_hypotheses = {
            "tone": (0.0, "up"),
            f"up_{bandwidth_hz * 0.64:.0f}Hz": (bandwidth_hz * 0.64, "up"),
            f"{sweep}_{bandwidth_hz:.0f}Hz": (bandwidth_hz, sweep),
            f"up_{bandwidth_hz * 1.6:.0f}Hz": (bandwidth_hz * 1.6, "up"),
            f"{'down' if sweep == 'up' else 'up'}_{bandwidth_hz:.0f}Hz": (
                bandwidth_hz,
                "down" if sweep == "up" else "up",
            ),
        }
        score_templates = {
            name: (_build_template(fs_hz, pulse_len_us, bw, sw, args.ramp_time_us) if bw > 0.0 else np.ones(template.size, dtype=np.complex64) / np.sqrt(template.size))
            for name, (bw, sw) in score_hypotheses.items()
        }
        template_scores: dict[str, float] = {name: [] for name in score_templates}
        widths_us = []
        score_records = records[: min(len(records), max(args.score_records, 1))]
        for rec in score_records:
            seq_data = np.asarray(src[rec]["antennas_iq_data"][score_index], dtype=np.complex64)
            for seq_i in range(seq_data.shape[0]):
                for name, tpl in score_templates.items():
                    comp = np.abs(np.convolve(seq_data[seq_i], np.conjugate(tpl[::-1]), mode="same"))
                    vals = []
                    for s in pulse_starts:
                        lo = max(0, s - 10)
                        hi = min(comp.size, s + tpl.size + 10)
                        vals.append(float(comp[lo:hi].max()))
                    template_scores[name].append(float(np.median(vals)))

                comp = np.abs(np.convolve(seq_data[seq_i], np.conjugate(template[::-1]), mode="same"))
                pulse_vals = []
                for s in pulse_starts:
                    lo = max(0, s - 10)
                    hi = min(comp.size, s + template.size + 10)
                    pulse_vals.append(float(comp[lo:hi].max()))
                s = pulse_starts[int(np.argmax(pulse_vals))]
                lo = max(0, s - 10)
                hi = min(comp.size, s + template.size + 10)
                seg = comp[lo:hi]
                k = int(np.argmax(seg))
                half = seg[k] * 0.5
                left = k
                while left > 0 and seg[left] >= half:
                    left -= 1
                right = k
                while right < seg.size - 1 and seg[right] >= half:
                    right += 1
                widths_us.append((right - left) / fs_hz * 1.0e6)

        template_scores = {name: float(np.median(vals)) for name, vals in template_scores.items()}
        width_stats_us = {
            "median": float(np.median(widths_us)),
            "mean": float(np.mean(widths_us)),
            "min": float(np.min(widths_us)),
            "max": float(np.max(widths_us)),
        }

        rti_db = 10.0 * np.log10(rti_power + 1.0e-12)
        mean_profile_db = 10.0 * np.log10(np.mean(rti_power, axis=0) + 1.0e-12)

        summary = _build_summary(
            args,
            first,
            records,
            seq_dt,
            selected_antennas,
            selected_power,
            template_scores,
            width_stats_us,
            range_km,
            elapsed_s,
        )
        summary["strongest_antenna"] = strongest_antenna
        summary["score_antenna"] = score_antenna
        summary["pulse_sequence_tau_units"] = pulse_sequence.tolist()
        summary["pulse_index"] = pulse_index
        summary["pulse_start_samples"] = pulse_starts.tolist()
        summary["selected_array"] = args.array
        summary["top_antennas"] = int(top_n)
        summary["int_time_mean_s"] = float(np.mean(int_time_s))
        summary["int_time_std_s"] = float(np.std(int_time_s))
        summary["available_max_range_km_from_pulse"] = float(available_max_km)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    np.savez(
        output_dir / "compressed_last_pulse.npz",
        range_km=range_km.astype(np.float32),
        elapsed_s=elapsed_s.astype(np.float32),
        rti_power_db=rti_db.astype(np.float32),
        mean_profile_db=mean_profile_db.astype(np.float32),
        selected_antennas=selected_antennas.astype(np.int32),
    )

    ant_fig = plt.figure(figsize=(10, 4))
    ax = ant_fig.add_subplot(111)
    all_ids = summary["selected_antennas"]
    ax.bar(np.arange(len(all_ids)), 10.0 * np.log10(selected_power + 1.0e-12), color="#1f77b4")
    ax.set_xticks(np.arange(len(all_ids)))
    ax.set_xticklabels([str(a) for a in all_ids])
    ax.set_xlabel("Selected antenna ID")
    ax.set_ylabel("Mean power (dB)")
    ax.set_title("Top antennas used for incoherent pulse-compressed average")
    ant_fig.tight_layout()
    ant_fig.savefig(output_dir / "antenna_power.png", dpi=160)
    plt.close(ant_fig)

    score_fig = plt.figure(figsize=(9, 4))
    ax = score_fig.add_subplot(111)
    labels = list(template_scores.keys())
    values = [template_scores[k] for k in labels]
    ax.bar(np.arange(len(labels)), values, color="#d95f02")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Median matched-filter peak")
    ax.set_title(f"Strongest-channel template check (ant {summary['score_antenna']})")
    score_fig.tight_layout()
    score_fig.savefig(output_dir / "matched_filter_scores.png", dpi=160)
    plt.close(score_fig)

    example_fig = plt.figure(figsize=(10, 6))
    ax1 = example_fig.add_subplot(211)
    ax1.plot(example_raw["time_ms"], example_raw["amplitude"], color="#2c7fb8", lw=1.0)
    ax1.set_ylabel("|IQ|")
    ax1.set_title(f"Example raw sequence around pulse {pulse_index} (ant {strongest_antenna})")
    ax2 = example_fig.add_subplot(212)
    ax2.plot(range_km, example_comp, color="#d95f02", lw=1.0)
    ax2.set_xlabel("Range (km)")
    ax2.set_ylabel("Compressed power (dB)")
    ax2.set_title("Example last-pulse compressed profile")
    example_fig.tight_layout()
    example_fig.savefig(output_dir / "example_last_pulse.png", dpi=160)
    plt.close(example_fig)

    profile_fig = plt.figure(figsize=(10, 4))
    ax = profile_fig.add_subplot(111)
    ax.plot(range_km, mean_profile_db, color="#1b9e77", lw=1.2)
    ax.set_xlabel("Range (km)")
    ax.set_ylabel("Mean compressed power (dB)")
    ax.set_title("Mean last-pulse compressed profile")
    profile_fig.tight_layout()
    profile_fig.savefig(output_dir / "mean_range_profile.png", dpi=160)
    plt.close(profile_fig)

    rti_fig = plt.figure(figsize=(10, 5))
    ax = rti_fig.add_subplot(111)
    vmin = float(np.percentile(rti_db, 10.0))
    vmax = float(np.percentile(rti_db, 99.5))
    im = ax.imshow(
        rti_db,
        origin="lower",
        aspect="auto",
        extent=[float(range_km[0]), float(range_km[-1]), float(elapsed_s[0]), float(elapsed_s[-1])],
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
    )
    ax.set_xlabel("Range (km)")
    ax.set_ylabel("Elapsed time (s)")
    ax.set_title("Last-pulse LFM compressed RTI")
    cbar = rti_fig.colorbar(im, ax=ax)
    cbar.set_label("Power (dB)")
    rti_fig.tight_layout()
    rti_fig.savefig(output_dir / "last_pulse_rti.png", dpi=160)
    plt.close(rti_fig)

    print(json.dumps(summary, indent=2))
    return output_dir


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Pulse-compress FullFOV LFM antennas_iq files and generate plots")
    p.add_argument("--input", required=True, help="Path to antennas_iq HDF5 file")
    p.add_argument("--output-dir", help="Directory for plots and summary output")
    p.add_argument("--record", help="Specific record name to process")
    p.add_argument("--records", type=int, default=0, help="Use latest N records; 0 means all")
    p.add_argument("--array", choices=["main", "all"], default="main")
    p.add_argument("--top-antennas", type=int, default=8, help="Number of strongest channels to average")
    p.add_argument("--pulse-index", default="last", help="Pulse index to analyze or 'last'")
    p.add_argument(
        "--pulse-sequence",
        default=",".join(str(int(x)) for x in DEFAULT_FULL_FOV_SEQUENCE),
        help="Fallback pulse sequence in tau units if the file does not carry it",
    )
    p.add_argument("--range-min-km", type=float, help="Minimum plotted range in km")
    p.add_argument("--range-max-km", type=float, help="Maximum plotted range in km")
    p.add_argument("--ramp-time-us", type=float, default=10.0, help="Transmit ramp time in microseconds")
    p.add_argument("--score-records", type=int, default=20, help="Records to use for template-score diagnostics")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
