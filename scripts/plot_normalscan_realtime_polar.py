#!/usr/bin/env python3
"""Capture realtime normalscan FIT data and render a 2-panel polar plot.

This script listens to Borealis realtime FIT packets, accumulates records for a
short interval, and plots median power and Doppler on a beam/range polar grid.
By default it always renders the full beam extent (0..23), leaving sectors blank
where no fitted scatter is present.
"""

from __future__ import annotations

import argparse
import bz2
import datetime as dt
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pydarnio
import zmq


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--duration-s", type=float, default=70.0, help="Capture duration in seconds")
    p.add_argument("--socket", default="tcp://192.168.112.127:9696", help="Realtime PUB socket")
    p.add_argument("--cp", type=int, default=151, help="Control program id to filter")
    p.add_argument("--poll-timeout-ms", type=int, default=2000, help="ZMQ receive timeout in milliseconds")

    p.add_argument("--beam-min", type=int, default=0, help="Minimum beam number for fixed full-extent plotting")
    p.add_argument("--beam-max", type=int, default=23, help="Maximum beam number for fixed full-extent plotting")
    p.add_argument("--beams-from-data", action="store_true", help="Only plot beams seen in captured records")
    p.add_argument("--default-beam-spacing-deg", type=float, default=3.24, help="Fallback beam spacing when azimuth metadata is incomplete")

    p.add_argument("--default-frang", type=float, default=180.0, help="Fallback first range in km")
    p.add_argument("--default-rsep", type=float, default=45.0, help="Fallback range separation in km")
    p.add_argument("--default-nrang", type=int, default=75, help="Fallback number of range gates")

    p.add_argument("--output", default=None, help="Output PNG path (default: /tmp/wal_normalscan_polar_power_doppler_live_<UTC>.png)")
    p.add_argument("--meta-output", default=None, help="Output JSON metadata path")
    p.add_argument("--title-prefix", default="WAL normalscan", help="Figure title prefix")
    p.add_argument("--dpi", type=int, default=160, help="Figure DPI")
    return p.parse_args()


def _read_fit_records(payload: bytes) -> list[dict]:
    out = pydarnio.read_fitacf(payload)
    if isinstance(out, tuple):
        if not out:
            return []
        recs = out[0]
        return recs if isinstance(recs, list) else []
    return out if isinstance(out, list) else []


def _derive_beam_azimuths(
    beam_ids: list[int],
    beam_az_samples: dict[int, list[float]],
    default_spacing_deg: float,
) -> dict[int, float]:
    beam_az_deg: dict[int, float] = {}

    known_beams = []
    known_az = []
    for b in sorted(beam_az_samples.keys()):
        vals = np.asarray(beam_az_samples[b], dtype=np.float32)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            known_beams.append(b)
            known_az.append(float(np.median(vals)))

    if len(known_beams) >= 2:
        x = np.asarray(known_beams, dtype=np.float32)
        y = np.asarray(known_az, dtype=np.float32)
        for b in beam_ids:
            beam_az_deg[b] = float(np.interp(b, x, y, left=y[0] + (b - x[0]) * default_spacing_deg, right=y[-1] + (b - x[-1]) * default_spacing_deg))
        return beam_az_deg

    if len(known_beams) == 1:
        ref_b = known_beams[0]
        ref_az = known_az[0]
        for b in beam_ids:
            beam_az_deg[b] = float(ref_az + (b - ref_b) * default_spacing_deg)
        return beam_az_deg

    if not beam_ids:
        return beam_az_deg

    mid = 0.5 * (beam_ids[0] + beam_ids[-1])
    for b in beam_ids:
        beam_az_deg[b] = float((b - mid) * default_spacing_deg)
    return beam_az_deg


def main() -> int:
    args = parse_args()

    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    sub.setsockopt(zmq.RCVTIMEO, args.poll_timeout_ms)
    sub.connect(args.socket)

    start = dt.datetime.utcnow()
    end_ts = start.timestamp() + args.duration_s

    p_acc: dict[tuple[int, int], list[float]] = defaultdict(list)
    v_acc: dict[tuple[int, int], list[float]] = defaultdict(list)
    beam_az_samples: dict[int, list[float]] = defaultdict(list)

    frangs: list[float] = []
    rseps: list[float] = []
    nrangs: list[int] = []
    all_beams_seen: list[int] = []

    msg_count = 0
    cp_rec_count = 0

    while dt.datetime.utcnow().timestamp() < end_ts:
        try:
            msg = sub.recv()
        except zmq.error.Again:
            continue

        msg_count += 1
        try:
            recs = _read_fit_records(bz2.decompress(msg))
        except Exception:
            continue

        for rec in recs:
            if rec.get("cp") != args.cp:
                continue

            bm = rec.get("bmnum")
            if bm is None:
                continue
            bm = int(bm)
            all_beams_seen.append(bm)
            cp_rec_count += 1

            az = rec.get("bmazm")
            if az is not None and np.isfinite(az):
                beam_az_samples[bm].append(float(az))

            fr = rec.get("frang")
            rs = rec.get("rsep")
            nr = rec.get("nrang")
            if fr is not None:
                frangs.append(float(fr))
            if rs is not None:
                rseps.append(float(rs))
            if nr is not None:
                nrangs.append(int(nr))

            slist = np.asarray(rec.get("slist", []), dtype=np.int32)
            p_l = np.asarray(rec.get("p_l", []), dtype=np.float32)
            v = np.asarray(rec.get("v", []), dtype=np.float32)
            n = min(len(slist), len(p_l), len(v))
            for i in range(n):
                g = int(slist[i])
                p_acc[(bm, g)].append(float(p_l[i]))
                v_acc[(bm, g)].append(float(v[i]))

    stop = dt.datetime.utcnow()

    if cp_rec_count == 0:
        raise RuntimeError(f"No cp={args.cp} records received in capture window")

    if args.beams_from_data:
        beam_ids = sorted(set(all_beams_seen))
    else:
        beam_ids = list(range(args.beam_min, args.beam_max + 1))

    if not beam_ids:
        raise RuntimeError("No beams available for plotting")

    beam_az_deg = _derive_beam_azimuths(beam_ids, beam_az_samples, args.default_beam_spacing_deg)
    az_cent = np.deg2rad(np.asarray([beam_az_deg[b] for b in beam_ids], dtype=np.float32))

    if len(az_cent) == 1:
        dth = np.deg2rad(args.default_beam_spacing_deg)
        az_edges = np.asarray([az_cent[0] - 0.5 * dth, az_cent[0] + 0.5 * dth], dtype=np.float32)
    else:
        mids = 0.5 * (az_cent[1:] + az_cent[:-1])
        left = az_cent[0] - (mids[0] - az_cent[0])
        right = az_cent[-1] + (az_cent[-1] - mids[-1])
        az_edges = np.concatenate([[left], mids, [right]]).astype(np.float32)

    max_gate_seen = max((g for (_, g) in p_acc.keys()), default=-1)
    frang = float(np.median(np.asarray(frangs, dtype=np.float32))) if frangs else float(args.default_frang)
    rsep = float(np.median(np.asarray(rseps, dtype=np.float32))) if rseps else float(args.default_rsep)
    nrang_data = int(np.median(np.asarray(nrangs, dtype=np.float32))) if nrangs else int(args.default_nrang)
    nrang = max(int(args.default_nrang), nrang_data, max_gate_seen + 1)

    gates = np.arange(nrang, dtype=np.int32)
    r_cent = frang + gates * rsep
    r_edges = np.concatenate([[max(0.0, r_cent[0] - 0.5 * rsep)], r_cent[:-1] + 0.5 * rsep, [r_cent[-1] + 0.5 * rsep]]).astype(np.float32)

    power = np.full((nrang, len(beam_ids)), np.nan, dtype=np.float32)
    doppler = np.full((nrang, len(beam_ids)), np.nan, dtype=np.float32)
    beam_to_col = {b: i for i, b in enumerate(beam_ids)}

    for (b, g), vals in p_acc.items():
        if b in beam_to_col and 0 <= g < nrang and vals:
            power[g, beam_to_col[b]] = float(np.median(np.asarray(vals, dtype=np.float32)))

    for (b, g), vals in v_acc.items():
        if b in beam_to_col and 0 <= g < nrang and vals:
            doppler[g, beam_to_col[b]] = float(np.median(np.asarray(vals, dtype=np.float32)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={"projection": "polar"}, constrained_layout=True)

    for ax in axes:
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_thetamin(float(np.rad2deg(az_edges.min())))
        ax.set_thetamax(float(np.rad2deg(az_edges.max())))
        ax.set_rmax(float(r_edges[-1]))
        ax.grid(True, alpha=0.3)

    th2d, r2d = np.meshgrid(az_edges, r_edges)

    pvals = power[np.isfinite(power)]
    if pvals.size:
        pvmax = float(np.percentile(pvals, 98))
        pvmin = max(0.0, float(np.percentile(pvals, 10)))
        if pvmax <= pvmin:
            pvmin, pvmax = 0.0, max(20.0, float(pvals.max()))
    else:
        pvmin, pvmax = 0.0, 20.0

    im0 = axes[0].pcolormesh(th2d, r2d, power, cmap="viridis", vmin=pvmin, vmax=pvmax, shading="auto")
    axes[0].set_title("Normalscan Power (dB)")
    cb0 = fig.colorbar(im0, ax=axes[0], pad=0.08)
    cb0.set_label("Power (dB)")

    vvals = doppler[np.isfinite(doppler)]
    if vvals.size:
        vlim = float(np.percentile(np.abs(vvals), 98))
        vlim = max(100.0, min(vlim, 1500.0))
    else:
        vlim = 500.0

    im1 = axes[1].pcolormesh(th2d, r2d, doppler, cmap="RdBu_r", vmin=-vlim, vmax=vlim, shading="auto")
    axes[1].set_title("Normalscan Doppler (m/s)")
    cb1 = fig.colorbar(im1, ax=axes[1], pad=0.08)
    cb1.set_label("Velocity (m/s)")

    fig.suptitle(
        f"{args.title_prefix} cp={args.cp} realtime capture {start.strftime('%Y-%m-%d %H:%M:%S')}Z to {stop.strftime('%H:%M:%S')}Z"
    )

    stamp = stop.strftime("%Y%m%dT%H%M%SZ")
    output_png = Path(args.output).expanduser() if args.output else Path(f"/tmp/wal_normalscan_polar_power_doppler_live_{stamp}.png")
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=args.dpi)
    plt.close(fig)

    meta = {
        "start_utc": start.isoformat() + "Z",
        "stop_utc": stop.isoformat() + "Z",
        "duration_s": (stop - start).total_seconds(),
        "socket": args.socket,
        "cp": args.cp,
        "messages_received": msg_count,
        "cp_records": cp_rec_count,
        "all_beams_seen": sorted(set(all_beams_seen)),
        "plot_beams": beam_ids,
        "beam_az_deg": {str(k): float(v) for k, v in beam_az_deg.items()},
        "frang_km": frang,
        "rsep_km": rsep,
        "nrang": nrang,
        "output_png": str(output_png),
    }

    output_json = Path(args.meta_output).expanduser() if args.meta_output else Path(str(output_png).replace(".png", ".json"))
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(meta, indent=2))

    print(str(output_png))
    print(str(output_json))
    print(json.dumps(meta))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
