# Standalone Imaging Tools

This directory now includes two standalone (non-Borealis-runtime) tools:

- `imaging_ab_compare.py`: conventional beamforming + Capon/MUSIC A/B outputs
- `self_calibrate_array.py`: per-channel self-calibration estimation

## 1) Self-calibration

### Active-reference calibration (recommended)
Use a dedicated calibration run where one transmit antenna sends a stable tone/waveform and all RX channels listen.

Example (main array):

```bash
./borealis_env3.11/bin/python3 scripts/self_calibrate_array.py \
  --input /data/borealis_data/YYYYMMDD/<file>.antennas_iq.h5 \
  --records 10 \
  --array main \
  --mode crossref \
  --reference-antenna 2 \
  --output tmp/main_selfcal.npz \
  --summary-json tmp/main_selfcal.json
```

### Passive-reference calibration
If you do not have a controlled transmitter, use dominant-subspace mode (works best with a strong dominant source).

```bash
./borealis_env3.11/bin/python3 scripts/self_calibrate_array.py \
  --input /data/borealis_data/YYYYMMDD/<file>.antennas_iq.h5 \
  --records 5 \
  --array main \
  --mode dominant \
  --reference-antenna 2 \
  --output tmp/main_selfcal.npz
```

Optional: remove known geometric steering if source azimuth is known:

```bash
--known-az-deg 0
```

## 2) A/B Imaging: Conventional vs Capon/MUSIC

```bash
./borealis_env3.11/bin/python3 scripts/imaging_ab_compare.py \
  --input /data/borealis_data/YYYYMMDD/<file>.antennas_iq.h5 \
  --records 1 \
  --method both \
  --az-min -35 --az-max 35 --az-step 0.5 \
  --range-start-km 180 --range-stop-km 400 --range-step-km 2 \
  --window-samples 5 \
  --diag-loading 1e-2 \
  --model-order 1 \
  --calibration tmp/main_selfcal.npz \
  --output tmp/imaging_ab.h5
```

Outputs include:

- `scheduled_beam_power_db` (standard scheduled beamformed power)
- `conventional_scan_power_db` (conventional steered scan over az grid)
- `capon_power_db`
- `music_power_db`
- `capon_peak_azimuth_deg`, `music_peak_azimuth_deg`

## Notes

- These tools are for fast experimentation and validation outside realtime Borealis.
- For operational realtime deployment, use these outputs to tune model order, loading, range/bin windows, and calibration quality first.

## 3) Near-realtime runner

Use the runner to continuously process new records as they arrive in an `antennas_iq` file:

```bash
./borealis_env3.11/bin/python3 scripts/imaging_ab_realtime_runner.py \
  --input-file /data/borealis_data/YYYYMMDD/<file>.antennas_iq.h5 \
  --output tmp/realtime_ab.h5 \
  --state tmp/realtime_ab_state.json \
  --use-calibration \
  --calibration-file tmp/realtime_selfcal.npz \
  --cal-records 5 \
  --cal-mode dominant \
  --cal-reference-antenna 2 \
  --method both \
  --az-min -35 --az-max 35 --az-step 0.5 \
  --range-start-km 180 --range-stop-km 400 --range-step-km 2
```

`run-once` smoke test:

```bash
./borealis_env3.11/bin/python3 scripts/imaging_ab_realtime_runner.py \
  ...same args... \
  --max-records-per-cycle 1 \
  --run-once
```

Runner state file stores processed records and latest calibration metrics.

## 4) WAL Operations Log (2026-03-07 UTC)

### Active-reference calibration capture

A short single-TX calibration run was executed using `active_reference_cal` at `12 MHz`:

```bash
./borealis_env3.11/bin/python3 scripts/steamed_hams.py \
  active_reference_cal release discretionary \
  --kwargs freq=12000 tx_ant=9 intt=1000 num_ranges=25
```

Capture file:

- `/data/borealis_data/20260307/20260307.0036.24.wal.0.antennas_iq.h5`

Calibration outputs from that capture:

- Main array (crossref, ref 9): `tmp/wal_active_ref_crossref_ref9.npz`
- Interferometer array (crossref, ref 17): `tmp/wal_active_ref_intf_crossref_ref17.npz`

Runtime calibration files pinned to these active-reference products:

- `tmp/realtime_selfcal.npz` (main)
- `tmp/realtime_intf_selfcal.npz` (intf)

### Weekend full-FOV run configuration

Radar set to `full_fov` at `12 MHz`:

```bash
./borealis_env3.11/bin/python3 scripts/steamed_hams.py \
  full_fov release common --kwargs freq=12000
```

Verified on 2026-03-07 UTC that both outputs are being written under FullFOV:

- `*.antennas_iq.h5`
- `*.rawacf`

Quick verification command:

```bash
ls -lt /data/borealis_data/$(date -u +%Y%m%d)
```

and (experiment metadata check):

```bash
./borealis_env3.11/bin/python3 - <<'PY'
import glob, h5py
f = sorted(glob.glob('/data/borealis_data/*/*.antennas_iq.h5'))[-1]
with h5py.File(f, 'r') as h:
    v = h['metadata']['experiment_name'][()]
    print(f, v.decode() if isinstance(v, bytes) else v)
PY
```
