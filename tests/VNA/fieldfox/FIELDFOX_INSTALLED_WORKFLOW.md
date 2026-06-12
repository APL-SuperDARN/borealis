# FieldFox Installed-Path Workflow

These scripts are designed to run with plain `python3` on either:

- macOS
- Ubuntu 20.04 on `borealis`

They use only the Python standard library.

## Scripts

- `fieldfox_measure_installed_batch.py`
  - full arrival workflow
  - configures the FieldFox for installed-path screening
  - pauses for front-panel OSL calibration
  - walks through all `16` main-array channels and `4` interferometer channels
  - writes a ranked summary plus SVG plots and CSV tables under `analysis/`
- `fieldfox_measure_installed_channel.py`
  - per-channel rerun tool
  - captures one installed-path set and prints a quick health summary

Shared helpers live in `fieldfox_installed_common.py`.

## Default assumptions

- FieldFox IP: `192.168.10.1`
- SCPI socket port: auto-tries `5025` then `5024`
- Mode: `CAT/TDR`
- Return loss sweep: `8-20 MHz`
- DTF frequency span: `8-20 MHz`
- Points: `801`
- Output power: `-15 dBm`
- Velocity factor: `0.85`
- Main-array DTF stop distance: `175 m`
- Interferometer DTF stop distance: `275 m`

The batch script uses a full preset by default before setup. That is appropriate for the initial baseline run because the next step is to perform a fresh OSL calibration.

The single-channel script does not preset or reconfigure the instrument unless you ask it to.

## Typical use

Initial baseline run:

```bash
python3 wallops_2026_04/scripts/fieldfox_measure_installed_batch.py --radar WAL
```

Single-channel rerun after the instrument is already calibrated and configured:

```bash
python3 wallops_2026_04/scripts/fieldfox_measure_installed_channel.py \
  --radar WAL \
  --kind antenna \
  --channel 3
```

Single-channel rerun that also reapplies the installed-path setup first:

```bash
python3 wallops_2026_04/scripts/fieldfox_measure_installed_channel.py \
  --radar WAL \
  --kind interferometer \
  --channel 2 \
  --configure-setup
```

## Output locations

Per-channel measurement files are written directly into:

- `antenna_<n>/from_control_room/`
- `interferometer_<n>/from_control_room/`

The batch summary is written under a new directory in:

- `analysis/installed_screening_<YYYYMMDD>/`

That summary directory contains:

- `screening_report.txt`
- `main_array_summary.csv`
- `interferometer_summary.csv`
- SVG overlays for VSWR, return loss, and DTF
- SVG ranking plots

## Notes

- The `.sta` file is retrieved from the FieldFox memory after the save command completes.
- The `.s1p` and `VSWR.csv` files are generated locally from the measured complex S11 data.
- The `DTF.csv` file is saved by the FieldFox and then copied locally.
- If you want to preserve the current front-panel state and calibration, use the single-channel script without `--preset-first` and without `--configure-setup`.
