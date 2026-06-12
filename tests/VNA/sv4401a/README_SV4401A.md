# SV4401A Wallops measurement script

## Files
- `sv4401a_wallops_measure.py` - guided per-channel installed-path capture script

## Platforms
This script is intended to work on:
- macOS
- Ubuntu Linux

It uses:
- Python 3
- `pyserial`
- the documented SV4401A USB serial console API

Install dependency:
```bash
python3 -m pip install pyserial
```

## What it does
- detects likely USB serial ports on macOS or Ubuntu
- guides the user through the physical OSL calibration sequence in the correct order:
  - OPEN
  - SHORT
  - LOAD
- sends the documented `cal` console commands to the SV4401A
- performs a basic calibration sanity check by measuring the LOAD after calibration
- walks channel-by-channel through `ANT1`-`ANT16` and `INT1`-`INT4` or a requested subset
- prompts the user before each physical reconnection
- captures one installed-path `S11` sweep per channel
- writes:
  - per-channel CSV
  - per-channel `.s1p`
  - per-channel JSON summary
  - session summary JSON

## Important limitation
The manual clearly documents automation for VNA sweep collection and calibration.
It does **not** clearly document console automation for:
- true passive spectrum-analyzer mode
- direct TDR trace export over the serial command API

So this script is currently a guided, reliable `S11` / `VSWR` / `return-loss` automation tool.

## Calibration confirmation
The SV4401A manual documents `cal open`, `cal short`, `cal load`, `cal done`, and `cal on`, but it does not document a direct console query for calibration state.

So the script confirms calibration in the best practical way available from the documented API:
- it asks you to leave the `50 ohm` LOAD connected after calibration
- it captures a check sweep
- it reports the resulting best return loss and minimum VSWR
- it marks the check as `PASS` if the load looks reasonably good

This is a sanity check, not a formal instrument self-report.

## Example
```bash
python3 tmp/sv4401a_scripts/sv4401a_wallops_measure.py --port /dev/ttyUSB0
```

On macOS, the port may look like:
```bash
python3 tmp/sv4401a_scripts/sv4401a_wallops_measure.py --port /dev/cu.usbmodemXXXX
```

Subset example:
```bash
python3 tmp/sv4401a_scripts/sv4401a_wallops_measure.py --channels ANT1,ANT2,ANT3,INT1
```

Skip calibration example:
```bash
python3 tmp/sv4401a_scripts/sv4401a_wallops_measure.py --skip-calibration
```
