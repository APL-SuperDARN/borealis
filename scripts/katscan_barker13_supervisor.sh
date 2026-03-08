#!/bin/bash
export TERM=xterm
set -u

export BOREALISPATH=/home/radar/borealis
export PYTHON_VERSION=3.11
export RADAR_ID=wal

LOGFILE=/home/radar/logs/katscan_barker13_supervisor.log
mkdir -p /home/radar/logs

echo "$(date -u +%F\ %T)Z supervisor started" >> "$LOGFILE"

while true; do
  # Healthy run: C++ USRP driver is alive
  if pgrep -x usrp_driver >/dev/null; then
    sleep 30
    continue
  fi

  echo "$(date -u +%F\ %T)Z attempting steamed_hams launch" >> "$LOGFILE"

  screen -S borealis -X quit >/dev/null 2>&1 || true
  pkill -f 'src/(usrp_driver|rx_signal_processing|data_write|radar_control|realtime|brian)\.py' >/dev/null 2>&1 || true
  pkill -x usrp_driver >/dev/null 2>&1 || true

  # Use script(1) to provide a TTY required by steamed_hams/screen.
  timeout 120 script -q -c 'cd /home/radar/borealis && ./borealis_env3.11/bin/python3 scripts/steamed_hams.py katscan_barker13 release common --kwargs freq=12000' /dev/null >> "$LOGFILE" 2>&1 || true

  sleep 25
done
