#!/usr/bin/python

"""
restart_borealis
~~~~~~~~~~~~~~

Python script to check data being written and restart Borealis in case it's not

:copyright: 2020 SuperDARN Canada
:author: Kevin Krieger
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime as dt, timedelta, timezone
from email.message import EmailMessage
from email.utils import formatdate
from pathlib import Path
from textwrap import indent


PRODUCTION_DATA_DIRECTORY = "/data/borealis_data"
HOME_LOG_DIR = Path("/home/radar/logs")
BOREALIS_LOG_DIR = Path("/data/borealis_logs")
LOG_TAIL_LINES = 80
EMAILS_FILE = Path("/home/radar/emails.txt")
ATTACHMENT_LOG_COUNT_HOME = 4
ATTACHMENT_LOG_COUNT_BOREALIS = 6
SKIP_LOG_NAMES = {"slack_dataflow_notif.log", "katscan_barker13_supervisor.log"}
ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


def utc_now():
    return dt.now(timezone.utc)


def format_timestamp(timestamp):
    return dt.fromtimestamp(timestamp, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_command(command):
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    output, error = process.communicate()
    return process.returncode, output, error


def newest_files(directory, limit):
    try:
        files = [path for path in directory.iterdir() if path.is_file()]
    except OSError:
        return []
    return sorted(files, key=lambda path: path.stat().st_mtime, reverse=True)[:limit]


def is_readable_log(path):
    if path.name in SKIP_LOG_NAMES:
        return False
    try:
        sample = path.read_text(errors="replace")[:4000]
    except OSError:
        return False
    if not sample.strip():
        return False
    if "?[" in sample:
        return False
    if ANSI_ESCAPE_RE.search(sample):
        return False
    return True


def gather_log_attachments():
    attachments = []

    home_count = 0
    for path in newest_files(HOME_LOG_DIR, limit=ATTACHMENT_LOG_COUNT_HOME * 4):
        if not is_readable_log(path):
            continue
        attachments.append(path)
        home_count += 1
        if home_count >= ATTACHMENT_LOG_COUNT_HOME:
            break

    borealis_count = 0
    for path in newest_files(BOREALIS_LOG_DIR, limit=ATTACHMENT_LOG_COUNT_BOREALIS * 4):
        if not is_readable_log(path):
            continue
        attachments.append(path)
        borealis_count += 1
        if borealis_count >= ATTACHMENT_LOG_COUNT_BOREALIS:
            break

    return attachments


def load_notification_recipients():
    try:
        with open(EMAILS_FILE, "r", errors="replace") as handle:
            recipients = [line.strip() for line in handle if line.strip() and not line.lstrip().startswith("#")]
    except OSError as exc:
        print(f"Unable to read notification recipient file {EMAILS_FILE}: {exc}")
        return []
    return recipients


def send_notification_email(subject, body, attachments):
    contacts = load_notification_recipients()
    if not contacts:
        print(f"No notification recipients found in {EMAILS_FILE}; skipping outage email")
        return False

    sent_any = False
    for address in contacts:
        message = EmailMessage()
        message["Subject"] = subject
        message["From"] = f"radar@{os.uname().nodename}"
        message["To"] = address
        message["Date"] = formatdate(localtime=True)
        message.set_content(body)

        for path in attachments:
            try:
                payload = path.read_bytes()
            except OSError as exc:
                print(f"Failed to read attachment {path}: {exc}")
                continue
            message.add_attachment(
                payload,
                maintype="text",
                subtype="plain",
                filename=path.name,
            )

        process = subprocess.Popen(
            ["/usr/sbin/sendmail", "-t", "-oi"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        _, error = process.communicate(message.as_bytes())
        if process.returncode == 0:
            sent_any = True
        else:
            print(f"Failed to send outage email to {address}: {error.decode(errors='replace')}")
    return sent_any


def build_email_body(hostname, config_path, data_directory, newest_file, last_data_write, action_lines, attachments):
    lines = [
        f"hostname: {hostname}",
        f"utc_time: {utc_now().strftime('%Y-%m-%dT%H:%M:%SZ')}",
        f"config_path: {config_path}",
        f"data_directory: {data_directory}",
        f"newest_seen_data_file: {newest_file}",
        f"newest_seen_data_age_seconds: {last_data_write}",
        "restart_action:",
    ]
    lines.extend(f"  {line}" for line in action_lines)
    lines.append("")
    lines.append("attached_logs:")
    if attachments:
        lines.extend(f"  {path}" for path in attachments)
    else:
        lines.append("  <no readable log attachments selected>")
    return "\n".join(lines).rstrip() + "\n"


def get_latest_data_file(data_directory):
    base_path = Path(data_directory)
    candidate_dirs = []
    now = utc_now()
    for offset_days in (0, -1):
        day = (now + timedelta(days=offset_days)).strftime("%Y%m%d")
        day_dir = base_path / day
        if day_dir.is_dir():
            candidate_dirs.append(day_dir)

    newest_file = None
    newest_file_write_time = None
    for day_dir in candidate_dirs:
        for path in day_dir.iterdir():
            if not path.is_file():
                continue
            path_mtime = path.stat().st_mtime
            if newest_file_write_time is None or path_mtime > newest_file_write_time:
                newest_file = path
                newest_file_write_time = path_mtime

    if newest_file is None:
        start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return None, float(start_of_day.timestamp())
    return newest_file, newest_file_write_time


def get_args():
    if not os.environ["BOREALISPATH"]:
        raise ValueError("BOREALISPATH env variable not set")
    if not os.environ["RADAR_ID"]:
        raise ValueError("RADAR_ID env variable not set")
    borealis_path = os.environ["BOREALISPATH"]
    radar_id = os.environ["RADAR_ID"]

    config_path = f"{borealis_path}/config/{radar_id}/{radar_id}_config.ini"
    try:
        with open(config_path, "r") as data:
            raw_config = json.load(data)
    except IOError:
        raise (f"IOError on config file at {config_path}")

    parser = argparse.ArgumentParser(
        description="Python script to check data being written and restart Borealis in case it's not"
    )
    parser.add_argument(
        "-r",
        "--restart-after-seconds",
        type=int,
        default=300,
        help="How many seconds can the data file be out of date before attempting to restart the radar? Default 300 seconds (5 minutes)",
    )
    parser.add_argument(
        "-p",
        "--borealis-path",
        required=False,
        dest="borealis_path",
        default=borealis_path,
        help="Path to Borealis directory. Default BOREALISPATH environment variable",
    )
    parser.add_argument(
        "-d",
        "--data-directory",
        required=False,
        dest="data_directory",
        default=raw_config["data_directory"],
        help="Path to Borealis data directory. Defaults to data_directory within config file",
    )
    parser.add_argument(
        "--config-path",
        required=False,
        dest="config_path",
        default=config_path,
        help="Path to Borealis config file. Defaults to config/<RADAR_ID>/<RADAR_ID>_config.ini",
    )
    return parser.parse_args()


def main():
    args = get_args()
    restart_after_seconds = args.restart_after_seconds
    borealis_path = args.borealis_path
    data_directory = args.data_directory
    config_path = args.config_path
    hostname = os.uname().nodename

    if data_directory != PRODUCTION_DATA_DIRECTORY:
        print(
            f"Configured data_directory is {data_directory}, not {PRODUCTION_DATA_DIRECTORY}; "
            "assuming non-operational mode and skipping restart check"
        )
        sys.exit(0)

    newest_file, new_file_write_time = get_latest_data_file(data_directory)
    now_utc_seconds = float(utc_now().timestamp())
    last_data_write = round(now_utc_seconds - new_file_write_time, 2)
    newest_file_str = str(newest_file) if newest_file else "<none found for current/previous UTC day>"

    print(
        f"Last write time: {format_timestamp(new_file_write_time)}, "
        f"Current time: {format_timestamp(now_utc_seconds)}, "
        f"Difference: {last_data_write} s, "
        f"Newest file: {newest_file_str}"
    )

    if float(last_data_write) <= float(restart_after_seconds):
        print(
            f"{last_data_write} s within {restart_after_seconds} s threshold - no restart necessary"
        )
        sys.exit(0)

    print(
        f"{last_data_write} s greater than {restart_after_seconds} s threshold - attempting to restart Borealis"
    )

    action_lines = [
        f"stale data threshold exceeded ({last_data_write} s > {restart_after_seconds} s)",
        f"using start script {borealis_path}/scripts/start_radar.sh",
    ]

    stop_returncode, stop_output, stop_error = run_command([f"{borealis_path}/scripts/stop_radar.sh"])
    print("Borealis stop_radar.sh called")
    print(indent(stop_output, "    "))
    if stop_error:
        print("Error with stop_radar.sh:")
        print(indent(stop_error, "      "))
    action_lines.append(f"stop_radar.sh return code: {stop_returncode}")

    time.sleep(1)

    start_returncode, start_output, start_error = run_command([f"{borealis_path}/scripts/start_radar.sh"])
    print("Borealis start_radar.sh called")
    print(indent(start_output, "    "))
    if start_error:
        print("Error with start_radar:")
        print(indent(start_error, "      "))
    action_lines.append(f"start_radar.sh return code: {start_returncode}")

    attachments = gather_log_attachments()
    subject = (
        f"Borealis outage detected on {hostname} at {utc_now().strftime('%Y-%m-%dT%H:%M:%SZ')}; restarting"
    )
    body = build_email_body(
        hostname=hostname,
        config_path=config_path,
        data_directory=data_directory,
        newest_file=newest_file_str,
        last_data_write=last_data_write,
        action_lines=action_lines,
        attachments=[str(path) for path in attachments],
    )
    send_notification_email(subject, body, attachments)

    sys.exit(1)


if __name__ == "__main__":
    main()
