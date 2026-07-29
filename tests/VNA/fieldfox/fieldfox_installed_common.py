#!/usr/bin/env python3

from __future__ import annotations

import datetime as dt
import html
import math
import socket
import statistics
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR

DEFAULT_FIELDFOX_HOST = "192.168.10.1"
DEFAULT_FIELDFOX_PORT = 5025
DEFAULT_FIELDFOX_PORT_FALLBACKS = (5025, 5024)
DEFAULT_TIMEOUT_SEC = 12.0

DEFAULT_START_HZ = 8_000_000.0
DEFAULT_STOP_HZ = 20_000_000.0
DEFAULT_POINTS = 801
DEFAULT_IF_BW_HZ = 10_000.0
DEFAULT_POWER_DBM = -15.0
DEFAULT_VELOCITY_FACTOR = 0.85

DEFAULT_MAIN_STOP_M = 175.0
DEFAULT_INTERFEROMETER_STOP_M = 275.0

VSWR_TARGET_MHZ = 14.25

PALETTE = [
    "#0b5d5e",
    "#c8553d",
    "#5c7c2f",
    "#7a4eab",
    "#0077b6",
    "#b56576",
    "#2a9d8f",
    "#f4a261",
    "#6d597a",
    "#8d0801",
    "#4c956c",
    "#6c757d",
    "#283618",
    "#9c6644",
    "#007f5f",
    "#1d3557",
    "#9a031e",
    "#3a86ff",
    "#fb8500",
    "#588157",
]


class FieldFoxError(RuntimeError):
    pass


@dataclass
class MeasurementSettings:
    start_hz: float = DEFAULT_START_HZ
    stop_hz: float = DEFAULT_STOP_HZ
    points: int = DEFAULT_POINTS
    if_bw_hz: float = DEFAULT_IF_BW_HZ
    power_dbm: float = DEFAULT_POWER_DBM
    velocity_factor: float = DEFAULT_VELOCITY_FACTOR


@dataclass
class ChannelSpec:
    kind: str
    number: int

    def token(self) -> str:
        return "ANT" if self.kind == "antenna" else "INT"

    def noun(self) -> str:
        return "Antenna" if self.kind == "antenna" else "Interferometer"

    def label(self) -> str:
        return "%s %d" % (self.noun(), self.number)

    def short_label(self) -> str:
        return "%s%d" % (self.token(), self.number)

    def output_dir(self, root: Path) -> Path:
        if self.kind == "antenna":
            return root / ("antenna_%d" % self.number) / "from_control_room"
        return root / ("interferometer_%d" % self.number) / "from_control_room"

    def distance_stop_m(self) -> float:
        if self.kind == "antenna":
            return DEFAULT_MAIN_STOP_M
        return DEFAULT_INTERFEROMETER_STOP_M


@dataclass
class CapturePaths:
    stem: str
    state_path: Path
    s1p_path: Path
    dtf_path: Path
    vswr_path: Path

    def error_log_path(self) -> Path:
        return self.state_path.with_name(self.state_path.stem + "_fieldfox_errors.log")


@dataclass
class FieldFoxErrorEvent:
    timestamp: str
    context: str
    message: str


@dataclass
class CalibrationStatus:
    rl_state: str
    dtf_state: str
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class S1PData:
    freq_hz: List[float]
    s11: List[complex]
    rl_db: List[float]
    vswr: List[float]


@dataclass
class DTFData:
    freq_hz: List[float]
    rl_db: List[float]
    dist_m: List[float]
    dtf_db: List[float]
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class ChannelSummary:
    channel: str
    kind: str
    healthy: bool
    verdict: str
    min_vswr: float
    min_vswr_freq_mhz: float
    avg_vswr: float
    best_rl_db: float
    best_rl_freq_mhz: float
    rl_14_db: float
    vswr_14: float
    end_dtf_dist_m: float
    end_dtf_db: float
    mid_dtf_dist_m: float
    mid_dtf_db: float
    estimated_length_m: float
    quick_score: float
    urgency_score: float
    notes: List[str]
    vswr_rmse_to_ref: Optional[float] = None
    rl_rmse_to_ref: Optional[float] = None
    vswr_rmse_to_peer_median: Optional[float] = None


@dataclass
class ChannelResult:
    spec: ChannelSpec
    paths: CapturePaths
    s1p: S1PData
    dtf: DTFData
    summary: ChannelSummary


def repo_root() -> Path:
    return ROOT


def format_float(value: float, digits: int) -> str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf"
    return ("%0." + str(digits) + "f") % value


def format_hz(value: float) -> str:
    if abs(value) >= 1e6:
        return "%s MHz" % format_float(value / 1e6, 3)
    if abs(value) >= 1e3:
        return "%s kHz" % format_float(value / 1e3, 3)
    return "%s Hz" % format_float(value, 3)


def iso_timestamp() -> str:
    return dt.datetime.now().replace(microsecond=0).isoformat()


def normalize_radar_id(value: str) -> str:
    return "".join(ch for ch in value.strip().upper() if ch.isalnum() or ch in ("_", "-"))


def default_date_tag() -> str:
    return dt.date.today().strftime("%Y%m%d")


def build_stem(radar_id: str, spec: ChannelSpec) -> str:
    return "%s_%s_%d_INSTALLED" % (radar_id, spec.token(), spec.number)


def capture_paths_for_channel(root: Path, radar_id: str, spec: ChannelSpec, date_tag: Optional[str]) -> CapturePaths:
    radar_id = normalize_radar_id(radar_id)
    date_tag = date_tag or default_date_tag()
    output_dir = spec.output_dir(root)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = build_stem(radar_id, spec)
    sequence = 1
    while True:
        suffix = "_%02d" % sequence if sequence > 1 else ""
        state_path = output_dir / ("%s_%s%s.sta" % (stem, date_tag, suffix))
        s1p_path = output_dir / ("%s_S11_%s%s.s1p" % (stem, date_tag, suffix))
        dtf_path = output_dir / ("%s_DTF_%s%s.csv" % (stem, date_tag, suffix))
        vswr_path = output_dir / ("%s_VSWR_%s%s.csv" % (stem, date_tag, suffix))
        if not (state_path.exists() or s1p_path.exists() or dtf_path.exists() or vswr_path.exists()):
            return CapturePaths(
                stem=stem,
                state_path=state_path,
                s1p_path=s1p_path,
                dtf_path=dtf_path,
                vswr_path=vswr_path,
            )
        sequence += 1


def frequency_axis(settings: MeasurementSettings) -> List[float]:
    if settings.points <= 1:
        return [settings.start_hz]
    step = (settings.stop_hz - settings.start_hz) / float(settings.points - 1)
    return [settings.start_hz + idx * step for idx in range(settings.points)]


class FieldFoxClient:
    def __init__(self, host: str, port: int = DEFAULT_FIELDFOX_PORT, timeout_sec: float = DEFAULT_TIMEOUT_SEC) -> None:
        self.host = host
        self.port = port
        self.timeout_sec = timeout_sec
        self.sock = socket.create_connection((host, port), timeout_sec)
        self.sock.settimeout(timeout_sec)
        self.fp = self.sock.makefile("rwb", buffering=0)
        self.error_history: List[FieldFoxErrorEvent] = []
        self.distance_stop_m: Optional[float] = None

    def close(self) -> None:
        try:
            self.fp.close()
        finally:
            self.sock.close()

    def __enter__(self) -> "FieldFoxClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _send(self, command: str) -> None:
        payload = command.strip().encode("ascii") + b"\n"
        self.fp.write(payload)
        self.fp.flush()

    def _read_exact(self, nbytes: int) -> bytes:
        chunks = []
        remaining = nbytes
        while remaining > 0:
            data = self.fp.read(remaining)
            if not data:
                raise FieldFoxError("Unexpected EOF while reading %d bytes from FieldFox" % nbytes)
            chunks.append(data)
            remaining -= len(data)
        return b"".join(chunks)

    def query(self, command: str) -> str:
        self._send(command)
        data = self.fp.readline()
        if not data:
            raise FieldFoxError("No response received for %s" % command)
        return data.decode("utf-8", "replace").strip()

    def query_ascii_values(self, command: str) -> List[float]:
        text = self.query(command)
        values = []
        for piece in text.split(","):
            token = piece.strip()
            if not token:
                continue
            values.append(float(token))
        return values

    def query_binary_block(self, command: str) -> bytes:
        self._send(command)
        prefix = self.fp.read(1)
        if not prefix:
            raise FieldFoxError("No response received for %s" % command)
        if prefix != b"#":
            tail = self.fp.readline()
            text = (prefix + tail).decode("utf-8", "replace").strip()
            raise FieldFoxError("Expected IEEE block data for %s, got %r" % (command, text))
        digits_char = self.fp.read(1)
        if not digits_char:
            raise FieldFoxError("Truncated block header for %s" % command)
        try:
            digits = int(digits_char.decode("ascii"))
        except ValueError:
            raise FieldFoxError("Invalid block header digit for %s: %r" % (command, digits_char))
        if digits == 0:
            chunks = []
            while True:
                data = self.fp.readline()
                if not data:
                    break
                chunks.append(data)
                if data.endswith(b"\n"):
                    break
            return b"".join(chunks).rstrip(b"\r\n")
        size = int(self._read_exact(digits).decode("ascii"))
        payload = self._read_exact(size)
        terminator = self.fp.read(1)
        if terminator == b"\r":
            self.fp.read(1)
        elif terminator not in (b"", b"\n"):
            pass
        return payload

    def command_errors(self, command: str) -> List[str]:
        self._send(command)
        completed = self.query("*OPC?")
        if completed.strip() != "1":
            raise FieldFoxError("Unexpected *OPC? response after %s: %r" % (command, completed))
        return self.system_errors(context=command)

    def command(self, command: str, required: bool = True) -> bool:
        errors = self.command_errors(command)
        if errors and required:
            raise FieldFoxError("FieldFox error after %s: %s" % (command, "; ".join(errors)))
        return not errors

    def system_errors(self, context: str = "system poll") -> List[str]:
        errors = []
        while True:
            try:
                response = self.query("SYST:ERR?")
            except Exception:
                return errors
            if not response:
                return errors
            normalized = response.replace('"', "").strip()
            if normalized.startswith("+0") or normalized.startswith("0"):
                return errors
            errors.append(normalized)
            self.error_history.append(
                FieldFoxErrorEvent(
                    timestamp=iso_timestamp(),
                    context=context,
                    message=normalized,
                )
            )

    def error_events(self) -> List[FieldFoxErrorEvent]:
        return list(self.error_history)

    def idn(self) -> str:
        return self.query("*IDN?")

    def single_sweep(self) -> None:
        response = self.query("INIT:IMM;*OPC?")
        if response.strip() != "1":
            raise FieldFoxError("Unexpected sweep completion response: %r" % response)

    def save_remote_file(self, save_command: str, remote_name: str, local_path: Path) -> None:
        quoted = remote_name.replace('"', "")
        self.command('%s "%s"' % (save_command, quoted))
        payload = self.query_binary_block('MMEM:DATA? "%s"' % quoted)
        local_path.write_bytes(payload)


def open_fieldfox(host: str, port: int = DEFAULT_FIELDFOX_PORT, timeout_sec: float = DEFAULT_TIMEOUT_SEC) -> FieldFoxClient:
    return FieldFoxClient(host=host, port=port, timeout_sec=timeout_sec)


def open_fieldfox_with_fallback(
    host: str,
    port: Optional[int] = None,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
) -> FieldFoxClient:
    ports_to_try = [port] if port is not None else list(DEFAULT_FIELDFOX_PORT_FALLBACKS)
    errors = []
    for candidate in ports_to_try:
        try:
            return FieldFoxClient(host=host, port=candidate, timeout_sec=timeout_sec)
        except Exception as exc:
            errors.append("%s:%d -> %s" % (host, candidate, exc))
    raise FieldFoxError("Unable to connect to FieldFox. Tried %s" % "; ".join(errors))


def _run_first_supported(client: FieldFoxClient, commands: Sequence[str], description: str) -> str:
    failures = []
    for command in commands:
        try:
            if client.command(command, required=False):
                return command
            failures.append("%s -> not supported" % command)
        except Exception as exc:
            failures.append("%s -> %s" % (command, exc))
    raise FieldFoxError("No supported SCPI form for %s. Tried %s" % (description, "; ".join(failures)))


def _select_trace(client: FieldFoxClient, trace_index: int) -> str:
    return _run_first_supported(
        client,
        (
            "CALC:PAR%d:SEL" % trace_index,
            "CALC:PAR:SEL %d" % trace_index,
            "CALC1:PAR%d:SEL" % trace_index,
        ),
        "trace %d selection" % trace_index,
    )


def _select_mode(client: FieldFoxClient, mode_name: str) -> str:
    return _run_first_supported(
        client,
        (
            'INST "%s"' % mode_name,
            "INST '%s'" % mode_name,
            'INST:SEL "%s"' % mode_name,
            "INST:SEL '%s'" % mode_name,
            "INST:SEL %s" % mode_name,
        ),
        "instrument mode %s" % mode_name,
    )


def _disable_all_correction(client: FieldFoxClient) -> str:
    return _run_first_supported(
        client,
        (
            "SENS:CORR:STAT OFF",
            "SENS:CORR OFF",
            "CORR OFF",
        ),
        "correction disable",
    )


def _configure_with_dtf2(client: FieldFoxClient) -> None:
    _select_mode(client, "CAT")
    client.command("CALC:PAR:DEF DTF2")
    _select_trace(client, 1)
    _select_trace(client, 2)
    _select_trace(client, 1)


def _configure_with_manual_traces(client: FieldFoxClient) -> None:
    _select_mode(client, "CAT")
    client.command("CALC:PAR:COUN 2")
    _select_trace(client, 1)
    client.command("CALC:PAR:DEF RLOS")
    _select_trace(client, 2)
    client.command("CALC:PAR:DEF DTF1")
    _select_trace(client, 1)


def configure_installed_measurement(
    client: FieldFoxClient,
    settings: MeasurementSettings,
    distance_stop_m: float,
    preset_first: bool = False,
) -> None:
    if preset_first:
        client.command("SYST:PRES")
    client.command("FORM:DATA ASC,0", required=False)
    try:
        _configure_with_dtf2(client)
    except Exception:
        _configure_with_manual_traces(client)
    _disable_all_correction(client)
    client.command("SOUR:POW %s" % format_float(settings.power_dbm, 2), required=False)
    client.command("SENS:BWID %.0f" % settings.if_bw_hz, required=False)
    client.command("CALC:SMO 0", required=False)
    client.command("SENS:CORR:RVEL:COAX %.5f" % settings.velocity_factor, required=False)
    _select_trace(client, 2)
    client.command("CALC:TRAN:FREQ BPAS")
    client.command("CALC:TRAN:DIST:UNIT MET", required=False)
    client.command("CALC:TRAN:DIST:STAR 0", required=False)
    client.command("CALC:TRAN:DIST:STOP %s" % format_float(distance_stop_m, 3))
    client.command("CALC:TRAN:DIST:WIND KBES", required=False)
    client.command("SENS:FREQ:STAR %.0f" % settings.start_hz)
    client.command("SENS:FREQ:STOP %.0f" % settings.stop_hz)
    client.command("SENS:SWE:POIN %d" % settings.points)
    _select_trace(client, 1)
    client.distance_stop_m = distance_stop_m
    client.command("INIT:CONT OFF")
    client.single_sweep()


def set_distance_stop(
    client: FieldFoxClient,
    distance_stop_m: float,
    settings: Optional[MeasurementSettings] = None,
    allow_calibration_disable: bool = False,
) -> bool:
    settings = settings or MeasurementSettings()
    if client.distance_stop_m is not None and abs(client.distance_stop_m - distance_stop_m) < 0.05:
        _select_trace(client, 1)
        return False
    _select_trace(client, 2)
    stop_command = "CALC:TRAN:DIST:STOP %s" % format_float(distance_stop_m, 3)
    errors = client.command_errors(stop_command)
    if errors:
        if not (
            allow_calibration_disable
            and all("stimulus outside calibrated range" in error.lower() for error in errors)
        ):
            raise FieldFoxError("FieldFox error after %s: %s" % (stop_command, "; ".join(errors)))
    client.command("SENS:FREQ:STAR %.0f" % settings.start_hz)
    client.command("SENS:FREQ:STOP %.0f" % settings.stop_hz)
    client.command("SENS:SWE:POIN %d" % settings.points)
    _select_trace(client, 1)
    client.distance_stop_m = distance_stop_m
    if errors:
        return True
    client.single_sweep()
    return False


def capture_s11_from_instrument(
    client: FieldFoxClient,
    settings: MeasurementSettings,
) -> S1PData:
    def normalize_trace_values(data: Sequence[float], points: int, label: str) -> List[float]:
        if len(data) == points:
            return list(data)
        if len(data) == points * 2:
            evens = list(data[0::2])
            odds = list(data[1::2])
            if safe_mean(abs(value) for value in odds) < 1e-9:
                return evens
            if safe_mean(abs(value) for value in evens) < 1e-9:
                return odds
        raise FieldFoxError("Expected %d %s values, got %d" % (points, label, len(data)))

    client.command("FORM:DATA ASC,0", required=False)
    _select_trace(client, 1)
    try:
        client.command("CALC:FORM MLOG", required=False)
        logmag_data = normalize_trace_values(
            client.query_ascii_values("CALC:DATA:FDAT?"),
            settings.points,
            "log-magnitude",
        )
        client.command("CALC:FORM PHAS", required=False)
        phase_deg_data = normalize_trace_values(
            client.query_ascii_values("CALC:DATA:FDAT?"),
            settings.points,
            "phase",
        )
        client.command("CALC:FORM MLOG", required=False)
        s11 = []
        for logmag_value, phase_deg in zip(logmag_data, phase_deg_data):
            if logmag_value > 0.0:
                mag = 10.0 ** (-logmag_value / 20.0)
            else:
                mag = 10.0 ** (logmag_value / 20.0)
            angle_rad = math.radians(phase_deg)
            s11.append(complex(mag * math.cos(angle_rad), mag * math.sin(angle_rad)))
    except Exception as exc:
        try:
            data = client.query_ascii_values("CALC:DATA:SDAT?")
            if len(data) != settings.points * 2:
                raise FieldFoxError("Expected %d SDATA values, got %d" % (settings.points * 2, len(data)))
            s11 = []
            idx = 0
            while idx < len(data):
                s11.append(complex(data[idx], data[idx + 1]))
                idx += 2
        except Exception as sdat_exc:
            raise FieldFoxError(
                "Unable to read S11 data using CALC:DATA:FDAT? or CALC:DATA:SDAT?: %s / %s"
                % (exc, sdat_exc)
            )
    freq_hz = frequency_axis(settings)
    rl_db = []
    vswr = []
    for gamma in s11:
        mag = abs(gamma)
        rl_db.append(-20.0 * math.log10(max(mag, 1e-15)))
        if mag >= 1.0:
            vswr.append(float("inf"))
        else:
            vswr.append((1.0 + mag) / (1.0 - mag))
    return S1PData(freq_hz=freq_hz, s11=s11, rl_db=rl_db, vswr=vswr)


def write_touchstone(
    path: Path,
    s1p: S1PData,
    instrument_id: str,
    radar_id: str,
    spec: ChannelSpec,
) -> None:
    lines = [
        "! Generated by fieldfox_measure_installed_channel.py",
        "! Instrument: %s" % instrument_id,
        "! Radar: %s" % radar_id,
        "! Channel: %s" % spec.label(),
        "! Timestamp: %s" % iso_timestamp(),
        "# Hz S DB R 50",
    ]
    for freq_hz, gamma in zip(s1p.freq_hz, s1p.s11):
        mag = max(abs(gamma), 1e-15)
        db_value = 20.0 * math.log10(mag)
        angle_deg = math.degrees(math.atan2(gamma.imag, gamma.real))
        lines.append(
            "%s %s %s"
            % (
                format_float(freq_hz, 6),
                format_float(db_value, 6),
                format_float(angle_deg, 6),
            )
        )
    path.write_text("\n".join(lines) + "\n")


def write_vswr_csv(
    path: Path,
    s1p: S1PData,
    instrument_id: str,
    radar_id: str,
    spec: ChannelSpec,
    source_s1p: Path,
) -> None:
    lines = [
        "! FILETYPE CSV",
        "! GENERATED_BY fieldfox_measure_installed_channel.py",
        "! TIMESTAMP %s" % iso_timestamp(),
        "! INSTRUMENT %s" % instrument_id,
        "! RADAR %s" % radar_id,
        "! CHANNEL %s" % spec.short_label(),
        "! SOURCE_S11 %s" % source_s1p.name,
        "! DATA Freq,VSWR",
        "! FREQ UNIT Hz",
        "! DATA UNIT VSWR",
        "BEGIN",
    ]
    for freq_hz, vswr in zip(s1p.freq_hz, s1p.vswr):
        lines.append("%s,%s" % (format_float(freq_hz, 6), format_float(vswr, 12)))
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")


def parse_fieldfox_csv_metadata(path: Path) -> Dict[str, str]:
    metadata = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line.startswith("! "):
            continue
        text = line[2:]
        if "," in text:
            key, value = text.split(",", 1)
        elif " " in text:
            key, value = text.split(" ", 1)
        else:
            continue
        metadata[key.strip()] = value.strip()
    return metadata


def parse_s1p(path: Path) -> S1PData:
    fmt = None
    unit_scale = 1.0
    freq_hz = []
    s11 = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("!"):
            continue
        if line.startswith("#"):
            parts = line.split()
            if len(parts) >= 4:
                unit = parts[1].upper()
                fmt = parts[3].upper()
                unit_scale = {
                    "HZ": 1.0,
                    "KHZ": 1e3,
                    "MHZ": 1e6,
                    "GHZ": 1e9,
                }.get(unit, 1.0)
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        freq = float(parts[0]) * unit_scale
        val1 = float(parts[1])
        val2 = float(parts[2])
        if fmt == "DB":
            mag = 10.0 ** (val1 / 20.0)
            angle_rad = math.radians(val2)
            gamma = complex(mag * math.cos(angle_rad), mag * math.sin(angle_rad))
        elif fmt == "RI":
            gamma = complex(val1, val2)
        elif fmt == "MA":
            angle_rad = math.radians(val2)
            gamma = complex(val1 * math.cos(angle_rad), val1 * math.sin(angle_rad))
        else:
            raise ValueError("Unsupported Touchstone format in %s" % path)
        freq_hz.append(freq)
        s11.append(gamma)
    rl_db = []
    vswr = []
    for gamma in s11:
        mag = abs(gamma)
        rl_db.append(-20.0 * math.log10(max(mag, 1e-15)))
        if mag >= 1.0:
            vswr.append(float("inf"))
        else:
            vswr.append((1.0 + mag) / (1.0 - mag))
    return S1PData(freq_hz=freq_hz, s11=s11, rl_db=rl_db, vswr=vswr)


def parse_dtf_csv(path: Path) -> DTFData:
    metadata = parse_fieldfox_csv_metadata(path)
    current_kind = None
    in_block = False
    freq_hz = []
    rl_db = []
    dist_m = []
    dtf_db = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("! DATA "):
            if "Dist" in line:
                current_kind = "Dist"
            elif "Freq" in line:
                current_kind = "Freq"
            continue
        if line == "BEGIN":
            in_block = True
            continue
        if line == "END":
            in_block = False
            continue
        if not in_block or line.startswith("!"):
            continue
        x_text, y_text = line.split(",", 1)
        x_value = float(x_text)
        y_value = float(y_text)
        if current_kind == "Freq":
            freq_hz.append(x_value)
            rl_db.append(y_value)
        elif current_kind == "Dist":
            dist_m.append(x_value)
            dtf_db.append(y_value)
    if not dist_m:
        raise ValueError("No distance-domain DTF block found in %s" % path)
    return DTFData(freq_hz=freq_hz, rl_db=rl_db, dist_m=dist_m, dtf_db=dtf_db, metadata=metadata)


def interp_at(x_values: Sequence[float], y_values: Sequence[float], x_target: float) -> float:
    if not x_values:
        return float("nan")
    if x_target <= x_values[0]:
        return y_values[0]
    if x_target >= x_values[-1]:
        return y_values[-1]
    index = 1
    lo = 0
    hi = len(x_values) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if x_values[mid] < x_target:
            lo = mid + 1
        else:
            hi = mid - 1
    index = max(1, min(lo, len(x_values) - 1))
    x0 = x_values[index - 1]
    x1 = x_values[index]
    y0 = y_values[index - 1]
    y1 = y_values[index]
    if x1 == x0:
        return y0
    return y0 + (y1 - y0) * (x_target - x0) / (x1 - x0)


def safe_mean(values: Sequence[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return float("inf")
    return sum(finite) / float(len(finite))


def safe_max(values: Sequence[float], default: float) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return default
    return max(finite)


def correction_state_is_on_u(value: str) -> bool:
    return value.strip().upper() == "ON U"


def read_displayed_calibration_status(client: FieldFoxClient) -> CalibrationStatus:
    remote_name = "FIELDFOX_INSTALL_CAL_CHECK.csv"
    with tempfile.TemporaryDirectory(prefix="fieldfox_cal_") as tmpdir:
        temp_path = Path(tmpdir) / remote_name
        _select_trace(client, 2)
        client.save_remote_file("MMEM:STOR:FDAT", remote_name, temp_path)
        metadata = parse_fieldfox_csv_metadata(temp_path)
    _select_trace(client, 1)
    return CalibrationStatus(
        rl_state=metadata.get("CORRECTION1", "").strip(),
        dtf_state=metadata.get("CORRECTION2", "").strip(),
        metadata=metadata,
    )


def ensure_installed_calibration_on_u(
    client: FieldFoxClient,
    prompt_for_retry: bool = True,
) -> CalibrationStatus:
    while True:
        status = read_displayed_calibration_status(client)
        if correction_state_is_on_u(status.rl_state) and correction_state_is_on_u(status.dtf_state):
            return status
        message = (
            "Calibration check failed: RL is %s and DTF is %s. "
            "Both displayed traces must show CAL ON U before continuing."
            % (status.rl_state or "(blank)", status.dtf_state or "(blank)")
        )
        if not prompt_for_retry:
            raise FieldFoxError(message)
        print(message)
        input(
            "Finish or apply the calibration so RL and DTF both show CAL ON U, "
            "then press Enter to re-check. "
        )


def rmse(values_a: Sequence[float], values_b: Sequence[float]) -> float:
    if len(values_a) != len(values_b) or not values_a:
        return float("nan")
    squared = 0.0
    count = 0
    for aval, bval in zip(values_a, values_b):
        if not math.isfinite(aval) or not math.isfinite(bval):
            continue
        diff = aval - bval
        squared += diff * diff
        count += 1
    if count == 0:
        return float("nan")
    return math.sqrt(squared / float(count))


def interpolate_series(x_old: Sequence[float], y_old: Sequence[float], x_new: Sequence[float]) -> List[float]:
    return [interp_at(x_old, y_old, target) for target in x_new]


def mean_series(series_list: Sequence[Sequence[float]]) -> List[float]:
    if not series_list:
        return []
    length = len(series_list[0])
    output = []
    for idx in range(length):
        values = [series[idx] for series in series_list if idx < len(series)]
        output.append(sum(values) / float(len(values)))
    return output


def median_series(series_list: Sequence[Sequence[float]]) -> List[float]:
    if not series_list:
        return []
    length = len(series_list[0])
    output = []
    for idx in range(length):
        values = [series[idx] for series in series_list if idx < len(series)]
        output.append(statistics.median(values))
    return output


def find_region_min(
    x_values: Sequence[float],
    y_values: Sequence[float],
    x_min: float,
    x_max: float,
) -> Tuple[float, float]:
    best_x = float("nan")
    best_y = float("nan")
    for x_value, y_value in zip(x_values, y_values):
        if x_value < x_min or x_value > x_max:
            continue
        if math.isnan(best_y) or y_value < best_y:
            best_x = x_value
            best_y = y_value
    return best_x, best_y


def dtf_windows(dist_m: Sequence[float]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    if not dist_m:
        return (10.0, 100.0), (120.0, 180.0)
    max_dist = max(dist_m)
    end_start = max(120.0, max_dist - 60.0)
    mid_stop = max(20.0, end_start - 5.0)
    return (10.0, mid_stop), (end_start, max_dist)


def absolute_main_score(summary: ChannelSummary) -> float:
    score = 0.0
    if summary.avg_vswr > 4.0:
        score += 70.0
    elif summary.avg_vswr > 3.0:
        score += 45.0
    elif summary.avg_vswr > 2.4:
        score += 25.0
    elif summary.avg_vswr > 2.0:
        score += 10.0

    if summary.min_vswr > 3.0:
        score += 60.0
    elif summary.min_vswr > 2.0:
        score += 35.0
    elif summary.min_vswr > 1.5:
        score += 15.0

    delta_f = abs(summary.min_vswr_freq_mhz - VSWR_TARGET_MHZ)
    if delta_f > 4.0:
        score += 30.0
    elif delta_f > 2.0:
        score += 18.0
    elif delta_f > 1.0:
        score += 8.0

    if summary.rl_14_db < 6.0:
        score += 45.0
    elif summary.rl_14_db < 10.0:
        score += 30.0
    elif summary.rl_14_db < 15.0:
        score += 15.0

    if math.isfinite(summary.mid_dtf_db) and math.isfinite(summary.end_dtf_db):
        if summary.mid_dtf_db <= summary.end_dtf_db + 2.0:
            score += 35.0
        elif summary.mid_dtf_db <= summary.end_dtf_db + 6.0:
            score += 20.0
    return score


def absolute_interferometer_score(summary: ChannelSummary) -> float:
    score = 0.0
    if summary.avg_vswr > 4.0:
        score += 55.0
    elif summary.avg_vswr > 3.0:
        score += 35.0
    elif summary.avg_vswr > 2.4:
        score += 20.0

    if summary.min_vswr > 3.0:
        score += 35.0
    elif summary.min_vswr > 2.0:
        score += 20.0

    if summary.vswr_14 > 1.8:
        score += 25.0
    elif summary.vswr_14 > 1.5:
        score += 12.0

    if summary.rl_14_db < 10.0:
        score += 20.0
    elif summary.rl_14_db < 14.0:
        score += 8.0

    delta_f = abs(summary.min_vswr_freq_mhz - VSWR_TARGET_MHZ)
    if delta_f > 5.0:
        score += 25.0
    elif delta_f > 3.0:
        score += 15.0

    if math.isfinite(summary.mid_dtf_db) and math.isfinite(summary.end_dtf_db):
        if summary.mid_dtf_db <= summary.end_dtf_db + 2.0:
            score += 25.0
        elif summary.mid_dtf_db <= summary.end_dtf_db + 6.0:
            score += 12.0
    return score


def summarize_channel(spec: ChannelSpec, s1p: S1PData, dtf: DTFData) -> ChannelSummary:
    in_band = [
        idx
        for idx, freq_hz in enumerate(s1p.freq_hz)
        if DEFAULT_START_HZ <= freq_hz <= DEFAULT_STOP_HZ
    ]
    if not in_band:
        in_band = list(range(len(s1p.freq_hz)))

    min_vswr = float("inf")
    min_vswr_freq = float("nan")
    for idx in in_band:
        freq_hz = s1p.freq_hz[idx]
        vswr = s1p.vswr[idx]
        if not math.isfinite(vswr):
            continue
        if vswr < min_vswr:
            min_vswr = vswr
            min_vswr_freq = freq_hz / 1e6

    best_rl_idx = max(in_band, key=lambda idx: s1p.rl_db[idx])
    best_rl_db = s1p.rl_db[best_rl_idx]
    best_rl_freq_mhz = s1p.freq_hz[best_rl_idx] / 1e6
    avg_vswr = safe_mean(s1p.vswr[idx] for idx in in_band)
    rl_14_db = interp_at(s1p.freq_hz, s1p.rl_db, 14.0e6)
    vswr_14 = interp_at(s1p.freq_hz, s1p.vswr, 14.0e6)

    (mid_start, mid_stop), (end_start, end_stop) = dtf_windows(dtf.dist_m)
    mid_dtf_dist_m, mid_dtf_db = find_region_min(dtf.dist_m, dtf.dtf_db, mid_start, mid_stop)
    end_dtf_dist_m, end_dtf_db = find_region_min(dtf.dist_m, dtf.dtf_db, end_start, end_stop)

    quick = 0.0
    notes = []
    if spec.kind == "antenna":
        provisional = ChannelSummary(
            channel=spec.short_label(),
            kind=spec.kind,
            healthy=False,
            verdict="NOT HEALTHY",
            min_vswr=min_vswr,
            min_vswr_freq_mhz=min_vswr_freq,
            avg_vswr=avg_vswr,
            best_rl_db=best_rl_db,
            best_rl_freq_mhz=best_rl_freq_mhz,
            rl_14_db=rl_14_db,
            vswr_14=vswr_14,
            end_dtf_dist_m=end_dtf_dist_m,
            end_dtf_db=end_dtf_db,
            mid_dtf_dist_m=mid_dtf_dist_m,
            mid_dtf_db=mid_dtf_db,
            estimated_length_m=end_dtf_dist_m,
            quick_score=0.0,
            urgency_score=0.0,
            notes=[],
        )
        quick = absolute_main_score(provisional)
        healthy = (
            avg_vswr <= 2.5
            and vswr_14 <= 1.7
            and rl_14_db >= 11.0
            and min_vswr <= 1.6
            and abs(min_vswr_freq - VSWR_TARGET_MHZ) <= 2.5
            and (
                not math.isfinite(mid_dtf_db)
                or not math.isfinite(end_dtf_db)
                or mid_dtf_db > end_dtf_db + 4.0
            )
        )
        if avg_vswr > 4.0 or min_vswr > 3.0:
            notes.append(
                "Broadband mismatch is severe: minimum VSWR is %s and average VSWR is %s."
                % (format_float(min_vswr, 3), format_float(avg_vswr, 3))
            )
        elif vswr_14 > 1.8 or rl_14_db < 10.0:
            notes.append(
                "The 14 MHz region is weak: VSWR@14 MHz is %s and RL@14 MHz is %s dB."
                % (format_float(vswr_14, 3), format_float(rl_14_db, 2))
            )
        if abs(min_vswr_freq - VSWR_TARGET_MHZ) > 2.5:
            notes.append(
                "The best match shifts to %s MHz instead of the usual 14-14.5 MHz region."
                % format_float(min_vswr_freq, 3)
            )
        if math.isfinite(mid_dtf_db) and math.isfinite(end_dtf_db) and mid_dtf_db <= end_dtf_db + 4.0:
            notes.append(
                "A mid-run DTF feature at %s m is unusually strong relative to the far-end feature."
                % format_float(mid_dtf_dist_m, 2)
            )
    else:
        provisional = ChannelSummary(
            channel=spec.short_label(),
            kind=spec.kind,
            healthy=False,
            verdict="NOT HEALTHY",
            min_vswr=min_vswr,
            min_vswr_freq_mhz=min_vswr_freq,
            avg_vswr=avg_vswr,
            best_rl_db=best_rl_db,
            best_rl_freq_mhz=best_rl_freq_mhz,
            rl_14_db=rl_14_db,
            vswr_14=vswr_14,
            end_dtf_dist_m=end_dtf_dist_m,
            end_dtf_db=end_dtf_db,
            mid_dtf_dist_m=mid_dtf_dist_m,
            mid_dtf_db=mid_dtf_db,
            estimated_length_m=end_dtf_dist_m,
            quick_score=0.0,
            urgency_score=0.0,
            notes=[],
        )
        quick = absolute_interferometer_score(provisional)
        healthy = (
            avg_vswr <= 2.8
            and vswr_14 <= 1.7
            and rl_14_db >= 12.0
            and min_vswr <= 1.5
            and abs(min_vswr_freq - VSWR_TARGET_MHZ) <= 4.0
        )
        if vswr_14 > 1.6 or rl_14_db < 12.0:
            notes.append(
                "The 14 MHz region looks weaker than ideal for an interferometer channel: VSWR@14 MHz is %s and RL@14 MHz is %s dB."
                % (format_float(vswr_14, 3), format_float(rl_14_db, 2))
            )
        if abs(min_vswr_freq - VSWR_TARGET_MHZ) > 4.0:
            notes.append(
                "The best match shifts to %s MHz instead of staying near the midband region."
                % format_float(min_vswr_freq, 3)
            )
        if math.isfinite(mid_dtf_db) and math.isfinite(end_dtf_db) and mid_dtf_db <= end_dtf_db + 6.0:
            notes.append(
                "The DTF trace shows a mid-run feature near %s m that deserves comparison with the other interferometer channels."
                % format_float(mid_dtf_dist_m, 2)
            )

    healthy = healthy and quick < 40.0
    verdict = "HEALTHY" if healthy else "NOT HEALTHY"
    return ChannelSummary(
        channel=spec.short_label(),
        kind=spec.kind,
        healthy=healthy,
        verdict=verdict,
        min_vswr=min_vswr,
        min_vswr_freq_mhz=min_vswr_freq,
        avg_vswr=avg_vswr,
        best_rl_db=best_rl_db,
        best_rl_freq_mhz=best_rl_freq_mhz,
        rl_14_db=rl_14_db,
        vswr_14=vswr_14,
        end_dtf_dist_m=end_dtf_dist_m,
        end_dtf_db=end_dtf_db,
        mid_dtf_dist_m=mid_dtf_dist_m,
        mid_dtf_db=mid_dtf_db,
        estimated_length_m=end_dtf_dist_m,
        quick_score=quick,
        urgency_score=quick,
        notes=notes,
    )


def render_quick_summary(result: ChannelResult, root: Path) -> str:
    summary = result.summary
    label_width = 7
    value_width = 7
    lines = [
        "%s: %s" % (summary.channel, summary.verdict),
        "  Files:",
        "    %s" % result.paths.state_path.relative_to(root),
        "    %s" % result.paths.s1p_path.relative_to(root),
        "    %s" % result.paths.dtf_path.relative_to(root),
        "    %s" % result.paths.vswr_path.relative_to(root),
        "  Best RL: %s dB @ %s MHz"
        % (format_float(summary.best_rl_db, 2), format_float(summary.best_rl_freq_mhz, 3)),
        "  Minimum VSWR: %s @ %s MHz"
        % (format_float(summary.min_vswr, 3), format_float(summary.min_vswr_freq_mhz, 3)),
        "  14 MHz: RL %s dB, VSWR %s"
        % (format_float(summary.rl_14_db, 2), format_float(summary.vswr_14, 3)),
        "  Estimated cable length (far-end DTF feature): %s m (%s dB)"
        % (format_float(summary.end_dtf_dist_m, 2), format_float(summary.end_dtf_db, 2)),
    ]
    if math.isfinite(summary.mid_dtf_dist_m):
        lines.append(
            "  Strongest mid-run DTF feature: %s m (%s dB)"
            % (format_float(summary.mid_dtf_dist_m, 2), format_float(summary.mid_dtf_db, 2))
        )
    freq_mhz_values = [9.0 + 0.5 * idx for idx in range(15)]
    lines.append("  RL/VSWR table (9-16 MHz in 0.5 MHz steps):")
    lines.append(
        "    "
        + ("%-*s" % (label_width, "MHz"))
        + "|"
        + "".join("%*s" % (value_width, format_float(freq_mhz, 2)) for freq_mhz in freq_mhz_values)
    )
    lines.append("    " + ("-" * label_width) + "|" + ("-" * (value_width * len(freq_mhz_values))))
    lines.append(
        "    "
        + ("%-*s" % (label_width, "VSWR"))
        + "|"
        + "".join(
            "%*s" % (value_width, format_float(interp_at(result.s1p.freq_hz, result.s1p.vswr, freq_mhz * 1.0e6), 2))
            for freq_mhz in freq_mhz_values
        )
    )
    lines.append(
        "    "
        + ("%-*s" % (label_width, "RL(dB)"))
        + "|"
        + "".join(
            "%*s" % (value_width, format_float(interp_at(result.s1p.freq_hz, result.s1p.rl_db, freq_mhz * 1.0e6), 2))
            for freq_mhz in freq_mhz_values
        )
    )
    if summary.notes:
        lines.append("  Notes:")
        for note in summary.notes:
            lines.append("    - %s" % note)
    return "\n".join(lines)


def render_fieldfox_error_log(
    radar_id: str,
    spec: ChannelSpec,
    instrument_id: str,
    host: str,
    port: Optional[int],
    status: str,
    error_events: Sequence[FieldFoxErrorEvent],
    failure_message: Optional[str] = None,
) -> str:
    lines = [
        "FieldFox Error Log",
        "Timestamp: %s" % iso_timestamp(),
        "Radar: %s" % radar_id,
        "Channel: %s" % spec.label(),
        "Instrument: %s" % instrument_id,
        "Host: %s" % host,
        "Port: %s" % (port if port is not None else "auto"),
        "Run Status: %s" % status,
    ]
    if failure_message:
        lines.append("Failure: %s" % failure_message)
    lines.append("")
    if not error_events:
        lines.append("No FieldFox system errors were observed.")
    else:
        lines.append("Observed FieldFox system errors:")
        for idx, event in enumerate(error_events, start=1):
            lines.append("%d. %s | %s | %s" % (idx, event.timestamp, event.context, event.message))
    lines.append("")
    return "\n".join(lines)


def capture_installed_channel(
    client: FieldFoxClient,
    root: Path,
    radar_id: str,
    spec: ChannelSpec,
    settings: MeasurementSettings,
    instrument_id: str,
    date_tag: Optional[str] = None,
    distance_stop_m: Optional[float] = None,
    reassert_distance_stop: bool = True,
) -> ChannelResult:
    distance_stop_m = distance_stop_m if distance_stop_m is not None else spec.distance_stop_m()
    if reassert_distance_stop:
        recalibration_required = set_distance_stop(client, distance_stop_m, settings=settings)
        if recalibration_required:
            raise FieldFoxError(
                "Changing the DTF stop distance to %s m disabled correction. "
                "Re-run the OSL calibration for this stop-distance setup before capturing."
                % format_float(distance_stop_m, 1)
            )
    client.single_sweep()

    paths = capture_paths_for_channel(root, radar_id, spec, date_tag=date_tag)

    client.save_remote_file("MMEM:STOR:STAT", paths.state_path.name, paths.state_path)
    _select_trace(client, 2)
    client.save_remote_file("MMEM:STOR:FDAT", paths.dtf_path.name, paths.dtf_path)
    _select_trace(client, 1)

    try:
        client.save_remote_file("MMEM:STOR:SNP", paths.s1p_path.name, paths.s1p_path)
        s1p = parse_s1p(paths.s1p_path)
    except Exception:
        s1p = capture_s11_from_instrument(client, settings)
        write_touchstone(paths.s1p_path, s1p, instrument_id=instrument_id, radar_id=radar_id, spec=spec)
    write_vswr_csv(
        paths.vswr_path,
        s1p,
        instrument_id=instrument_id,
        radar_id=radar_id,
        spec=spec,
        source_s1p=paths.s1p_path,
    )

    dtf = parse_dtf_csv(paths.dtf_path)
    summary = summarize_channel(spec, s1p, dtf)
    return ChannelResult(spec=spec, paths=paths, s1p=s1p, dtf=dtf, summary=summary)


def build_main_reference(results: Sequence[ChannelResult]) -> Tuple[List[float], List[float], List[float], List[str]]:
    if not results:
        return [], [], [], []
    candidates = [item for item in results if item.summary.healthy]
    if len(candidates) < 2:
        candidates = list(results)
    ordered = sorted(
        candidates,
        key=lambda item: (
            item.summary.quick_score,
            abs(item.summary.min_vswr_freq_mhz - VSWR_TARGET_MHZ),
            item.summary.vswr_14,
            item.summary.avg_vswr,
        ),
    )
    ref_count = min(4, len(candidates))
    chosen = ordered[:ref_count]
    ref_freq = list(chosen[0].s1p.freq_hz)
    vswr_curves = [interpolate_series(item.s1p.freq_hz, item.s1p.vswr, ref_freq) for item in chosen]
    rl_curves = [interpolate_series(item.s1p.freq_hz, item.s1p.rl_db, ref_freq) for item in chosen]
    return ref_freq, mean_series(vswr_curves), mean_series(rl_curves), [item.summary.channel for item in chosen]


def build_interferometer_reference(results: Sequence[ChannelResult]) -> Tuple[List[float], List[float]]:
    if not results:
        return [], []
    ref_freq = list(results[0].s1p.freq_hz)
    vswr_curves = [interpolate_series(item.s1p.freq_hz, item.s1p.vswr, ref_freq) for item in results]
    return ref_freq, median_series(vswr_curves)


def apply_batch_scoring(results: Sequence[ChannelResult]) -> Dict[str, List[str]]:
    main_results = [item for item in results if item.spec.kind == "antenna"]
    int_results = [item for item in results if item.spec.kind != "antenna"]

    main_ref_freq, main_ref_vswr, main_ref_rl, main_ref_channels = build_main_reference(main_results)
    int_ref_freq, int_ref_vswr = build_interferometer_reference(int_results)

    for item in main_results:
        score = absolute_main_score(item.summary)
        if main_ref_freq and main_ref_vswr:
            vswr_curve = interpolate_series(item.s1p.freq_hz, item.s1p.vswr, main_ref_freq)
            rl_curve = interpolate_series(item.s1p.freq_hz, item.s1p.rl_db, main_ref_freq)
            item.summary.vswr_rmse_to_ref = rmse(vswr_curve, main_ref_vswr)
            item.summary.rl_rmse_to_ref = rmse(rl_curve, main_ref_rl)
            if item.summary.vswr_rmse_to_ref is not None:
                if item.summary.vswr_rmse_to_ref > 2.0:
                    score += 45.0
                elif item.summary.vswr_rmse_to_ref > 1.2:
                    score += 28.0
                elif item.summary.vswr_rmse_to_ref > 0.7:
                    score += 15.0
        item.summary.urgency_score = score

    for item in int_results:
        score = absolute_interferometer_score(item.summary)
        if int_ref_freq and int_ref_vswr:
            vswr_curve = interpolate_series(item.s1p.freq_hz, item.s1p.vswr, int_ref_freq)
            item.summary.vswr_rmse_to_peer_median = rmse(vswr_curve, int_ref_vswr)
            peer_rmse = item.summary.vswr_rmse_to_peer_median
            if peer_rmse is not None:
                if peer_rmse > 2.0:
                    score += 35.0
                elif peer_rmse > 1.0:
                    score += 20.0
                elif peer_rmse > 0.35:
                    score += 8.0
        item.summary.urgency_score = score

    return {"main_reference_channels": main_ref_channels}


def _nice_ticks(start: float, stop: float, count: int) -> List[float]:
    if count <= 1 or stop <= start:
        return [start]
    span = stop - start
    raw_step = span / float(count - 1)
    magnitude = 10 ** math.floor(math.log10(raw_step)) if raw_step > 0 else 1
    for factor in (1, 2, 5, 10):
        step = factor * magnitude
        if step >= raw_step:
            break
    first = math.floor(start / step) * step
    ticks = []
    value = first
    while value <= stop + step:
        if value >= start - 1e-9:
            ticks.append(round(value, 9))
        value += step
    return ticks


def _svg_line_chart(
    title: str,
    xlabel: str,
    ylabel: str,
    series: Sequence[Tuple[str, Sequence[float], Sequence[float]]],
    out_path: Path,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
) -> None:
    width = 1400
    height = 840
    margin_left = 80
    margin_right = 320
    margin_top = 60
    margin_bottom = 75
    plot_left = margin_left
    plot_top = margin_top
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    x_values = []
    y_values = []
    for _, xs, ys in series:
        x_values.extend(float(value) for value in xs)
        y_values.extend(float(value) for value in ys if math.isfinite(value))
    if not x_values or not y_values:
        out_path.write_text("")
        return
    x_min = min(x_values)
    x_max = max(x_values)
    if y_min is None:
        y_min = min(y_values)
    if y_max is None:
        y_max = max(y_values)
    if y_max <= y_min:
        y_max = y_min + 1.0
    x_ticks = _nice_ticks(x_min, x_max, 7)
    y_ticks = _nice_ticks(y_min, y_max, 7)

    def scale_x(value: float) -> float:
        return plot_left + (value - x_min) * plot_width / float(x_max - x_min)

    def scale_y(value: float) -> float:
        clipped = min(max(value, y_min), y_max)
        return plot_top + plot_height - (clipped - y_min) * plot_height / float(y_max - y_min)

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d">' % (width, height, width, height),
        '<rect x="0" y="0" width="%d" height="%d" fill="#ffffff"/>' % (width, height),
        '<text x="%d" y="34" font-size="24" font-family="Menlo,Monaco,monospace" fill="#111111">%s</text>'
        % (margin_left, html.escape(title)),
    ]

    for tick in x_ticks:
        x_pos = scale_x(tick)
        parts.append('<line x1="%0.2f" y1="%d" x2="%0.2f" y2="%d" stroke="#e5e7eb" stroke-width="1"/>' % (x_pos, plot_top, x_pos, plot_top + plot_height))
        parts.append(
            '<text x="%0.2f" y="%d" text-anchor="middle" font-size="12" font-family="Menlo,Monaco,monospace" fill="#444444">%s</text>'
            % (x_pos, plot_top + plot_height + 22, html.escape(format_float(tick, 2)))
        )
    for tick in y_ticks:
        y_pos = scale_y(tick)
        parts.append('<line x1="%d" y1="%0.2f" x2="%d" y2="%0.2f" stroke="#e5e7eb" stroke-width="1"/>' % (plot_left, y_pos, plot_left + plot_width, y_pos))
        parts.append(
            '<text x="%d" y="%0.2f" text-anchor="end" font-size="12" font-family="Menlo,Monaco,monospace" fill="#444444">%s</text>'
            % (plot_left - 10, y_pos + 4, html.escape(format_float(tick, 2)))
        )

    parts.append('<rect x="%d" y="%d" width="%d" height="%d" fill="none" stroke="#111111" stroke-width="1.5"/>' % (plot_left, plot_top, plot_width, plot_height))
    parts.append(
        '<text x="%d" y="%d" text-anchor="middle" font-size="14" font-family="Menlo,Monaco,monospace" fill="#111111">%s</text>'
        % (plot_left + plot_width // 2, height - 18, html.escape(xlabel))
    )
    parts.append(
        '<text transform="translate(22 %d) rotate(-90)" text-anchor="middle" font-size="14" font-family="Menlo,Monaco,monospace" fill="#111111">%s</text>'
        % (plot_top + plot_height // 2, html.escape(ylabel))
    )

    for idx, (label, xs, ys) in enumerate(series):
        points = []
        for x_value, y_value in zip(xs, ys):
            if not math.isfinite(y_value):
                continue
            points.append("%0.2f,%0.2f" % (scale_x(float(x_value)), scale_y(float(y_value))))
        if len(points) < 2:
            continue
        color = PALETTE[idx % len(PALETTE)]
        parts.append('<polyline fill="none" stroke="%s" stroke-width="2" points="%s"/>' % (color, " ".join(points)))

    legend_x = plot_left + plot_width + 24
    legend_y = plot_top
    parts.append('<text x="%d" y="%d" font-size="16" font-family="Menlo,Monaco,monospace" fill="#111111">Legend</text>' % (legend_x, legend_y + 2))
    for idx, (label, _, _) in enumerate(series):
        color = PALETTE[idx % len(PALETTE)]
        row_y = legend_y + 24 + idx * 20
        parts.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="%s" stroke-width="3"/>' % (legend_x, row_y, legend_x + 18, row_y, color))
        parts.append(
            '<text x="%d" y="%d" font-size="13" font-family="Menlo,Monaco,monospace" fill="#111111">%s</text>'
            % (legend_x + 26, row_y + 4, html.escape(label))
        )

    parts.append("</svg>")
    out_path.write_text("\n".join(parts) + "\n")


def _svg_bar_chart(
    title: str,
    rows: Sequence[Tuple[str, float]],
    out_path: Path,
) -> None:
    ordered = list(rows)
    width = 1100
    row_height = 28
    height = max(320, 110 + row_height * len(ordered))
    margin_left = 240
    margin_right = 60
    margin_top = 60
    margin_bottom = 40
    plot_width = width - margin_left - margin_right
    if not ordered:
        out_path.write_text("")
        return
    max_score = max(score for _, score in ordered)
    max_score = max(max_score, 1.0)
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d">' % (width, height, width, height),
        '<rect x="0" y="0" width="%d" height="%d" fill="#ffffff"/>' % (width, height),
        '<text x="%d" y="34" font-size="24" font-family="Menlo,Monaco,monospace" fill="#111111">%s</text>'
        % (margin_left, html.escape(title)),
    ]
    ticks = _nice_ticks(0.0, max_score, 6)
    for tick in ticks:
        x_pos = margin_left + plot_width * tick / float(max_score)
        parts.append('<line x1="%0.2f" y1="%d" x2="%0.2f" y2="%d" stroke="#e5e7eb" stroke-width="1"/>' % (x_pos, margin_top, x_pos, height - margin_bottom))
        parts.append(
            '<text x="%0.2f" y="%d" text-anchor="middle" font-size="12" font-family="Menlo,Monaco,monospace" fill="#444444">%s</text>'
            % (x_pos, height - 14, html.escape(format_float(tick, 1)))
        )
    for idx, (label, score) in enumerate(ordered):
        y = margin_top + idx * row_height
        bar_width = plot_width * score / float(max_score)
        color = "#b91c1c" if score >= 120 else "#ea580c" if score >= 50 else "#15803d"
        parts.append('<rect x="%d" y="%d" width="%0.2f" height="16" rx="2" fill="%s"/>' % (margin_left, y, bar_width, color))
        parts.append(
            '<text x="%d" y="%d" text-anchor="end" font-size="13" font-family="Menlo,Monaco,monospace" fill="#111111">%s</text>'
            % (margin_left - 10, y + 13, html.escape(label))
        )
        parts.append(
            '<text x="%0.2f" y="%d" font-size="13" font-family="Menlo,Monaco,monospace" fill="#111111">%s</text>'
            % (margin_left + bar_width + 8, y + 13, html.escape(format_float(score, 1)))
        )
    parts.append("</svg>")
    out_path.write_text("\n".join(parts) + "\n")


def unique_report_dir(root: Path, date_tag: Optional[str] = None) -> Path:
    date_tag = date_tag or default_date_tag()
    base = root / "analysis" / ("installed_screening_%s" % date_tag)
    if not base.exists():
        base.mkdir(parents=True, exist_ok=True)
        return base
    sequence = 2
    while True:
        candidate = root / "analysis" / ("installed_screening_%s_%02d" % (date_tag, sequence))
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        sequence += 1


def write_batch_outputs(
    root: Path,
    radar_id: str,
    results: Sequence[ChannelResult],
    instrument_id: str,
    date_tag: Optional[str] = None,
) -> Dict[str, Path]:
    metadata = apply_batch_scoring(results)
    report_dir = unique_report_dir(root, date_tag=date_tag)

    main_results = sorted(
        [item for item in results if item.spec.kind == "antenna"],
        key=lambda item: item.summary.urgency_score,
        reverse=True,
    )
    int_results = sorted(
        [item for item in results if item.spec.kind != "antenna"],
        key=lambda item: item.summary.urgency_score,
        reverse=True,
    )

    def write_csv(path: Path, selected: Sequence[ChannelResult]) -> None:
        lines = [
            "channel,kind,verdict,healthy,urgency_score,quick_score,min_vswr,min_vswr_freq_mhz,avg_vswr,best_rl_db,best_rl_freq_mhz,rl_14_db,vswr_14,estimated_length_m,end_dtf_dist_m,end_dtf_db,mid_dtf_dist_m,mid_dtf_db,vswr_rmse_to_ref,rl_rmse_to_ref,vswr_rmse_to_peer_median,notes"
        ]
        for item in selected:
            summary = item.summary
            row = [
                summary.channel,
                summary.kind,
                summary.verdict,
                "yes" if summary.healthy else "no",
                format_float(summary.urgency_score, 3),
                format_float(summary.quick_score, 3),
                format_float(summary.min_vswr, 6),
                format_float(summary.min_vswr_freq_mhz, 6),
                format_float(summary.avg_vswr, 6),
                format_float(summary.best_rl_db, 6),
                format_float(summary.best_rl_freq_mhz, 6),
                format_float(summary.rl_14_db, 6),
                format_float(summary.vswr_14, 6),
                format_float(summary.estimated_length_m, 6),
                format_float(summary.end_dtf_dist_m, 6),
                format_float(summary.end_dtf_db, 6),
                format_float(summary.mid_dtf_dist_m, 6),
                format_float(summary.mid_dtf_db, 6),
                "" if summary.vswr_rmse_to_ref is None else format_float(summary.vswr_rmse_to_ref, 6),
                "" if summary.rl_rmse_to_ref is None else format_float(summary.rl_rmse_to_ref, 6),
                "" if summary.vswr_rmse_to_peer_median is None else format_float(summary.vswr_rmse_to_peer_median, 6),
                '"' + "; ".join(summary.notes).replace('"', "'") + '"',
            ]
            lines.append(",".join(row))
        path.write_text("\n".join(lines) + "\n")

    main_csv = report_dir / "main_array_summary.csv"
    int_csv = report_dir / "interferometer_summary.csv"
    write_csv(main_csv, main_results)
    write_csv(int_csv, int_results)

    main_vswr_series = [(item.summary.channel, [freq / 1e6 for freq in item.s1p.freq_hz], item.s1p.vswr) for item in main_results]
    main_rl_series = [(item.summary.channel, [freq / 1e6 for freq in item.s1p.freq_hz], item.s1p.rl_db) for item in main_results]
    main_dtf_series = [(item.summary.channel, item.dtf.dist_m, item.dtf.dtf_db) for item in main_results]
    int_vswr_series = [(item.summary.channel, [freq / 1e6 for freq in item.s1p.freq_hz], item.s1p.vswr) for item in int_results]
    int_rl_series = [(item.summary.channel, [freq / 1e6 for freq in item.s1p.freq_hz], item.s1p.rl_db) for item in int_results]
    int_dtf_series = [(item.summary.channel, item.dtf.dist_m, item.dtf.dtf_db) for item in int_results]

    main_vswr_plot = report_dir / "main_array_vswr.svg"
    main_rl_plot = report_dir / "main_array_return_loss.svg"
    main_dtf_plot = report_dir / "main_array_dtf.svg"
    _svg_line_chart(
        "Main Array Installed VSWR",
        "Frequency (MHz)",
        "VSWR",
        main_vswr_series,
        main_vswr_plot,
        y_min=1.0,
        y_max=max(6.0, min(8.0, max(safe_max(item.s1p.vswr, 1.0) for item in main_results if item.s1p.vswr))),
    )
    _svg_line_chart(
        "Main Array Installed Return Loss",
        "Frequency (MHz)",
        "Return Loss (dB)",
        main_rl_series,
        main_rl_plot,
        y_min=0.0,
        y_max=max(30.0, min(40.0, max(safe_max(item.s1p.rl_db, 0.0) for item in main_results if item.s1p.rl_db))),
    )
    _svg_line_chart(
        "Main Array Installed DTF",
        "Distance (m)",
        "DTF (dB)",
        main_dtf_series,
        main_dtf_plot,
        y_min=0.0,
        y_max=max(30.0, max(safe_max(item.dtf.dtf_db, 0.0) for item in main_results if item.dtf.dtf_db)),
    )

    int_vswr_plot = report_dir / "interferometer_vswr.svg"
    int_rl_plot = report_dir / "interferometer_return_loss.svg"
    int_dtf_plot = report_dir / "interferometer_dtf.svg"
    if int_results:
        _svg_line_chart(
            "Interferometer Installed VSWR",
            "Frequency (MHz)",
            "VSWR",
            int_vswr_series,
            int_vswr_plot,
            y_min=1.0,
            y_max=max(4.0, min(8.0, max(safe_max(item.s1p.vswr, 1.0) for item in int_results if item.s1p.vswr))),
        )
        _svg_line_chart(
            "Interferometer Installed Return Loss",
            "Frequency (MHz)",
            "Return Loss (dB)",
            int_rl_series,
            int_rl_plot,
            y_min=0.0,
            y_max=max(25.0, min(40.0, max(safe_max(item.s1p.rl_db, 0.0) for item in int_results if item.s1p.rl_db))),
        )
        _svg_line_chart(
            "Interferometer Installed DTF",
            "Distance (m)",
            "DTF (dB)",
            int_dtf_series,
            int_dtf_plot,
            y_min=0.0,
            y_max=max(30.0, max(safe_max(item.dtf.dtf_db, 0.0) for item in int_results if item.dtf.dtf_db)),
        )

    main_rank_plot = report_dir / "main_array_ranking.svg"
    int_rank_plot = report_dir / "interferometer_ranking.svg"
    _svg_bar_chart("Main Array Triage Ranking", [(item.summary.channel, item.summary.urgency_score) for item in main_results], main_rank_plot)
    if int_results:
        _svg_bar_chart("Interferometer Triage Ranking", [(item.summary.channel, item.summary.urgency_score) for item in int_results], int_rank_plot)

    report_lines = [
        "FieldFox installed-path screening report",
        "",
        "Radar ID: %s" % radar_id,
        "Instrument: %s" % instrument_id,
        "Generated: %s" % iso_timestamp(),
        "",
        "Configuration assumptions",
        "- Mode: CAT/TDR",
        "- Return-loss sweep: 8-20 MHz, 801 points, -15 dBm, 10 kHz IF BW when supported.",
        "- DTF setup: 8-20 MHz frequency span, velocity factor %.2f." % DEFAULT_VELOCITY_FACTOR,
        "- Main-array DTF span target: %.0f m. Interferometer DTF span target: %.0f m." % (DEFAULT_MAIN_STOP_M, DEFAULT_INTERFEROMETER_STOP_M),
        "",
        "How the ranking works",
        "- Each channel gets an absolute screening score from VSWR, return loss, match-frequency shift, and whether a mid-run DTF feature rivals the far-end feature.",
        "- Main-array channels then get an extra penalty when their VSWR shape differs from the average of the best-looking measured main channels.",
        "- Interferometer channels get an extra penalty when their VSWR shape differs from the interferometer peer median.",
        "- Higher score means address that channel sooner.",
        "",
        "Main-array reference channels: %s" % (", ".join(metadata.get("main_reference_channels", [])) or "none"),
        "",
        "Main-array ranking (worst to best)",
    ]
    for idx, item in enumerate(main_results, start=1):
        summary = item.summary
        report_lines.append(
            "%d. %s: %s; urgency %s; min VSWR %s @ %s MHz; VSWR@14 %s; RL@14 %s dB; est. length %s m"
            % (
                idx,
                summary.channel,
                summary.verdict,
                format_float(summary.urgency_score, 1),
                format_float(summary.min_vswr, 3),
                format_float(summary.min_vswr_freq_mhz, 3),
                format_float(summary.vswr_14, 3),
                format_float(summary.rl_14_db, 2),
                format_float(summary.estimated_length_m, 2),
            )
        )
        report_lines.append(
            "   DTF: end %s m / %s dB; mid %s m / %s dB"
            % (
                format_float(summary.end_dtf_dist_m, 2),
                format_float(summary.end_dtf_db, 2),
                format_float(summary.mid_dtf_dist_m, 2),
                format_float(summary.mid_dtf_db, 2),
            )
        )
        if summary.vswr_rmse_to_ref is not None:
            report_lines.append(
                "   VSWR RMSE to main-array reference: %s"
                % format_float(summary.vswr_rmse_to_ref, 3)
            )
        for note in summary.notes:
            report_lines.append("   Note: %s" % note)
    report_lines.append("")
    report_lines.append("Interferometer ranking (worst to best)")
    for idx, item in enumerate(int_results, start=1):
        summary = item.summary
        report_lines.append(
            "%d. %s: %s; urgency %s; min VSWR %s @ %s MHz; VSWR@14 %s; RL@14 %s dB; est. length %s m"
            % (
                idx,
                summary.channel,
                summary.verdict,
                format_float(summary.urgency_score, 1),
                format_float(summary.min_vswr, 3),
                format_float(summary.min_vswr_freq_mhz, 3),
                format_float(summary.vswr_14, 3),
                format_float(summary.rl_14_db, 2),
                format_float(summary.estimated_length_m, 2),
            )
        )
        report_lines.append(
            "   DTF: end %s m / %s dB; mid %s m / %s dB"
            % (
                format_float(summary.end_dtf_dist_m, 2),
                format_float(summary.end_dtf_db, 2),
                format_float(summary.mid_dtf_dist_m, 2),
                format_float(summary.mid_dtf_db, 2),
            )
        )
        if summary.vswr_rmse_to_peer_median is not None:
            report_lines.append(
                "   VSWR RMSE to interferometer peer median: %s"
                % format_float(summary.vswr_rmse_to_peer_median, 3)
            )
        for note in summary.notes:
            report_lines.append("   Note: %s" % note)

    report_txt = report_dir / "screening_report.txt"
    report_txt.write_text("\n".join(report_lines) + "\n")

    return {
        "report_dir": report_dir,
        "report_txt": report_txt,
        "main_csv": main_csv,
        "int_csv": int_csv,
        "main_vswr_plot": main_vswr_plot,
        "main_rl_plot": main_rl_plot,
        "main_dtf_plot": main_dtf_plot,
        "main_rank_plot": main_rank_plot,
        "int_vswr_plot": int_vswr_plot,
        "int_rl_plot": int_rl_plot,
        "int_dtf_plot": int_dtf_plot,
        "int_rank_plot": int_rank_plot,
    }
