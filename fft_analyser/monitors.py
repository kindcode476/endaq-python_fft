"""
Management of a bank of vibration monitors.

A :py:class:`MonitorBank` holds up to :py:data:`MAX_MONITORS` monitors and
serves the UI with decoded waveforms, whether they come from the X2 Cloud
API or from ``.bin`` files on disk.  Waveforms are cached in-process
because a five-second three-axis record is ~1.5 MB of float64 - far too
large to shuttle through a browser store on every callback.

Real accelerometer records carry a **static gravity component**: a
sensor at rest reads about 1 g on whichever axis points down.  In the
vendor's own three-axis sample the DC vector is 1.043 g, and it is four
times larger than the entire 10-1000 Hz vibration content.  Left in, it
lands in the 0 Hz bin and dominates every amplitude reading and any
overall RMS figure.  :py:data:`REAL_DATA_DEFAULTS` therefore removes the
mean for measured data, which is what condition-monitoring practice
assumes; use ``detrend="none"`` only when the DC level is what you are
looking at (checking sensor orientation, for instance).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import datetime as _dt
import pathlib
import typing

import numpy as np
import pandas as pd

from .analysis import analyze, band_rms, signal_quality, velocity_band_rms
from .pure_mems import MEMSWaveform, decode_bin, decode_bin_file

#: pureMEMS sensor full scale: ±16 g in m/s². Samples at ~this level mean
#: the sensor saturated and the spectrum's harmonics may be spurious.
SENSOR_FULL_SCALE = 16.0 * 9.80665

__all__ = [
    "MAX_MONITORS",
    "REAL_DATA_DEFAULTS",
    "ISO_BAND",
    "Monitor",
    "MonitorBank",
    "overall_levels",
]

#: how many monitors the bank (and the UI) holds
MAX_MONITORS = 5

#: analysis defaults appropriate to measured accelerometer data
REAL_DATA_DEFAULTS = {
    "detrend": "mean",      # strip the gravity offset (see module docstring)
    "window": "hann",
    "averaging": "linear",
    "nperseg": 8192,
    "overlap": 0.5,
}

#: the broadband band used for the overall level, in Hz.  10-1000 Hz is
#: the ISO 10816 / ISO 20816 machine-vibration band.
ISO_BAND = (10.0, 1000.0)


@dataclass
class Monitor:
    """One vibration monitor and its most recent waveform."""

    key: str                                   #: stable id (the address, or file path)
    name: str                                  #: display name
    address: str = ""                          #: X2 technical address
    site_id: str = ""
    source: str = "file"                       #: "file" or "cloud"
    waveform: typing.Optional[MEMSWaveform] = None
    captured: typing.Optional[_dt.datetime] = None
    file_name: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.waveform is not None and not self.error

    @property
    def axes(self) -> typing.List[str]:
        return list(self.waveform.data.columns) if self.waveform is not None else []

    def channel(self, axis: typing.Optional[str] = None) -> pd.DataFrame:
        """One axis as a single-column, time-indexed frame."""
        if self.waveform is None:
            raise ValueError(f"monitor {self.name!r} has no waveform")
        cols = self.waveform.data.columns
        col = axis if axis in cols else cols[-1]     # default to Z / the only axis
        return self.waveform.data[[col]]

    def summary(self, axis: typing.Optional[str] = None) -> dict:
        """Headline numbers for the monitor strip."""
        if not self.ok:
            return {"name": self.name, "error": self.error or "no data"}
        df = self.channel(axis)
        col = df.columns[0]
        x = df[col].to_numpy()
        result = analyze(df, quantity="psd", **REAL_DATA_DEFAULTS)
        rms = float(np.sqrt(np.mean((x - x.mean()) ** 2)))
        return {
            "name": self.name,
            "axis": col,
            "fs": self.waveform.fs,
            "n": len(x),
            "duration": self.waveform.duration,
            "dc_offset": float(x.mean()),
            "dc_g": float(x.mean() / 9.80665),
            "rms": rms,
            "peak": float(np.abs(x - x.mean()).max()),
            "crest": float(np.abs(x - x.mean()).max() / rms) if rms else float("nan"),
            "band_rms": float(band_rms(result, *ISO_BAND)),
            "vel_rms_mm_s": float(velocity_band_rms(result, *ISO_BAND)),
            "quality": signal_quality(df, full_scale=SENSOR_FULL_SCALE),
            "temperature": (float(np.mean(self.waveform.temperature_c))
                            if len(self.waveform.temperature_c) else None),
            "captured": self.captured,
            "truncated": self.waveform.truncated,
            "error": "",
        }


class MonitorBank:
    """
    A fixed-size bank of monitors, filled from files or from the cloud.

    The bank never issues a write: the cloud path goes through
    :py:class:`~fft_analyser.x2_client.X2Client`, which only permits
    read-only endpoints.
    """

    def __init__(self, max_monitors: int = MAX_MONITORS):
        self.max_monitors = max_monitors
        self.monitors: typing.List[Monitor] = []
        self.client = None
        self.site_id: typing.Optional[str] = None
        self.available: typing.List[dict] = []     # addresses discovered at the site

    # ── offline: .bin files ──

    def add_file(self, path, name: typing.Optional[str] = None) -> Monitor:
        """Decode a ``.bin`` file into the next free slot."""
        path = pathlib.Path(path)
        mon = Monitor(key=str(path), name=name or path.stem, source="file",
                      file_name=path.name)
        try:
            mon.waveform = decode_bin_file(path)
            stat = path.stat()
            mon.captured = _dt.datetime.fromtimestamp(stat.st_mtime)
        except Exception as exc:
            mon.error = f"{type(exc).__name__}: {exc}"
        self._place(mon)
        return mon

    def add_bytes(self, raw: bytes, name: str) -> Monitor:
        """Decode an uploaded ``.bin`` payload into the next free slot."""
        mon = Monitor(key=name, name=name, source="file", file_name=name)
        try:
            mon.waveform = decode_bin(raw)
            mon.captured = _dt.datetime.now()
        except Exception as exc:
            mon.error = f"{type(exc).__name__}: {exc}"
        self._place(mon)
        return mon

    # ── online: X2 Cloud (read-only) ──

    def connect(self, username: str, password: str, base_url: str,
                site_id) -> typing.List[dict]:
        """
        Log in and discover the vibration monitors at a site.

        Returns the address records found.  Nothing is sent to any sensor.
        """
        from .x2_client import X2Client

        self.client = X2Client(username, password, base_url=base_url)
        self.client.login()
        self.site_id = str(site_id)
        self.available = self.client.get_vibration_addresses(site_id)
        return self.available

    def load_from_cloud(self, addresses: typing.Sequence[str]) -> typing.List[Monitor]:
        """
        Download the most recent uploaded waveform for each address.

        This reads files the sensors have *already* uploaded; it never
        asks a sensor to measure or upload.
        """
        if self.client is None or self.site_id is None:
            raise RuntimeError("connect() first")

        by_address = {str(a.get("Address")): a for a in self.available}
        self.monitors = []
        for address in list(addresses)[: self.max_monitors]:
            record = by_address.get(str(address), {})
            mon = Monitor(
                key=str(address),
                name=str(record.get("Name") or address),
                address=str(address),
                site_id=self.site_id,
                source="cloud",
            )
            try:
                files = [f for f in self.client.get_address_files(self.site_id, address)
                         if f.is_waveform]
                if not files:
                    mon.error = "no waveform files uploaded"
                else:
                    newest = files[0]
                    mon.waveform = self.client.download_waveform(newest)
                    mon.captured = newest.changed
                    mon.file_name = newest.name
            except Exception as exc:
                mon.error = f"{type(exc).__name__}: {exc}"
            self.monitors.append(mon)
        return self.monitors

    # ── bank management ──

    def _place(self, mon: Monitor) -> None:
        existing = {m.key: i for i, m in enumerate(self.monitors)}
        if mon.key in existing:
            self.monitors[existing[mon.key]] = mon
        elif len(self.monitors) < self.max_monitors:
            self.monitors.append(mon)
        else:
            self.monitors[-1] = mon      # replace the oldest slot

    def clear(self) -> None:
        self.monitors = []

    def get(self, key: str) -> typing.Optional[Monitor]:
        for m in self.monitors:
            if m.key == key:
                return m
        return None

    def __len__(self) -> int:
        return len(self.monitors)

    def __iter__(self):
        return iter(self.monitors)


def overall_levels(monitors: typing.Iterable[Monitor],
                   axis: typing.Optional[str] = None) -> pd.DataFrame:
    """
    A comparison table of overall levels across the bank - the view an
    operator scans first.
    """
    rows = []
    for m in monitors:
        s = m.summary(axis)
        if s.get("error"):
            rows.append({"monitor": s["name"], "status": s["error"]})
            continue
        notes = []
        if s["truncated"]:
            notes.append("truncated record")
        notes.extend(s["quality"].flags)
        rows.append({
            "monitor": s["name"],
            "axis": s["axis"],
            "Vel RMS 10-1000 Hz (mm/s)": s["vel_rms_mm_s"],
            "band RMS 10-1000 Hz (m/s²)": s["band_rms"],
            "RMS (m/s²)": s["rms"],
            "peak (m/s²)": s["peak"],
            "crest": s["crest"],
            "DC (g)": s["dc_g"],
            "temp (°C)": s["temperature"],
            "status": "; ".join(notes) if notes else "ok",
        })
    return pd.DataFrame(rows)
