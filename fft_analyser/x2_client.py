"""
Strictly read-only client for the X2 Cloud API (X2 Wireless / TritonCloud).

**This module cannot write to the site or the sensors.** It is built so
that is a structural property, not a convention:

- Every data request goes through :py:meth:`X2Client._get`, which issues
  ``GET`` only and refuses any path that does not match the read
  allowlist in :py:data:`READ_PATHS`.
- The two mutating endpoints in the API - ``POST
  site/{id}/address/{addr}/control/{cmd}`` (actuates relays, acknowledges
  alarms, enables/disables inputs) and ``POST
  site/{id}/address/{addr}/config`` (rewrites device settings, including
  the measurement interval and duration) - have **no method here at
  all**.  There is nothing to call by accident.
- The only ``POST`` is ``login``, which the API requires to obtain a
  session cookie.  It creates a session; it does not reach the devices.
  ``logout`` is deliberately not implemented either, so no request can
  disturb a session another tool may be sharing.

Nothing this client does asks a sensor to measure, upload, or change
state; it reads waveform files the devices have already uploaded to the
cloud.

Reference: "X2 Cloud API" workbook, revision 11 (2024-04-09).

Usage::

    client = X2Client(username="...", password="...")
    client.login()
    sites = client.get_sites()
    monitors = client.get_vibration_addresses(site_id)
    files = client.get_address_files(site_id, monitors[0]["Address"])
    wave = client.download_waveform(files[0])        # -> MEMSWaveform
"""

from __future__ import annotations

from dataclasses import dataclass, field
import datetime as _dt
import posixpath
import re
import typing
import urllib.parse

import requests

from .pure_mems import MEMSWaveform, decode_bin

__all__ = [
    "PRIMARY_BASE",
    "SECONDARY_BASE",
    "READ_PATHS",
    "X2Error",
    "X2AuthError",
    "MonitorFile",
    "X2Client",
]

PRIMARY_BASE = "https://api.x2wireless.com"
SECONDARY_BASE = "https://api.tritoncloud.se"

#: The complete set of paths this client may request.  Anything else is
#: refused before a socket is opened.  All are GET / read-only.
READ_PATHS = (
    r"^site$",
    r"^site/[^/]+$",
    r"^site/[^/]+/address$",
    r"^site/[^/]+/address/[^/]+/files$",
    r"^site/[^/]+/stats$",
    r"^site/[^/]+/logs$",
    r"^customersites$",
)
_READ_RE = tuple(re.compile(p) for p in READ_PATHS)

#: Product type codes that produce vibration time waveforms
#: ("Product types" sheet).
VIBRATION_PRODUCT_TYPES = {21: "VIBRATION", 202: "MLT_PURE"}

#: Trend statistics worth plotting alongside the spectra
#: ("Statistic types" sheet).
MLT_STATS = {
    "extra_raw_mlt_rms_vel": "X/Y/Z RMS velocity (mm/s)",
    "extra_raw_mlt_rms_acc": "Z RMS acceleration (m/s²)",
    "extra_raw_mlt_peak_acc": "Z 0-peak acceleration (m/s²)",
    "extra_raw_mlt_rpm": "RPM",
    "extra_raw_temp": "Temperature (°C)",
    "extra_raw_battery": "Battery (%)",
}


class X2Error(RuntimeError):
    """An API request failed."""


class X2AuthError(X2Error):
    """Login failed or the session is not valid."""


@dataclass
class MonitorFile:
    """One waveform file already uploaded by a sensor."""

    name: str
    size: int
    changed: typing.Optional[_dt.datetime]
    pre_signed_link: str
    direct_link: str = ""
    public: bool = False
    address: str = ""
    site_id: str = ""

    @property
    def is_waveform(self) -> bool:
        return self.name.lower().endswith(".bin")

    @classmethod
    def from_json(cls, obj: dict, site_id: str = "", address: str = "") -> "MonitorFile":
        changed = None
        raw = obj.get("changed")
        if raw:
            try:
                changed = _dt.datetime.strptime(str(raw), "%Y-%m-%d %H:%M:%S")
            except ValueError:
                changed = None
        try:
            size = int(obj.get("size") or 0)
        except (TypeError, ValueError):
            size = 0
        return cls(
            name=str(obj.get("name", "")),
            size=size,
            changed=changed,
            pre_signed_link=str(obj.get("pre_signed_link") or ""),
            direct_link=str(obj.get("direct_link") or ""),
            public=bool(obj.get("public")),
            address=address,
            site_id=str(site_id),
        )


class X2Client:
    """
    Read-only X2 Cloud API client.

    :param username: API username (``u`` in the login payload)
    :param password: API password (``p``)
    :param base_url: API root; defaults to the documented primary host.
        The workbook's per-endpoint examples still show the legacy
        ``https://login.x2wireless.com/TritonCloud/webapp_mobile/api``
        form - revision 11 updated the header but not the examples, so
        pass that explicitly if your tenant still serves it.
    :param timeout: per-request timeout in seconds
    """

    def __init__(
            self,
            username: str,
            password: str,
            base_url: str = PRIMARY_BASE,
            timeout: float = 30.0,
            session: typing.Optional[requests.Session] = None,
    ):
        self.username = username
        self._password = password
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()
        self.customer_id: typing.Optional[str] = None
        self._logged_in = False

    # ── authentication (the only non-GET request in this module) ──

    def login(self) -> dict:
        """
        Obtain a session cookie.  Creates a session only - it sends
        nothing to any sensor.
        """
        url = f"{self.base_url}/login"
        try:
            resp = self.session.post(
                url, data={"u": self.username, "p": self._password},
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise X2Error(f"could not reach {url}: {exc}") from exc

        if resp.status_code != 200:
            raise X2AuthError(f"login returned HTTP {resp.status_code}")
        try:
            payload = resp.json()
        except ValueError as exc:
            raise X2AuthError("login response was not JSON") from exc

        if payload.get("Status") != "login_ok":
            raise X2AuthError(f"login rejected: {payload.get('Status')!r}")
        self.customer_id = payload.get("customerid")
        self._logged_in = True
        return payload

    # ── the single read path ──

    def _get(self, path: str, params: typing.Optional[dict] = None):
        """Issue a GET against an allowlisted read-only path."""
        clean = path.strip("/")
        if not any(rx.match(clean) for rx in _READ_RE):
            raise X2Error(
                f"refusing to request {path!r}: not a read-only endpoint. "
                f"This client only permits {list(READ_PATHS)}"
            )
        if not self._logged_in:
            raise X2AuthError("call login() first")

        url = f"{self.base_url}/{clean}"
        try:
            resp = self.session.get(url, params=params or {}, timeout=self.timeout)
        except requests.RequestException as exc:
            raise X2Error(f"GET {url} failed: {exc}") from exc

        if resp.status_code == 403:
            raise X2AuthError(f"GET {clean}: permission denied (403)")
        if resp.status_code != 200:
            raise X2Error(f"GET {clean}: HTTP {resp.status_code}")
        try:
            payload = resp.json()
        except ValueError as exc:
            raise X2Error(f"GET {clean}: response was not JSON") from exc

        if isinstance(payload, dict) and payload.get("ErrorCode"):
            raise X2Error(f"GET {clean}: {payload.get('ErrorCode')} "
                          f"{payload.get('ErrorMessage', '')}".strip())
        return payload

    # ── reads ──

    def get_sites(self, custom_id: typing.Optional[str] = None) -> list:
        """All sites the user can see."""
        payload = self._get("site", {"custom_id": custom_id} if custom_id else None)
        return _as_list(payload, "Sites")

    def get_site(self, site_id) -> dict:
        """Extended information about one site."""
        return self._get(f"site/{site_id}")

    def get_addresses(self, site_id) -> list:
        """Every sensor address at a site."""
        return _as_list(self._get(f"site/{site_id}/address"), "Addresslist")

    def get_vibration_addresses(self, site_id) -> list:
        """
        Just the vibration monitors (product types VIBRATION and
        MLT_PURE), which are the ones that upload time waveforms.
        """
        out = []
        for addr in self.get_addresses(site_id):
            if _is_vibration(addr):
                out.append(addr)
        return out

    def get_address_files(self, site_id, address) -> typing.List[MonitorFile]:
        """
        List the files a sensor has already uploaded, newest first.

        The ``pre_signed_link`` on each entry is valid for one hour.
        """
        quoted = urllib.parse.quote(str(address), safe="")
        payload = self._get(f"site/{site_id}/address/{quoted}/files")
        files = [MonitorFile.from_json(o, site_id, address)
                 for o in _as_list(payload, "Files")]
        files.sort(key=lambda f: (f.changed or _dt.datetime.min), reverse=True)
        return files

    def get_stats(
            self,
            site_id,
            address,
            stat_type: typing.Optional[str] = None,
            date_from: typing.Optional[_dt.datetime] = None,
            date_to: typing.Optional[_dt.datetime] = None,
    ) -> list:
        """
        Trend statistics for a sensor (RMS velocity, RMS/peak
        acceleration, RPM, temperature, battery).
        """
        params = {"address": address}
        if stat_type:
            params["type"] = stat_type
        if date_from:
            params["datefrom"] = date_from.strftime("%Y-%m-%d %H:%M:%S")
        if date_to:
            params["dateto"] = date_to.strftime("%Y-%m-%d %H:%M:%S")
        return _as_list(self._get(f"site/{site_id}/stats", params), "Series")

    # ── waveform download ──

    def download_waveform(self, file: MonitorFile) -> MEMSWaveform:
        """
        Fetch and decode one uploaded ``.bin`` waveform.

        The download is a plain GET of the pre-signed object-storage link
        the API handed back; it does not touch the sensor.
        """
        link = file.pre_signed_link or file.direct_link
        if not link:
            raise X2Error(f"file {file.name!r} has no download link")
        try:
            resp = self.session.get(link, timeout=self.timeout)
        except requests.RequestException as exc:
            raise X2Error(f"downloading {file.name!r} failed: {exc}") from exc
        if resp.status_code != 200:
            raise X2Error(f"downloading {file.name!r}: HTTP {resp.status_code} "
                          "(pre-signed links expire one hour after listing)")
        return decode_bin(resp.content)

    def latest_waveform(self, site_id, address) -> typing.Optional[MEMSWaveform]:
        """The most recently uploaded waveform for a sensor, or None."""
        for f in self.get_address_files(site_id, address):
            if f.is_waveform:
                return self.download_waveform(f)
        return None


# ── helpers ──

def _as_list(payload, key: str) -> list:
    """The API returns either a bare list or an object wrapping one."""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for candidate in (key, key.lower(), "data"):
            value = payload.get(candidate)
            if isinstance(value, list):
                return value
        # a single object where a list was expected
        return [payload]
    return []


def _is_vibration(address: dict) -> bool:
    """Identify a vibration monitor from its address record."""
    for key in ("TypeInt", "Type", "ProductType", "AddressType"):
        value = address.get(key)
        if isinstance(value, (int, float)) and int(value) in VIBRATION_PRODUCT_TYPES:
            return True
        if isinstance(value, str):
            upper = value.upper()
            if any(name in upper for name in VIBRATION_PRODUCT_TYPES.values()):
                return True
    # extra_device_type is "PRODUCT;VER_MAJ;VER_MIN;YEAR;MONTH;DAY"
    for extra in address.get("ExtraInfo", []) or []:
        if extra.get("ExtraRawKey") == "extra_device_type":
            head = str(extra.get("ExtraValue", "")).split(";")[0].strip().upper()
            if head in {str(v) for v in VIBRATION_PRODUCT_TYPES} | \
                       set(VIBRATION_PRODUCT_TYPES.values()):
                return True
    # any sensor reporting MLT trend keys is an MLT monitor
    for extra in address.get("ExtraInfo", []) or []:
        if str(extra.get("ExtraRawKey", "")).startswith("extra_mlt_"):
            return True
    return False
