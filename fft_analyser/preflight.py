"""
Preflight check for a live deployment: ``python -m fft_analyser.preflight``.

Answers, in order, the questions that actually stop a live connection:

1. Are the dependencies importable?
2. Can this machine *reach* the API host at all?  (A sandbox, a corporate
   egress proxy or a firewall will refuse the TLS tunnel long before any
   credential is checked - the failure looks like an auth problem but
   isn't.)
3. Do the credentials authenticate?
4. Does the site contain vibration monitors, and can their file listings
   be read?

Every step is read-only.  The only request that is not a GET is the login
the API requires for a session cookie; nothing is sent to any sensor.
"""

from __future__ import annotations

import os
import socket
import sys
import urllib.parse

OK = "  ok    "
BAD = "  FAIL  "
WARN = "  warn  "


def _line(mark: str, text: str) -> None:
    print(f"{mark}{text}")


def check_imports() -> bool:
    missing = []
    for mod in ("numpy", "scipy", "pandas", "requests", "dash", "plotly"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        _line(BAD, f"missing packages: {', '.join(missing)}")
        _line("", '        fix: pip install -e ".[analyser]"')
        return False
    _line(OK, "dependencies importable")
    return True


def check_reachable(base_url: str) -> bool:
    host = urllib.parse.urlparse(base_url).hostname or base_url
    try:
        addrs = socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        _line(BAD, f"DNS lookup for {host} failed: {exc}")
        _line("", "        this machine cannot resolve the API host")
        return False
    _line(OK, f"DNS: {host} -> {addrs[0][4][0]}")

    import requests
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/", timeout=15)
        _line(OK, f"TLS connection to {host} (HTTP {resp.status_code})")
        return True
    except requests.exceptions.ProxyError as exc:
        _line(BAD, f"an egress proxy refused the tunnel to {host}")
        _line("", f"        {exc}")
        _line("", "        run this from a network that permits the API host")
        return False
    except requests.exceptions.SSLError as exc:
        _line(BAD, f"TLS verification failed for {host}: {exc}")
        return False
    except requests.RequestException as exc:
        _line(BAD, f"cannot reach {host}: {exc}")
        return False


def check_login(base_url: str, user: str, password: str):
    from .x2_client import X2AuthError, X2Client, X2Error
    client = X2Client(user, password, base_url=base_url)
    try:
        info = client.login()
    except X2AuthError as exc:
        _line(BAD, f"login rejected: {exc}")
        return None
    except X2Error as exc:
        _line(BAD, f"login request failed: {exc}")
        return None
    _line(OK, f"logged in as {info.get('username') or user} "
              f"(customer {info.get('customerid')})")
    return client


def check_site(client, site_id: str) -> bool:
    from .x2_client import X2Error
    try:
        addresses = client.get_addresses(site_id)
    except X2Error as exc:
        _line(BAD, f"reading site {site_id} failed: {exc}")
        return False
    _line(OK, f"site {site_id}: {len(addresses)} address(es)")

    try:
        monitors = client.get_vibration_addresses(site_id)
    except X2Error as exc:
        _line(BAD, f"listing vibration monitors failed: {exc}")
        return False

    if not monitors:
        _line(WARN, "no vibration monitors recognised at this site")
        _line("", "        the detector looks for product types 21/202 and")
        _line("", "        extra_mlt_* keys; print an address record to check:")
        if addresses:
            sample = {k: addresses[0].get(k) for k in
                      ("Address", "Name", "Type", "TypeInt", "AddressType")}
            _line("", f"        {sample}")
        return False

    _line(OK, f"{len(monitors)} vibration monitor(s):")
    ok_any = False
    for mon in monitors[:10]:
        address = mon.get("Address")
        try:
            files = [f for f in client.get_address_files(site_id, address)
                     if f.is_waveform]
        except X2Error as exc:
            _line("", f"          {mon.get('Name') or address}: "
                      f"file listing failed - {exc}")
            continue
        if files:
            ok_any = True
            newest = files[0]
            _line("", f"          {mon.get('Name') or address}: "
                      f"{len(files)} waveform(s), newest {newest.name} "
                      f"({newest.size:,} B, {newest.changed})")
        else:
            _line("", f"          {mon.get('Name') or address}: "
                      "no waveform files uploaded yet")
    if not ok_any:
        _line(WARN, "no monitor has an uploaded waveform to analyse yet")
    return ok_any


def main() -> int:
    base = os.environ.get("X2_BASE_URL", "https://api.x2wireless.com")
    user = os.environ.get("X2_USERNAME")
    password = os.environ.get("X2_PASSWORD")
    site = os.environ.get("X2_SITE_ID")

    print(f"FFT analyser preflight - {base}\n")
    if not check_imports():
        return 1
    if not check_reachable(base):
        return 2
    if not (user and password and site):
        _line(WARN, "X2_USERNAME / X2_PASSWORD / X2_SITE_ID not set - "
                    "stopping after the reachability check")
        return 0
    client = check_login(base, user, password)
    if client is None:
        return 3
    if not check_site(client, site):
        return 4
    print("\nAll checks passed - `python -m fft_analyser --serve --connect` "
          "will come up live.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
