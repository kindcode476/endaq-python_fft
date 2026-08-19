"""A stand-in for the X2 Cloud API, so the Worker can be exercised for real.

Mimics only what the Worker uses: login, address list, file list, and the
pre-signed download. Serves the genuine vendor .bin so the whole chain -
Worker fetch, browser decode, FFT - runs on real data.
"""
import json
import http.server
import urllib.parse

BIN = "/home/user/endaq-python_fft/fft_analyser/sample_data/pure_mems/3axis_xyz_12800Hz.bin"
BIN2 = "/home/user/endaq-python_fft/fft_analyser/sample_data/pure_mems/1axis_z_12800Hz.bin"
GOOD = {"u": "operator", "p": "hunter2"}
COOKIE = "PHPSESSID=faketestsession"

SITES = [
    {"SiteID": "3", "Name": "Riverside Plant", "Description": "Main pumping station",
     "Location": "Riverside", "AddressCount": 4},
    {"SiteID": "11", "Name": "North Depot", "Description": "",
     "Location": "North", "AddressCount": 12},
]

ADDRESSES = [
    {"Address": "1.MLT.0001", "Name": "Pump A drive end", "TypeInt": 202,
     "ExtraInfo": [{"ExtraRawKey": "extra_last_upload", "ExtraValue": "2026-08-19 17:05:00"},
                   {"ExtraRawKey": "extra_mlt_z_rms_vel", "ExtraValue": "1.8"}]},
    {"Address": "1.MLT.0002", "Name": "Fan B bearing", "TypeInt": 21,
     "ExtraInfo": [{"ExtraRawKey": "extra_last_upload", "ExtraValue": "2026-08-19 16:40:00"}]},
    {"Address": "1.SMOKE.9", "Name": "Corridor smoke", "TypeInt": 1, "ExtraInfo": []},
    {"Address": "1.CLIMATE.4", "Name": "Office climate", "TypeInt": 9,
     "ExtraInfo": [{"ExtraRawKey": "extra_temperature", "ExtraValue": "21"}]},
]


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, status=200, cookie=False):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        if cookie:
            self.send_header("Set-Cookie", f"{COOKIE}; Path=/; HttpOnly")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if not self.path.endswith("/login"):
            self._json({"ErrorCode": "404"}, 404)
            return
        n = int(self.headers.get("content-length", 0))
        form = urllib.parse.parse_qs(self.rfile.read(n).decode())
        if form.get("u", [""])[0] == GOOD["u"] and form.get("p", [""])[0] == GOOD["p"]:
            self._json({"Status": "login_ok", "username": "operator",
                        "customerid": "7"}, cookie=True)
        else:
            self._json({"Status": "login_failed"})

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.strip("/")

        if path.startswith("blob/"):
            src = BIN2 if path.endswith("b") else BIN
            with open(src, "rb") as fh:
                data = fh.read()
            self.send_response(200)
            self.send_header("content-type", "application/octet-stream")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        # everything below requires the session cookie
        if COOKIE not in (self.headers.get("cookie") or ""):
            self._json({"ErrorCode": "403", "ErrorMessage": "no session"}, 403)
            return

        parts = path.split("/")
        if parts == ["site"]:
            self._json(SITES)
            return
        if parts[:1] == ["site"] and parts[2:] == ["address"]:
            self._json(ADDRESSES)
            return
        if parts[:1] == ["site"] and len(parts) == 5 and parts[2] == "address" and parts[4] == "files":
            addr = urllib.parse.unquote(parts[3])
            suffix = "b" if addr.endswith("0002") else "a"
            self._json([
                {"name": "waveform_latest.bin", "size": "173824",
                 "changed": "2026-08-19 17:05:00",
                 "pre_signed_link": f"http://127.0.0.1:9099/blob/{suffix}?sig=abc"},
                {"name": "waveform_older.bin", "size": "173824",
                 "changed": "2026-08-19 11:00:00",
                 "pre_signed_link": f"http://127.0.0.1:9099/blob/{suffix}?sig=old"},
            ])
            return
        self._json({"ErrorCode": "404"}, 404)


if __name__ == "__main__":
    http.server.HTTPServer(("127.0.0.1", 9099), Handler).serve_forever()
