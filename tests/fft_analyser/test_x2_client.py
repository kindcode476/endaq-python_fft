"""
Tests for the X2 Cloud client - above all, that it cannot write.

No test here touches the network: every request is served by a fake
session that records what was asked for.
"""

import datetime as dt
import json

import pytest

from fft_analyser import x2_client
from fft_analyser.x2_client import (
    READ_PATHS,
    MonitorFile,
    X2AuthError,
    X2Client,
    X2Error,
)


class FakeResponse:
    def __init__(self, payload=None, status=200, content=b""):
        self.status_code = status
        self._payload = payload
        self.content = content

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class FakeSession:
    """Records every request; serves canned payloads."""

    def __init__(self, payloads=None):
        self.payloads = payloads or {}
        self.calls = []

    def post(self, url, data=None, timeout=None):
        self.calls.append(("POST", url, data))
        if url.endswith("/login"):
            return FakeResponse({"Status": "login_ok", "customerid": "42",
                                 "username": "tester"})
        return FakeResponse({"Status": "error"}, status=404)

    def get(self, url, params=None, timeout=None):
        self.calls.append(("GET", url, params))
        for key, payload in self.payloads.items():
            if key in url:
                if isinstance(payload, bytes):
                    return FakeResponse(content=payload)
                return FakeResponse(payload)
        return FakeResponse([], status=200)

    @property
    def methods(self):
        return {c[0] for c in self.calls}

    @property
    def urls(self):
        return [c[1] for c in self.calls]


@pytest.fixture
def client():
    session = FakeSession()
    c = X2Client("user", "pw", session=session)
    c.login()
    return c


class TestReadOnlyGuarantee:
    """The client must be structurally incapable of altering the site."""

    def test_no_control_or_config_method_exists(self):
        forbidden = ("control", "config", "post", "put", "delete", "patch",
                     "acknowledge", "enable", "disable", "set_", "write",
                     "send", "command", "logout")
        public = [n for n in dir(X2Client) if not n.startswith("_")]
        for name in public:
            assert not any(name.lower().startswith(f) for f in forbidden), \
                f"X2Client.{name} looks like it could mutate remote state"

    @pytest.mark.parametrize("path", [
        "site/3/address/1.ADDR/control/output_high_1",
        "site/3/address/1.ADDR/control/acknowledge",
        "site/3/address/1.ADDR/config",
        "site/3/address/1.ADDR/control/disable_input",
        "logout",
        "../../admin",
    ])
    def test_get_refuses_non_read_paths(self, client, path):
        with pytest.raises(X2Error, match="read-only"):
            client._get(path)
        # nothing left the process
        assert all("control" not in u and "config" not in u
                   for u in client.session.urls)

    def test_only_request_that_is_not_a_get_is_login(self, client):
        client.get_sites()
        client.get_addresses(3)
        client.get_address_files(3, "1.ADDR")
        posts = [c for c in client.session.calls if c[0] == "POST"]
        assert len(posts) == 1
        assert posts[0][1].endswith("/login")

    def test_allowlist_covers_only_documented_reads(self):
        assert set(READ_PATHS) == {
            r"^site$",
            r"^site/[^/]+$",
            r"^site/[^/]+/address$",
            r"^site/[^/]+/address/[^/]+/files$",
            r"^site/[^/]+/stats$",
            r"^site/[^/]+/logs$",
            r"^customersites$",
        }

    def test_requests_require_login(self):
        c = X2Client("u", "p", session=FakeSession())
        with pytest.raises(X2AuthError):
            c.get_sites()


class TestAuth:

    def test_login_sets_customer_id(self):
        c = X2Client("u", "p", session=FakeSession())
        c.login()
        assert c.customer_id == "42"

    def test_failed_login_raises(self):
        session = FakeSession()
        session.post = lambda url, data=None, timeout=None: FakeResponse(
            {"Status": "login_failed"})
        c = X2Client("u", "p", session=session)
        with pytest.raises(X2AuthError):
            c.login()

    def test_password_is_not_exposed_on_the_instance(self):
        c = X2Client("u", "secret", session=FakeSession())
        assert "secret" not in repr(c.__dict__.get("username", ""))
        assert not hasattr(c, "password")


class TestReads:

    def test_get_sites_unwraps_object_or_list(self):
        for payload in ([{"SiteID": 3}], {"Sites": [{"SiteID": 3}]}):
            c = X2Client("u", "p", session=FakeSession({"/site": payload}))
            c.login()
            assert c.get_sites() == [{"SiteID": 3}]

    def test_vibration_addresses_filtered_by_product_type(self):
        payload = [
            {"Address": "1.A", "TypeInt": 202, "Name": "Pump MLT"},
            {"Address": "1.B", "TypeInt": 1, "Name": "Smoke"},
            {"Address": "1.C", "TypeInt": 21, "Name": "Vib"},
            {"Address": "1.D", "TypeInt": 9, "Name": "Climate",
             "ExtraInfo": [{"ExtraRawKey": "extra_mlt_z_rms_vel",
                            "ExtraValue": "1.2"}]},
        ]
        c = X2Client("u", "p", session=FakeSession({"/address": payload}))
        c.login()
        found = [a["Address"] for a in c.get_vibration_addresses(3)]
        assert found == ["1.A", "1.C", "1.D"]

    def test_files_are_sorted_newest_first(self):
        payload = [
            {"name": "old.bin", "size": "10", "changed": "2024-01-01 00:00:00",
             "pre_signed_link": "http://x/old"},
            {"name": "new.bin", "size": "20", "changed": "2024-06-01 12:00:00",
             "pre_signed_link": "http://x/new"},
        ]
        c = X2Client("u", "p", session=FakeSession({"/files": payload}))
        c.login()
        files = c.get_address_files(3, "1.ADDR")
        assert [f.name for f in files] == ["new.bin", "old.bin"]
        assert files[0].size == 20
        assert files[0].changed == dt.datetime(2024, 6, 1, 12, 0, 0)
        assert files[0].is_waveform

    def test_address_is_url_quoted_in_files_path(self, client):
        client.get_address_files(3, "1.VIRTUAL.FLW/9999")
        assert "1.VIRTUAL.FLW%2F9999" in client.session.urls[-1]

    def test_stats_passes_documented_parameters(self, client):
        client.get_stats(3, "1.ADDR", "extra_raw_mlt_rms_vel",
                         dt.datetime(2024, 1, 2, 3, 4, 5),
                         dt.datetime(2024, 1, 3, 0, 0, 0))
        _, url, params = client.session.calls[-1]
        assert url.endswith("/site/3/stats")
        assert params == {
            "address": "1.ADDR",
            "type": "extra_raw_mlt_rms_vel",
            "datefrom": "2024-01-02 03:04:05",
            "dateto": "2024-01-03 00:00:00",
        }

    def test_permission_denied_becomes_auth_error(self):
        session = FakeSession()
        session.get = lambda url, params=None, timeout=None: FakeResponse(
            {"ErrorCode": "403"}, status=403)
        c = X2Client("u", "p", session=session)
        c.login()
        with pytest.raises(X2AuthError):
            c.get_sites()


class TestWaveformDownload:

    def test_download_decodes_a_real_binary(self):
        import pathlib
        raw = (pathlib.Path(__file__).parents[2] / "fft_analyser" / "sample_data"
               / "pure_mems" / "1axis_z_12800Hz.bin").read_bytes()
        c = X2Client("u", "p", session=FakeSession({"presigned": raw}))
        c.login()
        f = MonitorFile(name="data.bin", size=len(raw), changed=None,
                        pre_signed_link="https://storage/presigned?sig=abc")
        wave = c.download_waveform(f)
        assert wave.fs == 12800.0
        assert len(wave.data) == 64000

    def test_missing_link_raises(self, client):
        f = MonitorFile(name="data.bin", size=1, changed=None, pre_signed_link="")
        with pytest.raises(X2Error, match="no download link"):
            client.download_waveform(f)

    def test_expired_link_reports_clearly(self):
        session = FakeSession()
        session.get = lambda url, params=None, timeout=None: FakeResponse(status=403)
        c = X2Client("u", "p", session=session)
        c.login()
        f = MonitorFile(name="d.bin", size=1, changed=None,
                        pre_signed_link="https://storage/expired")
        with pytest.raises(X2Error, match="expire"):
            c.download_waveform(f)
