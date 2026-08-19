"""
Interactive Dash UI for the FFT analyser.

Run with ``python -m fft_analyser`` (requires the optional ``dash``
dependency: ``pip install dash``), then open http://127.0.0.1:8050.

The front panel follows professional dynamic-signal-analyser conventions:
selectable spectrum quantity (peak/RMS amplitude, power, PSD, ASD),
window with live figures of merit (NENBW/ENBW, sidelobe, scalloping),
Welch overlap averaging (linear / peak hold), zero-padding, dB scales,
single-tone quality metrics (THD, THD+N, SNR, SINAD, ENOB, SFDR), a
spectrogram, and - because every built-in test signal carries its ground
truth - a live verification table scoring the analyser against known
spectral content.

Four data sources are available: the built-in ground-truth test signals,
a CSV upload, pureMEMS ``.bin`` waveforms, and a bank of up to five live
monitors read from the X2 Cloud API.  The cloud path is **read-only** -
see :py:mod:`fft_analyser.x2_client`.
"""

from __future__ import annotations

import base64
import datetime as _dt
import io
import os

import numpy as np
import pandas as pd

import dash
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go

from . import signals
from .analysis import (
    WINDOWS,
    DETRENDS,
    QUANTITIES,
    analyze,
    spectrogram,
    band_rms,
    window_figures,
    find_spectral_peaks,
    compare_to_expected,
    parseval_rms_error,
)
from .metrics import tone_metrics
from .monitors import (
    ISO_BAND,
    MAX_MONITORS,
    REAL_DATA_DEFAULTS,
    MonitorBank,
    overall_levels,
)
from .x2_client import PRIMARY_BASE, SECONDARY_BASE

#: The live monitor bank. Waveforms are megabytes each, so they stay in
#: process rather than round-tripping through a browser store.
BANK = MonitorBank(MAX_MONITORS)

# palette roles (validated reference palette, light mode)
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948"]
SEQ_BLUES = [[0.0, "#cde2fb"], [0.25, "#86b6ef"], [0.5, "#3987e5"],
             [0.75, "#1c5cab"], [1.0, "#0d366b"]]
STATUS_GOOD = "#006300"
STATUS_CRITICAL = "#d03b3b"
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

MAX_TIME_POINTS = 4096  # display decimation only; analysis uses all samples

SEGMENT_OPTIONS = [("whole record", 0), ("8192", 8192), ("4096", 4096),
                   ("2048", 2048), ("1024", 1024), ("512", 512), ("256", 256)]
OVERLAP_OPTIONS = [("auto (window)", -1), ("0 %", 0), ("25 %", 25),
                   ("50 %", 50), ("66 %", 66), ("75 %", 75), ("90 %", 90)]


def _layout(fig: go.Figure, xtitle: str, ytitle: str) -> go.Figure:
    fig.update_layout(
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT, color=INK_2, size=13),
        margin=dict(l=64, r=16, t=8, b=44),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#ffffff", font=dict(family=FONT, color=INK)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    font=dict(color=INK_2)),
    )
    axis = dict(
        gridcolor=GRID, linecolor=BASELINE, zerolinecolor=BASELINE,
        title_font=dict(color=MUTED, size=12), tickfont=dict(color=MUTED, size=11),
        showspikes=True, spikecolor=BASELINE, spikethickness=1, spikedash="dot",
    )
    fig.update_xaxes(title_text=xtitle, **axis)
    fig.update_yaxes(title_text=ytitle, **axis)
    return fig


def _tile(value_id: str, label: str) -> html.Div:
    return html.Div(
        [html.Div(id=value_id, className="tile-value"),
         html.Div(label, className="tile-label")],
        className="tile",
    )


def _control(label: str, component, wide: bool = False) -> html.Div:
    return html.Div([html.Label(label, className="ctl-label"), component],
                    className="ctl ctl-wide" if wide else "ctl")


def _monitor_card(mon, axis, selected: bool) -> html.Button:
    """
    One monitor in the strip - a button, so the whole tile selects it.
    State reads before the number: colour rail, then level, then identity.
    """
    s = mon.summary(axis)
    sel = " active" if selected else ""
    if s.get("error"):
        return html.Button(
            id={"type": "mon-btn", "key": mon.key}, n_clicks=0,
            className=f"mon bad{sel}", children=[
                html.Div(s["name"], className="mon-name", title=s["name"]),
                html.Div("—", className="mon-val"),
                html.Div(s["error"], className="mon-sub"),
            ])
    # a truncated record is still usable, but say so
    cls = "warn" if s["truncated"] else "ok"
    captured = s["captured"].strftime("%d %b %H:%M") if s["captured"] else "—"
    sub = f"{s['axis']} · {s['fs']:.0f} Hz · {s['duration']:.1f} s · {captured}"
    if s["truncated"]:
        sub += " · truncated"
    return html.Button(
        id={"type": "mon-btn", "key": mon.key}, n_clicks=0,
        className=f"mon {cls}{sel}", children=[
            html.Div(s["name"], className="mon-name", title=s["name"]),
            html.Div(f"{s['band_rms']:.3g} m/s²", className="mon-val"),
            html.Div(f"band RMS {ISO_BAND[0]:g}–{ISO_BAND[1]:g} Hz · crest "
                     f"{s['crest']:.1f}", className="mon-sub"),
            html.Div(sub, className="mon-sub"),
        ])


def _build_signal(name: str, noise_rms: float) -> signals.TestSignal:
    sig = signals.TEST_SIGNALS[name]()
    if noise_rms > 0:
        rng = np.random.default_rng(20260701)
        for c in sig.data.columns:
            sig.data[c] += rng.normal(0.0, noise_rms, len(sig.data))
        # adding white noise sets a known PSD floor: 2*sigma^2/fs
        floor = 2 * noise_rms ** 2 / sig.fs
        if sig.expected_psd_level:
            floor += sig.expected_psd_level
        sig.expected_psd_level = floor
        sig.description += f" + added noise (rms {noise_rms:g})"
    return sig


def _parse_upload(contents: str) -> pd.DataFrame:
    """Decode an uploaded CSV: first column = time in seconds, rest = channels."""
    _, payload = contents.split(",", 1)
    raw = base64.b64decode(payload)
    df = pd.read_csv(io.BytesIO(raw))
    df = df.set_index(df.columns[0])
    df.index.name = "time (s)"
    return df.apply(pd.to_numeric, errors="coerce").dropna(how="any")


def _expected_in_quantity(freq: float, amp_pk: float, quantity: str,
                          enbw_hz: float) -> float:
    """Convert a ground-truth (freq, peak amplitude) line into the displayed
    quantity.  For density quantities the conversion depends on ENBW - which
    is exactly the professional caveat that sinusoid heights in PSD/ASD are
    resolution- and window-dependent."""
    if quantity == "amplitude_peak":
        return amp_pk
    rms = amp_pk if freq == 0 else amp_pk / np.sqrt(2.0)
    if quantity == "amplitude_rms":
        return rms
    power = rms * rms
    if quantity == "power":
        return power
    if quantity == "psd":
        return power / enbw_hz
    return float(np.sqrt(power / enbw_hz))  # asd


def _to_display(values, quantity: str, yscale: str, floor_ref: float):
    values = np.asarray(values, dtype=float)
    if yscale != "db":
        return values
    is_power = QUANTITIES[quantity][2]
    factor = 10.0 if is_power else 20.0
    return factor * np.log10(np.maximum(values, floor_ref))


def create_app() -> Dash:
    app = Dash(__name__, title="FFT Analyser")

    app.layout = html.Div(className="page", children=[
        html.Div(className="header", children=[
            html.H1("FFT Analyser"),
            html.P("Measurement-grade spectra on the endaq-python foundation - "
                   "scaling per Heinzel 2002 / SciPy, windows per Harris 1978, "
                   "tone metrics per IEEE 1241", className="subtitle"),
        ]),

        html.Div(className="controls", children=[
            _control("Source", dcc.RadioItems(
                id="source", value="builtin",
                options=[{"label": " Test signal", "value": "builtin"},
                         {"label": " CSV upload", "value": "upload"},
                         {"label": " pureMEMS .bin", "value": "bin"},
                         {"label": " Live monitors", "value": "cloud"}],
                className="radio")),
            _control("Test signal", dcc.Dropdown(
                id="signal-name", clearable=False,
                value=list(signals.TEST_SIGNALS)[0],
                options=[{"label": k, "value": k} for k in signals.TEST_SIGNALS]),
                wide=True),
            _control("Added noise (rms)", dcc.Slider(
                id="noise", min=0.0, max=0.5, step=0.05, value=0.0,
                marks={0: "0", 0.25: "0.25", 0.5: "0.5"},
                tooltip={"placement": "bottom"})),
            _control("Upload CSV (time, ch1, ...)", dcc.Upload(
                id="upload", className="upload",
                children=html.Div(id="upload-label",
                                  children="Drop or select a file")), wide=True),
            _control("Upload .bin waveforms (up to 5)", dcc.Upload(
                id="upload-bin", multiple=True, className="upload",
                children=html.Div(id="upload-bin-label",
                                  children="Drop or select pureMEMS .bin files")),
                wide=True),
        ]),

        # ── X2 Cloud connection (read-only) ──
        html.Div(id="cloud-panel", className="controls", style={"display": "none"},
                 children=[
            _control("API host", dcc.Dropdown(
                id="x2-base", clearable=False,
                value=os.environ.get("X2_BASE_URL", PRIMARY_BASE),
                options=[{"label": "api.x2wireless.com (primary)", "value": PRIMARY_BASE},
                         {"label": "api.tritoncloud.se (secondary)", "value": SECONDARY_BASE},
                         {"label": "legacy login.x2wireless.com path",
                          "value": "https://login.x2wireless.com/TritonCloud/webapp_mobile/api"}]),
                wide=True),
            _control("Username", dcc.Input(id="x2-user", type="text",
                                           placeholder="API user",
                                           className="text-input")),
            _control("Password", dcc.Input(id="x2-pass", type="password",
                                           placeholder="API password",
                                           className="text-input")),
            _control("Site ID", dcc.Input(id="x2-site", type="text",
                                          placeholder="e.g. 3",
                                          className="text-input")),
            _control("​", html.Button("Connect (read-only)", id="x2-connect",
                                      n_clicks=0, className="btn")),
            _control(f"Monitors (choose up to {MAX_MONITORS})", dcc.Dropdown(
                id="x2-monitors", multi=True, placeholder="connect first",
                options=[]), wide=True),
            _control("​", html.Button("Fetch latest waveforms", id="x2-fetch",
                                      n_clicks=0, className="btn")),
            html.Div(id="x2-status", className="cloud-status"),
        ]),

        # ── the monitor bank ──
        html.Div(id="bank-panel", style={"display": "none"}, children=[
            html.Div(className="card", children=[
                html.H2("Monitor bank"),
                html.P(f"Up to {MAX_MONITORS} monitors. Overall level is the "
                       f"{ISO_BAND[0]:g}–{ISO_BAND[1]:g} Hz band RMS (ISO 20816 band), "
                       "computed with the gravity offset removed.",
                       className="hint"),
                html.Div(id="monitor-strip", className="monitor-strip"),
                dash_table.DataTable(
                    id="bank-table",
                    style_as_list_view=True,
                    style_table={"overflowX": "auto"},
                    style_header={
                        "backgroundColor": SURFACE, "color": MUTED,
                        "fontFamily": FONT, "fontSize": "12px",
                        "fontWeight": "600", "borderBottom": f"1px solid {BASELINE}"},
                    style_cell={
                        "backgroundColor": SURFACE, "color": INK,
                        "fontFamily": FONT, "fontSize": "13px",
                        "padding": "7px 12px", "textAlign": "right",
                        "borderBottom": f"1px solid {GRID}"},
                    style_cell_conditional=[
                        {"if": {"column_id": "monitor"}, "textAlign": "left"},
                        {"if": {"column_id": "status"}, "textAlign": "left"},
                    ]),
            ]),
            html.Div(className="controls", children=[
                _control("Axis", dcc.RadioItems(
                    id="active-axis", value="Z", className="radio",
                    options=[{"label": f" {a}", "value": a} for a in ("X", "Y", "Z")])),
                _control("Compare monitors", dcc.RadioItems(
                    id="overlay", value="off", className="radio",
                    options=[{"label": " off", "value": "off"},
                             {"label": " overlay spectra", "value": "on"}])),
                _control("Live feed", dcc.RadioItems(
                    id="live", value="off", className="radio",
                    options=[{"label": " off", "value": "off"},
                             {"label": " on", "value": "on"}])),
                _control("Check for new uploads every", dcc.Dropdown(
                    id="live-interval", clearable=False,
                    value=int(os.environ.get("X2_POLL_SECONDS", 300)),
                    options=[{"label": "30 s", "value": 30},
                             {"label": "1 min", "value": 60},
                             {"label": "5 min", "value": 300},
                             {"label": "15 min", "value": 900},
                             {"label": "1 hour", "value": 3600}])),
                html.Div(id="live-status", className="cloud-status"),
            ]),
        ]),

        html.Div(className="controls", children=[
            _control("Quantity", dcc.Dropdown(
                id="quantity", value="amplitude_peak", clearable=False,
                options=[{"label": QUANTITIES[q][0], "value": q}
                         for q in QUANTITIES]), wide=True),
            _control("Window", dcc.Dropdown(
                id="window", value="hann", clearable=False,
                options=[{"label": w, "value": w} for w in WINDOWS])),
            _control("Detrend", dcc.Dropdown(
                id="detrend", value="none", clearable=False,
                options=[{"label": d, "value": d} for d in DETRENDS])),
            _control("Segment length", dcc.Dropdown(
                id="nperseg", value=0, clearable=False,
                options=[{"label": lbl, "value": v} for lbl, v in SEGMENT_OPTIONS])),
            _control("Overlap", dcc.Dropdown(
                id="overlap", value=-1, clearable=False,
                options=[{"label": lbl, "value": v} for lbl, v in OVERLAP_OPTIONS])),
            _control("Averaging", dcc.Dropdown(
                id="averaging", value="linear", clearable=False,
                options=[{"label": "linear (RMS)", "value": "linear"},
                         {"label": "peak hold", "value": "peak_hold"},
                         {"label": "none (single record)", "value": "none"}])),
            _control("Zero-pad", dcc.Dropdown(
                id="pad", value=1, clearable=False,
                options=[{"label": f"{p}x", "value": p} for p in (1, 2, 4, 8)])),
            _control("Amplitude scale", dcc.RadioItems(
                id="yscale", value="linear",
                options=[{"label": " linear", "value": "linear"},
                         {"label": " dB", "value": "db"}],
                className="radio")),
            _control("Frequency axis", dcc.RadioItems(
                id="xscale", value="linear",
                options=[{"label": " linear", "value": "linear"},
                         {"label": " log", "value": "log"}],
                className="radio")),
            _control("Max peaks", dcc.Input(
                id="npeaks", type="number", value=10, min=1, max=50, step=1,
                className="num-input")),
            _control("Harmonic markers", dcc.RadioItems(
                id="harmonics", value="off",
                options=[{"label": " off", "value": "off"},
                         {"label": " on", "value": "on"}],
                className="radio")),
        ]),

        html.P(id="signal-description", className="description"),
        html.P(id="window-info", className="window-info"),

        html.Div(className="tiles", children=[
            _tile("tile-samples", "samples"),
            _tile("tile-fs", "sample rate"),
            _tile("tile-binwidth", "bin width Δf"),
            _tile("tile-enbw", "ENBW / bin"),
            _tile("tile-averages", "averages"),
            _tile("tile-parseval", "Parseval RMS error"),
            _tile("tile-verdict", "ground-truth check"),
        ]),

        html.Div(className="card", children=[
            html.H2("Time domain"),
            dcc.Graph(id="time-plot", config={"displayModeBar": False}),
        ]),
        html.Div(className="card", children=[
            html.H2("Spectrum"),
            dcc.Graph(id="spectrum-plot", config={"displayModeBar": False}),
        ]),
        html.Div(className="card", children=[
            html.H2("Tone metrics"),
            html.P("Single-tone quality figures from main-lobe power integration "
                   "(blackman-harris window). Meaningful when one tone dominates.",
                   className="hint"),
            html.Div(id="metrics-grid", className="metrics-grid"),
        ]),
        html.Div(className="card", children=[
            html.H2("Spectrogram"),
            html.P("Short-time PSD (dB re units²/Hz), same scaling conventions.",
                   className="hint"),
            dcc.Graph(id="spectrogram-plot", config={"displayModeBar": False}),
        ]),
        html.Div(className="card", children=[
            html.H2("Peak verification"),
            html.P("Each expected spectral line vs. the nearest detected peak, "
                   "scored in peak-amplitude units. Pass = frequency within "
                   "1 Hz and amplitude within 5 %.", className="hint"),
            dash_table.DataTable(
                id="peaks-table",
                style_as_list_view=True,
                style_table={"overflowX": "auto"},
                style_header={
                    "backgroundColor": SURFACE, "color": MUTED,
                    "fontFamily": FONT, "fontSize": "12px",
                    "fontWeight": "600", "borderBottom": f"1px solid {BASELINE}"},
                style_cell={
                    "backgroundColor": SURFACE, "color": INK,
                    "fontFamily": FONT, "fontSize": "13px",
                    "padding": "8px 12px", "textAlign": "right",
                    "borderBottom": f"1px solid {GRID}"},
                style_data_conditional=[
                    {"if": {"filter_query": '{result} contains "pass"',
                            "column_id": "result"},
                     "color": STATUS_GOOD, "fontWeight": "600"},
                    {"if": {"filter_query": '{result} contains "FAIL"',
                            "column_id": "result"},
                     "color": STATUS_CRITICAL, "fontWeight": "600"},
                ]),
        ]),
        dcc.Store(id="uploaded-data"),
        dcc.Store(id="bank-version", data=0),
        dcc.Store(id="active-monitor", data=None),
        dcc.Interval(id="live-tick", interval=60_000, disabled=True),
    ])

    _add_css(app)

    @app.callback(Output("uploaded-data", "data"),
                  Output("upload-label", "children"),
                  Output("source", "value"),
                  Input("upload", "contents"),
                  State("upload", "filename"),
                  prevent_initial_call=True)
    def store_upload(contents, filename):
        try:
            df = _parse_upload(contents)
        except Exception as exc:
            return dash.no_update, f"Could not read file: {exc}", dash.no_update
        return ({"records": df.reset_index().to_dict("records"),
                 "index": df.index.name, "name": filename},
                f"Loaded: {filename} ({len(df)} rows)", "upload")

    # ── source-dependent panel visibility ──

    @app.callback(Output("cloud-panel", "style"),
                  Output("bank-panel", "style"),
                  Output("detrend", "value"),
                  Output("nperseg", "value"),
                  Output("averaging", "value"),
                  Input("source", "value"))
    def toggle_panels(source):
        show = {"display": "flex"}
        hide = {"display": "none"}
        bank_show = {"display": "block"}
        if source in ("cloud", "bin"):
            # measured accelerometer data: strip the ~1 g static offset and
            # average, or the 0 Hz bin dominates every reading
            return (show if source == "cloud" else hide, bank_show,
                    REAL_DATA_DEFAULTS["detrend"],
                    REAL_DATA_DEFAULTS["nperseg"],
                    REAL_DATA_DEFAULTS["averaging"])
        # synthetic signals are exact: leave them untouched and unaveraged
        return hide, hide, "none", 0, "linear"

    # ── .bin uploads populate the bank ──

    @app.callback(Output("upload-bin-label", "children"),
                  Output("bank-version", "data", allow_duplicate=True),
                  Output("source", "value", allow_duplicate=True),
                  Input("upload-bin", "contents"),
                  State("upload-bin", "filename"),
                  State("bank-version", "data"),
                  prevent_initial_call=True)
    def load_bin_uploads(contents_list, filenames, version):
        if not contents_list:
            return dash.no_update, dash.no_update, dash.no_update
        BANK.clear()
        loaded, failed = [], []
        for contents, name in zip(contents_list, filenames or []):
            try:
                _, payload = contents.split(",", 1)
                mon = BANK.add_bytes(base64.b64decode(payload), name)
                (loaded if mon.ok else failed).append(name)
            except Exception:
                failed.append(name)
        msg = f"Loaded {len(loaded)} waveform(s)"
        if failed:
            msg += f" · failed: {', '.join(failed)}"
        return msg, (version or 0) + 1, "bin"

    # ── X2 Cloud: connect (read-only) ──

    @app.callback(Output("x2-monitors", "options"),
                  Output("x2-status", "children"),
                  Output("x2-status", "className"),
                  Input("x2-connect", "n_clicks"),
                  State("x2-base", "value"),
                  State("x2-user", "value"),
                  State("x2-pass", "value"),
                  State("x2-site", "value"),
                  prevent_initial_call=True)
    def connect_cloud(_clicks, base, user, password, site_id):
        if not (user and password and site_id):
            return [], "Enter username, password and site ID.", "cloud-status err"
        try:
            found = BANK.connect(user, password, base, site_id)
        except Exception as exc:
            return [], f"Connection failed: {exc}", "cloud-status err"
        options = [{"label": f"{a.get('Name') or a.get('Address')} "
                             f"({a.get('Address')})",
                    "value": str(a.get("Address"))} for a in found]
        if not options:
            return [], (f"Connected to site {site_id}, but no vibration monitors "
                        "were found there."), "cloud-status err"
        return options, (f"Connected read-only. {len(options)} vibration monitor(s) "
                         f"at site {site_id}. Select up to {MAX_MONITORS}, then "
                         "fetch."), "cloud-status ok"

    # ── X2 Cloud: fetch already-uploaded waveforms ──

    @app.callback(Output("x2-status", "children", allow_duplicate=True),
                  Output("x2-status", "className", allow_duplicate=True),
                  Output("bank-version", "data", allow_duplicate=True),
                  Input("x2-fetch", "n_clicks"),
                  State("x2-monitors", "value"),
                  State("bank-version", "data"),
                  prevent_initial_call=True)
    def fetch_cloud(_clicks, addresses, version):
        if not addresses:
            return "Select at least one monitor first.", "cloud-status err", dash.no_update
        try:
            mons = BANK.load_from_cloud(addresses[:MAX_MONITORS])
        except Exception as exc:
            return f"Fetch failed: {exc}", "cloud-status err", dash.no_update
        ok = [m for m in mons if m.ok]
        bad = [f"{m.name}: {m.error}" for m in mons if not m.ok]
        msg = f"Fetched {len(ok)} of {len(mons)} waveform(s)."
        cls = "cloud-status ok" if ok else "cloud-status err"
        if bad:
            msg += " Problems — " + "; ".join(bad)
            cls = "cloud-status err" if not ok else "cloud-status"
        return msg, cls, (version or 0) + 1

    # ── the bank view ──

    @app.callback(Output("monitor-strip", "children"),
                  Output("bank-table", "data"),
                  Output("bank-table", "columns"),
                  Output("active-monitor", "data"),
                  Output("active-axis", "options"),
                  Output("active-axis", "value"),
                  Input("bank-version", "data"),
                  Input("active-axis", "value"),
                  Input("active-monitor", "data"),
                  State("active-monitor", "data"))
    def refresh_bank(_version, axis, _selected, current):
        mons = list(BANK)
        if not mons:
            return ([html.Div("No monitors loaded.", className="hint")],
                    [], [], None, [{"label": " Z", "value": "Z"}], "Z")

        keys = [m.key for m in mons]
        value = current if current in keys else keys[0]

        cards = [_monitor_card(m, axis, m.key == value) for m in mons]
        table = overall_levels(mons, axis)
        rows = table.round(4).to_dict("records")
        cols = [{"name": c, "id": c} for c in table.columns]

        active = BANK.get(value)
        axes = active.axes if active and active.axes else ["Z"]
        axis_options = [{"label": f" {a}", "value": a} for a in axes]
        axis_value = axis if axis in axes else axes[-1]
        return cards, rows, cols, value, axis_options, axis_value

    # ── clicking a monitor tile selects it ──

    @app.callback(Output("active-monitor", "data", allow_duplicate=True),
                  Input({"type": "mon-btn", "key": dash.ALL}, "n_clicks"),
                  prevent_initial_call=True)
    def select_monitor(_clicks):
        ctx = dash.callback_context
        if not ctx.triggered:
            return dash.no_update
        # Rebuilding the strip re-fires this callback for every freshly
        # rendered button with n_clicks=0. Only a real click carries a
        # non-zero count; anything else would reset the selection.
        if not ctx.triggered[0].get("value"):
            return dash.no_update
        triggered = ctx.triggered_id
        if not isinstance(triggered, dict):
            return dash.no_update
        return triggered.get("key")

    # ── live feed: poll for newer uploads (read-only) ──

    @app.callback(Output("live-tick", "disabled"),
                  Output("live-tick", "interval"),
                  Input("live", "value"),
                  Input("live-interval", "value"),
                  Input("source", "value"))
    def configure_live(live, seconds, source):
        # only the cloud source has anything new to poll for
        on = live == "on" and source == "cloud"
        return (not on), int(seconds or 60) * 1000

    @app.callback(Output("bank-version", "data", allow_duplicate=True),
                  Output("live-status", "children"),
                  Input("live-tick", "n_intervals"),
                  State("bank-version", "data"),
                  State("live", "value"),
                  prevent_initial_call=True)
    def poll_live(_ticks, version, live):
        if live != "on" or BANK.client is None:
            return dash.no_update, ""
        stamp = _dt.datetime.now().strftime("%H:%M:%S")
        addresses = [m.address for m in BANK if m.address]
        if not addresses:
            return dash.no_update, f"Live: nothing to poll (checked {stamp})"
        try:
            before = {m.key: m.captured for m in BANK}
            BANK.load_from_cloud(addresses)
        except Exception as exc:
            return dash.no_update, f"Live: check failed at {stamp} — {exc}"
        fresh = [m.name for m in BANK if m.captured != before.get(m.key)]
        if fresh:
            return (version or 0) + 1, (f"Live: new upload(s) at {stamp} — "
                                        + ", ".join(fresh))
        return dash.no_update, f"Live: no new uploads (checked {stamp})"

    @app.callback(
        Output("time-plot", "figure"),
        Output("spectrum-plot", "figure"),
        Output("spectrogram-plot", "figure"),
        Output("peaks-table", "data"),
        Output("peaks-table", "columns"),
        Output("metrics-grid", "children"),
        Output("signal-description", "children"),
        Output("window-info", "children"),
        Output("tile-samples", "children"),
        Output("tile-fs", "children"),
        Output("tile-binwidth", "children"),
        Output("tile-enbw", "children"),
        Output("tile-averages", "children"),
        Output("tile-parseval", "children"),
        Output("tile-verdict", "children"),
        Output("tile-verdict", "style"),
        Input("source", "value"),
        Input("signal-name", "value"),
        Input("noise", "value"),
        Input("quantity", "value"),
        Input("window", "value"),
        Input("detrend", "value"),
        Input("nperseg", "value"),
        Input("overlap", "value"),
        Input("averaging", "value"),
        Input("pad", "value"),
        Input("yscale", "value"),
        Input("xscale", "value"),
        Input("npeaks", "value"),
        Input("harmonics", "value"),
        Input("uploaded-data", "data"),
        Input("bank-version", "data"),
        Input("active-monitor", "data"),
        Input("active-axis", "value"),
        Input("overlay", "value"),
    )
    def update(source, signal_name, noise, quantity, window, detrend,
               nperseg, overlap, averaging, pad, yscale, xscale, npeaks,
               harmonics, uploaded, _bank_version, active_monitor,
               active_axis, overlay):
        expected_peaks, expected_band, expected_psd = [], None, None
        units = "units"

        if source in ("bin", "cloud"):
            mon = BANK.get(active_monitor) if active_monitor else None
            if mon is None and len(BANK):
                mon = list(BANK)[0]
            if mon is None or not mon.ok:
                reason = (mon.error if mon is not None else
                          "No waveforms loaded yet.")
                return _idle_view(reason)
            df = mon.channel(active_axis)
            units = "m/s²"
            captured = (mon.captured.strftime("%Y-%m-%d %H:%M")
                        if mon.captured else "unknown time")
            description = (f"{mon.name} · axis {df.columns[0]} · "
                           f"{mon.waveform.fs:.0f} Hz · "
                           f"{mon.waveform.duration:.2f} s · captured {captured}"
                           + (f" · file {mon.file_name}" if mon.file_name else "")
                           + (" · record truncated" if mon.waveform.truncated else ""))
        elif source == "upload" and uploaded:
            df = pd.DataFrame(uploaded["records"]).set_index(uploaded["index"])
            description = f"Uploaded file: {uploaded['name']}"
        else:
            sig = _build_signal(signal_name, float(noise or 0.0))
            df = sig.data
            expected_peaks, expected_band = sig.expected_peaks, sig.expected_band
            expected_psd = sig.expected_psd_level
            description = sig.description

        n = len(df)
        result = analyze(
            df, quantity=quantity, window=window, detrend=detrend,
            nperseg=int(nperseg) or None,
            overlap=None if int(overlap) < 0 else int(overlap) / 100.0,
            averaging=averaging, pad_factor=int(pad),
        )

        # peak detection & verification always run in peak-amplitude units
        amp_view = result.as_quantity("amplitude_peak")
        peaks_amp = find_spectral_peaks(amp_view, n_peaks=int(npeaks or 10))
        parseval = parseval_rms_error(df)
        m = tone_metrics(df)

        figures = window_figures(window, result.nperseg)
        info = WINDOWS[window]
        window_line = (
            f"{window}: NENBW {result.nenbw_bins:.4f} bins · "
            f"ENBW {result.enbw_hz:.4g} Hz · sidelobe {info.sidelobe_db:g} dB · "
            f"scalloping {figures['scalloping_db']:+.2f} dB · "
            f"recommended overlap {info.recommended_overlap:.0%} · "
            f"best for: {info.use_for}"
        )

        time_fig = _time_figure(df, units)
        if source in ("bin", "cloud") and overlay == "on" and len(BANK) > 1:
            spec_fig = _overlay_figure(
                active_axis, quantity, window, detrend,
                int(nperseg) or None,
                None if int(overlap) < 0 else int(overlap) / 100.0,
                averaging, int(pad), yscale, xscale, units)
        else:
            spec_fig = _spectrum_figure(result, peaks_amp, expected_peaks,
                                        expected_band, expected_psd, yscale,
                                        xscale, harmonics == "on",
                                        m.fundamental_freq, units)
        sgram_fig = _spectrogram_figure(df, units)
        rows, cols = _table(peaks_amp, expected_peaks, detrend)
        verdict, verdict_style = _verdict(rows, expected_peaks)
        metrics_children = _metrics_grid(m, result, df)

        return (time_fig, spec_fig, sgram_fig, rows, cols, metrics_children,
                description, window_line,
                f"{n:,}", f"{result.fs:,.0f} Hz", f"{result.bin_width:.3g} Hz",
                f"{result.enbw_hz:.3g} Hz", f"{result.n_segments}",
                f"{parseval:.2e}", verdict, verdict_style)

    return app


def _idle_view(message: str):
    """Everything blank but the explanation - used before data is loaded."""
    blank = _layout(go.Figure(), "", "")
    blank.update_layout(height=200)
    return (blank, blank, blank, [], [], [], message, "",
            "—", "—", "—", "—", "—", "—", "n/a", {})


def _time_figure(df: pd.DataFrame, units: str = "units") -> go.Figure:
    step = max(1, len(df) // MAX_TIME_POINTS)
    view = df.iloc[::step]
    fig = go.Figure()
    for i, c in enumerate(df.columns):
        fig.add_trace(go.Scatter(
            x=view.index, y=view[c], name=str(c), mode="lines",
            line=dict(color=SERIES[i % len(SERIES)], width=2),
            hovertemplate="%{y:.4g}<extra>" + str(c) + "</extra>"))
    fig = _layout(fig, "time (s)", f"amplitude ({units})")
    fig.update_layout(showlegend=len(df.columns) > 1, height=260)
    return fig


def _overlay_figure(axis, quantity, window, detrend, nperseg, overlap,
                    averaging, pad, yscale, xscale, units) -> go.Figure:
    """All loaded monitors' spectra on one axis - the comparison view."""
    db = yscale == "db"
    fig = go.Figure()
    top = 0.0
    curves = []
    for mon in BANK:
        if not mon.ok:
            continue
        df = mon.channel(axis)
        res = analyze(df, quantity=quantity, window=window, detrend=detrend,
                      nperseg=nperseg, overlap=overlap, averaging=averaging,
                      pad_factor=pad)
        series = res.spectrum[res.spectrum.columns[0]]
        top = max(top, float(series.max()))
        curves.append((mon.name, res, series))

    floor = top * 1e-10 + 1e-300
    for i, (name, res, series) in enumerate(curves):
        values = series.to_numpy()
        y = _to_db(values, quantity, floor) if db else values
        fig.add_trace(go.Scatter(
            x=series.index, y=y, name=name, mode="lines",
            line=dict(color=SERIES[i % len(SERIES)], width=2),
            hovertemplate="%{y:.4g}<extra>" + name + "</extra>"))

    label = QUANTITIES[quantity][0]
    unit = QUANTITIES[quantity][1].format(u=units)
    fig = _layout(fig, "frequency (Hz)",
                  f"{label} (dB re 1 {unit})" if db else f"{label} ({unit})")
    fig.update_layout(height=380, showlegend=True)
    if xscale == "log":
        fig.update_xaxes(type="log")
    return fig


def _to_db(values, quantity, floor):
    factor = 10.0 if QUANTITIES[quantity][2] else 20.0
    return factor * np.log10(np.maximum(np.asarray(values, dtype=float), floor))


def _spectrum_figure(result, peaks_amp, expected_peaks, expected_band,
                     expected_psd, yscale, xscale, show_harmonics,
                     f0, units="units") -> go.Figure:
    spectrum = result.spectrum
    quantity = result.quantity
    top = float(np.nanmax(spectrum.to_numpy()))
    floor_ref = top * 1e-10 + 1e-300
    db = yscale == "db"

    def disp(v):
        return _to_display(v, quantity, yscale, floor_ref)

    fig = go.Figure()

    if expected_band is not None:
        x0 = expected_band[0]
        if xscale == "log":
            x0 = max(x0, float(spectrum.index[1]))
        fig.add_vrect(x0=x0, x1=expected_band[1], fillcolor=GRID, opacity=0.45,
                      line_width=0, layer="below",
                      annotation_text="expected band",
                      annotation_position="inside bottom left",
                      annotation_font=dict(color=MUTED, size=11))

    if show_harmonics and f0 > 0:
        h = 2
        while h * f0 < float(spectrum.index[-1]) and h <= 9:
            fig.add_vline(x=h * f0, line_color=BASELINE, line_dash="dot",
                          line_width=1, annotation_text=f"H{h}",
                          annotation_position="top",
                          annotation_font=dict(color=MUTED, size=10))
            h += 1

    for i, c in enumerate(spectrum.columns):
        fig.add_trace(go.Scatter(
            x=spectrum.index, y=disp(spectrum[c]), name=str(c), mode="lines",
            line=dict(color=SERIES[i % len(SERIES)], width=2),
            hovertemplate="%{y:.4g}<extra>" + str(c) + "</extra>"))

    if len(peaks_amp):
        # mark detected peaks on the displayed curve (nearest bin lookup)
        freqs = spectrum.index.to_numpy(dtype=float)
        col0 = spectrum.columns[0]
        idx = np.clip(np.searchsorted(freqs, peaks_amp["frequency (Hz)"]),
                      0, len(freqs) - 1)
        fig.add_trace(go.Scatter(
            x=freqs[idx], y=disp(spectrum[col0].to_numpy()[idx]),
            name="detected peaks", mode="markers",
            marker=dict(color=SERIES[0], size=9, symbol="circle",
                        line=dict(color=SURFACE, width=2)),
            hovertemplate="%{x:.2f} Hz<extra>detected</extra>"))

    if expected_peaks:
        fx = [f for f, _ in expected_peaks]
        fy = [_expected_in_quantity(f, a, quantity, result.enbw_hz)
              for f, a in expected_peaks]
        fig.add_trace(go.Scatter(
            x=fx, y=disp(fy), name="expected (ground truth)", mode="markers",
            marker=dict(color=INK_2, size=10, symbol="x-thin",
                        line=dict(color=INK_2, width=2)),
            hovertemplate="%{x:.2f} Hz<extra>expected</extra>"))

    if expected_psd is not None and quantity in ("psd", "asd"):
        level = expected_psd if quantity == "psd" else float(np.sqrt(expected_psd))
        fig.add_hline(y=float(disp(np.array([level]))[0]), line_color=INK_2,
                      line_dash="dash", line_width=1,
                      annotation_text="expected noise floor (2σ²/fs)",
                      annotation_position="top right",
                      annotation_font=dict(color=INK_2, size=11))

    unit = QUANTITIES[quantity][1].format(u=units)
    label = f"{result.ylabel} ({unit})"
    if db:
        label = f"{result.ylabel} (dB re 1 {unit})"
    fig = _layout(fig, "frequency (Hz)", label)
    fig.update_layout(height=380)
    if xscale == "log":
        fig.update_xaxes(type="log")
    return fig


def _spectrogram_figure(df: pd.DataFrame, units: str = "units") -> go.Figure:
    sgram = spectrogram(df)
    z = 10 * np.log10(np.maximum(sgram.to_numpy().T,
                                 np.nanmax(sgram.to_numpy()) * 1e-10 + 1e-300))
    fig = go.Figure(go.Heatmap(
        x=sgram.index.to_numpy(), y=sgram.columns.to_numpy(), z=z,
        colorscale=SEQ_BLUES,
        colorbar=dict(title=dict(text=f"dB re 1 {units}²/Hz",
                                 font=dict(color=MUTED, size=11)),
                      tickfont=dict(color=MUTED, size=10), thickness=12),
        hovertemplate="t=%{x:.3f} s, f=%{y:.1f} Hz: %{z:.1f} dB<extra></extra>"))
    fig = _layout(fig, "time (s)", "frequency (Hz)")
    fig.update_layout(height=320, hovermode="closest")
    fig.update_xaxes(showspikes=False)
    fig.update_yaxes(showspikes=False)
    return fig


def _metric_item(label: str, value: str) -> html.Div:
    return html.Div([html.Div(value, className="metric-value"),
                     html.Div(label, className="metric-label")],
                    className="metric")


def _fmt_db(v: float) -> str:
    return "∞" if np.isinf(v) else f"{v:.1f} dB"


def _metrics_grid(m, result, df) -> list:
    total_rms = band_rms(result)
    # single-tone figures are meaningless on broadband machine vibration;
    # say so rather than printing an authoritative-looking ENOB
    dominant = m.sfdr_dbc >= 20 if np.isfinite(m.sfdr_dbc) else True
    header = [] if dominant else [html.Div(
        "No dominant tone — the single-tone figures below (THD, SINAD, ENOB) "
        "do not apply to broadband vibration.",
        className="metrics-caveat")]
    return header + [
        _metric_item("fundamental", f"{m.fundamental_freq:.2f} Hz"),
        _metric_item("fund. amplitude (pk)", f"{m.fundamental_peak:.4g}"),
        _metric_item("THD", "~0" if m.thd_pct < 1e-6 else f"{m.thd_pct:.3f} %"),
        _metric_item("THD+N", "~0" if m.thd_n_pct < 1e-6 else f"{m.thd_n_pct:.3f} %"),
        _metric_item("SNR", _fmt_db(m.snr_db)),
        _metric_item("SINAD", _fmt_db(m.sinad_db)),
        _metric_item("ENOB", "-" if np.isinf(m.enob_bits) else f"{m.enob_bits:.2f} bits"),
        _metric_item("SFDR", _fmt_db(m.sfdr_dbc) + ("" if np.isinf(m.sfdr_dbc) else "c")),
        _metric_item("signal RMS (time dom.)", f"{m.rms:.4g}"),
        _metric_item("RMS from PSD ∫", f"{total_rms:.4g}"),
        _metric_item("crest factor", f"{m.crest_factor:.3f}"),
        _metric_item("harmonics in THD", f"{m.n_harmonics}"),
    ]


def _table(peaks, expected_peaks, detrend):
    if not expected_peaks:
        if not len(peaks):
            return [], [{"name": "no peaks detected", "id": "result"}]
        rows = [{"measured freq (Hz)": f"{r['frequency (Hz)']:.3f}",
                 "measured amplitude (pk)": f"{r['amplitude']:.4g}",
                 "result": "-"} for _, r in peaks.iterrows()]
        cols = [{"name": k, "id": k} for k in rows[0]]
        return rows, cols

    # a detrended signal legitimately loses its DC line: skip 0 Hz truth rows
    expected = [(f, a) for f, a in expected_peaks if f > 0 or detrend == "none"]
    cmp = compare_to_expected(peaks, expected)
    rows = []
    for _, r in cmp.iterrows():
        rows.append({
            "expected freq (Hz)": f"{r['expected freq (Hz)']:.2f}",
            "measured freq (Hz)": f"{r['measured freq (Hz)']:.3f}",
            "freq error (Hz)": f"{r['freq error (Hz)']:+.3f}",
            "expected amplitude": f"{r['expected amplitude']:.4g}",
            "measured amplitude": f"{r['measured amplitude']:.4g}",
            "amplitude error (%)": f"{r['amplitude error (%)']:+.2f}",
            "result": "✓ pass" if r["pass"] else "✗ FAIL",
        })
    cols = [{"name": k, "id": k} for k in rows[0]]
    return rows, cols


def _verdict(rows, expected_peaks):
    if not expected_peaks:
        return "n/a", {}
    failed = sum(1 for r in rows if "FAIL" in r["result"])
    if failed:
        return (f"✗ {failed}/{len(rows)} failed",
                {"color": STATUS_CRITICAL})
    return f"✓ {len(rows)}/{len(rows)} pass", {"color": STATUS_GOOD}


def _add_css(app: Dash) -> None:
    app.index_string = app.index_string.replace("</head>", """
<style>
  body { margin: 0; background: %(PAGE)s; color: %(INK)s;
         font-family: %(FONT)s; }
  .page { max-width: 1180px; margin: 0 auto; padding: 20px 24px 48px; }
  .header h1 { font-size: 22px; margin: 8px 0 2px; color: %(INK)s; }
  .subtitle { color: %(INK_2)s; margin: 0 0 16px; font-size: 14px; }
  .controls { display: flex; flex-wrap: wrap; gap: 12px 20px;
              background: %(SURFACE)s; border: 1px solid rgba(11,11,11,0.10);
              border-radius: 10px; padding: 14px 16px; margin-bottom: 12px; }
  .ctl { min-width: 130px; flex: 1 1 130px; }
  .ctl-wide { min-width: 210px; flex: 1.6 1 210px; }
  .ctl-label { display: block; font-size: 11px; font-weight: 600;
               color: %(MUTED)s; text-transform: uppercase;
               letter-spacing: 0.04em; margin-bottom: 4px; }
  .radio label { margin-right: 12px; font-size: 13px; color: %(INK_2)s; }
  .num-input { width: 70px; border: 1px solid %(BASELINE)s; border-radius: 6px;
               padding: 5px 8px; font: inherit; background: #fff; }
  .text-input { width: 100%%; border: 1px solid %(BASELINE)s; border-radius: 6px;
                padding: 5px 8px; font: inherit; background: #fff; }
  .btn { font: inherit; font-size: 13px; padding: 6px 14px; cursor: pointer;
         background: #2a78d6; color: #fff; border: 0; border-radius: 6px; }
  .btn:hover { background: #1c5cab; }
  .cloud-status { flex-basis: 100%%; font-size: 12.5px; color: %(INK_2)s;
                  margin-top: 2px; }
  .cloud-status.err { color: #d03b3b; }
  .cloud-status.ok { color: #006300; }
  .monitor-strip { display: grid; gap: 10px; margin: 4px 0 14px;
                   grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); }
  .mon {
    border: 1px solid rgba(11,11,11,0.10); border-left: 3px solid %(MUTED)s;
    border-radius: 8px; padding: 9px 12px; background: %(SURFACE)s;
    font: inherit; text-align: left; cursor: pointer; width: 100%%;
    transition: box-shadow .12s, border-color .12s;
  }
  .mon:hover { box-shadow: 0 1px 6px rgba(11,11,11,0.12); }
  .mon:focus-visible { outline: 2px solid #2a78d6; outline-offset: 2px; }
  .mon.active {
    border-color: #2a78d6; box-shadow: 0 0 0 2px rgba(42,120,214,0.28);
  }
  .mon.ok { border-left-color: #006300; }
  .mon.warn { border-left-color: #a8641a; }
  .mon.bad { border-left-color: #d03b3b; }
  .mon .mon-name { font-size: 13px; font-weight: 600; color: %(INK)s;
                   overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .mon .mon-val { font-size: 17px; font-weight: 650; margin-top: 3px; }
  .mon .mon-sub { font-size: 11px; color: %(MUTED)s; margin-top: 1px; }
  .upload { border: 1px dashed %(BASELINE)s; border-radius: 8px;
            padding: 7px 10px; font-size: 12px; color: %(INK_2)s;
            cursor: pointer; text-align: center; }
  .description { color: %(INK_2)s; font-size: 13px; margin: 0 2px 4px; }
  .window-info { color: %(MUTED)s; font-size: 12px; margin: 0 2px 14px; }
  .tiles { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 14px; }
  .tile { flex: 1 1 120px; background: %(SURFACE)s; border-radius: 10px;
          border: 1px solid rgba(11,11,11,0.10); padding: 12px 16px; }
  .tile-value { font-size: 18px; font-weight: 650; color: %(INK)s; }
  .tile-label { font-size: 11px; color: %(MUTED)s; text-transform: uppercase;
                letter-spacing: 0.04em; margin-top: 2px; }
  .card { background: %(SURFACE)s; border: 1px solid rgba(11,11,11,0.10);
          border-radius: 10px; padding: 14px 16px; margin-bottom: 14px; }
  .card h2 { font-size: 15px; margin: 0 0 6px; color: %(INK)s; }
  .hint { font-size: 12px; color: %(MUTED)s; margin: 0 0 10px; }
  .metrics-grid { display: grid; gap: 10px 18px;
                  grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); }
  .metric-value { font-size: 16px; font-weight: 650; color: %(INK)s; }
  .metric-label { font-size: 11px; color: %(MUTED)s; text-transform: uppercase;
                  letter-spacing: 0.04em; margin-top: 1px; }
  .metrics-caveat { grid-column: 1 / -1; font-size: 12px; color: #a8641a;
                    border: 1px solid currentColor; border-radius: 6px;
                    padding: 5px 10px; }
</style>
</head>""" % {"PAGE": PAGE, "SURFACE": SURFACE, "INK": INK, "INK_2": INK_2,
              "MUTED": MUTED, "BASELINE": BASELINE, "FONT": FONT})


if __name__ == "__main__":
    create_app().run(debug=True)
