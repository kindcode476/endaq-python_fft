"""
Interactive Dash UI for the FFT analyser.

Run with ``python -m fft_analyser`` (requires the optional ``dash``
dependency: ``pip install dash``), then open http://127.0.0.1:8050.

The UI is a live correctness harness as much as a viewer: every built-in
test signal carries its ground-truth spectral lines, which are overlaid on
the measured spectrum and scored in the verification table below it.
"""

from __future__ import annotations

import base64
import io

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
    compute_spectrum,
    find_spectral_peaks,
    compare_to_expected,
    parseval_rms_error,
)

# palette roles (validated reference palette, light mode)
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948"]
STATUS_GOOD = "#006300"   # success text step (readable on light surface)
STATUS_CRITICAL = "#d03b3b"
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

MAX_TIME_POINTS = 4096  # display decimation only; analysis uses all samples


def _layout(fig: go.Figure, xtitle: str, ytitle: str) -> go.Figure:
    fig.update_layout(
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT, color=INK_2, size=13),
        margin=dict(l=56, r=16, t=8, b=44),
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


def _control(label: str, component) -> html.Div:
    return html.Div([html.Label(label, className="ctl-label"), component],
                    className="ctl")


def _build_signal(name: str, noise_rms: float) -> signals.TestSignal:
    sig = signals.TEST_SIGNALS[name]()
    if noise_rms > 0:
        rng = np.random.default_rng(20260701)
        for c in sig.data.columns:
            sig.data[c] += rng.normal(0.0, noise_rms, len(sig.data))
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


def create_app() -> Dash:
    app = Dash(__name__, title="FFT Analyser")

    app.layout = html.Div(className="page", children=[
        html.Div(className="header", children=[
            html.H1("FFT Analyser"),
            html.P("Built on endaq.calc.fft - amplitude-corrected spectra with "
                   "ground-truth verification", className="subtitle"),
        ]),

        html.Div(className="controls", children=[
            _control("Source", dcc.RadioItems(
                id="source", value="builtin",
                options=[{"label": " Test signal", "value": "builtin"},
                         {"label": " Uploaded CSV", "value": "upload"}],
                className="radio")),
            _control("Test signal", dcc.Dropdown(
                id="signal-name", clearable=False,
                value=list(signals.TEST_SIGNALS)[0],
                options=[{"label": k, "value": k} for k in signals.TEST_SIGNALS])),
            _control("Added noise (rms)", dcc.Slider(
                id="noise", min=0.0, max=0.5, step=0.05, value=0.0,
                marks={0: "0", 0.25: "0.25", 0.5: "0.5"},
                tooltip={"placement": "bottom"})),
            _control("Window", dcc.Dropdown(
                id="window", value="hann", clearable=False,
                options=[{"label": w, "value": w} for w in WINDOWS])),
            _control("Detrend", dcc.Dropdown(
                id="detrend", value="none", clearable=False,
                options=[{"label": d, "value": d} for d in DETRENDS])),
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
            _control("Upload CSV (time, ch1, ...)", dcc.Upload(
                id="upload", className="upload",
                children=html.Div(id="upload-label", children="Drop or select a file"))),
        ]),

        html.P(id="signal-description", className="description"),

        html.Div(className="tiles", children=[
            _tile("tile-samples", "samples"),
            _tile("tile-fs", "sample rate"),
            _tile("tile-binwidth", "bin width"),
            _tile("tile-parseval", "Parseval RMS error"),
            _tile("tile-verdict", "ground-truth check"),
        ]),

        html.Div(className="card", children=[
            html.H2("Time domain"),
            dcc.Graph(id="time-plot", config={"displayModeBar": False}),
        ]),
        html.Div(className="card", children=[
            html.H2("Amplitude spectrum"),
            dcc.Graph(id="spectrum-plot", config={"displayModeBar": False}),
        ]),
        html.Div(className="card", children=[
            html.H2("Peak verification"),
            html.P("Each expected spectral line vs. the nearest detected peak. "
                   "Pass = frequency within 1 Hz and amplitude within 5 %.",
                   className="hint"),
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

    @app.callback(
        Output("time-plot", "figure"),
        Output("spectrum-plot", "figure"),
        Output("peaks-table", "data"),
        Output("peaks-table", "columns"),
        Output("signal-description", "children"),
        Output("tile-samples", "children"),
        Output("tile-fs", "children"),
        Output("tile-binwidth", "children"),
        Output("tile-parseval", "children"),
        Output("tile-verdict", "children"),
        Output("tile-verdict", "style"),
        Input("source", "value"),
        Input("signal-name", "value"),
        Input("noise", "value"),
        Input("window", "value"),
        Input("detrend", "value"),
        Input("pad", "value"),
        Input("yscale", "value"),
        Input("xscale", "value"),
        Input("npeaks", "value"),
        Input("uploaded-data", "data"),
    )
    def update(source, signal_name, noise, window, detrend, pad,
               yscale, xscale, npeaks, uploaded):
        expected_peaks, expected_band = [], None
        if source == "upload" and uploaded:
            df = pd.DataFrame(uploaded["records"]).set_index(uploaded["index"])
            description = f"Uploaded file: {uploaded['name']}"
        else:
            sig = _build_signal(signal_name, float(noise or 0.0))
            df = sig.data
            expected_peaks, expected_band = sig.expected_peaks, sig.expected_band
            description = sig.description

        n = len(df)
        dt = float(np.mean(np.diff(df.index.to_numpy(dtype=float))))
        fs = 1.0 / dt

        spectrum = compute_spectrum(df, window=window, detrend=detrend,
                                    pad_factor=int(pad))
        peaks = find_spectral_peaks(spectrum, n_peaks=int(npeaks or 10))
        parseval = parseval_rms_error(df)

        time_fig = _time_figure(df)
        spec_fig = _spectrum_figure(spectrum, peaks, expected_peaks,
                                    expected_band, yscale, xscale)
        rows, cols = _table(peaks, expected_peaks, detrend)
        verdict, verdict_style = _verdict(rows, expected_peaks)

        return (time_fig, spec_fig, rows, cols, description,
                f"{n:,}", f"{fs:,.0f} Hz", f"{fs / (n * int(pad)):.3g} Hz",
                f"{parseval:.2e}", verdict, verdict_style)

    return app


def _time_figure(df: pd.DataFrame) -> go.Figure:
    step = max(1, len(df) // MAX_TIME_POINTS)
    view = df.iloc[::step]
    fig = go.Figure()
    for i, c in enumerate(df.columns):
        fig.add_trace(go.Scatter(
            x=view.index, y=view[c], name=str(c), mode="lines",
            line=dict(color=SERIES[i % len(SERIES)], width=2),
            hovertemplate="%{y:.4g}<extra>" + str(c) + "</extra>"))
    fig = _layout(fig, "time (s)", "amplitude")
    fig.update_layout(showlegend=len(df.columns) > 1, height=280)
    return fig


def _spectrum_figure(spectrum, peaks, expected_peaks, expected_band,
                     yscale, xscale) -> go.Figure:
    db = yscale == "db"
    floor = spectrum.to_numpy().max() * 1e-8 + 1e-30

    def y_of(values):
        values = np.asarray(values, dtype=float)
        return 20 * np.log10(np.maximum(values, floor)) if db else values

    fig = go.Figure()

    if expected_band is not None:
        fig.add_vrect(x0=max(expected_band[0], spectrum.index[1] if xscale == "log" else expected_band[0]),
                      x1=expected_band[1], fillcolor=GRID, opacity=0.45,
                      line_width=0, layer="below",
                      annotation_text="expected band",
                      annotation_position="inside bottom left",
                      annotation_font=dict(color=MUTED, size=11))

    for i, c in enumerate(spectrum.columns):
        fig.add_trace(go.Scatter(
            x=spectrum.index, y=y_of(spectrum[c]), name=str(c), mode="lines",
            line=dict(color=SERIES[i % len(SERIES)], width=2),
            hovertemplate="%{y:.4g}<extra>" + str(c) + "</extra>"))

    if len(peaks):
        fig.add_trace(go.Scatter(
            x=peaks["frequency (Hz)"], y=y_of(peaks["amplitude"]),
            name="detected peaks", mode="markers",
            marker=dict(color=SERIES[0], size=9, symbol="circle",
                        line=dict(color=SURFACE, width=2)),
            hovertemplate="%{x:.2f} Hz, %{y:.4g}<extra>detected</extra>"))

    if expected_peaks:
        fx = [f for f, _ in expected_peaks]
        fa = [a for _, a in expected_peaks]
        fig.add_trace(go.Scatter(
            x=fx, y=y_of(fa), name="expected (ground truth)", mode="markers",
            marker=dict(color=INK_2, size=10, symbol="x-thin",
                        line=dict(color=INK_2, width=2)),
            hovertemplate="%{x:.2f} Hz, %{y:.4g}<extra>expected</extra>"))

    fig = _layout(fig, "frequency (Hz)", "amplitude (dB)" if db else "amplitude")
    fig.update_layout(height=360)
    if xscale == "log":
        fig.update_xaxes(type="log")
    return fig


def _table(peaks, expected_peaks, detrend):
    if not expected_peaks:
        if not len(peaks):
            return [], [{"name": "no peaks detected", "id": "result"}]
        rows = [{"measured freq (Hz)": f"{r['frequency (Hz)']:.3f}",
                 "measured amplitude": f"{r['amplitude']:.4g}",
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
              border-radius: 10px; padding: 14px 16px; margin-bottom: 14px; }
  .ctl { min-width: 150px; flex: 1 1 150px; }
  .ctl-label { display: block; font-size: 11px; font-weight: 600;
               color: %(MUTED)s; text-transform: uppercase;
               letter-spacing: 0.04em; margin-bottom: 4px; }
  .radio label { margin-right: 12px; font-size: 13px; color: %(INK_2)s; }
  .num-input { width: 70px; border: 1px solid %(BASELINE)s; border-radius: 6px;
               padding: 5px 8px; font: inherit; background: #fff; }
  .upload { border: 1px dashed %(BASELINE)s; border-radius: 8px;
            padding: 7px 10px; font-size: 12px; color: %(INK_2)s;
            cursor: pointer; text-align: center; }
  .description { color: %(INK_2)s; font-size: 13px; margin: 0 2px 14px; }
  .tiles { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 14px; }
  .tile { flex: 1 1 140px; background: %(SURFACE)s; border-radius: 10px;
          border: 1px solid rgba(11,11,11,0.10); padding: 12px 16px; }
  .tile-value { font-size: 20px; font-weight: 650; color: %(INK)s; }
  .tile-label { font-size: 11px; color: %(MUTED)s; text-transform: uppercase;
                letter-spacing: 0.04em; margin-top: 2px; }
  .card { background: %(SURFACE)s; border: 1px solid rgba(11,11,11,0.10);
          border-radius: 10px; padding: 14px 16px; margin-bottom: 14px; }
  .card h2 { font-size: 15px; margin: 0 0 6px; color: %(INK)s; }
  .hint { font-size: 12px; color: %(MUTED)s; margin: 0 0 10px; }
</style>
</head>""" % {"PAGE": PAGE, "SURFACE": SURFACE, "INK": INK, "INK_2": INK_2,
              "MUTED": MUTED, "BASELINE": BASELINE, "FONT": FONT})


if __name__ == "__main__":
    create_app().run(debug=True)
