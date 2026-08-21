"""
Three-way validation: analytical theory vs this module vs SciPy.

``python -m fft_analyser.validation`` regenerates the table in
``VALIDATION.md``.  Every row is a signal whose answer is known in closed
form, measured independently by this module's engine and by SciPy's
spectral estimators (which this module does not call for its spectra).
``tests/fft_analyser/test_validation_table.py`` asserts every row's error
bound, so the published table can never drift from the code.
"""

from __future__ import annotations

import typing

import numpy as np
import pandas as pd
import scipy.signal

from .analysis import (
    analyze,
    band_rms,
    velocity_band_rms,
    window_figures,
    parseval_rms_error,
)
from .metrics import tone_metrics

FS = 2048.0
N = 16384


def _df(x: np.ndarray, fs: float = FS) -> pd.DataFrame:
    t = np.arange(len(x)) / fs
    return pd.DataFrame({"signal": x}, index=pd.Index(t, name="time (s)"))


def _tone(freq: float = 100.0, amp: float = 1.0) -> pd.DataFrame:
    t = np.arange(N) / FS
    return _df(amp * np.sin(2 * np.pi * freq * t))


def _scipy_welch(x, scaling, nperseg=4096):
    return scipy.signal.welch(x, FS, window="hann", nperseg=nperseg,
                              noverlap=nperseg // 2, detrend=False,
                              scaling=scaling)


def _scipy_enbw_hz(nperseg=4096):
    w = scipy.signal.get_window("hann", nperseg, fftbins=True)
    return FS * (w ** 2).sum() / w.sum() ** 2


class Row(typing.NamedTuple):
    test: str
    theory: float
    ours: float
    reference: typing.Optional[float]  #: None = no direct SciPy analogue
    tolerance: float  #: max acceptable relative error for this row

    @property
    def max_error(self) -> float:
        # absolute error when the theoretical value is zero (Parseval row)
        errs = [abs(self.ours - self.theory) / abs(self.theory)
                if self.theory else abs(self.ours - self.theory)]
        if self.reference is not None:
            errs.append(abs(self.ours - self.reference) / abs(self.reference))
        return max(errs)


def validation_rows() -> typing.List[Row]:
    rows: typing.List[Row] = []

    # window figure of merit: periodic hann NENBW is exactly 3/2
    f = window_figures("hann", 4096)
    rows.append(Row("hann NENBW (bins)", 1.5, f["nenbw_bins"],
                    _scipy_enbw_hz() * 4096 / FS, 1e-9))

    # a 1 m/s² peak tone at 100 Hz, bin-centered, hann window
    tone = _tone()
    x = tone["signal"].to_numpy()
    res = analyze(tone, quantity="power", window="hann", nperseg=4096,
                  overlap=0.5, averaging="linear", detrend="none")
    _, p_spec = _scipy_welch(x, "spectrum")
    _, p_dens = _scipy_welch(x, "density")
    k = int(np.argmax(p_spec))

    rows.append(Row("tone amplitude, peak (m/s²)", 1.0,
                    res.as_quantity("amplitude_peak")["signal"].max(),
                    float(np.sqrt(2 * p_spec[k])), 1e-6))
    rows.append(Row("tone amplitude, RMS (m/s²)", 1.0 / np.sqrt(2),
                    res.as_quantity("amplitude_rms")["signal"].max(),
                    float(np.sqrt(p_spec[k])), 1e-6))

    v_theory = 1000.0 / (np.sqrt(2) * 2 * np.pi * 100.0)
    freqs = res.ps.index.to_numpy(dtype=float)
    rows.append(Row("tone velocity @ 100 Hz (mm/s RMS)", v_theory,
                    res.as_quantity("velocity_rms")["signal"].max(),
                    float(np.sqrt(p_dens[k] * _scipy_enbw_hz()) * 1000.0
                          / (2 * np.pi * freqs[np.argmax(p_dens)])), 1e-6))

    rows.append(Row("tone band RMS 10-1000 Hz (m/s² RMS)", 1.0 / np.sqrt(2),
                    band_rms(res, 10.0, 1000.0),
                    _band_ref(p_dens, 10.0, 1000.0), 1e-4))
    rows.append(Row("tone Vel RMS 10-1000 Hz (mm/s)", v_theory,
                    velocity_band_rms(res, 10.0, 1000.0),
                    _vel_band_ref(p_dens, 10.0, 1000.0), 1e-4))

    # white noise: single-sided PSD floor is 2 sigma^2 / fs
    rng = np.random.default_rng(11)
    sigma = 0.5
    noise = _df(sigma * rng.standard_normal(N))
    res_n = analyze(noise, quantity="psd", window="hann", nperseg=1024,
                    overlap=0.5, averaging="linear", detrend="none")
    _, p_noise = _scipy_welch(noise["signal"].to_numpy(), "density",
                              nperseg=1024)
    band = ((res_n.spectrum.index > 200) & (res_n.spectrum.index < 800))
    rows.append(Row("white-noise PSD floor (u²/Hz)", 2 * sigma ** 2 / FS,
                    float(res_n.spectrum["signal"][band].median()),
                    float(np.median(p_noise[band])), 5e-2))

    # Parseval: integral of the rectangular-window PSD vs mean square
    rows.append(Row("Parseval: rel. err. of ∫PSD df vs mean-square", 0.0,
                    parseval_rms_error(tone), None, 1e-9))

    # THD of a 2% + 1% harmonic mix: sqrt(0.02² + 0.01²) = 2.2360680 %
    t = np.arange(N) / FS
    dist = _df(np.sin(2 * np.pi * 100 * t)
               + 0.02 * np.sin(2 * np.pi * 200 * t)
               + 0.01 * np.sin(2 * np.pi * 300 * t))
    rows.append(Row("THD of 2%+1% harmonic mix (%)", 100 * np.sqrt(5) / 100,
                    tone_metrics(dist).thd_pct, None, 1e-4))

    return rows


def _band_ref(p_dens, lo, hi):
    f = np.fft.rfftfreq(4096, 1 / FS)
    mask = (f >= lo) & (f <= hi)
    return float(np.sqrt(p_dens[mask].sum() * (f[1] - f[0])))


def _vel_band_ref(p_dens, lo, hi):
    f = np.fft.rfftfreq(4096, 1 / FS)
    mask = (f >= lo) & (f <= hi)
    pv = p_dens[mask] * (1000.0 / (2 * np.pi * f[mask])) ** 2
    return float(np.sqrt(pv.sum() * (f[1] - f[0])))


def _fmt(v: typing.Optional[float]) -> str:
    if v is None:
        return "—"
    if v == 0:
        return "0"
    return f"{v:.7g}"


def render_markdown() -> str:
    lines = [
        "# Validation: theory vs this module vs SciPy",
        "",
        "Generated by `python -m fft_analyser.validation`; every row's",
        "error bound is enforced by",
        "`tests/fft_analyser/test_validation_table.py`, so this table",
        "cannot drift from the code. \"SciPy\" columns come from",
        "`scipy.signal.welch`/`periodogram` run independently on the same",
        "samples — this module computes its spectra from raw FFTs and",
        "never calls them.",
        "",
        "Signals: 1 m/s² peak tone at 100 Hz (bin-centered), hann window,",
        f"4096-sample segments at 50 % overlap, fs = {FS:g} Hz;",
        "white noise σ = 0.5; harmonic mix per the row label.",
        "",
        "| Test | Theory | This module | SciPy | Max rel. error |",
        "|---|---|---|---|---|",
    ]
    for r in validation_rows():
        err = r.max_error if r.theory != 0 else abs(r.ours)
        lines.append(f"| {r.test} | {_fmt(r.theory)} | {_fmt(r.ours)} | "
                     f"{_fmt(r.reference)} | {err:.1e} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    import pathlib
    out = pathlib.Path(__file__).parent / "VALIDATION.md"
    out.write_text(render_markdown())
    print(render_markdown())
    print(f"written to {out}")


if __name__ == "__main__":
    main()
