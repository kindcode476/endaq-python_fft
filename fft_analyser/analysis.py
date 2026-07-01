"""
The FFT analysis pipeline, built on :py:func:`endaq.calc.fft.rfft`.

Beyond the raw library call this adds the corrections a measurement-grade
analyser needs:

- **Window amplitude correction** - the spectrum is divided by the window's
  coherent gain (``mean(w)``), so a sinusoid of amplitude A peaks at A no
  matter which window is selected.
- **DC / Nyquist bin correction** - the "unit" normalization in
  :py:mod:`endaq.calc.fft` doubles *every* bin, but the 0 Hz bin (and the
  Nyquist bin, for even FFT lengths) has no negative-frequency twin and must
  not be doubled.  Without this a DC offset of 2 reads as 4.
- **Zero-padding amplitude correction** - padding to ``nfft > len(df)``
  interpolates the spectrum, but "unit" scaling divides by the padded length,
  attenuating every amplitude by ``len(df)/nfft``.  The correction restores
  true amplitudes at any padding factor.
- **Sub-bin peak estimation** - detected peaks are refined by parabolic
  interpolation on the dB magnitudes of the three bins around each maximum,
  giving frequency estimates well below the bin spacing.
"""

from __future__ import annotations

import typing

import numpy as np
import pandas as pd
import scipy.signal

from endaq.calc import fft as endaq_fft

__all__ = [
    "WINDOWS",
    "DETRENDS",
    "compute_spectrum",
    "find_spectral_peaks",
    "compare_to_expected",
    "parseval_rms_error",
]

#: window name -> scipy.signal.get_window identifier
WINDOWS: typing.Dict[str, str] = {
    "rectangular": "boxcar",
    "hann": "hann",
    "hamming": "hamming",
    "blackman": "blackman",
    "flattop": "flattop",
}

DETRENDS = ("none", "mean", "linear")


def _detrend(values: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return values
    if mode == "mean":
        return values - values.mean()
    if mode == "linear":
        return scipy.signal.detrend(values, type="linear")
    raise ValueError(f"detrend must be one of {DETRENDS}, was {mode!r}")


def compute_spectrum(
        df: pd.DataFrame,
        window: str = "hann",
        detrend: str = "none",
        pad_factor: int = 1,
) -> pd.DataFrame:
    """
    Compute an amplitude-corrected, single-sided magnitude spectrum.

    :param df: time-indexed input data (seconds), one column per channel
    :param window: one of :py:data:`WINDOWS`; amplitude-corrected so sinusoid
        peaks keep their time-domain amplitude
    :param detrend: "none", "mean" (subtract mean) or "linear" (remove
        least-squares line) before windowing
    :param pad_factor: zero-pad the FFT to ``pad_factor * len(df)`` for a
        finer frequency grid (amplitudes stay correct)
    :return: DataFrame of peak-equivalent amplitudes indexed by frequency (Hz)
    """
    if window not in WINDOWS:
        raise ValueError(f"window must be one of {sorted(WINDOWS)}, was {window!r}")
    if not isinstance(pad_factor, int) or pad_factor < 1:
        raise ValueError(f"pad_factor must be a positive integer, was {pad_factor!r}")

    n = len(df)
    w = scipy.signal.get_window(WINDOWS[window], n)
    conditioned = pd.DataFrame(
        {c: _detrend(df[c].to_numpy(dtype=float), detrend) * w for c in df.columns},
        index=df.index,
    )

    nfft = n * pad_factor
    spectrum = endaq_fft.rfft(conditioned, norm="unit", nfft=nfft, optimize=False)

    # unit norm applies 2/nfft: undo the window's coherent gain (mean(w) =
    # sum(w)/n) and the padded-length attenuation in one factor
    spectrum *= nfft / (n * w.mean())

    # bins without a negative-frequency twin must not carry the factor of 2
    spectrum.iloc[0] /= 2.0
    if nfft % 2 == 0:
        spectrum.iloc[-1] /= 2.0

    return spectrum


def find_spectral_peaks(
        spectrum: typing.Union[pd.DataFrame, pd.Series],
        column: typing.Optional[str] = None,
        n_peaks: int = 10,
        min_rel_prominence_db: float = -40.0,
        interpolate: bool = True,
) -> pd.DataFrame:
    """
    Locate spectral peaks, optionally refining them below bin resolution.

    :param spectrum: output of :py:func:`compute_spectrum`
    :param column: which channel to analyse (defaults to the first)
    :param n_peaks: keep at most this many peaks, strongest first
    :param min_rel_prominence_db: reject peaks whose prominence is more than
        this many dB below the spectrum maximum (screens out window sidelobes)
    :param interpolate: parabolic interpolation on dB magnitudes for sub-bin
        frequency/amplitude estimates
    :return: DataFrame with ``frequency (Hz)`` and ``amplitude`` columns,
        sorted by descending amplitude
    """
    if isinstance(spectrum, pd.DataFrame):
        series = spectrum[column if column is not None else spectrum.columns[0]]
    else:
        series = spectrum

    freqs = series.index.to_numpy(dtype=float)
    mag = series.to_numpy(dtype=float)

    peak_mag = mag.max()
    if peak_mag <= 0:
        return pd.DataFrame(columns=["frequency (Hz)", "amplitude"])

    prominence = peak_mag * 10.0 ** (min_rel_prominence_db / 20.0)
    idx, _ = scipy.signal.find_peaks(mag, prominence=prominence)

    # the DC bin can't be found by find_peaks (no left neighbour): include it
    # when it stands above the same threshold relative to its neighbourhood
    if mag[0] > mag[1] and mag[0] - mag[1] >= prominence:
        idx = np.concatenate(([0], idx))

    results = []
    floor = peak_mag * 1e-12  # keep log() finite on zero bins
    for i in idx:
        f_est, a_est = freqs[i], mag[i]
        if interpolate and 0 < i < len(mag) - 1 and mag[i - 1] > floor and mag[i + 1] > floor:
            y0, y1, y2 = 20 * np.log10([mag[i - 1], mag[i], mag[i + 1]])
            denom = y0 - 2 * y1 + y2
            if denom < 0:  # true local max in dB space
                delta = 0.5 * (y0 - y2) / denom
                f_est = freqs[i] + delta * (freqs[1] - freqs[0])
                a_est = 10.0 ** ((y1 - 0.25 * (y0 - y2) * delta) / 20.0)
        results.append((f_est, a_est))

    out = pd.DataFrame(results, columns=["frequency (Hz)", "amplitude"])
    return out.sort_values("amplitude", ascending=False).head(n_peaks).reset_index(drop=True)


def compare_to_expected(
        peaks: pd.DataFrame,
        expected_peaks: typing.Sequence[typing.Tuple[float, float]],
        freq_tolerance: float = 1.0,
        amp_tolerance_pct: float = 5.0,
) -> pd.DataFrame:
    """
    Match detected peaks against ground truth - the core correctness check.

    :param peaks: output of :py:func:`find_spectral_peaks`
    :param expected_peaks: (frequency, amplitude) ground-truth pairs
    :param freq_tolerance: max |frequency error| in Hz to count as a match
    :param amp_tolerance_pct: max |amplitude error| in percent to pass
    :return: one row per expected peak: expected/measured values, errors, and
        a boolean ``pass`` column
    """
    rows = []
    for f_exp, a_exp in expected_peaks:
        if len(peaks):
            nearest = (peaks["frequency (Hz)"] - f_exp).abs().idxmin()
            f_meas = peaks.loc[nearest, "frequency (Hz)"]
            a_meas = peaks.loc[nearest, "amplitude"]
        else:
            f_meas = a_meas = np.nan
        f_err = f_meas - f_exp
        a_err_pct = (a_meas - a_exp) / a_exp * 100.0 if a_exp else np.nan
        rows.append({
            "expected freq (Hz)": f_exp,
            "measured freq (Hz)": f_meas,
            "freq error (Hz)": f_err,
            "expected amplitude": a_exp,
            "measured amplitude": a_meas,
            "amplitude error (%)": a_err_pct,
            "pass": bool(
                np.isfinite(f_meas)
                and abs(f_err) <= freq_tolerance
                and abs(a_err_pct) <= amp_tolerance_pct
            ),
        })
    return pd.DataFrame(rows)


def parseval_rms_error(df: pd.DataFrame, column: typing.Optional[str] = None) -> float:
    """
    Energy-conservation check (Parseval's theorem).

    Computes the signal RMS both in the time domain and from a rectangular-
    window spectrum, and returns the relative error between them.  For a
    correct FFT implementation this is ~1e-15; anything above ~1e-6 indicates
    a scaling bug somewhere in the pipeline.
    """
    col = column if column is not None else df.columns[0]
    x = df[col].to_numpy(dtype=float)

    spectrum = compute_spectrum(df[[col]], window="rectangular", detrend="none")
    m = spectrum[col].to_numpy(dtype=float)

    # single-sided peak amplitudes -> mean square: DC and (even-n) Nyquist
    # bins contribute m^2, doubled bins contribute m^2/2
    interior = slice(1, -1) if len(x) % 2 == 0 else slice(1, None)
    mean_square = m[0] ** 2 + np.sum(m[interior] ** 2) / 2.0
    if len(x) % 2 == 0:
        mean_square += m[-1] ** 2

    rms_time = np.sqrt(np.mean(x ** 2))
    rms_freq = np.sqrt(mean_square)
    if rms_time == 0:
        return 0.0 if rms_freq == 0 else np.inf
    return abs(rms_freq - rms_time) / rms_time
