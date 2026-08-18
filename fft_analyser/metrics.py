"""
Single-tone signal-quality metrics, following audio-analyzer / ADC-test
conventions (Audio Precision; IEEE Std 1241/1057).

All ratios are computed from *component powers* integrated over the analysis
window's main lobe in the PSD - the same approach professional analysers
use, so a component's total (leaked) power is captured regardless of where
it falls between bins:

- **THD** - ratio of the RSS of harmonic amplitudes (2nd..Nth) to the
  fundamental amplitude.
- **THD+N** - ratio of everything-but-the-fundamental (excluding DC) to the
  fundamental; the reciprocal of SINAD.
- **SNR** - fundamental vs. noise only (harmonics excluded).
- **SINAD** - fundamental vs. noise + distortion, in dB.
- **ENOB** - effective number of bits, ``(SINAD_dB - 1.76) / 6.02``
  (IEEE 1241, full-scale sine convention).
- **SFDR** - fundamental vs. the strongest spur anywhere in the spectrum
  (harmonic or not), in dBc.

A high-dynamic-range window (blackman-harris by default) keeps window
sidelobes from masquerading as distortion.
"""

from __future__ import annotations

from dataclasses import dataclass
import typing

import numpy as np
import pandas as pd

from .analysis import WINDOWS, analyze

__all__ = ["ToneMetrics", "tone_metrics"]


@dataclass
class ToneMetrics:
    fundamental_freq: float  #: Hz
    fundamental_peak: float  #: peak amplitude of the fundamental
    fundamental_rms: float
    thd_pct: float
    thd_db: float  #: 20*log10(THD)  (negative for low distortion)
    thd_n_pct: float
    thd_n_db: float
    snr_db: float
    sinad_db: float
    enob_bits: float
    sfdr_dbc: float
    rms: float  #: time-domain RMS
    peak: float  #: time-domain |max|
    crest_factor: float
    n_harmonics: int  #: harmonics included in THD


def _component_power(psd: np.ndarray, df_hz: float, k: int, lobe: int) -> float:
    lo, hi = max(0, k - lobe), min(len(psd), k + lobe + 1)
    return float(psd[lo:hi].sum() * df_hz)


def tone_metrics(
        df: pd.DataFrame,
        column: typing.Optional[str] = None,
        window: str = "blackman-harris",
        max_harmonics: int = 10,
) -> ToneMetrics:
    """
    Measure THD, THD+N, SNR, SINAD, ENOB and SFDR of the dominant tone.

    :param df: time-indexed input data
    :param column: channel to analyse (defaults to the first)
    :param window: analysis window; needs high sidelobe suppression so
        window leakage isn't counted as distortion
    :param max_harmonics: highest harmonic order included in THD
    """
    col = column if column is not None else df.columns[0]
    x = df[col].to_numpy(dtype=float)

    result = analyze(df[[col]], quantity="psd", window=window,
                     detrend="mean", averaging="none")
    psd = result.spectrum[col].to_numpy(dtype=float)
    freqs = result.spectrum.index.to_numpy(dtype=float)
    dfreq = result.bin_width
    lobe = WINDOWS[window].lobe_half_bins

    # fundamental: strongest bin outside the DC lobe
    search = psd.copy()
    search[:lobe + 1] = 0.0
    k1 = int(np.argmax(search))

    # refine the fundamental frequency by parabolic interpolation (dB domain)
    f1 = freqs[k1]
    if 0 < k1 < len(psd) - 1 and psd[k1 - 1] > 0 and psd[k1 + 1] > 0:
        y0, y1v, y2 = 10 * np.log10(psd[k1 - 1:k1 + 2])
        denom = y0 - 2 * y1v + y2
        if denom < 0:
            f1 = freqs[k1] + 0.5 * (y0 - y2) / denom * dfreq

    total_power = float(psd[lobe + 1:].sum() * dfreq)  # everything but DC
    p_fund = _component_power(psd, dfreq, k1, lobe)

    # harmonic powers: h*f1 for h = 2.., while inside the spectrum and not
    # overlapping the fundamental's own lobe
    p_harm = 0.0
    n_harm = 0
    harmonic_bins = []
    for h in range(2, max_harmonics + 1):
        kh = int(round(h * f1 / dfreq))
        if kh + lobe >= len(psd):
            break
        if abs(kh - k1) <= 2 * lobe:
            continue
        p_harm += _component_power(psd, dfreq, kh, lobe)
        harmonic_bins.append(kh)
        n_harm += 1

    noise_dist = max(total_power - p_fund, 0.0)
    noise_only = max(noise_dist - p_harm, 0.0)

    def ratio_db(num: float, den: float) -> float:
        if den <= 0:
            return np.inf
        if num <= 0:
            return -np.inf
        return 10 * np.log10(num / den)

    thd = np.sqrt(p_harm / p_fund) if p_fund > 0 else np.nan
    thd_n = np.sqrt(noise_dist / p_fund) if p_fund > 0 else np.nan
    sinad_db = ratio_db(p_fund, noise_dist)

    # SFDR: strongest single component (integrated over one lobe) that is
    # neither DC nor the fundamental
    spur_mask = np.ones(len(psd), bool)
    spur_mask[:lobe + 1] = False
    spur_mask[max(0, k1 - lobe):k1 + lobe + 1] = False
    k_spur = int(np.argmax(np.where(spur_mask, psd, 0.0)))
    p_spur = _component_power(psd, dfreq, k_spur, lobe)

    rms = float(np.sqrt(np.mean(x ** 2)))
    peak = float(np.max(np.abs(x)))

    return ToneMetrics(
        fundamental_freq=float(f1),
        fundamental_peak=float(np.sqrt(2 * p_fund)),
        fundamental_rms=float(np.sqrt(p_fund)),
        thd_pct=float(thd * 100),
        thd_db=float(20 * np.log10(thd)) if thd > 0 else -np.inf,
        thd_n_pct=float(thd_n * 100),
        thd_n_db=float(20 * np.log10(thd_n)) if thd_n > 0 else -np.inf,
        snr_db=float(ratio_db(p_fund, noise_only)),
        sinad_db=float(sinad_db),
        enob_bits=float((sinad_db - 1.76) / 6.02),
        sfdr_dbc=float(ratio_db(p_fund, p_spur)),
        rms=rms,
        peak=peak,
        crest_factor=float(peak / rms) if rms > 0 else np.nan,
        n_harmonics=n_harm,
    )
