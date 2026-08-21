"""
The FFT analysis engine, following professional dynamic-signal-analyser
conventions.

Scaling conventions (Heinzel/Ruediger/Schilling 2002; identical to
:py:func:`scipy.signal.welch`'s ``spectrum``/``density`` scalings):

With window ``w`` of length N, ``S1 = sum(w)`` and ``S2 = sum(w**2)``, and
``X[k]`` the raw DFT of the windowed segment, the canonical single-sided
**power spectrum** (units: input-units² RMS) is::

    PS[k] = c_k * |X[k]|**2 / S1**2        c_k = 2, except 1 at DC & Nyquist

and the **power spectral density** (units²/Hz) is::

    PSD[k] = c_k * |X[k]|**2 / (fs * S2) = PS[k] / ENBW

where ``ENBW = fs * S2 / S1**2`` is the window's equivalent noise bandwidth
in Hz (``NENBW = N * S2 / S1**2`` in bins).  Derived quantities:
RMS amplitude = sqrt(PS), peak amplitude = sqrt(2*PS) (sqrt(PS) at
DC/Nyquist), ASD = sqrt(PSD).

For machine-vibration work (ISO 20816 convention) the spectrum is read as
**velocity in mm/s RMS**, obtained from an acceleration signal by
frequency-domain integration: a sinusoid ``a(t) = A*sin(2*pi*f*t)`` in m/s²
has velocity amplitude ``A/(2*pi*f)`` m/s, so::

    V_rms[k] = sqrt(PS[k]) / (2*pi*f[k]) * 1000       [mm/s RMS]

with bins below :py:data:`VELOCITY_CUTOFF_HZ` zeroed - the 1/f weighting
diverges toward DC, and ~2 Hz is the conventional velocity high-pass.  This
quantity is only meaningful when the input is acceleration in m/s².

The professional reading rules these scalings encode:

- **Sinusoids** are read from amplitude/power spectra - their peak height is
  independent of the FFT length and window (after coherent-gain correction).
- **Broadband noise** is read from PSD/ASD - the density level is
  independent of the FFT length and window (via the S2/ENBW normalization),
  while its height in an amplitude spectrum is meaningless (it grows with
  bin width).

Averaging follows Welch's method: the data is split into overlapping
segments and the *power* spectra are averaged (RMS averaging - reduces the
variance of the estimate by ~1/sqrt(M) without lowering the noise floor) or
max-held (peak hold).  50% overlap is the standard default for hann;
wide-main-lobe windows profit from more (Heinzel 2002).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import typing

import numpy as np
import pandas as pd
import scipy.fft
import scipy.signal

__all__ = [
    "WINDOWS",
    "DETRENDS",
    "QUANTITIES",
    "VELOCITY_CUTOFF_HZ",
    "WindowInfo",
    "window_figures",
    "SpectrumResult",
    "analyze",
    "spectrogram",
    "band_rms",
    "velocity_band_rms",
    "SignalQuality",
    "signal_quality",
    "compute_spectrum",
    "find_spectral_peaks",
    "compare_to_expected",
    "parseval_rms_error",
]


@dataclass(frozen=True)
class WindowInfo:
    """Static properties of an analysis window.

    ``sidelobe_db`` and ``recommended_overlap`` are literature values
    (Harris 1978; Heinzel 2002 Table 2 - hann/hamming/blackman-harris; the
    flat-top family per D'Antona & Ferrero 2006); the exact ENBW and
    scalloping loss are *computed* from the coefficients by
    :py:func:`window_figures`, which the test suite checks against the same
    literature.  ``lobe_half_bins`` is the half-width of the main lobe in
    bins, used to integrate a spectral component's total (leaked) power.
    """
    key: str
    scipy_name: typing.Union[str, tuple]
    sidelobe_db: float
    recommended_overlap: float
    lobe_half_bins: int
    use_for: str


#: Analysis windows offered, with their published figures of merit.
WINDOWS: typing.Dict[str, WindowInfo] = {info.key: info for info in [
    WindowInfo("rectangular", "boxcar", -13.3, 0.0, 1,
               "bin-exact tones, transients fully inside the record"),
    WindowInfo("hann", "hann", -31.5, 0.5, 2,
               "general purpose (the standard default)"),
    WindowInfo("hamming", "hamming", -42.7, 0.5, 2,
               "closely spaced tones of similar level"),
    WindowInfo("blackman", "blackman", -58.1, 0.62, 3,
               "moderate dynamic range"),
    WindowInfo("blackman-harris", "blackmanharris", -92.0, 0.661, 4,
               "high dynamic range / distortion analysis"),
    WindowInfo("flattop", "flattop", -93.0, 0.76, 5,
               "amplitude calibration (<0.01 dB scalloping loss)"),
    WindowInfo("kaiser (β=14)", ("kaiser", 14.0), -105.9, 0.7, 4,
               "very high dynamic range"),
]}

DETRENDS = ("none", "mean", "linear")

#: quantity key -> (label, unit template, is_power_quantity)
#: ``velocity_rms`` assumes acceleration input in m/s², so its unit is
#: absolute (mm/s) rather than a ``{u}`` template.
QUANTITIES: typing.Dict[str, typing.Tuple[str, str, bool]] = {
    "velocity_rms": ("velocity (RMS)", "mm/s RMS", False),
    "amplitude_peak": ("amplitude (peak)", "{u}", False),
    "amplitude_rms": ("amplitude (RMS)", "{u} RMS", False),
    "power": ("power spectrum", "{u}² RMS", True),
    "psd": ("power spectral density", "{u}²/Hz", True),
    "asd": ("amplitude spectral density", "{u}/√Hz", False),
}

#: bins below this frequency are zeroed in the velocity spectrum - the
#: 1/(2*pi*f) integration diverges toward DC, and ~2 Hz is the standard
#: high-pass cutoff for machine-vibration velocity readings (ISO 20816).
VELOCITY_CUTOFF_HZ = 2.0


def get_window_samples(window: str, n: int) -> np.ndarray:
    if window not in WINDOWS:
        raise ValueError(f"window must be one of {sorted(WINDOWS)}, was {window!r}")
    return scipy.signal.get_window(WINDOWS[window].scipy_name, n, fftbins=True)


def window_figures(window: str, n: int) -> typing.Dict[str, float]:
    """
    Compute a window's exact figures of merit from its coefficients:
    ``s1``, ``s2``, ``nenbw_bins`` (= N*S2/S1**2), ``coherent_gain``
    (= S1/N) and ``scalloping_db`` (worst-case level error for a tone
    half-way between bins).
    """
    w = get_window_samples(window, n)
    s1, s2 = float(w.sum()), float((w * w).sum())
    nenbw = n * s2 / s1 ** 2
    half_bin = np.abs(np.sum(w * np.exp(-1j * np.pi * np.arange(n) / n))) / s1
    return {
        "s1": s1,
        "s2": s2,
        "nenbw_bins": nenbw,
        "coherent_gain": s1 / n,
        "scalloping_db": float(20 * np.log10(half_bin)),
    }


def _detrend(values: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return values
    if mode == "mean":
        return values - values.mean()
    if mode == "linear":
        return scipy.signal.detrend(values, type="linear")
    raise ValueError(f"detrend must be one of {DETRENDS}, was {mode!r}")


@dataclass
class SpectrumResult:
    """A computed spectrum plus the measurement metadata an analyser reports."""
    spectrum: pd.DataFrame  #: the spectrum in the requested quantity
    ps: pd.DataFrame  #: the canonical power spectrum (units² RMS)
    quantity: str
    window: str
    fs: float
    bin_width: float  #: frequency spacing Δf in Hz
    enbw_hz: float  #: equivalent noise bandwidth of one bin, in Hz
    nenbw_bins: float
    nperseg: int
    n_segments: int
    overlap: float
    averaging: str
    detrend: str
    pad_factor: int
    has_nyquist: bool = True  #: last bin is exactly fs/2 (even FFT length)

    @property
    def ylabel(self) -> str:
        return QUANTITIES[self.quantity][0]

    def as_quantity(self, quantity: str) -> pd.DataFrame:
        """Convert the canonical power spectrum into any supported quantity."""
        return _ps_to_quantity(self.ps, quantity, self.enbw_hz, self.has_nyquist)


def _ps_to_quantity(ps: pd.DataFrame, quantity: str, enbw_hz: float,
                    has_nyquist: bool) -> pd.DataFrame:
    if quantity not in QUANTITIES:
        raise ValueError(f"quantity must be one of {sorted(QUANTITIES)}, was {quantity!r}")
    if quantity == "power":
        return ps.copy()
    if quantity == "amplitude_rms":
        return ps.pow(0.5)
    if quantity == "amplitude_peak":
        out = (2.0 * ps).pow(0.5)
        # DC (and Nyquist, when the FFT length is even) has no
        # negative-frequency twin: its peak value IS sqrt(PS)
        out.iloc[0] = np.sqrt(ps.iloc[0])
        if has_nyquist:
            out.iloc[-1] = np.sqrt(ps.iloc[-1])
        return out
    if quantity == "psd":
        return ps / enbw_hz
    if quantity == "velocity_rms":
        # Frequency-domain integration of acceleration: V_rms = A_rms/(2*pi*f),
        # in mm/s.  Only meaningful when the input is acceleration in m/s².
        # ``ps`` is indexed by frequency in Hz.
        f = ps.index.to_numpy(dtype=float)
        scale = np.zeros_like(f)
        band = f >= VELOCITY_CUTOFF_HZ
        scale[band] = 1000.0 / (2.0 * np.pi * f[band])
        return ps.pow(0.5).mul(scale, axis=0)
    return (ps / enbw_hz).pow(0.5)  # asd


def _sample_rate(df: pd.DataFrame) -> float:
    t = df.index.to_numpy(dtype=float)
    if len(t) < 2:
        raise ValueError("need at least 2 samples")
    return 1.0 / float(np.mean(np.diff(t)))


@dataclass(frozen=True)
class SignalQuality:
    """The input-integrity report of :py:func:`signal_quality`.

    The FFT assumes uniformly spaced, unclipped samples; this gate makes
    that assumption checkable instead of implicit.  ``flags`` is empty when
    the record is clean.
    """
    n: int
    fs: float  #: 1 / median(dt) - robust to isolated gaps
    dt_median: float
    max_jitter_frac: float  #: max |dt - median| / median over normal steps
    n_gaps: int  #: steps longer than 1.5x the median (dropped samples)
    n_duplicates: int  #: steps shorter than half the median (or repeated stamps)
    monotonic: bool
    clipped: int  #: samples at >= clip_frac x full_scale (0 if no full_scale)
    flags: typing.Tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.flags


def signal_quality(
        df: pd.DataFrame,
        full_scale: typing.Optional[float] = None,
        jitter_tol: float = 0.01,
        clip_frac: float = 0.98,
) -> SignalQuality:
    """
    Police the assumptions the FFT silently makes about its input.

    Checks the time base (monotonic, uniform within ``jitter_tol``, no
    gaps or duplicated stamps) and - when the sensor's ``full_scale`` is
    given, in the same units as the data - counts samples at the rail
    (``|x| >= clip_frac * full_scale``), the signature of a saturated
    sensor.  A clipped accelerometer produces spurious harmonics that look
    diagnosable, so this belongs before any spectrum reading.

    Returns a :py:class:`SignalQuality`; ``.ok`` is True when no flag was
    raised.  This function never raises on dirty data - callers decide
    whether a flag warns or rejects.
    """
    t = df.index.to_numpy(dtype=float)
    n = len(t)
    if n < 2:
        raise ValueError("need at least 2 samples")
    dt = np.diff(t)
    dt_median = float(np.median(dt))
    if dt_median <= 0:
        raise ValueError("timestamps are not increasing")
    monotonic = bool((dt > 0).all())
    gaps = int((dt > 1.5 * dt_median).sum())
    duplicates = int((dt < 0.5 * dt_median).sum())
    # judge jitter only over normal steps - a gap is its own flag, not jitter
    normal = dt[(dt >= 0.5 * dt_median) & (dt <= 1.5 * dt_median)]
    max_jitter = float(np.max(np.abs(normal - dt_median)) / dt_median) \
        if len(normal) else 0.0

    clipped = 0
    if full_scale is not None:
        rail = clip_frac * float(full_scale)
        for col in df.columns:
            x = df[col].to_numpy(dtype=float)
            clipped += int((np.abs(x) >= rail).sum())

    flags = []
    if not monotonic:
        flags.append("timestamps are not strictly increasing")
    if duplicates:
        flags.append(f"{duplicates} duplicated/compressed timestamp step(s)")
    if gaps:
        flags.append(f"{gaps} gap(s) in the sample stream (dropped samples)")
    if max_jitter > jitter_tol:
        flags.append(f"timestamp jitter up to {max_jitter:.1%} of the "
                     f"sample interval")
    if clipped:
        flags.append(f"{clipped} sample(s) at the sensor rail - "
                     f"possible clipping; spectrum harmonics may be spurious")

    return SignalQuality(
        n=n, fs=1.0 / dt_median, dt_median=dt_median,
        max_jitter_frac=max_jitter, n_gaps=gaps, n_duplicates=duplicates,
        monotonic=monotonic, clipped=clipped, flags=tuple(flags))


def _segment_starts(n: int, nperseg: int, step: int) -> np.ndarray:
    return np.arange(0, n - nperseg + 1, step)


def analyze(
        df: pd.DataFrame,
        quantity: str = "amplitude_peak",
        window: str = "hann",
        detrend: str = "none",
        nperseg: typing.Optional[int] = None,
        overlap: typing.Optional[float] = None,
        averaging: typing.Literal["none", "linear", "peak_hold"] = "linear",
        pad_factor: int = 1,
) -> SpectrumResult:
    """
    Compute a single-sided spectrum with professional analyser scaling.

    :param df: time-indexed input (seconds), one column per channel
    :param quantity: one of :py:data:`QUANTITIES`: ``"amplitude_peak"``,
        ``"amplitude_rms"``, ``"power"`` (units² RMS), ``"psd"`` (units²/Hz)
        or ``"asd"`` (units/√Hz)
    :param window: one of :py:data:`WINDOWS`
    :param detrend: per-segment detrend: "none", "mean" or "linear"
    :param nperseg: segment length; default = whole record (single FFT)
    :param overlap: fractional segment overlap in [0, 0.95]; default = the
        window's recommended overlap (0.5 for hann)
    :param averaging: ``"linear"`` (mean of segment power spectra - RMS
        averaging), ``"peak_hold"`` (bin-wise maximum) or ``"none"``
        (require a single segment)
    :param pad_factor: zero-pad each segment FFT to ``pad_factor * nperseg``
        for a finer frequency grid (amplitudes stay correct; the true
        resolution is still set by ``nperseg``)
    """
    n = len(df)
    if not isinstance(pad_factor, int) or pad_factor < 1:
        raise ValueError(f"pad_factor must be a positive integer, was {pad_factor!r}")
    if averaging not in ("none", "linear", "peak_hold"):
        raise ValueError(f'averaging must be "none", "linear" or "peak_hold", was {averaging!r}')

    nperseg = n if nperseg is None else int(min(nperseg, n))
    if nperseg < 2:
        raise ValueError("nperseg must be at least 2")
    if overlap is None:
        overlap = WINDOWS[window].recommended_overlap if window in WINDOWS else 0.5
    if not 0.0 <= overlap <= 0.95:
        raise ValueError(f"overlap must be within [0, 0.95], was {overlap}")

    fs = _sample_rate(df)
    w = get_window_samples(window, nperseg)
    figures = window_figures(window, nperseg)
    s1, s2 = figures["s1"], figures["s2"]
    enbw_hz = fs * s2 / s1 ** 2

    step = max(1, int(round(nperseg * (1.0 - overlap))))
    starts = _segment_starts(n, nperseg, step)
    if averaging == "none":
        starts = starts[:1]

    nfft = nperseg * pad_factor
    freqs = scipy.fft.rfftfreq(nfft, 1.0 / fs)
    c = np.full(len(freqs), 2.0)
    c[0] = 1.0
    if nfft % 2 == 0:
        c[-1] = 1.0

    ps_data = {}
    for col in df.columns:
        x = df[col].to_numpy(dtype=float)
        acc = None
        for start in starts:
            seg = _detrend(x[start:start + nperseg], detrend) * w
            mag2 = np.abs(scipy.fft.rfft(seg, n=nfft)) ** 2
            if acc is None:
                acc = mag2
            elif averaging == "peak_hold":
                acc = np.maximum(acc, mag2)
            else:
                acc = acc + mag2
        if averaging == "linear":
            acc = acc / len(starts)
        ps_data[col] = c * acc / s1 ** 2

    ps = pd.DataFrame(ps_data, index=pd.Index(freqs, name="frequency (Hz)"),
                      columns=df.columns)

    return SpectrumResult(
        spectrum=_ps_to_quantity(ps, quantity, enbw_hz, nfft % 2 == 0),
        ps=ps,
        quantity=quantity,
        window=window,
        fs=fs,
        bin_width=fs / nfft,
        enbw_hz=enbw_hz,
        nenbw_bins=figures["nenbw_bins"],
        nperseg=nperseg,
        n_segments=len(starts),
        overlap=overlap,
        averaging=averaging,
        detrend=detrend,
        pad_factor=pad_factor,
        has_nyquist=nfft % 2 == 0,
    )


def spectrogram(
        df: pd.DataFrame,
        column: typing.Optional[str] = None,
        window: str = "hann",
        nperseg: typing.Optional[int] = None,
        overlap: float = 0.5,
        detrend: str = "none",
) -> pd.DataFrame:
    """
    Short-time PSD map for waterfall/heatmap display: rows = segment center
    times, columns = frequencies, values in units²/Hz (same scaling as
    :py:func:`analyze` with ``quantity="psd"``).
    """
    col = column if column is not None else df.columns[0]
    n = len(df)
    if nperseg is None:
        nperseg = int(max(64, 2 ** np.floor(np.log2(max(n // 16, 64)))))
    nperseg = min(nperseg, n)

    fs = _sample_rate(df)
    w = get_window_samples(window, nperseg)
    s1 = w.sum()
    s2 = float((w * w).sum())
    enbw_hz = fs * s2 / s1 ** 2
    step = max(1, int(round(nperseg * (1.0 - overlap))))

    freqs = scipy.fft.rfftfreq(nperseg, 1.0 / fs)
    c = np.full(len(freqs), 2.0)
    c[0] = 1.0
    if nperseg % 2 == 0:
        c[-1] = 1.0

    x = df[col].to_numpy(dtype=float)
    t = df.index.to_numpy(dtype=float)
    rows, times = [], []
    for start in _segment_starts(n, nperseg, step):
        seg = _detrend(x[start:start + nperseg], detrend) * w
        ps = c * np.abs(scipy.fft.rfft(seg)) ** 2 / s1 ** 2
        rows.append(ps / enbw_hz)
        times.append(t[start + nperseg // 2])

    return pd.DataFrame(rows, index=pd.Index(times, name="time (s)"),
                        columns=pd.Index(freqs, name="frequency (Hz)"))


def band_rms(
        result: SpectrumResult,
        f_low: float = 0.0,
        f_high: typing.Optional[float] = None,
        column: typing.Optional[str] = None,
) -> float:
    """
    RMS of the signal content within a frequency band, integrated from the
    PSD: ``sqrt(sum(PSD * df))`` - the standard band-power cursor of a
    dynamic signal analyser.
    """
    col = column if column is not None else result.ps.columns[0]
    psd = result.as_quantity("psd")[col]
    freqs = psd.index.to_numpy(dtype=float)
    if f_high is None:
        f_high = freqs[-1]
    mask = (freqs >= f_low) & (freqs <= f_high)
    # the ENBW-normalized density integrates bin power; DC & Nyquist carry
    # half a bin each on the single-sided axis - close enough at Δf scale
    return float(np.sqrt(psd.to_numpy()[mask].sum() * result.bin_width))


def velocity_band_rms(
        result: SpectrumResult,
        f_low: float = 10.0,
        f_high: float = 1000.0,
        column: typing.Optional[str] = None,
) -> float:
    """
    Overall velocity level in **mm/s RMS** over an explicit frequency band -
    the headline number of ISO 20816 machine-vibration evaluation and of
    commercial condition-monitoring platforms ("Vel RMS").

    The input spectrum must come from acceleration in m/s².  The velocity
    PSD is ``PSD_a * (1000/(2*pi*f))**2`` and the band level is
    ``sqrt(integral)``, exactly parallel to :py:func:`band_rms`.  The band
    is explicit and separate from the display high-pass
    (:py:data:`VELOCITY_CUTOFF_HZ`): 10-1000 Hz is the standard band;
    ISO 20816 uses 2-1000 Hz for some large machine classes.
    """
    if f_low <= 0:
        raise ValueError("f_low must be > 0 - velocity diverges at DC")
    col = column if column is not None else result.ps.columns[0]
    psd_a = result.as_quantity("psd")[col]
    freqs = psd_a.index.to_numpy(dtype=float)
    mask = (freqs >= f_low) & (freqs <= f_high)
    psd_v = psd_a.to_numpy()[mask] * (1000.0 / (2.0 * np.pi * freqs[mask])) ** 2
    return float(np.sqrt(psd_v.sum() * result.bin_width))


def compute_spectrum(
        df: pd.DataFrame,
        window: str = "hann",
        detrend: str = "none",
        pad_factor: int = 1,
) -> pd.DataFrame:
    """
    Single-record, amplitude-corrected peak-amplitude spectrum (a sinusoid
    of amplitude A peaks at A).  Kept as the simple entry point; see
    :py:func:`analyze` for quantities, averaging and overlap.
    """
    return analyze(df, quantity="amplitude_peak", window=window,
                   detrend=detrend, averaging="none",
                   pad_factor=pad_factor).spectrum


def find_spectral_peaks(
        spectrum: typing.Union[pd.DataFrame, pd.Series],
        column: typing.Optional[str] = None,
        n_peaks: int = 10,
        min_rel_prominence_db: float = -60.0,
        interpolate: bool = True,
) -> pd.DataFrame:
    """
    Locate spectral peaks, optionally refining them below bin resolution.

    :param spectrum: an amplitude-quantity spectrum (e.g.
        :py:func:`compute_spectrum` output, or ``result.spectrum``)
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

    Computes the signal RMS both in the time domain and by integrating a
    rectangular-window PSD, and returns the relative error between them.
    For a correct implementation this is ~1e-15; anything above ~1e-6
    indicates a scaling bug somewhere in the pipeline.
    """
    col = column if column is not None else df.columns[0]
    x = df[col].to_numpy(dtype=float)

    result = analyze(df[[col]], quantity="psd", window="rectangular",
                     detrend="none", averaging="none")
    rms_freq = np.sqrt(result.spectrum[col].to_numpy().sum() * result.bin_width)

    rms_time = np.sqrt(np.mean(x ** 2))
    if rms_time == 0:
        return 0.0 if rms_freq == 0 else np.inf
    return abs(rms_freq - rms_time) / rms_time
