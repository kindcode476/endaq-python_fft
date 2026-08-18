"""
Test-signal generators with known ground truth.

Every generator returns a :py:class:`TestSignal` whose ``expected_peaks``
describe the discrete spectral lines the analyser must recover (frequency in
Hz, amplitude in signal units).  Because the analysis pipeline uses the
"unit" normalization from :py:mod:`endaq.calc.fft`, a sinusoid of amplitude A
must appear in the spectrum as a peak of height A - which makes correctness
directly checkable, both in pytest and visually in the UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import typing

import numpy as np
import pandas as pd

__all__ = [
    "TestSignal",
    "multi_tone",
    "single_tone",
    "dc_plus_tone",
    "square_wave",
    "impulse",
    "linear_chirp",
    "white_noise",
    "distorted_tone",
    "quantized_tone",
    "two_tone",
    "TEST_SIGNALS",
]


@dataclass
class TestSignal:
    """A generated time series plus the spectral content it should produce."""

    name: str
    description: str
    data: pd.DataFrame  #: time-indexed (seconds), one column per channel
    fs: float  #: sample rate in Hz
    #: ground-truth spectral lines as (frequency Hz, amplitude) pairs;
    #: empty for signals without discrete lines (chirp, noise)
    expected_peaks: typing.List[typing.Tuple[float, float]] = field(default_factory=list)
    #: for broadband signals: (f_low, f_high) where energy should concentrate
    expected_band: typing.Optional[typing.Tuple[float, float]] = None
    #: for noise: the single-sided PSD floor (units²/Hz) the analyser must
    #: report independent of window and FFT length (= 2*sigma²/fs)
    expected_psd_level: typing.Optional[float] = None


def _time(fs: float, duration: float) -> np.ndarray:
    return np.arange(int(round(fs * duration))) / fs


def _noise(n: int, rms: float, seed: int) -> np.ndarray:
    if rms <= 0:
        return np.zeros(n)
    return np.random.default_rng(seed).normal(0.0, rms, n)


def multi_tone(
        freqs: typing.Sequence[float] = (50.0, 120.0, 240.0),
        amps: typing.Sequence[float] = (1.0, 0.5, 0.25),
        phases: typing.Optional[typing.Sequence[float]] = None,
        fs: float = 2048.0,
        duration: float = 2.0,
        noise_rms: float = 0.0,
        seed: int = 12345,
) -> TestSignal:
    """
    Sum of sinusoids - the primary correctness probe: each (freq, amp) pair
    must reappear as a spectral peak of that exact height.
    """
    if len(freqs) != len(amps):
        raise ValueError("freqs and amps must have the same length")
    if phases is None:
        phases = np.zeros(len(freqs))
    t = _time(fs, duration)
    y = _noise(len(t), noise_rms, seed)
    for f, a, p in zip(freqs, amps, phases):
        if f >= fs / 2:
            raise ValueError(f"tone at {f} Hz is at/above Nyquist ({fs / 2} Hz)")
        y = y + a * np.sin(2 * np.pi * f * t + p)
    tones = ", ".join(f"{a:g} @ {f:g} Hz" for f, a in zip(freqs, amps))
    return TestSignal(
        name="Multi-tone",
        description=f"Sinusoids: {tones}" + (f" + noise (rms {noise_rms:g})" if noise_rms else ""),
        data=pd.DataFrame({"signal": y}, index=pd.Index(t, name="time (s)")),
        fs=fs,
        expected_peaks=list(zip(freqs, amps)),
    )


def single_tone(
        freq: float = 100.0,
        amp: float = 1.0,
        fs: float = 2048.0,
        duration: float = 2.0,
        noise_rms: float = 0.0,
        seed: int = 12345,
) -> TestSignal:
    """A single sinusoid of known frequency and amplitude."""
    sig = multi_tone((freq,), (amp,), fs=fs, duration=duration,
                     noise_rms=noise_rms, seed=seed)
    sig.name = "Single tone"
    return sig


def dc_plus_tone(
        offset: float = 2.0,
        freq: float = 100.0,
        amp: float = 1.0,
        fs: float = 2048.0,
        duration: float = 2.0,
) -> TestSignal:
    """
    A sinusoid riding on a DC offset - exercises DC-bin scaling (the 0 Hz bin
    must read ``offset``, not ``2*offset``) and the detrend options.
    """
    sig = multi_tone((freq,), (amp,), fs=fs, duration=duration)
    sig.data["signal"] += offset
    sig.name = "DC + tone"
    sig.description = f"{amp:g} @ {freq:g} Hz on a DC offset of {offset:g}"
    sig.expected_peaks = [(0.0, offset), (freq, amp)]
    return sig


def square_wave(
        freq: float = 50.0,
        amp: float = 1.0,
        fs: float = 8192.0,
        duration: float = 2.0,
        n_harmonics: int = 5,
) -> TestSignal:
    """
    Square wave - its Fourier series puts lines at odd harmonics k with
    amplitude ``4*amp/(pi*k)``, a stringent multi-line accuracy check.
    """
    t = _time(fs, duration)
    y = amp * np.sign(np.sin(2 * np.pi * freq * t))
    harmonics = [(freq * k, 4 * amp / (np.pi * k))
                 for k in range(1, 2 * n_harmonics, 2)
                 if freq * k < fs / 2]
    return TestSignal(
        name="Square wave",
        description=f"{amp:g} square @ {freq:g} Hz (odd harmonics at 4A/pi/k)",
        data=pd.DataFrame({"signal": y}, index=pd.Index(t, name="time (s)")),
        fs=fs,
        expected_peaks=harmonics,
    )


def impulse(
        amp: float = 1.0,
        fs: float = 2048.0,
        duration: float = 1.0,
) -> TestSignal:
    """
    Unit impulse - the magnitude spectrum must be flat at ``2*amp/N``
    (single-sided unit normalization), checking broadband flatness.
    """
    t = _time(fs, duration)
    y = np.zeros(len(t))
    y[len(t) // 2] = amp
    return TestSignal(
        name="Impulse",
        description=f"Impulse of amplitude {amp:g}: flat spectrum at 2A/N = {2 * amp / len(t):.3g}",
        data=pd.DataFrame({"signal": y}, index=pd.Index(t, name="time (s)")),
        fs=fs,
        expected_band=(0.0, fs / 2),
    )


def linear_chirp(
        f0: float = 20.0,
        f1: float = 400.0,
        amp: float = 1.0,
        fs: float = 2048.0,
        duration: float = 2.0,
) -> TestSignal:
    """Linear sweep - spectral energy must sit inside [f0, f1] and nowhere else."""
    t = _time(fs, duration)
    phase = 2 * np.pi * (f0 * t + (f1 - f0) / (2 * duration) * t ** 2)
    return TestSignal(
        name="Linear chirp",
        description=f"Sweep {f0:g} -> {f1:g} Hz over {duration:g} s",
        data=pd.DataFrame({"signal": amp * np.sin(phase)},
                          index=pd.Index(t, name="time (s)")),
        fs=fs,
        expected_band=(f0, f1),
    )


def white_noise(
        rms: float = 1.0,
        fs: float = 2048.0,
        duration: float = 2.0,
        seed: int = 12345,
) -> TestSignal:
    """Gaussian white noise - a flat, featureless floor (no discrete peaks)."""
    t = _time(fs, duration)
    return TestSignal(
        name="White noise",
        description=(f"Gaussian noise, rms {rms:g} - single-sided PSD floor "
                     f"= 2σ²/fs = {2 * rms ** 2 / fs:.3g} units²/Hz"),
        data=pd.DataFrame({"signal": _noise(len(t), rms, seed)},
                          index=pd.Index(t, name="time (s)")),
        fs=fs,
        expected_band=(0.0, fs / 2),
        expected_psd_level=2 * rms ** 2 / fs,
    )


def distorted_tone(
        freq: float = 100.0,
        amp: float = 1.0,
        harmonics: typing.Optional[typing.Dict[int, float]] = None,
        fs: float = 8192.0,
        duration: float = 2.0,
) -> TestSignal:
    """
    Fundamental plus harmonics at known relative levels - ground truth for
    THD: ``THD = sqrt(sum(r_h^2))`` where ``r_h`` is each harmonic's
    amplitude relative to the fundamental.
    """
    if harmonics is None:
        harmonics = {2: 0.02, 3: 0.01}  # THD = sqrt(5)% ≈ 2.236%
    freqs = [freq] + [h * freq for h in harmonics]
    amps = [amp] + [r * amp for r in harmonics.values()]
    sig = multi_tone(freqs, amps, fs=fs, duration=duration)
    thd = float(np.sqrt(sum(r * r for r in harmonics.values())))
    sig.name = "Distorted tone"
    sig.description = (f"{amp:g} @ {freq:g} Hz with harmonics "
                       + ", ".join(f"H{h}={r * 100:g}%" for h, r in harmonics.items())
                       + f" (THD = {thd * 100:.3f}%)")
    return sig


def quantized_tone(
        bits: int = 8,
        freq: float = 100.372,
        amp: float = 1.0,
        fs: float = 8192.0,
        duration: float = 2.0,
) -> TestSignal:
    """
    An ideal ``bits``-bit quantized full-scale sine - ground truth for
    SINAD/ENOB: an ideal quantizer yields ``SINAD = 6.02*bits + 1.76`` dB,
    i.e. ENOB = ``bits``.  The default frequency is deliberately unrelated
    to the bin grid so quantization error spreads as noise rather than
    collapsing onto harmonic bins.
    """
    sig = multi_tone((freq,), (amp,), fs=fs, duration=duration)
    q = amp / (2 ** (bits - 1))
    sig.data["signal"] = np.round(sig.data["signal"] / q) * q
    sig.name = f"Quantized tone ({bits}-bit)"
    sig.description = (f"{amp:g} @ {freq:g} Hz through an ideal {bits}-bit "
                       f"quantizer (expect ENOB ≈ {bits})")
    return sig


def two_tone(
        f1: float = 100.0,
        f2: float = 110.0,
        amps: typing.Tuple[float, float] = (1.0, 1.0),
        fs: float = 2048.0,
        duration: float = 2.0,
) -> TestSignal:
    """Two closely spaced tones - the classic frequency-resolution probe."""
    sig = multi_tone((f1, f2), amps, fs=fs, duration=duration)
    sig.name = "Two-tone"
    sig.description = (f"Tones at {f1:g} & {f2:g} Hz - resolving them needs "
                       f"bin width (x NENBW) below {abs(f2 - f1):g} Hz")
    return sig


#: Registry used by the UI dropdown; each entry builds a signal with defaults.
TEST_SIGNALS: typing.Dict[str, typing.Callable[..., TestSignal]] = {
    "Multi-tone (50/120/240 Hz)": multi_tone,
    "Single tone (100 Hz)": single_tone,
    "Two-tone (100/110 Hz)": two_tone,
    "Distorted tone (THD 2.24%)": distorted_tone,
    "Quantized tone (8-bit)": quantized_tone,
    "DC + tone": dc_plus_tone,
    "Square wave (50 Hz)": square_wave,
    "Impulse": impulse,
    "Linear chirp (20-400 Hz)": linear_chirp,
    "White noise": white_noise,
}
