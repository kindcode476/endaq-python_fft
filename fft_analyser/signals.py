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
        description=f"Gaussian noise, rms {rms:g}",
        data=pd.DataFrame({"signal": _noise(len(t), rms, seed)},
                          index=pd.Index(t, name="time (s)")),
        fs=fs,
        expected_band=(0.0, fs / 2),
    )


#: Registry used by the UI dropdown; each entry builds a signal with defaults.
TEST_SIGNALS: typing.Dict[str, typing.Callable[..., TestSignal]] = {
    "Multi-tone (50/120/240 Hz)": multi_tone,
    "Single tone (100 Hz)": single_tone,
    "DC + tone": dc_plus_tone,
    "Square wave (50 Hz)": square_wave,
    "Impulse": impulse,
    "Linear chirp (20-400 Hz)": linear_chirp,
    "White noise": white_noise,
}
