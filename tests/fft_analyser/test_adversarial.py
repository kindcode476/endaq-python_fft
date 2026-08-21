"""
Adversarial inputs: signals chosen to break an analyser rather than
flatter it - unresolvable tone pairs, extreme dynamic range, drifting DC,
near-Nyquist content, clipping, corrupted time bases and short records.
Each test states the physically correct outcome and pins it.
"""

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from fft_analyser.analysis import (
    analyze,
    band_rms,
    find_spectral_peaks,
    signal_quality,
)
from fft_analyser.metrics import tone_metrics


def _df(x, fs):
    t = np.arange(len(x)) / fs
    return pd.DataFrame({"signal": x}, index=pd.Index(t, name="time (s)"))


class TestUnresolvableTonePair:
    """Two tones closer than the window main lobe cannot be separated -
    the analyser must not invent two peaks, and must conserve their energy;
    lengthening the record (finer bins) must then resolve them."""

    FS = 2048.0

    def _pair(self, n):
        # 0.8 bins apart at the SHORT record length - inside any main lobe
        df_short = self.FS / 8192
        f1, f2 = 100.0, 100.0 + 0.8 * df_short
        t = np.arange(n) / self.FS
        x = np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)
        return _df(x, self.FS)

    def test_short_record_reports_one_peak_with_correct_energy(self):
        df = self._pair(8192)
        result = analyze(df, quantity="power", window="hann",
                         averaging="none")
        peaks = find_spectral_peaks(result.as_quantity("amplitude_peak"),
                                    n_peaks=5)
        near = peaks[(peaks["frequency (Hz)"] > 90)
                     & (peaks["frequency (Hz)"] < 110)]
        assert len(near) == 1  # no false resolution
        # energy is still conserved (Parseval over the record). NOTE: the
        # infinite-time RMS sqrt((a1²+a2²)/2) is NOT the right expectation
        # here - a coherent pair 0.8 bins apart beats, and one finite
        # record catches an arbitrary slice of that beat.
        rect = analyze(df, quantity="psd", window="rectangular",
                       averaging="none")
        x = df["signal"].to_numpy()
        npt.assert_allclose(band_rms(rect, 0.0, None),
                            np.sqrt(np.mean(x ** 2)), rtol=1e-9)

    def test_long_record_resolves_the_pair(self):
        result = analyze(self._pair(16 * 8192), quantity="power",
                         window="hann", averaging="none")
        peaks = find_spectral_peaks(result.as_quantity("amplitude_peak"),
                                    n_peaks=5)
        near = peaks[(peaks["frequency (Hz)"] > 90)
                     & (peaks["frequency (Hz)"] < 110)]
        assert len(near) == 2


class TestExtremeDynamicRange:
    """A -80 dBc tone beside a full-scale one: readable under a low-sidelobe
    window, buried under rectangular leakage - the reason windows exist."""

    def _signal(self):
        fs, n = 2048.0, 16384
        t = np.arange(n) / fs
        f_small = 317 * fs / n  # bin-centered, away from harmonics of 100
        x = 1.0 * np.sin(2 * np.pi * 100.0 * t) \
            + 1e-4 * np.sin(2 * np.pi * f_small * t)
        return _df(x, fs), f_small

    def _amp_at(self, result, freq):
        spec = result.as_quantity("amplitude_peak")["signal"]
        idx = np.argmin(np.abs(spec.index.to_numpy() - freq))
        return float(spec.iloc[idx])

    def test_blackman_harris_recovers_minus_80dBc_tone(self):
        df, f_small = self._signal()
        result = analyze(df, window="blackman-harris", averaging="none")
        npt.assert_allclose(self._amp_at(result, f_small), 1e-4, rtol=5e-2)

    def test_rectangular_buries_it_in_leakage(self):
        df, f_small = self._signal()
        # 100 Hz is NOT bin-centered at n=16384 (bin 800.0? 100/(2048/16384)
        # = 800 exactly) - shift the big tone off-bin to force leakage
        fs, n = 2048.0, 16384
        t = np.arange(n) / fs
        x = 1.0 * np.sin(2 * np.pi * 100.06 * t) \
            + 1e-4 * np.sin(2 * np.pi * f_small * t)
        result = analyze(_df(x, fs), window="rectangular", averaging="none")
        measured = self._amp_at(result, f_small)
        assert abs(measured - 1e-4) / 1e-4 > 0.5  # leakage-dominated


class TestDriftingDC:

    def _signal(self):
        fs, n = 2048.0, 8192
        t = np.arange(n) / fs
        x = np.linspace(0.0, 5.0, n) + 1.0 * np.sin(2 * np.pi * 100.0 * t)
        return _df(x, fs)

    def test_linear_detrend_recovers_the_tone(self):
        result = analyze(self._signal(), window="hann", detrend="linear",
                         averaging="none")
        peak = find_spectral_peaks(result.spectrum, n_peaks=1).iloc[0]
        npt.assert_allclose(peak["frequency (Hz)"], 100.0, atol=0.3)
        npt.assert_allclose(peak["amplitude"], 1.0, rtol=1e-3)

    def test_mean_detrend_leaves_low_frequency_contamination(self):
        result = analyze(self._signal(), window="hann", detrend="mean",
                         averaging="none")
        spec = result.spectrum["signal"]
        low = spec[(spec.index > 0) & (spec.index < 5.0)].max()
        assert low > 0.1  # the drift leaks into the low bins
        # ... while linear detrend suppresses it by orders of magnitude
        clean = analyze(self._signal(), window="hann", detrend="linear",
                        averaging="none").spectrum["signal"]
        assert clean[(clean.index > 0) & (clean.index < 5.0)].max() < low / 100


class TestNearNyquist:

    def test_bin_centered_tone_just_below_nyquist(self):
        fs, n = 2048.0, 8192
        f = fs / 2 - 2 * fs / n  # two bins below Nyquist: interior bin
        t = np.arange(n) / fs
        x = np.sin(2 * np.pi * f * t)
        result = analyze(_df(x, fs), window="hann", averaging="none")
        peak = find_spectral_peaks(result.spectrum, n_peaks=1).iloc[0]
        npt.assert_allclose(peak["frequency (Hz)"], f, atol=fs / n)
        npt.assert_allclose(peak["amplitude"], 1.0, rtol=1e-6)


class TestExtremeDCToACRatio:

    def test_millivolt_tone_under_ten_units_of_dc(self):
        fs, n = 2048.0, 8192
        t = np.arange(n) / fs
        x = 10.0 + 1e-3 * np.sin(2 * np.pi * 100.0 * t)  # DC 10,000x the AC
        result = analyze(_df(x, fs), window="hann", detrend="mean",
                         averaging="none")
        peak = find_spectral_peaks(result.spectrum, n_peaks=1).iloc[0]
        npt.assert_allclose(peak["frequency (Hz)"], 100.0, atol=0.3)
        npt.assert_allclose(peak["amplitude"], 1e-3, rtol=1e-3)
        assert result.spectrum["signal"].iloc[0] < 1e-12  # DC bin removed


class TestClipping:

    def _clipped(self, amp=2.0, rail=1.0):
        fs, n = 2048.0, 8192
        t = np.arange(n) / fs
        return _df(np.clip(amp * np.sin(2 * np.pi * 100.0 * t), -rail, rail),
                   fs)

    def test_quality_gate_flags_the_rail(self):
        q = signal_quality(self._clipped(), full_scale=1.0)
        assert q.clipped > 0
        assert not q.ok
        assert any("clipping" in f for f in q.flags)

    def test_unclipped_signal_passes(self):
        fs, n = 2048.0, 8192
        t = np.arange(n) / fs
        q = signal_quality(_df(0.5 * np.sin(2 * np.pi * 100.0 * t), fs),
                           full_scale=1.0)
        assert q.ok and q.clipped == 0

    def test_clipping_manufactures_harmonics(self):
        # the spurious-harmonic mechanism the gate warns about, made visible
        m = tone_metrics(self._clipped())
        assert m.thd_pct > 5.0


class TestTimeBaseIntegrity:

    def _uniform_index(self, n=4096, fs=2048.0):
        return np.arange(n) / fs

    def _quality(self, t):
        x = np.sin(2 * np.pi * 100.0 * np.asarray(t))
        return signal_quality(
            pd.DataFrame({"signal": x}, index=pd.Index(t, name="time (s)")))

    def test_clean_uniform_record_is_ok(self):
        q = self._quality(self._uniform_index())
        assert q.ok
        npt.assert_allclose(q.fs, 2048.0, rtol=1e-9)

    def test_gap_is_flagged(self):
        t = self._uniform_index()
        t[2000:] += 50.0 / 2048.0  # 50 dropped samples mid-record
        q = self._quality(t)
        assert q.n_gaps == 1
        assert any("gap" in f for f in q.flags)

    def test_duplicated_timestamps_are_flagged(self):
        t = self._uniform_index()
        t[1000] = t[999]  # a repeated stamp
        q = self._quality(t)
        assert q.n_duplicates >= 1
        assert not q.ok

    def test_jitter_is_flagged(self):
        rng = np.random.default_rng(3)
        t = self._uniform_index()
        t += rng.uniform(-0.05, 0.05, len(t)) / 2048.0  # 5% jitter
        t.sort()
        q = self._quality(t)
        assert q.max_jitter_frac > 0.01
        assert any("jitter" in f for f in q.flags)

    def test_non_monotonic_is_flagged(self):
        t = self._uniform_index()
        t[500], t[501] = t[501], t[500]
        q = self._quality(t)
        assert not q.monotonic
        assert not q.ok


class TestShortRecordClamp:
    """A record shorter than the requested segment must clamp, not crash -
    pinned so the behaviour survives refactors."""

    def test_segment_clamps_to_the_record(self):
        fs, n = 1000.0, 1000
        t = np.arange(n) / fs
        x = np.sin(2 * np.pi * 100.0 * t)  # bin-centered at nperseg=1000
        result = analyze(_df(x, fs), window="hann", nperseg=8192,
                         averaging="linear")
        assert result.nperseg == n
        assert result.n_segments == 1
        peak = find_spectral_peaks(result.spectrum, n_peaks=1).iloc[0]
        npt.assert_allclose(peak["amplitude"], 1.0, rtol=1e-6)
