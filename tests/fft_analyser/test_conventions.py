"""
Tests that pin the analyser to published professional conventions:
window figures of merit per Harris 1978 / Heinzel 2002, scaling identities
per Heinzel 2002 / scipy.signal.welch, and Welch averaging behaviour.
"""

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from fft_analyser import signals
from fft_analyser.analysis import (
    WINDOWS,
    QUANTITIES,
    analyze,
    band_rms,
    spectrogram,
    window_figures,
)


class TestWindowFiguresOfMerit:
    """Computed figures must reproduce the published tables."""

    # (window, NENBW bins, scalloping loss dB) - Harris 1978 / Heinzel 2002
    LITERATURE = [
        ("rectangular", 1.0000, -3.92),
        ("hann", 1.5000, -1.42),
        ("hamming", 1.3628, -1.75),
        ("blackman", 1.7268, -1.10),
        ("blackman-harris", 2.0044, -0.83),
        ("flattop", 3.7702, -0.01),
    ]

    @pytest.mark.parametrize("window, nenbw, scallop", LITERATURE)
    def test_nenbw_matches_literature(self, window, nenbw, scallop):
        f = window_figures(window, 4096)
        npt.assert_allclose(f["nenbw_bins"], nenbw, atol=5e-4)
        npt.assert_allclose(f["scalloping_db"], scallop, atol=0.01)

    def test_hann_nenbw_is_exactly_1_5(self):
        # periodic hann: S1 = N/2, S2 = 3N/8 -> NENBW = 3/2 exactly
        for n in (256, 1000, 4096):
            npt.assert_allclose(window_figures("hann", n)["nenbw_bins"], 1.5,
                                rtol=1e-12)

    def test_every_offered_window_has_finite_figures(self):
        for window in WINDOWS:
            f = window_figures(window, 1024)
            assert f["s1"] > 0 and f["s2"] > 0
            assert 1.0 <= f["nenbw_bins"] < 4.0


class TestScalingIdentities:
    """The Heinzel 2002 / scipy.signal.welch identities between quantities."""

    @pytest.fixture
    def tone_result(self):
        return analyze(signals.single_tone().data, quantity="power",
                       window="hann")

    def test_ps_over_psd_equals_enbw(self, tone_result):
        ps = tone_result.as_quantity("power")["signal"]
        psd = tone_result.as_quantity("psd")["signal"]
        mask = psd > psd.max() * 1e-6
        npt.assert_allclose((ps[mask] / psd[mask]).to_numpy(),
                            tone_result.enbw_hz, rtol=1e-9)

    def test_amplitude_rms_is_peak_over_sqrt2(self, tone_result):
        pk = tone_result.as_quantity("amplitude_peak")["signal"]
        rms = tone_result.as_quantity("amplitude_rms")["signal"]
        # interior bins only (DC/Nyquist follow the no-twin convention)
        npt.assert_allclose(pk.iloc[1:-1].to_numpy(),
                            rms.iloc[1:-1].to_numpy() * np.sqrt(2), rtol=1e-9)

    def test_power_is_rms_squared_and_asd_is_sqrt_psd(self, tone_result):
        rms = tone_result.as_quantity("amplitude_rms")["signal"].to_numpy()
        power = tone_result.as_quantity("power")["signal"].to_numpy()
        psd = tone_result.as_quantity("psd")["signal"].to_numpy()
        asd = tone_result.as_quantity("asd")["signal"].to_numpy()
        npt.assert_allclose(power, rms ** 2, rtol=1e-9)
        npt.assert_allclose(asd ** 2, psd, rtol=1e-9)

    def test_tone_psd_peak_is_rms_power_over_enbw(self):
        # sinusoid read from a density: a_rms^2 / ENBW (resolution-dependent
        # by design - the reason densities are not used to read tones)
        result = analyze(signals.single_tone().data, quantity="psd",
                         window="hann")
        peak = result.spectrum["signal"].max()
        npt.assert_allclose(peak, 0.5 / result.enbw_hz, rtol=1e-6)

    def test_enbw_scales_inversely_with_segment_length(self):
        sig = signals.white_noise(duration=4.0)
        r1 = analyze(sig.data, window="hann", nperseg=1024)
        r2 = analyze(sig.data, window="hann", nperseg=2048)
        npt.assert_allclose(r1.enbw_hz / r2.enbw_hz, 2.0, rtol=1e-9)


class TestSinusoidVsNoiseReadingRules:
    """The core professional convention: tones are window/length-invariant in
    amplitude spectra; noise floors are window/length-invariant in densities."""

    @pytest.mark.parametrize("window", list(WINDOWS))
    def test_tone_amplitude_invariant_to_window(self, window):
        result = analyze(signals.single_tone().data, window=window,
                         quantity="amplitude_peak")
        npt.assert_allclose(result.spectrum["signal"].max(), 1.0, rtol=1e-6)

    @pytest.mark.parametrize("window", ["rectangular", "hann", "flattop"])
    def test_noise_psd_floor_invariant_to_window(self, window):
        sig = signals.white_noise(rms=1.0, duration=8.0)
        result = analyze(sig.data, quantity="psd", window=window,
                         nperseg=2048, averaging="linear")
        floor = result.spectrum["signal"].iloc[5:-5].mean()
        npt.assert_allclose(floor, sig.expected_psd_level, rtol=0.03)

    @pytest.mark.parametrize("nperseg", [512, 1024, 4096])
    def test_noise_psd_floor_invariant_to_fft_length(self, nperseg):
        sig = signals.white_noise(rms=0.5, duration=8.0)
        result = analyze(sig.data, quantity="psd", window="hann",
                         nperseg=nperseg)
        floor = result.spectrum["signal"].iloc[5:-5].mean()
        npt.assert_allclose(floor, sig.expected_psd_level, rtol=0.05)

    def test_psd_integral_equals_signal_power(self):
        # sum(PSD * df) over the band = total mean square, any window
        sig = signals.multi_tone(noise_rms=0.3)
        x = sig.data["signal"].to_numpy()
        for window in ("rectangular", "hann"):
            result = analyze(sig.data, quantity="psd", window=window,
                             averaging="none")
            power = result.spectrum["signal"].sum() * result.bin_width
            npt.assert_allclose(power, np.mean(x ** 2), rtol=1e-2)


class TestWelchAveraging:

    def test_segment_count(self):
        sig = signals.white_noise(duration=4.0)  # 8192 samples
        r = analyze(sig.data, nperseg=1024, overlap=0.5)
        assert r.n_segments == 15  # (8192 - 1024)/512 + 1
        r = analyze(sig.data, nperseg=1024, overlap=0.0)
        assert r.n_segments == 8

    def test_linear_averaging_reduces_variance(self):
        sig = signals.white_noise(rms=1.0, duration=16.0)
        single = analyze(sig.data, quantity="psd", window="hann",
                         nperseg=1024, averaging="none").spectrum["signal"]
        avg = analyze(sig.data, quantity="psd", window="hann",
                      nperseg=1024, averaging="linear")
        s = avg.spectrum["signal"].iloc[5:-5]
        rel_single = single.iloc[5:-5].std() / single.iloc[5:-5].mean()
        rel_avg = s.std() / s.mean()
        # ~1/sqrt(M) improvement; allow slack for overlap correlation
        assert rel_avg < rel_single / np.sqrt(avg.n_segments) * 2.0
        assert rel_avg < 0.25 * rel_single

    def test_peak_hold_bounds_linear_average(self):
        sig = signals.white_noise(duration=8.0)
        lin = analyze(sig.data, quantity="psd", nperseg=1024,
                      averaging="linear").spectrum["signal"]
        ph = analyze(sig.data, quantity="psd", nperseg=1024,
                     averaging="peak_hold").spectrum["signal"]
        assert (ph >= lin - 1e-15).all()

    def test_averaging_none_uses_one_segment(self):
        sig = signals.white_noise(duration=4.0)
        r = analyze(sig.data, nperseg=1024, averaging="none")
        assert r.n_segments == 1

    def test_invalid_parameters_raise(self):
        df = signals.single_tone().data
        with pytest.raises(ValueError):
            analyze(df, averaging="quadratic")
        with pytest.raises(ValueError):
            analyze(df, overlap=0.99)
        with pytest.raises(ValueError):
            analyze(df, quantity="loudness")


class TestBandRMS:

    def test_tone_band_rms(self):
        result = analyze(signals.single_tone().data, quantity="psd",
                         window="hann")
        npt.assert_allclose(band_rms(result, 90, 110), 1 / np.sqrt(2),
                            rtol=1e-6)

    def test_full_band_rms_matches_time_domain(self):
        sig = signals.multi_tone(noise_rms=0.2)
        result = analyze(sig.data, quantity="psd", window="rectangular",
                         averaging="none")
        x = sig.data["signal"].to_numpy()
        npt.assert_allclose(band_rms(result), np.sqrt(np.mean(x ** 2)),
                            rtol=1e-6)

    def test_out_of_band_tone_excluded(self):
        result = analyze(signals.single_tone(freq=100.0).data,
                         quantity="psd", window="hann")
        assert band_rms(result, 300, 500) < 1e-3


class TestSpectrogram:

    def test_chirp_ridge_follows_sweep(self):
        sig = signals.linear_chirp(f0=50.0, f1=400.0, duration=2.0)
        sgram = spectrogram(sig.data, nperseg=256)
        ridge = sgram.idxmax(axis=1).to_numpy()
        t = sgram.index.to_numpy()
        expected = 50.0 + (400.0 - 50.0) / 2.0 * t
        npt.assert_allclose(ridge, expected, atol=25.0)

    def test_scaling_matches_analyze_psd(self):
        sig = signals.white_noise(rms=1.0, duration=8.0)
        sgram = spectrogram(sig.data, nperseg=1024, overlap=0.5)
        floor = sgram.to_numpy()[:, 5:-5].mean()
        npt.assert_allclose(floor, sig.expected_psd_level, rtol=0.05)
