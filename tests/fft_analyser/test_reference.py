"""
Independent-reference validation: the same signals through this module AND
through SciPy's spectral estimators (an implementation this module does not
call for its own spectra), compared numerically.

The engine computes its spectra from raw rFFTs with hand-rolled
segmentation and S1/S2 scaling; ``scipy.signal.welch`` / ``periodogram``
are therefore genuinely independent implementations of the same published
conventions.  Agreement here is evidence the conventions were implemented,
not merely cited.
"""

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
import scipy.signal

from fft_analyser import signals
from fft_analyser.analysis import (
    VELOCITY_CUTOFF_HZ,
    analyze,
    band_rms,
    velocity_band_rms,
    window_figures,
)


def _df(x, fs):
    t = np.arange(len(x)) / fs
    return pd.DataFrame({"signal": x}, index=pd.Index(t, name="time (s)"))


@pytest.fixture
def noisy_tone():
    rng = np.random.default_rng(7)
    fs, n = 2048.0, 16384
    t = np.arange(n) / fs
    x = np.sin(2 * np.pi * 100.0 * t) + 0.2 * rng.standard_normal(n)
    return x, fs


class TestAgainstSciPyWelch:
    """Bin-for-bin agreement with scipy.signal.welch on identical settings."""

    @pytest.mark.parametrize("window", ["hann", "blackman", "flattop"])
    def test_psd_matches_welch_density(self, noisy_tone, window):
        x, fs = noisy_tone
        nperseg = 4096
        ours = analyze(_df(x, fs), quantity="psd", window=window,
                       nperseg=nperseg, overlap=0.5, averaging="linear",
                       detrend="none")
        f_ref, p_ref = scipy.signal.welch(
            x, fs, window=window, nperseg=nperseg, noverlap=nperseg // 2,
            detrend=False, scaling="density")
        npt.assert_allclose(ours.spectrum.index.to_numpy(), f_ref, rtol=1e-12)
        npt.assert_allclose(ours.spectrum["signal"].to_numpy(), p_ref,
                            rtol=1e-9)

    def test_power_matches_welch_spectrum(self, noisy_tone):
        x, fs = noisy_tone
        ours = analyze(_df(x, fs), quantity="power", window="hann",
                       nperseg=4096, overlap=0.5, averaging="linear",
                       detrend="none")
        _, p_ref = scipy.signal.welch(
            x, fs, window="hann", nperseg=4096, noverlap=2048,
            detrend=False, scaling="spectrum")
        npt.assert_allclose(ours.spectrum["signal"].to_numpy(), p_ref,
                            rtol=1e-9)

    def test_mean_detrend_matches_welch_constant(self, noisy_tone):
        x, fs = noisy_tone
        ours = analyze(_df(x + 3.0, fs), quantity="psd", window="hann",
                       nperseg=4096, overlap=0.5, averaging="linear",
                       detrend="mean")
        _, p_ref = scipy.signal.welch(
            x + 3.0, fs, window="hann", nperseg=4096, noverlap=2048,
            detrend="constant", scaling="density")
        npt.assert_allclose(ours.spectrum["signal"].to_numpy(), p_ref,
                            rtol=1e-9)

    def test_single_segment_matches_periodogram(self, noisy_tone):
        x, fs = noisy_tone
        ours = analyze(_df(x, fs), quantity="psd", window="rectangular",
                       averaging="none", detrend="none")
        _, p_ref = scipy.signal.periodogram(
            x, fs, window="boxcar", detrend=False, scaling="density")
        npt.assert_allclose(ours.spectrum["signal"].to_numpy(), p_ref,
                            rtol=1e-9)

    def test_velocity_matches_scipy_derived(self, noisy_tone):
        # velocity from OUR engine vs the same conversion applied to the
        # SciPy density: v = sqrt(psd * enbw)/(2*pi*f) * 1000
        x, fs = noisy_tone
        nperseg = 4096
        ours = analyze(_df(x, fs), quantity="velocity_rms", window="hann",
                       nperseg=nperseg, overlap=0.5, averaging="linear",
                       detrend="none")
        f_ref, p_ref = scipy.signal.welch(
            x, fs, window="hann", nperseg=nperseg, noverlap=nperseg // 2,
            detrend=False, scaling="density")
        w = scipy.signal.get_window("hann", nperseg, fftbins=True)
        enbw = fs * (w ** 2).sum() / w.sum() ** 2
        band = f_ref >= VELOCITY_CUTOFF_HZ
        v_ref = np.sqrt(p_ref[band] * enbw) * 1000.0 / (2 * np.pi * f_ref[band])
        npt.assert_allclose(ours.spectrum["signal"].to_numpy()[band], v_ref,
                            rtol=1e-9)

    def test_band_rms_matches_scipy_integral(self, noisy_tone):
        x, fs = noisy_tone
        ours = analyze(_df(x, fs), quantity="psd", window="hann",
                       nperseg=4096, overlap=0.5, averaging="linear",
                       detrend="none")
        f_ref, p_ref = scipy.signal.welch(
            x, fs, window="hann", nperseg=4096, noverlap=2048,
            detrend=False, scaling="density")
        df_hz = f_ref[1] - f_ref[0]
        mask = (f_ref >= 10.0) & (f_ref <= 1000.0)
        npt.assert_allclose(band_rms(ours, 10.0, 1000.0),
                            np.sqrt(p_ref[mask].sum() * df_hz), rtol=1e-9)


class TestVelocityBandRMS:

    def test_tone_velocity_band_rms_is_analytic(self):
        # a lone 1 m/s² pk tone at 100 Hz carries the band's entire energy:
        # Vel RMS(10-1000) = (1/sqrt2)/(2*pi*100)*1000 mm/s
        sig = signals.single_tone(freq=100.0, amp=1.0)
        result = analyze(sig.data, quantity="psd", window="hann",
                         nperseg=4096, overlap=0.5, averaging="linear")
        expected = 1000.0 / (np.sqrt(2.0) * 2.0 * np.pi * 100.0)
        npt.assert_allclose(velocity_band_rms(result, 10.0, 1000.0),
                            expected, rtol=1e-3)

    def test_out_of_band_content_excluded(self):
        # a 5 Hz tone must not contribute to the 10-1000 Hz level
        fs, n = 2048.0, 16384
        t = np.arange(n) / fs
        x = np.sin(2 * np.pi * 5.0 * t) + 0.001 * np.sin(2 * np.pi * 100.0 * t)
        result = analyze(_df(x, fs), quantity="psd", window="hann",
                         nperseg=8192, averaging="linear")
        expected = 0.001 * 1000.0 / (np.sqrt(2.0) * 2.0 * np.pi * 100.0)
        npt.assert_allclose(velocity_band_rms(result, 10.0, 1000.0),
                            expected, rtol=2e-2)

    def test_band_must_exclude_dc(self):
        sig = signals.single_tone()
        result = analyze(sig.data, quantity="psd")
        with pytest.raises(ValueError):
            velocity_band_rms(result, 0.0, 1000.0)

    def test_enbw_figures_match_scipy_windows(self):
        for window, scipy_name in [("hann", "hann"), ("flattop", "flattop"),
                                   ("blackman-harris", "blackmanharris")]:
            f = window_figures(window, 4096)
            w = scipy.signal.get_window(scipy_name, 4096, fftbins=True)
            npt.assert_allclose(f["nenbw_bins"],
                                4096 * (w ** 2).sum() / w.sum() ** 2,
                                rtol=1e-12)
