import numpy as np
import numpy.testing as npt
import pytest

from fft_analyser import signals
from fft_analyser.analysis import (
    WINDOWS,
    compute_spectrum,
    find_spectral_peaks,
    compare_to_expected,
    parseval_rms_error,
)


class TestAmplitudeAccuracy:
    """A sinusoid of amplitude A must peak at A - for every window."""

    @pytest.mark.parametrize("window", list(WINDOWS))
    def test_bin_centered_tones_recover_exact_amplitudes(self, window):
        sig = signals.multi_tone()  # 1.0/0.5/0.25 at 50/120/240 Hz, 2 s @ 2048 Hz
        spectrum = compute_spectrum(sig.data, window=window)
        peaks = find_spectral_peaks(spectrum, n_peaks=3)
        result = compare_to_expected(peaks, sig.expected_peaks,
                                     freq_tolerance=0.1, amp_tolerance_pct=0.1)
        assert result["pass"].all(), result.to_string()

    def test_off_bin_tone_with_flattop_and_padding(self):
        # worst case for scalloping loss: tone half-way between bins
        sig = signals.single_tone(freq=100.25, amp=1.0, fs=2048.0, duration=2.0)
        spectrum = compute_spectrum(sig.data, window="flattop", pad_factor=4)
        peak = find_spectral_peaks(spectrum, n_peaks=1).iloc[0]
        assert abs(peak["frequency (Hz)"] - 100.25) < 0.15
        assert abs(peak["amplitude"] - 1.0) < 0.01

    def test_zero_padding_preserves_amplitude(self):
        sig = signals.single_tone()
        for pad in (1, 2, 4, 8):
            spectrum = compute_spectrum(sig.data, window="hann", pad_factor=pad)
            peak = find_spectral_peaks(spectrum, n_peaks=1).iloc[0]
            npt.assert_allclose(peak["amplitude"], 1.0, rtol=1e-3)

    def test_square_wave_harmonic_amplitudes(self):
        sig = signals.square_wave()
        spectrum = compute_spectrum(sig.data, window="flattop")
        peaks = find_spectral_peaks(spectrum, n_peaks=len(sig.expected_peaks))
        result = compare_to_expected(peaks, sig.expected_peaks,
                                     freq_tolerance=0.5, amp_tolerance_pct=1.0)
        assert result["pass"].all(), result.to_string()


class TestDCHandling:

    def test_dc_bin_is_not_doubled(self):
        sig = signals.dc_plus_tone(offset=2.0)
        spectrum = compute_spectrum(sig.data, window="rectangular")
        npt.assert_allclose(spectrum["signal"].iloc[0], 2.0, rtol=1e-9)

    def test_mean_detrend_removes_dc(self):
        sig = signals.dc_plus_tone(offset=2.0)
        spectrum = compute_spectrum(sig.data, window="rectangular",
                                    detrend="mean")
        assert spectrum["signal"].iloc[0] < 1e-9

    def test_nyquist_bin_is_not_doubled(self):
        # a cos at exactly Nyquist lands entirely in the last bin
        fs, n = 1000.0, 1000
        t = np.arange(n) / fs
        import pandas as pd
        df = pd.DataFrame({"signal": np.cos(2 * np.pi * (fs / 2) * t)},
                          index=pd.Index(t, name="time (s)"))
        spectrum = compute_spectrum(df, window="rectangular")
        npt.assert_allclose(spectrum["signal"].iloc[-1], 1.0, rtol=1e-9)


class TestEnergyAndBroadband:

    def test_parseval_on_noisy_multitone(self):
        sig = signals.multi_tone(noise_rms=0.3)
        assert parseval_rms_error(sig.data) < 1e-9

    def test_parseval_on_odd_length(self):
        sig = signals.multi_tone()
        assert parseval_rms_error(sig.data.iloc[:-1]) < 1e-9

    def test_impulse_spectrum_is_flat(self):
        sig = signals.impulse(amp=1.0)
        spectrum = compute_spectrum(sig.data, window="rectangular")
        mags = spectrum["signal"].to_numpy()[1:-1]  # skip half-height DC/Nyquist
        expected = 2.0 / len(sig.data)
        npt.assert_allclose(mags, expected, rtol=1e-9)

    def test_chirp_energy_stays_in_band(self):
        sig = signals.linear_chirp(f0=20.0, f1=400.0)
        spectrum = compute_spectrum(sig.data, window="hann")
        power = spectrum["signal"].to_numpy() ** 2
        freqs = spectrum.index.to_numpy()
        margin = 10.0  # Hz of leakage allowance at the sweep edges
        in_band = (freqs >= 20.0 - margin) & (freqs <= 400.0 + margin)
        assert power[in_band].sum() / power.sum() > 0.99


class TestPeakDetection:

    def test_minus_40dbc_harmonic_is_detected(self):
        # regression: a 1% (-40 dBc) line sat exactly on the old -40 dB
        # prominence default and was dropped
        sig = signals.distorted_tone(harmonics={2: 0.02, 3: 0.01})
        spectrum = compute_spectrum(sig.data, window="hann")
        peaks = find_spectral_peaks(spectrum)
        result = compare_to_expected(peaks, sig.expected_peaks)
        assert result["pass"].all(), result.to_string()

    def test_noise_produces_no_false_ground_truth_peaks(self):
        sig = signals.multi_tone(noise_rms=0.2)
        spectrum = compute_spectrum(sig.data, window="hann")
        peaks = find_spectral_peaks(spectrum, n_peaks=3)
        result = compare_to_expected(peaks, sig.expected_peaks)
        assert result["pass"].all(), result.to_string()

    def test_missing_peak_fails_comparison(self):
        sig = signals.single_tone(freq=100.0)
        spectrum = compute_spectrum(sig.data, window="hann")
        peaks = find_spectral_peaks(spectrum, n_peaks=1)
        result = compare_to_expected(peaks, [(300.0, 1.0)])
        assert not result["pass"].any()

    def test_empty_spectrum_yields_empty_peaks(self):
        import pandas as pd
        df = pd.DataFrame({"signal": np.zeros(256)},
                          index=pd.Index(np.arange(256) / 256.0,
                                         name="time (s)"))
        peaks = find_spectral_peaks(compute_spectrum(df))
        assert len(peaks) == 0


class TestValidation:

    def test_invalid_window_raises(self):
        sig = signals.single_tone()
        with pytest.raises(ValueError):
            compute_spectrum(sig.data, window="kaiser-bessel-deluxe")

    def test_invalid_detrend_raises(self):
        sig = signals.single_tone()
        with pytest.raises(ValueError):
            compute_spectrum(sig.data, detrend="cubic")

    @pytest.mark.parametrize("pad", [0, -1, 1.5, "2"])
    def test_invalid_pad_factor_raises(self, pad):
        sig = signals.single_tone()
        with pytest.raises(ValueError):
            compute_spectrum(sig.data, pad_factor=pad)


class TestSampleData:

    def test_committed_csv_matches_its_ground_truth(self):
        import pathlib
        import pandas as pd
        path = (pathlib.Path(__file__).parents[2] / "fft_analyser"
                / "sample_data" / "three_tone_noisy.csv")
        df = pd.read_csv(path).set_index("time (s)")
        spectrum = compute_spectrum(df, window="hann")
        peaks = find_spectral_peaks(spectrum, n_peaks=3)
        result = compare_to_expected(
            peaks, [(50.0, 1.0), (120.0, 0.5), (240.0, 0.25)],
            freq_tolerance=1.0, amp_tolerance_pct=5.0)
        assert result["pass"].all(), result.to_string()


class TestAppSmoke:

    def test_create_app_builds(self):
        pytest.importorskip("dash")
        from fft_analyser.app import create_app
        app = create_app()
        assert app.layout is not None
        # csv upload, panel visibility, bin upload, cloud connect, cloud fetch,
        # bank refresh, monitor select, live config, live poll, main analysis
        assert len(app.callback_map) == 10
