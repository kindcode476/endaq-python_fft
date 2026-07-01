import numpy as np
import numpy.testing as npt
import pytest

from fft_analyser import signals


class TestGenerators:

    @pytest.mark.parametrize("factory", list(signals.TEST_SIGNALS.values()))
    def test_registry_signals_are_well_formed(self, factory):
        sig = factory()
        assert len(sig.data) > 0
        assert sig.data.index.name == "time (s)"
        assert sig.fs > 0
        # index spacing must match the declared sample rate
        npt.assert_allclose(np.diff(sig.data.index.to_numpy()), 1.0 / sig.fs,
                            rtol=1e-9)
        # ground truth must be below Nyquist
        for f, a in sig.expected_peaks:
            assert 0 <= f < sig.fs / 2
            assert a > 0

    def test_multi_tone_time_domain_matches_ground_truth(self):
        sig = signals.multi_tone(freqs=(10.0,), amps=(2.0,), fs=1000.0,
                                 duration=1.0)
        t = sig.data.index.to_numpy()
        npt.assert_allclose(sig.data["signal"].to_numpy(),
                            2.0 * np.sin(2 * np.pi * 10.0 * t), atol=1e-12)
        assert sig.expected_peaks == [(10.0, 2.0)]

    def test_multi_tone_rejects_mismatched_args(self):
        with pytest.raises(ValueError):
            signals.multi_tone(freqs=(10, 20), amps=(1.0,))

    def test_multi_tone_rejects_tone_at_nyquist(self):
        with pytest.raises(ValueError):
            signals.multi_tone(freqs=(500.0,), amps=(1.0,), fs=1000.0)

    def test_square_wave_harmonic_series(self):
        sig = signals.square_wave(freq=50.0, amp=2.0, n_harmonics=3)
        expected = [(50.0, 8 / np.pi), (150.0, 8 / (3 * np.pi)),
                    (250.0, 8 / (5 * np.pi))]
        for (f, a), (f_exp, a_exp) in zip(sig.expected_peaks, expected):
            npt.assert_allclose([f, a], [f_exp, a_exp])

    def test_square_wave_truncates_harmonics_at_nyquist(self):
        sig = signals.square_wave(freq=300.0, fs=2048.0, n_harmonics=10)
        assert all(f < 1024.0 for f, _ in sig.expected_peaks)

    def test_noise_is_deterministic(self):
        a = signals.white_noise(seed=7).data
        b = signals.white_noise(seed=7).data
        npt.assert_array_equal(a.to_numpy(), b.to_numpy())
