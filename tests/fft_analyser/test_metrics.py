"""
Tone-metric tests against analytical ground truth (IEEE 1241 / audio
analyzer conventions).
"""

import numpy as np
import numpy.testing as npt
import pytest

from fft_analyser import signals
from fft_analyser.metrics import tone_metrics


class TestTHD:

    def test_known_harmonic_mix(self):
        # 2% H2 + 1% H3 -> THD = sqrt(0.02^2 + 0.01^2) = 2.2361%
        sig = signals.distorted_tone(harmonics={2: 0.02, 3: 0.01})
        m = tone_metrics(sig.data)
        npt.assert_allclose(m.thd_pct, 100 * np.sqrt(5) / 100, rtol=1e-3)
        npt.assert_allclose(m.fundamental_freq, 100.0, atol=0.05)
        npt.assert_allclose(m.fundamental_peak, 1.0, rtol=1e-3)

    def test_thd_of_clean_tone_is_negligible(self):
        m = tone_metrics(signals.single_tone().data)
        assert m.thd_pct < 1e-6

    def test_square_wave_thd(self):
        # odd harmonics 1/k: THD = sqrt(sum 1/k^2, odd k>=3) ~ 0.4834
        m = tone_metrics(signals.square_wave(fs=65536.0, freq=50.0).data,
                         max_harmonics=40)
        expected = np.sqrt(sum(1.0 / k ** 2 for k in range(3, 81, 2)))
        npt.assert_allclose(m.thd_pct / 100, expected, rtol=0.02)

    def test_sinad_is_reciprocal_of_thd_n(self):
        sig = signals.distorted_tone()
        m = tone_metrics(sig.data)
        npt.assert_allclose(m.sinad_db, -20 * np.log10(m.thd_n_pct / 100),
                            rtol=1e-9)


class TestSNR:

    def test_tone_in_known_noise(self):
        # A=1 pk (P=0.5), sigma=0.1 (P=0.01) -> SNR = 10log10(50) = 16.99 dB
        sig = signals.single_tone(amp=1.0, noise_rms=0.1, duration=4.0)
        m = tone_metrics(sig.data)
        npt.assert_allclose(m.snr_db, 10 * np.log10(0.5 / 0.01), atol=0.3)


class TestENOB:

    @pytest.mark.parametrize("bits", [6, 8, 12])
    def test_ideal_quantizer_enob(self, bits):
        m = tone_metrics(signals.quantized_tone(bits=bits).data)
        npt.assert_allclose(m.enob_bits, bits, atol=0.25)
        npt.assert_allclose(m.sinad_db, 6.02 * bits + 1.76, atol=1.5)


class TestSFDR:

    def test_known_spur(self):
        # spur at -60 dBc, non-harmonic frequency
        sig = signals.multi_tone((100.0, 333.0), (1.0, 1e-3), fs=8192.0,
                                 duration=2.0)
        m = tone_metrics(sig.data)
        npt.assert_allclose(m.sfdr_dbc, 60.0, atol=1.0)

    def test_harmonic_can_be_the_spur(self):
        sig = signals.distorted_tone(harmonics={2: 0.02})  # -34 dBc
        m = tone_metrics(sig.data)
        npt.assert_allclose(m.sfdr_dbc, -20 * np.log10(0.02), atol=0.5)


class TestTimeDomainFigures:

    def test_sine_crest_factor(self):
        m = tone_metrics(signals.single_tone().data)
        npt.assert_allclose(m.crest_factor, np.sqrt(2), rtol=1e-3)
        npt.assert_allclose(m.rms, 1 / np.sqrt(2), rtol=1e-3)

    def test_square_crest_factor(self):
        m = tone_metrics(signals.square_wave().data)
        # sign(sin) hits an exact zero sample now and then, so RMS is a hair
        # below the amplitude
        npt.assert_allclose(m.crest_factor, 1.0, rtol=1e-3)
