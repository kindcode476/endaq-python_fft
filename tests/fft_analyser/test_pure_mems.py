"""
Decoder tests against the vendor's own sample waveforms.

The two ``.bin`` files in ``fft_analyser/sample_data/pure_mems/`` ship
with the vendor's decode kit; the expectations below were taken from
running the vendor's reference script over them.
"""

import pathlib

import numpy as np
import numpy.testing as npt
import pytest

from fft_analyser.pure_mems import G_SCALE, decode_bin, decode_bin_file
from fft_analyser.analysis import analyze, band_rms

SAMPLES = pathlib.Path(__file__).parents[2] / "fft_analyser" / "sample_data" / "pure_mems"
ONE_AXIS = SAMPLES / "1axis_z_12800Hz.bin"
THREE_AXIS = SAMPLES / "3axis_xyz_12800Hz.bin"


@pytest.fixture(scope="module")
def wave():
    return decode_bin_file(ONE_AXIS)


@pytest.fixture(scope="module")
def wave3():
    return decode_bin_file(THREE_AXIS)


class TestSingleAxisFile:

    def test_header(self, wave):
        # values produced by the vendor's reference decoder
        assert wave.sampling_freq == 26706
        assert wave.resampling_freq == 12800
        assert wave.fs == 12800.0
        assert wave.axis == "Z"

    def test_sample_count_and_duration(self, wave):
        assert len(wave.data) == 64000
        npt.assert_allclose(wave.duration, 5.0)
        assert list(wave.data.columns) == ["Z"]

    def test_first_samples_match_reference(self, wave):
        expected = [-0.55545, -0.60334, -0.55067, -0.51715, -0.48842]
        npt.assert_allclose(wave.data["Z"].to_numpy()[:5], expected, atol=1e-5)

    def test_time_index_matches_sample_rate(self, wave):
        npt.assert_allclose(np.diff(wave.data.index.to_numpy()), 1 / 12800.0,
                            rtol=1e-9)

    def test_temperature_channel(self, wave):
        assert len(wave.temperature_c) == 41
        npt.assert_allclose(wave.temperature_c[:3], [24.0, 24.0, 24.0])

    def test_not_truncated(self, wave):
        assert wave.truncated is False

    def test_values_are_in_physical_units(self, wave):
        # ±16 g full scale; a healthy machine sits far below that
        peak = np.abs(wave.data["Z"].to_numpy()).max()
        assert 0 < peak < 16 * 9.80665
        # every value must be an exact multiple of the LSB
        counts = wave.data["Z"].to_numpy() / G_SCALE
        npt.assert_allclose(counts, np.round(counts), atol=1e-6)


class TestThreeAxisFile:

    def test_header(self, wave3):
        assert wave3.sampling_freq == 26746
        assert wave3.resampling_freq == 12800
        assert wave3.axis == "XYZ"

    def test_three_equal_length_channels(self, wave3):
        assert list(wave3.data.columns) == ["X", "Y", "Z"]
        assert len(wave3.data) == 63965
        assert wave3.data.notna().all().all()

    def test_first_x_samples_match_reference(self, wave3):
        expected = [-1.90578, -2.03028, -1.94888, -1.92973, -2.04465]
        npt.assert_allclose(wave3.data["X"].to_numpy()[:5], expected, atol=1e-5)

    def test_truncation_is_reported(self, wave3):
        # this file ends without an end-of-stream marker; the vendor's own
        # script prints an indexing warning here and emits ONE extra sample
        # decoded from the partial trailing field. We drop that sample and
        # flag the record instead of inventing a value for it.
        assert wave3.truncated is True


class TestDecoderRobustness:

    def test_rejects_empty_input(self):
        with pytest.raises(ValueError):
            decode_bin(b"")

    def test_rejects_truncated_header(self):
        with pytest.raises(ValueError):
            decode_bin(b"\xff")

    def test_decode_bytes_matches_decode_file(self):
        from_bytes = decode_bin(ONE_AXIS.read_bytes())
        from_file = decode_bin_file(ONE_AXIS)
        npt.assert_array_equal(from_bytes.data.to_numpy(), from_file.data.to_numpy())


class TestSpectrumOfRealData:
    """The analyser must produce sane, self-consistent numbers on real data."""

    def test_parseval_holds_on_real_data(self, wave):
        from fft_analyser.analysis import parseval_rms_error
        assert parseval_rms_error(wave.data) < 1e-9

    def test_band_rms_matches_time_domain_rms(self, wave):
        result = analyze(wave.data, quantity="psd", window="rectangular",
                         averaging="none")
        x = wave.data["Z"].to_numpy()
        npt.assert_allclose(band_rms(result), np.sqrt(np.mean(x ** 2)), rtol=1e-6)

    def test_spectrum_spans_to_nyquist(self, wave):
        result = analyze(wave.data, nperseg=4096)
        npt.assert_allclose(result.freqs_max if hasattr(result, "freqs_max")
                            else result.spectrum.index[-1], 6400.0, rtol=1e-6)

    def test_averaging_reduces_scatter_on_real_data(self, wave):
        single = analyze(wave.data, quantity="psd", nperseg=4096,
                         averaging="none").spectrum["Z"]
        avg = analyze(wave.data, quantity="psd", nperseg=4096,
                      averaging="linear")
        assert avg.n_segments > 5
        s, o = avg.spectrum["Z"].iloc[5:-5], single.iloc[5:-5]
        assert (s.std() / s.mean()) < (o.std() / o.mean())
