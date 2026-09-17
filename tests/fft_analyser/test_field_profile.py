"""The field analysis profile from the client's vibration engineer:
5 Hz integration high-pass, and at least 4-6 Welch averages of the TWF
at 20% overlap ('Twf is long so should be ok')."""

import numpy as np
import numpy.testing as npt
import pandas as pd

from fft_analyser.analysis import (
    VELOCITY_CUTOFF_HZ,
    analyze,
    segment_for_averages,
)
from fft_analyser.monitors import REAL_DATA_DEFAULTS


def _df(x, fs):
    t = np.arange(len(x)) / fs
    return pd.DataFrame({"Z": x}, index=pd.Index(t, name="time (s)"))


class TestFieldHighPass:

    def test_cutoff_is_the_field_spec(self):
        assert VELOCITY_CUTOFF_HZ == 5.0

    def test_low_frequency_rumble_is_removed(self):
        # 3 Hz rumble (the noise the engineer saw) + a 48.5 Hz machine tone
        fs, n = 12800.0, 25600
        t = np.arange(n) / fs
        x = 0.5 * np.sin(2 * np.pi * 3.0 * t) \
            + 1.0 * np.sin(2 * np.pi * 48.5 * t)
        res = analyze(_df(x, fs), quantity="velocity_rms", window="hann",
                      nperseg=8192, overlap=0.2, averaging="linear",
                      detrend="mean")
        spec = res.spectrum.iloc[:, 0]
        f = spec.index.to_numpy(dtype=float)
        # everything below 5 Hz is zeroed - the rumble cannot appear
        assert spec.to_numpy()[f < 5.0].max() == 0.0
        # while the machine tone still reads correctly (~3.28 mm/s RMS)
        expected = 1000.0 / (np.sqrt(2) * 2 * np.pi * 48.5)
        npt.assert_allclose(spec[np.abs(f - 48.5) < 3].max(), expected,
                            rtol=0.05)


class TestAveragesProfile:

    def test_defaults_carry_the_20_percent_overlap(self):
        assert REAL_DATA_DEFAULTS["overlap"] == 0.2

    def test_long_record_keeps_full_segment(self):
        # 5 s upload at 12.8 kHz: 8192 @ 20 % gives 9 averages
        assert segment_for_averages(64000) == 8192

    def test_short_record_shrinks_until_averages_reached(self):
        # 2 s upload: 8192 @ 20 % gives only 3 averages -> halve to 4096
        assert segment_for_averages(25600) == 4096

    def test_tiny_record_falls_back_to_smallest_segment(self):
        assert segment_for_averages(300) == 256

    def test_the_chosen_segment_actually_delivers_the_averages(self):
        for n in (25600, 64000, 128000):
            nperseg = segment_for_averages(n)
            fs = 12800.0
            res = analyze(_df(np.random.default_rng(1).standard_normal(n), fs),
                          nperseg=nperseg, overlap=0.2, averaging="linear")
            assert res.n_segments >= 4, (n, nperseg, res.n_segments)
