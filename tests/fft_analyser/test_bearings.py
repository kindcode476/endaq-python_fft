"""Bearing defect frequencies: identities, published multipliers, guards."""

import numpy as np
import numpy.testing as npt
import pytest

from fft_analyser.bearings import BearingGeometry, bearing_frequencies, PRESETS


class TestKinematicIdentities:

    @pytest.mark.parametrize("name", list(PRESETS))
    def test_bpfo_is_n_times_ftf_and_pair_sums_to_n_speed(self, name):
        g = PRESETS[name]
        f = bearing_frequencies(g, 10.0)
        npt.assert_allclose(f["BPFO"], g.n_rollers * f["FTF"], rtol=1e-12)
        npt.assert_allclose(f["BPFO"] + f["BPFI"],
                            g.n_rollers * 10.0, rtol=1e-12)

    def test_frequencies_scale_linearly_with_speed(self):
        g = PRESETS["6205"]
        f1 = bearing_frequencies(g, 10.0)
        f2 = bearing_frequencies(g, 25.0)
        for k in f1:
            npt.assert_allclose(f2[k] / f1[k], 2.5, rtol=1e-12)

    def test_contact_angle_moves_bpfo_up(self):
        flat = bearing_frequencies(BearingGeometry(9, 8.0, 40.0, 0.0), 10.0)
        angled = bearing_frequencies(BearingGeometry(9, 8.0, 40.0, 30.0), 10.0)
        assert angled["BPFO"] > flat["BPFO"]
        assert angled["BPFI"] < flat["BPFI"]


class TestPublishedMultipliers:
    """Nominal multipliers of shaft speed for the 6205, as published in
    common bearing-frequency tables."""

    def test_6205_multipliers(self):
        f = bearing_frequencies(PRESETS["6205"], 1.0)
        npt.assert_allclose(f["BPFO"], 3.572, atol=0.01)
        npt.assert_allclose(f["BPFI"], 5.428, atol=0.01)
        npt.assert_allclose(f["BSF"], 2.322, atol=0.01)
        npt.assert_allclose(f["FTF"], 0.397, atol=0.005)

    def test_analyst_workflow_case(self):
        # the brief's user journey: 1480 RPM = 24.67 Hz shaft speed
        f = bearing_frequencies(PRESETS["6310"], 1480.0 / 60.0)
        assert 70.0 < f["BPFO"] < 80.0     # ~3.07 x 24.67 = 75.6 Hz
        assert 115.0 < f["BPFI"] < 130.0   # ~4.93 x 24.67 = 121.7 Hz


class TestGuards:

    def test_bad_geometry_raises(self):
        with pytest.raises(ValueError):
            BearingGeometry(1, 8.0, 40.0)
        with pytest.raises(ValueError):
            BearingGeometry(9, 40.0, 8.0)
        with pytest.raises(ValueError):
            BearingGeometry(9, 8.0, 40.0, 95.0)

    def test_bad_speed_raises(self):
        with pytest.raises(ValueError):
            bearing_frequencies(PRESETS["6205"], 0.0)
