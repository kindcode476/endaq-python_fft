"""
End-to-end diagnostic scenario from the cbmWorx brief's analyst workflow:
a 1480 RPM machine with imbalance (1x), misalignment (2x) and a seeded
outer-race bearing defect (6310, BPFO impact train ringing a 3 kHz
resonance) - every planted fault carries analytic ground truth, and the
analyser must recover all of them at the right frequency and level.

This is the whole toolchain in one test: velocity/displacement
integration, Vel RMS, peak detection, and the bearing-frequency
calculator aligning with the measured bearing signature.
"""

import numpy as np
import numpy.testing as npt
import pandas as pd

from fft_analyser.analysis import (
    analyze,
    find_spectral_peaks,
    velocity_band_rms,
)
from fft_analyser.bearings import PRESETS, bearing_frequencies

FS = 12800.0
N = 64000  # 5 s, matching the pureMEMS uploads
SPEED_HZ = 1480.0 / 60.0  # the brief's example machine
V1_MM_S = 2.0   # imbalance severity at 1x, mm/s RMS
V2_MM_S = 1.2   # misalignment at 2x, mm/s RMS
RESONANCE_HZ = 3000.0  # structure rung by each bearing impact
BURST_AMP = 8.0        # m/s² per impact
BURST_TAU = 0.0015     # s


def _accel_for_velocity(v_mm_s_rms: float, freq: float) -> float:
    """Peak acceleration (m/s²) of a tone with the given velocity RMS."""
    return v_mm_s_rms / 1000.0 * (2.0 * np.pi * freq) * np.sqrt(2.0)


def make_scenario():
    bpfo = bearing_frequencies(PRESETS["6310"], SPEED_HZ)["BPFO"]
    t = np.arange(N) / FS
    x = _accel_for_velocity(V1_MM_S, SPEED_HZ) \
        * np.sin(2 * np.pi * SPEED_HZ * t)
    x += _accel_for_velocity(V2_MM_S, 2 * SPEED_HZ) \
        * np.sin(2 * np.pi * 2 * SPEED_HZ * t + 1.0)
    # outer-race impacts: one decaying resonance burst per BPFO period,
    # phase-coherent with the exact (fractional-sample) impact times
    span = int(8 * BURST_TAU * FS)
    for k in range(int(bpfo * N / FS) + 1):
        t_k = k / bpfo
        i0 = int(np.ceil(t_k * FS))
        i1 = min(N, i0 + span)
        if i0 >= N:
            break
        dt = t[i0:i1] - t_k
        x[i0:i1] += BURST_AMP * np.exp(-dt / BURST_TAU) \
            * np.sin(2 * np.pi * RESONANCE_HZ * dt)
    rng = np.random.default_rng(42)
    x += 0.05 * rng.standard_normal(N)
    df = pd.DataFrame({"Z": x}, index=pd.Index(t, name="time (s)"))
    return df, bpfo


DF, BPFO = make_scenario()


class TestRotationalFaults:
    """1x and 2x must read at their planted velocity severities."""

    def _velocity_result(self):
        return analyze(DF, quantity="velocity_rms", window="hann",
                       nperseg=8192, overlap=0.5, averaging="linear",
                       detrend="mean")

    def test_running_speed_lines_recovered(self):
        res = self._velocity_result()
        peaks = find_spectral_peaks(res.spectrum, n_peaks=6)
        f = peaks["frequency (Hz)"].to_numpy()
        a = peaks["amplitude"].to_numpy()

        i1 = int(np.argmin(np.abs(f - SPEED_HZ)))
        assert abs(f[i1] - SPEED_HZ) < 0.5
        npt.assert_allclose(a[i1], V1_MM_S, rtol=0.05)

        i2 = int(np.argmin(np.abs(f - 2 * SPEED_HZ)))
        assert abs(f[i2] - 2 * SPEED_HZ) < 0.5
        npt.assert_allclose(a[i2], V2_MM_S, rtol=0.05)

    def test_flattop_reads_the_1x_acceleration_to_half_percent(self):
        # flattop is the amplitude-calibration window: the raw bin maximum
        # reads the level to ~0.5 % even for an off-bin tone (its top is
        # deliberately flat; parabolic refinement needs curvature and does
        # not apply). The precision claim is made on the ACCELERATION
        # spectrum - in a velocity spectrum at 24.7 Hz the 1/f weighting
        # varies several percent across flattop's wide main lobe, an
        # inherent property of integrated spectra at low frequency.
        res = analyze(DF, quantity="amplitude_peak", window="flattop",
                      nperseg=8192, overlap=0.7, averaging="linear",
                      detrend="mean", pad_factor=2)
        spec = res.spectrum.iloc[:, 0]
        near_1x = spec[np.abs(spec.index.to_numpy() - SPEED_HZ) < 5.0]
        expected_a_pk = _accel_for_velocity(V1_MM_S, SPEED_HZ)
        npt.assert_allclose(near_1x.max(), expected_a_pk, rtol=5e-3)

    def test_displacement_at_1x_matches_analytic(self):
        res = analyze(DF, quantity="displacement_um", window="hann",
                      nperseg=8192, overlap=0.5, averaging="linear",
                      detrend="mean")
        peaks = find_spectral_peaks(res.spectrum, n_peaks=3)
        f = peaks["frequency (Hz)"].to_numpy()
        a = peaks["amplitude"].to_numpy()
        i1 = int(np.argmin(np.abs(f - SPEED_HZ)))
        expected_um = V1_MM_S / (2 * np.pi * SPEED_HZ) * 1000.0
        npt.assert_allclose(a[i1], expected_um, rtol=0.05)

    def test_overall_vel_rms_reflects_the_planted_severities(self):
        res = analyze(DF, quantity="psd", window="hann", nperseg=8192,
                      overlap=0.5, averaging="linear", detrend="mean")
        overall = velocity_band_rms(res, 10.0, 1000.0)
        floor = np.sqrt(V1_MM_S ** 2 + V2_MM_S ** 2)  # 2.332 mm/s
        assert floor * 0.98 < overall < floor * 1.15


class TestBearingSignature:
    """The measured bearing comb must align with the CALCULATED BPFO -
    the bearing-overlay workflow, closed end to end."""

    def test_resonance_band_comb_spacing_equals_bpfo(self):
        res = analyze(DF, quantity="amplitude_peak", window="hann",
                      nperseg=8192, overlap=0.5, averaging="linear",
                      detrend="mean")
        spec = res.spectrum.iloc[:, 0]
        band = spec[(spec.index > RESONANCE_HZ - 500)
                    & (spec.index < RESONANCE_HZ + 500)]
        peaks = find_spectral_peaks(band.to_frame(), n_peaks=10)
        f = np.sort(peaks["frequency (Hz)"].to_numpy())
        spacings = np.diff(f)
        # keep first-neighbour spacings only (a missed peak doubles one)
        spacings = spacings[spacings < 1.5 * BPFO]
        assert len(spacings) >= 5
        npt.assert_allclose(np.median(spacings), BPFO, rtol=0.02)

    def test_calculated_bpfo_matches_the_brief(self):
        # 6310 at 1480 RPM: the brief's workflow expects ~75.6 Hz
        npt.assert_allclose(BPFO, 3.065 * SPEED_HZ, rtol=1e-3)
        assert 75.0 < BPFO < 76.5
