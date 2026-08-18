"""
fft_analyser - a measurement-grade FFT analyser for the endaq-python
foundation, following professional dynamic-signal-analyser conventions
(scaling per Heinzel 2002 / scipy.signal.welch; windows per Harris 1978;
tone metrics per IEEE 1241 / audio-analyzer practice).

The package has four parts:

- :py:mod:`fft_analyser.signals` - deterministic test-signal generators,
  each carrying the ground-truth spectral content it should produce
  (spectral lines, PSD noise floors, THD/ENOB values).
- :py:mod:`fft_analyser.analysis` - the spectrum engine: five scaled
  quantities (peak/RMS amplitude, power spectrum, PSD, ASD) with proper
  S1/S2 window normalization and ENBW, Welch-style overlap averaging
  (linear / peak-hold), spectrogram, band RMS, peak detection and
  ground-truth validation.
- :py:mod:`fft_analyser.metrics` - single-tone quality metrics: THD,
  THD+N, SNR, SINAD, ENOB, SFDR, crest factor.
- :py:mod:`fft_analyser.app` - a Dash UI exposing the full front panel
  (requires the optional ``dash`` dependency; run ``python -m fft_analyser``).
"""

from . import signals
from . import analysis
from . import metrics

from .signals import TestSignal, TEST_SIGNALS
from .analysis import (
    WINDOWS,
    QUANTITIES,
    window_figures,
    SpectrumResult,
    analyze,
    spectrogram,
    band_rms,
    compute_spectrum,
    find_spectral_peaks,
    compare_to_expected,
    parseval_rms_error,
)
from .metrics import ToneMetrics, tone_metrics

__version__ = "0.2.0"


def run_app(**kwargs):
    """Launch the Dash UI (imported lazily so ``dash`` stays optional)."""
    from .app import create_app

    create_app().run(**kwargs)
