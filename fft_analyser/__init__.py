"""
fft_analyser - An interactive FFT analyser built on :py:mod:`endaq.calc.fft`.

The package has three parts:

- :py:mod:`fft_analyser.signals` - deterministic test-signal generators, each
  carrying the ground-truth spectral content it should produce, so the FFT
  pipeline can be verified end-to-end.
- :py:mod:`fft_analyser.analysis` - the analysis pipeline: detrending,
  amplitude-corrected windowing, single-sided spectra via
  :py:func:`endaq.calc.fft.rfft`, peak detection with sub-bin interpolation,
  and validation helpers (ground-truth comparison, Parseval energy check).
- :py:mod:`fft_analyser.app` - a Dash UI wrapping the pipeline (requires the
  optional ``dash`` dependency; run with ``python -m fft_analyser``).
"""

from . import signals
from . import analysis

from .signals import TestSignal, TEST_SIGNALS
from .analysis import (
    WINDOWS,
    compute_spectrum,
    find_spectral_peaks,
    compare_to_expected,
    parseval_rms_error,
)

__version__ = "0.1.0"


def run_app(**kwargs):
    """Launch the Dash UI (imported lazily so ``dash`` stays optional)."""
    from .app import create_app

    create_app().run(**kwargs)
