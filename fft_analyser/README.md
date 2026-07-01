# FFT Analyser

An interactive FFT analyser built on [`endaq.calc.fft`](../endaq/calc/fft.py),
with built-in test signals whose ground-truth spectra are overlaid on the
measured results — so you can *see* (and pytest can *prove*) that the
algorithm is correct.

![FFT Analyser UI](docs/screenshot.png)

## Running the UI

```bash
pip install endaq dash        # dash is the only extra dependency
python -m fft_analyser        # then open http://127.0.0.1:8050
```

Options: `--host`, `--port`, `--debug`.

## What's in the UI

- **Source** — built-in test signals (with an adjustable added-noise level),
  or upload your own CSV (first column: time in seconds; remaining columns:
  channels). A ready-made example lives at
  [`sample_data/three_tone_noisy.csv`](sample_data/three_tone_noisy.csv)
  (tones of amplitude 1.0/0.5/0.25 at 50/120/240 Hz plus noise).
- **Analysis controls** — window (rectangular/hann/hamming/blackman/flattop),
  detrend (none/mean/linear), zero-padding (1–8×), linear/dB amplitude,
  linear/log frequency axis.
- **Verification** — the spectrum plot overlays detected peaks (●) and the
  signal's ground-truth lines (✕); the table underneath scores each expected
  line against the nearest detected peak (pass = within 1 Hz and 5 %
  amplitude), and a stat tile reports the Parseval energy-conservation error
  (≈1e-16 for a correct pipeline).

## Test signals and what they prove

| Signal | Ground truth | What it verifies |
|---|---|---|
| Multi-tone | lines at 50/120/240 Hz, amplitudes 1.0/0.5/0.25 | frequency & amplitude accuracy |
| Single tone | one known line | basic scaling |
| DC + tone | 0 Hz bin = offset, line at 100 Hz | DC-bin scaling, detrend |
| Square wave | odd harmonics at 4A/πk | multi-line accuracy across a decade |
| Impulse | flat spectrum at 2A/N | broadband flatness |
| Linear chirp | energy confined to 20–400 Hz | band leakage |
| White noise | featureless floor | false-peak rejection |

## The analysis pipeline (`fft_analyser.analysis`)

`compute_spectrum()` wraps `endaq.calc.fft.rfft` and adds the corrections a
measurement-grade analyser needs:

1. **Window amplitude correction** — divides by the window's coherent gain,
   so a sinusoid of amplitude A peaks at A under every window.
2. **DC/Nyquist bin correction** — the library's "unit" normalization doubles
   every bin, but the 0 Hz bin (and the Nyquist bin for even lengths) has no
   negative-frequency twin; without this correction a DC offset of 2 reads
   as 4.
3. **Zero-padding amplitude correction** — padding refines the frequency grid
   without attenuating amplitudes.
4. **Sub-bin peak interpolation** — parabolic interpolation on dB magnitudes
   estimates peak frequency/amplitude well below the bin spacing
   (`find_spectral_peaks()`).

Validation helpers: `compare_to_expected()` (ground-truth scoring, used by
both the UI table and the tests) and `parseval_rms_error()` (energy
conservation).

## Tests

```bash
python -m pytest tests/fft_analyser
```

The suite generates each test signal, runs the full pipeline, and asserts
the ground truth is recovered: exact amplitudes for bin-centered tones under
all five windows, <1 % amplitude error for the worst-case off-bin tone
(flattop + 4× padding), square-wave harmonic series within 1 %, flat impulse
spectrum, chirp energy confinement, DC/Nyquist scaling, Parseval < 1e-9, and
error handling for bad parameters.

## Library fixes made alongside this module

- `endaq.calc.fft.dct()` and `dst()` were swapped — each called the other's
  SciPy transform. Fixed, with regression tests in
  `tests/calc/test_fft.py::TestDCTDST`.
- `endaq.calc.fft.fft()` docstring recommended itself for real input where it
  meant `rfft()`.
