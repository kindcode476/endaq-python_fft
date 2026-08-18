# FFT Analyser

A measurement-grade FFT analyser on the [endaq-python](../README.md)
foundation, following the conventions of professional dynamic-signal
analysers — scaling per Heinzel 2002 / SciPy, windows per Harris 1978, tone
metrics per IEEE 1241 / audio-analyzer practice. The conventions, formulas
and references are documented in [CONVENTIONS.md](CONVENTIONS.md); every one
of them is pinned by an automated test.

Built-in test signals carry their ground truth (spectral lines, PSD noise
floors, THD/ENOB values), which the UI overlays on the measured results and
scores live — so you can *see*, and pytest can *prove*, that the analyser is
correct.

![FFT Analyser UI](docs/screenshot.png)

## Running the UI

```bash
pip install endaq dash        # dash is the only extra dependency
python -m fft_analyser        # then open http://127.0.0.1:8050
```

Options: `--host`, `--port`, `--debug`.

## The front panel

- **Quantity** — peak amplitude, RMS amplitude, power spectrum (u² RMS),
  PSD (u²/Hz) or ASD (u/√Hz), each with correct S1/S2 window normalization
  and DC/Nyquist single-sided handling.
- **Window** — rectangular, hann, hamming, blackman, blackman-harris,
  flattop, kaiser(β=14), with live figures of merit displayed (NENBW/ENBW,
  sidelobe level, scalloping loss, recommended overlap).
- **Averaging** — Welch segmentation with selectable segment length and
  overlap; linear (RMS) averaging or peak hold; averages count shown.
- **Display** — linear/dB amplitude, linear/log frequency, zero-padding
  (1–8×), detrend (none/mean/linear), harmonic markers.
- **Tone metrics** — fundamental, THD, THD+N, SNR, SINAD, ENOB, SFDR,
  RMS (time-domain and from the PSD integral), crest factor.
- **Spectrogram** — short-time PSD in dB with the same scaling.
- **Verification** — ground-truth overlay (✕), detected peaks (●), a
  pass/fail table per expected line, and a Parseval energy self-check tile.
- **Sources** — built-in test signals (with adjustable added noise), or
  upload a CSV (first column: time in seconds; other columns: channels).
  Example: [`sample_data/three_tone_noisy.csv`](sample_data/three_tone_noisy.csv).

## Test signals and what they prove

| Signal | Ground truth | What it verifies |
|---|---|---|
| Multi-tone | lines at 50/120/240 Hz, amps 1.0/0.5/0.25 | frequency & amplitude accuracy |
| Single / two-tone | known lines | scaling; frequency resolution |
| Distorted tone | THD = √(Σr²) = 2.236 % | THD/THD+N/SINAD/SFDR |
| Quantized tone | ENOB ≈ bits (6.02b+1.76 dB SINAD) | SINAD/ENOB |
| DC + tone | 0 Hz bin = offset | DC-bin scaling, detrend |
| Square wave | odd harmonics at 4A/πk | multi-line accuracy, THD ≈ 48.3 % |
| Impulse | flat spectrum at 2A/N | broadband flatness |
| Linear chirp | energy confined to 20–400 Hz | leakage; spectrogram ridge |
| White noise | PSD floor = 2σ²/fs | density scaling, window/length invariance |

## Library API

```python
from fft_analyser import analyze, tone_metrics, band_rms, spectrogram

result = analyze(df, quantity="psd", window="hann",
                 nperseg=4096, overlap=0.5, averaging="linear")
result.spectrum        # DataFrame indexed by frequency
result.enbw_hz         # equivalent noise bandwidth per bin
result.n_segments      # averages taken
band_rms(result, 10, 1000)   # band-limited RMS from the PSD integral
m = tone_metrics(df)         # THD, THD+N, SNR, SINAD, ENOB, SFDR, crest...
```

`compute_spectrum()`, `find_spectral_peaks()`, `compare_to_expected()` and
`parseval_rms_error()` from v0.1 are unchanged.

## Tests

```bash
python -m pytest tests/fft_analyser
```

The suite pins every documented convention: window figures of merit against
the Harris/Heinzel tables, all scaling identities (PS/PSD = ENBW,
peak = √2·RMS, ASD² = PSD), tone-amplitude invariance across windows,
noise-floor invariance across windows and FFT lengths, PSD integral =
signal power, Welch segment counts and 1/√M variance reduction, band RMS,
spectrogram ridge tracking, THD/SNR/SINAD/ENOB/SFDR against analytical
ground truth, DC/Nyquist scaling, Parseval, and error handling.

## Library fixes made alongside this module

- `endaq.calc.fft.dct()` and `dst()` were swapped — each called the other's
  SciPy transform. Fixed, with regression tests in
  `tests/calc/test_fft.py::TestDCTDST`.
- `endaq.calc.fft.fft()` docstring recommended itself for real input where
  it meant `rfft()`.
