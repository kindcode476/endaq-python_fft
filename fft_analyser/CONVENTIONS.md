# Conventions of a measurement-grade FFT analyser

This module follows the conventions established in the canonical spectral-
estimation literature and implemented by professional dynamic-signal /
audio analysers (SRS SR785-class instruments, Audio Precision APx, Dewesoft,
Brüel & Kjær) and by SciPy's spectral functions. This document states each
convention, the formula used, and where it comes from. Every numbered
convention is pinned by an automated test in `tests/fft_analyser/`.

## 1. Scaling: the S1/S2 window-sum normalization

With a window `w` of length N, define

```
S1 = sum(w[n])          S2 = sum(w[n]²)
```

and let `X[k]` be the raw DFT of the windowed segment. The single-sided
quantities are (Heinzel, Rüdiger & Schilling 2002, §5; identical to
`scipy.signal.welch`'s `spectrum` / `density` scalings):

| Quantity | Formula | Units | Read from it |
|---|---|---|---|
| Power spectrum (PS) | `c_k·|X[k]|²/S1²` | u² RMS | sinusoids |
| Power spectral density (PSD) | `c_k·|X[k]|²/(fs·S2)` | u²/Hz | broadband noise |
| Amplitude (RMS) | `√PS` | u RMS | sinusoids |
| Amplitude (peak) | `√(2·PS)` (interior bins) | u | sinusoids |
| Amplitude spectral density (ASD) | `√PSD` | u/√Hz | noise |

with `c_k = 2` for interior bins and `c_k = 1` at DC and (for even FFT
lengths) Nyquist — those bins have no negative-frequency twin and must not
be doubled.

**Equivalent noise bandwidth** ties the two families together:

```
ENBW = fs·S2/S1²  [Hz]      NENBW = N·S2/S1²  [bins]      PS = PSD × ENBW
```

The professional reading rules this scaling encodes (Heinzel 2002; Audio
Precision "FFT Scaling for Noise"; Dewesoft FFT guide):

- **Sinusoid levels are read from amplitude/power spectra.** After
  coherent-gain (S1) correction their height is independent of window and
  FFT length. (Verified: a unit tone reads 1.000 under all seven windows.)
- **Noise levels are read from PSD/ASD.** After S2 normalization the density
  floor is independent of window and FFT length. (Verified: white noise of
  variance σ² reads `2σ²/fs` under rectangular, hann and flattop windows and
  at every segment length.)
- Reading a sinusoid off a PSD gives the resolution-dependent value
  `a_rms²/ENBW` — the UI will happily demonstrate this trap.

## 2. Windows and their figures of merit

Figures of merit per Harris (1978) and Heinzel (2002) Table 2 — NENBW and
scalloping loss are *computed from the coefficients* at runtime and verified
against the published values in the test suite:

| Window | NENBW (bins) | Max sidelobe | Scalloping loss | Recommended overlap | Use |
|---|---|---|---|---|---|
| rectangular | 1.0000 | −13.3 dB | −3.92 dB | 0 % | bin-exact tones, full transients |
| hann | 1.5000 | −31.5 dB | −1.42 dB | 50 % | general purpose default |
| hamming | 1.3628 | −42.7 dB | −1.75 dB | 50 % | close tones of similar level |
| blackman | 1.7268 | −58.1 dB | −1.10 dB | ~62 % | moderate dynamic range |
| blackman-harris | 2.0044 | −92.0 dB | −0.83 dB | 66.1 % | high dynamic range |
| flattop | 3.7702 | −93.0 dB | −0.01 dB | ~76 % | amplitude calibration |
| kaiser (β=14) | 2.1611 | −105.9 dB | −0.71 dB | ~70 % | very high dynamic range |

The UI displays the live NENBW/ENBW, sidelobe, scalloping and recommended
overlap for the selected window, as bench analysers do.

## 3. Averaging (Welch's method)

- Data is segmented with fractional **overlap** (default = the window's
  recommended overlap) and the **power** spectra of the segments are
  combined — never the magnitudes (SRS "About FFT Spectrum Analyzers").
- **Linear (RMS) averaging** = mean of segment power spectra: reduces the
  variance of the estimate ~1/√M without lowering the actual noise floor.
  (Verified: relative PSD scatter shrinks by ~1/√M.)
- **Peak hold** = bin-wise maximum across segments.
- 50 % overlap with hann recovers nearly all information; wide-main-lobe
  windows profit from more (Welch 1967; Heinzel 2002 §9; SciPy notes).

## 4. Zero-padding

Padding to `pad_factor × nperseg` interpolates the spectral grid. The S1/S2
normalization is computed on the *unpadded* window, so levels stay correct;
true resolution is still set by the segment length. Displayed bin width Δf
reflects the padded grid; ENBW does not change.

## 5. Single-tone quality metrics

Computed by integrating the PSD over the analysis window's main lobe around
each component (so off-bin leakage is fully captured), using a
blackman-harris window so window sidelobes don't masquerade as distortion —
the approach of audio analysers and MATLAB's `thd()`:

| Metric | Definition | Reference |
|---|---|---|
| THD | √(ΣP_harmonics)/√P_fund | Audio Precision conventions |
| THD+N | √(P_total−P_fund−P_DC)/√P_fund | Audio Precision |
| SNR | P_fund/(P_total−P_fund−ΣP_harm−P_DC), dB | IEEE 1057/1241 |
| SINAD | P_fund/(P_total−P_fund−P_DC), dB (= −THD+N in dB) | IEEE 1241 |
| ENOB | (SINAD_dB − 1.76)/6.02 | IEEE 1241 (full-scale sine) |
| SFDR | fundamental vs. strongest spur (harmonic or not), dBc | ADC/RF practice |
| Crest factor | |peak| / RMS | vibration practice |

Verified against constructions with analytical answers: a 2 %+1 % harmonic
mix reads THD = √5 % = 2.2361 %; an ideal b-bit quantized sine reads
ENOB ≈ b (6.02b+1.76 dB SINAD); a −60 dBc spur reads SFDR = 60 dBc; a tone
in σ=0.1 noise reads SNR = 16.99 dB.

## 6. Other conventions carried by the module

- **Detrending** ("none" / "mean" / "linear") is applied per segment before
  windowing, as in `scipy.signal.welch`; DC-coupled measurements keep the
  0 Hz bin honest (`c_0 = 1`, never doubled).
- **Band power cursor**: RMS in a band = `√(∫ PSD df)` — the standard band
  cursor readout. (Verified: integrating a unit tone's band returns
  0.7071; the full band returns the time-domain RMS.)
- **Parseval self-check**: rectangular-window PSD integrates to the
  time-domain mean square to ~1e-15 relative error.
- **Sub-bin peak estimation**: parabolic interpolation on dB magnitudes of
  the three bins around a maximum (standard analyser marker refinement).
- **Spectrogram** uses the same PSD scaling per segment, so its levels agree
  with the averaged spectrum.

## References

- G. Heinzel, A. Rüdiger, R. Schilling, *Spectrum and spectral density
  estimation by the Discrete Fourier transform (DFT), including a
  comprehensive list of window functions and some new flat-top windows*,
  Max-Planck-Institut für Gravitationsphysik (Albert-Einstein-Institut),
  Hannover, 2002 — the canonical scaling/window whitepaper.
- F. J. Harris, *On the Use of Windows for Harmonic Analysis with the
  Discrete Fourier Transform*, Proc. IEEE 66(1), 1978 — window figures of
  merit.
- P. D. Welch, *The Use of Fast Fourier Transform for the Estimation of
  Power Spectra*, IEEE Trans. Audio Electroacoust. AU-15, 1967 — overlap
  averaging.
- Stanford Research Systems, *About FFT Spectrum Analyzers* (App. Note) —
  instrument averaging/overlap/window conventions.
- Audio Precision, *FFT Spectrum and Spectral Densities — Same Data,
  Different Scaling* and *FFT Scaling for Noise* — spectrum-vs-density
  reading rules.
- IEEE Std 1057/1241 — SINAD/ENOB definitions for digitizer testing.
- D'Antona & Ferrero, *Digital Signal Processing for Measurement Systems*,
  Springer 2006 — the flat-top window family (as implemented by
  `scipy.signal.windows.flattop`).
- `scipy.signal.welch` / `scipy.signal.windows` — the open-source reference
  implementation of the same S1/S2 scalings, against which this module's
  conventions were cross-checked.
