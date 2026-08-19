"""
Decoder for pureMEMS / MLT time-waveform ``.bin`` files.

The format (per the vendor's reference decoder and "pureMEMS binary file
decode" note) is a bit stream of variable-length prefix-coded, delta-
compressed samples:

===========  ==========  ==============================================
prefix       data bits   meaning
===========  ==========  ==============================================
``0``        6           delta; first data bit ``1`` means negative
``10``       6           delta; first data bit ``0`` means negative
``110``      7           delta; first data bit ``0`` means negative
``1110``     16          delta; first data bit ``1`` means negative
``11110``    16          temperature (value/16 °C), not delta-coded
``11111``    -           end of stream
===========  ==========  ==============================================

The header is two prefix-coded unsigned integers - the ADC sampling
frequency and the resampling frequency - followed by 2 bits of axis
selection (``00`` XYZ interleaved, ``01`` X, ``10`` Y, ``11`` Z).  **The
resampling frequency is the true sample rate of the decoded waveform**;
the sampling frequency is the raw ADC rate before decimation.  When the
resampling frequency is 0 the two are the same.

Deltas accumulate in int16 with wraparound.  Counts convert to
acceleration in m/s² by ``value * 16 * 9.80665 / 0x8000`` (±16 g full
scale over a 16-bit signed range).

This implementation is a straight port of the vendor's reference script,
rewritten to walk the bit stream with an index instead of re-slicing it,
which turns an O(n²) decode into a linear one (~7.7 s -> ~0.2 s for a
5 s three-axis record).  ``tests/fft_analyser/test_pure_mems.py`` asserts
it reproduces the reference output exactly on the vendor's own sample
files.
"""

from __future__ import annotations

from dataclasses import dataclass
import typing

import numpy as np
import pandas as pd

__all__ = ["MEMSWaveform", "decode_bin", "decode_bin_file", "G_SCALE"]

#: counts -> m/s² (±16 g full scale across a signed 16-bit range)
G_SCALE = 16.0 * 9.80665 / 0x8000

#: prefix -> (data bits, bit value that marks a negative number)
_PREFIX = {
    "0":     (6, 1),
    "10":    (6, 0),
    "110":   (7, 0),
    "1110":  (16, 1),
    "11110": (16, 1),   # temperature
    "11111": (-1, 0),   # end of stream
}

_AXIS = {"00": "XYZ", "01": "X", "10": "Y", "11": "Z"}


@dataclass
class MEMSWaveform:
    """A decoded waveform plus the metadata needed to analyse it."""

    data: pd.DataFrame          #: time-indexed (seconds), m/s², one column per axis
    fs: float                   #: true sample rate (the resampling frequency), Hz
    sampling_freq: int          #: raw ADC rate before decimation, Hz
    resampling_freq: int        #: decimated rate, Hz
    axis: str                   #: "XYZ", "X", "Y" or "Z"
    temperature_c: np.ndarray   #: temperature samples (°C), recorded at ~8 Hz
    truncated: bool = False     #: stream ended without an end-of-file marker

    @property
    def duration(self) -> float:
        return len(self.data) / self.fs

    def __repr__(self) -> str:
        return (f"MEMSWaveform(axis={self.axis!r}, fs={self.fs:g} Hz, "
                f"n={len(self.data)}, duration={self.duration:.3f} s"
                + (", truncated" if self.truncated else "") + ")")


class _BitReader:
    """
    Sequential bit reader over a byte string.

    The bits are held as one immutable ``"0101..."`` string and read by
    index.  Slicing at a fixed offset costs only the field width, whereas
    the reference decoder re-slices the *whole* remaining stream after
    every field, which is what makes it quadratic.
    """

    __slots__ = ("s", "pos", "n")

    def __init__(self, raw: bytes):
        bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))
        self.s = bits.astype("U1").tobytes().decode("utf-32-le")
        self.pos = 0
        self.n = len(self.s)

    def read_prefix(self) -> typing.Optional[str]:
        """Read up to 5 bits, stopping at the first 0. None if out of data."""
        start = self.pos
        for i in range(5):
            if start + i >= self.n:
                self.pos = self.n
                return None
            if self.s[start + i] == "0":
                self.pos = start + i + 1
                return self.s[start:self.pos]
        self.pos = start + 5
        return "11111"  # five 1s: end of stream

    def read_uint(self, nbits: int) -> typing.Optional[int]:
        if self.pos + nbits > self.n:
            return None
        chunk = self.s[self.pos:self.pos + nbits]
        self.pos += nbits
        return int(chunk, 2)

    def peek_bit(self) -> int:
        return 1 if self.pos < self.n and self.s[self.pos] == "1" else 0


def _read_value(reader: _BitReader, nbits: int, neg_marker: int
                ) -> typing.Optional[int]:
    """Read an nbits field, applying the format's sign convention."""
    if reader.pos + nbits > reader.n:
        return None
    sign_bit = reader.peek_bit()
    value = reader.read_uint(nbits)
    if value is None:
        return None
    if sign_bit == neg_marker:
        value -= 1 << nbits
    return value


def decode_bin(raw: bytes) -> MEMSWaveform:
    """
    Decode a pureMEMS ``.bin`` waveform.

    :param raw: the file contents
    :return: a :py:class:`MEMSWaveform` with acceleration in m/s²
    """
    reader = _BitReader(raw)

    # ── header: two prefix-coded unsigned ints, then 2 axis bits ──
    header = []
    for _ in range(2):
        prefix = reader.read_prefix()
        if prefix is None or _PREFIX[prefix][0] < 0:
            raise ValueError("truncated header in pureMEMS binary")
        nbits = _PREFIX[prefix][0]
        value = reader.read_uint(nbits)
        if value is None:
            raise ValueError("truncated header in pureMEMS binary")
        header.append(value)
    sampling_freq, resampling_freq = header
    if resampling_freq == 0:
        resampling_freq = sampling_freq

    axis_bits = reader.s[reader.pos:reader.pos + 2]
    reader.pos += 2
    axis = _AXIS.get(axis_bits)
    if axis is None:
        raise ValueError(f"unknown axis code {axis_bits!r}")

    n_axes = 3 if axis == "XYZ" else 1
    channels: list[list[float]] = [[] for _ in range(n_axes)]
    prev = [0] * n_axes
    temperature: list[float] = []
    cur_axis = 0
    truncated = False

    while True:
        prefix = reader.read_prefix()
        if prefix is None:
            truncated = True
            break
        nbits, neg_marker = _PREFIX[prefix]
        if nbits < 0:                      # clean end-of-stream marker
            break
        value = _read_value(reader, nbits, neg_marker)
        if value is None:                  # ran out mid-field
            truncated = True
            break

        if prefix == "11110":              # temperature, absolute
            temperature.append(value / 16.0)
            continue

        if not channels[cur_axis]:         # first sample of a channel is absolute
            acc = value
        else:                              # the rest are int16-wrapping deltas
            acc = value + prev[cur_axis]
            if acc > 32767:
                acc -= 65536
            elif acc < -32768:
                acc += 65536
        channels[cur_axis].append(acc * G_SCALE)
        prev[cur_axis] = acc
        if n_axes == 3:
            cur_axis = (cur_axis + 1) % 3

    # a truncated three-axis record can end mid-triplet: keep whole samples only
    if n_axes == 3:
        keep = min(len(c) for c in channels)
        channels = [c[:keep] for c in channels]

    names = ["X", "Y", "Z"] if axis == "XYZ" else [axis]
    n = len(channels[0])
    index = pd.Index(np.arange(n) / resampling_freq, name="time (s)")
    df = pd.DataFrame({nm: np.asarray(ch, dtype=float)
                       for nm, ch in zip(names, channels)}, index=index)

    return MEMSWaveform(
        data=df,
        fs=float(resampling_freq),
        sampling_freq=int(sampling_freq),
        resampling_freq=int(resampling_freq),
        axis=axis,
        temperature_c=np.asarray(temperature, dtype=float),
        truncated=truncated,
    )


def decode_bin_file(path) -> MEMSWaveform:
    """Decode a pureMEMS ``.bin`` waveform from disk."""
    with open(path, "rb") as fh:
        return decode_bin(fh.read())
