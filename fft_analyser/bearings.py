"""
Rolling-element bearing defect frequencies.

The four kinematic fault frequencies of a rolling-element bearing follow
from its geometry and the shaft speed (standard formulas; see e.g. the
cbmWorx brief §3, SKF/Timken references):

    r    = (d/D)·cos(θ)
    FTF  = s/2 · (1 − r)          cage (fundamental train) frequency
    BPFO = n·s/2 · (1 − r)        ball-pass frequency, outer race
    BPFI = n·s/2 · (1 + r)        ball-pass frequency, inner race
    BSF  = D/(2d)·s · (1 − r²)    ball (roller) spin frequency

with ``n`` rolling elements, roller diameter ``d``, pitch diameter ``D``,
contact angle ``θ`` and shaft speed ``s`` in Hz.  Useful identities the
tests pin: ``BPFO = n·FTF`` and ``BPFO + BPFI = n·s``.

``PRESETS`` carries *nominal* geometry for a handful of very common
deep-groove bearings so an analyst can start immediately; nominal geometry
varies slightly between manufacturers and vintages, so for diagnostic
decisions verify against the manufacturer's datasheet or enter the
geometry directly.  (A full cross-referenced bearing database lives in the
condition-monitoring platform, not in this module.)
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import typing

__all__ = ["BearingGeometry", "bearing_frequencies", "PRESETS"]


@dataclass(frozen=True)
class BearingGeometry:
    """Nominal kinematic geometry of a rolling-element bearing."""
    n_rollers: int
    d_roller_mm: float  #: ball / roller diameter
    d_pitch_mm: float  #: pitch (cage) diameter
    contact_angle_deg: float = 0.0

    def __post_init__(self):
        if self.n_rollers < 2:
            raise ValueError("a bearing needs at least 2 rolling elements")
        if not 0 < self.d_roller_mm < self.d_pitch_mm:
            raise ValueError("roller diameter must be positive and smaller "
                             "than the pitch diameter")
        if not 0 <= self.contact_angle_deg < 90:
            raise ValueError("contact angle must be in [0, 90) degrees")


def bearing_frequencies(geometry: BearingGeometry,
                        speed_hz: float) -> typing.Dict[str, float]:
    """The four defect frequencies in Hz for a shaft speed in Hz."""
    if speed_hz <= 0:
        raise ValueError("speed_hz must be positive")
    r = (geometry.d_roller_mm / geometry.d_pitch_mm) \
        * math.cos(math.radians(geometry.contact_angle_deg))
    ftf = 0.5 * speed_hz * (1.0 - r)
    return {
        "FTF": ftf,
        "BPFO": geometry.n_rollers * ftf,
        "BPFI": geometry.n_rollers * 0.5 * speed_hz * (1.0 + r),
        "BSF": (geometry.d_pitch_mm / (2.0 * geometry.d_roller_mm))
               * speed_hz * (1.0 - r * r),
    }


#: Nominal geometry for common deep-groove ball bearings (contact angle 0).
#: NOMINAL values - verify against the manufacturer's data before making
#: diagnostic decisions; equivalents exist across OEMs (SKF/NTN/FAG/...).
PRESETS: typing.Dict[str, BearingGeometry] = {
    "6203": BearingGeometry(8, 6.75, 28.5),
    "6205": BearingGeometry(9, 7.94, 38.5),
    "6206": BearingGeometry(9, 9.53, 46.0),
    "6207": BearingGeometry(9, 11.11, 53.5),
    "6309": BearingGeometry(8, 17.46, 72.5),
    "6310": BearingGeometry(8, 19.05, 81.5),
}
