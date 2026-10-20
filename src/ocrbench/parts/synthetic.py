"""Draw invented part photos and printable scale markers.

The photos are a bright part on a dark mat next to an ArUco marker, at a
known scale, so tests and demos know the true sizes. They are drawings, not
photos of Kimura's parts: results on them say the code works, not how well
the check works on real parts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Final, Literal

import cv2
import numpy as np

from ocrbench.parts.vision import DEFAULT_ARUCO_DICTIONARY, Image

_MAT: Final[int] = 55
_METAL: Final[int] = 200
_QUIET_ZONE: Final[float] = 0.25
"""White border around the printed marker, as a share of its side."""


@dataclass(frozen=True)
class SyntheticPart:
    """The true shape of an invented part, in millimetres.

    ``round``: a disc seen from above, ``diameter_mm`` across, with
    ``holes`` given as (diameter, distance from centre) pairs.
    ``profile``: a shaft on its side, ``length_mm`` long, ``diameter_mm``
    thick at one end and ``end_diameter_mm`` at the other (a taper when they
    differ).
    """

    view: Literal["round", "profile"]
    diameter_mm: float
    length_mm: float = 0.0
    end_diameter_mm: float | None = None
    holes: tuple[tuple[float, float], ...] = field(default_factory=tuple)


def marker_image(
    *, marker_id: int, side_px: int, dictionary: int = DEFAULT_ARUCO_DICTIONARY
) -> Image:
    """An ArUco marker with its white quiet zone, as a grayscale image."""
    marker = cv2.aruco.generateImageMarker(
        cv2.aruco.getPredefinedDictionary(dictionary), marker_id, side_px
    )
    border = round(side_px * _QUIET_ZONE)
    return np.asarray(
        cv2.copyMakeBorder(marker, border, border, border, border, cv2.BORDER_CONSTANT, value=255),
        dtype=np.uint8,
    )


def printable_marker(*, marker_id: int, marker_mm: float, dpi: int = 300) -> Image:
    """A marker sized so that printing at 100% at ``dpi`` gives ``marker_mm``.

    Measure the printed black square with a ruler before use: printers that
    "fit to page" change the size, and the scale would then be wrong.
    """
    side_px = round(marker_mm / 25.4 * dpi)
    return marker_image(marker_id=marker_id, side_px=side_px)


def render_photo(
    part: SyntheticPart,
    *,
    mm_per_px: float = 0.25,
    marker_mm: float = 50.0,
    marker_id: int = 0,
    angle_deg: float = 0.0,
    size: tuple[int, int] = (1600, 1200),
    noise: float = 4.0,
    seed: int = 0,
) -> Image:
    """A BGR photo of ``part`` with a marker of ``marker_mm`` at ``mm_per_px``."""
    width, height = size
    photo = np.full((height, width), _MAT, dtype=np.uint8)
    marker = marker_image(marker_id=marker_id, side_px=round(marker_mm / mm_per_px))
    photo[40 : 40 + marker.shape[0], 40 : 40 + marker.shape[1]] = marker
    centre = (width * 0.62, height * 0.5)
    if part.view == "round":
        radius = part.diameter_mm / 2 / mm_per_px
        cv2.circle(
            photo, _point(centre), round(radius), _METAL, thickness=cv2.FILLED, lineType=cv2.LINE_AA
        )
        for index, (hole_mm, offset_mm) in enumerate(part.holes):
            theta = math.radians(angle_deg) + index * 2 * math.pi / max(1, len(part.holes))
            spot = (
                centre[0] + offset_mm / mm_per_px * math.cos(theta),
                centre[1] + offset_mm / mm_per_px * math.sin(theta),
            )
            cv2.circle(
                photo, _point(spot), round(hole_mm / 2 / mm_per_px), _MAT, cv2.FILLED, cv2.LINE_AA
            )
    else:
        end = part.end_diameter_mm if part.end_diameter_mm is not None else part.diameter_mm
        half_l = part.length_mm / 2 / mm_per_px
        corners = np.array(
            [
                (-half_l, -part.diameter_mm / 2 / mm_per_px),
                (half_l, -end / 2 / mm_per_px),
                (half_l, end / 2 / mm_per_px),
                (-half_l, part.diameter_mm / 2 / mm_per_px),
            ]
        )
        theta = math.radians(angle_deg)
        rotation = np.array(
            [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]
        )
        placed = corners @ rotation.T + np.array(centre)
        cv2.fillPoly(photo, [np.round(placed).astype(np.int32)], _METAL, lineType=cv2.LINE_AA)
    rng = np.random.default_rng(seed)
    noisy = photo.astype(np.float64) + rng.normal(0, noise, photo.shape)
    blurred = cv2.GaussianBlur(np.clip(noisy, 0, 255).astype(np.uint8), (3, 3), 0)
    return np.asarray(cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR), dtype=np.uint8)


def _point(xy: tuple[float, float]) -> tuple[int, int]:
    return round(xy[0]), round(xy[1])


__all__ = ["SyntheticPart", "marker_image", "printable_marker", "render_photo"]
