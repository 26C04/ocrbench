"""Find the scale, segment the part and measure it in millimetres.

The photo is taken from above, with the part lying on a plain mat next to a
printed ArUco marker of known size. The marker gives millimetres per pixel;
the segmenter gives the part's mask; :func:`measure_mask` turns the mask into
sizes.

Two views are told apart by the outline. A disc or flange lying flat shows a
circle (``round`` view): its diameter is measured, its length cannot be, and
through holes are visible. A shaft lying on its side shows a profile
(``profile`` view): its length and diameter are both measured, holes are not.

Known limits, absorbed by calibration rather than corrected here: the top of
a tall part is closer to the camera and looks slightly larger, and a camera
that is not straight above the mat stretches one direction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, Protocol

import cv2
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

VISION_VERSION: Final[str] = "1"
DEFAULT_ARUCO_DICTIONARY: Final[int] = cv2.aruco.DICT_4X4_50
# Above this circularity, with a near-square bounding box, the outline is a
# disc seen from above.
_ROUND_CIRCULARITY: Final[float] = 0.85
_ROUND_ASPECT: Final[float] = 1.15
# A hole smaller than this share of the part's area is noise, not a hole.
_MIN_HOLE_SHARE: Final[float] = 0.0005

Mask = NDArray[np.bool_]
Image = NDArray[np.uint8]


class VisionError(Exception):
    """Raised when a photo cannot be measured; the message is content-free."""


@dataclass(frozen=True)
class Scale:
    """Millimetres per pixel and where it came from."""

    mm_per_px: float
    source: Literal["marker", "fixed"]
    marker_id: int | None = None
    marker_side_px: float | None = None
    marker_polygon: NDArray[np.float32] | None = None


class HoleMeasurement(BaseModel):
    """One hole seen inside the part's outline."""

    model_config = ConfigDict(extra="forbid")

    diameter_mm: float = Field(gt=0)
    circularity: float = Field(ge=0)


class Measurement(BaseModel):
    """Sizes of the part in one photo, in millimetres."""

    model_config = ConfigDict(extra="forbid")

    view: Literal["round", "profile"]
    long_mm: float = Field(gt=0)
    short_mm: float = Field(gt=0)
    area_mm2: float = Field(gt=0)
    circularity: float = Field(ge=0)
    holes: list[HoleMeasurement] = Field(default_factory=list)
    end_width_ratio: float | None = Field(default=None, ge=0, le=1)
    mm_per_px: float = Field(gt=0)
    scale_source: Literal["marker", "fixed"]
    vision_version: str = VISION_VERSION


class Segmenter(Protocol):
    """Gives the part's mask in a photo; ``exclude`` are polygons to ignore."""

    @property
    def segmenter_id(self) -> str:
        """Stable name recorded with each check."""

    def segment(self, image: Image, *, exclude: list[NDArray[np.float32]]) -> Mask:
        """Return a boolean mask, ``True`` on the part, same height and width as ``image``."""


def load_image(path: Path) -> Image:
    """Read a photo as an 8-bit BGR array."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise VisionError("cannot read the photo")
    return np.asarray(image, dtype=np.uint8)


def _gray(image: Image) -> Image:
    if image.ndim == 2:
        return image
    return np.asarray(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), dtype=np.uint8)


def marker_scale(
    image: Image,
    *,
    marker_mm: float,
    dictionary: int = DEFAULT_ARUCO_DICTIONARY,
    marker_id: int | None = None,
) -> Scale:
    """Find one ArUco marker of side ``marker_mm`` and return the scale.

    With several markers in the photo, ``marker_id`` picks one; without it the
    photo must show exactly one marker, so a stray marker can never set the
    scale silently.
    """
    if marker_mm <= 0:
        raise VisionError("marker size must be positive")
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(dictionary), cv2.aruco.DetectorParameters()
    )
    corners, ids, _rejected = detector.detectMarkers(_gray(image))
    found = [] if ids is None else [int(value) for value in ids.flatten()]
    if marker_id is not None:
        chosen = [index for index, value in enumerate(found) if value == marker_id]
    else:
        chosen = list(range(len(found)))
    if not chosen:
        raise VisionError("no scale marker found in the photo")
    if len(chosen) > 1:
        raise VisionError("more than one scale marker found; pass the marker id")
    polygon = np.asarray(corners[chosen[0]], dtype=np.float32).reshape(4, 2)
    sides = [float(np.linalg.norm(polygon[i] - polygon[(i + 1) % 4])) for i in range(4)]
    side_px = sum(sides) / 4
    return Scale(
        mm_per_px=marker_mm / side_px,
        source="marker",
        marker_id=found[chosen[0]],
        marker_side_px=side_px,
        marker_polygon=polygon,
    )


def fixed_scale(mm_per_px: float) -> Scale:
    """Scale from a fixed camera height, measured once with a ruler."""
    if mm_per_px <= 0:
        raise VisionError("mm per pixel must be positive")
    return Scale(mm_per_px=mm_per_px, source="fixed")


def _largest_component(mask: Image) -> Mask:
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        raise VisionError("no part found in the photo")
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = 1 + int(np.argmax(areas))
    return np.asarray(labels == keep, dtype=np.bool_)


# The printed marker has a white quiet zone around its black square; blank out
# the square grown by this factor so the paper is never taken for the part.
MARKER_EXCLUDE_GROWTH: Final[float] = 1.8


def _fill_polygons(mask: Image, polygons: list[NDArray[np.float32]]) -> None:
    for polygon in polygons:
        centre = polygon.mean(axis=0)
        grown = centre + (polygon - centre) * MARKER_EXCLUDE_GROWTH
        cv2.fillPoly(mask, [np.round(grown).astype(np.int32)], 0)


class ThresholdSegmenter:
    """Otsu threshold for a part on a plain, evenly lit mat.

    Good enough for a fixed station with a contrasting mat, and the reference
    the SAM3 segmenter is compared with. The background is whichever side of
    the threshold covers most of the photo's border.
    """

    def __init__(self, *, blur_px: int = 5, close_px: int = 5) -> None:
        self._blur = blur_px | 1
        self._close = max(1, close_px)

    @property
    def segmenter_id(self) -> str:
        return f"threshold(blur={self._blur},close={self._close})"

    def segment(self, image: Image, *, exclude: list[NDArray[np.float32]]) -> Mask:
        gray = cv2.GaussianBlur(_gray(image), (self._blur, self._blur), 0)
        _threshold, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary: Image = np.asarray(otsu, dtype=np.uint8)
        border = np.concatenate([binary[0], binary[-1], binary[:, 0], binary[:, -1]])
        if np.count_nonzero(border) > border.size / 2:
            binary = np.asarray(cv2.bitwise_not(binary), dtype=np.uint8)
        _fill_polygons(binary, exclude)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self._close, self._close))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
        return _largest_component(np.asarray(opened, dtype=np.uint8))


class MaskFileSegmenter:
    """Use a mask made elsewhere, for example by Roman's SAM3 script.

    Any non-zero pixel is the part. The mask must have the photo's size.
    """

    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def segmenter_id(self) -> str:
        return "mask-file"

    def segment(self, image: Image, *, exclude: list[NDArray[np.float32]]) -> Mask:
        mask = cv2.imread(str(self._path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise VisionError("cannot read the mask file")
        if mask.shape != image.shape[:2]:
            raise VisionError("mask size does not match the photo")
        binary = np.asarray(mask > 0, dtype=np.uint8)
        _fill_polygons(binary, exclude)
        return _largest_component(binary)


def _end_width_ratio(
    outer: Image, rect: tuple[tuple[float, float], tuple[float, float], float]
) -> float | None:
    """Narrower end width over wider end width along the part's long axis.

    1.0 for a straight cylinder; lower for a taper or a stepped shaft.
    """
    (cx, cy), (width, height), angle = rect
    if width < height:
        angle += 90
        width, height = height, width
    rotation = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    size = math.ceil(math.hypot(outer.shape[0], outer.shape[1]))
    rotation[0, 2] += size / 2 - cx
    rotation[1, 2] += size / 2 - cy
    aligned = cv2.warpAffine(outer, rotation, (size, size), flags=cv2.INTER_NEAREST)
    columns = np.count_nonzero(aligned, axis=0)
    filled = np.flatnonzero(columns)
    if filled.size < 10:
        return None
    span = columns[filled[0] : filled[-1] + 1].astype(np.float64)
    band = max(1, span.size // 10)
    ends = (float(np.median(span[band : 2 * band])), float(np.median(span[-2 * band : -band])))
    widest = max(ends)
    return min(ends) / widest if widest > 0 else None


def measure_mask(mask: Mask, scale: Scale) -> Measurement:
    """Measure the part's outline and holes in millimetres."""
    binary = np.asarray(mask, dtype=np.uint8) * 255
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not contours or hierarchy is None:
        raise VisionError("no part found in the mask")
    tree = hierarchy[0]
    outer_index = max(
        (index for index in range(len(contours)) if tree[index][3] == -1),
        key=lambda index: cv2.contourArea(contours[index]),
    )
    outer = contours[outer_index]
    outer_area = float(cv2.contourArea(outer))
    perimeter = float(cv2.arcLength(outer, closed=True))
    if outer_area <= 0 or perimeter <= 0:
        raise VisionError("the part's outline is empty")
    circularity = 4 * math.pi * outer_area / perimeter**2
    (rect_cx, rect_cy), (rect_w, rect_h), rect_angle = cv2.minAreaRect(outer)
    rect = ((float(rect_cx), float(rect_cy)), (float(rect_w), float(rect_h)), float(rect_angle))
    long_px, short_px = max(rect[1]), min(rect[1])
    if short_px <= 0:
        raise VisionError("the part's outline is a line")
    holes: list[HoleMeasurement] = []
    child = tree[outer_index][2]
    while child != -1:
        hole_area = float(cv2.contourArea(contours[child]))
        hole_perimeter = float(cv2.arcLength(contours[child], closed=True))
        if hole_area >= _MIN_HOLE_SHARE * outer_area and hole_perimeter > 0:
            holes.append(
                HoleMeasurement(
                    diameter_mm=2 * math.sqrt(hole_area / math.pi) * scale.mm_per_px,
                    circularity=4 * math.pi * hole_area / hole_perimeter**2,
                )
            )
        child = tree[child][0]
    holes.sort(key=lambda hole: hole.diameter_mm, reverse=True)
    round_view = circularity >= _ROUND_CIRCULARITY and long_px / short_px <= _ROUND_ASPECT
    filled_outer = np.zeros_like(binary)
    cv2.drawContours(filled_outer, [outer], -1, 255, thickness=cv2.FILLED)
    # The outline's area minus its holes, so a ring is not counted as a disc.
    solid_px = float(np.count_nonzero(binary))
    return Measurement(
        view="round" if round_view else "profile",
        long_mm=long_px * scale.mm_per_px,
        short_mm=short_px * scale.mm_per_px,
        area_mm2=solid_px * scale.mm_per_px**2,
        circularity=min(circularity, 1.0),
        holes=holes,
        end_width_ratio=None if round_view else _end_width_ratio(filled_outer, rect),
        mm_per_px=scale.mm_per_px,
        scale_source=scale.source,
    )


__all__ = [
    "DEFAULT_ARUCO_DICTIONARY",
    "MARKER_EXCLUDE_GROWTH",
    "VISION_VERSION",
    "HoleMeasurement",
    "Image",
    "Mask",
    "MaskFileSegmenter",
    "Measurement",
    "Scale",
    "Segmenter",
    "ThresholdSegmenter",
    "VisionError",
    "fixed_scale",
    "load_image",
    "marker_scale",
    "measure_mask",
]
