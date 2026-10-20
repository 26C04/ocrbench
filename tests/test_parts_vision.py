"""Tests for the scale marker, segmenters and measurement, on invented photos."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from ocrbench.parts.synthetic import (
    SyntheticPart,
    marker_image,
    printable_marker,
    render_photo,
)
from ocrbench.parts.vision import (
    MaskFileSegmenter,
    Measurement,
    ThresholdSegmenter,
    VisionError,
    fixed_scale,
    load_image,
    marker_scale,
    measure_mask,
)

FLANGE = SyntheticPart("round", 120, holes=((30, 0), (11, 45), (11, 45), (11, 45)))
SHAFT = SyntheticPart("profile", 36, length_mm=100)
TAPER = SyntheticPart("profile", 40, length_mm=180, end_diameter_mm=20)


def _measure(part: SyntheticPart, *, angle_deg: float = 0.0) -> Measurement:
    photo = render_photo(part, angle_deg=angle_deg)
    scale = marker_scale(photo, marker_mm=50)
    assert scale.marker_polygon is not None
    mask = ThresholdSegmenter().segment(photo, exclude=[scale.marker_polygon])
    return measure_mask(mask, scale)


def test_marker_gives_the_true_scale() -> None:
    scale = marker_scale(render_photo(FLANGE, mm_per_px=0.2), marker_mm=50)

    assert scale.source == "marker"
    assert scale.marker_id == 0
    assert scale.mm_per_px == pytest.approx(0.2, rel=0.01)


def test_flange_seen_from_above_is_round_with_its_holes() -> None:
    measurement = _measure(FLANGE, angle_deg=12)

    assert measurement.view == "round"
    assert measurement.long_mm == pytest.approx(120, rel=0.02)
    assert measurement.short_mm == pytest.approx(120, rel=0.02)
    diameters = [hole.diameter_mm for hole in measurement.holes]
    assert len(diameters) == 4
    assert diameters[0] == pytest.approx(30, abs=1.5)
    assert diameters[1:] == pytest.approx([11, 11, 11], abs=1.5)
    assert measurement.end_width_ratio is None


@pytest.mark.parametrize("angle", [0.0, 25.0, -60.0])
def test_shaft_on_its_side_is_a_profile(angle: float) -> None:
    measurement = _measure(SHAFT, angle_deg=angle)

    assert measurement.view == "profile"
    assert measurement.long_mm == pytest.approx(100, rel=0.02)
    assert measurement.short_mm == pytest.approx(36, rel=0.03)
    assert measurement.holes == []
    assert measurement.end_width_ratio == pytest.approx(1.0, abs=0.05)


def test_taper_lowers_the_end_width_ratio() -> None:
    measurement = _measure(TAPER, angle_deg=-20)

    assert measurement.end_width_ratio is not None
    assert measurement.end_width_ratio < 0.8


def test_photo_without_marker_cannot_be_scaled() -> None:
    blank = np.full((300, 300, 3), 60, dtype=np.uint8)

    with pytest.raises(VisionError, match="no scale marker"):
        marker_scale(blank, marker_mm=50)


def test_two_markers_need_an_id() -> None:
    photo = render_photo(SHAFT)
    second = marker_image(marker_id=3, side_px=120)
    photo[900 : 900 + second.shape[0], 40 : 40 + second.shape[1]] = second[..., None]

    with pytest.raises(VisionError, match="more than one"):
        marker_scale(photo, marker_mm=50)
    assert marker_scale(photo, marker_mm=50, marker_id=3).marker_id == 3
    with pytest.raises(VisionError, match="no scale marker"):
        marker_scale(photo, marker_mm=50, marker_id=9)


def test_scale_values_must_be_positive() -> None:
    with pytest.raises(VisionError):
        marker_scale(render_photo(SHAFT), marker_mm=0)
    with pytest.raises(VisionError):
        fixed_scale(0)
    assert fixed_scale(0.3).source == "fixed"


def test_empty_photo_has_no_part() -> None:
    empty = np.full((200, 200, 3), 60, dtype=np.uint8)

    with pytest.raises(VisionError, match="no part"):
        ThresholdSegmenter().segment(empty, exclude=[])


def test_mask_file_segmenter(tmp_path: Path) -> None:
    photo = render_photo(SHAFT)
    scale = marker_scale(photo, marker_mm=50)
    assert scale.marker_polygon is not None
    mask = ThresholdSegmenter().segment(photo, exclude=[scale.marker_polygon])
    path = tmp_path / "mask.png"
    cv2.imwrite(str(path), mask.astype(np.uint8) * 255)

    from_file = MaskFileSegmenter(path).segment(photo, exclude=[scale.marker_polygon])

    assert MaskFileSegmenter(path).segmenter_id == "mask-file"
    assert np.array_equal(from_file, mask)
    with pytest.raises(VisionError, match="does not match"):
        MaskFileSegmenter(path).segment(photo[:100], exclude=[])
    with pytest.raises(VisionError, match="cannot read"):
        MaskFileSegmenter(tmp_path / "missing.png").segment(photo, exclude=[])


def test_load_image(tmp_path: Path) -> None:
    path = tmp_path / "photo.png"
    cv2.imwrite(str(path), render_photo(SHAFT))

    assert load_image(path).shape == (1200, 1600, 3)
    with pytest.raises(VisionError):
        load_image(tmp_path / "missing.png")


def test_fixed_scale_measures_without_marker() -> None:
    photo = render_photo(SHAFT, mm_per_px=0.25)
    polygon = marker_scale(photo, marker_mm=50).marker_polygon
    assert polygon is not None
    mask = ThresholdSegmenter().segment(photo, exclude=[polygon])

    measurement = measure_mask(mask, fixed_scale(0.25))

    assert measurement.scale_source == "fixed"
    assert measurement.long_mm == pytest.approx(100, rel=0.02)


def test_empty_mask_is_rejected() -> None:
    with pytest.raises(VisionError):
        measure_mask(np.zeros((50, 50), dtype=np.bool_), fixed_scale(1))


def test_printable_marker_size() -> None:
    image = printable_marker(marker_id=0, marker_mm=50, dpi=300)

    black_square = round(50 / 25.4 * 300)
    assert image.shape[0] == black_square + 2 * round(black_square * 0.25)
