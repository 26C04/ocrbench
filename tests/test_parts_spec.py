"""Tests for reading the part spec from sheet text. All values are invented."""

from __future__ import annotations

import pytest

from ocrbench.parts.spec import SPEC_RULES_VERSION, HoleSpec, read_part_spec

SHEET = "\n".join(
    [
        "伝票No. 1000001",
        "受注先 Nidecﾏｼﾝﾂｰﾙ株式会社 送り先 外注課",
        "部番 HX123456789",
        "受注No 012345-2026 部品名 SEAL HOLDER * 営業：担当",  # noqa: RUF001
        "材質",
        "SCM415",
        "素材 φ 125.0 × 40.0",
        "4-φ11貫通 φ30貫通",
        "電子カルテ 12345",
    ]
)


def test_reads_every_field_of_a_sheet() -> None:
    spec = read_part_spec(SHEET)

    assert spec.part_no == "HX123456789"
    assert spec.part_name == "SEAL HOLDER"
    assert spec.material == "SCM415"
    assert (spec.stock_d_mm, spec.stock_l_mm) == (125.0, 40.0)
    assert spec.stock_candidates == 1
    assert spec.stock_aspect == pytest.approx(0.32)
    assert spec.through_holes == [
        HoleSpec(diameter_mm=11, count=4),
        HoleSpec(diameter_mm=30, count=1),
    ]
    assert spec.karte_no == "12345"
    assert not spec.mentions_taper
    assert spec.rules_version == SPEC_RULES_VERSION


def test_full_width_text_and_x_separator_are_folded() -> None:
    spec = read_part_spec("部品名 ＳＨＡＦＴ\n材質 Ｓ４５Ｃ\nφ３６．０ｘ１００\nテーパ部")  # noqa: RUF001

    assert spec.part_name == "SHAFT"
    assert spec.material == "S45C"
    assert (spec.stock_d_mm, spec.stock_l_mm) == (36.0, 100.0)
    assert spec.mentions_taper
    assert spec.stock_aspect == pytest.approx(100 / 36)


def test_missing_values_are_none_not_guesses() -> None:
    spec = read_part_spec("部番 HX1\n何もない")

    assert spec.part_name is None
    assert spec.material is None
    assert spec.stock_d_mm is None and spec.stock_l_mm is None
    assert spec.stock_aspect is None
    assert spec.through_holes == []
    assert spec.karte_no is None


def test_two_different_materials_give_none() -> None:
    assert read_part_spec("SCM415\nS45C").material is None
    assert read_part_spec("SCM415 SCM415").material == "SCM415"


def test_first_stock_size_is_used_and_all_are_counted() -> None:
    spec = read_part_spec("φ50×80\n穴 φ10×20")

    assert (spec.stock_d_mm, spec.stock_l_mm) == (50.0, 80.0)
    assert spec.stock_candidates == 2


def test_hole_note_is_not_a_stock_size() -> None:
    spec = read_part_spec("φ11貫通")

    assert spec.stock_d_mm is None
    assert spec.through_holes == [HoleSpec(diameter_mm=11, count=1)]


def test_repeated_hole_note_keeps_largest_count() -> None:
    spec = read_part_spec("6-φ9キリ貫通\nφ9貫通")

    assert spec.through_holes == [HoleSpec(diameter_mm=9, count=6)]


def test_inner_diameter_value_and_mention() -> None:
    assert read_part_spec("内径 φ42.5").inner_d_mm == 42.5
    mention = read_part_spec("内径仕上げ")
    assert mention.mentions_inner_d and mention.inner_d_mm is None


def test_material_inside_a_longer_code_is_ignored() -> None:
    assert read_part_spec("XSCM4150").material is None
