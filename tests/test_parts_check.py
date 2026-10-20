"""Tests for the part-check rules, the registry and calibration. All values are invented."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ocrbench.parts.calibrate import CalibrationError, calibrate, load_records
from ocrbench.parts.check import (
    PartCheck,
    RuleResult,
    RuleStatus,
    Tolerances,
    Verdict,
    assign_dimensions,
    check_part,
    load_tolerances,
)
from ocrbench.parts.registry import PartRegistry, RegistryError, RegistryMatch
from ocrbench.parts.spec import HoleSpec, PartSpec
from ocrbench.parts.vision import HoleMeasurement, Measurement

DEFAULTS = Tolerances()


def _spec(**overrides: object) -> PartSpec:
    values: dict[str, object] = {
        "part_no": "HX1",
        "part_name": "FLANGE",
        "material": "SCM415",
        "stock_d_mm": 125.0,
        "stock_l_mm": 40.0,
        "stock_candidates": 1,
    }
    values.update(overrides)
    return PartSpec.model_validate(values)


def _round(diameter: float, holes: tuple[float, ...] = ()) -> Measurement:
    return Measurement(
        view="round",
        long_mm=diameter,
        short_mm=diameter,
        area_mm2=3.14159 * (diameter / 2) ** 2,
        circularity=0.9,
        holes=[HoleMeasurement(diameter_mm=hole, circularity=0.9) for hole in holes],
        mm_per_px=0.25,
        scale_source="marker",
    )


def _profile(long_mm: float, short_mm: float, *, end_ratio: float = 1.0) -> Measurement:
    return Measurement(
        view="profile",
        long_mm=long_mm,
        short_mm=short_mm,
        area_mm2=long_mm * short_mm,
        circularity=0.5,
        end_width_ratio=end_ratio,
        mm_per_px=0.25,
        scale_source="marker",
    )


def _status(rules: list[RuleResult], name: str) -> RuleStatus:
    return next(rule.status for rule in rules if rule.rule == name)


def test_right_flange_passes() -> None:
    result = check_part(_spec(), _round(119), tolerances=DEFAULTS)

    assert result.verdict is Verdict.PASS
    assert result.reasons == ["tolerances are uncalibrated placeholders"]
    assert _status(result.rules, "envelope_d") is RuleStatus.PASS
    assert _status(result.rules, "envelope_l") is RuleStatus.NOT_CHECKED


def test_part_larger_than_stock_fails() -> None:
    result = check_part(_spec(), _round(160), tolerances=DEFAULTS)

    assert result.verdict is Verdict.FAIL
    assert "larger than stock" in result.reasons[0]


def test_part_far_smaller_than_stock_fails() -> None:
    result = check_part(_spec(), _round(40), tolerances=DEFAULTS)

    assert result.verdict is Verdict.FAIL
    assert _status(result.rules, "closeness_d") is RuleStatus.FAIL


def test_slack_allows_small_measurement_error() -> None:
    limit = 125 * (1 + DEFAULTS.envelope_slack_rel) + DEFAULTS.envelope_slack_mm

    assert check_part(_spec(), _round(limit - 0.1), tolerances=DEFAULTS).verdict is Verdict.PASS
    assert check_part(_spec(), _round(limit + 0.1), tolerances=DEFAULTS).verdict is Verdict.FAIL


def test_calibrated_tolerances_drop_the_placeholder_reason() -> None:
    result = check_part(_spec(), _round(119), tolerances=Tolerances(calibrated=True))

    assert result.reasons == []


def test_shaft_profile_checks_both_sizes() -> None:
    spec = _spec(part_name="SHAFT", stock_d_mm=36.0, stock_l_mm=105.0)

    good = check_part(spec, _profile(100, 34), tolerances=DEFAULTS)
    too_long = check_part(spec, _profile(140, 34), tolerances=DEFAULTS)

    assert good.verdict is Verdict.PASS
    assert _status(good.rules, "envelope_l") is RuleStatus.PASS
    assert too_long.verdict is Verdict.FAIL
    assert "length 140.0 mm is larger" in too_long.reasons[0]


def test_disc_standing_on_edge_maps_long_side_to_diameter() -> None:
    spec = _spec()

    assert assign_dimensions(spec, _profile(120, 38)) == (120, 38)
    assert assign_dimensions(_spec(stock_d_mm=36.0, stock_l_mm=105.0), _profile(100, 34)) == (
        34,
        100,
    )
    assert assign_dimensions(_spec(stock_d_mm=None, stock_l_mm=None), _profile(100, 34)) == (
        34,
        100,
    )
    assert assign_dimensions(spec, _round(119)) == (119, None)


def test_through_holes_must_all_be_found() -> None:
    spec = _spec(
        through_holes=[HoleSpec(diameter_mm=11, count=4), HoleSpec(diameter_mm=30, count=1)]
    )

    all_holes = check_part(spec, _round(119, (30.6, 11.4, 11.3, 11.5, 11.2)), tolerances=DEFAULTS)
    three_small = check_part(spec, _round(119, (30.6, 11.4, 11.3, 11.5)), tolerances=DEFAULTS)

    assert all_holes.verdict is Verdict.PASS
    assert three_small.verdict is Verdict.FAIL
    assert "not found: 1×φ11" in three_small.reasons[0]


def test_holes_not_checked_in_profile_or_without_note() -> None:
    with_hole = _spec(through_holes=[HoleSpec(diameter_mm=11, count=1)])

    profile = check_part(with_hole, _profile(120, 38), tolerances=DEFAULTS)
    no_note = check_part(_spec(), _round(119), tolerances=DEFAULTS)

    assert _status(profile.rules, "through_holes") is RuleStatus.NOT_CHECKED
    assert _status(no_note.rules, "through_holes") is RuleStatus.NOT_CHECKED


def test_taper_is_information_only() -> None:
    spec = _spec(stock_d_mm=40.0, stock_l_mm=185.0, mentions_taper=True)

    profile = check_part(spec, _profile(180, 39, end_ratio=0.5), tolerances=DEFAULTS)
    round_view = check_part(spec, _round(39), tolerances=DEFAULTS)

    assert _status(profile.rules, "taper") is RuleStatus.INFO
    assert profile.verdict is Verdict.PASS
    assert _status(round_view.rules, "taper") is RuleStatus.NOT_CHECKED


def test_no_stock_and_no_registry_cannot_check() -> None:
    result = check_part(_spec(stock_d_mm=None, stock_l_mm=None), _round(119), tolerances=DEFAULTS)

    assert result.verdict is Verdict.CANNOT_CHECK
    assert "no size rule could run" in result.reasons[0]


def test_no_measurement_cannot_check() -> None:
    result = check_part(
        _spec(), None, tolerances=DEFAULTS, measurement_error="no scale marker found"
    )

    assert result.verdict is Verdict.CANNOT_CHECK
    assert result.reasons == ["no scale marker found"]


def test_registry_match_alone_allows_a_pass() -> None:
    spec = _spec(stock_d_mm=None, stock_l_mm=None)
    match = RegistryMatch(3, 3, True, "3 earlier part(s) in this view: sizes and holes match")

    result = check_part(spec, _round(119), tolerances=DEFAULTS, registry_match=match)

    assert result.verdict is Verdict.PASS
    assert result.registry_references == 3


def test_registry_statuses() -> None:
    first = RegistryMatch(0, 0, False, "")
    other_view = RegistryMatch(2, 0, False, "")
    mismatch = RegistryMatch(2, 2, False, "area differs")

    def run(match: RegistryMatch | None) -> PartCheck:
        return check_part(_spec(), _round(119), tolerances=DEFAULTS, registry_match=match)

    assert _status(run(None).rules, "registry") is RuleStatus.NOT_CHECKED
    assert _status(run(first).rules, "registry") is RuleStatus.NOT_CHECKED
    assert _status(run(other_view).rules, "registry") is RuleStatus.NOT_CHECKED
    assert run(mismatch).verdict is Verdict.FAIL


def test_load_tolerances(tmp_path: Path) -> None:
    path = tmp_path / "tolerances.json"
    path.write_text(json.dumps({"calibrated": True, "envelope_slack_rel": 0.01}), encoding="utf-8")

    assert load_tolerances(None) == Tolerances()
    assert load_tolerances(path).envelope_slack_rel == 0.01
    path.write_text("{bad", encoding="utf-8")
    with pytest.raises(ValueError, match="tolerances"):
        load_tolerances(path)


def test_registry_round_trip(tmp_path: Path) -> None:
    with PartRegistry(tmp_path / "registry.sqlite") as registry:
        assert registry.compare("HX1", _round(119), size_rel=0.05).references == 0
        registry.add("HX1", _round(119, (30,)))
        assert registry.add("HX1", _round(120, (30,))) == 2
        registry.add("HX2", _profile(100, 34))

        same = registry.compare("HX1", _round(119.5, (30.2,)), size_rel=0.05)
        bigger = registry.compare("HX1", _round(140, (30,)), size_rel=0.05)
        holes = registry.compare("HX1", _round(119.5), size_rel=0.05)
        view = registry.compare("HX2", _round(34), size_rel=0.05)

        assert registry.count() == 3
        assert registry.distinct_part_numbers() == 2
    assert same.consistent and same.comparable == 2
    assert not bigger.consistent and "long side differs" in bigger.detail
    assert not holes.consistent and "0 holes, earlier parts had 1" in holes.detail
    assert view.references == 1 and view.comparable == 0


def test_registry_refuses_the_repository() -> None:
    with pytest.raises(RegistryError, match="outside the repository"):
        PartRegistry(Path(__file__).parent / "registry.sqlite")


def test_calibration_from_confirmed_parts() -> None:
    hole_spec = _spec(through_holes=[HoleSpec(diameter_mm=30, count=1)])
    confirmed = [
        check_part(hole_spec, _round(125 * ratio, (30 * (1 + 0.01 * index),)), tolerances=DEFAULTS)
        for index, ratio in enumerate([0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.93, 0.91])
    ]
    wrong = [check_part(hole_spec, _round(160, (30,)), tolerances=DEFAULTS)]

    report = calibrate(confirmed, mismatches=wrong)

    assert report.tolerances.calibrated
    assert report.tolerances.envelope_slack_rel == pytest.approx(0.02)
    assert report.tolerances.min_fraction_of_stock == pytest.approx(0.88)
    assert report.tolerances.hole_tolerance_rel == pytest.approx(0.11)
    assert report.records == 10 and report.correct_rejected == 0
    assert report.mismatches == 1 and report.mismatches_caught == 1


def test_calibration_needs_enough_records() -> None:
    with pytest.raises(CalibrationError, match="at least 10"):
        calibrate([check_part(_spec(), _round(119), tolerances=DEFAULTS)])


def test_load_records(tmp_path: Path) -> None:
    good = tmp_path / "good.json"
    good.write_text(
        check_part(_spec(), _round(119), tolerances=DEFAULTS).model_dump_json(), encoding="utf-8"
    )
    bad = tmp_path / "bad.json"
    bad.write_text("[]", encoding="utf-8")

    assert len(load_records([good])) == 1
    with pytest.raises(CalibrationError, match=r"bad\.json"):
        load_records([bad])


def test_shape_class_rule() -> None:
    disc = _spec()
    bar = _spec(stock_d_mm=36.0, stock_l_mm=105.0)
    middle = _spec(stock_d_mm=50.0, stock_l_mm=50.0)

    assert (
        _status(check_part(disc, _round(119), tolerances=DEFAULTS).rules, "shape_class")
        is RuleStatus.PASS
    )
    on_edge = check_part(disc, _profile(100, 36), tolerances=DEFAULTS)
    assert on_edge.verdict is Verdict.FAIL
    assert "standing on edge" in on_edge.reasons[0]
    on_end = check_part(bar, _round(35), tolerances=DEFAULTS)
    assert "standing on end" in on_end.reasons[0]
    assert (
        _status(check_part(middle, _round(48), tolerances=DEFAULTS).rules, "shape_class")
        is RuleStatus.NOT_CHECKED
    )
