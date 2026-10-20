"""Decide whether a measured part is consistent with its sheet.

Each rule gives ``pass``, ``fail``, ``not_checked`` (the sheet or the photo
lacks what the rule needs) or ``info`` (measured and shown, never decides).
The verdict is ``fail`` if any rule fails, ``cannot_check`` if no size rule
could run, and ``pass`` otherwise. A failed or uncheckable part goes to the
worker with the reasons; the check never overrides a person.

Rules:

- envelope: the part fits inside its raw stock (``measured <= stock``, plus a
  slack for measurement error).
- closeness: the part is not far smaller than its stock (a finished part
  removes some material, not most of it).
- shape class: a disc-shaped stock (L/D below 0.5) lies flat and shows a
  round outline; a long stock (L/D above 2) lies on its side and shows a
  profile. Without this, a short shaft on its side passes as a flange
  standing on edge, since both outlines fit the same envelope.
- through holes: in a round view, every ``φd貫通`` on the sheet has a hole of
  about that size, as many times as printed.
- registry: when this part number was confirmed before, the sizes, the area
  and the number of holes match the earlier parts.
- taper: the end-width ratio is shown when the sheet mentions a taper (info).

The default :class:`Tolerances` are placeholders, marked ``calibrated:
false`` in every record, until ``ocrbench parts calibrate`` measures them on
confirmed parts.
"""

from __future__ import annotations

import json
from collections import Counter
from enum import StrEnum
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ocrbench.parts.registry import RegistryMatch
from ocrbench.parts.spec import PartSpec
from ocrbench.parts.vision import Measurement

CHECK_SCHEMA_VERSION: Final[int] = 1


class Tolerances(BaseModel):
    """Limits used by the rules. Defaults are placeholders until calibrated."""

    model_config = ConfigDict(extra="forbid")

    calibrated: bool = False
    envelope_slack_rel: float = Field(default=0.03, ge=0, le=1)
    envelope_slack_mm: float = Field(default=2.0, ge=0)
    min_fraction_of_stock: float = Field(default=0.5, gt=0, le=1)
    hole_tolerance_rel: float = Field(default=0.10, ge=0, le=1)
    hole_tolerance_mm: float = Field(default=1.5, ge=0)
    registry_size_rel: float = Field(default=0.05, ge=0, le=1)
    disc_aspect_max: float = Field(default=0.5, gt=0)
    long_aspect_min: float = Field(default=2.0, gt=0)
    calibration_records: int = Field(default=0, ge=0)


class RuleStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    NOT_CHECKED = "not_checked"
    INFO = "info"


class Verdict(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    CANNOT_CHECK = "cannot_check"


class RuleResult(BaseModel):
    """One rule's outcome and its reason in plain words."""

    model_config = ConfigDict(extra="forbid")

    rule: str
    status: RuleStatus
    detail: str


class PartCheck(BaseModel):
    """Everything one check decided, ready to save as JSON."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = CHECK_SCHEMA_VERSION
    verdict: Verdict
    reasons: list[str]
    rules: list[RuleResult]
    spec: PartSpec
    measurement: Measurement | None
    segmenter: str | None
    tolerances: Tolerances
    registry_references: int = 0


_SIZE_RULES: Final[frozenset[str]] = frozenset(
    {"envelope_d", "envelope_l", "closeness_d", "closeness_l", "registry"}
)


def load_tolerances(path: Path | None) -> Tolerances:
    """Read a tolerances JSON file; ``None`` gives the uncalibrated defaults."""
    if path is None:
        return Tolerances()
    try:
        return Tolerances.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ValidationError) as error:
        raise ValueError(f"cannot read tolerances file: {type(error).__name__}") from None


def assign_dimensions(spec: PartSpec, measurement: Measurement) -> tuple[float, float | None]:
    """Map the measured long and short sides to the stock's D and L.

    In a round view only D is visible. In a profile view the side that matches
    the stock's own shape is taken as L: the long side when the stock is
    longer than wide, the short side when it is a disc standing on edge.
    """
    if measurement.view == "round":
        return (measurement.long_mm + measurement.short_mm) / 2, None
    if spec.stock_aspect is not None and spec.stock_aspect < 1:
        return measurement.long_mm, measurement.short_mm
    return measurement.short_mm, measurement.long_mm


def _size_rules(
    name: Literal["d", "l"],
    measured: float | None,
    stock: float | None,
    tolerances: Tolerances,
) -> list[RuleResult]:
    label = "diameter" if name == "d" else "length"
    if stock is None:
        reason = f"sheet gives no stock {label}"
        return [
            RuleResult(rule=f"envelope_{name}", status=RuleStatus.NOT_CHECKED, detail=reason),
            RuleResult(rule=f"closeness_{name}", status=RuleStatus.NOT_CHECKED, detail=reason),
        ]
    if measured is None:
        reason = f"{label} is not visible in this view"
        return [
            RuleResult(rule=f"envelope_{name}", status=RuleStatus.NOT_CHECKED, detail=reason),
            RuleResult(rule=f"closeness_{name}", status=RuleStatus.NOT_CHECKED, detail=reason),
        ]
    limit = stock * (1 + tolerances.envelope_slack_rel) + tolerances.envelope_slack_mm
    floor = stock * tolerances.min_fraction_of_stock
    fits = measured <= limit
    close = measured >= floor
    return [
        RuleResult(
            rule=f"envelope_{name}",
            status=RuleStatus.PASS if fits else RuleStatus.FAIL,
            detail=(
                f"{label} {measured:.1f} mm {'fits' if fits else 'is larger than'} "
                f"stock {stock:.1f} mm (limit {limit:.1f})"
            ),
        ),
        RuleResult(
            rule=f"closeness_{name}",
            status=RuleStatus.PASS if close else RuleStatus.FAIL,
            detail=(
                f"{label} {measured:.1f} mm is {measured / stock:.0%} of stock "
                f"{stock:.1f} mm (minimum {tolerances.min_fraction_of_stock:.0%})"
            ),
        ),
    ]


def _shape_class_rule(
    spec: PartSpec, measurement: Measurement, tolerances: Tolerances
) -> RuleResult:
    aspect = spec.stock_aspect
    if aspect is None:
        return RuleResult(
            rule="shape_class", status=RuleStatus.NOT_CHECKED, detail="sheet gives no stock size"
        )
    if aspect < tolerances.disc_aspect_max:
        expected, shape = "round", "a disc"
    elif aspect > tolerances.long_aspect_min:
        expected, shape = "profile", "a long bar"
    else:
        return RuleResult(
            rule="shape_class",
            status=RuleStatus.NOT_CHECKED,
            detail=f"stock L/D {aspect:.2f} can lie either way",
        )
    if measurement.view == expected:
        return RuleResult(
            rule="shape_class",
            status=RuleStatus.PASS,
            detail=f"stock is {shape} (L/D {aspect:.2f}) and the outline is {expected}",
        )
    posture = "standing on edge" if expected == "round" else "standing on end"
    return RuleResult(
        rule="shape_class",
        status=RuleStatus.FAIL,
        detail=(
            f"stock is {shape} (L/D {aspect:.2f}) but the outline is {measurement.view}: "
            f"wrong part, or the part is {posture}"
        ),
    )


def _counted(names: list[str]) -> str:
    """``["φ11", "φ11", "φ30"]`` -> ``"2×φ11, 1×φ30"``."""
    return ", ".join(f"{count}×{name}" for name, count in Counter(names).items())


def _hole_rule(spec: PartSpec, measurement: Measurement, tolerances: Tolerances) -> RuleResult:
    if not spec.through_holes:
        return RuleResult(
            rule="through_holes",
            status=RuleStatus.NOT_CHECKED,
            detail="sheet lists no through hole",
        )
    if measurement.view != "round":
        return RuleResult(
            rule="through_holes",
            status=RuleStatus.NOT_CHECKED,
            detail="holes are not visible in a profile view",
        )
    unused = [hole.diameter_mm for hole in measurement.holes]
    missing: list[str] = []
    for wanted in spec.through_holes:
        allowed = max(
            wanted.diameter_mm * tolerances.hole_tolerance_rel, tolerances.hole_tolerance_mm
        )
        for _ in range(wanted.count):
            match = next(
                (seen for seen in unused if abs(seen - wanted.diameter_mm) <= allowed), None
            )
            if match is None:
                missing.append(f"φ{wanted.diameter_mm:g}")
            else:
                unused.remove(match)
    wanted_text = ", ".join(f"{hole.count}×φ{hole.diameter_mm:g}" for hole in spec.through_holes)
    if missing:
        return RuleResult(
            rule="through_holes",
            status=RuleStatus.FAIL,
            detail=f"sheet lists {wanted_text}; not found: {_counted(missing)}",
        )
    return RuleResult(
        rule="through_holes", status=RuleStatus.PASS, detail=f"sheet lists {wanted_text}; all found"
    )


def _taper_rule(spec: PartSpec, measurement: Measurement) -> RuleResult:
    if not spec.mentions_taper:
        return RuleResult(
            rule="taper", status=RuleStatus.NOT_CHECKED, detail="sheet mentions no taper"
        )
    if measurement.end_width_ratio is None:
        return RuleResult(
            rule="taper",
            status=RuleStatus.NOT_CHECKED,
            detail="end widths not measurable in this view",
        )
    return RuleResult(
        rule="taper",
        status=RuleStatus.INFO,
        detail=f"sheet mentions a taper; end widths ratio {measurement.end_width_ratio:.2f}",
    )


def _registry_rule(match: RegistryMatch | None) -> RuleResult:
    if match is None:
        return RuleResult(
            rule="registry", status=RuleStatus.NOT_CHECKED, detail="no registry given"
        )
    if match.references == 0:
        return RuleResult(
            rule="registry",
            status=RuleStatus.NOT_CHECKED,
            detail="first time this part number is checked",
        )
    if match.comparable == 0:
        return RuleResult(
            rule="registry",
            status=RuleStatus.NOT_CHECKED,
            detail=f"{match.references} earlier parts, none photographed in this view",
        )
    return RuleResult(
        rule="registry",
        status=RuleStatus.PASS if match.consistent else RuleStatus.FAIL,
        detail=match.detail,
    )


def check_part(
    spec: PartSpec,
    measurement: Measurement | None,
    *,
    tolerances: Tolerances,
    segmenter: str | None = None,
    registry_match: RegistryMatch | None = None,
    measurement_error: str | None = None,
) -> PartCheck:
    """Apply every rule and give the verdict with its reasons."""
    rules: list[RuleResult] = []
    if measurement is None:
        reason = measurement_error or "the photo could not be measured"
        return PartCheck(
            verdict=Verdict.CANNOT_CHECK,
            reasons=[reason],
            rules=[RuleResult(rule="measure", status=RuleStatus.NOT_CHECKED, detail=reason)],
            spec=spec,
            measurement=None,
            segmenter=segmenter,
            tolerances=tolerances,
        )
    measured_d, measured_l = assign_dimensions(spec, measurement)
    rules.extend(_size_rules("d", measured_d, spec.stock_d_mm, tolerances))
    rules.extend(_size_rules("l", measured_l, spec.stock_l_mm, tolerances))
    rules.append(_shape_class_rule(spec, measurement, tolerances))
    rules.append(_hole_rule(spec, measurement, tolerances))
    rules.append(_taper_rule(spec, measurement))
    rules.append(_registry_rule(registry_match))
    failed = [rule.detail for rule in rules if rule.status is RuleStatus.FAIL]
    sized = [rule for rule in rules if rule.rule in _SIZE_RULES and rule.status is RuleStatus.PASS]
    if failed:
        verdict, reasons = Verdict.FAIL, failed
    elif not sized:
        verdict = Verdict.CANNOT_CHECK
        reasons = ["no size rule could run: the sheet gives no stock size and no earlier part"]
    else:
        verdict, reasons = Verdict.PASS, []
    if not tolerances.calibrated:
        reasons.append("tolerances are uncalibrated placeholders")
    return PartCheck(
        verdict=verdict,
        reasons=reasons,
        rules=rules,
        spec=spec,
        measurement=measurement,
        segmenter=segmenter,
        tolerances=tolerances,
        registry_references=registry_match.references if registry_match is not None else 0,
    )


__all__ = [
    "CHECK_SCHEMA_VERSION",
    "PartCheck",
    "RuleResult",
    "RuleStatus",
    "Tolerances",
    "Verdict",
    "assign_dimensions",
    "check_part",
    "load_tolerances",
]
