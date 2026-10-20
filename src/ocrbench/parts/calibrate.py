"""Set the tolerances from parts a worker has confirmed.

Input: the check records (``ocrbench parts check --out``) of parts that were
confirmed correct, and optionally records of known mismatches (a part
photographed with the wrong sheet on purpose). Output: a tolerances file
with ``calibrated: true``.

The rule for each limit is simple and visible: take the extreme seen on the
correct parts and add a margin. The report then says how many correct parts
the new limits would reject (should be 0) and how many mismatches they catch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from pydantic import ValidationError

from ocrbench.parts.check import PartCheck, Tolerances, Verdict, assign_dimensions, check_part

MIN_RECORDS: Final[int] = 10
_MARGIN: Final[float] = 0.02
"""Added on top of the worst case seen, as a share of the stock size."""


class CalibrationError(Exception):
    """Raised when the records cannot give trustworthy tolerances."""


@dataclass(frozen=True)
class CalibrationReport:
    tolerances: Tolerances
    records: int
    size_ratios: int
    hole_errors: int
    correct_rejected: int
    mismatches: int
    mismatches_caught: int


def load_records(paths: list[Path]) -> list[PartCheck]:
    records: list[PartCheck] = []
    for path in paths:
        try:
            records.append(PartCheck.model_validate(json.loads(path.read_text(encoding="utf-8"))))
        except (OSError, json.JSONDecodeError, ValidationError) as error:
            raise CalibrationError(
                f"cannot read check record {path.name}: {type(error).__name__}"
            ) from None
    return records


def calibrate(
    confirmed: list[PartCheck],
    *,
    mismatches: list[PartCheck] | None = None,
    min_records: int = MIN_RECORDS,
) -> CalibrationReport:
    """Derive tolerances from confirmed parts and test them on both sets."""
    ratios: list[float] = []
    hole_errors: list[float] = []
    usable = [record for record in confirmed if record.measurement is not None]
    for record in usable:
        assert record.measurement is not None
        measured_d, measured_l = assign_dimensions(record.spec, record.measurement)
        for measured, stock in (
            (measured_d, record.spec.stock_d_mm),
            (measured_l, record.spec.stock_l_mm),
        ):
            if measured is not None and stock is not None:
                ratios.append(measured / stock)
        if record.measurement.view == "round":
            for wanted in record.spec.through_holes:
                nearest = min(
                    (
                        abs(hole.diameter_mm - wanted.diameter_mm) / wanted.diameter_mm
                        for hole in record.measurement.holes
                    ),
                    default=None,
                )
                if nearest is not None:
                    hole_errors.append(nearest)
    if len(usable) < min_records or len(ratios) < min_records:
        raise CalibrationError(
            f"need at least {min_records} confirmed parts with a stock size; "
            f"got {len(usable)} parts, {len(ratios)} size readings"
        )
    defaults = Tolerances()
    tolerances = Tolerances(
        calibrated=True,
        envelope_slack_rel=max(0.0, max(ratios) - 1) + _MARGIN,
        envelope_slack_mm=0.0,
        min_fraction_of_stock=max(0.05, min(ratios) - _MARGIN),
        hole_tolerance_rel=(max(hole_errors) + _MARGIN)
        if hole_errors
        else defaults.hole_tolerance_rel,
        hole_tolerance_mm=0.0 if hole_errors else defaults.hole_tolerance_mm,
        registry_size_rel=defaults.registry_size_rel,
        calibration_records=len(usable),
    )

    def rejected(records: list[PartCheck]) -> int:
        return sum(
            check_part(record.spec, record.measurement, tolerances=tolerances).verdict
            is Verdict.FAIL
            for record in records
            if record.measurement is not None
        )

    wrong = mismatches or []
    return CalibrationReport(
        tolerances=tolerances,
        records=len(usable),
        size_ratios=len(ratios),
        hole_errors=len(hole_errors),
        correct_rejected=rejected(usable),
        mismatches=len(wrong),
        mismatches_caught=rejected(wrong),
    )


__all__ = ["MIN_RECORDS", "CalibrationError", "CalibrationReport", "calibrate", "load_records"]
