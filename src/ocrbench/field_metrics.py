"""ADR-0002 metrics for the 4 EDI fields: per-sheet scores, run summaries,
layer attribution, paired comparison, and the cloud decision rule (§4, §5, §7).

Statistics reuse the pure-Python helpers in :mod:`ocrbench.metrics`.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from ocrbench.customers import customer_key
from ocrbench.edi_fields import (
    EDI_FIELD_NAMES,
    ORDER_NO_RULE,
    EdiFields,
    OrderNoRule,
    comparable_fields,
    ground_truth_problems,
    normalize_order_no,
)
from ocrbench.metrics import bootstrap_ci, mcnemar_exact, percentile, wilson_interval
from ocrbench.normalize import UNIT_SUFFIXES
from ocrbench.pipelines import PipelineResult, PipelineStatus

CLOUD_MARGIN: Final[float] = 0.02
"""ADR-0002 §7: the cloud pipeline must beat local by about 2 points (open decision D5)."""

_WHITESPACE = re.compile(r"\s+")
_DIGIT_GROUPING = re.compile(r"(?<=\d),(?=\d{3})")


class FieldOutcome(StrEnum):
    """How one predicted field compares with the ground truth."""

    CORRECT = "correct"
    WRONG = "wrong"
    MISSING = "missing"


class Layer(StrEnum):
    """Which layer of a two-layer pipeline lost a field (ADR-0002 §5)."""

    LAYER1 = "layer1"
    LAYER2 = "layer2"


@dataclass(frozen=True)
class SheetScore:
    """The score of one pipeline on one sheet."""

    sheet_id: str
    source_kind: str | None
    status: PipelineStatus
    outcomes: dict[str, FieldOutcome]
    failed_layer: dict[str, Layer]
    total_ms: float
    cost_usd: float

    @property
    def correct_count(self) -> int:
        return sum(outcome is FieldOutcome.CORRECT for outcome in self.outcomes.values())

    @property
    def missing_count(self) -> int:
        return sum(outcome is FieldOutcome.MISSING for outcome in self.outcomes.values())

    @property
    def all_correct(self) -> bool:
        return self.correct_count == len(EDI_FIELD_NAMES)

    @property
    def silent_errors(self) -> int:
        """Wrong values on a sheet marked ``ok``: the ones that would reach the EDI.

        A sheet marked for review is checked by a person, so its wrong values
        are not silent.
        """
        if self.status is not PipelineStatus.OK:
            return 0
        return sum(outcome is FieldOutcome.WRONG for outcome in self.outcomes.values())


def _search_text(text: str) -> str:
    return _WHITESPACE.sub("", unicodedata.normalize("NFKC", text))


_UNIT_PATTERN = "|".join(
    re.escape(unit) for unit in sorted({u.upper() for u in UNIT_SUFFIXES}, key=len, reverse=True)
)
_QUANTITY_LABEL = r"個\s*数"  # the label may be split over two lines in a narrow cell


def _quantity_in(text: str, quantity: int) -> bool:
    """Return whether ``quantity`` is printed as a quantity, not just as some number.

    Small numbers such as 1, 2 or 8 also appear on a sheet as row numbers and
    order counts, so a bare number is not evidence that Layer 1 read the
    quantity. It counts only when followed by a unit (``8PC``, ``8 個``) or
    directly after the 個数 label (``個数 8``, also split as ``個 / 数``), after NFKC folding.
    """
    normalized = _DIGIT_GROUPING.sub("", unicodedata.normalize("NFKC", text)).upper()
    number = rf"(?<!\d){quantity}(?!\d)"
    with_unit = rf"{number}\s*(?:{_UNIT_PATTERN})"
    after_label = rf"{_QUANTITY_LABEL}[\s:|]*{quantity}(?!\d)"
    return re.search(with_unit, normalized) is not None or (
        re.search(after_label, _WHITESPACE.sub(" ", normalized)) is not None
    )


def value_in_layer1_text(
    gt: EdiFields, field: str, layer1_text: str, rule: OrderNoRule = ORDER_NO_RULE
) -> bool:
    """Return whether the correct value for ``field`` appears in the OCR text.

    Both sides are NFKC-normalized and whitespace is ignored, so full-width
    characters and line breaks inside a value do not count against Layer 1.
    Quantity must appear as a whole number, not inside a longer one.
    """
    if field == "quantity":
        return gt.quantity is not None and _quantity_in(layer1_text, gt.quantity)
    raw = {
        "customer": gt.customer,
        "part_no": gt.part_no,
        "order_no": normalize_order_no(gt.order_no, rule),
    }[field]
    if raw is None:
        return False
    needle = customer_key(raw) if field == "customer" else _search_text(raw)
    return bool(needle) and needle in _search_text(layer1_text)


def score_sheet(
    sheet_id: str,
    gt: EdiFields,
    result: PipelineResult,
    *,
    source_kind: str | None = None,
    rule: OrderNoRule = ORDER_NO_RULE,
) -> SheetScore:
    """Score one pipeline result against usable ground truth.

    A field the pipeline left empty is ``MISSING``. A non-empty field that
    does not match is ``WRONG``, including a customer name that maps to no EDI
    site (counted conservatively). When the pipeline returned Layer 1 text,
    every field that is not correct is attributed to a layer.
    """
    problems = ground_truth_problems(gt, rule)
    if problems:
        raise ValueError(f"ground truth for {sheet_id} cannot be scored: {'; '.join(problems)}")

    gt_values = comparable_fields(gt, rule)
    pred_values = comparable_fields(result.fields, rule) if result.fields is not None else None
    raw_pred = (
        (
            result.fields.customer,
            result.fields.part_no,
            result.fields.order_no,
            result.fields.quantity,
        )
        if result.fields is not None
        else None
    )

    outcomes: dict[str, FieldOutcome] = {}
    failed_layer: dict[str, Layer] = {}
    for index, field in enumerate(EDI_FIELD_NAMES):
        if pred_values is not None and pred_values[index] == gt_values[index]:
            outcomes[field] = FieldOutcome.CORRECT
            continue
        present = raw_pred is not None and raw_pred[index] not in (None, "")
        outcomes[field] = FieldOutcome.WRONG if present else FieldOutcome.MISSING
        if result.layer1_text is not None:
            found = value_in_layer1_text(gt, field, result.layer1_text, rule)
            failed_layer[field] = Layer.LAYER2 if found else Layer.LAYER1

    return SheetScore(
        sheet_id=sheet_id,
        source_kind=source_kind,
        status=result.status,
        outcomes=outcomes,
        failed_layer=failed_layer,
        total_ms=result.total_ms,
        cost_usd=result.cost_usd,
    )


@dataclass(frozen=True)
class FieldRunSummary:
    """ADR-0002 §4 metrics for one pipeline over one set of sheets."""

    n_sheets: int
    field_accuracy: float
    field_accuracy_ci: tuple[float, float]
    sheets_correct: int
    sheet_accuracy: float
    sheet_accuracy_ci: tuple[float, float]
    per_field_accuracy: dict[str, float]
    per_field_ci: dict[str, tuple[float, float]]
    silent_errors: int
    sheets_with_silent_errors: int
    missing_fields: int
    status_counts: dict[str, int]
    layer_failures: dict[str, dict[str, int]]
    by_source_kind: dict[str, float]
    time_p50_ms: float
    time_p95_ms: float
    cost_per_sheet_usd: float
    total_cost_usd: float


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def summarize(scores: Sequence[SheetScore]) -> FieldRunSummary:
    """Aggregate sheet scores; intervals resample whole sheets with a fixed seed."""
    if not scores:
        raise ValueError("scores must not be empty")
    ids = [score.sheet_id for score in scores]
    if len(set(ids)) != len(ids):
        raise ValueError("each sheet may be scored only once per run")

    n = len(scores)
    field_count = len(EDI_FIELD_NAMES)
    per_sheet = [score.correct_count / field_count for score in scores]
    sheets_correct = sum(score.all_correct for score in scores)

    per_field_accuracy: dict[str, float] = {}
    per_field_ci: dict[str, tuple[float, float]] = {}
    for field in EDI_FIELD_NAMES:
        hits = [float(score.outcomes[field] is FieldOutcome.CORRECT) for score in scores]
        per_field_accuracy[field] = _mean(hits)
        per_field_ci[field] = bootstrap_ci(hits, _mean)

    layer_failures = {
        layer.value: {
            field: sum(score.failed_layer.get(field) is layer for score in scores)
            for field in EDI_FIELD_NAMES
        }
        for layer in Layer
    }
    kinds = sorted({score.source_kind for score in scores if score.source_kind is not None})
    by_source_kind = {
        kind: _mean([p for p, s in zip(per_sheet, scores, strict=True) if s.source_kind == kind])
        for kind in kinds
    }
    times = [score.total_ms for score in scores]
    total_cost = sum(score.cost_usd for score in scores)

    return FieldRunSummary(
        n_sheets=n,
        field_accuracy=_mean(per_sheet),
        field_accuracy_ci=bootstrap_ci(per_sheet, _mean),
        sheets_correct=sheets_correct,
        sheet_accuracy=sheets_correct / n,
        sheet_accuracy_ci=wilson_interval(sheets_correct, n),
        per_field_accuracy=per_field_accuracy,
        per_field_ci=per_field_ci,
        silent_errors=sum(score.silent_errors for score in scores),
        sheets_with_silent_errors=sum(score.silent_errors > 0 for score in scores),
        missing_fields=sum(score.missing_count for score in scores),
        status_counts=dict(sorted(Counter(score.status.value for score in scores).items())),
        layer_failures=layer_failures,
        by_source_kind=by_source_kind,
        time_p50_ms=percentile(times, 50),
        time_p95_ms=percentile(times, 95),
        cost_per_sheet_usd=total_cost / n,
        total_cost_usd=total_cost,
    )


@dataclass(frozen=True)
class FieldComparison:
    """Paired comparison of pipeline B against pipeline A on the same sheets."""

    n_sheets: int
    field_accuracy_difference: float
    difference_ci: tuple[float, float]
    sheet_a_only: int
    sheet_b_only: int
    sheet_mcnemar_p_value: float


def _paired(
    a: Sequence[SheetScore], b: Sequence[SheetScore]
) -> list[tuple[SheetScore, SheetScore]]:
    by_id_a = {score.sheet_id: score for score in a}
    by_id_b = {score.sheet_id: score for score in b}
    if len(by_id_a) != len(a) or len(by_id_b) != len(b):
        raise ValueError("each sheet may be scored only once per run")
    if not by_id_a or by_id_a.keys() != by_id_b.keys():
        raise ValueError("paired comparison needs the same non-empty set of sheets")
    return [(by_id_a[sheet_id], by_id_b[sheet_id]) for sheet_id in sorted(by_id_a)]


def compare(a: Sequence[SheetScore], b: Sequence[SheetScore]) -> FieldComparison:
    """Compare B with A: positive differences mean B is more accurate.

    The field-accuracy interval is a sheet-resampled bootstrap of the paired
    per-sheet differences; whole-sheet correctness uses exact McNemar.
    """
    pairs = _paired(a, b)
    field_count = len(EDI_FIELD_NAMES)
    differences = [(sb.correct_count - sa.correct_count) / field_count for sa, sb in pairs]
    a_only = sum(sa.all_correct and not sb.all_correct for sa, sb in pairs)
    b_only = sum(sb.all_correct and not sa.all_correct for sa, sb in pairs)
    return FieldComparison(
        n_sheets=len(pairs),
        field_accuracy_difference=_mean(differences),
        difference_ci=bootstrap_ci(differences, _mean),
        sheet_a_only=a_only,
        sheet_b_only=b_only,
        sheet_mcnemar_p_value=mcnemar_exact(a_only, b_only),
    )


class CloudChoice(StrEnum):
    """Outcome of the ADR-0002 §7 decision rule."""

    LOCAL = "local"
    CLOUD = "cloud"


@dataclass(frozen=True)
class CloudDecision:
    """The decision with the comparison behind it and plain-language reasons."""

    choice: CloudChoice
    comparison: FieldComparison
    margin: float
    reasons: tuple[str, ...]


def decide_cloud(
    best_local: Sequence[SheetScore],
    cloud: Sequence[SheetScore],
    *,
    margin: float = CLOUD_MARGIN,
) -> CloudDecision:
    """Prefer local unless the cloud pipeline clearly wins (ADR-0002 §7, D5).

    The cloud pipeline is chosen only when the lower end of the paired
    interval for its field-accuracy advantage is at least ``margin``. Any
    weaker evidence, including a better but uncertain result, keeps local.
    """
    if not 0.0 <= margin < 1.0:
        raise ValueError("margin must be in [0, 1)")
    comparison = compare(best_local, cloud)
    low, high = comparison.difference_ci
    difference = comparison.field_accuracy_difference
    summary = (
        f"cloud minus local field accuracy {difference:+.1%} "
        f"(95% interval {low:+.1%} to {high:+.1%}, {comparison.n_sheets} sheets)"
    )
    if low >= margin:
        return CloudDecision(
            CloudChoice.CLOUD,
            comparison,
            margin,
            (summary, f"the whole interval clears the {margin:.0%} margin"),
        )
    reason = (
        "cloud is not more accurate"
        if difference <= 0
        else f"the interval does not clear the {margin:.0%} margin, so local wins by default"
    )
    return CloudDecision(CloudChoice.LOCAL, comparison, margin, (summary, reason))
