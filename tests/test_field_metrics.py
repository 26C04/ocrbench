"""Tests for the ADR-0002 field metrics, layer attribution, and cloud decision rule."""

from __future__ import annotations

import pytest

from ocrbench.edi_fields import EdiFields, OrderNoRule
from ocrbench.field_metrics import (
    CloudChoice,
    FieldOutcome,
    Layer,
    SheetScore,
    compare,
    decide_cloud,
    score_sheet,
    summarize,
    value_in_layer1_text,
)
from ocrbench.pipelines import PipelineResult, PipelineStatus

GT = EdiFields(
    customer="Nidecマシンツール株式会社",
    part_no="HR237101021",
    order_no="023235-2025",
    quantity=1,
)
ALL = ("customer", "part_no", "order_no", "quantity")


def _result(
    fields: EdiFields | None = GT,
    *,
    status: PipelineStatus = PipelineStatus.OK,
    layer1_text: str | None = None,
    total_ms: float = 1000.0,
    cost_usd: float = 0.0,
) -> PipelineResult:
    return PipelineResult(
        status=status,
        fields=fields,
        layer1_text=layer1_text,
        layer1_ms=None,
        layer2_ms=None,
        total_ms=total_ms,
        cost_usd=cost_usd,
        attempts=1,
    )


def _pred(**changes: object) -> EdiFields:
    return GT.model_copy(update=changes)


def test_perfect_sheet() -> None:
    score = score_sheet("s1", GT, _result(), source_kind="scan")

    assert score.outcomes == dict.fromkeys(ALL, FieldOutcome.CORRECT)
    assert score.all_correct
    assert score.correct_count == 4
    assert score.silent_errors == 0
    assert score.failed_layer == {}


def test_wrong_missing_and_normalized_matches() -> None:
    pred = _pred(
        customer="Ｎｉｄｅｃﾏｼﾝﾂｰﾙ株式会社", part_no="HR237101O21", order_no=None, quantity=2
    )

    score = score_sheet("s1", GT, _result(pred))

    assert score.outcomes == {
        "customer": FieldOutcome.CORRECT,
        "part_no": FieldOutcome.WRONG,
        "order_no": FieldOutcome.MISSING,
        "quantity": FieldOutcome.WRONG,
    }
    assert score.silent_errors == 2
    assert score.missing_count == 1


def test_unmapped_customer_counts_as_wrong() -> None:
    score = score_sheet("s1", GT, _result(_pred(customer="ニデック")))

    assert score.outcomes["customer"] is FieldOutcome.WRONG


def test_no_fields_means_every_field_is_missing() -> None:
    score = score_sheet("s1", GT, _result(None, status=PipelineStatus.FORMAT_ERROR))

    assert set(score.outcomes.values()) == {FieldOutcome.MISSING}
    assert score.silent_errors == 0


def test_wrong_values_on_a_review_sheet_are_not_silent() -> None:
    score = score_sheet("s1", GT, _result(_pred(quantity=7), status=PipelineStatus.NEEDS_REVIEW))

    assert score.outcomes["quantity"] is FieldOutcome.WRONG
    assert score.silent_errors == 0


def test_order_number_rule_is_applied_to_both_sides() -> None:
    pred = _pred(order_no="023235")

    keep = score_sheet("s1", GT, _result(pred), rule=OrderNoRule.KEEP)
    strip = score_sheet("s1", GT, _result(pred), rule=OrderNoRule.STRIP_YEAR)

    assert keep.outcomes["order_no"] is FieldOutcome.WRONG
    assert strip.outcomes["order_no"] is FieldOutcome.CORRECT


def test_unusable_ground_truth_is_rejected() -> None:
    with pytest.raises(ValueError, match="s9 cannot be scored: part_no: missing"):
        score_sheet("s9", _pred(part_no=None), _result())


def test_layer_attribution() -> None:
    full_width_order = "\uff10\uff12\uff13\uff12\uff13\uff15-2025"
    text = (
        "受注先 Nidec マシンツール\n株式会社\n部番 HR237101021\n"
        f"受注No {full_width_order}\n個数 12 PC"
    )
    pred = _pred(customer=None, part_no="HR237101O21", order_no="023236-2025", quantity=None)

    score = score_sheet("s1", GT, _result(pred, layer1_text=text))

    assert score.failed_layer == {
        "customer": Layer.LAYER2,
        "part_no": Layer.LAYER2,
        "order_no": Layer.LAYER2,
        "quantity": Layer.LAYER1,
    }


@pytest.mark.parametrize(
    ("text", "found"),
    [
        ("個数 1 PC", True),
        ("個数 1PC", True),
        ("個数 12", False),
        ("1,000", False),
        ("\uff11 pcs", True),
        ("個数\n\uff11", True),
        ("1 個", True),
        # A bare number is not the quantity: row numbers and counts share small values.
        ("\uff11", False),
        ("1 CL-2 0.70 個数 BPC", False),
        ("受注数 1", False),
    ],
)
def test_quantity_must_appear_as_a_whole_number(text: str, found: bool) -> None:
    assert value_in_layer1_text(GT, "quantity", text) is found


def test_order_number_search_follows_the_rule() -> None:
    assert value_in_layer1_text(GT, "order_no", "023235", OrderNoRule.STRIP_YEAR)
    assert not value_in_layer1_text(GT, "order_no", "023235", OrderNoRule.KEEP)


def _scores(pattern: list[int], *, kind: str = "scan", cost: float = 0.0) -> list[SheetScore]:
    """Build scores where pattern[i] is how many of the 4 fields sheet i gets right."""
    scores = []
    for index, correct in enumerate(pattern):
        updates = dict(zip(ALL[correct:], (None, "X", "X", 9)[correct:], strict=True))
        pred = _pred(**{key: value for key, value in updates.items() if key != "customer"})
        if correct == 0:
            pred = _pred(customer=None, part_no="X", order_no="X", quantity=9)
        scores.append(
            score_sheet(
                f"s{index:02d}",
                GT,
                _result(pred, total_ms=float(100 * (index + 1)), cost_usd=cost),
                source_kind=kind,
            )
        )
    return scores


def test_summarize_golden_values() -> None:
    scores = _scores([4, 4, 4, 3]) + _scores([4], kind="photo")
    scores[-1] = SheetScore(
        "photo", "photo", PipelineStatus.OK, scores[-1].outcomes, {}, 5000.0, 0.004
    )

    summary = summarize(scores)

    assert summary.n_sheets == 5
    assert summary.field_accuracy == pytest.approx(19 / 20)
    assert summary.sheets_correct == 4
    assert summary.sheet_accuracy == pytest.approx(0.8)
    low, high = summary.sheet_accuracy_ci
    assert 0.0 <= low < 0.8 < high <= 1.0
    assert summary.per_field_accuracy["quantity"] == pytest.approx(0.8)
    assert summary.per_field_accuracy["part_no"] == 1.0
    assert summary.silent_errors == 1
    assert summary.sheets_with_silent_errors == 1
    assert summary.missing_fields == 0
    assert summary.status_counts == {"ok": 5}
    assert summary.by_source_kind == {"photo": 1.0, "scan": pytest.approx(15 / 16)}
    assert summary.time_p50_ms == 300.0
    assert summary.time_p95_ms == 5000.0
    assert summary.total_cost_usd == pytest.approx(0.004)
    assert summary.cost_per_sheet_usd == pytest.approx(0.0008)
    assert summarize(scores) == summary


def test_summarize_counts_layer_failures() -> None:
    text = "部番 HR237101021"
    scores = [
        score_sheet("a", GT, _result(_pred(part_no="X"), layer1_text=text)),
        score_sheet("b", GT, _result(_pred(quantity=5), layer1_text=text)),
    ]

    failures = summarize(scores).layer_failures

    assert failures["layer2"]["part_no"] == 1
    assert failures["layer1"]["quantity"] == 1
    assert failures["layer1"]["part_no"] == 0


def test_summarize_rejects_empty_and_duplicate_sheets() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        summarize([])
    one = score_sheet("same", GT, _result())
    with pytest.raises(ValueError, match="only once"):
        summarize([one, one])


def test_compare_pairs_by_sheet_id() -> None:
    a = _scores([4, 4, 3, 2, 4])
    b = list(reversed(_scores([4, 4, 4, 4, 2])))

    comparison = compare(a, b)

    assert comparison.n_sheets == 5
    assert comparison.field_accuracy_difference == pytest.approx((0 + 0 + 1 + 2 - 2) / 20)
    assert comparison.sheet_a_only == 1
    assert comparison.sheet_b_only == 2
    assert 0.0 < comparison.sheet_mcnemar_p_value <= 1.0


def test_compare_needs_the_same_sheets() -> None:
    with pytest.raises(ValueError, match="same non-empty set"):
        compare(_scores([4, 4]), _scores([4]))
    with pytest.raises(ValueError, match="same non-empty set"):
        compare([], [])


def test_cloud_wins_only_with_a_clear_margin() -> None:
    local = _scores([2] * 30)
    cloud = _scores([4] * 30)

    decision = decide_cloud(local, cloud)

    assert decision.choice is CloudChoice.CLOUD
    assert decision.comparison.difference_ci[0] >= 0.02
    assert "clears the 2% margin" in decision.reasons[1]


def test_slightly_better_cloud_keeps_local() -> None:
    local = _scores([4] * 23 + [3])
    cloud = _scores([4] * 24)

    decision = decide_cloud(local, cloud)

    assert decision.comparison.field_accuracy_difference > 0
    assert decision.choice is CloudChoice.LOCAL
    assert "local wins by default" in decision.reasons[1]


def test_worse_cloud_keeps_local_and_margin_is_validated() -> None:
    decision = decide_cloud(_scores([4] * 5), _scores([3] * 5))

    assert decision.choice is CloudChoice.LOCAL
    assert decision.reasons[1] == "cloud is not more accurate"
    with pytest.raises(ValueError, match="margin"):
        decide_cloud(_scores([4]), _scores([4]), margin=1.5)


def test_quantity_label_split_over_two_lines_counts_as_read() -> None:
    assert value_in_layer1_text(GT, "quantity", "個\n数\n1")
    assert value_in_layer1_text(GT, "quantity", "個 数 1")
