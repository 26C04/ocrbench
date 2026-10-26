"""Tests for the ADR-0002 EDI field record, its parser, and comparison rules."""

import json

import pytest
from pydantic import ValidationError

from ocrbench.customers import CUSTOMER_SITES, customer_key, resolve_customer_site
from ocrbench.edi_fields import (
    EDI_FIELD_NAMES,
    ORDER_NO_RULE,
    EdiFields,
    OrderNoRule,
    comparable_fields,
    compare_fields,
    edi_format_schema,
    format_problems,
    ground_truth_problems,
    normalize_order_no,
    parse_edi_output,
)
from ocrbench.parsing import ParseStatus

NIDEC = "Nidecマシンツール株式会社"


def _fields(
    customer: str | None = NIDEC,
    part_no: str | None = "HR237101021",
    order_no: str | None = "023235-2025",
    quantity: int | None = 1,
) -> EdiFields:
    return EdiFields(customer=customer, part_no=part_no, order_no=order_no, quantity=quantity)


def test_schema_requires_exactly_the_four_fields_and_forbids_others() -> None:
    schema = edi_format_schema()

    assert set(schema["required"]) == set(EDI_FIELD_NAMES)
    assert schema["additionalProperties"] is False


@pytest.mark.parametrize(
    "payload",
    [
        {"customer": NIDEC, "part_no": "A", "order_no": "1", "quantity": 1, "material": "SS400"},
        {"customer": NIDEC, "part_no": "A", "order_no": "1"},
        {"customer": NIDEC, "part_no": "A", "order_no": "1", "quantity": "1"},
        {"customer": NIDEC, "part_no": "A", "order_no": "1", "quantity": 0},
        {"customer": NIDEC, "part_no": "A", "order_no": "1", "quantity": 1.0},
    ],
)
def test_invalid_records_are_rejected(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        EdiFields.model_validate(payload)


def test_parse_accepts_bare_json_with_nulls() -> None:
    raw = json.dumps({"customer": None, "part_no": "A", "order_no": None, "quantity": None})

    result = parse_edi_output(raw)

    assert result.status is ParseStatus.OK
    assert result.fields == EdiFields(customer=None, part_no="A", order_no=None, quantity=None)
    assert result.error is None


@pytest.mark.parametrize(
    "raw",
    [
        '```json\n{"customer": null, "part_no": null, "order_no": null, "quantity": null}\n```',
        'Here: {"customer": null, "part_no": null, "order_no": null, "quantity": null}',
        "",
    ],
)
def test_parse_does_not_rescue_text_around_json(raw: str) -> None:
    result = parse_edi_output(raw)

    assert result.status is ParseStatus.NOT_JSON
    assert result.fields is None
    assert result.raw_output == raw


def test_parse_reports_schema_violation_with_bounded_error() -> None:
    result = parse_edi_output(json.dumps({"customer": "x" * 2000, "extra": 1}))

    assert result.status is ParseStatus.SCHEMA_VIOLATION
    assert result.error is not None
    assert len(result.error) <= 500


@pytest.mark.parametrize(
    ("value", "rule", "expected"),
    [
        (" 023235-2025 ", OrderNoRule.KEEP, "023235-2025"),
        (" 023235-2025 ", OrderNoRule.STRIP_YEAR, "023235"),
        ("023235", OrderNoRule.STRIP_YEAR, "023235"),
        ("023235-25", OrderNoRule.STRIP_YEAR, "023235-25"),
        ("2025-2025", OrderNoRule.STRIP_YEAR, "2025"),
        ("-2025", OrderNoRule.STRIP_YEAR, None),
        ("   ", OrderNoRule.KEEP, None),
        (None, OrderNoRule.STRIP_YEAR, None),
    ],
)
def test_normalize_order_no(value: str | None, rule: OrderNoRule, expected: str | None) -> None:
    assert normalize_order_no(value, rule) == expected


def test_order_no_rule_defaults_to_keep_until_d1_is_settled() -> None:
    assert ORDER_NO_RULE is OrderNoRule.KEEP


@pytest.mark.parametrize(
    "name",
    [
        NIDEC,
        f"  {NIDEC}　",
        "Nidec マシンツール 株式会社",
        "Ｎｉｄｅｃマシンツール株式会社",
        "Nidecﾏｼﾝﾂｰﾙ株式会社",
    ],
)
def test_customer_variants_resolve_to_the_same_site(name: str) -> None:
    assert resolve_customer_site(name) == "hwao_nmtj"


@pytest.mark.parametrize("name", [None, "", "NIDECマシンツール株式会社", "ニデック株式会社"])
def test_unknown_customer_resolves_to_none(name: str | None) -> None:
    assert resolve_customer_site(name) is None


def test_customer_table_is_keyed_by_normalized_names() -> None:
    assert all(customer_key(key) == key for key in CUSTOMER_SITES)
    with pytest.raises(TypeError):
        CUSTOMER_SITES["x"] = "y"  # type: ignore[index]


def test_comparable_fields_normalizes_each_field() -> None:
    fields = _fields(customer=" Nidecﾏｼﾝﾂｰﾙ株式会社 ", part_no=" HR237101021 ")

    assert comparable_fields(fields) == ("hwao_nmtj", "HR237101021", "023235-2025", 1)
    assert comparable_fields(fields, OrderNoRule.STRIP_YEAR)[2] == "023235"


def test_compare_fields_scores_each_field_independently() -> None:
    gt = _fields()
    pred = _fields(customer="Ｎｉｄｅｃマシンツール株式会社", part_no="HR237101O21", quantity=2)

    assert compare_fields(gt, pred) == {
        "customer": True,
        "part_no": False,
        "order_no": True,
        "quantity": False,
    }


def test_order_no_rule_changes_the_comparison() -> None:
    gt = _fields(order_no="023235-2025")
    pred = _fields(order_no="023235")

    assert compare_fields(gt, pred, OrderNoRule.KEEP)["order_no"] is False
    assert compare_fields(gt, pred, OrderNoRule.STRIP_YEAR)["order_no"] is True


def test_missing_values_never_match() -> None:
    gt = _fields(customer="未登録の顧客", part_no=None)
    pred = _fields(customer="別の未登録顧客", part_no=None)

    result = compare_fields(gt, pred)

    assert result["customer"] is False
    assert result["part_no"] is False


def test_ground_truth_problems() -> None:
    assert ground_truth_problems(_fields()) == []
    assert ground_truth_problems(
        _fields(customer="未登録の顧客", part_no=" ", order_no=None, quantity=None)
    ) == [
        "customer: no EDI site is mapped for this name",
        "part_no: missing",
        "order_no: missing",
        "quantity: missing",
    ]
    assert ground_truth_problems(_fields(customer=None)) == ["customer: missing"]


def test_format_problems_accepts_the_printed_nidec_formats() -> None:
    assert format_problems(_fields(order_no="000001-2026")) == []
    assert format_problems(_fields(order_no="000001")) == []
    assert format_problems(_fields(part_no=" HR000000001 ")) == []
    assert format_problems(_fields(part_no=None, order_no=None)) == []
    assert format_problems(_fields(customer=None)) == []


@pytest.mark.parametrize(
    ("fields", "problem"),
    [
        (_fields(part_no="COLLAR"), "part_no does not have the hwao_nmtj format"),
        (_fields(part_no="H0000000001"), "part_no does not have the hwao_nmtj format"),
        (_fields(order_no="1224272"), "order_no does not have the hwao_nmtj format"),
        (_fields(order_no="000001+2026"), "order_no does not have the hwao_nmtj format"),
        (_fields(customer="Nidecモーター株式会社"), "customer does not match any known EDI site"),
    ],
)
def test_format_problems_flags_values_from_the_wrong_box(fields: EdiFields, problem: str) -> None:
    assert format_problems(fields) == [problem]
