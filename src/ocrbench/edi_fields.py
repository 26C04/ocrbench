"""The 4 EDI fields scored by ADR-0002 and their comparison rules."""

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Any, Final

from pydantic import BaseModel, ConfigDict, Field

from ocrbench.customers import SITE_FORMATS, resolve_customer_site
from ocrbench.normalize import strip_outer_whitespace
from ocrbench.parsing import ParseStatus

EDI_FIELD_NAMES: Final[tuple[str, ...]] = ("customer", "part_no", "order_no", "quantity")

_YEAR_SUFFIX_PATTERN = re.compile(r"-[0-9]{4}$")
_MAX_ERROR_LENGTH = 500


class EdiFields(BaseModel):
    """The values one sheet contributes to an hWAO shipment entry.

    Every field is ``None`` when it cannot be read; pipelines must not guess.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    customer: str | None
    part_no: str | None
    order_no: str | None
    quantity: Annotated[int, Field(ge=1)] | None


class OrderNoRule(StrEnum):
    """How the ``-YYYY`` suffix of a printed order number is compared."""

    KEEP = "keep"
    STRIP_YEAR = "strip_year"


ORDER_NO_RULE: Final[OrderNoRule] = OrderNoRule.KEEP
"""Rule in force until ADR-0002 open decision D1 is settled against hWAO."""

type ComparableFields = tuple[str | None, str | None, str | None, int | None]


def edi_format_schema() -> dict[str, Any]:
    """Return the JSON Schema supplied as Ollama's ``format`` argument."""
    schema = EdiFields.model_json_schema()

    assert schema.get("additionalProperties") is False
    assert set(schema.get("required", ())) == set(EDI_FIELD_NAMES)

    return schema


@dataclass(frozen=True)
class EdiParseResult:
    """A raw Layer 2 response and its strict validation outcome."""

    status: ParseStatus
    fields: EdiFields | None
    error: str | None
    raw_output: str


def parse_edi_output(raw: str) -> EdiParseResult:
    """Parse raw model output as ``EdiFields`` without extracting or repairing JSON."""
    try:
        decoded = json.loads(raw)
    except Exception as error:
        return EdiParseResult(ParseStatus.NOT_JSON, None, str(error)[:_MAX_ERROR_LENGTH], raw)
    try:
        fields = EdiFields.model_validate(decoded)
    except Exception as error:
        return EdiParseResult(
            ParseStatus.SCHEMA_VIOLATION, None, str(error)[:_MAX_ERROR_LENGTH], raw
        )
    return EdiParseResult(ParseStatus.OK, fields, None, raw)


def normalize_order_no(value: str | None, rule: OrderNoRule = ORDER_NO_RULE) -> str | None:
    """Trim an order number and apply the year-suffix rule.

    ``STRIP_YEAR`` removes one trailing ``-`` plus four digits, so
    ``023235-2025`` compares as ``023235``. A value that becomes empty is
    treated as unreadable.
    """
    if value is None:
        return None
    trimmed = strip_outer_whitespace(value)
    if rule is OrderNoRule.STRIP_YEAR:
        trimmed = _YEAR_SUFFIX_PATTERN.sub("", trimmed)
    return trimmed or None


def comparable_fields(fields: EdiFields, rule: OrderNoRule = ORDER_NO_RULE) -> ComparableFields:
    """Return the normalized values compared field by field.

    ``customer`` becomes its EDI site identifier, so an unknown customer
    compares as ``None``.
    """
    part_no = None if fields.part_no is None else strip_outer_whitespace(fields.part_no)
    return (
        resolve_customer_site(fields.customer),
        part_no or None,
        normalize_order_no(fields.order_no, rule),
        fields.quantity,
    )


def compare_fields(
    gt: EdiFields, pred: EdiFields, rule: OrderNoRule = ORDER_NO_RULE
) -> dict[str, bool]:
    """Return per-field correctness of a prediction against ground truth.

    A field is correct only when the normalized ground truth is present and
    equal to the normalized prediction, so two unreadable values never match.
    """
    gt_values = comparable_fields(gt, rule)
    pred_values = comparable_fields(pred, rule)
    return {
        name: gt_value is not None and gt_value == pred_value
        for name, gt_value, pred_value in zip(EDI_FIELD_NAMES, gt_values, pred_values, strict=True)
    }


def ground_truth_problems(gt: EdiFields, rule: OrderNoRule = ORDER_NO_RULE) -> list[str]:
    """List reasons a ground-truth record cannot be scored, empty when usable.

    ADR-0002 only scores sheets a person can read, so every field must be
    present after normalization and the customer must map to an EDI site.
    """
    problems: list[str] = []
    for name, value in zip(EDI_FIELD_NAMES, comparable_fields(gt, rule), strict=True):
        if value is not None:
            continue
        if name == "customer" and gt.customer is not None:
            problems.append("customer: no EDI site is mapped for this name")
        else:
            problems.append(f"{name}: missing")
    return problems


def format_problems(fields: EdiFields) -> list[str]:
    """Return why read values cannot go to the EDI unchecked, in plain words.

    The customer must map to a known EDI site, and the site's part and order
    numbers must have their printed format. Missing values are not reported
    here; the caller already treats them as not read.
    """
    if fields.customer is None:
        return []
    site = resolve_customer_site(fields.customer)
    if site is None:
        return ["customer does not match any known EDI site"]
    problems: list[str] = []
    for name, pattern in SITE_FORMATS.get(site, {}).items():
        value = getattr(fields, name)
        if value is not None and pattern.fullmatch(strip_outer_whitespace(value)) is None:
            problems.append(f"{name} does not have the {site} format")
    return problems
