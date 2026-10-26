"""The record one sheet hands to the EDI registration script (Issue 2).

``ocrbench extract`` runs a chosen pipeline on one sheet and prints this
record as JSON. It is the contract between the OCR side and the Playwright
script: the script may register a shipment only when ``ready`` is true, and
must send every other sheet to the worker for review. See
``docs/edi-handover-format.md``.
"""

from __future__ import annotations

import re
from typing import Any, Final

from ocrbench.customers import CUSTOMER_TABLE_VERSION, resolve_customer_site
from ocrbench.edi_fields import EDI_FIELD_NAMES, ORDER_NO_RULE, format_problems
from ocrbench.pipelines import PipelineResult, PipelineStatus

HANDOVER_SCHEMA_VERSION: Final[int] = 1
_YEAR_SUFFIX = re.compile(r"-[0-9]{4}$")


def handover_record(result: PipelineResult, *, pipeline_id: str) -> dict[str, Any]:
    """Build the JSON-ready record for one sheet.

    ``ready`` is true only when the pipeline marked the sheet ``ok``, all 4
    fields are present, and the customer maps to a known EDI site.
    ``review_reasons`` lists, in plain words, why a sheet is not ready.
    """
    fields = (
        result.fields.model_dump() if result.fields is not None else dict.fromkeys(EDI_FIELD_NAMES)
    )
    site = resolve_customer_site(fields["customer"])
    order_no = fields["order_no"]
    reasons: list[str] = []
    if result.status is not PipelineStatus.OK:
        reasons.append(f"pipeline status is {result.status.value}")
    reasons.extend(f"{name} was not read" for name in EDI_FIELD_NAMES if fields[name] is None)
    if result.fields is not None:
        reasons.extend(format_problems(result.fields))
    if result.error:
        reasons.append(result.error)
    return {
        "schema_version": HANDOVER_SCHEMA_VERSION,
        "pipeline_id": pipeline_id,
        "ready": not reasons,
        "review_reasons": reasons,
        "fields": fields,
        "edi": {
            "site": site,
            "order_no_printed": order_no,
            "order_no_without_year": (
                _YEAR_SUFFIX.sub("", order_no.strip()) if order_no is not None else None
            ),
            "quantity": fields["quantity"],
        },
        "rules": {
            "customer_table_version": CUSTOMER_TABLE_VERSION,
            "order_no_rule": ORDER_NO_RULE.value,
        },
        "timing_ms": {
            "layer1": result.layer1_ms,
            "layer2": result.layer2_ms,
            "total": result.total_ms,
        },
        "cost_usd": result.cost_usd,
    }


__all__ = ["HANDOVER_SCHEMA_VERSION", "handover_record"]
