"""Write ocrbench handover records as a report the automation-harness EDI script reads.

Fuke's ``harness ship register|run <site> --ocr report.json`` and the harness
web UI take the JSON that ``harness ocr read --out`` writes: a ``slips`` list
where each slip has a ``status`` (``ok`` or ``needs_review``) and an ``edi``
block of registration keys. Only ``ok`` slips whose ``site_id`` is the site
being shipped are registered; the harness then checks each one against the
EDI row (exactly one row for the 手配番号, same part number, quantity not above
the open quantity) and stops on any mismatch.

This module maps one ``handover_record`` per sheet onto that shape, so the
chosen benchmark pipeline (A3G) can feed the EDI script without changes on the
harness side. A sheet that is not ``ready`` becomes ``needs_review`` with its
reasons, and its keys are left empty, as the harness does for its own reads.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Final

from ocrbench.customers import canonical_name

HARNESS_FORM: Final[str] = "ocrbench"
_EMPTY_KEYS: Final[dict[str, None]] = dict.fromkeys(
    ("site_id", "customer", "order_no", "order_no_full", "part_no", "quantity", "unit")
)


def harness_slip(record: dict[str, Any], *, source: str, page: int = 1) -> dict[str, Any]:
    """Map one ``handover_record`` onto a harness ``SlipResult`` dict."""
    ready = bool(record["ready"])
    edi = record["edi"]
    fields = record["fields"]
    keys: dict[str, Any] = dict(_EMPTY_KEYS)
    if ready:
        keys = {
            **keys,
            "site_id": edi["site"],
            "customer": canonical_name(edi["site"]),
            "order_no": edi["order_no_without_year"],
            "order_no_full": edi["order_no_printed"],
            "part_no": fields["part_no"],
            "quantity": edi["quantity"],
        }
    return {
        "source": source,
        "page": page,
        "pages": [page],
        "status": "ok" if ready else "needs_review",
        "review_reasons": list(record["review_reasons"]),
        "edi": keys,
    }


def harness_report(
    slips: list[dict[str, Any]], *, pipeline_id: str, now: datetime | None = None
) -> dict[str, Any]:
    """Wrap slips in the harness ``OcrReport`` shape."""
    ok = sum(slip["status"] == "ok" for slip in slips)
    return {
        "form": HARNESS_FORM,
        "model": pipeline_id,
        "created": (now or datetime.now(UTC)).isoformat(timespec="seconds"),
        "slips": slips,
        "summary": {"slips": len(slips), "ok": ok, "needs_review": len(slips) - ok},
    }


__all__ = ["HARNESS_FORM", "harness_report", "harness_slip"]
