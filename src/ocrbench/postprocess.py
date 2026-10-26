"""Optional checks applied to a pipeline's 4 fields before its status is decided.

Both are off unless a pipeline config turns them on, so earlier runs stay
reproducible:

- ``snap_customer``: a near-miss customer name becomes its EDI site's printed
  name (:func:`ocrbench.customers.snap_customer`).
- ``rules_check``: the label rules read the same Layer 1 text, and each field
  is kept only when the LLM and the rules agree, when one of them is empty, or
  when exactly one of two different values has the customer's printed format.
  Otherwise the field is left empty, which sends the sheet to review.
"""

from __future__ import annotations

from ocrbench.customers import SITE_FORMATS, resolve_customer_site, snap_customer
from ocrbench.edi_fields import EdiFields
from ocrbench.normalize import strip_outer_whitespace


def snapped(fields: EdiFields) -> EdiFields:
    """Return the fields with the customer snapped to a known name where it is close."""
    return fields.model_copy(update={"customer": snap_customer(fields.customer)})


def _fits(site: str | None, name: str, value: str) -> bool:
    pattern = SITE_FORMATS.get(site or "", {}).get(name)
    return pattern is not None and pattern.fullmatch(value) is not None


def _pick_text(site: str | None, name: str, llm: str | None, rules: str | None) -> str | None:
    a = strip_outer_whitespace(llm) if llm is not None else None
    b = strip_outer_whitespace(rules) if rules is not None else None
    if not a or not b or a == b:
        return a or b or None
    fits_a, fits_b = _fits(site, name, a), _fits(site, name, b)
    if fits_a != fits_b:
        return a if fits_a else b
    return None


def cross_check(llm: EdiFields, rules: EdiFields) -> EdiFields:
    """Combine the LLM's fields with the label rules' reading of the same text."""
    customer = llm.customer or rules.customer
    site = resolve_customer_site(llm.customer) or resolve_customer_site(rules.customer)
    if llm.quantity is None or rules.quantity is None:
        quantity = llm.quantity if llm.quantity is not None else rules.quantity
    else:
        quantity = llm.quantity if llm.quantity == rules.quantity else None
    return EdiFields(
        customer=customer,
        part_no=_pick_text(site, "part_no", llm.part_no, rules.part_no),
        order_no=_pick_text(site, "order_no", llm.order_no, rules.order_no),
        quantity=quantity,
    )
