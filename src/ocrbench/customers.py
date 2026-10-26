"""Versioned mapping from printed customer names to EDI site identifiers.

ADR-0002 §2 (open decision D2) scores ``customer`` by the EDI system it
selects rather than by its exact characters. Add a row here when a new
customer is onboarded, and bump ``CUSTOMER_TABLE_VERSION`` so run manifests
record which table produced a score.

Each site may also list the printed formats of its part and order numbers.
A value that does not fit is sent to review instead of to the EDI, so a
misread field is caught before it becomes a silent error.
"""

import re
import unicodedata
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

CUSTOMER_TABLE_VERSION: Final[int] = 2

_CUSTOMER_SITES: Final[dict[str, str]] = {
    "Nidecマシンツール株式会社": "hwao_nmtj",
}

_WHITESPACE_PATTERN = re.compile(r"\s+")


def customer_key(name: str) -> str:
    """Return the lookup key for a printed customer name.

    NFKC folds full-width Latin letters and half-width katakana into one form,
    and all whitespace is removed, so layout and OCR width differences do not
    change which EDI site a name selects. Case is preserved.
    """
    return _WHITESPACE_PATTERN.sub("", unicodedata.normalize("NFKC", name))


CUSTOMER_SITES: Final[Mapping[str, str]] = MappingProxyType(
    {customer_key(name): site for name, site in _CUSTOMER_SITES.items()}
)
"""Read-only table keyed by :func:`customer_key`."""


def resolve_customer_site(name: str | None) -> str | None:
    """Return the EDI site for a printed customer name, or ``None`` if unknown."""
    if name is None:
        return None
    return CUSTOMER_SITES.get(customer_key(name))


SITE_FORMATS: Final[Mapping[str, Mapping[str, re.Pattern[str]]]] = MappingProxyType(
    {
        "hwao_nmtj": MappingProxyType(
            {
                # Two capital letters and 9 digits, e.g. HR000000000.
                "part_no": re.compile(r"[A-Z]{2}[0-9]{9}"),
                # 6 digits, optionally followed by the printed -YYYY (open decision D1).
                "order_no": re.compile(r"[0-9]{6}(?:-[0-9]{4})?"),
            }
        ),
    }
)
"""Printed formats per EDI site, checked with ``fullmatch`` on the trimmed value."""
