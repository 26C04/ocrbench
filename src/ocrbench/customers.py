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
from difflib import SequenceMatcher
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


LOOSE_CUTOFF: Final[float] = 0.65
LOOSE_MARGIN: Final[float] = 0.1
_LEGAL_FORMS = re.compile(
    r"(株式会社|有限会社|\(株\)|\(有\)|CO\.,?\s*LTD\.?|INC\.?)", re.IGNORECASE
)
_LOOSE_SEPARATORS = re.compile(r"[\s・.,]")


def loose_key(name: str) -> str:
    """Comparison key without legal-form words, spaces or case (as in the template harness)."""
    folded = unicodedata.normalize("NFKC", name)
    return _LOOSE_SEPARATORS.sub("", _LEGAL_FORMS.sub("", folded)).casefold()


def loose_site(name: str | None) -> str | None:
    """Return the closest EDI site when it is similar enough and clearly ahead of the next.

    Half-width katakana such as ﾏｼﾝﾂｰﾙ rarely survive OCR character for
    character, so an exact match misses names a person reads at once. With one
    customer in the table, any similar name maps to it, a similarly named
    company included; the format check is what still guards the other fields.
    """
    if name is None or not loose_key(name):
        return None
    best: dict[str, float] = {}
    for key, site in CUSTOMER_SITES.items():
        score = SequenceMatcher(None, loose_key(name), loose_key(key)).ratio()
        best[site] = max(best.get(site, 0.0), score)
    ranked = sorted(best.values(), reverse=True)
    runner_up = ranked[1] if len(ranked) > 1 else 0.0
    if ranked[0] >= LOOSE_CUTOFF and ranked[0] - runner_up >= LOOSE_MARGIN:
        return max(best, key=lambda site: best[site])
    return None


TEXT_CUTOFF: Final[float] = 0.7


def find_site_in_text(text: str) -> str | None:
    """Return the EDI site whose printed name appears, closely enough, in a whole OCR text.

    Each line is compared window by window with each site's name, legal forms
    and spaces removed, so the name is found even when other text shares its
    line. The best site must reach ``TEXT_CUTOFF`` and beat the next by
    ``LOOSE_MARGIN``. This is a fallback for when the extracted customer field
    maps to no site.
    """
    best: dict[str, float] = {}
    lines = [loose_key(line) for line in text.splitlines()]
    for key, site in CUSTOMER_SITES.items():
        target = loose_key(key)
        for line in lines:
            for start in range(max(1, len(line) - len(target) + 1)):
                window = line[start : start + len(target)]
                score = SequenceMatcher(None, window, target).ratio()
                best[site] = max(best.get(site, 0.0), score)
    if not best:
        return None
    ranked = sorted(best.values(), reverse=True)
    runner_up = ranked[1] if len(ranked) > 1 else 0.0
    if ranked[0] >= TEXT_CUTOFF and ranked[0] - runner_up >= LOOSE_MARGIN:
        return max(best, key=lambda site: best[site])
    return None


def canonical_name(site: str) -> str:
    """Return the table's printed name for an EDI site."""
    return next(key for key, value in _CUSTOMER_SITES.items() if value == site)


def snap_customer(name: str | None) -> str | None:
    """Replace a near-miss customer name with its site's printed name; leave others as read."""
    if name is None or resolve_customer_site(name) is not None:
        return name
    site = loose_site(name)
    return canonical_name(site) if site is not None else name
