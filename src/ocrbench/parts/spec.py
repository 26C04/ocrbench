"""Read the part description printed on an instruction sheet.

The sheet's Layer 1 text (the PDF text layer or OCR output) carries more than
the 4 EDI fields: the part name after 部品名, the material, the raw stock
size as ``φD × L`` and machining notes such as ``φ11貫通`` (through hole),
``内径`` (inner diameter) and テーパ (taper). Like :mod:`ocrbench.label_rules`
this never guesses: a value that is missing or printed twice with different
values is ``None``, and the check reports it as not checkable.

Change :data:`SPEC_RULES_VERSION` whenever a pattern changes.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from ocrbench.label_rules import extract_fields

SPEC_RULES_VERSION: Final[str] = "1"
_NUMBER: Final[str] = r"([0-9]+(?:\.[0-9]+)?)"
_TIMES: Final[str] = r"[×xX*]"
_PART_NAME: Final[re.Pattern[str]] = re.compile(
    r"部品名[ \t　:|]*([^\n|]+?)[ \t　*]*(?=営業|\||\r?\n|$)"
)
# The raw stock of a round bar: "φ 36.0 × 100.0". A hole note such as
# "φ11貫通" has no "×", so it never matches.
_STOCK: Final[re.Pattern[str]] = re.compile(rf"[φΦ⌀Ø]\s*{_NUMBER}\s*{_TIMES}\s*{_NUMBER}")
# "φ11貫通" or "6-φ11貫通" (six holes of φ11).
_THROUGH_HOLE: Final[re.Pattern[str]] = re.compile(
    rf"(?:([0-9]+)\s*-\s*)?[φΦ⌀Ø]\s*{_NUMBER}\s*(?:キリ\s*)?貫通"
)
_INNER: Final[re.Pattern[str]] = re.compile(rf"内径\s*[φΦ⌀Ø]?\s*(?:{_NUMBER})?")
_TAPER: Final[re.Pattern[str]] = re.compile(r"テ\s*[-ー]?\s*ーパ|テーパ|TAPER", re.IGNORECASE)
_MATERIAL: Final[re.Pattern[str]] = re.compile(
    r"(?<![A-Z0-9])(SNCM[0-9]{3}|SCM[0-9]{3}|SUJ[0-9]|SUS[0-9]{3}[A-Z]?|S[0-9]{2}C|SS[0-9]{2,3}"
    r"|FCD[0-9]{2,3}|FC[0-9]{2,3}|A[0-9]{4})(?![A-Z0-9])"
)
_KARTE: Final[re.Pattern[str]] = re.compile(r"電子カルテ\s*(?:NO\.?|No\.?)?\s*([0-9]+)")


class HoleSpec(BaseModel):
    """A through hole printed on the sheet: diameter and how many."""

    model_config = ConfigDict(extra="forbid")

    diameter_mm: float = Field(gt=0)
    count: int = Field(ge=1)


class PartSpec(BaseModel):
    """What the sheet says about the physical part; ``None`` where not certain."""

    model_config = ConfigDict(extra="forbid")

    part_no: str | None
    part_name: str | None
    material: str | None
    stock_d_mm: float | None = Field(default=None, gt=0)
    stock_l_mm: float | None = Field(default=None, gt=0)
    stock_candidates: int = Field(default=0, ge=0)
    through_holes: list[HoleSpec] = Field(default_factory=list)
    inner_d_mm: float | None = Field(default=None, gt=0)
    mentions_inner_d: bool = False
    mentions_taper: bool = False
    karte_no: str | None = None
    rules_version: str = SPEC_RULES_VERSION

    @property
    def stock_aspect(self) -> float | None:
        """Stock length over diameter; below 0.5 is a disc, above 2 a long part."""
        if self.stock_d_mm is None or self.stock_l_mm is None:
            return None
        return self.stock_l_mm / self.stock_d_mm


def _single(values: list[str]) -> str | None:
    distinct = {value.strip() for value in values if value.strip()}
    return distinct.pop() if len(distinct) == 1 else None


def read_part_spec(text: str) -> PartSpec:
    """Return the part spec found in one sheet's Layer 1 text.

    The stock size is the first ``φD × L`` on the sheet, which is where Kimura's
    layout prints the raw material; ``stock_candidates`` counts every
    ``φD × L`` found so a report can flag sheets where that assumption is weak.
    """
    folded = unicodedata.normalize("NFKC", text)
    stock = _STOCK.findall(folded)
    holes: dict[float, int] = {}
    for count, diameter in _THROUGH_HOLE.findall(folded):
        size = float(diameter)
        holes[size] = max(holes.get(size, 0), int(count) if count else 1)
    inner = [match.group(1) for match in _INNER.finditer(folded) if match.group(1)]
    inner_value = _single(inner)
    name = _single([value.rstrip("*").strip() for value in _PART_NAME.findall(folded)])
    return PartSpec(
        part_no=extract_fields(text).part_no,
        part_name=name.upper() if name is not None else None,
        material=_single(_MATERIAL.findall(folded.upper())),
        stock_d_mm=float(stock[0][0]) if stock else None,
        stock_l_mm=float(stock[0][1]) if stock else None,
        stock_candidates=len(stock),
        through_holes=[
            HoleSpec(diameter_mm=size, count=count) for size, count in sorted(holes.items())
        ],
        inner_d_mm=float(inner_value) if inner_value is not None else None,
        mentions_inner_d="内径" in folded,
        mentions_taper=_TAPER.search(folded) is not None,
        karte_no=_single(_KARTE.findall(folded)),
    )


__all__ = ["SPEC_RULES_VERSION", "HoleSpec", "PartSpec", "read_part_spec"]
