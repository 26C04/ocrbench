"""A deterministic Layer 2: read each field from the text after its printed label.

This is the no-LLM baseline for two-layer pipelines. Kimura's instruction
sheet prints every EDI field next to a fixed label (受注先, 部番, 受注No, 個数),
so a rule that reads the value after the label is cheap, fast and fully
explainable. It never guesses: a missing label, an unreadable value, or two
different values for the same label give ``None`` and the sheet is marked
for review.

Change :data:`RULES_VERSION` whenever a pattern changes, so run manifests show
which rules produced a score.
"""

from __future__ import annotations

import re
import unicodedata
from time import perf_counter_ns
from typing import Final

from ocrbench.edi_fields import EdiFields
from ocrbench.normalize import normalize_num_pieces
from ocrbench.pipelines import (
    OcrEngine,
    PipelineResult,
    PipelineStatus,
    _status_for,
    elapsed_ms,
)
from ocrbench.preprocess import PreprocessResult

RULES_VERSION: Final[str] = "2"
# Separators between a label and its value never cross a line break, so an
# empty cell cannot borrow the next line's value.
_SEP: Final[str] = r"[ \t\u3000:|]*"
# A value must end at whitespace, a table border, 。 or the end of the text, so
# a misread character inside it (2O26, 1O個) rejects the value instead of
# cutting it short.
_END: Final[str] = r"(?=[\s|。]|$)"
_LABELS: Final[tuple[str, ...]] = (
    "受注先",
    "送り先",
    "発送先",
    "部番",
    "部品名",
    "受注NO",
    "受注日",
    "個数",
    "営業",
)
_PATTERNS: Final[dict[str, re.Pattern[str]]] = {
    # The customer runs to the next label (送り先), a table border or the line end;
    # spaces inside the name are kept (customer matching ignores them).
    "customer": re.compile(rf"受注先{_SEP}([^\n|]+?)[ \t\u3000]*(?=送り先|\||\r?$)", re.MULTILINE),
    "part_no": re.compile(rf"部番{_SEP}([A-Z0-9][A-Z0-9-]{{3,}}){_END}"),
    # A space followed by "-" means the order number was split; reject it.
    "order_no": re.compile(rf"受注\s*NO\.?{_SEP}([0-9][0-9-]*[0-9]){_END}(?![ \t]*-)"),
    # 個数 may be split over two lines (個 / 数) in a narrow table cell.
    "quantity": re.compile(rf"個\s*数{_SEP}([0-9][0-9,]*[ \t]*(?:PCS|PC|個|本|枚)?){_END}"),
}


def _normalized(text: str) -> str:
    """Fold full-width and half-width forms (NFKC) and upper-case Latin letters.

    Upper-casing only affects Latin letters, so a lower-case ``pc`` or ``No``
    matches; the extracted customer is taken from the unfolded text instead,
    so its printed form is kept.
    """
    return unicodedata.normalize("NFKC", text).upper()


def _is_label(value: str) -> bool:
    compact = re.sub(r"\s+", "", value).upper()
    return any(compact.startswith(label) for label in _LABELS)


def _single(pattern: re.Pattern[str], text: str) -> str | None:
    values = {match.strip() for match in pattern.findall(text)}
    values.discard("")
    if len(values) != 1:
        return None
    value = values.pop()
    return None if _is_label(value) else value


def extract_fields(text: str) -> EdiFields:
    """Return the 4 EDI fields found after their labels; ``None`` where not certain."""
    folded = _normalized(text)
    customer = _single(_PATTERNS["customer"], unicodedata.normalize("NFKC", text))
    quantity_text = _single(_PATTERNS["quantity"], folded)
    return EdiFields(
        customer=customer,
        part_no=_single(_PATTERNS["part_no"], folded),
        order_no=_single(_PATTERNS["order_no"], folded),
        quantity=normalize_num_pieces(quantity_text) if quantity_text is not None else None,
    )


class OcrRulesPipeline:
    """Two-layer candidate whose Layer 2 is :func:`extract_fields` instead of an LLM."""

    def __init__(self, *, pipeline_id: str, ocr: OcrEngine) -> None:
        self._pipeline_id = pipeline_id
        self._ocr = ocr

    @property
    def pipeline_id(self) -> str:
        return self._pipeline_id

    def describe(self) -> dict[str, str]:
        details = {f"ocr.{key}": value for key, value in self._ocr.describe().items()}
        return {
            "kind": "ocr_rules",
            "pipeline_id": self._pipeline_id,
            "ocr_engine": self._ocr.engine_id,
            **details,
            "rules_version": RULES_VERSION,
        }

    def extract(self, image: PreprocessResult) -> PipelineResult:
        started = perf_counter_ns()
        try:
            ocr = self._ocr.transcribe(image)
        except Exception as caught:
            return PipelineResult(
                status=PipelineStatus.ERROR,
                fields=None,
                layer1_text=None,
                layer1_ms=elapsed_ms(started),
                layer2_ms=None,
                total_ms=elapsed_ms(started),
                cost_usd=float(getattr(caught, "cost_usd", 0.0)),
                attempts=0,
                error=f"layer 1 failed: {type(caught).__name__}",
            )
        layer2_started = perf_counter_ns()
        fields = extract_fields(ocr.text)
        return PipelineResult(
            status=_status_for(fields),
            fields=fields,
            layer1_text=ocr.text,
            layer1_ms=ocr.duration_ms,
            layer2_ms=elapsed_ms(layer2_started),
            total_ms=elapsed_ms(started),
            cost_usd=ocr.cost_usd,
            attempts=1,
        )


__all__ = ["RULES_VERSION", "OcrRulesPipeline", "extract_fields"]
