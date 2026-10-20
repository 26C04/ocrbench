"""Split a multi-page PDF into one-page PDFs, keeping each page's text layer.

The dataset and the preprocessor take one sheet per file, but Kimura's system
prints many sheets into one PDF. Pages are copied, not rendered, so a digital
PDF keeps its text layer for the ``pdf_text`` engine.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

import pypdfium2 as pdfium  # type: ignore[import-untyped]

from ocrbench.config import OcrBenchError
from ocrbench.edi_answers import ANSWER_COLUMNS
from ocrbench.label_rules import extract_fields


class PdfSplitError(OcrBenchError):
    """Raised when a PDF cannot be split without overwriting anything."""


def split_pdf(pdf_path: Path, out_dir: Path, *, prefix: str = "sheet") -> list[Path]:
    """Write ``<prefix>-001.pdf`` … into ``out_dir`` and return the paths in page order.

    Nothing is written if any target file already exists.
    """
    if not prefix or any(char in prefix for char in "/\\:"):
        raise PdfSplitError("prefix must be a plain file-name part")
    try:
        source = pdfium.PdfDocument(pdf_path)
    except Exception as error:
        raise PdfSplitError(f"cannot open the PDF: {type(error).__name__}") from None
    try:
        count = len(source)
        if count == 0:
            raise PdfSplitError("the PDF has no pages")
        width = max(3, len(str(count)))
        targets = [out_dir / f"{prefix}-{index + 1:0{width}d}.pdf" for index in range(count)]
        existing = [path.name for path in targets if path.exists()]
        if existing:
            raise PdfSplitError(f"{len(existing)} target files already exist, e.g. {existing[0]}")
        out_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        try:
            for index, target in enumerate(targets):
                page_document = pdfium.PdfDocument.new()
                try:
                    page_document.import_pages(source, [index])
                    with target.open("xb") as stream:
                        written.append(target)  # created: remove it if anything below fails
                        page_document.save(stream)
                finally:
                    page_document.close()
        except BaseException:
            for path in written:
                path.unlink(missing_ok=True)
            raise
        return written
    finally:
        source.close()


def write_answers_template(paths: list[Path], template: Path) -> None:
    """Write an answers CSV with the ``file`` column filled and every other column empty."""
    with template.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(ANSWER_COLUMNS)
        for path in paths:
            writer.writerow([path.name, *([""] * (len(ANSWER_COLUMNS) - 1))])


@dataclass(frozen=True)
class PdfAnswers:
    """Answer rows read from a digital PDF, and the pages a person must still fill in."""

    rows: list[list[str]]
    incomplete_pages: list[int]


def answers_from_pdf(
    pdf_path: Path,
    *,
    file_pattern: str,
    source_kind: str,
    final_share: float,
    seed: int,
) -> PdfAnswers:
    """Build answers-CSV rows from a digital PDF's text layer, one row per page.

    ``file_pattern`` names each page's image, e.g. ``scan-{page:03d}.jpg``
    (pages count from 1), so the rows can describe scans or photos of the
    printed sheets. Fields are read with :func:`ocrbench.label_rules.extract_fields`;
    a field it cannot read with certainty is left blank, and ``import-answers``
    then refuses the file until a person fills it in. ``final_share`` of the
    pages, chosen with a seeded shuffle, go to the final split.
    """
    if source_kind not in ("scan", "photo"):
        raise PdfSplitError("source kind must be 'scan' or 'photo'")
    if not 0.0 <= final_share < 1.0:
        raise PdfSplitError("final share must be at least 0 and below 1")
    try:
        file_pattern.format(page=1)
    except KeyError, IndexError, ValueError:
        raise PdfSplitError("file pattern must use {page}, e.g. scan-{page:03d}.jpg") from None
    try:
        document = pdfium.PdfDocument(pdf_path)
    except Exception as error:
        raise PdfSplitError(f"cannot open the PDF: {type(error).__name__}") from None
    try:
        texts = []
        for index in range(len(document)):
            page = document[index]
            try:
                text_page = page.get_textpage()
                try:
                    texts.append(text_page.get_text_range())
                finally:
                    text_page.close()
            finally:
                page.close()
    finally:
        document.close()
    if not texts:
        raise PdfSplitError("the PDF has no pages")
    if not any(text.strip() for text in texts):
        raise PdfSplitError("the PDF has no text layer (is it a scan?)")

    order = list(range(len(texts)))
    random.Random(seed).shuffle(order)
    final_pages = set(order[: round(len(texts) * final_share)])
    names = [file_pattern.format(page=index + 1) for index in range(len(texts))]
    if len(set(names)) != len(names):
        raise PdfSplitError("file pattern gives two pages the same name; use {page}")

    rows: list[list[str]] = []
    incomplete: list[int] = []
    for index, text in enumerate(texts):
        fields = extract_fields(text)
        values = [fields.customer, fields.part_no, fields.order_no, fields.quantity]
        if any(value is None for value in values):
            incomplete.append(index + 1)
        rows.append(
            [
                names[index],
                source_kind,
                "final" if index in final_pages else "dev",
                *("" if value is None else str(value) for value in values),
            ]
        )
    return PdfAnswers(rows, incomplete)


def write_answers(answers: PdfAnswers, out: Path) -> None:
    """Write the rows as an answers CSV, refusing to overwrite an existing file."""
    with out.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(ANSWER_COLUMNS)
        writer.writerows(answers.rows)


__all__ = [
    "PdfAnswers",
    "PdfSplitError",
    "answers_from_pdf",
    "split_pdf",
    "write_answers",
    "write_answers_template",
]
