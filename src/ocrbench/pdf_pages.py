"""Split a multi-page PDF into one-page PDFs, keeping each page's text layer.

The dataset and the preprocessor take one sheet per file, but Kimura's system
prints many sheets into one PDF. Pages are copied, not rendered, so a digital
PDF keeps its text layer for the ``pdf_text`` engine.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pypdfium2 as pdfium  # type: ignore[import-untyped]

from ocrbench.config import OcrBenchError
from ocrbench.edi_answers import ANSWER_COLUMNS


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
                        page_document.save(stream)
                finally:
                    page_document.close()
                written.append(target)
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


__all__ = ["PdfSplitError", "split_pdf", "write_answers_template"]
