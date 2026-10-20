"""Decode the barcodes printed on a sheet, such as the 伝票No (``* D1224272 *``).

A barcode is read exactly or not at all, so when Kimura can look a 伝票No up,
it gives an exact key for the order without any OCR. The per-process barcodes
on the sheet are decoded too; the caller decides which one it needs.
"""

from __future__ import annotations

from dataclasses import dataclass

import zxingcpp

from ocrbench.parts.vision import Image


@dataclass(frozen=True)
class DecodedBarcode:
    text: str
    format: str
    x: int
    y: int


def read_barcodes(image: Image) -> list[DecodedBarcode]:
    """Every linear or 2D barcode zxing-cpp can read, top to bottom, left to right."""
    found = [
        DecodedBarcode(
            text=result.text,
            format=str(result.format).removeprefix("BarcodeFormat."),
            x=int(result.position.top_left.x),
            y=int(result.position.top_left.y),
        )
        for result in zxingcpp.read_barcodes(image)
        if result.valid and result.text
    ]
    return sorted(found, key=lambda barcode: (barcode.y, barcode.x))


__all__ = ["DecodedBarcode", "read_barcodes"]
