"""Reusable, code-generated test inputs that do not contain sensitive data."""

import csv
import hashlib
import json
from pathlib import Path

from PIL import Image

from ocrbench.dataset import DocEntry, Manifest, SourceKind, Split
from ocrbench.edi_fields import EdiFields
from ocrbench.schema import LineItem, OrderDocument


def make_pdf_bytes(page_count: int) -> bytes:
    """Build a minimal PDF with the requested number of blank Letter pages."""

    page_ids = list(range(3, 3 + page_count))
    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {page_count} >>".encode(),
    ]
    objects.extend(b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>" for _ in page_ids)

    content = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for object_id, body in enumerate(objects, start=1):
        offsets.append(len(content))
        content.extend(f"{object_id} 0 obj\n".encode())
        content.extend(body)
        content.extend(b"\nendobj\n")

    xref_offset = len(content)
    content.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    content.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        content.extend(f"{offset:010d} 00000 n \n".encode())
    content.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode()
    )
    return bytes(content)


def make_order_document(index: int) -> OrderDocument:
    """Return deterministic synthetic ground truth for one test document."""
    return OrderDocument(
        customer_name=f"Synthetic Customer {index:02d}",
        order_no=f"ORDER-{index:04d}",
        delivery_date=f"2026-08-{index + 1:02d}",
        items=[
            LineItem(
                part_no=f"PART-{index:04d}",
                material=f"Synthetic Material {index:02d}",
                num_pieces=index + 1,
            )
        ],
    )


def make_mini_dataset(root: Path) -> Manifest:
    """Create an eight-document synthetic dataset for integration tests.

    The manifest contains four development documents (two per source kind),
    two selection documents, and two final documents. Images, ground truth,
    and the manifest are generated at runtime; no binary or real data fixture
    is required in the repository.
    """
    raw_dir = root / "raw"
    gt_dir = root / "gt"
    raw_dir.mkdir(parents=True)
    gt_dir.mkdir()

    assignments: tuple[tuple[Split, SourceKind, int], ...] = (
        ("dev", "scan", 2),
        ("dev", "photo", 2),
        ("selection", "scan", 1),
        ("selection", "photo", 1),
        ("final", "scan", 1),
        ("final", "photo", 1),
    )
    docs: list[DocEntry] = []
    index = 0
    for split, source_kind, count in assignments:
        for _ in range(count):
            temporary_image = raw_dir / f"synthetic-{index}.png"
            color = ((index * 37) % 256, (index * 73) % 256, (index * 109) % 256)
            Image.new("RGB", (32, 24), color).save(temporary_image, format="PNG")
            sha256 = hashlib.sha256(temporary_image.read_bytes()).hexdigest()
            doc_id = f"doc-{sha256[:12]}"
            anonymous_image = raw_dir / f"{doc_id}.png"
            temporary_image.rename(anonymous_image)

            ground_truth = make_order_document(index)
            (gt_dir / f"{doc_id}.json").write_text(
                json.dumps(
                    ground_truth.model_dump(mode="json"),
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            docs.append(
                DocEntry(
                    doc_id=doc_id,
                    source_kind=source_kind,
                    file=anonymous_image.name,
                    sha256=sha256,
                    split=split,
                )
            )
            index += 1

    docs.sort(key=lambda entry: entry.doc_id)
    manifest = Manifest(schema_version=1, docs=docs)
    (root / "manifest.json").write_text(
        json.dumps(
            manifest.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


EDI_CUSTOMER = "Nidecマシンツール株式会社"
EDI_ROWS: tuple[tuple[str, str, str, str, str, str, str], ...] = (
    ("s0.png", "scan", "dev", EDI_CUSTOMER, "HR900000001", "900001-2025", "1 PC"),
    ("s1.png", "photo", "dev", EDI_CUSTOMER, "HR900000002", "900002-2025", "2"),
    ("s2.png", "scan", "dev", EDI_CUSTOMER, "HR900000003", "900003-2026", "3PC"),
    ("s3.png", "scan", "final", EDI_CUSTOMER, "HR900000004", "900004-2026", "4"),
    ("s4.png", "photo", "final", EDI_CUSTOMER, "HR900000005", "900005-2025", "5"),
)
"""Invented answers: 3 dev and 2 final sheets, scans and photos in both splits."""


def make_edi_dataset(root: Path, answers_dir: Path) -> Manifest:
    """Build a synthetic ADR-0002 ``edi_fields`` dataset through the real answers import."""
    from ocrbench.edi_answers import ANSWER_COLUMNS, import_answers

    (root / "raw").mkdir(parents=True)
    for index, row in enumerate(EDI_ROWS):
        color = (40 * index + 20, 255 - 30 * index, 90)
        Image.new("RGB", (48, 32), color).save(root / "raw" / row[0], format="PNG")
    answers = answers_dir / "answers.csv"
    with answers.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(ANSWER_COLUMNS)
        writer.writerows(EDI_ROWS)
    return import_answers(root, answers)


def read_edi_ground_truth(root: Path, doc_id: str) -> EdiFields:
    """Return the synthetic ground truth written by :func:`make_edi_dataset`."""
    text = (root / "gt" / f"{doc_id}.json").read_text(encoding="utf-8")
    return EdiFields.model_validate_json(text)


def make_text_pdf_bytes(pages: list[list[str]]) -> bytes:
    """Build a digital PDF whose pages carry the given lines as a real text layer.

    Any Unicode text works, Japanese included: each distinct character gets a
    one-byte code, and the font's ``/ToUnicode`` map turns the codes back into
    the original characters on extraction. Glyph shapes are irrelevant here.
    """
    alphabet = sorted({char for lines in pages for line in lines for char in line})
    # Codes start at 0x21: pdfium treats control and space codes specially.
    if len(alphabet) > 0xFE - 0x21 + 1:
        raise ValueError("too many distinct characters for a one-byte test font")
    code = {char: 0x21 + index for index, char in enumerate(alphabet)}
    page_ids = [3 + 2 * index for index in range(len(pages))]
    font_id = 3 + 2 * len(pages)
    cmap_id = font_id + 1
    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode(),
    ]
    for page_id, lines in zip(page_ids, pages, strict=True):
        ops = ["BT", "/F1 12 Tf", "72 760 Td", "14 TL"]
        ops.extend(f"<{''.join(f'{code[char]:02X}' for char in line)}> Tj T*" for line in lines)
        ops.append("ET")
        stream = "\n".join(ops).encode("ascii")
        resources = f"/Resources << /Font << /F1 {font_id} 0 R >> >>"
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] {resources} "
            f"/Contents {page_id + 1} 0 R >>".encode()
        )
        objects.append(f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream")
    objects.append(
        f"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /ToUnicode {cmap_id} 0 R >>".encode()
    )
    entries = "\n".join(
        f"<{index:02X}> <{''.join(f'{unit:04X}' for unit in _utf16_units(char))}>"
        for char, index in code.items()
    )
    cmap = (
        "/CIDInit /ProcSet findresource begin 12 dict begin begincmap\n"
        "/CMapName /OcrbenchTest def /CMapType 2 def\n"
        "1 begincodespacerange <00> <FF> endcodespacerange\n"
        f"{len(code)} beginbfchar\n{entries}\nendbfchar\n"
        "endcmap CMapName currentdict /CMap defineresource pop end end"
    ).encode("ascii")
    objects.append(f"<< /Length {len(cmap)} >>\nstream\n".encode() + cmap + b"\nendstream")

    content = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for object_id, body in enumerate(objects, start=1):
        offsets.append(len(content))
        content.extend(f"{object_id} 0 obj\n".encode())
        content.extend(body)
        content.extend(b"\nendobj\n")
    xref_offset = len(content)
    content.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    content.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        content.extend(f"{offset:010d} 00000 n \n".encode())
    trailer = f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
    content.extend(f"{trailer}startxref\n{xref_offset}\n%%EOF\n".encode())
    return bytes(content)


def _utf16_units(char: str) -> list[int]:
    encoded = char.encode("utf-16-be")
    return [int.from_bytes(encoded[i : i + 2], "big") for i in range(0, len(encoded), 2)]
