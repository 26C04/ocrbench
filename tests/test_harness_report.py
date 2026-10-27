"""Tests for the report the automation-harness EDI script reads (synthetic sheets only)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from PIL import Image

from ocrbench.cli import main
from ocrbench.harness_report import harness_report, harness_slip
from tests.helpers import make_text_pdf_bytes
from tests.test_label_rules import _sheet_lines

READY = {
    "ready": True,
    "review_reasons": [],
    "fields": {
        "customer": "Nidecﾏｼﾝﾂｰﾙ株式会社",
        "part_no": "HR900000001",
        "order_no": "900001-2026",
        "quantity": 4,
    },
    "edi": {
        "site": "hwao_nmtj",
        "order_no_printed": "900001-2026",
        "order_no_without_year": "900001",
        "quantity": 4,
    },
}


def test_ready_sheet_becomes_an_ok_slip_with_the_harness_keys() -> None:
    slip = harness_slip(READY, source="sheet.jpg")
    assert slip["status"] == "ok" and slip["pages"] == [1]
    assert slip["edi"] == {
        "site_id": "hwao_nmtj",
        "customer": "Nidecマシンツール株式会社",
        "order_no": "900001",
        "order_no_full": "900001-2026",
        "part_no": "HR900000001",
        "quantity": 4,
        "unit": None,
    }


def test_review_sheet_keeps_its_reasons_and_no_keys() -> None:
    review = {**READY, "ready": False, "review_reasons": ["customer was not read"]}
    slip = harness_slip(review, source="sheet.jpg")
    assert slip["status"] == "needs_review"
    assert slip["review_reasons"] == ["customer was not read"]
    assert set(slip["edi"].values()) == {None}
    report = harness_report(
        [slip, harness_slip(READY, source="b.jpg")],
        pipeline_id="A3G",
        now=datetime(2026, 10, 27, tzinfo=UTC),
    )
    assert report["summary"] == {"slips": 2, "ok": 1, "needs_review": 1}
    assert report["created"] == "2026-10-27T00:00:00+00:00"


def test_cli_writes_a_report_for_a_folder_of_sheets(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "a0r.json"
    config.write_text(
        json.dumps({"kind": "ocr_rules", "pipeline_id": "A0R", "ocr": {"engine": "pdf_text"}}),
        encoding="utf-8",
    )
    sheets = tmp_path / "today"
    sheets.mkdir()
    (sheets / "a.pdf").write_bytes(
        make_text_pdf_bytes([_sheet_lines("HR900000001", "900001-2026", "4PC")])
    )
    Image.new("RGB", (8, 8), "white").save(sheets / "b.png")
    (sheets / "c.png").write_bytes(b"not an image")
    (sheets / "notes.txt").write_text("skip me", encoding="utf-8")
    out = tmp_path / "report.json"

    arguments = ["harness-report", "--pipeline", str(config), "--images", str(sheets)]
    assert main([*arguments, "--out", str(out)]) == 3
    printed = capsys.readouterr().out
    assert "1 ready, 2 to review (of 3 sheets)" in printed
    assert "HR900000001" not in printed
    report = json.loads(out.read_text(encoding="utf-8"))
    assert [slip["source"] for slip in report["slips"]] == ["a.pdf", "b.png", "c.png"]
    assert report["slips"][0]["edi"]["order_no"] == "900001"
    assert "cannot read the sheet" in report["slips"][2]["review_reasons"][0]

    assert main([*arguments, "--out", str(out)]) == 2
    assert (
        main(
            [
                "harness-report",
                "--pipeline",
                str(config),
                "--images",
                str(tmp_path / "x"),
                "--out",
                str(tmp_path / "r2.json"),
            ]
        )
        == 2
    )
