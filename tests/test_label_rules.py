"""Tests for the PDF text-layer engine, label rules, PDF splitting, and their pipelines.

All sheet text is invented; the layout mimics the labels of the instruction sheet.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from ocrbench.cli import main
from ocrbench.edi_fields import EdiFields
from ocrbench.label_rules import RULES_VERSION, OcrRulesPipeline, extract_fields
from ocrbench.ocr_engines import OcrEngineError, PdfTextLayerEngine
from ocrbench.pdf_pages import PdfSplitError, split_pdf, write_answers_template
from ocrbench.pipeline_config import OcrRulesConfig, build_pipeline, parse_pipeline_config
from ocrbench.pipeline_results import load_pipeline_run
from ocrbench.pipelines import FakeOcrEngine, PipelineStatus
from ocrbench.preprocess import PreprocessResult, preprocess
from tests.helpers import make_text_pdf_bytes

CUSTOMER_HALF_WIDTH = "Nidecﾏｼﾝﾂｰﾙ株式会社"


def _sheet_lines(part_no: str, order_no: str, quantity: str) -> list[str]:
    """Text in the order a digital instruction sheet's text layer gives it."""
    return [
        "伝票No. 1000001",
        "生産指示書 受注日 2026 年 10 月 07 日",
        f"受注先 {CUSTOMER_HALF_WIDTH} 送り先 外注課 発送先",
        f"部番 {part_no}",
        f"受注No {order_no} 部品名 SAMPLE PART 営業：担当",  # noqa: RUF001
        "工程名 段取",
        "合計 作業者 作業日 機械工数 個",
        f"数 {quantity}",
        "1 CL-2 1.00 0.70 6.60",
        "2 UK 0.00 0.10 0.80",
        "受注数 1",
    ]


def _image(text_layer: str | None) -> PreprocessResult:
    return PreprocessResult(b"png", 1, 1, "pdf", 0.1, "v1", text_layer=text_layer)


def test_rules_read_every_field_from_the_sheet_text() -> None:
    fields = extract_fields("\n".join(_sheet_lines("HR900000001", "900001-2026", "8PC")))
    assert fields == EdiFields(
        customer="Nidecマシンツール株式会社",
        part_no="HR900000001",
        order_no="900001-2026",
        quantity=8,
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # OCR-style text: table borders, no spaces, full-width digits, lower-case unit.
        (
            "| 受注先NidecY》ツ9ール株式会社|送り先 | 外注課\n| 部番 HR90000000l |\n"
            "受注No ９００００１-2026 。 | 部品名\n個数 ８ pc",  # noqa: RUF001
            EdiFields(
                customer="NidecY》ツ9ール株式会社",
                part_no="HR90000000L",  # a misread value is returned, then scored as wrong
                order_no="900001-2026",
                quantity=8,
            ),
        ),
        # Labels missing entirely: nothing is guessed.
        (
            "HR900000001 900001-2026 8PC",
            EdiFields(customer=None, part_no=None, order_no=None, quantity=None),
        ),
        # Two different part numbers after the label: ambiguous, so null.
        (
            "部番 HR900000001\n部番 HR900000002\n個数 3PC",
            EdiFields(customer=None, part_no=None, order_no=None, quantity=3),
        ),
        # A misread digit before PC is not a quantity.
        ("個数 BPC", EdiFields(customer=None, part_no=None, order_no=None, quantity=None)),
    ],
)
def test_rules_never_guess(text: str, expected: EdiFields) -> None:
    assert extract_fields(text) == expected


def test_rules_pipeline_statuses_and_description() -> None:
    complete = "\n".join(_sheet_lines("HR900000001", "900001-2026", "2PC"))
    pipeline = OcrRulesPipeline(
        pipeline_id="A1R", ocr=FakeOcrEngine([complete, "部番 HR900000001"])
    )

    first = pipeline.extract(_image(None))
    second = pipeline.extract(_image(None))

    assert first.status is PipelineStatus.OK
    assert first.layer1_text == complete
    assert first.layer2_ms is not None
    assert second.status is PipelineStatus.NEEDS_REVIEW
    assert second.fields is not None and second.fields.part_no == "HR900000001"
    description = pipeline.describe()
    assert description["kind"] == "ocr_rules"
    assert description["rules_version"] == RULES_VERSION
    assert description["ocr.name"] == "fake-ocr"


def test_rules_pipeline_reports_a_layer1_failure() -> None:
    result = OcrRulesPipeline(pipeline_id="A0R", ocr=PdfTextLayerEngine()).extract(_image(None))
    assert result.status is PipelineStatus.ERROR
    assert result.error == "layer 1 failed: OcrEngineError"
    assert result.fields is None


def test_pdf_text_engine() -> None:
    engine = PdfTextLayerEngine()
    assert engine.transcribe(_image("部番 HR1")).text == "部番 HR1"
    with pytest.raises(OcrEngineError, match="no PDF text layer"):
        engine.transcribe(_image(None))
    assert engine.engine_id == "pdf-text"
    assert engine.describe()["name"] == "pdf-text-layer"


def test_preprocess_keeps_the_text_layer_of_a_digital_pdf(tmp_path: Path) -> None:
    digital = tmp_path / "digital.pdf"
    digital.write_bytes(make_text_pdf_bytes([_sheet_lines("HR900000001", "900001-2026", "4PC")]))
    image = tmp_path / "image.png"
    from PIL import Image

    Image.new("RGB", (4, 4), "white").save(image)

    result = preprocess(digital)
    assert result.text_layer is not None
    assert "部番 HR900000001" in result.text_layer
    assert preprocess(image).text_layer is None


def test_split_pdf_keeps_text_and_never_overwrites(tmp_path: Path) -> None:
    source = tmp_path / "many.pdf"
    source.write_bytes(make_text_pdf_bytes([[f"部番 HR90000000{index}"] for index in range(1, 4)]))
    out = tmp_path / "raw"

    pages = split_pdf(source, out, prefix="kimura")

    assert [page.name for page in pages] == ["kimura-001.pdf", "kimura-002.pdf", "kimura-003.pdf"]
    assert preprocess(pages[1]).text_layer == "部番 HR900000002"
    with pytest.raises(PdfSplitError, match="already exist"):
        split_pdf(source, out, prefix="kimura")
    with pytest.raises(PdfSplitError, match="prefix"):
        split_pdf(source, out, prefix="../x")
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"not a pdf")
    with pytest.raises(PdfSplitError, match="cannot open"):
        split_pdf(broken, tmp_path / "other")

    template = tmp_path / "answers.csv"
    write_answers_template(pages, template)
    rows = list(csv.reader(template.open(encoding="utf-8")))
    assert rows[0][0] == "file"
    assert [row[0] for row in rows[1:]] == [page.name for page in pages]
    assert all(value == "" for row in rows[1:] for value in row[1:])


def test_config_kinds() -> None:
    config = parse_pipeline_config(
        '{"kind": "ocr_rules", "pipeline_id": "A0R", "ocr": {"engine": "pdf_text"}}'
    )
    assert isinstance(config, OcrRulesConfig)

    def no_adapter() -> object:
        raise AssertionError("rules pipelines must not need Ollama")

    built = build_pipeline(config, adapter=no_adapter, prompt=lambda name: None)  # type: ignore[arg-type,return-value]
    assert built.cloud is False
    assert built.record == {}
    assert built.pipeline.describe()["ocr_engine"] == "pdf-text"


def test_cli_end_to_end_on_a_split_digital_pdf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Multi-page digital PDF → split → answers → import → A0R run → report."""
    sheets = [
        ("HR900000001", "900001-2026", "8PC"),
        ("HR900000002", "900002-2026", "2PC"),
        ("HQ900000003", "900003-2026", "10PC"),
    ]
    source = tmp_path / "printed.pdf"
    source.write_bytes(make_text_pdf_bytes([_sheet_lines(*sheet) for sheet in sheets]))
    root = tmp_path / "data"
    template = tmp_path / "template.csv"
    assert (
        main(
            [
                "dataset",
                "split-pdf",
                "--pdf",
                str(source),
                "--out-dir",
                str(root / "raw"),
                "--answers-template",
                str(template),
            ]
        )
        == 0
    )
    assert "Wrote 3 one-page PDFs" in capsys.readouterr().out
    assert (
        main(
            [
                "dataset",
                "split-pdf",
                "--pdf",
                str(source),
                "--out-dir",
                str(root / "x"),
                "--answers-template",
                str(template),
            ]
        )
        == 2
    )
    capsys.readouterr()

    rows = list(csv.reader(template.open(encoding="utf-8")))
    answers = tmp_path / "answers.csv"
    with answers.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(rows[0])
        for row, (part, order, quantity) in zip(rows[1:], sheets, strict=True):
            writer.writerow([row[0], "scan", "dev", CUSTOMER_HALF_WIDTH, part, order, quantity])
    assert main(["dataset", "import-answers", "--root", str(root), "--answers", str(answers)]) == 0

    runs = tmp_path / "runs"
    monkeypatch.setenv("OCRBENCH_DATA_DIR", str(root))
    monkeypatch.setenv("OCRBENCH_RUNS_DIR", str(runs))
    config = tmp_path / "a0r.json"
    config.write_text(
        json.dumps({"kind": "ocr_rules", "pipeline_id": "A0R", "ocr": {"engine": "pdf_text"}}),
        encoding="utf-8",
    )
    capsys.readouterr()
    assert (
        main(["run", "--pipeline", str(config), "--split", "dev", "--run-id", "a0r", "--json"]) == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["field_accuracy"] == 1.0
    assert payload["summary"]["silent_errors"] == 0

    (run_dir,) = list(runs.iterdir())
    assert load_pipeline_run(run_dir).manifest.pipeline["rules_version"] == RULES_VERSION
    assert main(["report", "--run-dir", str(run_dir)]) == 0
    assert "| Field accuracy | 100.0% (12/12 fields;" in (run_dir / "report.md").read_text(
        encoding="utf-8"
    )
