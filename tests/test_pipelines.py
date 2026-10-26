"""Tests for the ADR-0002 pipeline interface and its adapters (no network)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from ocrbench import config
from ocrbench.edi_fields import EdiFields, edi_format_schema
from ocrbench.ollama_adapter import FakeOllamaAdapter
from ocrbench.pipelines import (
    OCR_TEXT_HEADER,
    ExternalCommandPipeline,
    FakeOcrEngine,
    FakePipeline,
    OcrLlmPipeline,
    OcrResult,
    Pipeline,
    PipelineResult,
    PipelineStatus,
    VisionPipeline,
    describe_pipeline,
    layer2_prompt,
    order_document_to_edi,
)
from ocrbench.preprocess import PreprocessResult
from ocrbench.schema import LineItem, OrderDocument

IMAGE = PreprocessResult(
    png_bytes=b"\x89PNG-fake",
    width=10,
    height=10,
    source_kind="image",
    duration_ms=0.5,
    version="v1",
)
FULL = {
    "customer": "Nidecマシンツール株式会社",
    "part_no": "HR237101021",
    "order_no": "023235",
    "quantity": 1,
}


def _ocr_llm(
    responses: list[str], *, cost: float = 0.0
) -> tuple[OcrLlmPipeline, FakeOllamaAdapter]:
    llm = FakeOllamaAdapter(responses)
    pipeline = OcrLlmPipeline(
        pipeline_id="A1",
        ocr=FakeOcrEngine(["受注先 Nidec 部番 HR237101021"], cost_usd=cost),
        llm=llm,
        model="qwen3:8b",
        instructions="Extract the 4 fields.\n",
    )
    return pipeline, llm


def test_layer2_prompt_separates_instructions_from_ocr_text() -> None:
    assert layer2_prompt("Do it.\n\n", "abc") == f"Do it.\n\n{OCR_TEXT_HEADER}\nabc"


def test_ocr_llm_success_uses_text_call_schema_seed_and_ocr_cost() -> None:
    pipeline, llm = _ocr_llm([json.dumps(FULL)], cost=0.004)

    result = pipeline.extract(IMAGE)

    assert result.status is PipelineStatus.OK
    assert result.fields == EdiFields.model_validate(FULL)
    assert result.layer1_text == "受注先 Nidec 部番 HR237101021"
    assert result.cost_usd == 0.004
    assert result.attempts == 1
    assert result.layer1_ms == 1.0
    assert result.layer2_ms is not None
    call = llm.calls[0]
    assert call.image_size == 0
    assert call.schema == edi_format_schema()
    assert call.seed == config.SEED
    assert call.temperature == 0.0
    assert call.prompt.endswith(f"{OCR_TEXT_HEADER}\n受注先 Nidec 部番 HR237101021")


def test_ocr_llm_retries_once_after_a_format_error() -> None:
    pipeline, llm = _ocr_llm(["```json{}```", json.dumps(FULL)])

    result = pipeline.extract(IMAGE)

    assert result.status is PipelineStatus.OK
    assert result.attempts == 2
    assert result.raw_outputs == ("```json{}```", json.dumps(FULL))
    assert llm.calls[0].prompt == llm.calls[1].prompt


def test_ocr_llm_reports_format_error_after_two_failures() -> None:
    pipeline, _ = _ocr_llm(["not json", json.dumps({**FULL, "extra": 1})])

    result = pipeline.extract(IMAGE)

    assert result.status is PipelineStatus.FORMAT_ERROR
    assert result.fields is None
    assert result.attempts == 2


def test_ocr_llm_marks_null_fields_for_review() -> None:
    pipeline, _ = _ocr_llm([json.dumps({**FULL, "quantity": None})])

    result = pipeline.extract(IMAGE)

    assert result.status is PipelineStatus.NEEDS_REVIEW
    assert result.fields is not None
    assert result.fields.quantity is None


class _BrokenOcr:
    engine_id = "broken"

    def describe(self) -> dict[str, str]:
        return {"name": "broken"}

    def transcribe(self, image: PreprocessResult) -> OcrResult:
        raise RuntimeError("secret sheet value 023235")


class _BrokenLlm:
    def generate_structured_text(self, **kwargs: object) -> object:
        raise ConnectionError("secret sheet value 023235")


def test_ocr_llm_layer1_error_does_not_leak_message() -> None:
    pipeline = OcrLlmPipeline(
        pipeline_id="A1", ocr=_BrokenOcr(), llm=FakeOllamaAdapter(), model="m", instructions="i"
    )

    result = pipeline.extract(IMAGE)

    assert result.status is PipelineStatus.ERROR
    assert result.error == "layer 1 failed: RuntimeError"
    assert result.attempts == 0
    assert result.cost_usd == 0.0


class _BilledOcr(_BrokenOcr):
    def transcribe(self, image: PreprocessResult) -> OcrResult:
        from ocrbench.ocr_engines import OcrEngineError

        raise OcrEngineError("unusable response", cost_usd=0.004)


def test_ocr_llm_layer1_error_keeps_the_billed_cost() -> None:
    pipeline = OcrLlmPipeline(
        pipeline_id="A4", ocr=_BilledOcr(), llm=FakeOllamaAdapter(), model="m", instructions="i"
    )

    result = pipeline.extract(IMAGE)

    assert result.status is PipelineStatus.ERROR
    assert result.cost_usd == 0.004


def test_ocr_llm_layer2_error_keeps_layer1_text() -> None:
    pipeline = OcrLlmPipeline(
        pipeline_id="A1",
        ocr=FakeOcrEngine(["text"]),
        llm=_BrokenLlm(),  # type: ignore[arg-type]
        model="m",
        instructions="i",
    )

    result = pipeline.extract(IMAGE)

    assert result.status is PipelineStatus.ERROR
    assert result.error == "layer 2 failed: ConnectionError"
    assert result.layer1_text == "text"
    assert result.attempts == 1


def test_ocr_llm_describe_includes_engine_and_model() -> None:
    pipeline, _ = _ocr_llm([])

    assert describe_pipeline(pipeline) == {
        "kind": "ocr_llm",
        "llm_model": "qwen3:8b",
        "ocr.name": "fake-ocr",
        "ocr.version": "fake",
        "ocr_engine": "fake-ocr",
        "pipeline_id": "A1",
        "seed": str(config.SEED),
    }


def _document(items: list[LineItem]) -> OrderDocument:
    return OrderDocument(
        customer_name="Nidecマシンツール株式会社",
        order_no="023235",
        delivery_date=None,
        items=items,
    )


def test_order_document_with_one_item_maps_to_edi_fields() -> None:
    fields, status = order_document_to_edi(
        _document([LineItem(part_no="HR237101021", material="SS400", num_pieces=1)])
    )

    assert fields == EdiFields.model_validate(FULL)
    assert status is PipelineStatus.OK


@pytest.mark.parametrize("count", [0, 2])
def test_order_document_without_exactly_one_item_needs_review(count: int) -> None:
    items = [LineItem(part_no=f"P{i}", material=None, num_pieces=1) for i in range(count)]

    fields, status = order_document_to_edi(_document(items))

    assert status is PipelineStatus.NEEDS_REVIEW
    assert (fields.part_no, fields.quantity) == (None, None)
    assert fields.order_no == "023235"


def test_vision_pipeline_sends_image_and_maps_result() -> None:
    document = _document([LineItem(part_no="HR237101021", material=None, num_pieces=1)])
    adapter = FakeOllamaAdapter(["oops", document.model_dump_json()])
    pipeline = VisionPipeline(pipeline_id="B", adapter=adapter, model="gemma4:12b", prompt="p")

    result = pipeline.extract(IMAGE)

    assert result.status is PipelineStatus.OK
    assert result.fields == EdiFields.model_validate(FULL)
    assert result.attempts == 2
    assert result.layer1_text is None
    assert [call.image_size for call in adapter.calls] == [len(IMAGE.png_bytes)] * 2
    assert describe_pipeline(pipeline)["vision_model"] == "gemma4:12b"


def test_vision_pipeline_format_error_and_adapter_error() -> None:
    bad = VisionPipeline(
        pipeline_id="B", adapter=FakeOllamaAdapter(["x", "y"]), model="m", prompt="p"
    )
    assert bad.extract(IMAGE).status is PipelineStatus.FORMAT_ERROR

    broken = VisionPipeline(pipeline_id="B", adapter=FakeOllamaAdapter([]), model="m", prompt="p")
    result = broken.extract(IMAGE)
    assert result.status is PipelineStatus.ERROR
    assert result.error == "vision model failed: AssertionError"


def test_vision_pipeline_edi_fields_output_reads_the_4_fields_directly() -> None:
    adapter = FakeOllamaAdapter(["oops", json.dumps(FULL)])
    pipeline = VisionPipeline(
        pipeline_id="B", adapter=adapter, model="gemma4:12b", prompt="p", output="edi_fields"
    )

    result = pipeline.extract(IMAGE)

    assert result.status is PipelineStatus.OK
    assert result.fields == EdiFields.model_validate(FULL)
    assert result.attempts == 2
    assert adapter.calls[0].schema == edi_format_schema()
    assert describe_pipeline(pipeline)["output"] == "edi_fields"


def test_vision_pipeline_edi_fields_output_flags_missing_and_failures() -> None:
    partial = {**FULL, "quantity": None}
    flagged = VisionPipeline(
        pipeline_id="B",
        adapter=FakeOllamaAdapter([json.dumps(partial)]),
        model="m",
        prompt="p",
        output="edi_fields",
    ).extract(IMAGE)
    assert flagged.status is PipelineStatus.NEEDS_REVIEW
    assert flagged.fields is not None and flagged.fields.quantity is None

    bad = VisionPipeline(
        pipeline_id="B",
        adapter=FakeOllamaAdapter(["x", "y"]),
        model="m",
        prompt="p",
        output="edi_fields",
    )
    assert bad.extract(IMAGE).status is PipelineStatus.FORMAT_ERROR

    broken = VisionPipeline(
        pipeline_id="B", adapter=FakeOllamaAdapter([]), model="m", prompt="p", output="edi_fields"
    ).extract(IMAGE)
    assert broken.status is PipelineStatus.ERROR
    assert broken.error == "vision model failed: AssertionError"


def test_vision_pipeline_sends_a_badly_formatted_value_to_review() -> None:
    wrong_box = {**FULL, "part_no": "COLLAR"}
    result = VisionPipeline(
        pipeline_id="B",
        adapter=FakeOllamaAdapter([json.dumps(wrong_box)]),
        model="m",
        prompt="p",
        output="edi_fields",
    ).extract(IMAGE)
    assert result.status is PipelineStatus.NEEDS_REVIEW
    assert result.fields is not None and result.fields.part_no == "COLLAR"


def _script(tmp_path: Path, body: str) -> list[str]:
    script = tmp_path / "harness.py"
    script.write_text(body, encoding="utf-8")
    return [sys.executable, str(script)]


def _external(tmp_path: Path, body: str, timeout_s: float = 30.0) -> ExternalCommandPipeline:
    return ExternalCommandPipeline(
        pipeline_id="C", command=_script(tmp_path, body), timeout_s=timeout_s, version="cf186ea"
    )


def test_external_command_receives_the_image_and_returns_fields(tmp_path: Path) -> None:
    body = f"""
import json, pathlib, sys
data = pathlib.Path(sys.argv[1]).read_bytes()
assert data == {IMAGE.png_bytes!r}
print(json.dumps({{"status": "ok", "fields": {FULL!r}, "layer1_text": "t", "cost_usd": 0.5}}))
"""
    result = _external(tmp_path, body).extract(IMAGE)

    assert result.status is PipelineStatus.OK
    assert result.fields == EdiFields.model_validate(FULL)
    assert result.layer1_text == "t"
    assert result.cost_usd == 0.5


def test_external_needs_review_without_fields(tmp_path: Path) -> None:
    body = 'import json; print(json.dumps({"status": "needs_review", "fields": None}))'

    result = _external(tmp_path, body).extract(IMAGE)

    assert result.status is PipelineStatus.NEEDS_REVIEW
    assert result.fields is None


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        ("import sys; print('023235 leaked', file=sys.stderr); sys.exit(3)", "exit code 3"),
        ("print('023235 not json')", "invalid output"),
        ('import json; print(json.dumps({"status": "ok", "fields": None}))', "invalid output"),
        ('import json; print(json.dumps({"status": "done", "fields": None}))', "invalid output"),
    ],
)
def test_external_failures_are_errors_without_output_content(
    tmp_path: Path, body: str, reason: str
) -> None:
    result = _external(tmp_path, body).extract(IMAGE)

    assert result.status is PipelineStatus.ERROR
    assert result.error == reason
    assert result.fields is None


def test_external_timeout(tmp_path: Path) -> None:
    result = _external(tmp_path, "import time; time.sleep(5)", timeout_s=0.2).extract(IMAGE)

    assert result.status is PipelineStatus.ERROR
    assert result.error == "timed out after 0.2 s"


def test_external_missing_program(tmp_path: Path) -> None:
    pipeline = ExternalCommandPipeline(pipeline_id="C", command=[str(tmp_path / "does-not-exist")])

    result = pipeline.extract(IMAGE)

    assert result.status is PipelineStatus.ERROR
    assert result.error is not None
    assert result.error.startswith("could not start: ")


def test_external_rejects_empty_command_and_describes_itself(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="command must not be empty"):
        ExternalCommandPipeline(pipeline_id="C", command=[])

    description = describe_pipeline(_external(tmp_path, ""))
    assert description["kind"] == "external"
    assert description["version"] == "cf186ea"


def test_fakes_satisfy_the_protocols_and_run_out_loudly() -> None:
    scripted = PipelineResult(
        status=PipelineStatus.OK,
        fields=EdiFields.model_validate(FULL),
        layer1_text=None,
        layer1_ms=None,
        layer2_ms=None,
        total_ms=1.0,
        cost_usd=0.0,
        attempts=1,
    )
    pipeline: Pipeline = FakePipeline([scripted])

    assert pipeline.extract(IMAGE) is scripted
    assert describe_pipeline(pipeline) == {"kind": "fake", "pipeline_id": "fake"}
    with pytest.raises(AssertionError, match="exhausted"):
        pipeline.extract(IMAGE)

    engine = FakeOcrEngine([])
    with pytest.raises(AssertionError, match="exhausted"):
        engine.transcribe(IMAGE)
