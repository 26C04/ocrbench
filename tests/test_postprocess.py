"""Tests for the optional customer snap and label-rules cross-check."""

from __future__ import annotations

import json

from ocrbench.customers import find_site_in_text, snap_customer
from ocrbench.edi_fields import EdiFields
from ocrbench.ollama_adapter import FakeOllamaAdapter
from ocrbench.pipelines import (
    OcrLlmPipeline,
    OcrResult,
    PipelineStatus,
    VisionPipeline,
    describe_pipeline,
)
from ocrbench.postprocess import cross_check, customer_from_text
from ocrbench.preprocess import PreprocessResult

NIDEC = "Nidecマシンツール株式会社"
IMAGE = PreprocessResult(
    png_bytes=b"png", width=1, height=1, source_kind="image", duration_ms=0.0, version="v1"
)
TEXT = "受注先 Nidecﾏｼﾝﾂ-ﾙ株式会社 送り先 外注課\n部番 HR000000001\n受注No 000001-2026\n個数 4PC"


def _fields(**overrides: object) -> EdiFields:
    values: dict[str, object] = {
        "customer": NIDEC,
        "part_no": "HR000000001",
        "order_no": "000001-2026",
        "quantity": 4,
    }
    return EdiFields.model_validate({**values, **overrides})


def test_snap_customer_fixes_near_misses_only() -> None:
    assert snap_customer("Nidecﾏｼﾝﾂ-ﾙ株式会社") == NIDEC
    assert snap_customer(NIDEC) == NIDEC
    assert snap_customer("山田大作") == "山田大作"
    assert snap_customer(None) is None


def test_cross_check_prefers_agreement_then_the_printed_format() -> None:
    llm = _fields(part_no="COLLAR", order_no="1224272", quantity=4)
    assert cross_check(llm, _fields()) == _fields()
    assert cross_check(_fields(part_no=None), _fields()).part_no == "HR000000001"
    assert cross_check(_fields(), _fields(order_no=None)).order_no == "000001-2026"
    both_fit = cross_check(_fields(part_no="HR000000009"), _fields())
    assert both_fit.part_no is None
    assert cross_check(_fields(quantity=5), _fields()).quantity is None
    unknown = cross_check(_fields(customer="X", part_no="A"), _fields(customer="X", part_no="B"))
    assert unknown.part_no is None


class _Ocr:
    engine_id = "text"

    def describe(self) -> dict[str, str]:
        return {"name": "text"}

    def transcribe(self, image: PreprocessResult) -> OcrResult:
        return OcrResult(text=TEXT, duration_ms=1.0)


def test_ocr_llm_options_rescue_a_wrong_box_answer() -> None:
    wrong = {
        "customer": "Nidecﾏｼﾝﾂ-ﾙ株式会社",
        "part_no": "COLLAR",
        "order_no": "000001-2026",
        "quantity": 4,
    }
    plain = OcrLlmPipeline(
        pipeline_id="A3",
        ocr=_Ocr(),
        llm=FakeOllamaAdapter([json.dumps(wrong)]),
        model="m",
        instructions="i",
    )
    assert plain.extract(IMAGE).status is PipelineStatus.NEEDS_REVIEW
    assert "rules_check" not in describe_pipeline(plain)

    checked = OcrLlmPipeline(
        pipeline_id="A3X",
        ocr=_Ocr(),
        llm=FakeOllamaAdapter([json.dumps(wrong)]),
        model="m",
        instructions="i",
        snap_customer=True,
        rules_check=True,
    )
    result = checked.extract(IMAGE)
    assert result.status is PipelineStatus.OK
    assert result.fields == _fields()
    assert describe_pipeline(checked)["rules_check"] == "on"


def test_vision_snap_customer() -> None:
    near = {**_fields().model_dump(), "customer": "Nidecﾏｼﾝﾂｰﾙ"}
    result = VisionPipeline(
        pipeline_id="BX",
        adapter=FakeOllamaAdapter([json.dumps(near)]),
        model="m",
        prompt="p",
        output="edi_fields",
        snap_customer=True,
    ).extract(IMAGE)
    assert result.status is PipelineStatus.OK
    assert result.fields is not None and result.fields.customer == NIDEC


def test_find_site_in_text_finds_the_name_inside_a_longer_line() -> None:
    assert find_site_in_text(TEXT) == "hwao_nmtj"
    assert find_site_in_text("お客様 NIDEC ﾏｼﾝﾂｰﾙ(株) 御中\n個数 4") == "hwao_nmtj"
    assert find_site_in_text("受注先 山田製作所株式会社\n部番 HR000000001") is None
    assert find_site_in_text("") is None


def test_customer_from_text_only_fills_an_unmapped_customer() -> None:
    assert customer_from_text(_fields(customer="社"), TEXT).customer == NIDEC
    assert customer_from_text(_fields(customer=None), TEXT).customer == NIDEC
    assert customer_from_text(_fields(customer="X"), "個数 4").customer == "X"
    assert customer_from_text(_fields(), "") == _fields()


def test_ocr_llm_customer_from_text_option() -> None:
    lost = {**_fields().model_dump(), "customer": "木村工業"}
    pipeline = OcrLlmPipeline(
        pipeline_id="A3HT",
        ocr=_Ocr(),
        llm=FakeOllamaAdapter([json.dumps(lost)]),
        model="m",
        instructions="i",
        customer_from_text=True,
    )
    result = pipeline.extract(IMAGE)
    assert result.fields is not None and result.fields.customer == NIDEC
    assert describe_pipeline(pipeline)["customer_from_text"] == "on"
