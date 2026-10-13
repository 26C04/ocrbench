"""Tests for the Layer 1 OCR engines, with every external program or service faked."""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import sys
import types
import urllib.error
import urllib.request
from typing import Any

import pytest
from PIL import Image, ImageDraw, ImageFont

from ocrbench import config
from ocrbench.ocr_engines import (
    DEFAULT_MISTRAL_USD_PER_PAGE,
    MISTRAL_OCR_URL,
    TRANSCRIBE_PROMPT,
    CloudNotAllowedError,
    MistralOcrEngine,
    OcrEngineError,
    PaddleOcrEngine,
    QwenVlTranscribeEngine,
    TesseractEngine,
    ensure_cloud_allowed,
)
from ocrbench.ollama_adapter import FakeOllamaAdapter
from ocrbench.preprocess import PreprocessResult

IMAGE = PreprocessResult(
    png_bytes=b"\x89PNG-fake",
    width=10,
    height=10,
    source_kind="image",
    duration_ms=0.5,
    version="v1",
)


class _FakeRunner:
    def __init__(self, *results: subprocess.CompletedProcess[bytes] | BaseException) -> None:
        self.results = list(results)
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def __call__(self, command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        self.calls.append((command, kwargs))
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


def _done(stdout: bytes = b"", returncode: int = 0) -> subprocess.CompletedProcess[bytes]:
    return subprocess.CompletedProcess([], returncode, stdout=stdout, stderr=b"")


def test_tesseract_sends_png_on_stdin_and_reads_stdout() -> None:
    runner = _FakeRunner(_done("受注No 023235\n".encode()))
    engine = TesseractEngine(runner=runner, languages="jpn", page_segmentation_mode=4)

    result = engine.transcribe(IMAGE)

    assert result.text == "受注No 023235\n"
    assert result.cost_usd == 0.0
    command, kwargs = runner.calls[0]
    assert command == ["tesseract", "stdin", "stdout", "-l", "jpn", "--psm", "4"]
    assert kwargs["input"] == IMAGE.png_bytes
    assert kwargs["check"] is False


def test_tesseract_describe_reads_version_once() -> None:
    runner = _FakeRunner(_done(b"tesseract 5.3.4\n leptonica-1.82.0\n"))
    engine = TesseractEngine(runner=runner)

    assert engine.describe() == {
        "name": "tesseract",
        "version": "tesseract 5.3.4",
        "languages": "jpn+eng",
        "psm": "6",
    }
    assert engine.describe()["version"] == "tesseract 5.3.4"
    assert len(runner.calls) == 1
    assert engine.engine_id == "tesseract"


@pytest.mark.parametrize(
    ("outcome", "message"),
    [
        (_done(returncode=1), "tesseract exited with code 1"),
        (FileNotFoundError(), "tesseract not found at 'tesseract'"),
        (subprocess.TimeoutExpired("tesseract", 5), "tesseract timed out after 120 s"),
    ],
)
def test_tesseract_failures(
    outcome: subprocess.CompletedProcess[bytes] | BaseException, message: str
) -> None:
    engine = TesseractEngine(runner=_FakeRunner(outcome))

    with pytest.raises(OcrEngineError, match=message):
        engine.transcribe(IMAGE)


def test_qwen_vl_transcribes_with_fixed_prompt_and_seed() -> None:
    adapter = FakeOllamaAdapter([json.dumps({"text": "部番 HR237101021"})])
    engine = QwenVlTranscribeEngine(adapter=adapter)

    result = engine.transcribe(IMAGE)

    assert result.text == "部番 HR237101021"
    call = adapter.calls[0]
    assert call.model == "qwen2.5vl:7b"
    assert call.prompt == TRANSCRIBE_PROMPT
    assert call.seed == config.SEED
    assert call.temperature == 0.0
    assert call.image_size == len(IMAGE.png_bytes)
    assert call.schema["required"] == ["text"]
    assert engine.describe()["model"] == "qwen2.5vl:7b"
    assert engine.engine_id == "qwen-vl"


@pytest.mark.parametrize("content", ["not json", "[]", '{"text": 1}', '{"text": "a", "extra": 1}'])
def test_qwen_vl_rejects_malformed_transcriptions(content: str) -> None:
    engine = QwenVlTranscribeEngine(adapter=FakeOllamaAdapter([content]))

    with pytest.raises(OcrEngineError):
        engine.transcribe(IMAGE)


@pytest.fixture
def cloud_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OCRBENCH_ALLOW_CLOUD", "1")
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key-123")


@pytest.mark.parametrize(
    ("allow", "key", "confirmed", "missing"),
    [
        (None, "k", True, "OCRBENCH_ALLOW_CLOUD=1"),
        ("true", "k", True, "OCRBENCH_ALLOW_CLOUD=1"),
        ("1", None, True, "MISTRAL_API_KEY"),
        ("1", "  ", True, "MISTRAL_API_KEY"),
        ("1", "k", False, "--confirm-cloud"),
    ],
)
def test_each_missing_cloud_lock_refuses(
    monkeypatch: pytest.MonkeyPatch,
    allow: str | None,
    key: str | None,
    confirmed: bool,
    missing: str,
) -> None:
    for name, value in (("OCRBENCH_ALLOW_CLOUD", allow), ("MISTRAL_API_KEY", key)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)

    with pytest.raises(CloudNotAllowedError, match=missing):
        ensure_cloud_allowed(confirmed=confirmed)
    with pytest.raises(CloudNotAllowedError):
        MistralOcrEngine(confirmed=confirmed, post=_never_post)


def _never_post(request: urllib.request.Request, timeout_s: float) -> bytes:
    raise AssertionError("must not reach the network")


@pytest.mark.usefixtures("cloud_env")
def test_mistral_posts_png_and_charges_per_page() -> None:
    sent: list[urllib.request.Request] = []

    def post(request: urllib.request.Request, timeout_s: float) -> bytes:
        sent.append(request)
        return json.dumps({"pages": [{"index": 0, "markdown": "受注先 Nidec"}]}).encode()

    engine = MistralOcrEngine(confirmed=True, post=post)
    result = engine.transcribe(IMAGE)

    assert result.text == "受注先 Nidec"
    assert result.cost_usd == DEFAULT_MISTRAL_USD_PER_PAGE
    request = sent[0]
    assert request.full_url == MISTRAL_OCR_URL
    assert request.get_header("Authorization") == "Bearer test-key-123"
    body = json.loads(request.data)  # type: ignore[arg-type]
    assert body["model"] == "mistral-ocr-latest"
    assert body["document"]["image_url"].startswith("data:image/png;base64,")
    description = engine.describe()
    assert "test-key-123" not in json.dumps(description)
    assert description["usd_per_page"] == "0.004"
    assert engine.engine_id == "mistral-ocr"


@pytest.mark.usefixtures("cloud_env")
def test_mistral_rechecks_locks_before_each_call(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = MistralOcrEngine(confirmed=True, post=_never_post)
    monkeypatch.setenv("OCRBENCH_ALLOW_CLOUD", "0")

    with pytest.raises(CloudNotAllowedError):
        engine.transcribe(IMAGE)


@pytest.mark.usefixtures("cloud_env")
@pytest.mark.parametrize(
    ("error", "message"),
    [
        (urllib.error.HTTPError(MISTRAL_OCR_URL, 401, "no", {}, None), "HTTP 401"),  # type: ignore[arg-type]
        (urllib.error.URLError("down"), "could not be reached"),
        (TimeoutError(), "could not be reached"),
    ],
)
def test_mistral_network_errors_hide_details(error: BaseException, message: str) -> None:
    def post(request: urllib.request.Request, timeout_s: float) -> bytes:
        raise error

    with pytest.raises(OcrEngineError, match=message) as caught:
        MistralOcrEngine(confirmed=True, post=post).transcribe(IMAGE)
    assert "test-key-123" not in str(caught.value)


@pytest.mark.usefixtures("cloud_env")
@pytest.mark.parametrize("body", [b"not json", b"{}", b'{"pages": [{"index": 0}]}'])
def test_mistral_unexpected_responses(body: bytes) -> None:
    engine = MistralOcrEngine(confirmed=True, post=lambda request, timeout_s: body)

    with pytest.raises(OcrEngineError, match="unexpected response"):
        engine.transcribe(IMAGE)


def _fake_paddle(monkeypatch: pytest.MonkeyPatch, result: object, *, v3: bool = True) -> None:
    class Reader:
        def __init__(self, lang: str) -> None:
            self.lang = lang

        def ocr(self, path: str) -> object:
            if isinstance(result, Exception):
                raise result
            return result

    if v3:
        Reader.predict = Reader.ocr  # type: ignore[attr-defined]
    module = types.ModuleType("paddleocr")
    module.PaddleOCR = Reader  # type: ignore[attr-defined]
    module.__version__ = "3.0.0"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "paddleocr", module)


def test_paddle_reads_v3_result_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_paddle(monkeypatch, [{"rec_texts": ["受注先", "Nidec"]}])
    engine = PaddleOcrEngine()

    assert engine.transcribe(IMAGE).text == "受注先\nNidec"
    assert engine.describe() == {"name": "paddleocr", "version": "3.0.0", "lang": "japan"}
    assert engine.engine_id == "paddleocr"


def test_paddle_reads_v2_result_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    box = [[0, 0], [1, 0], [1, 1], [0, 1]]
    _fake_paddle(monkeypatch, [[[box, ("部番", 0.99)], [box, ("HR237101021", 0.98)]]], v3=False)

    assert PaddleOcrEngine().transcribe(IMAGE).text == "部番\nHR237101021"


def test_paddle_failure_and_missing_install(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_paddle(monkeypatch, RuntimeError("023235 leaked"))
    with pytest.raises(OcrEngineError, match="PaddleOCR failed on this sheet"):
        PaddleOcrEngine().transcribe(IMAGE)

    monkeypatch.setitem(sys.modules, "paddleocr", None)
    with pytest.raises(OcrEngineError, match="PaddleOCR is not installed"):
        PaddleOcrEngine().transcribe(IMAGE)


def _png_with_text(text: str) -> bytes:
    image = Image.new("RGB", (1400, 160), "white")
    ImageDraw.Draw(image).text((40, 50), text, fill="black", font=ImageFont.load_default(56))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.mark.ocr_engines
@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract is not installed")
def test_real_tesseract_reads_a_part_number() -> None:
    png = _png_with_text("HR237101021 023235-2025")
    image = PreprocessResult(
        png_bytes=png, width=1400, height=160, source_kind="image", duration_ms=0.0, version="v1"
    )

    result = TesseractEngine(languages="eng").transcribe(image)

    assert "HR237101021" in result.text
