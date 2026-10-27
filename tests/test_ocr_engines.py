"""Tests for the Layer 1 OCR engines, with every external program or service faked."""

from __future__ import annotations

import http.client
import importlib.metadata
import io
import json
import shutil
import subprocess
import sys
import types
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
from PIL import Image, ImageDraw, ImageFont

from ocrbench import config
from ocrbench.ocr_engines import (
    DEFAULT_MISTRAL_USD_PER_PAGE,
    GLM_OCR_PROMPT,
    MISTRAL_OCR_URL,
    TRANSCRIBE_PROMPT,
    CloudNotAllowedError,
    MistralOcrEngine,
    OcrEngineError,
    PaddleOcrEngine,
    QwenVlTranscribeEngine,
    TesseractEngine,
    enlarge,
    ensure_cloud_allowed,
    fit_long_side,
    keep_top_part,
    remove_ruling_lines,
)
from ocrbench.ollama_adapter import ChatResult, FakeOllamaAdapter
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


def _tessdata(tmp_path: Path, *langs: str) -> subprocess.CompletedProcess[bytes]:
    for lang in langs:
        (tmp_path / f"{lang}.traineddata").write_bytes(f"model-{lang}".encode())
    listing = f'List of available languages in "{tmp_path}/" ({len(langs)}):\n'
    return _done((listing + "\n".join(langs) + "\n").encode())


def test_tesseract_describe_records_version_and_language_data_once(tmp_path: Path) -> None:
    runner = _FakeRunner(
        _done(b"tesseract 5.3.4\n leptonica-1.82.0\n"), _tessdata(tmp_path, "jpn", "eng")
    )
    engine = TesseractEngine(runner=runner)

    description = engine.describe()

    assert description["version"] == "tesseract 5.3.4"
    assert description["languages"] == "jpn+eng"
    assert description["psm"] == "6"
    assert description["data.jpn"].startswith("sha256:")
    assert description["data.jpn"] != description["data.eng"]
    assert engine.describe() == description
    assert [call[0][1] for call in runner.calls] == ["--version", "--list-langs"]
    assert engine.engine_id == "tesseract"


def test_tesseract_language_data_changes_with_the_file(tmp_path: Path) -> None:
    first = TesseractEngine(languages="jpn", runner=_FakeRunner(_tessdata(tmp_path, "jpn")))
    before = first.language_data()["jpn"]
    (tmp_path / "jpn.traineddata").write_bytes(b"best model")
    second = TesseractEngine(languages="jpn", runner=_FakeRunner(_tessdata(tmp_path)))

    assert second.language_data()["jpn"] != before


@pytest.mark.parametrize(
    ("outcome", "message"),
    [
        (_done(b"", returncode=1), "tesseract version unavailable"),
        (_done(b"   \n", returncode=0), "tesseract version unavailable"),
    ],
)
def test_tesseract_version_failures_raise_and_are_not_cached(
    outcome: subprocess.CompletedProcess[bytes], message: str
) -> None:
    runner = _FakeRunner(outcome, _done(b"tesseract 5.3.4\n"))
    engine = TesseractEngine(runner=runner)

    with pytest.raises(OcrEngineError, match=message):
        engine.version()
    assert engine.version() == "tesseract 5.3.4"


def test_tesseract_language_data_failures(tmp_path: Path) -> None:
    missing = TesseractEngine(languages="jpn", runner=_FakeRunner(_tessdata(tmp_path)))
    with pytest.raises(OcrEngineError, match="language data 'jpn' is not installed"):
        missing.language_data()

    no_folder = TesseractEngine(runner=_FakeRunner(_done(b"jpn\neng\n")))
    with pytest.raises(OcrEngineError, match="did not report its tessdata folder"):
        no_folder.language_data()


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
    assert engine.describe() == {
        "name": "qwen-vl-transcribe",
        "model": "qwen2.5vl:7b",
        "digest": "sha256:fake",
        "seed": str(config.SEED),
        "max_side": "original",
        "num_ctx": "default",
    }
    assert engine.engine_id == "qwen-vl"
    assert adapter.num_ctx_calls == [None]


def test_qwen_vl_shrinks_large_images_and_sets_the_context() -> None:
    big = Image.new("RGB", (3000, 2000), "white")
    buffer = io.BytesIO()
    big.save(buffer, format="PNG")
    page = PreprocessResult(
        png_bytes=buffer.getvalue(),
        width=3000,
        height=2000,
        source_kind="image",
        duration_ms=0.0,
        version="v1",
    )
    adapter = FakeOllamaAdapter([json.dumps({"text": "t"})])
    engine = QwenVlTranscribeEngine(adapter=adapter, max_side=1500, num_ctx=12288)

    engine.transcribe(page)

    assert adapter.num_ctx_calls == [12288]
    assert engine.describe()["max_side"] == "1500"
    with Image.open(io.BytesIO(fit_long_side(page.png_bytes, 1500))) as shrunk:
        assert shrunk.size == (1500, 1000)
    assert fit_long_side(page.png_bytes, 4000) == page.png_bytes
    assert fit_long_side(page.png_bytes, None) == page.png_bytes


def _png(width: int, height: int) -> PreprocessResult:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buffer, format="PNG")
    return PreprocessResult(
        png_bytes=buffer.getvalue(),
        width=width,
        height=height,
        source_kind="image",
        duration_ms=0.0,
        version="v1",
    )


def test_keep_top_part_crops_to_the_header() -> None:
    page = _png(1000, 1400)
    with Image.open(io.BytesIO(keep_top_part(page.png_bytes, 0.4))) as top:
        assert top.size == (1000, 560)
    assert keep_top_part(page.png_bytes, None) == page.png_bytes
    assert keep_top_part(page.png_bytes, 1.0) == page.png_bytes


class _LoopingOnce(FakeOllamaAdapter):
    """Fail the first call the way Ollama aborts a looping generation."""

    def __init__(self, message: str) -> None:
        super().__init__([json.dumps({"text": "t"})])
        self.message = message

    def generate_structured(self, **kwargs: Any) -> ChatResult:
        if not self.repeat_penalty_calls:
            self.repeat_penalty_calls.append(kwargs.get("repeat_penalty"))
            raise RuntimeError(self.message)
        return super().generate_structured(**kwargs)


def test_qwen_vl_retries_once_with_a_penalty_after_a_repeat_abort() -> None:
    adapter = _LoopingOnce("prediction aborted, token repeat limit reached (status code: 500)")
    engine = QwenVlTranscribeEngine(adapter=adapter, keep_top=0.4, repeat_retry=True)

    assert engine.transcribe(_png(100, 100)).text == "t"
    assert adapter.repeat_penalty_calls == [None, 1.15]
    assert engine.describe()["keep_top"] == "0.4"
    assert engine.describe()["repeat_retry"] == "1.15"


@pytest.mark.parametrize(
    ("retry", "message"), [(False, "token repeat limit reached"), (True, "connection refused")]
)
def test_qwen_vl_raises_other_failures_without_retrying(retry: bool, message: str) -> None:
    adapter = _LoopingOnce(message)
    engine = QwenVlTranscribeEngine(adapter=adapter, repeat_retry=retry)

    with pytest.raises(RuntimeError):
        engine.transcribe(_png(10, 10))
    assert adapter.repeat_penalty_calls == [None]
    assert "repeat_retry" not in engine.describe() or retry


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
        (http.client.IncompleteRead(b"partial"), "could not be reached"),
    ],
)
def test_mistral_network_errors_hide_details(error: BaseException, message: str) -> None:
    def post(request: urllib.request.Request, timeout_s: float) -> bytes:
        raise error

    with pytest.raises(OcrEngineError, match=message) as caught:
        MistralOcrEngine(confirmed=True, post=post).transcribe(IMAGE)
    assert "test-key-123" not in str(caught.value)
    assert caught.value.cost_usd == 0.0


@pytest.mark.usefixtures("cloud_env")
@pytest.mark.parametrize("body", [b"not json", b"{}", b'{"pages": [{"index": 0}]}'])
def test_mistral_unexpected_responses(body: bytes) -> None:
    engine = MistralOcrEngine(confirmed=True, post=lambda request, timeout_s: body)

    with pytest.raises(OcrEngineError, match="unexpected response") as caught:
        engine.transcribe(IMAGE)
    assert caught.value.cost_usd == DEFAULT_MISTRAL_USD_PER_PAGE


class _FakeArray:
    def __init__(self, image: Image.Image) -> None:
        self.first_pixel = image.getpixel((0, 0))


def _fake_paddle(
    monkeypatch: pytest.MonkeyPatch, result: object, *, v3: bool = True, broken: bool = False
) -> tuple[list[object], list[dict[str, str]]]:
    inputs: list[object] = []
    constructed: list[dict[str, str]] = []

    class Reader:
        def __init__(self, **options: str) -> None:
            constructed.append(options)
            if broken:
                raise OSError("model download failed")

        def ocr(self, image: object) -> object:
            inputs.append(image)
            if isinstance(result, Exception):
                raise result
            return result

    if v3:
        Reader.predict = Reader.ocr  # type: ignore[attr-defined]
    module = types.ModuleType("paddleocr")
    module.PaddleOCR = Reader  # type: ignore[attr-defined]
    numpy = types.ModuleType("numpy")
    numpy.asarray = _FakeArray  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "paddleocr", module)
    monkeypatch.setitem(sys.modules, "numpy", numpy)
    return inputs, constructed


def _red_png() -> PreprocessResult:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), (255, 0, 0)).save(buffer, format="PNG")
    return PreprocessResult(buffer.getvalue(), 4, 4, "image", 0.0, "v1")


def test_paddle_reads_v3_result_from_an_in_memory_bgr_array(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs, constructed = _fake_paddle(monkeypatch, [{"rec_texts": ["受注先", "Nidec"]}])
    engine = PaddleOcrEngine(ocr_version="PP-OCRv5")

    assert engine.transcribe(_red_png()).text == "受注先\nNidec"
    array = inputs[0]
    assert isinstance(array, _FakeArray)
    assert array.first_pixel == (0, 0, 255)
    assert constructed == [{"lang": "japan", "ocr_version": "PP-OCRv5"}]
    assert engine.engine_id == "paddleocr"


def test_paddle_describe_needs_no_model_load(monkeypatch: pytest.MonkeyPatch) -> None:
    versions = {"paddleocr": "3.0.0"}

    def fake_version(package: str) -> str:
        if package not in versions:
            raise importlib.metadata.PackageNotFoundError(package)
        return versions[package]

    monkeypatch.setattr(importlib.metadata, "version", fake_version)

    assert PaddleOcrEngine().describe() == {
        "name": "paddleocr",
        "version": "3.0.0",
        "paddlepaddle": "not installed",
        "lang": "japan",
        "ocr_version": "default",
    }


def test_paddle_reads_v2_result_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    box = [[0, 0], [1, 0], [1, 1], [0, 1]]
    _fake_paddle(monkeypatch, [[[box, ("部番", 0.99)], [box, ("HR237101021", 0.98)]]], v3=False)

    assert PaddleOcrEngine().transcribe(_red_png()).text == "部番\nHR237101021"


def test_paddle_sheet_failure_hides_details(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_paddle(monkeypatch, RuntimeError("023235 leaked"))

    with pytest.raises(OcrEngineError, match="PaddleOCR failed on this sheet") as caught:
        PaddleOcrEngine().transcribe(_red_png())
    assert "023235" not in str(caught.value)


def test_paddle_start_failure_is_wrapped_and_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    _, constructed = _fake_paddle(monkeypatch, [], broken=True)
    engine = PaddleOcrEngine()

    for _ in range(2):
        with pytest.raises(OcrEngineError, match="PaddleOCR could not start"):
            engine.transcribe(_red_png())
    assert len(constructed) == 1


def test_paddle_missing_install(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "paddleocr", None)

    with pytest.raises(OcrEngineError, match="PaddleOCR is not installed"):
        PaddleOcrEngine().transcribe(_red_png())


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


def _ruled_png() -> bytes:
    page = Image.new("L", (400, 200), 230)
    draw = ImageDraw.Draw(page)
    draw.rectangle((20, 20, 380, 180), outline=0, width=3)
    draw.line((20, 100, 380, 100), fill=0, width=3)
    draw.line((200, 120, 230, 120), fill=0, width=4)  # a short stroke, like part of a glyph
    output = io.BytesIO()
    page.save(output, format="PNG")
    return output.getvalue()


def test_remove_ruling_lines_erases_long_lines_and_keeps_short_strokes() -> None:
    with Image.open(io.BytesIO(remove_ruling_lines(_ruled_png()))) as cleaned:
        assert cleaned.mode == "L"
        pixel = cast(Callable[[tuple[int, int]], int], cleaned.getpixel)
        assert pixel((300, 100)) > 200  # horizontal rule
        assert pixel((21, 60)) > 200  # vertical rule
        assert pixel((215, 120)) < 50  # short stroke stays
    with pytest.raises(ValueError, match="min_run_px"):
        remove_ruling_lines(_ruled_png(), min_run_px=1)


def test_tesseract_remove_lines_option_changes_the_input_and_is_recorded(
    tmp_path: Path,
) -> None:
    image = PreprocessResult(_ruled_png(), 400, 200, "image", 0.5, "v1")
    runner = _FakeRunner(_done(b"text"), _done(b"tesseract 5.3.4\n"), _tessdata(tmp_path, "jpn"))
    engine = TesseractEngine(languages="jpn", remove_lines=True, line_min_run_px=40, runner=runner)

    engine.transcribe(image)

    sent = runner.calls[0][1]["input"]
    assert sent != image.png_bytes
    assert sent == remove_ruling_lines(image.png_bytes, min_run_px=40)
    assert engine.describe()["remove_lines"] == "min_run_px=40"
    assert TesseractEngine(runner=_FakeRunner()).__dict__["_remove_lines"] is False


def test_glm_ocr_plain_text_skips_the_json_schema() -> None:
    adapter = FakeOllamaAdapter(["受注先 Nidec\n個数 4"])
    engine = QwenVlTranscribeEngine(
        adapter=adapter, model="glm-ocr", prompt=GLM_OCR_PROMPT, plain_text=True
    )

    assert engine.transcribe(_png(10, 10)).text == "受注先 Nidec\n個数 4"
    assert adapter.calls[0].schema == {}
    assert adapter.calls[0].prompt == GLM_OCR_PROMPT
    assert engine.describe()["output"] == "plain_text"
    with pytest.raises(OcrEngineError, match="empty"):
        QwenVlTranscribeEngine(adapter=FakeOllamaAdapter(["  "]), plain_text=True).transcribe(
            _png(10, 10)
        )


def test_upscale_enlarges_the_header_before_reading() -> None:
    adapter = FakeOllamaAdapter(["text"])
    engine = QwenVlTranscribeEngine(adapter=adapter, plain_text=True, keep_top=0.5, upscale=2.0)
    page = _png(100, 200)

    engine.transcribe(page)

    assert adapter.calls[0].image_size == len(enlarge(keep_top_part(page.png_bytes, 0.5), 2.0))
    with Image.open(io.BytesIO(enlarge(page.png_bytes, 2.0))) as big:
        assert big.size == (200, 400)
    assert enlarge(page.png_bytes, None) == page.png_bytes
    assert engine.describe()["upscale"] == "2.0"
