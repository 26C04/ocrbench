"""Layer 1 OCR engines for the two-layer pipelines (ADR-0002 §1, §6).

None of these adds a required dependency: Tesseract runs as a command-line
program, Qwen-VL goes through the existing Ollama adapter, Mistral OCR uses
the standard library's HTTP client, and PaddleOCR is imported only when used.
Error messages name the failure, never the sheet's content.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Final

from ocrbench import config
from ocrbench.config import OcrBenchError
from ocrbench.ollama_adapter import OllamaAdapter
from ocrbench.pipelines import OcrResult
from ocrbench.preprocess import PreprocessResult


class OcrEngineError(OcrBenchError):
    """Raised when an OCR engine cannot produce text for a sheet."""


class CloudNotAllowedError(OcrEngineError):
    """Raised when a cloud engine is used without every required approval."""


def _elapsed_ms(started_ns: int) -> float:
    return (perf_counter_ns() - started_ns) / 1_000_000


type Runner = Callable[..., subprocess.CompletedProcess[bytes]]


class TesseractEngine:
    """Candidate A1: the Tesseract command-line program (Japanese + English)."""

    def __init__(
        self,
        *,
        executable: str = "tesseract",
        languages: str = "jpn+eng",
        page_segmentation_mode: int = 6,
        timeout_s: float = 120.0,
        runner: Runner = subprocess.run,
    ) -> None:
        self._executable = executable
        self._languages = languages
        self._psm = page_segmentation_mode
        self._timeout_s = timeout_s
        self._run = runner
        self._version: str | None = None

    @property
    def engine_id(self) -> str:
        return "tesseract"

    def describe(self) -> dict[str, str]:
        return {
            "name": "tesseract",
            "version": self.version(),
            "languages": self._languages,
            "psm": str(self._psm),
        }

    def version(self) -> str:
        """Return the first line of ``tesseract --version``."""
        if self._version is None:
            completed = self._invoke([self._executable, "--version"], None)
            output = completed.stdout or completed.stderr
            lines = output.decode("utf-8", "replace").splitlines()
            self._version = lines[0].strip() if lines else ""
        return self._version

    def transcribe(self, image: PreprocessResult) -> OcrResult:
        started = perf_counter_ns()
        command = [
            self._executable,
            "stdin",
            "stdout",
            "-l",
            self._languages,
            "--psm",
            str(self._psm),
        ]
        completed = self._invoke(command, image.png_bytes)
        if completed.returncode != 0:
            raise OcrEngineError(f"tesseract exited with code {completed.returncode}")
        return OcrResult(
            text=completed.stdout.decode("utf-8", "replace"),
            duration_ms=_elapsed_ms(started),
        )

    def _invoke(
        self, command: list[str], stdin: bytes | None
    ) -> subprocess.CompletedProcess[bytes]:
        try:
            return self._run(
                command,
                input=stdin,
                capture_output=True,
                timeout=self._timeout_s,
                check=False,
            )
        except FileNotFoundError:
            raise OcrEngineError(
                f"tesseract not found at {self._executable!r}; see docs/windows-setup.md"
            ) from None
        except subprocess.TimeoutExpired:
            raise OcrEngineError(f"tesseract timed out after {self._timeout_s:g} s") from None


TRANSCRIBE_PROMPT: Final[str] = (
    "Transcribe every piece of text on this production instruction sheet exactly as "
    "printed, line by line, top to bottom. Do not interpret, translate, correct, or "
    'summarize anything. Return JSON of the form {"text": "..."}.'
)
_TRANSCRIBE_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
    "additionalProperties": False,
}


class QwenVlTranscribeEngine:
    """Candidate A3: a vision model used only to transcribe text, via Ollama."""

    def __init__(
        self,
        *,
        adapter: OllamaAdapter,
        model: str = "qwen2.5vl:7b",
        prompt: str = TRANSCRIBE_PROMPT,
        seed: int = config.SEED,
    ) -> None:
        self._adapter = adapter
        self._model = model
        self._prompt = prompt
        self._seed = seed

    @property
    def engine_id(self) -> str:
        return "qwen-vl"

    def describe(self) -> dict[str, str]:
        return {"name": "qwen-vl-transcribe", "model": self._model, "seed": str(self._seed)}

    def transcribe(self, image: PreprocessResult) -> OcrResult:
        started = perf_counter_ns()
        chat = self._adapter.generate_structured(
            model=self._model,
            prompt=self._prompt,
            image_png=image.png_bytes,
            format_schema=dict(_TRANSCRIBE_SCHEMA),
            seed=self._seed,
            temperature=0.0,
        )
        try:
            decoded = json.loads(chat.content)
        except json.JSONDecodeError:
            raise OcrEngineError("transcription was not JSON") from None
        text = decoded.get("text") if isinstance(decoded, dict) else None
        if not isinstance(text, str) or set(decoded) != {"text"}:
            raise OcrEngineError("transcription did not match the expected shape")
        return OcrResult(text=text, duration_ms=_elapsed_ms(started))


MISTRAL_OCR_URL: Final[str] = "https://api.mistral.ai/v1/ocr"
MISTRAL_KEY_VARIABLE: Final[str] = "MISTRAL_API_KEY"
ALLOW_CLOUD_VARIABLE: Final[str] = "OCRBENCH_ALLOW_CLOUD"
DEFAULT_MISTRAL_USD_PER_PAGE: Final[float] = 0.004
"""About USD 4 per 1,000 pages (October 2026); pass the current price explicitly."""

type HttpPost = Callable[[urllib.request.Request, float], bytes]


def _urllib_post(request: urllib.request.Request, timeout_s: float) -> bytes:
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        body: bytes = response.read()
    return body


def ensure_cloud_allowed(*, confirmed: bool) -> str:
    """Check all three cloud locks and return the API key (ADR-0002 §6).

    The locks are ``OCRBENCH_ALLOW_CLOUD=1``, a non-empty ``MISTRAL_API_KEY``,
    and ``confirmed`` (the CLI's ``--confirm-cloud``). Kimura's written
    approval must be recorded in ``docs/`` before anyone opens them.
    """
    missing: list[str] = []
    if os.environ.get(ALLOW_CLOUD_VARIABLE) != "1":
        missing.append(f"{ALLOW_CLOUD_VARIABLE}=1")
    key = os.environ.get(MISTRAL_KEY_VARIABLE, "").strip()
    if not key:
        missing.append(MISTRAL_KEY_VARIABLE)
    if not confirmed:
        missing.append("--confirm-cloud")
    if missing:
        raise CloudNotAllowedError(
            "cloud OCR is locked; missing: "
            + ", ".join(missing)
            + " (Kimura's written approval is required first)"
        )
    return key


class MistralOcrEngine:
    """Candidate A4: Mistral OCR, a paid cloud API, locked behind three conditions."""

    def __init__(
        self,
        *,
        confirmed: bool,
        model: str = "mistral-ocr-latest",
        usd_per_page: float = DEFAULT_MISTRAL_USD_PER_PAGE,
        timeout_s: float = 120.0,
        post: HttpPost = _urllib_post,
    ) -> None:
        ensure_cloud_allowed(confirmed=confirmed)
        self._confirmed = confirmed
        self._model = model
        self._usd_per_page = usd_per_page
        self._timeout_s = timeout_s
        self._post = post

    @property
    def engine_id(self) -> str:
        return "mistral-ocr"

    def describe(self) -> dict[str, str]:
        return {
            "name": "mistral-ocr",
            "model": self._model,
            "usd_per_page": f"{self._usd_per_page:g}",
        }

    def transcribe(self, image: PreprocessResult) -> OcrResult:
        key = ensure_cloud_allowed(confirmed=self._confirmed)
        started = perf_counter_ns()
        encoded = base64.b64encode(image.png_bytes).decode("ascii")
        payload = {
            "model": self._model,
            "document": {"type": "image_url", "image_url": f"data:image/png;base64,{encoded}"},
        }
        request = urllib.request.Request(
            MISTRAL_OCR_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            body = self._post(request, self._timeout_s)
        except urllib.error.HTTPError as error:
            raise OcrEngineError(f"Mistral OCR returned HTTP {error.code}") from None
        except urllib.error.URLError, TimeoutError, OSError:
            raise OcrEngineError("Mistral OCR could not be reached") from None
        try:
            decoded = json.loads(body)
            pages = decoded["pages"]
            text = "\n".join(str(page["markdown"]) for page in pages)
        except ValueError, KeyError, TypeError:
            raise OcrEngineError("Mistral OCR returned an unexpected response") from None
        return OcrResult(
            text=text,
            duration_ms=_elapsed_ms(started),
            cost_usd=self._usd_per_page * max(1, len(pages)),
        )


def _paddle_lines(result: Any) -> list[str]:
    """Collect recognized lines from PaddleOCR 3.x or 2.x result shapes."""
    lines: list[str] = []
    for page in result or []:
        texts = None
        if isinstance(page, dict) or hasattr(page, "get"):
            texts = page.get("rec_texts")
        if texts is not None:
            lines.extend(str(text) for text in texts)
            continue
        for entry in _as_iterable(page):
            if isinstance(entry, (list, tuple)) and len(entry) == 2:
                recognized = entry[1]
                if isinstance(recognized, (list, tuple)) and recognized:
                    lines.append(str(recognized[0]))
    return lines


def _as_iterable(value: Any) -> Iterable[Any]:
    return value if isinstance(value, (list, tuple)) else []


class PaddleOcrEngine:
    """Candidate A2: PaddleOCR, imported only when first used.

    PaddleOCR is not a declared dependency: install it on the benchmark
    machine as described in ``docs/windows-setup.md``.
    """

    def __init__(self, *, lang: str = "japan") -> None:
        self._lang = lang
        self._reader: Any = None
        self._version = ""

    @property
    def engine_id(self) -> str:
        return "paddleocr"

    def describe(self) -> dict[str, str]:
        return {"name": "paddleocr", "version": self._version, "lang": self._lang}

    def _load(self) -> Any:
        if self._reader is None:
            try:
                import paddleocr  # type: ignore[import-not-found]
            except ImportError:
                raise OcrEngineError(
                    "PaddleOCR is not installed; see docs/windows-setup.md"
                ) from None
            self._version = str(getattr(paddleocr, "__version__", ""))
            self._reader = paddleocr.PaddleOCR(lang=self._lang)
        return self._reader

    def transcribe(self, image: PreprocessResult) -> OcrResult:
        reader = self._load()
        started = perf_counter_ns()
        with tempfile.TemporaryDirectory(prefix="ocrbench-") as directory:
            sheet = Path(directory) / "sheet.png"
            sheet.write_bytes(image.png_bytes)
            try:
                run = reader.predict if hasattr(reader, "predict") else reader.ocr
                result = run(str(sheet))
            except Exception:
                raise OcrEngineError("PaddleOCR failed on this sheet") from None
        return OcrResult(text="\n".join(_paddle_lines(result)), duration_ms=_elapsed_ms(started))
