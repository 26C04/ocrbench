"""Layer 1 OCR engines for the two-layer pipelines (ADR-0002 §1, §6).

None of these adds a required dependency: Tesseract runs as a command-line
program, Qwen-VL goes through the existing Ollama adapter, Mistral OCR uses
the standard library's HTTP client, and PaddleOCR is imported only when used.
Error messages name the failure, never the sheet's content.
"""

from __future__ import annotations

import base64
import hashlib
import http.client
import importlib
import importlib.metadata
import io
import json
import os
import re
import subprocess
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Final

from PIL import Image, ImageChops, ImageFilter

from ocrbench import config
from ocrbench.config import OcrBenchError
from ocrbench.ollama_adapter import OllamaAdapter
from ocrbench.pipelines import OcrResult, elapsed_ms
from ocrbench.preprocess import PreprocessResult


class OcrEngineError(OcrBenchError):
    """Raised when an OCR engine cannot produce text for a sheet.

    ``cost_usd`` is what the failed call was still billed, so a paid engine's
    cost per sheet stays correct when its response cannot be used.
    """

    def __init__(self, message: str, *, cost_usd: float = 0.0) -> None:
        super().__init__(message)
        self.cost_usd = cost_usd


class CloudNotAllowedError(OcrEngineError):
    """Raised when a cloud engine is used without every required approval."""


_TESSDATA_PATTERN = re.compile(r'"([^"]+)"')

type Runner = Callable[..., subprocess.CompletedProcess[bytes]]

DEFAULT_LINE_MIN_RUN_PX: Final[int] = 60
_LINE_INK_THRESHOLD: Final[int] = 160


def remove_ruling_lines(png_bytes: bytes, *, min_run_px: int = DEFAULT_LINE_MIN_RUN_PX) -> bytes:
    """Erase table ruling lines so Tesseract can read text inside the cells.

    Tesseract's layout analysis skips text enclosed by ruled boxes. A pixel
    belongs to a line when it lies in a dark horizontal or vertical run of at
    least ``min_run_px`` pixels; those pixels are filled with the local
    background brightness. Glyph strokes are shorter than the run length, so
    text is kept. Pillow only, no extra dependency; returns a grayscale PNG.
    """
    if min_run_px < 2:
        raise ValueError("min_run_px must be at least 2")
    with Image.open(io.BytesIO(png_bytes)) as source:
        gray = source.convert("L")
    width, height = gray.size
    dark = gray.point(lambda value: 255 if value < _LINE_INK_THRESHOLD else 0)

    def long_runs(horizontal: bool) -> Image.Image:
        # Two grids, the second shifted by half a block, so line ends keep shorter stubs.
        mask = Image.new("L", (width, height), 0)
        for offset in (0, min_run_px // 2):
            box = (offset, 0, width, height) if horizontal else (0, offset, width, height)
            part = dark.crop(box)
            size = (
                (max(1, part.width // min_run_px), part.height)
                if horizontal
                else (part.width, max(1, part.height // min_run_px))
            )
            solid = part.resize(size, Image.Resampling.BOX).point(
                lambda value: 255 if value >= 230 else 0
            )
            layer = Image.new("L", (width, height), 0)
            layer.paste(solid.resize(part.size, Image.Resampling.NEAREST), box[:2])
            mask = ImageChops.lighter(mask, layer)
        return mask

    horizontal = long_runs(True)
    vertical = long_runs(False)
    mask = ImageChops.lighter(horizontal, vertical).filter(ImageFilter.MaxFilter(3))
    small = gray.resize((max(1, width // 8), max(1, height // 8)), Image.Resampling.BOX)
    background = small.filter(ImageFilter.MaxFilter(5)).resize(
        (width, height), Image.Resampling.BILINEAR
    )
    output = io.BytesIO()
    Image.composite(background, gray, mask).save(output, format="PNG")
    return output.getvalue()


class TesseractEngine:
    """Candidate A1: the Tesseract command-line program (Japanese + English)."""

    def __init__(
        self,
        *,
        executable: str = "tesseract",
        languages: str = "jpn+eng",
        page_segmentation_mode: int = 6,
        timeout_s: float = 120.0,
        remove_lines: bool = False,
        line_min_run_px: int = DEFAULT_LINE_MIN_RUN_PX,
        runner: Runner = subprocess.run,
    ) -> None:
        self._executable = executable
        self._languages = languages
        self._psm = page_segmentation_mode
        self._remove_lines = remove_lines
        self._line_min_run_px = line_min_run_px
        self._timeout_s = timeout_s
        self._run = runner
        self._version: str | None = None
        self._language_data: dict[str, str] | None = None

    @property
    def engine_id(self) -> str:
        return "tesseract"

    def describe(self) -> dict[str, str]:
        return {
            "name": "tesseract",
            "version": self.version(),
            "languages": self._languages,
            "psm": str(self._psm),
            "remove_lines": (
                f"min_run_px={self._line_min_run_px}" if self._remove_lines else "off"
            ),
            **{f"data.{lang}": digest for lang, digest in self.language_data().items()},
        }

    def version(self) -> str:
        """Return the first line of ``tesseract --version``."""
        if self._version is None:
            lines = self._lines([self._executable, "--version"], "version")
            self._version = lines[0].strip()
        return self._version

    def language_data(self) -> dict[str, str]:
        """Return a short SHA-256 of each ``.traineddata`` file in use.

        Engine version alone does not identify the model: swapping the fast
        and best Japanese data changes accuracy without changing the version.
        """
        if self._language_data is None:
            lines = self._lines([self._executable, "--list-langs"], "language list")
            match = _TESSDATA_PATTERN.search(lines[0])
            if match is None:
                raise OcrEngineError("tesseract did not report its tessdata folder")
            folder = Path(match.group(1))
            data: dict[str, str] = {}
            for lang in self._languages.split("+"):
                path = folder / f"{lang}.traineddata"
                if not path.is_file():
                    raise OcrEngineError(f"tesseract language data {lang!r} is not installed")
                data[lang] = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()[:12]
            self._language_data = data
        return self._language_data

    def _lines(self, command: list[str], what: str) -> list[str]:
        completed = self._invoke(command, None)
        lines = completed.stdout.decode("utf-8", "replace").splitlines()
        if completed.returncode != 0 or not lines or not lines[0].strip():
            raise OcrEngineError(f"tesseract {what} unavailable; see docs/windows-setup.md")
        return lines

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
        png = (
            remove_ruling_lines(image.png_bytes, min_run_px=self._line_min_run_px)
            if self._remove_lines
            else image.png_bytes
        )
        completed = self._invoke(command, png)
        if completed.returncode != 0:
            raise OcrEngineError(f"tesseract exited with code {completed.returncode}")
        return OcrResult(
            text=completed.stdout.decode("utf-8", "replace"),
            duration_ms=elapsed_ms(started),
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
        self._digest: str | None = None

    @property
    def engine_id(self) -> str:
        return "qwen-vl"

    def describe(self) -> dict[str, str]:
        if self._digest is None:
            self._digest = self._adapter.resolve_digest(self._model)
        return {
            "name": "qwen-vl-transcribe",
            "model": self._model,
            "digest": self._digest,
            "seed": str(self._seed),
        }

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
        return OcrResult(text=text, duration_ms=elapsed_ms(started))


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
        except OSError, http.client.HTTPException:
            raise OcrEngineError("Mistral OCR could not be reached") from None
        try:
            decoded = json.loads(body)
            pages = decoded["pages"]
            text = "\n".join(str(page["markdown"]) for page in pages)
        except ValueError, KeyError, TypeError:
            # A successful HTTP response was billed even though it is unusable.
            raise OcrEngineError(
                "Mistral OCR returned an unexpected response", cost_usd=self._usd_per_page
            ) from None
        return OcrResult(
            text=text,
            duration_ms=elapsed_ms(started),
            cost_usd=self._usd_per_page * max(1, len(pages)),
        )


def _paddle_lines(result: Any) -> list[str]:
    """Collect recognized lines from PaddleOCR 3.x or 2.x result shapes."""
    lines: list[str] = []
    for page in result or []:
        texts = page.get("rec_texts") if hasattr(page, "get") else None
        if texts is not None:
            lines.extend(str(text) for text in texts)
            continue
        for entry in page if isinstance(page, (list, tuple)) else ():
            if isinstance(entry, (list, tuple)) and len(entry) == 2:
                recognized = entry[1]
                if isinstance(recognized, (list, tuple)) and recognized:
                    lines.append(str(recognized[0]))
    return lines


def _installed_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


class PaddleOcrEngine:
    """Candidate A2: PaddleOCR, imported only when first used.

    PaddleOCR is not a declared dependency: install it on the benchmark
    machine as described in ``docs/windows-setup.md``. Sheets are passed in
    memory as BGR arrays (PaddleOCR's OpenCV convention), never written to disk.
    """

    def __init__(self, *, lang: str = "japan", ocr_version: str | None = None) -> None:
        self._lang = lang
        self._ocr_version = ocr_version
        self._reader: Any = None
        self._numpy: Any = None
        self._load_error: str | None = None

    @property
    def engine_id(self) -> str:
        return "paddleocr"

    def describe(self) -> dict[str, str]:
        return {
            "name": "paddleocr",
            "version": _installed_version("paddleocr"),
            "paddlepaddle": _installed_version("paddlepaddle"),
            "lang": self._lang,
            "ocr_version": self._ocr_version or "default",
        }

    def _load(self) -> Any:
        if self._load_error is not None:
            raise OcrEngineError(self._load_error)
        if self._reader is None:
            try:
                paddleocr = importlib.import_module("paddleocr")
                self._numpy = importlib.import_module("numpy")
            except ImportError:
                self._load_error = "PaddleOCR is not installed; see docs/windows-setup.md"
                raise OcrEngineError(self._load_error) from None
            options = {"lang": self._lang}
            if self._ocr_version is not None:
                options["ocr_version"] = self._ocr_version
            try:
                self._reader = paddleocr.PaddleOCR(**options)
            except Exception:
                self._load_error = (
                    "PaddleOCR could not start (models or GPU runtime); see docs/windows-setup.md"
                )
                raise OcrEngineError(self._load_error) from None
        return self._reader

    def transcribe(self, image: PreprocessResult) -> OcrResult:
        reader = self._load()
        started = perf_counter_ns()
        with Image.open(io.BytesIO(image.png_bytes)) as decoded:
            red, green, blue = decoded.convert("RGB").split()
            bgr = self._numpy.asarray(Image.merge("RGB", (blue, green, red)))
        try:
            run = reader.predict if hasattr(reader, "predict") else reader.ocr
            result = run(bgr)
        except Exception:
            raise OcrEngineError("PaddleOCR failed on this sheet") from None
        return OcrResult(text="\n".join(_paddle_lines(result)), duration_ms=elapsed_ms(started))
