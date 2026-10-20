"""Candidate pipelines that turn one preprocessed sheet into the 4 EDI fields.

ADR-0002 §1: every candidate implements :class:`Pipeline`, so the evaluator
can score two-layer OCR → LLM pipelines, a single vision model, and external
programs such as the template harness with the same code.
"""

from __future__ import annotations

import copy
import json
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Final, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from ocrbench import config
from ocrbench.edi_fields import EdiFields, edi_format_schema, parse_edi_output
from ocrbench.ollama_adapter import ChatResult, OllamaAdapter, TextLlm
from ocrbench.parsing import ParseStatus, parse_model_output
from ocrbench.preprocess import PreprocessResult
from ocrbench.schema import OrderDocument, ollama_format_schema

_TEMPERATURE: Final[float] = 0.0
_RETRY_STATUSES: Final[frozenset[ParseStatus]] = frozenset(
    {ParseStatus.NOT_JSON, ParseStatus.SCHEMA_VIOLATION}
)
OCR_TEXT_HEADER: Final[str] = "--- OCR text ---"
"""Line separating the Layer 2 instructions from the Layer 1 text."""


class PipelineStatus(StrEnum):
    """Outcome of one pipeline run on one sheet."""

    OK = "ok"
    NEEDS_REVIEW = "needs_review"
    FORMAT_ERROR = "format_error"
    ERROR = "error"


@dataclass(frozen=True)
class PipelineResult:
    """Fields, Layer 1 text, timings, and cost from one pipeline run.

    ``fields`` is ``None`` only for ``FORMAT_ERROR`` and ``ERROR``. ``error``
    holds a short, content-free reason so it never leaks sheet values.
    """

    status: PipelineStatus
    fields: EdiFields | None
    layer1_text: str | None
    layer1_ms: float | None
    layer2_ms: float | None
    total_ms: float
    cost_usd: float
    attempts: int
    raw_outputs: tuple[str, ...] = ()
    error: str | None = None


class Pipeline(Protocol):
    """A candidate that extracts the 4 EDI fields from one sheet image."""

    @property
    def pipeline_id(self) -> str:
        """Stable identifier, e.g. ``A1`` or ``tesseract+qwen3:8b``."""

    def describe(self) -> dict[str, str]:
        """Configuration recorded in the run manifest (no secrets)."""

    def extract(self, image: PreprocessResult) -> PipelineResult:
        """Run the pipeline on one preprocessed sheet."""


@dataclass(frozen=True)
class OcrResult:
    """Raw text from one Layer 1 run."""

    text: str
    duration_ms: float
    cost_usd: float = 0.0


class OcrEngine(Protocol):
    """Layer 1: image to raw text, without interpreting meaning."""

    @property
    def engine_id(self) -> str:
        """Stable identifier, e.g. ``tesseract``."""

    def describe(self) -> dict[str, str]:
        """Engine name and versions recorded in the run manifest (no secrets)."""

    def transcribe(self, image: PreprocessResult) -> OcrResult:
        """Return the text found on one sheet."""


def layer2_prompt(instructions: str, ocr_text: str) -> str:
    """Join the Layer 2 instructions and the Layer 1 text into one message."""
    return f"{instructions.rstrip()}\n\n{OCR_TEXT_HEADER}\n{ocr_text}"


def elapsed_ms(started_ns: int) -> float:
    """Milliseconds since a ``perf_counter_ns`` start; shared by every timed layer."""
    return (perf_counter_ns() - started_ns) / 1_000_000


def _status_for(fields: EdiFields) -> PipelineStatus:
    values = (fields.customer, fields.part_no, fields.order_no, fields.quantity)
    return PipelineStatus.OK if all(v is not None for v in values) else PipelineStatus.NEEDS_REVIEW


def _error_name(error: BaseException) -> str:
    return type(error).__name__


class OcrLlmPipeline:
    """Two-layer candidate: an OCR engine followed by a local text LLM."""

    def __init__(
        self,
        *,
        pipeline_id: str,
        ocr: OcrEngine,
        llm: TextLlm,
        model: str,
        instructions: str,
        seed: int = config.SEED,
    ) -> None:
        self._pipeline_id = pipeline_id
        self._ocr = ocr
        self._llm = llm
        self._model = model
        self._instructions = instructions
        self._seed = seed

    @property
    def pipeline_id(self) -> str:
        return self._pipeline_id

    def describe(self) -> dict[str, str]:
        details = {f"ocr.{key}": value for key, value in self._ocr.describe().items()}
        return {
            "kind": "ocr_llm",
            "pipeline_id": self._pipeline_id,
            "ocr_engine": self._ocr.engine_id,
            **details,
            "llm_model": self._model,
            "seed": str(self._seed),
        }

    def extract(self, image: PreprocessResult) -> PipelineResult:
        started = perf_counter_ns()
        try:
            ocr = self._ocr.transcribe(image)
        except Exception as caught:
            return PipelineResult(
                status=PipelineStatus.ERROR,
                fields=None,
                layer1_text=None,
                layer1_ms=elapsed_ms(started),
                layer2_ms=None,
                total_ms=elapsed_ms(started),
                cost_usd=float(getattr(caught, "cost_usd", 0.0)),
                attempts=0,
                error=f"layer 1 failed: {_error_name(caught)}",
            )

        layer2_started = perf_counter_ns()
        prompt = layer2_prompt(self._instructions, ocr.text)
        schema = edi_format_schema()
        raw_outputs: list[str] = []
        error: str | None = None
        fields: EdiFields | None = None
        for _attempt in range(2):
            try:
                chat = self._llm.generate_structured_text(
                    model=self._model,
                    prompt=prompt,
                    format_schema=copy.deepcopy(schema),
                    seed=self._seed,
                    temperature=_TEMPERATURE,
                )
            except Exception as caught:
                error = f"layer 2 failed: {_error_name(caught)}"
                break
            raw_outputs.append(chat.content)
            parsed = parse_edi_output(chat.content)
            if parsed.status not in _RETRY_STATUSES:
                fields = parsed.fields
                break

        if error is not None:
            status = PipelineStatus.ERROR
        elif fields is None:
            status = PipelineStatus.FORMAT_ERROR
        else:
            status = _status_for(fields)
        return PipelineResult(
            status=status,
            fields=fields,
            layer1_text=ocr.text,
            layer1_ms=ocr.duration_ms,
            layer2_ms=elapsed_ms(layer2_started),
            total_ms=elapsed_ms(started),
            cost_usd=ocr.cost_usd,
            attempts=len(raw_outputs) + (1 if error is not None else 0),
            raw_outputs=tuple(raw_outputs),
            error=error,
        )


def order_document_to_edi(document: OrderDocument) -> tuple[EdiFields, PipelineStatus]:
    """Map the ADR-0001 order document onto the 4 EDI fields.

    One sheet carries one shipment line. A document with several items cannot
    be mapped without guessing, so its part number and quantity are left empty
    and the sheet is marked for review.
    """
    item = document.items[0] if len(document.items) == 1 else None
    fields = EdiFields(
        customer=document.customer_name,
        part_no=item.part_no if item is not None else None,
        order_no=document.order_no,
        quantity=item.num_pieces if item is not None else None,
    )
    if len(document.items) > 1:
        return fields, PipelineStatus.NEEDS_REVIEW
    return fields, _status_for(fields)


class VisionPipeline:
    """Single-stage candidate B: a vision model reads the image directly."""

    def __init__(
        self,
        *,
        pipeline_id: str,
        adapter: OllamaAdapter,
        model: str,
        prompt: str,
        seed: int = config.SEED,
    ) -> None:
        self._pipeline_id = pipeline_id
        self._adapter = adapter
        self._model = model
        self._prompt = prompt
        self._seed = seed

    @property
    def pipeline_id(self) -> str:
        return self._pipeline_id

    def describe(self) -> dict[str, str]:
        return {
            "kind": "vision",
            "pipeline_id": self._pipeline_id,
            "vision_model": self._model,
            "seed": str(self._seed),
        }

    def extract(self, image: PreprocessResult) -> PipelineResult:
        started = perf_counter_ns()
        schema = ollama_format_schema()
        raw_outputs: list[str] = []
        error: str | None = None
        document: OrderDocument | None = None
        for _attempt in range(2):
            try:
                chat: ChatResult = self._adapter.generate_structured(
                    model=self._model,
                    prompt=self._prompt,
                    image_png=image.png_bytes,
                    format_schema=copy.deepcopy(schema),
                    seed=self._seed,
                    temperature=_TEMPERATURE,
                )
            except Exception as caught:
                error = f"vision model failed: {_error_name(caught)}"
                break
            raw_outputs.append(chat.content)
            parsed = parse_model_output(chat.content)
            if parsed.status not in _RETRY_STATUSES:
                document = parsed.document
                break

        fields: EdiFields | None = None
        if error is not None:
            status = PipelineStatus.ERROR
        elif document is None:
            status = PipelineStatus.FORMAT_ERROR
        else:
            fields, status = order_document_to_edi(document)
        elapsed = elapsed_ms(started)
        return PipelineResult(
            status=status,
            fields=fields,
            layer1_text=None,
            layer1_ms=None,
            layer2_ms=elapsed,
            total_ms=elapsed,
            cost_usd=0.0,
            attempts=len(raw_outputs) + (1 if error is not None else 0),
            raw_outputs=tuple(raw_outputs),
            error=error,
        )


class _ExternalOutput(BaseModel):
    """The one JSON object an external pipeline prints on stdout."""

    model_config = ConfigDict(extra="forbid", strict=True)

    status: Literal["ok", "needs_review"]
    fields: EdiFields | None
    layer1_text: str | None = None
    cost_usd: float = Field(default=0.0, ge=0.0)


class ExternalCommandPipeline:
    """Candidate implemented by another program, such as the template harness.

    The command is run as ``[*command, <path to sheet.png>]``. It must print
    exactly one JSON object on stdout::

        {"status": "ok" | "needs_review",
         "fields": {"customer": ..., "part_no": ..., "order_no": ..., "quantity": ...} | null,
         "layer1_text": "..." | null,
         "cost_usd": 0.0}

    ``layer1_text`` and ``cost_usd`` are optional. A non-zero exit code, a
    timeout, or any other output is an ``ERROR``. stdout and stderr are never
    copied into results, because they may contain sheet values.
    """

    def __init__(
        self,
        *,
        pipeline_id: str,
        command: Sequence[str],
        timeout_s: float = 120.0,
        version: str = "",
    ) -> None:
        if not command:
            raise ValueError("command must not be empty")
        self._pipeline_id = pipeline_id
        self._command = tuple(command)
        self._timeout_s = timeout_s
        self._version = version

    @property
    def pipeline_id(self) -> str:
        return self._pipeline_id

    def describe(self) -> dict[str, str]:
        return {
            "kind": "external",
            "pipeline_id": self._pipeline_id,
            "command": self._command[0],
            "version": self._version,
        }

    def extract(self, image: PreprocessResult) -> PipelineResult:
        started = perf_counter_ns()
        with tempfile.TemporaryDirectory(prefix="ocrbench-") as directory:
            sheet = Path(directory) / "sheet.png"
            sheet.write_bytes(image.png_bytes)
            try:
                completed = subprocess.run(
                    [*self._command, str(sheet)],
                    capture_output=True,
                    timeout=self._timeout_s,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return self._error(started, f"timed out after {self._timeout_s:g} s")
            except OSError as caught:
                return self._error(started, f"could not start: {_error_name(caught)}")

        if completed.returncode != 0:
            return self._error(started, f"exit code {completed.returncode}")
        try:
            output = _ExternalOutput.model_validate(json.loads(completed.stdout))
        except Exception:
            return self._error(started, "invalid output")

        if output.status == "ok" and output.fields is None:
            return self._error(started, "invalid output")
        status = (
            PipelineStatus.NEEDS_REVIEW
            if output.status == "needs_review" or output.fields is None
            else _status_for(output.fields)
        )
        elapsed = elapsed_ms(started)
        return PipelineResult(
            status=status,
            fields=output.fields,
            layer1_text=output.layer1_text,
            layer1_ms=None,
            layer2_ms=None,
            total_ms=elapsed,
            cost_usd=output.cost_usd,
            attempts=1,
        )

    def _error(self, started: int, reason: str) -> PipelineResult:
        return PipelineResult(
            status=PipelineStatus.ERROR,
            fields=None,
            layer1_text=None,
            layer1_ms=None,
            layer2_ms=None,
            total_ms=elapsed_ms(started),
            cost_usd=0.0,
            attempts=1,
            error=reason,
        )


@dataclass
class FakeOcrEngine:
    """Deterministic Layer 1 for tests: returns scripted texts in order."""

    texts: Sequence[str]
    cost_usd: float = 0.0
    engine_id: str = "fake-ocr"
    calls: list[int] = field(default_factory=list)

    def describe(self) -> dict[str, str]:
        return {"name": self.engine_id, "version": "fake"}

    def transcribe(self, image: PreprocessResult) -> OcrResult:
        index = len(self.calls)
        if index >= len(self.texts):
            raise AssertionError("FakeOcrEngine scripted texts are exhausted")
        self.calls.append(len(image.png_bytes))
        return OcrResult(text=self.texts[index], duration_ms=1.0, cost_usd=self.cost_usd)


@dataclass
class FakePipeline:
    """Deterministic pipeline for runner and report tests."""

    results: Sequence[PipelineResult]
    pipeline_id: str = "fake"
    calls: int = 0

    def describe(self) -> dict[str, str]:
        return {"kind": "fake", "pipeline_id": self.pipeline_id}

    def extract(self, image: PreprocessResult) -> PipelineResult:
        del image
        if self.calls >= len(self.results):
            raise AssertionError("FakePipeline scripted results are exhausted")
        result = self.results[self.calls]
        self.calls += 1
        return result


def describe_pipeline(pipeline: Pipeline) -> dict[str, Any]:
    """Return a JSON-ready description, sorted for deterministic manifests."""
    return dict(sorted(pipeline.describe().items()))
