"""Build a :class:`~ocrbench.pipelines.Pipeline` from a JSON configuration file.

One file describes one candidate, so the configuration of every run can be
kept beside the code and recorded before the final split is run (ADR-0002 §3,
freeze rule). Examples::

    {"kind": "ocr_llm", "pipeline_id": "A1",
     "ocr": {"engine": "tesseract", "languages": "jpn+eng", "psm": 6},
     "llm_model": "qwen3:8b", "prompt_name": "edi-fields"}

    {"kind": "vision", "pipeline_id": "B", "model": "gemma4:12b",
     "prompt_name": "edi-fields-vision", "output": "edi_fields"}

    {"kind": "external", "pipeline_id": "C", "command": ["python", "harness.py"],
     "timeout_s": 120, "version": "1.0"}

    {"kind": "ocr_rules", "pipeline_id": "A0R", "ocr": {"engine": "pdf_text"}}

OCR engines are ``tesseract``, ``paddleocr``, ``qwen_vl``, ``mistral`` and
``pdf_text`` (the text layer of a digital PDF, no OCR). ``ocr_rules`` replaces
the Layer 2 LLM with the label rules in :mod:`ocrbench.label_rules`.
Mistral is a cloud engine and needs all three locks (ADR-0002 §6). Prompts
come from the prompt registry, so their content hash is recorded.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Final, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ocrbench.config import OcrBenchError
from ocrbench.label_rules import OcrRulesPipeline
from ocrbench.ocr_engines import (
    DEFAULT_LINE_MIN_RUN_PX,
    DEFAULT_MISTRAL_USD_PER_PAGE,
    GLM_OCR_PROMPT,
    TRANSCRIBE_PROMPT,
    MistralOcrEngine,
    PaddleOcrEngine,
    PdfTextLayerEngine,
    QwenVlTranscribeEngine,
    TesseractEngine,
)
from ocrbench.ollama_adapter import OllamaAdapter, TextLlm
from ocrbench.pipelines import (
    ExternalCommandPipeline,
    OcrEngine,
    OcrLlmPipeline,
    Pipeline,
    VisionPipeline,
)
from ocrbench.prompts import PromptVersion

_PIPELINE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+-]{0,63}$")
_UNION_TAGS: Final = frozenset(
    {
        "ocr_llm",
        "ocr_rules",
        "vision",
        "external",
        "tesseract",
        "paddleocr",
        "qwen_vl",
        "mistral",
        "pdf_text",
    }
)
"""Discriminator values pydantic inserts into error locations; not key names."""


class PipelineConfigError(OcrBenchError):
    """Raised when a pipeline configuration file cannot be used."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class TesseractConfig(_Strict):
    engine: Literal["tesseract"]
    executable: str = "tesseract"
    languages: str = "jpn+eng"
    psm: Annotated[int, Field(ge=0, le=13)] = 6
    timeout_s: Annotated[float, Field(gt=0)] = 120.0
    remove_lines: bool = False
    line_min_run_px: Annotated[int, Field(ge=2)] = DEFAULT_LINE_MIN_RUN_PX


class PaddleOcrConfig(_Strict):
    engine: Literal["paddleocr"]
    lang: str = "japan"
    ocr_version: str | None = None


class QwenVlConfig(_Strict):
    engine: Literal["qwen_vl"]
    model: str = "qwen2.5vl:7b"
    max_side: Annotated[int, Field(ge=448, le=8192)] | None = None
    num_ctx: Annotated[int, Field(ge=2048, le=131072)] | None = None
    keep_top: Annotated[float, Field(gt=0.1, le=1.0)] | None = None
    repeat_retry: bool = False
    output: Literal["json", "glm_ocr"] = "json"
    """``glm_ocr``: plain text with GLM-OCR's own prompt instead of the JSON transcription."""


class MistralConfig(_Strict):
    engine: Literal["mistral"]
    model: str = "mistral-ocr-latest"
    usd_per_page: Annotated[float, Field(ge=0)] = DEFAULT_MISTRAL_USD_PER_PAGE


class PdfTextConfig(_Strict):
    engine: Literal["pdf_text"]


type OcrConfig = Annotated[
    TesseractConfig | PaddleOcrConfig | QwenVlConfig | MistralConfig | PdfTextConfig,
    Field(discriminator="engine"),
]


class _PipelineBase(_Strict):
    pipeline_id: str

    @field_validator("pipeline_id")
    @classmethod
    def validate_pipeline_id(cls, value: str) -> str:
        if _PIPELINE_ID_PATTERN.fullmatch(value) is None:
            raise ValueError("pipeline_id must match ^[A-Za-z0-9][A-Za-z0-9_.:+-]{0,63}$")
        return value


class OcrLlmConfig(_PipelineBase):
    kind: Literal["ocr_llm"]
    ocr: OcrConfig
    llm_model: Annotated[str, Field(min_length=1)]
    prompt_name: Annotated[str, Field(min_length=1)]
    snap_customer: bool = False
    rules_check: bool = False
    customer_from_text: bool = False


class OcrRulesConfig(_PipelineBase):
    kind: Literal["ocr_rules"]
    ocr: OcrConfig
    snap_customer: bool = False


class VisionConfig(_PipelineBase):
    kind: Literal["vision"]
    model: Annotated[str, Field(min_length=1)]
    prompt_name: Annotated[str, Field(min_length=1)]
    output: Literal["order_document", "edi_fields"] = "order_document"
    snap_customer: bool = False


class ExternalConfig(_PipelineBase):
    kind: Literal["external"]
    command: Annotated[list[Annotated[str, Field(min_length=1)]], Field(min_length=1)]
    timeout_s: Annotated[float, Field(gt=0)] = 120.0
    version: str = ""


class _ConfigFile(_Strict):
    config: Annotated[
        OcrLlmConfig | OcrRulesConfig | VisionConfig | ExternalConfig,
        Field(discriminator="kind"),
    ]


type PipelineConfig = OcrLlmConfig | OcrRulesConfig | VisionConfig | ExternalConfig


def parse_pipeline_config(text: str) -> PipelineConfig:
    """Validate one configuration document; errors name keys, never values."""
    try:
        document = json.loads(text)
    except json.JSONDecodeError:
        raise PipelineConfigError("pipeline configuration is not valid JSON") from None
    try:
        return _ConfigFile.model_validate({"config": document}).config
    except ValidationError as error:
        locations = sorted(
            {
                ".".join(str(part) for part in item["loc"][1:] if part not in _UNION_TAGS)
                or "(root)"
                for item in error.errors()
            }
        )
        raise PipelineConfigError(
            "pipeline configuration is invalid at: " + ", ".join(locations)
        ) from None


def load_pipeline_config(path: Path) -> PipelineConfig:
    """Read and validate a pipeline configuration file."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise PipelineConfigError("cannot read the pipeline configuration as UTF-8") from error
    return parse_pipeline_config(text)


@dataclass(frozen=True)
class BuiltPipeline:
    """A ready pipeline plus what the run manifest must record beside ``describe()``."""

    pipeline: Pipeline
    record: dict[str, str]
    cloud: bool


def _ocr_engine(
    config: OcrConfig, *, adapter: Callable[[], OllamaAdapter], confirm_cloud: bool
) -> OcrEngine:
    match config:
        case TesseractConfig():
            return TesseractEngine(
                executable=config.executable,
                languages=config.languages,
                page_segmentation_mode=config.psm,
                timeout_s=config.timeout_s,
                remove_lines=config.remove_lines,
                line_min_run_px=config.line_min_run_px,
            )
        case PaddleOcrConfig():
            return PaddleOcrEngine(lang=config.lang, ocr_version=config.ocr_version)
        case QwenVlConfig():
            return QwenVlTranscribeEngine(
                adapter=adapter(),
                model=config.model,
                max_side=config.max_side,
                num_ctx=config.num_ctx,
                keep_top=config.keep_top,
                repeat_retry=config.repeat_retry,
                prompt=GLM_OCR_PROMPT if config.output == "glm_ocr" else TRANSCRIBE_PROMPT,
                plain_text=config.output == "glm_ocr",
            )
        case MistralConfig():
            return MistralOcrEngine(
                confirmed=confirm_cloud, model=config.model, usd_per_page=config.usd_per_page
            )
        case PdfTextConfig():
            return PdfTextLayerEngine()


def _prompt_record(prompt: PromptVersion) -> dict[str, str]:
    return {"prompt_name": prompt.name, "prompt_hash": prompt.hash}


def build_pipeline(
    config: PipelineConfig,
    *,
    adapter: Callable[[], OllamaAdapter],
    prompt: Callable[[str], PromptVersion],
    confirm_cloud: bool = False,
) -> BuiltPipeline:
    """Create the configured pipeline.

    ``adapter`` is called only when the pipeline needs Ollama, so an external
    command runs without a local server. Model digests are resolved here, so a
    missing model fails before any sheet is read.
    """
    match config:
        case OcrLlmConfig():
            cloud = isinstance(config.ocr, MistralConfig)
            engine = _ocr_engine(config.ocr, adapter=adapter, confirm_cloud=confirm_cloud)
            instructions = prompt(config.prompt_name)
            ollama = adapter()
            # Both Ollama adapters (real and fake) also implement TextLlm.
            llm = cast(TextLlm, ollama)
            ocr_llm = OcrLlmPipeline(
                pipeline_id=config.pipeline_id,
                ocr=engine,
                llm=llm,
                model=config.llm_model,
                instructions=instructions.text,
                snap_customer=config.snap_customer,
                rules_check=config.rules_check,
                customer_from_text=config.customer_from_text,
            )
            record = {
                **_prompt_record(instructions),
                "llm_digest": ollama.resolve_digest(config.llm_model),
            }
            return BuiltPipeline(ocr_llm, record, cloud)
        case OcrRulesConfig():
            rules = OcrRulesPipeline(
                pipeline_id=config.pipeline_id,
                ocr=_ocr_engine(config.ocr, adapter=adapter, confirm_cloud=confirm_cloud),
                snap_customer=config.snap_customer,
            )
            return BuiltPipeline(rules, {}, isinstance(config.ocr, MistralConfig))
        case VisionConfig():
            vision_prompt = prompt(config.prompt_name)
            ollama = adapter()
            vision = VisionPipeline(
                pipeline_id=config.pipeline_id,
                adapter=ollama,
                model=config.model,
                prompt=vision_prompt.text,
                output=config.output,
                snap_customer=config.snap_customer,
            )
            record = {
                **_prompt_record(vision_prompt),
                "vision_digest": ollama.resolve_digest(config.model),
            }
            return BuiltPipeline(vision, record, False)
        case ExternalConfig():
            external = ExternalCommandPipeline(
                pipeline_id=config.pipeline_id,
                command=config.command,
                timeout_s=config.timeout_s,
                version=config.version,
            )
            return BuiltPipeline(external, {}, False)


__all__ = [
    "BuiltPipeline",
    "ExternalConfig",
    "OcrLlmConfig",
    "OcrRulesConfig",
    "PipelineConfig",
    "PipelineConfigError",
    "VisionConfig",
    "build_pipeline",
    "load_pipeline_config",
    "parse_pipeline_config",
]
