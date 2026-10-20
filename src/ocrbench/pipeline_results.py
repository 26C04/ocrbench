"""Strict persistence models for ADR-0002 pipeline runs.

A pipeline run directory holds ``manifest.json`` and ``results.json``, like an
ADR-0001 run, but the documents are scored on the 4 EDI fields. The
``kind`` key tells the two formats apart. Run directories stay local: they
contain predicted values and Layer 1 text.
"""

from __future__ import annotations

import re
import stat
from pathlib import Path
from typing import Annotated, Literal, Self, cast

from pydantic import Field, ValidationError, field_validator, model_validator

from ocrbench.edi_fields import EDI_FIELD_NAMES, EdiFields, OrderNoRule
from ocrbench.field_metrics import FieldOutcome, Layer, SheetScore
from ocrbench.pipelines import PipelineStatus
from ocrbench.results import (
    ResultsError,
    _is_link_or_reparse,
    _read_json,
    _regular_artifact,
    _strict_json,
    _StrictModel,
)

PIPELINE_RUN_KIND: Literal["pipeline_run"] = "pipeline_run"
_DOC_ID_PATTERN = re.compile(r"^doc-[0-9a-f]{12}$")
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_TIMESTAMP_PATTERN = re.compile(r"^\d{8}T\d{6}Z$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_TIMING_FIELDS = ("preprocess", "layer1", "layer2", "total")

type NonNegative = Annotated[float, Field(ge=0.0)]


class PipelineRunManifest(_StrictModel):
    """Everything needed to reproduce one pipeline run (no secrets, no sheet values)."""

    kind: Literal["pipeline_run"]
    run_id: str
    started_at_utc: str
    pipeline_id: str
    pipeline: dict[str, str]
    cloud: bool
    split: Literal["dev", "final"]
    dataset_fingerprint: str
    preprocess_version: str
    order_no_rule: OrderNoRule
    code_commit: str | None
    code_dirty: bool | None
    os_info: str
    gpu_info: str | None

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, value: str) -> str:
        if _RUN_ID_PATTERN.fullmatch(value) is None:
            raise ValueError("run_id is unsafe or invalid")
        return value

    @field_validator("started_at_utc")
    @classmethod
    def validate_timestamp(cls, value: str) -> str:
        if _TIMESTAMP_PATTERN.fullmatch(value) is None:
            raise ValueError("started_at_utc must use YYYYMMDDTHHMMSSZ")
        return value

    @field_validator("dataset_fingerprint")
    @classmethod
    def validate_fingerprint(cls, value: str) -> str:
        if _SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError("dataset_fingerprint must be 64 lowercase hex characters")
        return value

    @model_validator(mode="after")
    def validate_pipeline_identity(self) -> Self:
        if self.pipeline.get("pipeline_id") != self.pipeline_id:
            raise ValueError("pipeline description does not match pipeline_id")
        return self


class SheetResult(_StrictModel):
    """One sheet's prediction, score, timings, and cost."""

    sheet_id: str
    source_kind: Literal["scan", "photo"]
    status: PipelineStatus
    fields: EdiFields | None
    layer1_text: str | None
    raw_outputs: list[str]
    outcomes: dict[str, FieldOutcome]
    failed_layer: dict[str, Layer]
    timings_ms: dict[str, NonNegative | None]
    cost_usd: NonNegative
    attempts: Annotated[int, Field(ge=0)]
    error: Annotated[str, Field(min_length=1, max_length=500)] | None

    @field_validator("sheet_id")
    @classmethod
    def validate_sheet_id(cls, value: str) -> str:
        if _DOC_ID_PATTERN.fullmatch(value) is None:
            raise ValueError("sheet_id must be an anonymous doc-<12 hex> identifier")
        return value

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        if tuple(self.outcomes) != EDI_FIELD_NAMES:
            raise ValueError("outcomes must list the 4 EDI fields in order")
        for field in self.failed_layer:
            if self.outcomes.get(field) in (None, FieldOutcome.CORRECT):
                raise ValueError("failed_layer may name only fields that are not correct")
        if self.failed_layer and self.layer1_text is None:
            raise ValueError("failed_layer requires Layer 1 text")
        if set(self.timings_ms) != set(_TIMING_FIELDS) or self.timings_ms["total"] is None:
            raise ValueError("timings_ms must hold preprocess, layer1, layer2 and total")
        if self.status is PipelineStatus.OK and self.fields is None:
            raise ValueError("an ok sheet must have fields")
        failed = self.status in (PipelineStatus.FORMAT_ERROR, PipelineStatus.ERROR)
        if failed and self.fields is not None:
            raise ValueError("a format_error or error sheet must not have fields")
        return self

    def score(self) -> SheetScore:
        """Rebuild the :class:`~ocrbench.field_metrics.SheetScore` used for metrics."""
        return SheetScore(
            sheet_id=self.sheet_id,
            source_kind=self.source_kind,
            status=self.status,
            outcomes=dict(self.outcomes),
            failed_layer=dict(self.failed_layer),
            total_ms=cast(float, self.timings_ms["total"]),
            cost_usd=self.cost_usd,
        )


class PipelineRunResults(_StrictModel):
    """The complete ``results.json`` of a pipeline run."""

    schema_version: Literal[1]
    kind: Literal["pipeline_run"]
    manifest: PipelineRunManifest
    sheets: list[SheetResult]

    @field_validator("sheets")
    @classmethod
    def validate_sheets(cls, value: list[SheetResult]) -> list[SheetResult]:
        ids = [sheet.sheet_id for sheet in value]
        if not ids:
            raise ValueError("a run must contain at least one sheet")
        if len(set(ids)) != len(ids):
            raise ValueError("sheet IDs must be unique")
        if ids != sorted(ids):
            raise ValueError("sheets must be sorted by sheet_id")
        return value

    def scores(self) -> list[SheetScore]:
        return [sheet.score() for sheet in self.sheets]


def _run_directory(run_dir: Path) -> Path:
    directory = Path(run_dir)
    try:
        if _is_link_or_reparse(directory):
            raise ResultsError("run directory must not be a symlink or reparse alias")
        details = directory.stat(follow_symlinks=False)
    except FileNotFoundError as error:
        raise ResultsError("run directory does not exist") from error
    except ResultsError:
        raise
    except OSError as error:
        raise ResultsError("cannot inspect run directory") from error
    if not stat.S_ISDIR(details.st_mode):
        raise ResultsError("run path is not a directory")
    return directory


def is_pipeline_run(run_dir: Path) -> bool:
    """Return whether ``results.json`` declares a pipeline run; never raises."""
    try:
        directory = _run_directory(run_dir)
        document = _read_json(_regular_artifact(directory, "results.json"), "results.json")
    except ResultsError:
        return False
    return isinstance(document, dict) and document.get("kind") == PIPELINE_RUN_KIND


def load_pipeline_run(run_dir: Path) -> PipelineRunResults:
    """Load a pipeline run after strict path, schema, and cross-file checks."""
    directory = _run_directory(run_dir)
    manifest_document = _read_json(_regular_artifact(directory, "manifest.json"), "manifest.json")
    results_document = _read_json(_regular_artifact(directory, "results.json"), "results.json")
    try:
        manifest = PipelineRunManifest.model_validate_json(_strict_json(manifest_document))
        results = PipelineRunResults.model_validate_json(_strict_json(results_document))
    except ValidationError as error:
        raise ResultsError("run artifacts do not match the pipeline results schema") from error
    if results.manifest != manifest:
        raise ResultsError("manifest.json does not match results.json manifest")
    if directory.name != f"{manifest.started_at_utc}_{manifest.run_id}":
        raise ResultsError("run directory name does not match its manifest identity")
    return results


__all__ = [
    "PIPELINE_RUN_KIND",
    "PipelineRunManifest",
    "PipelineRunResults",
    "SheetResult",
    "is_pipeline_run",
    "load_pipeline_run",
]
