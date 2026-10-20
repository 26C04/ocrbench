"""Run one ADR-0002 pipeline over one split of an ``edi_fields`` dataset.

Each sheet is preprocessed, passed to :meth:`Pipeline.extract`, and scored
with :func:`ocrbench.field_metrics.score_sheet`. The run is published with
the same atomic, no-overwrite logic as :mod:`ocrbench.runner`, which stays
unchanged for ADR-0001 runs.
"""

from __future__ import annotations

import json
import os
import platform
import stat
import subprocess
import uuid
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter_ns
from typing import Final, Literal

from ocrbench.dataset import DatasetError, DocEntry, Manifest, dataset_fingerprint, load_manifest
from ocrbench.edi_fields import ORDER_NO_RULE, EdiFields
from ocrbench.field_metrics import score_sheet
from ocrbench.pipeline_results import (
    PIPELINE_RUN_KIND,
    PipelineRunManifest,
    PipelineRunResults,
    SheetResult,
)
from ocrbench.pipelines import (
    Pipeline,
    PipelineResult,
    PipelineStatus,
    describe_pipeline,
    elapsed_ms,
)
from ocrbench.preprocess import PREPROCESS_VERSION, PreprocessError, PreprocessResult, preprocess
from ocrbench.runner import (
    RunnerError,
    _check_output_path,
    _gpu_info,
    _is_link_or_reparse,
    _publish_run,
    _reject_duplicate_keys,
    _utc_now,
    _validated_run_id,
)
from ocrbench.splitguard import ensure_final_allowed

_REPOSITORY_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
_GROUND_TRUTH: Final = "edi_fields"

type PipelineSplit = Literal["dev", "final"]


def load_edi_ground_truth(data_root: Path, doc_id: str) -> EdiFields:
    """Read one anonymous GT file without following an alias or exposing its values."""
    gt_directory = data_root / "gt"
    path = gt_directory / f"{doc_id}.json"
    try:
        if _is_link_or_reparse(gt_directory) or _is_link_or_reparse(path):
            raise RunnerError(f"Ground truth is unsafe for {doc_id}")
        root = gt_directory.resolve(strict=True)
        resolved = path.resolve(strict=True)
        details = path.stat(follow_symlinks=False)
        if resolved.parent != root or not stat.S_ISREG(details.st_mode):
            raise RunnerError(f"Ground truth is unsafe for {doc_id}")
        with path.open("r", encoding="utf-8", newline="") as stream:
            value = json.load(stream, object_pairs_hook=_reject_duplicate_keys)
        return EdiFields.model_validate(value)
    except RunnerError:
        raise
    except OSError, UnicodeError, json.JSONDecodeError, ValueError:
        raise RunnerError(f"Ground truth is invalid for {doc_id}") from None


def _git(*arguments: str) -> str | None:
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=_REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=creation_flags,
        )
    except OSError, subprocess.SubprocessError:
        return None
    return completed.stdout if completed.returncode == 0 else None


def code_version() -> tuple[str | None, bool | None]:
    """Return the checked-out commit and whether tracked files differ from it.

    Both are ``None`` when Git or the repository is unavailable, for example
    in an installed package.
    """
    commit = _git("rev-parse", "HEAD")
    if commit is None or not commit.strip():
        return None, None
    status = _git("status", "--porcelain", "--untracked-files=no")
    return commit.strip(), (None if status is None else bool(status.strip()))


def _validated_dataset(
    data_root: Path, supplied: Manifest, split: PipelineSplit
) -> tuple[Manifest, str]:
    try:
        manifest = Manifest.model_validate(supplied.model_dump(mode="python"))
    except AttributeError, ValueError:
        raise RunnerError("Supplied dataset manifest is invalid") from None
    try:
        on_disk = load_manifest(data_root, ground_truth=_GROUND_TRUTH)
    except DatasetError:
        raise RunnerError("On-disk edi_fields dataset validation failed") from None
    if on_disk != manifest:
        raise RunnerError("Supplied manifest does not match the validated on-disk manifest")
    if not any(entry.split == split for entry in manifest.docs):
        raise RunnerError(f"Dataset has no documents in split {split!r}")
    try:
        fingerprint = dataset_fingerprint(data_root, manifest)
    except DatasetError:
        raise RunnerError("Dataset fingerprinting failed") from None
    return manifest, fingerprint


def _revalidate_dataset(data_root: Path, manifest: Manifest, fingerprint: str) -> None:
    try:
        on_disk = load_manifest(data_root, ground_truth=_GROUND_TRUTH)
        current = dataset_fingerprint(data_root, manifest)
    except DatasetError:
        raise RunnerError("Dataset changed or became invalid during benchmark execution") from None
    if on_disk != manifest or current != fingerprint:
        raise RunnerError("Dataset changed during benchmark execution; results were not published")


def _extract(pipeline: Pipeline, prepared: PreprocessResult) -> PipelineResult:
    """Call the pipeline; an unexpected exception becomes a content-free ``ERROR``."""
    started = perf_counter_ns()
    try:
        return pipeline.extract(prepared)
    except Exception as caught:
        return PipelineResult(
            status=PipelineStatus.ERROR,
            fields=None,
            layer1_text=None,
            layer1_ms=None,
            layer2_ms=None,
            total_ms=elapsed_ms(started),
            cost_usd=float(getattr(caught, "cost_usd", 0.0)),
            attempts=0,
            error=f"pipeline raised {type(caught).__name__}",
        )


def _sheet_result(pipeline: Pipeline, data_root: Path, entry: DocEntry) -> SheetResult:
    ground_truth = load_edi_ground_truth(data_root, entry.doc_id)
    try:
        prepared = preprocess(data_root / "raw" / entry.file)
    except PreprocessError:
        raise RunnerError(f"Preprocessing failed for {entry.doc_id}") from None
    result = _extract(pipeline, prepared)
    score = score_sheet(entry.doc_id, ground_truth, result, source_kind=entry.source_kind)
    return SheetResult(
        sheet_id=entry.doc_id,
        source_kind=entry.source_kind,
        status=result.status,
        fields=result.fields,
        layer1_text=result.layer1_text,
        raw_outputs=list(result.raw_outputs),
        outcomes=score.outcomes,
        failed_layer=score.failed_layer,
        timings_ms={
            "preprocess": prepared.duration_ms,
            "layer1": result.layer1_ms,
            "layer2": result.layer2_ms,
            "total": result.total_ms,
        },
        cost_usd=result.cost_usd,
        attempts=result.attempts,
        error=result.error,
    )


def run_pipeline(
    *,
    pipeline: Pipeline,
    data_root: Path,
    manifest: Manifest,
    split: PipelineSplit,
    runs_root: Path,
    run_id: str | None = None,
    confirm_final_flag: bool = False,
    record: Mapping[str, str] | None = None,
    cloud: bool = False,
) -> Path:
    """Run ``pipeline`` on every sheet of ``split`` and return the published run directory.

    ``record`` adds entries such as the prompt hash and model digests to the
    pipeline description in the manifest; it must not hold secrets. ``cloud``
    marks a pipeline that sends sheets to a cloud service, for ``decide_cloud``.
    """
    if split not in ("dev", "final"):
        raise RunnerError(f"Unsupported split for a pipeline run: {split!r}")
    if split == "final":
        ensure_final_allowed(confirm_final_flag=confirm_final_flag)
    selected_run_id = _validated_run_id(run_id) if run_id is not None else uuid.uuid4().hex[:8]
    validated, fingerprint = _validated_dataset(data_root, manifest, split)
    started_at = _utc_now().strftime("%Y%m%dT%H%M%SZ")
    directory_name = f"{started_at}_{selected_run_id}"
    _check_output_path(runs_root, directory_name)

    try:
        description = describe_pipeline(pipeline)
    except Exception as caught:
        raise RunnerError(
            f"Pipeline could not describe its configuration: {type(caught).__name__}"
        ) from None
    extra = dict(record or {})
    clashes = sorted(set(extra) & set(description))
    if clashes:
        raise RunnerError("Run record repeats pipeline description keys: " + ", ".join(clashes))
    commit, dirty = code_version()
    try:
        run_manifest = PipelineRunManifest(
            kind=PIPELINE_RUN_KIND,
            run_id=selected_run_id,
            started_at_utc=started_at,
            pipeline_id=pipeline.pipeline_id,
            pipeline=dict(sorted({**description, **extra}.items())),
            cloud=cloud,
            split=split,
            dataset_fingerprint=fingerprint,
            preprocess_version=PREPROCESS_VERSION,
            order_no_rule=ORDER_NO_RULE,
            code_commit=commit,
            code_dirty=dirty,
            os_info=platform.platform(),
            gpu_info=_gpu_info(),
        )
    except ValueError:
        raise RunnerError("Pipeline metadata does not satisfy the run manifest") from None

    entries = sorted(
        (entry for entry in validated.docs if entry.split == split),
        key=lambda entry: entry.doc_id,
    )
    sheets = [_sheet_result(pipeline, data_root, entry) for entry in entries]
    try:
        results = PipelineRunResults(
            schema_version=1, kind=PIPELINE_RUN_KIND, manifest=run_manifest, sheets=sheets
        )
        manifest_payload = (run_manifest.model_dump_json(indent=2) + "\n").encode("utf-8")
        results_payload = (results.model_dump_json(indent=2) + "\n").encode("utf-8")
    except UnicodeEncodeError, ValueError:
        raise RunnerError("Run results could not be serialized safely") from None

    _revalidate_dataset(data_root, validated, fingerprint)
    return _publish_run(
        runs_root=runs_root,
        directory_name=directory_name,
        manifest_payload=manifest_payload,
        results_payload=results_payload,
    )


__all__ = ["code_version", "load_edi_ground_truth", "run_pipeline"]
