"""Reports for ADR-0002 pipeline runs: ``report.md``, the anonymous summary, and
the local-only detail report.

``report.md`` and the summary hold aggregates and anonymous per-sheet outcome
categories only. Predicted values, ground truth, Layer 1 text and raw model
output appear only in the detail report, which is restricted to the dev split
and to ``OCRBENCH_RUNS_DIR``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any, Final

from ocrbench.edi_fields import EDI_FIELD_NAMES, EdiFields
from ocrbench.field_metrics import FieldRunSummary, summarize
from ocrbench.pipeline_results import PipelineRunResults, load_pipeline_run
from ocrbench.pipeline_runner import load_edi_ground_truth
from ocrbench.reporting import (
    _analysis_template,
    _atomic_create_text,
    _atomic_write_text,
    _ci,
    _milliseconds,
    _percent,
    _require_external_run_directory,
    _string_leaves,
    _table_row,
)
from ocrbench.splitguard import ensure_detail_report_allowed

TARGET_WARM_MS: Final[float] = 30_000.0
"""ADR-0001 §7 / ADR-0002 §4 target: 30 seconds per sheet."""
MANUAL_MS: Final[float] = 300_000.0
"""About 5 minutes of manual work per sheet (ADR-0002 §4 requirement)."""
_MIN_SENSITIVE_LENGTH: Final[int] = 4
_FIELD_LABELS: Final[dict[str, str]] = {
    "customer": "Customer (受注先)",
    "part_no": "Part number (部番)",
    "order_no": "Order number (受注No)",
    "quantity": "Quantity (個数)",
}


def _usd(value: float) -> str:
    return f"USD {value:.4f}"


def _optional_ms(values: list[float | None]) -> str:
    present = [value for value in values if value is not None]
    return _milliseconds(sum(present) / len(present)) if present else "N/A"


def render_pipeline_report(results: PipelineRunResults, summary: FieldRunSummary) -> str:
    """Render the aggregate report with every ADR-0002 §4 and §5 metric, without sheet data."""
    if summary.n_sheets != len(results.sheets):
        raise ValueError("summary does not match the run")
    manifest = results.manifest
    n = summary.n_sheets
    field_total = n * len(EDI_FIELD_NAMES)
    correct_fields = round(summary.field_accuracy * field_total)
    slow = sum(sheet.score().total_ms > TARGET_WARM_MS for sheet in results.sheets)
    over_manual = sum(sheet.score().total_ms > MANUAL_MS for sheet in results.sheets)

    lines = [
        f"# Pipeline Benchmark Report: {manifest.pipeline_id}",
        "",
        "## Run identity",
        "",
        "| Field | Value |",
        "| --- | --- |",
    ]
    identity = (
        ("Run ID", manifest.run_id),
        ("Started at (UTC)", manifest.started_at_utc),
        ("Pipeline", manifest.pipeline_id),
        ("Split", manifest.split),
        ("Sheets", str(n)),
        ("Cloud", "yes" if manifest.cloud else "no"),
        ("Dataset fingerprint", manifest.dataset_fingerprint),
        ("Preprocess version", manifest.preprocess_version),
        ("Order number rule", manifest.order_no_rule.value),
        ("Code commit", manifest.code_commit or "unknown"),
        (
            "Uncommitted changes",
            "unknown" if manifest.code_dirty is None else "yes" if manifest.code_dirty else "no",
        ),
        ("Operating system", manifest.os_info),
        ("GPU", manifest.gpu_info or "N/A"),
    )
    lines.extend(_table_row(label, value) for label, value in identity)
    lines.extend(["", "## Pipeline configuration", "", "| Key | Value |", "| --- | --- |"])
    lines.extend(_table_row(key, value) for key, value in sorted(manifest.pipeline.items()))
    lines.extend(
        [
            "",
            "## Primary metric",
            "",
            "| Metric | Value |",
            "| --- | --- |",
            _table_row(
                "Field accuracy",
                f"{_percent(summary.field_accuracy)} ({correct_fields}/{field_total} fields; "
                f"bootstrap 95% CI {_ci(summary.field_accuracy_ci)})",
            ),
            "",
            "## Secondary metrics",
            "",
            "| Metric | Value |",
            "| --- | --- |",
            _table_row(
                "Sheets fully correct",
                f"{_percent(summary.sheet_accuracy)} ({summary.sheets_correct}/{n}; "
                f"Wilson 95% CI {_ci(summary.sheet_accuracy_ci)})",
            ),
            _table_row(
                "Silent errors (wrong values on sheets marked ok)",
                f"{summary.silent_errors} fields on {summary.sheets_with_silent_errors} sheets",
            ),
            _table_row("Abstentions (empty fields)", f"{summary.missing_fields} fields"),
            "",
            "## Per-field accuracy",
            "",
            "| Field | Accuracy | Bootstrap 95% CI |",
            "| --- | ---: | --- |",
        ]
    )
    lines.extend(
        _table_row(
            _FIELD_LABELS[field],
            _percent(summary.per_field_accuracy[field]),
            _ci(summary.per_field_ci[field]),
        )
        for field in EDI_FIELD_NAMES
    )
    lines.extend(["", "## Sheet status", "", "| Status | Sheets |", "| --- | ---: |"])
    lines.extend(_table_row(status, str(count)) for status, count in summary.status_counts.items())
    lines.extend(
        [
            "",
            "## Which layer failed",
            "",
            "A field is a Layer 1 error when its correct value is not in the OCR text, and a "
            "Layer 2 error when it is but the extracted field is still wrong. Pipelines without "
            "Layer 1 text are not attributed.",
            "",
            "| Field | Layer 1 (OCR) | Layer 2 (LLM) |",
            "| --- | ---: | ---: |",
        ]
    )
    lines.extend(
        _table_row(
            _FIELD_LABELS[field],
            str(summary.layer_failures["layer1"][field]),
            str(summary.layer_failures["layer2"][field]),
        )
        for field in EDI_FIELD_NAMES
    )
    lines.extend(
        [
            "",
            "## Source-kind breakdown",
            "",
            "| Source kind | Sheets | Field accuracy |",
            "| --- | ---: | ---: |",
        ]
    )
    kinds: list[str] = [sheet.source_kind for sheet in results.sheets]
    lines.extend(
        _table_row(kind, str(kinds.count(kind)), _percent(accuracy))
        for kind, accuracy in summary.by_source_kind.items()
    )
    lines.extend(
        [
            "",
            "## Time per sheet",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            _table_row("p50", _milliseconds(summary.time_p50_ms)),
            _table_row("p95", _milliseconds(summary.time_p95_ms)),
            _table_row(
                "Mean Layer 1", _optional_ms([s.timings_ms["layer1"] for s in results.sheets])
            ),
            _table_row(
                "Mean Layer 2", _optional_ms([s.timings_ms["layer2"] for s in results.sheets])
            ),
            _table_row("Sheets above 30 s target", f"{slow}/{n}"),
            _table_row("Sheets above 5 min manual time", f"{over_manual}/{n}"),
            "",
            "## Cost",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            _table_row("Cost per sheet", _usd(summary.cost_per_sheet_usd)),
            _table_row("Total cost", _usd(summary.total_cost_usd)),
            "",
            "This aggregate report intentionally excludes sheet IDs, predicted values, OCR text, "
            "raw model output, and ground-truth values.",
            "",
        ]
    )
    return "\n".join(lines)


def write_pipeline_report(run_dir: Path) -> Path:
    """Write ``report.md`` for a pipeline run and create ``analysis.md`` if absent."""
    directory = Path(run_dir)
    results = load_pipeline_run(directory)
    report_path = directory / "report.md"
    _atomic_write_text(report_path, render_pipeline_report(results, summarize(results.scores())))
    _atomic_create_text(directory / "analysis.md", _analysis_template())
    return report_path


def _sensitive_strings(results: PipelineRunResults) -> set[str]:
    values: set[str] = set()
    for sheet in results.sheets:
        values.update(sheet.raw_outputs)
        if sheet.layer1_text:
            values.add(sheet.layer1_text)
        if sheet.fields is not None:
            values.update(_string_leaves(sheet.fields.model_dump(mode="json")))
    return {value for value in values if len(value.strip()) >= _MIN_SENSITIVE_LENGTH}


def export_pipeline_summary(results: PipelineRunResults) -> dict[str, Any]:
    """Return the allowlisted, cloud-shareable summary of a pipeline run.

    Per sheet it holds only the anonymous ID, source kind, status, outcome
    category per field, failed layer, time, and cost. Any predicted value,
    OCR text, or raw output of four or more characters found in the payload
    aborts the export.
    """
    summary = summarize(results.scores())
    manifest = results.manifest
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "pipeline_summary",
        "manifest": manifest.model_dump(mode="json"),
        "summary": asdict(summary),
        "sheets": [
            {
                "sheet_id": sheet.sheet_id,
                "source_kind": sheet.source_kind,
                "status": sheet.status.value,
                "outcomes": {field: outcome.value for field, outcome in sheet.outcomes.items()},
                "failed_layer": {field: layer.value for field, layer in sheet.failed_layer.items()},
                "total_ms": sheet.timings_ms["total"],
                "cost_usd": sheet.cost_usd,
            }
            for sheet in results.sheets
        ],
    }
    try:
        json.dumps(payload, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("anonymous export is not strict JSON") from error
    forbidden = _sensitive_strings(results)
    exported = _string_leaves(payload)
    if any(secret in text for secret in forbidden for text in exported):
        raise ValueError("anonymous export contains forbidden prediction or OCR content")
    return payload


def write_pipeline_summary(run_dir: Path, out_path: Path) -> Path:
    """Write the anonymous summary of a dev pipeline run as strict JSON."""
    results = load_pipeline_run(Path(run_dir))
    if results.manifest.split != "dev":
        raise ValueError("agent-facing summary export is restricted to Development runs")
    destination = Path(out_path)
    payload = export_pipeline_summary(results)
    _atomic_write_text(
        destination, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return destination


def _cell(value: object) -> str:
    return "null" if value is None else str(value)


def render_pipeline_detail_report(
    results: PipelineRunResults, ground_truth: Mapping[str, EdiFields]
) -> str:
    """Render the local-only report with ground truth, predictions, and OCR text."""
    if set(ground_truth) != {sheet.sheet_id for sheet in results.sheets}:
        raise ValueError("ground truth must contain exactly the run sheet IDs")
    lines = [
        f"# Pipeline Detailed Report: {results.manifest.pipeline_id}",
        "",
        "CONFIDENTIAL: This local-only report contains ground truth, predictions, OCR text, and "
        "raw model output. Do not commit it, share it, or send it to cloud services.",
    ]
    for sheet in results.sheets:
        gt = ground_truth[sheet.sheet_id].model_dump()
        predicted = sheet.fields.model_dump() if sheet.fields is not None else {}
        lines.extend(
            [
                "",
                f"## {sheet.sheet_id} ({sheet.source_kind})",
                "",
                f"Status: {sheet.status.value}; attempts: {sheet.attempts}; "
                f"total: {_milliseconds(sheet.score().total_ms)}"
                + (f"; error: {sheet.error}" if sheet.error else ""),
                "",
                "| Field | Ground truth | Predicted | Outcome | Failed layer |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        lines.extend(
            _table_row(
                field,
                _cell(gt[field]),
                _cell(predicted.get(field)),
                sheet.outcomes[field].value,
                sheet.failed_layer[field].value if field in sheet.failed_layer else "",
            )
            for field in EDI_FIELD_NAMES
        )
        if sheet.layer1_text is not None:
            fence = "````"
            while fence in sheet.layer1_text:
                fence += "`"
            lines.extend(["", "Layer 1 text:", "", fence, sheet.layer1_text, fence])
    lines.append("")
    return "\n".join(lines)


def write_pipeline_detail_report(run_dir: Path, data_root: Path) -> Path:
    """Write ``report_detail.md`` for a dev pipeline run inside ``OCRBENCH_RUNS_DIR``."""
    directory = Path(run_dir)
    results = load_pipeline_run(directory)
    ensure_detail_report_allowed(results.manifest.split)
    _require_external_run_directory(directory)
    ground_truth = {
        sheet.sheet_id: load_edi_ground_truth(data_root, sheet.sheet_id) for sheet in results.sheets
    }
    output = directory / "report_detail.md"
    _atomic_write_text(output, render_pipeline_detail_report(results, ground_truth))
    return output


__all__ = [
    "export_pipeline_summary",
    "render_pipeline_detail_report",
    "render_pipeline_report",
    "write_pipeline_detail_report",
    "write_pipeline_report",
    "write_pipeline_summary",
]
