"""Preview the ``customer_from_text`` option on finished Development runs.

Usage::

    uv run python -m scripts.customer_text_preview --run-dirs RUN [RUN ...]

When a run's customer field maps to no EDI site, the option searches the
run's stored Layer 1 text for the site's printed name
(:func:`ocrbench.customers.find_site_in_text`). This rescoring shows what the
option would have changed without running the models again.

Only aggregate counts are printed, and only Development runs with Layer 1
text are accepted, so no sheet value and no Final result is shown. Nothing is
written.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ocrbench.config import OcrBenchError
from ocrbench.edi_fields import EdiFields, compare_fields, format_problems
from ocrbench.pipeline_results import load_pipeline_run
from ocrbench.pipeline_runner import load_edi_ground_truth
from ocrbench.pipelines import PipelineStatus
from ocrbench.postprocess import customer_from_text


@dataclass(frozen=True)
class TextPreview:
    """One run's counts as recorded and with the customer searched in the OCR text."""

    pipeline_id: str
    n_sheets: int
    customer_before: int
    customer_after: int
    fields_before: int
    fields_after: int
    ready_before: int
    ready_after: int
    silent_after: int


def _ready(fields: EdiFields) -> bool:
    values = (fields.customer, fields.part_no, fields.order_no, fields.quantity)
    return None not in values and not format_problems(fields)


def preview_run(run_dir: Path, data_root: Path) -> TextPreview:
    """Rescore one completed Development run with ``customer_from_text`` applied."""
    results = load_pipeline_run(run_dir)
    if results.manifest.split != "dev":
        raise OcrBenchError(f"{run_dir}: only Development runs can be previewed")
    counts = dict.fromkeys(
        (
            "customer_before",
            "customer_after",
            "fields_before",
            "fields_after",
            "ready_before",
            "ready_after",
            "silent_after",
        ),
        0,
    )
    for sheet in results.sheets:
        if sheet.status is PipelineStatus.OK:
            counts["ready_before"] += 1
        if sheet.fields is None:
            continue
        if sheet.layer1_text is None:
            raise OcrBenchError(f"{run_dir}: the run has no Layer 1 text to search")
        gt = load_edi_ground_truth(data_root, sheet.sheet_id)
        before = compare_fields(gt, sheet.fields)
        fields = customer_from_text(sheet.fields, sheet.layer1_text)
        after = compare_fields(gt, fields)
        counts["customer_before"] += before["customer"]
        counts["customer_after"] += after["customer"]
        counts["fields_before"] += sum(before.values())
        counts["fields_after"] += sum(after.values())
        errored = sheet.status in (PipelineStatus.ERROR, PipelineStatus.FORMAT_ERROR)
        if not errored and _ready(fields):
            counts["ready_after"] += 1
            counts["silent_after"] += 4 - sum(after.values())
    return TextPreview(results.manifest.pipeline_id, len(results.sheets), **counts)


def render(previews: Sequence[TextPreview]) -> str:
    lines = [
        "# Preview: customer searched in the OCR text (customer_from_text)",
        "",
        "| Candidate | Customer now | Customer with option | Fields now | Fields with option "
        "| Ready now | Ready with option | Silent errors with option |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for p in previews:
        n, fields = p.n_sheets, 4 * p.n_sheets
        lines.append(
            f"| {p.pipeline_id} | {p.customer_before}/{n} | {p.customer_after}/{n} "
            f"| {p.fields_before}/{fields} | {p.fields_after}/{fields} "
            f"| {p.ready_before}/{n} | {p.ready_after}/{n} | {p.silent_after} |"
        )
    lines += [
        "",
        "Ready = the sheet would pass every check and go to the EDI. Silent errors with option ="
        " wrong values on those ready sheets.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--data-root", type=Path, default=None, help="default: OCRBENCH_DATA_DIR")
    arguments = parser.parse_args(argv)
    data_root = arguments.data_root or Path(os.environ.get("OCRBENCH_DATA_DIR", ""))
    try:
        previews = [preview_run(path, data_root) for path in arguments.run_dirs]
    except OcrBenchError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(render(previews), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
