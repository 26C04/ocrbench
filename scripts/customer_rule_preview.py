"""Preview open decision D2: rescore dev runs with a looser customer rule.

Usage::

    uv run python -m scripts.customer_rule_preview --run-dirs RUN [RUN ...]

The official rule maps the printed customer name to an EDI site only when it
matches the table in ``ocrbench.customers`` exactly (after width and space
normalisation). The loose rule is the template harness's: legal-form words
such as 株式会社 are dropped, and the closest site name wins when its
similarity is at least 0.65 and beats the runner-up by 0.1.

Only aggregate counts are printed, and only Development runs are accepted, so
no sheet value and no Final result is shown. Nothing is written: the official
scores do not change until the group decides D2.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ocrbench.config import OcrBenchError
from ocrbench.customers import canonical_name, loose_site, resolve_customer_site
from ocrbench.edi_fields import EdiFields, compare_fields, format_problems
from ocrbench.pipeline_results import load_pipeline_run
from ocrbench.pipeline_runner import load_edi_ground_truth
from ocrbench.pipelines import PipelineStatus


@dataclass(frozen=True)
class RulePreview:
    """One run's counts under the official and the loose customer rule."""

    pipeline_id: str
    n_sheets: int
    customer_exact: int
    customer_loose: int
    fields_exact: int
    fields_loose: int
    ready_exact: int
    ready_loose: int
    silent_loose: int


def preview_run(run_dir: Path, data_root: Path) -> RulePreview:
    """Rescore one completed Development run under both customer rules."""
    results = load_pipeline_run(run_dir)
    if results.manifest.split != "dev":
        raise OcrBenchError(f"{run_dir}: only Development runs can be previewed")
    counts = dict.fromkeys(
        ("customer_exact", "customer_loose", "fields_exact", "fields_loose"), 0
    ) | {"ready_exact": 0, "ready_loose": 0, "silent_loose": 0}
    for sheet in results.sheets:
        if sheet.status is PipelineStatus.OK:
            counts["ready_exact"] += 1
        if sheet.fields is None:
            continue
        gt = load_edi_ground_truth(data_root, sheet.sheet_id)
        exact = compare_fields(gt, sheet.fields)
        site = loose_site(sheet.fields.customer)
        customer_loose = site is not None and site == resolve_customer_site(gt.customer)
        others = sum(exact[name] for name in ("part_no", "order_no", "quantity"))
        counts["customer_exact"] += exact["customer"]
        counts["customer_loose"] += customer_loose
        counts["fields_exact"] += exact["customer"] + others
        counts["fields_loose"] += customer_loose + others
        if sheet.status in (PipelineStatus.ERROR, PipelineStatus.FORMAT_ERROR) or site is None:
            continue
        mapped = EdiFields.model_validate(
            {**sheet.fields.model_dump(), "customer": canonical_name(site)}
        )
        present = None not in (mapped.part_no, mapped.order_no, mapped.quantity)
        if present and not format_problems(mapped):
            counts["ready_loose"] += 1
            counts["silent_loose"] += (not customer_loose) + (3 - others)
    return RulePreview(results.manifest.pipeline_id, len(results.sheets), **counts)


def render(previews: Sequence[RulePreview]) -> str:
    lines = [
        "# D2 preview: customer rule, official (exact) vs loose (harness-style)",
        "",
        "| Candidate | Customer exact | Customer loose | Fields exact | Fields loose "
        "| Ready sheets exact | Ready sheets loose | Silent errors loose |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for p in previews:
        fields, n = 4 * p.n_sheets, p.n_sheets
        lines.append(
            f"| {p.pipeline_id} | {p.customer_exact}/{n} | {p.customer_loose}/{n} "
            f"| {p.fields_exact}/{fields} | {p.fields_loose}/{fields} "
            f"| {p.ready_exact}/{n} | {p.ready_loose}/{n} | {p.silent_loose} |"
        )
    lines += [
        "",
        "Ready = the sheet would pass every check and go to the EDI. Silent errors loose ="
        " wrong values on those ready sheets. With one customer in the table, any name close to"
        " it maps to that site, including a different company with a similar name.",
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
