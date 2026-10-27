"""Preview combining two finished Development runs: two reads that must agree.

Usage::

    uv run python -m scripts.agree_preview --run-a RUN --run-b RUN

Per sheet, the customer is taken from whichever run's name maps to an EDI
site (run A first). Part number, order number and quantity are kept when both
runs agree or one is empty, and left empty (so the sheet goes to review) when
they differ. This rescores what such a combined pipeline would have done,
without running the models again.

Only aggregate counts are printed, and only Development runs on the same
sheets are accepted, so no sheet value and no Final result is shown.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ocrbench.config import OcrBenchError
from ocrbench.customers import resolve_customer_site
from ocrbench.edi_fields import EdiFields, compare_fields, format_problems
from ocrbench.normalize import strip_outer_whitespace
from ocrbench.pipeline_results import SheetResult, load_pipeline_run
from ocrbench.pipeline_runner import load_edi_ground_truth
from ocrbench.pipelines import PipelineStatus

_FAILED = (PipelineStatus.ERROR, PipelineStatus.FORMAT_ERROR)


@dataclass(frozen=True)
class AgreePreview:
    """Counts for run A, run B and their combination on the same sheets."""

    label: str
    n_sheets: int
    customer: int
    fields: int
    sheets_correct: int
    ready: int
    silent: int


def _agree[T](a: T | None, b: T | None) -> T | None:
    if a is None or b is None:
        return a if b is None else b
    if isinstance(a, str) and isinstance(b, str):
        return a if strip_outer_whitespace(a) == strip_outer_whitespace(b) else None
    return a if a == b else None


def combine(a: EdiFields | None, b: EdiFields | None) -> EdiFields | None:
    """Combine two reads of one sheet by the rule in the module docstring."""
    if a is None or b is None:
        return a if b is None else b
    mapped = [name for name in (a.customer, b.customer) if resolve_customer_site(name)]
    return EdiFields(
        customer=mapped[0] if mapped else a.customer,
        part_no=_agree(a.part_no, b.part_no),
        order_no=_agree(a.order_no, b.order_no),
        quantity=_agree(a.quantity, b.quantity),
    )


def _ready(fields: EdiFields) -> bool:
    values = (fields.customer, fields.part_no, fields.order_no, fields.quantity)
    return None not in values and not format_problems(fields)


def _usable(sheet: SheetResult) -> EdiFields | None:
    return None if sheet.status in _FAILED else sheet.fields


def preview(run_a: Path, run_b: Path, data_root: Path) -> list[AgreePreview]:
    """Score run A, run B and the combination on their shared Development sheets."""
    a, b = load_pipeline_run(run_a), load_pipeline_run(run_b)
    if a.manifest.split != "dev" or b.manifest.split != "dev":
        raise OcrBenchError("only Development runs can be previewed")
    if a.manifest.dataset_fingerprint != b.manifest.dataset_fingerprint:
        raise OcrBenchError("the two runs used different datasets")
    by_id = {sheet.sheet_id: sheet for sheet in b.sheets}
    if set(by_id) != {sheet.sheet_id for sheet in a.sheets}:
        raise OcrBenchError("the two runs cover different sheets")
    labels = (a.manifest.pipeline_id, b.manifest.pipeline_id, "combined")
    totals = {label: [0, 0, 0, 0, 0] for label in labels}
    for sheet_a in a.sheets:
        sheet_b = by_id[sheet_a.sheet_id]
        gt = load_edi_ground_truth(data_root, sheet_a.sheet_id)
        reads = (_usable(sheet_a), _usable(sheet_b))
        for label, fields in zip(labels, (*reads, combine(*reads)), strict=True):
            if fields is None:
                continue
            right = compare_fields(gt, fields)
            counts = totals[label]
            counts[0] += right["customer"]
            counts[1] += sum(right.values())
            counts[2] += all(right.values())
            if _ready(fields):
                counts[3] += 1
                counts[4] += 4 - sum(right.values())
    n = len(a.sheets)
    return [AgreePreview(label, n, *totals[label]) for label in labels]


def render(rows: Sequence[AgreePreview]) -> str:
    lines = [
        "# Preview: two reads that must agree",
        "",
        "| Pipeline | Field accuracy | Customer | Sheets fully correct | Ready | Silent errors |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        n = r.n_sheets
        lines.append(
            f"| {r.label} | {r.fields / (4 * n):.1%} ({r.fields}/{4 * n}) | {r.customer}/{n} "
            f"| {r.sheets_correct}/{n} | {r.ready}/{n} | {r.silent} |"
        )
    lines += [
        "",
        "Ready = the sheet would pass every check and go to the EDI. Silent errors = wrong"
        " values on those ready sheets.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None, help="default: OCRBENCH_DATA_DIR")
    arguments = parser.parse_args(argv)
    data_root = arguments.data_root or Path(os.environ.get("OCRBENCH_DATA_DIR", ""))
    try:
        rows = preview(arguments.run_a, arguments.run_b, data_root)
    except OcrBenchError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(render(rows), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
