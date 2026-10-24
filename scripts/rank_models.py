"""Rank Development runs for the ADR §8-1/§8-2 five-model first-pass evaluation.

Usage::

    uv run python -m scripts.rank_models --run-dirs RUN [RUN ...] [--out ranking.md]

Every run must be a completed Development run on the same dataset fingerprint
and prompt hash. Models whose warm p50 latency exceeds the 30-second target are
ineligible. Eligible models are ordered with the same aggregate key used by
``ocrbench select pick-best``: exact-match count, JSON-invalid count, the sum of
``order_no``/``part_no``/``num_pieces`` accuracy, output tokens, then warm p50.
Runs with CPU/RAM offload (``participation == "reference"``) are listed but
ranked after every primary run, so they are never chosen ahead of one.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ocrbench.config import OcrBenchError
from ocrbench.metrics import RunSummary, aggregate
from ocrbench.results import RunResults, load_run

LATENCY_TARGET_MS = 30_000.0
TOP_N = 2


@dataclass(frozen=True)
class RankedRun:
    """One model's Development run with the values used to rank it."""

    run_dir: Path
    results: RunResults
    summary: RunSummary

    @property
    def model_tag(self) -> str:
        return self.results.manifest.model_tag

    @property
    def is_primary(self) -> bool:
        return self.results.manifest.participation == "primary"

    @property
    def invalid_json_count(self) -> int:
        valid = round(self.summary.n_docs * self.summary.schema_valid_rate)
        return self.summary.n_docs - valid

    @property
    def key_field_accuracy(self) -> float:
        return (
            self.summary.header_field_accuracy["order_no"]
            + self.summary.item_field_accuracy["part_no"]
            + self.summary.item_field_accuracy["num_pieces"]
        )

    @property
    def meets_latency(self) -> bool:
        p50 = self.summary.latency_warm_p50_ms
        return p50 is not None and p50 <= LATENCY_TARGET_MS


def _sort_key(run: RankedRun) -> tuple[int, int, int, int, float, int, float, str]:
    p50 = run.summary.latency_warm_p50_ms
    return (
        0 if run.meets_latency else 1,
        0 if run.is_primary else 1,
        -run.summary.exact_match_count,
        run.invalid_json_count,
        -run.key_field_accuracy,
        run.summary.token_totals["output"],
        math.inf if p50 is None else p50,
        run.model_tag,
    )


def load_runs(run_dirs: Sequence[Path]) -> list[RankedRun]:
    """Load runs and reject any set that would not be a fair comparison."""
    runs: list[RankedRun] = []
    for path in run_dirs:
        results = load_run(path)
        if results.manifest.split != "dev":
            raise OcrBenchError(f"{path}: only Development runs can be ranked")
        runs.append(RankedRun(path, results, aggregate(results.documents)))
    for field in ("dataset_fingerprint", "prompt_hash"):
        values = {getattr(run.results.manifest, field) for run in runs}
        if len(values) > 1:
            raise OcrBenchError(f"runs differ in {field}; rerun them on the same inputs")
    tags = [run.model_tag for run in runs]
    duplicates = sorted({tag for tag in tags if tags.count(tag) > 1})
    if duplicates:
        raise OcrBenchError("more than one run for model(s): " + ", ".join(duplicates))
    return sorted(runs, key=_sort_key)


def _seconds(value_ms: float | None) -> str:
    return "n/a" if value_ms is None else f"{value_ms / 1000:.1f}"


def render(runs: Sequence[RankedRun]) -> str:
    """Render the ranking and the selected top models as Markdown."""
    first = runs[0].results.manifest
    lines = [
        "# Phase 1 model ranking (Development)",
        "",
        f"- Documents per run: {runs[0].summary.n_docs}",
        f"- Prompt: {first.prompt_name}/{first.prompt_hash}",
        f"- Dataset fingerprint: {first.dataset_fingerprint}",
        f"- Latency target: warm p50 <= {LATENCY_TARGET_MS / 1000:.0f} s",
        "",
        "| Rank | Model | Exact match | JSON invalid | order_no | part_no | num_pieces "
        "| Warm p50 s | Warm p95 s | Output tokens | GPU | 30 s |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for rank, run in enumerate(runs, start=1):
        summary = run.summary
        lines.append(
            f"| {rank} | `{run.model_tag}` "
            f"| {summary.exact_match_count}/{summary.n_docs} ({summary.exact_match_rate:.1%}) "
            f"| {run.invalid_json_count} "
            f"| {summary.header_field_accuracy['order_no']:.1%} "
            f"| {summary.item_field_accuracy['part_no']:.1%} "
            f"| {summary.item_field_accuracy['num_pieces']:.1%} "
            f"| {_seconds(summary.latency_warm_p50_ms)} "
            f"| {_seconds(summary.latency_warm_p95_ms)} "
            f"| {summary.token_totals['output']} "
            f"| {'100% GPU' if run.is_primary else 'offload (reference)'} "
            f"| {'yes' if run.meets_latency else 'NO'} |"
        )

    selected = [run for run in runs if run.meets_latency][:TOP_N]
    lines += ["", "## Selected for prompt improvement (ADR §8-2)", ""]
    if not selected:
        lines.append("No model met the 30-second target. Do not proceed; review latency.")
    for run in selected:
        note = "" if run.is_primary else " (reference only: not 100% GPU on this machine)"
        lines.append(f"- `{run.model_tag}`{note}")
    if len(selected) < TOP_N:
        lines.append(f"- Only {len(selected)} model(s) met the latency target.")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, help="also write the Markdown ranking here")
    arguments = parser.parse_args(argv)
    try:
        report = render(load_runs(arguments.run_dirs))
    except OcrBenchError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(report, end="")
    if arguments.out is not None:
        arguments.out.write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
