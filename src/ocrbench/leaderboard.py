"""Rank several pipeline runs on the same sheets and apply the ADR-0002 decision rule.

Every run must cover the same split, dataset fingerprint, order-number rule,
and sheet IDs, so the ranking is a fair, paired comparison. The output holds
aggregates only, never sheet IDs or values.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ocrbench.field_metrics import (
    CloudDecision,
    FieldComparison,
    FieldRunSummary,
    compare,
    decide_cloud,
    summarize,
)
from ocrbench.pipeline_results import PipelineRunResults


class LeaderboardError(ValueError):
    """Raised when runs cannot be ranked together."""


@dataclass(frozen=True)
class Entry:
    """One ranked run: its summary and its paired comparison with the leader."""

    rank: int
    pipeline_id: str
    run_id: str
    cloud: bool
    summary: FieldRunSummary
    versus_leader: FieldComparison | None


@dataclass(frozen=True)
class Leaderboard:
    split: str
    n_sheets: int
    entries: tuple[Entry, ...]
    cloud_decision: CloudDecision | None
    best_local: str | None
    best_cloud: str | None


def _check_comparable(runs: Sequence[PipelineRunResults]) -> None:
    if len(runs) < 2:
        raise LeaderboardError("a leaderboard needs at least two pipeline runs")
    first = runs[0]
    for field in ("split", "dataset_fingerprint", "order_no_rule"):
        if any(getattr(run.manifest, field) != getattr(first.manifest, field) for run in runs):
            raise LeaderboardError(f"runs must share manifest field {field!r}")
    sheet_ids = {sheet.sheet_id for sheet in first.sheets}
    if any({sheet.sheet_id for sheet in run.sheets} != sheet_ids for run in runs):
        raise LeaderboardError("runs must contain the same sheet IDs")
    run_ids = [run.manifest.run_id for run in runs]
    if len(set(run_ids)) != len(run_ids):
        raise LeaderboardError("each run may appear only once")


def _sort_key(item: tuple[PipelineRunResults, FieldRunSummary]) -> tuple[float, int, float, str]:
    run, summary = item
    # Most accurate first; ties go to fewer silent errors, then to the faster run.
    return (
        -summary.field_accuracy,
        summary.silent_errors,
        summary.time_p50_ms,
        run.manifest.run_id,
    )


def build_leaderboard(runs: Sequence[PipelineRunResults]) -> Leaderboard:
    """Rank runs by field accuracy, then silent errors, then median time."""
    _check_comparable(runs)
    ranked = sorted(((run, summarize(run.scores())) for run in runs), key=_sort_key)
    leader_scores = ranked[0][0].scores()
    entries = tuple(
        Entry(
            rank=index + 1,
            pipeline_id=run.manifest.pipeline_id,
            run_id=run.manifest.run_id,
            cloud=run.manifest.cloud,
            summary=summary,
            versus_leader=None if index == 0 else compare(leader_scores, run.scores()),
        )
        for index, (run, summary) in enumerate(ranked)
    )
    best_local = next((run for run, _ in ranked if not run.manifest.cloud), None)
    best_cloud = next((run for run, _ in ranked if run.manifest.cloud), None)
    decision = (
        decide_cloud(best_local.scores(), best_cloud.scores())
        if best_local is not None and best_cloud is not None
        else None
    )
    return Leaderboard(
        split=runs[0].manifest.split,
        n_sheets=len(runs[0].sheets),
        entries=entries,
        cloud_decision=decision,
        best_local=best_local.manifest.pipeline_id if best_local is not None else None,
        best_cloud=best_cloud.manifest.pipeline_id if best_cloud is not None else None,
    )


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _signed(value: float) -> str:
    return f"{value * 100:+.1f}"


def render_leaderboard(board: Leaderboard) -> str:
    """Markdown table of the ranking plus the decision, without sheet data."""
    lines = [
        f"# Pipeline Leaderboard ({board.split}, {board.n_sheets} sheets)",
        "",
        "Ranked by field accuracy, then fewer silent errors, then median time. "
        '"vs leader" is the paired field-accuracy difference in points with its 95% interval; '
        "an interval that includes 0 means the gap may be noise.",
        "",
        "| Rank | Pipeline | Field accuracy | Sheets fully correct | Silent errors | "
        "Empty fields | Time p50 | Cost / sheet | vs leader |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for entry in board.entries:
        summary = entry.summary
        versus = "leader"
        if entry.versus_leader is not None:
            low, high = entry.versus_leader.difference_ci
            versus = (
                f"{_signed(entry.versus_leader.field_accuracy_difference)} "
                f"({_signed(low)} to {_signed(high)})"
            )
        name = entry.pipeline_id + (" (cloud)" if entry.cloud else "")
        lines.append(
            f"| {entry.rank} | {name} | {_pct(summary.field_accuracy)} | "
            f"{summary.sheets_correct}/{summary.n_sheets} | {summary.silent_errors} | "
            f"{summary.missing_fields} | {summary.time_p50_ms / 1000:.1f} s | "
            f"USD {summary.cost_per_sheet_usd:.4f} | {versus} |"
        )
    lines.append("")
    if board.cloud_decision is not None:
        decision = board.cloud_decision
        lines.extend(
            [
                "## Local or cloud (ADR-0002 §7)",
                "",
                f"Best local: {board.best_local}. Best cloud: {board.best_cloud}. "
                f"Decision: **{decision.choice.value}**.",
                "",
                *[f"- {reason}" for reason in decision.reasons],
                "",
            ]
        )
    else:
        lines.extend(["No cloud run is included, so the local-or-cloud rule was not applied.", ""])
    return "\n".join(lines)


def write_leaderboard(board: Leaderboard, path: Path) -> Path:
    """Write the leaderboard Markdown, refusing to overwrite an existing file."""
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(render_leaderboard(board))
    return Path(path)


__all__ = [
    "Entry",
    "Leaderboard",
    "LeaderboardError",
    "build_leaderboard",
    "render_leaderboard",
    "write_leaderboard",
]
