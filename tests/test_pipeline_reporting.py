"""T24 acceptance: two fake pipelines on a synthetic dataset produce ADR-0002 reports."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import ocrbench.cli as cli_module
from ocrbench.cli import main
from ocrbench.dataset import Manifest
from ocrbench.pipeline_config import BuiltPipeline, ExternalConfig
from ocrbench.pipeline_reporting import (
    export_pipeline_summary,
    render_pipeline_detail_report,
    render_pipeline_report,
)
from ocrbench.pipeline_results import load_pipeline_run
from ocrbench.pipelines import FakePipeline, PipelineResult, PipelineStatus
from tests.helpers import make_edi_dataset, read_edi_ground_truth

_REPORT_SECTIONS = (
    "## Run identity",
    "## Pipeline configuration",
    "Field accuracy",
    "Sheets fully correct",
    "Wilson 95% CI",
    "Silent errors",
    "Abstentions",
    "## Per-field accuracy",
    "Customer (受注先)",
    "Quantity (個数)",
    "## Which layer failed",
    "## Source-kind breakdown",
    "| p50 |",
    "| p95 |",
    "Cost per sheet",
    "Total cost",
)
_WRONG_PART = "HR-WRONG-777"


def _result(
    fields: object, *, status: PipelineStatus, layer1: str | None, cost: float
) -> PipelineResult:
    return PipelineResult(
        status=status,
        fields=fields,  # type: ignore[arg-type]
        layer1_text=layer1,
        layer1_ms=4.0 if layer1 is not None else None,
        layer2_ms=6.0,
        total_ms=10.0,
        cost_usd=cost,
        attempts=1,
    )


def _pipelines(root: Path, manifest: Manifest, split: str) -> dict[str, BuiltPipeline]:
    """A1: OCR+LLM with one silent error; B: vision model that abstains on quantity."""
    ids = sorted(entry.doc_id for entry in manifest.docs if entry.split == split)
    truths = [read_edi_ground_truth(root, doc_id) for doc_id in ids]
    a1 = []
    for index, gt in enumerate(truths):
        text = f"{gt.customer}\n{gt.part_no}\n{gt.order_no}\n{gt.quantity} PC"
        fields = gt.model_copy(update={"part_no": _WRONG_PART}) if index == 0 else gt
        a1.append(_result(fields, status=PipelineStatus.OK, layer1=text, cost=0.0))
    b = [
        _result(
            gt.model_copy(update={"quantity": None}),
            status=PipelineStatus.NEEDS_REVIEW,
            layer1=None,
            cost=0.0,
        )
        for gt in truths
    ]
    return {
        "A1": BuiltPipeline(FakePipeline(a1, pipeline_id="A1"), {"prompt_hash": "0" * 12}, False),
        "B": BuiltPipeline(FakePipeline(b, pipeline_id="B"), {}, False),
    }


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Manifest, Path, Path]:
    root = tmp_path / "data"
    manifest = make_edi_dataset(root, tmp_path)
    runs = tmp_path / "runs"
    for variable, value in (
        ("OCRBENCH_DATA_DIR", root),
        ("OCRBENCH_FINAL_DIR", root),
        ("OCRBENCH_RUNS_DIR", runs),
    ):
        monkeypatch.setenv(variable, str(value))
    return root, manifest, runs, tmp_path


def _run(
    monkeypatch: pytest.MonkeyPatch,
    env: tuple[Path, Manifest, Path, Path],
    split: str,
    *extra: str,
) -> dict[str, Path]:
    root, manifest, runs, tmp_path = env
    built = _pipelines(root, manifest, split)

    def fake_build(config: ExternalConfig, **_: object) -> BuiltPipeline:
        return built[config.pipeline_id]

    monkeypatch.setattr(cli_module, "build_pipeline", fake_build)
    run_dirs: dict[str, Path] = {}
    for pipeline_id in built:
        config = tmp_path / f"{pipeline_id}.json"
        config.write_text(
            json.dumps({"kind": "external", "pipeline_id": pipeline_id, "command": ["x"]}),
            encoding="utf-8",
        )
        run_id = f"{pipeline_id.lower()}-{split}"
        arguments = ["run", "--pipeline", str(config), "--split", split, "--run-id", run_id]
        assert main([*arguments, *extra]) == 0
        (run_dirs[pipeline_id],) = [p for p in runs.iterdir() if p.name.endswith(f"_{run_id}")]
    return run_dirs


def test_two_fake_pipelines_produce_reports_with_every_adr_0002_metric(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    env: tuple[Path, Manifest, Path, Path],
) -> None:
    root, _manifest, _runs, tmp_path = env
    run_dirs = _run(monkeypatch, env, "dev")
    capsys.readouterr()

    for pipeline_id, run_dir in run_dirs.items():
        assert main(["report", "--run-dir", str(run_dir), "--json"]) == 0
        assert json.loads(capsys.readouterr().out)["kind"] == "aggregate"
        report = (run_dir / "report.md").read_text(encoding="utf-8")
        assert report.startswith(f"# Pipeline Benchmark Report: {pipeline_id}")
        for section in _REPORT_SECTIONS:
            assert section in report, section
        assert (run_dir / "analysis.md").is_file()
        assert _WRONG_PART not in report
        assert "HR9000" not in report
        assert "doc-" not in report

    a1_report = (run_dirs["A1"] / "report.md").read_text(encoding="utf-8")
    assert "| Field accuracy | 91.7% (11/12 fields;" in a1_report
    assert "| Sheets fully correct | 66.7% (2/3;" in a1_report
    assert (
        "| Silent errors (wrong values on sheets marked ok) | 1 fields on 1 sheets |" in a1_report
    )
    assert "| Part number (部番) | 0 | 1 |" in a1_report
    b_report = (run_dirs["B"] / "report.md").read_text(encoding="utf-8")
    assert "| Abstentions (empty fields) | 3 fields |" in b_report
    assert "| Silent errors (wrong values on sheets marked ok) | 0 fields on 0 sheets |" in b_report
    assert "| needs_review | 3 |" in b_report

    summary_path = tmp_path / "out" / "a1-summary.json"
    assert (
        main(["export-summary", "--run-dir", str(run_dirs["A1"]), "--out", str(summary_path)]) == 0
    )
    capsys.readouterr()
    exported = summary_path.read_text(encoding="utf-8")
    assert _WRONG_PART not in exported
    assert "HR9000" not in exported
    assert "900001" not in exported
    summary = json.loads(exported)
    assert summary["kind"] == "pipeline_summary"
    assert summary["summary"]["silent_errors"] == 1
    assert summary["sheets"][0]["outcomes"]["customer"] == "correct"

    assert main(["report", "--run-dir", str(run_dirs["A1"]), "--detail"]) == 0
    capsys.readouterr()
    detail = (run_dirs["A1"] / "report_detail.md").read_text(encoding="utf-8")
    assert detail.startswith("# Pipeline Detailed Report: A1")
    assert "CONFIDENTIAL" in detail
    assert _WRONG_PART in detail
    assert "Layer 1 text:" in detail
    assert "HR900000001" in detail

    assert main(["compare-runs", "--run-dirs", *map(str, run_dirs.values()), "--json"]) == 0
    comparison = json.loads(capsys.readouterr().out)
    assert comparison["pipeline_a"] == "A1"
    assert comparison["field_accuracy_difference"] == pytest.approx(0.75 - 11 / 12)
    del root


def test_final_runs_get_aggregate_reports_only(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    env: tuple[Path, Manifest, Path, Path],
) -> None:
    monkeypatch.setenv("OCRBENCH_ALLOW_FINAL", "1")
    run_dirs = _run(monkeypatch, env, "final", "--confirm-final")
    capsys.readouterr()
    run_dir = run_dirs["A1"]

    assert main(["report", "--run-dir", str(run_dir)]) == 0
    assert "Field accuracy" in (run_dir / "report.md").read_text(encoding="utf-8")
    assert main(["report", "--run-dir", str(run_dir), "--detail"]) == 2
    assert "Development" in capsys.readouterr().err
    out = env[3] / "final-summary.json"
    assert main(["export-summary", "--run-dir", str(run_dir), "--out", str(out)]) == 2
    assert "Development" in capsys.readouterr().err
    assert not out.exists()
    assert not (run_dir / "report_detail.md").exists()


def test_export_refuses_a_leaked_value(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    env: tuple[Path, Manifest, Path, Path],
) -> None:
    run_dirs = _run(monkeypatch, env, "dev")
    capsys.readouterr()
    results = load_pipeline_run(run_dirs["A1"])
    leaked = results.model_copy(
        update={
            "manifest": results.manifest.model_copy(
                update={"pipeline": {**results.manifest.pipeline, "note": _WRONG_PART}}
            )
        }
    )
    with pytest.raises(ValueError, match="forbidden"):
        export_pipeline_summary(leaked)
    assert export_pipeline_summary(results)["schema_version"] == 1


def test_renderers_check_their_inputs(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    env: tuple[Path, Manifest, Path, Path],
) -> None:
    run_dirs = _run(monkeypatch, env, "dev")
    capsys.readouterr()
    results = load_pipeline_run(run_dirs["B"])
    from ocrbench.field_metrics import summarize

    one_sheet = summarize(results.scores()[:1])
    with pytest.raises(ValueError, match="does not match"):
        render_pipeline_report(results, one_sheet)
    with pytest.raises(ValueError, match="exactly the run sheet IDs"):
        render_pipeline_detail_report(results, {})
    report = render_pipeline_report(results, summarize(results.scores()))
    assert "| Mean Layer 1 | N/A |" in report
