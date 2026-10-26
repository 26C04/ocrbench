"""Tests for the D2 customer-rule preview on synthetic Development runs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import ocrbench.cli as cli_module
from ocrbench.cli import main
from ocrbench.pipeline_config import BuiltPipeline, ExternalConfig
from ocrbench.pipelines import FakePipeline, PipelineResult, PipelineStatus
from scripts import customer_rule_preview as preview
from tests.helpers import make_edi_dataset, read_edi_ground_truth


def test_loose_site_accepts_close_names_and_rejects_others() -> None:
    assert preview.loose_site("Nidecﾏｼﾝﾂｰﾙ株式会社") == "hwao_nmtj"
    assert preview.loose_site("Nidecマシンツ-ル株式会社") == "hwao_nmtj"
    assert preview.loose_site("山田大作") is None
    assert preview.loose_site("株式会社") is None
    assert preview.loose_site(None) is None


def _run_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    root = tmp_path / "data"
    manifest = make_edi_dataset(root, tmp_path)
    runs = tmp_path / "runs"
    for name, value in (
        ("OCRBENCH_DATA_DIR", root),
        ("OCRBENCH_FINAL_DIR", root),
        ("OCRBENCH_RUNS_DIR", runs),
    ):
        monkeypatch.setenv(name, str(value))
    ids = sorted(entry.doc_id for entry in manifest.docs if entry.split == "dev")
    gts = [read_edi_ground_truth(root, doc_id) for doc_id in ids]
    blurred = gts[0].model_copy(update={"customer": "Nidecﾏｼﾝﾂ-ﾙ株式会社"})
    results = [
        PipelineResult(PipelineStatus.NEEDS_REVIEW, blurred, None, None, 1.0, 1.0, 0.0, 1),
        *[PipelineResult(PipelineStatus.OK, g, None, None, 1.0, 1.0, 0.0, 1) for g in gts[1:]],
    ]
    built = BuiltPipeline(FakePipeline(results, pipeline_id="P"), {}, False)

    def fake_build(config: ExternalConfig, **_: object) -> BuiltPipeline:
        return built

    monkeypatch.setattr(cli_module, "build_pipeline", fake_build)
    config = tmp_path / "P.json"
    config.write_text(
        json.dumps({"kind": "external", "pipeline_id": "P", "command": ["x"]}), encoding="utf-8"
    )
    assert main(["run-all", "--pipelines", str(config), "--split", "dev", "--run-prefix", "d"]) == 0
    (run_dir,) = runs.iterdir()
    return run_dir, root


def test_preview_counts_the_sheet_the_loose_rule_rescues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir, root = _run_dir(tmp_path, monkeypatch)
    capsys.readouterr()

    result = preview.preview_run(run_dir, root)

    n = result.n_sheets
    assert (result.customer_exact, result.customer_loose) == (n - 1, n)
    assert result.fields_loose == result.fields_exact + 1
    assert (result.ready_exact, result.ready_loose, result.silent_loose) == (n - 1, n, 0)

    assert preview.main(["--run-dirs", str(run_dir), "--data-root", str(root)]) == 0
    printed = capsys.readouterr().out
    assert f"| P | {n - 1}/{n} | {n}/{n} |" in printed
    assert "Nidec" not in printed and "doc-" not in printed
