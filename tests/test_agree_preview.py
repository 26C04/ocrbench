"""Tests for the two-reads preview on synthetic Development runs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import ocrbench.cli as cli_module
from ocrbench.cli import main
from ocrbench.edi_fields import EdiFields
from ocrbench.pipeline_config import BuiltPipeline, ExternalConfig
from ocrbench.pipelines import FakePipeline, PipelineResult, PipelineStatus
from scripts import agree_preview as preview
from tests.helpers import make_edi_dataset, read_edi_ground_truth

NIDEC = "Nidecマシンツール株式会社"


def _f(**overrides: object) -> EdiFields:
    values: dict[str, object] = {
        "customer": NIDEC,
        "part_no": "HR000000001",
        "order_no": "000001-2026",
        "quantity": 4,
    }
    return EdiFields.model_validate({**values, **overrides})


def test_combine_takes_a_mapped_customer_and_needs_agreement() -> None:
    assert preview.combine(_f(customer="社"), _f()) == _f()
    differ = preview.combine(_f(order_no="000009-2026"), _f())
    one_empty = preview.combine(_f(quantity=None), _f())
    assert differ is not None and differ.order_no is None
    assert one_empty is not None and one_empty.quantity == 4
    assert preview.combine(None, _f()) == _f()
    assert preview.combine(_f(), None) == _f()


def test_preview_scores_both_runs_and_the_combination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
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

    def ok(fields: EdiFields, status: PipelineStatus = PipelineStatus.OK) -> PipelineResult:
        return PipelineResult(status, fields, "t", 1.0, 1.0, 1.0, 0.0, 1)

    lost = gts[0].model_copy(update={"customer": "社"})
    wrong = gts[1].model_copy(update={"part_no": "HR999999999"})
    built = {
        "A": BuiltPipeline(
            FakePipeline(
                [ok(lost, PipelineStatus.NEEDS_REVIEW), ok(wrong), *map(ok, gts[2:])],
                pipeline_id="A",
            ),
            {},
            False,
        ),
        "B": BuiltPipeline(FakePipeline([ok(g) for g in gts], pipeline_id="B"), {}, False),
    }

    def fake_build(config: ExternalConfig, **_: object) -> BuiltPipeline:
        return built[config.pipeline_id]

    monkeypatch.setattr(cli_module, "build_pipeline", fake_build)
    paths = []
    for name in built:
        path = tmp_path / f"{name}.json"
        path.write_text(
            json.dumps({"kind": "external", "pipeline_id": name, "command": ["x"]}),
            encoding="utf-8",
        )
        paths.append(str(path))
    assert main(["run-all", "--pipelines", *paths, "--split", "dev", "--run-prefix", "d"]) == 0
    run_a, run_b = sorted(runs.iterdir(), key=lambda p: p.name.split("-")[-1])
    capsys.readouterr()

    rows = preview.preview(run_a, run_b, root)

    n = len(gts)
    a, _, combined = rows
    assert (a.customer, a.silent) == (n - 1, 1)
    assert (combined.customer, combined.silent, combined.ready) == (n, 0, n - 1)
    assert combined.fields == 4 * n - 1
    assert (
        preview.main(["--run-a", str(run_a), "--run-b", str(run_b), "--data-root", str(root)]) == 0
    )
    printed = capsys.readouterr().out
    assert "| combined |" in printed and "Nidec" not in printed and "doc-" not in printed
