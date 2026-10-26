"""Synthetic coverage for the five-model first-pass ranking helper."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from PIL import Image

import ocrbench.cli as cli_module
from ocrbench.cli import main
from ocrbench.ollama_adapter import FakeOllamaAdapter
from scripts import rank_models

_GT: dict[str, object] = {
    "customer_name": "Synthetic Customer",
    "order_no": "SYNTHETIC-ORDER-0001",
    "delivery_date": "2026-08-01",
    "items": [{"part_no": "SYNTHETIC-PART", "material": "Synthetic Alloy", "num_pieces": 1}],
}
_WRONG: dict[str, object] = {**_GT, "order_no": "WRONG"}


def _prepare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Build a two-document Development dataset and return the runs root."""
    data_root = tmp_path / "data"
    (data_root / "raw").mkdir(parents=True)
    (data_root / "gt").mkdir()
    for index in range(2):
        Image.new("RGB", (24, 18), (index * 40, 20, 30)).save(data_root / "raw" / f"in-{index}.png")
    assignments = tmp_path / "assignments.csv"
    with assignments.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["file", "source_kind", "split"])
        writer.writerow(["in-0.png", "scan", "dev"])
        writer.writerow(["in-1.png", "photo", "dev"])
    runs_root = tmp_path / "runs"
    monkeypatch.setenv("OCRBENCH_DATA_DIR", str(data_root))
    monkeypatch.setenv("OCRBENCH_RUNS_DIR", str(runs_root))
    monkeypatch.setenv("OCRBENCH_ADAPTER", "fake")
    assert (
        main(
            [
                "dataset",
                "build-manifest",
                "--root",
                str(data_root),
                "--assignments",
                str(assignments),
            ]
        )
        == 0
    )
    manifest = json.loads((data_root / "manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["docs"]:
        (data_root / "gt" / f"{entry['doc_id']}.json").write_text(json.dumps(_GT), encoding="utf-8")
    return runs_root


def _run(
    monkeypatch: pytest.MonkeyPatch,
    runs_root: Path,
    model: str,
    responses: list[dict[str, object]],
    *,
    fully_on_gpu: bool = True,
) -> Path:
    adapter = FakeOllamaAdapter([json.dumps(item) for item in responses], fully_on_gpu=fully_on_gpu)
    monkeypatch.setattr(cli_module, "create_adapter", lambda: adapter)
    run_id = model.replace(":", "-").replace(".", "-")
    assert (
        main(
            ["run", "--model", model, "--split", "dev", "--prompt-name", "base", "--run-id", run_id]
        )
        == 0
    )
    (run_dir,) = runs_root.glob(f"*_{run_id}")
    return run_dir


def test_ranks_primary_runs_by_exact_match_and_picks_top_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    runs_root = _prepare(tmp_path, monkeypatch)
    weak = _run(monkeypatch, runs_root, "gemma4:12b", [_WRONG, _WRONG])
    strong = _run(monkeypatch, runs_root, "qwen3.5:9b", [_GT, _GT])
    offload = _run(monkeypatch, runs_root, "ministral-3:14b", [_GT, _GT], fully_on_gpu=False)
    middle = _run(monkeypatch, runs_root, "minicpm-v4.5:8b", [_GT, _WRONG])
    capsys.readouterr()

    out = tmp_path / "ranking.md"
    assert (
        rank_models.main(
            ["--run-dirs", str(weak), str(offload), str(middle), str(strong), "--out", str(out)]
        )
        == 0
    )
    report = capsys.readouterr().out
    assert out.read_text(encoding="utf-8") == report

    order = [run.model_tag for run in rank_models.load_runs([weak, offload, middle, strong])]
    assert order == ["qwen3.5:9b", "minicpm-v4.5:8b", "gemma4:12b", "ministral-3:14b"]
    selected = report.split("## Selected")[1]
    assert "`qwen3.5:9b`" in selected
    assert "`minicpm-v4.5:8b`" in selected
    assert "ministral" not in selected
    assert "offload (reference)" in report


def test_rejects_duplicate_model_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    runs_root = _prepare(tmp_path, monkeypatch)
    run_dir = _run(monkeypatch, runs_root, "gemma4:12b", [_GT, _GT])
    capsys.readouterr()

    assert rank_models.main(["--run-dirs", str(run_dir), str(run_dir)]) == 2
    assert "more than one run" in capsys.readouterr().err
