"""Tests for the leaderboard, `ocrbench doctor`, `run-all` and `leaderboard` (synthetic only)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import ocrbench.cli as cli_module
from ocrbench.cli import main
from ocrbench.dataset import Manifest
from ocrbench.edi_fields import EdiFields
from ocrbench.leaderboard import LeaderboardError, build_leaderboard, render_leaderboard
from ocrbench.pipeline_config import BuiltPipeline, ExternalConfig
from ocrbench.pipeline_results import PipelineRunResults, load_pipeline_run
from ocrbench.pipelines import FakePipeline, PipelineResult, PipelineStatus
from tests.helpers import make_edi_dataset, read_edi_ground_truth


def _ok(fields: EdiFields, *, ms: float = 10.0, cost: float = 0.0) -> PipelineResult:
    return PipelineResult(PipelineStatus.OK, fields, None, None, ms, ms, cost, 1)


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Manifest, Path]:
    root = tmp_path / "data"
    manifest = make_edi_dataset(root, tmp_path)
    runs = tmp_path / "runs"
    for name, value in (
        ("OCRBENCH_DATA_DIR", root),
        ("OCRBENCH_FINAL_DIR", root),
        ("OCRBENCH_RUNS_DIR", runs),
    ):
        monkeypatch.setenv(name, str(value))
    return root, manifest, runs


def _truths(root: Path, manifest: Manifest, split: str = "dev") -> list[EdiFields]:
    ids = sorted(entry.doc_id for entry in manifest.docs if entry.split == split)
    return [read_edi_ground_truth(root, doc_id) for doc_id in ids]


def _candidates(root: Path, manifest: Manifest, split: str = "dev") -> dict[str, BuiltPipeline]:
    """P: perfect but slow. Q: as accurate as P but one silent error. R: abstains. M: cloud."""
    gts = _truths(root, manifest, split)
    wrong = [gts[0].model_copy(update={"part_no": "HR-WRONG"}), *gts[1:]]
    q = [_ok(gts[0].model_copy(update={"part_no": None})), *[_ok(g) for g in gts[1:]]]
    return {
        "P": BuiltPipeline(
            FakePipeline([_ok(g, ms=50.0) for g in gts], pipeline_id="P"), {}, False
        ),
        "S": BuiltPipeline(FakePipeline([_ok(g) for g in wrong], pipeline_id="S"), {}, False),
        "Q": BuiltPipeline(FakePipeline(q, pipeline_id="Q"), {}, False),
        "M": BuiltPipeline(
            FakePipeline([_ok(g, cost=0.004) for g in gts], pipeline_id="M"), {}, True
        ),
    }


def _install(
    monkeypatch: pytest.MonkeyPatch, built: dict[str, BuiltPipeline], tmp_path: Path
) -> list[Path]:
    def fake_build(config: ExternalConfig, **_: object) -> BuiltPipeline:
        if config.pipeline_id == "BROKEN":
            raise ConnectionError("Failed to connect to Ollama.")
        return built[config.pipeline_id]

    monkeypatch.setattr(cli_module, "build_pipeline", fake_build)
    paths = []
    for name in [*built, "BROKEN"]:
        path = tmp_path / f"{name}.json"
        path.write_text(
            json.dumps({"kind": "external", "pipeline_id": name, "command": ["x"]}),
            encoding="utf-8",
        )
        paths.append(path)
    return paths


def _runs(runs: Path) -> list[PipelineRunResults]:
    return [load_pipeline_run(path) for path in sorted(runs.iterdir())]


def test_run_all_continues_past_a_failure_and_ranks_the_rest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    env: tuple[Path, Manifest, Path],
) -> None:
    root, manifest, runs = env
    paths = _install(monkeypatch, _candidates(root, manifest), tmp_path)
    out = tmp_path / "board.md"

    arguments = ["run-all", "--pipelines", *map(str, paths), "--split", "dev"]
    assert main([*arguments, "--run-prefix", "night1", "--leaderboard-out", str(out)]) == 1
    captured = capsys.readouterr()
    assert "Finished 4 of 5 candidates." in captured.out
    assert "FAILED" in captured.out and "ConnectionError" in captured.out
    assert "Running" in captured.err
    assert sorted(path.name.split("_", 1)[1] for path in runs.iterdir()) == [
        "night1-M",
        "night1-P",
        "night1-Q",
        "night1-S",
    ]

    board = out.read_text(encoding="utf-8")
    rows = [line for line in board.splitlines() if line.startswith("| ") and line[2].isdigit()]
    # Equal accuracy: no silent error beats one silent error; then faster beats slower.
    assert [row.split(" | ")[1] for row in rows] == ["M (cloud)", "P", "Q", "S"]
    assert "| 1 | M (cloud) | 100.0% | 3/3 | 0 |" in board
    assert "Decision: **local**" in board
    assert "Best local: P. Best cloud: M." in board
    assert "HR" not in board and "doc-" not in board


def test_leaderboard_ordering_and_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    env: tuple[Path, Manifest, Path],
) -> None:
    root, manifest, runs = env
    built = _candidates(root, manifest)
    del built["M"]
    paths = _install(monkeypatch, built, tmp_path)
    assert (
        main(
            ["run-all", "--pipelines", *map(str, paths[:-1]), "--split", "dev", "--run-prefix", "n"]
        )
        == 0
    )
    capsys.readouterr()

    board = build_leaderboard(_runs(runs))
    assert [entry.pipeline_id for entry in board.entries] == ["P", "Q", "S"]
    assert board.entries[0].versus_leader is None
    second = board.entries[1].versus_leader
    assert second is not None and second.field_accuracy_difference < 0
    assert board.cloud_decision is None and board.best_cloud is None
    assert "rule was not applied" in render_leaderboard(board)

    dirs = [str(path) for path in sorted(runs.iterdir())]
    assert main(["leaderboard", "--run-dirs", *dirs, "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [entry["pipeline_id"] for entry in payload["entries"]] == ["P", "Q", "S"]
    assert "cloud_decision" not in payload
    out = tmp_path / "b.md"
    assert main(["leaderboard", "--run-dirs", *dirs, "--out", str(out)]) == 0
    assert main(["leaderboard", "--run-dirs", *dirs, "--out", str(out)]) == 1  # never overwrites
    capsys.readouterr()


def test_leaderboard_rejects_unfair_comparisons(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    env: tuple[Path, Manifest, Path],
) -> None:
    root, manifest, runs = env
    built = _candidates(root, manifest)
    paths = _install(monkeypatch, {"P": built["P"]}, tmp_path)
    assert (
        main(["run-all", "--pipelines", str(paths[0]), "--split", "dev", "--run-prefix", "dev"])
        == 0
    )
    monkeypatch.setenv("OCRBENCH_ALLOW_FINAL", "1")
    final_built = _candidates(root, manifest, "final")
    _install(monkeypatch, {"P": final_built["P"]}, tmp_path)
    assert (
        main(
            [
                "run-all",
                "--pipelines",
                str(paths[0]),
                "--split",
                "final",
                "--run-prefix",
                "fin",
                "--confirm-final",
            ]
        )
        == 0
    )
    capsys.readouterr()
    dev_run, final_run = sorted(_runs(runs), key=lambda run: run.manifest.split)

    with pytest.raises(LeaderboardError, match="at least two"):
        build_leaderboard([dev_run])
    with pytest.raises(LeaderboardError, match="'split'"):
        build_leaderboard([dev_run, final_run])
    with pytest.raises(LeaderboardError, match="only once"):
        build_leaderboard([dev_run, dev_run])
    shorter = dev_run.model_copy(update={"sheets": dev_run.sheets[:1]})
    with pytest.raises(LeaderboardError, match="same sheet IDs"):
        build_leaderboard([dev_run, shorter])

    final_dir = str(next(p for p in runs.iterdir() if "fin-" in p.name))
    monkeypatch.delenv("OCRBENCH_ALLOW_FINAL")
    assert main(["leaderboard", "--run-dirs", final_dir, final_dir]) == 2
    assert "OCRBENCH_ALLOW_FINAL" in capsys.readouterr().err
    assert (
        main(["run-all", "--pipelines", str(paths[0]), "--split", "final", "--run-prefix", "x"])
        == 2
    )
    capsys.readouterr()


def test_run_all_rejects_colliding_run_ids(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], env: tuple[Path, Manifest, Path]
) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    for folder in ("a", "b"):
        (tmp_path / folder / "same.json").write_text("{}", encoding="utf-8")
    paths = [str(tmp_path / "a" / "same.json"), str(tmp_path / "b" / "same.json")]
    assert main(["run-all", "--pipelines", *paths, "--split", "dev", "--run-prefix", "p"]) == 2
    assert "same run ID" in capsys.readouterr().err


def test_doctor_reports_each_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class NotInstalled(FakePipeline):
        def describe(self) -> dict[str, str]:
            return {"kind": "fake", "pipeline_id": "P2", "ocr.version": "not installed"}

    built = {
        "OK": BuiltPipeline(FakePipeline([], pipeline_id="OK"), {}, False),
        "P2": BuiltPipeline(NotInstalled([], pipeline_id="P2"), {}, False),
    }
    paths = _install(monkeypatch, built, tmp_path)
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")

    assert main(["doctor", "--pipelines", *map(str, paths), str(bad_json)]) == 1
    out = capsys.readouterr().out
    assert "ready    OK" in out
    assert "PROBLEM  P2" in out and "not installed: ocr.version" in out
    assert "PROBLEM  BROKEN" in out and "ConnectionError: Failed to connect to Ollama." in out
    assert "PipelineConfigError" in out
    assert "1 of 4 candidates are ready." in out

    assert main(["doctor", "--pipelines", str(paths[0]), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ready"] == 1 and payload["candidates"][0]["ok"] is True
