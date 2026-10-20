"""Tests for ADR-0002 pipeline runs, their results, and the run/compare CLI (synthetic only)."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

import ocrbench.cli as cli_module
import ocrbench.pipeline_runner as runner_module
from ocrbench.cli import main
from ocrbench.dataset import Manifest, load_manifest
from ocrbench.edi_fields import EdiFields
from ocrbench.field_metrics import FieldOutcome, Layer
from ocrbench.ocr_engines import CloudNotAllowedError
from ocrbench.ollama_adapter import FakeOllamaAdapter
from ocrbench.pipeline_config import (
    BuiltPipeline,
    ExternalConfig,
    OcrLlmConfig,
    PipelineConfigError,
    VisionConfig,
    build_pipeline,
    load_pipeline_config,
    parse_pipeline_config,
)
from ocrbench.pipeline_results import is_pipeline_run, load_pipeline_run
from ocrbench.pipeline_runner import code_version, run_pipeline
from ocrbench.pipelines import FakePipeline, PipelineResult, PipelineStatus
from ocrbench.preprocess import PreprocessResult, preprocess
from ocrbench.prompts import PromptVersion
from ocrbench.runner import RunnerError
from ocrbench.splitguard import FinalAccessError
from tests.helpers import make_edi_dataset, read_edi_ground_truth


def _split_ids(manifest: Manifest, split: str) -> list[str]:
    return sorted(entry.doc_id for entry in manifest.docs if entry.split == split)


def _ok(fields: EdiFields, *, layer1_text: str | None = None, cost: float = 0.0) -> PipelineResult:
    return PipelineResult(
        status=PipelineStatus.OK,
        fields=fields,
        layer1_text=layer1_text,
        layer1_ms=1.0 if layer1_text is not None else None,
        layer2_ms=2.0,
        total_ms=3.0,
        cost_usd=cost,
        attempts=1,
    )


@pytest.fixture
def dataset(tmp_path: Path) -> tuple[Path, Manifest]:
    root = tmp_path / "data"
    return root, make_edi_dataset(root, tmp_path)


def _dev_results(root: Path, manifest: Manifest) -> list[PipelineResult]:
    """Sheet 0 correct, sheet 1 wrong part number (Layer 2), sheet 2 a format error."""
    ids = _split_ids(manifest, "dev")
    first = read_edi_ground_truth(root, ids[0])
    second = read_edi_ground_truth(root, ids[1])
    wrong = second.model_copy(update={"part_no": "HR999999999"})
    layer1 = f"{second.customer} {second.part_no} {second.order_no} {second.quantity}"
    return [
        _ok(first, layer1_text="unused"),
        _ok(wrong, layer1_text=layer1),
        PipelineResult(
            status=PipelineStatus.FORMAT_ERROR,
            fields=None,
            layer1_text="",
            layer1_ms=1.0,
            layer2_ms=1.0,
            total_ms=2.0,
            cost_usd=0.0,
            attempts=2,
        ),
    ]


def test_run_pipeline_scores_every_dev_sheet_and_records_the_configuration(
    dataset: tuple[Path, Manifest], tmp_path: Path
) -> None:
    root, manifest = dataset
    pipeline = FakePipeline(_dev_results(root, manifest), pipeline_id="A1")

    run_dir = run_pipeline(
        pipeline=pipeline,
        data_root=root,
        manifest=manifest,
        split="dev",
        runs_root=tmp_path / "runs",
        run_id="first",
        record={"prompt_hash": "abc123def456"},
    )

    assert pipeline.calls == 3
    assert run_dir.name.endswith("_first")
    assert sorted(path.name for path in run_dir.iterdir()) == ["manifest.json", "results.json"]
    results = load_pipeline_run(run_dir)
    assert is_pipeline_run(run_dir)
    manifest_record = results.manifest
    assert manifest_record.pipeline == {
        "kind": "fake",
        "pipeline_id": "A1",
        "prompt_hash": "abc123def456",
    }
    assert manifest_record.split == "dev"
    assert manifest_record.order_no_rule == "keep"
    assert manifest_record.cloud is False
    assert len(manifest_record.dataset_fingerprint) == 64

    first, second, third = results.sheets
    assert all(outcome is FieldOutcome.CORRECT for outcome in first.outcomes.values())
    assert first.failed_layer == {}
    assert second.outcomes["part_no"] is FieldOutcome.WRONG
    assert second.failed_layer == {"part_no": Layer.LAYER2}
    assert third.status is PipelineStatus.FORMAT_ERROR
    assert set(third.outcomes.values()) == {FieldOutcome.MISSING}
    assert third.failed_layer == dict.fromkeys(third.outcomes, Layer.LAYER1)
    assert first.timings_ms["preprocess"] is not None
    assert [score.sheet_id for score in results.scores()] == _split_ids(manifest, "dev")


def test_pipeline_exception_becomes_a_content_free_error_sheet(
    dataset: tuple[Path, Manifest], tmp_path: Path
) -> None:
    root, manifest = dataset

    class Exploding:
        pipeline_id = "boom"

        def describe(self) -> dict[str, str]:
            return {"kind": "test", "pipeline_id": "boom"}

        def extract(self, image: PreprocessResult) -> PipelineResult:
            raise RuntimeError("secret sheet value HR900000001")

    run_dir = run_pipeline(
        pipeline=Exploding(),
        data_root=root,
        manifest=manifest,
        split="dev",
        runs_root=tmp_path / "runs",
    )

    text = (run_dir / "results.json").read_text(encoding="utf-8")
    assert "secret" not in text
    sheets = load_pipeline_run(run_dir).sheets
    assert {sheet.status for sheet in sheets} == {PipelineStatus.ERROR}
    assert sheets[0].error == "pipeline raised RuntimeError"


def test_final_split_needs_both_confirmations(
    dataset: tuple[Path, Manifest], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, manifest = dataset
    pipeline = FakePipeline(
        [_ok(read_edi_ground_truth(root, i)) for i in _split_ids(manifest, "final")]
    )
    monkeypatch.delenv("OCRBENCH_ALLOW_FINAL", raising=False)

    with pytest.raises(FinalAccessError):
        run_pipeline(
            pipeline=pipeline,
            data_root=root,
            manifest=manifest,
            split="final",
            runs_root=tmp_path / "runs",
            confirm_final_flag=True,
        )
    assert pipeline.calls == 0

    monkeypatch.setenv("OCRBENCH_ALLOW_FINAL", "1")
    run_dir = run_pipeline(
        pipeline=pipeline,
        data_root=root,
        manifest=manifest,
        split="final",
        runs_root=tmp_path / "runs",
        confirm_final_flag=True,
    )
    assert len(load_pipeline_run(run_dir).sheets) == 2


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"split": "selection"}, "Unsupported split"),
        ({"run_id": "../escape"}, "run_id"),
        ({"record": {"kind": "other"}}, "repeats pipeline description keys: kind"),
    ],
)
def test_invalid_run_requests_fail_before_any_sheet_is_read(
    dataset: tuple[Path, Manifest],
    tmp_path: Path,
    change: dict[str, object],
    message: str,
) -> None:
    root, manifest = dataset
    pipeline = FakePipeline([])
    arguments: dict[str, object] = {
        "pipeline": pipeline,
        "data_root": root,
        "manifest": manifest,
        "split": "dev",
        "runs_root": tmp_path / "runs",
        **change,
    }

    with pytest.raises(RunnerError, match=message):
        run_pipeline(**arguments)  # type: ignore[arg-type]
    assert pipeline.calls == 0


def test_existing_output_and_wrong_ground_truth_kind_are_rejected(
    dataset: tuple[Path, Manifest], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, manifest = dataset
    runs = tmp_path / "runs"
    run_pipeline(
        pipeline=FakePipeline(_dev_results(root, manifest)),
        data_root=root,
        manifest=manifest,
        split="dev",
        runs_root=runs,
        run_id="same",
    )
    stamp = datetime.strptime(next(runs.iterdir()).name.split("_")[0], "%Y%m%dT%H%M%SZ")
    monkeypatch.setattr(runner_module, "_utc_now", lambda: stamp.replace(tzinfo=UTC))
    again = FakePipeline(_dev_results(root, manifest))
    with pytest.raises(RunnerError, match="already exists"):
        run_pipeline(
            pipeline=again,
            data_root=root,
            manifest=manifest,
            split="dev",
            runs_root=runs,
            run_id="same",
        )
    assert again.calls == 0

    other = load_manifest(root, ground_truth="edi_fields").model_copy(update={"docs": []})
    with pytest.raises(RunnerError, match="does not match"):
        run_pipeline(
            pipeline=FakePipeline([]),
            data_root=root,
            manifest=other,
            split="dev",
            runs_root=runs,
        )


def test_order_document_dataset_is_not_accepted(tmp_path: Path) -> None:
    from tests.helpers import make_mini_dataset

    root = tmp_path / "legacy"
    manifest = make_mini_dataset(root)
    with pytest.raises(RunnerError, match="edi_fields"):
        run_pipeline(
            pipeline=FakePipeline([]),
            data_root=root,
            manifest=manifest,
            split="dev",
            runs_root=tmp_path / "runs",
        )


def test_tampered_results_are_rejected(dataset: tuple[Path, Manifest], tmp_path: Path) -> None:
    root, manifest = dataset
    run_dir = run_pipeline(
        pipeline=FakePipeline(_dev_results(root, manifest)),
        data_root=root,
        manifest=manifest,
        split="dev",
        runs_root=tmp_path / "runs",
    )
    manifest_path = run_dir / "manifest.json"
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    document["cloud"] = True
    manifest_path.write_text(json.dumps(document), encoding="utf-8")

    from ocrbench.results import ResultsError

    with pytest.raises(ResultsError, match="does not match"):
        load_pipeline_run(run_dir)
    assert not is_pipeline_run(tmp_path / "missing")


def test_code_version_reports_the_commit_or_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    commit, _dirty = code_version()
    assert commit is None or len(commit) == 40

    monkeypatch.setattr(runner_module, "_git", lambda *arguments: None)
    assert code_version() == (None, None)


# --- pipeline configuration -------------------------------------------------------------


def _prompt(name: str) -> PromptVersion:
    text = f"Extract the 4 EDI fields ({name}). Return JSON only."
    return PromptVersion(name, hashlib.sha256(text.encode()).hexdigest()[:12], text)


def test_config_kinds_parse_and_build() -> None:
    adapter = FakeOllamaAdapter(digest="sha256:layer2")
    ocr_llm = parse_pipeline_config(
        json.dumps(
            {
                "kind": "ocr_llm",
                "pipeline_id": "A3",
                "ocr": {"engine": "qwen_vl"},
                "llm_model": "qwen3:8b",
                "prompt_name": "edi-fields",
            }
        )
    )
    assert isinstance(ocr_llm, OcrLlmConfig)
    built = build_pipeline(ocr_llm, adapter=lambda: adapter, prompt=_prompt)
    assert built.cloud is False
    assert built.record == {
        "llm_digest": "sha256:layer2",
        "prompt_hash": _prompt("edi-fields").hash,
        "prompt_name": "edi-fields",
    }
    assert built.pipeline.describe()["ocr_engine"] == "qwen-vl"

    for engine in ({"engine": "tesseract", "psm": 4}, {"engine": "paddleocr"}):
        config = parse_pipeline_config(
            json.dumps(
                {
                    "kind": "ocr_llm",
                    "pipeline_id": "A",
                    "ocr": engine,
                    "llm_model": "qwen3:8b",
                    "prompt_name": "edi-fields",
                }
            )
        )
        assert build_pipeline(config, adapter=lambda: adapter, prompt=_prompt).cloud is False

    vision = parse_pipeline_config(
        '{"kind": "vision", "pipeline_id": "B", "model": "gemma4:12b", "prompt_name": "base"}'
    )
    assert isinstance(vision, VisionConfig)
    assert build_pipeline(vision, adapter=lambda: adapter, prompt=_prompt).record[
        "vision_digest"
    ] == ("sha256:layer2")

    def no_adapter() -> FakeOllamaAdapter:
        raise AssertionError("an external command must not need Ollama")

    external = parse_pipeline_config(
        '{"kind": "external", "pipeline_id": "C", "command": ["harness"], "version": "1"}'
    )
    assert isinstance(external, ExternalConfig)
    built_external = build_pipeline(external, adapter=no_adapter, prompt=_prompt)
    assert built_external.record == {}
    assert built_external.pipeline.describe()["command"] == "harness"


def test_mistral_config_is_cloud_and_locked(monkeypatch: pytest.MonkeyPatch) -> None:
    config = parse_pipeline_config(
        json.dumps(
            {
                "kind": "ocr_llm",
                "pipeline_id": "A4",
                "ocr": {"engine": "mistral", "usd_per_page": 0.004},
                "llm_model": "qwen3:8b",
                "prompt_name": "edi-fields",
            }
        )
    )
    monkeypatch.delenv("OCRBENCH_ALLOW_CLOUD", raising=False)
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    with pytest.raises(CloudNotAllowedError):
        build_pipeline(config, adapter=FakeOllamaAdapter, prompt=_prompt, confirm_cloud=True)

    monkeypatch.setenv("OCRBENCH_ALLOW_CLOUD", "1")
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key-not-real")
    built = build_pipeline(config, adapter=FakeOllamaAdapter, prompt=_prompt, confirm_cloud=True)
    assert built.cloud is True
    assert "test-key-not-real" not in json.dumps(built.pipeline.describe())


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("not json", "not valid JSON"),
        ('{"kind": "vision", "pipeline_id": "B"}', "at: model, prompt_name"),
        (
            '{"kind": "ocr_llm", "pipeline_id": "A", "ocr": {"engine": "tesseract", "psm": 99},'
            ' "llm_model": "m", "prompt_name": "p"}',
            "at: ocr.psm",
        ),
        ('{"kind": "external", "pipeline_id": "../x", "command": ["SECRET-CMD"]}', "pipeline_id"),
        ('{"kind": "teleport", "pipeline_id": "X"}', "(root)"),
        (
            '{"kind": "external", "pipeline_id": "C", "command": ["c"], "extra": "SECRET"}',
            "extra",
        ),
    ],
)
def test_invalid_configs_name_keys_not_values(text: str, message: str) -> None:
    with pytest.raises(PipelineConfigError, match=r"invalid|not valid") as caught:
        parse_pipeline_config(text)
    assert message in str(caught.value)
    assert "SECRET" not in str(caught.value)


def test_unreadable_config_file(tmp_path: Path) -> None:
    with pytest.raises(PipelineConfigError, match="cannot read"):
        load_pipeline_config(tmp_path / "missing.json")


# --- CLI ----------------------------------------------------------------------------------


@pytest.fixture
def cli_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dataset: tuple[Path, Manifest]
) -> tuple[Path, Manifest, Path]:
    root, manifest = dataset
    runs = tmp_path / "runs"
    monkeypatch.setenv("OCRBENCH_DATA_DIR", str(root))
    monkeypatch.setenv("OCRBENCH_FINAL_DIR", str(root))
    monkeypatch.setenv("OCRBENCH_RUNS_DIR", str(runs))
    monkeypatch.setenv("OCRBENCH_PROMPTS_DIR", str(tmp_path / "prompts"))
    return root, manifest, runs


def _register_prompt(tmp_path: Path, name: str, capsys: pytest.CaptureFixture[str]) -> None:
    text = "Extract customer, part_no, order_no and quantity. Return JSON only."
    prompt_file = tmp_path / f"{name}.txt"
    prompt_file.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(text.encode()).hexdigest()[:12]
    assert main(["prompt", "add", "--name", name, "--file", str(prompt_file)]) == 0
    assert main(["prompt", "activate", "--name", name, "--hash", digest]) == 0
    capsys.readouterr()


def _external_harness(tmp_path: Path, root: Path, manifest: Manifest) -> list[str]:
    """A synthetic external command that answers by the preprocessed sheet's hash."""
    answers: dict[str, object] = {}
    for entry in manifest.docs:
        png = preprocess(root / "raw" / entry.file).png_bytes
        fields = read_edi_ground_truth(root, entry.doc_id).model_dump(mode="json")
        answers[hashlib.sha256(png).hexdigest()] = fields
    mapping = tmp_path / "harness-answers.json"
    mapping.write_text(json.dumps(answers), encoding="utf-8")
    script = tmp_path / "harness.py"
    script.write_text(
        "import hashlib, json, sys\n"
        "answers = json.load(open(sys.argv[1], encoding='utf-8'))\n"
        "digest = hashlib.sha256(open(sys.argv[2], 'rb').read()).hexdigest()\n"
        "print(json.dumps({'status': 'ok', 'fields': answers[digest]}))\n",
        encoding="utf-8",
    )
    return [sys.executable, str(script), str(mapping)]


def _write_config(path: Path, config: dict[str, object]) -> Path:
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def _only_run(runs: Path, run_id: str) -> Path:
    (match,) = [path for path in runs.iterdir() if path.name.endswith(f"_{run_id}")]
    return match


def test_cli_runs_two_pipelines_and_compares_them(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    cli_env: tuple[Path, Manifest, Path],
) -> None:
    root, manifest, runs = cli_env
    _register_prompt(tmp_path, "edi-fields", capsys)

    external = _write_config(
        tmp_path / "c.json",
        {
            "kind": "external",
            "pipeline_id": "C",
            "command": _external_harness(tmp_path, root, manifest),
            "timeout_s": 60,
        },
    )
    assert (
        main(["run", "--pipeline", str(external), "--split", "dev", "--run-id", "c", "--json"]) == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["n_sheets"] == 3
    assert payload["summary"]["field_accuracy"] == 1.0

    # A3 with a fake Ollama: transcription, then Layer 2 fields, per sheet.
    responses: list[str] = []
    for doc_id in _split_ids(manifest, "dev"):
        gt = read_edi_ground_truth(root, doc_id)
        responses.append(json.dumps({"text": f"{gt.customer}\n{gt.order_no}\n{gt.quantity}"}))
        responses.append(json.dumps(gt.model_copy(update={"part_no": None}).model_dump()))
    adapter = FakeOllamaAdapter(responses, digest="sha256:fake-layer2")
    monkeypatch.setattr(cli_module, "create_adapter", lambda: adapter)
    a3 = _write_config(
        tmp_path / "a3.json",
        {
            "kind": "ocr_llm",
            "pipeline_id": "A3",
            "ocr": {"engine": "qwen_vl"},
            "llm_model": "qwen3:8b",
            "prompt_name": "edi-fields",
        },
    )
    assert main(["run", "--pipeline", str(a3), "--split", "dev", "--run-id", "a3"]) == 0
    human = capsys.readouterr().out
    assert '"field_accuracy": 0.75' in human
    a3_run = load_pipeline_run(_only_run(runs, "a3"))
    assert a3_run.manifest.pipeline["llm_digest"] == "sha256:fake-layer2"
    assert a3_run.manifest.pipeline["prompt_name"] == "edi-fields"
    assert all(sheet.failed_layer == {"part_no": Layer.LAYER1} for sheet in a3_run.sheets)

    c_dir, a3_dir = _only_run(runs, "c"), _only_run(runs, "a3")
    assert main(["compare-runs", "--run-dirs", str(a3_dir), str(c_dir), "--json"]) == 0
    comparison = json.loads(capsys.readouterr().out)
    assert comparison["kind"] == "field_comparison"
    assert comparison["pipeline_a"] == "A3"
    assert comparison["field_accuracy_difference"] == pytest.approx(0.25)
    assert comparison["sheet_b_only"] == 3
    assert "cloud_decision" not in comparison

    assert main(["compare-runs", "--run-dirs", str(a3_dir), str(c_dir)]) == 0
    assert "B=C minus A=A3" in capsys.readouterr().out


def test_cli_compare_applies_the_cloud_rule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    cli_env: tuple[Path, Manifest, Path],
) -> None:
    root, manifest, runs = cli_env
    ids = _split_ids(manifest, "dev")
    perfect = [_ok(read_edi_ground_truth(root, i)) for i in ids]
    built: dict[str, BuiltPipeline] = {
        "local": BuiltPipeline(FakePipeline(perfect, pipeline_id="A1"), {}, False),
        "cloud": BuiltPipeline(
            FakePipeline(
                [_ok(read_edi_ground_truth(root, i), cost=0.004) for i in ids], pipeline_id="A4"
            ),
            {},
            True,
        ),
    }

    def fake_build(config: ExternalConfig, **_: object) -> BuiltPipeline:
        return built[config.pipeline_id]

    monkeypatch.setattr(cli_module, "build_pipeline", fake_build)
    for name in built:
        config = _write_config(
            tmp_path / f"{name}.json",
            {"kind": "external", "pipeline_id": name, "command": ["unused"]},
        )
        assert main(["run", "--pipeline", str(config), "--split", "dev", "--run-id", name]) == 0
    capsys.readouterr()

    cloud_dir, local_dir = _only_run(runs, "cloud"), _only_run(runs, "local")
    assert main(["compare-runs", "--run-dirs", str(cloud_dir), str(local_dir), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    decision = payload["cloud_decision"]
    assert decision["choice"] == "local"
    assert decision["cloud_pipeline"] == "A4"
    assert decision["local_pipeline"] == "A1"
    assert "cloud is not more accurate" in decision["reasons"][1]


def test_cli_final_pipeline_run_prints_only_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    cli_env: tuple[Path, Manifest, Path],
) -> None:
    root, manifest, runs = cli_env
    config = _write_config(
        tmp_path / "c.json",
        {
            "kind": "external",
            "pipeline_id": "C",
            "command": _external_harness(tmp_path, root, manifest),
        },
    )
    arguments = ["run", "--pipeline", str(config), "--split", "final"]
    monkeypatch.delenv("OCRBENCH_ALLOW_FINAL", raising=False)
    assert main(arguments) == 2
    assert "OCRBENCH_ALLOW_FINAL" in capsys.readouterr().err

    monkeypatch.setenv("OCRBENCH_ALLOW_FINAL", "1")
    assert main([*arguments, "--confirm-final", "--run-id", "f1"]) == 0
    output = capsys.readouterr().out
    assert output.startswith("Completed 2 sheets.")
    assert "field_accuracy" not in output

    assert main([*arguments, "--confirm-final", "--run-id", "f2"]) == 0
    capsys.readouterr()
    dirs = [str(_only_run(runs, "f1")), str(_only_run(runs, "f2"))]
    monkeypatch.delenv("OCRBENCH_ALLOW_FINAL")
    assert main(["compare-runs", "--run-dirs", *dirs]) == 2
    capsys.readouterr()


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--model", "gemma4:12b"], "cannot be combined"),
        (["--split", "selection"], "dev and final"),
    ],
)
def test_cli_pipeline_argument_errors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    cli_env: tuple[Path, Manifest, Path],
    extra: list[str],
    message: str,
) -> None:
    config = _write_config(
        tmp_path / "c.json", {"kind": "external", "pipeline_id": "C", "command": ["x"]}
    )
    split = [] if "--split" in extra else ["--split", "dev"]
    assert main(["run", "--pipeline", str(config), *split, *extra]) == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["run", "--split", "dev", "--model", "gemma4:12b"], "--model and --prompt-name"),
        (
            [
                "run",
                "--split",
                "dev",
                "--model",
                "gemma4:12b",
                "--prompt-name",
                "b",
                "--confirm-cloud",
            ],
            "only to --pipeline",
        ),
    ],
)
def test_cli_legacy_run_argument_errors(
    capsys: pytest.CaptureFixture[str], arguments: list[str], message: str
) -> None:
    assert main(arguments) == 2
    assert message in capsys.readouterr().err


def test_cli_cloud_pipeline_without_locks_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    cli_env: tuple[Path, Manifest, Path],
) -> None:
    _register_prompt(tmp_path, "edi-fields", capsys)
    monkeypatch.delenv("OCRBENCH_ALLOW_CLOUD", raising=False)
    monkeypatch.setattr(cli_module, "create_adapter", FakeOllamaAdapter)
    config = _write_config(
        tmp_path / "a4.json",
        {
            "kind": "ocr_llm",
            "pipeline_id": "A4",
            "ocr": {"engine": "mistral"},
            "llm_model": "qwen3:8b",
            "prompt_name": "edi-fields",
        },
    )
    assert main(["run", "--pipeline", str(config), "--split", "dev"]) == 2
    assert "cloud OCR is locked" in capsys.readouterr().err
    assert not cli_env[2].exists()


def test_cli_compare_rejects_mixed_and_mismatched_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    cli_env: tuple[Path, Manifest, Path],
) -> None:
    root, manifest, runs = cli_env
    config = _write_config(
        tmp_path / "c.json",
        {
            "kind": "external",
            "pipeline_id": "C",
            "command": _external_harness(tmp_path, root, manifest),
        },
    )
    assert main(["run", "--pipeline", str(config), "--split", "dev", "--run-id", "d"]) == 0
    monkeypatch.setenv("OCRBENCH_ALLOW_FINAL", "1")
    final = ["run", "--pipeline", str(config), "--split", "final", "--confirm-final"]
    assert main([*final, "--run-id", "f"]) == 0
    capsys.readouterr()
    dev_dir, final_dir = _only_run(runs, "d"), _only_run(runs, "f")

    checks: list[tuple[list[str], str]] = [
        ([str(dev_dir)], "exactly two"),
        ([str(dev_dir), str(final_dir)], "'split'"),
        ([str(dev_dir), str(tmp_path / "not-a-run")], "cannot be compared"),
    ]
    for run_dirs, message in checks:
        assert main(["compare-runs", "--run-dirs", *run_dirs]) == 2
        assert message in capsys.readouterr().err


def test_cli_pipeline_run_needs_configured_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("OCRBENCH_DATA_DIR", raising=False)
    config = _write_config(
        tmp_path / "c.json", {"kind": "external", "pipeline_id": "C", "command": ["x"]}
    )
    assert main(["run", "--pipeline", str(config), "--split", "dev"]) == 2
    assert "OCRBENCH_DATA_DIR" in capsys.readouterr().err


def _sheet_document() -> dict[str, object]:
    return {
        "sheet_id": "doc-0123456789ab",
        "source_kind": "scan",
        "status": "ok",
        "fields": {"customer": "c", "part_no": "p", "order_no": "o", "quantity": 1},
        "layer1_text": None,
        "raw_outputs": [],
        "outcomes": dict.fromkeys(("customer", "part_no", "order_no", "quantity"), "correct"),
        "failed_layer": {},
        "timings_ms": {"preprocess": 1.0, "layer1": None, "layer2": None, "total": 1.0},
        "cost_usd": 0.0,
        "attempts": 1,
        "error": None,
    }


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"sheet_id": "sheet-1"}, "anonymous"),
        ({"outcomes": {"customer": "correct"}}, "4 EDI fields"),
        ({"failed_layer": {"customer": "layer1"}}, "not correct"),
        ({"timings_ms": {"total": 1.0}}, "timings_ms"),
        ({"fields": None}, "ok sheet"),
        ({"status": "error"}, "must not have fields"),
    ],
)
def test_sheet_result_contract(change: dict[str, object], message: str) -> None:
    from ocrbench.pipeline_results import SheetResult

    assert SheetResult.model_validate_json(json.dumps(_sheet_document())).score().all_correct
    with pytest.raises(ValueError, match=message):
        SheetResult.model_validate_json(json.dumps({**_sheet_document(), **change}))


@pytest.mark.parametrize(
    "path", sorted((Path(__file__).parents[1] / "pipelines").glob("*.json")), ids=lambda p: p.stem
)
def test_example_pipeline_configs_are_valid(path: Path) -> None:
    config = load_pipeline_config(path)
    assert path.stem.startswith(config.pipeline_id)
