"""Tests for the candidate C adapter around the template harness."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from ocrbench.pipeline_config import ExternalConfig, parse_pipeline_config
from scripts import harness_candidate

SLIP: dict[str, Any] = {
    "status": "ok",
    "edi": {
        "site_id": "hwao_nmtj",
        "customer": "nidecマシンツール",
        "order_no": "000001",
        "order_no_full": "000001-2026",
        "part_no": "HR000000001",
        "quantity": 4,
        "unit": "PC",
    },
    "fields": {
        "customer": {
            "reads": {"header": "Nidecﾏｼﾝﾂｰﾙ株式会社", "verify": "Nidecﾏｼﾝﾂｰﾙ株式会社"},
            "agreement": "header=verify",
        }
    },
}


def test_convert_maps_the_harness_edi_keys() -> None:
    assert harness_candidate.convert({"slips": [SLIP]}) == {
        "status": "ok",
        "fields": {
            "customer": "Nidecマシンツール株式会社",
            "part_no": "HR000000001",
            "order_no": "000001-2026",
            "quantity": 4,
        },
        "cost_usd": 0.0,
    }


def test_convert_printed_customer_and_review() -> None:
    review = {**SLIP, "status": "needs_review", "edi": {**SLIP["edi"], "site_id": None}}
    result = harness_candidate.convert({"slips": [review]}, "printed")
    assert result["status"] == "needs_review"
    assert result["fields"]["customer"] == "Nidecﾏｼﾝﾂｰﾙ株式会社"
    assert harness_candidate.convert({"slips": [review]})["fields"]["customer"] is None


def test_convert_without_exactly_one_slip_needs_review() -> None:
    assert harness_candidate.convert({"slips": []})["fields"] is None
    assert harness_candidate.convert({"slips": [SLIP, SLIP]})["status"] == "needs_review"


def test_main_runs_the_harness_and_prints_only_the_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    harness = tmp_path / "automation-harness"
    harness.mkdir()
    (harness / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        calls.append(command)
        out = Path(command[command.index("--out") + 1])
        out.write_text(json.dumps({"slips": [SLIP]}), encoding="utf-8")
        return subprocess.CompletedProcess(command, 10, b"secret sheet values", b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert harness_candidate.main([str(tmp_path / "sheet.png"), "--harness", str(harness)]) == 0
    printed = capsys.readouterr().out
    assert json.loads(printed)["fields"]["part_no"] == "HR000000001"
    assert "secret" not in printed
    assert calls[0][:4] == ["uv", "run", "--project", str(harness)]


def test_main_reports_a_missing_checkout_or_failed_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert harness_candidate.main(["sheet.png", "--harness", str(tmp_path / "none")]) == 2
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 6, b"", b""),
    )
    assert harness_candidate.main(["sheet.png", "--harness", str(tmp_path)]) == 1


def test_c_config_is_an_external_candidate() -> None:
    config = parse_pipeline_config(Path("pipelines/C-harness.json").read_text(encoding="utf-8"))
    assert isinstance(config, ExternalConfig)
    assert config.command[-1] == "scripts.harness_candidate"
