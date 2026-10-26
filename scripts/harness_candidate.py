"""Run the teammate's template harness as benchmark candidate C (ADR-0002 §1).

``pipelines/C-harness.json`` calls this module through ``ExternalCommandPipeline``,
which appends the path of one preprocessed sheet image. The module runs
``harness ocr read`` from the automation-harness checkout, converts its report
to the ocrbench stdout contract, and prints exactly one JSON object. Nothing
else is printed: the harness's own output may contain sheet values.

The harness checkout defaults to ``automation-harness`` beside this repository
and can be set with ``OCRBENCH_HARNESS_DIR``. It needs its own ``uv sync``.

Customer is scored by the EDI site the harness chose (``--customer site``, the
default), which is how the harness is designed and what open decision D2
recommends. ``--customer printed`` passes the verbatim transcription instead,
so it can be scored by the same exact-name rule as the other candidates.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

CustomerMode = Literal["site", "printed"]

SITE_CUSTOMER_NAMES: dict[str, str] = {"hwao_nmtj": "Nidecマシンツール株式会社"}
"""Canonical printed name per harness site id, as in ocrbench's customer table."""

_REVIEW_EXIT_CODE = 10
_VIEW_ORDER = ("header", "verify", "row")


def _default_harness_dir() -> Path:
    configured = os.environ.get("OCRBENCH_HARNESS_DIR")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[2] / "automation-harness"


def _printed_customer(slip: dict[str, Any]) -> str | None:
    field = slip.get("fields", {}).get("customer", {})
    reads: dict[str, str] = field.get("reads", {})
    agreement: str = field.get("agreement", "")
    agreed = [view for view in _VIEW_ORDER if view in agreement]
    for view in (*agreed, *_VIEW_ORDER):
        value = reads.get(view, "").strip()
        if value:
            return value
    return None


def convert(report: dict[str, Any], customer_mode: CustomerMode = "site") -> dict[str, Any]:
    """Map one harness report for a single image onto the ocrbench output contract."""
    slips: list[dict[str, Any]] = report.get("slips", [])
    if len(slips) != 1:
        return {"status": "needs_review", "fields": None, "cost_usd": 0.0}
    slip = slips[0]
    edi: dict[str, Any] = slip.get("edi", {})
    if customer_mode == "site":
        customer = SITE_CUSTOMER_NAMES.get(edi.get("site_id") or "")
    else:
        customer = _printed_customer(slip)
    quantity = edi.get("quantity")
    fields = {
        "customer": customer,
        "part_no": edi.get("part_no"),
        "order_no": edi.get("order_no_full"),
        "quantity": quantity if isinstance(quantity, int) and quantity >= 1 else None,
    }
    status = "ok" if slip.get("status") == "ok" else "needs_review"
    return {"status": status, "fields": fields, "cost_usd": 0.0}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image", type=Path, help="one sheet image (appended by ocrbench)")
    parser.add_argument("--harness", type=Path, default=None, help="automation-harness checkout")
    parser.add_argument("--customer", choices=("site", "printed"), default="site")
    parser.add_argument("--model", default=None, help="vision model (harness default otherwise)")
    arguments = parser.parse_args(argv)
    harness = arguments.harness or _default_harness_dir()
    if not (harness / "pyproject.toml").is_file():
        print("error: automation-harness checkout not found", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(prefix="ocrbench-c-") as directory:
        out = Path(directory) / "report.json"
        command = ["uv", "run", "--project", str(harness), "harness", "ocr", "read"]
        command += [
            str(arguments.image.resolve()),
            "--out",
            str(out),
            "--no-secondary",
            "--no-crops",
        ]
        if arguments.model:
            command += ["--model", arguments.model]
        completed = subprocess.run(command, cwd=harness, capture_output=True, check=False)
        if completed.returncode not in (0, _REVIEW_EXIT_CODE) or not out.is_file():
            print(f"error: harness exited with code {completed.returncode}", file=sys.stderr)
            return 1
        report = json.loads(out.read_text(encoding="utf-8"))

    print(json.dumps(convert(report, arguments.customer), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
