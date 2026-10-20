# Handover: OCR benchmark (Issue 1)

- Owner: ENKHBAYAR Anu-Ujin, AI integration lead, Group 26C04
- Date: 20 October 2026
- Branch: `claude/serene-brahmagupta-pjokpe`, pull request
  [26C04/ocrbench#23](https://github.com/26C04/ocrbench/pull/23) into `develop`
  (open, mergeable, CI green on `282553e`, no reviews yet)

This file lets a teammate or a new Claude session continue without the chat
history. Read [CLAUDE.md](CLAUDE.md) and [ADR-0002](adr/0002-two-layer-edi-field-benchmark.md)
next; where this file and an ADR disagree, the ADR wins.

## 0. Next session: start here

1. Read §1 (data rules) and §4 (what to build).
2. Build **Step 5 part 2: the pipeline run command** (§4, item 3–4). It was
   planned for 16 Oct and is **not started**, so the first real run (planned
   20 Oct) is waiting on it.
3. Then part 3 (report) and part 4 (synthetic dry run, T25).
4. Push each part to this branch (PR 23). Never merge or approve the PR.

Dates that matter: slides v1 **21 Oct**, slides v2 26 Oct, rehearsal 27 Oct,
presentation to Kimura's engineers **28 Oct**, midterm **30 Oct**.

## 1. Read this first: real data rules

Kimura has sent the real instruction sheets, and a test EDI website has been
provided. From now on:

- **Real sheets, correct answers, model outputs and run results never enter
  the repository, a chat with a cloud AI (including Claude), or OneDrive.**
  Keep them in a plain local folder outside the repo, for example
  `D:\ocrbench-data` (dev) and `D:\ocrbench-final` (final test), pointed to by
  `OCRBENCH_DATA_DIR`, `OCRBENCH_FINAL_DIR` and `OCRBENCH_RUNS_DIR`.
- **Claude Code must not open files in those folders**, even in a local
  session: anything Claude reads is sent to the model. Ask Claude to write and
  run code; look at results yourself; share only anonymous totals with it.
- **Test EDI website credentials** go in a password manager or the current
  terminal's environment variables only. Never in a file in the repo, a commit
  message, a screenshot, or a chat.
- **No sheet image goes to Mistral (cloud)** until Kimura's written approval
  is recorded in `docs/` (date and scope only). The code enforces this with
  three locks (`OCRBENCH_ALLOW_CLOUD=1`, `MISTRAL_API_KEY`, `--confirm-cloud`).

## 2. Where things stand

| Step | Task | What it delivers | Status |
|---|---|---|---|
| 0 | T19 | ADR-0002 (two-layer design, 4 EDI fields), CLAUDE.md, task specs T19–T25 | Done, `99b1e79` |
| 1 | T20 | `edi_fields.py`, `customers.py`: strict 4-field format, customer → EDI site table, "-YYYY" switch, "1 PC" fix | Done, `268b680` |
| 2 | T21 | `pipelines.py`: one interface for all candidates (OCR+LLM, vision model, external program) | Done, `4e763e2` |
| 3 | T22 | `ocr_engines.py`: Tesseract, PaddleOCR, Qwen-VL, Mistral (3 locks) + code-review fixes | Done, `d1f6278`, `1fcb35a` |
| 4 | T23 | `field_metrics.py`: field accuracy, silent errors, layer attribution, paired comparison, cloud rule | Done, `b125206` |
| — | — | Merge of `develop` (teammate's PR 24, `tasks/README.md` owner table) | Done, `7f475db` |
| 5.1 | T24 | `edi_answers.py`: answers CSV import, EDI ground truth, `--min-per-split` | Done, `282553e` |
| 5.2 | T24 | Pipeline runner + `ocrbench run --pipeline` + `compare-runs` | **Next, not started** |
| 5.3 | T24 | `report.md` with ADR-0002 metrics | To do |
| 5.4 | T25 | Synthetic dry run of every local candidate | To do |
| 6 | H6 | Real benchmark on Kimura's sheets | Waiting on step 5 |
| 7 | H7 | Mistral comparison | Waiting on Kimura's cloud approval |

Quality gates on the branch head: ruff, ruff format, strict mypy, **577 tests**
passing (~90% coverage), CI green. Tests by step: 442 (July) → 480 → 505 →
531 → 540 → 562 → 577. Nothing from ADR-0001 was deleted; the July code is
candidate B. The tag `v0.1-vlm-benchmark` (on `3912b7c`) exists only locally
in the first session; create it from GitHub Releases if needed (tag pushes from
cloud sessions are refused by the proxy).

## 3. Decisions and open questions

Decided (ADR-0002, status Proposed until the group confirms D2–D5):

- Two layers: OCR reads text, a **local** LLM extracts the 4 fields.
- Score only customer, part number, order number, quantity. Exact match after
  normalization. Unreadable means `null`, never a guess.
- Customer is scored by the EDI site it maps to (`hwao_nmtj`), so full-width
  and half-width text count the same.
- Main score: field accuracy. Also silent errors (wrong values on sheets
  marked ok), which layer failed, time and cost per sheet.
- Local by default; Mistral only if the low end of its paired 95% interval
  beats the best local pipeline by 2 points (`CLOUD_MARGIN`).
- **D1 (owner's call, 16 Oct): keep the year ending** ("-2025"). This is the
  current code (`ORDER_NO_RULE = KEEP` in `src/ocrbench/edi_fields.py`); no
  change needed. How "-2026" sheets behave will be seen from the sheets
  themselves. Flipping to `STRIP_YEAR` stays a one-constant change.

Still open (owner handles these; do not put them in reports for the
professor):

- Sheet count and scan/photo mix: the owner will check. It sets the dev/final
  split (planned 40% / 60%) and the `--min-per-split` value.
- Roles: `develop`'s `tasks/README.md` (PR 24) lists Anu-Ujin as "発表資料作成"
  (slides) and Fuke as code/model selection. The owner's teammate said the OCR
  part is the owner's job. Not yet confirmed with the group leader.
- PR 23 has no reviewer yet. Nothing merges without explicit approval.

## 4. Step 5 remaining work (T24–T25)

Done in part 1 (`282553e`): answers CSV → anonymized dataset + EDI ground truth.

```powershell
uv run ocrbench dataset import-answers --root D:\ocrbench-data --answers D:\answers.csv
uv run ocrbench dataset validate --root D:\ocrbench-data --ground-truth edi_fields --min-per-split 10 --strict-counts
uv run ocrbench dataset fingerprint --root D:\ocrbench-data --ground-truth edi_fields
```

CSV columns `file,source_kind,split,customer,part_no,order_no,quantity`,
values typed as printed. Nothing is renamed or written unless every row is
valid; errors name rows and columns, never values.

Still to build, in order:

3. **Runner.** Run any `Pipeline` (`pipelines.py`) over one split of an
   `edi_fields` dataset (`load_manifest(root, ground_truth="edi_fields")`),
   score each sheet with `field_metrics.score_sheet`, and write
   `results.json` plus a run manifest: pipeline `describe()` output, engine
   and model versions/digests, dataset fingerprint, code commit. Suggested:
   add new modules (`pipeline_runner.py`, `pipeline_results.py`) beside the
   existing OrderDocument-based `runner.py` / `results.py`, which stay
   unchanged (ADR-0002: add, do not delete). Reuse `splitguard` for the
   final-split protection.
4. **CLI.** `ocrbench run --pipeline <config.json>` (pipeline ID, engine,
   model, prompt, or external command), `--confirm-cloud` for A4,
   `--confirm-final` for the final split. `ocrbench compare-runs` prints
   `field_metrics.compare()` and `decide_cloud()`. Existing commands keep
   working unchanged (their CLI tests must still pass).
5. **Report.** `report.md` and the anonymous summary with the ADR-0002
   metrics (field accuracy, sheets fully correct with Wilson interval,
   per-field accuracy, silent errors, layer attribution, time p50/p95, cost).
   Per-sheet detail stays local.
6. **Synthetic dry run (T25).** Every local candidate on fake sheets under
   `tests/fixtures/public/` before touching real sheets.

Acceptance (T24 spec): a fake end-to-end test where two fake pipelines on a
synthetic dataset produce a report with all ADR-0002 metrics. Specs:
`tasks/T24-runner-cli-reports.md`, `tasks/T25-synthetic-dry-run.md`.

## 5. Things to verify on the Windows benchmark machine

- Tesseract 5 with Japanese data (`docs/windows-setup.md` §11); run
  `uv run pytest -m ocr_engines --no-cov`.
- PaddleOCR: may not support Python 3.14 yet. If it doesn't, record that A2
  could not run. Also confirm it accepts BGR arrays, as assumed in
  `PaddleOcrEngine`.
- Ollama models: the Layer 2 LLM (for example `qwen3:8b`) and `qwen2.5vl:7b`
  for A3. Record digests.

## 6. Working with the repository

- Checks before every push: `uv sync`, `uv run ruff check .`,
  `uv run ruff format --check .`, `uv run mypy`, `uv run pytest`.
- GitHub: the Claude GitHub App is now installed on the **26C04** org (the
  owner is an org admin), so cloud sessions push directly to this branch.
  Pushing tags or other branches is refused by the session proxy.
- Only this branch is pushed to. Never merge, approve, or push to `develop`
  or `main`.
- When `develop` moves, merge it into this branch (no rebase or force-push)
  and keep both sides of any conflict.

## 7. How the owner likes reports and pages

- Visual pages (charts, diagrams, timelines) over long text; plain, simple
  explanations, but engineers' material must say what was achieved and chosen.
- Put "ENKHBAYAR Anu-Ujin · AI integration lead, OCR/AI benchmark" on every
  page and deck; leave the email off.
- Reports for the professor: no open-questions section, no "cloud approval
  pending", no sheet counts.
- Numbers on pages come from the repo (test counts, line counts, commit
  dates), never invented, and never from real sheet data.

## 8. Documents made so far

| What | Where |
|---|---|
| Design record | `adr/0002-two-layer-edi-field-benchmark.md` |
| Weekly report, week 42 (visual page, for the professor) | https://claude.ai/artifact/688Y1ic4vT8254MKuiu9fe |
| Weekly report, 13–20 Oct (editable doc) | https://claude.ai/code/artifact/06f76399-45bb-4051-847d-e2222f13773b |
| Project plan and timeline (page) | https://claude.ai/artifact/Y6znNNBzKUJcNtSewUW81g |
| Engineers' progress report (12 slides) | https://claude.ai/artifact/Q9orKactnXoZAWbM7jbnQ6 |
| Steps 3–4 update (page) | https://claude.ai/artifact/GUuSf9m9eX7C2HxBx1MnoF |
| First progress deck (15 slides) | https://claude.ai/artifact/M1UBzhHE7wY9VDUVbBsz4m |
| ADR-0002 explained (page) | https://claude.ai/artifact/EPryi8fRJAsWV2B7Qdgmp7 |
| Benchmark plan (page) | https://claude.ai/artifact/XckDaH52JE83U5nTmLumoD |
| Session log (page) | https://claude.ai/artifact/KEDALi2ZX55rfm5qcek35b |

The pages are private to the owner until shared from each page's Share menu.
Earlier files (weekly report PDF of 13 Oct, progress .pptx) were made in the
first session's scratch folder and are not in the repo; the owner has the
downloaded copies.
