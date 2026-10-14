# Handover: OCR benchmark (Issue 1)

- Owner: ENKHBAYAR Anu-Ujin, AI integration lead, Group 26C04
- Date: 14 October 2026
- Branch: `claude/serene-brahmagupta-pjokpe`, pull request
  [26C04/ocrbench#23](https://github.com/26C04/ocrbench/pull/23) into `develop` (open, CI green)

This file lets a teammate or a new Claude session continue without the chat
history. Read [CLAUDE.md](CLAUDE.md) and [ADR-0002](adr/0002-two-layer-edi-field-benchmark.md)
next; where this file and an ADR disagree, the ADR wins.

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
  is recorded in `docs/` (date and scope only).

## 2. Where things stand

| Step | Task | What it delivers | Status |
|---|---|---|---|
| 0 | T19 | ADR-0002 (two-layer design, 4 EDI fields), CLAUDE.md, task specs T19–T25 | Done, `99b1e79` |
| 1 | T20 | `edi_fields.py`, `customers.py`: strict 4-field format, customer → EDI site table, "-YYYY" switch, "1 PC" fix | Done, `268b680` |
| 2 | T21 | `pipelines.py`: one interface for all candidates (OCR+LLM, vision model, external program) | Done, `4e763e2` |
| 3 | T22 | `ocr_engines.py`: Tesseract, PaddleOCR, Qwen-VL, Mistral (3 locks) + code-review fixes | Done, `d1f6278`, `1fcb35a` |
| 4 | T23 | `field_metrics.py`: field accuracy, silent errors, layer attribution, paired comparison, cloud rule | Done, `b125206` |
| 5 | T24–T25 | Runner, CLI, reports for pipelines; ground truth from CSV; synthetic dry run | **Next** |
| 6 | H6 | Real benchmark on Kimura's sheets | Waiting on step 5 |
| 7 | H7 | Mistral comparison | Waiting on Kimura's cloud approval |

Quality gates on the branch head: ruff, ruff format, strict mypy, 562 tests
passing (90% coverage), CI green. Nothing from ADR-0001 was deleted; the July
code is preserved and is candidate B. The tag `v0.1-vlm-benchmark` (on
`3912b7c`) was created locally but **not pushed**; push it when convenient.

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

Open:

| ID | Question | How to settle | Current setting |
|---|---|---|---|
| D1 | Does hWAO's 手配番号 include the year ending ("-2025", "-2026")? | Log in to the **test EDI website**, open 出荷入力 → データ取得, and search one order number with and without the ending | `ORDER_NO_RULE = KEEP` in `src/ocrbench/edi_fields.py`; the strip rule removes any `-` plus 4 digits, so 2026 sheets work either way |
| D2–D5 | Customer scoring, fields, split shares, cloud rule | Group agreement in PR #23 | Recommendations as above; split 40% dev / 60% final |

The year ending is probably the order year. If order numbers restart each
year, keeping it may be required to tell orders apart. Include sheets from both
years in the dataset.

## 4. Next: Step 5 (T24–T25)

What to build, in order:

1. **Ground truth from one CSV** per dataset root, instead of one JSON per
   sheet. Columns:

   ```csv
   file,source_kind,split,customer,part_no,order_no,quantity
   ```

   `source_kind` is `scan` or `photo`; `split` is `dev` or `final`. Type
   values exactly as printed (order number with its year ending, quantity as
   printed, e.g. `1 PC`). Validate every row with `ground_truth_problems()`.
   Final-split answers must be checked by a second person.

   **Done.** Put the sheets in `<root>/raw/`, then run:

   ```powershell
   uv run ocrbench dataset import-answers --root D:\ocrbench-data --answers D:\answers.csv
   uv run ocrbench dataset validate --root D:\ocrbench-data --ground-truth edi_fields --min-per-split 10 --strict-counts
   uv run ocrbench dataset fingerprint --root D:\ocrbench-data --ground-truth edi_fields
   ```

   Nothing is renamed or written unless every row is valid. Errors name the
   row and column but never print the values. Keep `answers.csv` outside the
   repo, next to the data.
2. **Dataset:** reuse `dataset.build_manifest` (anonymous file names, SHA-256,
   fingerprint). Replace the fixed 40/20/40 strict count check with a minimum
   per split for `dev` and `final`. **Done** (`--min-per-split`; a split with
   no scans or no photos is a warning only).
3. **Runner:** run any `Pipeline` over a split, score with `score_sheet`,
   write `results.json` and the run manifest (pipeline `describe()` output,
   versions, digests, dataset fingerprint).
4. **CLI:** `ocrbench run --pipeline <config>` (plus `--confirm-cloud` for A4,
   `--confirm-final` for the final split); `ocrbench compare-runs` prints
   `compare()` and `decide_cloud()`.
5. **Reports:** `report.md` with the ADR-0002 metrics; detailed per-sheet
   output stays local.
6. **Synthetic dry run (T25)** of every local candidate before touching real
   sheets.

Specs: `tasks/T24-runner-cli-reports.md`, `tasks/T25-synthetic-dry-run.md`.

## 5. Things to verify on the Windows benchmark machine

- Tesseract 5 with Japanese data (`docs/windows-setup.md` §11); run
  `uv run pytest -m ocr_engines --no-cov`.
- PaddleOCR: may not support Python 3.14 yet. If it doesn't, record that A2
  could not run. Also confirm it accepts BGR arrays, as assumed in
  `PaddleOcrEngine`.
- Ollama models: the Layer 2 LLM (for example `qwen3:8b`) and `qwen2.5vl:7b`
  for A3. Record digests.

## 6. Working with the repository

- Checks before every push: `uv run ruff check .`, `uv run ruff format --check .`,
  `uv run mypy`, `uv run pytest`.
- GitHub access: the Claude GitHub App is installed only on the personal
  account, not on the **26C04** organization, so cloud Claude sessions cannot
  push. Each step was pushed with GitHub Desktop from a zip of changed files.
  An org owner approving the app removes this step.

## 7. Documents made so far

| What | Where |
|---|---|
| Design record | `adr/0002-two-layer-edi-field-benchmark.md` |
| Benchmark plan (page) | https://claude.ai/artifact/XckDaH52JE83U5nTmLumoD |
| ADR-0002 explained (page) | https://claude.ai/artifact/EPryi8fRJAsWV2B7Qdgmp7 |
| Session log (page) | https://claude.ai/artifact/KEDALi2ZX55rfm5qcek35b |
| Steps 3–4 update (page) | https://claude.ai/artifact/GUuSf9m9eX7C2HxBx1MnoF |
| First progress deck (15 slides) | https://claude.ai/artifact/M1UBzhHE7wY9VDUVbBsz4m |
| Engineers' progress report (12 slides) | https://claude.ai/artifact/Q9orKactnXoZAWbM7jbnQ6 |

The pages are private to the owner until shared from each page's Share menu.
