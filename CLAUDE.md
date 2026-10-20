# Capstone 1 · Group 26C04 · KUAS

AI Automation of the Packaging & EDI Workflow
**Client:** Kimura Industry Co., Ltd.
**Member:** ENKHBAYAR Anu-Ujin (2023m610@kuas.ac.jp) — AI integration lead, OCR/AI benchmark

> **Source of truth for the benchmark design:** [ADR-0002](adr/0002-two-layer-edi-field-benchmark.md)
> (Proposed), which revises parts of [ADR-0001](adr/0001-local-ocr-benchmark-architecture.md).
> Where this file and an ADR disagree, the ADR wins. Open decisions D1–D5 are listed in ADR-0002.

---

## Project goal

Automate the manual workflow: worker reads a production instruction sheet (生産指示書), logs into a customer's EDI website, and enters shipment data by hand. Our system replaces this with:

- **Phase 1, Issue 1 (my work):** OCR + LLM reads the sheet → outputs 4 structured fields
- **Phase 1, Issue 2:** Playwright browser agent logs into EDI and enters the data
- **Phase 2:** Computer vision (SAM3) identifies physical parts from camera images

---

## My component: OCR/AI benchmark (Issue 1)

**Goal:** Find the most accurate, affordable way to extract 4 fields from Kimura's instruction sheets.

### The 2-layer system (Prof. Sato's design)

```
Sheet image → [Layer 1: OCR] → raw text → [Layer 2: local LLM] → 4 fields
```

- **Layer 1 (OCR):** Image → text. Does not interpret meaning. Candidates: Mistral OCR, PaddleOCR, Qwen-VL, Tesseract
- **Layer 2 (LLM):** Text → structured JSON. Always runs locally (free, data stays in-house).

### 4 fields to extract (for Nidec Machine Tool customer)

| Field | Japanese | Example | Notes |
|-------|----------|---------|-------|
| Customer | 受注先 | Nidecマシンツール株式会社 | Determines which EDI system |
| Part number | 部番 | HR237101021 | Cross-check with EDI row |
| Order number | 受注No | 023235-2025 | Keep the "-2025" ending (owner's call, 16 Oct; ADR-0002 D1) |
| Quantity | 個数 | 1 | Strip "PC" unit |

### Benchmark scoring metrics
1. Field accuracy — % of 4 fields exactly right (main score)
2. Sheets fully correct — % of sheets with all 4 fields right
3. Per-field accuracy — which field is hardest
4. OCR character errors — which layer failed
5. Time per sheet — must beat ~5 min manual
6. Cost per sheet — Mistral is paid (~$4/1000 pages); local = free

### Decision rule
Prefer fully local unless Mistral is clearly better (>~2 percentage points).

---

## EDI workflow (Issue 2 — teammate's work, but I need to know it)

Customer: **Nidec Machine Tool**, system: **hWAO NMTJ Edition**

Steps the script automates:
1. Log in with company code, user ID, password
2. Menu: truck icon → 出荷入力
3. Click データ取得 → filter 手配番号 by order number
4. Fill: tick 入力, set 納品日 = today, set 出荷数 = quantity
5. Click 登録 → OK → OK
6. Menu: truck icon → 納品書印刷 → tick 印刷 → print PDF

**Important:** Use Playwright (not AI agent) for EDI — more reliable on a fixed website.

---

## Tech stack

- **OCR candidates:** Mistral OCR API, PaddleOCR, Qwen-VL, Tesseract
- **Layer 2 LLM:** Local model via Ollama (Qwen or similar)
- **EDI automation:** Playwright (Python)
- **Phase 2:** SAM3 object detection
- **Repo:** GitHub Organization (team-shared; do not commit credentials or instruction sheet images)

---

## Key constraints

- **Never commit** to the shared repo: EDI credentials, instruction sheet images (possible cloud approval pending from Kimura), personal API keys
- **Claude must never open, read, or display** real sheet images, ground-truth files, model outputs, or run results (anything under `OCRBENCH_DATA_DIR`, `OCRBENCH_FINAL_DIR`, `OCRBENCH_RUNS_DIR`), and must never handle EDI credentials. Write and run code against them; report only anonymous totals
- Mistral OCR requires Kimura's approval before sending sheet images to the cloud
- Layer 2 (LLM) must always run locally regardless of Layer 1 choice
- Score on exact matches only — one wrong digit = wrong EDI entry

---

## Current status (as of 20 Oct 2026)

- Steps 0–4 and Step 5 part 1 (answers CSV import) done in PR #23 (CI green, 577 tests, no reviews yet)
- **Next: Step 5 part 2, the pipeline run command** (planned 16 Oct, not started), then the report and the T25 dry run
- **Kimura has sent the real instruction sheets.** They stay in local folders outside the repo and OneDrive
- ADR-0002 D1: the owner decided to **keep** the "-2025" ending (current code, `ORDER_NO_RULE = KEEP`)
- Still waiting on Kimura: cloud approval for Mistral OCR (keep it out of reports for the professor)
- Dates: slides v1 21 Oct, engineers 28 Oct, midterm 30 Oct
- Full status, next steps and report preferences: [HANDOVER.md](HANDOVER.md) (start at §0)

---

## Glossary

- **生産指示書** — production instruction sheet (the paper we read)
- **出荷入力** — shipment entry screen in the EDI
- **納品書** — delivery note (printed at end)
- **手配番号** — order/arrangement number (search key in EDI)
- **受注No** — order number on the sheet (same as 手配番号; whether to strip "-2025" is ADR-0002 D1)
- **Ollama** — local LLM runner
- **SAM3** — Segment Anything Model 3, used for object detection in Phase 2

---

## Working in this repo

- Setup and checks (all four must pass before every push; CI runs the same):
  `uv sync`, `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy`, `uv run pytest`
- Tests needing a real Ollama server: `uv run pytest -m ollama --no-cov` (never in CI).
- Code: `src/ocrbench/`. Task specs: `tasks/` (T01–T18 done; T19–T25 implement ADR-0002).
- The code before ADR-0002 is tagged `v0.1-vlm-benchmark`. ADR-0002 adds alongside existing
  code; do not delete existing features without a separate decision.
- Real sheets, ground truth, predictions, and run outputs live outside the repo
  (`OCRBENCH_DATA_DIR`, `OCRBENCH_FINAL_DIR`, `OCRBENCH_RUNS_DIR`). Only synthetic fixtures
  under `tests/fixtures/public/` may be committed.
