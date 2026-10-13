# ADR-0002: Two-layer OCR benchmark for the 4 EDI fields

- Status: Proposed
- Date: 2026-10-13
- Revises: [ADR-0001](0001-local-ocr-benchmark-architecture.md) §1, §2, §4, §8, §9, and parts of §5 and §6
- Tag of the code before this ADR: `v0.1-vlm-benchmark`

## Context

ADR-0001 (July 2026) designed a benchmark in which an Ollama vision model reads a
production instruction sheet (生産指示書) and writes a full order document as JSON:
customer, order number, delivery date, and a list of items with part number,
material, and quantity. T01–T18 implemented that design and the quality gates pass.

Since then, three things changed.

1. **Prof. Sato's two-layer design.** The project now specifies two layers.
   Layer 1 is OCR (image → raw text) and does not interpret meaning. Layer 2 is a
   local LLM (raw text → structured fields). Layer 2 must always run locally. The
   Layer 1 candidates are Mistral OCR, PaddleOCR, Qwen-VL, and Tesseract.
2. **The goal is the EDI entry.** The first customer is Nidec Machine Tool on the
   hWAO NMTJ Edition EDI. The shipment-entry step needs only 4 values from the
   sheet: customer (selects the EDI system), part number (cross-checked with the
   EDI row), order number (the 手配番号 search key), and quantity (出荷数). The EDI
   sets 納品日 to the current date, so the delivery date on the sheet is not used.
3. **A third approach exists.** A teammate's template harness (report of
   2026-10-12) aligns the sheet to a printed-rule template, crops the needed boxes,
   reads them with a local vision model, normalizes them in code, and marks a page
   `needs_review` instead of guessing. Its reported numbers come from synthetic
   images and thresholds chosen while looking at evaluation data.

The data we expect is also smaller: Kimura has been asked for 30–50 real sheets,
not the 100 that ADR-0001 §4 planned.

A benchmark that only measures one vision model reading the whole page cannot
answer the questions the project now has: which Layer 1 engine is best, whether
the two-layer design beats a single vision model or the template harness, and
whether paid cloud OCR is worth its cost.

## Decision

### 1. `ocrbench` is a pipeline-agnostic evaluator

`ocrbench` scores any **pipeline** that turns one sheet image into the 4 EDI
fields. All candidates read the same sheets and are scored by the same code.

| ID | Pipeline | Runs |
|---|---|---|
| A1 | Tesseract → local LLM | local |
| A2 | PaddleOCR → local LLM | local |
| A3 | Qwen-VL in transcribe-only mode → local LLM | local |
| A4 | Mistral OCR → local LLM | cloud, only under §6 |
| B | Vision model, image straight to the fields (the ADR-0001 design) | local |
| C | Template harness, called as an external command | local |

- All A pipelines share one Layer 2 configuration: the same local Ollama model,
  prompt, and inference settings. Differences between A1–A4 therefore come from
  Layer 1.
- Layer 2 keeps the ADR-0001 §7 inference protocol: temperature 0, fixed seed,
  thinking off, Ollama `format` set to the JSON Schema, concurrency 1.
- Every pipeline also returns the raw Layer 1 text (when it has one), its
  timings, and its cost, so the evaluator can attribute errors (§5).
- B reuses the existing code. C is integrated through a small command-line
  contract (stdin/stdout JSON), so the two codebases do not need to be merged.

### 2. The scored record is the 4 EDI fields

```json
{
  "customer": "Nidecマシンツール株式会社",
  "part_no": "HR237101021",
  "order_no": "023235",
  "quantity": 1
}
```

- Every field may be `null` when it cannot be read. Pipelines must not guess.
- Unknown keys are rejected (`additionalProperties: false`). Text around the
  JSON, Markdown, and code fences are format errors, as in ADR-0001 §2.
- Delivery date and material are not scored. The existing `OrderDocument`
  schema stays in the code and is still used by pipeline B internally.
- Comparison stays exact after the normalization below. One wrong digit is a
  wrong EDI entry.

**Normalization.** Pipeline output must first pass the schema (for example,
`quantity` must already be an integer). The rules below are then applied to
both ground truth and pipeline output before comparison:

| Field | Rule |
|---|---|
| customer | Map to the EDI site identifier through a versioned table in the repo, e.g. `Nidecマシンツール株式会社` → `hwao_nmtj` (proposed, see Open decision D2). Unmapped names score as wrong. |
| part_no | Strip outer whitespace only. |
| order_no | Strip outer whitespace. Whether the `-2025` year suffix is removed is **Open decision D1**. The rule lives in one function so it can change in one place. |
| quantity | Integer ≥ 1. Remove thousands separators, full-width digits, and a trailing unit, including `PC`, `PCS`, `pc`, `pcs`, `個`, with or without a space (`1 PC`, `4PC`). |

### 3. Dataset: 30–50 sheets, two splits

| Split | Share | Directory | Purpose |
|---|---:|---|---|
| dev | ~40% | `OCRBENCH_DATA_DIR` | Tuning prompts, crops, and settings |
| final (test) | ~60% | `OCRBENCH_FINAL_DIR` | Frozen evaluation, run once per candidate |

- The split names `dev` and `final` match the existing code. The `selection`
  split is not used.
- Exact count checks become a configurable minimum instead of the fixed
  40/20/40 table. Scan and photo should both appear in each split when the data
  allows.
- The final split keeps every ADR-0001 §5 protection: separate Windows user,
  NTFS ACL, `OCRBENCH_ALLOW_FINAL=1` plus `--confirm-final`, and
  independent checking of its ground truth by a second person.
- **Freeze rule.** Before the final split is run, the code is tagged and every
  candidate's configuration is recorded. The final split is run once per
  candidate. A fix made after seeing final results requires a new tag and is
  reported as a second, non-primary result.

### 4. Metrics

Primary metric:

- **Field accuracy**: correct fields / (4 × sheets). A `null` counts as wrong.

Secondary metrics:

- Sheets fully correct (all 4 fields right), with the Wilson 95% interval.
- Per-field accuracy, with a sheet-level bootstrap interval (fixed seed).
- **Silent errors**: fields that are non-null and wrong. These are the errors
  that would reach the EDI. Reported separately from abstentions (`null` or
  `needs_review`), because an abstention costs a manual check while a silent
  error causes a wrong registration.
- Paired comparison between two pipelines on the same sheets (exact McNemar,
  already implemented).
- Time per sheet (warm, wall clock, image load to validated fields) with p50
  and p95. Requirement: faster than the ~5 minutes of manual work. Target:
  ADR-0001 §7's 30 seconds warm.
- Cost per sheet. Local pipelines record 0 (electricity is not counted).
  Mistral OCR uses the configured per-page price (about USD 4 per 1,000 pages
  in October 2026).

### 5. Which layer failed

For A pipelines, each wrong field is attributed to a layer without requiring a
full hand transcription of every sheet:

- **Layer 1 error**: the normalized correct value does not appear in the
  normalized Layer 1 text.
- **Layer 2 error**: the correct value appears in the Layer 1 text, but the
  final field is wrong.

The containment check normalizes Unicode (NFKC) and whitespace on both sides
before searching. A character error rate (CER) against full transcriptions may
be added later if the group writes them; it is not required by this ADR.

### 6. Cloud OCR is allowed only behind a lock

This revises ADR-0001 §5 for Layer 1 only. Layer 2 always runs locally.

- No sheet image is sent to a cloud service until Kimura has approved it in
  writing and the approval is recorded in `docs/` (date and scope, no personal
  data).
- The Mistral adapter refuses to run unless all three are present:
  `OCRBENCH_ALLOW_CLOUD=1`, an API key in the environment, and the CLI flag
  `--confirm-cloud`. The API key is never written to the repository, run
  outputs, or logs.
- Results from A4 are reported with their cost per sheet.
- Cloud agents still receive only anonymous aggregates, never images, ground
  truth, or misread values (ADR-0001 §5 unchanged).

### 7. Decision rule

Prefer a fully local pipeline. Choose Mistral OCR (A4) only if it is better on
field accuracy **and** the paired comparison against the best local pipeline
shows the difference is real, i.e. the lower end of its interval clears the
best local result by about 2 percentage points (see Open decision D5).

With 30–50 sheets, one wrong sheet moves sheet accuracy by 2–4 points, so a
2-point difference on its own is within measurement noise. When the evidence is
inconclusive, the local pipeline wins.

### 8. Scope changes

- The automatic prompt-improvement loop (ADR-0001 §8) and blind selection are
  not part of this workflow. Their code (`selection.py`, `ocrbench select`,
  `ocrbench loop`) and tests stay in the repository and keep passing. Removing
  them requires a separate decision.
- Dependencies: the 4 core runtime dependencies stay, and the OCR engines add
  none. Tesseract runs as an external program, Qwen-VL goes through the existing
  Ollama adapter, Mistral OCR uses the standard library's HTTP client, and
  PaddleOCR is installed separately on the benchmark machine and imported only
  when used. A default install and CI need none of them; CI uses fakes only.

## Open decisions

These are recorded with a recommendation. They are settled by the group and
this ADR is updated before it moves to Accepted.

| ID | Question | Recommendation | How to settle |
|---|---|---|---|
| D1 | Remove `-2025` from `order_no`? The project brief says remove it; the harness report keeps the year. | Use whatever the hWAO 手配番号 search accepts. | Look at one real 手配番号 in the hWAO shipment-entry screen. |
| D2 | Score `customer` as the mapped EDI site or as exact characters? | Mapped EDI site. | Group decision. |
| D3 | Score only the 4 EDI fields? | Yes; drop delivery date and material. | Group decision. |
| D4 | Split shares for 30–50 sheets? | dev ~40%, final ~60%. | Group decision once the sheet count is known. |
| D5 | How to apply the "Mistral must be ~2 points better" rule with few sheets? | Local wins unless Mistral is better and its interval clearly clears the best local result. | Group decision with Prof. Sato. |

## Relationship to ADR-0001

| ADR-0001 section | Effect of this ADR |
|---|---|
| §1 Scope | Revised: pipelines with an OCR layer are in scope, not only vision models. Printed single-page sheets remain the scope. |
| §2 Schema | Revised: the scored record is the 4 EDI fields (§2 above). |
| §3 Preprocessing | Unchanged. All pipelines receive the same preprocessed image. |
| §4 Dataset | Revised: 30–50 sheets, dev and final splits (§3 above). |
| §5 Confidentiality | Unchanged, except cloud OCR under the lock in §6 above. |
| §6 Models | Partly revised: candidates are pipelines (§1 above). The GPU and digest rules still apply to local models. |
| §7 Inference | Unchanged; applies to every local model. |
| §8 Prompt loop | Not used in this workflow; code kept (§8 above). |
| §9 Metrics | Revised: field accuracy is primary (§4 above). |
| §10 Reports | Unchanged layout; reports gain the new metrics. |
| §11 Quality | Unchanged. |

## Consequences

### Positive

- The benchmark answers the questions the project has now: best OCR engine,
  two layers versus one vision model versus the template harness, and whether
  cloud OCR is worth paying for.
- Layer attribution shows where to spend effort.
- The silent-error metric matches the EDI risk: a wrong registration is worse
  than a manual check.
- Existing code is reused: dataset handling, strict JSON parsing, statistics,
  reporting, split guards, CI.

### Negative

- With 30–50 sheets, intervals are wide (about ±6 points on field accuracy for
  ~24 final sheets). Small differences between candidates will not be
  distinguishable.
- More moving parts: optional OCR dependencies, an external-command contract,
  and a cloud adapter that must stay locked.
- Customer scoring depends on a mapping table that must be maintained when new
  customers are added.

## Implementation

Tasks T19–T25 in [the task list](../tasks/README.md) implement this ADR. Human
tasks H6–H8 cover the real-data run, the Mistral run, and settling D1.
