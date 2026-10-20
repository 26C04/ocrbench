# T25: Synthetic dry run

Goal: prove that every local candidate (A1–A3, B, and C when its command is
ready) runs end to end on the Windows benchmark machine **before** any real
sheet is used. The sheets are generated in code with invented values, so the
results are not benchmark results and are not reported as such.

Keep the synthetic data and its runs in their own folders, never in the real
data folders and never in the repository.

## 1. Point ocrbench at synthetic folders (current terminal only)

```powershell
$env:OCRBENCH_DATA_DIR  = "D:\ocrbench-synthetic"
$env:OCRBENCH_FINAL_DIR = "D:\ocrbench-synthetic"
$env:OCRBENCH_RUNS_DIR  = "D:\ocrbench-synthetic-runs"
```

Open a new terminal afterwards to get back to the real folders.

## 2. Generate and import the sheets

```powershell
uv run python -m scripts.make_synthetic_sheets --out D:\ocrbench-synthetic --count 10
uv run ocrbench dataset import-answers --root D:\ocrbench-synthetic --answers D:\ocrbench-synthetic\answers.csv
uv run ocrbench dataset validate --root D:\ocrbench-synthetic --ground-truth edi_fields --min-per-split 4 --strict-counts
```

10 sheets give 4 dev and 6 final, half scans (clean PNG) and half photos
(tilted, blurred, uneven light, JPEG). The generator finds MS Gothic, Yu
Gothic or Meiryo automatically; pass `--font` otherwise. It refuses to write
inside the repository.

## 3. Check the engines

```powershell
uv run pytest -m ocr_engines --no-cov     # Tesseract with jpn data
ollama pull qwen3:8b                      # Layer 2 for A1–A4
ollama pull qwen2.5vl:7b                  # A3 transcription
ollama pull gemma4:12b                    # B
```

PaddleOCR (A2) is optional: install it as in `docs/windows-setup.md`. If it
does not install on Python 3.14, record that A2 could not run.

## 4. Run every local candidate on dev

The Layer 2 prompt `edi-fields` is already in the registry (`prompts/`).

```powershell
foreach ($p in "A1-tesseract", "A1L-tesseract-lines-removed", "A2-paddleocr", "A3-qwen-vl", "B-vision") {
    uv run ocrbench run --pipeline "pipelines\$p.json" --split dev --run-id "dry-$($p.ToLower())"
}
```

For C, write a config with `"kind": "external"` and the harness command; the
stdout contract is on `ExternalCommandPipeline` in `src/ocrbench/pipelines.py`.

## 5. Reports and one comparison

```powershell
Get-ChildItem D:\ocrbench-synthetic-runs -Directory | ForEach-Object {
    uv run ocrbench report --run-dir $_.FullName
}
uv run ocrbench compare-runs --run-dirs <A1 run dir> <A1L run dir>
```

Each `report.md` records the engine versions, Tesseract data hashes, Ollama
model digests, prompt hash, code commit, time p50/p95 and cost. These reports
are the T25 deliverable; keep them outside the repository. Because the sheets
are synthetic, their reports may be shared for review.

Optionally run the final split once to check the final-split locks
(`--split final --confirm-final` with `$env:OCRBENCH_ALLOW_FINAL = "1"`).

## What the cloud session already found (Tesseract 5.3.4, 10 synthetic sheets)

Layer 1 only: does each correct value appear in the OCR text?

| Setting | Scans (5 sheets) | Photos (5 sheets) |
|---|---|---|
| `psm 6` (A1 default) | 0 of 20 values | 0 of 20 values |
| `psm 4`, `remove_lines` (A1L) | 20 of 20 values | 3 of 20 values |

Tesseract skips text inside ruled table cells; erasing the ruling lines first
(`"remove_lines": true`) fixes this on scans. Photos still fail. If the real
sheets are ruled tables, A1 without line removal will score close to zero, so
run both A1 and A1L on dev and keep the better one as A1 before the final
split is frozen. These numbers come from invented sheets and say nothing about
accuracy on real sheets.
