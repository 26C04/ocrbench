# Phase 1 Runbook: Setup to First Five-Model Ranking

This runbook covers ADR-0001 Implementation Order steps 2–4: the Windows
smoke test, the Development dataset, and the first evaluation of all five
approved models with the shared `base` prompt (ADR §8-1, §8-2). It builds on
[windows-setup.md](windows-setup.md); follow that guide for anything not
repeated here.

## Before the session (downloads take the longest)

1. Install Git, uv and Ollama, then pull the five approved models
   ([windows-setup.md](windows-setup.md) §2–4). Leave the PC on while they
   download.
2. Clone the repository and run `uv sync` and `uv run pytest`.
3. Create `D:\OCRBench\data\raw`, `D:\OCRBench\runs` and
   `D:\OCRBench\samples`, then `setx OCRBENCH_DATA_DIR` and
   `setx OCRBENCH_RUNS_DIR` (§5).

## Digitize the Development documents

Phase 1 needs the 40 Development documents: 20 scans and 20 photos of
different originals. Selection (10 + 10) can be prepared at the same time.
Final documents belong in the separate isolated Final folder and are not
handled here.

- Scans: PDF, PNG or JPG from the scanner.
- Photos: JPG or PNG. On iPhone set **Settings → Camera → Formats → Most
  Compatible**, because HEIC files are rejected.
- Copy every file into `D:\OCRBench\data\raw`. File names do not matter;
  `build-manifest` renames them to anonymous IDs. Avoid commas in names.

## Enter ground truth

Open `scripts\gt_entry.html` in Microsoft Edge or Google Chrome by
double-clicking it. It works offline and writes only to the folder you choose.

1. Click **Open dataset folder** and choose `D:\OCRBench\data`.
2. For each document, choose scan/photo and the split, then type exactly what
   is printed. Leave a value empty only if it is missing or unreadable.
3. **Ctrl+S** saves `gt\<doc-id>.json`, refreshes `assignments.csv` and moves
   to the next unfinished document. The header shows progress against the
   20 + 20 Development target.

## Freeze the dataset

```powershell
uv run ocrbench dataset build-manifest --root D:\OCRBench\data --assignments D:\OCRBench\data\assignments.csv
uv run ocrbench dataset validate --root D:\OCRBench\data
uv run ocrbench dataset fingerprint --root D:\OCRBench\data
```

`validate` warns while Selection or Final are still incomplete; that does not
block a Development run. `--strict-counts` is required only once all 100
documents exist. If you add documents later, reopen the entry form and save
once so `assignments.csv` lists the renamed files, then rebuild.

## Run phase 1

Put one anonymous sample image (no customer data) in `D:\OCRBench\samples`,
then from the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\phase1.ps1 -SampleImage D:\OCRBench\samples\anonymous.png
```

For each model, the script smoke-tests it, runs it on the Development split,
writes `report.md` and unloads it before the next one. It then writes
`phase1-ranking-<timestamp>.md` to `OCRBENCH_RUNS_DIR` and saves a full
transcript next to it. Expect 1–2 hours of unattended run time.

The ranking excludes models whose warm p50 exceeds 30 seconds, and orders the
rest with the `pick-best` key: exact matches, JSON-invalid count, the
`order_no`/`part_no`/`num_pieces` accuracy sum, output tokens, then latency.
Runs that were not 100% GPU are marked `reference` and ranked after every
primary run (ADR §6). The ranking can be regenerated at any time:

```powershell
uv run python -m scripts.rank_models --run-dirs <run-dir> <run-dir> ...
```

## Record

Keep the transcript, the ranking file, the Ollama version, each model's
digest, the dataset fingerprint and the GPU model with the benchmark notes.
A ranking produced on hardware other than the ADR-0001 reference machine is a
preliminary result until it is reproduced there.
