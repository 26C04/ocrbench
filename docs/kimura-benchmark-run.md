# Running the benchmark on Kimura's sheets (Windows benchmark PC)

This is the H6 run: every local candidate on scans and phone photos of
Kimura's sample sheets. The answer key comes from Kimura's PDF, so nobody has
to type the answers.

Rules that still apply:

- Sheets, the answer key, and run results stay in the local data folders
  (`D:\ocrbench-data`, `D:\ocrbench-runs`). Never put them in the repository
  or OneDrive, and never open them with Claude: share only `leaderboard.md`
  and the aggregate `report.md` files.
- Tune prompts and settings on **dev** only. Run **final** once per candidate,
  after the configs are frozen (ADR-0002 §3).

## 1. Scan and photograph the printed sheets

Print the PDF. Then capture every page in page order:

- Scan at 300 dpi (JPEG or PNG) into `D:\incoming\scans`.
- Photograph each page with a phone, the way a worker would, into
  `D:\incoming\photos`.

Name the files by page number, so the answer key can be matched to them:

```powershell
$i = 1; Get-ChildItem D:\incoming\scans\*.jpg | Sort-Object Name | ForEach-Object {
    Rename-Item $_.FullName ("scan-{0:D3}.jpg" -f $i); $i++ }
$i = 1; Get-ChildItem D:\incoming\photos\*.jpg | Sort-Object Name | ForEach-Object {
    Rename-Item $_.FullName ("photo-{0:D3}.jpg" -f $i); $i++ }
```

Check that each folder now holds one file per page, in page order.

## 2. Build the answer key from the PDF

```powershell
$pdf = "D:\incoming\sheets.pdf"
uv run ocrbench dataset answers-from-pdf --pdf $pdf --out D:\incoming\scans.csv  --file-pattern "scan-{page:03d}.jpg"  --source-kind scan
uv run ocrbench dataset answers-from-pdf --pdf $pdf --out D:\incoming\photos.csv --file-pattern "photo-{page:03d}.jpg" --source-kind photo
Get-Content -Encoding utf8 D:\incoming\scans.csv | Set-Content -Encoding utf8 D:\incoming\answers.csv
Get-Content -Encoding utf8 D:\incoming\photos.csv | Select-Object -Skip 1 | Add-Content -Encoding utf8 D:\incoming\answers.csv
```

- `-Encoding utf8` on every `Get-Content` matters: without it Windows
  PowerShell 5.1 reads the file in the Japanese code page and garbles the
  customer name, and `import-answers` then rejects every row.
- About 60% of pages go to **final**, chosen by a fixed seed. The scan and
  the photo of the same page always land in the same split.
- If a command lists pages with blank fields, type those values from the
  printed sheet. `import-answers` refuses the file until every field is
  filled.
- A second person checks the final rows against the printed sheets (ADR-0002 §3).

## 3. Import and check the dataset

```powershell
$env:OCRBENCH_DATA_DIR  = "D:\ocrbench-data"
$env:OCRBENCH_FINAL_DIR = "D:\ocrbench-data"
$env:OCRBENCH_RUNS_DIR  = "D:\ocrbench-runs"
Copy-Item D:\incoming\scans\*.jpg, D:\incoming\photos\*.jpg D:\ocrbench-data\raw\
uv run ocrbench dataset import-answers --root D:\ocrbench-data --answers D:\incoming\answers.csv
uv run ocrbench dataset validate --root D:\ocrbench-data --ground-truth edi_fields --min-per-split 10
uv run ocrbench dataset fingerprint --root D:\ocrbench-data --ground-truth edi_fields
```

(Before the final run, move the final split to its own ACL-protected folder
as in `docs/isolation-windows.md` and point `OCRBENCH_FINAL_DIR` there.)

## 4. Check every candidate, then run them all on dev

```powershell
uv run ocrbench doctor --pipelines pipelines\A1L-tesseract-lines-removed.json pipelines\A2-paddleocr.json pipelines\A3-qwen-vl.json pipelines\B-vision.json pipelines\A1R-tesseract-rules.json
uv run ocrbench run-all --split dev --run-prefix dev1 --leaderboard-out D:\ocrbench-runs\leaderboard-dev1.md `
    --pipelines pipelines\A1L-tesseract-lines-removed.json pipelines\A2-paddleocr.json pipelines\A3-qwen-vl.json pipelines\B-vision.json pipelines\A1R-tesseract-rules.json
```

- `doctor` reads no sheets. It reports a missing program, an Ollama model
  that is not pulled, or a prompt that is not registered.
- `run-all` keeps going if one candidate fails, then prints the leaderboard:
  field accuracy, sheets fully correct, silent errors, time, cost, and the gap
  to the leader with its 95% interval.
- For one run in detail: `uv run ocrbench report --run-dir <run>` (scan and
  photo accuracy are shown separately).

## 5. Final, once

After tuning on dev, tag the code and record the configs (freeze rule), then:

```powershell
$env:OCRBENCH_ALLOW_FINAL = "1"
uv run ocrbench run-all --split final --confirm-final --run-prefix final1 --leaderboard-out D:\ocrbench-runs\leaderboard-final1.md `
    --pipelines <the frozen configs>
```

A fix made after seeing final results needs a new tag and is reported as a
second, non-primary result.
