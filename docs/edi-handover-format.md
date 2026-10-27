# Handover format: sheet → EDI script

The contract between Issue 1 (reading the sheet) and Issue 2 (Fuke's
Playwright registration script). One sheet in, one JSON record out.

```powershell
uv run ocrbench extract --pipeline pipelines\<chosen pipeline>.json --image D:\incoming\sheet.jpg
```

- Exit code `0`: the record is `ready`; the script may register it.
- Exit code `3`: the sheet needs review; the script must **not** register it
  and should show the worker `review_reasons`.
- Exit code `1` or `2`: the command failed (missing file, bad config, engine
  not installed); nothing was read.

Add `--json` for one compact line instead of indented JSON. The record contains
the sheet's real values, so run it on the local PC only and never log it to a
shared place.

## The record

Example with invented values:

```json
{
  "schema_version": 1,
  "pipeline_id": "A1L",
  "ready": true,
  "review_reasons": [],
  "fields": {
    "customer": "Nidecマシンツール株式会社",
    "part_no": "HR900000001",
    "order_no": "900001-2026",
    "quantity": 4
  },
  "edi": {
    "site": "hwao_nmtj",
    "order_no_printed": "900001-2026",
    "order_no_without_year": "900001",
    "quantity": 4
  },
  "rules": {"customer_table_version": 1, "order_no_rule": "keep"},
  "timing_ms": {"layer1": 1820.4, "layer2": 950.2, "total": 2790.0},
  "cost_usd": 0.0
}
```

| Key | Meaning |
|---|---|
| `ready` | `true` only when the pipeline marked the sheet ok, all 4 fields were read, and the customer maps to a known EDI site. |
| `review_reasons` | Plain-language reasons the sheet is not ready, e.g. `"quantity was not read"`, `"customer does not match any known EDI site"`. Empty when ready. |
| `fields` | The 4 values as read. Any of them can be `null`: unreadable values are never guessed. `quantity` is a whole number with the unit (`PC`) removed. |
| `edi.site` | Which EDI system to open. Today only `hwao_nmtj` (Nidec Machine Tool, hWAO NMTJ Edition) exists; the table is `src/ocrbench/customers.py`. |
| `edi.order_no_printed` | The order number exactly as printed, e.g. `023235-2025`. |
| `edi.order_no_without_year` | The same without a trailing `-YYYY`, e.g. `023235`. The EDI manual filters 手配番号 by this form. Use whichever hWAO actually matches (open decision D1). |
| `edi.quantity` | The value for 出荷数. |
| `rules` | Which customer table and order-number rule produced the record, for audit. |
| `timing_ms`, `cost_usd` | Time per layer and the cloud cost (0 for local pipelines). |

## What the script should do

1. Run `extract`. If the exit code is not `0`, stop and show `review_reasons`
   to the worker.
2. Open the EDI site named in `edi.site`, then 出荷入力 → データ取得.
3. Filter 手配番号 by the order number. If no row or more than one row
   matches, stop and ask the worker; never pick one.
4. Check that the row's part number matches `fields.part_no`. The EDI shows it
   as `HR237101021/0105`, so compare the part before the `/`. If it differs,
   stop and ask the worker. This catches a misread order number that happens
   to match another real order.
5. Tick 入力, set 納品日 = today and 出荷数 = `edi.quantity`, then 登録 → OK → OK.
6. 納品書印刷: print the delivery note for the row just registered.

The `schema_version` changes only if a key is removed or its meaning
changes; new keys may be added without notice.

## Feeding the automation-harness EDI script

Fuke's `harness ship` (in `26C04/automation-harness`) already does steps 2–6:
it finds exactly one 出荷入力 row by 手配番号, requires 入力状況 = 未, checks
the part number (`base_equals`) and that 出荷数 does not exceed 注文残数, and
stops on any mismatch. It reads the JSON report of `harness ocr read --out`.
`ocrbench harness-report` writes the same shape from any pipeline, so the
chosen pipeline (A3G) feeds it without changes on the harness side:

```powershell
uv run ocrbench harness-report --pipeline pipelines\A3G-glm-ocr-header.json `
    --images C:\incoming\today --out C:\edi\today-report.json
cd ..\automation-harness
uv run harness ship register hwao_nmtj --ocr C:\edi\today-report.json   # rehearsal, sends nothing
uv run harness ship run hwao_nmtj --ocr C:\edi\today-report.json --commit  # register + delivery note
```

Each sheet becomes one slip: `ok` with its `edi` keys when the record is
`ready`, otherwise `needs_review` with the `review_reasons` and no keys. The
harness skips `needs_review` slips. The report can also be opened in the
harness web UI (`uv run --extra ui harness ui`), where a worker checks each
row against the sheet before shipping. The report holds sheet values: keep it
on the local PC, outside the repository and OneDrive. `--images` takes files
or folders (PDF, JPEG, PNG); exit code 3 means at least one sheet needs review.
