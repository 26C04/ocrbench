# Part verification (Phase 2): check a photographed part against its sheet

There is no database linking parts to CAD files yet, so the **sheet itself is the reference**.
The sheet prints:

- the part name (部品名)
- the material
- the raw stock size `φD × L` the part was cut from
- notes such as `4-φ11貫通` (four through holes of φ11)

A finished part must fit inside its stock, be close to it in size, and have the holes the sheet
lists. A photo with a known scale lets us check all of this with plain geometry. Every part a worker
confirms goes into a registry, so the next sheet with the same part number is also compared with
earlier shipments.

```
sheet ──► Phase 1 pipeline (OCR) ──► part spec: part no, name, material, stock φD×L, holes
photo ──► ArUco marker (mm/px) + segmenter (threshold / SAM3 / mask file) ──► sizes in mm
                         │
                         ▼
        rules: envelope · closeness · shape class · through holes · registry · taper (info)
                         │
                         ▼
        pass (exit 0) · fail with reasons (exit 3) · cannot check (exit 4) ──► worker
                         │ worker confirms
                         ▼
                 registry (SQLite: part no + sizes, no images)
```

Code: `src/ocrbench/parts/`. Every command prints JSON with real part numbers and sizes. Run it
on the desk PC only. Photos, records, masks and the registry stay outside the repository: the
commands refuse to write them inside it.

## Station setup

1. Print the scale marker at 100% (no "fit to page"):
   `uv run ocrbench parts marker --out D:\parts\marker.png --mm 50`.
   Measure the black square with a ruler. It must be 50 mm.
2. Use a plain mat that contrasts with metal, such as dark grey or black. Light it evenly.
3. Fix the camera straight above the mat. The marker lies flat beside the part.
4. Lay the part in its natural resting position:
   - discs and flanges flat (the photo shows a circle)
   - shafts on their side (the photo shows a profile)

## Commands

| Command | What it does |
|---|---|
| `parts spec --sheet S --pipeline P` | Reads the part spec from a sheet with a Phase 1 pipeline. `--text` (read text) and `--spec` (saved spec) also work. |
| `parts measure --photo F --marker-mm 50` | Measures the part in mm. `--save-mask` writes the mask. |
| `parts check --sheet S --pipeline P --photo F --marker-mm 50 [--registry R] [--tolerances T] [--out rec.json]` | Gives the verdict. Exit 0 is pass, 3 is fail, 4 is cannot check. |
| `parts confirm --record rec.json --registry R` | Stores a part the worker confirmed. |
| `parts registry-stats --registry R` | Counts confirmed parts and part numbers. |
| `parts calibrate --confirmed rec*.json [--mismatch bad*.json] --out T` | Sets the tolerances from confirmed parts. |
| `parts marker --out M --mm 50` | Writes a printable ArUco marker (PNG with its DPI set). |
| `parts barcodes --image S` | Decodes the barcodes on a sheet, such as the 伝票No. |

Segmenters (`--segmenter`):

- `threshold` (default) is an Otsu threshold for a plain mat.
- `sam3` runs SAM3 through `transformers`; see below.
- `--mask FILE` uses a mask made elsewhere, for example by a separate SAM3 script.

Use `--mm-per-px` instead of `--marker-mm` when the camera height is fixed and measured once.

Try it without real parts:

```powershell
uv run python -m scripts.make_synthetic_parts --out D:\parts-demo
uv run ocrbench parts check --text D:\parts-demo\sheet.txt --photo D:\parts-demo\right.png --marker-mm 50
```

The demo has one right photo and three wrong ones. Only `right.png` passes:

- `too_big.png` is larger than the stock.
- `no_bolt_holes.png` lacks the four φ11 holes.
- `shaft.png` is a bar where the sheet describes a disc.

## Rules

| Rule | Passes when | Not checked when |
|---|---|---|
| envelope D / L | measured ≤ stock × (1 + slack) + slack mm | no stock size; L is not visible in a round view |
| closeness D / L | measured ≥ stock × minimum fraction | same |
| shape class | a disc stock (L/D < 0.5) shows a round outline; a long stock (L/D > 2) shows a profile | no stock size; L/D between 0.5 and 2 |
| through holes | every `φd貫通` is found in a round view, as many times as printed | no hole note; profile view |
| registry | sizes, area and hole count match the median of earlier confirmed parts in the same view | no registry; first time this part number is seen |
| taper | information only: the ratio of the two end widths | no taper on the sheet; round view |

The verdict:

- **fail** if any rule fails
- **cannot check** if no size rule could run (no stock size and no earlier part), or the photo
  could not be measured (no marker, no part found)
- **pass** otherwise

A failed or uncheckable part goes to the worker with the reasons. The check never overrides a
person.

## Calibration: the tolerances are placeholders until measured

The default tolerances are guesses, and every record says `"calibrated": false`. Before the check
is trusted:

1. Photograph about 20 confirmed part and sheet pairs at Kimura with `parts check --out`.
2. Also photograph a few deliberate mismatches: a part with someone else's sheet.
3. Run `parts calibrate`:

   ```powershell
   uv run ocrbench parts calibrate --confirmed D:\parts\ok\*.json --mismatch D:\parts\bad\*.json --out D:\parts\tolerances.json
   ```

   It takes the worst case seen on the correct parts and adds a 2% margin. It then reports how many
   correct parts the new limits would reject (this should be 0) and how many mismatches they catch.
   With fewer than 10 records it refuses.
4. Pass `--tolerances D:\parts\tolerances.json` to every `parts check`.

## SAM3 on the GPU PC

```powershell
uv pip install transformers torch   # not project dependencies: torch is large
hf auth login                       # facebook/sam3 is gated: accept its licence once on Hugging Face
uv run ocrbench parts check ... --segmenter sam3 --sam3-prompt "metal part"
```

The adapter uses `Sam3Model` and `Sam3Processor` from `transformers` 5.x. It drops instances that
lie mostly on the marker, then keeps the one with the highest score. To compare SAM3 with the
threshold segmenter, run `parts measure` with both on the same photos.

## What the sample sheets give

These counts come from reading the 51 sheets of Kimura's sample PDF with `parts spec`'s rules. Only
counts are shown here.

| Field | Sheets |
|---|---|
| part number, part name | 51 of 51 |
| material | 49 of 51 |
| stock size `φD × L` | 48 of 51 (one sheet has 3 candidates; the first is used and flagged) |
| through-hole note | 23 of 51 |
| taper mention | 3 of 51 |
| 内径 mention | 6 of 51 (a value next to it on 2) |
| 電子カルテ number | 50 of 51 |

## Known limits

- **Height.** The top of a tall part is closer to the camera and looks slightly larger. A tilted
  camera stretches one direction. Calibration absorbs both; a fixed camera keeps them constant.
- **Same envelope.** Two different parts with the same stock size and the same holes pass the
  sheet rules. The registry, and later CAD (when Kimura links files to 部番 or 図番), close that gap.
- **Profile view.** Holes are not visible from the side.
- **Shiny surfaces.** Reflections can break the threshold mask. Use SAM3 or a matt mat, and check
  the mask with `--save-mask`.
