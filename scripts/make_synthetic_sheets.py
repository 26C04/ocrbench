"""Generate synthetic production instruction sheets for the T25 dry run.

Every value is invented from a seeded random generator; the layout is a
generic labelled form, not a copy of a real sheet. The output folder gets
``raw/<name>.png|jpg`` and ``answers.csv`` in the format of
``ocrbench dataset import-answers``. Results on these sheets are not
benchmark results (T25). The folder must be outside the repository, so the
images can never be committed::

    uv run python -m scripts.make_synthetic_sheets --out D:\\ocrbench-synthetic --count 10
    uv run ocrbench dataset import-answers --root D:\\ocrbench-synthetic `
        --answers D:\\ocrbench-synthetic\\answers.csv
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from PIL import Image, ImageDraw, ImageFilter, ImageFont

REPOSITORY_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
CUSTOMER: Final[str] = "Nidecマシンツール株式会社"
"""The one customer mapped to an EDI site in ``ocrbench.customers``."""
FONT_CANDIDATES: Final[tuple[str, ...]] = (
    r"C:\Windows\Fonts\msgothic.ttc",
    r"C:\Windows\Fonts\YuGothM.ttc",
    r"C:\Windows\Fonts\meiryo.ttc",
    "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf",
    "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
    "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
)
ANSWER_HEADER: Final[tuple[str, ...]] = (
    "file",
    "source_kind",
    "split",
    "customer",
    "part_no",
    "order_no",
    "quantity",
)
_PAGE: Final[tuple[int, int]] = (1240, 1754)
_MATERIALS: Final[tuple[str, ...]] = ("SS400", "S45C", "SUS304", "A5052", "SPCC")


class SyntheticSheetError(Exception):
    """Raised when synthetic sheets cannot be generated safely."""


@dataclass(frozen=True)
class SyntheticSheet:
    """One invented sheet: the printed values and how it is saved."""

    file: str
    source_kind: str
    split: str
    part_no: str
    order_no: str
    quantity_printed: str
    delivery_date: str
    material: str

    def answer_row(self) -> tuple[str, ...]:
        return (
            self.file,
            self.source_kind,
            self.split,
            CUSTOMER,
            self.part_no,
            self.order_no,
            self.quantity_printed,
        )


def find_font(explicit: Path | None) -> Path:
    """Return a font that can draw Japanese text."""
    candidates = [explicit] if explicit is not None else [Path(p) for p in FONT_CANDIDATES]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate
    raise SyntheticSheetError("no Japanese font found; pass --font with a .ttf or .ttc file")


def plan_sheets(
    count: int, *, seed: int, dev_share: float = 0.4, photo_share: float = 0.5
) -> list[SyntheticSheet]:
    """Invent ``count`` sheets; both splits get scans and photos when count allows."""
    if count < 2:
        raise SyntheticSheetError("count must be at least 2 (one dev and one final sheet)")
    if not 0.0 < dev_share < 1.0 or not 0.0 <= photo_share <= 1.0:
        raise SyntheticSheetError("dev_share must be in (0, 1) and photo_share in [0, 1]")
    rng = random.Random(seed)
    n_dev = min(count - 1, max(1, round(count * dev_share)))
    sheets: list[SyntheticSheet] = []
    for index in range(count):
        split = "dev" if index < n_dev else "final"
        position = index if split == "dev" else index - n_dev
        size = n_dev if split == "dev" else count - n_dev
        photo = position >= round(size * (1 - photo_share))
        quantity = rng.randint(1, 60)
        unit = rng.choice(("PC", " PC", "", "個"))
        sheets.append(
            SyntheticSheet(
                file=f"synthetic-{index:03d}.{'jpg' if photo else 'png'}",
                source_kind="photo" if photo else "scan",
                split=split,
                part_no=f"HR{rng.randint(0, 999_999_999):09d}",
                order_no=f"{rng.randint(0, 999_999):06d}-{rng.choice((2025, 2026))}",
                quantity_printed=f"{quantity}{unit}",
                delivery_date=f"2026/{rng.randint(1, 12):02d}/{rng.randint(1, 28):02d}",
                material=rng.choice(_MATERIALS),
            )
        )
    return sheets


def render_sheet(sheet: SyntheticSheet, font_path: Path, *, seed: int) -> Image.Image:
    """Draw one sheet; photos are rotated, blurred and unevenly lit."""
    page = Image.new("RGB", _PAGE, "white")
    draw = ImageDraw.Draw(page)
    title = ImageFont.truetype(str(font_path), 64)
    label = ImageFont.truetype(str(font_path), 34)
    value = ImageFont.truetype(str(font_path), 42)
    draw.text((420, 90), "生産指示書", font=title, fill="black")
    draw.text((860, 200), "発行 2026/10/01", font=label, fill="black")
    rows = (
        ("受注先", CUSTOMER),
        ("受注No", sheet.order_no),
        ("部番", sheet.part_no),
        ("品名", "ブラケット（合成データ）"),  # noqa: RUF001 - printed full-width
        ("材質", sheet.material),
        ("個数", sheet.quantity_printed),
        ("納期", sheet.delivery_date),
        ("備考", "SYNTHETIC SHEET - NOT REAL DATA"),
    )
    top, height, left, split_x, right = 300, 130, 100, 360, 1140
    for row, (name, text) in enumerate(rows):
        y = top + row * height
        draw.rectangle((left, y, right, y + height), outline="black", width=3)
        draw.line((split_x, y, split_x, y + height), fill="black", width=3)
        draw.text((left + 24, y + 44), name, font=label, fill="black")
        draw.text((split_x + 30, y + 38), text, font=value, fill="black")
    if sheet.source_kind == "scan":
        return page.convert("L").convert("RGB")

    rng = random.Random(seed)
    photo = page.rotate(rng.uniform(-3.0, 3.0), expand=True, fillcolor=(205, 200, 190))
    shade = Image.linear_gradient("L").resize(photo.size).point(lambda v: 150 + v * 105 // 255)
    photo = Image.composite(photo, Image.new("RGB", photo.size, (60, 55, 50)), shade)
    return photo.filter(ImageFilter.GaussianBlur(rng.uniform(0.6, 1.4)))


def _inside_repository(path: Path) -> bool:
    return path.resolve().is_relative_to(REPOSITORY_ROOT)


def generate(
    out: Path,
    *,
    count: int,
    seed: int = 20261020,
    dev_share: float = 0.4,
    photo_share: float = 0.5,
    font: Path | None = None,
) -> list[SyntheticSheet]:
    """Write ``raw/`` images and ``answers.csv`` into an empty folder outside the repo."""
    if _inside_repository(out):
        raise SyntheticSheetError(
            "--out must be outside the repository so images are never committed"
        )
    if out.exists() and any(out.iterdir()):
        raise SyntheticSheetError("--out must be a new or empty folder")
    font_path = find_font(font)
    sheets = plan_sheets(count, seed=seed, dev_share=dev_share, photo_share=photo_share)
    raw = out / "raw"
    raw.mkdir(parents=True)
    for index, sheet in enumerate(sheets):
        image = render_sheet(sheet, font_path, seed=seed + index)
        if sheet.source_kind == "photo":
            image.save(raw / sheet.file, format="JPEG", quality=85)
        else:
            image.save(raw / sheet.file, format="PNG")
    with (out / "answers.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(ANSWER_HEADER)
        writer.writerows(sheet.answer_row() for sheet in sheets)
    return sheets


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--out", type=Path, required=True, help="new folder outside the repo")
    parser.add_argument("--count", type=int, default=10, help="number of sheets (default 10)")
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--dev-share", type=float, default=0.4, help="share of dev sheets")
    parser.add_argument("--photo-share", type=float, default=0.5, help="share of photo sheets")
    parser.add_argument("--font", type=Path, help="Japanese .ttf/.ttc font (auto-detected)")
    arguments = parser.parse_args(argv)
    try:
        sheets = generate(
            arguments.out,
            count=arguments.count,
            seed=arguments.seed,
            dev_share=arguments.dev_share,
            photo_share=arguments.photo_share,
            font=arguments.font,
        )
    except (SyntheticSheetError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    n_dev = sum(sheet.split == "dev" for sheet in sheets)
    print(
        f"Wrote {len(sheets)} synthetic sheets ({n_dev} dev, {len(sheets) - n_dev} final) "
        f"and answers.csv to {arguments.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
