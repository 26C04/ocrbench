"""Write a demo folder for ``ocrbench parts check``: an invented sheet and photos.

Every value is invented. The folder gets ``sheet.txt`` (a flange: stock
φ125 × 40, 4-φ11 and φ30 through holes), ``marker.png`` to print, and four
photos: the right part, a part too large for the stock, the right size with
the bolt holes missing, and a shaft. The folder must be outside the
repository::

    uv run python -m scripts.make_synthetic_parts --out D:\\parts-demo
    uv run ocrbench parts check --text D:\\parts-demo\\sheet.txt `
        --photo D:\\parts-demo\\right.png --marker-mm 50
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import cv2

from ocrbench.parts.synthetic import SyntheticPart, printable_marker, render_photo

REPOSITORY_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
SHEET_TEXT: Final[str] = "\n".join(
    [
        "受注先 Nidecマシンツール株式会社",
        "部番 HX000000001",
        "部品名 FLANGE 営業",
        "材質 SCM415",
        "φ 125.0 × 40.0",
        "4-φ11貫通 φ30貫通",
        "",
    ]
)
BOLTS: Final[tuple[tuple[float, float], ...]] = ((30, 0), (11, 45), (11, 45), (11, 45), (11, 45))
PHOTOS: Final[dict[str, tuple[SyntheticPart, float]]] = {
    "right.png": (SyntheticPart("round", 118, holes=BOLTS), 10.0),
    "too_big.png": (SyntheticPart("round", 160, holes=BOLTS), 0.0),
    "no_bolt_holes.png": (SyntheticPart("round", 118, holes=((30, 0),)), 0.0),
    "shaft.png": (SyntheticPart("profile", 36, length_mm=100), 25.0),
}


def write_demo(out: Path) -> list[Path]:
    """Write the demo folder and return the files written."""
    resolved = out.expanduser().resolve()
    if resolved.is_relative_to(REPOSITORY_ROOT):
        raise SystemExit("error: write the demo outside the repository")
    resolved.mkdir(parents=True, exist_ok=True)
    written = [resolved / "sheet.txt", resolved / "marker.png"]
    written[0].write_text(SHEET_TEXT, encoding="utf-8")
    cv2.imwrite(str(written[1]), printable_marker(marker_id=0, marker_mm=50))
    for seed, (name, (part, angle)) in enumerate(PHOTOS.items()):
        path = resolved / name
        cv2.imwrite(str(path), render_photo(part, angle_deg=angle, seed=seed))
        written.append(path)
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="folder outside the repository")
    arguments = parser.parse_args(argv)
    for path in write_demo(arguments.out):
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
