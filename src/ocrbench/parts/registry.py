"""Remember confirmed parts, so repeat part numbers are checked against them.

Every part a worker confirms is stored with its part number and measured
sizes. The next photo with that part number is compared with the median of
the earlier ones. This is the reference database Kimura does not have, built
from normal work. Repeat orders are common: in Kimura's one-day sample, 9 of
42 part numbers appeared more than once.

The registry is a SQLite file on the desk PC. It holds part numbers and sizes
only, never photos, and must stay outside the repository like all real data.
"""

from __future__ import annotations

import json
import sqlite3
import statistics
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from ocrbench.parts.vision import Measurement

REGISTRY_SCHEMA_VERSION: Final[int] = 1
_REPOSITORY_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
_SCHEMA: Final[str] = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS confirmed_parts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    part_no TEXT NOT NULL,
    confirmed_at TEXT NOT NULL,
    view TEXT NOT NULL,
    long_mm REAL NOT NULL,
    short_mm REAL NOT NULL,
    area_mm2 REAL NOT NULL,
    hole_count INTEGER NOT NULL,
    measurement_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS confirmed_parts_part_no ON confirmed_parts (part_no);
"""


class RegistryError(Exception):
    """Raised when the registry cannot be opened or written."""


@dataclass(frozen=True)
class RegistryMatch:
    """How a new measurement compares with earlier confirmed parts."""

    references: int
    comparable: int
    consistent: bool
    detail: str


def _relative_gap(measured: float, reference: float) -> float:
    return abs(measured - reference) / reference if reference > 0 else float("inf")


class PartRegistry:
    """A SQLite file of confirmed parts. Use as a context manager."""

    def __init__(self, path: Path) -> None:
        resolved = path.expanduser().resolve()
        if resolved.is_relative_to(_REPOSITORY_ROOT):
            raise RegistryError(
                "the registry holds real part numbers; keep it outside the repository"
            )
        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(resolved)
            self._connection.executescript(_SCHEMA)
            self._connection.execute(
                "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?)",
                (str(REGISTRY_SCHEMA_VERSION),),
            )
            self._connection.commit()
        except (OSError, sqlite3.Error) as error:
            raise RegistryError(f"cannot open the registry: {type(error).__name__}") from None

    def __enter__(self) -> PartRegistry:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._connection.close()

    def add(
        self, part_no: str, measurement: Measurement, *, confirmed_at: datetime | None = None
    ) -> int:
        """Store one confirmed part; returns how many are now stored for ``part_no``."""
        when = (confirmed_at or datetime.now(UTC)).isoformat(timespec="seconds")
        with self._connection:
            self._connection.execute(
                "INSERT INTO confirmed_parts (part_no, confirmed_at, view, long_mm, short_mm,"
                " area_mm2, hole_count, measurement_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    part_no,
                    when,
                    measurement.view,
                    measurement.long_mm,
                    measurement.short_mm,
                    measurement.area_mm2,
                    len(measurement.holes),
                    measurement.model_dump_json(),
                ),
            )
        return self.count(part_no)

    def count(self, part_no: str | None = None) -> int:
        if part_no is None:
            row = self._connection.execute("SELECT COUNT(*) FROM confirmed_parts").fetchone()
        else:
            row = self._connection.execute(
                "SELECT COUNT(*) FROM confirmed_parts WHERE part_no = ?", (part_no,)
            ).fetchone()
        return int(row[0])

    def distinct_part_numbers(self) -> int:
        row = self._connection.execute(
            "SELECT COUNT(DISTINCT part_no) FROM confirmed_parts"
        ).fetchone()
        return int(row[0])

    def references(self, part_no: str) -> list[Measurement]:
        rows = self._connection.execute(
            "SELECT measurement_json FROM confirmed_parts WHERE part_no = ? ORDER BY id", (part_no,)
        ).fetchall()
        return [Measurement.model_validate(json.loads(row[0])) for row in rows]

    def compare(self, part_no: str, measurement: Measurement, *, size_rel: float) -> RegistryMatch:
        """Compare with the median of earlier parts photographed in the same view.

        Long side, short side and area must each be within ``size_rel`` of the
        median (area within twice that, since it grows with the square), and a
        round view must show the same number of holes as most earlier parts.
        """
        earlier = self.references(part_no)
        same_view = [reference for reference in earlier if reference.view == measurement.view]
        if not same_view:
            return RegistryMatch(len(earlier), 0, False, "no earlier part in this view")
        medians = {
            "long side": statistics.median(reference.long_mm for reference in same_view),
            "short side": statistics.median(reference.short_mm for reference in same_view),
            "area": statistics.median(reference.area_mm2 for reference in same_view),
        }
        measured = {
            "long side": measurement.long_mm,
            "short side": measurement.short_mm,
            "area": measurement.area_mm2,
        }
        problems: list[str] = []
        for name, reference in medians.items():
            limit = size_rel * (2 if name == "area" else 1)
            gap = _relative_gap(measured[name], reference)
            if gap > limit:
                problems.append(f"{name} differs by {gap:.0%} (limit {limit:.0%})")
        if measurement.view == "round":
            usual = statistics.mode(len(reference.holes) for reference in same_view)
            if len(measurement.holes) != usual:
                problems.append(f"{len(measurement.holes)} holes, earlier parts had {usual}")
        summary = f"{len(same_view)} earlier part(s) in this view"
        if problems:
            return RegistryMatch(
                len(earlier), len(same_view), False, f"{summary}: {'; '.join(problems)}"
            )
        return RegistryMatch(
            len(earlier), len(same_view), True, f"{summary}: sizes and holes match"
        )


__all__ = ["REGISTRY_SCHEMA_VERSION", "PartRegistry", "RegistryError", "RegistryMatch"]
