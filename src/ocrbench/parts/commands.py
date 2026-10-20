"""``ocrbench parts …``: the Phase 2 part check on the desk PC.

The records these commands print contain real part numbers and sizes, like
``ocrbench extract``: run them on the local PC only, and every file they write
must be outside the repository.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

import cv2
import numpy as np

from ocrbench.parts.barcode import read_barcodes
from ocrbench.parts.calibrate import MIN_RECORDS, CalibrationError, calibrate, load_records
from ocrbench.parts.check import PartCheck, Verdict, check_part, load_tolerances
from ocrbench.parts.registry import PartRegistry, RegistryMatch
from ocrbench.parts.spec import PartSpec, read_part_spec
from ocrbench.parts.synthetic import printable_marker
from ocrbench.parts.vision import (
    Image,
    MaskFileSegmenter,
    Measurement,
    Scale,
    Segmenter,
    ThresholdSegmenter,
    VisionError,
    fixed_scale,
    load_image,
    marker_scale,
    measure_mask,
)

EXIT_FAIL: Final[int] = 3
EXIT_CANNOT_CHECK: Final[int] = 4
_REPOSITORY_ROOT: Final[Path] = Path(__file__).resolve().parents[3]

SheetTextReader = Callable[[Path, Path, bool], str]
"""(sheet, pipeline config, confirm cloud) -> the pipeline's Layer 1 text."""
Leaf = Callable[..., argparse.ArgumentParser]


class PartsCommandError(ValueError):
    """A rejected input for a parts command (exit code 2)."""


def _outside_repository(path: Path, what: str) -> Path:
    if path.expanduser().resolve().is_relative_to(_REPOSITORY_ROOT):
        raise PartsCommandError(f"{what} holds real data; write it outside the repository")
    return path


def _print_json(arguments: argparse.Namespace, payload: Any) -> None:
    compact = bool(getattr(arguments, "json", False))
    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=None if compact else 2,
            separators=(",", ":") if compact else None,
        )
    )


def _add_spec_source(parser: argparse.ArgumentParser) -> None:
    source = parser.add_argument_group("sheet (one of)")
    source.add_argument("--sheet", type=Path, help="the sheet: PDF, JPEG or PNG (needs --pipeline)")
    source.add_argument("--pipeline", type=Path, help="pipeline config whose OCR reads the sheet")
    source.add_argument("--confirm-cloud", action="store_true")
    source.add_argument("--text", type=Path, help="the sheet's text, already read (UTF-8)")
    source.add_argument("--spec", type=Path, help="a spec JSON from `ocrbench parts spec`")


def _add_photo(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--photo", type=Path, required=True, help="photo of the part from above")
    scale = parser.add_mutually_exclusive_group(required=True)
    scale.add_argument("--marker-mm", type=float, help="side of the printed ArUco marker in mm")
    scale.add_argument("--mm-per-px", type=float, help="fixed camera scale, measured once")
    parser.add_argument("--marker-id", type=int, help="which marker sets the scale, if several")
    parser.add_argument(
        "--segmenter",
        choices=("threshold", "sam3", "mask"),
        default="threshold",
        help="how the part is found (default threshold; mask needs --mask)",
    )
    parser.add_argument("--mask", type=Path, help="a mask image made elsewhere (non-zero = part)")
    parser.add_argument("--sam3-prompt", default="metal part")
    parser.add_argument("--sam3-model", default="facebook/sam3")


def _spec_from(arguments: argparse.Namespace, read_sheet_text: SheetTextReader) -> PartSpec:
    sources = [arguments.sheet is not None, arguments.text is not None, arguments.spec is not None]
    if sum(sources) != 1:
        raise PartsCommandError("give exactly one of --sheet, --text or --spec")
    if arguments.spec is not None:
        try:
            return PartSpec.model_validate(
                json.loads(Path(arguments.spec).read_text(encoding="utf-8"))
            )
        except (OSError, json.JSONDecodeError, ValueError) as error:
            raise PartsCommandError(f"cannot read spec file: {type(error).__name__}") from None
    if arguments.text is not None:
        try:
            return read_part_spec(Path(arguments.text).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError) as error:
            raise PartsCommandError(f"cannot read text file: {type(error).__name__}") from None
    if arguments.pipeline is None:
        raise PartsCommandError("--sheet needs --pipeline")
    return read_part_spec(
        read_sheet_text(arguments.sheet, arguments.pipeline, bool(arguments.confirm_cloud))
    )


def _segmenter(arguments: argparse.Namespace) -> Segmenter:
    if arguments.mask is not None:
        if arguments.segmenter == "sam3":
            raise PartsCommandError("--mask and --segmenter sam3 exclude each other")
        return MaskFileSegmenter(arguments.mask)
    if arguments.segmenter == "mask":
        raise PartsCommandError("--segmenter mask needs --mask")
    if arguments.segmenter == "sam3":
        from ocrbench.parts.sam3 import Sam3Segmenter, TransformersSam3

        return Sam3Segmenter(
            prompt=arguments.sam3_prompt, backend=TransformersSam3(model_id=arguments.sam3_model)
        )
    return ThresholdSegmenter()


def _measure(arguments: argparse.Namespace) -> tuple[Measurement, Segmenter, Image, Scale]:
    image = load_image(arguments.photo)
    if arguments.marker_mm is not None:
        scale = marker_scale(image, marker_mm=arguments.marker_mm, marker_id=arguments.marker_id)
    else:
        scale = fixed_scale(arguments.mm_per_px)
    segmenter = _segmenter(arguments)
    exclude = [scale.marker_polygon] if scale.marker_polygon is not None else []
    mask = segmenter.segment(image, exclude=exclude)
    if getattr(arguments, "save_mask", None) is not None:
        target = _outside_repository(arguments.save_mask, "--save-mask")
        cv2.imwrite(str(target), np.asarray(mask, dtype=np.uint8) * 255)
    return measure_mask(mask, scale), segmenter, image, scale


def make_handlers(
    read_sheet_text: SheetTextReader,
) -> dict[str, Callable[[argparse.Namespace], int]]:
    """The handler for each ``parts`` subcommand, bound to the CLI's sheet reader."""

    def spec(arguments: argparse.Namespace) -> int:
        _print_json(arguments, _spec_from(arguments, read_sheet_text).model_dump(mode="json"))
        return 0

    def measure(arguments: argparse.Namespace) -> int:
        try:
            measurement, segmenter, _image, _scale = _measure(arguments)
        except VisionError as error:
            raise PartsCommandError(str(error)) from None
        _print_json(
            arguments, {"segmenter": segmenter.segmenter_id, **measurement.model_dump(mode="json")}
        )
        return 0

    def check(arguments: argparse.Namespace) -> int:
        part_spec = _spec_from(arguments, read_sheet_text)
        tolerances = load_tolerances(arguments.tolerances)
        if arguments.out is not None:
            _outside_repository(arguments.out, "--out")
        measurement: Measurement | None = None
        segmenter_id: str | None = None
        error_text: str | None = None
        try:
            measurement, segmenter, _image, _scale = _measure(arguments)
            segmenter_id = segmenter.segmenter_id
        except VisionError as error:
            error_text = str(error)
        match: RegistryMatch | None = None
        if arguments.registry is not None and measurement is not None:
            if part_spec.part_no is None:
                match = RegistryMatch(0, 0, False, "part number not read from the sheet")
            else:
                with PartRegistry(arguments.registry) as registry:
                    match = registry.compare(
                        part_spec.part_no, measurement, size_rel=tolerances.registry_size_rel
                    )
        result = check_part(
            part_spec,
            measurement,
            tolerances=tolerances,
            segmenter=segmenter_id,
            registry_match=match,
            measurement_error=error_text,
        )
        payload = result.model_dump(mode="json")
        if arguments.out is not None:
            arguments.out.parent.mkdir(parents=True, exist_ok=True)
            arguments.out.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        _print_json(arguments, payload)
        if result.verdict is Verdict.PASS:
            return 0
        return EXIT_FAIL if result.verdict is Verdict.FAIL else EXIT_CANNOT_CHECK

    def confirm(arguments: argparse.Namespace) -> int:
        try:
            record = PartCheck.model_validate(
                json.loads(Path(arguments.record).read_text(encoding="utf-8"))
            )
        except (OSError, json.JSONDecodeError, ValueError) as error:
            raise PartsCommandError(f"cannot read check record: {type(error).__name__}") from None
        part_no = arguments.part_no or record.spec.part_no
        if part_no is None:
            raise PartsCommandError("the record has no part number; pass --part-no")
        if record.measurement is None:
            raise PartsCommandError("the record has no measurement to store")
        with PartRegistry(arguments.registry) as registry:
            stored = registry.add(part_no, record.measurement)
            total = registry.count()
        _print_json(arguments, {"stored_for_part_no": stored, "registry_total": total})
        return 0

    def registry_stats(arguments: argparse.Namespace) -> int:
        with PartRegistry(arguments.registry) as registry:
            payload = {
                "confirmed_parts": registry.count(),
                "part_numbers": registry.distinct_part_numbers(),
            }
        _print_json(arguments, payload)
        return 0

    def calibrate_command(arguments: argparse.Namespace) -> int:
        _outside_repository(arguments.out, "--out")
        try:
            report = calibrate(
                load_records(list(arguments.confirmed)),
                mismatches=load_records(list(arguments.mismatch or [])),
                min_records=arguments.min_records,
            )
        except CalibrationError as error:
            raise PartsCommandError(str(error)) from None
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        arguments.out.write_text(report.tolerances.model_dump_json(indent=2), encoding="utf-8")
        _print_json(
            arguments,
            {
                "records": report.records,
                "size_readings": report.size_ratios,
                "hole_readings": report.hole_errors,
                "confirmed_rejected": report.correct_rejected,
                "mismatches": report.mismatches,
                "mismatches_caught": report.mismatches_caught,
                "tolerances": report.tolerances.model_dump(mode="json"),
            },
        )
        return 0

    def marker(arguments: argparse.Namespace) -> int:
        image = printable_marker(marker_id=arguments.id, marker_mm=arguments.mm, dpi=arguments.dpi)
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(arguments.out), image, [cv2.IMWRITE_PNG_COMPRESSION, 9]):
            raise PartsCommandError("cannot write the marker image")
        _set_png_dpi(arguments.out, arguments.dpi)
        _print_json(
            arguments, {"out": str(arguments.out), "marker_mm": arguments.mm, "dpi": arguments.dpi}
        )
        return 0

    def barcodes(arguments: argparse.Namespace) -> int:
        from ocrbench.preprocess import PreprocessError, preprocess

        try:
            prepared = preprocess(arguments.image)
        except PreprocessError as error:
            raise PartsCommandError(f"cannot read the image: {type(error).__name__}") from None
        decoded = cv2.imdecode(
            np.frombuffer(prepared.png_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE
        )
        found = read_barcodes(np.asarray(decoded, dtype=np.uint8))
        _print_json(arguments, {"barcodes": [vars(barcode) for barcode in found]})
        return 0

    return {
        "spec": spec,
        "measure": measure,
        "check": check,
        "confirm": confirm,
        "registry-stats": registry_stats,
        "calibrate": calibrate_command,
        "marker": marker,
        "barcodes": barcodes,
    }


def _set_png_dpi(path: Path, dpi: int) -> None:
    """Record the print resolution so "100%" prints the marker at its true size."""
    from PIL import Image as PilImage

    with PilImage.open(path) as image:
        image.load()
        image.save(path, dpi=(dpi, dpi))


def register(
    commands: argparse._SubParsersAction[Any],
    *,
    leaf: Leaf,
    shared: argparse.ArgumentParser,
    read_sheet_text: SheetTextReader,
) -> None:
    """Add ``ocrbench parts`` and its subcommands to the CLI."""
    handlers = make_handlers(read_sheet_text)
    parts = commands.add_parser(
        "parts",
        parents=[shared],
        help="Phase 2: check a photographed part against its sheet",
        description="Phase 2: check a photographed part against its sheet.",
    )
    sub = parts.add_subparsers(dest="parts_command", metavar="COMMAND")

    spec = leaf(sub, "spec", shared=shared, help_text="read the part spec printed on a sheet")
    _add_spec_source(spec)
    spec.set_defaults(handler=handlers["spec"])

    measure = leaf(sub, "measure", shared=shared, help_text="measure the part in a photo, in mm")
    _add_photo(measure)
    measure.add_argument("--save-mask", type=Path, help="write the part mask (outside the repo)")
    measure.set_defaults(handler=handlers["measure"])

    check = leaf(
        sub,
        "check",
        shared=shared,
        help_text="check a photo against its sheet; exit 0 pass, 3 fail, 4 cannot check",
    )
    _add_spec_source(check)
    _add_photo(check)
    check.add_argument("--tolerances", type=Path, help="tolerances JSON from `parts calibrate`")
    check.add_argument("--registry", type=Path, help="registry of confirmed parts (SQLite)")
    check.add_argument("--save-mask", type=Path, help="write the part mask (outside the repo)")
    check.add_argument("--out", type=Path, help="also save the record as JSON (outside the repo)")
    check.set_defaults(handler=handlers["check"])

    confirm = leaf(
        sub, "confirm", shared=shared, help_text="store a worker-confirmed part in the registry"
    )
    confirm.add_argument(
        "--record", type=Path, required=True, help="record from `parts check --out`"
    )
    confirm.add_argument("--registry", type=Path, required=True)
    confirm.add_argument("--part-no", help="part number, when the sheet's could not be read")
    confirm.set_defaults(handler=handlers["confirm"])

    stats = leaf(sub, "registry-stats", shared=shared, help_text="count the parts in a registry")
    stats.add_argument("--registry", type=Path, required=True)
    stats.set_defaults(handler=handlers["registry-stats"])

    calibrate_parser = leaf(
        sub,
        "calibrate",
        shared=shared,
        help_text="set tolerances from worker-confirmed check records",
    )
    calibrate_parser.add_argument("--confirmed", type=Path, nargs="+", required=True)
    calibrate_parser.add_argument(
        "--mismatch", type=Path, nargs="*", help="records of known wrong pairs"
    )
    calibrate_parser.add_argument(
        "--out", type=Path, required=True, help="tolerances JSON to write"
    )
    calibrate_parser.add_argument("--min-records", type=int, default=MIN_RECORDS)
    calibrate_parser.set_defaults(handler=handlers["calibrate"])

    marker = leaf(
        sub, "marker", shared=shared, help_text="write a printable ArUco scale marker (PNG)"
    )
    marker.add_argument("--out", type=Path, required=True)
    marker.add_argument(
        "--mm", type=float, default=50.0, help="side of the black square (default 50)"
    )
    marker.add_argument("--id", type=int, default=0)
    marker.add_argument("--dpi", type=int, default=300)
    marker.set_defaults(handler=handlers["marker"])

    barcode = leaf(sub, "barcodes", shared=shared, help_text="decode the barcodes on a sheet")
    barcode.add_argument("--image", type=Path, required=True, help="the sheet: PDF, JPEG or PNG")
    barcode.set_defaults(handler=handlers["barcodes"])


__all__ = ["EXIT_CANNOT_CHECK", "EXIT_FAIL", "PartsCommandError", "register"]
