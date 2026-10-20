"""Tests for ``ocrbench parts``, SAM3 instance selection and barcodes. All data is invented."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import zxingcpp
from numpy.typing import NDArray

from ocrbench.cli import main
from ocrbench.parts import commands
from ocrbench.parts.barcode import read_barcodes
from ocrbench.parts.sam3 import Sam3Segmenter, TransformersSam3
from ocrbench.parts.synthetic import SyntheticPart, render_photo
from ocrbench.parts.vision import VisionError, marker_scale

SHEET = "部番 HX123456789\n部品名 FLANGE 営業\n材質 SCM415\nφ 125.0 × 40.0\n4-φ11貫通 φ30貫通\n"
RIGHT = SyntheticPart("round", 118, holes=((30, 0), (11, 45), (11, 45), (11, 45), (11, 45)))
WRONG = SyntheticPart("round", 160, holes=((30, 0),))


@pytest.fixture
def desk(tmp_path: Path) -> Path:
    (tmp_path / "sheet.txt").write_text(SHEET, encoding="utf-8")
    cv2.imwrite(str(tmp_path / "right.png"), render_photo(RIGHT, angle_deg=10, seed=1))
    cv2.imwrite(str(tmp_path / "wrong.png"), render_photo(WRONG, seed=2))
    cv2.imwrite(str(tmp_path / "blank.png"), np.full((400, 400, 3), 60, dtype=np.uint8))
    return tmp_path


def _json(capsys: pytest.CaptureFixture[str]) -> dict[str, object]:
    return json.loads(capsys.readouterr().out)  # type: ignore[no-any-return]


def _check(desk: Path, photo: str, *extra: str) -> list[str]:
    return [
        "--json", "parts", "check", "--text", str(desk / "sheet.txt"),
        "--photo", str(desk / photo), "--marker-mm", "50", *extra,
    ]  # fmt: skip


def test_check_pass_fail_and_cannot_check(desk: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(_check(desk, "right.png")) == 0
    assert _json(capsys)["verdict"] == "pass"
    assert main(_check(desk, "wrong.png")) == commands.EXIT_FAIL
    assert _json(capsys)["verdict"] == "fail"
    assert main(_check(desk, "blank.png")) == commands.EXIT_CANNOT_CHECK
    assert _json(capsys)["reasons"] == ["no scale marker found in the photo"]


def test_confirm_then_registry_catches_a_wrong_repeat(
    desk: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    registry = str(desk / "registry.sqlite")
    record = desk / "right.json"
    assert main(_check(desk, "right.png", "--registry", registry, "--out", str(record))) == 0
    capsys.readouterr()

    assert (
        main(["--json", "parts", "confirm", "--record", str(record), "--registry", registry]) == 0
    )
    assert _json(capsys) == {"registry_total": 1, "stored_for_part_no": 1}
    assert main(_check(desk, "right.png", "--registry", registry)) == 0
    rules = {rule["rule"]: rule for rule in _json(capsys)["rules"]}  # type: ignore[attr-defined]
    assert rules["registry"]["status"] == "pass"

    assert main(["--json", "parts", "registry-stats", "--registry", registry]) == 0
    stats = _json(capsys)
    assert stats["confirmed_parts"] == 1 and stats["part_numbers"] == 1


def test_spec_and_measure_commands(desk: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--json", "parts", "spec", "--text", str(desk / "sheet.txt")]) == 0
    spec = _json(capsys)
    assert spec["stock_d_mm"] == 125.0 and spec["part_name"] == "FLANGE"
    (desk / "spec.json").write_text(json.dumps(spec), encoding="utf-8")

    mask = desk / "mask.png"
    measure = [
        "--json",
        "parts",
        "measure",
        "--photo",
        str(desk / "right.png"),
        "--marker-mm",
        "50",
    ]
    assert main([*measure, "--save-mask", str(mask)]) == 0
    measured = _json(capsys)
    assert measured["view"] == "round" and mask.is_file()

    assert main([*measure, "--mask", str(mask)]) == 0
    assert _json(capsys)["segmenter"] == "mask-file"
    check = ["--json", "parts", "check", "--spec", str(desk / "spec.json")]
    assert main([*check, "--photo", str(desk / "right.png"), "--mm-per-px", "0.25"]) == 0


def test_input_errors_exit_2(desk: Path, capsys: pytest.CaptureFixture[str]) -> None:
    photo = ["--photo", str(desk / "right.png"), "--marker-mm", "50"]
    text = ["--text", str(desk / "sheet.txt")]

    assert main(["parts", "check", *photo]) == 2
    assert main(["parts", "check", *text, "--spec", str(desk / "x.json"), *photo]) == 2
    assert main(["parts", "check", "--sheet", str(desk / "right.png"), *photo]) == 2
    assert main(["parts", "check", "--spec", str(desk / "missing.json"), *photo]) == 2
    assert main(["parts", "check", "--text", str(desk / "missing.txt"), *photo]) == 2
    assert main(["parts", "measure", *photo, "--segmenter", "mask"]) == 2
    assert main(["parts", "measure", *photo, "--segmenter", "sam3", "--mask", "m.png"]) == 2
    assert main(["parts", "measure", "--photo", str(desk / "blank.png"), "--marker-mm", "50"]) == 2
    assert (
        main(["parts", "check", *text, *photo, "--out", str(Path(__file__).parent / "x.json")]) == 2
    )
    assert (
        main(["parts", "confirm", "--record", str(desk / "sheet.txt"), "--registry", "r.db"]) == 2
    )
    assert "outside the repository" in capsys.readouterr().err


def test_confirm_needs_part_no_and_measurement(desk: Path) -> None:
    from ocrbench.parts.check import Tolerances, check_part
    from ocrbench.parts.spec import read_part_spec

    no_part_no = check_part(read_part_spec("φ50×80"), None, tolerances=Tolerances())
    record = desk / "record.json"
    record.write_text(no_part_no.model_dump_json(), encoding="utf-8")
    registry = str(desk / "registry.sqlite")

    assert main(["parts", "confirm", "--record", str(record), "--registry", registry]) == 2
    assert (
        main(
            [
                "parts",
                "confirm",
                "--record",
                str(record),
                "--registry",
                registry,
                "--part-no",
                "HX9",
            ]
        )
        == 2
    )


def test_calibrate_command(desk: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (desk / "sheet.txt").write_text("部番 HX1\nφ 125.0 × 40.0\n", encoding="utf-8")
    records = []
    for index in range(10):
        photo = desk / f"part{index}.png"
        cv2.imwrite(str(photo), render_photo(SyntheticPart("round", 110 + index), seed=index))
        record = desk / f"part{index}.json"
        assert main(_check(desk, photo.name, "--out", str(record))) == 0
        records.append(str(record))
    capsys.readouterr()
    out = desk / "tolerances.json"

    assert main(["--json", "parts", "calibrate", "--confirmed", *records, "--out", str(out)]) == 0
    report = _json(capsys)
    assert report["records"] == 10 and report["confirmed_rejected"] == 0
    assert json.loads(out.read_text(encoding="utf-8"))["calibrated"] is True
    assert main(["parts", "calibrate", "--confirmed", records[0], "--out", str(out)]) == 2


def test_marker_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "marker.png"

    assert main(["--json", "parts", "marker", "--out", str(out), "--mm", "40", "--dpi", "200"]) == 0
    assert _json(capsys)["marker_mm"] == 40.0
    photo = cv2.imread(str(out), cv2.IMREAD_GRAYSCALE)
    assert marker_scale(np.asarray(photo), marker_mm=40).marker_id == 0


def test_barcodes_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    barcode = zxingcpp.create_barcode("D1234567", zxingcpp.BarcodeFormat.Code39)
    image = np.asarray(barcode.to_image(scale=3), dtype=np.uint8)
    sheet = np.full((600, 900), 255, dtype=np.uint8)
    sheet[100 : 100 + image.shape[0], 100 : 100 + image.shape[1]] = image
    path = tmp_path / "sheet.png"
    cv2.imwrite(str(path), sheet)

    assert [found.text for found in read_barcodes(sheet)] == ["D1234567"]
    assert main(["--json", "parts", "barcodes", "--image", str(path)]) == 0
    found = _json(capsys)["barcodes"]
    assert found[0]["text"] == "D1234567" and found[0]["format"] == "Code 39"  # type: ignore[index]
    assert main(["parts", "barcodes", "--image", str(tmp_path / "missing.png")]) == 2


def _sam3_photo() -> tuple[NDArray[np.uint8], NDArray[np.float32]]:
    photo = render_photo(RIGHT)
    polygon = marker_scale(photo, marker_mm=50).marker_polygon
    assert polygon is not None
    return photo, polygon


def test_sam3_picks_best_instance_off_the_marker() -> None:
    photo, polygon = _sam3_photo()
    height, width = photo.shape[:2]
    on_marker = np.zeros((height, width), dtype=np.bool_)
    on_marker[40:300, 40:300] = True
    part = np.zeros((height, width), dtype=np.bool_)
    part[400:800, 800:1200] = True
    weak = np.zeros((height, width), dtype=np.bool_)
    weak[100:150, 900:950] = True

    def backend(image: NDArray[np.uint8], prompt: str) -> list[tuple[float, NDArray[np.bool_]]]:
        assert prompt == "metal part"
        return [(0.99, on_marker), (0.9, part), (0.5, weak), (0.95, np.zeros_like(part))]

    mask = Sam3Segmenter(backend=backend).segment(photo, exclude=[polygon])

    assert np.array_equal(mask, part)
    assert Sam3Segmenter(backend=backend).segmenter_id == "sam3(model=custom,prompt=metal part)"


def test_sam3_errors() -> None:
    photo, polygon = _sam3_photo()

    with pytest.raises(VisionError, match="found no part"):
        Sam3Segmenter(backend=lambda image, prompt: []).segment(photo, exclude=[polygon])
    with pytest.raises(VisionError, match="size does not match"):
        Sam3Segmenter(backend=lambda image, prompt: [(0.9, np.ones((5, 5), np.bool_))]).segment(
            photo, exclude=[]
        )


class _Tensor:
    def __init__(self, value: NDArray[np.generic]) -> None:
        self._value = value

    def detach(self) -> _Tensor:
        return self

    def cpu(self) -> _Tensor:
        return self

    def numpy(self) -> NDArray[np.generic]:
        return self._value


class _Inputs(dict[str, object]):
    def to(self, device: str) -> _Inputs:
        self["device"] = device
        return self


def test_transformers_backend_with_fake_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    photo, _polygon = _sam3_photo()
    mask = np.zeros((1, *photo.shape[:2]), dtype=np.int64)
    mask[0, 10:20, 10:20] = 1

    class Processor:
        @classmethod
        def from_pretrained(cls, model_id: str) -> Processor:
            return cls()

        def __call__(self, **kwargs: object) -> _Inputs:
            assert kwargs["text"] == "metal part"
            return _Inputs(pixel_values=1)

        def post_process_instance_segmentation(
            self, outputs: object, **kwargs: object
        ) -> list[dict[str, _Tensor]]:
            assert kwargs["target_sizes"] == [photo.shape[:2]]
            return [{"scores": _Tensor(np.array([0.8])), "masks": _Tensor(mask)}]

    class Model:
        @classmethod
        def from_pretrained(cls, model_id: str) -> Model:
            assert model_id == "facebook/sam3"
            return cls()

        def to(self, device: str) -> Model:
            return self

        def eval(self) -> Model:
            return self

        def __call__(self, **kwargs: object) -> object:
            return object()

    class NoGrad:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *_args: object) -> None:
            return None

    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False), no_grad=NoGrad)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(
        sys.modules, "transformers", SimpleNamespace(Sam3Model=Model, Sam3Processor=Processor)
    )

    instances = TransformersSam3()(photo, "metal part")

    assert len(instances) == 1
    assert instances[0][0] == pytest.approx(0.8)
    assert int(np.count_nonzero(instances[0][1])) == 100


def test_transformers_backend_without_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)

    with pytest.raises(VisionError, match="needs transformers and torch"):
        TransformersSam3()(np.zeros((10, 10, 3), np.uint8), "metal part")


def test_demo_script(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from scripts.make_synthetic_parts import main as demo

    assert demo(["--out", str(tmp_path / "demo")]) == 0
    capsys.readouterr()
    sheet = str(tmp_path / "demo" / "sheet.txt")
    expected = {"right.png": 0, "too_big.png": 3, "no_bolt_holes.png": 3, "shaft.png": 3}
    for photo, code in expected.items():
        arguments = ["parts", "check", "--text", sheet, "--marker-mm", "50"]
        assert main([*arguments, "--photo", str(tmp_path / "demo" / photo)]) == code, photo
    with pytest.raises(SystemExit):
        demo(["--out", str(Path(__file__).parent / "demo")])
