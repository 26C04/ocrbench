"""SAM3 segmenter: find the part in a photo from a text prompt.

Uses ``Sam3Model`` and ``Sam3Processor`` from Hugging Face ``transformers``
(5.x). They are not project dependencies because ``torch`` is large and
needs the GPU PC; install them there with::

    uv pip install transformers torch

The weights (``facebook/sam3``) are gated on Hugging Face: accept the licence
once and log in with ``hf auth login`` before the first run.

SAM3 may return several instances. The marker's paper is never the part, so
instances lying mostly on the marker are dropped, and the highest-scoring
remaining one is used.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any, Final

import cv2
import numpy as np
from numpy.typing import NDArray

from ocrbench.parts.vision import MARKER_EXCLUDE_GROWTH, Image, Mask, VisionError

DEFAULT_MODEL: Final[str] = "facebook/sam3"
DEFAULT_PROMPT: Final[str] = "metal part"
_MAX_MARKER_SHARE: Final[float] = 0.5

Instances = list[tuple[float, NDArray[np.bool_]]]
"""(score, mask) pairs, one per instance SAM3 found."""
Backend = Callable[[Image, str], Instances]


class TransformersSam3:
    """Runs SAM3 through ``transformers``; loads the model on first use."""

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_MODEL,
        device: str | None = None,
        threshold: float = 0.4,
        mask_threshold: float = 0.5,
    ) -> None:
        self.model_id = model_id
        self._device = device
        self._threshold = threshold
        self._mask_threshold = mask_threshold
        self._loaded: tuple[Any, Any, Any] | None = None

    def _load(self) -> tuple[Any, Any, Any]:
        if self._loaded is None:
            try:
                torch = importlib.import_module("torch")
                transformers = importlib.import_module("transformers")
            except ImportError:
                raise VisionError(
                    "SAM3 needs transformers and torch: uv pip install transformers torch"
                ) from None
            device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            model = transformers.Sam3Model.from_pretrained(self.model_id).to(device).eval()
            processor = transformers.Sam3Processor.from_pretrained(self.model_id)
            self._device = device
            self._loaded = (torch, model, processor)
        return self._loaded

    def __call__(self, image: Image, prompt: str) -> Instances:
        torch, model, processor = self._load()
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        inputs = processor(images=rgb, text=prompt, return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = model(**inputs)
        result = processor.post_process_instance_segmentation(
            outputs,
            threshold=self._threshold,
            mask_threshold=self._mask_threshold,
            target_sizes=[image.shape[:2]],
        )[0]
        scores = result["scores"].detach().cpu().numpy()
        masks = result["masks"].detach().cpu().numpy()
        return [
            (float(score), np.asarray(mask > 0)) for score, mask in zip(scores, masks, strict=True)
        ]


class Sam3Segmenter:
    """The :class:`~ocrbench.parts.vision.Segmenter` backed by SAM3."""

    def __init__(self, *, prompt: str = DEFAULT_PROMPT, backend: Backend | None = None) -> None:
        self._prompt = prompt
        self._backend: Backend = backend if backend is not None else TransformersSam3()

    @property
    def segmenter_id(self) -> str:
        model = getattr(self._backend, "model_id", "custom")
        return f"sam3(model={model},prompt={self._prompt})"

    def segment(self, image: Image, *, exclude: list[NDArray[np.float32]]) -> Mask:
        blocked = np.zeros(image.shape[:2], dtype=np.uint8)
        for polygon in exclude:
            centre = polygon.mean(axis=0)
            grown = centre + (polygon - centre) * MARKER_EXCLUDE_GROWTH
            cv2.fillPoly(blocked, [np.round(grown).astype(np.int32)], 1)
        candidates: Instances = []
        for score, mask in self._backend(image, self._prompt):
            if mask.shape != image.shape[:2]:
                raise VisionError("SAM3 mask size does not match the photo")
            area = int(np.count_nonzero(mask))
            if area == 0:
                continue
            on_marker = int(np.count_nonzero(mask & (blocked > 0)))
            if on_marker / area <= _MAX_MARKER_SHARE:
                candidates.append((score, mask))
        if not candidates:
            raise VisionError("SAM3 found no part in the photo")
        _score, best = max(candidates, key=lambda candidate: candidate[0])
        return np.asarray(best & (blocked == 0), dtype=np.bool_)


__all__ = ["DEFAULT_MODEL", "DEFAULT_PROMPT", "Sam3Segmenter", "TransformersSam3"]
