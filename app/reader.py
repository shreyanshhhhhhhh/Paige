"""Read words on a page with a detection model and a recognition model."""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

DETECTOR = "db_mobilenet_v3_large"
RECOGNIZER = "crnn_mobilenet_v3_small"
BOX_COLOR = (40, 61, 226)


@dataclass(frozen=True)
class TextLine:
    text: str
    confidence: float
    box: tuple[int, int, int, int]


class PageReader:
    """Lazy loader for the docTR OCR predictor."""

    def __init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                from doctr.models import ocr_predictor

                model = ocr_predictor(
                    det_arch=DETECTOR,
                    reco_arch=RECOGNIZER,
                    pretrained=True,
                )
                model.eval()
                self._model = model
        return self._model

    def read(self, image_bgr: np.ndarray) -> list[TextLine]:
        import cv2
        import torch

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        model = self.load()
        with torch.inference_mode():
            result = model([rgb])
        return lines_from_export(result.export(), image_bgr.shape[1], image_bgr.shape[0])


def lines_from_export(exported: dict, width: int, height: int) -> list[TextLine]:
    lines: list[TextLine] = []
    pages = exported.get("pages") or []
    if not pages:
        return lines
    for block in pages[0].get("blocks") or []:
        for line in block.get("lines") or []:
            words = [word for word in line.get("words") or [] if word.get("value")]
            if not words:
                continue
            text = " ".join(word["value"] for word in words)
            confidence = sum(float(word.get("confidence") or 0.0) for word in words) / len(words)
            box = _pixel_box(line.get("geometry") or words[0].get("geometry"), width, height)
            lines.append(TextLine(text=text, confidence=round(confidence, 4), box=box))
    return lines


def draw_lines(image_bgr: np.ndarray, lines: list[TextLine]) -> np.ndarray:
    import cv2

    canvas = image_bgr.copy()
    for line in lines:
        x1, y1, x2, y2 = line.box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), BOX_COLOR, 2)
    return canvas


def _pixel_box(geometry, width: int, height: int) -> tuple[int, int, int, int]:
    (x1, y1), (x2, y2) = geometry
    return (
        _clamp(int(round(x1 * width)), 0, width - 1),
        _clamp(int(round(y1 * height)), 0, height - 1),
        _clamp(int(round(x2 * width)), 0, width - 1),
        _clamp(int(round(y2 * height)), 0, height - 1),
    )


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(value, high))
