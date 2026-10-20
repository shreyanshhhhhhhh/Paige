"""Web app: upload a photo of a page, get the words back."""

from __future__ import annotations

import base64
import io
import time
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from app.reader import DETECTOR, RECOGNIZER, PageReader, TextLine, draw_lines
from app.scanner import prepare_page

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "app" / "static"
SAMPLES = ROOT / "samples"
MAX_BYTES = 8 * 1024 * 1024

app = FastAPI(title="Paige")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
app.mount("/samples", StaticFiles(directory=SAMPLES), name="samples")

_reader = PageReader()


def get_reader() -> PageReader:
    return _reader


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health(reader: PageReader = Depends(get_reader)) -> dict:
    return {
        "ok": True,
        "model_loaded": reader.loaded,
        "detector": DETECTOR,
        "recognizer": RECOGNIZER,
    }


@app.post("/api/read")
async def read_page(
    file: UploadFile = File(...),
    reader: PageReader = Depends(get_reader),
) -> dict:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Upload an image of a page.")
    if len(raw) > MAX_BYTES:
        raise HTTPException(status_code=400, detail="Image must be under 8 MB.")

    try:
        image = decode_image(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    started = time.perf_counter()
    page_found, prepared = prepare_page(image)
    lines = reader.read(prepared)
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    preview = draw_lines(prepared, lines)

    return {
        "text": "\n".join(line.text for line in lines),
        "lines": [_line_payload(line) for line in lines],
        "page_found": page_found,
        "elapsed_ms": elapsed_ms,
        "width": int(prepared.shape[1]),
        "height": int(prepared.shape[0]),
        "detector": DETECTOR,
        "recognizer": RECOGNIZER,
        "preview": _jpeg_data_url(preview),
    }


def decode_image(raw: bytes) -> np.ndarray:
    import cv2

    try:
        with Image.open(io.BytesIO(raw)) as img:
            rgb = np.array(img.convert("RGB"))
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Upload a JPEG, PNG, or WebP image.") from exc
    if rgb.size == 0 or min(rgb.shape[0], rgb.shape[1]) < 16:
        raise ValueError("That image is too small to read.")
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _line_payload(line: TextLine) -> dict:
    return {
        "text": line.text,
        "confidence": line.confidence,
        "box": list(line.box),
    }


def _jpeg_data_url(image_bgr: np.ndarray) -> str:
    import cv2

    ok, encoded = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise HTTPException(status_code=500, detail="Could not draw the page preview.")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"
