"""Build the sample page photos used by the demo button."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "samples"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def render_page() -> Image.Image:
    image = Image.new("RGB", (1200, 1600), (250, 248, 242))
    draw = ImageDraw.Draw(image)
    title = ImageFont.truetype(FONT, 92)
    body = ImageFont.truetype(FONT, 54)
    draw.text((90, 140), "PAIGE", fill=(28, 26, 22), font=title)
    lines = [
        "This page was photographed",
        "and read by a vision model.",
        "The detector finds each line.",
        "The recognizer reads the words.",
    ]
    y = 320
    for line in lines:
        draw.text((90, y), line, fill=(28, 26, 22), font=body)
        y += 110
    return image


def place_on_desk(page: Image.Image) -> np.ndarray:
    desk = np.full((1400, 1800, 3), (42, 46, 52), dtype=np.uint8)
    page_bgr = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
    source = np.array(
        [[0, 0], [page.width - 1, 0], [page.width - 1, page.height - 1], [0, page.height - 1]],
        dtype=np.float32,
    )
    destination = np.array([[280, 180], [1500, 260], [1380, 1180], [220, 1080]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(source, destination)
    warped = cv2.warpPerspective(page_bgr, matrix, (desk.shape[1], desk.shape[0]))
    mask = warped.any(axis=2)
    desk[mask] = warped[mask]
    return desk


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    page = render_page()
    page.save(OUT / "straight_page.png")
    angled = place_on_desk(page)
    cv2.imwrite(str(OUT / "angled_page.png"), angled)
    print(f"Wrote samples to {OUT}")


if __name__ == "__main__":
    main()
