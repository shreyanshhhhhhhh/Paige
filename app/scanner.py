"""Find a page in a photo and flatten it."""

from __future__ import annotations

import numpy as np

CORNER_COUNT = 4
MIN_PAGE_AREA = 0.18
MAX_PAGE_AREA = 0.97
MIN_CORNER_ANGLE = 45.0
MAX_CORNER_ANGLE = 140.0


def order_corners(points: np.ndarray) -> np.ndarray:
    """Return corners as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(points, dtype=np.float32).reshape(CORNER_COUNT, 2)
    ordered = np.zeros((CORNER_COUNT, 2), dtype=np.float32)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    ordered[0] = pts[np.argmin(sums)]
    ordered[2] = pts[np.argmax(sums)]
    ordered[1] = pts[np.argmin(diffs)]
    ordered[3] = pts[np.argmax(diffs)]
    return ordered


def find_page_corners(image_bgr: np.ndarray) -> np.ndarray | None:
    """Return four page corners in the original image, or None when no sheet stands out."""
    import cv2

    height, width = image_bgr.shape[:2]
    scale = 1.0 if max(height, width) <= 900 else 900 / max(height, width)
    small = cv2.resize(image_bgr, (int(width * scale), int(height * scale)))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = float(small.shape[0] * small.shape[1])
    best: np.ndarray | None = None
    best_area = 0.0

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < MIN_PAGE_AREA * image_area or area > MAX_PAGE_AREA * image_area:
            continue
        peri = cv2.arcLength(contour, True)
        for epsilon in (0.015, 0.02, 0.03, 0.04, 0.05):
            approx = cv2.approxPolyDP(contour, epsilon * peri, True)
            if len(approx) != CORNER_COUNT or not cv2.isContourConvex(approx):
                continue
            corners = order_corners(approx.reshape(CORNER_COUNT, 2).astype(np.float32) / scale)
            if area > best_area and _corners_look_like_a_page(corners):
                best = corners
                best_area = area
            break

    return best


def warp_page(image_bgr: np.ndarray, corners: np.ndarray, max_side: int = 1600) -> np.ndarray:
    """Perspective-correct a page so it fills the frame."""
    import cv2

    tl, tr, br, bl = order_corners(corners)
    width = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    height = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    width = max(width, 1)
    height = max(height, 1)
    scale = min(1.0, max_side / max(width, height))
    width = max(int(width * scale), 1)
    height = max(int(height * scale), 1)
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(order_corners(corners), destination)
    return cv2.warpPerspective(image_bgr, matrix, (width, height))


def limit_size(image_bgr: np.ndarray, max_side: int = 1600) -> np.ndarray:
    import cv2

    height, width = image_bgr.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale == 1.0:
        return image_bgr
    return cv2.resize(
        image_bgr,
        (max(int(width * scale), 1), max(int(height * scale), 1)),
        interpolation=cv2.INTER_AREA,
    )


def prepare_page(image_bgr: np.ndarray) -> tuple[bool, np.ndarray]:
    """Flatten the page when a sheet is visible. Otherwise shrink the original photo."""
    corners = find_page_corners(image_bgr)
    if corners is None:
        return False, limit_size(image_bgr)
    return True, limit_size(warp_page(image_bgr, corners))


def _corners_look_like_a_page(corners: np.ndarray) -> bool:
    tl, tr, br, bl = corners
    edges = (
        np.linalg.norm(tr - tl),
        np.linalg.norm(br - tr),
        np.linalg.norm(bl - br),
        np.linalg.norm(tl - bl),
    )
    if min(edges) < 40:
        return False
    angles = (
        _angle(bl, tl, tr),
        _angle(tl, tr, br),
        _angle(tr, br, bl),
        _angle(br, bl, tl),
    )
    return all(MIN_CORNER_ANGLE <= angle <= MAX_CORNER_ANGLE for angle in angles)


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    cosine = float(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
