import numpy as np

from app.scanner import find_page_corners, order_corners, prepare_page, warp_page


def test_order_corners_puts_top_left_first():
    points = np.array([[80, 10], [10, 80], [10, 10], [80, 80]], dtype=np.float32)
    ordered = order_corners(points)
    assert ordered.tolist() == [[10, 10], [80, 10], [80, 80], [10, 80]]


def test_find_and_flatten_an_angled_page():
    image = np.full((900, 1200, 3), 35, dtype=np.uint8)
    sheet = np.array([[220, 140], [980, 190], [930, 760], [250, 720]], dtype=np.int32)
    import cv2

    cv2.fillConvexPoly(image, sheet, (246, 243, 236))
    corners = find_page_corners(image)
    assert corners is not None
    warped = warp_page(image, corners)
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    assert gray.mean() > 180
    assert warped.shape[0] > 200 and warped.shape[1] > 200


def test_full_frame_page_is_left_alone():
    image = np.full((600, 800, 3), 240, dtype=np.uint8)
    found, prepared = prepare_page(image)
    assert found is False
    assert prepared.shape[0] == 600
    assert prepared.shape[1] == 800
