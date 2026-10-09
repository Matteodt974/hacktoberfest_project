import cv2
import numpy as np

from lecturify.ingest.frames import containment


def _canvas():
    return np.full((90, 160), 20, np.uint8)


def test_partial_drawing_is_contained_in_complete_one():
    partial, full = _canvas(), _canvas()
    cv2.line(partial, (10, 45), (80, 45), 255, 2)
    cv2.line(full, (10, 45), (80, 45), 255, 2)
    cv2.circle(full, (120, 45), 25, 255, 2)
    assert containment(partial, full) > 0.9
    assert containment(full, partial) < 0.7


def test_different_drawings_not_contained():
    a, b = _canvas(), _canvas()
    cv2.rectangle(a, (20, 20), (60, 70), 255, 2)
    cv2.circle(b, (120, 45), 25, 255, 2)
    assert containment(a, b) < 0.2
