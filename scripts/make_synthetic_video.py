"""Generate a 3Blue1Brown-like synthetic test video (dark bg, continuous animations, holds).

Used to test frame sampling offline. Ground truth: each scene ends with a hold on a complete
drawing; the sampler should keep ~1 frame per scene, near the end of each hold.
Usage: python scripts/make_synthetic_video.py out.mp4
"""
import math
import sys

import cv2
import numpy as np

W, H, FPS = 1280, 720, 24
BG = (28, 20, 20)
YEL, BLU, WHT = (60, 200, 255), (255, 200, 120), (235, 235, 235)


def put(img, text, org, scale=2.0, color=WHT, alpha=1.0):
    c = tuple(int(b + (x - b) * alpha) for x, b in zip(color, BG))
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, c, 3, cv2.LINE_AA)


def scene_circle(p, img):
    cv2.ellipse(img, (640, 380), (220, 220), 0, 0, 360 * min(1, p * 1.2), YEL, 6, cv2.LINE_AA)
    if p > 0.7:
        put(img, "A = pi r^2", (450, 120), alpha=min(1, (p - 0.7) / 0.3))


def scene_axes(p, img):
    cv2.line(img, (140, 600), (140 + int(1000 * min(1, p * 2)), 600), WHT, 3, cv2.LINE_AA)
    cv2.line(img, (140, 600), (140, 600 - int(500 * min(1, p * 2))), WHT, 3, cv2.LINE_AA)
    if p > 0.5:
        q = (p - 0.5) / 0.5
        pts = [(140 + int(x), 600 - int(0.0005 * x * x)) for x in np.linspace(0, 1000 * q, 200)]
        cv2.polylines(img, [np.array(pts, np.int32)], False, BLU, 5, cv2.LINE_AA)
    put(img, "f(x) = x^2", (800, 140), alpha=min(1, p * 1.5))


def scene_vector(p, img):
    for k in range(-6, 7):
        cv2.line(img, (640 + k * 90, 80), (640 + k * 90, 680), (70, 60, 60), 1)
        cv2.line(img, (100, 380 + k * 90), (1180, 380 + k * 90), (70, 60, 60), 1)
    tip = (640 + int(270 * min(1, p * 1.3)), 380 - int(180 * min(1, p * 1.3)))
    cv2.arrowedLine(img, (640, 380), tip, YEL, 6, cv2.LINE_AA, tipLength=0.08)
    if p > 0.6:
        put(img, "v = 3i + 2j", (760, 160), alpha=min(1, (p - 0.6) / 0.4))


def scene_matrix(p, img):
    put(img, "[ 1  -1 ]", (420, 330), 2.4, alpha=min(1, p * 2))
    put(img, "[ 2   1 ]", (420, 420), 2.4, alpha=min(1, p * 2))
    if p > 0.5:
        put(img, "det = 3", (520, 560), 2.0, YEL, alpha=min(1, (p - 0.5) * 2))


def scene_wave(p, img):
    pts = [(int(x), int(380 + 150 * math.sin(x / 80 + p * 12))) for x in range(80, 1200, 4)]
    cv2.polylines(img, [np.array(pts, np.int32)], False, BLU, 5, cv2.LINE_AA)  # always moving


def scene_sum(p, img):
    n = int(12 * min(1, p * 1.2))
    for i in range(n):
        cv2.rectangle(img, (150 + i * 80, 600 - (i + 1) * 35), (210 + i * 80, 600), YEL, -1)
    if p > 0.7:
        put(img, "sum k = n(n+1)/2", (330, 140), alpha=min(1, (p - 0.7) / 0.3))


def title(p, img):
    put(img, "Chapter 1: Synthetic", (300, 380), 2.2, alpha=min(1, p * 3))


# (scene fn, animate seconds, hold seconds)
SCENES = [(title, 2, 2), (scene_circle, 6, 4), (scene_axes, 7, 4), (scene_wave, 8, 0),
          (scene_vector, 6, 5), (None, 0, 2), (scene_matrix, 5, 4), (scene_sum, 7, 4)]


def main(out):
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    t, truth = 0.0, []
    for fn, anim, hold in SCENES:
        total = anim + hold
        for f in range(int(total * FPS)):
            img = np.zeros((H, W, 3), np.uint8)
            if fn is not None:
                img[:] = BG
                p = min(1.0, (f / FPS) / anim) if anim else 1.0
                fn(p, img)
            vw.write(img)
        if fn is not None and hold:
            truth.append((fn.__name__, round(t + anim, 1), round(t + total, 1)))
        t += total
    vw.release()
    print(f"wrote {out} ({t:.0f}s). Expected holds (scene, start, end):")
    for row in truth:
        print("  ", row)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "synthetic.mp4")
