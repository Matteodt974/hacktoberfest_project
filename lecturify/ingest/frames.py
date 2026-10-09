"""Frame sampling: keep the END of each animation (stable, complete drawing), deduplicated.

3Blue1Brown-style videos animate continuously on dark backgrounds, so classic scene-cut
detection and brightness heuristics fail. We instead:
  1. sample every `sample_every_s` seconds (grayscale thumbnail);
  2. motion[i] = mean |thumb_i - thumb_{i-1}|; a frame is stable when motion[i] and motion[i+1]
     are both under `motion_threshold`;
  3. content score = Canny edge density; reject near-empty frames;
  4. group consecutive stable frames with pHash distance <= 6, keep the LAST of each group;
  5. global dedup (pHash <= 4) keeping the highest edge density;
  6. cap at `max_frames` (auto-tightening), fall back to uniform sampling if < `min_frames`.
"""
from __future__ import annotations

import html
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image as PILImage

from ..config import Config
from .youtube import mmss

log = logging.getLogger(__name__)


@dataclass
class Sample:
    t: float
    motion: float
    edge_density: float
    phash: str
    idx: int = 0  # frame index in the video, to re-read the full-res frame later
    hash: imagehash.ImageHash | None = None
    small: np.ndarray | None = None  # grayscale thumbnail (~160 px), cheap to keep


def _edge_density(gray_small: np.ndarray) -> float:
    edges = cv2.Canny(gray_small, 60, 160)
    return float(np.count_nonzero(edges)) / edges.size


def _phash(gray_small: np.ndarray) -> imagehash.ImageHash:
    return imagehash.phash(PILImage.fromarray(gray_small))


def sample_video(video: Path, cfg: Config) -> list[Sample]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = n_frames / fps if n_frames else 0
    step = max(1, int(round(fps * cfg.sample_every_s)))

    samples: list[Sample] = []
    prev_small: np.ndarray | None = None
    idx = 0
    # Sequential grab() is far faster and more reliable than seeking with CAP_PROP_POS_FRAMES.
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 or idx / fps
            h, w = frame.shape[:2]
            small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (cfg.thumb_width, max(1, int(h * cfg.thumb_width / w))),
                               interpolation=cv2.INTER_AREA)
            motion = float(np.mean(cv2.absdiff(small, prev_small))) if prev_small is not None else 255.0
            samples.append(Sample(t=round(t, 2), motion=motion, edge_density=_edge_density(small), phash="",
                                  idx=idx, hash=_phash(small), small=small))
            prev_small = small
        idx += 1
    cap.release()
    log.info("Sampled %d frames (%.0fs video, every %.1fs)", len(samples), duration, cfg.sample_every_s)
    return samples


def _motion_report(samples: list[Sample]) -> str:
    m = np.array([s.motion for s in samples[1:]]) if len(samples) > 1 else np.array([0.0])
    q = np.percentile(m, [10, 25, 50, 75, 90])
    return "motion percentiles p10=%.2f p25=%.2f p50=%.2f p75=%.2f p90=%.2f" % tuple(q)


def select_frames(samples: list[Sample], cfg: Config, motion_threshold: float, min_edge: float) -> list[Sample]:
    n = len(samples)
    stable = []
    for i in range(1, n):
        nxt = samples[i + 1].motion if i + 1 < n else 0.0  # last frame: treat as settled
        if samples[i].motion < motion_threshold and nxt < motion_threshold and samples[i].edge_density >= min_edge:
            stable.append(samples[i])

    # Group consecutive stable frames that look alike; keep the LAST (most complete drawing).
    groups: list[list[tuple[Sample, imagehash.ImageHash]]] = []
    for s in stable:
        h = s.hash
        if groups:
            last_s, last_h = groups[-1][-1]
            contiguous = s.t - last_s.t <= cfg.sample_every_s * 1.5 + 0.01
            if contiguous and (h - last_h) <= cfg.group_phash_dist:
                groups[-1].append((s, h))
                continue
        groups.append([(s, h)])
    reps = [g[-1] for g in groups]

    # Global dedup: identical-looking frames anywhere in the video -> keep the richest one.
    kept: list[tuple[Sample, imagehash.ImageHash]] = []
    for s, h in reps:
        dup = next((k for k, (ks, kh) in enumerate(kept) if (h - kh) <= cfg.global_phash_dist), None)
        if dup is None:
            kept.append((s, h))
        elif s.edge_density > kept[dup][0].edge_density:
            kept[dup] = (s, h)
    out = []
    for s, h in sorted(kept, key=lambda x: x[0].t):
        s.phash = str(h)
        out.append(s)
    return drop_partial_drawings(out, cfg)


def _edges(small: np.ndarray) -> np.ndarray:
    return cv2.Canny(small, 60, 160) > 0


def containment(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of A's edge pixels also present (within 1 px) in B."""
    ea = _edges(a)
    if not ea.any():
        return 1.0
    eb = cv2.dilate(_edges(b).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    return float(np.count_nonzero(ea & eb)) / float(np.count_nonzero(ea))


def drop_partial_drawings(frames: list[Sample], cfg: Config) -> list[Sample]:
    """Drop A when a later frame B (within `subsume_window_s`) contains A's drawing and adds to it.

    Animations add elements progressively, so a half-drawn frame is "contained" in the finished one.
    """
    keep = []
    for i, a in enumerate(frames):
        subsumed = False
        for b in frames[i + 1:]:
            if b.t - a.t > cfg.subsume_window_s:
                break
            if b.edge_density >= a.edge_density and containment(a.small, b.small) >= cfg.subsume_containment:
                subsumed = True
                break
        if not subsumed:
            keep.append(a)
    return keep


def _read_frames(video: Path, wanted: set[int]) -> dict[int, np.ndarray]:
    """Second sequential pass: fetch only the full-resolution frames we keep."""
    cap = cv2.VideoCapture(str(video))
    out: dict[int, np.ndarray] = {}
    idx, last = 0, max(wanted) if wanted else -1
    while idx <= last and cap.grab():
        if idx in wanted:
            ok, frame = cap.retrieve()
            if ok:
                out[idx] = frame
        idx += 1
    cap.release()
    return out


def _uniform(samples: list[Sample], cfg: Config) -> list[Sample]:
    out, next_t = [], 0.0
    for s in samples:
        if s.t >= next_t and s.edge_density > 0:
            s.phash = str(s.hash)
            out.append(s)
            next_t = s.t + cfg.fallback_every_s
    return out


def extract_frames(video_id: str, cfg: Config) -> list[dict]:
    vdir = cfg.video_dir(video_id)
    out_json = vdir / "frames.json"
    if out_json.exists() and not cfg.force:
        return json.loads(out_json.read_text())
    video = vdir / "video.mp4"
    samples = sample_video(video, cfg)
    if not samples:
        raise RuntimeError("No frames could be read from the video.")
    log.info(_motion_report(samples))

    thr, min_edge = cfg.motion_threshold, cfg.min_edge_density
    chosen = select_frames(samples, cfg, thr, min_edge)
    tries = 0
    while len(chosen) > cfg.max_frames and tries < 6:  # too many: tighten
        thr *= 0.7
        min_edge *= 1.3
        chosen = select_frames(samples, cfg, thr, min_edge)
        tries += 1
    if len(chosen) > cfg.max_frames:
        chosen = sorted(chosen, key=lambda s: -s.edge_density)[: cfg.max_frames]
        chosen.sort(key=lambda s: s.t)
    if len(chosen) < cfg.min_frames:
        log.warning("Only %d stable frames; falling back to uniform sampling every %.0fs", len(chosen), cfg.fallback_every_s)
        chosen = _uniform(samples, cfg)
    log.info("Kept %d frames (motion_threshold=%.2f, min_edge_density=%.4f)", len(chosen), thr, min_edge)

    fdir = vdir / "frames"
    fdir.mkdir(exist_ok=True)
    for old in fdir.glob("f_*.jpg"):
        old.unlink()
    records = []
    full = _read_frames(video, {s.idx for s in chosen})
    for s in chosen:
        frame = full[s.idx]
        h, w = frame.shape[:2]
        img = frame if w <= cfg.frame_width else cv2.resize(
            frame, (cfg.frame_width, int(h * cfg.frame_width / w)), interpolation=cv2.INTER_AREA)
        name = f"f_{int(s.t):06d}.jpg"
        if (fdir / name).exists():  # two frames in the same second
            name = f"f_{int(s.t):06d}_{int((s.t % 1) * 100):02d}.jpg"
        cv2.imwrite(str(fdir / name), img, [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality])
        records.append({"file": f"frames/{name}", "t": s.t, "motion": round(s.motion, 3),
                        "edge_density": round(s.edge_density, 4), "phash": s.phash})
    out_json.write_text(json.dumps(records, indent=1))
    return records


def contact_sheet(video_id: str, cfg: Config) -> Path:
    """HTML contact sheet of every kept frame, for visual calibration."""
    vdir = cfg.video_dir(video_id)
    frames = json.loads((vdir / "frames.json").read_text())
    cells = "\n".join(
        f'<figure><img src="{html.escape(f["file"])}" loading="lazy"><figcaption><b>{mmss(f["t"])}</b> '
        f'· motion {f["motion"]:.2f} · edges {f["edge_density"]:.3f}</figcaption></figure>'
        for f in frames)
    page = f"""<!doctype html><meta charset="utf-8"><title>Frames · {html.escape(video_id)}</title>
<style>body{{font:14px system-ui;margin:16px;background:#f6f6f4}}h1{{font-size:18px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px}}
figure{{margin:0;background:#fff;padding:6px;border:1px solid #ddd}}img{{width:100%;display:block}}
figcaption{{font-size:12px;color:#444;margin-top:4px}}</style>
<h1>{len(frames)} frames kept — {html.escape(video_id)}</h1><div class="grid">{cells}</div>"""
    out = vdir / "contact_sheet.html"
    out.write_text(page)
    return out


def frame_records_as_dicts(samples: list[Sample]) -> list[dict]:  # debugging helper
    return [{k: v for k, v in asdict(s).items() if k != "hash"} for s in samples]
