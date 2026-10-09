"""'My screenshots' mode: 1–20 ordered images (+ optional context) → same notes format, no timestamps."""
from __future__ import annotations

import hashlib
import io
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image as PILImage

from ..config import Config
from ..llm import prompts
from ..llm.client import GemmaClient, Image
from ..llm.schemas import CourseOutline, ImagesOutline, SectionOutline
from ..render.html import render
from .notes import section_notes
from .run import Progress, _noop, save_stats

log = logging.getLogger(__name__)

MAX_IMAGES = 20


def _to_jpeg(data: bytes, width: int) -> bytes:
    img = PILImage.open(io.BytesIO(data))
    img = img.convert("RGB")
    if img.width > width:
        img = img.resize((width, round(img.height * width / img.width)), PILImage.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return buf.getvalue()


def _groups(outline: ImagesOutline, k: int) -> list[list[int]]:
    """Clean Gemma's grouping: valid 1-based indices, each image once, every image covered, sorted."""
    all_idx = [i for sec in outline.sections for i in sec.image_indices]
    shift = 1 if all_idx and min(all_idx) == 0 and max(all_idx) <= k - 1 else 0  # model answered 0-based
    seen: set[int] = set()
    groups: list[list[int]] = []
    for sec in outline.sections:
        g = sorted({i + shift for i in sec.image_indices if 1 <= i + shift <= k and i + shift not in seen})
        seen.update(g)
        groups.append(g)
    missing = [i for i in range(1, k + 1) if i not in seen]
    for i in missing:  # attach orphans to the section holding the previous image
        target = next((g for g in reversed(groups) if g and g[0] < i), groups[0] if groups else None)
        if target is None:
            groups.append([i])
        else:
            target.append(i)
            target.sort()
    return groups


def run_images(blobs: list[tuple[str, bytes]], cfg: Config, *, title: str = "", context: str = "",
               progress: Progress = _noop) -> Path:
    t0 = time.time()
    if not 1 <= len(blobs) <= MAX_IMAGES:
        raise ValueError(f"Upload between 1 and {MAX_IMAGES} images.")
    progress("frames", f"Preparing {len(blobs)} images…")
    jpegs = [_to_jpeg(b, cfg.frame_width) for _, b in blobs]
    h = hashlib.sha256(b"".join(hashlib.sha256(j).digest() for j in jpegs) + (title + "\0" + context).encode())
    item_id = f"images_{h.hexdigest()[:12]}"
    vdir = cfg.video_dir(item_id)
    fdir = vdir / "frames"
    fdir.mkdir(exist_ok=True)
    frames = []
    for i, j in enumerate(jpegs, 1):
        (fdir / f"img_{i:02d}.jpg").write_bytes(j)
        # "t" holds the image number; edge_density is unused here but kept for the shared candidate code
        frames.append({"file": f"frames/img_{i:02d}.jpg", "t": float(i), "edge_density": 0.0})
    (vdir / "frames.json").write_text(json.dumps(frames, indent=1))
    (vdir / "meta.json").write_text(json.dumps({
        "video_id": item_id, "title": title or "Screenshots", "channel": "", "duration": 0, "url": "",
        "images_mode": True}, indent=2))
    progress("frames", f"{len(frames)} images", frames=[f["file"] for f in frames], video_id=item_id)

    client = GemmaClient(concurrency=cfg.concurrency, max_retries=cfg.max_retries)
    k = len(jpegs)
    # Step A — one multimodal call groups the images into sections (smaller images when there are many)
    progress("outline", "Gemma is looking at all the images…")
    width = 1024 if k <= 10 else 512
    parts: list = []
    for i, j in enumerate(jpegs, 1):
        parts += [f"Image {i}", Image(_to_jpeg(j, width) if width < cfg.frame_width else j)]
    parts.append(prompts.IMAGES_OUTLINE_USER.format(
        title_line=f"Title given by the user: {title}\n" if title else "", language_name=cfg.language_name, k=k,
        context_line=f"The user also gave this context:\n---\n{context[:6000]}\n---\n" if context else "",
        max_sections=min(8, k)))
    io_outline = client.generate_json(task_name="images:outline", system=prompts.OUTLINE_SYSTEM, parts=parts,
                                      schema=ImagesOutline, model=cfg.model, thinking=cfg.thinking_outline,
                                      cache_dir=vdir / "llm_cache")
    groups = _groups(io_outline, k)
    sections, kept = [], []
    for sec, g in zip(io_outline.sections, groups):
        if g:
            sections.append(SectionOutline(title=sec.title, start_s=g[0], end_s=g[-1], summary=sec.summary,
                                           key_concepts=sec.key_concepts))
            kept.append(g)
    outline = CourseOutline(course_title=title or io_outline.course_title, summary=io_outline.summary,
                            prerequisites=io_outline.prerequisites, sections=sections)
    (vdir / "outline.json").write_text(json.dumps(outline.model_dump(), indent=2, ensure_ascii=False))

    # Step B — Pass 2 per section, with the group's images as candidates and the user's context as "transcript"
    n = len(sections)
    progress("notes", f"0 / {n}", done=0, total=n)
    done = 0

    def work(i: int) -> dict:
        nonlocal done
        cands = [frames[j - 1] for j in kept[i]][: cfg.max_images_per_section]
        notes = section_notes(i, sections[i], outline, cands, context[:6000], vdir, client, cfg, timestamps=False)
        done += 1
        progress("notes", f"{done} / {n}", done=done, total=n)
        return {"index": i, "candidates": cands, "notes": notes.model_dump()}

    with ThreadPoolExecutor(max_workers=cfg.concurrency) as ex:
        results = list(ex.map(work, range(n)))
    (vdir / "notes.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
    save_stats(vdir, client, cfg, k, t0)
    progress("render", "Rendering HTML…")
    out = render(vdir, cfg)
    progress("render", "Done", result=str(out))
    return out
