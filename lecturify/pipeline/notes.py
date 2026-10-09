"""Pass 2 — multimodal: per section, Gemma LOOKS at candidate frames, picks the best figure,
transcribes the formulas shown on them into LaTeX, and writes the notes."""
from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from ..config import Config
from ..ingest.youtube import mmss, transcript_for_llm
from ..llm import prompts
from ..llm.client import GemmaClient, Image
from ..llm.schemas import CourseOutline, SectionNotes, SectionOutline

log = logging.getLogger(__name__)


def select_candidates(frames: list[dict], start: float, end: float, k: int) -> list[dict]:
    """Frames inside [start, end]; if more than k, one per equal time slice (highest edge density);
    if none, the frame closest in time."""
    inside = [f for f in frames if start <= f["t"] < end] or [f for f in frames if start <= f["t"] <= end]
    if not inside:
        if not frames:
            return []
        mid = (start + end) / 2
        return [min(frames, key=lambda f: abs(f["t"] - mid))]
    if len(inside) <= k:
        return inside
    width = (end - start) / k
    picked: list[dict] = []
    for b in range(k):
        lo, hi = start + b * width, start + (b + 1) * width
        bucket = [f for f in inside if lo <= f["t"] < hi or (b == k - 1 and f["t"] == hi)]
        if bucket:
            picked.append(max(bucket, key=lambda f: f["edge_density"]))
    # empty slices: top up with the richest remaining frames
    rest = sorted((f for f in inside if f not in picked), key=lambda f: -f["edge_density"])
    picked += rest[: k - len(picked)]
    return sorted(picked, key=lambda f: f["t"])


def _guard(notes: SectionNotes, cands: list[dict]) -> SectionNotes:
    k = len(cands)
    if not 1 <= notes.best_image <= k:
        best = max(range(k), key=lambda i: cands[i]["edge_density"]) + 1 if k else 1
        log.warning("best_image=%s out of range 1..%d → using %d", notes.best_image, k, best)
        notes.best_image = best
    for f in notes.formulas:
        if f.image_index is not None and not 1 <= f.image_index <= k:
            f.image_index = None
        if f.source == "transcript":
            f.image_index = None
    return notes


def section_notes(i: int, sec: SectionOutline, outline: CourseOutline, cands: list[dict], context: str,
                  vdir: Path, client: GemmaClient, cfg: Config, timestamps: bool = True) -> SectionNotes:
    n = len(outline.sections)
    header = prompts.NOTES_HEADER.format(
        course_title=outline.course_title, i=i + 1, n=n, section_title=sec.title,
        start_mmss=mmss(sec.start_s) if timestamps else "", end_mmss=mmss(sec.end_s) if timestamps else "",
        language_name=cfg.language_name, k=len(cands))
    parts: list = [header]
    for j, f in enumerate(cands, 1):
        label = f"Image {j} — t={mmss(f['t'])}" if timestamps else f"Image {j}"
        parts += [label, Image.from_path(vdir / f["file"])]
    parts.append(prompts.NOTES_TASKS.format(transcript=context or "(no transcript)"))
    notes = client.generate_json(
        task_name=f"pass2:section{i + 1}", system=prompts.NOTES_SYSTEM, parts=parts, schema=SectionNotes,
        model=cfg.model, thinking=cfg.thinking_notes, cache_dir=vdir / "llm_cache")
    return _guard(notes, cands)


def make_notes(video_id: str, outline: CourseOutline, cues: list[dict], frames: list[dict], client: GemmaClient,
               cfg: Config, progress: Callable[[int, int], None] | None = None) -> list[dict]:
    vdir = cfg.video_dir(video_id)
    out = vdir / "notes.json"
    if out.exists() and not cfg.force:
        return json.loads(out.read_text())
    n = len(outline.sections)
    done = 0

    def work(i: int) -> dict:
        nonlocal done
        sec = outline.sections[i]
        cands = select_candidates(frames, sec.start_s, sec.end_s, cfg.max_images_per_section)
        notes = section_notes(i, sec, outline, cands, transcript_for_llm(cues, sec.start_s, sec.end_s),
                              vdir, client, cfg)
        done += 1
        if progress:
            progress(done, n)
        log.info("Section %d/%d: chose image %d/%d, %d formulas (%d from images)", i + 1, n, notes.best_image,
                 len(cands), len(notes.formulas), sum(f.source == "image" for f in notes.formulas))
        return {"index": i, "candidates": cands, "notes": notes.model_dump()}

    with ThreadPoolExecutor(max_workers=cfg.concurrency) as ex:
        results = list(ex.map(work, range(n)))
    out.write_text(json.dumps(results, indent=1, ensure_ascii=False))
    return results
