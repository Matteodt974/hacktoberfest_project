"""Pass 1 — Gemma reads the full transcript and splits the lecture into sections."""
from __future__ import annotations

import json
import logging

from ..config import Config
from ..ingest.youtube import mmss, transcript_for_llm
from ..llm import prompts
from ..llm.client import GemmaClient
from ..llm.schemas import CourseOutline, SectionOutline

log = logging.getLogger(__name__)

MIN_SECTION_S = 30.0


def normalize_sections(sections: list[SectionOutline], duration: float) -> list[SectionOutline]:
    """Sort, clamp to [0, duration], merge sections < 30 s, make them contiguous and covering."""
    secs = sorted((s.model_copy() for s in sections), key=lambda s: s.start_s)
    for s in secs:
        s.start_s = min(max(0.0, float(s.start_s)), duration)
        s.end_s = min(max(0.0, float(s.end_s)), duration)
    secs = [s for s in secs if s.end_s > s.start_s or s is secs[-1]]
    if not secs:
        return [SectionOutline(title="Lecture", start_s=0, end_s=duration)]
    # contiguous: each section ends where the next starts; first at 0, last at duration
    secs[0].start_s = 0.0
    for a, b in zip(secs, secs[1:]):
        a.end_s = b.start_s
    secs[-1].end_s = duration
    # merge too-short sections into a neighbour (previous if any, else next)
    i = 0
    while i < len(secs) and len(secs) > 1:
        s = secs[i]
        if s.end_s - s.start_s < MIN_SECTION_S:
            if i > 0:
                prev = secs[i - 1]
                prev.end_s = s.end_s
                prev.key_concepts += [k for k in s.key_concepts if k not in prev.key_concepts]
            else:
                nxt = secs[1]
                nxt.start_s = s.start_s
                nxt.title = s.title if not nxt.title else nxt.title
                nxt.key_concepts = s.key_concepts + [k for k in nxt.key_concepts if k not in s.key_concepts]
            secs.pop(i)
            continue
        i += 1
    return secs


def make_outline(video_id: str, meta: dict, cues: list[dict], client: GemmaClient, cfg: Config) -> CourseOutline:
    vdir = cfg.video_dir(video_id)
    out = vdir / "outline.json"
    if out.exists() and not cfg.force:
        return CourseOutline.model_validate_json(out.read_text())

    duration = float(meta.get("duration") or (cues[-1]["end"] if cues else 0))
    n_min, n_max = (4, 10) if duration >= 240 else (2, 5)
    user = prompts.OUTLINE_USER.format(
        title=meta.get("title", ""), channel=meta.get("channel", ""), duration_mmss=mmss(duration),
        duration_s=duration, language_name=cfg.language_name, min_sections=n_min, max_sections=n_max,
        transcript=transcript_for_llm(cues),
    )
    outline = client.generate_json(
        task_name="pass1:outline", system=prompts.OUTLINE_SYSTEM, parts=[user], schema=CourseOutline,
        model=cfg.model, thinking=cfg.thinking_outline, cache_dir=vdir / "llm_cache",
    )
    outline.sections = normalize_sections(outline.sections, duration)
    if not (n_min <= len(outline.sections) <= n_max):
        log.warning("Outline has %d sections (target %d–%d)", len(outline.sections), n_min, n_max)
    out.write_text(json.dumps(outline.model_dump(), indent=2, ensure_ascii=False))
    log.info("Outline: %d sections — %s", len(outline.sections),
             " | ".join(f"{mmss(s.start_s)} {s.title}" for s in outline.sections))
    return outline
