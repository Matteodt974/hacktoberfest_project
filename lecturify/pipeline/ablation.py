"""Ablation ("eyes closed"): the same Pass 2 task, same model and settings, but WITHOUT any image.

Comparing it with the real (multimodal) notes measures what Gemma 4 gains by looking at the frames: formulas that
only exist on screen, the chosen figure, etc. Only the text-only side is stored (ablation.json); the comparison is
computed at render time against the current notes.json, so it never goes stale.
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from ..config import Config
from ..ingest.youtube import mmss
from ..llm import prompts
from ..llm.client import GemmaClient
from ..llm.schemas import CourseOutline, SectionOutline, TextOnlyNotes

log = logging.getLogger(__name__)

_DROP = re.compile(r"\\(?:displaystyle|left|right|quad|qquad|[,;:!])")
_SINGLE_BRACED = re.compile(r"\{(\\?[A-Za-z0-9])\}")


def norm_latex(s: str) -> str:
    """Strict normalisation for matching formulas across the two runs (loose matching would inflate the result)."""
    s = (s or "").strip().strip("$").strip()
    s = re.sub(r"\\[dt]frac", r"\\frac", s)
    s = _DROP.sub("", s)
    s = re.sub(r"\s+", "", s)
    prev = None
    while prev != s:  # {x} -> x, repeatedly ( ^{{2}} -> ^2 )
        prev, s = s, _SINGLE_BRACED.sub(r"\1", s)
    return s.rstrip(".,;")


def _words(md: str) -> int:
    return len(re.findall(r"[A-Za-zÀ-ÿ]+", md or ""))


def compare(with_nt: dict, without_nt: dict, has_figure: bool = True) -> dict:
    """Counts for one section. Formulas are matched one-to-one on their normalised LaTeX."""
    pool = Counter(norm_latex(f["latex"]) for f in without_nt.get("formulas", []))
    only_img = only_tr = 0
    for f in with_nt.get("formulas", []):
        key = norm_latex(f["latex"])
        if pool[key] > 0:
            pool[key] -= 1
        elif f.get("source") == "image":
            only_img += 1
        else:
            only_tr += 1
    return {
        "formulas_with": len(with_nt.get("formulas", [])),
        "from_images": sum(f.get("source") == "image" for f in with_nt.get("formulas", [])),
        "formulas_without": len(without_nt.get("formulas", [])),
        "only_with_image": only_img,
        "only_with_transcript": only_tr,  # same source on both sides: sampling noise, not vision
        "only_without": sum(pool.values()),
        "words_with": _words(with_nt.get("notes_markdown", "")),
        "words_without": _words(without_nt.get("notes_markdown", "")),
        "figure": bool(has_figure),
    }


def ablation_section(i: int, sec: SectionOutline, outline: CourseOutline, context: str, vdir: Path,
                     client: GemmaClient, cfg: Config, timestamps: bool = True) -> TextOnlyNotes:
    header = prompts.ABLATION_HEADER.format(
        course_title=outline.course_title, i=i + 1, n=len(outline.sections), section_title=sec.title,
        start_mmss=mmss(sec.start_s) if timestamps else "", end_mmss=mmss(sec.end_s) if timestamps else "",
        language_name=cfg.language_name)
    if not timestamps:
        header = header.replace(" (–)", "")
    parts = [header, prompts.ABLATION_TASKS.format(transcript=context or "(no transcript)")]  # text only, no Image
    notes = client.generate_json(task_name=f"ablation:section{i + 1}", system=prompts.ABLATION_SYSTEM, parts=parts,
                                 schema=TextOnlyNotes, model=cfg.model, thinking=cfg.thinking_notes,
                                 cache_dir=vdir / "llm_cache")
    for f in notes.formulas:  # it saw no image: a claimed "image" source is impossible
        f.source, f.image_index = "transcript", None
    return notes


def make_ablation(vdir: Path, outline: CourseOutline, notes: list[dict], context_for: Callable[[dict], str],
                  cfg: Config, progress: Callable[[int, int], None] | None = None, timestamps: bool = True,
                  client: GemmaClient | None = None) -> dict:
    out = vdir / "ablation.json"
    if out.exists() and not cfg.force:
        return json.loads(out.read_text())
    # A separate client: its cost is reported in ablation.json and never overwrites the notes' stats.json.
    client = client or GemmaClient(concurrency=cfg.concurrency, max_retries=cfg.max_retries)
    n, done = len(notes), 0

    def one(r: dict) -> dict:
        nonlocal done
        i = r["index"]
        res: dict = {"index": i}
        try:
            ctx = context_for(r)
            if r.get("failed"):
                res["skipped"] = "pass2 failed"
            elif not (ctx or "").strip():
                res["skipped"] = "no text context"  # screenshots mode without context: nothing to compare with
            else:
                res["notes"] = ablation_section(i, outline.sections[i], outline, ctx, vdir, client, cfg,
                                                timestamps).model_dump()
        except Exception as e:  # noqa: BLE001 — the ablation is optional, never fail the run for it
            log.error("Ablation failed for section %d: %s", i + 1, e)
            res["error"] = str(e)[:200]
        done += 1
        if progress:
            progress(done, n)
        return res

    with ThreadPoolExecutor(max_workers=cfg.concurrency) as ex:
        sections = list(ex.map(one, notes))
    result = {"model": cfg.model, "thinking": cfg.thinking_notes, "cost": client.stats.as_dict(),
              "sections": sections}
    if not any("error" in s for s in sections):  # don't cache a partial ablation; per-call cache makes reruns cheap
        out.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    log.info("Ablation: %d/%d sections done (%s)", sum("notes" in s for s in sections), n, client.stats.as_dict())
    return result
