"""Pass 3 (stretch) — Chain-of-Verification, factored variant.

Dhuliawala et al., "Chain-of-Verification Reduces Hallucination in Large Language Models" (2023), arXiv:2309.11495.
  1. draft      = Pass 2 notes;
  2. plan       = Gemma writes one verification question per formula and per key point;
  3. execute    = each question is answered in a SEPARATE call that sees only the source image + transcript,
                  never the draft (the paper's key idea: the model cannot copy its own mistakes);
  4. revise     = Gemma compares draft vs independent answers → verified / corrected / doubtful.
"""
from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from ..config import Config
from ..llm import prompts
from ..llm.client import GemmaClient, Image
from ..llm.schemas import VAnswer, VPlan, VRevision

log = logging.getLogger(__name__)

MAX_QUESTIONS = 8


def draft_text(notes: dict) -> tuple[str, list[str]]:
    lines, targets = [], []
    for k, f in enumerate(notes.get("formulas", []), 1):
        targets.append(f"formula:{k}")
        lines.append(f"[formula:{k}] LaTeX: {f['latex']}  — meaning: {f.get('meaning', '')}")
    for k, p in enumerate(notes.get("key_points", []), 1):
        targets.append(f"key_point:{k}")
        lines.append(f"[key_point:{k}] {p}")
    return "\n".join(lines), targets


def _images_for(target: str, notes: dict, cands: list[dict]) -> list[dict]:
    """Frames a target is checked against: a formula's source image; for a key point, all the section's frames."""
    if target.startswith("formula:"):
        k = int(target.split(":")[1]) - 1
        f = notes["formulas"][k] if k < len(notes["formulas"]) else {}
        idx = f.get("image_index")
        if idx and 1 <= idx <= len(cands):
            return [cands[idx - 1]]
    return list(cands)


def verify_section(r: dict, context: str, vdir: Path, client: GemmaClient, cfg: Config) -> dict:
    notes, cands, i = r["notes"], r["candidates"], r["index"]
    draft, targets = draft_text(notes)
    if not targets or r.get("failed"):
        return {"index": i, "items": []}
    cache = vdir / "llm_cache"

    # 2. plan
    plan = client.generate_json(task_name=f"cove:plan{i + 1}", system=prompts.VERIFY_SYSTEM,
                                parts=[prompts.VERIFY_PLAN.format(draft=draft)], schema=VPlan, model=cfg.model,
                                thinking="minimal", cache_dir=cache)
    questions = {q.target: q.question for q in plan.questions if q.target in targets}
    qlist = [(t, questions[t]) for t in targets if t in questions][:MAX_QUESTIONS]

    # 3. execute — factored: one independent call per question, no draft in the context
    def answer(tq: tuple[str, str]) -> tuple[str, str, VAnswer]:
        t, q = tq
        parts: list = []
        for k, frame in enumerate(_images_for(t, notes, cands), 1):
            parts += [f"IMAGE {k}:", Image.from_path(vdir / frame["file"])]
        parts.append(prompts.VERIFY_ANSWER.format(transcript=context or "(none)", question=q))
        try:
            a = client.generate_json(task_name=f"cove:answer{i + 1}:{t}", system=prompts.VERIFY_SYSTEM, parts=parts,
                                     schema=VAnswer, model=cfg.model, thinking="minimal", cache_dir=cache)
        except Exception as e:  # noqa: BLE001 — one unanswerable question must not sink the section
            log.warning("CoVe answer failed for section %d %s: %s", i + 1, t, str(e)[:150])
            a = VAnswer(found=False, answer="")
        return t, q, a

    with ThreadPoolExecutor(max_workers=cfg.concurrency) as ex:
        answers = list(ex.map(answer, qlist))

    # 4. revise
    qa = "\n".join(f"[{t}] Q: {q}\n    A: {'NOT FOUND' if not a.found else a.answer}"
                   f"{'  LaTeX: ' + a.latex if a.latex else ''}" for t, q, a in answers)
    rev = client.generate_json(
        task_name=f"cove:revise{i + 1}", system=prompts.VERIFY_SYSTEM,
        parts=[prompts.VERIFY_REVISE.format(draft=draft, qa=qa or "(no answers)", targets=", ".join(targets))],
        schema=VRevision, model=cfg.model, thinking="high", cache_dir=cache)
    by_target = {it.target: it for it in rev.items}
    qa_by_target = {t: (q, a) for t, q, a in answers}
    items = []
    for t in targets:
        it = by_target.get(t)
        q, a = qa_by_target.get(t, ("", None))
        status = it.status if it else "doubtful"
        correction = (it.correction or "").strip() if it and status == "corrected" else None
        if status == "corrected" and not correction:
            status = "doubtful"  # a correction without content is not a correction
        items.append({"target": t, "status": status, "correction": correction, "note": it.note if it else "",
                      "question": q, "answer": (a.answer if a and a.found else "not found") if a else ""})
    return {"index": i, "items": items}


def verify_notes(vdir: Path, notes: list[dict], context_for: Callable[[dict], str], client: GemmaClient,
                 cfg: Config, progress: Callable[[int, int], None] | None = None) -> dict:
    out = vdir / "verification.json"
    if out.exists() and not cfg.force:
        return json.loads(out.read_text())
    n, done = len(notes), 0

    def one(r: dict) -> dict:
        nonlocal done
        try:
            res = verify_section(r, context_for(r), vdir, client, cfg)
        except Exception as e:  # noqa: BLE001 — verification is optional, never fail the run for it
            log.error("CoVe failed for section %d: %s", r["index"] + 1, e)
            res = {"index": r["index"], "items": [], "error": str(e)[:200]}
        done += 1
        if progress:
            progress(done, n)
        return res

    # sections in parallel; the client's semaphore bounds the number of simultaneous Gemma calls
    with ThreadPoolExecutor(max_workers=2) as ex:
        sections = list(ex.map(one, notes))
    counts = {s: sum(it["status"] == s for sec in sections for it in sec["items"])
              for s in ("verified", "corrected", "doubtful")}
    result = {"method": "Chain-of-Verification (factored), arXiv:2309.11495", "counts": counts, "sections": sections}
    if not any(sec.get("error") for sec in sections):  # don't cache a partial verification
        out.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    log.info("CoVe: %s", counts)
    return result
