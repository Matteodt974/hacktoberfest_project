"""Orchestrator: URL → ingest → frames → Pass 1 → Pass 2 → output.html, with progress callbacks."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Callable, Protocol

from ..config import Config
from ..ingest import frames as frames_mod
from ..ingest import youtube
from ..ingest.youtube import transcript_for_llm
from ..llm.client import GemmaClient
from ..render.html import render
from .notes import make_notes
from .outline import make_outline

log = logging.getLogger(__name__)

STEPS = ["download", "transcript", "frames", "outline", "notes", "ablation", "verify", "render"]
STEP_LABELS = {
    "download": "Download", "transcript": "Subtitles", "frames": "Frame extraction",
    "outline": "Structure (Gemma)", "notes": "Section notes (Gemma, multimodal)",
    "ablation": "Ablation (Gemma, same task without the images)",
    "verify": "Verification (Gemma, Chain-of-Verification)", "render": "Render",
}


class Progress(Protocol):
    def __call__(self, step: str, message: str = "", **extra) -> None: ...


def _noop(step: str, message: str = "", **extra) -> None:
    log.info("[%s] %s", step, message)


def save_stats(vdir: Path, client: GemmaClient | None, cfg: Config, n_frames: int | None, t0: float) -> None:
    # Cached responses replay their original cost, so these stats describe what produced the notes.
    # If outline/notes came straight from their JSON files, the client saw nothing: keep the previous stats.
    path = vdir / "stats.json"
    prev = json.loads(path.read_text()) if path.exists() else {}
    stats = {**prev, "model": prev.get("model", cfg.model), "lang": cfg.lang, "n_frames": n_frames}
    if client and (client.stats.calls or client.stats.cache_hits or not prev):
        stats.update(client.stats.as_dict(), model=cfg.model, wall_s=round(time.time() - t0, 1))
    path.write_text(json.dumps(stats, indent=2))


def run_verify(vdir: Path, notes: list[dict], context_for, client: GemmaClient, cfg: Config,
               progress: Progress) -> None:
    from .verify import verify_notes

    progress("verify", f"0 / {len(notes)}", done=0, total=len(notes))
    verify_notes(vdir, notes, context_for, client, cfg,
                 progress=lambda k, n: progress("verify", f"{k} / {n}", done=k, total=n))


def run_ablation(vdir: Path, outline, notes: list[dict], context_for, cfg: Config, progress: Progress,
                 timestamps: bool = True) -> None:
    from .ablation import make_ablation

    progress("ablation", f"0 / {len(notes)}", done=0, total=len(notes))
    make_ablation(vdir, outline, notes, context_for, cfg, timestamps=timestamps,
                  progress=lambda k, n: progress("ablation", f"{k} / {n}", done=k, total=n))


def run_video(url: str, cfg: Config, progress: Progress = _noop) -> Path:
    """Every step is cached on disk, so a rerun on an already-ingested video needs no YouTube access."""
    t0 = time.time()
    vid = youtube.parse_video_id(url)
    vdir = cfg.video_dir(vid)

    progress("download", "Reading video metadata…")
    meta = youtube.fetch_meta(vid, cfg)
    progress("download", f"Downloading “{meta['title']}”…")
    if not (vdir / "frames.json").exists() or cfg.force:
        youtube.download_video(vid, cfg)
    progress("transcript", "Fetching subtitles…")
    cues = youtube.fetch_transcript(vid, cfg)
    progress("frames", "Extracting stable frames…")
    frames = frames_mod.extract_frames(vid, cfg)
    progress("frames", f"{len(frames)} frames kept", frames=[f["file"] for f in frames], video_id=vid)
    return _llm_and_render(vid, vdir, meta, cues, frames, cfg, progress, t0)


def _llm_and_render(vid, vdir, meta, cues, frames, cfg: Config, progress: Progress, t0: float) -> Path:
    client = GemmaClient(concurrency=cfg.concurrency, max_retries=cfg.max_retries)
    progress("outline", "Gemma is reading the whole transcript…")
    outline = make_outline(vid, meta, cues, client, cfg)
    progress("outline", f"{len(outline.sections)} sections")
    progress("notes", f"0 / {len(outline.sections)}", done=0, total=len(outline.sections))
    notes = make_notes(vid, outline, cues, frames, client, cfg,
                       progress=lambda k, n: progress("notes", f"{k} / {n}", done=k, total=n))
    def section_transcript(r: dict) -> str:
        sec = outline.sections[r["index"]]
        return transcript_for_llm(cues, sec.start_s, sec.end_s)

    if cfg.ablation:
        run_ablation(vdir, outline, notes, section_transcript, cfg, progress)
    if cfg.verify:
        run_verify(vdir, notes, section_transcript, client, cfg, progress)
    save_stats(vdir, client, cfg, len(frames), t0)
    progress("render", "Rendering HTML…")
    out = render(vdir, cfg)
    progress("render", "Done", result=str(out))
    log.info("Done in %.0fs → %s  (Gemma: %s)", time.time() - t0, out, client.stats.as_dict())
    return out
