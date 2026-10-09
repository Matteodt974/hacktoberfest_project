"""Flask UI: submit a YouTube URL (or screenshots), watch progress, browse the notes library."""
from __future__ import annotations

import json
import logging
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from flask import Flask, abort, jsonify, redirect, render_template, request, send_file, url_for

from ..config import DEFAULT_MODEL, SUPPORTED_MODELS, Config
from ..pipeline.run import STEP_LABELS, STEPS

log = logging.getLogger(__name__)

STATIC = Path(__file__).parent / "static"


@dataclass
class Job:
    id: str
    kind: str  # "video" | "images"
    status: str = "running"  # running | done | error
    step: str = "download"
    message: str = "Starting…"
    progress: dict = field(default_factory=dict)
    error: str | None = None
    result_id: str | None = None
    frames_preview: list[tuple[str, str]] = field(default_factory=list)  # (item_id, file name)
    steps_done: list[str] = field(default_factory=list)
    verify: bool = False

    def as_dict(self) -> dict:
        return {
            "status": self.status, "step": self.step, "message": self.message, "progress": self.progress,
            "error": self.error, "steps_done": self.steps_done,
            "frames_preview": [url_for("frame", item_id=i, name=n) for i, n in self.frames_preview],
            "result_url": url_for("notes", item_id=self.result_id) if self.result_id and self.status == "done" else None,
        }


JOBS: dict[str, Job] = {}


def _friendly(e: Exception) -> str:
    msg = str(e).strip() or type(e).__name__
    low = msg.lower()
    if "gemini_api_key" in low:
        return "No Gemini API key: put GEMINI_API_KEY=... in the .env file and restart."
    if "sign in to confirm" in low or "403" in low and "youtube" in low:
        return "YouTube refused the download from this network. Try again, or use the “My screenshots” mode."
    return msg.splitlines()[0][:400]


def library(cfg: Config) -> list[dict]:
    items = []
    if not cfg.data_dir.exists():
        return items
    for d in cfg.data_dir.iterdir():
        out, meta_p, outline_p = d / "output.html", d / "meta.json", d / "outline.json"
        if not (d.is_dir() and out.exists() and meta_p.exists()):
            continue
        meta = json.loads(meta_p.read_text())
        outline = json.loads(outline_p.read_text()) if outline_p.exists() else {}
        items.append({
            "id": d.name, "title": outline.get("course_title") or meta.get("title", d.name),
            "source": meta.get("title", ""), "channel": meta.get("channel", ""),
            "sections": len(outline.get("sections", [])), "mtime": out.stat().st_mtime,
            "images_mode": meta.get("images_mode", False),
        })
    return sorted(items, key=lambda x: -x["mtime"])


def create_app(cfg: Config | None = None) -> Flask:
    base_cfg = cfg or Config()
    app = Flask(__name__, static_folder=str(STATIC), template_folder=str(Path(__file__).parent / "templates"))
    app.config["MAX_CONTENT_LENGTH"] = 80 * 1024 * 1024

    def job_cfg(form) -> Config:
        model = form.get("model", DEFAULT_MODEL)
        lang = form.get("lang", "en")
        return Config(model=model if model in SUPPORTED_MODELS else DEFAULT_MODEL,
                      lang=lang if lang in ("en", "fr") else "en", data_dir=base_cfg.data_dir,
                      verify=bool(form.get("verify")))

    def start(job: Job, target) -> None:
        def progress(step: str, message: str = "", **extra) -> None:
            if STEPS.index(step) >= STEPS.index(job.step):  # steps only move forward
                job.steps_done = STEPS[: STEPS.index(step)]
                job.step = step
            job.message = message
            if "frames" in extra and "video_id" in extra:
                # no url_for here: this runs in the worker thread, outside the Flask app context
                job.frames_preview = [(extra["video_id"], Path(f).name) for f in extra["frames"]]
            if "done" in extra:
                job.progress = {"done": extra["done"], "total": extra["total"]}
            if "result" in extra:
                job.result_id = Path(extra["result"]).parent.name

        def runner():
            try:
                target(progress)
                job.steps_done = list(STEPS)
                job.status = "done"
            except Exception as e:  # noqa: BLE001 — shown to the user, logged in full
                log.error("Job %s failed: %s\n%s", job.id, e, traceback.format_exc())
                job.status, job.error = "error", _friendly(e)

        threading.Thread(target=runner, daemon=True).start()

    @app.get("/")
    def index():
        return render_template("index.html", items=library(base_cfg), models=SUPPORTED_MODELS)

    @app.post("/jobs")
    def create_job():
        from ..pipeline.run import run_video

        url = (request.form.get("url") or "").strip()
        if not url:
            return render_template("index.html", items=library(base_cfg), models=SUPPORTED_MODELS,
                                   error="Paste a YouTube URL."), 400
        jcfg = job_cfg(request.form)
        jcfg.force = bool(request.form.get("force"))
        job = Job(id=uuid.uuid4().hex[:10], kind="video", verify=jcfg.verify)
        JOBS[job.id] = job
        start(job, lambda p: run_video(url, jcfg, p))
        return redirect(url_for("job_view", job_id=job.id))

    @app.post("/jobs/images")
    def create_images_job():
        from ..pipeline.images_mode import run_images

        files = [f for f in request.files.getlist("images") if f and f.filename]
        if not 1 <= len(files) <= 20:
            return render_template("index.html", items=library(base_cfg), models=SUPPORTED_MODELS,
                                   error="Upload between 1 and 20 images.", tab="images"), 400
        blobs = [(f.filename, f.read()) for f in files]
        jcfg = job_cfg(request.form)
        title, context = request.form.get("title", "").strip(), request.form.get("context", "").strip()
        job = Job(id=uuid.uuid4().hex[:10], kind="images", step="frames", verify=jcfg.verify)
        JOBS[job.id] = job
        start(job, lambda p: run_images(blobs, jcfg, title=title, context=context, progress=p))
        return redirect(url_for("job_view", job_id=job.id))

    @app.get("/jobs/<job_id>")
    def job_status(job_id: str):
        job = JOBS.get(job_id) or abort(404)
        return jsonify(job.as_dict())

    @app.get("/jobs/<job_id>/view")
    def job_view(job_id: str):
        job = JOBS.get(job_id) or abort(404)
        steps = [(s, STEP_LABELS[s]) for s in STEPS
                 if not (job.kind == "images" and s in ("download", "transcript")) and (s != "verify" or job.verify)]
        return render_template("progress.html", job=job, steps=steps)

    @app.get("/notes/<item_id>")
    def notes(item_id: str):
        d = (base_cfg.data_dir / item_id).resolve()
        if d.parent != base_cfg.data_dir.resolve() or not (d / "output.html").exists():
            abort(404)
        # Re-render with the local KaTeX copy so the demo works without the CDN.
        from ..render.html import render
        render(d, Config(data_dir=base_cfg.data_dir), katex_base=url_for("static", filename="katex").rstrip("/"),
               out_name="output_web.html")
        return send_file(d / "output_web.html")

    @app.get("/frames/<item_id>/<name>")
    def frame(item_id: str, name: str):
        p = (base_cfg.data_dir / item_id / "frames" / name).resolve()
        if p.parent.parent.parent != base_cfg.data_dir.resolve() or not p.exists():
            abort(404)
        return send_file(p)

    @app.get("/notes/<item_id>/download")
    def download(item_id: str):
        d = (base_cfg.data_dir / item_id).resolve()
        if d.parent != base_cfg.data_dir.resolve() or not (d / "output.html").exists():
            abort(404)
        return send_file(d / "output.html", as_attachment=True, download_name=f"lecturify-{item_id}.html")

    return app
