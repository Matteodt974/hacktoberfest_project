"""Command-line interface: python -m lecturify <command> ..."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import DEFAULT_MODEL, SUPPORTED_MODELS, Config


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default=DEFAULT_MODEL, choices=SUPPORTED_MODELS)
    p.add_argument("--lang", default="en", choices=["en", "fr"])


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "google_genai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    ap = argparse.ArgumentParser(prog="lecturify", description="Turn lecture videos into illustrated course notes with Gemma 4.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("smoke-test", help="check Gemma 4 access (text, image, JSON, multi-image)")
    _common(p)

    p = sub.add_parser("run", help="full pipeline: YouTube URL (or cached video id) → output.html")
    p.add_argument("url")
    _common(p)
    p.add_argument("--max-images", type=int, default=6)
    p.add_argument("--force", action="store_true", help="recompute every step (LLM calls stay cached)")
    p.add_argument("--verify", action="store_true", help="Pass 3: Chain-of-Verification of formulas and key points")

    p = sub.add_parser("render", help="re-render data/<video_id>/output.html offline from cached JSON")
    p.add_argument("video_id")

    p = sub.add_parser("images", help="notes from 1-20 screenshots (files or a directory), no YouTube needed")
    p.add_argument("paths", nargs="+", type=Path)
    p.add_argument("--title", default="")
    p.add_argument("--context-file", type=Path)
    p.add_argument("--verify", action="store_true", help="Pass 3: Chain-of-Verification")
    _common(p)

    p = sub.add_parser("web", help="start the web UI")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--host", default="127.0.0.1")

    p = sub.add_parser("ingest", help="download metadata, transcript and video (needs YouTube access)")
    p.add_argument("url")
    _common(p)
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("import-local", help="import a local video + subtitles (.vtt/.json) without YouTube")
    p.add_argument("video_id")
    p.add_argument("video", type=Path)
    p.add_argument("subs", type=Path)
    p.add_argument("--title")
    p.add_argument("--channel", default="")

    p = sub.add_parser("frames", help="extract stable frames from data/<video_id>/video.mp4")
    p.add_argument("video_id")
    p.add_argument("--force", action="store_true")
    p.add_argument("--motion-threshold", type=float)

    p = sub.add_parser("frames-debug", help="extract frames (if needed) and write an HTML contact sheet")
    p.add_argument("video_id")
    p.add_argument("--force", action="store_true")
    p.add_argument("--motion-threshold", type=float)

    p = sub.add_parser("bundle", help="zip data/<video_id>/ without the video (to move a cache between machines)")
    p.add_argument("video_id")

    args = ap.parse_args(argv)
    cfg = Config(model=getattr(args, "model", DEFAULT_MODEL), lang=getattr(args, "lang", "en"),
                 force=getattr(args, "force", False), verify=getattr(args, "verify", False))
    if getattr(args, "motion_threshold", None):
        cfg.motion_threshold = args.motion_threshold

    try:
        _dispatch(args, cfg)
    except Exception as e:  # readable errors instead of stack traces
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            raise
        logging.getLogger("lecturify").error("%s: %s", type(e).__name__, e)
        sys.exit(1)


def _dispatch(args, cfg: Config) -> None:

    if args.cmd == "smoke-test":
        from .smoke import run_smoke
        res = run_smoke(cfg)
        sys.exit(0 if all(v.get("ok", True) for v in res.values() if isinstance(v, dict) and "ok" in v) else 1)

    elif args.cmd == "run":
        from .pipeline.run import run_video
        cfg.max_images_per_section = args.max_images
        print(f"Notes: {run_video(args.url, cfg)}")

    elif args.cmd == "render":
        from .render.html import render
        print(f"Notes: {render(cfg.video_dir(args.video_id), cfg)}")

    elif args.cmd == "images":
        from .pipeline.images_mode import run_images
        files: list[Path] = []
        for p in args.paths:
            files += sorted(x for x in p.iterdir() if x.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")) \
                if p.is_dir() else [p]
        context = args.context_file.read_text() if args.context_file else ""
        out = run_images([(f.name, f.read_bytes()) for f in files], cfg, title=args.title, context=context)
        print(f"Notes: {out}")

    elif args.cmd == "web":
        from .web.app import create_app
        print(f"Lecturify running on http://{args.host}:{args.port}")
        create_app(cfg).run(host=args.host, port=args.port, debug=False, threaded=True)

    elif args.cmd == "ingest":
        from .ingest.youtube import ingest
        vid, meta, cues, video = ingest(args.url, cfg)
        print(f"{vid}: '{meta['title']}' ({meta['duration']:.0f}s), {len(cues)} transcript cues, video at {video}")

    elif args.cmd == "import-local":
        from .ingest.youtube import import_local
        import_local(args.video_id, args.video, args.subs, cfg, title=args.title, channel=args.channel)
        print(f"Imported into {cfg.video_dir(args.video_id)}")

    elif args.cmd in ("frames", "frames-debug"):
        from .ingest.frames import contact_sheet, extract_frames
        frames = extract_frames(args.video_id, cfg)
        print(f"{len(frames)} frames kept")
        if args.cmd == "frames-debug":
            print(f"Contact sheet: {contact_sheet(args.video_id, cfg)}")

    elif args.cmd == "bundle":
        import zipfile
        vdir = cfg.video_dir(args.video_id)
        out = cfg.data_dir / f"{args.video_id}_bundle.zip"
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(vdir.rglob("*")):
                if f.is_file() and f.suffix not in (".mp4", ".webm", ".mkv") and not f.name.startswith("subs."):
                    z.write(f, f.relative_to(cfg.data_dir))
        print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
