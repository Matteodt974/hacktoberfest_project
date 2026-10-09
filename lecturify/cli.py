"""Command-line interface: python -m lecturify <command> ..."""
from __future__ import annotations

import argparse
import logging
import sys

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

    args = ap.parse_args(argv)
    cfg = Config(model=args.model, lang=getattr(args, "lang", "en"))

    if args.cmd == "smoke-test":
        from .smoke import run_smoke
        res = run_smoke(cfg)
        sys.exit(0 if all(v.get("ok", True) for v in res.values() if isinstance(v, dict) and "ok" in v) else 1)
