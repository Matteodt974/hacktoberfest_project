"""Render notes.json + outline.json into one self-contained output.html (images inlined as base64)."""
from __future__ import annotations

import base64
import html
import json
import re
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from markdown_it import MarkdownIt
from markupsafe import Markup

from ..config import Config
from ..ingest.youtube import mmss

TEMPLATES = Path(__file__).parent / "templates"
REPO_URL = "https://github.com/Matteodt974/hacktoberfest_project"
KATEX_CDN = "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist"

# Display math first ($$..$$, \[..\]), then inline ($..$, \(..\)). A "$" preceded by a backslash is literal.
_MATH_RE = re.compile(
    r"(\$\$.+?\$\$|\\\[.+?\\\]|\\\(.+?\\\)|(?<![\\$])\$(?!\s)[^\n$]+?(?<![\s\\])\$(?!\d))",
    re.DOTALL,
)
_md = MarkdownIt("commonmark", {"html": False, "linkify": False, "typographer": False}).enable("table")


def markdown_with_math(text: str) -> str:
    """Markdown → HTML without letting the Markdown parser touch the LaTeX.

    Each math span is swapped for an inert placeholder, Markdown is rendered, then the raw LaTeX is
    restored (HTML-escaped) so KaTeX auto-render can typeset it in the browser.
    """
    spans: list[str] = []

    def stash(m: re.Match) -> str:
        spans.append(m.group(0))
        return f"MATHPLACEHOLDER{len(spans) - 1}X"

    protected = _MATH_RE.sub(stash, text or "")
    out = _md.render(protected)
    return re.sub(r"MATHPLACEHOLDER(\d+)X", lambda m: html.escape(spans[int(m.group(1))], quote=False), out)


def strip_leading_title(md: str, title: str) -> str:
    """Drop a first-line heading that just repeats the section title."""
    lines = (md or "").lstrip().split("\n")
    if lines and lines[0].startswith("#"):
        heading = lines[0].lstrip("#").strip().lower()
        if heading and (heading in title.lower() or title.lower() in heading or len(heading.split()) <= 8):
            return "\n".join(lines[1:]).lstrip()
    return md


_CHECK = {"verified": ("✓", "verified"), "corrected": ("✎", "corrected"), "doubtful": ("⚠", "doubtful")}


def _check(item: dict | None) -> dict:
    """Display info for one CoVe result (empty when verification was not run)."""
    if not item:
        return {"status": None, "icon": "", "label": "", "tip": "", "correction": None}
    icon, label = _CHECK.get(item["status"], ("⚠", "doubtful"))
    if item["status"] == "doubtful" and item.get("answer") == "not found":
        label = "not in source"  # often true general knowledge, but the video/slides don't state it
    tip = f"Q: {item.get('question', '')}\nIndependent answer: {item.get('answer', '')}\n{item.get('note', '')}"
    return {"status": item["status"], "icon": icon, "label": label, "tip": tip.strip(),
            "correction": item.get("correction") if item["status"] == "corrected" else None}


def _data_uri(path: Path, cache: dict[Path, str]) -> str:
    if path not in cache:
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        cache[path] = f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()
    return cache[path]


def build_context(vdir: Path, cfg: Config) -> dict:
    meta = json.loads((vdir / "meta.json").read_text())
    outline = json.loads((vdir / "outline.json").read_text())
    notes = {r["index"]: r for r in json.loads((vdir / "notes.json").read_text())}
    stats = json.loads((vdir / "stats.json").read_text()) if (vdir / "stats.json").exists() else {}
    verification = json.loads((vdir / "verification.json").read_text()) if (vdir / "verification.json").exists() else None
    vmap = {(sec["index"], it["target"]): it for sec in (verification or {}).get("sections", []) for it in sec["items"]}
    timestamps = not meta.get("images_mode", False)
    uris: dict[Path, str] = {}
    sections = []
    n_img_formulas = 0
    for i, sec in enumerate(outline["sections"]):
        r = notes.get(i)
        if not r:
            continue
        nt = r["notes"]
        cands = []
        for j, f in enumerate(r["candidates"], 1):
            cands.append({"n": j, "t": mmss(f["t"]) if timestamps else "", "src": _data_uri(vdir / f["file"], uris),
                          "chosen": j == nt["best_image"]})
        best = cands[nt["best_image"] - 1] if cands else None
        formulas = []
        for fk, f in enumerate(nt["formulas"], 1):
            check = _check(vmap.get((i, f"formula:{fk}")))
            from_img = f["source"] == "image"
            n_img_formulas += from_img
            badge = (f"read from image {f['image_index']}" if from_img and f.get("image_index")
                     else "read from an image" if from_img else "from transcript")
            formulas.append({"latex": check["correction"] or f["latex"], "original": f["latex"],
                             "meaning": f.get("meaning", ""), "from_image": from_img, "badge": badge, "check": check})
        sections.append({
            "n": i + 1, "id": f"s{i + 1}", "title": sec["title"],
            "start": mmss(sec["start_s"]) if timestamps else "", "end": mmss(sec["end_s"]) if timestamps else "",
            "yt": f"https://www.youtube.com/watch?v={meta['video_id']}&t={int(sec['start_s'])}s" if timestamps else "",
            "summary": sec.get("summary", ""),
            "figure": best, "caption": nt.get("figure_caption", ""), "reason": nt.get("image_choice_reason", ""),
            "formulas": formulas,
            "notes_html": Markup(markdown_with_math(strip_leading_title(nt["notes_markdown"], sec["title"]))),
            "key_points": [
                {"html": Markup(markdown_with_math(c["correction"] or k).strip().removeprefix("<p>").removesuffix("</p>")),
                 "check": c, "original": k}
                for kk, k in enumerate(nt.get("key_points", []), 1)
                for c in [_check(vmap.get((i, f"key_point:{kk}")))]],
            "candidates": cands,
        })
    return {
        "meta": meta, "outline": outline, "sections": sections, "timestamps": timestamps,
        "model": stats.get("model", cfg.model), "stats": stats, "repo_url": REPO_URL,
        "n_frames": stats.get("n_frames"), "n_img_formulas": n_img_formulas,
        "n_seen": sum(len(s["candidates"]) for s in sections),
        "verification": verification,
        "lang": stats.get("lang", cfg.lang),
    }


def render(vdir: Path, cfg: Config, katex_base: str = KATEX_CDN, out_name: str = "output.html") -> Path:
    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape(["html", "j2"]))
    page = env.get_template("notes.html.j2").render(katex_base=katex_base, **build_context(vdir, cfg))
    out = vdir / out_name
    out.write_text(page, encoding="utf-8")
    return out
