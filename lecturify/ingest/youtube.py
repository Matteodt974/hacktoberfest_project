"""YouTube ingestion: URL parsing, metadata, video download, transcript."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ..config import Config

log = logging.getLogger(__name__)

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


class IngestError(RuntimeError):
    pass


def parse_video_id(url: str) -> str:
    """Extract the 11-char video id from watch?v=, youtu.be/, shorts/, embed/, live/ URLs (or a bare id)."""
    url = url.strip()
    if _ID_RE.match(url):
        return url
    u = urlparse(url if "://" in url else "https://" + url)
    host = (u.hostname or "").lower().removeprefix("www.").removeprefix("m.")
    if host == "youtu.be":
        cand = u.path.lstrip("/").split("/")[0]
    elif host.endswith("youtube.com") or host.endswith("youtube-nocookie.com"):
        qs = parse_qs(u.query)
        if "v" in qs:
            cand = qs["v"][0]
        else:
            parts = [p for p in u.path.split("/") if p]
            cand = parts[1] if len(parts) >= 2 and parts[0] in ("shorts", "embed", "live", "v") else ""
    else:
        cand = ""
    if not _ID_RE.match(cand):
        raise IngestError(f"Could not find a YouTube video id in: {url}")
    return cand


def mmss(seconds: float) -> str:
    s = int(max(0, seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


# ------------------------------------------------------------------ metadata + video
def fetch_meta(video_id: str, cfg: Config) -> dict:
    out = cfg.video_dir(video_id) / "meta.json"
    if out.exists() and not cfg.force:
        return json.loads(out.read_text())
    import yt_dlp

    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True}) as ydl:
        try:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
        except Exception as e:  # yt-dlp raises many types
            raise IngestError(f"Could not read video metadata: {e}") from e
    meta = {
        "video_id": video_id,
        "title": info.get("title") or video_id,
        "channel": info.get("channel") or info.get("uploader") or "",
        "duration": float(info.get("duration") or 0),
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "thumbnail": info.get("thumbnail"),
    }
    check_duration(meta["duration"], cfg)
    out.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return meta


def check_duration(duration: float, cfg: Config) -> None:
    if duration > cfg.max_duration_s:
        raise IngestError(f"Video is {mmss(duration)} long; the limit is {mmss(cfg.max_duration_s)}.")
    if duration > cfg.warn_duration_s:
        log.warning("Long video (%s): generation will take a while.", mmss(duration))


def download_video(video_id: str, cfg: Config) -> Path:
    vdir = cfg.video_dir(video_id)
    out = vdir / "video.mp4"
    if out.exists() and not cfg.force:
        return out
    import yt_dlp

    opts = {
        "quiet": True,
        "no_warnings": True,
        # video only is enough (no audio sent to Gemma); <=720p; prefer mp4
        # Prefer H.264: OpenCV wheels often cannot decode AV1 (frames.ensure_decodable transcodes as a fallback)
        "format": "bv*[height<=720][vcodec^=avc1]/bv*[height<=720][ext=mp4]/bv*[height<=720]/b[height<=720]/b",
        "outtmpl": str(vdir / "video.%(ext)s"),
        "merge_output_format": "mp4",
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        try:
            ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
        except Exception as e:
            raise IngestError(f"Video download failed: {e}") from e
    if not out.exists():
        cands = sorted(vdir.glob("video.*"))
        if not cands:
            raise IngestError("Video download produced no file.")
        cands[0].rename(out)  # OpenCV reads webm too; keep a stable name
    return out


# ------------------------------------------------------------------ transcript
def _langs(lang: str) -> list[str]:
    seen: list[str] = []
    for l in (lang, "en", "fr"):
        if l not in seen:
            seen.append(l)
    return seen


def _from_transcript_api(video_id: str, langs: list[str]) -> list[dict]:
    from youtube_transcript_api import YouTubeTranscriptApi

    api = YouTubeTranscriptApi()
    if hasattr(api, "fetch"):  # v1.x
        fetched = api.fetch(video_id, languages=langs)
        return [{"start": s.start, "end": s.start + s.duration, "text": s.text} for s in fetched]
    rows = YouTubeTranscriptApi.get_transcript(video_id, languages=langs)  # pragma: no cover (old API)
    return [{"start": r["start"], "end": r["start"] + r["duration"], "text": r["text"]} for r in rows]


_VTT_TIME = re.compile(r"(\d+):(\d{2}):(\d{2})\.(\d{3})|(\d{2}):(\d{2})\.(\d{3})")


def _vtt_ts(s: str) -> float:
    m = _VTT_TIME.search(s)
    if not m:
        raise ValueError(s)
    if m.group(1):
        h, mi, se, ms = (int(x) for x in m.group(1, 2, 3, 4))
    else:
        h, (mi, se, ms) = 0, (int(x) for x in m.group(5, 6, 7))
    return h * 3600 + mi * 60 + se + ms / 1000


def parse_vtt(text: str) -> list[dict]:
    """Parse WebVTT (incl. YouTube auto-captions, which repeat rolling lines) into deduplicated cues."""
    cues: list[dict] = []
    blocks = re.split(r"\n\s*\n", text.replace("\r\n", "\n"))
    for b in blocks:
        lines = [l for l in b.strip().split("\n") if l.strip()]
        idx = next((i for i, l in enumerate(lines) if "-->" in l), None)
        if idx is None:
            continue
        a, _, z = lines[idx].partition("-->")
        try:
            start, end = _vtt_ts(a), _vtt_ts(z)
        except ValueError:
            continue
        body = " ".join(re.sub(r"<[^>]+>", "", l).strip() for l in lines[idx + 1:])
        body = re.sub(r"\s+", " ", body).strip()
        if body:
            cues.append({"start": start, "end": end, "text": body})
    # Dedupe rolling auto-captions: drop text already emitted at the end of the previous output.
    out: list[dict] = []
    prev_words: list[str] = []
    for c in cues:
        words = c["text"].split()
        k = 0
        for n in range(min(len(prev_words), len(words)), 0, -1):
            if prev_words[-n:] == words[:n]:
                k = n
                break
        new = words[k:]
        if not new:
            continue
        out.append({"start": c["start"], "end": c["end"], "text": " ".join(new)})
        prev_words = (prev_words + new)[-60:]
    return out


def _from_ytdlp_subs(video_id: str, langs: list[str], vdir: Path) -> list[dict]:
    import yt_dlp

    opts = {
        "quiet": True, "no_warnings": True, "skip_download": True,
        "writesubtitles": True, "writeautomaticsub": True,
        "subtitleslangs": langs, "subtitlesformat": "vtt",
        "outtmpl": str(vdir / "subs.%(ext)s"),
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
    for l in langs:  # manual subs and auto subs share the naming scheme; prefer requested order
        for f in sorted(vdir.glob(f"subs.{l}*.vtt")):
            cues = parse_vtt(f.read_text(encoding="utf-8", errors="replace"))
            if cues:
                return cues
    return []


def fetch_transcript(video_id: str, cfg: Config) -> list[dict]:
    vdir = cfg.video_dir(video_id)
    out = vdir / "transcript.json"
    if out.exists() and not cfg.force:
        return json.loads(out.read_text())
    langs = _langs(cfg.lang)
    cues: list[dict] = []
    try:
        cues = _from_transcript_api(video_id, langs)
        log.info("Transcript via youtube-transcript-api: %d cues", len(cues))
    except Exception as e:  # noqa: BLE001
        log.warning("youtube-transcript-api failed (%s); trying yt-dlp subtitles", str(e).splitlines()[0][:200])
        try:
            cues = _from_ytdlp_subs(video_id, langs, vdir)
            log.info("Transcript via yt-dlp subtitles: %d cues", len(cues))
        except Exception as e2:  # noqa: BLE001
            log.warning("yt-dlp subtitles failed: %s", str(e2)[:200])
    if not cues:
        raise IngestError("This video has no usable subtitles. Try the 'My screenshots' mode instead.")
    cues = [{"start": round(float(c["start"]), 2), "end": round(float(c["end"]), 2),
             "text": re.sub(r"\s+", " ", c["text"]).strip()} for c in cues if c["text"].strip()]
    out.write_text(json.dumps(cues, indent=1, ensure_ascii=False))
    return cues


def transcript_for_llm(cues: list[dict], start: float = 0.0, end: float = float("inf"), window_s: float = 12.0) -> str:
    """Group cues into ~window_s chunks, each line prefixed with [mm:ss]."""
    lines: list[str] = []
    cur_t: float | None = None
    buf: list[str] = []
    for c in cues:
        if c["end"] < start or c["start"] > end:
            continue
        if cur_t is None:
            cur_t = c["start"]
        elif c["start"] - cur_t >= window_s:
            lines.append(f"[{mmss(cur_t)}] {' '.join(buf)}")
            cur_t, buf = c["start"], []
        buf.append(c["text"])
    if buf and cur_t is not None:
        lines.append(f"[{mmss(cur_t)}] {' '.join(buf)}")
    return "\n".join(lines)


def ingest(url: str, cfg: Config) -> tuple[str, dict, list[dict], Path]:
    vid = parse_video_id(url)
    meta = fetch_meta(vid, cfg)
    cues = fetch_transcript(vid, cfg)
    video = download_video(vid, cfg)
    return vid, meta, cues, video


def import_local(video_id: str, video_path: Path, subs_path: Path, cfg: Config, *,
                 title: str | None = None, channel: str = "", duration: float | None = None) -> None:
    """Offline import (no YouTube access): copy a local video + .vtt/.json transcript into data/<id>/."""
    import shutil

    import cv2

    vdir = cfg.video_dir(video_id)
    if video_path.resolve() != (vdir / "video.mp4").resolve():
        shutil.copy(video_path, vdir / "video.mp4")
    if subs_path.suffix == ".json":
        cues = json.loads(subs_path.read_text())
    else:
        cues = parse_vtt(subs_path.read_text(encoding="utf-8", errors="replace"))
    (vdir / "transcript.json").write_text(json.dumps(cues, indent=1, ensure_ascii=False))
    if duration is None:
        cap = cv2.VideoCapture(str(vdir / "video.mp4"))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        cap.release()
    meta_path = vdir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update({"video_id": video_id, "title": title or meta.get("title") or video_id,
                 "channel": channel or meta.get("channel", ""), "duration": float(duration),
                 "url": f"https://www.youtube.com/watch?v={video_id}"})
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
