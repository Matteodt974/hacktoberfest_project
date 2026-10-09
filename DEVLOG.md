# DEVLOG — Lecturify

Decisions, discoveries and pitfalls, in chronological order.

## 2026-10-09 — M0 (skeleton + smoke test)

### Environment
- Development runs in a cloud container (Linux, Python 3.13.16, ffmpeg present), venv in `.venv/`.
- Installed: google-genai 2.29.0, youtube-transcript-api 1.2.4 (v1.x API: `YouTubeTranscriptApi().fetch(...)`),
  yt-dlp 2026.8.19, opencv-python-headless 5.0.0.93, pydantic 2.14, Pillow 12.3, ImageHash 4.3.2.
- Network: `generativelanguage.googleapis.com` reachable; **`www.youtube.com` is blocked by the container's egress policy**
  (CONNECT 403). Ingestion (M1) cannot be run here as-is.

### Decisions
- Renamed `.env.exemple` → `.env.example` (spec name).
- LaTeX-in-JSON defense in 2 layers (`lecturify/llm/jsonfix.py`):
  1. **Pre-parse** repair: single backslashes that start a LaTeX command (`\frac`, `\theta`, `\beta`, `\rho`,
     known `\n…` commands like `\nabla`/`\neq`, invalid escapes like `\vec`) are doubled before `json.loads`.
     A plain `\n` followed by prose (`\nThe`) stays a newline.
  2. **Post-parse** sanitizer: control chars → backslash sequences (`latex` fields also repair `\n`).
  - 22 offline unit tests pass.
- LLM cache key = sha256(model, system, thinking, schema name + JSON schema, text parts, image sha256s).
