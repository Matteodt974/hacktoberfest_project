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

### M0 smoke test results (`python -m lecturify smoke-test`, model `gemma-4-26b-a4b-it`)
| Test | Result |
|---|---|
| (a) text call | OK, 1.9 s |
| thinking_level="high" | OK (17×23 = 391), 4.6 s |
| system_instruction | OK (answered in French as instructed) |
| (b) **inline image** (`Part.from_bytes`), PIL formula on dark background | OK — read `E = mc^2` **and** `\nabla \cdot E = \rho / \varepsilon_0` correctly. No upload needed. |
| (c) JSON mode `response_mime_type="application/json"` | **Supported** (clean JSON) |
| `response_schema=<pydantic>` | Flaky: 500 then OK on retry; one output contained an HTML-entity artifact (`&amp;ine;`). |
| (d) 6 images / 10 images in one request | **Both OK**, every image identified in order (4.0 s / 4.7 s) |
| (e) burst: 8 calls, 4 parallel | All OK in 4.9 s wall; 2 transient HTTP 500 absorbed by retries |

**Decisions from M0**
- Images: **inline** `types.Part.from_bytes` (no Files API).
- JSON: `response_mime_type="application/json"` + our robust parser; **no `response_schema`** (flaky, artifacts).
- The API returns **intermittent HTTP 500** (~1 in 5 calls during the test) → the 5-retry exponential backoff is
  essential, and 500 is treated as retryable.
- Concurrency: default 2 kept for safety; 4 parallel worked fine (no 429 seen) → raised default to 3.
- Up to 10 images/request works; we keep `max_images_per_section = 6` (spec) for latency and token cost.
- YouTube (yt-dlp and youtube-transcript-api) still blocked from the container (proxy 403) → ingestion must run on
  Matteo's laptop; offline import path planned for M1.

## M1 — Ingestion (code done, real run pending on Matteo's laptop)
- `lecturify/ingest/youtube.py`: URL parsing (watch/youtu.be/shorts/embed/live/m./bare id), yt-dlp metadata
  (`meta.json`, 20 min warning / 45 min refusal), video-only ≤720p download, transcript via
  youtube-transcript-api v1.x (`YouTubeTranscriptApi().fetch`) with yt-dlp VTT fallback (manual then auto subs,
  rolling-caption dedup by longest word overlap), `transcript_for_llm` grouping ~12 s lines prefixed `[mm:ss]`.
- **Blocker**: the cloud container cannot reach YouTube (egress proxy 403 on youtube.com/googlevideo/ytimg).
  Decision: ingestion + frame extraction run on Matteo's laptop; `python -m lecturify bundle <id>` zips
  `data/<id>/` without the video so the cache can be moved here; `import-local` imports a local mp4 + .vtt.
- Demo video: https://www.youtube.com/watch?v=LPZh9BOjkQs (3Blue1Brown, Essence of linear algebra playlist).

## M2 — Frame sampling (algorithm validated on a synthetic video; real contact sheet pending)
- Two passes over the video: pass 1 computes grayscale 160 px thumbnails, motion, Canny density and pHash for
  every 1 s sample (low memory); pass 2 re-reads only the kept full-res frames.
- pHash computed on the grayscale thumbnail (pHash downsizes to 32×32 anyway).
- **Added rule (not in spec): partial-drawing subsumption.** Slow animations (motion < threshold) produced
  several half-drawn "stable" frames. A frame A is dropped if a later frame B within 20 s has ≥ edge density and
  contains ≥ 85 % of A's (Canny) edge pixels. That is exactly "keep the most complete drawing".
- Removed an "auto-loosen when few frames" loop I had added: it pushed the threshold to 6.75 and made a
  continuously-moving wave count as stable. Spec behavior kept: tighten above 150, uniform fallback below 5.
- Synthetic test (`scripts/make_synthetic_video.py`, 66 s, 3B1B-like dark background, 6 animate-then-hold scenes,
  1 always-moving wave, 1 black screen): **6 frames kept, exactly one per hold, all complete drawings**;
  wave and black screen rejected.
- **Bug found on Matteo's laptop:** `No frames could be read from the video`. Root cause reproduced here: YouTube
  serves many 720p video-only streams as **AV1**; OpenCV wheels open the file but cannot decode a single frame.
  Fix: (1) yt-dlp format now prefers H.264 (`vcodec^=avc1`); (2) `frames.ensure_decodable` transcodes to H.264
  (`video_h264.mp4`) with the static ffmpeg shipped by `imageio-ffmpeg` (new dependency, no system ffmpeg needed).
  Verified: the synthetic video re-encoded to AV1 → same 6 frames. `bundle` now skips every video file.
