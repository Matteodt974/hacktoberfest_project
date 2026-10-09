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

## M2 — validated on the real demo video
- Matteo ran ingest + frames on his laptop (YouTube served **AV1** → fixed above): **27 frames kept** for the
  8-min video. Visual check of the contact sheet: complete diagrams, no black/blank frames; 2–3 near-duplicates
  (e.g. 02:06 vs 02:21) which Pass 2 absorbs by choosing one image per section.
- ⚠️ The demo URL's `v=LPZh9BOjkQs` is **"Large Language Models explained briefly"** (3Blue1Brown), not a
  linear-algebra episode (the `list=` param is ignored). It has few formulas → weak demo of "formulas read from
  images". Recommendation to Matteo: also ingest a math-heavy video (e.g. "The determinant", `Ip3X9LOh2dk`).
- Data moved here with `python -m lecturify bundle` (1.6 MB zip, no video).

## M3 — Pass 1 (outline)
- `pipeline/outline.py`: full `[mm:ss]` transcript in one call (2k tokens in); post-validation sorts, clamps,
  makes sections contiguous and covering [0, duration], merges sections < 30 s.
- Section target 4–10 (2–5 if video < 4 min).
- Real run: **5 coherent sections** (Next-word prediction / Training / RLHF+GPUs / Transformer / Emergence),
  85 s latency with `thinking_level="high"`.
- Network: one `httpx.ReadError: Connection reset by peer` on the first try → transport errors are now retried
  like 429/5xx.

## M4 — Pass 2 (multimodal notes)
- One Gemma call per section: header → `Image k — t=mm:ss` + inline JPEG (≤ 6, one per time slice, highest edge
  density) → section transcript → tasks + JSON shape. 3 sections in parallel.
- Real run on the demo video: Gemma picked a figure per section with a sensible reason, and **read content off the
  frames**: embedding vectors (`\begin{bmatrix} +1.0 \\ +4.3 …`) from 05:09, the arithmetic from the "1 billion
  computations per second" frame, `\vec{E}_1 … \vec{E}_8` from the series end card.
- v1 prompt over-transcribed (10 trivial additions, 12 near-identical vectors) and repeated the section title as `#`.
  v2 prompt: "skip decorative numbers, 1–2 representative items for repeated ones, ≤ 6 formulas, no title heading".
  Result: 0–3 meaningful formulas per section.
- Observed: one response missing `notes_markdown` and one malformed JSON → the single repair round fixed both.
- API instability is real (HTTP 500 bursts, `ReadError` connection resets, one call needed 4 attempts) →
  retries 7, backoff capped at 30 s. Latency per multimodal call: 17–100 s with `thinking_level="high"`.

## M5 — Render → CLI MVP
- `render/html.py`: Markdown with math protected by placeholders (unit-tested: `a_1 … b_2` no italics, backslashes
  kept, `<` escaped, `$5 … $10` not math), leading title heading stripped, frames inlined as base64 (2.5 MB file).
- Template: header, TOC, one card per section (figure + caption, formulas with source badges, notes, key points,
  "What Gemma saw" panel with candidates, chosen one highlighted + reason), slides mode (← → Esc), print CSS (one
  section per page, panel hidden), footer with model + stats.
- KaTeX: CDN (jsDelivr) in `output.html`; a vendored copy (`web/static/katex`, MIT, woff2 only, 612 KB, fetched from
  the npm registry) for the web UI. `render(..., katex_base=...)` switches between them.
- Stats: cached Gemma responses now store their cost (calls, latency, tokens) and replay it, so the footer shows what
  produced the notes even on a cached rerun (older cache entries count 1 call, no latency).
- Verified in headless Chromium: 7 formulas typeset, 0 KaTeX errors, figures and panel OK, slides OK.
- `python -m lecturify run <url>` works fully offline once a video is cached (every step skips on its JSON);
  rebuilding outline/notes from `llm_cache` reproduced `notes.json` byte for byte.

## M6 — Flask UI
- Routes per spec (`/`, `POST /jobs`, `GET /jobs/<id>`, `/jobs/<id>/view`, `/notes/<id>`) + `/jobs/images`,
  `/frames/<id>/<name>` (live thumbnail strip) and `/notes/<id>/download`. In-memory job registry, worker thread.
- `/notes/<id>` re-renders with the **local KaTeX copy** (`output_web.html`) → the demo works without CDN access.
- Bug found by driving the UI in Chromium: `url_for` called from the worker thread → "Working outside of application
  context" → every job failed. Fixed: the thread stores raw (item_id, file) pairs, URLs are built in the request.
- Verified in headless Chromium: submit URL → progress page → auto-redirect to the notes; KaTeX 7/0 errors;
  path traversal on `/notes/..%2F..` → 404. Readable error messages instead of stack traces.

## M7 — Screenshots mode, slides, print, "What Gemma saw"
- `pipeline/images_mode.py`: 1–20 images resized to 1024 px → **step A**, one multimodal call grouping images into
  sections (512 px when > 10 images) → **step B**, Pass 2 per group with the user's context as transcript. Cached under
  `data/images_<hash>/`. CLI `python -m lecturify images <files|dir> --title --context-file`, web tab "My screenshots".
- Bug found on the first real run: **Gemma returned 0-based `image_indices`** although images are labeled "Image 1…",
  which shifted every section title by one. Fix: prompt states "1-based, Image 1 is 1, last is k", and `_groups`
  detects a 0-based answer (min 0, max k-1) and shifts it. Unit-tested.
- Test deck: 4 dark math slides rendered with matplotlib (limit definition, power rule, chain rule, gradient).
  Gemma transcribed **all 6 formulas exactly** (`\lim_{h \to 0} \frac{f(x+h)-f(x)}{h}`, `\nabla f = (\partial f/\partial x, …)`),
  3 coherent sections, 4 calls / 206 s model latency. Browser upload path tested too (Playwright).
- Slides mode (one section per screen, ← → Esc) and print CSS (each section starts a page, panel and toolbar hidden)
  were built into the M5 template; verified via screenshots and a Chromium PDF (12 pages for 5 sections).
- Formulas are laid out side by side (flex) instead of one tall vector per row.
