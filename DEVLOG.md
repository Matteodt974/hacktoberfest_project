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
