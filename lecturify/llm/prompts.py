"""All prompt templates sent to Gemma 4."""
from __future__ import annotations

OUTLINE_SYSTEM = (
    "You are an expert university teaching assistant who turns lecture videos into structured course notes. "
    "You are precise and you never invent content."
)

OUTLINE_USER = """Video title: {title}
Channel: {channel}
Duration: {duration_mmss} ({duration_s:.0f} seconds)
Write everything in: {language_name}

Below is the full timestamped transcript of the video. Split the lecture into {min_sections}–{max_sections} coherent
sections that follow its teaching progression (one concept or reasoning step per section). Sections must be
contiguous, ordered, cover the whole video (first starts at 0, last ends at {duration_s:.0f}), and use the transcript
timestamps for their boundaries. start_s and end_s are numbers of SECONDS (e.g. 04:12 -> 252).

Return ONLY a JSON object with exactly this shape:
{{
  "course_title": string,
  "summary": string,               // 3-4 sentences
  "prerequisites": [string],
  "sections": [
    {{"title": string, "start_s": number, "end_s": number,
     "summary": string, "key_concepts": [string]}}
  ]
}}

TRANSCRIPT:
{transcript}
"""

NOTES_SYSTEM = (
    "You are an expert university teaching assistant. You turn a section of a math/science lecture video into "
    "high-quality course-handout notes. You look carefully at the images, you are faithful to the source, and you "
    "never invent formulas."
)

NOTES_HEADER = """Course: {course_title}
Section {i}/{n}: "{section_title}" ({start_mmss}–{end_mmss})
Write everything in: {language_name}

You are given {k} frames extracted from this section of the video, each labeled with its number and timestamp,
followed by the transcript of the section.
"""

NOTES_TASKS = """
TRANSCRIPT OF THIS SECTION:
{transcript}

Tasks:
1. Look at every image. Pick the single image that best illustrates the core idea of this section.
   Prefer a complete, information-rich diagram over a partial animation, a title card or a logo.
2. Transcribe into LaTeX EVERY mathematical formula, equation or matrix that is visible in the images, exactly as
   displayed (source "image", with its image_index). You may also add a formula that is explicitly stated in the
   transcript (source "transcript", image_index null). Never add a formula that appears in neither.
   Keep it meaningful: skip decorative or random-looking numbers, and when an image shows many similar items
   (e.g. a row of vectors or a column of arithmetic), transcribe only 1–2 representative ones and say so in
   "meaning". At most 6 formulas. If there is no formula at all, return an empty list.
3. Write concise course notes (120–250 words) in the style of a university handout, in Markdown.
   Do NOT repeat the section title as a heading; use at most "####" sub-headings, short paragraphs and lists.
   Refer to what the figures show when useful. Use $...$ for inline math and $$...$$ for display math.
4. List 2–4 key takeaways.

IMPORTANT — JSON escaping: inside JSON strings, write every LaTeX backslash as TWO backslashes
(write \\\\frac, \\\\vec, \\\\theta — never a single backslash).

Return ONLY a JSON object with exactly this shape:
{{
  "best_image": integer,              // 1-based index of the chosen image
  "image_choice_reason": string,      // one sentence: why this image, mentioning what it shows
  "figure_caption": string,
  "formulas": [{{"latex": string, "meaning": string,
                "source": "image" | "transcript", "image_index": integer | null}}],
  "notes_markdown": string,
  "key_points": [string]
}}
"""
