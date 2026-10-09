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

IMAGES_OUTLINE_USER = """{title_line}Write everything in: {language_name}

You are given {k} images, in order: screenshots of a lecture (slides, whiteboard, video frames), each labeled with its
number. {context_line}
Group the images into 1–{max_sections} ordered sections that follow the teaching progression. Every image belongs to
exactly one section; sections use consecutive images. "image_indices" are the image numbers exactly as labeled,
1-based: "Image 1" is 1, and the last image is {k}. Then propose a course title, a 3–4 sentence summary and the
prerequisites.

Return ONLY a JSON object with exactly this shape:
{{
  "course_title": string,
  "summary": string,
  "prerequisites": [string],
  "sections": [{{"title": string, "image_indices": [integer], "summary": string, "key_concepts": [string]}}]
}}
"""

# ---------------------------------------------------------------- CoVe (Chain-of-Verification)
VERIFY_SYSTEM = (
    "You are a meticulous fact-checker for university course notes. You only trust what is visible in the given "
    "image or stated in the given transcript."
)

VERIFY_PLAN = """Below is a DRAFT of course notes for one section of a lecture. For EACH formula and EACH key point,
write ONE verification question that a checker can answer by looking only at the source image and transcript.

Rules:
- The question must NOT contain the draft's answer (never copy the formula or the claim into the question).
  Good: "Which formula giving the derivative of a composite function is written on the image? Transcribe it in LaTeX."
  Bad: "Is the chain rule d/dx g(h(x)) = g'(h(x))h'(x)?"
- For a formula, ask for the exact transcription of what is displayed.
- For a key point, ask the factual question whose answer would confirm or refute it.
- Ask about what the LECTURE shows or says (the slide, the figure, the speaker). Never mention "the notes",
  "the draft" or "the summary": the checker has never seen them.
- Use the exact target ids given below.

DRAFT:
{draft}

Return ONLY a JSON object: {{"questions": [{{"target": string, "question": string}}]}}
"""

VERIFY_ANSWER = """Answer the question using ONLY the image(s) above and the transcript below. Do not guess.
If the information is not in the image or the transcript, set "found" to false.
If the answer is a formula, put its exact LaTeX in "latex" (every backslash written as TWO backslashes in JSON).

TRANSCRIPT:
{transcript}

QUESTION: {question}

Return ONLY a JSON object: {{"found": boolean, "answer": string, "latex": string | null}}
"""

VERIFY_REVISE = """You are given a DRAFT of course notes and independent VERIFICATION answers obtained from the source
image and transcript only (the verifier never saw the draft).

For each target, decide:
- "verified": the verification answer agrees with the draft (same formula up to notation, same fact);
- "corrected": the verification answer clearly contradicts the draft and is itself specific; give the corrected LaTeX
  (formula) or corrected sentence (key point) in "correction";
- "doubtful": the verifier could not find it, or the evidence is ambiguous.
Add a short "note" (one sentence) explaining your decision.

DRAFT:
{draft}

VERIFICATION:
{qa}

Return ONLY a JSON object: {{"items": [{{"target": string, "status": "verified" | "corrected" | "doubtful",
"correction": string | null, "note": string}}]}} with one item per target: {targets}
"""
