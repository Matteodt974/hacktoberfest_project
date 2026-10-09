"""M0 smoke test: lifts the unknowns about Gemma 4 on the Gemini API.

(a) plain text call, (b) one PIL-generated image with a formula, (c) JSON mode,
(d) 6 then 10 images in a single request, (e) a small parallel burst (rate limits).
"""
from __future__ import annotations

import io
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor

from PIL import Image as PILImage, ImageDraw, ImageFont

from .config import Config
from .llm.client import GemmaClient, Image, LLMError
from .llm.schemas import SmokeFormula, SmokeImages

log = logging.getLogger(__name__)

SHAPES = ["circle", "square", "triangle", "star", "cross", "arrow", "ellipse", "diamond", "line", "grid"]


def _font(size: int):
    for name in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def _jpeg(img: PILImage.Image) -> Image:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=90)
    return Image(buf.getvalue())


def formula_image() -> Image:
    """Dark background like 3Blue1Brown, with a formula written on it."""
    img = PILImage.new("RGB", (900, 400), (20, 20, 28))
    d = ImageDraw.Draw(img)
    d.text((80, 80), "E = mc²", font=_font(90), fill=(240, 240, 240))
    d.text((80, 230), "∇ · E = ρ / ε₀", font=_font(70), fill=(120, 200, 255))
    return _jpeg(img)


def shape_image(i: int) -> Image:
    img = PILImage.new("RGB", (512, 512), (20, 20, 28))
    d = ImageDraw.Draw(img)
    c = (255, 200, 60)
    s = SHAPES[i % len(SHAPES)]
    if s == "circle":
        d.ellipse((120, 120, 392, 392), outline=c, width=10)
    elif s == "square":
        d.rectangle((120, 120, 392, 392), outline=c, width=10)
    elif s == "triangle":
        d.polygon([(256, 100), (100, 400), (412, 400)], outline=c, width=10)
    elif s == "star":
        d.text((150, 120), "★", font=_font(260), fill=c)
    elif s == "cross":
        d.line((120, 120, 392, 392), fill=c, width=14); d.line((392, 120, 120, 392), fill=c, width=14)
    elif s == "arrow":
        d.line((100, 256, 380, 256), fill=c, width=14); d.polygon([(380, 200), (440, 256), (380, 312)], fill=c)
    elif s == "ellipse":
        d.ellipse((60, 180, 452, 332), outline=c, width=10)
    elif s == "diamond":
        d.polygon([(256, 80), (432, 256), (256, 432), (80, 256)], outline=c, width=10)
    elif s == "line":
        d.line((80, 430, 430, 80), fill=c, width=14)
    else:
        for k in range(5):
            d.line((80 + k * 88, 80, 80 + k * 88, 432), fill=c, width=6)
            d.line((80, 80 + k * 88, 432, 80 + k * 88), fill=c, width=6)
    return _jpeg(img)


def run_smoke(cfg: Config) -> dict:
    results: dict = {"model": cfg.model}
    client = GemmaClient(concurrency=4, json_mode=False)

    def step(name, fn):
        t0 = time.time()
        try:
            out = fn()
            results[name] = {"ok": True, "latency_s": round(time.time() - t0, 1), "result": out}
        except Exception as e:  # noqa: BLE001 — we want to record every failure
            results[name] = {"ok": False, "latency_s": round(time.time() - t0, 1), "error": f"{type(e).__name__}: {e}"[:600]}
        print(f"[{'OK ' if results[name]['ok'] else 'ERR'}] {name}: {json.dumps(results[name], ensure_ascii=False)[:400]}")

    step("a_text", lambda: client.generate_text(
        task_name="smoke:text", model=cfg.model, parts=["Reply with exactly: Gemma is alive."], thinking=None)[0])

    step("a2_text_thinking_high", lambda: client.generate_text(
        task_name="smoke:think", model=cfg.model, parts=["What is 17*23? Reply with the number only."], thinking="high")[0])

    step("a3_system_instruction", lambda: client.generate_text(
        task_name="smoke:system", model=cfg.model, system="You always answer in French.",
        parts=["Say hello."], thinking="minimal")[0])

    img = formula_image()
    step("b_image_formula_inline", lambda: client.generate_text(
        task_name="smoke:image", model=cfg.model, thinking="minimal",
        parts=["Image 1 —", img, "Transcribe every formula visible in Image 1 into LaTeX, one per line."])[0])

    prompt_json = ("Image 1 —", img,
                   'Return ONLY a JSON object {"latex": string, "description": string} for the FIRST formula in the '
                   "image. Write every LaTeX backslash as two backslashes.")

    def json_mode_on():
        raw, _ = client.generate_text(task_name="smoke:json_mode", model=cfg.model, thinking="minimal",
                                      parts=list(prompt_json), json_mode=True)
        return raw

    step("c_json_mode_response_mime_type", json_mode_on)

    def json_parsed():
        r = client.generate_json(task_name="smoke:json_parse", system="You are precise.", model=cfg.model,
                                 parts=list(prompt_json), schema=SmokeFormula, thinking="minimal")
        return r.model_dump()

    step("c2_generate_json_parsed", json_parsed)

    def schema_mode():
        from google.genai import types
        resp = client._client.models.generate_content(
            model=cfg.model, contents=client._contents(list(prompt_json)),
            config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=SmokeFormula))
        return resp.text

    step("c3_response_schema", schema_mode)

    for n in (6, 10):
        def multi(n=n):
            parts: list = []
            for i in range(n):
                parts += [f"Image {i + 1} —", shape_image(i)]
            parts.append(f'There are {n} images. Return ONLY JSON {{"count": int, "descriptions": [one short '
                         f"description of the main shape per image, in order]}}.")
            r = client.generate_json(task_name=f"smoke:{n}imgs", system="You are precise.", model=cfg.model,
                                     parts=parts, schema=SmokeImages, thinking="minimal")
            return {"expected": SHAPES[:n], **r.model_dump()}
        step(f"d_{n}_images", multi)

    def burst():
        t0 = time.time()
        def one(i):
            try:
                client.generate_text(task_name=f"smoke:burst{i}", model=cfg.model,
                                     parts=[f"Reply with the number {i}."], thinking=None)
                return "ok"
            except LLMError as e:
                return str(e)[:120]
        with ThreadPoolExecutor(4) as ex:
            out = list(ex.map(one, range(8)))
        return {"results": out, "wall_s": round(time.time() - t0, 1)}

    step("e_burst_8_calls_4_parallel", burst)

    results["stats"] = client.stats.as_dict()
    out = cfg.data_dir / "smoke_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved {out}")
    return results
