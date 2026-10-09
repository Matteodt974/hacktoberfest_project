"""Thin wrapper around google-genai for Gemma 4: retries, robust JSON, disk cache, stats."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar, Union

from pydantic import BaseModel, ValidationError

from .jsonfix import parse_json, sanitize_tree

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class Image:
    """A JPEG/PNG image to send to Gemma, inline."""
    data: bytes
    mime_type: str = "image/jpeg"

    @classmethod
    def from_path(cls, path: Path | str) -> "Image":
        p = Path(path)
        mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
        return cls(p.read_bytes(), mime)

    @property
    def sha(self) -> str:
        return hashlib.sha256(self.data).hexdigest()


Part = Union[str, Image]


@dataclass
class Stats:
    calls: int = 0
    cache_hits: int = 0
    latency_s: float = 0.0
    prompt_tokens: int = 0
    output_tokens: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, *, latency: float = 0.0, hit: bool = False, usage: dict | None = None) -> None:
        with self._lock:
            if hit:
                self.cache_hits += 1
                return
            self.calls += 1
            self.latency_s += latency
            if usage:
                self.prompt_tokens += usage.get("prompt_token_count") or 0
                self.output_tokens += usage.get("candidates_token_count") or 0

    def replay(self, cost: dict) -> None:
        """Count the original cost of a cached response, so stats describe what produced the notes."""
        with self._lock:
            self.cache_hits += 1
            self.calls += cost.get("calls", 1)
            self.latency_s += cost.get("latency_s", 0.0)
            self.prompt_tokens += cost.get("prompt_tokens", 0)
            self.output_tokens += cost.get("output_tokens", 0)

    def as_dict(self) -> dict:
        return {
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "latency_s": round(self.latency_s, 1),
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
        }


class LLMError(RuntimeError):
    pass


class GemmaClient:
    def __init__(self, *, concurrency: int = 2, max_retries: int = 7, json_mode: bool | None = None):
        from google import genai  # imported lazily so offline commands work without a key

        if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
            raise LLMError("GEMINI_API_KEY is not set. Put it in .env (see .env.example).")
        self._client = genai.Client()
        self._sem = threading.Semaphore(concurrency)
        self.max_retries = max_retries
        if json_mode is None:
            json_mode = os.environ.get("LECTURIFY_JSON_MODE", "1") != "0"
        self.json_mode = json_mode
        self.stats = Stats()

    # ------------------------------------------------------------------ low level
    def _contents(self, parts: list[Part]):
        from google.genai import types

        out = []
        for p in parts:
            if isinstance(p, Image):
                out.append(types.Part.from_bytes(data=p.data, mime_type=p.mime_type))
            else:
                out.append(types.Part.from_text(text=p))
        return [types.Content(role="user", parts=out)]

    def _config(self, system: str | None, thinking: str | None, json_mode: bool):
        from google.genai import types

        kw: dict = {}
        if system:
            kw["system_instruction"] = system
        if thinking:
            kw["thinking_config"] = types.ThinkingConfig(thinking_level=thinking)
        if json_mode:
            kw["response_mime_type"] = "application/json"
        return types.GenerateContentConfig(**kw)

    def generate_text(
        self, *, task_name: str, parts: list[Part], model: str, system: str | None = None,
        thinking: str | None = "minimal", json_mode: bool = False,
    ) -> tuple[str, dict]:
        """One raw call with retries on 429/5xx. Returns (text, usage)."""
        import httpx
        from google.genai import errors

        contents = self._contents(parts)
        config = self._config(system, thinking, json_mode)
        delay = 2.0
        for attempt in range(1, self.max_retries + 1):
            t0 = time.time()
            try:
                with self._sem:
                    resp = self._client.models.generate_content(model=model, contents=contents, config=config)
            except (errors.APIError, httpx.TransportError) as e:
                code = getattr(e, "code", None) if isinstance(e, errors.APIError) else type(e).__name__
                retryable = (code == 429 or (isinstance(code, int) and code >= 500)
                             or isinstance(e, httpx.TransportError))  # connection reset, timeouts
                if not retryable or attempt == self.max_retries:
                    raise LLMError(f"{task_name}: Gemini API error {code}: {e}") from e
                sleep = delay + random.uniform(0, delay / 2)
                log.warning("%s: API %s (attempt %d/%d), retrying in %.1fs", task_name, code, attempt, self.max_retries, sleep)
                time.sleep(sleep)
                delay = min(delay * 2, 30.0)
                continue
            latency = time.time() - t0
            usage = {}
            if getattr(resp, "usage_metadata", None):
                usage = resp.usage_metadata.model_dump(exclude_none=True)
            self.stats.add(latency=latency, usage=usage)
            usage["_latency_s"] = latency
            text = resp.text or ""
            log.info("LLM %-22s model=%s latency=%.1fs cache=miss tokens_in=%s out=%s",
                     task_name, model, latency, usage.get("prompt_token_count"), usage.get("candidates_token_count"))
            return text, usage
        raise LLMError(f"{task_name}: exhausted retries")

    # ------------------------------------------------------------------ JSON
    @staticmethod
    def _cache_key(model: str, system: str | None, parts: list[Part], thinking: str | None, schema: type) -> str:
        h = hashlib.sha256()
        for chunk in (model, system or "", thinking or "", schema.__name__,
                      json.dumps(schema.model_json_schema(), sort_keys=True)):
            h.update(chunk.encode())
            h.update(b"\x00")
        for p in parts:
            h.update(("img:" + p.sha) .encode() if isinstance(p, Image) else ("txt:" + p).encode())
            h.update(b"\x00")
        return h.hexdigest()[:32]

    def generate_json(
        self, *, task_name: str, system: str, parts: list[Part], schema: type[T], model: str,
        thinking: str | None = "high", cache_dir: Path | None = None,
    ) -> T:
        key = self._cache_key(model, system, parts, thinking, schema)
        cache_file = cache_dir / f"{key}.json" if cache_dir else None
        if cache_file and cache_file.exists():
            cached = json.loads(cache_file.read_text())
            self.stats.replay(cached.get("cost", {}))
            log.info("LLM %-22s model=%s cache=hit", task_name, model)
            return schema.model_validate(cached["parsed"])

        cost = {"calls": 0, "latency_s": 0.0, "prompt_tokens": 0, "output_tokens": 0}

        def account(usage: dict) -> None:
            cost["calls"] += 1
            cost["latency_s"] = round(cost["latency_s"] + usage.get("_latency_s", 0.0), 2)
            cost["prompt_tokens"] += usage.get("prompt_token_count") or 0
            cost["output_tokens"] += usage.get("candidates_token_count") or 0

        raw, usage = self._call_json(task_name, system, parts, model, thinking)
        account(usage)
        try:
            parsed = self._validate(raw, schema)
        except (ValueError, ValidationError) as err:
            log.warning("%s: invalid JSON (%s), one repair attempt", task_name, str(err)[:200])
            repair = parts + [
                "\n\nYOUR PREVIOUS ANSWER:\n" + raw,
                f"\n\nYour answer was not valid JSON or did not match the schema: {str(err)[:800]}\n"
                "Return ONLY the corrected JSON object. Remember: write every LaTeX backslash as two backslashes.",
            ]
            raw, usage = self._call_json(task_name + ":repair", system, repair, model, thinking)
            account(usage)
            try:
                parsed = self._validate(raw, schema)
            except (ValueError, ValidationError) as err2:
                raise LLMError(f"{task_name}: model did not return valid JSON after repair: {err2}") from err2

        if cache_file:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(
                {"task": task_name, "model": model, "cost": cost, "raw": raw, "parsed": parsed.model_dump()},
                ensure_ascii=False, indent=1))
        return parsed

    def _call_json(self, task_name, system, parts, model, thinking) -> tuple[str, dict]:
        if self.json_mode:
            try:
                return self.generate_text(task_name=task_name, parts=parts, model=model, system=system,
                                          thinking=thinking, json_mode=True)
            except LLMError as e:
                if "400" not in str(e):
                    raise
                log.warning("JSON mode rejected by API, falling back to plain text: %s", str(e)[:200])
                self.json_mode = False
        return self.generate_text(task_name=task_name, parts=parts, model=model, system=system,
                                  thinking=thinking, json_mode=False)

    @staticmethod
    def _validate(raw: str, schema: type[T]) -> T:
        data = sanitize_tree(parse_json(raw))
        return schema.model_validate(data)
