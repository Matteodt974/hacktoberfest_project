"""Robust JSON extraction + LaTeX-escape repair for LLM output.

The critical trap: inside a JSON string, "\\frac" written with a SINGLE backslash is
*valid* JSON meaning form-feed + "rac". Same for \\theta (tab), \\nabla (newline),
\\beta (backspace), \\rho (carriage return). The JSON parses without error and the
formulas are silently corrupted. We defend in two layers:

1. `fix_latex_escapes` runs on the raw text BEFORE json.loads and doubles single
   backslashes that start a LaTeX command (and any invalid JSON escape like \\vec).
2. `sanitize_latex` / `sanitize_text` run AFTER parsing and turn leftover control
   characters back into backslash sequences.
"""
from __future__ import annotations

import json
import re
from typing import Any

# LaTeX commands that start with "n" — "\n" is otherwise a legit JSON newline.
_N_COMMANDS = {
    "nabla", "neq", "ne", "nu", "not", "neg", "ni", "newline", "nexists", "nmid",
    "nparallel", "nleq", "ngeq", "nsubseteq", "notin", "natural", "nearrow",
    "nwarrow", "ncong", "nsim", "nolimits", "normalsize", "newcommand", "nleftarrow",
    "nrightarrow", "nLeftarrow", "nRightarrow", "nless", "ngtr",
}
_VALID_JSON_ESCAPES = set('"\\/bfnrtu')


def fix_latex_escapes(raw: str) -> str:
    """Double single backslashes that are LaTeX commands rather than JSON escapes."""
    out: list[str] = []
    i, n = 0, len(raw)
    while i < n:
        c = raw[i]
        if c != "\\":
            out.append(c)
            i += 1
            continue
        if i + 1 >= n:
            out.append("\\\\")
            i += 1
            continue
        nxt = raw[i + 1]
        if nxt == "\\":  # already-escaped backslash: keep both
            out.append("\\\\")
            i += 2
            continue
        m = re.match(r"[A-Za-z]+", raw[i + 1:])
        word = m.group(0) if m else ""
        if nxt not in _VALID_JSON_ESCAPES:
            # Invalid JSON escape (\vec, \sum, \alpha, \{, \, ...) -> literal backslash
            out.append("\\\\")
        elif nxt in "bf" and len(word) >= 2:
            out.append("\\\\")  # \beta \frac \bar \forall
        elif nxt in "rt" and len(word) >= 2:
            out.append("\\\\")  # \rho \right \theta \times \text
        elif nxt == "n" and word in _N_COMMANDS:
            out.append("\\\\")  # \nabla \neq
        elif nxt == "u" and not re.match(r"u[0-9a-fA-F]{4}", raw[i + 1:i + 6]):
            out.append("\\\\")  # \underline \uparrow
        else:
            out.append("\\")
        i += 1
    return "".join(out)


def extract_json_text(text: str) -> str:
    """Strip ``` fences and keep from the first '{' to the last '}'."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object found in model output")
    return t[start:end + 1]


def parse_json(text: str) -> Any:
    fixed = fix_latex_escapes(extract_json_text(text))
    try:
        return json.loads(fixed, strict=False)
    except json.JSONDecodeError as err:
        # Typical model slips: unescaped quotes inside strings, missing commas, trailing commas.
        from json_repair import repair_json

        obj = repair_json(fixed, return_objects=True)
        if not isinstance(obj, dict) or not obj:
            raise ValueError(f"invalid JSON: {err}") from err
        return obj


_CTRL_PARTIAL = {"\x0c": "\\f", "\x08": "\\b", "\t": "\\t", "\r": "\\r"}


def sanitize_text(s: str) -> str:
    """Repair control chars in prose/markdown (real newlines are kept)."""
    for k, v in _CTRL_PARTIAL.items():
        s = s.replace(k, v)
    return s


def sanitize_latex(s: str) -> str:
    """Repair control chars in a single-line LaTeX formula (newlines too)."""
    s = sanitize_text(s).replace("\n", "\\n")
    return s.strip()


def sanitize_tree(obj: Any, latex_keys: frozenset[str] = frozenset({"latex"})) -> Any:
    """Recursively sanitize every string; keys in `latex_keys` get the strict repair."""
    if isinstance(obj, dict):
        return {
            k: (sanitize_latex(v) if k in latex_keys and isinstance(v, str) else sanitize_tree(v, latex_keys))
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [sanitize_tree(v, latex_keys) for v in obj]
    if isinstance(obj, str):
        return sanitize_text(obj)
    return obj
