import pytest

from lecturify.llm.jsonfix import extract_json_text, parse_json, sanitize_latex, sanitize_text, sanitize_tree


@pytest.mark.parametrize("cmd", ["frac", "theta", "nabla", "beta", "vec", "times", "rho", "text", "neq", "underline"])
def test_single_backslash_latex_survives(cmd):
    raw = '{"latex": "\\%s{x}"}' % cmd  # model wrote ONE backslash
    assert parse_json(raw)["latex"] == "\\" + cmd + "{x}"


def test_double_backslash_kept():
    raw = r'{"latex": "\\frac{a}{b} + \\theta"}'
    assert parse_json(raw)["latex"] == r"\frac{a}{b} + \theta"


def test_real_newline_escape_in_markdown_kept():
    raw = r'{"notes_markdown": "First line.\nThe second line with $\nabla f$."}'
    assert parse_json(raw)["notes_markdown"] == "First line.\nThe second line with $\\nabla f$."


def test_unicode_escape_kept():
    assert parse_json(r'{"a": "café"}')["a"] == "café"


def test_fences_and_prose_stripped():
    assert extract_json_text('Sure!\n```json\n{"a": 1}\n```') == '{"a": 1}'


@pytest.mark.parametrize("corrupt,fixed", [
    ("\x0crac{a}{b}", r"\frac{a}{b}"),
    ("\theta", r"\theta"),
    ("\nabla f", r"\nabla f"),
    ("\x08eta", r"\beta"),
    ("\rho", r"\rho"),
    ("a \times b", r"a \times b"),
])
def test_sanitize_latex_post_parse(corrupt, fixed):
    assert sanitize_latex(corrupt) == fixed


def test_sanitize_text_keeps_newlines():
    assert sanitize_text("a\nb \x0crac") == "a\nb \\frac"


def test_sanitize_tree_latex_key_only_strict():
    out = sanitize_tree({"latex": "\nabla", "notes_markdown": "x\ny", "formulas": [{"latex": "\theta"}]})
    assert out == {"latex": r"\nabla", "notes_markdown": "x\ny", "formulas": [{"latex": r"\theta"}]}
