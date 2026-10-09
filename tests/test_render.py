from lecturify.render.html import markdown_with_math, strip_leading_title


def test_subscripts_not_turned_into_italics():
    out = markdown_with_math("Let $a_1 + b_2$ and $c_1 * d_2$ be sums.")
    assert "<em>" not in out
    assert "$a_1 + b_2$" in out and "$c_1 * d_2$" in out


def test_backslashes_preserved_in_display_math():
    out = markdown_with_math("Text\n\n$$\\frac{a}{b} \\\\ \\vec{v}_1$$\n\nmore *emph*")
    assert "$$\\frac{a}{b} \\\\ \\vec{v}_1$$" in out
    assert "<em>emph</em>" in out


def test_math_is_html_escaped():
    out = markdown_with_math("If $a<b$ then $b>a$.")
    assert "$a&lt;b$" in out and "$b&gt;a$" in out


def test_dollar_amounts_not_math():
    out = markdown_with_math("It costs $5 and then $10.")
    assert "$5 and then $10" in out


def test_strip_repeated_title():
    md = "# Training and Parameter Tuning\n\nBody text."
    assert strip_leading_title(md, "Training and Parameter Tuning") == "Body text."
    assert strip_leading_title("Body only.", "X") == "Body only."
