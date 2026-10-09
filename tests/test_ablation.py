import json

from lecturify.config import Config
from lecturify.llm import prompts
from lecturify.llm.client import Image, Stats
from lecturify.llm.schemas import CourseOutline, Formula, SectionOutline, TextOnlyNotes
from lecturify.pipeline.ablation import compare, make_ablation, norm_latex
from lecturify.render.html import build_context


def test_norm_latex_is_strict_but_ignores_notation_noise():
    assert norm_latex(r"\dfrac{d}{dx} x^{2} = 2x.") == norm_latex(r"\frac{d}{dx}x^2=2x")
    assert norm_latex(r"\left( a \, b \right)") == norm_latex("(ab)")
    assert norm_latex("a^2+b^2") != norm_latex("a^2-b^2")


def F(latex, source="image"):
    return {"latex": latex, "source": source}


def test_compare_matches_one_to_one_and_splits_by_source():
    with_nt = {"formulas": [F("a=b"), F("a=b"), F("c=d", "transcript"), F("e=f")], "notes_markdown": "one two three"}
    without = {"formulas": [F("a = b", "transcript"), F("x=y", "transcript")], "notes_markdown": "one"}
    c = compare(with_nt, without, has_figure=True)
    # one "a=b" cancels one; the second "a=b" and "e=f" exist only with images; "c=d" is transcript noise
    assert c["only_with_image"] == 2 and c["only_with_transcript"] == 1 and c["only_without"] == 1
    assert (c["formulas_with"], c["formulas_without"], c["from_images"]) == (4, 2, 3)
    assert (c["words_with"], c["words_without"], c["figure"]) == (3, 1, True)


def test_prompts_share_every_non_image_clause_with_pass2():
    shared = ["Never add a formula", "Keep it meaningful: skip decorative or random-looking numbers",
              "At most 6 formulas. If there is no formula at all, return an empty list.",
              "Write concise course notes (120–250 words) in the style of a university handout, in Markdown.",
              'Do NOT repeat the section title as a heading; use at most "####" sub-headings',
              "Use $...$ for inline math and $$...$$ for display math.", "List 2–4 key takeaways.",
              "write every LaTeX backslash as TWO backslashes", "TRANSCRIPT OF THIS SECTION:\n{transcript}"]
    for clause in shared:
        assert clause in prompts.NOTES_TASKS and clause in prompts.ABLATION_TASKS, clause
    assert "image" not in prompts.ABLATION_SYSTEM.lower()
    assert "Image" not in prompts.ABLATION_HEADER and "frames" not in prompts.ABLATION_HEADER


OUTLINE = CourseOutline(course_title="c", sections=[SectionOutline(title=f"s{i}", start_s=i * 10, end_s=i * 10 + 10)
                                                    for i in range(3)])


class FakeClient:
    def __init__(self, fail=None):
        self.fail, self.parts, self.stats = fail, [], Stats()

    def generate_json(self, *, task_name, parts, **kw):
        self.parts.append(parts)
        if task_name == self.fail:
            raise RuntimeError("boom")
        return TextOnlyNotes(formulas=[Formula(latex="x", source="image", image_index=2)], notes_markdown="n")


def test_text_only_never_sees_an_image_and_sources_are_forced(tmp_path):
    cfg = Config(data_dir=tmp_path, concurrency=1)
    notes = [{"index": i, "notes": {}, "candidates": []} for i in range(3)]
    fc = FakeClient()
    res = make_ablation(tmp_path, OUTLINE, notes, lambda r: f"transcript {r['index']}", cfg, client=fc)
    assert len(fc.parts) == 3
    assert not any(isinstance(p, Image) for parts in fc.parts for p in parts)
    assert "transcript 1" in fc.parts[1][1]  # the same transcript string Pass 2 gets
    f = res["sections"][0]["notes"]["formulas"][0]
    assert (f["source"], f["image_index"]) == ("transcript", None)
    assert (tmp_path / "ablation.json").exists()


def test_skips_and_partial_failures(tmp_path):
    cfg = Config(data_dir=tmp_path, concurrency=1)
    notes = [{"index": 0, "notes": {}, "failed": True}, {"index": 1, "notes": {}}, {"index": 2, "notes": {}}]
    ctx = {0: "t", 1: "", 2: "t"}
    res = make_ablation(tmp_path, OUTLINE, notes, lambda r: ctx[r["index"]], cfg, client=FakeClient("ablation:section3"))
    s = {x["index"]: x for x in res["sections"]}
    assert s[0]["skipped"] == "pass2 failed" and s[1]["skipped"] == "no text context" and "error" in s[2]
    assert not (tmp_path / "ablation.json").exists()  # partial result is not cached


def _write_item(d, ablation=None):
    (d / "frames").mkdir(parents=True)
    (d / "frames/a.jpg").write_bytes(b"\xff\xd8x")
    (d / "meta.json").write_text(json.dumps({"video_id": "v", "title": "t", "url": "", "duration": 30}))
    (d / "outline.json").write_text(OUTLINE.model_dump_json())
    cand = [{"file": "frames/a.jpg", "t": 1.0, "edge_density": 0}]
    nt = {"best_image": 1, "formulas": [{"latex": "E=mc^2", "source": "image", "image_index": 1, "meaning": ""}],
          "notes_markdown": "x", "key_points": [], "figure_caption": "", "image_choice_reason": ""}
    (d / "notes.json").write_text(json.dumps([{"index": i, "candidates": cand, "notes": nt} for i in range(3)]))
    if ablation is not None:
        (d / "ablation.json").write_text(json.dumps(ablation))


def test_build_context_without_with_and_partial_ablation(tmp_path):
    a = tmp_path / "a"
    _write_item(a)
    ctx = build_context(a, Config())
    assert ctx["ablation_totals"] is None and all(s["ablation"] is None for s in ctx["sections"])

    b = tmp_path / "b"
    _write_item(b, {"cost": {"calls": 2}, "sections": [
        {"index": 0, "notes": {"formulas": [], "notes_markdown": "y", "key_points": []}},
        {"index": 1, "error": "boom"}]})  # section 2 missing entirely
    ctx = build_context(b, Config())
    assert [s["ablation"] is not None for s in ctx["sections"]] == [True, False, False]
    t = ctx["ablation_totals"]
    assert (t["compared"], t["sections"], t["only_with_image"], t["figures"]) == (1, 3, 1, 1)
