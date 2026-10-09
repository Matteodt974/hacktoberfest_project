from lecturify.config import Config
from lecturify.llm.schemas import VAnswer, VItem, VPlan, VQuestion, VRevision
from lecturify.pipeline.verify import draft_text, verify_notes

NOTES = {"best_image": 1, "formulas": [{"latex": "a^2+b^2=c^2", "meaning": "Pythagoras", "source": "image",
                                         "image_index": 1}],
         "key_points": ["Right triangles only."], "notes_markdown": "x"}


def test_draft_targets():
    text, targets = draft_text(NOTES)
    assert targets == ["formula:1", "key_point:1"]
    assert "[formula:1]" in text and "a^2+b^2=c^2" in text


class FakeClient:
    def __init__(self):
        self.answer_prompts = []

    def generate_json(self, *, task_name, parts, schema, **kw):
        if schema is VPlan:
            return VPlan(questions=[VQuestion(target="formula:1", question="Which formula is on the image?"),
                                    VQuestion(target="key_point:1", question="Which triangles?")])
        if schema is VAnswer:
            self.answer_prompts.append(" ".join(p for p in parts if isinstance(p, str)))
            return VAnswer(found=True, answer="x", latex="a^2+b^2=c^2")
        return VRevision(items=[VItem(target="formula:1", status="verified"),
                                VItem(target="key_point:1", status="corrected", correction=None)])


def test_factored_answers_never_see_the_draft_and_empty_correction_is_doubtful(tmp_path):
    cfg = Config(data_dir=tmp_path)
    vdir = cfg.video_dir("v")
    (vdir / "frames").mkdir()
    (vdir / "frames/a.jpg").write_bytes(b"\xff\xd8x")
    r = {"index": 0, "candidates": [{"file": "frames/a.jpg", "t": 1.0, "edge_density": 0}], "notes": NOTES}
    fc = FakeClient()
    res = verify_notes(vdir, [r], lambda _: "transcript", fc, cfg)
    assert len(fc.answer_prompts) == 2
    assert all("Right triangles only" not in p and "Pythagoras" not in p for p in fc.answer_prompts)
    statuses = {it["target"]: it["status"] for it in res["sections"][0]["items"]}
    assert statuses == {"formula:1": "verified", "key_point:1": "doubtful"}
    assert res["counts"] == {"verified": 1, "corrected": 0, "doubtful": 1}
