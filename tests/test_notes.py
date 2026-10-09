import json

from lecturify.config import Config
from lecturify.llm.schemas import CourseOutline, SectionNotes, SectionOutline
from lecturify.pipeline.notes import make_notes, select_candidates

FRAMES = [{"file": f"frames/f_{t:06d}.jpg", "t": float(t), "edge_density": e}
          for t, e in [(5, .1), (12, .3), (14, .2), (30, .05), (31, .4), (55, .1), (70, .2), (90, .9)]]


def test_candidates_inside_section_only():
    assert [f["t"] for f in select_candidates(FRAMES, 10, 40, 6)] == [12, 14, 30, 31]


def test_candidates_one_per_slice_prefer_edges():
    got = select_candidates(FRAMES, 0, 100, 3)  # slices [0,33) [33,66) [66,100)
    assert [f["t"] for f in got] == [31, 55, 90]


def test_candidates_fallback_nearest():
    assert [f["t"] for f in select_candidates(FRAMES, 40, 50, 6)] == [55]


class FlakyClient:
    def __init__(self, fail_section):
        self.fail_section, self.calls = fail_section, 0

    def generate_json(self, *, task_name, **kw):
        self.calls += 1
        if task_name == f"pass2:section{self.fail_section}":
            raise RuntimeError("boom")
        return SectionNotes(best_image=1, notes_markdown=f"notes for {task_name}")


def test_failed_section_gets_placeholder_then_is_retried(tmp_path):
    cfg = Config(data_dir=tmp_path, concurrency=1)
    vdir = cfg.video_dir("vid")
    (vdir / "frames").mkdir()
    for f in FRAMES:
        (vdir / f["file"]).write_bytes(b"\xff\xd8fake")
    outline = CourseOutline(course_title="c", sections=[SectionOutline(title=f"s{i}", start_s=i * 50, end_s=(i + 1) * 50)
                                                        for i in range(2)])
    res = make_notes("vid", outline, [], FRAMES, FlakyClient(fail_section=2), cfg)
    assert [r.get("failed", False) for r in res] == [False, True]
    assert "could not be generated" in res[1]["notes"]["notes_markdown"]
    ok = FlakyClient(fail_section=99)
    res2 = make_notes("vid", outline, [], FRAMES, ok, cfg)
    assert ok.calls == 1  # only the failed section is regenerated
    assert not any(r.get("failed") for r in res2)
    assert json.loads((vdir / "notes.json").read_text())[1]["notes"]["notes_markdown"] == "notes for pass2:section2"
