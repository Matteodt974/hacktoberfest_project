from lecturify.llm.schemas import SectionOutline
from lecturify.pipeline.outline import normalize_sections


def S(t, a, b):
    return SectionOutline(title=t, start_s=a, end_s=b)


def test_sorted_contiguous_and_covering():
    out = normalize_sections([S("b", 100, 200), S("a", 5, 90), S("c", 210, 999)], duration=300)
    assert [s.title for s in out] == ["a", "b", "c"]
    assert out[0].start_s == 0 and out[-1].end_s == 300
    assert all(x.end_s == y.start_s for x, y in zip(out, out[1:]))


def test_short_sections_merged():
    out = normalize_sections([S("a", 0, 100), S("tiny", 100, 110), S("c", 110, 300)], duration=300)
    assert [s.title for s in out] == ["a", "c"]
    assert out[0].end_s == 110


def test_short_first_section_merged_into_next():
    out = normalize_sections([S("intro", 0, 10), S("b", 10, 200), S("c", 200, 300)], duration=300)
    assert len(out) == 2 and out[0].start_s == 0
