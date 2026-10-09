import pytest

from lecturify.ingest.youtube import IngestError, mmss, parse_video_id, parse_vtt, transcript_for_llm


@pytest.mark.parametrize("url", [
    "https://www.youtube.com/watch?v=LPZh9BOjkQs&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=6",
    "https://youtu.be/LPZh9BOjkQs?t=42",
    "youtube.com/shorts/LPZh9BOjkQs",
    "https://www.youtube.com/embed/LPZh9BOjkQs",
    "https://m.youtube.com/watch?feature=share&v=LPZh9BOjkQs",
    "LPZh9BOjkQs",
])
def test_parse_video_id(url):
    assert parse_video_id(url) == "LPZh9BOjkQs"


@pytest.mark.parametrize("url", ["https://example.com/watch?v=LPZh9BOjkQs", "https://youtube.com/watch?v=short", ""])
def test_parse_video_id_rejects(url):
    with pytest.raises(IngestError):
        parse_video_id(url)


def test_mmss():
    assert mmss(0) == "00:00"
    assert mmss(312.7) == "05:12"
    assert mmss(3725) == "1:02:05"


AUTO_VTT = """WEBVTT
Kind: captions

00:00:00.000 --> 00:00:02.000 align:start
hey<00:00:00.500><c> everyone</c>

00:00:02.000 --> 00:00:04.000
hey everyone
today we talk

00:00:04.000 --> 00:00:06.000
today we talk
about matrices
"""


def test_parse_vtt_dedupes_rolling_auto_captions():
    cues = parse_vtt(AUTO_VTT)
    assert " ".join(c["text"] for c in cues) == "hey everyone today we talk about matrices"
    assert cues[0]["start"] == 0.0


def test_transcript_for_llm_groups_and_prefixes():
    cues = [{"start": float(t), "end": t + 2.0, "text": f"w{t}"} for t in range(0, 40, 2)]
    out = transcript_for_llm(cues, window_s=12)
    lines = out.splitlines()
    assert lines[0].startswith("[00:00] w0 w2")
    assert lines[1].startswith("[00:12]")
    assert transcript_for_llm(cues, start=20, end=25).startswith("[00:18]")
