from lecturify.llm.schemas import ImageGroup, ImagesOutline
from lecturify.pipeline.images_mode import _groups


def O(*groups):
    return ImagesOutline(course_title="c", sections=[ImageGroup(title=f"s{i}", image_indices=g) for i, g in enumerate(groups)])


def test_one_based_kept():
    assert _groups(O([1], [2], [3, 4]), 4) == [[1], [2], [3, 4]]


def test_zero_based_answer_is_shifted():
    assert _groups(O([0], [1], [2], [3]), 4) == [[1], [2], [3], [4]]


def test_orphans_and_duplicates():
    # 2 is a duplicate and 9 is out of range → section 2 ends up empty (dropped later); orphans 3, 4 join section 1
    assert _groups(O([1, 2], [2, 9]), 4) == [[1, 2, 3, 4], []]
