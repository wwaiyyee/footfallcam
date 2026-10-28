import pytest

from staff_video.config import Config
from staff_video.temporal import StaffResolver, inclusive_intervals


def person(track=1, score=.9):
    return {"tracker_id": track, "tag_score": score}


def test_identity_requires_repeated_tag_evidence_and_persists_without_tag():
    resolver = StaffResolver(Config())
    first = resolver.resolve(0, [person()])[0]
    assert first["tag_observed"] and not first["staff_present"]
    confirmed = resolver.resolve(1, [person()])[0]
    assert confirmed["status"] == "Tag observed"
    tracked = resolver.resolve(2, [person(score=.1)])[0]
    assert tracked["status"] == "Staff tracked"
    assert resolver.resolve(3, []) == []
    stranger = resolver.resolve(4, [person(track=2, score=.1)])[0]
    assert not stranger["staff_present"]


def test_expired_identity_and_confirmation_window_do_not_leak():
    resolver = StaffResolver(Config(confirmation_window=3, max_track_gap=3))
    resolver.resolve(0, [person()])
    assert not resolver.resolve(3, [person()])[0]["staff_present"]
    assert resolver.resolve(4, [person()])[0]["staff_present"]
    recycled = resolver.resolve(8, [person(score=.1)])[0]
    assert not recycled["staff_present"] and recycled["track_id"] == "1.2"


def test_unassigned_tag_candidate_is_not_confirmed_and_same_frame_is_not_double_counted():
    resolver = StaffResolver(Config())
    assert not resolver.resolve(0, [person(track=None)])[0]["staff_present"]
    resolver.resolve(1, [person()])
    assert not resolver.resolve(1, [person()])[0]["staff_present"]


@pytest.mark.parametrize("presence,expected", [([], []), ([False], []), ([True], [(0, 0)]),
    ([True, True, False, True], [(0, 1), (3, 3)]), ([False, True, True], [(1, 2)])])
def test_inclusive_intervals(presence, expected):
    frames = [{"frame_index": index, "timestamp_seconds": index / 25, "duration_seconds": .04,
               "staff_present": present} for index, present in enumerate(presence)]
    intervals = inclusive_intervals(frames)
    assert [(row["start_frame"], row["end_frame"]) for row in intervals] == expected
    for row in intervals:
        assert row["frame_count"] == row["end_frame"] - row["start_frame"] + 1
        assert row["end_seconds_exclusive"] == pytest.approx((row["end_frame"] + 1) / 25)


def test_interval_splits_missing_frame_indices():
    frames = [{"frame_index": index, "timestamp_seconds": index / 25, "duration_seconds": .04,
               "staff_present": True} for index in [0, 2]]
    assert len(inclusive_intervals(frames)) == 2
