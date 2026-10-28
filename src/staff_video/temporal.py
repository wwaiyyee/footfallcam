from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .config import Config


@dataclass
class TrackState:
    last_frame: int
    hits: deque = field(default_factory=deque)
    confirmed: bool = False
    generation: int = 1


class StaffResolver:
    def __init__(self, config: Config):
        self.config = config
        self.states: dict[int, TrackState] = {}
        self.generations: dict[int, int] = {}

    def resolve(self, frame_index: int, people: list[dict]) -> list[dict]:
        expired = [track for track, state in self.states.items()
                   if frame_index - state.last_frame > self.config.max_track_gap]
        for track in expired:
            del self.states[track]
        for person in people:
            track = person["tracker_id"]
            person["status"] = "Uncertain"
            person["staff_present"] = False
            person["track_id"] = None
            person["tag_observed"] = person["tag_score"] >= self.config.tag_threshold
            if track is None:
                continue
            if track not in self.states:
                generation = self.generations.get(track, 0) + 1
                self.generations[track] = generation
                self.states[track] = TrackState(frame_index, generation=generation)
            state = self.states[track]
            state.last_frame = frame_index
            while state.hits and frame_index - state.hits[0] >= self.config.confirmation_window:
                state.hits.popleft()
            if person["tag_observed"] and (not state.hits or state.hits[-1] != frame_index):
                state.hits.append(frame_index)
            state.confirmed = state.confirmed or len(state.hits) >= self.config.confirmation_hits
            person["track_id"] = f"{track}.{state.generation}"
            person["staff_present"] = state.confirmed
            if state.confirmed:
                person["status"] = "Tag observed" if person["tag_observed"] else "Staff tracked"
        return people


def inclusive_intervals(frames: list[dict], track_id: str | None = None) -> list[dict]:
    intervals = []
    start = None
    previous = None
    for frame in frames:
        present = frame["staff_present"] if track_id is None else any(
            person["staff_present"] and person["track_id"] == track_id for person in frame["people"]
        )
        if present:
            if start is None:
                start = frame
            elif previous is not None and frame["frame_index"] != previous["frame_index"] + 1:
                intervals.append(_interval(start, previous, track_id))
                start = frame
            previous = frame
        elif start is not None:
            intervals.append(_interval(start, previous, track_id))
            start = previous = None
    if start is not None:
        intervals.append(_interval(start, previous, track_id))
    return intervals


def _interval(start, end, track_id):
    result = {
        "start_frame": start["frame_index"], "end_frame": end["frame_index"],
        "start_seconds": start["timestamp_seconds"],
        "end_seconds_exclusive": end["timestamp_seconds"] + end["duration_seconds"],
        "frame_count": end["frame_index"] - start["frame_index"] + 1,
    }
    if track_id is not None:
        result["track_id"] = track_id
    return result
