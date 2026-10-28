import csv
import json
from pathlib import Path
import shutil
import zipfile

import cv2
import numpy as np
import pandas as pd
import pytest

from staff_video.config import Config, ROOT
from staff_video.pipeline import analyze_video
from staff_video.tag import TagMatch
from staff_video.video import browser_encode, exact_frame, inspect_video


@pytest.fixture
def synthetic_video(tmp_path):
    path = tmp_path / "SYNTHETIC_TEST_ONLY.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (96, 72))
    assert writer.isOpened()
    for index in range(6):
        frame = np.full((72, 96, 3), index * 30, np.uint8)
        writer.write(frame)
    writer.release()
    return path


class SyntheticTracker:
    def __init__(self, *args):
        self.index = -1

    def detect(self, frame):
        self.index += 1
        if self.index == 3:
            return []
        return [{"tracker_id": 1, "box": [10, 10, 60, 65], "detector_score": .9,
                 "center_x": 35, "center_y": 37.5, "bottom_center_x": 35, "bottom_center_y": 65}]


class SyntheticMatcher:
    def __init__(self, config):
        self.reference_paths = [Path(config.tag_reference)]
        self.index = -1

    def match(self, *args):
        self.index += 1
        return TagMatch(.95 if self.index < 2 else .1, (20, 20, 30, 26))


def test_shared_pipeline_exports_real_decoded_frame_count_without_inventing_missing_boxes(synthetic_video, tmp_path, monkeypatch):
    monkeypatch.setattr("staff_video.pipeline.PersonTracker", SyntheticTracker)
    monkeypatch.setattr("staff_video.pipeline.TagMatcher", SyntheticMatcher)
    monkeypatch.setattr("staff_video.pipeline.browser_encode", shutil.copyfile)
    events = []
    result = analyze_video(synthetic_video, tmp_path / "outputs", Config(), events.append)
    assert result.summary["processed_frames"] == 6
    assert result.summary["staff_present_frames"] == 4
    assert result.summary["staff_present_seconds"] == pytest.approx(.16)
    assert "evaluation" not in result.summary
    assert result.frames[3]["people"] == [] and not result.frames[3]["staff_present"]
    with (result.directory / "frames.csv").open(newline="") as source:
        frame_rows = list(csv.DictReader(source))
    assert [int(row["frame_index"]) for row in frame_rows] == list(range(6))
    assert frame_rows[3]["detected_people"] == "0"
    assert [(interval["start_frame"], interval["end_frame"]) for interval in result.intervals] == [(1, 2), (4, 5)]
    assert (result.directory / "artifacts.zip").is_file()
    assert events[-1]["stage"] == "Complete"


def test_person_evaluation_is_saved_in_summary_and_zip(synthetic_video, tmp_path, monkeypatch):
    monkeypatch.setattr("staff_video.pipeline.PersonTracker", SyntheticTracker)
    monkeypatch.setattr("staff_video.pipeline.TagMatcher", SyntheticMatcher)
    monkeypatch.setattr("staff_video.pipeline.browser_encode", shutil.copyfile)
    labels = pd.DataFrame([
        {"frame_index": 0, "x1": 10, "y1": 10, "x2": 60, "y2": 65, "is_staff": 1, "tag_visible": 1},
        {"frame_index": 3, "x1": None, "y1": None, "x2": None, "y2": None, "is_staff": None, "tag_visible": None},
    ])
    result = analyze_video(synthetic_video, tmp_path / "evaluated", Config(), person_labels=labels)
    metrics = result.summary["person_evaluation"]
    assert metrics["labelled_frames"] == 2
    assert metrics["person_detection"]["precision"] == 1
    assert json.loads((result.directory / "summary.json").read_text())["person_evaluation"] == metrics
    with zipfile.ZipFile(result.directory / "artifacts.zip") as archive:
        assert json.loads(archive.read("summary.json"))["person_evaluation"] == metrics


def test_exact_sequential_frame_and_h264_export(synthetic_video, tmp_path):
    metadata = inspect_video(synthetic_video)
    assert metadata.reported_frames == 6 and metadata.fps == 25
    assert exact_frame(synthetic_video, 4).mean() == pytest.approx(120, abs=6)
    with pytest.raises(ValueError):
        exact_frame(synthetic_video, 6)
    target = tmp_path / "browser.mp4"
    browser_encode(synthetic_video, target)
    assert inspect_video(target).codec.lower() in {"avc1", "h264"}


def test_decode_error_is_understandable(tmp_path):
    invalid = tmp_path / "broken.mp4"
    invalid.write_bytes(b"not a video")
    with pytest.raises(ValueError, match="Cannot decode"):
        inspect_video(invalid)


def test_bad_model_checksum_fails_before_model_load(synthetic_video, tmp_path):
    model = tmp_path / "model.pt"
    model.write_bytes(b"invalid model")
    model.with_suffix(".json").write_text('{"sha256":"not-the-checksum"}')
    with pytest.raises(ValueError, match="checksum"):
        analyze_video(synthetic_video, tmp_path / "out", Config(model_path=str(model)))
    assert len(list((tmp_path / "out").glob("*/FAILED.txt"))) == 1
