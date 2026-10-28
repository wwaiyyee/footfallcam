import csv
import json
import zipfile

import pandas as pd
import pytest

from staff_video.evaluation import evaluate_labels, evaluate_people
from staff_video.exports import build_zip, export_tables


def test_metrics_only_use_supplied_rows_and_undefined_metrics_are_null():
    frames = [{"frame_index": index, "staff_present": value} for index, value in enumerate([True, False, False])]
    labels = pd.DataFrame({"frame_index": [0, 1], "staff_present": [1, 1]})
    metrics = evaluate_labels(frames, labels)
    assert metrics["precision"] == 1 and metrics["recall"] == .5
    assert metrics["coverage"] == pytest.approx(2 / 3)
    metrics = evaluate_labels(frames, pd.DataFrame({"frame_index": [1], "staff_present": [0]}))
    assert metrics["precision"] is None and metrics["recall"] is None


@pytest.mark.parametrize("indices,values", [([0, 0], [1, 0]), ([5], [1]), ([-1], [1]),
    ([.5], [0]), ([0], [2]), ([0], ["yes"]), ([None], [1])])
def test_invalid_reference_labels_rejected(indices, values):
    with pytest.raises(ValueError):
        evaluate_labels([{"frame_index": 0, "staff_present": True}], pd.DataFrame({"frame_index": indices, "staff_present": values}))


def test_person_staff_and_tag_metrics_use_box_matches_and_empty_frames():
    frames = [
        {"people": [{"box": [0, 0, 20, 20], "staff_present": True, "tag_observed": True},
                    {"box": [50, 0, 70, 20], "staff_present": False, "tag_observed": True}]},
        {"people": [{"box": [0, 0, 20, 20], "staff_present": True, "tag_observed": False}]},
        {"people": []},
    ]
    labels = pd.DataFrame([
        {"frame_index": 0, "x1": 0, "y1": 0, "x2": 20, "y2": 20, "is_staff": 1, "tag_visible": 1},
        {"frame_index": 0, "x1": 25, "y1": 0, "x2": 45, "y2": 20, "is_staff": 1, "tag_visible": 0},
        {"frame_index": 1, "x1": 0, "y1": 0, "x2": 20, "y2": 20, "is_staff": 1, "tag_visible": 1},
        {"frame_index": 2, "x1": None, "y1": None, "x2": None, "y2": None, "is_staff": None, "tag_visible": None},
    ])
    metrics = evaluate_people(frames, labels, 100, 100)
    assert metrics["labelled_frames"] == 3 and metrics["coverage"] == 1
    assert metrics["person_detection"]["true_positive"] == 2
    assert metrics["person_detection"]["false_positive"] == 1
    assert metrics["person_detection"]["false_negative"] == 1
    assert metrics["staff_identification"]["recall"] == pytest.approx(2 / 3)
    assert metrics["visible_tag_detection"]["precision"] == pytest.approx(.5)


@pytest.mark.parametrize("row", [
    {"frame_index": 0, "x1": 0, "y1": 0, "x2": None, "y2": 10, "is_staff": 1, "tag_visible": 1},
    {"frame_index": 0, "x1": 0, "y1": 0, "x2": 0, "y2": 10, "is_staff": 1, "tag_visible": 1},
    {"frame_index": 0, "x1": 0, "y1": 0, "x2": 101, "y2": 10, "is_staff": 1, "tag_visible": 1},
    {"frame_index": 0, "x1": 0, "y1": 0, "x2": 10, "y2": 10, "is_staff": None, "tag_visible": 1},
    {"frame_index": 1, "x1": 0, "y1": 0, "x2": 10, "y2": 10, "is_staff": 1, "tag_visible": 1},
])
def test_invalid_person_labels_rejected(row):
    with pytest.raises(ValueError):
        evaluate_people([{"people": []}], pd.DataFrame([row]), 100, 100)


def test_empty_staff_exports_keep_headers_and_zip_has_no_recursion(tmp_path):
    frames = [{"frame_index": 0, "timestamp_seconds": 0, "duration_seconds": .04,
               "staff_present": False, "staff_count": 0, "tag_observed_count": 0, "people": []}]
    export_tables(tmp_path, frames, {"synthetic": True, "fixture_note": "Synthetic unit-test data only"})
    assert list(csv.DictReader((tmp_path / "intervals.csv").open())) == []
    assert "center_x" in (tmp_path / "detections.csv").read_text()
    assert list(csv.DictReader((tmp_path / "frames.csv").open()))[0]["detected_people"] == "0"
    assert json.loads((tmp_path / "summary.json").read_text())["synthetic"]
    target = build_zip(tmp_path)
    with zipfile.ZipFile(target) as archive:
        assert "summary.json" in archive.namelist() and "artifacts.zip" not in archive.namelist()
