from pathlib import Path

import cv2
import numpy as np
import pytest

from staff_video.config import Config, ROOT, load_config
from staff_video.detector import restore_box, track_identities
from staff_video.tag import TagMatcher


def test_reference_pattern_matches_but_solid_clothing_does_not():
    matcher = TagMatcher(Config(tag_min_width=24, tag_max_width=24, tag_rotation_step=90))
    reference = cv2.imread(str(ROOT / "assets/tag_reference.png"), cv2.IMREAD_GRAYSCALE)
    reference = cv2.resize(reference, (24, round(24 * reference.shape[0] / reference.shape[1])))
    frame = np.full((100, 100, 3), 80, np.uint8)
    assert matcher.match(frame, (0, 0, 100, 100)).score < .82
    frame[30:30 + reference.shape[0], 30:54] = cv2.cvtColor(reference, cv2.COLOR_GRAY2BGR)
    match = matcher.match(frame, (0, 0, 100, 100))
    assert match.score > .98
    assert match.box[:2] == (30, 30)


@pytest.mark.parametrize("settings", [dict(tag_threshold=float("nan")), dict(confirmation_hits=0),
    dict(confirmation_window=1, confirmation_hits=2), dict(tag_min_width=40, tag_max_width=12),
    dict(tag_rotation_step=35), dict(detector_rotations=[45]), dict(detector_rotations=[0, 0])])
def test_bad_config_rejected(settings):
    with pytest.raises(ValueError):
        Config(**settings)


def test_config_paths_are_relative_to_yaml_not_working_directory(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("model_path: weights.pt\n")
    assert load_config(config).model_path == str(tmp_path / "weights.pt")
    config.write_text("unrecognised: 4\n")
    with pytest.raises(ValueError, match="Unknown"):
        load_config(config)


def test_default_model_is_yolo11s_in_python_and_yaml():
    expected = str(ROOT / "models/yolo11s.pt")
    assert Config().model_path == expected
    assert load_config(ROOT / "config/default.yaml").model_path == expected


@pytest.mark.parametrize("angle,rotated", [(0, [10, 20, 40, 60]), (90, [20, 60, 60, 90]),
    (180, [60, 20, 90, 60]), (270, [20, 10, 60, 40])])
def test_rotated_boxes_return_to_source_pixels(angle, rotated):
    assert restore_box(rotated, angle, 100, 80) == [10, 20, 40, 60]


def test_bytetrack_subset_indices_map_to_the_correct_source_detection():
    candidates = np.array([[0, 0, 10, 10, .2, 0], [20, 20, 30, 30, .9, 0], [40, 40, 50, 50, .8, 0]])
    tracked = np.array([[20, 20, 30, 30, 17, .9, 0, 0], [40, 40, 50, 50, 18, .8, 0, 1],
                        [0, 0, 10, 10, 19, .2, 0, 0]])
    assert track_identities(tracked, candidates, .25, .1) == {1: 17, 2: 18, 0: 19}
