import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from staff_video.config import Config, ROOT
from staff_video.exports import build_zip, export_tables
from staff_video.pipeline import AnalysisResult


@pytest.fixture
def synthetic_result(tmp_path):
    source = tmp_path / "SYNTHETIC_UI_TEST_ONLY.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 25, (96, 72))
    for index in range(3):
        writer.write(np.full((72, 96, 3), index * 60, np.uint8))
    writer.release()
    directory = tmp_path / "run"
    directory.mkdir()
    frames = [{"frame_index": index, "timestamp_seconds": index * .04, "duration_seconds": .04,
               "staff_present": index > 0, "staff_count": int(index > 0), "tag_observed_count": 0,
               "people": []} for index in range(3)]
    summary = {"synthetic": True, "source": {"filename": source.name}, "created_utc": "SYNTHETIC TEST",
               "settings": Config().to_dict(), "processed_frames": 3, "staff_present_frames": 2,
               "staff_present_seconds": .08, "staff_track_count": 1, "processing_seconds": 1,
               "decoded_duration_seconds": .12, "warnings": []}
    shutil.copyfile(source, directory / "annotated.mp4")
    export_tables(directory, frames, summary)
    build_zip(directory)
    return AnalysisResult(source, directory, summary, frames)


def forbid_inference(*args, **kwargs):
    pytest.fail("Ordinary UI rerun must not start inference.")


def test_initial_ui_does_not_analyze_and_shows_explicit_action(monkeypatch):
    monkeypatch.setattr("staff_video.pipeline.analyze_video", forbid_inference)
    app = AppTest.from_file(str(ROOT / "app.py")).run()
    assert not app.exception
    assert any(button.label == "Analyze video" for button in app.button)
    mode = next(selectbox for selectbox in app.selectbox if selectbox.label == "Person detection mode")
    assert mode.value.startswith("Coverage-focused · YOLO11s")
    assert any("YOLO11m" in option for option in mode.options)
    mode.select(mode.options[1]).run()
    assert not app.exception
    app.slider[0].set_value(.25).run()
    assert not app.exception


def test_completed_result_survives_inspection_and_setting_changes(synthetic_result, monkeypatch):
    monkeypatch.setattr("staff_video.pipeline.analyze_video", forbid_inference)
    app = AppTest.from_file(str(ROOT / "app.py"))
    app.session_state["result"] = synthetic_result
    app.session_state["frame_index"] = 0
    app.session_state["playback_time"] = 0.0
    app.session_state["interval_selection"] = None
    app.run()
    assert not app.exception
    assert any("SYNTHETIC" in warning.value for warning in app.warning)
    assert any(len(table.value) == 3 and "People detected" in table.value.columns for table in app.dataframe)
    next(button for button in app.button if button.label == "Next →").click().run()
    assert not app.exception and app.session_state["frame_index"] == 1
    app.slider[0].set_value(.25).run()
    assert not app.exception
    assert app.session_state["result"].summary["processed_frames"] == 3
    selector = next(selectbox for selectbox in app.selectbox if selectbox.key == "interval_selection")
    selector.select(0).run()
    assert not app.exception and app.session_state["frame_index"] == 1


def test_selecting_a_frame_row_opens_that_exact_frame(synthetic_result, monkeypatch):
    original_dataframe = st.dataframe

    def select_second_row(data, *args, **kwargs):
        rendered = original_dataframe(data, *args, **kwargs)
        if kwargs.get("key", "").startswith("frame_table_"):
            return SimpleNamespace(selection=SimpleNamespace(rows=[1]))
        return rendered

    monkeypatch.setattr(st, "dataframe", select_second_row)
    app = AppTest.from_file(str(ROOT / "app.py"))
    app.session_state["result"] = synthetic_result
    app.session_state["frame_index"] = 0
    app.session_state["playback_time"] = 0.0
    app.session_state["interval_selection"] = None
    app.run()
    assert not app.exception and app.session_state["frame_index"] == 1
