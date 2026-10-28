from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import yaml

from .config import Config, ROOT


class PersonTracker:
    def __init__(self, config: Config, tracker_file: Path, fps: float = 25):
        if not Path(config.model_path).is_file():
            raise FileNotFoundError("Local model weights are missing. Run scripts/setup_model.py; inference never downloads weights.")
        runtime = ROOT / ".runtime/ultralytics"
        runtime.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(runtime))
        from ultralytics import YOLO
        from ultralytics.trackers.byte_tracker import BYTETracker

        self.model = YOLO(config.model_path, task="detect")
        self.config = config
        self.tracker_file = tracker_file
        self.tracker_settings = yaml.safe_load(tracker_file.read_text())
        self.tracker = BYTETracker(SimpleNamespace(**self.tracker_settings), frame_rate=round(fps))

    def detect(self, frame) -> list[dict]:
        from ultralytics.engine.results import Boxes

        height, width = frame.shape[:2]
        candidates = []
        for angle in self.config.detector_rotations:
            rotated = np.ascontiguousarray(np.rot90(frame, angle // 90))
            result = self.model.predict(
                rotated, classes=[0], conf=self.config.detector_threshold, iou=self.config.detector_iou,
                imgsz=self.config.image_size, device=self.config.device, verbose=False,
            )[0]
            if result.boxes is None:
                continue
            for values in result.boxes.data.cpu().numpy():
                box = restore_box(values[:4], angle, width, height)
                candidates.append([*box, float(values[4]), 0])
        candidates = np.asarray(candidates, dtype=np.float32).reshape(-1, 6)
        if len(candidates):
            xywh = candidates[:, :4].copy()
            xywh[:, 2:] -= xywh[:, :2]
            keep = cv2.dnn.NMSBoxes(xywh.tolist(), candidates[:, 4].tolist(),
                                    self.config.detector_threshold, self.config.detector_iou)
            candidates = candidates[np.asarray(keep).reshape(-1).astype(int)]
        tracked = self.tracker.update(Boxes(candidates, (height, width)), frame)
        identities = track_identities(tracked, candidates, self.tracker_settings["track_high_thresh"],
                                      self.tracker_settings["track_low_thresh"])
        people = []
        for index, candidate in enumerate(candidates):
            box, score, identity = candidate[:4], candidate[4], identities.get(index)
            x1, y1, x2, y2 = [int(round(float(value))) for value in box]
            x1, x2 = max(0, min(x1, width - 1)), max(0, min(x2, width))
            y1, y2 = max(0, min(y1, height - 1)), max(0, min(y2, height))
            if x2 <= x1 or y2 <= y1:
                continue
            people.append({"tracker_id": None if identity is None else int(identity),
                           "box": [x1, y1, x2, y2], "detector_score": float(score),
                           "center_x": (x1 + x2) / 2, "center_y": (y1 + y2) / 2,
                           "bottom_center_x": (x1 + x2) / 2, "bottom_center_y": y2})
        return people


def restore_box(box, angle, width, height):
    left, top, right, bottom = [float(value) for value in box]
    if angle == 0:
        return [left, top, right, bottom]
    if angle == 90:
        return [width - bottom, left, width - top, right]
    if angle == 180:
        return [width - right, height - bottom, width - left, height - top]
    return [top, height - right, bottom, height - left]


def track_identities(tracked, candidates, high_threshold, low_threshold):
    high_indices = np.flatnonzero(candidates[:, 4] >= high_threshold)
    low_indices = np.flatnonzero((candidates[:, 4] > low_threshold) & (candidates[:, 4] < high_threshold))
    identities = {}
    for row in tracked:
        indices = high_indices if row[5] >= high_threshold else low_indices
        local_index = int(row[-1])
        if not 0 <= local_index < len(indices):
            raise RuntimeError("ByteTrack returned an invalid subset detection index.")
        identities[int(indices[local_index])] = int(row[4])
    return identities
