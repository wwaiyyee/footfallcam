from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
import math

import yaml


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Config:
    model_path: str = str(ROOT / "models/yolo11s.pt")
    tag_reference: str = str(ROOT / "assets/tag_reference.png")
    sample_path: str = str(ROOT / "sample.mp4")
    detector_threshold: float = 0.20
    detector_iou: float = 0.50
    detector_rotations: tuple = (0, 90, 180, 270)
    image_size: int = 960
    device: str = "cpu"
    tag_threshold: float = 0.88
    tag_min_width: int = 12
    tag_max_width: int = 36
    tag_width_step: int = 4
    tag_rotation_step: int = 30
    confirmation_hits: int = 2
    confirmation_window: int = 20
    max_track_gap: int = 15
    tracker_buffer: int = 15
    tracker_match_threshold: float = 0.80

    def __post_init__(self):
        if not self.detector_rotations or len(set(self.detector_rotations)) != len(self.detector_rotations) or any(
            angle not in (0, 90, 180, 270) for angle in self.detector_rotations
        ):
            raise ValueError("detector_rotations must contain unique quarter-turn angles.")
        for name in ("detector_threshold", "detector_iou", "tag_threshold", "tracker_match_threshold"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1].")
        for name in ("image_size", "tag_min_width", "tag_max_width", "tag_width_step",
                     "tag_rotation_step", "confirmation_hits", "confirmation_window",
                     "max_track_gap", "tracker_buffer"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.tag_max_width < self.tag_min_width:
            raise ValueError("tag_max_width must be at least tag_min_width.")
        if self.tag_rotation_step > 90 or 360 % self.tag_rotation_step:
            raise ValueError("tag_rotation_step must divide 360 and be at most 90.")
        if self.confirmation_hits > self.confirmation_window:
            raise ValueError("confirmation_hits cannot exceed confirmation_window.")

    def changed(self, **values):
        return replace(self, **values)

    def to_dict(self):
        return asdict(self)


def load_config(path: Path | str | None = None) -> Config:
    if path is None:
        return Config()
    path = Path(path).resolve()
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    unknown = set(data) - {field.name for field in fields(Config)}
    if unknown:
        raise ValueError(f"Unknown settings: {', '.join(sorted(unknown))}")
    for name in ("model_path", "tag_reference", "sample_path"):
        if name in data:
            data[name] = str((path.parent / data[name]).resolve())
    return Config(**data)
