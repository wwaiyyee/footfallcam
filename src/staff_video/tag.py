from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import cv2
import numpy as np

from .config import Config


@dataclass(frozen=True)
class TagMatch:
    score: float
    box: tuple[int, int, int, int] | None


def rotate_bound(image, angle):
    height, width = image.shape
    radians = math.radians(angle)
    new_width = int(math.ceil(abs(width * math.cos(radians)) + abs(height * math.sin(radians))))
    new_height = int(math.ceil(abs(height * math.cos(radians)) + abs(width * math.sin(radians))))
    transform = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    transform[0, 2] += (new_width - width) / 2
    transform[1, 2] += (new_height - height) / 2
    return cv2.warpAffine(image, transform, (new_width, new_height), borderValue=128)


class TagMatcher:
    def __init__(self, config: Config):
        reference = cv2.imread(str(Path(config.tag_reference)), cv2.IMREAD_GRAYSCALE)
        if reference is None or reference.std() < 15:
            raise ValueError("Missing or low-contrast tag reference. Supply a clear tag-only PNG through the tag_reference setting.")
        reference_path = Path(config.tag_reference)
        self.reference_paths = [reference_path] + sorted(reference_path.parent.glob(f"{reference_path.stem}_[0-9]*.png"))
        self.templates = []
        for path in self.reference_paths:
            reference = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if reference is None or reference.std() < 15:
                raise ValueError(f"Invalid tag exemplar: {path.name}")
            for width in range(config.tag_min_width, config.tag_max_width + 1, config.tag_width_step):
                height = max(6, round(width * reference.shape[0] / reference.shape[1]))
                resized = cv2.resize(reference, (width, height), interpolation=cv2.INTER_AREA)
                for angle in range(0, 360, config.tag_rotation_step):
                    template = rotate_bound(resized, angle)
                    self.templates.append(template)

    def match(self, frame, person_box) -> TagMatch:
        x1, y1, x2, y2 = person_box
        region = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        if region.size == 0:
            return TagMatch(0.0, None)
        best_score = 0.0
        best_box = None
        for template in self.templates:
            height, width = template.shape
            if height > region.shape[0] or width > region.shape[1]:
                continue
            scores = cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED)
            _, score, _, location = cv2.minMaxLoc(scores)
            left, top = location
            candidate = region[top:top + height, left:left + width]
            if candidate.std() < 25 or candidate.max() - int(candidate.min()) < 80:
                continue
            if np.isfinite(score) and score > best_score:
                best_score = float(score)
                best_box = (x1 + left, y1 + top, x1 + left + width, y1 + top + height)
        return TagMatch(best_score, best_box)
