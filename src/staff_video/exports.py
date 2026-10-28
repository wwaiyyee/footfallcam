from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import zipfile

from .temporal import inclusive_intervals


FRAME_COLUMNS = ["frame_index", "timestamp_seconds", "duration_seconds", "detected_people", "tag_observed_count", "staff_count", "staff_present"]
DETECTION_COLUMNS = ["frame_index", "timestamp_seconds", "tracker_id", "track_id", "status", "staff_present",
                     "tag_observed", "detector_score", "tag_score", "x1", "y1", "x2", "y2", "center_x",
                     "center_y", "bottom_center_x", "bottom_center_y", "tag_x1", "tag_y1", "tag_x2", "tag_y2"]
INTERVAL_COLUMNS = ["start_frame", "end_frame", "start_seconds", "end_seconds_exclusive", "frame_count"]


def sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows, columns):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def detection_rows(frames):
    for frame in frames:
        for person in frame["people"]:
            row = {**person, "frame_index": frame["frame_index"], "timestamp_seconds": frame["timestamp_seconds"]}
            row.update(zip(["x1", "y1", "x2", "y2"], person["box"]))
            row.update(zip(["tag_x1", "tag_y1", "tag_x2", "tag_y2"], person["tag_box"] or [None] * 4))
            yield row


def frame_rows(frames):
    for frame in frames:
        yield {**frame, "detected_people": len(frame["people"])}


def export_tables(directory: Path, frames: list[dict], summary: dict):
    write_csv(directory / "frames.csv", frame_rows(frames), FRAME_COLUMNS)
    write_csv(directory / "detections.csv", detection_rows(frames), DETECTION_COLUMNS)
    intervals = inclusive_intervals(frames)
    write_csv(directory / "intervals.csv", intervals, INTERVAL_COLUMNS)
    tracks = sorted({person["track_id"] for frame in frames for person in frame["people"] if person["staff_present"]})
    track_intervals = [interval for track in tracks for interval in inclusive_intervals(frames, track)]
    write_csv(directory / "track_intervals.csv", track_intervals, ["track_id", *INTERVAL_COLUMNS])
    (directory / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (directory / "records.json").write_text(json.dumps(frames, allow_nan=False) + "\n")
    (directory / "EXPORT_README.txt").write_text(
        ("SYNTHETIC TEST FIXTURE; not a real analysis.\n" if summary.get("synthetic") else "Real local analysis; no synthetic detections.\n") +
        "Frame indices are zero-based; interval start/end frames are inclusive.\n"
        "Time interval ends are exclusive. Coordinates are source pixels, origin top-left.\n"
        "Person boxes are [x1,y1,x2,y2), center is a box center, NOT a floor-plane location.\n"
        "Detections CSV includes all detected people. Filter staff_present=True for staff coordinates.\n"
        "Uncertain means no confirmed tag-based staff identity, not a guaranteed non-staff person.\n"
        "Tag scores are template correlation, detector scores are detector output, neither is measured accuracy.\n"
        "A confirmed track is retained only across bounded tracking gaps; no boxes are invented in missing frames.\n"
        "Track count is NOT a guaranteed unique employee count. Annotations are silent constant-frame-rate H.264.\n"
        "Source timestamps and exact frame indices in CSV/records are authoritative, not playback seeking.\n"
    )


def build_zip(directory: Path):
    target = directory / "artifacts.zip"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path != target and not path.name.startswith("."):
                archive.write(path, path.relative_to(directory))
    return target
