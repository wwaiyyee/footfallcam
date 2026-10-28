from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import time
from typing import Callable
from uuid import uuid4

import cv2
import yaml

from .config import Config
from .detector import PersonTracker
from .evaluation import evaluate_labels, evaluate_people
from .exports import build_zip, export_tables, sha256
from .overlay import annotate
from .tag import TagMatcher
from .temporal import StaffResolver, inclusive_intervals
from .video import browser_encode, decoded_frames, inspect_video


@dataclass
class AnalysisResult:
    source: Path
    directory: Path
    summary: dict
    frames: list[dict]

    @property
    def intervals(self):
        return inclusive_intervals(self.frames)


def load_result(directory: Path | str, source: Path | str) -> AnalysisResult:
    directory, source = Path(directory).resolve(), Path(source).resolve()
    if (directory / "FAILED.txt").exists():
        raise ValueError("This run did not complete.")
    summary = json.loads((directory / "summary.json").read_text())
    frames = json.loads((directory / "records.json").read_text())
    if sha256(source) != summary["source_sha256"]:
        raise ValueError("This saved run belongs to a different video (source checksum mismatch).")
    if not frames or len(frames) != summary["processed_frames"] or any(
        record["frame_index"] != index for index, record in enumerate(frames)
    ):
        raise ValueError("Saved frame records are incomplete.")
    if not (directory / "annotated.mp4").is_file() or not (directory / "artifacts.zip").is_file():
        raise ValueError("Saved video or export ZIP is missing.")
    return AnalysisResult(source, directory, summary, frames)


def tracker_configuration(config: Config) -> dict:
    return {"tracker_type": "bytetrack", "track_high_thresh": max(.25, config.detector_threshold),
            "track_low_thresh": min(.1, config.detector_threshold),
            "new_track_thresh": max(.25, config.detector_threshold),
            "track_buffer": config.tracker_buffer, "match_thresh": config.tracker_match_threshold,
            "fuse_score": True}


def analyze_video(source: Path | str, output_root: Path | str, config: Config,
                  progress: Callable[[dict], None] | None = None, reference_labels=None,
                  person_labels=None) -> AnalysisResult:
    started = time.perf_counter()
    source = Path(source).resolve()
    metadata = inspect_video(source)
    directory = Path(output_root).resolve() / f"run-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    directory.mkdir(parents=True)
    frames = []

    def emit(stage, done=0, total=metadata.reported_frames):
        if progress:
            progress({"stage": stage, "processed_frames": done, "total_frames": total,
                      "elapsed_seconds": time.perf_counter() - started})

    try:
        emit("Loading local model and tag reference")
        model_hash = sha256(config.model_path)
        manifest_path = Path(config.model_path).with_suffix(".json")
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest["sha256"] != model_hash:
                raise ValueError("Model checksum does not match its setup manifest.")
        tracker_file = directory / "bytetrack.yaml"
        tracker_file.write_text(yaml.safe_dump(tracker_configuration(config)))
        detector = PersonTracker(config, tracker_file, metadata.fps)
        matcher = TagMatcher(config)
        resolver = StaffResolver(config)
        for index, timestamp, frame in decoded_frames(source, metadata.fps):
            people = detector.detect(frame)
            for person in people:
                match = matcher.match(frame, person["box"])
                person.update(tag_score=match.score, tag_box=match.box)
            people = resolver.resolve(index, people)
            count = sum(person["staff_present"] for person in people)
            frames.append({"frame_index": index, "timestamp_seconds": timestamp,
                           "duration_seconds": 1 / metadata.fps, "people": people,
                           "staff_present": bool(count), "staff_count": count,
                           "tag_observed_count": sum(person["tag_observed"] for person in people)})
            if index % 5 == 0:
                emit("Detecting people, matching tags, and tracking", index + 1)
        if not frames:
            raise ValueError("No frames were decoded.")
        if metadata.reported_frames and len(frames) < metadata.reported_frames:
            raise ValueError(f"Video stopped decoding at {len(frames)} of {metadata.reported_frames} reported frames. No complete result was exported.")
        for previous, current in zip(frames, frames[1:]):
            previous["duration_seconds"] = current["timestamp_seconds"] - previous["timestamp_seconds"]
        emit("Rendering annotations", 0, len(frames))
        temporary_video = directory / ".intermediate.mp4"
        writer = cv2.VideoWriter(str(temporary_video), cv2.VideoWriter_fourcc(*"mp4v"), metadata.fps,
                                 (metadata.width, metadata.height))
        if not writer.isOpened():
            raise RuntimeError("OpenCV could not create the annotated video.")
        observed = [record for record in frames if any(person["status"] == "Tag observed" for person in record["people"])]
        tracked = [record for record in frames if any(person["status"] == "Staff tracked" for person in record["people"])]
        example_indices = {0, len(frames) // 2, len(frames) - 1}
        if observed:
            example_indices.add(max(observed, key=lambda record: max(person["tag_score"] for person in record["people"] if person["staff_present"]))["frame_index"])
        if tracked:
            example_indices.add(tracked[len(tracked) // 2]["frame_index"])
        (directory / "examples").mkdir()
        rendered = 0
        try:
            for index, _, frame in decoded_frames(source, metadata.fps):
                if index >= len(frames):
                    raise ValueError("Source changed between detection and rendering.")
                annotated = annotate(frame, frames[index])
                writer.write(annotated)
                rendered += 1
                if index in example_indices:
                    cv2.imwrite(str(directory / "examples" / f"frame_{index:06d}_source.jpg"), frame)
                    cv2.imwrite(str(directory / "examples" / f"frame_{index:06d}_annotated.jpg"), annotated)
                if index % 25 == 0:
                    emit("Rendering annotations", index + 1, len(frames))
        finally:
            writer.release()
        if rendered != len(frames):
            raise ValueError("Second decode did not reproduce every processed frame.")
        emit("Encoding browser-compatible H.264", len(frames), len(frames))
        browser_encode(temporary_video, directory / "annotated.mp4")
        temporary_video.unlink()
        tracks = {person["track_id"] for record in frames for person in record["people"] if person["staff_present"]}
        warnings = ["No reference labels supplied; accuracy has not been measured."] if reference_labels is None and person_labels is None else []
        if person_labels is None:
            warnings.append("Person, staff, and visible-tag precision/recall have not been measured against labelled boxes.")
        if not tracks:
            warnings.append("No staff track was confirmed. This is not proof that staff are absent; inspect detector coverage and tag scale/threshold.")
        if metadata.ffprobe is None:
            warnings.append("FFprobe unavailable; decoded metadata comes from OpenCV. H.264 uses bundled FFmpeg when needed.")
        warnings.append("Overhead distortion, small tags, occlusion, similar patterns, and tracker identity switches can cause errors.")
        summary = {
            "schema_version": 1, "synthetic": False, "created_utc": datetime.now(timezone.utc).isoformat(),
            "source": metadata.to_dict(), "source_sha256": sha256(source), "settings": config.to_dict(),
            "model": {"filename": Path(config.model_path).name, "sha256": model_hash},
            "tag_reference_sha256": sha256(config.tag_reference), "tracker": tracker_configuration(config),
            "tag_exemplars": [{"filename": path.name, "sha256": sha256(path)} for path in matcher.reference_paths],
            "processed_frames": len(frames), "staff_present_frames": sum(record["staff_present"] for record in frames),
            "staff_present_seconds": sum(record["duration_seconds"] for record in frames if record["staff_present"]),
            "decoded_duration_seconds": frames[-1]["timestamp_seconds"] + frames[-1]["duration_seconds"],
            "staff_track_count": len(tracks), "staff_track_ids": sorted(tracks),
            "tag_observed_frames": sum(record["tag_observed_count"] > 0 for record in frames),
            "coordinate_convention": "Source pixels, origin top-left; half-open xyxy boxes; xy is person-box center, not world position.",
            "frame_convention": "Zero-based, inclusive frame intervals; exclusive time interval ends. No retroactive staff labels before confirmation.",
            "video_export": "Silent H.264 / yuv420p / constant nominal FPS. Use records for source timestamps and exact frames.",
            "runtime": {"python": platform.python_version(), **{name: importlib.metadata.version(name)
                        for name in ["ultralytics", "torch", "opencv-python", "numpy", "streamlit", "pandas", "plotly"]}},
            "warnings": warnings,
        }
        if reference_labels is not None:
            summary["evaluation"] = evaluate_labels(frames, reference_labels)
        if person_labels is not None:
            summary["person_evaluation"] = evaluate_people(frames, person_labels, metadata.width, metadata.height)
        summary["processing_seconds"] = time.perf_counter() - started
        emit("Exporting results", len(frames), len(frames))
        export_tables(directory, frames, summary)
        build_zip(directory)
        emit("Complete", len(frames), len(frames))
        return AnalysisResult(source, directory, summary, frames)
    except Exception as error:
        (directory / "FAILED.txt").write_text(f"Incomplete run; do not use partial artifacts as results.\n{type(error).__name__}: {error}\n")
        raise
