from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math
import shutil
import subprocess

import cv2


@dataclass(frozen=True)
class VideoInfo:
    filename: str
    width: int
    height: int
    fps: float
    reported_frames: int
    duration_seconds: float
    codec: str
    ffprobe: dict | None = None

    def to_dict(self):
        return asdict(self)


def inspect_video(path: Path | str) -> VideoInfo:
    path = Path(path)
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot decode {path.name}. Choose a readable video supported by OpenCV.")
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f"{path.name} has no decodable frames.")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Video has no valid frame rate; re-encode it with FFmpeg first.")
        count = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(chr((fourcc >> (8 * shift)) & 255) for shift in range(4))
        probe = None
        if shutil.which("ffprobe"):
            try:
                completed = subprocess.run(
                    ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams",
                     "-show_format", "-of", "json", str(path.resolve())],
                    capture_output=True, text=True, check=True, timeout=30,
                )
                probe = json.loads(completed.stdout)
            except (subprocess.SubprocessError, ValueError, OSError):
                probe = {"warning": "FFprobe failed; metadata is from OpenCV."}
        return VideoInfo(path.name, frame.shape[1], frame.shape[0], fps, count, count / fps, codec, probe)
    finally:
        capture.release()


def decoded_frames(path: Path | str, fps: float):
    capture = cv2.VideoCapture(str(path))
    previous_time = -1.0
    index = 0
    try:
        if not capture.isOpened():
            raise ValueError("Video decoder could not be opened.")
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if not math.isfinite(timestamp) or timestamp <= previous_time:
                timestamp = max(index / fps, previous_time + 1 / fps)
            yield index, timestamp, frame
            index += 1
            previous_time = timestamp
    finally:
        capture.release()


def exact_frame(path: Path | str, index: int):
    if index < 0:
        raise ValueError("Frame index must be nonnegative.")
    capture = cv2.VideoCapture(str(path))
    try:
        for current in range(index + 1):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Cannot decode frame {index}; stopped at frame {current}.")
        return frame
    finally:
        capture.release()


def browser_encode(source: Path, target: Path):
    executable = shutil.which("ffmpeg")
    if executable is None:
        import imageio_ffmpeg

        executable = imageio_ffmpeg.get_ffmpeg_exe()
    try:
        subprocess.run(
            [executable, "-y", "-v", "error", "-i", str(source), "-an", "-c:v", "libx264",
             "-preset", "fast", "-crf", "21", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)],
            check=True, capture_output=True, timeout=1800,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"H.264 conversion failed: {error.stderr.decode(errors='replace')[-1500:]}") from error
