# Staff Identification Video System — Build Instructions

## 1. Purpose and scope

Build a local application that identifies frames containing staff in a retail sensor video and locates each staff member. A person is identified as staff through the specified black-and-white name tag, then tracked while they remain visible.

Provide both:

1. An interactive browser interface for uploading, processing, inspecting, and exporting video results.
2. A command-line interface that runs the same processing pipeline on another video.

This document is an implementation specification, not an existing implementation or a claim of measured accuracy. Implement the work when asked to build the project. Keep this file at the repository root so a coding assistant can discover it.

### Assignment requirements

- Required: identify which frames contain staff.
- Bonus, included in this build: output staff image coordinates.
- The candidate writes the 1–2 page solution report personally. The system must not generate report prose or PDFs. Provide code, visual examples, and instructions for running an unseen video on a laptop.
- Use visual evidence from the tag. Clothing colour, faces, filenames, and memorized timestamps must not determine staff identity.

### Initial reference material

- Video: `/Users/chloelee/Downloads/sample.mp4`
- Reported video metadata: approximately 53.64 seconds, 960 × 720, nominally 25 fps.
- Video frames have not yet been successfully inspected in this planning session. The first build milestone must verify decoding and visually inspect the footage.
- Only one example video was initially supplied. Do not assume a training dataset exists.

These are initial input locations only. Accept any input path through configuration or upload; do not embed these absolute paths in application logic.

## 2. Product experience

Use a single, clear Streamlit application with three sequential areas: input, processing, and results. Default to local processing on a laptop.

### Input and processing

- Upload an MP4 or choose a configured local sample.
- Show the filename, dimensions, duration, and decoded metadata when available.
- Provide an explicit **Analyze video** button. Uploading alone must not launch expensive inference.
- Show processing progress, elapsed time, and an understandable error if decoding or model loading fails.
- Keep a completed result visible across ordinary Streamlit reruns. Changing inspection controls must not rerun inference.
- Keep advanced settings collapsed: detector thresholds, tag threshold, and tracking settings. Describe these as tunable values, not measured probabilities.
- Changed analysis settings take effect only on a new analysis run. Display the settings used for the current result.

### Results and interaction

- Display a browser-playable annotated video with person boxes, staff track IDs, and coordinates.
- Visually distinguish **Tag observed**, **Staff tracked**, and **Uncertain** using text as well as colour.
- Show a presence timeline and a table of inclusive staff frame intervals.
- Allow selecting an interval and opening playback near its start. If the player cannot seek reliably through its API, provide a timestamp control and an exact frame inspector.
- Provide an exact frame inspector with frame number, previous/next controls, source frame, and annotated frame. Video playback is not the source of truth for exact frame selection.
- Show summary statistics: processed frames, frames labelled staff-present, duration of those frames, staff track count, and processing time. Track count is not a guaranteed count of unique employees.
- Allow downloading CSV results, JSON summary/configuration, annotated video, and an artifact ZIP.
- Display evaluation metrics only when reference labels have been supplied.
- Never show invented detections or scores in a real analysis result. Any demonstration fixture must be visibly marked as synthetic.

Do not add authentication, a database, cloud deployment, or a separate JavaScript frontend to the initial build.

## 3. Technology and architecture

Use:

- Python for the application and processing pipeline.
- Streamlit for the local frontend.
- OpenCV and NumPy for image operations, reference matching, and overlays.
- A lightweight pretrained person detector through Ultralytics, with an explicitly configured ByteTrack tracker (`bytetrack.yaml`). Validate its performance on the overhead view.
- pandas for result tables and Plotly for the presence timeline.
- FFmpeg/FFprobe for inspecting codecs and producing a browser-compatible H.264 MP4 when available.
- pytest for meaningful pipeline, export, and temporal-logic checks.

Choose mutually compatible dependency versions during setup and record the versions actually tested. Provide a reproducible environment file. Download model weights during setup and store their identity and checksum; the interview demo should run using local weights.

```text
Streamlit UI ─┐
              ├─> Shared pipeline
CLI ──────────┘      ├─ Decode frames and timestamps
                     ├─ Detect and track people
                     ├─ Detect tag and associate it with a person
                     ├─ Resolve staff status across time
                     ├─ Calculate coordinates
                     └─ Export results and annotated video
```

Keep Streamlit imports out of the processing package. The UI calls the pipeline and consumes its progress events and completed artifacts. A synchronous pipeline with progress callbacks is sufficient initially.

### Suggested repository structure

```text
AGENTS.md
README.md
pyproject.toml
config/default.yaml
app.py
src/staff_video/
  __init__.py
  cli.py
