# Staff identification video system

A local Streamlit workspace and CLI for finding **tag-confirmed staff** in overhead retail video. Both use the same Python pipeline. No authentication, database, cloud inference, facial recognition, or separate frontend.

The supplied build instructions are preserved in `AGENTS.md`. The pasted specification ends partway through its suggested directory tree; the implementation follows all supplied requirements.

## In plain English

1. Upload a video (or choose the local sample), then click **Analyze video**. Uploading alone does not start processing.
2. The app checks **every decoded frame**: it finds people, looks for the specified black-and-white name tag inside their boxes, and tracks tag-confirmed people across frames.
3. Watch the annotated video or inspect any exact frame. Green means a tag match is visible now; blue means the same staff track was confirmed earlier; grey means **uncertain**, not definitely non-staff.
4. Download `frames.csv` for one result per frame, `detections.csv` for person/staff image coordinates, and `artifacts.zip` for the full result. The CLI provides the same analysis for a new video.

The system exports evidence and results, but does **not** generate a written report. Write any assessment report yourself.

YOLO11s is the default because it gives better person coverage than the original small baseline while taking about **half the time of YOLO11m** in our 100-frame test. Neither model has proven accuracy on unseen videos; labelled evaluation is needed before operational use.

## Quick start

Use **Python 3.9-3.12** (Python 3.11 recommended for a new environment; the supplied laptop was tested with Python 3.9.6 on Apple Silicon). Python 3.13/3.14 are not supported by these pinned Torch versions.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.lock
python -m pip install --no-deps -e '.[dev]'
python scripts/setup_model.py
python -m streamlit run app.py
```

If your supported interpreter is named `python3`, use that instead of `python3.11`. The checked-in lock records the tested dependency closure, while `pyproject.toml` pins the portable direct dependencies. On another OS, install with `python -m pip install -e '.[dev]'` if a platform-specific transitive lock dependency is unavailable. Do not use the system Python 3.14 conda environment with this lock.

Setup needs internet access for dependencies and the 18 MB default YOLO11s detector. Inference uses **local weights only** and does not automatically download a missing model. Setup records its SHA-256 in `models/yolo11s.json`; subsequent setup and every analysis verify that checksum. To try the other modes, run `python scripts/setup_model.py --variant n` or `python scripts/setup_model.py --variant m`. Treat custom `.pt` model files as trusted executable input, not arbitrary user uploads. Review Ultralytics' AGPL-3.0/enterprise license before redistribution or commercial integration.

The three small tag-only exemplars in `assets/` are included; no external reference document is needed to run the system. Their hashes and crop provenance are recorded in `assets/tag_reference.json`. To analyze footage with a different name tag, supply a clear tag-only PNG through the `tag_reference` setting. Two exemplar patches come from snapshots of the development sample, so results on that sample are **not independent validation**.

### Browser workflow

1. Use the configured local sample or upload an MP4. Inspect dimensions, duration, and the source preview.
2. Optionally expand advanced settings. YOLO11s with four orientations is the default; YOLO11n is faster but misses more people in sampled frames, and YOLO11m is slower. Select **Analyze video** explicitly; uploading does not run inference.
3. Review the annotated H.264 video, presence timeline, inclusive intervals, and summary. Selecting an interval updates playback start and the exact frame selection.
4. Use **Exact frame inspector** to browse the table with one row per decoded frame, including zero-detection frames. Filter by staff presence, current tag match, or no detected people; selecting a row opens its source and annotated images. The person table reports each detected person's track ID, current tag match, staff state, and image coordinates. Zero-based frame number and previous/next controls remain available. The inspector sequentially decodes the source so keyframe seeking is never treated as exact.
5. Download frame/coordinate/interval CSVs, full frame records JSON, summary/settings JSON, video, or the complete artifact ZIP. Advanced setting changes do not mutate an existing result or rerun inference.

Completed results persist across normal Streamlit reruns. After restarting the app, use **Reopen a completed local run** to load disk artifacts without inference; the source checksum must match. The original uploaded file remains in `outputs/uploads/`; do not remove it if you still need exact source-frame inspection. Nothing is sent to a server beyond this local Streamlit process.

### Run an unseen video

```bash
staff-video inspect /path/to/unseen.mp4
staff-video analyze /path/to/unseen.mp4 --output outputs
staff-video analyze /path/to/unseen.mp4 --config config/default.yaml --tag-threshold 0.88
```

An alternative without installing the entrypoint is `PYTHONPATH=src python -m staff_video.cli analyze /path/to/unseen.mp4`. Use `--model`, `--tag-reference`, or `--device cpu|mps|0` to override paths/device. CPU is the tested default. `mps` and CUDA device `0` are optional and have not been benchmarked here. YAML paths resolve relative to the YAML file, not the working directory; CLI path overrides resolve relative to the current directory. Change `sample_path` in the config to select another local sample in the UI. No source paths or timestamps determine staff identity.

## How it works

```text
Streamlit / CLI -> shared synchronous pipeline
  decode every frame + timestamps
  -> YOLO11s person detection on four rotated views by default
  -> map boxes to source pixels + cross-view nonmaximum suppression
  -> one explicitly configured Ultralytics BYTETracker (bytetrack.yaml)
  -> multiscale, rotated grayscale tag-template matching in each person box
  -> repeated tag observations confirm a staff track
  -> frame labels / image coordinates / CSV / JSON / annotated H.264 / ZIP
```

**Why rotated person detection?** Initial visual inspection found upright-only detection missed inverted people in the overhead view. The default uses YOLO11s on 0°, 90°, 180°, and 270° views, merges results, and updates ByteTrack once per source frame. YOLO11m uses the same views at about twice the runtime in the tested excerpt; YOLO11n uses only 0°/180°. More candidates can include duplicate/partial boxes and are **not** evidence of high precision or recall. The tracker is configured from the exported `bytetrack.yaml`, not Ultralytics' changeable default tracker. Its native tracker buffer is scaled by source FPS; `max_track_gap` is a separate strict source-frame limit for retaining staff identity.

**Tag evidence.** Match the distinctive black-and-white pattern using normalized cross-correlation across tag widths and rotations; reject low-contrast patches. The maximum score and candidate box are exported for every person. Defaults require at least two matches at or above **0.88** within 20 frames on the same track. The conservative threshold was selected after inspecting sample candidates; it is a tunable similarity threshold, not a calibrated probability or measured accuracy. Neither clothing colour, faces, nor filenames classify staff.

**Temporal policy.** Confirmation is causal: earlier frames are not retrospectively labelled staff. A confirmed identity remains attached to its active track even when the tag is hidden. Retain it across at most 15 source-frame gaps by default, then require new tag evidence. Track generations prevent a stale staff label from leaking to a recycled ID. Missing detections produce no person box or staff-present claim for that frame. A track switch can fragment one employee into multiple IDs or, incorrectly, transfer identity; this is a known baseline limitation.

| Overlay | Meaning |
| --- | --- |
| Tag observed (green) | Confirmed staff identity and a current matching tag |
| Staff tracked (blue) | Previously tag-confirmed, currently visible on the same track |
| Uncertain (grey) | Not enough tag evidence; **not** a guarantee of non-staff |
| Yellow small box | Current tag candidate, including pending confirmation |

## Output contract

Each independent run gets an `outputs/run-<UTC>-<id>/` directory:

| Artifact | Contents |
| --- | --- |
| `frames.csv` | One row per processed frame, including zero-detection frames: timestamp, duration, detected person count, current tag match count, staff presence/count |
| `detections.csv` | All detected people, track IDs, status, raw scores, tag box, person box, coordinates; filter `staff_present=True` for staff |
| `intervals.csv` | Union of staff-present frames as inclusive frame intervals |
| `track_intervals.csv` | Inclusive intervals for each staff track generation |
| `summary.json` | Statistics, settings, source/model/reference hashes, exact runtime versions, warnings, optional evaluation |
| `records.json` | Full per-frame records used by the inspector and saved-result reopening |
| `annotated.mp4` | Silent H.264/yuv420p video with fast-start metadata |
| `examples/` | Exact source/annotated pairs, including strongest observed evidence when present |
| `bytetrack.yaml` | Exact tracker configuration used |
| `artifacts.zip` | All final artifacts above, without source video or model weights |

Frames are **zero-based**; `start_frame` and `end_frame` are inclusive. Time interval ends are exclusive. Pixel coordinates have a top-left origin. Person boxes are half-open `[x1,y1,x2,y2)`; `center_x,center_y` are bounding-box centres. Bottom-centre coordinates are also supplied, but **neither is a calibrated floor-plane/world position**, especially in this fisheye view. Track count is not a guaranteed count of unique employees.

Staff-present duration sums the actual labelled frame durations without double-counting simultaneous staff. Source timestamps use decoded presentation positions with a nominal-FPS fallback when unavailable/nonmonotonic. The annotated video uses constant nominal FPS, so playback timing may differ on variable-frame-rate inputs. The frame records and exact inspector remain authoritative. A detected truncated decode fails the run rather than exporting a falsely complete analysis; failure directories contain `FAILED.txt`.

FFprobe provides codec/container metadata when installed. OpenCV provides decoded metadata otherwise. System FFmpeg is preferred for H.264; the pinned `imageio-ffmpeg` binary is the fallback. Install system FFmpeg/FFprobe with your OS package manager for fuller metadata. No fabricated detections, demonstration scores, or accuracy metrics are inserted into real results.

### Reference-label evaluation

Supply a CSV with `frame_index,staff_present`, where presence is 0 or 1 and indices are unique valid integers. Partial labels are accepted and coverage is reported; unlabelled frames are excluded.

```bash
staff-video analyze another.mp4 --labels labels.csv
```

The UI also evaluates an uploaded label CSV against the completed run without inference and offers evaluation JSON. It does not rewrite the original analysis ZIP; the CLI's `--labels` includes evaluation in the summary/ZIP. Precision, recall, F1, accuracy, and confusion counts exist **only with reference labels**. Undefined ratios are null, not invented zeroes. These are frame-presence metrics, not person-location or track-identity accuracy.

To measure person detection, staff identification, and visible-tag detection separately, annotate **every visible person** in each selected frame using source-pixel boxes and a CSV headed `frame_index,x1,y1,x2,y2,is_staff,tag_visible`. The two flags are 0 or 1; `tag_visible` means the specified tag is visibly present on that person, whether or not the algorithm found it. For an annotated frame containing no people, include one row with `frame_index` and all remaining cells blank. Omit unannotated frames. The evaluator matches boxes at IoU ≥ 0.5 and reports TP/FP/FN, precision, recall, F1, and annotated-frame coverage for all three stages. A missed person with a visible tag counts against both person and tag recall. Track identity continuity is not scored. Upload this CSV under **Person and tag labels** in the UI, or run:

```bash
staff-video analyze another.mp4 --person-labels person_labels.csv
```

The CLI stores these metrics in `summary.json` and the ZIP; UI post-run evaluation does not modify saved artifacts. Label several **unseen** videos with varied lighting, camera angles, crowds, and tag visibility before choosing an operational threshold. Do not tune and validate on the same footage.

## Verification

```bash
python -m pytest -q
```

Tests cover tag-vs-solid-patch matching, rotated coordinates, temporal confirmation/expiry, inclusive intervals, malformed labels, CSV/ZIP contracts, exact decoding, real H.264 conversion, model checksum failures, and Streamlit rerun persistence. Synthetic data is confined to explicitly named temporary test fixtures and visibly marked when injected into the UI.

### Submission handoff

For the assessment, write the two-page report yourself and provide it separately from this codebase and `outputs/run-20261028T113438Z-1e8c9ec8/artifacts.zip`. In that completed YOLO11s run, `frames.csv` is the required frame-by-frame answer; `detections.csv` contains bonus coordinates; `annotated.mp4` and `examples/` show visual evidence you can review. The app and CLI never write report prose or a report PDF. Keep `models/yolo11s.pt` and its checksum manifest locally for an offline interview; `.pt` weights are ignored by Git. Run `python -m streamlit run app.py` or `staff-video analyze /path/to/new.mp4 --output outputs` on the interview video. Results are algorithm estimates, not manually verified ground truth.

## Limits and next steps

- This is an interpretable laptop baseline with one supplied video, not a trained or validated production staff detector. No labelled training dataset or complete reference frame labels were supplied.
- Small/blurred tags, extreme perspective, partial occlusion, and lookalike desk/clothing patterns can cause misses or false positives. Bounding boxes include background, so person association is not perfect.
- Pretrained person detection has limited overhead/fisheye coverage even with rotation augmentation. Do not equate no confirmed output with staff absence. Coverage mode costs more CPU time and can introduce false/duplicate boxes.
- Causal confirmation intentionally misses the early part of a person's appearance until evidence is observed. Dropped detections create gaps rather than invented locations.
- On new videos, inspect person coverage and tag candidates before tuning thresholds. Independently annotate held-out videos and person boxes before quoting accuracy; consider a dedicated tag detector and overhead person fine-tuning only when real training data becomes available.
- Video, uploads, and exports remain on local disk and may contain personal data. Apply your organisation's retention/access rules; remove `outputs/` manually when no longer needed.

Code is under `src/staff_video/`; Streamlit is imported only by `app.py`, never by the processing package.

## Verified sample run

The current-default YOLO11s/four-view run is `outputs/run-20261028T113438Z-1e8c9ec8/`. It processed **all 1,341 frames** of `sample.mp4` and exported 1,341 rows in `frames.csv`, 15,340 detected-person boxes, an annotated H.264 video, exact frame records, and an artifact ZIP. The algorithm marked **39 staff-present frames / 1.56 seconds** across **4 track IDs**, with **33 frames containing a tag match**. Its inclusive staff intervals are **858-864, 1015-1029, 1117-1121, and 1199-1210**. End-to-end processing took **1,452.2 seconds (24.2 minutes)** on CPU. These are detector outputs, not verified ground truth: person boxes still miss or duplicate people, staff tracks fragment, and tag precision/recall have not been measured. This run is a reproducible assignment answer, **not a claim that every staff frame was found**.

The completed, corrected development run is `outputs/run-20261026T095036Z-17654fd5/` (generated locally, not version-controlled). It used the **earlier YOLO11n/two-view baseline**, not the new default. All **1,341 frames** decoded and produced a browser-compatible annotated export. That baseline labelled **39 frames / 1.56 seconds** staff-present across **4 track IDs**, in **781.8 seconds** through encoding on the supplied CPU setup. These are algorithm outputs, not reference truth: visual review shows limited overhead coverage and fragmented tracks, so this is **not a complete ground-truth answer or a production-ready accuracy claim**. Two analyses ran concurrently for part of this test; the recorded processing time is not an isolated-device benchmark.

On the first 100 frames of the original source, the old baseline exported 399 person candidates and no staff-present frames. A coverage-mode run on a re-encoded 100-frame excerpt exported 922 person candidates, 14 staff-present frames, and 9 frames with a tag match in about 81 seconds. This is a directional comparison, **not a controlled accuracy benchmark**, because the excerpt was re-encoded.

We then ran **YOLO11s and YOLO11m on that exact same 100-frame excerpt** (matching source SHA-256, thresholds, four orientations, tracker, and tag reference; only the detector weights changed):

| Detector | Person boxes across 100 frames | Frames with a tag match | Staff-present frames | Processing time |
| --- | ---: | ---: | ---: | ---: |
| YOLO11s | 922 | 9 | 14 | 80.7 s |
| YOLO11m | 1,224 | 9 | 14 | 158.9 s |

YOLO11m produced 302 more boxes (33%) and took about 1.97× as long. Visual comparison on sampled frames shows some additional people boxed, but some boxes are duplicates or partial and some people remain unboxed. The tag/staff frame sets were **identical** for this excerpt. These are output counts, **not person, tag, or staff precision/recall**; there are no independent labels for this video. YOLO11s remains the default for a better speed/coverage tradeoff; YOLO11m is optional. Neither model has been validated on other camera footage.

The test suite passes in a separately installed, clean environment, and `pip check` reports no broken requirements. The earlier `run-20261026T094123Z-52984c5f` was explicitly invalidated after a ByteTrack subset-index mapping correction; it cannot be reopened as a completed result.
