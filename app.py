from __future__ import annotations

import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import cv2
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from staff_video.config import ROOT, load_config
from staff_video.evaluation import evaluate_labels, evaluate_people
from staff_video.exports import frame_rows
from staff_video.overlay import annotate
from staff_video.pipeline import analyze_video, load_result
from staff_video.video import exact_frame, inspect_video


st.set_page_config(page_title="Staff video analysis", page_icon="◉", layout="wide")
st.markdown("""<style>
.block-container {max-width:1380px;padding-top:2rem;padding-bottom:3rem}
h1 {letter-spacing:-1.5px;font-weight:750!important} h2 {letter-spacing:-.6px}
[data-testid="stMetric"] {background:white;border:1px solid #e1e7ec;border-radius:12px;padding:16px}
[data-testid="stMetricLabel"] {color:#657584}
[data-testid="stFileUploader"] {border-radius:12px}
.eyebrow {font-size:12px;font-weight:700;letter-spacing:2px;color:#087f8c;margin-bottom:10px}
.intro {color:#657584;max-width:850px;font-size:17px;line-height:1.6}
.pill {display:inline-block;padding:5px 11px;border:1px solid #dbe5e8;border-radius:20px;
font-size:12px;color:#526675;background:white;margin-right:6px}
</style>""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def metadata_for(path, size, modified):
    return inspect_video(path).to_dict()


@st.cache_data(show_spinner=False, max_entries=8)
def inspect_exact(path, index, modified):
    return exact_frame(path, index)


def move_frame(amount):
    result = st.session_state.result
    st.session_state.frame_index = min(len(result.frames) - 1, max(0, st.session_state.frame_index + amount))


def choose_interval():
    intervals = st.session_state.result.intervals
    chosen = st.session_state.interval_selection
    if chosen is not None:
        interval = intervals[chosen]
        st.session_state.frame_index = interval["start_frame"]
        st.session_state.playback_time = float(interval["start_seconds"])


def main():
    base = load_config(ROOT / "config/default.yaml")
    with st.sidebar:
        st.markdown("### Staff video analysis")
        st.caption("LOCAL VIDEO INTELLIGENCE")
        st.divider()
        st.markdown("**Evidence, not appearance.**")
        st.caption("Staff identity starts with the specified black-and-white tag. Tracking carries that evidence forward while the person remains visible.")
        st.image(str(ROOT / "assets/tag_reference.png"), caption="Identity-bearing tag reference", width=150)
        st.divider()
        st.markdown("**Reading the overlays**")
        st.markdown("🟢 **Tag observed**  \nConfirmed staff; tag matches now.")
        st.markdown("🔵 **Staff tracked**  \nConfirmed earlier; same active track.")
        st.markdown("⚪ **Uncertain**  \nStaff identity is not confirmed.")
        st.caption("A yellow box marks a current tag candidate. A candidate alone does not confirm identity.")
        st.divider()
        st.caption("No cloud inference · No face recognition\n\nTrack IDs are not unique employee counts.")

    st.markdown('<div class="eyebrow">RETAIL SENSOR / VIDEO WORKSPACE</div>', unsafe_allow_html=True)
    st.title("See the staff. Follow the evidence.")
    st.markdown('<p class="intro">Find tag-confirmed staff in overhead video, inspect the exact frames, and take the evidence with you.</p>', unsafe_allow_html=True)
    st.markdown('<span class="pill">Local processing</span><span class="pill">Frame-level evidence</span><span class="pill">CSV + video export</span>', unsafe_allow_html=True)
    st.write("")
    st.subheader("01 / Add a video")
    options = ["Upload an MP4"]
    if Path(base.sample_path).is_file():
        options.insert(0, "Use local sample")
    choice = st.radio("Video source", options, horizontal=True, label_visibility="collapsed")
    source = None
    if choice == "Use local sample":
        source = Path(base.sample_path)
    else:
        uploaded = st.file_uploader("Drop a sensor recording here", type=["mp4"], help="Up to 500 MB. Uploading does not start analysis.")
        if uploaded:
            content = uploaded.getvalue()
            upload_directory = ROOT / "outputs/uploads"
            upload_directory.mkdir(parents=True, exist_ok=True)
            safe_name = Path(uploaded.name).name
            source = upload_directory / f"{hashlib.sha256(content).hexdigest()[:16]}-{safe_name}"
            if not source.exists():
                source.write_bytes(content)
    info = None
    if source:
        try:
            info = metadata_for(str(source), source.stat().st_size, source.stat().st_mtime_ns)
            columns = st.columns([3, 1, 1, 1])
            columns[0].markdown(f"**{source.name}**")
            columns[1].caption(f'{info["width"]} × {info["height"]} px')
            columns[2].caption(f'{info["duration_seconds"]:.2f} seconds')
            columns[3].caption(f'{info["fps"]:g} fps · {info["reported_frames"]:,} frames')
            with st.expander("Source preview & decoded metadata"):
                st.video(str(source))
                st.json(info)
        except (OSError, ValueError) as error:
            st.error(str(error))

    st.subheader("02 / Analyze locally")
    with st.expander("Advanced settings · tunable, not measured probabilities", expanded=False):
        model_modes = {
            "Coverage-focused · YOLO11s, four orientations": ("yolo11s.pt", (0, 90, 180, 270)),
            "Higher-capacity · YOLO11m, four orientations": ("yolo11m.pt", (0, 90, 180, 270)),
            "Faster · YOLO11n, two orientations": ("yolo11n.pt", (0, 180)),
        }
        detection_mode = st.selectbox("Person detection mode", list(model_modes))
        left, middle, right = st.columns(3)
        detector_threshold = left.slider("Person detector threshold", .05, .80, float(base.detector_threshold), .05)
        tag_threshold = middle.slider("Tag correlation threshold", .50, .99, float(base.tag_threshold), .01)
        confirmation_hits = right.number_input("Tag observations to confirm", 1, 20, base.confirmation_hits)
        left, middle, right = st.columns(3)
        confirmation_window = left.number_input("Confirmation window (frames)", 1, 120, base.confirmation_window)
        max_track_gap = middle.number_input("Retain identity across gap (frames)", 1, 120, base.max_track_gap)
        tracker_buffer = right.number_input("ByteTrack buffer (at 30 fps)", 1, 120, base.tracker_buffer)
        left, middle, right = st.columns(3)
        tracker_match_threshold = left.slider("ByteTrack association threshold", .10, 1.0, float(base.tracker_match_threshold), .05)
        tag_min_width = middle.number_input("Smallest tag width (pixels)", 8, 64, base.tag_min_width)
        tag_max_width = right.number_input("Largest tag width (pixels)", 8, 100, base.tag_max_width)
        st.caption("YOLO11s and YOLO11m check four orientations; faster YOLO11n checks upright and 180°. All modes merge boxes and update one ByteTrack instance. A larger model is slower and does not guarantee better accuracy on this camera.")
    config = None
    try:
        model_filename, rotations = model_modes[detection_mode]
        config = base.changed(model_path=str(ROOT / "models" / model_filename),
                              detector_rotations=rotations,
                              detector_threshold=detector_threshold, tag_threshold=tag_threshold,
                              confirmation_hits=confirmation_hits, confirmation_window=confirmation_window,
                              max_track_gap=max_track_gap, tracker_buffer=tracker_buffer,
                              tracker_match_threshold=tracker_match_threshold,
                              tag_min_width=tag_min_width, tag_max_width=tag_max_width)
    except ValueError as error:
        st.error(str(error))
    st.caption("Settings apply only when you start a new analysis. All frames are processed; long clips may take many minutes on CPU. YOLO11m takes about twice as long as YOLO11s in the 100-frame test.")
    if st.button("Analyze video", type="primary", disabled=info is None or config is None, icon="▶"):
        progress_bar = st.progress(0, text="Preparing local analysis…")
        status = st.empty()

        def update(event):
            stage = event["stage"]
            fraction = min(1.0, event["processed_frames"] / max(1, event["total_frames"]))
            overall = fraction * .78 if stage.startswith("Detecting") else .78 + fraction * .17 if stage.startswith("Rendering") else .97 if stage.startswith(("Encoding", "Exporting")) else 1.0 if stage == "Complete" else 0
            progress_bar.progress(overall, text=stage)
            status.caption(f'{event["processed_frames"]:,} / {event["total_frames"]:,} frames · {event["elapsed_seconds"]:.1f}s elapsed')

        try:
            result = analyze_video(source, ROOT / "outputs", config, update)
            st.session_state.result = result
            st.session_state.frame_index = result.intervals[0]["start_frame"] if result.intervals else 0
            st.session_state.playback_time = result.frames[st.session_state.frame_index]["timestamp_seconds"]
            st.session_state.interval_selection = None
            st.success("Analysis complete. Inspection controls do not rerun inference.")
        except Exception as error:
            progress_bar.empty()
            st.error(f"Analysis could not finish: {error}. Check that local weights/reference exist and the video is decodable. Any previous completed result is retained.")
    saved_runs = sorted((path for path in (ROOT / "outputs").glob("run-*/summary.json")
                         if not (path.parent / "FAILED.txt").exists()), reverse=True)
    if saved_runs and source:
        with st.expander("Reopen a completed local run · no inference"):
            saved = st.selectbox("Saved run", [path.parent for path in saved_runs], format_func=lambda path: path.name)
            if st.button("Open saved result"):
                try:
                    result = load_result(saved, source)
                    st.session_state.result = result
                    st.session_state.frame_index = result.intervals[0]["start_frame"] if result.intervals else 0
                    st.session_state.playback_time = float(result.frames[st.session_state.frame_index]["timestamp_seconds"])
                    st.session_state.interval_selection = None
                    st.success("Saved result opened. Source checksum verified; no inference ran.")
                except (OSError, ValueError, KeyError) as error:
                    st.error(str(error))
    result = st.session_state.get("result")
    if result is None:
        st.info("Ready when you are. Choose a recording, then select Analyze video. No detections are shown until a real analysis finishes.")
        return

    st.divider()
    st.subheader("03 / Inspect the evidence")
    summary = result.summary
    if summary.get("synthetic"):
        st.warning("SYNTHETIC TEST FIXTURE — not a real video analysis.")
    st.caption(f'Completed result: {summary["source"]["filename"]} · {summary["created_utc"]} · run {result.directory.name}')
    if source != result.source or (config and config.to_dict() != summary["settings"]):
        st.warning("Input or settings differ from this completed result. Select Analyze video to apply changes; the current result is unchanged.")
    metrics = st.columns(5)
    for column, label, value in zip(metrics,
            ["Processed frames", "Staff-present frames", "Staff-present time", "Staff tracks", "Processing time"],
            [f'{summary["processed_frames"]:,}', f'{summary["staff_present_frames"]:,}',
             f'{summary["staff_present_seconds"]:.2f}s', str(summary["staff_track_count"]), f'{summary["processing_seconds"]:.1f}s']):
        column.metric(label, value)
    st.caption("Staff tracks ≠ unique employees. Staff-present time is the union of labelled frame durations, not summed person-time.")
    for warning in summary["warnings"]:
        st.caption(f"Note: {warning}")
    if "person_evaluation" not in summary:
        st.warning("Person, staff, and tag accuracy is not measured for this run. More detections do not prove correct detections; annotate frames and evaluate below before operational use.")
    with st.expander("Settings & provenance used for this result"):
        st.json(summary)

    st.markdown("#### Presence over time")
    figure = go.Figure(go.Scatter(
        x=[frame["timestamp_seconds"] for frame in result.frames] + [summary["decoded_duration_seconds"]],
        y=[frame["staff_count"] for frame in result.frames] + [result.frames[-1]["staff_count"]],
        mode="lines", line={"color": "#087f8c", "width": 2, "shape": "hv"}, fill="tozeroy",
        fillcolor="rgba(8,127,140,.13)", hovertemplate="%{x:.2f}s · %{y} confirmed staff<extra></extra>",
    ))
    figure.update_layout(height=200, margin=dict(l=0, r=10, t=10, b=5), paper_bgcolor="rgba(0,0,0,0)",
                         plot_bgcolor="rgba(0,0,0,0)", xaxis_title="Source time (seconds)", yaxis_title="Staff", yaxis=dict(dtick=1, rangemode="tozero"))
    st.plotly_chart(figure, config={"displayModeBar": False})

    intervals = result.intervals
    if intervals:
        st.dataframe(pd.DataFrame(intervals), hide_index=True, width="stretch")
        st.selectbox("Jump to an inclusive staff interval", range(len(intervals)), index=None,
                     key="interval_selection", on_change=choose_interval,
                     format_func=lambda index: f'Frames {intervals[index]["start_frame"]}–{intervals[index]["end_frame"]} · {intervals[index]["start_seconds"]:.2f}s')
    else:
        st.warning("No confirmed staff intervals. Review exact frames and candidate scores before concluding that staff are absent.")

    video_tab, frame_tab, exports_tab = st.tabs(["Annotated playback", "Exact frame inspector", "Exports & evaluation"])
    with frame_tab:
        st.markdown("#### Results for every frame")
        st.caption("One row per decoded frame, including frames with zero detections. Tag matches show current visual evidence; staff counts require track confirmation. Select a row to inspect its exact source and annotated images.")
        frame_filter = st.selectbox("Show frames", ["All frames", "Staff present", "Tag match observed", "No people detected"])
        all_frame_rows = list(frame_rows(result.frames))
        if frame_filter == "Staff present":
            visible_frames = [frame for frame in all_frame_rows if frame["staff_present"]]
        elif frame_filter == "Tag match observed":
            visible_frames = [frame for frame in all_frame_rows if frame["tag_observed_count"]]
        elif frame_filter == "No people detected":
            visible_frames = [frame for frame in all_frame_rows if not frame["detected_people"]]
        else:
            visible_frames = all_frame_rows
        frame_table = pd.DataFrame([{
            "Frame": frame["frame_index"], "Time (s)": round(frame["timestamp_seconds"], 3),
            "People detected": frame["detected_people"], "Tag matches now": frame["tag_observed_count"],
            "Confirmed staff": frame["staff_count"], "Staff present": frame["staff_present"],
        } for frame in visible_frames], columns=["Frame", "Time (s)", "People detected", "Tag matches now", "Confirmed staff", "Staff present"])
        selection = st.dataframe(frame_table, hide_index=True, width="stretch", height=300,
                                 on_select="rerun", selection_mode="single-row",
                                 key=f"frame_table_{result.directory.name}_{frame_filter}")
        selected_rows = selection.selection.rows
        if selected_rows:
            selection_key = (result.directory.name, frame_filter, selected_rows[0])
            if st.session_state.get("last_selected_frame_row") != selection_key:
                selected_frame = visible_frames[selected_rows[0]]
                st.session_state.frame_index = selected_frame["frame_index"]
                st.session_state.playback_time = float(selected_frame["timestamp_seconds"])
                st.session_state.last_selected_frame_row = selection_key
        st.caption(f"Showing {len(visible_frames):,} of {len(result.frames):,} processed frames.")
        st.markdown("#### Exact frame inspector")
        previous, number, following = st.columns([1, 4, 1], vertical_alignment="bottom")
        previous.button("← Previous", on_click=move_frame, args=(-1,), width="stretch")
        frame_index = number.number_input("Exact frame (zero-based)", 0, len(result.frames) - 1, key="frame_index", step=1)
        following.button("Next →", on_click=move_frame, args=(1,), width="stretch")
        record = result.frames[frame_index]
        st.caption(f'Frame {frame_index} · source timestamp {record["timestamp_seconds"]:.3f}s · {record["staff_count"]} confirmed staff')
        try:
            original = inspect_exact(str(result.source), frame_index, result.source.stat().st_mtime_ns)
            left, right = st.columns(2)
            left.image(cv2.cvtColor(original, cv2.COLOR_BGR2RGB), caption="Source · exact sequential decode", width="stretch")
            right.image(cv2.cvtColor(annotate(original, record), cv2.COLOR_BGR2RGB), caption="Annotated · same source frame", width="stretch")
        except (OSError, ValueError) as error:
            st.error(f"Source frame unavailable: {error}")
        st.markdown("**People detected in this frame**")
        rows = [{"Track ID": person["track_id"], "Status": person["status"],
                 "Tag matched now": person["tag_observed"], "Staff confirmed": person["staff_present"],
                 "Tag score": round(person["tag_score"], 4), "Person center x": person["center_x"],
                 "Person center y": person["center_y"], "Person box": person["box"],
                 "Tag box": person["tag_box"]} for person in record["people"]]
        if rows:
            st.dataframe(rows, width="stretch", hide_index=True)
        else:
            st.info("No person was detected in this frame.")
        st.caption("Coordinates are source-image pixels from the top-left; xy is the person-box center, not a floor-plane location. Tag score is template correlation, not a probability.")
    with video_tab:
        playback_time = st.number_input("Playback start (seconds)", 0.0, float(summary["decoded_duration_seconds"]), key="playback_time", step=.1)
        st.video(str(result.directory / "annotated.mp4"), start_time=playback_time)
        st.caption("Playback seeking is approximate. Use the frame inspector for exact selection. Green: Tag observed · Blue: Staff tracked · Grey: Uncertain.")
    with exports_tab:
        downloads = [("Artifact ZIP", "artifacts.zip", "application/zip"), ("Frame CSV", "frames.csv", "text/csv"),
                     ("Coordinates CSV", "detections.csv", "text/csv"), ("Intervals CSV", "intervals.csv", "text/csv"),
                     ("All frame records JSON", "records.json", "application/json"),
                     ("Summary & settings", "summary.json", "application/json"), ("Annotated video", "annotated.mp4", "video/mp4")]
        columns = st.columns(3)
        for index, (label, filename, mime) in enumerate(downloads):
            columns[index % 3].download_button(label, (result.directory / filename).read_bytes(), filename,
                                               mime=mime, width="stretch")
        st.caption(f"Artifacts are also saved locally: {result.directory}")
        st.markdown("#### Evaluate against your reference labels")
        st.caption("Optional CSV: frame_index,staff_present (0 or 1). Only labelled frames are scored; coverage is reported. No metrics exist without reference labels.")
        labels = st.file_uploader("Reference labels", type=["csv"], key="reference_labels")
        if labels:
            try:
                evaluation = evaluate_labels(result.frames, pd.read_csv(labels))
                st.json(evaluation)
                import json

                st.download_button("Download evaluation JSON", json.dumps(evaluation, indent=2), "evaluation.json", "application/json")
            except (ValueError, pd.errors.ParserError) as error:
                st.error(str(error))
        st.caption("Optional person CSV: frame_index,x1,y1,x2,y2,is_staff,tag_visible. Annotate every visible person in each selected frame; use one row with blank box/flags for an empty frame. Unlabelled frames are excluded.")
        person_labels = st.file_uploader("Person and tag labels", type=["csv"], key="person_labels")
        if person_labels:
            try:
                person_evaluation = evaluate_people(result.frames, pd.read_csv(person_labels),
                                                    summary["source"]["width"], summary["source"]["height"])
                st.json(person_evaluation)
                import json

                st.download_button("Download person evaluation JSON", json.dumps(person_evaluation, indent=2),
                                   "person_evaluation.json", "application/json")
            except (ValueError, pd.errors.ParserError) as error:
                st.error(str(error))


if __name__ == "__main__":
    main()
