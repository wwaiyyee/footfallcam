from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import math


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _scores(true_positive, false_positive, false_negative):
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": _ratio(true_positive, true_positive + false_positive),
        "recall": _ratio(true_positive, true_positive + false_negative),
        "f1": _ratio(2 * true_positive, 2 * true_positive + false_positive + false_negative),
    }


def _box_iou(first, second):
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / (first_area + second_area - intersection) if intersection else 0.0


def evaluate_people(frames: list[dict], labels: pd.DataFrame, width: int, height: int,
                    minimum_iou: float = 0.5) -> dict:
    if not math.isfinite(minimum_iou) or not 0 < minimum_iou <= 1:
        raise ValueError("Matching IoU threshold must be in (0, 1].")
    columns = {"frame_index", "x1", "y1", "x2", "y2", "is_staff", "tag_visible"}
    if labels.empty or not columns.issubset(labels.columns):
        raise ValueError("Person labels need frame_index,x1,y1,x2,y2,is_staff,tag_visible. Use one row per visible person.")
    indices = pd.to_numeric(labels["frame_index"], errors="coerce")
    if indices.isna().any() or (indices % 1 != 0).any() or not indices.astype(int).between(0, len(frames) - 1).all():
        raise ValueError("Person label frame indices must be integers within the processed video.")
    prepared = labels.copy()
    prepared["frame_index"] = indices.astype(int)
    coordinate_columns = ["x1", "y1", "x2", "y2"]
    for column in coordinate_columns + ["is_staff", "tag_visible"]:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    empty_boxes = prepared[coordinate_columns].isna().all(axis=1)
    partial_boxes = prepared[coordinate_columns].isna().any(axis=1) & ~empty_boxes
    if partial_boxes.any() or (empty_boxes & prepared[["is_staff", "tag_visible"]].notna().any(axis=1)).any():
        raise ValueError("Each row needs a complete person box or an entirely blank person/label entry for an empty frame.")
    if (prepared.loc[~empty_boxes, "x1"] < 0).any() or (prepared.loc[~empty_boxes, "y1"] < 0).any() or (
        prepared.loc[~empty_boxes, "x2"] > width
    ).any() or (prepared.loc[~empty_boxes, "y2"] > height).any() or (
        prepared.loc[~empty_boxes, "x2"] <= prepared.loc[~empty_boxes, "x1"]
    ).any() or (prepared.loc[~empty_boxes, "y2"] <= prepared.loc[~empty_boxes, "y1"]).any():
        raise ValueError("Person boxes must have positive area and remain inside source-image bounds.")
    for column in ("is_staff", "tag_visible"):
        if not prepared.loc[~empty_boxes, column].isin([0, 1]).all():
            raise ValueError(f"{column} must be 0 or 1 for every labelled person.")
    if empty_boxes.any() and prepared.loc[empty_boxes, "frame_index"].duplicated().any():
        raise ValueError("An empty-frame label may occur only once per frame.")
    if set(prepared.loc[empty_boxes, "frame_index"]) & set(prepared.loc[~empty_boxes, "frame_index"]):
        raise ValueError("An empty-frame row cannot be mixed with person rows for the same frame.")
    labelled_frames = set(prepared["frame_index"])
    counts = {name: {"tp": 0, "fp": 0, "fn": 0} for name in ("person", "staff", "visible_tag")}
    for frame_index in sorted(labelled_frames):
        people = frames[frame_index]["people"]
        truth = prepared[(prepared["frame_index"] == frame_index) & ~empty_boxes]
        truth_rows = truth.to_dict("records")
        matches = []
        if people and truth_rows:
            overlaps = np.array([[_box_iou(person["box"], [row[column] for column in coordinate_columns])
                                  for row in truth_rows] for person in people])
            valid_scores = np.where(overlaps >= minimum_iou, 1 + overlaps, 0)
            predicted_indices, truth_indices = linear_sum_assignment(-valid_scores)
            matches = [(int(predicted), int(actual)) for predicted, actual in zip(predicted_indices, truth_indices)
                       if overlaps[predicted, actual] >= minimum_iou]
        matched_predictions = {predicted for predicted, _ in matches}
        matched_truth = {actual for _, actual in matches}
        counts["person"]["tp"] += len(matches)
        counts["person"]["fp"] += len(people) - len(matches)
        counts["person"]["fn"] += len(truth_rows) - len(matches)
        for prediction_key, truth_key, category in (("staff_present", "is_staff", "staff"),
                                                    ("tag_observed", "tag_visible", "visible_tag")):
            for predicted, actual in matches:
                observed = bool(people[predicted][prediction_key])
                expected = bool(truth_rows[actual][truth_key])
                counts[category]["tp"] += int(observed and expected)
                counts[category]["fp"] += int(observed and not expected)
                counts[category]["fn"] += int(expected and not observed)
            counts[category]["fp"] += sum(bool(person[prediction_key]) for index, person in enumerate(people)
                                            if index not in matched_predictions)
            counts[category]["fn"] += sum(bool(row[truth_key]) for index, row in enumerate(truth_rows)
                                            if index not in matched_truth)
    return {
        "labelled_frames": len(labelled_frames), "coverage": len(labelled_frames) / len(frames),
        "matching_iou_threshold": minimum_iou,
        "person_detection": _scores(counts["person"]["tp"], counts["person"]["fp"], counts["person"]["fn"]),
        "staff_identification": _scores(counts["staff"]["tp"], counts["staff"]["fp"], counts["staff"]["fn"]),
        "visible_tag_detection": _scores(counts["visible_tag"]["tp"], counts["visible_tag"]["fp"], counts["visible_tag"]["fn"]),
        "scope": "Only fully annotated frame indices are evaluated. Missing people count against staff/tag recall. Track continuity is not scored.",
    }


def evaluate_labels(frames: list[dict], labels: pd.DataFrame) -> dict:
    required = {"frame_index", "staff_present"}
    if not required.issubset(labels.columns) or labels.empty:
        raise ValueError("Reference CSV needs at least one row and columns frame_index,staff_present (0 or 1).")
    indices = pd.to_numeric(labels["frame_index"], errors="coerce")
    values = pd.to_numeric(labels["staff_present"], errors="coerce")
    if indices.isna().any() or (indices % 1 != 0).any() or indices.duplicated().any():
        raise ValueError("Reference frame indices must be unique integers.")
    if not values.isin([0, 1]).all():
        raise ValueError("Reference staff_present values must be 0 or 1.")
    predictions = {frame["frame_index"]: bool(frame["staff_present"]) for frame in frames}
    if not set(indices.astype(int)).issubset(predictions):
        raise ValueError("Reference frame indices are outside the processed video.")
    true_positive = false_positive = false_negative = true_negative = 0
    for index, truth in zip(indices.astype(int), values.astype(bool)):
        predicted = predictions[index]
        true_positive += int(predicted and truth)
        false_positive += int(predicted and not truth)
        false_negative += int(not predicted and truth)
        true_negative += int(not predicted and not truth)
    return {
        "labelled_frames": len(labels), "coverage": len(labels) / len(frames),
        "true_positive": true_positive, "false_positive": false_positive,
        "false_negative": false_negative, "true_negative": true_negative,
        "precision": _ratio(true_positive, true_positive + false_positive),
        "recall": _ratio(true_positive, true_positive + false_negative),
        "f1": _ratio(2 * true_positive, 2 * true_positive + false_positive + false_negative),
        "accuracy": _ratio(true_positive + true_negative, len(labels)),
        "scope": "Only supplied reference rows; unlabelled frames excluded.",
    }
