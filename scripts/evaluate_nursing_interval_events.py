"""Evaluate nursing interval predictions as matched continuous-time events.

The evaluator compares the original confirmed continuous intervals with an
interval prediction JSON.  It clips both sides to each video's true duration,
then performs deterministic greedy one-to-one matching in descending IoU order.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.nursing_interval_annotations import parse_confirmed_annotations  # noqa: E402


DEFAULT_MODEL_DIR = ROOT / "reproduced" / "intervals"
DEFAULT_ANNOTATIONS = ROOT / "data/DS4_intervals/annotations/confirm_milking.md"
DEFAULT_PREDICTIONS = DEFAULT_MODEL_DIR / "oof_intervals.json"
DEFAULT_MANIFEST = None
DEFAULT_OUTPUT = DEFAULT_MODEL_DIR / "event_evaluation"

CLASS_NAMES = ("milk", "partial", "collective")
ANNOTATION_FIELDS = {
    "milk": "milk_intervals",
    "partial": "partial_intervals",
    "collective": "collective_intervals",
}
PREDICTION_FIELDS = {
    "milk": "milk_intervals",
    "partial": "partial_arching_intervals",
    "collective": "collective_arching_intervals",
}
MATCH_COLUMNS = [
    "video",
    "class",
    "true_start_sec",
    "true_end_sec",
    "true_duration_sec",
    "pred_start_sec",
    "pred_end_sec",
    "pred_duration_sec",
    "iou",
    "start_delta_sec",
    "end_delta_sec",
    "duration_delta_sec",
    "start_abs_delta_sec",
    "end_abs_delta_sec",
    "duration_abs_delta_sec",
    "boundary_abs_mean_sec",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(value), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _endpoint(event: Any, name: str) -> float:
    value = event[name] if isinstance(event, Mapping) else getattr(event, name)
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Event {name} must be finite, got {value!r}")
    return number


def _normalized_event(event: Any, *, duration_sec: float, context: str) -> dict[str, float] | None:
    start = _endpoint(event, "start_sec")
    end = _endpoint(event, "end_sec")
    if end <= start:
        raise ValueError(f"{context} has non-positive interval {start:g}-{end:g}")
    clipped_start = max(0.0, start)
    clipped_end = min(float(duration_sec), end)
    if clipped_end <= clipped_start:
        return None
    return {
        "start_sec": clipped_start,
        "end_sec": clipped_end,
        "duration_sec": clipped_end - clipped_start,
    }


def interval_iou(true_event: Mapping[str, Any], pred_event: Mapping[str, Any]) -> float:
    """Return intersection-over-union for two half-open continuous intervals."""
    true_start = _endpoint(true_event, "start_sec")
    true_end = _endpoint(true_event, "end_sec")
    pred_start = _endpoint(pred_event, "start_sec")
    pred_end = _endpoint(pred_event, "end_sec")
    if true_end <= true_start or pred_end <= pred_start:
        raise ValueError("IoU events must have positive duration")
    intersection = max(0.0, min(true_end, pred_end) - max(true_start, pred_start))
    union = (true_end - true_start) + (pred_end - pred_start) - intersection
    return float(intersection / union) if union > 0.0 else 0.0


def greedy_match_events(
    true_events: Sequence[Mapping[str, Any]],
    pred_events: Sequence[Mapping[str, Any]],
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Greedily match all event pairs by descending IoU, one-to-one."""
    threshold = float(iou_threshold)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("iou_threshold must be in [0, 1]")
    candidates = [
        (true_index, pred_index, interval_iou(true_event, pred_event))
        for true_index, true_event in enumerate(true_events)
        for pred_index, pred_event in enumerate(pred_events)
    ]
    candidates.sort(key=lambda item: (-item[2], item[0], item[1]))
    used_true: set[int] = set()
    used_pred: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for true_index, pred_index, iou in candidates:
        if iou + 1e-12 < threshold:
            break
        if true_index in used_true or pred_index in used_pred:
            continue
        used_true.add(true_index)
        used_pred.add(pred_index)
        matches.append((true_index, pred_index, float(iou)))
    return {
        "matches": matches,
        "unmatched_true_indices": [index for index in range(len(true_events)) if index not in used_true],
        "unmatched_pred_indices": [index for index in range(len(pred_events)) if index not in used_pred],
    }


def summarize_delta(values: Sequence[float]) -> dict[str, Any]:
    """Summarize signed Δt values and their absolute magnitudes."""
    array = np.asarray(list(values), dtype=float)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("Delta values must be a finite one-dimensional sequence")
    if not len(array):
        return {
            "count": 0,
            "signed_mean": None,
            "signed_median": None,
            "signed_std": None,
            "absolute_mae": None,
            "absolute_median": None,
            "absolute_p90": None,
            "absolute_max": None,
        }
    absolute = np.abs(array)
    return {
        "count": int(len(array)),
        "signed_mean": float(np.mean(array)),
        "signed_median": float(np.median(array)),
        "signed_std": float(np.std(array, ddof=0)),
        "absolute_mae": float(np.mean(absolute)),
        "absolute_median": float(np.median(absolute)),
        "absolute_p90": float(np.percentile(absolute, 90)),
        "absolute_max": float(np.max(absolute)),
    }


def _summarize_boundary(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=float)
    if not len(array):
        return {"count": 0, "mean": None, "median": None, "p90": None}
    if not np.isfinite(array).all():
        raise ValueError("Boundary errors must be finite")
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
    }


def _delta_statistics(matches: pd.DataFrame) -> dict[str, Any]:
    return {
        "start_delta_sec": summarize_delta(matches.get("start_delta_sec", pd.Series(dtype=float))),
        "end_delta_sec": summarize_delta(matches.get("end_delta_sec", pd.Series(dtype=float))),
        "duration_delta_sec": summarize_delta(matches.get("duration_delta_sec", pd.Series(dtype=float))),
        "boundary_abs_mean_sec": _summarize_boundary(
            matches.get("boundary_abs_mean_sec", pd.Series(dtype=float))
        ),
    }


def _prediction_map(predictions: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    records = predictions.get("videos")
    if not isinstance(records, list):
        raise ValueError("Prediction JSON must contain a videos list")
    result: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping) or not str(record.get("video", "")).strip():
            raise ValueError("Each prediction video record must have a non-empty video")
        video = str(record["video"])
        if video in result:
            raise ValueError(f"Duplicate prediction video: {video!r}")
        result[video] = record
    return result


def _clipped_class_events(
    source: Any,
    field: str,
    *,
    duration_sec: float,
    context: str,
) -> list[dict[str, float]]:
    values = source.get(field) if isinstance(source, Mapping) else getattr(source, field)
    if values is None or isinstance(values, (str, bytes, Mapping)):
        raise ValueError(f"{context}.{field} must be an interval sequence")
    clipped: list[dict[str, float]] = []
    for index, event in enumerate(values):
        normalized = _normalized_event(
            event,
            duration_sec=duration_sec,
            context=f"{context}.{field}[{index}]",
        )
        if normalized is not None:
            clipped.append(normalized)
    return clipped


def _match_row(
    video: str,
    class_name: str,
    true_event: Mapping[str, float],
    pred_event: Mapping[str, float],
    iou: float,
) -> dict[str, Any]:
    start_delta = float(pred_event["start_sec"] - true_event["start_sec"])
    end_delta = float(pred_event["end_sec"] - true_event["end_sec"])
    duration_delta = float(pred_event["duration_sec"] - true_event["duration_sec"])
    return {
        "video": video,
        "class": class_name,
        "true_start_sec": true_event["start_sec"],
        "true_end_sec": true_event["end_sec"],
        "true_duration_sec": true_event["duration_sec"],
        "pred_start_sec": pred_event["start_sec"],
        "pred_end_sec": pred_event["end_sec"],
        "pred_duration_sec": pred_event["duration_sec"],
        "iou": float(iou),
        "start_delta_sec": start_delta,
        "end_delta_sec": end_delta,
        "duration_delta_sec": duration_delta,
        "start_abs_delta_sec": abs(start_delta),
        "end_abs_delta_sec": abs(end_delta),
        "duration_abs_delta_sec": abs(duration_delta),
        "boundary_abs_mean_sec": (abs(start_delta) + abs(end_delta)) / 2.0,
    }


def evaluate_events(
    annotations: Mapping[str, Any],
    predictions: Mapping[str, Any],
    source_durations_sec: Mapping[str, float],
    iou_threshold: float = 0.5,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate all videos/classes and return aggregate metrics plus TP matches."""
    threshold = float(iou_threshold)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("iou_threshold must be in [0, 1]")
    prediction_by_video = _prediction_map(predictions)
    truth_videos = {str(video) for video in annotations}
    prediction_videos = set(prediction_by_video)
    if truth_videos != prediction_videos:
        raise ValueError(
            "Prediction/truth video sets differ: "
            f"truth_only={sorted(truth_videos - prediction_videos)}, "
            f"prediction_only={sorted(prediction_videos - truth_videos)}"
        )
    duration_videos = {str(video) for video in source_durations_sec}
    if duration_videos != truth_videos:
        raise ValueError(
            "Manifest duration/truth video sets differ: "
            f"truth_without_duration={sorted(truth_videos - duration_videos)}, "
            f"duration_without_truth={sorted(duration_videos - truth_videos)}"
        )

    class_counts = {
        name: {"tp": 0, "fp": 0, "fn": 0, "true_event_count": 0, "pred_event_count": 0}
        for name in CLASS_NAMES
    }
    match_rows: list[dict[str, Any]] = []
    for video in sorted(truth_videos):
        duration = float(source_durations_sec[video])
        if not math.isfinite(duration) or duration < 0.0:
            raise ValueError(f"Invalid source duration for {video!r}: {duration!r}")
        truth_source = annotations[video]
        prediction_source = prediction_by_video[video]
        for class_name in CLASS_NAMES:
            truth_events = _clipped_class_events(
                truth_source,
                ANNOTATION_FIELDS[class_name],
                duration_sec=duration,
                context=f"truth[{video!r}]",
            )
            predicted_events = _clipped_class_events(
                prediction_source,
                PREDICTION_FIELDS[class_name],
                duration_sec=duration,
                context=f"prediction[{video!r}]",
            )
            matched = greedy_match_events(truth_events, predicted_events, threshold)
            counts = class_counts[class_name]
            counts["true_event_count"] += len(truth_events)
            counts["pred_event_count"] += len(predicted_events)
            counts["tp"] += len(matched["matches"])
            counts["fn"] += len(matched["unmatched_true_indices"])
            counts["fp"] += len(matched["unmatched_pred_indices"])
            for true_index, pred_index, iou in matched["matches"]:
                match_rows.append(
                    _match_row(
                        video,
                        class_name,
                        truth_events[true_index],
                        predicted_events[pred_index],
                        iou,
                    )
                )

    matches = pd.DataFrame(match_rows, columns=MATCH_COLUMNS)
    if not matches.empty:
        class_order = {name: index for index, name in enumerate(CLASS_NAMES)}
        matches["_class_order"] = matches["class"].map(class_order)
        matches.sort_values(
            ["video", "_class_order", "true_start_sec", "pred_start_sec"],
            inplace=True,
        )
        matches.drop(columns="_class_order", inplace=True)
        matches.reset_index(drop=True, inplace=True)

    class_metrics: dict[str, Any] = {}
    for class_name in CLASS_NAMES:
        counts = class_counts[class_name]
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        subset = matches[matches["class"] == class_name]
        class_metrics[class_name] = {
            **counts,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_iou": float(subset["iou"].mean()) if len(subset) else None,
            "delta_statistics": _delta_statistics(subset),
        }

    micro_tp = sum(row["tp"] for row in class_counts.values())
    micro_fp = sum(row["fp"] for row in class_counts.values())
    micro_fn = sum(row["fn"] for row in class_counts.values())
    micro_precision = micro_tp / (micro_tp + micro_fp) if micro_tp + micro_fp else 0.0
    micro_recall = micro_tp / (micro_tp + micro_fn) if micro_tp + micro_fn else 0.0
    micro_f1 = (
        2.0 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )
    report = {
        "created_utc": _utc_now(),
        "evaluation_unit": "continuous half-open events [start_sec, end_sec)",
        "matching": {
            "method": "all-pair IoU sorted descending, deterministic greedy one-to-one",
            "iou_threshold": threshold,
            "threshold_comparison": "IoU >= threshold",
        },
        "delta_semantics": {
            "formula": "delta_t = predicted_time - true_time",
            "start_delta_sec": "positive means the predicted event starts later; negative means earlier",
            "end_delta_sec": "positive means the predicted event ends later; negative means earlier",
            "duration_delta_sec": "positive means the predicted event is longer; negative means shorter",
            "boundary_abs_mean_sec": "mean(abs(start_delta_sec), abs(end_delta_sec)) for each TP",
        },
        "video_count": len(truth_videos),
        "classes": class_metrics,
        "micro": {
            "tp": micro_tp,
            "fp": micro_fp,
            "fn": micro_fn,
            "precision": micro_precision,
            "recall": micro_recall,
            "f1": micro_f1,
            "mean_iou": float(matches["iou"].mean()) if len(matches) else None,
            "all_tp_delta_statistics": _delta_statistics(matches),
        },
    }
    return report, matches


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    annotations_path = args.annotations.expanduser().resolve()
    predictions_path = args.predictions.expanduser().resolve()
    output = args.output.expanduser().resolve()
    annotations = parse_confirmed_annotations(annotations_path)
    predictions = _read_json(predictions_path)
    if args.manifest is not None and args.manifest.expanduser().resolve().is_file():
        manifest_path = args.manifest.expanduser().resolve()
        manifest = _read_json(manifest_path)
        durations = manifest.get("source_durations_sec")
        if not isinstance(durations, Mapping):
            raise ValueError(f"Manifest has no source_durations_sec mapping: {manifest_path}")
    else:
        durations = {
            str(entry["video"]): float(entry["source_duration_sec"])
            for entry in predictions.get("videos", [])
            if isinstance(entry, Mapping) and entry.get("source_duration_sec") is not None
        }
    report, matches = evaluate_events(
        annotations,
        predictions,
        durations,
        iou_threshold=float(args.iou_threshold),
    )
    report["inputs"] = {
        "annotations": annotations_path,
        "predictions": predictions_path,
    }
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "event_metrics_deltat.json"
    matches_path = output / "tp_event_matches.csv"
    _write_json(metrics_path, report)
    matches.to_csv(matches_path, index=False, encoding="utf-8-sig")
    print(
        f"[done] videos={report['video_count']} tp={report['micro']['tp']} "
        f"fp={report['micro']['fp']} fn={report['micro']['fn']} "
        f"f1={report['micro']['f1']:.6f}",
        flush=True,
    )
    print(f"[output] metrics={metrics_path} matches={matches_path}", flush=True)


if __name__ == "__main__":
    main()
