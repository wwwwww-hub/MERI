"""Evaluate milk as events and both arching classes as occupied time.

Milk release keeps the established greedy one-to-one event matching at a
configurable IoU threshold. Partial and collective arching are evaluated
separately from the duration of their temporal overlap, without pairing or
counting individual arching events.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_nursing_interval_events import evaluate_events  # noqa: E402
from scripts.nursing_interval_annotations import parse_confirmed_annotations  # noqa: E402


DEFAULT_MODEL_DIR = ROOT / "reproduced" / "intervals"
DEFAULT_ANNOTATIONS = ROOT / "data/DS4_intervals/annotations/confirm_milking.md"
DEFAULT_PREDICTIONS = DEFAULT_MODEL_DIR / "oof_intervals.json"
DEFAULT_MANIFEST = None
DEFAULT_OUTPUT = DEFAULT_MODEL_DIR / "hybrid_evaluation"

ARCHING_FIELDS = {
    "partial": ("partial_intervals", "partial_arching_intervals"),
    "collective": ("collective_intervals", "collective_arching_intervals"),
}
TEMPORAL_FIELDS = {
    "milk": ("milk_intervals", "milk_intervals"),
    **ARCHING_FIELDS,
}

PER_VIDEO_COLUMNS = (
    "video",
    "annotation_index",
    "class",
    "source_duration_sec",
    "evaluation_duration_sec",
    "excluded_duration_sec",
    "true_duration_sec",
    "pred_duration_sec",
    "tp_overlap_sec",
    "fp_duration_sec",
    "fn_duration_sec",
    "tn_duration_sec",
    "time_precision",
    "time_recall",
    "time_f1",
    "time_iou",
    "time_accuracy",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _endpoint(event: Any, name: str) -> float:
    value = event.get(name) if isinstance(event, Mapping) else getattr(event, name)
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Event {name} must be finite, got {value!r}")
    return number


def interval_union(
    events: Sequence[Any],
    duration_sec: float,
) -> list[tuple[float, float]]:
    """Clip, sort and merge intervals into a non-overlapping temporal union."""

    duration = float(duration_sec)
    if not math.isfinite(duration) or duration < 0.0:
        raise ValueError(f"duration_sec must be finite and non-negative, got {duration_sec!r}")
    clipped: list[tuple[float, float]] = []
    for event in events:
        start = _endpoint(event, "start_sec")
        end = _endpoint(event, "end_sec")
        if end <= start:
            raise ValueError(f"Event has non-positive interval {start:g}-{end:g}")
        start = max(0.0, start)
        end = min(duration, end)
        if end > start:
            clipped.append((start, end))
    clipped.sort()
    merged: list[list[float]] = []
    for start, end in clipped:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def interval_union_duration(intervals: Sequence[tuple[float, float]]) -> float:
    return float(sum(end - start for start, end in intervals))


def interval_overlap_duration(
    first: Sequence[tuple[float, float]],
    second: Sequence[tuple[float, float]],
) -> float:
    """Return exact overlap duration for two sorted, non-overlapping unions."""

    first_index = 0
    second_index = 0
    overlap = 0.0
    while first_index < len(first) and second_index < len(second):
        first_start, first_end = first[first_index]
        second_start, second_end = second[second_index]
        overlap += max(0.0, min(first_end, second_end) - max(first_start, second_start))
        if first_end <= second_end:
            first_index += 1
        else:
            second_index += 1
    return float(overlap)


def interval_intersection(
    first: Sequence[tuple[float, float]],
    second: Sequence[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Return the overlap of two sorted, non-overlapping temporal unions."""

    result: list[tuple[float, float]] = []
    first_index = second_index = 0
    while first_index < len(first) and second_index < len(second):
        start = max(first[first_index][0], second[second_index][0])
        end = min(first[first_index][1], second[second_index][1])
        if end > start:
            result.append((start, end))
        if first[first_index][1] <= second[second_index][1]:
            first_index += 1
        else:
            second_index += 1
    return result


def _ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator > 0.0 else None


def temporal_overlap_metrics(
    true_events: Sequence[Any],
    pred_events: Sequence[Any],
    duration_sec: float,
    excluded_events: Sequence[Any] = (),
) -> dict[str, float | None]:
    """Calculate duration-weighted binary metrics for one class and video."""

    duration = float(duration_sec)
    excluded = interval_union(excluded_events, duration)
    excluded_duration = interval_union_duration(excluded)
    evaluation_duration = duration - excluded_duration
    truth = interval_union(true_events, duration)
    prediction = interval_union(pred_events, duration)
    true_duration = interval_union_duration(truth) - interval_overlap_duration(truth, excluded)
    pred_duration = interval_union_duration(prediction) - interval_overlap_duration(prediction, excluded)
    tp = interval_overlap_duration(truth, prediction) - interval_overlap_duration(
        interval_intersection(truth, prediction), excluded
    )
    fp = max(0.0, pred_duration - tp)
    fn = max(0.0, true_duration - tp)
    tn = max(0.0, evaluation_duration - tp - fp - fn)
    return {
        "source_duration_sec": duration,
        "evaluation_duration_sec": evaluation_duration,
        "excluded_duration_sec": excluded_duration,
        "true_duration_sec": true_duration,
        "pred_duration_sec": pred_duration,
        "tp_overlap_sec": tp,
        "fp_duration_sec": fp,
        "fn_duration_sec": fn,
        "tn_duration_sec": tn,
        "time_precision": _ratio(tp, tp + fp),
        "time_recall": _ratio(tp, tp + fn),
        "time_f1": _ratio(2.0 * tp, 2.0 * tp + fp + fn),
        "time_iou": _ratio(tp, tp + fp + fn),
        "time_accuracy": _ratio(tp + tn, evaluation_duration),
    }


def _prediction_map(predictions: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    records = predictions.get("videos")
    if not isinstance(records, list):
        raise ValueError("Prediction JSON must contain a videos list")
    result: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping) or not str(record.get("video", "")).strip():
            raise ValueError("Each prediction record must have a non-empty video")
        video = str(record["video"])
        if video in result:
            raise ValueError(f"Duplicate prediction video: {video!r}")
        result[video] = record
    return result


def _micro_metrics(rows: pd.DataFrame) -> dict[str, float | None]:
    totals = {
        name: float(rows[name].sum())
        for name in (
            "source_duration_sec",
            "evaluation_duration_sec",
            "excluded_duration_sec",
            "true_duration_sec",
            "pred_duration_sec",
            "tp_overlap_sec",
            "fp_duration_sec",
            "fn_duration_sec",
            "tn_duration_sec",
        )
    }
    tp = totals["tp_overlap_sec"]
    fp = totals["fp_duration_sec"]
    fn = totals["fn_duration_sec"]
    tn = totals["tn_duration_sec"]
    return {
        **totals,
        "time_precision": _ratio(tp, tp + fp),
        "time_recall": _ratio(tp, tp + fn),
        "time_f1": _ratio(2.0 * tp, 2.0 * tp + fp + fn),
        "time_iou": _ratio(tp, tp + fp + fn),
        "time_accuracy": _ratio(tp + tn, tp + fp + fn + tn),
    }


def _macro_metrics(rows: pd.DataFrame) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for name in ("time_precision", "time_recall", "time_f1", "time_iou", "time_accuracy"):
        values = pd.to_numeric(rows[name], errors="coerce").dropna().to_numpy(dtype=float)
        metrics[name] = float(np.mean(values)) if len(values) else None
        metrics[f"{name}_defined_video_count"] = int(len(values))
    return metrics


def evaluate_hybrid(
    annotations: Mapping[str, Any],
    predictions: Mapping[str, Any],
    source_durations_sec: Mapping[str, float],
    milk_iou_threshold: float = 0.5,
    exclude_unannotated_prediction_videos: bool = False,
    exclude_arching_cross_confusion: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Return the mixed-unit report, per-video arching rows and milk TP matches."""

    prediction_by_video = _prediction_map(predictions)
    truth_videos = {str(video) for video in annotations}
    prediction_only = sorted(set(prediction_by_video) - truth_videos)
    duration_only = sorted({str(video) for video in source_durations_sec} - truth_videos)
    if truth_videos - set(prediction_by_video):
        raise ValueError("Prediction and annotation video sets differ")
    if truth_videos - {str(video) for video in source_durations_sec}:
        raise ValueError("Duration and annotation video sets differ")
    if (prediction_only or duration_only) and not exclude_unannotated_prediction_videos:
        raise ValueError("Prediction and annotation video sets differ")
    filtered_predictions = {
        **predictions,
        "videos": [prediction_by_video[video] for video in sorted(truth_videos)],
    }
    filtered_durations = {
        video: float(source_durations_sec[video]) for video in sorted(truth_videos)
    }
    event_report, event_matches = evaluate_events(
        annotations,
        filtered_predictions,
        filtered_durations,
        iou_threshold=milk_iou_threshold,
    )
    milk_matches = event_matches[event_matches["class"] == "milk"].reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    cross_confusion_duration_by_video: dict[str, float] = {}
    for video in sorted(truth_videos):
        annotation = annotations[video]
        prediction = prediction_by_video[video]
        duration = float(source_durations_sec[video])
        excluded_arching_events: list[dict[str, float]] = []
        if exclude_arching_cross_confusion:
            true_partial = interval_union(annotation.partial_intervals, duration)
            true_collective = interval_union(annotation.collective_intervals, duration)
            pred_partial = interval_union(
                prediction.get("partial_arching_intervals", []), duration
            )
            pred_collective = interval_union(
                prediction.get("collective_arching_intervals", []), duration
            )
            confused = interval_intersection(true_partial, pred_collective) + interval_intersection(
                true_collective, pred_partial
            )
            excluded_arching_events = [
                {"start_sec": start, "end_sec": end} for start, end in confused
            ]
            excluded_arching_events = [
                {"start_sec": start, "end_sec": end}
                for start, end in interval_union(excluded_arching_events, duration)
            ]
        cross_confusion_duration_by_video[video] = interval_union_duration(
            [(item["start_sec"], item["end_sec"]) for item in excluded_arching_events]
        )
        for class_name, (annotation_field, prediction_field) in TEMPORAL_FIELDS.items():
            true_events = getattr(annotation, annotation_field)
            pred_events = prediction.get(prediction_field, [])
            if not isinstance(pred_events, list):
                raise ValueError(f"prediction[{video!r}].{prediction_field} must be a list")
            rows.append(
                {
                    "video": video,
                    "annotation_index": int(getattr(annotation, "index", 0)),
                    "class": class_name,
                    **temporal_overlap_metrics(
                        true_events,
                        pred_events,
                        duration,
                        excluded_events=(excluded_arching_events if class_name in ARCHING_FIELDS else ()),
                    ),
                }
            )

    per_video = pd.DataFrame(rows, columns=PER_VIDEO_COLUMNS)
    class_reports: dict[str, Any] = {}
    for class_name in TEMPORAL_FIELDS:
        subset = per_video[per_video["class"] == class_name]
        class_reports[class_name] = {
            "micro_duration_weighted": _micro_metrics(subset),
            "macro_video_average": _macro_metrics(subset),
        }

    report = {
        "schema_version": 1,
        "created_utc": _utc_now(),
        "evaluation_design": {
            "milk_event": "unchanged event-level deterministic greedy one-to-one matching",
            "milk_time": "continuous-time overlap for milk, reported alongside event metrics",
            "partial": "continuous-time overlap for the partial class; no event pairing",
            "collective": "continuous-time overlap for the collective class; no event pairing",
            "interval_semantics": "half-open intervals [start_sec, end_sec), clipped to source duration",
            "micro": "sum durations over all videos before calculating metrics",
            "macro": "mean of defined per-video metrics; undefined zero-denominator values are excluded",
            "arching_cross_confusion": (
                "exclude truth-partial/pred-collective and truth-collective/pred-partial time "
                "from both arching classes" if exclude_arching_cross_confusion else "included"
            ),
        },
        "video_count": len(truth_videos),
        "excluded_prediction_videos": prediction_only,
        "excluded_duration_videos": duration_only,
        "exclude_arching_cross_confusion": bool(exclude_arching_cross_confusion),
        "arching_cross_confusion_excluded_sec": float(sum(cross_confusion_duration_by_video.values())),
        "arching_cross_confusion_excluded_by_video_sec": cross_confusion_duration_by_video,
        "milk_event_metrics": {
            "iou_threshold": float(milk_iou_threshold),
            **event_report["classes"]["milk"],
        },
        "time_overlap_metrics": class_reports,
        "arching_time_overlap_metrics": {
            class_name: class_reports[class_name] for class_name in ARCHING_FIELDS
        },
    }
    return report, per_video, milk_matches


def _percent(value: Any) -> str:
    return "n.a." if value is None else f"{100.0 * float(value):.2f}%"


def _write_summary(path: Path, report: Mapping[str, Any]) -> None:
    milk = report["milk_event_metrics"]
    lines = [
        "# 放乳事件与分级拱乳时间重叠评价",
        "",
        f"视频数：{report['video_count']}。放乳按事件级 IoU≥{milk['iou_threshold']:.2f} 评价；两种拱乳强度分别按连续时间重叠评价。",
        "",
    ]
    if report.get("exclude_arching_cross_confusion"):
        lines.extend(
            [
                "部分拱乳与集体拱乳相互混淆的连续时间同时从两类拱乳统计中排除；放乳统计不变。",
                f"排除的相互混淆时间：{float(report['arching_cross_confusion_excluded_sec']):.1f} s。",
                "",
            ]
        )
    lines.extend(
        [
            "## 放乳（事件级）",
            "",
            "| TP | FP | FN | Precision | Recall | F1 |",
            "|---:|---:|---:|---:|---:|---:|",
            f"| {milk['tp']} | {milk['fp']} | {milk['fn']} | {_percent(milk['precision'])} | {_percent(milk['recall'])} | {_percent(milk['f1'])} |",
            "",
            "## 两类拱乳行为（时间级，全部视频累计）",
            "",
            "| 类别 | 人工时长(s) | 预测时长(s) | 重叠TP(s) | FP(s) | FN(s) | 时间Precision | 时间Recall | 时间F1 | 时间IoU |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    labels = {"partial": "部分拱乳", "collective": "集体拱乳"}
    for class_name in ("partial", "collective"):
        metric = report["time_overlap_metrics"][class_name]["micro_duration_weighted"]
        lines.append(
            f"| {labels[class_name]} | {metric['true_duration_sec']:.1f} | {metric['pred_duration_sec']:.1f} | "
            f"{metric['tp_overlap_sec']:.1f} | {metric['fp_duration_sec']:.1f} | {metric['fn_duration_sec']:.1f} | "
            f"{_percent(metric['time_precision'])} | {_percent(metric['time_recall'])} | "
            f"{_percent(metric['time_f1'])} | {_percent(metric['time_iou'])} |"
        )
    lines.extend(
        [
            "",
            "时间Recall表示人工标注行为时长中被模型命中的比例。时间Precision用于约束把过多非该类别时间预测为该类别的问题。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--milk-iou-threshold", type=float, default=0.5)
    parser.add_argument(
        "--exclude-unannotated-prediction-videos",
        action="store_true",
        help="Ignore prediction/manifest videos absent from the annotation file.",
    )
    parser.add_argument(
        "--exclude-arching-cross-confusion",
        action="store_true",
        help="Exclude truth-partial/pred-collective and truth-collective/pred-partial time from both arching class metrics.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
    report, per_video, milk_matches = evaluate_hybrid(
        annotations,
        predictions,
        durations,
        milk_iou_threshold=float(args.milk_iou_threshold),
        exclude_unannotated_prediction_videos=bool(args.exclude_unannotated_prediction_videos),
        exclude_arching_cross_confusion=bool(args.exclude_arching_cross_confusion),
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "hybrid_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    per_video.to_csv(output / "time_overlap_by_video.csv", index=False, encoding="utf-8-sig")
    per_video[per_video["class"].isin(ARCHING_FIELDS)].to_csv(
        output / "arching_time_overlap_by_video.csv", index=False, encoding="utf-8-sig"
    )
    milk_matches.to_csv(output / "milk_tp_event_matches.csv", index=False, encoding="utf-8-sig")
    _write_summary(output / "summary.md", report)

    milk_time = report["time_overlap_metrics"]["milk"]["micro_duration_weighted"]
    partial = report["time_overlap_metrics"]["partial"]["micro_duration_weighted"]
    collective = report["time_overlap_metrics"]["collective"]["micro_duration_weighted"]
    milk = report["milk_event_metrics"]
    print(
        "[done] "
        f"videos={report['video_count']} "
        f"milk_f1={milk['f1']:.6f} "
        f"milk_time_f1={milk_time['time_f1']:.6f} "
        f"partial_time_f1={partial['time_f1']:.6f} "
        f"collective_time_f1={collective['time_f1']:.6f} "
        f"output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
