"""Extract normal optical-flow intensity inside the piglet/sow nursing contact band.

The script reuses the sow and piglet segmentation models plus the nursing-band
geometry from ``extract_nursing_band.py``.  It does not classify massage.  It
only writes an auditable per-frame-pair normal-flow signal for later rule/model
development.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.extract_nursing_band import (  # noqa: E402
    DEFAULT_PIGLET_WEIGHTS,
    DEFAULT_SOW_WEIGHTS,
    VIDEO_EXTS,
    InstanceMask,
    extract_band,
    mask_for_shape,
    partition_piglets_by_sow_overlap,
    predict_labeled_instances,
    select_sow_instance,
    sow_mask_change_ratio,
)


# Directory for input nursing video clips (not bundled in release)
DEFAULT_INPUT = ROOT / "working_data" / "nursing_clips"
DEFAULT_OUTPUT = ROOT / "runs" / "nursing_normal_flow"
FLOW_METRIC_FIELDS = (
    "normal_signed_mean_px_s",
    "normal_signed_median_px_s",
    "normal_abs_mean_px_s",
    "normal_abs_median_px_s",
    "normal_abs_p75_px_s",
    "normal_abs_p90_px_s",
    "normal_rms_px_s",
    "normal_outward_fraction",
)
CSV_FIELDS = (
    "video",
    "sample_index",
    "prev_frame_index",
    "frame_index",
    "interval_start_sec",
    "timestamp_sec",
    "interval_end_sec",
    "dt_sec",
    "source_fps",
    "actual_flow_fps",
    "source_mask_refreshed",
    "current_mask_refreshed",
    "roi_segment_id",
    "roi_refreshed",
    "flow_valid",
    "status",
    "source_piglet_count",
    "source_contact_piglet_count",
    "source_contact_pixel_count",
    "valid_pixel_count",
    "valid_band_fraction",
    *FLOW_METRIC_FIELDS,
)


@dataclass(frozen=True)
class RoiState:
    segment_id: int
    band: np.ndarray
    sow: InstanceMask
    arc_points: np.ndarray
    normal_field_crop: np.ndarray
    crop_xyxy: tuple[int, int, int, int]
    band_pixel_count: int


@dataclass(frozen=True)
class FlowFrameState:
    frame_index: int
    timestamp_sec: float
    segment_id: int
    mask_refreshed: bool
    gray_crop: np.ndarray
    contact_mask_crop: np.ndarray
    piglet_count: int
    contact_piglet_count: int
    contact_pixel_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Video file or directory.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--flow-fps", type=float, default=12.5)
    parser.add_argument("--mask-fps", type=float, default=4.0)
    parser.add_argument("--sow-check-sec", type=float, default=5.0)
    parser.add_argument("--roi-change-threshold", type=float, default=0.10)
    parser.add_argument("--min-roi-contacts", type=int, default=3)
    parser.add_argument("--flow-crop-margin", type=int, default=64)
    parser.add_argument("--contact-erode-px", type=int, default=2)
    parser.add_argument("--sow-exclusion-px", type=int, default=2)
    parser.add_argument("--min-flow-pixels", type=int, default=128)
    parser.add_argument("--normal-step-px", type=float, default=1.0)
    parser.add_argument("--normal-span-px", type=float, default=15.0)
    parser.add_argument("--normal-probe-px", type=float, default=6.0)
    parser.add_argument("--progress-sec", type=float, default=30.0)

    parser.add_argument("--piglet-weights", type=Path, default=DEFAULT_PIGLET_WEIGHTS)
    parser.add_argument("--sow-weights", type=Path, default=DEFAULT_SOW_WEIGHTS)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default="")
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--retina-masks", action="store_true")
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--sow-label", default="sow")
    parser.add_argument("--piglet-label", default="piglet")
    parser.add_argument("--min-sow-area", type=int, default=1000)
    parser.add_argument("--min-piglet-area", type=int, default=80)
    parser.add_argument("--max-piglet-sow-overlap", type=float, default=0.60)

    # Keep these defaults aligned with extract_nursing_band.py because extract_band() consumes them.
    parser.add_argument("--max-contact-dist", type=float, default=48.0)
    parser.add_argument("--contact-margin", type=float, default=12.0)
    parser.add_argument("--cluster-gap", type=float, default=360.0)
    parser.add_argument("--head-projection-quantile", type=float, default=0.9)
    parser.add_argument("--min-main-piglets", type=int, default=3)
    parser.add_argument("--arc-padding", type=float, default=70.0)
    parser.add_argument("--band-width", type=int, default=80)
    parser.add_argument("--band-outer-width", type=float, default=75.0)
    parser.add_argument("--band-inner-width", type=float, default=35.0)
    parser.add_argument("--centerline-thickness", type=int, default=5)
    parser.add_argument("--contour-smooth-window", type=float, default=25.0)
    parser.add_argument("--turn-window", type=float, default=60.0)
    parser.add_argument("--max-turn-angle", type=float, default=50.0)
    parser.add_argument("--min-turn-radius", type=float, default=35.0)
    parser.add_argument("--max-axis-deviation", type=float, default=30.0)
    parser.add_argument("--max-contour-bridge", type=float, default=180.0)
    parser.add_argument("--max-endpoint-extension", type=float, default=140.0)

    parser.add_argument("--plot-dpi", type=int, default=160)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def iter_videos(input_path: Path, recursive: bool = False) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in VIDEO_EXTS:
            raise ValueError(f"Unsupported video: {input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input does not exist: {input_path}")
    pattern = "**/*" if recursive else "*"
    videos = sorted(path for path in input_path.glob(pattern) if path.is_file() and path.suffix.lower() in VIDEO_EXTS)
    if not videos:
        raise FileNotFoundError(f"No videos found in {input_path}")
    return videos


def output_base(video_path: Path, input_path: Path, output_dir: Path) -> Path:
    """Return a collision-safe output stem while preserving recursive subdirectories."""
    relative = video_path.relative_to(input_path) if input_path.is_dir() else Path(video_path.name)
    extension = relative.suffix.lower().lstrip(".") or "video"
    return output_dir / relative.parent / f"{relative.stem}__{extension}"


def _resample_polyline(points_xy: np.ndarray, step_px: float) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    if len(points) < 2 or step_px <= 0:
        raise ValueError("A positive step and at least two arc points are required.")
    keep = np.ones(len(points), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-5
    points = points[keep]
    if len(points) < 2:
        raise ValueError("Arc points collapse to one unique point.")
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    total = float(cumulative[-1])
    if total <= 1e-5:
        raise ValueError("Arc length is zero.")
    samples = np.arange(0.0, total, float(step_px), dtype=np.float32)
    if len(samples) == 0 or total - float(samples[-1]) > 1e-4:
        samples = np.append(samples, total)
    x = np.interp(samples, cumulative, points[:, 0])
    y = np.interp(samples, cumulative, points[:, 1])
    return np.column_stack((x, y)).astype(np.float32)


def arc_outward_normals(
    arc_points: np.ndarray,
    sow_mask: np.ndarray,
    step_px: float = 1.0,
    span_px: float = 15.0,
    probe_px: float = 6.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return resampled arc points and consistently outward unit normals."""
    points = _resample_polyline(arc_points, step_px)
    offset = max(1, int(round(max(step_px, span_px / 2.0) / step_px)))
    indices = np.arange(len(points))
    before = points[np.maximum(0, indices - offset)]
    after = points[np.minimum(len(points) - 1, indices + offset)]
    tangents = after - before
    tangent_norm = np.linalg.norm(tangents, axis=1)
    if np.any(tangent_norm <= 1e-6):
        raise ValueError("Could not estimate a finite tangent along the nursing arc.")
    tangents /= tangent_norm[:, None]
    normals = np.column_stack((-tangents[:, 1], tangents[:, 0])).astype(np.float32)

    sow = sow_mask.astype(bool, copy=False)
    height, width = sow.shape

    def inside_mask(probes: np.ndarray) -> np.ndarray:
        xy = np.rint(probes).astype(np.int32)
        valid = (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)
        inside = np.zeros(len(xy), dtype=bool)
        inside[valid] = sow[xy[valid, 1], xy[valid, 0]]
        return inside

    plus_inside = inside_mask(points + normals * float(probe_px))
    minus_inside = inside_mask(points - normals * float(probe_px))
    outward_votes = int(np.count_nonzero(~plus_inside & minus_inside))
    inward_votes = int(np.count_nonzero(plus_inside & ~minus_inside))
    if inward_votes > outward_votes:
        normals = -normals
    elif inward_votes == outward_votes:
        ys, xs = np.nonzero(sow)
        if len(xs) == 0:
            raise ValueError("Sow mask is empty.")
        centroid = np.asarray([float(xs.mean()), float(ys.mean())], dtype=np.float32)
        if float(np.median(np.sum(normals * (points - centroid), axis=1))) < 0:
            normals = -normals
    return points, normals


def normal_field_from_arc(
    image_shape: tuple[int, int],
    band: np.ndarray,
    arc_points: np.ndarray,
    sow_mask: np.ndarray,
    step_px: float = 1.0,
    span_px: float = 15.0,
    probe_px: float = 6.0,
) -> np.ndarray:
    """Assign each band pixel the unit normal of its nearest nursing-arc seed."""
    height, width = image_shape
    points, normals = arc_outward_normals(arc_points, sow_mask, step_px, span_px, probe_px)
    xy = np.rint(points).astype(np.int32)
    valid = (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)
    xy = xy[valid]
    normals = normals[valid]
    if len(xy) == 0:
        raise ValueError("No nursing-arc seeds fall inside the image.")

    grouped: dict[tuple[int, int], list[np.ndarray]] = {}
    for pixel, normal in zip(xy, normals):
        grouped.setdefault((int(pixel[0]), int(pixel[1])), []).append(normal)
    seed_image = np.ones((height, width), dtype=np.uint8)
    seed_normals: dict[tuple[int, int], np.ndarray] = {}
    for (x, y), values in grouped.items():
        vector = np.mean(np.asarray(values), axis=0)
        length = float(np.linalg.norm(vector))
        if length <= 1e-6:
            continue
        seed_image[y, x] = 0
        seed_normals[(x, y)] = (vector / length).astype(np.float32)
    if not seed_normals:
        raise ValueError("No valid arc normals could be rasterized.")

    _, labels = cv2.distanceTransformWithLabels(
        seed_image,
        cv2.DIST_L2,
        cv2.DIST_MASK_5,
        labelType=cv2.DIST_LABEL_PIXEL,
    )
    lookup = np.zeros((int(labels.max()) + 1, 2), dtype=np.float32)
    counts = np.zeros(len(lookup), dtype=np.int32)
    for (x, y), normal in seed_normals.items():
        label = int(labels[y, x])
        if label <= 0:
            continue
        lookup[label] += normal
        counts[label] += 1
    populated = counts > 0
    lookup[populated] /= counts[populated, None]
    lengths = np.linalg.norm(lookup, axis=1)
    nonzero = lengths > 1e-6
    lookup[nonzero] /= lengths[nonzero, None]

    field = lookup[labels]
    band_bool = mask_for_shape(band, image_shape)
    field[~band_bool] = 0.0
    field_norm = np.linalg.norm(field, axis=2)
    valid_field = band_bool & (field_norm > 1e-6)
    field[valid_field] /= field_norm[valid_field, None]
    return field.astype(np.float32)


def safe_contact_mask(
    piglets: list[InstanceMask],
    band: np.ndarray,
    sow_mask: np.ndarray,
    erode_px: int = 2,
    sow_exclusion_px: int = 2,
) -> tuple[np.ndarray, int]:
    """Build a conservative visible-piglet mask on the outside half of the band."""
    shape = band.shape[:2]
    band_bool = mask_for_shape(band, shape)
    sow = mask_for_shape(sow_mask, shape)
    radius = max(0, int(sow_exclusion_px))
    if radius:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        sow = cv2.dilate(sow.astype(np.uint8), kernel).astype(bool)
    support = band_bool & ~sow

    union = np.zeros(shape, dtype=bool)
    contact_count = 0
    erode_radius = max(0, int(erode_px))
    erode_kernel = (
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_radius + 1, 2 * erode_radius + 1))
        if erode_radius
        else None
    )
    for piglet in piglets:
        mask = mask_for_shape(piglet.mask, shape)
        if erode_kernel is not None:
            mask = cv2.erode(mask.astype(np.uint8), erode_kernel).astype(bool)
        contact = mask & support
        if np.any(contact):
            contact_count += 1
            union |= contact
    return union, contact_count


def farneback_flow(previous_gray: np.ndarray, current_gray: np.ndarray) -> np.ndarray:
    if previous_gray.shape != current_gray.shape:
        raise ValueError("Farneback input frames must have the same shape.")
    previous = cv2.GaussianBlur(previous_gray, (5, 5), 0)
    current = cv2.GaussianBlur(current_gray, (5, 5), 0)
    return cv2.calcOpticalFlowFarneback(
        previous,
        current,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=21,
        iterations=3,
        poly_n=7,
        poly_sigma=1.5,
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
    )


def normal_flow_statistics(
    flow: np.ndarray,
    normal_field: np.ndarray,
    contact_mask: np.ndarray,
    dt_sec: float,
    min_pixels: int = 128,
) -> dict[str, float | int | bool | None]:
    if dt_sec <= 0:
        raise ValueError("dt_sec must be positive.")
    if flow.shape != normal_field.shape or flow.shape[:2] != contact_mask.shape:
        raise ValueError("Flow, normal field, and contact mask shapes do not match.")
    field_norm = np.linalg.norm(normal_field, axis=2)
    finite = np.isfinite(flow).all(axis=2) & np.isfinite(normal_field).all(axis=2)
    valid = contact_mask.astype(bool) & finite & (field_norm > 1e-6)
    count = int(np.count_nonzero(valid))
    empty: dict[str, float | int | bool | None] = {
        "flow_valid": False,
        "valid_pixel_count": count,
        **{key: None for key in FLOW_METRIC_FIELDS},
    }
    if count < max(1, int(min_pixels)):
        return empty
    signed = np.sum(flow[valid] * normal_field[valid], axis=1).astype(np.float64) / float(dt_sec)
    absolute = np.abs(signed)
    return {
        "flow_valid": True,
        "valid_pixel_count": count,
        "normal_signed_mean_px_s": float(np.mean(signed)),
        "normal_signed_median_px_s": float(np.median(signed)),
        "normal_abs_mean_px_s": float(np.mean(absolute)),
        "normal_abs_median_px_s": float(np.median(absolute)),
        "normal_abs_p75_px_s": float(np.percentile(absolute, 75)),
        "normal_abs_p90_px_s": float(np.percentile(absolute, 90)),
        "normal_rms_px_s": float(np.sqrt(np.mean(signed**2))),
        "normal_outward_fraction": float(np.mean(signed > 0)),
    }


def _crop_bounds(mask: np.ndarray, margin: int) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("Cannot crop an empty nursing band.")
    height, width = mask.shape[:2]
    pad = max(0, int(margin))
    return (
        max(0, int(xs.min()) - pad),
        max(0, int(ys.min()) - pad),
        min(width, int(xs.max()) + pad + 1),
        min(height, int(ys.max()) + pad + 1),
    )


def _crop(array: np.ndarray, bounds: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = bounds
    return array[y0:y1, x0:x1]


def build_roi_state(
    sow: InstanceMask,
    piglets: list[InstanceMask],
    image_shape: tuple[int, int],
    args: argparse.Namespace,
    segment_id: int,
) -> RoiState:
    band, _, arc_points, accepted_sow, _, _, selected, _ = extract_band([sow, *piglets], image_shape, args)
    if len(selected) < max(1, int(args.min_roi_contacts)):
        raise ValueError(f"Only {len(selected)} contacts support the candidate ROI.")
    normal_field = normal_field_from_arc(
        image_shape,
        band,
        arc_points,
        accepted_sow.mask,
        step_px=args.normal_step_px,
        span_px=args.normal_span_px,
        probe_px=args.normal_probe_px,
    )
    bounds = _crop_bounds(band, args.flow_crop_margin)
    return RoiState(
        segment_id=segment_id,
        band=band,
        sow=accepted_sow,
        arc_points=arc_points,
        normal_field_crop=_crop(normal_field, bounds),
        crop_xyxy=bounds,
        band_pixel_count=int(np.count_nonzero(band)),
    )


def _infer_sow(model: Any, frame: np.ndarray, args: argparse.Namespace) -> InstanceMask:
    instances = predict_labeled_instances(model, frame, args, args.sow_label, start_index=0)
    return select_sow_instance(instances, args)


def _infer_raw_piglets(model: Any, frame: np.ndarray, args: argparse.Namespace) -> list[InstanceMask]:
    instances = predict_labeled_instances(model, frame, args, args.piglet_label, start_index=1)
    return [instance for instance in instances if instance.area >= args.min_piglet_area]


def _filter_piglets(
    piglets: list[InstanceMask],
    sow: InstanceMask,
    max_overlap_ratio: float,
) -> list[InstanceMask]:
    kept, _, _ = partition_piglets_by_sow_overlap(piglets, sow, max_overlap_ratio=max_overlap_ratio)
    return kept


def _empty_flow_metrics() -> dict[str, float | int | bool | None]:
    return {
        "flow_valid": False,
        "valid_pixel_count": 0,
        **{key: None for key in FLOW_METRIC_FIELDS},
    }


def _make_row(
    *,
    video_name: str,
    sample_index: int,
    current_frame_index: int,
    current_timestamp_sec: float,
    source_fps: float,
    actual_flow_fps: float,
    mask_refreshed: bool,
    roi: RoiState | None,
    roi_refreshed: bool,
    status: str,
    metrics: dict[str, float | int | bool | None],
    previous: FlowFrameState | None,
) -> dict[str, Any]:
    valid_pixels = int(metrics.get("valid_pixel_count", 0) or 0)
    row: dict[str, Any] = {
        "video": video_name,
        "sample_index": sample_index,
        "prev_frame_index": previous.frame_index if previous is not None else "",
        "frame_index": current_frame_index,
        "interval_start_sec": round(previous.timestamp_sec, 6) if previous is not None else "",
        "timestamp_sec": round((previous.timestamp_sec + current_timestamp_sec) / 2.0, 6)
        if previous is not None
        else round(current_timestamp_sec, 6),
        "interval_end_sec": round(current_timestamp_sec, 6),
        "dt_sec": round(current_timestamp_sec - previous.timestamp_sec, 6) if previous is not None else "",
        "source_fps": source_fps,
        "actual_flow_fps": actual_flow_fps,
        "source_mask_refreshed": int(previous.mask_refreshed) if previous is not None else 0,
        "current_mask_refreshed": int(mask_refreshed),
        "roi_segment_id": roi.segment_id if roi is not None else "",
        "roi_refreshed": int(roi_refreshed),
        "flow_valid": int(bool(metrics.get("flow_valid"))),
        "status": status,
        "source_piglet_count": previous.piglet_count if previous is not None else 0,
        "source_contact_piglet_count": previous.contact_piglet_count if previous is not None else 0,
        "source_contact_pixel_count": previous.contact_pixel_count if previous is not None else 0,
        "valid_pixel_count": valid_pixels,
        "valid_band_fraction": float(valid_pixels / roi.band_pixel_count) if roi is not None and roi.band_pixel_count else 0.0,
    }
    row.update({key: metrics.get(key) for key in FLOW_METRIC_FIELDS})
    return row


def _rolling_nanmedian(values: np.ndarray, window: int) -> np.ndarray:
    output = np.full(len(values), np.nan, dtype=float)
    width = max(1, int(window))
    for index in range(len(values)):
        chunk = values[max(0, index - width + 1) : index + 1]
        finite = chunk[np.isfinite(chunk)]
        if len(finite):
            output[index] = float(np.median(finite))
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_rows(path: Path, rows: list[dict[str, Any]], video_name: str, actual_flow_fps: float, dpi: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = np.asarray([float(row["timestamp_sec"]) for row in rows], dtype=float)
    p75 = np.asarray(
        [float(row["normal_abs_p75_px_s"]) if row.get("normal_abs_p75_px_s") not in {None, ""} else np.nan for row in rows],
        dtype=float,
    )
    pixels = np.asarray([float(row.get("valid_pixel_count", 0) or 0) for row in rows], dtype=float)
    smooth = _rolling_nanmedian(p75, max(1, int(round(actual_flow_fps))))
    refresh_times = [float(row["timestamp_sec"]) for row in rows if int(row.get("roi_refreshed", 0) or 0)]

    figure, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    axes[0].plot(times / 60.0, p75, color="#A7B0BE", linewidth=0.7, alpha=0.65, label="raw abs normal-flow P75")
    axes[0].plot(times / 60.0, smooth, color="#2563EB", linewidth=1.6, label="1 s rolling median")
    axes[0].set_ylabel("Normal flow intensity (px/s)")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.2)
    axes[1].plot(times / 60.0, pixels, color="#0F766E", linewidth=0.9)
    axes[1].set_ylabel("Valid pixels")
    axes[1].set_xlabel("Source video time (min)")
    axes[1].grid(alpha=0.2)
    for timestamp in refresh_times:
        for axis in axes:
            axis.axvline(timestamp / 60.0, color="#7C3AED", linestyle=":", linewidth=0.8, alpha=0.7)
    figure.suptitle(video_name)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def process_video(piglet_model: Any, sow_model: Any, video_path: Path, args: argparse.Namespace) -> tuple[Path, Path]:
    stem = output_base(video_path, args.input, args.output)
    csv_path = stem.with_name(f"{stem.name}_normal_flow.csv")
    plot_path = stem.with_name(f"{stem.name}_normal_flow_plot.png")
    for path in (csv_path, plot_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite to replace it.")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if source_fps <= 0 or frame_count <= 0:
        capture.release()
        raise ValueError(f"Invalid video metadata for {video_path}")
    flow_step = max(1, int(round(source_fps / args.flow_fps)))
    actual_flow_fps = source_fps / flow_step
    mask_stride = max(1, int(round(actual_flow_fps / args.mask_fps)))
    expected_dt = flow_step / source_fps
    start_frame = min(frame_count - 1, max(0, int(round(args.start_sec * source_fps))))
    if args.max_seconds is None:
        end_frame = frame_count
    else:
        end_frame = min(frame_count, start_frame + max(1, int(math.ceil(args.max_seconds * source_fps))))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    rows: list[dict[str, Any]] = []
    roi: RoiState | None = None
    latest_sow: InstanceMask | None = None
    latest_piglets: list[InstanceMask] = []
    latest_contact_mask: np.ndarray | None = None
    latest_contact_count = 0
    previous: FlowFrameState | None = None
    next_roi_segment_id = 0
    next_sow_check_sec = args.start_sec
    sample_index = 0
    frame_index = start_frame
    last_progress_sec = args.start_sec - args.progress_sec

    while frame_index < end_frame:
        ok, frame = capture.read()
        if not ok:
            break
        if (frame_index - start_frame) % flow_step != 0:
            frame_index += 1
            continue
        timestamp_sec = frame_index / source_fps
        mask_refreshed = sample_index % mask_stride == 0
        roi_refreshed = False
        event_status = "ok"
        candidate_sow: InstanceMask | None = None

        if mask_refreshed:
            sow_due = latest_sow is None or timestamp_sec + 1e-9 >= next_sow_check_sec
            if sow_due:
                try:
                    candidate_sow = _infer_sow(sow_model, frame, args)
                    if latest_sow is None:
                        latest_sow = candidate_sow
                except Exception as exc:
                    event_status = f"sow_inference_failed:{type(exc).__name__}"
                next_sow_check_sec = timestamp_sec + max(expected_dt, args.sow_check_sec)
            try:
                raw_piglets = _infer_raw_piglets(piglet_model, frame, args)
            except Exception as exc:
                raw_piglets = []
                event_status = f"piglet_inference_failed:{type(exc).__name__}"

            reference_sow = candidate_sow or (roi.sow if roi is not None else latest_sow)
            filtered_for_candidate = (
                _filter_piglets(raw_piglets, reference_sow, args.max_piglet_sow_overlap)
                if reference_sow is not None
                else []
            )
            if roi is None and reference_sow is not None:
                try:
                    roi = build_roi_state(
                        reference_sow,
                        filtered_for_candidate,
                        frame.shape[:2],
                        args,
                        segment_id=next_roi_segment_id,
                    )
                    next_roi_segment_id += 1
                    latest_sow = roi.sow
                    latest_piglets = _filter_piglets(raw_piglets, roi.sow, args.max_piglet_sow_overlap)
                    roi_refreshed = True
                    event_status = "initial_roi"
                except Exception as exc:
                    latest_piglets = filtered_for_candidate
                    event_status = f"no_roi:{type(exc).__name__}"
            elif roi is not None:
                refreshed = False
                if candidate_sow is not None:
                    change = sow_mask_change_ratio(roi.sow.mask, candidate_sow.mask)
                    if change >= args.roi_change_threshold:
                        candidate_piglets = filtered_for_candidate
                        try:
                            roi = build_roi_state(
                                candidate_sow,
                                candidate_piglets,
                                frame.shape[:2],
                                args,
                                segment_id=next_roi_segment_id,
                            )
                            next_roi_segment_id += 1
                            latest_sow = roi.sow
                            latest_piglets = candidate_piglets
                            roi_refreshed = True
                            refreshed = True
                            event_status = "roi_refreshed"
                        except Exception as exc:
                            event_status = f"roi_refresh_rejected:{type(exc).__name__}"
                            latest_sow = candidate_sow
                            latest_piglets = candidate_piglets
                            latest_contact_mask = None
                            latest_contact_count = 0
                            previous = None
                            roi = None
                if not refreshed and roi is not None:
                    latest_piglets = _filter_piglets(raw_piglets, roi.sow, args.max_piglet_sow_overlap)

            if roi is not None:
                latest_contact_mask, latest_contact_count = safe_contact_mask(
                    latest_piglets,
                    roi.band,
                    roi.sow.mask,
                    erode_px=args.contact_erode_px,
                    sow_exclusion_px=args.sow_exclusion_px,
                )

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if roi is None or latest_contact_mask is None:
            metrics = _empty_flow_metrics()
            rows.append(
                _make_row(
                    video_name=video_path.name,
                    sample_index=sample_index,
                    current_frame_index=frame_index,
                    current_timestamp_sec=timestamp_sec,
                    source_fps=source_fps,
                    actual_flow_fps=actual_flow_fps,
                    mask_refreshed=mask_refreshed,
                    roi=roi,
                    roi_refreshed=roi_refreshed,
                    status=event_status if event_status != "ok" else "no_roi",
                    metrics=metrics,
                    previous=None,
                )
            )
            previous = None
        else:
            gray_crop = _crop(gray, roi.crop_xyxy)
            contact_crop = _crop(latest_contact_mask, roi.crop_xyxy)
            current = FlowFrameState(
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
                segment_id=roi.segment_id,
                mask_refreshed=mask_refreshed,
                gray_crop=gray_crop,
                contact_mask_crop=contact_crop,
                piglet_count=len(latest_piglets),
                contact_piglet_count=latest_contact_count,
                contact_pixel_count=int(np.count_nonzero(contact_crop)),
            )
            if roi_refreshed or previous is None or previous.segment_id != roi.segment_id:
                metrics = _empty_flow_metrics()
                status = event_status if roi_refreshed else "flow_initialized"
            else:
                dt_sec = timestamp_sec - previous.timestamp_sec
                if dt_sec <= 0 or dt_sec > expected_dt * 1.75:
                    metrics = _empty_flow_metrics()
                    status = "time_gap_reset"
                else:
                    flow = farneback_flow(previous.gray_crop, gray_crop)
                    metrics = normal_flow_statistics(
                        flow,
                        roi.normal_field_crop,
                        previous.contact_mask_crop,
                        dt_sec=dt_sec,
                        min_pixels=args.min_flow_pixels,
                    )
                    status = "ok" if metrics["flow_valid"] else "insufficient_contact"
                    if event_status != "ok":
                        status = f"{status}|{event_status}"
            rows.append(
                _make_row(
                    video_name=video_path.name,
                    sample_index=sample_index,
                    current_frame_index=frame_index,
                    current_timestamp_sec=timestamp_sec,
                    source_fps=source_fps,
                    actual_flow_fps=actual_flow_fps,
                    mask_refreshed=mask_refreshed,
                    roi=roi,
                    roi_refreshed=roi_refreshed,
                    status=status,
                    metrics=metrics,
                    previous=None if roi_refreshed else previous,
                )
            )
            previous = current

        if args.progress_sec > 0 and timestamp_sec - last_progress_sec >= args.progress_sec:
            print(
                f"[INFO] {video_path.name}: t={timestamp_sec:.1f}s "
                f"rows={len(rows)} valid={sum(int(row['flow_valid']) for row in rows)}"
            )
            last_progress_sec = timestamp_sec
        sample_index += 1
        frame_index += 1

    capture.release()
    if not rows:
        raise RuntimeError(f"No sampled frames were read from {video_path}")
    write_csv(csv_path, rows)
    plot_rows(plot_path, rows, video_path.name, actual_flow_fps, args.plot_dpi)
    valid_count = sum(int(row["flow_valid"]) for row in rows)
    print(f"[INFO] wrote {csv_path} ; {plot_path} (rows={len(rows)}, valid={valid_count})")
    return csv_path, plot_path


def validate_args(args: argparse.Namespace) -> None:
    if args.flow_fps <= 0 or args.mask_fps <= 0:
        raise ValueError("--flow-fps and --mask-fps must be positive.")
    if args.start_sec < 0 or (args.max_seconds is not None and args.max_seconds <= 0):
        raise ValueError("--start-sec must be non-negative and --max-seconds must be positive.")
    if args.min_flow_pixels <= 0 or args.min_roi_contacts <= 0:
        raise ValueError("Minimum pixel/contact counts must be positive.")
    if not 0 <= args.max_piglet_sow_overlap <= 1:
        raise ValueError("--max-piglet-sow-overlap must be between 0 and 1.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    videos = iter_videos(args.input, args.recursive)
    if args.max_videos is not None:
        videos = videos[: max(0, args.max_videos)]
    if not videos:
        raise ValueError("No videos selected.")
    if args.dry_run:
        print(f"Input: {args.input}")
        print(f"Videos: {len(videos)}")
        print(f"Flow FPS requested: {args.flow_fps}")
        print(f"Mask FPS requested: {args.mask_fps}")
        print(f"Output: {args.output}")
        for video in videos:
            print(video)
        return
    for weights in (args.piglet_weights, args.sow_weights):
        if not weights.exists():
            raise FileNotFoundError(f"Weights not found: {weights}")

    from ultralytics import YOLO

    args.output.mkdir(parents=True, exist_ok=True)
    piglet_model = YOLO(str(args.piglet_weights), task="segment")
    sow_model = YOLO(str(args.sow_weights), task="segment")
    failed = 0
    for index, video_path in enumerate(videos, start=1):
        try:
            print(f"[INFO] processing {index}/{len(videos)}: {video_path}")
            process_video(piglet_model, sow_model, video_path, args)
        except Exception as exc:
            failed += 1
            if args.fail_on_error:
                raise
            print(f"[WARN] failed {video_path}: {type(exc).__name__}: {exc}")
    print(f"Done. videos={len(videos)}, failed={failed}, output={args.output}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
