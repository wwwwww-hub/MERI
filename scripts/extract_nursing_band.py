"""Extract nursing-band masks from a directory of sow/piglet images."""

from __future__ import annotations

import argparse
import csv
import json
import math
import queue
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PIGLET_WEIGHTS = ROOT / "models" / "segmentation" / "piglet.pt"
DEFAULT_SOW_WEIGHTS = ROOT / "models" / "segmentation" / "sow.pt"
DEFAULT_OUTPUT = ROOT / "runs" / "nursing_band"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"}


@dataclass
class InstanceMask:
    index: int
    class_id: int
    label: str
    confidence: float
    mask: np.ndarray

    @property
    def area(self) -> int:
        return int(self.mask.sum())


@dataclass
class Contact:
    piglet_index: int
    confidence: float
    min_distance_px: float
    contact_center: tuple[float, float]
    contour_index: int
    contour_s: float
    contour_point: tuple[int, int]


@dataclass(frozen=True)
class PigletGeometry:
    centroid: tuple[float, float]
    bbox: tuple[int, int, int, int]
    area: int


@dataclass(frozen=True)
class EdgeConfirmationStats:
    edge_candidates: int
    pending_edge_candidates: int
    confirmed_edge_candidates: int = 0
    edge_candidate_area_px: int = 0
    pending_edge_area_px: int = 0
    edge_hit_ratio_mean: float = 0.0


@dataclass(frozen=True)
class InitialRoiCandidate:
    timestamp_sec: float
    band: np.ndarray
    sow: InstanceMask
    selected_contact_count: int
    core_band: np.ndarray | None = None


@dataclass(frozen=True)
class RoiRefreshCandidate:
    timestamp_sec: float
    band: np.ndarray
    core_band: np.ndarray
    sow: InstanceMask
    selected_contacts: tuple[Contact, ...]
    current_contact_coverage: float
    candidate_contact_coverage: float


@dataclass(frozen=True)
class RoiContactRecoverySignature:
    """Stable, pose-normalized location of an alternative contact cluster."""

    side: int
    longitudinal_center_ratio: float


@dataclass(frozen=True)
class HampelResult:
    values: list[float]
    flags: list[bool]
    replaced: list[bool]
    local_medians: list[float]
    thresholds: list[float]


class AsyncVideoWriter:
    def __init__(self, path: Path, fps: float, size: tuple[int, int], queue_size: int) -> None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(path), fourcc, float(fps), size)
        if not self.writer.isOpened():
            raise ValueError(f"Could not open demo video writer: {path}")
        self.frames: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=max(2, queue_size))
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._write_loop, name="nursing-demo-writer", daemon=True)
        self.thread.start()

    def _write_loop(self) -> None:
        try:
            while True:
                frame = self.frames.get()
                if frame is None:
                    break
                self.writer.write(frame)
        except BaseException as exc:
            self.error = exc
        finally:
            self.writer.release()

    def _put(self, frame: np.ndarray | None) -> None:
        while True:
            if self.error is not None:
                raise RuntimeError("Demo video writer failed.") from self.error
            try:
                self.frames.put(frame, timeout=0.1)
                return
            except queue.Full:
                continue

    def write(self, frame: np.ndarray) -> None:
        self._put(frame)

    def close(self) -> None:
        self._put(None)
        self.thread.join()
        if self.error is not None:
            raise RuntimeError("Demo video writer failed.") from self.error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input image/video file or directory.")
    parser.add_argument("--mode", choices=("auto", "image", "video"), default="auto")
    parser.add_argument("--piglet-weights", type=Path, default=DEFAULT_PIGLET_WEIGHTS, help="Piglet YOLO segmentation weights.")
    parser.add_argument("--sow-weights", type=Path, default=DEFAULT_SOW_WEIGHTS, help="Sow YOLO segmentation weights.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output directory.")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default="", help="CUDA device like 0, cpu, or empty for auto.")
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--retina-masks", action="store_true")
    parser.add_argument("--sow-label", default="sow")
    parser.add_argument("--piglet-label", default="piglet")
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--max-contact-dist", type=float, default=48.0)
    parser.add_argument("--contact-margin", type=float, default=12.0)
    parser.add_argument("--cluster-gap", type=float, default=360.0)
    parser.add_argument("--head-projection-quantile", type=float, default=0.9)
    parser.add_argument("--min-main-piglets", type=int, default=3)
    parser.add_argument("--arc-padding", type=float, default=70.0)
    parser.add_argument("--band-width", type=int, default=80)
    parser.add_argument(
        "--band-outer-width",
        type=float,
        default=75.0,
        help="Wide ROI reach in pixels on the piglet/outside side of the sow contour.",
    )
    parser.add_argument(
        "--band-inner-width",
        type=float,
        default=35.0,
        help="Wide ROI reach in pixels inside the sow mask; kept smaller to suppress thigh false positives.",
    )
    parser.add_argument("--centerline-thickness", type=int, default=5)
    parser.add_argument(
        "--contour-smooth-window",
        type=float,
        default=25.0,
        help="Arc-length window in pixels used to smooth the sow contour before drawing the ROI.",
    )
    parser.add_argument(
        "--turn-window",
        type=float,
        default=60.0,
        help="Arc-length span in pixels used to measure local contour turns.",
    )
    parser.add_argument(
        "--max-turn-angle",
        type=float,
        default=50.0,
        help="Maximum local turn in degrees; set <= 0 to disable the angle limit.",
    )
    parser.add_argument(
        "--min-turn-radius",
        type=float,
        default=35.0,
        help="Minimum local radius of curvature in pixels; set <= 0 to disable the curvature limit.",
    )
    parser.add_argument(
        "--max-axis-deviation",
        type=float,
        default=30.0,
        help="Maximum contour-tangent deviation from the sow long axis in degrees; set <= 0 to disable.",
    )
    parser.add_argument(
        "--max-contour-bridge",
        type=float,
        default=180.0,
        help="Maximum rejected contour gap in pixels that may be replaced by a straight supported bridge.",
    )
    parser.add_argument(
        "--max-endpoint-extension",
        type=float,
        default=140.0,
        help="Maximum straight extension in pixels used to recover selected contacts beyond a cleaned endpoint.",
    )
    parser.add_argument("--min-sow-area", type=int, default=1000)
    parser.add_argument("--min-piglet-area", type=int, default=80)
    parser.add_argument("--recursive", action="store_true", help="Search input directory recursively.")
    parser.add_argument("--sample-fps", type=float, default=1.0, help="Piglet mask refresh rate for videos.")
    parser.add_argument(
        "--sow-check-sec",
        "--sow-refresh-sec",
        dest="sow_check_sec",
        type=float,
        default=5.0,
        help="Seconds between sow-mask change checks; --sow-refresh-sec is kept as a compatibility alias.",
    )
    parser.add_argument(
        "--sow-mask-change-threshold",
        type=float,
        default=0.10,
        help="Candidate ROI refresh threshold for sow-mask symmetric-difference/union.",
    )
    parser.add_argument(
        "--sow-change-confirm-checks",
        type=int,
        default=2,
        help="Number of consecutive above-threshold sow checks required before an ROI refresh.",
    )
    parser.add_argument(
        "--sow-mask-change-immediate-threshold",
        type=float,
        default=0.30,
        help="Refresh immediately when sow-mask change reaches this larger threshold.",
    )
    parser.add_argument("--roi-lock-min-contacts", type=int, default=4)
    parser.add_argument("--roi-lock-coverage", type=float, default=0.80)
    parser.add_argument("--roi-unlock-coverage", type=float, default=0.65)
    parser.add_argument("--roi-lock-confirm-checks", type=int, default=2)
    parser.add_argument("--roi-unlock-confirm-checks", type=int, default=2)
    parser.add_argument("--roi-candidate-min-iou", type=float, default=0.90)
    parser.add_argument("--roi-candidate-coverage-gain", type=float, default=0.15)
    parser.add_argument("--roi-minor-refresh-cooldown-sec", type=float, default=30.0)
    parser.add_argument(
        "--roi-contact-recovery-current-max",
        type=float,
        default=0.20,
        help=(
            "Start ROI self-recovery when the accepted ROI covers no more than this fraction of contacts."
        ),
    )
    parser.add_argument(
        "--roi-contact-recovery-candidate-min",
        type=float,
        default=0.80,
        help="Minimum contact coverage required from an alternative ROI during self-recovery.",
    )
    parser.add_argument(
        "--roi-contact-recovery-confirm-checks",
        type=int,
        default=3,
        help="Consecutive sow checks required before switching to a stable alternative ROI.",
    )
    parser.add_argument(
        "--roi-contact-recovery-max-center-shift-ratio",
        type=float,
        default=0.25,
        help=(
            "Maximum sow-body-normalized longitudinal shift between consecutive recovery candidates."
        ),
    )
    parser.add_argument("--roi-low-contact-confirm-checks", type=int, default=3)
    parser.add_argument("--roi-low-contact-sow-threshold", type=float, default=0.15)
    parser.add_argument("--sow-gross-change-threshold", type=float, default=0.25)
    parser.add_argument("--sow-gross-centroid-shift-ratio", type=float, default=0.05)
    parser.add_argument("--sow-gross-bbox-shift-ratio", type=float, default=0.08)
    parser.add_argument("--sow-gross-angle-deg", type=float, default=12.0)
    parser.add_argument(
        "--initial-roi-confirm-sec",
        type=float,
        default=5.0,
        help="Keep the initial ROI provisional for this many seconds while collecting candidates.",
    )
    parser.add_argument(
        "--initial-roi-candidate-sec",
        type=float,
        default=1.0,
        help="Seconds between initial provisional ROI candidates.",
    )
    parser.add_argument(
        "--initial-roi-min-candidates",
        type=int,
        default=3,
        help="Minimum mutually consistent candidates required to confirm the initial ROI.",
    )
    parser.add_argument(
        "--initial-roi-min-iou",
        type=float,
        default=0.45,
        help="Minimum mask IoU for two initial ROI candidates to support the same consensus.",
    )
    parser.add_argument(
        "--initial-roi-min-consensus-ratio",
        type=float,
        default=0.60,
        help="Minimum fraction of collected initial candidates that must support the selected ROI.",
    )
    parser.add_argument(
        "--max-piglet-sow-overlap",
        type=float,
        default=0.6,
        help="Strict QC channel: reject a piglet candidate when more than this fraction lies inside the sow mask.",
    )
    parser.add_argument(
        "--hard-max-piglet-sow-overlap",
        type=float,
        default=0.85,
        help=(
            "Permissive behavior channel: only reject a candidate above this sow-overlap fraction; "
            "must be >= --max-piglet-sow-overlap."
        ),
    )
    parser.add_argument(
        "--roi-edge-margin",
        type=int,
        default=10,
        help="Pixels eroded from the nursing ROI to define its stable inner region.",
    )
    parser.add_argument(
        "--roi-edge-min-inner-area",
        type=int,
        default=50,
        help="A contacting piglet with less inner-ROI overlap is treated as an edge candidate.",
    )
    parser.add_argument(
        "--edge-confirm-window",
        type=int,
        default=None,
        help="Legacy fixed-frame edge-confirmation window; provide together with --edge-confirm-hits.",
    )
    parser.add_argument(
        "--edge-confirm-hits",
        type=int,
        default=None,
        help="Legacy fixed-frame required hits; provide together with --edge-confirm-window.",
    )
    parser.add_argument(
        "--edge-confirm-window-sec",
        type=float,
        default=1.5,
        help="Time span represented by the edge-confirmation window when legacy frame arguments are omitted.",
    )
    parser.add_argument(
        "--edge-confirm-hit-ratio",
        type=float,
        default=4.0 / 7.0,
        help="Required fraction of sampled frames inside the time-scaled edge-confirmation window.",
    )
    parser.add_argument(
        "--edge-track-max-distance",
        type=float,
        default=80.0,
        help="Maximum centroid distance in pixels when matching piglets across sampled frames.",
    )
    parser.add_argument(
        "--edge-track-max-area-ratio",
        type=float,
        default=2.5,
        help="Maximum larger/smaller mask-area ratio allowed for a temporal match.",
    )
    parser.add_argument(
        "--postprocess-bin-sec",
        type=float,
        default=1.0,
        help="Time-bin width in seconds for the processed interaction CSV and plot.",
    )
    parser.add_argument("--hampel-window-sec", type=float, default=7.0)
    parser.add_argument("--hampel-sigma", type=float, default=3.0)
    parser.add_argument("--hampel-relative-threshold", type=float, default=0.1)
    parser.add_argument("--hampel-total-floor", type=float, default=1500.0)
    parser.add_argument("--hampel-union-ratio-floor", type=float, default=0.05)
    parser.add_argument("--hampel-max-pulse-sec", type=float, default=2.0)
    parser.add_argument("--count-confirm-sec", type=float, default=3.0)
    parser.add_argument("--quantitative-median-sec", type=float, default=3.0)
    parser.add_argument("--display-sg-window-sec", type=float, default=7.0)
    parser.add_argument("--display-sg-polyorder", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8, help="Piglet inference batch size for videos.")
    parser.add_argument("--plot-dpi", type=int, default=160)
    parser.add_argument("--save-overlays", action="store_true", help="Save sampled video overlay frames for visual checking.")
    parser.add_argument("--overlay-limit", type=int, default=120, help="Maximum sampled overlay frames to save per video.")
    parser.add_argument("--save-demo-video", action="store_true", help="Save an MP4 demo with masks, ROI, and contact overlay.")
    parser.add_argument("--demo-fps", type=float, default=6.0, help="Playback FPS for exported demo videos.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-on-error", action="store_true", help="Abort on the first failed image.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def iter_images(input_dir: Path, recursive: bool) -> list[Path]:
    if input_dir.is_file():
        if input_dir.suffix.lower() not in IMAGE_EXTS:
            raise ValueError(f"Input is not a supported image: {input_dir}")
        return [input_dir]
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input must be a directory or image file: {input_dir}")

    pattern = "**/*" if recursive else "*"
    images = sorted(path for path in input_dir.glob(pattern) if path.is_file() and path.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise FileNotFoundError(f"No images found in {input_dir}.")
    return images


def iter_videos(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in VIDEO_EXTS:
            raise ValueError(f"Input is not a supported video: {input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise NotADirectoryError(f"Input must be a directory or video file: {input_path}")

    pattern = "**/*" if recursive else "*"
    videos = sorted(path for path in input_path.glob(pattern) if path.is_file() and path.suffix.lower() in VIDEO_EXTS)
    if not videos:
        raise FileNotFoundError(f"No videos found in {input_path}.")
    return videos


def resolve_mode(input_path: Path, mode: str, recursive: bool) -> str:
    if mode != "auto":
        return mode
    if input_path.is_file():
        suffix = input_path.suffix.lower()
        if suffix in VIDEO_EXTS:
            return "video"
        if suffix in IMAGE_EXTS:
            return "image"
    if input_path.is_dir():
        pattern = "**/*" if recursive else "*"
        if any(path.is_file() and path.suffix.lower() in VIDEO_EXTS for path in input_path.glob(pattern)):
            return "video"
        if any(path.is_file() and path.suffix.lower() in IMAGE_EXTS for path in input_path.glob(pattern)):
            return "image"
    raise FileNotFoundError(f"Could not find supported images or videos in {input_path}.")


def output_stem(image_path: Path, input_dir: Path, output_dir: Path, recursive: bool) -> Path:
    if input_dir.is_file():
        return output_dir / image_path.stem
    if recursive:
        rel_parent = image_path.parent.relative_to(input_dir)
        return output_dir / rel_parent / image_path.stem
    return output_dir / image_path.stem

def result_instances(
    result: Any,
    mask_threshold: float,
    label_override: str | None = None,
    start_index: int = 0,
) -> list[InstanceMask]:
    if result.masks is None or result.boxes is None:
        return []

    height, width = result.orig_shape
    masks = result.masks.data.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    confs = result.boxes.conf.detach().cpu().numpy()
    names = result.names or {}
    instances: list[InstanceMask] = []

    for index, raw_mask in enumerate(masks):
        mask = raw_mask.astype(np.float32)
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
        class_id = int(classes[index]) if index < len(classes) else 0
        label = label_override or (names.get(class_id, str(class_id)) if isinstance(names, dict) else str(class_id))
        confidence = float(confs[index]) if index < len(confs) else 1.0
        instances.append(
            InstanceMask(
                index=start_index + index,
                class_id=class_id,
                label=str(label),
                confidence=confidence,
                mask=mask >= mask_threshold,
            )
        )
    return instances


def largest_contour(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No contour found for sow mask.")
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    if len(contour) < 3:
        raise ValueError("Sow contour is too small.")
    return contour


def contour_arclength(contour: np.ndarray) -> tuple[np.ndarray, float]:
    next_points = np.roll(contour, -1, axis=0)
    edge_lengths = np.linalg.norm(next_points - contour, axis=1)
    cumulative = np.zeros(len(contour), dtype=np.float32)
    cumulative[1:] = np.cumsum(edge_lengths[:-1])
    return cumulative, float(edge_lengths.sum())


def nearest_contour_index(point_xy: tuple[float, float], contour: np.ndarray) -> int:
    point = np.asarray(point_xy, dtype=np.float32)
    return int(np.argmin(np.sum((contour - point) ** 2, axis=1)))


def build_contacts(
    sow_mask: np.ndarray,
    sow_contour: np.ndarray,
    contour_s: np.ndarray,
    piglets: list[InstanceMask],
    max_contact_dist: float,
    contact_margin: float,
    head_projection_quantile: float,
) -> list[Contact]:
    outside_sow = (~sow_mask).astype(np.uint8)
    dist_to_sow = cv2.distanceTransform(outside_sow, cv2.DIST_L2, 5)
    sow_ys, sow_xs = np.nonzero(sow_mask)
    sow_centroid = np.asarray([float(sow_xs.mean()), float(sow_ys.mean())], dtype=np.float32)
    contacts: list[Contact] = []

    for piglet in piglets:
        ys, xs = np.nonzero(piglet.mask)
        if len(xs) == 0:
            continue

        centroid = np.asarray([float(xs.mean()), float(ys.mean())], dtype=np.float32)
        head_direction = sow_centroid - centroid
        head_direction_norm = float(np.linalg.norm(head_direction))
        if head_direction_norm > 1e-6:
            head_direction = head_direction / head_direction_norm

        distances = dist_to_sow[ys, xs]
        min_distance = float(distances.min())
        if min_distance > max_contact_dist:
            continue

        near_threshold = min(max_contact_dist, min_distance + contact_margin)
        near = distances <= near_threshold
        if int(near.sum()) < 3:
            take = max(3, min(len(distances), int(math.ceil(len(distances) * 0.05))))
            near_indices = np.argpartition(distances, take - 1)[:take]
            near_xs = xs[near_indices]
            near_ys = ys[near_indices]
        else:
            near_xs = xs[near]
            near_ys = ys[near]

        if head_direction_norm > 1e-6 and len(xs) >= 3:
            all_points = np.column_stack([xs, ys]).astype(np.float32)
            projections = (all_points - centroid) @ head_direction
            quantile = min(0.99, max(0.5, float(head_projection_quantile)))
            threshold = float(np.quantile(projections, quantile))
            head_side = projections >= threshold
            if int(head_side.sum()) >= 3:
                head_xs = xs[head_side]
                head_ys = ys[head_side]
                head_distances = dist_to_sow[head_ys, head_xs]
                head_min_distance = float(head_distances.min())
                head_near = head_distances <= min(max_contact_dist, head_min_distance + contact_margin)
                if int(head_near.sum()) >= 3:
                    near_xs = head_xs[head_near]
                    near_ys = head_ys[head_near]

        center = (float(np.median(near_xs)), float(np.median(near_ys)))
        contour_index = nearest_contour_index(center, sow_contour)
        contour_point = sow_contour[contour_index]
        contacts.append(
            Contact(
                piglet_index=piglet.index,
                confidence=piglet.confidence,
                min_distance_px=min_distance,
                contact_center=center,
                contour_index=contour_index,
                contour_s=float(contour_s[contour_index]),
                contour_point=(int(round(contour_point[0])), int(round(contour_point[1]))),
            )
        )
    return contacts


def cluster_contacts(contacts: list[Contact], total_s: float, max_gap: float) -> list[list[Contact]]:
    if len(contacts) <= 1:
        return [contacts] if contacts else []

    ordered = sorted(contacts, key=lambda contact: contact.contour_s)
    clusters: list[list[Contact]] = [[ordered[0]]]
    for prev, curr in zip(ordered, ordered[1:]):
        if curr.contour_s - prev.contour_s <= max_gap:
            clusters[-1].append(curr)
        else:
            clusters.append([curr])

    wrap_gap = ordered[0].contour_s + total_s - ordered[-1].contour_s
    if len(clusters) > 1 and wrap_gap <= max_gap:
        clusters[0] = clusters[-1] + clusters[0]
        clusters.pop()
    return clusters


def minimal_arc(values: list[float], total_s: float) -> tuple[float, float, float]:
    if not values:
        raise ValueError("Cannot build arc from empty contacts.")
    if len(values) == 1:
        return values[0], values[0], 0.0

    ordered = sorted(value % total_s for value in values)
    gaps = [ordered[i + 1] - ordered[i] for i in range(len(ordered) - 1)]
    gaps.append(ordered[0] + total_s - ordered[-1])
    gap_index = int(np.argmax(gaps))
    start = ordered[(gap_index + 1) % len(ordered)]
    end = ordered[gap_index]
    return float(start), float(end), float((end - start) % total_s)


def interval_points(contour: np.ndarray, contour_s: np.ndarray, start_s: float, end_s: float) -> np.ndarray:
    start_idx = int(np.searchsorted(contour_s, start_s, side="left")) % len(contour)
    end_idx = int(np.searchsorted(contour_s, end_s, side="right")) % len(contour)
    if start_s <= end_s:
        points = contour[start_idx:max(start_idx + 2, end_idx)]
    else:
        points = np.concatenate([contour[start_idx:], contour[: max(1, end_idx)]], axis=0)
    if len(points) < 2:
        points = contour[[start_idx, (start_idx + 1) % len(contour)]]
    return np.round(points).astype(np.int32)


def open_polyline_arclength(points: np.ndarray) -> tuple[np.ndarray, float]:
    if len(points) < 2:
        return np.zeros(len(points), dtype=np.float32), 0.0
    edge_lengths = np.linalg.norm(np.diff(points.astype(np.float32), axis=0), axis=1)
    cumulative = np.zeros(len(points), dtype=np.float32)
    cumulative[1:] = np.cumsum(edge_lengths)
    return cumulative, float(cumulative[-1])


def resample_open_polyline(points: np.ndarray, step_px: float) -> np.ndarray:
    points = points.astype(np.float32)
    if len(points) < 2:
        return points

    keep = np.concatenate([[True], np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-6])
    points = points[keep]
    cumulative, total = open_polyline_arclength(points)
    if len(points) < 2 or total <= step_px:
        return points

    sample_s = np.arange(0.0, total, step_px, dtype=np.float32)
    if len(sample_s) == 0 or total - float(sample_s[-1]) > 1e-3:
        sample_s = np.append(sample_s, total)
    xs = np.interp(sample_s, cumulative, points[:, 0])
    ys = np.interp(sample_s, cumulative, points[:, 1])
    return np.column_stack([xs, ys]).astype(np.float32)


def smooth_open_polyline(points: np.ndarray, window_samples: int) -> np.ndarray:
    if len(points) < 3 or window_samples <= 1:
        return points.astype(np.float32)

    window_samples = min(int(window_samples), len(points) if len(points) % 2 == 1 else len(points) - 1)
    window_samples = max(1, window_samples)
    if window_samples % 2 == 0:
        window_samples -= 1
    if window_samples <= 1:
        return points.astype(np.float32)

    radius = window_samples // 2
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    sigma = max(1.0, window_samples / 5.0)
    weights = np.exp(-0.5 * (offsets / sigma) ** 2)
    weights /= weights.sum()
    padded = np.pad(points.astype(np.float32), ((radius, radius), (0, 0)), mode="edge")
    smoothed = np.empty_like(points, dtype=np.float32)
    smoothed[:, 0] = np.convolve(padded[:, 0], weights, mode="valid")
    smoothed[:, 1] = np.convolve(padded[:, 1], weights, mode="valid")
    return smoothed


def contour_turn_metrics(points: np.ndarray, span_px: float) -> tuple[np.ndarray, np.ndarray]:
    angles = np.zeros(len(points), dtype=np.float32)
    radii = np.full(len(points), np.inf, dtype=np.float32)
    cumulative, total = open_polyline_arclength(points)
    if len(points) < 3 or total <= 0:
        return angles, radii

    half_span = max(1.0, float(span_px) / 2.0)
    for index, center_s in enumerate(cumulative):
        if center_s < half_span or total - center_s < half_span:
            continue
        left = int(np.searchsorted(cumulative, center_s - half_span, side="left"))
        right = int(np.searchsorted(cumulative, center_s + half_span, side="left"))
        right = min(right, len(points) - 1)
        if left >= index or right <= index:
            continue

        incoming = points[index] - points[left]
        outgoing = points[right] - points[index]
        incoming_len = float(np.linalg.norm(incoming))
        outgoing_len = float(np.linalg.norm(outgoing))
        chord_len = float(np.linalg.norm(points[right] - points[left]))
        if min(incoming_len, outgoing_len, chord_len) <= 1e-6:
            continue

        cosine = float(np.dot(incoming, outgoing) / (incoming_len * outgoing_len))
        angles[index] = math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))
        twice_area = abs(float(incoming[0] * outgoing[1] - incoming[1] * outgoing[0]))
        if twice_area > 1e-6:
            radii[index] = incoming_len * outgoing_len * chord_len / (2.0 * twice_area)
    return angles, radii


def mask_long_axis(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if len(xs) < 3:
        raise ValueError("Sow mask has too few pixels to estimate its long axis.")
    points = np.column_stack([xs, ys]).astype(np.float32)
    covariance = np.cov(points - points.mean(axis=0), rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, int(np.argmax(eigenvalues))].astype(np.float32)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-6:
        raise ValueError("Could not estimate the sow long axis.")
    return axis / norm


def contour_axis_deviation(points: np.ndarray, axis: np.ndarray, span_px: float) -> np.ndarray:
    deviations = np.zeros(len(points), dtype=np.float32)
    cumulative, total = open_polyline_arclength(points)
    if len(points) < 2 or total <= 0:
        return deviations

    half_span = max(1.0, float(span_px) / 2.0)
    for index, center_s in enumerate(cumulative):
        left_s = max(0.0, float(center_s - half_span))
        right_s = min(total, float(center_s + half_span))
        left = int(np.searchsorted(cumulative, left_s, side="left"))
        right = min(int(np.searchsorted(cumulative, right_s, side="left")), len(points) - 1)
        if right <= left:
            continue
        tangent = points[right] - points[left]
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm <= 1e-6:
            continue
        cosine = abs(float(np.dot(tangent / tangent_norm, axis)))
        deviations[index] = math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))
    return deviations


def contiguous_true_runs(values: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(values.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def contact_support_indices(
    points: np.ndarray,
    contact_points: np.ndarray,
    support_distance: float,
) -> set[int]:
    if len(points) == 0 or len(contact_points) == 0:
        return set()
    distances_sq = np.sum((contact_points[:, None, :] - points[None, :, :]) ** 2, axis=2)
    return set(np.flatnonzero(np.min(distances_sq, axis=1) <= support_distance**2).tolist())


def vector_axis_deviation(vector: np.ndarray, axis: np.ndarray) -> float:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-6:
        return 0.0
    cosine = abs(float(np.dot(vector / norm, axis)))
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def bridge_supported_contour_runs(
    points: np.ndarray,
    runs: list[tuple[int, int]],
    contact_points: np.ndarray,
    support_distance: float,
    axis: np.ndarray,
    max_gap_px: float,
    max_axis_deviation: float,
    sample_step: float,
) -> tuple[list[np.ndarray], int, float]:
    if not runs:
        return [], 0, 0.0

    groups: list[np.ndarray] = []
    current_start, current_end = runs[0]
    current = points[current_start:current_end]
    current_support = contact_support_indices(current, contact_points, support_distance)
    bridge_count = 0
    bridged_gap_px = 0.0

    for next_start, next_end in runs[1:]:
        next_points = points[next_start:next_end]
        next_support = contact_support_indices(next_points, contact_points, support_distance)
        gap_points = points[current_end - 1 : next_start + 1]
        _, gap_length = open_polyline_arclength(gap_points)
        chord = next_points[0] - current[-1]
        chord_length = float(np.linalg.norm(chord))
        chord_axis_deviation = vector_axis_deviation(chord, axis)
        combined_support = current_support | next_support
        can_bridge = (
            max_gap_px > 0
            and gap_length <= max_gap_px
            and chord_length <= max_gap_px
            and chord_axis_deviation <= max_axis_deviation
            and bool(current_support)
            and bool(next_support)
            and len(combined_support) >= 2
        )

        if can_bridge:
            bridge_samples = max(2, int(math.ceil(chord_length / sample_step)) + 1)
            bridge = np.linspace(current[-1], next_points[0], bridge_samples, dtype=np.float32)
            current = np.concatenate([current, bridge[1:-1], next_points], axis=0)
            current_support = combined_support
            current_end = next_end
            bridge_count += 1
            bridged_gap_px += gap_length
        else:
            groups.append(current)
            current = next_points
            current_support = next_support
            current_end = next_end

    groups.append(current)
    return groups, bridge_count, bridged_gap_px


def extend_polyline_to_supported_contacts(
    points: np.ndarray,
    contact_points: np.ndarray,
    support_distance: float,
    axis: np.ndarray,
    max_extension_px: float,
    max_axis_deviation: float,
    sample_step: float,
) -> tuple[np.ndarray, int, float]:
    extended = points.astype(np.float32)
    extension_count = 0
    extension_length_px = 0.0
    if len(extended) < 2 or len(contact_points) == 0 or max_extension_px <= 0:
        return extended, extension_count, extension_length_px

    for at_start in (True, False):
        supported = contact_support_indices(extended, contact_points, support_distance)
        endpoint = extended[0] if at_start else extended[-1]
        interior_index = min(3, len(extended) - 1)
        interior = extended[interior_index] - endpoint if at_start else extended[-1 - interior_index] - endpoint
        candidates: list[tuple[float, np.ndarray]] = []

        for contact_index, contact_point in enumerate(contact_points):
            if contact_index in supported:
                continue
            distances_sq = np.sum((extended - contact_point) ** 2, axis=1)
            nearest_index = int(np.argmin(distances_sq))
            if at_start and nearest_index > 2:
                continue
            if not at_start and nearest_index < len(extended) - 3:
                continue

            outward = contact_point - endpoint
            distance = float(np.linalg.norm(outward))
            if distance <= 1e-6 or distance > max_extension_px:
                continue
            if float(np.dot(outward, interior)) >= 0:
                continue
            if vector_axis_deviation(outward, axis) > max_axis_deviation:
                continue
            candidates.append((distance, contact_point))

        if not candidates:
            continue

        distance, target = max(candidates, key=lambda item: item[0])
        extension_samples = max(2, int(math.ceil(distance / sample_step)) + 1)
        if at_start:
            extension = np.linspace(target, extended[0], extension_samples, dtype=np.float32)
            extended = np.concatenate([extension[:-1], extended], axis=0)
        else:
            extension = np.linspace(extended[-1], target, extension_samples, dtype=np.float32)
            extended = np.concatenate([extended, extension[1:]], axis=0)
        extension_count += 1
        extension_length_px += distance

    return extended, extension_count, extension_length_px


def clean_contour_arc(
    arc_points: np.ndarray,
    selected_contacts: list[Contact],
    sow_mask: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, float]]:
    if args.contour_smooth_window < 0:
        raise ValueError("--contour-smooth-window must be >= 0.")
    if args.turn_window <= 0:
        raise ValueError("--turn-window must be > 0.")
    if args.max_contour_bridge < 0:
        raise ValueError("--max-contour-bridge must be >= 0.")
    if args.max_endpoint_extension < 0:
        raise ValueError("--max-endpoint-extension must be >= 0.")

    sample_step = min(8.0, max(3.0, args.turn_window / 12.0))
    resampled = resample_open_polyline(arc_points, sample_step)
    smooth_samples = max(1, int(round(args.contour_smooth_window / sample_step)))
    if smooth_samples % 2 == 0:
        smooth_samples += 1
    smoothed = smooth_open_polyline(resampled, smooth_samples)
    angles, radii = contour_turn_metrics(smoothed, args.turn_window)
    long_axis = mask_long_axis(sow_mask)
    axis_deviations = contour_axis_deviation(smoothed, long_axis, args.turn_window)

    invalid = np.zeros(len(smoothed), dtype=bool)
    if args.max_turn_angle > 0:
        invalid |= angles > args.max_turn_angle
    if args.min_turn_radius > 0:
        invalid |= radii < args.min_turn_radius
    if args.max_axis_deviation > 0:
        invalid |= axis_deviations > args.max_axis_deviation
    violation_count = int(invalid.sum())

    if violation_count:
        trim_samples = max(1, int(round(max(sample_step, args.contour_smooth_window / 2.0) / sample_step)))
        kernel = np.ones(trim_samples * 2 + 1, dtype=np.uint8)
        invalid = np.convolve(invalid.astype(np.uint8), kernel, mode="same") > 0

    min_length = max(float(args.turn_window), float(args.band_width))
    candidates: list[tuple[np.ndarray, int, float]] = []
    short_candidates: list[tuple[np.ndarray, int, float]] = []
    contact_points = np.asarray([contact.contour_point for contact in selected_contacts], dtype=np.float32)
    support_distance = max(float(args.band_width), float(args.max_contact_dist))
    bridge_axis_deviation = args.max_axis_deviation if args.max_axis_deviation > 0 else 30.0
    groups, bridge_count, bridged_gap_px = bridge_supported_contour_runs(
        smoothed,
        contiguous_true_runs(~invalid),
        contact_points,
        support_distance,
        long_axis,
        args.max_contour_bridge,
        bridge_axis_deviation,
        sample_step,
    )
    for candidate in groups:
        _, length = open_polyline_arclength(candidate)
        if len(candidate) < 2:
            continue
        support = len(contact_support_indices(candidate, contact_points, support_distance))
        item = (candidate, support, length)
        short_candidates.append(item)
        if length >= min_length:
            candidates.append(item)

    if candidates:
        cleaned, support, clean_length = max(candidates, key=lambda item: (item[1], item[2]))
    elif short_candidates:
        cleaned, support, clean_length = max(short_candidates, key=lambda item: (item[1], item[2]))
    else:
        cleaned = smoothed
        support = len(selected_contacts)
        _, clean_length = open_polyline_arclength(cleaned)

    endpoint_axis_deviation = min(45.0, bridge_axis_deviation + 15.0)
    cleaned, endpoint_extension_count, endpoint_extension_length = extend_polyline_to_supported_contacts(
        cleaned,
        contact_points,
        support_distance,
        long_axis,
        args.max_endpoint_extension,
        endpoint_axis_deviation,
        sample_step,
    )
    support = len(contact_support_indices(cleaned, contact_points, support_distance))
    _, clean_length = open_polyline_arclength(cleaned)

    finite_radii = radii[np.isfinite(radii)]
    clean_axis_deviations = contour_axis_deviation(cleaned, long_axis, args.turn_window)
    metrics = {
        "raw_arc_length_px": open_polyline_arclength(arc_points)[1],
        "clean_arc_length_px": float(clean_length),
        "turn_violation_points": float(violation_count),
        "max_observed_turn_deg": float(angles.max()) if len(angles) else 0.0,
        "min_observed_turn_radius_px": float(finite_radii.min()) if len(finite_radii) else 0.0,
        "max_axis_deviation_deg": float(axis_deviations.max()) if len(axis_deviations) else 0.0,
        "clean_max_axis_deviation_deg": (
            float(clean_axis_deviations.max()) if len(clean_axis_deviations) else 0.0
        ),
        "bridged_gap_count": float(bridge_count),
        "bridged_gap_length_px": float(bridged_gap_px),
        "endpoint_extension_count": float(endpoint_extension_count),
        "endpoint_extension_length_px": float(endpoint_extension_length),
        "clean_contact_support": float(support),
    }
    return np.round(cleaned).astype(np.int32), metrics


def draw_polyline(shape: tuple[int, int], points: np.ndarray, thickness: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.polylines(mask, [points.reshape(-1, 1, 2)], False, 255, max(1, int(thickness)))
    return mask


def close_band(mask: np.ndarray, width: float) -> np.ndarray:
    close_radius = max(3, int(round(float(width) / 8.0)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_radius * 2 + 1, close_radius * 2 + 1))
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)


def build_core_band(image_shape: tuple[int, int], arc_points: np.ndarray, width: int) -> np.ndarray:
    return close_band(draw_polyline(image_shape, arc_points, width), width)


def build_asymmetric_band(
    image_shape: tuple[int, int],
    arc_points: np.ndarray,
    sow_mask: np.ndarray,
    outer_width: float,
    inner_width: float,
    centerline_thickness: int = 5,
) -> np.ndarray:
    """Build a contour band that reaches farther outside the sow than inside it."""
    centerline = draw_polyline(image_shape, arc_points, centerline_thickness)
    distance_input = np.where(centerline > 0, 0, 1).astype(np.uint8)
    distance = cv2.distanceTransform(distance_input, cv2.DIST_L2, 5)
    sow = mask_for_shape(sow_mask, image_shape)
    outside = np.logical_and(~sow, distance <= float(outer_width))
    inside = np.logical_and(sow, distance <= float(inner_width))
    band = np.logical_or(np.logical_or(outside, inside), centerline > 0).astype(np.uint8) * 255
    return close_band(band, max(float(outer_width), float(inner_width)))


def select_sow_instance(instances: list[InstanceMask], args: argparse.Namespace) -> InstanceMask:
    sow_candidates = [
        instance
        for instance in instances
        if instance.label.lower() == args.sow_label.lower() and instance.area >= args.min_sow_area
    ]
    if not sow_candidates:
        raise ValueError(f"No usable sow mask found with label={args.sow_label!r}.")
    return max(sow_candidates, key=lambda instance: instance.area)


def select_instances(instances: list[InstanceMask], args: argparse.Namespace) -> tuple[InstanceMask, list[InstanceMask]]:
    sow = select_sow_instance(instances, args)
    piglets = [
        instance
        for instance in instances
        if instance.label.lower() == args.piglet_label.lower() and instance.area >= args.min_piglet_area
    ]
    if not piglets:
        raise ValueError(f"No usable piglet masks found with label={args.piglet_label!r}.")
    return sow, piglets


def extract_band(
    instances: list[InstanceMask],
    image_shape: tuple[int, int],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, InstanceMask, list[InstanceMask], list[Contact], list[Contact], dict[str, float]]:
    sow, piglets = select_instances(instances, args)
    contour = largest_contour(sow.mask)
    contour_s, total_s = contour_arclength(contour)
    contacts = build_contacts(
        sow.mask,
        contour,
        contour_s,
        piglets,
        max_contact_dist=args.max_contact_dist,
        contact_margin=args.contact_margin,
        head_projection_quantile=args.head_projection_quantile,
    )
    if not contacts:
        raise ValueError("No piglet contact points passed the distance threshold.")

    clusters = cluster_contacts(contacts, total_s, args.cluster_gap)
    selected = max(clusters, key=lambda cluster: (len(cluster), -minimal_arc([c.contour_s for c in cluster], total_s)[2]))
    if len(selected) < args.min_main_piglets and len(contacts) >= args.min_main_piglets:
        selected = contacts

    arc_start, arc_end, _ = minimal_arc([contact.contour_s for contact in selected], total_s)
    start_s = (arc_start - args.arc_padding) % total_s
    end_s = (arc_end + args.arc_padding) % total_s
    arc_points = interval_points(contour, contour_s, start_s, end_s)
    arc_points, contour_metrics = clean_contour_arc(arc_points, selected, sow.mask, args)

    centerline = draw_polyline(image_shape, arc_points, args.centerline_thickness)
    band = build_asymmetric_band(
        image_shape,
        arc_points,
        sow.mask,
        outer_width=args.band_outer_width,
        inner_width=args.band_inner_width,
        centerline_thickness=args.centerline_thickness,
    )

    interval = {
        "start_s": float(start_s),
        "end_s": float(end_s),
        "total_s": float(total_s),
        "length_s": float((end_s - start_s) % total_s),
        **contour_metrics,
    }
    return band, centerline, arc_points, sow, piglets, contacts, selected, interval


def blend_mask(image: np.ndarray, mask: np.ndarray, color_bgr: tuple[int, int, int], alpha: float) -> np.ndarray:
    pixels = mask.astype(bool)
    if not np.any(pixels):
        return image
    selected = image[pixels]
    color = np.empty_like(selected)
    color[:] = color_bgr
    image[pixels] = cv2.addWeighted(selected, 1.0 - alpha, color, alpha, 0)
    return image


def union_instance_masks(instances: list[InstanceMask], shape: tuple[int, int]) -> np.ndarray:
    union = np.zeros(shape, dtype=bool)
    for instance in instances:
        mask = instance.mask
        if mask.shape != shape:
            mask = cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        union |= mask
    return union


def mask_for_shape(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask.astype(bool, copy=False)
    return cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)


def sow_pose_change_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    reference_bool = reference.astype(bool, copy=False)
    candidate_bool = mask_for_shape(candidate, reference_bool.shape)

    def geometry(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        ys, xs = np.nonzero(mask)
        if len(xs) < 3:
            raise ValueError("Sow mask has too few pixels for pose comparison.")
        centroid = np.asarray([float(xs.mean()), float(ys.mean())], dtype=np.float32)
        bbox = np.asarray([float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())], dtype=np.float32)
        body_length = max(1.0, float(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)))
        return centroid, bbox, body_length, mask_long_axis(mask)

    ref_centroid, ref_bbox, ref_length, ref_axis = geometry(reference_bool)
    candidate_centroid, candidate_bbox, candidate_length, candidate_axis = geometry(candidate_bool)
    body_length = max(1.0, (ref_length + candidate_length) / 2.0)
    axis_dot = abs(float(np.dot(ref_axis, candidate_axis)))
    angle = math.degrees(math.acos(float(np.clip(axis_dot, -1.0, 1.0))))
    return {
        "centroid_shift_ratio": float(np.linalg.norm(candidate_centroid - ref_centroid) / body_length),
        "bbox_shift_ratio": float(np.max(np.abs(candidate_bbox - ref_bbox)) / body_length),
        "long_axis_angle_deg": float(angle),
    }


def contact_point_coverage(contacts: list[Contact] | tuple[Contact, ...], band: np.ndarray | None) -> float:
    if not contacts or band is None:
        return 0.0
    band_bool = band.astype(bool, copy=False)
    covered = 0
    for contact in contacts:
        x = int(round(contact.contact_center[0]))
        y = int(round(contact.contact_center[1]))
        if 0 <= y < band_bool.shape[0] and 0 <= x < band_bool.shape[1] and band_bool[y, x]:
            covered += 1
    return float(covered / len(contacts))


def roi_contact_recovery_signature(
    sow_mask: np.ndarray,
    contacts: list[Contact] | tuple[Contact, ...],
) -> RoiContactRecoverySignature | None:
    """Describe a contact cluster by sow side and longitudinal position.

    The sow long-axis sign is canonicalized so the side remains comparable when
    independently inferred sow masks make PCA return the opposite eigenvector.
    """
    if not contacts:
        return None
    ys, xs = np.nonzero(sow_mask)
    if len(xs) < 3:
        return None
    axis = mask_long_axis(sow_mask).astype(np.float32)
    if axis[0] < 0 or (abs(float(axis[0])) <= 1e-6 and axis[1] < 0):
        axis = -axis
    normal = np.asarray([-axis[1], axis[0]], dtype=np.float32)
    centroid = np.asarray([float(xs.mean()), float(ys.mean())], dtype=np.float32)
    points = np.asarray([contact.contact_center for contact in contacts], dtype=np.float32)
    offsets = points - centroid
    lateral_center = float(np.median(offsets @ normal))
    if abs(lateral_center) <= 1e-6:
        return None
    body_length = max(1.0, float(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)))
    longitudinal_center_ratio = float(np.median(offsets @ axis) / body_length)
    return RoiContactRecoverySignature(
        side=1 if lateral_center > 0 else -1,
        longitudinal_center_ratio=longitudinal_center_ratio,
    )


def update_roi_contact_recovery(
    *,
    previous_streak: int,
    previous_signature: RoiContactRecoverySignature | None,
    candidate_signature: RoiContactRecoverySignature | None,
    contact_count: int,
    current_coverage: float,
    candidate_coverage: float,
    min_contacts: int,
    current_max: float,
    candidate_min: float,
    max_center_shift_ratio: float,
    cooldown_active: bool,
) -> tuple[int, RoiContactRecoverySignature | None, bool, bool]:
    """Advance the contact-loss recovery streak.

    Returns ``(streak, signature, candidate_consistent, evidence_present)``.
    Candidate consistency deliberately uses sow-relative side and longitudinal
    position instead of pixel-mask IoU, which is too sensitive to contour jitter.
    """
    evidence_present = bool(
        not cooldown_active
        and candidate_signature is not None
        and contact_count >= min_contacts
        and current_coverage <= current_max
        and candidate_coverage >= candidate_min
    )
    if not evidence_present:
        return 0, None, False, False
    if previous_signature is None or previous_streak <= 0:
        return 1, candidate_signature, False, True

    candidate_consistent = bool(
        previous_signature.side == candidate_signature.side
        and abs(
            previous_signature.longitudinal_center_ratio
            - candidate_signature.longitudinal_center_ratio
        )
        <= max_center_shift_ratio
    )
    if not candidate_consistent:
        return 1, candidate_signature, False, True
    return previous_streak + 1, candidate_signature, True, True


def is_gross_sow_pose_change(
    sow_change_ratio: float,
    pose_metrics: dict[str, float],
    args: argparse.Namespace,
) -> bool:
    return bool(
        sow_change_ratio >= args.sow_mask_change_immediate_threshold
        or sow_change_ratio >= args.sow_gross_change_threshold
        or pose_metrics["centroid_shift_ratio"] >= args.sow_gross_centroid_shift_ratio
        or pose_metrics["bbox_shift_ratio"] >= args.sow_gross_bbox_shift_ratio
        or pose_metrics["long_axis_angle_deg"] >= args.sow_gross_angle_deg
    )


def build_roi_refresh_candidate(
    sow: InstanceMask,
    piglets: list[InstanceMask],
    current_band: np.ndarray | None,
    image_shape: tuple[int, int],
    timestamp_sec: float,
    args: argparse.Namespace,
) -> RoiRefreshCandidate:
    new_band, _, arc_points, new_sow, _, _, selected, _ = extract_band(
        [sow, *piglets],
        image_shape,
        args,
    )
    selected_tuple = tuple(selected)
    core_band = build_core_band(image_shape, arc_points, args.band_width)
    return RoiRefreshCandidate(
        timestamp_sec=timestamp_sec,
        band=new_band,
        core_band=core_band,
        sow=new_sow,
        selected_contacts=selected_tuple,
        current_contact_coverage=contact_point_coverage(selected_tuple, current_band),
        candidate_contact_coverage=contact_point_coverage(selected_tuple, new_band),
    )


def sow_mask_change_ratio(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Return spatial mask change as symmetric difference divided by union (1 - IoU)."""
    reference_bool = reference.astype(bool, copy=False)
    candidate_bool = mask_for_shape(candidate, reference_bool.shape)
    union = np.logical_or(reference_bool, candidate_bool)
    union_area = int(union.sum())
    if union_area == 0:
        return 0.0
    changed = int(np.logical_xor(reference_bool, candidate_bool).sum())
    return float(changed / union_area)


def binary_mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    first_bool = first.astype(bool, copy=False)
    second_bool = mask_for_shape(second, first_bool.shape)
    union = np.logical_or(first_bool, second_bool)
    union_area = int(union.sum())
    if union_area == 0:
        return 1.0
    return float(np.logical_and(first_bool, second_bool).sum() / union_area)


def select_initial_roi_consensus(
    candidates: list[InitialRoiCandidate],
    min_candidates: int,
    min_iou: float,
    min_consensus_ratio: float,
) -> tuple[InitialRoiCandidate | None, int, float]:
    """Return the medoid-like initial ROI candidate with sufficient temporal support."""
    if not candidates:
        return None, 0, 0.0

    required = max(min_candidates, int(math.ceil(len(candidates) * min_consensus_ratio)))
    best_candidate: InitialRoiCandidate | None = None
    best_support = 0
    best_mean_iou = 0.0
    best_key: tuple[int, float, int, int] | None = None
    for candidate in candidates:
        supporting_ious: list[float] = []
        for other in candidates:
            iou = binary_mask_iou(candidate.band, other.band)
            if iou >= min_iou:
                supporting_ious.append(iou)
        support = len(supporting_ious)
        mean_iou = float(np.mean(supporting_ious)) if supporting_ious else 0.0
        key = (support, mean_iou, candidate.selected_contact_count, int(candidate.band.astype(bool).sum()))
        if best_key is None or key > best_key:
            best_key = key
            best_candidate = candidate
            best_support = support
            best_mean_iou = mean_iou

    if best_support < required:
        return None, best_support, best_mean_iou
    return best_candidate, best_support, best_mean_iou


def filter_piglets_by_sow_overlap(
    piglets: list[InstanceMask],
    sow: InstanceMask | None,
    max_overlap_ratio: float,
) -> tuple[list[InstanceMask], int]:
    """Reject candidates whose masks lie mostly inside the current sow mask."""
    kept, rejected, _ = partition_piglets_by_sow_overlap(piglets, sow, max_overlap_ratio)
    return kept, len(rejected)


def partition_piglets_by_sow_overlap(
    piglets: list[InstanceMask],
    sow: InstanceMask | None,
    max_overlap_ratio: float,
) -> tuple[list[InstanceMask], list[InstanceMask], list[float]]:
    """Partition piglets while retaining continuous overlap evidence for model/QC output."""
    if sow is None:
        return list(piglets), [], [0.0] * len(piglets)

    kept: list[InstanceMask] = []
    rejected: list[InstanceMask] = []
    overlap_ratios: list[float] = []
    for piglet in piglets:
        area = piglet.area
        if area <= 0:
            overlap_ratios.append(0.0)
            kept.append(piglet)
            continue
        sow_mask = mask_for_shape(sow.mask, piglet.mask.shape)
        overlap_ratio = float(np.logical_and(piglet.mask, sow_mask).sum() / area)
        overlap_ratios.append(overlap_ratio)
        if overlap_ratio > max_overlap_ratio:
            rejected.append(piglet)
        else:
            kept.append(piglet)
    return kept, rejected, overlap_ratios


def piglet_geometry(instance: InstanceMask) -> PigletGeometry | None:
    ys, xs = np.nonzero(instance.mask)
    if len(xs) == 0:
        return None
    return PigletGeometry(
        centroid=(float(xs.mean()), float(ys.mean())),
        bbox=(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
        area=len(xs),
    )


def bbox_iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    if right < left or bottom < top:
        return 0.0
    intersection = float((right - left + 1) * (bottom - top + 1))
    first_area = float((first[2] - first[0] + 1) * (first[3] - first[1] + 1))
    second_area = float((second[2] - second[0] + 1) * (second[3] - second[1] + 1))
    return intersection / (first_area + second_area - intersection)


def match_piglet_pairs_one_to_one(
    current: list[InstanceMask],
    previous: list[InstanceMask],
    max_centroid_distance: float,
    max_area_ratio: float,
) -> list[tuple[int, int]]:
    """Match current instances to one previous frame without relying on per-frame detector indices."""
    current_geometry = [piglet_geometry(instance) for instance in current]
    previous_geometry = [piglet_geometry(instance) for instance in previous]
    pairs: list[tuple[float, float, int, int]] = []
    for current_index, current_item in enumerate(current_geometry):
        if current_item is None:
            continue
        for previous_index, previous_item in enumerate(previous_geometry):
            if previous_item is None:
                continue
            smaller_area = min(current_item.area, previous_item.area)
            if smaller_area <= 0 or max(current_item.area, previous_item.area) / smaller_area > max_area_ratio:
                continue
            distance = math.hypot(
                current_item.centroid[0] - previous_item.centroid[0],
                current_item.centroid[1] - previous_item.centroid[1],
            )
            if distance > max_centroid_distance:
                continue
            pairs.append((-bbox_iou(current_item.bbox, previous_item.bbox), distance, current_index, previous_index))

    matches: list[tuple[int, int]] = []
    matched_current: set[int] = set()
    matched_previous: set[int] = set()
    for _, _, current_index, previous_index in sorted(pairs):
        if current_index in matched_current or previous_index in matched_previous:
            continue
        matched_current.add(current_index)
        matched_previous.add(previous_index)
        matches.append((current_index, previous_index))
    return matches


def match_piglets_one_to_one(
    current: list[InstanceMask],
    previous: list[InstanceMask],
    max_centroid_distance: float,
    max_area_ratio: float,
) -> set[int]:
    return {
        current_index
        for current_index, _ in match_piglet_pairs_one_to_one(
            current,
            previous,
            max_centroid_distance=max_centroid_distance,
            max_area_ratio=max_area_ratio,
        )
    }


class EdgePigletConfirmer:
    """Confirm shallow ROI-edge detections against the preceding sampled frames."""

    def __init__(
        self,
        window: int,
        required_hits: int,
        edge_margin: int,
        min_inner_area: int,
        max_centroid_distance: float,
        max_area_ratio: float,
    ) -> None:
        self.window = window
        self.required_hits = required_hits
        self.edge_margin = edge_margin
        self.min_inner_area = min_inner_area
        self.max_centroid_distance = max_centroid_distance
        self.max_area_ratio = max_area_ratio
        self.history: deque[list[InstanceMask]] = deque(maxlen=max(0, window - 1))

    def reset(self) -> None:
        self.history.clear()

    def update(
        self,
        piglets: list[InstanceMask],
        band: np.ndarray | None,
    ) -> tuple[list[InstanceMask], EdgeConfirmationStats]:
        previous_matches = [
            match_piglets_one_to_one(
                piglets,
                previous,
                max_centroid_distance=self.max_centroid_distance,
                max_area_ratio=self.max_area_ratio,
            )
            for previous in self.history
        ]

        inner_band: np.ndarray | None = None
        band_bool: np.ndarray | None = None
        if band is not None:
            band_bool = band.astype(bool)
            diameter = self.edge_margin * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
            inner_band = cv2.erode(band_bool.astype(np.uint8), kernel).astype(bool)

        accepted: list[InstanceMask] = []
        edge_candidates = 0
        pending_edge_candidates = 0
        confirmed_edge_candidates = 0
        edge_candidate_area_px = 0
        pending_edge_area_px = 0
        edge_hit_ratios: list[float] = []
        for index, piglet in enumerate(piglets):
            is_edge_candidate = False
            interaction_area = 0
            if band_bool is not None and inner_band is not None:
                piglet_mask = mask_for_shape(piglet.mask, band_bool.shape)
                interaction_area = int(np.logical_and(piglet_mask, band_bool).sum())
                inner_area = int(np.logical_and(piglet_mask, inner_band).sum())
                is_edge_candidate = interaction_area > 0 and inner_area < self.min_inner_area

            if not is_edge_candidate:
                accepted.append(piglet)
                continue

            edge_candidates += 1
            edge_candidate_area_px += interaction_area
            hits = 1 + sum(index in matches for matches in previous_matches)
            edge_hit_ratios.append(float(hits / max(1, self.window)))
            if hits >= self.required_hits:
                accepted.append(piglet)
                confirmed_edge_candidates += 1
            else:
                pending_edge_candidates += 1
                pending_edge_area_px += interaction_area

        self.history.append(list(piglets))
        return accepted, EdgeConfirmationStats(
            edge_candidates=edge_candidates,
            pending_edge_candidates=pending_edge_candidates,
            confirmed_edge_candidates=confirmed_edge_candidates,
            edge_candidate_area_px=edge_candidate_area_px,
            pending_edge_area_px=pending_edge_area_px,
            edge_hit_ratio_mean=float(np.mean(edge_hit_ratios)) if edge_hit_ratios else 0.0,
        )


def make_overlay(
    image: np.ndarray,
    band: np.ndarray,
    arc_points: np.ndarray,
    sow: InstanceMask,
    piglets: list[InstanceMask],
    contacts: list[Contact],
    selected: list[Contact],
) -> np.ndarray:
    overlay = image.copy()
    overlay = blend_mask(overlay, sow.mask, (255, 170, 140), 0.22)
    piglet_union = union_instance_masks(piglets, image.shape[:2])
    overlay = blend_mask(overlay, piglet_union, (170, 170, 255), 0.28)
    overlay = blend_mask(overlay, band, (0, 255, 180), 0.45)

    cv2.polylines(overlay, [arc_points.reshape(-1, 1, 2)], False, (0, 180, 0), 3)
    selected_ids = {contact.piglet_index for contact in selected}
    for contact in contacts:
        color = (0, 255, 255) if contact.piglet_index in selected_ids else (0, 130, 255)
        center = (int(round(contact.contact_center[0])), int(round(contact.contact_center[1])))
        cv2.circle(overlay, center, 5, color, -1)
        cv2.circle(overlay, contact.contour_point, 5, (0, 255, 0), -1)
        cv2.line(overlay, center, contact.contour_point, color, 1)
    return overlay


def metadata(
    image_path: Path,
    args: argparse.Namespace,
    interval: dict[str, float],
    sow: InstanceMask,
    piglets: list[InstanceMask],
    contacts: list[Contact],
    selected: list[Contact],
) -> dict[str, Any]:
    selected_ids = {contact.piglet_index for contact in selected}
    return {
        "image": str(image_path),
        "weights": {
            "piglet": str(args.piglet_weights),
            "sow": str(args.sow_weights),
        },
        "nursing_interval": {key: round(value, 3) for key, value in interval.items()},
        "sow": {
            "instance_index": sow.index,
            "label": sow.label,
            "confidence": round(sow.confidence, 6),
            "area_px": sow.area,
        },
        "piglets": [
            {
                "instance_index": piglet.index,
                "label": piglet.label,
                "confidence": round(piglet.confidence, 6),
                "area_px": piglet.area,
            }
            for piglet in piglets
        ],
        "contacts": [
            {
                "piglet_index": contact.piglet_index,
                "selected": contact.piglet_index in selected_ids,
                "min_distance_px": round(contact.min_distance_px, 3),
                "contact_center": [round(v, 2) for v in contact.contact_center],
                "contour_point": list(contact.contour_point),
                "contour_s": round(contact.contour_s, 3),
            }
            for contact in contacts
        ],
        "params": {
            "max_contact_dist": args.max_contact_dist,
            "contact_margin": args.contact_margin,
            "cluster_gap": args.cluster_gap,
            "head_projection_quantile": args.head_projection_quantile,
            "arc_padding": args.arc_padding,
            "core_band_width": args.band_width,
            "band_outer_width": args.band_outer_width,
            "band_inner_width": args.band_inner_width,
            "contour_smooth_window": args.contour_smooth_window,
            "turn_window": args.turn_window,
            "max_turn_angle": args.max_turn_angle,
            "min_turn_radius": args.min_turn_radius,
            "max_axis_deviation": args.max_axis_deviation,
            "max_contour_bridge": args.max_contour_bridge,
            "max_endpoint_extension": args.max_endpoint_extension,
        },
    }


def predict_labeled_instances(
    model: Any,
    image_source: Any,
    args: argparse.Namespace,
    label: str,
    start_index: int = 0,
) -> list[InstanceMask]:
    source = str(image_source) if isinstance(image_source, Path) else image_source
    results = model.predict(
        source=source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device or None,
        max_det=args.max_det,
        half=args.half and args.device != "cpu",
        retina_masks=args.retina_masks,
        verbose=False,
    )
    if not results:
        return []
    return result_instances(results[0], args.mask_threshold, label_override=label, start_index=start_index)


def predict_labeled_instances_batch(
    model: Any,
    image_sources: list[np.ndarray],
    args: argparse.Namespace,
    label: str,
    start_index: int = 0,
) -> list[list[InstanceMask]]:
    if not image_sources:
        return []
    results = model.predict(
        source=image_sources,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device or None,
        max_det=args.max_det,
        half=args.half and args.device != "cpu",
        retina_masks=args.retina_masks,
        batch=min(args.batch_size, len(image_sources)),
        verbose=False,
    )
    if len(results) != len(image_sources):
        raise RuntimeError(f"Expected {len(image_sources)} prediction results, got {len(results)}.")
    return [
        result_instances(result, args.mask_threshold, label_override=label, start_index=start_index)
        for result in results
    ]


def process_image(
    piglet_model: Any,
    sow_model: Any,
    image_path: Path,
    input_dir: Path,
    args: argparse.Namespace,
) -> tuple[Path, Path, Path, Path]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    height, width = image.shape[:2]

    stem = output_stem(image_path, input_dir, args.output, args.recursive)
    stem.parent.mkdir(parents=True, exist_ok=True)
    mask_path = stem.with_name(f"{stem.name}_nursing_band.png")
    centerline_path = stem.with_name(f"{stem.name}_nursing_centerline.png")
    overlay_path = stem.with_name(f"{stem.name}_nursing_overlay.jpg")
    json_path = stem.with_name(f"{stem.name}_nursing_band.json")
    for path in (mask_path, centerline_path, overlay_path, json_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} already exists. Use --overwrite.")

    sow_instances = predict_labeled_instances(sow_model, image_path, args, args.sow_label, start_index=0)
    piglet_instances = predict_labeled_instances(
        piglet_model,
        image_path,
        args,
        args.piglet_label,
        start_index=len(sow_instances),
    )
    instances = sow_instances + piglet_instances
    band, centerline, arc_points, sow, piglets, contacts, selected, interval = extract_band(instances, (height, width), args)
    overlay = make_overlay(image, band, arc_points, sow, piglets, contacts, selected)

    cv2.imwrite(str(mask_path), band)
    cv2.imwrite(str(centerline_path), centerline)
    cv2.imwrite(str(overlay_path), overlay)
    json_path.write_text(
        json.dumps(metadata(image_path, args, interval, sow, piglets, contacts, selected), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return mask_path, centerline_path, overlay_path, json_path


def interaction_stats(piglets: list[InstanceMask], band: np.ndarray | None) -> dict[str, float | int]:
    if band is None:
        return {
            "band_area_px": 0,
            "piglet_count": len(piglets),
            "active_piglet_count": 0,
            "total_interaction_area_px": 0,
            "union_interaction_area_px": 0,
            "mean_interaction_area_per_piglet_px": 0.0,
            "mean_interaction_area_active_piglet_px": 0.0,
            "total_interaction_area_ratio": 0.0,
            "union_interaction_area_ratio": 0.0,
        }

    band_bool = band.astype(bool)
    band_area = int(band_bool.sum())
    areas: list[int] = []
    for piglet in piglets:
        if piglet.mask.shape != band_bool.shape:
            resized = cv2.resize(
                piglet.mask.astype(np.uint8),
                (band_bool.shape[1], band_bool.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        else:
            resized = piglet.mask
        areas.append(int(np.logical_and(resized, band_bool).sum()))

    total = int(sum(areas))
    piglet_union = union_instance_masks(piglets, band_bool.shape)
    union_total = int(np.logical_and(piglet_union, band_bool).sum())
    active = [area for area in areas if area > 0]
    piglet_count = len(areas)
    active_count = len(active)
    return {
        "band_area_px": band_area,
        "piglet_count": piglet_count,
        "active_piglet_count": active_count,
        "total_interaction_area_px": total,
        "union_interaction_area_px": union_total,
        "mean_interaction_area_per_piglet_px": float(total / piglet_count) if piglet_count else 0.0,
        "mean_interaction_area_active_piglet_px": float(total / active_count) if active_count else 0.0,
        "total_interaction_area_ratio": float(total / band_area) if band_area else 0.0,
        "union_interaction_area_ratio": float(union_total / band_area) if band_area else 0.0,
    }


def prefix_interaction_stats(stats: dict[str, float | int], prefix: str) -> dict[str, float | int]:
    """Prefix a complete interaction-stat block for parallel model/QC channels."""
    normalized_prefix = prefix if not prefix or prefix.endswith("_") else f"{prefix}_"
    return {f"{normalized_prefix}{key}": value for key, value in stats.items()}


def interaction_union_mask(piglets: list[InstanceMask], band: np.ndarray | None) -> np.ndarray | None:
    if band is None:
        return None
    band_bool = band.astype(bool, copy=False)
    return np.logical_and(union_instance_masks(piglets, band_bool.shape), band_bool)


def mask_transition_metrics(
    current_mask: np.ndarray | None,
    previous_mask: np.ndarray | None,
    band: np.ndarray | None,
) -> dict[str, float | None]:
    """Describe geometry change without reducing it to an area-only difference."""
    empty = {"iou": None, "xor_union_ratio": None, "xor_band_ratio": None}
    if current_mask is None or previous_mask is None or band is None:
        return empty
    band_bool = band.astype(bool, copy=False)
    current = np.logical_and(mask_for_shape(current_mask, band_bool.shape), band_bool)
    previous = np.logical_and(mask_for_shape(previous_mask, band_bool.shape), band_bool)
    intersection_area = int(np.logical_and(current, previous).sum())
    union_area = int(np.logical_or(current, previous).sum())
    xor_area = int(np.logical_xor(current, previous).sum())
    band_area = int(band_bool.sum())
    if union_area == 0:
        iou = 1.0
        xor_union_ratio = 0.0
    else:
        iou = float(intersection_area / union_area)
        xor_union_ratio = float(xor_area / union_area)
    return {
        "iou": iou,
        "xor_union_ratio": xor_union_ratio,
        "xor_band_ratio": float(xor_area / band_area) if band_area else 0.0,
    }


def active_piglets_in_band(piglets: list[InstanceMask], band: np.ndarray | None) -> list[InstanceMask]:
    if band is None:
        return []
    band_bool = band.astype(bool, copy=False)
    return [
        piglet
        for piglet in piglets
        if np.logical_and(mask_for_shape(piglet.mask, band_bool.shape), band_bool).any()
    ]


def instance_motion_metrics(
    current: list[InstanceMask],
    previous: list[InstanceMask] | None,
    band: np.ndarray | None,
    dt_sec: float | None,
    max_centroid_distance: float,
    max_area_ratio: float,
) -> dict[str, float | int | None]:
    """Match active permissive instances and summarize motion in body lengths per second."""
    empty: dict[str, float | int | None] = {
        "matched_active_piglet_count": None,
        "active_match_fraction": None,
        "centroid_speed_median_body_per_sec": None,
        "centroid_speed_p75_body_per_sec": None,
        "matched_piglet_mask_iou_median": None,
    }
    if previous is None or band is None or dt_sec is None or dt_sec <= 0:
        return empty
    current_active = active_piglets_in_band(current, band)
    previous_active = active_piglets_in_band(previous, band)
    pairs = match_piglet_pairs_one_to_one(
        current_active,
        previous_active,
        max_centroid_distance=max_centroid_distance,
        max_area_ratio=max_area_ratio,
    )
    speeds: list[float] = []
    mask_ious: list[float] = []
    for current_index, previous_index in pairs:
        current_item = current_active[current_index]
        previous_item = previous_active[previous_index]
        current_geometry = piglet_geometry(current_item)
        previous_geometry = piglet_geometry(previous_item)
        if current_geometry is None or previous_geometry is None:
            continue
        distance = math.hypot(
            current_geometry.centroid[0] - previous_geometry.centroid[0],
            current_geometry.centroid[1] - previous_geometry.centroid[1],
        )
        body_scale = math.sqrt(max(1.0, (current_geometry.area + previous_geometry.area) / 2.0))
        speeds.append(float(distance / body_scale / dt_sec))
        current_mask = current_item.mask.astype(bool, copy=False)
        previous_mask = mask_for_shape(previous_item.mask, current_mask.shape)
        union_area = int(np.logical_or(current_mask, previous_mask).sum())
        intersection_area = int(np.logical_and(current_mask, previous_mask).sum())
        mask_ious.append(float(intersection_area / union_area) if union_area else 1.0)
    denominator = len(current_active) + len(previous_active)
    return {
        "matched_active_piglet_count": len(pairs),
        "active_match_fraction": float(2 * len(pairs) / denominator) if denominator else 1.0,
        "centroid_speed_median_body_per_sec": float(np.median(speeds)) if speeds else None,
        "centroid_speed_p75_body_per_sec": float(np.percentile(speeds, 75)) if speeds else None,
        "matched_piglet_mask_iou_median": float(np.median(mask_ious)) if mask_ious else None,
    }


def make_interaction_overlay(
    image: np.ndarray,
    band: np.ndarray | None,
    core_band: np.ndarray | None,
    sow: InstanceMask | None,
    piglets: list[InstanceMask],
    timestamp_sec: float,
    stats: dict[str, float | int],
    roi_state: str = "confirmed",
) -> np.ndarray:
    overlay = image.copy()
    if sow is not None:
        overlay = blend_mask(overlay, sow.mask, (255, 170, 140), 0.18)
    piglet_union = union_instance_masks(piglets, image.shape[:2])
    overlay = blend_mask(overlay, piglet_union, (170, 170, 255), 0.24)
    if band is not None:
        overlay = blend_mask(overlay, band, (0, 255, 180), 0.4)
        band_bool = band.astype(bool)
        interaction = np.logical_and(piglet_union, band_bool)
        overlay = blend_mask(overlay, interaction, (0, 220, 255), 0.65)
    if core_band is not None:
        contours, _ = cv2.findContours(core_band.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 255, 0), 1, cv2.LINE_AA)
    text = (
        f"t={timestamp_sec:.1f}s ROI={roi_state} sum={int(stats['total_interaction_area_px'])} "
        f"union={int(stats['union_interaction_area_px'])} "
        f"mean={float(stats['mean_interaction_area_per_piglet_px']):.1f} piglets={int(stats['piglet_count'])}"
    )
    cv2.putText(overlay, text, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(overlay, text, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return overlay


def write_interaction_csv(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    interaction_stat_keys = [
        "band_area_px",
        "piglet_count",
        "active_piglet_count",
        "total_interaction_area_px",
        "union_interaction_area_px",
        "mean_interaction_area_per_piglet_px",
        "mean_interaction_area_active_piglet_px",
        "total_interaction_area_ratio",
        "union_interaction_area_ratio",
    ]
    parallel_stat_fields = [
        f"{prefix}{key}"
        for prefix in ("prefilter_", "permissive_", "pre_edge_", "core_")
        for key in interaction_stat_keys
    ]
    motion_fields = [
        f"{prefix}{metric}_{lag}"
        for prefix in ("interaction_mask_", "permissive_interaction_mask_")
        for lag in ("prev", "1s")
        for metric in ("iou", "xor_union_ratio", "xor_band_ratio")
    ]
    fieldnames = [
        "sample_index",
        "timestamp_sec",
        "frame_index",
        "sow_checked",
        "sow_mask_iou",
        "sow_mask_change_ratio",
        "sow_mask_age_sec",
        "time_since_roi_refresh_sec",
        "sow_change_exceedance_streak",
        "roi_refresh_requested",
        "roi_refresh_reason",
        "band_refreshed",
        "roi_state",
        "initial_roi_candidate_count",
        "initial_roi_consensus_support",
        "initial_roi_consensus_iou",
        "roi_locked",
        "roi_lock_support_streak",
        "roi_unlock_streak",
        "contact_support_count",
        "current_contact_coverage",
        "candidate_contact_coverage",
        "candidate_contact_coverage_gain",
        "candidate_band_iou_current",
        "candidate_band_iou_previous",
        "candidate_sow_iou_previous",
        "sow_centroid_shift_ratio",
        "sow_bbox_shift_ratio",
        "sow_long_axis_angle_deg",
        "gross_pose_change",
        "minor_refresh_cooldown_active",
        "roi_contact_recovery_streak",
        "roi_contact_recovery_evidence",
        "roi_contact_recovery_candidate_consistent",
        "roi_contact_recovery_side",
        "roi_contact_recovery_longitudinal_center_ratio",
        "raw_piglet_count",
        "sow_overlap_rejected_count",
        "hard_sow_overlap_rejected_count",
        "sow_overlap_ratio_max",
        "sow_overlap_ratio_median",
        "sow_overlap_ratio_p75",
        "sow_overlap_rejected_union_interaction_area_px",
        "edge_candidate_count",
        "pending_edge_candidate_count",
        "confirmed_edge_candidate_count",
        "edge_candidate_area_px",
        "pending_edge_candidate_area_px",
        "edge_hit_ratio_mean",
        *parallel_stat_fields,
        *interaction_stat_keys,
        *motion_fields,
        "matched_active_piglet_count",
        "active_match_fraction",
        "centroid_speed_median_body_per_sec",
        "centroid_speed_p75_body_per_sec",
        "matched_piglet_mask_iou_median",
        "status",
        "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if value in {None, ""}:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values


def conservative_mode(values: list[float]) -> int:
    counts: dict[int, int] = {}
    for value in values:
        item = int(round(value))
        counts[item] = counts.get(item, 0) + 1
    if not counts:
        return 0
    best_count = max(counts.values())
    return min(item for item, count in counts.items() if count == best_count)


def interaction_rows_with_segments(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    segmented: list[dict[str, Any]] = []
    segment_id = -1
    accepted_band_seen = False
    for source_row in rows:
        row = dict(source_row)
        if truthy(row.get("band_refreshed")):
            if accepted_band_seen:
                segment_id += 1
            else:
                segment_id = 0
                accepted_band_seen = True
        if segment_id < 0:
            segment_id = 0
        row["roi_segment_id"] = segment_id
        segmented.append(row)
    return segmented


def aggregate_interaction_rows(
    rows: list[dict[str, Any]],
    bin_sec: float,
) -> list[dict[str, Any]]:
    groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in interaction_rows_with_segments(rows):
        timestamp_sec = float(row.get("timestamp_sec", 0.0) or 0.0)
        bin_index = int(math.floor(timestamp_sec / bin_sec + 1e-9))
        key = (int(row["roi_segment_id"]), bin_index)
        groups.setdefault(key, []).append(row)

    aggregated: list[dict[str, Any]] = []
    median_keys = (
        "band_area_px",
        "total_interaction_area_px",
        "union_interaction_area_px",
        "mean_interaction_area_per_piglet_px",
        "total_interaction_area_ratio",
        "union_interaction_area_ratio",
        "prefilter_total_interaction_area_px",
        "prefilter_union_interaction_area_px",
        "prefilter_union_interaction_area_ratio",
        "permissive_total_interaction_area_px",
        "permissive_union_interaction_area_px",
        "permissive_union_interaction_area_ratio",
        "pre_edge_total_interaction_area_px",
        "pre_edge_union_interaction_area_px",
        "pre_edge_union_interaction_area_ratio",
        "core_band_area_px",
        "core_total_interaction_area_px",
        "core_union_interaction_area_px",
        "core_mean_interaction_area_per_piglet_px",
        "core_total_interaction_area_ratio",
        "core_union_interaction_area_ratio",
        "interaction_mask_xor_union_ratio_prev",
        "interaction_mask_xor_band_ratio_prev",
        "interaction_mask_xor_union_ratio_1s",
        "interaction_mask_xor_band_ratio_1s",
        "permissive_interaction_mask_xor_union_ratio_prev",
        "permissive_interaction_mask_xor_band_ratio_prev",
        "permissive_interaction_mask_xor_union_ratio_1s",
        "permissive_interaction_mask_xor_band_ratio_1s",
        "active_match_fraction",
        "centroid_speed_median_body_per_sec",
        "centroid_speed_p75_body_per_sec",
        "matched_piglet_mask_iou_median",
        "sow_mask_age_sec",
        "time_since_roi_refresh_sec",
        "sow_overlap_ratio_max",
        "sow_overlap_ratio_median",
        "sow_overlap_ratio_p75",
        "sow_overlap_rejected_union_interaction_area_px",
        "edge_candidate_area_px",
        "pending_edge_candidate_area_px",
        "edge_hit_ratio_mean",
        "contact_support_count",
        "current_contact_coverage",
        "candidate_contact_coverage",
        "candidate_contact_coverage_gain",
        "candidate_band_iou_current",
        "candidate_band_iou_previous",
        "candidate_sow_iou_previous",
        "sow_centroid_shift_ratio",
        "sow_bbox_shift_ratio",
        "sow_long_axis_angle_deg",
    )
    for (segment_id, bin_index), group in groups.items():
        valid = sorted(
            [
            row
            for row in group
            if row.get("status") in {"ok", "warning"} and row.get("total_interaction_area_px") not in {None, ""}
            ],
            key=lambda row: float(row.get("timestamp_sec", 0.0) or 0.0),
        )
        refreshed_rows = [row for row in group if truthy(row.get("band_refreshed"))]
        output: dict[str, Any] = {
            "roi_segment_id": segment_id,
            "second_index": bin_index,
            "second_start_sec": round(bin_index * bin_sec, 6),
            "timestamp_sec": round(float(np.median([float(row["timestamp_sec"]) for row in valid])), 6)
            if valid
            else round((bin_index + 0.5) * bin_sec, 6),
            "source_sample_count": len(group),
            "valid_sample_count": len(valid),
            "band_refreshed": bool(refreshed_rows),
            "roi_refresh_reason": refreshed_rows[0].get("roi_refresh_reason", "") if refreshed_rows else "",
            "edge_candidate_sample_count": sum(int(row.get("edge_candidate_count", 0) or 0) > 0 for row in group),
            "pending_edge_sample_count": sum(
                int(row.get("pending_edge_candidate_count", 0) or 0) > 0 for row in group
            ),
            "confirmed_edge_sample_count": sum(
                int(row.get("confirmed_edge_candidate_count", 0) or 0) > 0 for row in group
            ),
            "sow_overlap_rejection_sample_count": sum(
                int(row.get("sow_overlap_rejected_count", 0) or 0) > 0 for row in group
            ),
            "hard_sow_overlap_rejection_sample_count": sum(
                int(row.get("hard_sow_overlap_rejected_count", 0) or 0) > 0 for row in group
            ),
            "roi_locked_sample_count": sum(truthy(row.get("roi_locked")) for row in group),
            "gross_pose_change_sample_count": sum(truthy(row.get("gross_pose_change")) for row in group),
            "minor_refresh_cooldown_sample_count": sum(
                truthy(row.get("minor_refresh_cooldown_active")) for row in group
            ),
            "postprocess_status": "ok" if valid else "no_valid_samples",
        }
        source_count = max(1, len(group))
        output["edge_candidate_sample_fraction"] = output["edge_candidate_sample_count"] / source_count
        output["pending_edge_sample_fraction"] = output["pending_edge_sample_count"] / source_count
        output["confirmed_edge_sample_fraction"] = output["confirmed_edge_sample_count"] / source_count
        output["sow_overlap_rejection_sample_fraction"] = (
            output["sow_overlap_rejection_sample_count"] / source_count
        )
        output["hard_sow_overlap_rejection_sample_fraction"] = (
            output["hard_sow_overlap_rejection_sample_count"] / source_count
        )
        output["roi_locked_sample_fraction"] = output["roi_locked_sample_count"] / source_count
        output["gross_pose_change_sample_fraction"] = output["gross_pose_change_sample_count"] / source_count
        output["minor_refresh_cooldown_sample_fraction"] = (
            output["minor_refresh_cooldown_sample_count"] / source_count
        )
        for key in median_keys:
            values = numeric_values(valid, key)
            output[f"raw_{key}"] = float(np.median(values)) if values else None

        distribution_keys = (
            "union_interaction_area_ratio",
            "prefilter_union_interaction_area_ratio",
            "permissive_union_interaction_area_ratio",
            "core_union_interaction_area_ratio",
            "interaction_mask_xor_union_ratio_prev",
            "interaction_mask_xor_band_ratio_prev",
            "permissive_interaction_mask_xor_union_ratio_prev",
            "permissive_interaction_mask_xor_band_ratio_prev",
        )
        for key in distribution_keys:
            values = numeric_values(valid, key)
            if not values:
                continue
            source = np.asarray(values, dtype=np.float64)
            output[f"raw_{key}_p10"] = float(np.percentile(source, 10))
            output[f"raw_{key}_p90"] = float(np.percentile(source, 90))
            output[f"raw_{key}_iqr"] = float(np.percentile(source, 75) - np.percentile(source, 25))
            output[f"raw_{key}_std"] = float(np.std(source))
            output[f"raw_{key}_min"] = float(np.min(source))
            output[f"raw_{key}_max"] = float(np.max(source))
            output[f"raw_{key}_within_bin_tv"] = float(np.abs(np.diff(source)).sum()) if len(source) > 1 else 0.0

        output["raw_piglet_count_mode"] = conservative_mode(numeric_values(valid, "piglet_count")) if valid else None
        output["raw_active_piglet_count_mode"] = (
            conservative_mode(numeric_values(valid, "active_piglet_count")) if valid else None
        )
        if output["raw_piglet_count_mode"] is not None:
            output["raw_active_piglet_fraction"] = (
                float(output["raw_active_piglet_count_mode"] / output["raw_piglet_count_mode"])
                if output["raw_piglet_count_mode"]
                else 0.0
            )
        for prefix in ("prefilter_", "permissive_", "core_"):
            piglet_values = numeric_values(valid, f"{prefix}piglet_count")
            active_values = numeric_values(valid, f"{prefix}active_piglet_count")
            if not piglet_values:
                continue
            piglet_mode = conservative_mode(piglet_values)
            active_mode = min(piglet_mode, conservative_mode(active_values)) if active_values else 0
            output[f"raw_{prefix}piglet_count_mode"] = piglet_mode
            output[f"raw_{prefix}active_piglet_count_mode"] = active_mode
            output[f"raw_{prefix}active_piglet_fraction"] = active_mode / piglet_mode if piglet_mode else 0.0
        for source_key, output_key in (
            ("piglet_count", "piglet_count"),
            ("active_piglet_count", "active_piglet_count"),
            ("prefilter_piglet_count", "prefilter_piglet_count"),
            ("prefilter_active_piglet_count", "prefilter_active_piglet_count"),
            ("permissive_piglet_count", "permissive_piglet_count"),
            ("permissive_active_piglet_count", "permissive_active_piglet_count"),
            ("core_piglet_count", "core_piglet_count"),
            ("core_active_piglet_count", "core_active_piglet_count"),
        ):
            values = [int(round(value)) for value in numeric_values(valid, source_key)]
            if not values:
                continue
            output[f"raw_{output_key}_min"] = min(values)
            output[f"raw_{output_key}_max"] = max(values)
            output[f"raw_{output_key}_range"] = max(values) - min(values)
            output[f"raw_{output_key}_transition_count"] = sum(
                current != previous for previous, current in zip(values, values[1:])
            )
        aggregated.append(output)
    return sorted(aggregated, key=lambda row: (int(row["roi_segment_id"]), int(row["second_index"])))


def seconds_to_odd_bins(seconds: float, bin_sec: float, minimum: int = 3) -> int:
    bins = max(minimum, int(round(seconds / bin_sec)))
    return bins if bins % 2 == 1 else bins + 1


def hampel_despike(
    values: list[float],
    window: int,
    sigma: float,
    relative_threshold: float,
    absolute_floor: float,
    max_pulse_bins: int,
) -> HampelResult:
    source = np.asarray(values, dtype=np.float64)
    count = len(source)
    if count == 0:
        return HampelResult([], [], [], [], [])

    half = window // 2
    medians = np.empty(count, dtype=np.float64)
    thresholds = np.empty(count, dtype=np.float64)
    flags = np.zeros(count, dtype=bool)
    for index in range(count):
        local = source[max(0, index - half) : min(count, index + half + 1)]
        local_median = float(np.median(local))
        mad = float(np.median(np.abs(local - local_median)))
        threshold = max(sigma * 1.4826 * mad, relative_threshold * abs(local_median), absolute_floor)
        medians[index] = local_median
        thresholds[index] = threshold
        flags[index] = abs(float(source[index]) - local_median) > threshold

    cleaned = source.copy()
    replaced = np.zeros(count, dtype=bool)
    start = 0
    while start < count:
        if not flags[start]:
            start += 1
            continue
        end = start
        while end + 1 < count and flags[end + 1]:
            end += 1
        if end - start + 1 <= max_pulse_bins:
            left = float(source[start - 1]) if start > 0 and not flags[start - 1] else None
            right = float(source[end + 1]) if end + 1 < count and not flags[end + 1] else None
            if left is not None and right is not None:
                replacements = np.linspace(left, right, end - start + 3, dtype=np.float64)[1:-1]
            elif left is not None:
                replacements = np.full(end - start + 1, left, dtype=np.float64)
            elif right is not None:
                replacements = np.full(end - start + 1, right, dtype=np.float64)
            else:
                replacements = medians[start : end + 1]
            cleaned[start : end + 1] = replacements
            replaced[start : end + 1] = True
        start = end + 1

    return HampelResult(
        values=cleaned.tolist(),
        flags=flags.tolist(),
        replaced=replaced.tolist(),
        local_medians=medians.tolist(),
        thresholds=thresholds.tolist(),
    )


def centered_median_smooth(values: list[float], window: int) -> list[float]:
    source = np.asarray(values, dtype=np.float64)
    if len(source) < window or window <= 1:
        return source.tolist()
    half = window // 2
    padded = np.pad(source, (half, half), mode="edge")
    return [float(np.median(padded[index : index + window])) for index in range(len(source))]


def savgol_smooth(values: list[float], window: int, polyorder: int) -> list[float]:
    source = np.asarray(values, dtype=np.float64)
    count = len(source)
    if count < window or window < 3:
        return source.tolist()
    half = window // 2
    degree = min(polyorder, window - 1)
    smoothed = np.empty(count, dtype=np.float64)
    for index in range(count):
        start = min(max(0, index - half), count - window)
        stop = start + window
        x = np.arange(start, stop, dtype=np.float64) - index
        design = np.vander(x, degree + 1, increasing=True)
        coefficients, _, _, _ = np.linalg.lstsq(design, source[start:stop], rcond=None)
        smoothed[index] = coefficients[0]
    return smoothed.tolist()


def debounce_counts(values: list[int], confirm_bins: int) -> list[int]:
    if not values:
        return []
    output: list[int] = []
    stable = int(values[0])
    index = 0
    while index < len(values):
        candidate = int(values[index])
        end = index + 1
        while end < len(values) and int(values[end]) == candidate:
            end += 1
        if candidate == stable or end - index >= confirm_bins:
            stable = candidate
        output.extend([stable] * (end - index))
        index = end
    return output


def valid_contiguous_runs(rows: list[dict[str, Any]]) -> list[list[int]]:
    runs: list[list[int]] = []
    current: list[int] = []
    previous_segment: int | None = None
    previous_bin: int | None = None
    for index, row in enumerate(rows):
        segment = int(row["roi_segment_id"])
        bin_index = int(row["second_index"])
        valid = row.get("postprocess_status") == "ok"
        contiguous = (
            valid
            and current
            and segment == previous_segment
            and previous_bin is not None
            and bin_index == previous_bin + 1
        )
        if not valid or (current and not contiguous):
            if current:
                runs.append(current)
            current = []
        if valid:
            current.append(index)
            previous_segment = segment
            previous_bin = bin_index
        else:
            previous_segment = None
            previous_bin = None
    if current:
        runs.append(current)
    return runs


def postprocess_interaction_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    processed = aggregate_interaction_rows(rows, args.postprocess_bin_sec)
    output_keys = (
        "total_hampel_candidate_flag",
        "union_ratio_hampel_candidate_flag",
        "outlier_flag",
        "outlier_reason",
        "total_hampel_local_median_px",
        "total_hampel_threshold_px",
        "union_ratio_hampel_local_median",
        "union_ratio_hampel_threshold",
        "despiked_total_interaction_area_px",
        "despiked_union_interaction_area_ratio",
        "debounced_piglet_count",
        "debounced_active_piglet_count",
        "quantitative_total_interaction_area_px",
        "quantitative_union_interaction_area_ratio",
        "quantitative_union_interaction_area_px",
        "quantitative_mean_interaction_area_per_piglet_px",
        "trend_total_interaction_area_px",
        "trend_union_interaction_area_ratio",
        "trend_union_interaction_area_px",
        "trend_mean_interaction_area_per_piglet_px",
        "trend_piglet_count",
        "trend_active_piglet_count",
        "trend_active_piglet_fraction",
        "display_total_interaction_area_px",
        "display_union_interaction_area_ratio",
        "display_union_interaction_area_px",
        "display_mean_interaction_area_per_piglet_px",
    )
    for row in processed:
        for key in output_keys:
            row[key] = None

    hampel_window = seconds_to_odd_bins(args.hampel_window_sec, args.postprocess_bin_sec)
    median_window = seconds_to_odd_bins(args.quantitative_median_sec, args.postprocess_bin_sec)
    sg_window = seconds_to_odd_bins(args.display_sg_window_sec, args.postprocess_bin_sec)
    max_pulse_bins = max(1, int(round(args.hampel_max_pulse_sec / args.postprocess_bin_sec)))
    confirm_bins = max(1, int(math.ceil(args.count_confirm_sec / args.postprocess_bin_sec)))

    for run in valid_contiguous_runs(processed):
        totals = [float(processed[index]["raw_total_interaction_area_px"]) for index in run]
        union_ratios = [float(processed[index]["raw_union_interaction_area_ratio"]) for index in run]
        piglet_counts = [int(processed[index]["raw_piglet_count_mode"]) for index in run]
        active_counts = [int(processed[index]["raw_active_piglet_count_mode"]) for index in run]

        total_hampel = hampel_despike(
            totals,
            window=hampel_window,
            sigma=args.hampel_sigma,
            relative_threshold=args.hampel_relative_threshold,
            absolute_floor=args.hampel_total_floor,
            max_pulse_bins=max_pulse_bins,
        )
        ratio_hampel = hampel_despike(
            union_ratios,
            window=hampel_window,
            sigma=args.hampel_sigma,
            relative_threshold=args.hampel_relative_threshold,
            absolute_floor=args.hampel_union_ratio_floor,
            max_pulse_bins=max_pulse_bins,
        )
        quantitative_totals = centered_median_smooth(total_hampel.values, median_window)
        quantitative_ratios = centered_median_smooth(ratio_hampel.values, median_window)
        display_totals = savgol_smooth(total_hampel.values, sg_window, args.display_sg_polyorder)
        display_ratios = savgol_smooth(ratio_hampel.values, sg_window, args.display_sg_polyorder)
        debounced_piglets = debounce_counts(piglet_counts, confirm_bins)
        debounced_active = debounce_counts(active_counts, confirm_bins)

        for offset, row_index in enumerate(run):
            row = processed[row_index]
            piglet_count = max(0, debounced_piglets[offset])
            active_count = min(piglet_count, max(0, debounced_active[offset]))
            band_area = float(row["raw_band_area_px"] or 0.0)
            quantitative_total = max(0.0, float(quantitative_totals[offset]))
            display_total = max(0.0, float(display_totals[offset]))
            quantitative_ratio = float(np.clip(quantitative_ratios[offset], 0.0, 1.0))
            display_ratio = float(np.clip(display_ratios[offset], 0.0, 1.0))
            reasons: list[str] = []
            if total_hampel.replaced[offset]:
                reasons.append("summed_total_hampel")
            if ratio_hampel.replaced[offset]:
                reasons.append("union_ratio_hampel")
            row.update(
                {
                    "total_hampel_candidate_flag": total_hampel.flags[offset],
                    "union_ratio_hampel_candidate_flag": ratio_hampel.flags[offset],
                    "outlier_flag": bool(reasons),
                    "outlier_reason": "|".join(reasons),
                    "total_hampel_local_median_px": total_hampel.local_medians[offset],
                    "total_hampel_threshold_px": total_hampel.thresholds[offset],
                    "union_ratio_hampel_local_median": ratio_hampel.local_medians[offset],
                    "union_ratio_hampel_threshold": ratio_hampel.thresholds[offset],
                    "despiked_total_interaction_area_px": max(0.0, total_hampel.values[offset]),
                    "despiked_union_interaction_area_ratio": float(np.clip(ratio_hampel.values[offset], 0.0, 1.0)),
                    "debounced_piglet_count": piglet_count,
                    "debounced_active_piglet_count": active_count,
                    "quantitative_total_interaction_area_px": quantitative_total,
                    "quantitative_union_interaction_area_ratio": quantitative_ratio,
                    "quantitative_union_interaction_area_px": quantitative_ratio * band_area,
                    "quantitative_mean_interaction_area_per_piglet_px": quantitative_total / piglet_count
                    if piglet_count
                    else 0.0,
                    "trend_total_interaction_area_px": quantitative_total,
                    "trend_union_interaction_area_ratio": quantitative_ratio,
                    "trend_union_interaction_area_px": quantitative_ratio * band_area,
                    "trend_mean_interaction_area_per_piglet_px": quantitative_total / piglet_count
                    if piglet_count
                    else 0.0,
                    "trend_piglet_count": piglet_count,
                    "trend_active_piglet_count": active_count,
                    "trend_active_piglet_fraction": active_count / piglet_count if piglet_count else 0.0,
                    "display_total_interaction_area_px": display_total,
                    "display_union_interaction_area_ratio": display_ratio,
                    "display_union_interaction_area_px": display_ratio * band_area,
                    "display_mean_interaction_area_per_piglet_px": display_total / piglet_count
                    if piglet_count
                    else 0.0,
                }
            )
    return processed


def write_processed_interaction_csv(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "roi_segment_id",
        "second_index",
        "second_start_sec",
        "timestamp_sec",
        "source_sample_count",
        "valid_sample_count",
        "band_refreshed",
        "roi_refresh_reason",
        "edge_candidate_sample_count",
        "pending_edge_sample_count",
        "sow_overlap_rejection_sample_count",
        "postprocess_status",
        "raw_band_area_px",
        "raw_total_interaction_area_px",
        "raw_union_interaction_area_px",
        "raw_mean_interaction_area_per_piglet_px",
        "raw_total_interaction_area_ratio",
        "raw_union_interaction_area_ratio",
        "raw_piglet_count_mode",
        "raw_active_piglet_count_mode",
        "total_hampel_candidate_flag",
        "union_ratio_hampel_candidate_flag",
        "outlier_flag",
        "outlier_reason",
        "total_hampel_local_median_px",
        "total_hampel_threshold_px",
        "union_ratio_hampel_local_median",
        "union_ratio_hampel_threshold",
        "despiked_total_interaction_area_px",
        "despiked_union_interaction_area_ratio",
        "debounced_piglet_count",
        "debounced_active_piglet_count",
        "quantitative_total_interaction_area_px",
        "quantitative_union_interaction_area_ratio",
        "quantitative_union_interaction_area_px",
        "quantitative_mean_interaction_area_per_piglet_px",
        "display_total_interaction_area_px",
        "display_union_interaction_area_ratio",
        "display_union_interaction_area_px",
        "display_mean_interaction_area_per_piglet_px",
    ]
    known = set(fieldnames)
    extra_fields = sorted({key for row in rows for key in row if key not in known})
    fieldnames.extend(extra_fields)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def plot_interaction(
    rows: list[dict[str, Any]],
    processed_rows: list[dict[str, Any]],
    plot_path: Path,
    title: str,
    dpi: int,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on local plotting install
        raise RuntimeError("matplotlib is required to write the interaction plot.") from exc

    segmented_rows = interaction_rows_with_segments(rows)
    good_rows = [row for row in segmented_rows if row.get("status") in {"ok", "warning"}]
    good_processed = [row for row in processed_rows if row.get("postprocess_status") == "ok"]
    if not good_rows:
        raise ValueError("No valid rows available for plotting.")

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    raw_specs = (
        ("union_interaction_area_ratio", axes[0], 100.0),
        ("total_interaction_area_px", axes[1], 1.0),
        ("mean_interaction_area_per_piglet_px", axes[2], 1.0),
    )
    quantitative_specs = (
        ("quantitative_union_interaction_area_ratio", axes[0], 100.0),
        ("quantitative_total_interaction_area_px", axes[1], 1.0),
        ("quantitative_mean_interaction_area_per_piglet_px", axes[2], 1.0),
    )
    display_specs = (
        ("display_union_interaction_area_ratio", axes[0], 100.0),
        ("display_total_interaction_area_px", axes[1], 1.0),
        ("display_mean_interaction_area_per_piglet_px", axes[2], 1.0),
    )

    raw_label_used = False
    for segment_id in sorted({int(row["roi_segment_id"]) for row in good_rows}):
        segment = [row for row in good_rows if int(row["roi_segment_id"]) == segment_id]
        times = [float(row["timestamp_sec"]) / 60.0 for row in segment]
        for key, axis, scale in raw_specs:
            axis.plot(
                times,
                [float(row[key]) * scale for row in segment],
                color="#9CA3AF",
                linewidth=0.7,
                alpha=0.45,
                label="Raw sampled" if not raw_label_used else None,
            )
        raw_label_used = True

    quantitative_label_used = False
    display_label_used = False
    for run in valid_contiguous_runs(processed_rows):
        segment = [processed_rows[index] for index in run]
        times = [float(row["timestamp_sec"]) / 60.0 for row in segment]
        for key, axis, scale in quantitative_specs:
            axis.plot(
                times,
                [float(row[key]) * scale for row in segment],
                color="#0F766E",
                linewidth=1.25,
                linestyle="--",
                alpha=0.85,
                label="1 Hz quantitative median" if not quantitative_label_used else None,
            )
        for key, axis, scale in display_specs:
            axis.plot(
                times,
                [float(row[key]) * scale for row in segment],
                color="#2563EB",
                linewidth=1.8,
                label="1 Hz Hampel + SG" if not display_label_used else None,
            )
        quantitative_label_used = True
        display_label_used = True

    outliers = [row for row in good_processed if truthy(row.get("outlier_flag"))]
    if outliers:
        outlier_times = [float(row["timestamp_sec"]) / 60.0 for row in outliers]
        outlier_values = (
            [float(row["raw_union_interaction_area_ratio"]) * 100.0 for row in outliers],
            [float(row["raw_total_interaction_area_px"]) for row in outliers],
            [float(row["raw_mean_interaction_area_per_piglet_px"]) for row in outliers],
        )
        for axis, values in zip(axes, outlier_values):
            axis.scatter(outlier_times, values, color="#DC2626", marker="x", s=24, linewidths=1.2, zorder=5)

    refresh_times = [
        float(row["timestamp_sec"]) / 60.0
        for row in rows
        if truthy(row.get("band_refreshed"))
        and row.get("roi_refresh_reason") not in {"initial", "initial_confirmed"}
    ]
    for axis in axes:
        for refresh_time in refresh_times:
            axis.axvline(refresh_time, color="#7C3AED", linewidth=0.9, linestyle=":", alpha=0.65)
        axis.grid(True, alpha=0.2)

    axes[0].set_ylabel("Unique ROI occupancy (%)")
    axes[0].set_ylim(bottom=0.0, top=100.0)
    axes[0].set_title(title)
    axes[0].legend(loc="upper right", frameon=False, ncol=3)
    axes[1].set_ylabel("Summed interaction (px)")
    axes[2].set_ylabel("Mean per piglet (px)")
    axes[2].set_xlabel("Time (min)")

    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=dpi)
    plt.close(fig)


def video_duration_sec(capture: cv2.VideoCapture, fps: float, frame_count: int) -> float:
    if fps > 0 and frame_count > 0:
        return float(frame_count / fps)
    milliseconds = float(capture.get(cv2.CAP_PROP_POS_MSEC))
    return milliseconds / 1000.0 if milliseconds > 0 else 0.0


def video_sample_frame_indices(fps: float, frame_count: int, sample_fps: float) -> list[int]:
    if fps <= 0 or frame_count <= 0:
        return []
    step = max(1, int(round(fps / sample_fps)))
    stop = max(1, frame_count - (2 * step) + 1)
    return list(range(0, stop, step))


def resolve_edge_confirmation(
    sample_interval_sec: float,
    window_sec: float,
    hit_ratio: float,
    legacy_window: int | None = None,
    legacy_hits: int | None = None,
) -> tuple[int, int]:
    """Resolve time-based edge confirmation to the sampled-frame grid.

    The legacy fixed-frame pair is retained for reproducibility. New runs should use
    seconds plus a hit ratio so the filter has comparable behavior at different
    ``--sample-fps`` values.
    """
    if (legacy_window is None) != (legacy_hits is None):
        raise ValueError("--edge-confirm-window and --edge-confirm-hits must be provided together.")
    if legacy_window is not None and legacy_hits is not None:
        if legacy_window <= 0 or not 1 <= legacy_hits <= legacy_window:
            raise ValueError("Legacy edge-confirmation values must satisfy 1 <= hits <= window.")
        return int(legacy_window), int(legacy_hits)
    if sample_interval_sec <= 0 or window_sec <= 0:
        raise ValueError("Time-scaled edge confirmation requires positive sample interval and window seconds.")
    if not 0.0 < hit_ratio <= 1.0:
        raise ValueError("--edge-confirm-hit-ratio must be in (0, 1].")
    window = max(1, int(math.floor(window_sec / sample_interval_sec + 1e-9)) + 1)
    hits = min(window, max(1, int(math.ceil(hit_ratio * window - 1e-12))))
    return window, hits


def read_sampled_frames_sequential(
    capture: cv2.VideoCapture,
    target_indices: list[int],
    decode_position: int,
) -> tuple[list[np.ndarray | None], int]:
    frames: list[np.ndarray | None] = []
    for target_index in target_indices:
        if target_index < decode_position:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(target_index))
            decode_position = target_index

        decode_ok = True
        while decode_position < target_index:
            if not capture.grab():
                decode_ok = False
                break
            decode_position += 1

        if not decode_ok:
            frames.append(None)
            continue

        ok, frame = capture.read()
        if ok and frame is not None:
            frames.append(frame)
            decode_position += 1
        else:
            frames.append(None)
    return frames, decode_position


def process_video(
    piglet_model: Any,
    sow_model: Any,
    video_path: Path,
    input_root: Path,
    args: argparse.Namespace,
) -> tuple[Path, Path, Path, Path, Path | None]:
    stem = output_stem(video_path, input_root, args.output, args.recursive)
    stem.parent.mkdir(parents=True, exist_ok=True)
    csv_path = stem.with_name(f"{stem.name}_interaction.csv")
    processed_csv_path = stem.with_name(f"{stem.name}_interaction_1hz.csv")
    json_path = stem.with_name(f"{stem.name}_interaction_summary.json")
    plot_path = stem.with_name(f"{stem.name}_interaction_plot.png")
    demo_path = stem.with_name(f"{stem.name}_interaction_demo.mp4")
    output_paths = [csv_path, processed_csv_path, json_path, plot_path]
    if args.save_demo_video:
        output_paths.append(demo_path)
    for path in output_paths:
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} already exists. Use --overwrite.")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = video_duration_sec(capture, fps, frame_count)
    if duration <= 0 and fps > 0 and frame_count > 0:
        duration = frame_count / fps
    if duration <= 0:
        raise ValueError(f"Could not determine video duration: {video_path}")
    if args.sample_fps <= 0:
        raise ValueError("--sample-fps must be > 0.")
    if args.sow_check_sec <= 0:
        raise ValueError("--sow-check-sec must be > 0.")
    if not 0.0 <= args.sow_mask_change_threshold <= 1.0:
        raise ValueError("--sow-mask-change-threshold must be between 0 and 1.")
    if args.sow_change_confirm_checks <= 0:
        raise ValueError("--sow-change-confirm-checks must be > 0.")
    if not args.sow_mask_change_threshold <= args.sow_mask_change_immediate_threshold <= 1.0:
        raise ValueError(
            "--sow-mask-change-immediate-threshold must be between --sow-mask-change-threshold and 1."
        )
    if args.band_width <= 0 or args.band_outer_width <= 0 or args.band_inner_width < 0:
        raise ValueError("ROI band widths must be positive; --band-inner-width may be zero.")
    if args.band_outer_width <= args.band_inner_width:
        raise ValueError("--band-outer-width must be greater than --band-inner-width.")
    for name in (
        "roi_lock_coverage",
        "roi_unlock_coverage",
        "roi_candidate_min_iou",
        "roi_candidate_coverage_gain",
        "roi_contact_recovery_current_max",
        "roi_contact_recovery_candidate_min",
        "roi_low_contact_sow_threshold",
        "sow_gross_change_threshold",
        "sow_gross_centroid_shift_ratio",
        "sow_gross_bbox_shift_ratio",
    ):
        if not 0.0 <= float(getattr(args, name)) <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be between 0 and 1.")
    if args.roi_unlock_coverage >= args.roi_lock_coverage:
        raise ValueError("--roi-unlock-coverage must be smaller than --roi-lock-coverage.")
    if args.roi_contact_recovery_current_max >= args.roi_contact_recovery_candidate_min:
        raise ValueError(
            "--roi-contact-recovery-current-max must be smaller than "
            "--roi-contact-recovery-candidate-min."
        )
    if args.roi_lock_min_contacts <= 0:
        raise ValueError("--roi-lock-min-contacts must be > 0.")
    if min(
        args.roi_lock_confirm_checks,
        args.roi_unlock_confirm_checks,
        args.roi_low_contact_confirm_checks,
        args.roi_contact_recovery_confirm_checks,
    ) <= 0:
        raise ValueError("ROI lock and low-contact confirmation checks must be > 0.")
    if (
        args.roi_minor_refresh_cooldown_sec < 0
        or args.roi_contact_recovery_max_center_shift_ratio < 0
        or args.sow_gross_angle_deg < 0
    ):
        raise ValueError(
            "ROI cooldown, recovery-shift, and gross-pose angle thresholds must be >= 0."
        )
    if args.initial_roi_confirm_sec <= 0 or args.initial_roi_candidate_sec <= 0:
        raise ValueError("Initial ROI confirmation durations must be > 0.")
    if args.initial_roi_min_candidates <= 0:
        raise ValueError("--initial-roi-min-candidates must be > 0.")
    if not 0.0 <= args.initial_roi_min_iou <= 1.0:
        raise ValueError("--initial-roi-min-iou must be between 0 and 1.")
    if not 0.0 < args.initial_roi_min_consensus_ratio <= 1.0:
        raise ValueError("--initial-roi-min-consensus-ratio must be in (0, 1].")
    if not 0.0 <= args.max_piglet_sow_overlap <= 1.0:
        raise ValueError("--max-piglet-sow-overlap must be between 0 and 1.")
    if not 0.0 <= args.hard_max_piglet_sow_overlap <= 1.0:
        raise ValueError("--hard-max-piglet-sow-overlap must be between 0 and 1.")
    if args.hard_max_piglet_sow_overlap < args.max_piglet_sow_overlap:
        raise ValueError("--hard-max-piglet-sow-overlap must be >= --max-piglet-sow-overlap.")
    if args.roi_edge_margin < 0:
        raise ValueError("--roi-edge-margin must be >= 0.")
    if args.roi_edge_min_inner_area < 0:
        raise ValueError("--roi-edge-min-inner-area must be >= 0.")
    if (args.edge_confirm_window is None) != (args.edge_confirm_hits is None):
        raise ValueError("--edge-confirm-window and --edge-confirm-hits must be provided together.")
    if args.edge_confirm_window_sec <= 0:
        raise ValueError("--edge-confirm-window-sec must be > 0.")
    if not 0.0 < args.edge_confirm_hit_ratio <= 1.0:
        raise ValueError("--edge-confirm-hit-ratio must be in (0, 1].")
    if args.edge_track_max_distance <= 0:
        raise ValueError("--edge-track-max-distance must be > 0.")
    if args.edge_track_max_area_ratio < 1:
        raise ValueError("--edge-track-max-area-ratio must be >= 1.")
    if args.postprocess_bin_sec <= 0:
        raise ValueError("--postprocess-bin-sec must be > 0.")
    if args.hampel_window_sec <= 0 or args.quantitative_median_sec <= 0 or args.display_sg_window_sec <= 0:
        raise ValueError("Postprocessing window durations must be > 0.")
    if args.hampel_sigma <= 0:
        raise ValueError("--hampel-sigma must be > 0.")
    if args.hampel_relative_threshold < 0 or args.hampel_total_floor < 0 or args.hampel_union_ratio_floor < 0:
        raise ValueError("Hampel thresholds must be >= 0.")
    if args.hampel_max_pulse_sec <= 0 or args.count_confirm_sec <= 0:
        raise ValueError("Postprocessing pulse and count-confirm durations must be > 0.")
    if args.display_sg_polyorder < 0:
        raise ValueError("--display-sg-polyorder must be >= 0.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0.")
    if args.save_demo_video and args.demo_fps <= 0:
        raise ValueError("--demo-fps must be > 0 when --save-demo-video is used.")

    sample_frame_indices = video_sample_frame_indices(fps, frame_count, args.sample_fps)
    if not sample_frame_indices:
        raise ValueError(f"Could not build sample frame indices for video: {video_path}")
    sample_interval_sec = (
        float((sample_frame_indices[1] - sample_frame_indices[0]) / fps)
        if len(sample_frame_indices) > 1 and fps > 0
        else float(1.0 / args.sample_fps)
    )
    edge_confirm_window, edge_confirm_hits = resolve_edge_confirmation(
        sample_interval_sec=sample_interval_sec,
        window_sec=args.edge_confirm_window_sec,
        hit_ratio=args.edge_confirm_hit_ratio,
        legacy_window=args.edge_confirm_window,
        legacy_hits=args.edge_confirm_hits,
    )
    band: np.ndarray | None = None
    core_band: np.ndarray | None = None
    roi_reference_sow: InstanceMask | None = None
    provisional_sow: InstanceMask | None = None
    initial_roi_started_sec: float | None = None
    next_initial_roi_candidate_sec = 0.0
    initial_roi_candidates: deque[InitialRoiCandidate] = deque()
    next_sow_check_sec = 0.0
    last_successful_sow_check_sec: float | None = None
    last_roi_refresh_sec: float | None = None
    sow_change_exceedance_streak = 0
    roi_locked = False
    roi_lock_support_streak = 0
    roi_unlock_streak = 0
    pending_roi_candidate: RoiRefreshCandidate | None = None
    roi_contact_recovery_streak = 0
    pending_roi_contact_recovery_signature: RoiContactRecoverySignature | None = None
    edge_confirmer = EdgePigletConfirmer(
        window=edge_confirm_window,
        required_hits=edge_confirm_hits,
        edge_margin=args.roi_edge_margin,
        min_inner_area=args.roi_edge_min_inner_area,
        max_centroid_distance=args.edge_track_max_distance,
        max_area_ratio=args.edge_track_max_area_ratio,
    )
    rows: list[dict[str, Any]] = []
    overlay_dir = stem.with_name(f"{stem.name}_overlays")
    overlays_written = 0
    demo_writer: AsyncVideoWriter | None = None
    demo_frames = 0
    decode_position = 0
    progress_interval = max(1, int(round(args.sample_fps * 30)))
    motion_history: deque[dict[str, Any]] = deque()

    for batch_start in range(0, len(sample_frame_indices), args.batch_size):
        batch_targets = sample_frame_indices[batch_start : batch_start + args.batch_size]
        batch_frames, decode_position = read_sampled_frames_sequential(capture, batch_targets, decode_position)
        valid_offsets = [offset for offset, frame in enumerate(batch_frames) if frame is not None]
        valid_frames = [batch_frames[offset] for offset in valid_offsets]
        piglets_by_offset: dict[int, list[InstanceMask]] = {}
        errors_by_offset: dict[int, str] = {}

        if valid_frames:
            try:
                batch_piglets = predict_labeled_instances_batch(
                    piglet_model,
                    valid_frames,
                    args,
                    args.piglet_label,
                    start_index=1,
                )
                piglets_by_offset.update(zip(valid_offsets, batch_piglets))
            except Exception as batch_exc:
                print(f"[WARN] Batch piglet inference failed; retrying frame by frame: {batch_exc}")
                for offset in valid_offsets:
                    frame = batch_frames[offset]
                    try:
                        piglets_by_offset[offset] = predict_labeled_instances(
                            piglet_model, frame, args, args.piglet_label, start_index=1
                        )
                    except Exception as frame_exc:
                        errors_by_offset[offset] = f"Piglet refresh failed: {frame_exc}"

        for offset, (target_frame_index, frame) in enumerate(zip(batch_targets, batch_frames)):
            sample_index = batch_start + offset
            timestamp_sec = float(target_frame_index / fps)
            if frame is None:
                edge_confirmer.update([], band)
                motion_history.clear()
                rows.append(
                    {
                        "sample_index": sample_index,
                        "timestamp_sec": round(timestamp_sec, 3),
                        "frame_index": "",
                        "sow_checked": False,
                        "roi_refresh_requested": False,
                        "roi_refresh_reason": "read_failed",
                        "band_refreshed": False,
                        "raw_piglet_count": 0,
                        "sow_overlap_rejected_count": 0,
                        "edge_candidate_count": 0,
                        "pending_edge_candidate_count": 0,
                        "status": "read_failed",
                        "error": "Could not read sampled frame.",
                    }
                )
                continue

            raw_piglets = piglets_by_offset.get(offset, [])
            error = errors_by_offset.get(offset, "")
            had_band = band is not None
            scheduled_sow_check = timestamp_sec + 1e-9 >= next_sow_check_sec
            need_sow_check = scheduled_sow_check or (band is None and provisional_sow is None)
            sow_checked = False
            candidate_sow: InstanceMask | None = None
            sow_change_ratio: float | None = None
            roi_refresh_requested = False
            roi_refresh_reason = "not_checked"
            band_refreshed = False
            initial_roi_candidate_count = len(initial_roi_candidates)
            initial_roi_consensus_support = 0
            initial_roi_consensus_iou = 0.0
            contact_support_count = 0
            current_contact_coverage = 0.0
            candidate_contact_coverage = 0.0
            candidate_contact_coverage_gain = 0.0
            candidate_band_iou_current: float | None = None
            candidate_band_iou_previous: float | None = None
            candidate_sow_iou_previous: float | None = None
            pose_metrics = {
                "centroid_shift_ratio": 0.0,
                "bbox_shift_ratio": 0.0,
                "long_axis_angle_deg": 0.0,
            }
            gross_pose_change = False
            minor_refresh_cooldown_active = False
            roi_contact_recovery_streak_observed = roi_contact_recovery_streak
            roi_contact_recovery_evidence = False
            roi_contact_recovery_candidate_consistent = False
            roi_contact_recovery_candidate_inconsistent = False
            roi_contact_recovery_signature_side: int | str = ""
            roi_contact_recovery_longitudinal_center_ratio: float | str = ""
            refresh_candidate: RoiRefreshCandidate | None = None

            if need_sow_check:
                sow_checked = True
                try:
                    sow_instances = predict_labeled_instances(sow_model, frame, args, args.sow_label, start_index=0)
                    candidate_sow = select_sow_instance(sow_instances, args)
                    last_successful_sow_check_sec = timestamp_sec
                except Exception as exc:
                    sow_change_exceedance_streak = 0
                    message = f"Sow mask check failed: {exc}"
                    error = f"{error}; {message}" if error else message
                    roi_refresh_reason = "sow_check_failed"

                if scheduled_sow_check:
                    while next_sow_check_sec <= timestamp_sec + 1e-9:
                        next_sow_check_sec += args.sow_check_sec

            if band is None and candidate_sow is not None:
                if provisional_sow is None:
                    initial_roi_started_sec = timestamp_sec
                    next_initial_roi_candidate_sec = timestamp_sec
                provisional_sow = candidate_sow

            sow_for_filter = (
                candidate_sow
                if candidate_sow is not None
                else provisional_sow
                if band is None
                else roi_reference_sow
            )
            permissive_piglets, hard_overlap_rejected, sow_overlap_ratios = partition_piglets_by_sow_overlap(
                raw_piglets,
                sow_for_filter,
                args.hard_max_piglet_sow_overlap,
            )
            spatial_piglets, strict_overlap_rejected, _ = partition_piglets_by_sow_overlap(
                raw_piglets,
                sow_for_filter,
                args.max_piglet_sow_overlap,
            )
            sow_overlap_rejected = len(strict_overlap_rejected)

            if had_band and candidate_sow is not None and roi_reference_sow is not None:
                sow_change_ratio = sow_mask_change_ratio(roi_reference_sow.mask, candidate_sow.mask)
                pose_metrics = sow_pose_change_metrics(roi_reference_sow.mask, candidate_sow.mask)
                gross_pose_change = is_gross_sow_pose_change(sow_change_ratio, pose_metrics, args)
                try:
                    height, width = frame.shape[:2]
                    refresh_candidate = build_roi_refresh_candidate(
                        candidate_sow,
                        spatial_piglets,
                        band,
                        (height, width),
                        timestamp_sec,
                        args,
                    )
                    contact_support_count = len(refresh_candidate.selected_contacts)
                    current_contact_coverage = refresh_candidate.current_contact_coverage
                    candidate_contact_coverage = refresh_candidate.candidate_contact_coverage
                    candidate_contact_coverage_gain = candidate_contact_coverage - current_contact_coverage
                    candidate_band_iou_current = binary_mask_iou(band, refresh_candidate.band)
                    if pending_roi_candidate is not None:
                        candidate_band_iou_previous = binary_mask_iou(
                            pending_roi_candidate.band,
                            refresh_candidate.band,
                        )
                        candidate_sow_iou_previous = binary_mask_iou(
                            pending_roi_candidate.sow.mask,
                            refresh_candidate.sow.mask,
                        )
                except Exception:
                    refresh_candidate = None

                contact_supported = (
                    contact_support_count >= args.roi_lock_min_contacts
                    and current_contact_coverage >= args.roi_lock_coverage
                )
                contact_lost = (
                    contact_support_count < args.roi_lock_min_contacts
                    or current_contact_coverage < args.roi_unlock_coverage
                )
                if contact_supported:
                    roi_lock_support_streak += 1
                    roi_unlock_streak = 0
                    if roi_lock_support_streak >= args.roi_lock_confirm_checks:
                        roi_locked = True
                elif roi_locked and contact_lost:
                    roi_unlock_streak += 1
                    roi_lock_support_streak = 0
                    if roi_unlock_streak >= args.roi_unlock_confirm_checks:
                        roi_locked = False
                        roi_unlock_streak = 0
                elif not roi_locked:
                    roi_lock_support_streak = 0

                minor_refresh_cooldown_active = bool(
                    last_roi_refresh_sec is not None
                    and timestamp_sec - last_roi_refresh_sec < args.roi_minor_refresh_cooldown_sec
                )

                recovery_signature = (
                    roi_contact_recovery_signature(
                        candidate_sow.mask,
                        refresh_candidate.selected_contacts,
                    )
                    if refresh_candidate is not None
                    else None
                )
                previous_recovery_streak = roi_contact_recovery_streak
                previous_recovery_signature = pending_roi_contact_recovery_signature
                (
                    roi_contact_recovery_streak,
                    pending_roi_contact_recovery_signature,
                    roi_contact_recovery_candidate_consistent,
                    roi_contact_recovery_evidence,
                ) = update_roi_contact_recovery(
                    previous_streak=previous_recovery_streak,
                    previous_signature=previous_recovery_signature,
                    candidate_signature=recovery_signature,
                    contact_count=contact_support_count,
                    current_coverage=current_contact_coverage,
                    candidate_coverage=candidate_contact_coverage,
                    min_contacts=args.roi_lock_min_contacts,
                    current_max=args.roi_contact_recovery_current_max,
                    candidate_min=args.roi_contact_recovery_candidate_min,
                    max_center_shift_ratio=args.roi_contact_recovery_max_center_shift_ratio,
                    cooldown_active=minor_refresh_cooldown_active,
                )
                roi_contact_recovery_candidate_inconsistent = bool(
                    roi_contact_recovery_evidence
                    and previous_recovery_streak > 0
                    and previous_recovery_signature is not None
                    and not roi_contact_recovery_candidate_consistent
                )
                roi_contact_recovery_streak_observed = roi_contact_recovery_streak
                if recovery_signature is not None:
                    roi_contact_recovery_signature_side = recovery_signature.side
                    roi_contact_recovery_longitudinal_center_ratio = (
                        recovery_signature.longitudinal_center_ratio
                    )

                if gross_pose_change:
                    sow_change_exceedance_streak = 0
                    pending_roi_candidate = None
                    if refresh_candidate is not None:
                        roi_refresh_requested = True
                        roi_refresh_reason = "sow_pose_changed_immediate"
                    else:
                        roi_refresh_reason = "gross_pose_candidate_failed"
                elif (
                    roi_contact_recovery_evidence
                    and roi_contact_recovery_streak >= args.roi_contact_recovery_confirm_checks
                    and refresh_candidate is not None
                ):
                    sow_change_exceedance_streak = 0
                    pending_roi_candidate = None
                    roi_refresh_requested = True
                    roi_refresh_reason = "contact_recovery_confirmed"
                elif roi_locked and contact_supported:
                    sow_change_exceedance_streak = 0
                    pending_roi_candidate = None
                    roi_refresh_reason = "roi_locked_contact_supported"
                elif sow_change_ratio > args.sow_mask_change_threshold:
                    if minor_refresh_cooldown_active:
                        sow_change_exceedance_streak = 0
                        pending_roi_candidate = None
                        roi_refresh_reason = "minor_refresh_cooldown"
                    elif refresh_candidate is None:
                        sow_change_exceedance_streak = 0
                        pending_roi_candidate = None
                        roi_refresh_reason = "refresh_candidate_failed"
                    else:
                        sow_change_exceedance_streak += 1
                        sufficient_contacts = contact_support_count >= args.roi_lock_min_contacts
                        required_checks = (
                            args.sow_change_confirm_checks
                            if sufficient_contacts
                            else args.roi_low_contact_confirm_checks
                        )
                        candidate_consistent = bool(
                            pending_roi_candidate is not None
                            and candidate_band_iou_previous is not None
                            and candidate_band_iou_previous >= args.roi_candidate_min_iou
                        )
                        coverage_supports_refresh = bool(
                            sufficient_contacts
                            and (
                                current_contact_coverage < args.roi_unlock_coverage
                                or candidate_contact_coverage_gain >= args.roi_candidate_coverage_gain
                            )
                        )
                        low_contact_supports_refresh = bool(
                            not sufficient_contacts
                            and sow_change_ratio >= args.roi_low_contact_sow_threshold
                        )
                        if (
                            sow_change_exceedance_streak >= required_checks
                            and candidate_consistent
                            and (coverage_supports_refresh or low_contact_supports_refresh)
                        ):
                            roi_refresh_requested = True
                            roi_refresh_reason = "sow_contact_changed_confirmed"
                        elif not candidate_consistent and pending_roi_candidate is not None:
                            roi_refresh_reason = "refresh_candidate_inconsistent"
                        elif sufficient_contacts and not coverage_supports_refresh:
                            roi_refresh_reason = "current_roi_contact_supported"
                        elif not sufficient_contacts and not low_contact_supports_refresh:
                            roi_refresh_reason = "low_contact_change_pending"
                        else:
                            roi_refresh_reason = "sow_contact_change_pending"
                        pending_roi_candidate = refresh_candidate
                else:
                    sow_change_exceedance_streak = 0
                    pending_roi_candidate = None
                    # Keep the accepted reference unchanged so several small shifts can accumulate
                    # past the threshold.
                    roi_refresh_reason = "sow_stable"

                if roi_contact_recovery_evidence and not roi_refresh_requested:
                    roi_refresh_reason = (
                        "contact_recovery_candidate_inconsistent"
                        if roi_contact_recovery_candidate_inconsistent
                        else "contact_recovery_pending"
                    )

            elif had_band and sow_checked:
                roi_contact_recovery_streak = 0
                pending_roi_contact_recovery_signature = None
                roi_contact_recovery_streak_observed = 0

            if band is None:
                roi_refresh_requested = True
                roi_refresh_reason = "initial_pending"
                due_initial_candidate = (
                    provisional_sow is not None
                    and timestamp_sec + 1e-9 >= next_initial_roi_candidate_sec
                )
                if due_initial_candidate:
                    while next_initial_roi_candidate_sec <= timestamp_sec + 1e-9:
                        next_initial_roi_candidate_sec += args.initial_roi_candidate_sec
                    try:
                        height, width = frame.shape[:2]
                        new_band, _, arc_points, new_sow, _, _, selected, _ = extract_band(
                            [provisional_sow, *spatial_piglets],
                            (height, width),
                            args,
                        )
                        initial_roi_candidates.append(
                            InitialRoiCandidate(
                                timestamp_sec=timestamp_sec,
                                band=new_band,
                                sow=new_sow,
                                selected_contact_count=len(selected),
                                core_band=build_core_band((height, width), arc_points, args.band_width),
                            )
                        )
                    except Exception:
                        # A frame without a usable contact cluster is normal during initialization.
                        roi_refresh_reason = "initial_candidate_failed"

                if initial_roi_started_sec is not None:
                    while (
                        initial_roi_candidates
                        and timestamp_sec - initial_roi_candidates[0].timestamp_sec
                        > args.initial_roi_confirm_sec + 1e-9
                    ):
                        initial_roi_candidates.popleft()
                    initial_roi_candidate_count = len(initial_roi_candidates)
                    elapsed_initial_sec = timestamp_sec - initial_roi_started_sec
                    if elapsed_initial_sec + 1e-9 >= args.initial_roi_confirm_sec:
                        consensus, initial_roi_consensus_support, initial_roi_consensus_iou = (
                            select_initial_roi_consensus(
                                list(initial_roi_candidates),
                                min_candidates=args.initial_roi_min_candidates,
                                min_iou=args.initial_roi_min_iou,
                                min_consensus_ratio=args.initial_roi_min_consensus_ratio,
                            )
                        )
                        if consensus is not None:
                            band = consensus.band
                            core_band = consensus.core_band if consensus.core_band is not None else consensus.band
                            roi_reference_sow = consensus.sow
                            provisional_sow = None
                            initial_roi_candidates.clear()
                            band_refreshed = True
                            roi_refresh_reason = "initial_confirmed"
                        elif roi_refresh_reason != "initial_candidate_failed":
                            roi_refresh_reason = "initial_consensus_pending"

            if band_refreshed:
                edge_confirmer.reset()
            piglets, edge_stats = edge_confirmer.update(spatial_piglets, band)

            if had_band and roi_refresh_requested and refresh_candidate is not None:
                band = refresh_candidate.band
                core_band = refresh_candidate.core_band
                roi_reference_sow = refresh_candidate.sow
                band_refreshed = True
                roi_locked = False
                roi_lock_support_streak = 0
                roi_unlock_streak = 0
                pending_roi_candidate = None
                edge_confirmer.reset()
                piglets, edge_stats = edge_confirmer.update(spatial_piglets, band)

            if band_refreshed:
                last_roi_refresh_sec = timestamp_sec
                sow_change_exceedance_streak = 0
                pending_roi_candidate = None
                roi_contact_recovery_streak = 0
                pending_roi_contact_recovery_signature = None
                motion_history.clear()

            prefilter_stats = interaction_stats(raw_piglets, band)
            permissive_stats = interaction_stats(permissive_piglets, band)
            pre_edge_stats = interaction_stats(spatial_piglets, band)
            stats = interaction_stats(piglets, band)
            core_stats = interaction_stats(piglets, core_band)
            rejected_stats = interaction_stats(strict_overlap_rejected, band)

            current_masks = {
                "strict": interaction_union_mask(piglets, band),
                "permissive": interaction_union_mask(permissive_piglets, band),
            }
            previous_motion = motion_history[-1] if motion_history else None
            if previous_motion is not None:
                previous_gap = timestamp_sec - float(previous_motion["timestamp_sec"])
                if previous_gap <= 0 or previous_gap > sample_interval_sec * 1.75:
                    motion_history.clear()
                    previous_motion = None

            motion_fields: dict[str, Any] = {}
            for channel, output_prefix in (("strict", "interaction_mask_"), ("permissive", "permissive_interaction_mask_")):
                previous_mask = previous_motion["masks"][channel] if previous_motion is not None else None
                transition = mask_transition_metrics(current_masks[channel], previous_mask, band)
                motion_fields.update(
                    {
                        f"{output_prefix}iou_prev": transition["iou"],
                        f"{output_prefix}xor_union_ratio_prev": transition["xor_union_ratio"],
                        f"{output_prefix}xor_band_ratio_prev": transition["xor_band_ratio"],
                    }
                )

            lag_item: dict[str, Any] | None = None
            if motion_history:
                candidate_lag = min(
                    motion_history,
                    key=lambda item: abs((timestamp_sec - float(item["timestamp_sec"])) - 1.0),
                )
                lag_age = timestamp_sec - float(candidate_lag["timestamp_sec"])
                if abs(lag_age - 1.0) <= max(0.2, sample_interval_sec * 1.5):
                    lag_item = candidate_lag
            for channel, output_prefix in (("strict", "interaction_mask_"), ("permissive", "permissive_interaction_mask_")):
                lag_mask = lag_item["masks"][channel] if lag_item is not None else None
                transition = mask_transition_metrics(current_masks[channel], lag_mask, band)
                motion_fields.update(
                    {
                        f"{output_prefix}iou_1s": transition["iou"],
                        f"{output_prefix}xor_union_ratio_1s": transition["xor_union_ratio"],
                        f"{output_prefix}xor_band_ratio_1s": transition["xor_band_ratio"],
                    }
                )

            motion_fields.update(
                instance_motion_metrics(
                    permissive_piglets,
                    previous_motion["permissive_piglets"] if previous_motion is not None else None,
                    band,
                    timestamp_sec - float(previous_motion["timestamp_sec"]) if previous_motion is not None else None,
                    max_centroid_distance=args.edge_track_max_distance,
                    max_area_ratio=args.edge_track_max_area_ratio,
                )
            )
            if band is not None:
                motion_history.append(
                    {
                        "timestamp_sec": timestamp_sec,
                        "masks": current_masks,
                        "permissive_piglets": list(permissive_piglets),
                    }
                )
                while motion_history and timestamp_sec - float(motion_history[0]["timestamp_sec"]) > 1.5:
                    motion_history.popleft()
            else:
                motion_history.clear()

            roi_state = "confirmed" if band is not None else "provisional" if provisional_sow is not None else "unavailable"
            status = (
                "ok"
                if band is not None and not error
                else "warning"
                if band is not None
                else "roi_provisional"
                if provisional_sow is not None
                else "no_band"
            )
            rows.append(
                {
                    "sample_index": sample_index,
                    "timestamp_sec": round(timestamp_sec, 3),
                    "frame_index": target_frame_index,
                    "sow_checked": sow_checked,
                    "sow_mask_iou": round(1.0 - sow_change_ratio, 6) if sow_change_ratio is not None else "",
                    "sow_mask_change_ratio": round(sow_change_ratio, 6) if sow_change_ratio is not None else "",
                    "sow_mask_age_sec": round(timestamp_sec - last_successful_sow_check_sec, 3)
                    if last_successful_sow_check_sec is not None
                    else "",
                    "time_since_roi_refresh_sec": round(timestamp_sec - last_roi_refresh_sec, 3)
                    if last_roi_refresh_sec is not None
                    else "",
                    "sow_change_exceedance_streak": sow_change_exceedance_streak,
                    "roi_refresh_requested": roi_refresh_requested,
                    "roi_refresh_reason": roi_refresh_reason,
                    "band_refreshed": band_refreshed,
                    "roi_state": roi_state,
                    "initial_roi_candidate_count": initial_roi_candidate_count,
                    "initial_roi_consensus_support": initial_roi_consensus_support,
                    "initial_roi_consensus_iou": initial_roi_consensus_iou,
                    "roi_locked": roi_locked,
                    "roi_lock_support_streak": roi_lock_support_streak,
                    "roi_unlock_streak": roi_unlock_streak,
                    "contact_support_count": contact_support_count,
                    "current_contact_coverage": current_contact_coverage,
                    "candidate_contact_coverage": candidate_contact_coverage,
                    "candidate_contact_coverage_gain": candidate_contact_coverage_gain,
                    "candidate_band_iou_current": candidate_band_iou_current,
                    "candidate_band_iou_previous": candidate_band_iou_previous,
                    "candidate_sow_iou_previous": candidate_sow_iou_previous,
                    "sow_centroid_shift_ratio": pose_metrics["centroid_shift_ratio"],
                    "sow_bbox_shift_ratio": pose_metrics["bbox_shift_ratio"],
                    "sow_long_axis_angle_deg": pose_metrics["long_axis_angle_deg"],
                    "gross_pose_change": gross_pose_change,
                    "minor_refresh_cooldown_active": minor_refresh_cooldown_active,
                    "roi_contact_recovery_streak": roi_contact_recovery_streak_observed,
                    "roi_contact_recovery_evidence": roi_contact_recovery_evidence,
                    "roi_contact_recovery_candidate_consistent": (
                        roi_contact_recovery_candidate_consistent
                    ),
                    "roi_contact_recovery_side": roi_contact_recovery_signature_side,
                    "roi_contact_recovery_longitudinal_center_ratio": (
                        roi_contact_recovery_longitudinal_center_ratio
                    ),
                    "raw_piglet_count": len(raw_piglets),
                    "sow_overlap_rejected_count": sow_overlap_rejected,
                    "hard_sow_overlap_rejected_count": len(hard_overlap_rejected),
                    "sow_overlap_ratio_max": max(sow_overlap_ratios) if sow_overlap_ratios else 0.0,
                    "sow_overlap_ratio_median": float(np.median(sow_overlap_ratios)) if sow_overlap_ratios else 0.0,
                    "sow_overlap_ratio_p75": float(np.percentile(sow_overlap_ratios, 75))
                    if sow_overlap_ratios
                    else 0.0,
                    "sow_overlap_rejected_union_interaction_area_px": rejected_stats[
                        "union_interaction_area_px"
                    ],
                    "edge_candidate_count": edge_stats.edge_candidates,
                    "pending_edge_candidate_count": edge_stats.pending_edge_candidates,
                    "confirmed_edge_candidate_count": edge_stats.confirmed_edge_candidates,
                    "edge_candidate_area_px": edge_stats.edge_candidate_area_px,
                    "pending_edge_candidate_area_px": edge_stats.pending_edge_area_px,
                    "edge_hit_ratio_mean": edge_stats.edge_hit_ratio_mean,
                    "status": status,
                    "error": error,
                    **prefix_interaction_stats(prefilter_stats, "prefilter_"),
                    **prefix_interaction_stats(permissive_stats, "permissive_"),
                    **prefix_interaction_stats(pre_edge_stats, "pre_edge_"),
                    **prefix_interaction_stats(core_stats, "core_"),
                    **motion_fields,
                    **stats,
                }
            )

            if args.save_overlays or args.save_demo_video:
                overlay = make_interaction_overlay(
                    frame,
                    band,
                    core_band,
                    roi_reference_sow if band is not None else provisional_sow,
                    piglets,
                    timestamp_sec,
                    stats,
                    roi_state=roi_state,
                )
                if args.save_overlays and overlays_written < args.overlay_limit:
                    overlay_dir.mkdir(parents=True, exist_ok=True)
                    overlay_path = overlay_dir / f"{stem.name}_{sample_index:05d}_{timestamp_sec:08.2f}s.jpg"
                    cv2.imwrite(str(overlay_path), overlay)
                    overlays_written += 1
                if args.save_demo_video:
                    if demo_writer is None:
                        height, width = overlay.shape[:2]
                        demo_writer = AsyncVideoWriter(
                            demo_path,
                            args.demo_fps,
                            (width, height),
                            queue_size=args.batch_size * 2,
                        )
                    demo_writer.write(overlay)
                    demo_frames += 1

            if (sample_index + 1) % progress_interval == 0:
                print(f"[INFO] {video_path.name}: sampled {sample_index + 1}/{len(sample_frame_indices)} frames")

    capture.release()
    if demo_writer is not None:
        demo_writer.close()

    write_interaction_csv(csv_path, rows)
    processed_rows = postprocess_interaction_rows(rows, args)
    write_processed_interaction_csv(processed_csv_path, processed_rows)
    plot_interaction(rows, processed_rows, plot_path, video_path.name, args.plot_dpi)

    valid_rows = [row for row in rows if row.get("status") in {"ok", "warning"}]
    total_values = [float(row["total_interaction_area_px"]) for row in valid_rows]
    union_values = [float(row["union_interaction_area_px"]) for row in valid_rows]
    union_ratio_values = [float(row["union_interaction_area_ratio"]) for row in valid_rows]
    mean_values = [float(row["mean_interaction_area_per_piglet_px"]) for row in valid_rows]
    valid_processed_rows = [row for row in processed_rows if row.get("postprocess_status") == "ok"]
    display_union_ratios = [float(row["display_union_interaction_area_ratio"]) for row in valid_processed_rows]
    display_means = [
        float(row["display_mean_interaction_area_per_piglet_px"]) for row in valid_processed_rows
    ]
    summary = {
        "feature_schema_version": 4,
        "video": str(video_path),
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": round(duration, 3),
        "sample_fps": args.sample_fps,
        "actual_sample_interval_sec": sample_interval_sec,
        "actual_sample_fps": 1.0 / sample_interval_sec if sample_interval_sec > 0 else 0.0,
        "sow_check_sec": args.sow_check_sec,
        "sow_mask_change_threshold": args.sow_mask_change_threshold,
        "sow_change_confirm_checks": args.sow_change_confirm_checks,
        "sow_mask_change_immediate_threshold": args.sow_mask_change_immediate_threshold,
        "roi_geometry": {
            "primary": "fixed asymmetric wide band",
            "core_band_width_px": args.band_width,
            "outer_width_px": args.band_outer_width,
            "inner_width_px": args.band_inner_width,
            "core_feature_prefix": "core_",
        },
        "roi_refresh_policy": {
            "lock_min_contacts": args.roi_lock_min_contacts,
            "lock_coverage": args.roi_lock_coverage,
            "unlock_coverage": args.roi_unlock_coverage,
            "lock_confirm_checks": args.roi_lock_confirm_checks,
            "unlock_confirm_checks": args.roi_unlock_confirm_checks,
            "candidate_min_iou": args.roi_candidate_min_iou,
            "candidate_coverage_gain": args.roi_candidate_coverage_gain,
            "minor_refresh_cooldown_sec": args.roi_minor_refresh_cooldown_sec,
            "contact_recovery_current_max": args.roi_contact_recovery_current_max,
            "contact_recovery_candidate_min": args.roi_contact_recovery_candidate_min,
            "contact_recovery_confirm_checks": args.roi_contact_recovery_confirm_checks,
            "contact_recovery_max_center_shift_ratio": (
                args.roi_contact_recovery_max_center_shift_ratio
            ),
            "contact_recovery_refresh_count": sum(
                row.get("roi_refresh_reason") == "contact_recovery_confirmed" for row in rows
            ),
            "low_contact_confirm_checks": args.roi_low_contact_confirm_checks,
            "low_contact_sow_threshold": args.roi_low_contact_sow_threshold,
            "gross_change_threshold": args.sow_gross_change_threshold,
            "gross_centroid_shift_ratio": args.sow_gross_centroid_shift_ratio,
            "gross_bbox_shift_ratio": args.sow_gross_bbox_shift_ratio,
            "gross_angle_deg": args.sow_gross_angle_deg,
            "locked_sample_count": sum(bool(row.get("roi_locked")) for row in rows),
        },
        "initial_roi_confirmation": {
            "confirm_sec": args.initial_roi_confirm_sec,
            "candidate_sec": args.initial_roi_candidate_sec,
            "min_candidates": args.initial_roi_min_candidates,
            "min_iou": args.initial_roi_min_iou,
            "min_consensus_ratio": args.initial_roi_min_consensus_ratio,
            "provisional_sample_count": sum(row.get("status") == "roi_provisional" for row in rows),
        },
        "max_piglet_sow_overlap": args.max_piglet_sow_overlap,
        "hard_max_piglet_sow_overlap": args.hard_max_piglet_sow_overlap,
        "edge_confirmation": {
            "effective_window_samples": edge_confirm_window,
            "effective_required_hits": edge_confirm_hits,
            "configured_window_sec": args.edge_confirm_window_sec,
            "configured_hit_ratio": args.edge_confirm_hit_ratio,
            "legacy_window_override": args.edge_confirm_window,
            "legacy_hits_override": args.edge_confirm_hits,
            "roi_edge_margin_px": args.roi_edge_margin,
            "min_inner_area_px": args.roi_edge_min_inner_area,
            "track_max_distance_px": args.edge_track_max_distance,
            "track_max_area_ratio": args.edge_track_max_area_ratio,
        },
        "postprocessing": {
            "role": "trend_auxiliary_and_display_only; MODEL_RAW and raw 1 Hz distributions remain available",
            "bin_sec": args.postprocess_bin_sec,
            "hampel_window_sec": args.hampel_window_sec,
            "hampel_sigma": args.hampel_sigma,
            "hampel_relative_threshold": args.hampel_relative_threshold,
            "hampel_total_floor_px": args.hampel_total_floor,
            "hampel_union_ratio_floor": args.hampel_union_ratio_floor,
            "hampel_max_pulse_sec": args.hampel_max_pulse_sec,
            "count_confirm_sec": args.count_confirm_sec,
            "quantitative_median_sec": args.quantitative_median_sec,
            "display_sg_window_sec": args.display_sg_window_sec,
            "display_sg_polyorder": args.display_sg_polyorder,
        },
        "batch_size": args.batch_size,
        "samples": len(rows),
        "valid_samples": len(valid_rows),
        "outputs": {
            "csv": str(csv_path),
            "processed_1hz_csv": str(processed_csv_path),
            "plot": str(plot_path),
            "overlays": str(overlay_dir) if args.save_overlays else None,
            "demo_video": str(demo_path) if args.save_demo_video else None,
        },
        "output_contract": {
            "model_raw_4hz": {
                "prefilter_prefix": "prefilter_",
                "permissive_prefix": "permissive_",
                "strict_prefix": "unprefixed interaction metrics",
                "core_prefix": "core_ strict interaction metrics on the legacy 80 px band",
                "motion_fields": "interaction_mask_*, permissive_interaction_mask_*, centroid_speed_*, active_match_fraction",
            },
            "model_1hz": "raw_* medians/modes plus distribution, within-bin motion and QC fraction fields",
            "trend_auxiliary": "despiked_*, trend_* and legacy quantitative_* aliases",
            "display_only": "display_*",
        },
        "demo_video_frames": demo_frames,
        "total_interaction_area_px": {
            "mean": float(np.mean(total_values)) if total_values else 0.0,
            "max": float(np.max(total_values)) if total_values else 0.0,
        },
        "union_interaction_area_px": {
            "mean": float(np.mean(union_values)) if union_values else 0.0,
            "max": float(np.max(union_values)) if union_values else 0.0,
        },
        "union_interaction_area_ratio": {
            "mean": float(np.mean(union_ratio_values)) if union_ratio_values else 0.0,
            "max": float(np.max(union_ratio_values)) if union_ratio_values else 0.0,
        },
        "mean_interaction_area_per_piglet_px": {
            "mean": float(np.mean(mean_values)) if mean_values else 0.0,
            "max": float(np.max(mean_values)) if mean_values else 0.0,
        },
        "processed_1hz": {
            "rows": len(processed_rows),
            "valid_rows": len(valid_processed_rows),
            "roi_segments": len({int(row["roi_segment_id"]) for row in processed_rows}),
            "outliers_replaced": sum(truthy(row.get("outlier_flag")) for row in processed_rows),
            "display_union_interaction_area_ratio": {
                "mean": float(np.mean(display_union_ratios)) if display_union_ratios else 0.0,
                "max": float(np.max(display_union_ratios)) if display_union_ratios else 0.0,
            },
            "display_mean_interaction_area_per_piglet_px": {
                "mean": float(np.mean(display_means)) if display_means else 0.0,
                "max": float(np.max(display_means)) if display_means else 0.0,
            },
        },
        "warnings": sum(1 for row in rows if row.get("status") == "warning"),
        "no_band_samples": sum(1 for row in rows if row.get("status") == "no_band"),
        "read_failed_samples": sum(1 for row in rows if row.get("status") == "read_failed"),
        "sow_mask_checks": sum(bool(row.get("sow_checked")) for row in rows),
        "roi_refresh_requests": sum(bool(row.get("roi_refresh_requested")) for row in rows),
        "roi_refreshes": sum(bool(row.get("band_refreshed")) for row in rows),
        "sow_overlap_rejections": sum(int(row.get("sow_overlap_rejected_count", 0) or 0) for row in rows),
        "hard_sow_overlap_rejections": sum(
            int(row.get("hard_sow_overlap_rejected_count", 0) or 0) for row in rows
        ),
        "pending_edge_candidates": sum(int(row.get("pending_edge_candidate_count", 0) or 0) for row in rows),
        "pending_edge_candidate_area_px": sum(
            int(row.get("pending_edge_candidate_area_px", 0) or 0) for row in rows
        ),
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return csv_path, processed_csv_path, json_path, plot_path, demo_path if args.save_demo_video else None


def main() -> None:
    args = parse_args()
    mode = resolve_mode(args.input, args.mode, args.recursive)
    inputs = iter_videos(args.input, args.recursive) if mode == "video" else iter_images(args.input, args.recursive)
    if args.dry_run:
        print(f"Input: {args.input}")
        print(f"Mode: {mode}")
        print(f"Files: {len(inputs)}")
        print(f"Piglet weights: {args.piglet_weights}")
        print(f"Sow weights: {args.sow_weights}")
        print(f"Output: {args.output}")
        if mode == "video":
            print(f"Sample FPS: {args.sample_fps}")
            print(f"Sow-mask check seconds: {args.sow_check_sec}")
            print(f"Sow-mask change threshold: {args.sow_mask_change_threshold}")
            print(
                "Sow-overlap channels: "
                f"strict<={args.max_piglet_sow_overlap:.2f}, permissive<={args.hard_max_piglet_sow_overlap:.2f}"
            )
            if args.edge_confirm_window is not None:
                edge_description = f"legacy {args.edge_confirm_hits}/{args.edge_confirm_window} sampled frames"
            else:
                edge_description = (
                    f"time-scaled window={args.edge_confirm_window_sec:.2f}s, "
                    f"hit-ratio={args.edge_confirm_hit_ratio:.3f}"
                )
            print(
                "Edge confirmation: "
                f"{edge_description}, margin={args.roi_edge_margin}px, "
                f"min-inner-area={args.roi_edge_min_inner_area}px"
            )
            print(
                "Postprocessing: "
                f"bin={args.postprocess_bin_sec}s, Hampel={args.hampel_window_sec}s, "
                f"median={args.quantitative_median_sec}s, SG={args.display_sg_window_sec}s"
            )
            print(f"Inference batch size: {args.batch_size}")
            print(f"Save demo video: {args.save_demo_video}")
            if args.save_demo_video:
                print(f"Demo FPS: {args.demo_fps}")
        return

    if not args.piglet_weights.exists():
        raise FileNotFoundError(f"Piglet weights not found: {args.piglet_weights}")
    if not args.sow_weights.exists():
        raise FileNotFoundError(f"Sow weights not found: {args.sow_weights}")

    from ultralytics import YOLO

    args.output.mkdir(parents=True, exist_ok=True)
    piglet_model = YOLO(str(args.piglet_weights), task="segment")
    sow_model = YOLO(str(args.sow_weights), task="segment")
    written = 0
    failed = 0

    for index, input_path in enumerate(inputs, start=1):
        try:
            if mode == "video":
                csv_path, processed_csv_path, json_path, plot_path, demo_path = process_video(
                    piglet_model, sow_model, input_path, args.input, args
                )
                paths = [str(csv_path), str(processed_csv_path), str(plot_path), str(json_path)]
                if demo_path is not None:
                    paths.append(str(demo_path))
                print(f"[INFO] {index}/{len(inputs)} wrote: {' ; '.join(paths)}")
            else:
                _, _, overlay_path, json_path = process_image(piglet_model, sow_model, input_path, args.input, args)
                print(f"[INFO] {index}/{len(inputs)} wrote: {overlay_path} ; {json_path}")
            written += 1
        except Exception as exc:
            failed += 1
            message = f"[WARN] {index}/{len(inputs)} failed: {input_path} ({exc})"
            if args.fail_on_error:
                raise RuntimeError(message) from exc
            print(message)

    print(f"Done. mode={mode}, written={written}, failed={failed}, output={args.output}")

if __name__ == "__main__":
    main()
