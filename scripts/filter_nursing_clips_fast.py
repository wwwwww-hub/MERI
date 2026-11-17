"""Fast second-pass filtering for candidate nursing videos.

The first-pass UniformerV2 clips may contain a recumbent sow with too few
piglets, or piglets sleeping close to the udder.  This script keeps a window
only when all three inexpensive signals agree:

1. a spatially coherent sow/piglet contact band is stable in time;
2. at least four piglets continuously support that band; and
3. contact piglets are not consistently low-activity.

Inference uses one two-class segmentation model (sow + piglet), FFmpeg NVDEC,
FP16 batches, and a bounded decode prefetch queue.  The default action is an
auditable report; source videos are never changed unless an explicit reject
action is requested.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Iterator, Sequence

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.extract_nursing_band import (  # noqa: E402
    VIDEO_EXTS,
    InstanceMask,
    build_asymmetric_band,
    build_contacts,
    clean_contour_arc,
    cluster_contacts,
    contour_arclength,
    interval_points,
    largest_contour,
    mask_for_shape,
    match_piglet_pairs_one_to_one,
    minimal_arc,
    partition_piglets_by_sow_overlap,
)


# Directory for candidate nursing clips exported by DS1 screening (not bundled in release)
DEFAULT_INPUT = ROOT / "working_data" / "nursing_clips"
DEFAULT_OUTPUT = ROOT / "runs" / "nursing_clip_filter_fast"
DEFAULT_WEIGHTS = ROOT / "models" / "comparators" / "yolo26_human_two_class.pt"
DEFAULT_FFMPEG = Path("ffmpeg")
DEFAULT_FFPROBE = Path("ffprobe")
SCHEMA_VERSION = 2
FILENAME_RANGE_RE = re.compile(r"_(?P<start>\d{2}-\d{2}-\d{2}-\d{3})_(?P<end>\d{2}-\d{2}-\d{2}-\d{3})$")


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    duration_sec: float
    fps: float
    codec: str


@dataclass(frozen=True)
class FramePacket:
    index: int
    timestamp_sec: float
    frame: np.ndarray


@dataclass(frozen=True)
class TimeRange:
    start_sec: float
    end_sec: float

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


@dataclass
class Observation:
    timestamp_sec: float
    piglet_count: int = 0
    band_present: bool = False
    band_iou: float | None = None
    contact_count: int = 0
    activity_score: float | None = None
    activity_ok: bool = False
    sow_detected: bool = True
    error: str = ""
    span_ratio: float | None = None
    radial_alignment_ratio: float | None = None
    appearance_motion_ratio: float | None = None
    mask_xor_ratio: float | None = None
    centroid_speed_p75_body_per_sec: float | None = None
    active_contact_count: int = 0
    brightness_shift: float | None = None
    adaptive_pixel_threshold: float | None = None


@dataclass(frozen=True)
class WindowEvidence:
    timestamp_sec: float
    band_presence_ratio: float
    stable_band_ratio: float
    contact_ratio: float
    activity_ratio: float
    qualifies: bool
    start_sec: float = 0.0
    end_sec: float = 0.0
    sample_count: int = 0
    raw_activity_ratio: float = 0.0
    low_activity_window: bool = False
    appearance_motion_p75: float | None = None
    mask_xor_p75: float | None = None
    centroid_speed_p75: float | None = None


@dataclass(frozen=True)
class BandCandidate:
    band: np.ndarray
    contact_piglets: tuple[InstanceMask, ...]
    contact_count: int
    span_ratio: float
    radial_alignment_ratio: float


@dataclass
class TemporalState:
    gray: np.ndarray | None = None
    sow_mask: np.ndarray | None = None

    contact_piglets: list[InstanceMask] = field(default_factory=list)
    contact_union: np.ndarray | None = None
    band: np.ndarray | None = None
    timestamp_sec: float | None = None


class DecodeError(RuntimeError):
    """Raised when the FFmpeg raw-video producer fails."""


class SourceChangedError(RuntimeError):
    """Raised when a source changed after its analysis snapshot was recorded."""


class InsufficientBandError(ValueError):
    """Expected no-band outcome, distinct from an implementation failure."""

    def __init__(
        self,
        reason: str,
        *,
        contact_count: int = 0,
        span_ratio: float | None = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.contact_count = int(contact_count)
        self.span_ratio = span_ratio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input video or directory.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Reports, per-video JSON, and clips.")
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--start-sec", type=float, default=0.0, help="Debug-only analysis start for a single video.")
    parser.add_argument("--max-seconds", type=float, default=None, help="Debug-only duration for a single video.")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true", help="Replace matching analysis outputs.")
    parser.add_argument("--save-features", action="store_true", help="Write one compact per-sample CSV per video.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-on-error", action="store_true")

    parser.add_argument(
        "--weights", type=Path, default=DEFAULT_WEIGHTS, help="Two-class sow + piglet segmentation model."
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64, help="RTX 4090 starting point; use 16 on a 12 GB GPU.")
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=32)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--sow-label", default="sow")
    parser.add_argument("--piglet-label", default="piglet")

    parser.add_argument("--ffmpeg", type=Path, default=DEFAULT_FFMPEG)
    parser.add_argument("--ffprobe", type=Path, default=DEFAULT_FFPROBE)
    parser.add_argument("--hwaccel", choices=("cuda", "none"), default="cuda")
    parser.add_argument("--decode-fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--decode-width", type=int, default=1024)
    parser.add_argument(
        "--geometry-width",
        type=int,
        default=512,
        help="GPU-downsampled mask width used for CPU contact geometry; model inference remains at --imgsz.",
    )
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--decode-queue", type=int, default=96)

    parser.add_argument("--reference-width", type=float, default=1920.0)
    parser.add_argument("--min-sow-area", type=int, default=1000)
    parser.add_argument("--min-piglet-area", type=int, default=80)
    parser.add_argument("--max-piglet-sow-overlap", type=float, default=0.60)
    parser.add_argument("--min-contact-piglets", type=int, default=4)
    parser.add_argument("--min-contact-span-ratio", type=float, default=0.25)
    parser.add_argument("--min-contact-area-px", type=int, default=20)
    parser.add_argument("--min-contact-area-ratio", type=float, default=0.01)
    parser.add_argument("--max-contact-dist", type=float, default=48.0)
    parser.add_argument("--contact-margin", type=float, default=12.0)
    parser.add_argument("--cluster-gap", type=float, default=360.0)
    parser.add_argument("--head-projection-quantile", type=float, default=0.90)
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

    parser.add_argument("--motion-width", type=int, default=384)
    parser.add_argument("--activity-pixel-delta", type=float, default=8.0)
    parser.add_argument("--activity-noise-mad", type=float, default=3.0)
    parser.add_argument("--active-piglet-changed-ratio", type=float, default=0.015)
    parser.add_argument("--low-activity-mask-xor", type=float, default=0.10)
    parser.add_argument("--low-activity-centroid-speed", type=float, default=0.035)
    parser.add_argument("--low-activity-appearance", type=float, default=0.015)

    parser.add_argument("--window-sec", type=float, default=12.0)
    parser.add_argument("--min-band-presence-ratio", type=float, default=0.75)
    parser.add_argument("--min-stable-band-ratio", type=float, default=0.60)
    parser.add_argument("--min-band-iou", type=float, default=0.45)
    parser.add_argument("--min-contact-ratio", type=float, default=0.70)
    parser.add_argument("--min-activity-ratio", type=float, default=0.30)
    parser.add_argument("--max-gap-sec", type=float, default=10.0)
    parser.add_argument("--min-keep-sec", type=float, default=15.0)
    parser.add_argument("--pad-sec", type=float, default=5.0)
    parser.add_argument("--whole-keep-ratio", type=float, default=0.90)
    parser.add_argument("--min-analysis-sec", type=float, default=8.0)
    parser.add_argument("--min-observation-ratio", type=float, default=0.75)
    parser.add_argument("--min-sow-detection-ratio", type=float, default=0.50)
    parser.add_argument(
        "--min-piglet-detection-ratio",
        type=float,
        default=0.50,
        help="Below this ratio, low-count evidence is sent to review instead of rejected.",
    )
    parser.add_argument("--review-margin", type=float, default=0.05)

    parser.add_argument("--emit-clips", action="store_true", help="Write only retained ranges for partial videos.")
    parser.add_argument("--clips-dir", type=Path, default=None)
    parser.add_argument("--trim-mode", choices=("nvenc", "copy", "x265"), default="nvenc")
    parser.add_argument("--overwrite-clips", action="store_true")
    parser.add_argument("--reject-action", choices=("none", "move", "delete"), default="none")
    parser.add_argument("--reject-dir", type=Path, default=None)
    parser.add_argument(
        "--confirm-delete",
        default="",
        help="Required literal DELETE_REJECTED when --reject-action delete is used.",
    )
    return parser.parse_args()


def _within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def validate_args(args: argparse.Namespace) -> None:
    if args.sample_fps <= 0 or args.batch_size <= 0 or args.decode_queue <= 0:
        raise ValueError("--sample-fps, --batch-size, and --decode-queue must be positive.")
    if args.decode_width <= 0 or args.geometry_width <= 0 or args.motion_width <= 0 or args.imgsz <= 0:
        raise ValueError("Image sizes must be positive.")
    if args.start_sec < 0 or (args.max_seconds is not None and args.max_seconds <= 0):
        raise ValueError("--start-sec must be non-negative and --max-seconds must be positive.")
    if args.min_contact_piglets < 4:
        raise ValueError("--min-contact-piglets must be at least 4 for this filtering task.")
    for name in (
        "max_piglet_sow_overlap",
        "min_contact_span_ratio",
        "min_contact_area_ratio",
        "active_piglet_changed_ratio",
        "low_activity_mask_xor",
        "low_activity_centroid_speed",
        "low_activity_appearance",
        "min_band_presence_ratio",
        "min_stable_band_ratio",
        "min_band_iou",
        "min_contact_ratio",
        "min_activity_ratio",
        "whole_keep_ratio",
        "min_observation_ratio",
        "min_sow_detection_ratio",
        "min_piglet_detection_ratio",
        "review_margin",
    ):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be between 0 and 1.")
    if args.window_sec <= 0 or args.min_keep_sec <= 0 or args.min_analysis_sec <= 0:
        raise ValueError("Temporal durations must be positive.")
    if args.max_gap_sec < 0 or args.pad_sec < 0:
        raise ValueError("Gap and padding durations cannot be negative.")
    for executable in (args.ffmpeg, args.ffprobe):
        if not executable.is_file():
            raise FileNotFoundError(f"Executable not found: {executable}")
    if args.reject_action == "move":
        if args.reject_dir is None:
            raise ValueError("--reject-dir is required for --reject-action move.")
        if args.input.is_dir() and _within(args.reject_dir, args.input):
            raise ValueError("--reject-dir must be outside the input tree.")
    if args.reject_action == "delete" and args.confirm_delete != "DELETE_REJECTED":
        raise ValueError("Deletion requires --confirm-delete DELETE_REJECTED.")
    mutating = args.emit_clips or args.reject_action != "none"
    if mutating and (args.start_sec > 0 or args.max_seconds is not None):
        raise ValueError("Partial/debug analysis cannot emit clips, move, or delete source videos.")


def iter_videos(input_path: Path, recursive: bool = True, excluded_roots: Sequence[Path] = ()) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in VIDEO_EXTS:
            raise ValueError(f"Unsupported video: {input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input does not exist: {input_path}")
    pattern = "**/*" if recursive else "*"
    excluded = [path.resolve() for path in excluded_roots]
    return sorted(
        path
        for path in input_path.glob(pattern)
        if path.is_file() and path.suffix.lower() in VIDEO_EXTS and not any(_within(path, root) for root in excluded)
    )


def _clock_to_seconds(value: str) -> float:
    hours, minutes, seconds, milliseconds = (int(item) for item in value.split("-"))
    return float(hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0)


def parse_filename_duration(path_or_name: str | Path) -> float | None:
    match = FILENAME_RANGE_RE.search(Path(path_or_name).stem)
    if match is None:
        return None
    start = _clock_to_seconds(match.group("start"))
    end = _clock_to_seconds(match.group("end"))
    return end - start if end > start else None


def _rate(value: str | None) -> float:
    if not value or value in {"0/0", "N/A"}:
        return 0.0
    numerator, separator, denominator = value.partition("/")
    if separator:
        denominator_value = float(denominator)
        return float(numerator) / denominator_value if denominator_value else 0.0
    return float(value)


def probe_video(video_path: Path, ffprobe: Path) -> VideoInfo:
    command = [
        str(ffprobe),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,codec_name,avg_frame_rate:format=duration",
        "-of",
        "json",
        str(video_path),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True, encoding="utf-8")
    payload = json.loads(completed.stdout)
    stream = payload["streams"][0]
    duration = float(payload.get("format", {}).get("duration") or 0.0)
    if duration <= 0:
        duration = float(parse_filename_duration(video_path) or 0.0)
    info = VideoInfo(
        width=int(stream["width"]),
        height=int(stream["height"]),
        duration_sec=duration,
        fps=_rate(stream.get("avg_frame_rate")),
        codec=str(stream.get("codec_name") or ""),
    )
    if min(info.width, info.height, info.duration_sec) <= 0:
        raise ValueError(f"Invalid video metadata: {video_path}")
    return info


def decoded_size(source_width: int, source_height: int, requested_width: int) -> tuple[int, int]:
    width = min(int(source_width), int(requested_width))
    width -= width % 2
    height = max(2, int(round(source_height * width / source_width)))
    height -= height % 2
    return width, height


def build_decode_command(
    ffmpeg: str | Path,
    video: str | Path,
    width: int,
    height: int,
    sample_fps: float,
    hwaccel: str,
    start_sec: float = 0.0,
    max_seconds: float | None = None,
) -> list[str]:
    command = [str(ffmpeg), "-hide_banner", "-loglevel", "error"]
    if hwaccel == "cuda":
        command.extend(("-hwaccel", "cuda"))
    if start_sec > 0:
        command.extend(("-ss", f"{start_sec:.6f}"))
    command.extend(("-i", str(video)))
    if max_seconds is not None:
        command.extend(("-t", f"{max_seconds:.6f}"))
    command.extend(
        (
            "-an",
            "-vf",
            f"fps={sample_fps:.9g}:start_time=0,scale={width}:{height}:flags=bilinear",
            "-pix_fmt",
            "bgr24",
            "-f",
            "rawvideo",
            "pipe:1",
        )
    )
    return command


class FFmpegFrameReader(Iterable[FramePacket]):
    """Read fixed-rate BGR frames while FFmpeg decodes ahead in a background thread."""

    _END = object()

    def __init__(
        self,
        command: Sequence[str],
        width: int,
        height: int,
        sample_fps: float,
        start_sec: float,
        queue_size: int,
    ) -> None:
        self.command = list(command)
        self.width = width
        self.height = height
        self.sample_fps = sample_fps
        self.start_sec = start_sec
        self.items: queue.Queue[FramePacket | BaseException | object] = queue.Queue(maxsize=max(2, queue_size))
        self.stop_event = threading.Event()
        self.process: subprocess.Popen[bytes] | None = None
        self.thread: threading.Thread | None = None
        self.stderr_thread: threading.Thread | None = None
        self.stderr_buffer = bytearray()
        self.stderr = ""

    def _put(self, item: FramePacket | BaseException | object) -> None:
        while not self.stop_event.is_set():
            try:
                self.items.put(item, timeout=0.1)
                return
            except queue.Full:
                continue

    @staticmethod
    def _read_exact(stream: Any, length: int) -> bytes:
        chunks: list[bytes] = []
        remaining = length
        while remaining:
            chunk = stream.read(remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _drain_stderr(self) -> None:
        assert self.process is not None and self.process.stderr is not None
        while True:
            chunk = self.process.stderr.read(8192)
            if not chunk:
                break
            self.stderr_buffer.extend(chunk)
            if len(self.stderr_buffer) > 262_144:
                del self.stderr_buffer[:-262_144]

    def _produce(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        frame_bytes = self.width * self.height * 3
        index = 0
        try:
            while not self.stop_event.is_set():
                raw = self._read_exact(self.process.stdout, frame_bytes)
                if not raw:
                    break
                if len(raw) != frame_bytes:
                    raise DecodeError(f"Short FFmpeg frame: expected {frame_bytes} bytes, got {len(raw)}")
                frame = np.frombuffer(raw, dtype=np.uint8).reshape(self.height, self.width, 3)
                self._put(FramePacket(index, self.start_sec + index / self.sample_fps, frame))
                index += 1
            return_code = self.process.wait()
            if self.stderr_thread is not None:
                self.stderr_thread.join(timeout=5.0)
            self.stderr = bytes(self.stderr_buffer).decode("utf-8", errors="replace").strip()
            if return_code != 0 and not self.stop_event.is_set():
                raise DecodeError(f"FFmpeg exited with {return_code}: {self.stderr[-2000:]}")
        except BaseException as exc:
            self._put(exc)
        finally:
            self._put(self._END)

    def __enter__(self) -> FFmpegFrameReader:
        self.process = subprocess.Popen(
            self.command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        self.stderr_thread = threading.Thread(target=self._drain_stderr, name="nursing-ffmpeg-stderr", daemon=True)
        self.stderr_thread.start()
        self.thread = threading.Thread(target=self._produce, name="nursing-ffmpeg-reader", daemon=True)
        self.thread.start()
        return self

    def __iter__(self) -> Iterator[FramePacket]:
        if self.thread is None:
            raise RuntimeError("FFmpegFrameReader must be used as a context manager.")
        while True:
            item = self.items.get()
            if item is self._END:
                break
            if isinstance(item, BaseException):
                raise item
            assert isinstance(item, FramePacket)
            yield item

    def close(self) -> None:
        self.stop_event.set()
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
        if self.thread is not None:
            self.thread.join(timeout=5.0)
        if self.process is not None and self.process.poll() is None:
            self.process.kill()
            self.process.wait(timeout=5.0)
        if self.stderr_thread is not None:
            self.stderr_thread.join(timeout=5.0)

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def _scaled_args(args: argparse.Namespace, width: int) -> SimpleNamespace:
    scale = width / float(args.reference_width)
    area_scale = scale * scale
    return SimpleNamespace(
        sow_label=args.sow_label,
        piglet_label=args.piglet_label,
        min_sow_area=max(1, int(round(args.min_sow_area * area_scale))),
        min_piglet_area=max(1, int(round(args.min_piglet_area * area_scale))),
        max_contact_dist=args.max_contact_dist * scale,
        contact_margin=args.contact_margin * scale,
        cluster_gap=args.cluster_gap * scale,
        head_projection_quantile=args.head_projection_quantile,
        arc_padding=args.arc_padding * scale,
        band_width=max(1, int(round(args.band_width * scale))),
        band_outer_width=args.band_outer_width * scale,
        band_inner_width=args.band_inner_width * scale,
        centerline_thickness=max(1, int(round(args.centerline_thickness * scale))),
        contour_smooth_window=args.contour_smooth_window * scale,
        turn_window=max(3.0, args.turn_window * scale),
        max_turn_angle=args.max_turn_angle,
        min_turn_radius=args.min_turn_radius * scale,
        max_axis_deviation=args.max_axis_deviation,
        max_contour_bridge=args.max_contour_bridge * scale,
        max_endpoint_extension=args.max_endpoint_extension * scale,
        min_contact_area_px=max(1, int(round(args.min_contact_area_px * area_scale))),
    )


def _select_instances(
    instances: list[InstanceMask],
    geometry: SimpleNamespace,
    max_sow_overlap: float,
) -> tuple[InstanceMask, list[InstanceMask]]:
    sows = [
        item
        for item in instances
        if item.label.casefold() == geometry.sow_label.casefold() and item.area >= geometry.min_sow_area
    ]
    if not sows:
        raise InsufficientBandError("no_sow")
    sow = max(sows, key=lambda item: item.area)
    raw_piglets = [
        item
        for item in instances
        if item.label.casefold() == geometry.piglet_label.casefold() and item.area >= geometry.min_piglet_area
    ]
    piglets, _, _ = partition_piglets_by_sow_overlap(raw_piglets, sow, max_sow_overlap)
    return sow, piglets


def result_instances_fast_batch(
    results: Sequence[Any],
    mask_threshold: float,
    target_shape: tuple[int, int],
) -> list[list[InstanceMask]]:
    """Threshold/downsample masks on the GPU before the CPU transfer.

    Ultralytics' generic helper copies float masks at the decoded frame size.
    Screening geometry needs far less resolution, so copying uint8 masks on a
    smaller grid removes most mask-transfer and full-image NumPy work without
    reducing the neural network input size.
    """
    import torch
    import torch.nn.functional as functional

    mask_chunks: list[Any] = []
    class_chunks: list[Any] = []
    confidence_chunks: list[Any] = []
    counts: list[int] = []
    for result in results:
        if result.masks is None or result.boxes is None:
            counts.append(0)
            continue
        masks = result.masks.data.detach()
        if tuple(masks.shape[-2:]) != tuple(target_shape):
            masks = functional.interpolate(
                masks[:, None].float(),
                size=target_shape,
                mode="bilinear",
                align_corners=False,
            )[:, 0]
        count = int(len(masks))
        counts.append(count)
        if count:
            mask_chunks.append((masks >= mask_threshold).byte())
            class_chunks.append(result.boxes.cls.detach())
            confidence_chunks.append(result.boxes.conf.detach())
    if mask_chunks:
        binary_masks = torch.cat(mask_chunks, dim=0).cpu().numpy()
        classes = torch.cat(class_chunks, dim=0).to("cpu").numpy().astype(int)
        confidences = torch.cat(confidence_chunks, dim=0).to("cpu").numpy()
    else:
        binary_masks = np.empty((0, *target_shape), dtype=np.uint8)
        classes = np.empty(0, dtype=int)
        confidences = np.empty(0, dtype=float)

    output: list[list[InstanceMask]] = []
    offset = 0
    for result, count in zip(results, counts):
        names = result.names or {}
        instances: list[InstanceMask] = []
        for local_index in range(count):
            flat_index = offset + local_index
            class_id = int(classes[flat_index])
            label = names.get(class_id, str(class_id)) if isinstance(names, dict) else str(class_id)
            instances.append(
                InstanceMask(
                    index=local_index,
                    class_id=class_id,
                    label=str(label),
                    confidence=float(confidences[flat_index]),
                    mask=binary_masks[flat_index].astype(bool, copy=False),
                )
            )
        output.append(instances)
        offset += count
    return output


def _mask_axis(mask: np.ndarray) -> tuple[np.ndarray, float]:
    ys, xs = np.nonzero(mask)
    if len(xs) < 3:
        return np.asarray([1.0, 0.0], dtype=np.float32), 1.0
    points = np.column_stack((xs, ys)).astype(np.float32)
    centered = points - points.mean(axis=0)
    covariance = centered.T @ centered / max(1, len(points) - 1)
    values, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, int(np.argmax(values))].astype(np.float32)
    if axis[int(np.argmax(np.abs(axis)))] < 0:
        axis = -axis
    length = max(1.0, float(np.ptp(centered @ axis)))
    return axis, length


def _radial_alignment_ratio(
    sow_contour: np.ndarray,
    contacts: Sequence[Any],
    piglets_by_index: dict[int, InstanceMask],
) -> float:
    aligned: list[float] = []
    count = len(sow_contour)
    radius = max(2, min(15, count // 50))
    for contact in contacts:
        piglet = piglets_by_index.get(contact.piglet_index)
        if piglet is None:
            continue
        piglet_axis, _ = _mask_axis(piglet.mask)
        before = sow_contour[(contact.contour_index - radius) % count]
        after = sow_contour[(contact.contour_index + radius) % count]
        tangent = after.astype(np.float32) - before.astype(np.float32)
        norm = float(np.linalg.norm(tangent))
        if norm <= 1e-6:
            continue
        normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float32) / norm
        aligned.append(abs(float(np.dot(piglet_axis, normal))))
    return float(np.mean(np.asarray(aligned) >= 0.60)) if aligned else 0.0


def strict_nursing_band(
    sow: InstanceMask,
    piglets: list[InstanceMask],
    image_shape: tuple[int, int],
    args: argparse.Namespace,
    geometry: SimpleNamespace,
) -> BandCandidate:
    """Build a band only from one coherent cluster with at least four contacts."""
    if not piglets:
        raise InsufficientBandError("no_piglets")
    contour = largest_contour(sow.mask)
    contour_s, total_s = contour_arclength(contour)
    contacts = build_contacts(
        sow.mask,
        contour,
        contour_s,
        piglets,
        max_contact_dist=geometry.max_contact_dist,
        contact_margin=geometry.contact_margin,
        head_projection_quantile=geometry.head_projection_quantile,
    )
    clusters = cluster_contacts(contacts, total_s, geometry.cluster_gap)
    if not clusters:
        raise InsufficientBandError("no_contact_cluster")
    selected = max(
        clusters,
        key=lambda group: (len(group), -minimal_arc([item.contour_s for item in group], total_s)[2]),
    )
    if len(selected) < args.min_contact_piglets:
        raise InsufficientBandError("contact_cluster_below_minimum", contact_count=len(selected))

    arc_start, arc_end, _ = minimal_arc([item.contour_s for item in selected], total_s)
    start_s = (arc_start - geometry.arc_padding) % total_s
    end_s = (arc_end + geometry.arc_padding) % total_s
    arc_points = interval_points(contour, contour_s, start_s, end_s)
    arc_points, _ = clean_contour_arc(arc_points, list(selected), sow.mask, geometry)
    band = build_asymmetric_band(
        image_shape,
        arc_points,
        sow.mask,
        outer_width=geometry.band_outer_width,
        inner_width=geometry.band_inner_width,
        centerline_thickness=geometry.centerline_thickness,
    )

    piglets_by_index = {item.index: item for item in piglets}
    retained_contacts: list[Any] = []
    retained_piglets: list[InstanceMask] = []
    band_bool = band.astype(bool)
    for contact in selected:
        piglet = piglets_by_index.get(contact.piglet_index)
        if piglet is None:
            continue
        overlap = int(np.logical_and(mask_for_shape(piglet.mask, image_shape), band_bool).sum())
        required = max(geometry.min_contact_area_px, int(math.ceil(args.min_contact_area_ratio * piglet.area)))
        if overlap >= required:
            retained_contacts.append(contact)
            retained_piglets.append(piglet)
    if len(retained_contacts) < args.min_contact_piglets:
        raise InsufficientBandError("band_overlap_below_minimum", contact_count=len(retained_contacts))

    sow_axis, sow_length = _mask_axis(sow.mask)
    contact_points = np.asarray([item.contact_center for item in retained_contacts], dtype=np.float32)
    projections = contact_points @ sow_axis
    span = float(np.quantile(projections, 0.90) - np.quantile(projections, 0.10)) if len(projections) > 1 else 0.0
    span_ratio = span / sow_length
    if span_ratio < args.min_contact_span_ratio:
        raise InsufficientBandError(
            "contact_span_too_short",
            contact_count=len(retained_contacts),
            span_ratio=float(span_ratio),
        )

    return BandCandidate(
        band=band,
        contact_piglets=tuple(retained_piglets),
        contact_count=len(retained_piglets),
        span_ratio=float(span_ratio),
        radial_alignment_ratio=_radial_alignment_ratio(contour, retained_contacts, piglets_by_index),
    )


def binary_mask_iou(first: np.ndarray | None, second: np.ndarray | None) -> float | None:
    if first is None or second is None:
        return None
    first_bool = first.astype(bool, copy=False)
    second_bool = mask_for_shape(second, first_bool.shape)
    union = int(np.logical_or(first_bool, second_bool).sum())
    if union == 0:
        return 1.0
    return float(np.logical_and(first_bool, second_bool).sum() / union)


def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    return cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)


def _union_masks(instances: Sequence[InstanceMask], shape: tuple[int, int]) -> np.ndarray:
    union = np.zeros(shape, dtype=bool)
    for item in instances:
        union |= _resize_mask(item.mask, shape)
    return union


def _instance_activity(
    frame: np.ndarray,
    sow: InstanceMask,
    candidate: BandCandidate,
    state: TemporalState,
    timestamp_sec: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    height, width = frame.shape[:2]
    motion_width = min(width, args.motion_width)
    motion_height = max(2, int(round(height * motion_width / width)))
    gray = cv2.cvtColor(
        cv2.resize(frame, (motion_width, motion_height), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY
    )
    sow_mask = _resize_mask(sow.mask, gray.shape)
    contact_union = _union_masks(candidate.contact_piglets, gray.shape)
    result: dict[str, Any] = {
        "gray": gray,
        "sow_mask": sow_mask,
        "contact_union": contact_union,
        "appearance_motion_ratio": None,
        "mask_xor_ratio": None,
        "centroid_speed_p75_body_per_sec": None,
        "active_contact_count": 0,
        "brightness_shift": None,
        "adaptive_pixel_threshold": None,
        "activity_score": None,
        "activity_ok": False,
    }
    if state.gray is None or state.timestamp_sec is None:
        return result
    dt = timestamp_sec - state.timestamp_sec
    expected = 1.0 / args.sample_fps
    if dt <= 0 or dt > expected * 1.75 or state.gray.shape != gray.shape:
        return result

    signed = gray.astype(np.int16) - state.gray.astype(np.int16)
    reference = np.logical_and(sow_mask, state.sow_mask) if state.sow_mask is not None else sow_mask
    if int(reference.sum()) < 256:
        reference = np.ones_like(gray, dtype=bool)
    reference_values = signed[reference].astype(np.float32)
    brightness_shift = float(np.median(reference_values))
    residual_reference = np.abs(reference_values - brightness_shift)
    noise_median = float(np.median(residual_reference))
    noise_mad = float(np.median(np.abs(residual_reference - noise_median)))
    pixel_threshold = max(
        float(args.activity_pixel_delta),
        noise_median + float(args.activity_noise_mad) * 1.4826 * noise_mad,
    )
    residual = np.abs(signed.astype(np.float32) - brightness_shift)

    per_piglet_motion: list[float] = []
    for piglet in candidate.contact_piglets:
        mask = _resize_mask(piglet.mask, gray.shape)
        values = residual[mask]
        if len(values) < 8:
            continue
        per_piglet_motion.append(float(np.mean(values >= pixel_threshold)))
    appearance = float(np.median(per_piglet_motion)) if per_piglet_motion else 0.0
    active_count = sum(value >= args.active_piglet_changed_ratio for value in per_piglet_motion)

    speeds: list[float] = []
    matched_mask_xors: list[float] = []
    if state.contact_piglets:
        distance_limit = 120.0 * width / args.reference_width
        pairs = match_piglet_pairs_one_to_one(
            list(candidate.contact_piglets),
            state.contact_piglets,
            max_centroid_distance=max(8.0, distance_limit),
            max_area_ratio=3.0,
        )
        for current_index, previous_index in pairs:
            current = candidate.contact_piglets[current_index]
            previous = state.contact_piglets[previous_index]
            cy, cx = np.nonzero(current.mask)
            py, px = np.nonzero(previous.mask)
            if not len(cx) or not len(px):
                continue
            distance = math.hypot(float(cx.mean() - px.mean()), float(cy.mean() - py.mean()))
            body_scale = math.sqrt(max(1.0, (current.area + previous.area) / 2.0))
            speeds.append(distance / body_scale / dt)
            current_mask = current.mask.astype(bool, copy=False)
            previous_mask = mask_for_shape(previous.mask, current_mask.shape)
            union = int(np.logical_or(current_mask, previous_mask).sum())
            matched_mask_xors.append(float(np.logical_xor(current_mask, previous_mask).sum() / union) if union else 0.0)
    centroid_speed = float(np.percentile(speeds, 75)) if speeds else 0.0
    # Use matched-instance mask change.  A raw union XOR is badly inflated when
    # the strict contact cluster gains or loses one sleeping piglet.
    mask_xor = float(np.percentile(matched_mask_xors, 75)) if matched_mask_xors else None
    mask_xor_value = float(mask_xor or 0.0)
    low_activity = (
        mask_xor_value < args.low_activity_mask_xor
        and centroid_speed < args.low_activity_centroid_speed
        and appearance < args.low_activity_appearance
    )
    normalized = max(
        mask_xor_value / max(1e-9, args.low_activity_mask_xor),
        centroid_speed / max(1e-9, args.low_activity_centroid_speed),
        appearance / max(1e-9, args.low_activity_appearance),
    )
    result.update(
        {
            "appearance_motion_ratio": appearance,
            "mask_xor_ratio": mask_xor,
            "centroid_speed_p75_body_per_sec": centroid_speed,
            "active_contact_count": active_count,
            "brightness_shift": brightness_shift,
            "adaptive_pixel_threshold": pixel_threshold,
            "activity_score": float(normalized),
            "activity_ok": not low_activity,
        }
    )
    return result


def _predict_batch_resilient(
    model: Any, frames: list[np.ndarray], args: argparse.Namespace
) -> list[Any | BaseException]:
    if not frames:
        return []
    try:
        results = model.predict(
            source=frames,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device or None,
            max_det=args.max_det,
            half=args.half and args.device != "cpu",
            retina_masks=False,
            batch=len(frames),
            verbose=False,
        )
        if len(results) != len(frames):
            raise RuntimeError(f"Expected {len(frames)} predictions, received {len(results)}")
        return list(results)
    except Exception as exc:
        if len(frames) == 1:
            return [exc]
        midpoint = len(frames) // 2
        return _predict_batch_resilient(model, frames[:midpoint], args) + _predict_batch_resilient(
            model, frames[midpoint:], args
        )


def analyze_packets(
    model: Any,
    packets: list[FramePacket],
    results: list[Any | BaseException],
    state: TemporalState,
    args: argparse.Namespace,
    geometry: SimpleNamespace,
    geometry_shape: tuple[int, int],
) -> list[Observation]:
    observations: list[Observation] = []
    valid_predictions = [prediction for prediction in results if not isinstance(prediction, BaseException)]
    converted = iter(result_instances_fast_batch(valid_predictions, args.mask_threshold, geometry_shape))
    for packet, prediction in zip(packets, results):
        observation = Observation(timestamp_sec=packet.timestamp_sec, sow_detected=False)
        candidate: BandCandidate | None = None
        sow: InstanceMask | None = None
        piglets: list[InstanceMask] = []
        if isinstance(prediction, BaseException):
            observation.error = f"predict:{type(prediction).__name__}:{prediction}"
            observations.append(observation)
            state.gray = None
            state.band = None
            state.contact_piglets.clear()
            state.contact_union = None
            state.timestamp_sec = packet.timestamp_sec
            continue
        try:
            instances = next(converted)
            sow, piglets = _select_instances(instances, geometry, args.max_piglet_sow_overlap)
            observation.sow_detected = True
            observation.piglet_count = len(piglets)
            candidate = strict_nursing_band(sow, piglets, geometry_shape, args, geometry)
            observation.band_present = True
            observation.contact_count = candidate.contact_count
            observation.span_ratio = candidate.span_ratio
            observation.radial_alignment_ratio = candidate.radial_alignment_ratio
            if state.band is not None and state.timestamp_sec is not None:
                gap = packet.timestamp_sec - state.timestamp_sec
                if gap <= (1.0 / args.sample_fps) * 1.75:
                    observation.band_iou = binary_mask_iou(candidate.band, state.band)
            activity = _instance_activity(packet.frame, sow, candidate, state, packet.timestamp_sec, args)
            for key, value in activity.items():
                if key not in {"gray", "sow_mask", "contact_union"}:
                    setattr(observation, key, value)
            state.gray = activity["gray"]
            state.sow_mask = activity["sow_mask"]
            state.contact_union = activity["contact_union"]
            state.contact_piglets = list(candidate.contact_piglets)
            state.band = candidate.band
        except InsufficientBandError as exc:
            reason = exc.reason
            observation.error = "" if reason != "no_sow" else reason
            observation.sow_detected = sow is not None
            observation.piglet_count = len(piglets)
            observation.contact_count = exc.contact_count
            observation.span_ratio = exc.span_ratio
            state.gray = None
            state.sow_mask = None
            state.contact_union = None
            state.contact_piglets.clear()
            state.band = None
        except Exception as exc:
            observation.error = f"geometry:{type(exc).__name__}:{exc}"
            state.gray = None
            state.sow_mask = None
            state.contact_union = None
            state.contact_piglets.clear()
            state.band = None
        state.timestamp_sec = packet.timestamp_sec
        observations.append(observation)
    return observations


def rolling_gate_metrics(
    observations: Sequence[Observation],
    window_sec: float,
    min_band_presence_ratio: float,
    min_stable_band_ratio: float,
    min_band_iou: float,
    min_contact_ratio: float,
    min_activity_ratio: float,
    low_activity_mask_xor: float = 0.10,
    low_activity_centroid_speed: float = 0.035,
    low_activity_appearance: float = 0.015,
) -> list[WindowEvidence]:
    if not observations:
        return []
    timestamps = np.asarray([item.timestamp_sec for item in observations], dtype=np.float64)
    evidence: list[WindowEvidence] = []
    left = 0
    right = 0
    radius = window_sec / 2.0
    for index, timestamp in enumerate(timestamps):
        while left < len(observations) and timestamps[left] < timestamp - radius - 1e-9:
            left += 1
        right = max(right, index)
        while right + 1 < len(observations) and timestamps[right + 1] <= timestamp + radius + 1e-9:
            right += 1
        group = observations[left : right + 1]
        total = len(group)
        band_presence = sum(item.band_present for item in group) / total
        comparisons = [item.band_iou for item in group if item.band_iou is not None]
        stable_ratio = (
            sum(float(value) >= min_band_iou for value in comparisons) / len(comparisons) if comparisons else 0.0
        )
        contact_ratio = sum(item.contact_count >= 4 for item in group) / total
        # Keep the gates diagnostically independent: low count must not force the
        # reported activity ratio to zero when activity evidence is available.
        activity_items = [item for item in group if item.activity_score is not None]
        raw_activity_ratio = (
            sum(item.activity_ok for item in activity_items) / len(activity_items) if activity_items else 0.0
        )
        appearance_p75 = _percentile((item.appearance_motion_ratio for item in group), 75)
        mask_xor_p75 = _percentile((item.mask_xor_ratio for item in group), 75)
        centroid_speed_p75 = _percentile((item.centroid_speed_p75_body_per_sec for item in group), 75)
        low_activity_window = bool(
            appearance_p75 is not None
            and mask_xor_p75 is not None
            and centroid_speed_p75 is not None
            and appearance_p75 < low_activity_appearance
            and mask_xor_p75 < low_activity_mask_xor
            and centroid_speed_p75 < low_activity_centroid_speed
        )
        activity_ratio = 0.0 if low_activity_window else raw_activity_ratio
        qualifies = (
            total >= 2
            and band_presence >= min_band_presence_ratio
            and stable_ratio >= min_stable_band_ratio
            and contact_ratio >= min_contact_ratio
            and activity_ratio >= min_activity_ratio
        )
        evidence.append(
            WindowEvidence(
                timestamp_sec=float(timestamp),
                start_sec=float(timestamps[left]),
                end_sec=float(timestamps[right]),
                sample_count=total,
                raw_activity_ratio=float(raw_activity_ratio),
                low_activity_window=low_activity_window,
                appearance_motion_p75=appearance_p75,
                mask_xor_p75=mask_xor_p75,
                centroid_speed_p75=centroid_speed_p75,
                band_presence_ratio=float(band_presence),
                stable_band_ratio=float(stable_ratio),
                contact_ratio=float(contact_ratio),
                activity_ratio=float(activity_ratio),
                qualifies=qualifies,
            )
        )
    return evidence


def merge_time_ranges(ranges: Sequence[TimeRange], max_gap_sec: float) -> list[TimeRange]:
    merged: list[TimeRange] = []
    for item in sorted(ranges, key=lambda value: value.start_sec):
        if item.end_sec <= item.start_sec:
            continue
        if merged and item.start_sec - merged[-1].end_sec <= max_gap_sec + 1e-9:
            merged[-1] = TimeRange(merged[-1].start_sec, max(merged[-1].end_sec, item.end_sec))
        else:
            merged.append(item)
    return merged


def qualifying_intervals(
    observations: Sequence[Observation],
    evidence: Sequence[WindowEvidence],
    duration_sec: float,
    sample_fps: float,
    max_gap_sec: float,
    min_keep_sec: float,
    pad_sec: float,
) -> list[TimeRange]:
    if not observations or not evidence:
        return []
    half_step = 0.5 / sample_fps
    raw = [
        TimeRange(max(0.0, item.timestamp_sec - half_step), min(duration_sec, item.timestamp_sec + half_step))
        for item in evidence
        if item.qualifies
    ]
    closed = merge_time_ranges(raw, max_gap_sec=max_gap_sec)
    long_enough = [item for item in closed if item.duration_sec + 1e-9 >= min_keep_sec]
    padded = [
        TimeRange(max(0.0, item.start_sec - pad_sec), min(duration_sec, item.end_sec + pad_sec)) for item in long_enough
    ]
    return merge_time_ranges(padded, max_gap_sec=0.0)


def _percentile(values: Iterable[float | None], quantile: float) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.percentile(finite, quantile)) if finite else None


def summarize_decision(
    observations: Sequence[Observation],
    evidence: Sequence[WindowEvidence],
    intervals: Sequence[TimeRange],
    duration_sec: float,
    analyzed_duration_sec: float,
    args: argparse.Namespace,
) -> tuple[str, str, dict[str, Any]]:
    expected = max(1, int(math.floor(analyzed_duration_sec * args.sample_fps + 1e-6)))
    observation_ratio = min(1.0, len(observations) / expected)
    valid_ratio = sum(not item.error for item in observations) / len(observations) if observations else 0.0
    sow_ratio = sum(item.sow_detected for item in observations) / len(observations) if observations else 0.0
    kept_duration = sum(item.duration_sec for item in intervals)
    kept_ratio = kept_duration / duration_sec if duration_sec > 0 else 0.0
    maxima = {
        "band_presence_ratio": max((item.band_presence_ratio for item in evidence), default=0.0),
        "stable_band_ratio": max((item.stable_band_ratio for item in evidence), default=0.0),
        "contact_ratio": max((item.contact_ratio for item in evidence), default=0.0),
        "activity_ratio": max((item.activity_ratio for item in evidence), default=0.0),
    }
    summary = {
        "samples": len(observations),
        "expected_samples": expected,
        "observation_ratio": observation_ratio,
        "valid_ratio": valid_ratio,
        "sow_detection_ratio": sow_ratio,
        "piglet_detection_ratio": (
            sum(item.piglet_count > 0 for item in observations) / len(observations) if observations else 0.0
        ),
        "band_presence_ratio": (
            sum(item.band_present for item in observations) / len(observations) if observations else 0.0
        ),
        "contact_count_median": _percentile((float(item.contact_count) for item in observations), 50),
        "contact_count_p75": _percentile((float(item.contact_count) for item in observations), 75),
        "appearance_motion_p75": _percentile((item.appearance_motion_ratio for item in observations), 75),
        "mask_xor_p75": _percentile((item.mask_xor_ratio for item in observations), 75),
        "centroid_speed_p75": _percentile((item.centroid_speed_p75_body_per_sec for item in observations), 75),
        "radial_alignment_median": _percentile((item.radial_alignment_ratio for item in observations), 50),
        "kept_duration_sec": kept_duration,
        "kept_ratio": kept_ratio,
        "best_window": maxima,
        "error_samples": sum(bool(item.error) for item in observations),
    }
    partial = args.start_sec > 0 or args.max_seconds is not None
    if intervals:
        decision = "keep" if kept_ratio >= args.whole_keep_ratio and not partial else "trim"
        return decision, "three_gates_passed", summary
    if partial:
        return "review", "partial_debug_analysis", summary
    if (
        analyzed_duration_sec < args.min_analysis_sec
        or observation_ratio < args.min_observation_ratio
        or valid_ratio < args.min_observation_ratio
        or sow_ratio < args.min_sow_detection_ratio
    ):
        return "review", "insufficient_reliable_observations", summary

    piglet_detection_ratio = float(summary["piglet_detection_ratio"])
    if piglet_detection_ratio == 0.0:
        # A complete piglet-model miss is not safe evidence for destructive filtering.
        return "review", "no_piglet_detections", summary
    if piglet_detection_ratio < args.min_piglet_detection_ratio:
        # Sparse detections cannot distinguish a genuinely small litter from a
        # detector failure, so never use them as automatic reject evidence.
        return "review", "insufficient_piglet_detections", summary

    thresholds = {
        "band_presence_ratio": args.min_band_presence_ratio,
        "stable_band_ratio": args.min_stable_band_ratio,
        "contact_ratio": args.min_contact_ratio,
        "activity_ratio": args.min_activity_ratio,
    }
    all_margins = {key: maxima[key] - threshold for key, threshold in thresholds.items()}
    if min(all_margins.values()) >= -args.review_margin:
        return "review", "borderline_three_gate_result", summary
    contact_median = summary.get("contact_count_median")
    if contact_median is not None and float(contact_median) <= 3.0 and maxima["contact_ratio"] < args.min_contact_ratio:
        return "reject", "contact_count_not_sustained_above_3", summary
    failed_key = min(all_margins, key=all_margins.get)
    reasons = {
        "band_presence_ratio": "unstable_or_missing_contact_band",
        "stable_band_ratio": "unstable_contact_band",
        "contact_ratio": "contact_count_not_sustained_above_3",
        "activity_ratio": "contact_piglets_low_activity",
    }
    return "reject", reasons[failed_key], summary


def _analysis_config(args: argparse.Namespace) -> dict[str, Any]:
    keys = (
        "weights",
        "imgsz",
        "half",
        "conf",
        "iou",
        "max_det",
        "mask_threshold",
        "sow_label",
        "piglet_label",
        "hwaccel",
        "decode_width",
        "geometry_width",
        "motion_width",
        "sample_fps",
        "reference_width",
        "min_sow_area",
        "min_piglet_area",
        "max_piglet_sow_overlap",
        "min_contact_piglets",
        "min_contact_span_ratio",
        "min_contact_area_px",
        "min_contact_area_ratio",
        "max_contact_dist",
        "contact_margin",
        "cluster_gap",
        "head_projection_quantile",
        "arc_padding",
        "band_width",
        "band_outer_width",
        "band_inner_width",
        "centerline_thickness",
        "contour_smooth_window",
        "turn_window",
        "max_turn_angle",
        "min_turn_radius",
        "max_axis_deviation",
        "max_contour_bridge",
        "max_endpoint_extension",
        "activity_pixel_delta",
        "activity_noise_mad",
        "active_piglet_changed_ratio",
        "low_activity_mask_xor",
        "low_activity_centroid_speed",
        "low_activity_appearance",
        "window_sec",
        "min_band_presence_ratio",
        "min_stable_band_ratio",
        "min_band_iou",
        "min_contact_ratio",
        "min_activity_ratio",
        "max_gap_sec",
        "min_keep_sec",
        "pad_sec",
        "whole_keep_ratio",
        "min_analysis_sec",
        "min_observation_ratio",
        "min_sow_detection_ratio",
        "min_piglet_detection_ratio",
        "review_margin",
        "start_sec",
        "max_seconds",
    )
    return {
        key: str(getattr(args, key)) if isinstance(getattr(args, key), Path) else getattr(args, key) for key in keys
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def analysis_signature(args: argparse.Namespace) -> str:
    stat = args.weights.stat()
    config = _analysis_config(args)
    # The model content fingerprint below is authoritative.  Its absolute
    # deployment path and filesystem mtime are not analysis semantics and can
    # change when an otherwise identical server/container is recreated.
    config.pop("weights", None)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "config": config,
        "weights_size": stat.st_size,
        "weights_sha256": _sha256(args.weights),
        "filter_code_sha256": _sha256(Path(__file__).resolve()),
        "band_code_sha256": _sha256(ROOT / "scripts" / "extract_nursing_band.py"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def relative_video_path(video_path: Path, input_path: Path) -> Path:
    return video_path.relative_to(input_path) if input_path.is_dir() else Path(video_path.name)


def result_path_for(video_path: Path, input_path: Path, output: Path) -> Path:
    relative = relative_video_path(video_path, input_path)
    return output / "results" / relative.parent / f"{relative.name}.nursing_filter.json"


def feature_path_for(video_path: Path, input_path: Path, output: Path) -> Path:
    relative = relative_video_path(video_path, input_path)
    return output / "features" / relative.parent / f"{relative.name}.nursing_filter.csv"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def write_features(path: Path, observations: Sequence[Observation], evidence: Sequence[WindowEvidence]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    evidence_by_timestamp = {round(item.timestamp_sec, 6): item for item in evidence}
    observation_fields = (
        list(asdict(observations[0]).keys()) if observations else list(Observation.__dataclass_fields__)
    )
    evidence_fields = [
        "window_band_presence_ratio",
        "window_stable_band_ratio",
        "window_contact_ratio",
        "window_activity_ratio",
        "window_raw_activity_ratio",
        "window_low_activity",
        "window_appearance_motion_p75",
        "window_mask_xor_p75",
        "window_centroid_speed_p75",
        "window_qualifies",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=[*observation_fields, *evidence_fields])
        writer.writeheader()
        for observation in observations:
            row = asdict(observation)
            item = evidence_by_timestamp.get(round(observation.timestamp_sec, 6))
            if item is not None:
                row.update(
                    {
                        "window_band_presence_ratio": item.band_presence_ratio,
                        "window_stable_band_ratio": item.stable_band_ratio,
                        "window_contact_ratio": item.contact_ratio,
                        "window_activity_ratio": item.activity_ratio,
                        "window_raw_activity_ratio": item.raw_activity_ratio,
                        "window_low_activity": item.low_activity_window,
                        "window_appearance_motion_p75": item.appearance_motion_p75,
                        "window_mask_xor_p75": item.mask_xor_p75,
                        "window_centroid_speed_p75": item.centroid_speed_p75,
                        "window_qualifies": item.qualifies,
                    }
                )
            writer.writerow(row)


def analyze_video(
    model: Any, video_path: Path, input_path: Path, args: argparse.Namespace, signature: str
) -> dict[str, Any]:
    started = time.perf_counter()
    source_stat = video_path.stat()
    info = probe_video(video_path, args.ffprobe)
    width, height = decoded_size(info.width, info.height, args.decode_width)
    geometry_width, geometry_height = decoded_size(width, height, min(width, args.geometry_width))
    geometry_shape = (geometry_height, geometry_width)
    available = max(0.0, info.duration_sec - args.start_sec)
    analyzed_duration = min(available, args.max_seconds) if args.max_seconds is not None else available
    if analyzed_duration <= 0:
        raise ValueError(f"Analysis start is beyond video duration: {video_path}")
    geometry = _scaled_args(args, geometry_width)
    state = TemporalState()
    observations: list[Observation] = []
    hwaccels = [args.hwaccel]
    if args.hwaccel == "cuda" and args.decode_fallback:
        hwaccels.append("none")
    decode_backend = hwaccels[0]
    last_error: BaseException | None = None
    for attempt, hwaccel in enumerate(hwaccels):
        command = build_decode_command(
            args.ffmpeg,
            video_path,
            width,
            height,
            args.sample_fps,
            hwaccel,
            start_sec=args.start_sec,
            max_seconds=analyzed_duration,
        )
        observations.clear()
        state = TemporalState()
        try:
            with FFmpegFrameReader(
                command,
                width,
                height,
                args.sample_fps,
                args.start_sec,
                args.decode_queue,
            ) as reader:
                batch: list[FramePacket] = []
                for packet in reader:
                    batch.append(packet)
                    if len(batch) < args.batch_size:
                        continue
                    results = _predict_batch_resilient(model, [item.frame for item in batch], args)
                    observations.extend(analyze_packets(model, batch, results, state, args, geometry, geometry_shape))
                    batch.clear()
                if batch:
                    results = _predict_batch_resilient(model, [item.frame for item in batch], args)
                    observations.extend(analyze_packets(model, batch, results, state, args, geometry, geometry_shape))
            decode_backend = hwaccel
            last_error = None
            break
        except DecodeError as exc:
            last_error = exc
            if attempt + 1 < len(hwaccels):
                print(f"[WARN] CUDA decode failed for {video_path.name}; retrying on CPU: {exc}")
                continue
            raise
    if last_error is not None:
        raise last_error

    evidence = rolling_gate_metrics(
        observations,
        window_sec=args.window_sec,
        min_band_presence_ratio=args.min_band_presence_ratio,
        min_stable_band_ratio=args.min_stable_band_ratio,
        min_band_iou=args.min_band_iou,
        min_contact_ratio=args.min_contact_ratio,
        min_activity_ratio=args.min_activity_ratio,
        low_activity_mask_xor=args.low_activity_mask_xor,
        low_activity_centroid_speed=args.low_activity_centroid_speed,
        low_activity_appearance=args.low_activity_appearance,
    )
    intervals = qualifying_intervals(
        observations,
        evidence,
        duration_sec=info.duration_sec,
        sample_fps=args.sample_fps,
        max_gap_sec=args.max_gap_sec,
        min_keep_sec=args.min_keep_sec,
        pad_sec=args.pad_sec,
    )
    decision, reason, summary = summarize_decision(
        observations, evidence, intervals, info.duration_sec, analyzed_duration, args
    )
    elapsed = time.perf_counter() - started
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "analysis_signature": signature,
        "source_video": str(video_path),
        "relative_video": str(relative_video_path(video_path, input_path)),
        "source_size": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "video": asdict(info),
        "analysis": {
            "start_sec": args.start_sec,
            "duration_sec": analyzed_duration,
            "complete": args.start_sec == 0 and analyzed_duration >= info.duration_sec - 0.1,
            "decode_width": width,
            "decode_height": height,
            "geometry_width": geometry_width,
            "geometry_height": geometry_height,
            "decode_backend": decode_backend,
            "elapsed_sec": elapsed,
            "source_realtime_factor": analyzed_duration / elapsed if elapsed > 0 else None,
        },
        "decision": decision,
        "reason": reason,
        "keep_intervals": [asdict(item) | {"duration_sec": item.duration_sec} for item in intervals],
        "summary": summary,
        "configuration": _analysis_config(args),
    }
    if args.save_features:
        features_path = feature_path_for(video_path, input_path, args.output)
        write_features(features_path, observations, evidence)
        payload["features_csv"] = str(features_path)
    return payload


def build_trim_command(
    ffmpeg: Path,
    source: Path,
    destination: Path,
    interval: TimeRange,
    mode: str,
    overwrite: bool,
) -> list[str]:
    command = [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-ss",
        f"{interval.start_sec:.6f}",
        "-i",
        str(source),
        "-t",
        f"{interval.duration_sec:.6f}",
        "-map",
        "0:v:0",
        "-an",
    ]
    if mode == "copy":
        command.extend(("-c:v", "copy", "-avoid_negative_ts", "make_zero"))
    elif mode == "nvenc":
        command.extend(("-c:v", "hevc_nvenc", "-preset", "p4", "-tune", "hq", "-rc", "vbr", "-cq", "21"))
    else:
        command.extend(("-c:v", "libx265", "-preset", "medium", "-crf", "21"))
    command.extend(("-movflags", "+faststart", str(destination)))
    return command


def ensure_source_unchanged(payload: dict[str, Any], video_path: Path) -> None:
    """Refuse a mutating action unless the source still matches the analyzed snapshot."""
    try:
        expected_size = int(payload["source_size"])
        expected_mtime_ns = int(payload["source_mtime_ns"])
        current = video_path.stat()
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise SourceChangedError(f"Cannot verify analyzed source {video_path}: {exc}") from exc
    if current.st_size != expected_size or current.st_mtime_ns != expected_mtime_ns:
        raise SourceChangedError(
            "Source changed after analysis: "
            f"{video_path} (size {expected_size}->{current.st_size}, "
            f"mtime_ns {expected_mtime_ns}->{current.st_mtime_ns})"
        )


def _clip_is_complete(path: Path, ffprobe: Path, expected_duration_sec: float) -> bool:
    """Check that a published or temporary clip has valid video metadata and duration."""
    try:
        if not path.is_file() or path.stat().st_size <= 0:
            return False
        info = probe_video(path, ffprobe)
    except (OSError, ValueError, IndexError, KeyError, subprocess.SubprocessError):
        return False
    tolerance = max(1.0, min(3.0, expected_duration_sec * 0.05))
    return (
        info.width > 0
        and info.height > 0
        and info.duration_sec > 0
        and abs(info.duration_sec - expected_duration_sec) <= tolerance
    )


def emit_retained_clips(
    payload: dict[str, Any], video_path: Path, input_path: Path, args: argparse.Namespace
) -> list[str]:
    if not args.emit_clips or payload["decision"] != "trim":
        return []
    ensure_source_unchanged(payload, video_path)
    recorded = {str(Path(path).resolve()) for path in payload.get("emitted_clips", [])}
    clips_root = args.clips_dir or (args.output / "clips")
    relative = relative_video_path(video_path, input_path)
    destination_dir = clips_root / relative.parent
    destination_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    for index, item in enumerate(payload["keep_intervals"], start=1):
        interval = TimeRange(float(item["start_sec"]), float(item["end_sec"]))
        destination = (
            destination_dir
            / f"{relative.stem}_keep_{index:03d}_{interval.start_sec:010.3f}_{interval.end_sec:010.3f}.mp4"
        )
        temporary = destination.with_name(f"{destination.stem}.part{destination.suffix}")
        if not args.overwrite_clips and _clip_is_complete(destination, args.ffprobe, interval.duration_sec):
            temporary.unlink(missing_ok=True)
            outputs.append(str(destination))
            continue
        if destination.exists() and not args.overwrite_clips and str(destination.resolve()) not in recorded:
            raise FileExistsError(f"Invalid clip exists; pass --overwrite-clips: {destination}")
        subprocess.run(
            build_trim_command(args.ffmpeg, video_path, temporary, interval, args.trim_mode, overwrite=True),
            check=True,
        )
        if not _clip_is_complete(temporary, args.ffprobe, interval.duration_sec):
            raise DecodeError(f"Emitted clip failed validation: {temporary}")
        os.replace(temporary, destination)
        outputs.append(str(destination))
    return outputs


def apply_reject_action(
    payload: dict[str, Any], video_path: Path, input_path: Path, args: argparse.Namespace
) -> str | None:
    if payload["decision"] != "reject" or args.reject_action == "none":
        return None
    ensure_source_unchanged(payload, video_path)
    if args.reject_action == "delete":
        video_path.unlink()
        return "deleted"
    assert args.reject_dir is not None
    destination = args.reject_dir / relative_video_path(video_path, input_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Quarantine target already exists: {destination}")
    shutil.move(str(video_path), str(destination))
    return f"moved:{destination}"


def _resume_payload(result_path: Path, video_path: Path, signature: str) -> dict[str, Any] | None:
    if not result_path.is_file():
        return None
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        stat = video_path.stat()
        if (
            payload.get("analysis_signature") == signature
            and payload.get("source_size") == stat.st_size
            and payload.get("source_mtime_ns") == stat.st_mtime_ns
        ):
            return payload
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None
    return None


def write_manifest(path: Path, payloads: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "source_video",
        "relative_video",
        "decision",
        "reason",
        "duration_sec",
        "kept_duration_sec",
        "kept_ratio",
        "keep_intervals_json",
        "observation_ratio",
        "sow_detection_ratio",
        "piglet_detection_ratio",
        "band_presence_ratio",
        "contact_count_median",
        "appearance_motion_p75",
        "mask_xor_p75",
        "centroid_speed_p75",
        "elapsed_sec",
        "source_realtime_factor",
        "error",
    )
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for payload in payloads:
            summary = payload.get("summary", {})
            analysis = payload.get("analysis", {})
            video = payload.get("video", {})
            writer.writerow(
                {
                    "source_video": payload.get("source_video", ""),
                    "relative_video": payload.get("relative_video", ""),
                    "decision": payload.get("decision", "error"),
                    "reason": payload.get("reason", ""),
                    "duration_sec": video.get("duration_sec", ""),
                    "kept_duration_sec": summary.get("kept_duration_sec", ""),
                    "kept_ratio": summary.get("kept_ratio", ""),
                    "keep_intervals_json": json.dumps(payload.get("keep_intervals", []), ensure_ascii=False),
                    "observation_ratio": summary.get("observation_ratio", ""),
                    "sow_detection_ratio": summary.get("sow_detection_ratio", ""),
                    "piglet_detection_ratio": summary.get("piglet_detection_ratio", ""),
                    "band_presence_ratio": summary.get("band_presence_ratio", ""),
                    "contact_count_median": summary.get("contact_count_median", ""),
                    "appearance_motion_p75": summary.get("appearance_motion_p75", ""),
                    "mask_xor_p75": summary.get("mask_xor_p75", ""),
                    "centroid_speed_p75": summary.get("centroid_speed_p75", ""),
                    "elapsed_sec": analysis.get("elapsed_sec", ""),
                    "source_realtime_factor": analysis.get("source_realtime_factor", ""),
                    "error": payload.get("error", ""),
                }
            )


def _configure_torch() -> None:
    try:
        import torch

        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def main() -> int:
    args = parse_args()
    validate_args(args)
    excluded = [args.output]
    if args.clips_dir is not None:
        excluded.append(args.clips_dir)
    if args.reject_dir is not None:
        excluded.append(args.reject_dir)
    videos = iter_videos(args.input, args.recursive, excluded)
    if args.max_videos is not None:
        videos = videos[: max(0, args.max_videos)]
    if not videos:
        raise FileNotFoundError(f"No videos selected from {args.input}")
    print(f"Input: {args.input}")
    print(f"Videos: {len(videos)}")
    print(f"Weights: {args.weights}")
    print(f"Sampling: {args.sample_fps:g} FPS, decode width {args.decode_width}, batch {args.batch_size}")
    print(f"Source action: {args.reject_action}; emit partial clips: {args.emit_clips}")
    if args.dry_run:
        for path in videos[:20]:
            print(path)
        if len(videos) > 20:
            print(f"... and {len(videos) - 20} more")
        return 0
    if not args.weights.is_file():
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    from ultralytics import YOLO

    _configure_torch()
    args.output.mkdir(parents=True, exist_ok=True)
    signature = analysis_signature(args)
    model = YOLO(str(args.weights), task="segment")
    labels = {str(value).casefold() for value in model.names.values()}
    required_labels = {args.sow_label.casefold(), args.piglet_label.casefold()}
    if not required_labels.issubset(labels):
        raise ValueError(f"Model classes {model.names} do not contain {sorted(required_labels)}")

    payloads: list[dict[str, Any]] = []
    failures = 0
    for index, video_path in enumerate(videos, start=1):
        result_path = result_path_for(video_path, args.input, args.output)
        try:
            payload = None
            if args.resume and not args.overwrite:
                payload = _resume_payload(result_path, video_path, signature)
            if payload is None:
                print(f"[INFO] {index}/{len(videos)} analyze {video_path}")
                payload = analyze_video(model, video_path, args.input, args, signature)
                _atomic_json(result_path, payload)
            else:
                print(f"[INFO] {index}/{len(videos)} resume {video_path.name}: {payload['decision']}")

            clips = emit_retained_clips(payload, video_path, args.input, args)
            if clips:
                payload["emitted_clips"] = clips
            reject_action = apply_reject_action(payload, video_path, args.input, args)
            if reject_action:
                payload["source_action"] = reject_action
            if clips or reject_action:
                _atomic_json(result_path, payload)
            payloads.append(payload)
            elapsed = payload.get("analysis", {}).get("elapsed_sec", 0.0)
            speed = payload.get("analysis", {}).get("source_realtime_factor")
            speed_text = f"{float(speed):.1f}x" if speed is not None else "n/a"
            print(
                f"[RESULT] {payload['decision']} {payload['reason']} ranges={len(payload['keep_intervals'])} "
                f"elapsed={float(elapsed):.1f}s speed={speed_text}"
            )
        except SourceChangedError as exc:
            failures += 1
            if payload is None:
                raise
            payload["decision"] = "review"
            payload["reason"] = "source_changed_after_analysis"
            payload["source_action"] = "skipped_source_changed"
            payload["error"] = str(exc)
            _atomic_json(result_path, payload)
            payloads.append(payload)
            print(f"[WARN] {video_path}: {exc}; no source action was taken", file=sys.stderr)
            if args.fail_on_error:
                raise
        except Exception as exc:
            failures += 1
            error_payload = {
                "schema_version": SCHEMA_VERSION,
                "analysis_signature": signature,
                "source_video": str(video_path),
                "relative_video": str(relative_video_path(video_path, args.input)),
                "decision": "error",
                "reason": "processing_failed",
                "keep_intervals": [],
                "error": f"{type(exc).__name__}: {exc}",
            }
            payloads.append(error_payload)
            print(f"[ERROR] {video_path}: {error_payload['error']}", file=sys.stderr)
            if args.fail_on_error:
                raise

    manifest_path = args.output / "nursing_filter_manifest.csv"
    write_manifest(manifest_path, payloads)
    counts: dict[str, int] = {}
    for payload in payloads:
        decision = str(payload.get("decision", "error"))
        counts[decision] = counts.get(decision, 0) + 1
    print(f"Done: {counts}; failures={failures}; manifest={manifest_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
