#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
import math
import os
import shutil
import sys
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import extract_nursing_tracked_normal_flow as tracked  # noqa: E402
from scripts import filter_nursing_clips_fast as fast  # noqa: E402


DEFAULT_OUTPUT = ROOT / "runs" / "nursing_tracked_normal_flow_screened"
SCREEN_SCHEMA_VERSION = 1
EPSILON = 1e-6


def _discover_executable(name: str, preferred: Path) -> Path:
    """Prefer the configured binary, then resolve a cross-platform PATH entry."""

    if preferred.is_file():
        return preferred
    resolved = shutil.which(name)
    return Path(resolved) if resolved else Path(name)


@dataclass(frozen=True, slots=True)
class TimeScope:
    start_sec: float
    end_sec: float

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


@dataclass(frozen=True, slots=True)
class ScreenResult:
    """Normalized fail-open geometry decision."""

    decision: str
    reason: str
    intervals: tuple[TimeScope, ...]
    reliable: bool


@dataclass(frozen=True, slots=True)
class ExtractionJob:
    video_path: Path
    start_sec: float
    end_sec: float
    candidate_index: int
    part_index: int
    output_suffix: str

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


@dataclass(slots=True)
class VideoPlan:
    video_path: Path
    scope: TimeScope
    jobs: list[ExtractionJob]
    sidecar_path: Path
    payload: dict[str, Any]
    compatible_resume: bool = False


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _coerce_scope(value: Any) -> TimeScope | None:
    if isinstance(value, TimeScope):
        return value
    if isinstance(value, dict):
        start = _finite_float(value.get("start_sec"))
        end = _finite_float(value.get("end_sec"))
    elif isinstance(value, (tuple, list)) and len(value) >= 2:
        start = _finite_float(value[0])
        end = _finite_float(value[1])
    else:
        start = _finite_float(getattr(value, "start_sec", None))
        end = _finite_float(getattr(value, "end_sec", None))
    if start is None or end is None or end <= start + EPSILON:
        return None
    return TimeScope(start, end)


def requested_scope(
    duration_sec: float,
    start_sec: float = 0.0,
    max_seconds: float | None = None,
) -> TimeScope:
    """Return the requested absolute source-time interval."""
    duration = _finite_float(duration_sec)
    start = _finite_float(start_sec)
    limit = None if max_seconds is None else _finite_float(max_seconds)
    if duration is None or duration <= 0:
        raise ValueError("Video duration must be positive.")
    if start is None or start < 0:
        raise ValueError("Requested start must be non-negative.")
    if limit is not None and limit <= 0:
        raise ValueError("Requested duration must be positive.")
    end = duration if limit is None else min(duration, start + limit)
    start = min(start, duration)
    if end <= start + EPSILON:
        raise ValueError("Requested start is beyond the video duration.")
    return TimeScope(start, end)


def normalize_intervals(
    intervals: Iterable[Any],
    scope: TimeScope | tuple[float, float],
    padding_sec: float = 0.0,
    merge_gap_sec: float = 0.0,
) -> list[TimeScope]:
    """Clip, pad, sort, and merge intervals inside ``scope``."""
    bounded = _coerce_scope(scope)
    if bounded is None:
        raise ValueError("Invalid normalization scope.")
    if padding_sec < 0 or merge_gap_sec < 0:
        raise ValueError("Padding and merge gap must be non-negative.")
    clipped: list[TimeScope] = []
    for value in intervals:
        interval = _coerce_scope(value)
        if interval is None:
            continue
        start = max(bounded.start_sec, interval.start_sec - padding_sec)
        end = min(bounded.end_sec, interval.end_sec + padding_sec)
        if end > start + EPSILON:
            clipped.append(TimeScope(start, end))
    merged: list[TimeScope] = []
    for interval in sorted(clipped, key=lambda item: (item.start_sec, item.end_sec)):
        if merged and interval.start_sec <= merged[-1].end_sec + merge_gap_sec + EPSILON:
            merged[-1] = TimeScope(
                merged[-1].start_sec,
                max(merged[-1].end_sec, interval.end_sec),
            )
        else:
            merged.append(interval)
    return merged


def screen_result_from_payload(payload: Any, scope: TimeScope) -> ScreenResult:
    """Map the fast-screen payload to a conservative extraction decision."""
    full = (scope,)
    if not isinstance(payload, dict):
        return ScreenResult("review", "malformed_screen_payload", full, False)
    decision = str(payload.get("decision", "")).casefold()
    reason = str(payload.get("reason", "") or decision or "unknown")
    if decision == "keep":
        return ScreenResult("keep", reason, full, True)
    if decision == "trim":
        raw = payload.get("keep_intervals")
        if not isinstance(raw, (list, tuple)) or any(_coerce_scope(item) is None for item in raw):
            return ScreenResult("review", "trim_with_malformed_intervals", full, False)
        intervals = normalize_intervals(raw, scope)
        if intervals:
            return ScreenResult("trim", reason, tuple(intervals), True)
        return ScreenResult("review", "trim_without_valid_intervals", full, False)
    if decision == "reject":
        raw = payload.get("keep_intervals")
        if not isinstance(raw, (list, tuple)) or any(_coerce_scope(item) is None for item in raw):
            return ScreenResult("review", "malformed_reject_intervals", full, False)
        intervals = normalize_intervals(raw, scope)
        if intervals:
            return ScreenResult("review", "reject_with_keep_intervals", full, False)
        return ScreenResult("reject", reason, (), True)
    if decision in {"review", "error"}:
        return ScreenResult(decision, reason, full, False)
    return ScreenResult("review", "unknown_screen_decision", full, False)


def _subtract_intervals_local(
    intervals: Sequence[TimeScope],
    vetoes: Sequence[TimeScope],
) -> list[TimeScope]:
    remaining = list(intervals)
    for veto in vetoes:
        next_remaining: list[TimeScope] = []
        for interval in remaining:
            if veto.end_sec <= interval.start_sec + EPSILON or veto.start_sec >= interval.end_sec - EPSILON:
                next_remaining.append(interval)
                continue
            if veto.start_sec > interval.start_sec + EPSILON:
                next_remaining.append(TimeScope(interval.start_sec, min(veto.start_sec, interval.end_sec)))
            if veto.end_sec < interval.end_sec - EPSILON:
                next_remaining.append(TimeScope(max(veto.end_sec, interval.start_sec), interval.end_sec))
        remaining = next_remaining
    return remaining


def job_output_suffix(
    candidate_index: int,
    part_index: int,
    start_sec: float,
    end_sec: float,
    configuration_tag: str = "",
) -> str:
    """Create a stable collision-free suffix from absolute source time."""

    def base36(value: int) -> str:
        alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
        if value == 0:
            return "0"
        digits: list[str] = []
        while value:
            value, remainder = divmod(value, 36)
            digits.append(alphabet[remainder])
        return "".join(reversed(digits))

    start_ms = max(0, int(round(start_sec * 1000.0)))
    end_ms = max(start_ms + 1, int(round(end_sec * 1000.0)))
    # UniformerV2 names plus channel subdirectories are already long on
    # Windows.  Base-36 times and unpadded indices keep even partial-file names
    # safely below MAX_PATH while the sidecar retains human-readable seconds.
    safe_tag = "".join(character for character in configuration_tag if character.isalnum())[:12]
    tag = f"_{safe_tag}" if safe_tag else ""
    return (
        f"__s{base36(candidate_index)}p{base36(part_index)}{tag}_"
        f"{base36(start_ms)}_{base36(end_ms)}"
    )


def _job_configuration_tag(signature: str, source_size: int, source_mtime_ns: int) -> str:
    """Bind resumable output names to both configuration and source revision."""

    identity = f"{signature}:{int(source_size)}:{int(source_mtime_ns)}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:8]


def plan_extraction_jobs(
    video_path: Path,
    screen_result: ScreenResult,
    scope: TimeScope,
    sparse_veto_intervals: Sequence[Any] = (),
    *,
    padding_sec: float = 10.0,
    chunk_seconds: float = 1200.0,
    chunk_overlap_sec: float = 10.0,
    configuration_tag: str = "",
) -> list[ExtractionJob]:
    """Apply reliable vetoes, pad retained ranges once, then make overlapping jobs."""
    if chunk_seconds < 0:
        raise ValueError("Screened extraction chunk length cannot be negative.")
    if chunk_overlap_sec < 0 or (chunk_seconds > 0 and chunk_overlap_sec >= chunk_seconds):
        raise ValueError("Chunk overlap must be non-negative and shorter than a chunk.")
    if chunk_seconds == 0 and chunk_overlap_sec != 0:
        raise ValueError("Chunk overlap must be zero when chunking is disabled.")
    candidates = normalize_intervals(screen_result.intervals, scope)
    vetoes = normalize_intervals(sparse_veto_intervals, scope)
    candidates = normalize_intervals(
        _subtract_intervals_local(candidates, vetoes),
        scope,
        padding_sec=padding_sec,
    )
    jobs: list[ExtractionJob] = []
    for candidate_index, candidate in enumerate(candidates, start=1):
        if chunk_seconds == 0:
            jobs.append(
                ExtractionJob(
                    Path(video_path),
                    candidate.start_sec,
                    candidate.end_sec,
                    candidate_index,
                    1,
                    job_output_suffix(
                        candidate_index,
                        1,
                        candidate.start_sec,
                        candidate.end_sec,
                        configuration_tag,
                    ),
                )
            )
            continue
        start = candidate.start_sec
        part_index = 1
        while start < candidate.end_sec - EPSILON:
            end = min(candidate.end_sec, start + chunk_seconds)
            jobs.append(
                ExtractionJob(
                    Path(video_path),
                    start,
                    end,
                    candidate_index,
                    part_index,
                    job_output_suffix(
                        candidate_index,
                        part_index,
                        start,
                        end,
                        configuration_tag,
                    ),
                )
            )
            if end >= candidate.end_sec - EPSILON:
                break
            start = end - chunk_overlap_sec
            part_index += 1
    return jobs


def _screen_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--screen-weights", type=Path, default=fast.DEFAULT_WEIGHTS)
    parser.add_argument(
        "--screen-ffmpeg",
        type=Path,
        default=_discover_executable("ffmpeg", fast.DEFAULT_FFMPEG),
    )
    parser.add_argument(
        "--screen-ffprobe",
        type=Path,
        default=_discover_executable("ffprobe", fast.DEFAULT_FFPROBE),
    )
    parser.add_argument("--screen-device", default=None)
    parser.add_argument("--screen-imgsz", type=int, default=1024)
    parser.add_argument("--screen-batch-size", type=int, default=48)
    parser.add_argument("--screen-sample-fps", type=float, default=1.0)
    parser.add_argument("--screen-decode-width", type=int, default=1024)
    parser.add_argument("--screen-geometry-width", type=int, default=512)
    parser.add_argument("--screen-motion-width", type=int, default=384)
    parser.add_argument("--screen-hwaccel", choices=("cuda", "none"), default="cuda")
    parser.add_argument("--screen-decode-fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--screen-min-keep-sec", type=float, default=8.0)
    parser.add_argument("--screen-candidate-padding-sec", type=float, default=15.0)
    parser.add_argument("--screen-chunk-seconds", type=float, default=1200.0)
    parser.add_argument("--screen-chunk-overlap-sec", type=float, default=10.0)
    parser.add_argument("--screen-resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--screen-resume-policy",
        choices=("compatible", "strict"),
        default="compatible",
        help=(
            "Reuse a structurally valid sidecar when the source-relative path and file size match, "
            "even if its absolute path, timestamp, or screening signature differs (default). "
            "Use strict to require exact source/signature metadata."
        ),
    )
    parser.add_argument("--screen-overwrite", action="store_true")
    screen_mode = parser.add_mutually_exclusive_group()
    screen_mode.add_argument(
        "--skip-screen",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run tracked extraction directly without screening models or caches (default: True); "
            "use --no-skip-screen or --screen to enable secondary screening."
        ),
    )
    screen_mode.add_argument(
        "--screen",
        dest="skip_screen",
        action="store_false",
        help="Enable secondary geometry screening before tracked extraction.",
    )
    screen_mode.add_argument(
        "--screen-only",
        action="store_true",
        help="Write screening plans without loading or running the full tracked extractor.",
    )
    parser.add_argument("--screen-sparse-flow", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--screen-sparse-flow-fps", type=float, default=4.0)
    parser.add_argument("--screen-sparse-burst-sec", type=float, default=2.0)
    parser.add_argument("--screen-sparse-period-sec", type=float, default=4.0)
    parser.add_argument("--screen-sparse-width", type=int, default=384)
    parser.add_argument("--screen-sparse-imgsz", type=int, default=640)
    parser.add_argument("--screen-sparse-min-sleep-sec", type=float, default=180.0)
    parser.add_argument(
        "--screen-sparse-analysis-chunk-sec",
        type=float,
        default=3600.0,
        help="Bound sparse-flow evidence memory by analyzing candidates in core chunks.",
    )
    parser.add_argument("--screen-sparse-strong-active-count", type=int, default=3)
    parser.add_argument("--screen-sparse-weak-active-count", type=int, default=2)
    parser.add_argument("--screen-sparse-low-active-count", type=int, default=2)
    parser.add_argument("--screen-sparse-protect-before-sec", type=float, default=90.0)
    parser.add_argument("--screen-sparse-protect-after-sec", type=float, default=60.0)
    parser.add_argument("--screen-sparse-weak-max-run-sec", type=float, default=30.0)
    parser.add_argument("--screen-sparse-weak-protect-sec", type=float, default=30.0)
    parser.add_argument("--screen-sparse-active-p75", type=float, default=0.025)
    parser.add_argument("--screen-sparse-active-moving-fraction", type=float, default=0.08)
    return parser


def parse_args() -> tuple[argparse.Namespace, argparse.Namespace]:
    parser = _screen_parser()
    screen_args, remaining = parser.parse_known_args()
    if screen_args.screen_only:
        screen_args.skip_screen = False
    if "-h" in remaining or "--help" in remaining:
        print("Screened-wrapper options:")
        parser.print_help()
        print("\nTracked extractor options:")
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *remaining]
        args = tracked.parse_args()
    finally:
        sys.argv = original_argv
    if not any(item == "--output" or item.startswith("--output=") for item in remaining):
        args.output = DEFAULT_OUTPUT
    return args, screen_args


def _validate_wrapper_args(args: argparse.Namespace, screen_args: argparse.Namespace) -> None:
    tracked.validate_args(args)
    if screen_args.screen_imgsz <= 0 or screen_args.screen_batch_size <= 0:
        raise ValueError("Screen image size and batch size must be positive.")
    if screen_args.screen_sample_fps <= 0:
        raise ValueError("--screen-sample-fps must be positive.")
    if min(
        screen_args.screen_decode_width,
        screen_args.screen_geometry_width,
        screen_args.screen_motion_width,
        screen_args.screen_sparse_width,
        screen_args.screen_sparse_imgsz,
    ) <= 0:
        raise ValueError("Screen widths must be positive.")
    if screen_args.screen_candidate_padding_sec < 0:
        raise ValueError("--screen-candidate-padding-sec cannot be negative.")
    if screen_args.screen_chunk_seconds < 0:
        raise ValueError("--screen-chunk-seconds cannot be negative.")
    if (
        screen_args.screen_chunk_overlap_sec < 0
        or (
            screen_args.screen_chunk_seconds > 0
            and screen_args.screen_chunk_overlap_sec >= screen_args.screen_chunk_seconds
        )
        or (
            screen_args.screen_chunk_seconds == 0
            and screen_args.screen_chunk_overlap_sec != 0
        )
    ):
        raise ValueError("Screen chunk overlap must be shorter than a chunk.")
    if screen_args.screen_sparse_flow_fps <= 0:
        raise ValueError("--screen-sparse-flow-fps must be positive.")
    if screen_args.screen_sparse_burst_sec <= 0:
        raise ValueError("--screen-sparse-burst-sec must be positive.")
    if screen_args.screen_sparse_period_sec < screen_args.screen_sparse_burst_sec:
        raise ValueError("Sparse-flow period must be at least its burst length.")
    if screen_args.screen_sparse_min_sleep_sec <= 0:
        raise ValueError("--screen-sparse-min-sleep-sec must be positive.")
    if (
        screen_args.screen_sparse_analysis_chunk_sec
        < screen_args.screen_sparse_min_sleep_sec
    ):
        raise ValueError(
            "--screen-sparse-analysis-chunk-sec must be at least "
            "--screen-sparse-min-sleep-sec."
        )
    if not (
        0 <= screen_args.screen_sparse_low_active_count
        <= screen_args.screen_sparse_weak_active_count
        < screen_args.screen_sparse_strong_active_count
    ):
        raise ValueError("Sparse low/weak/strong active counts must be ordered.")
    if min(
        screen_args.screen_sparse_protect_before_sec,
        screen_args.screen_sparse_protect_after_sec,
        screen_args.screen_sparse_weak_max_run_sec,
        screen_args.screen_sparse_weak_protect_sec,
    ) < 0:
        raise ValueError("Sparse-flow protection durations cannot be negative.")
    if screen_args.screen_sparse_active_p75 <= 0:
        raise ValueError("--screen-sparse-active-p75 must be positive.")
    if not 0 <= screen_args.screen_sparse_active_moving_fraction <= 1:
        raise ValueError("--screen-sparse-active-moving-fraction must be in [0, 1].")
    if screen_args.screen_overwrite and screen_args.screen_resume:
        screen_args.screen_resume = False


def _fast_defaults() -> argparse.Namespace:
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0]]
        return fast.parse_args()
    finally:
        sys.argv = original_argv


def _make_fast_args(
    args: argparse.Namespace,
    screen_args: argparse.Namespace,
    output: Path,
) -> argparse.Namespace:
    result = _fast_defaults()
    result.input = args.input
    result.output = output
    result.recursive = args.recursive
    result.max_videos = args.max_videos
    result.start_sec = args.start_sec
    result.max_seconds = args.max_seconds
    result.resume = False
    result.overwrite = False
    result.save_features = False
    result.dry_run = False
    result.fail_on_error = args.fail_on_error
    result.weights = screen_args.screen_weights
    result.ffmpeg = screen_args.screen_ffmpeg
    result.ffprobe = screen_args.screen_ffprobe
    result.device = screen_args.screen_device if screen_args.screen_device is not None else args.device
    result.imgsz = screen_args.screen_imgsz
    result.batch_size = screen_args.screen_batch_size
    result.sample_fps = screen_args.screen_sample_fps
    result.decode_width = screen_args.screen_decode_width
    result.geometry_width = screen_args.screen_geometry_width
    result.motion_width = screen_args.screen_motion_width
    result.hwaccel = screen_args.screen_hwaccel
    result.decode_fallback = screen_args.screen_decode_fallback
    result.min_keep_sec = screen_args.screen_min_keep_sec
    # Geometry is the first gate.  Never reject confirmed milking merely because
    # its motion is subtle; only the calibrated sparse-flow helper may veto it.
    result.min_activity_ratio = 0.0
    result.pad_sec = 0.0
    result.emit_clips = False
    result.clips_dir = None
    result.reject_action = "none"
    result.reject_dir = None
    result.confirm_delete = ""
    return result


def _relative_video(video_path: Path, input_path: Path) -> Path:
    return video_path.relative_to(input_path) if input_path.is_dir() else Path(video_path.name)


def _sidecar_path(video_path: Path, input_path: Path, output: Path) -> Path:
    relative = _relative_video(video_path, input_path)
    return output / "_screen" / relative.parent / f"{relative.name}.screened.json"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_fingerprint(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        return {"missing": True}
    stat = resolved.stat()
    return {
        "size": stat.st_size,
        "sha256": _file_sha256(resolved),
    }


def _wrapper_signature(
    fast_signature: str,
    screen_args: argparse.Namespace,
    extraction_args: argparse.Namespace,
) -> str:
    helper = ROOT / "scripts" / "nursing_sparse_flow_screen.py"
    operational = {
        "input",
        "output",
        "recursive",
        "max_videos",
        "resume",
        "overwrite",
        "fail_on_error",
        "dry_run",
        "start_sec",
        "max_seconds",
        "chunk_seconds",
        "output_suffix",
        # Runtime/performance controls do not change the scientific job plan
        # or the expected output schema, so they must not invalidate a costly
        # screen after a restart or a move to different hardware.
        "device",
        "yolo_batch_size",
        "tracker_batch_size",
        "torch_compile",
        "no_prefetch",
        "opencv_threads",
        "progress_sec",
        # Model identity is represented below by content fingerprints.  Paths
        # are deployment details and commonly change when a server is rebuilt.
        "piglet_weights",
        "sow_weights",
        "tracker_checkpoint",
    }
    extraction_config = {
        key: _jsonable(value)
        for key, value in sorted(vars(extraction_args).items())
        if key not in operational
    }
    extraction_files = (
        ROOT / "scripts" / "extract_nursing_tracked_normal_flow.py",
        ROOT / "scripts" / "extract_nursing_normal_flow_features.py",
        ROOT / "scripts" / "extract_nursing_normal_flow.py",
        ROOT / "scripts" / "extract_nursing_band.py",
        ROOT / "scripts" / "track_yolo26_masks_offline.py",
        ROOT / "ultralytics" / "trackers" / "mask_aware.py",
        ROOT / "ultralytics" / "trackers" / "osnet.py",
    )
    payload = {
        "schema_version": SCREEN_SCHEMA_VERSION,
        "fast_signature": fast_signature,
        "wrapper_sha256": _file_sha256(Path(__file__).resolve()),
        "sparse_helper_sha256": _file_sha256(helper) if helper.is_file() else None,
        "extraction_config": extraction_config,
        "extraction_code_sha256": {
            str(path.relative_to(ROOT)): _file_sha256(path) for path in extraction_files
        },
        "extraction_models": {
            "piglet": _file_fingerprint(extraction_args.piglet_weights),
            "sow": _file_fingerprint(extraction_args.sow_weights),
            "tracker": _file_fingerprint(extraction_args.tracker_checkpoint),
        },
        "candidate_padding_sec": screen_args.screen_candidate_padding_sec,
        "chunk_seconds": screen_args.screen_chunk_seconds,
        "chunk_overlap_sec": screen_args.screen_chunk_overlap_sec,
        "sparse_flow": screen_args.screen_sparse_flow,
        "sparse_flow_fps": screen_args.screen_sparse_flow_fps,
        "sparse_burst_sec": screen_args.screen_sparse_burst_sec,
        "sparse_period_sec": screen_args.screen_sparse_period_sec,
        "sparse_width": screen_args.screen_sparse_width,
        "sparse_imgsz": screen_args.screen_sparse_imgsz,
        "sparse_min_sleep_sec": screen_args.screen_sparse_min_sleep_sec,
        "sparse_analysis_chunk_sec": screen_args.screen_sparse_analysis_chunk_sec,
        "sparse_strong_active_count": screen_args.screen_sparse_strong_active_count,
        "sparse_weak_active_count": screen_args.screen_sparse_weak_active_count,
        "sparse_low_active_count": screen_args.screen_sparse_low_active_count,
        "sparse_protect_before_sec": screen_args.screen_sparse_protect_before_sec,
        "sparse_protect_after_sec": screen_args.screen_sparse_protect_after_sec,
        "sparse_weak_max_run_sec": screen_args.screen_sparse_weak_max_run_sec,
        "sparse_weak_protect_sec": screen_args.screen_sparse_weak_protect_sec,
        "sparse_active_p75": screen_args.screen_sparse_active_p75,
        "sparse_active_moving_fraction": screen_args.screen_sparse_active_moving_fraction,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _resume_payload(
    path: Path,
    video_path: Path,
    signature: str,
    policy: str = "compatible",
    input_path: Path | None = None,
) -> dict[str, Any] | None:
    if policy not in {"compatible", "strict"}:
        raise ValueError(f"unsupported screen resume policy: {policy}")
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        stat = video_path.stat()
        cached_signature = payload.get("screen_signature")
        cached_source = Path(str(payload.get("source_video", "")))
        cached_size = int(payload.get("source_size", -1))
        cached_mtime_ns = int(payload.get("source_mtime_ns", -1))
        exact_source = cached_source.resolve() == video_path.resolve()
        exact_match = (
            cached_signature == signature
            and exact_source
            and cached_size == stat.st_size
            and cached_mtime_ns == stat.st_mtime_ns
        )

        relative_match = False
        if input_path is not None:
            cached_relative = str(payload.get("relative_video", "")).replace("\\", "/")
            current_relative = str(_relative_video(video_path, input_path)).replace("\\", "/")
            relative_match = bool(cached_relative) and cached_relative == current_relative

        compatible_match = (
            policy == "compatible"
            and cached_size == stat.st_size
            and (exact_source or relative_match)
        )
        if (
            (exact_match or compatible_match)
            and isinstance(cached_signature, str)
            and bool(cached_signature)
            and isinstance(payload.get("jobs"), list)
        ):
            return payload
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    return None


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _sparse_work_units(
    candidates: Sequence[TimeScope],
    scope: TimeScope,
    *,
    chunk_sec: float,
    context_sec: float,
    phase_period_sec: float | None = None,
) -> list[tuple[TimeScope, TimeScope]]:
    """Return bounded ``(veto_core, evidence_context)`` sparse-flow units.

    When ``phase_period_sec`` is supplied, every context starts on the same
    absolute sampling grid rooted at ``scope.start_sec``.  Overlapping work
    units then inspect the same burst times near a chunk boundary.
    """

    units: list[tuple[TimeScope, TimeScope]] = []
    for candidate in candidates:
        core_start = candidate.start_sec
        while core_start < candidate.end_sec - EPSILON:
            core_end = min(candidate.end_sec, core_start + chunk_sec)
            core = TimeScope(core_start, core_end)
            analysis_start = max(scope.start_sec, core.start_sec - context_sec)
            analysis_end = min(scope.end_sec, core.end_sec + context_sec)
            if phase_period_sec is not None:
                if phase_period_sec <= 0 or not math.isfinite(phase_period_sec):
                    raise ValueError("Sparse-flow phase period must be positive and finite.")
                start_steps = math.floor(
                    (analysis_start - scope.start_sec) / phase_period_sec + EPSILON
                )
                end_steps = math.ceil(
                    (analysis_end - scope.start_sec) / phase_period_sec - EPSILON
                )
                analysis_start = max(
                    scope.start_sec,
                    scope.start_sec + start_steps * phase_period_sec,
                )
                analysis_end = min(
                    scope.end_sec,
                    scope.start_sec + end_steps * phase_period_sec,
                )
            analysis_scope = TimeScope(
                analysis_start,
                analysis_end,
            )
            units.append((core, analysis_scope))
            core_start = core_end
    return units


def _qualifying_candidate_vetoes(
    raw_vetoes: Sequence[Any],
    candidates: Sequence[TimeScope],
    scope: TimeScope,
    *,
    min_duration_sec: float,
) -> list[TimeScope]:
    """Clip evidence-context vetoes to candidates and reapply minimum length.

    A short geometry candidate must never disappear merely because the
    surrounding context is a long sleeping interval.  Context is only for
    finding activity anchors; the candidate itself must contain a qualifying
    continuous low-activity duration.
    """

    normalized_vetoes = normalize_intervals(raw_vetoes, scope)
    qualified: list[TimeScope] = []
    for candidate in candidates:
        for overlap in normalize_intervals(normalized_vetoes, candidate):
            if overlap.duration_sec + EPSILON >= min_duration_sec:
                qualified.append(overlap)
    return normalize_intervals(qualified, scope)


def _run_sparse_flow(
    model: Any,
    video_path: Path,
    screen_args: argparse.Namespace,
    scope: TimeScope,
    candidates: Sequence[TimeScope],
) -> tuple[list[TimeScope], dict[str, Any]]:
    if not screen_args.screen_sparse_flow or not candidates:
        return [], {"enabled": False, "reliable": False, "veto_intervals": []}
    try:
        from scripts import nursing_sparse_flow_screen as sparse
    except Exception as exc:
        return [], {
            "enabled": True,
            "reliable": False,
            "reason": "helper_unavailable_fail_open",
            "error": f"{type(exc).__name__}: {exc}",
            "veto_intervals": [],
        }

    helper_args = copy.copy(screen_args)
    helper_args.weights = screen_args.screen_weights
    helper_args.device = screen_args.screen_device
    helper_args.ffmpeg = screen_args.screen_ffmpeg
    helper_args.ffprobe = screen_args.screen_ffprobe
    helper_args.hwaccel = screen_args.screen_hwaccel
    helper_args.decode_fallback = screen_args.screen_decode_fallback
    helper_args.imgsz = screen_args.screen_sparse_imgsz
    helper_args.sparse_flow_imgsz = screen_args.screen_sparse_imgsz
    helper_args.half = True
    helper_args.sparse_flow_fps = screen_args.screen_sparse_flow_fps
    helper_args.sparse_flow_burst_sec = screen_args.screen_sparse_burst_sec
    helper_args.sparse_flow_period_sec = screen_args.screen_sparse_period_sec
    helper_args.sparse_flow_width = screen_args.screen_sparse_width
    helper_args.sparse_flow_anchor_batch_size = min(32, screen_args.screen_batch_size)
    helper_args.sparse_flow_active_p75_body_per_sec = screen_args.screen_sparse_active_p75
    helper_args.sparse_flow_active_moving_fraction = (
        screen_args.screen_sparse_active_moving_fraction
    )
    summaries: list[dict[str, Any]] = []
    raw_vetoes: list[Any] = []
    errors: list[str] = []
    reliable_count = 0
    context_sec = max(
        screen_args.screen_sparse_protect_before_sec,
        screen_args.screen_sparse_protect_after_sec,
        screen_args.screen_sparse_weak_protect_sec,
    )
    _, _, _, actual_sampling_period_sec = sparse._sampling_grid(
        screen_args.screen_sparse_flow_fps,
        screen_args.screen_sparse_burst_sec,
        screen_args.screen_sparse_period_sec,
    )
    work_units = _sparse_work_units(
        candidates,
        scope,
        chunk_sec=screen_args.screen_sparse_analysis_chunk_sec,
        context_sec=context_sec,
        phase_period_sec=actual_sampling_period_sec,
    )

    for core, analysis_scope in work_units:
        try:
            raw_result = sparse.analyze_sparse_flow(
                video_path,
                model,
                helper_args,
                (analysis_scope.start_sec, analysis_scope.end_sec),
            )
            error = str(getattr(raw_result, "error", "") or "")
            evidence = tuple(getattr(raw_result, "evidence", ()) or ())
            valid_evidence = [item for item in evidence if bool(getattr(item, "valid", False))]
            metrics = getattr(raw_result, "metrics", {}) or {}
            actual_period_sec = float(
                metrics.get("period_sec", screen_args.screen_sparse_period_sec)
            )
            active_histogram: dict[str, int] = {}
            invalid_reasons: dict[str, int] = {}
            piglet_p75_values: list[float] = []
            moving_fraction_values: list[float] = []
            burst_top1_p75: list[float] = []
            burst_median_p75: list[float] = []
            for item in evidence:
                if bool(getattr(item, "valid", False)):
                    key = str(int(getattr(item, "active_count", 0)))
                    active_histogram[key] = active_histogram.get(key, 0) + 1
                    piglets = getattr(item, "metrics", {}).get("piglets", ())
                    burst_values = [
                        float(piglet["p75_body_per_sec"])
                        for piglet in piglets
                        if piglet.get("valid") and piglet.get("p75_body_per_sec") is not None
                    ]
                    piglet_p75_values.extend(burst_values)
                    moving_fraction_values.extend(
                        float(piglet["moving_fraction"])
                        for piglet in piglets
                        if piglet.get("valid") and piglet.get("moving_fraction") is not None
                    )
                    if burst_values:
                        burst_top1_p75.append(max(burst_values))
                        burst_median_p75.append(float(np.median(burst_values)))
                else:
                    key = str(getattr(item, "error", "") or "unknown")
                    invalid_reasons[key] = invalid_reasons.get(key, 0) + 1

            def percentiles(values: Sequence[float]) -> dict[str, float]:
                if not values:
                    return {}
                array = np.asarray(values, dtype=np.float64)
                return {
                    "p10": float(np.percentile(array, 10)),
                    "p25": float(np.percentile(array, 25)),
                    "p50": float(np.percentile(array, 50)),
                    "p75": float(np.percentile(array, 75)),
                    "p90": float(np.percentile(array, 90)),
                }

            summary = {
                "core_scope": asdict(core),
                "analysis_scope": asdict(analysis_scope),
                "error": error,
                "metrics": _jsonable(metrics),
                "burst_count": len(evidence),
                "valid_burst_count": len(valid_evidence),
                "valid_burst_ratio": len(valid_evidence) / len(evidence) if evidence else 0.0,
                "active_count_histogram": active_histogram,
                "piglet_p75_body_per_sec": percentiles(piglet_p75_values),
                "piglet_moving_fraction": percentiles(moving_fraction_values),
                "burst_top1_p75_body_per_sec": percentiles(burst_top1_p75),
                "burst_median_p75_body_per_sec": percentiles(burst_median_p75),
                "invalid_reason_counts": invalid_reasons,
            }
            summaries.append(summary)
            if error or not valid_evidence:
                errors.append(error or "no_sparse_flow_evidence")
                continue
            valid_ratio = len(valid_evidence) / len(evidence) if evidence else 0.0
            if valid_ratio >= 0.5:
                reliable_count += 1
            else:
                errors.append(
                    f"low_valid_burst_ratio:{valid_ratio:.3f}@"
                    f"{analysis_scope.start_sec:.3f}-{analysis_scope.end_sec:.3f}"
                )
            analysis_vetoes = sparse.sleep_veto_intervals(
                (analysis_scope.start_sec, analysis_scope.end_sec),
                evidence,
                strong_active_count=screen_args.screen_sparse_strong_active_count,
                weak_active_count=screen_args.screen_sparse_weak_active_count,
                strong_protect_before_sec=screen_args.screen_sparse_protect_before_sec,
                strong_protect_after_sec=screen_args.screen_sparse_protect_after_sec,
                weak_max_run_sec=screen_args.screen_sparse_weak_max_run_sec,
                weak_protect_sec=screen_args.screen_sparse_weak_protect_sec,
                min_low_activity_sec=screen_args.screen_sparse_min_sleep_sec,
                nominal_period_sec=actual_period_sec,
                max_reliable_gap_sec=1.5 * actual_period_sec,
                low_active_count=screen_args.screen_sparse_low_active_count,
            )
            # Context is evidence-only: only veto the candidate's non-overlap
            # core.  Neighboring cores/candidates independently see the same
            # 90/60-second protection context.
            raw_vetoes.extend(normalize_intervals(analysis_vetoes, core))
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    vetoes = _qualifying_candidate_vetoes(
        raw_vetoes,
        candidates,
        scope,
        min_duration_sec=screen_args.screen_sparse_min_sleep_sec,
    )
    return vetoes, {
        "enabled": True,
        "min_sleep_sec": screen_args.screen_sparse_min_sleep_sec,
        "reliable": reliable_count == len(work_units),
        "reliable_analysis_count": reliable_count,
        "candidate_count": len(candidates),
        "analysis_count": len(work_units),
        "candidate_summaries": summaries,
        "errors": errors,
        "veto_intervals": [asdict(item) for item in vetoes],
    }


def _jobs_from_payload(video_path: Path, payload: dict[str, Any]) -> list[ExtractionJob] | None:
    scope = _coerce_scope(payload.get("requested_scope"))
    signature = str(payload.get("screen_signature", ""))
    try:
        source_size = int(payload["source_size"])
        source_mtime_ns = int(payload["source_mtime_ns"])
    except (KeyError, TypeError, ValueError):
        return None
    configuration_tag = _job_configuration_tag(
        signature,
        source_size,
        source_mtime_ns,
    )
    raw_jobs = payload.get("jobs")
    planning = payload.get("planning")
    if scope is None or not signature or not isinstance(raw_jobs, list) or not isinstance(planning, dict):
        return None
    jobs: list[ExtractionJob] = []
    seen_suffixes: set[str] = set()
    seen_parts: set[tuple[int, int]] = set()
    for raw in raw_jobs:
        try:
            start = _finite_float(raw["start_sec"])
            end = _finite_float(raw["end_sec"])
            candidate_index = int(raw["candidate_index"])
            part_index = int(raw["part_index"])
            suffix = str(raw["output_suffix"])
            if (
                start is None
                or end is None
                or end <= start + EPSILON
                or start < scope.start_sec - EPSILON
                or end > scope.end_sec + EPSILON
                or candidate_index < 1
                or part_index < 1
                or suffix in seen_suffixes
                or (candidate_index, part_index) in seen_parts
                or suffix
                != job_output_suffix(
                    candidate_index,
                    part_index,
                    start,
                    end,
                    configuration_tag,
                )
            ):
                return None
            jobs.append(
                ExtractionJob(
                    video_path,
                    start,
                    end,
                    candidate_index,
                    part_index,
                    suffix,
                )
            )
            seen_suffixes.add(suffix)
            seen_parts.add((candidate_index, part_index))
        except (KeyError, TypeError, ValueError):
            return None

    raw_retained = planning.get("retained_intervals")
    if not isinstance(raw_retained, list):
        return None
    retained: list[TimeScope] = []
    for raw in raw_retained:
        item = _coerce_scope(raw)
        if (
            item is None
            or item.start_sec < scope.start_sec - EPSILON
            or item.end_sec > scope.end_sec + EPSILON
        ):
            return None
        retained.append(item)
    retained = normalize_intervals(retained, scope)
    job_coverage = normalize_intervals(jobs, scope)
    same_coverage = len(retained) == len(job_coverage) and all(
        abs(expected.start_sec - actual.start_sec) <= EPSILON
        and abs(expected.end_sec - actual.end_sec) <= EPSILON
        for expected, actual in zip(retained, job_coverage)
    )
    decision = str(planning.get("final_decision", ""))
    geometry = payload.get("geometry")
    sparse_payload = payload.get("sparse_flow")
    if (
        not same_coverage
        or bool(retained) != bool(jobs)
        or (decision == "process") != bool(retained)
        or not isinstance(geometry, dict)
        or not isinstance(sparse_payload, dict)
    ):
        return None
    geometry_decision = str(geometry.get("decision", "")).casefold()
    geometry_reliable = geometry.get("reliable") is True
    if geometry_decision in {"keep", "trim", "reject"}:
        if not geometry_reliable:
            return None
    elif geometry_decision in {"review", "error"}:
        if geometry_reliable:
            return None
    else:
        return None

    if decision == "process":
        if geometry_decision == "reject":
            return None
        if geometry_decision in {"review", "error"} and retained != [scope]:
            return None
    elif decision == "reject_geometry":
        if geometry_decision != "reject" or retained:
            return None
    elif decision == "reject_long_low_activity":
        raw_vetoes = sparse_payload.get("veto_intervals")
        raw_candidates = geometry.get("candidate_intervals")
        min_sleep_sec = _finite_float(sparse_payload.get("min_sleep_sec"))
        if not isinstance(raw_vetoes, list) or not isinstance(raw_candidates, list):
            return None
        vetoes = normalize_intervals(raw_vetoes, scope)
        candidates = normalize_intervals(raw_candidates, scope)
        if (
            geometry_decision not in {"keep", "trim"}
            or min_sleep_sec is None
            or min_sleep_sec <= 0
            or not vetoes
            or any(item.duration_sec + EPSILON < min_sleep_sec for item in vetoes)
            or not candidates
            or _subtract_intervals_local(candidates, vetoes)
            or retained
        ):
            return None
    else:
        # ``reject_no_candidate`` is only a defensive planning fallback and
        # cannot arise from a valid normalized geometry result.  Rescreen it
        # instead of accepting an unauditable zero-job cache entry.
        return None
    return jobs


def _payload_scope(payload: dict[str, Any]) -> TimeScope | None:
    return _coerce_scope(payload.get("requested_scope"))


def _make_screen_payload(
    video_path: Path,
    input_path: Path,
    scope: TimeScope,
    signature: str,
    fast_payload: dict[str, Any],
    normalized: ScreenResult,
    sparse_payload: dict[str, Any],
    jobs: Sequence[ExtractionJob],
) -> dict[str, Any]:
    stat = video_path.stat()
    retained = normalize_intervals(jobs, scope)
    retained_duration = sum(item.duration_sec for item in retained)
    veto_count = len(sparse_payload.get("veto_intervals", ()))
    if retained:
        final_decision = "process"
    elif normalized.decision == "reject":
        final_decision = "reject_geometry"
    elif veto_count:
        final_decision = "reject_long_low_activity"
    else:
        final_decision = "reject_no_candidate"
    return {
        "schema_version": SCREEN_SCHEMA_VERSION,
        "screen_signature": signature,
        "source_video": str(video_path),
        "relative_video": str(_relative_video(video_path, input_path)),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "requested_scope": asdict(scope),
        "geometry": {
            "decision": normalized.decision,
            "reason": normalized.reason,
            "reliable": normalized.reliable,
            "candidate_intervals": [asdict(item) for item in normalized.intervals],
            "raw": fast_payload,
        },
        "sparse_flow": sparse_payload,
        "planning": {
            "final_decision": final_decision,
            "retained_intervals": [asdict(item) for item in retained],
            "retained_duration_sec": retained_duration,
            "rejected_duration_sec": max(0.0, scope.duration_sec - retained_duration),
        },
        "jobs": [asdict(job) | {"video_path": str(job.video_path)} for job in jobs],
        "extraction": {"status": "pending" if jobs else "not_required"},
    }


def _video_duration_fail_open(video_path: Path, ffprobe: Path) -> float:
    """Resolve duration even when the screening probe itself is unavailable."""
    try:
        return float(fast.probe_video(video_path, ffprobe).duration_sec)
    except Exception:
        import cv2

        capture = cv2.VideoCapture(str(video_path))
        try:
            if capture.isOpened():
                fps = float(capture.get(cv2.CAP_PROP_FPS))
                frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                if (
                    math.isfinite(fps)
                    and math.isfinite(frames)
                    and min(fps, frames) > 0
                ):
                    return frames / fps
        finally:
            capture.release()
        # UniformerV2 embeds source-relative bounds in its file names.  This is
        # less authoritative than the container/frame count, so use it last.
        parsed = fast.parse_filename_duration(video_path)
        if parsed is not None and parsed > 0:
            return float(parsed)
        raise ValueError(f"Could not resolve video duration: {video_path}")


def _screen_one(
    model: Any,
    video_path: Path,
    args: argparse.Namespace,
    screen_args: argparse.Namespace,
    fast_args: argparse.Namespace,
    fast_signature: str,
    wrapper_signature: str,
) -> VideoPlan:
    source_stat = video_path.stat()
    duration_sec = _video_duration_fail_open(video_path, fast_args.ffprobe)
    scope = requested_scope(duration_sec, args.start_sec, args.max_seconds)
    local_fast_args = copy.copy(fast_args)
    local_fast_args.start_sec = scope.start_sec
    local_fast_args.max_seconds = (
        None if args.start_sec == 0 and args.max_seconds is None else scope.duration_sec
    )
    try:
        fast_payload = fast.analyze_video(
            model,
            video_path,
            args.input,
            local_fast_args,
            fast_signature,
        )
        if (
            int(fast_payload.get("source_size", -1)) != source_stat.st_size
            or int(fast_payload.get("source_mtime_ns", -1)) != source_stat.st_mtime_ns
        ):
            raise fast.SourceChangedError("Source changed during geometry screening.")
        normalized = screen_result_from_payload(fast_payload, scope)
    except Exception as exc:
        fast_payload = {
            "decision": "error",
            "reason": "geometry_screen_failed_open",
            "error": f"{type(exc).__name__}: {exc}",
            "keep_intervals": [],
        }
        normalized = ScreenResult("error", "geometry_screen_failed_open", (scope,), False)

    # Sparse veto is never applied to unreliable geometry/review fallbacks.  This
    # prevents low-amplitude, manually confirmed milk-letdown motion from being
    # discarded merely because its gross activity resembles sleep.
    if normalized.reliable and normalized.decision in {"keep", "trim"}:
        sparse_candidates = normalize_intervals(normalized.intervals, scope)
        vetoes, sparse_payload = _run_sparse_flow(
            model,
            video_path,
            screen_args,
            scope,
            sparse_candidates,
        )
    else:
        vetoes = []
        sparse_payload = {
            "enabled": bool(screen_args.screen_sparse_flow),
            "reliable": False,
            "reason": "not_applied_to_fail_open_geometry",
            "veto_intervals": [],
        }
    current_stat = video_path.stat()
    if (
        current_stat.st_size != source_stat.st_size
        or current_stat.st_mtime_ns != source_stat.st_mtime_ns
    ):
        raise fast.SourceChangedError("Source changed while its screening plan was being built.")
    jobs = plan_extraction_jobs(
        video_path,
        normalized,
        scope,
        vetoes,
        padding_sec=screen_args.screen_candidate_padding_sec,
        chunk_seconds=screen_args.screen_chunk_seconds,
        chunk_overlap_sec=screen_args.screen_chunk_overlap_sec,
        configuration_tag=_job_configuration_tag(
            wrapper_signature,
            source_stat.st_size,
            source_stat.st_mtime_ns,
        ),
    )
    path = _sidecar_path(video_path, args.input, args.output)
    payload = _make_screen_payload(
        video_path,
        args.input,
        scope,
        wrapper_signature,
        fast_payload,
        normalized,
        sparse_payload,
        jobs,
    )
    _atomic_json(path, payload)
    return VideoPlan(video_path, scope, jobs, path, payload)


def _full_fail_open_plan(
    video_path: Path,
    args: argparse.Namespace,
    screen_args: argparse.Namespace,
    wrapper_signature: str,
    screening_error: Exception,
) -> VideoPlan:
    """Build a full-scope plan after an unexpected screening/planning error."""

    source_stat = video_path.stat()
    duration_sec = _video_duration_fail_open(video_path, screen_args.screen_ffprobe)
    current_stat = video_path.stat()
    if (
        current_stat.st_size != source_stat.st_size
        or current_stat.st_mtime_ns != source_stat.st_mtime_ns
    ):
        raise fast.SourceChangedError("Source changed during fail-open replanning.")
    scope = requested_scope(duration_sec, args.start_sec, args.max_seconds)
    normalized = ScreenResult("error", "screen_plan_failed_open", (scope,), False)
    sparse_payload = {
        "enabled": bool(screen_args.screen_sparse_flow),
        "reliable": False,
        "reason": "screen_plan_failed_open",
        "veto_intervals": [],
    }
    jobs = plan_extraction_jobs(
        video_path,
        normalized,
        scope,
        (),
        padding_sec=screen_args.screen_candidate_padding_sec,
        chunk_seconds=screen_args.screen_chunk_seconds,
        chunk_overlap_sec=screen_args.screen_chunk_overlap_sec,
        configuration_tag=_job_configuration_tag(
            wrapper_signature,
            source_stat.st_size,
            source_stat.st_mtime_ns,
        ),
    )
    fast_payload = {
        "decision": "error",
        "reason": "screen_plan_failed_open",
        "error": f"{type(screening_error).__name__}: {screening_error}",
        "keep_intervals": [],
    }
    path = _sidecar_path(video_path, args.input, args.output)
    payload = _make_screen_payload(
        video_path,
        args.input,
        scope,
        wrapper_signature,
        fast_payload,
        normalized,
        sparse_payload,
        jobs,
    )
    if (
        int(payload["source_size"]) != source_stat.st_size
        or int(payload["source_mtime_ns"]) != source_stat.st_mtime_ns
    ):
        raise fast.SourceChangedError("Source changed before fail-open plan publication.")
    _atomic_json(path, payload)
    return VideoPlan(video_path, scope, jobs, path, payload)


def _write_manifest(path: Path, plans: Sequence[VideoPlan]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields = (
        "source_video",
        "relative_video",
        "scope_start_sec",
        "scope_end_sec",
        "geometry_decision",
        "geometry_reason",
        "final_decision",
        "retained_duration_sec",
        "rejected_duration_sec",
        "sparse_reliable",
        "sparse_veto_count",
        "job_count",
        "extraction_status",
        "error",
    )
    with temporary.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for plan in plans:
            geometry = plan.payload.get("geometry", {})
            sparse = plan.payload.get("sparse_flow", {})
            planning = plan.payload.get("planning", {})
            extraction = plan.payload.get("extraction", {})
            writer.writerow(
                {
                    "source_video": str(plan.video_path),
                    "relative_video": plan.payload.get("relative_video", ""),
                    "scope_start_sec": plan.scope.start_sec,
                    "scope_end_sec": plan.scope.end_sec,
                    "geometry_decision": geometry.get("decision", ""),
                    "geometry_reason": geometry.get("reason", ""),
                    "final_decision": planning.get("final_decision", ""),
                    "retained_duration_sec": planning.get("retained_duration_sec", ""),
                    "rejected_duration_sec": planning.get("rejected_duration_sec", ""),
                    "sparse_reliable": sparse.get("reliable", False),
                    "sparse_veto_count": len(sparse.get("veto_intervals", ())),
                    "job_count": len(plan.jobs),
                    "extraction_status": extraction.get("status", ""),
                    "error": extraction.get("error", ""),
                }
            )
    os.replace(temporary, path)


def _load_full_resources(args: argparse.Namespace) -> tuple[Any, Any, tracked.TrackerResources]:
    import torch
    from ultralytics import YOLO
    from ultralytics.trackers.mask_aware import load_association_checkpoint

    for weights in (args.piglet_weights, args.sow_weights, args.tracker_checkpoint):
        if not weights.expanduser().is_file():
            raise FileNotFoundError(f"Weights not found: {weights}")
    device = tracked.select_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    association_model, checkpoint = load_association_checkpoint(args.tracker_checkpoint, device)
    training_sample_fps = float(checkpoint.get("training_args", {}).get("sample_fps", 0.0))
    if not math.isfinite(training_sample_fps) or training_sample_fps <= 0:
        raise ValueError("Tracker checkpoint does not contain a positive training_args.sample_fps.")
    checkpoint_path = args.tracker_checkpoint.expanduser().resolve()
    resources = tracked.TrackerResources(
        association_model,
        checkpoint,
        device,
        training_sample_fps,
        hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
    )
    piglet_model = YOLO(str(args.piglet_weights), task="segment")
    sow_model = YOLO(str(args.sow_weights), task="segment")
    if args.torch_compile != "none":
        compile_mode: bool | str = True if args.torch_compile == "default" else args.torch_compile
        piglet_model.overrides["compile"] = compile_mode
        sow_model.overrides["compile"] = compile_mode
    return piglet_model, sow_model, resources


def _run_extraction(
    plans: Sequence[VideoPlan],
    args: argparse.Namespace,
) -> tuple[int, int, int]:
    all_jobs = [(plan, job) for plan in plans for job in plan.jobs]
    if not all_jobs:
        print("[INFO] no retained intervals; full tracked models were not loaded")
        return 0, 0, 0
    piglet_model, sow_model, resources = _load_full_resources(args)
    processed = skipped = failed = 0
    for index, (plan, job) in enumerate(all_jobs, start=1):
        job_args = copy.copy(args)
        job_args.start_sec = job.start_sec
        job_args.max_seconds = job.duration_sec
        job_args.chunk_seconds = 0.0
        job_args.output_suffix = job.output_suffix
        if not job_args.overwrite:
            job_args.resume = True
        try:
            source_stat = job.video_path.stat()
            if (
                source_stat.st_size != int(plan.payload.get("source_size", -1))
                or (
                    not plan.compatible_resume
                    and source_stat.st_mtime_ns
                    != int(plan.payload.get("source_mtime_ns", -1))
                )
            ):
                raise fast.SourceChangedError(
                    "Source changed after screening; rerun to rebuild its plan."
                )
            print(
                f"[INFO] extract {index}/{len(all_jobs)} {job.video_path.name} "
                f"{job.start_sec:.3f}-{job.end_sec:.3f}s"
            )
            result = tracked.process_video(
                piglet_model,
                sow_model,
                resources,
                job.video_path,
                job_args,
            )
            if result is None:
                skipped += 1
            else:
                processed += 1
        except Exception as exc:
            failed += 1
            extraction = plan.payload.setdefault("extraction", {})
            extraction["status"] = "failed"
            extraction["error"] = f"{type(exc).__name__}: {exc}"
            _atomic_json(plan.sidecar_path, plan.payload)
            print(
                f"[WARN] extraction failed {job.video_path} {job.output_suffix}: "
                f"{type(exc).__name__}: {exc}"
            )
            if args.fail_on_error:
                raise
    for plan in plans:
        extraction = plan.payload.setdefault("extraction", {})
        if extraction.get("status") != "failed":
            extraction["status"] = "complete" if plan.jobs else "not_required"
        _atomic_json(plan.sidecar_path, plan.payload)
    return processed, skipped, failed


def _discover_input_videos(input_path: Path, recursive: bool) -> tuple[list[Path], bool]:
    """Select videos, automatically descending when a directory has no top-level videos."""

    try:
        return tracked.flow_base.iter_videos(input_path, recursive), recursive
    except FileNotFoundError:
        if recursive or not input_path.is_dir():
            raise
        return tracked.flow_base.iter_videos(input_path, True), True


def main() -> int:
    args, screen_args = parse_args()
    if screen_args.skip_screen:
        print("[INFO] screening skipped; running tracked extraction directly")
        tracked.main(args)
        return 0
    _validate_wrapper_args(args, screen_args)
    tracked._configure_runtime(args)
    if screen_args.screen_device is None:
        screen_args.screen_device = args.device
    videos, effective_recursive = _discover_input_videos(args.input, args.recursive)
    if effective_recursive and not args.recursive:
        args.recursive = True
        print("[INFO] no top-level videos found; recursive input scan enabled automatically")
    if args.max_videos is not None:
        videos = videos[: max(0, args.max_videos)]
    if not videos:
        raise ValueError("No videos selected.")
    print(f"Input: {args.input}")
    print(f"Videos: {len(videos)}")
    print(f"Output: {args.output}")
    print(
        f"Screen: geometry {screen_args.screen_sample_fps:g} FPS; "
        f"sparse flow={'on' if screen_args.screen_sparse_flow else 'off'} "
        f"({screen_args.screen_sparse_flow_fps:g} FPS, "
        f"{screen_args.screen_sparse_burst_sec:g}s/{screen_args.screen_sparse_period_sec:g}s bursts)"
    )
    print(
        f"Extraction jobs: {screen_args.screen_chunk_seconds:g}s max, "
        f"{screen_args.screen_chunk_overlap_sec:g}s overlap"
    )
    if args.dry_run:
        print("Dry run: screening and full models are not loaded.")
        for video in videos:
            print(video)
        return 0

    args.output.mkdir(parents=True, exist_ok=True)
    fast_args = _make_fast_args(args, screen_args, args.output / "_screen_fast")
    fast.validate_args(fast_args)
    if not screen_args.screen_weights.is_file():
        raise FileNotFoundError(f"Screen weights not found: {screen_args.screen_weights}")
    fast_signature = fast.analysis_signature(fast_args)
    wrapper_signature = _wrapper_signature(fast_signature, screen_args, args)

    plans_by_path: dict[Path, VideoPlan] = {}
    pending: list[Path] = []
    compatible_resume_count = 0
    for video_path in videos:
        sidecar = _sidecar_path(video_path, args.input, args.output)
        payload = (
            _resume_payload(
                sidecar,
                video_path,
                wrapper_signature,
                screen_args.screen_resume_policy,
                args.input,
            )
            if screen_args.screen_resume and not screen_args.screen_overwrite
            else None
        )
        scope = _payload_scope(payload) if payload is not None else None
        if payload is not None and scope is not None:
            resumed_jobs = _jobs_from_payload(video_path, payload)
            if resumed_jobs is None:
                pending.append(video_path)
                continue
            source_stat = video_path.stat()
            cached_source = Path(str(payload.get("source_video", "")))
            compatible_resume = (
                payload.get("screen_signature") != wrapper_signature
                or cached_source.resolve() != video_path.resolve()
                or int(payload.get("source_mtime_ns", -1))
                != source_stat.st_mtime_ns
            )
            if compatible_resume:
                compatible_resume_count += 1
            plans_by_path[video_path] = VideoPlan(
                video_path,
                scope,
                resumed_jobs,
                sidecar,
                payload,
                compatible_resume,
            )
            print(f"[INFO] screen resume {video_path.name}: jobs={len(resumed_jobs)}")
        else:
            pending.append(video_path)

    if compatible_resume_count:
        print(
            f"[WARN] compatible resume reused {compatible_resume_count} validated screen plans "
            "whose signature, source location, and/or source timestamp differs from the current "
            "deployment; use "
            "--screen-resume-policy strict to rebuild them."
        )

    model: Any | None = None
    if pending:
        from ultralytics import YOLO

        fast._configure_torch()
        model = YOLO(str(screen_args.screen_weights), task="segment")
        labels = {str(value).casefold() for value in model.names.values()}
        required = {fast_args.sow_label.casefold(), fast_args.piglet_label.casefold()}
        if not required.issubset(labels):
            raise ValueError(f"Screen model classes {model.names} do not contain {sorted(required)}")
        for index, video_path in enumerate(pending, start=1):
            print(f"[INFO] screen {index}/{len(pending)}: {video_path}")
            try:
                plans_by_path[video_path] = _screen_one(
                    model,
                    video_path,
                    args,
                    screen_args,
                    fast_args,
                    fast_signature,
                    wrapper_signature,
                )
            except Exception as exc:
                if args.fail_on_error:
                    raise
                try:
                    plans_by_path[video_path] = _full_fail_open_plan(
                        video_path,
                        args,
                        screen_args,
                        wrapper_signature,
                        exc,
                    )
                    print(
                        f"[WARN] screen planning failed open; extracting full scope "
                        f"{video_path}: {type(exc).__name__}: {exc}"
                    )
                except Exception as fallback_exc:
                    # An unreadable or repeatedly changing source cannot be
                    # extracted safely.  It remains an explicit failed video,
                    # never a successful screening rejection.
                    print(
                        f"[ERROR] could not build fail-open plan {video_path}: "
                        f"{type(fallback_exc).__name__}: {fallback_exc}; "
                        f"original={type(exc).__name__}: {exc}"
                    )

    if model is not None:
        del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    plans = [plans_by_path[path] for path in videos if path in plans_by_path]
    manifest = args.output / "screened_extraction_manifest.csv"
    _write_manifest(manifest, plans)
    if screen_args.screen_only:
        for plan in plans:
            extraction = plan.payload.setdefault("extraction", {})
            if extraction.get("status") == "pending":
                extraction["status"] = "planned_only"
                _atomic_json(plan.sidecar_path, plan.payload)
        _write_manifest(manifest, plans)
        missing_plans = len(videos) - len(plans)
        print(
            f"Done (screen only). videos={len(videos)}, planned={len(plans)}, "
            f"failed={missing_plans}, manifest={manifest}"
        )
        return 1 if missing_plans else 0
    processed, skipped, failed = _run_extraction(plans, args)
    _write_manifest(manifest, plans)
    missing_plans = len(videos) - len(plans)
    failed += missing_plans
    total_jobs = sum(len(plan.jobs) for plan in plans)
    print(
        f"Done. videos={len(videos)}, jobs={total_jobs}, processed={processed}, "
        f"skipped={skipped}, failed={failed}, manifest={manifest}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
