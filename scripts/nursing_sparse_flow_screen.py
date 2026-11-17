"""Conservative sparse-flow evidence for long nursing-candidate videos.

This module is deliberately an *evidence layer*, not a nursing classifier.
It may veto only long, continuously reliable, unprotected low-motion runs.
Short or weak milk-let-down activity is protected by :func:`sleep_veto_intervals`
and all inference, geometry, illumination, or reference-motion failures are
treated as unknown rather than as sleep.

The decoder first samples at 4 fps and then retains a two-second burst every
four seconds.  ``FFmpegFrameReader`` necessarily numbers the selected raw
frames contiguously, so absolute source timestamps are reconstructed from the
burst number and the position inside the burst.  Optical flow is never
calculated across burst boundaries.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

from scripts import filter_nursing_clips_fast as fast


Interval = tuple[float, float]


@dataclass(frozen=True)
class SparseBurstEvidence:
    """One sparse burst's auditable motion evidence.

    ``valid=False`` means unknown evidence and must never be interpreted as
    sleep.  ``active_count`` counts contact piglets whose compensated,
    body-normalized flow exceeds either of the deliberately sensitive motion
    thresholds.
    """

    timestamp: float
    valid: bool
    contact_count: int = 0
    active_count: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    @property
    def timestamp_sec(self) -> float:
        """Alias used by code that follows the rest of the extraction API."""

        return self.timestamp


@dataclass(frozen=True)
class SparseFlowResult:
    """Sparse-flow observations for one requested absolute-time scope."""

    scope: Interval
    evidence: tuple[SparseBurstEvidence, ...]
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    @property
    def valid_count(self) -> int:
        return sum(item.valid for item in self.evidence)


_GEOMETRY_DEFAULTS: dict[str, Any] = {
    "reference_width": 1920.0,
    "min_sow_area": 1000,
    "min_piglet_area": 80,
    "max_piglet_sow_overlap": 0.60,
    "min_contact_piglets": 4,
    "min_contact_span_ratio": 0.25,
    "min_contact_area_px": 20,
    "min_contact_area_ratio": 0.01,
    "max_contact_dist": 48.0,
    "contact_margin": 12.0,
    "cluster_gap": 360.0,
    "head_projection_quantile": 0.90,
    "arc_padding": 70.0,
    "band_width": 80,
    "band_outer_width": 75.0,
    "band_inner_width": 35.0,
    "centerline_thickness": 5,
    "contour_smooth_window": 25.0,
    "turn_window": 60.0,
    "max_turn_angle": 50.0,
    "min_turn_radius": 35.0,
    "max_axis_deviation": 30.0,
    "max_contour_bridge": 180.0,
    "max_endpoint_extension": 140.0,
    "sow_label": "sow",
    "piglet_label": "piglet",
}

_INFERENCE_DEFAULTS: dict[str, Any] = {
    "imgsz": 640,
    "conf": 0.20,
    "iou": 0.70,
    "device": "0",
    "max_det": 32,
    "half": True,
    "mask_threshold": 0.5,
}


def _value(args: argparse.Namespace, name: str, default: Any) -> Any:
    value = getattr(args, name, default)
    return default if value is None else value


def _analysis_args(args: argparse.Namespace) -> argparse.Namespace:
    """Fill the geometry/inference fields used by the existing fast helpers."""

    values = vars(args).copy()
    for name, default in {**_GEOMETRY_DEFAULTS, **_INFERENCE_DEFAULTS}.items():
        if values.get(name) is None:
            values[name] = default
    # Sparse anchors are only 384 px wide by default, so 640 px inference is
    # sufficient and materially cheaper than inheriting the 1024 px geometry
    # screen.  The integrated wrapper may still override this explicitly.
    values["imgsz"] = int(values.get("sparse_flow_imgsz", values["imgsz"]))
    return argparse.Namespace(**values)


def _scope_tuple(scope: Any, duration_sec: float) -> Interval:
    if scope is None:
        start, end = 0.0, duration_sec
    elif hasattr(scope, "start_sec") and hasattr(scope, "end_sec"):
        start, end = float(scope.start_sec), float(scope.end_sec)
    else:
        start, end = map(float, scope)
    start = min(max(0.0, start), duration_sec)
    end = min(max(start, end), duration_sec)
    if end <= start:
        raise ValueError(f"Empty sparse-flow scope: ({start}, {end})")
    return start, end


def _sampling_grid(
    sample_fps: float,
    burst_sec: float,
    period_sec: float,
) -> tuple[int, int, float, float]:
    """Return the integer-frame sampling grid and its actual durations.

    FFmpeg's ``select`` filter necessarily works in whole frames.  Keeping the
    rounded, frame-derived durations here prevents a small user-supplied
    fractional period from accumulating timestamp drift over a long video.
    """

    if sample_fps <= 0 or burst_sec <= 0 or period_sec <= 0 or burst_sec > period_sec:
        raise ValueError("Sparse flow requires fps > 0 and 0 < burst_sec <= period_sec.")
    frames_per_burst = max(2, int(round(burst_sec * sample_fps)))
    frames_per_period = max(1, int(round(period_sec * sample_fps)))
    if frames_per_burst > frames_per_period:
        raise ValueError("Sparse burst duration cannot exceed its rounded frame period.")
    return (
        frames_per_burst,
        frames_per_period,
        frames_per_burst / sample_fps,
        frames_per_period / sample_fps,
    )


def _burst_decode_command(
    ffmpeg: str | Path,
    video: str | Path,
    width: int,
    height: int,
    sample_fps: float,
    burst_sec: float,
    period_sec: float,
    hwaccel: str,
    start_sec: float,
    max_seconds: float,
) -> list[str]:
    frames_per_burst, frames_per_period, _, _ = _sampling_grid(
        sample_fps,
        burst_sec,
        period_sec,
    )
    # Commas inside the select expression are escaped for FFmpeg's filter
    # parser.  No shell quoting is involved because Popen receives an argv.
    filters = (
        f"fps={sample_fps:.9g}:start_time=0,"
        f"select=lt(mod(n\\,{frames_per_period})\\,{frames_per_burst}),"
        f"scale={width}:{height}:flags=bilinear"
    )
    command = [str(ffmpeg), "-hide_banner", "-loglevel", "error"]
    if hwaccel == "cuda":
        command.extend(("-hwaccel", "cuda"))
    if start_sec > 0:
        command.extend(("-ss", f"{start_sec:.6f}"))
    command.extend(("-i", str(video), "-t", f"{max_seconds:.6f}", "-an"))
    # Disable output timestamp synchronization; otherwise FFmpeg may duplicate
    # frames to fill the two-second gaps removed by ``select``.
    command.extend(
        ("-vf", filters, "-vsync", "0", "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1")
    )
    return command


def _resize_gray(frame: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if gray.shape != shape:
        gray = cv2.resize(gray, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(gray)


def _union_masks(instances: Sequence[Any], shape: tuple[int, int]) -> np.ndarray:
    union = np.zeros(shape, dtype=bool)
    for instance in instances:
        union |= fast.mask_for_shape(instance.mask, shape)
    return union


def _invalid(
    timestamp: float,
    error: str,
    *,
    contact_count: int = 0,
    metrics: dict[str, Any] | None = None,
) -> SparseBurstEvidence:
    return SparseBurstEvidence(
        timestamp=float(timestamp),
        valid=False,
        contact_count=int(contact_count),
        active_count=0,
        metrics=metrics or {},
        error=error,
    )


def _measure_burst(
    frames: Sequence[np.ndarray],
    timestamp: float,
    instances: list[Any],
    args: argparse.Namespace,
    geometry: Any,
    geometry_shape: tuple[int, int],
    flow_fps: float,
    flow_estimator: Any,
) -> SparseBurstEvidence:
    """Measure one burst; no state is accepted, preventing cross-burst flow."""

    contact_count = 0
    try:
        sow, piglets = fast._select_instances(
            instances,
            geometry,
            float(_value(args, "max_piglet_sow_overlap", 0.60)),
        )
        candidate = fast.strict_nursing_band(sow, piglets, geometry_shape, args, geometry)
        contact_count = candidate.contact_count
    except fast.InsufficientBandError as exc:
        return _invalid(
            timestamp,
            f"geometry:{exc.reason}",
            contact_count=exc.contact_count,
            metrics={"span_ratio": exc.span_ratio},
        )
    except Exception as exc:
        return _invalid(timestamp, f"geometry:{type(exc).__name__}:{exc}")

    sow_mask = fast.mask_for_shape(sow.mask, geometry_shape).astype(bool, copy=False)
    piglet_union = _union_masks(piglets, geometry_shape)
    reference_mask = np.logical_and(sow_mask, ~piglet_union)
    reference_erode = int(_value(args, "sparse_flow_reference_erode_px", 2))
    if reference_erode > 0:
        kernel_size = 2 * reference_erode + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        reference_mask = cv2.erode(reference_mask.astype(np.uint8), kernel).astype(bool)
    min_reference_pixels = int(_value(args, "sparse_flow_min_reference_pixels", 120))
    if int(reference_mask.sum()) < min_reference_pixels:
        return _invalid(
            timestamp,
            "reference_pixels_below_minimum",
            contact_count=contact_count,
            metrics={"reference_pixels": int(reference_mask.sum())},
        )

    gray_frames = [_resize_gray(frame, geometry_shape) for frame in frames]
    brightness_limit = float(_value(args, "sparse_flow_max_brightness_shift", 18.0))
    moving_speed = float(_value(args, "sparse_flow_moving_speed_body_per_sec", 0.018))
    active_p75 = float(_value(args, "sparse_flow_active_p75_body_per_sec", 0.025))
    active_fraction = float(_value(args, "sparse_flow_active_moving_fraction", 0.08))

    per_piglet_speeds: list[list[np.ndarray]] = [[] for _ in candidate.contact_piglets]
    brightness_shifts: list[float] = []
    reference_vectors: list[tuple[float, float]] = []
    try:
        for previous, current in zip(gray_frames, gray_frames[1:]):
            brightness_shift = abs(float(np.median(current)) - float(np.median(previous)))
            brightness_shifts.append(brightness_shift)
            if brightness_shift > brightness_limit:
                return _invalid(
                    timestamp,
                    "brightness_shift",
                    contact_count=contact_count,
                    metrics={
                        "brightness_shift_max": max(brightness_shifts),
                        "brightness_shift_limit": brightness_limit,
                    },
                )
            flow = flow_estimator.calculate(previous, current)
            reference = flow[reference_mask]
            if len(reference) < min_reference_pixels:
                return _invalid(
                    timestamp,
                    "reference_flow_below_minimum",
                    contact_count=contact_count,
                    metrics={"reference_pixels": int(len(reference))},
                )
            global_vector = np.median(reference, axis=0).astype(np.float32)
            reference_vectors.append((float(global_vector[0]), float(global_vector[1])))
            compensated = flow - global_vector.reshape(1, 1, 2)
            for index, piglet in enumerate(candidate.contact_piglets):
                mask = fast.mask_for_shape(piglet.mask, geometry_shape).astype(bool, copy=False)
                pixel_vectors = compensated[mask]
                if len(pixel_vectors) < 8:
                    continue
                body_scale = max(4.0, math.sqrt(float(mask.sum())))
                speed = np.linalg.norm(pixel_vectors, axis=1) * flow_fps / body_scale
                per_piglet_speeds[index].append(speed.astype(np.float32, copy=False))
    except Exception as exc:
        return _invalid(
            timestamp,
            f"flow:{type(exc).__name__}:{exc}",
            contact_count=contact_count,
        )

    piglet_metrics: list[dict[str, Any]] = []
    active_count = 0
    for piglet, chunks in zip(candidate.contact_piglets, per_piglet_speeds):
        if not chunks:
            piglet_metrics.append(
                {"instance_index": int(piglet.index), "valid": False, "p75": None, "moving_fraction": None}
            )
            continue
        speeds = np.concatenate(chunks)
        p75 = float(np.percentile(speeds, 75))
        moving_fraction = float(np.mean(speeds >= moving_speed))
        # OR is intentional: the evidence layer must be sensitive to subtle
        # milk-let-down arching and must prefer a false active anchor over a
        # false sleep veto.
        active = p75 >= active_p75 or moving_fraction >= active_fraction
        active_count += int(active)
        piglet_metrics.append(
            {
                "instance_index": int(piglet.index),
                "valid": True,
                "p75_body_per_sec": p75,
                "moving_fraction": moving_fraction,
                "active": active,
            }
        )

    if any(not item["valid"] for item in piglet_metrics):
        return _invalid(
            timestamp,
            "piglet_flow_pixels_below_minimum",
            contact_count=contact_count,
            metrics={"piglets": piglet_metrics},
        )

    return SparseBurstEvidence(
        timestamp=float(timestamp),
        valid=True,
        contact_count=int(contact_count),
        active_count=int(active_count),
        metrics={
            "pair_count": max(0, len(gray_frames) - 1),
            "reference_pixels": int(reference_mask.sum()),
            "brightness_shift_max": max(brightness_shifts, default=0.0),
            "reference_flow_median_vectors": reference_vectors,
            "piglets": piglet_metrics,
            "span_ratio": float(candidate.span_ratio),
            "radial_alignment_ratio": float(candidate.radial_alignment_ratio),
        },
    )


def _process_burst_batch(
    model: Any,
    bursts: Sequence[tuple[float, list[np.ndarray]]],
    args: argparse.Namespace,
    geometry: Any,
    geometry_shape: tuple[int, int],
    flow_fps: float,
    flow_estimator: Any,
) -> list[SparseBurstEvidence]:
    anchor_indices = [min(len(frames) - 1, len(frames) // 2) for _, frames in bursts]
    anchors = [frames[index] for (_, frames), index in zip(bursts, anchor_indices)]
    predictions = fast._predict_batch_resilient(model, anchors, args)
    valid_predictions = [item for item in predictions if not isinstance(item, BaseException)]
    try:
        converted = iter(
            fast.result_instances_fast_batch(valid_predictions, float(args.mask_threshold), geometry_shape)
        )
    except Exception as exc:
        return [
            _invalid(timestamp, f"mask_conversion:{type(exc).__name__}:{exc}")
            for timestamp, _ in bursts
        ]

    output: list[SparseBurstEvidence] = []
    for (timestamp, frames), prediction in zip(bursts, predictions):
        if isinstance(prediction, BaseException):
            output.append(_invalid(timestamp, f"predict:{type(prediction).__name__}:{prediction}"))
            continue
        try:
            instances = next(converted)
        except Exception as exc:
            output.append(_invalid(timestamp, f"mask_conversion:{type(exc).__name__}:{exc}"))
            continue
        output.append(
            _measure_burst(
                frames,
                timestamp,
                instances,
                args,
                geometry,
                geometry_shape,
                flow_fps,
                flow_estimator,
            )
        )
    return output


def _analyze_decode(
    video_path: Path,
    model: Any,
    args: argparse.Namespace,
    scope: Interval,
    source_shape: tuple[int, int],
    hwaccel: str,
) -> tuple[list[SparseBurstEvidence], dict[str, Any]]:
    # Lazy import keeps pure interval-policy tests independent of torch/CUDA
    # initialization while still using the tracked extractor's shared backend.
    from scripts.extract_nursing_tracked_normal_flow import DenseFlowEstimator

    flow_fps = float(_value(args, "sparse_flow_fps", 4.0))
    requested_burst_sec = float(_value(args, "sparse_flow_burst_sec", 2.0))
    requested_period_sec = float(_value(args, "sparse_flow_period_sec", 4.0))
    (
        frames_per_burst,
        frames_per_period,
        actual_burst_sec,
        actual_period_sec,
    ) = _sampling_grid(flow_fps, requested_burst_sec, requested_period_sec)

    requested_width = int(_value(args, "sparse_flow_width", 384))
    decode_width, decode_height = fast.decoded_size(source_shape[1], source_shape[0], requested_width)
    geometry_width = min(decode_width, int(_value(args, "sparse_flow_geometry_width", decode_width)))
    geometry_width, geometry_height = fast.decoded_size(decode_width, decode_height, geometry_width)
    geometry_shape = (geometry_height, geometry_width)
    geometry = fast._scaled_args(args, geometry_width)
    flow_estimator = DenseFlowEstimator("dis-fast", 0.5)

    command = _burst_decode_command(
        _value(args, "ffmpeg", fast.DEFAULT_FFMPEG),
        video_path,
        decode_width,
        decode_height,
        flow_fps,
        requested_burst_sec,
        requested_period_sec,
        hwaccel,
        scope[0],
        scope[1] - scope[0],
    )
    queue_size = int(_value(args, "sparse_flow_decode_queue", 96))
    anchor_batch = max(1, int(_value(args, "sparse_flow_anchor_batch_size", 32)))
    pending_bursts: list[tuple[float, list[np.ndarray]]] = []
    current_frames: list[np.ndarray] = []
    current_burst_index = 0
    evidence: list[SparseBurstEvidence] = []

    def flush() -> None:
        if not pending_bursts:
            return
        evidence.extend(
            _process_burst_batch(
                model,
                pending_bursts,
                args,
                geometry,
                geometry_shape,
                flow_fps,
                flow_estimator,
            )
        )
        pending_bursts.clear()

    with fast.FFmpegFrameReader(
        command,
        decode_width,
        decode_height,
        flow_fps,
        scope[0],
        queue_size,
    ) as reader:
        for packet in reader:
            # Selected output frames are contiguous.  Reconstruct the original
            # timeline explicitly instead of trusting packet.timestamp_sec.
            burst_index = packet.index // frames_per_burst
            position = packet.index % frames_per_burst
            if burst_index != current_burst_index:
                if current_frames:
                    timestamp = scope[0] + current_burst_index * actual_period_sec + (
                        len(current_frames) - 1
                    ) / (2.0 * flow_fps)
                    if len(current_frames) == frames_per_burst:
                        pending_bursts.append((timestamp, current_frames))
                    else:
                        evidence.append(
                            _invalid(
                                timestamp,
                                "short_burst",
                                metrics={"frame_count": len(current_frames)},
                            )
                        )
                current_frames = []
                current_burst_index = burst_index
            # This identity is an assertion of the timestamp reconstruction;
            # it also documents that no selected frame crosses a burst.
            expected_time = scope[0] + burst_index * actual_period_sec + position / flow_fps
            if expected_time >= scope[1] + 1.0 / flow_fps:
                continue
            current_frames.append(packet.frame)
            if len(pending_bursts) >= anchor_batch:
                flush()

    if current_frames:
        timestamp = scope[0] + current_burst_index * actual_period_sec + (
            len(current_frames) - 1
        ) / (2.0 * flow_fps)
        if len(current_frames) == frames_per_burst:
            pending_bursts.append((timestamp, current_frames))
        else:
            evidence.append(
                _invalid(timestamp, "short_burst", metrics={"frame_count": len(current_frames)})
            )
    flush()
    evidence.sort(key=lambda item: item.timestamp)
    return evidence, {
        "flow_fps": flow_fps,
        "requested_burst_sec": requested_burst_sec,
        "requested_period_sec": requested_period_sec,
        "burst_sec": actual_burst_sec,
        "period_sec": actual_period_sec,
        "frames_per_burst": frames_per_burst,
        "frames_per_period": frames_per_period,
        "decode_width": decode_width,
        "decode_height": decode_height,
        "geometry_width": geometry_width,
        "geometry_height": geometry_height,
        "hwaccel": hwaccel,
    }


def analyze_sparse_flow(
    video_path: str | Path,
    model: Any,
    args: argparse.Namespace,
    scope: Any = None,
) -> SparseFlowResult:
    """Collect sparse DIS evidence for ``scope`` without classifying the video.

    Parameters are read with ``getattr`` so an integrated entrypoint may expose
    only the options it wants.  Important optional names include
    ``sparse_flow_width``, ``sparse_flow_fps``, ``sparse_flow_burst_sec``,
    ``sparse_flow_period_sec`` and the activity thresholds used in
    :func:`_measure_burst`.
    """

    path = Path(video_path)
    analysis_args = _analysis_args(args)
    ffprobe = Path(_value(analysis_args, "ffprobe", fast.DEFAULT_FFPROBE))
    info = fast.probe_video(path, ffprobe)
    actual_scope = _scope_tuple(scope, info.duration_sec)
    requested_hwaccel = str(_value(analysis_args, "hwaccel", "cuda"))
    try:
        evidence, metrics = _analyze_decode(
            path,
            model,
            analysis_args,
            actual_scope,
            (info.height, info.width),
            requested_hwaccel,
        )
    except fast.DecodeError as exc:
        if requested_hwaccel != "cuda" or not bool(_value(analysis_args, "decode_fallback", True)):
            return SparseFlowResult(actual_scope, (), error=f"decode:{type(exc).__name__}:{exc}")
        try:
            evidence, metrics = _analyze_decode(
                path,
                model,
                analysis_args,
                actual_scope,
                (info.height, info.width),
                "none",
            )
            metrics["decode_fallback_from"] = "cuda"
        except Exception as fallback_exc:
            return SparseFlowResult(
                actual_scope,
                (),
                error=f"decode_fallback:{type(fallback_exc).__name__}:{fallback_exc}",
            )
    except Exception as exc:
        return SparseFlowResult(actual_scope, (), error=f"analysis:{type(exc).__name__}:{exc}")
    metrics.update(
        {
            "video_width": info.width,
            "video_height": info.height,
            "video_duration_sec": info.duration_sec,
            "valid_bursts": sum(item.valid for item in evidence),
            "total_bursts": len(evidence),
        }
    )
    return SparseFlowResult(actual_scope, tuple(evidence), metrics=metrics)


def _merge_intervals(intervals: Iterable[Interval], *, gap: float = 0.0) -> list[Interval]:
    normalized = sorted(
        (float(start), float(end))
        for start, end in intervals
        if math.isfinite(start) and math.isfinite(end) and end > start
    )
    merged: list[list[float]] = []
    for start, end in normalized:
        if not merged or start > merged[-1][1] + gap:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def sleep_veto_intervals(
    scope: Interval,
    evidence: Sequence[SparseBurstEvidence],
    *,
    strong_active_count: int = 3,
    weak_active_count: int = 2,
    strong_protect_before_sec: float = 90.0,
    strong_protect_after_sec: float = 60.0,
    weak_max_run_sec: float = 30.0,
    weak_protect_sec: float = 30.0,
    min_low_activity_sec: float = 180.0,
    max_reliable_gap_sec: float = 8.0,
    nominal_period_sec: float = 4.0,
    low_active_count: int = 2,
) -> list[Interval]:
    """Return only long, reliable sleep-like ranges safe to veto.

    Protection policy is intentionally asymmetric and conservative:

    * ``active_count >= 3`` protects 90 seconds before and 60 after;
    * short runs with ``active_count >= 2`` protect 30 seconds on both sides;
    * invalid/unknown evidence, excessive sample gaps, active anchors, and
      protected low-motion anchors break a possible veto run;
    * a resulting unprotected run with no more than two active piglets must
      cover at least 180 seconds.

    Thus a low-activity milk-let-down interval adjacent to even a brief
    calibrated activity anchor cannot be removed by sparse flow alone.
    """

    scope_start, scope_end = map(float, scope)
    if scope_end <= scope_start:
        return []
    samples = sorted(
        (item for item in evidence if scope_start <= item.timestamp <= scope_end),
        key=lambda item: item.timestamp,
    )
    if not samples:
        return []

    protections: list[Interval] = []
    for item in samples:
        if item.valid and item.active_count >= strong_active_count:
            protections.append(
                (
                    max(scope_start, item.timestamp - strong_protect_before_sec),
                    min(scope_end, item.timestamp + strong_protect_after_sec),
                )
            )

    weak_runs: list[list[SparseBurstEvidence]] = []
    current_weak: list[SparseBurstEvidence] = []
    previous_sample_timestamp: float | None = None
    # A 2,1,2,1,... pattern represents one persistent weak episode, not many
    # isolated short anchors whose +/- protection windows should be unioned.
    # Permit one intervening low sample, while invalid/missing evidence still
    # closes the episode immediately.
    weak_anchor_gap_sec = max(
        max_reliable_gap_sec,
        2.0 * nominal_period_sec + 1e-6,
    )

    def close_weak_run() -> None:
        nonlocal current_weak
        if current_weak:
            weak_runs.append(current_weak)
            current_weak = []

    for item in samples:
        sample_gap_ok = (
            previous_sample_timestamp is None
            or item.timestamp - previous_sample_timestamp <= max_reliable_gap_sec
        )
        is_weak = (
            item.valid
            and weak_active_count <= item.active_count < strong_active_count
        )
        if not sample_gap_ok or not item.valid or item.active_count >= strong_active_count:
            close_weak_run()
        elif is_weak:
            if (
                current_weak
                and item.timestamp - current_weak[-1].timestamp > weak_anchor_gap_sec
            ):
                close_weak_run()
            current_weak.append(item)
        else:
            if (
                current_weak
                and item.timestamp - current_weak[-1].timestamp > weak_anchor_gap_sec
            ):
                close_weak_run()
        previous_sample_timestamp = item.timestamp
    close_weak_run()
    for run in weak_runs:
        covered = run[-1].timestamp - run[0].timestamp + nominal_period_sec
        if covered <= weak_max_run_sec:
            protections.append(
                (
                    max(scope_start, run[0].timestamp - weak_protect_sec),
                    min(scope_end, run[-1].timestamp + weak_protect_sec),
                )
            )
    protections = _merge_intervals(protections)

    def protected(timestamp: float) -> bool:
        return any(start <= timestamp <= end for start, end in protections)

    low_runs: list[list[SparseBurstEvidence]] = []
    current_low: list[SparseBurstEvidence] = []
    previous_timestamp: float | None = None
    for item in samples:
        gap_ok = (
            previous_timestamp is None
            or item.timestamp - previous_timestamp <= max_reliable_gap_sec
        )
        low = item.valid and item.active_count <= low_active_count and not protected(item.timestamp)
        if low and gap_ok:
            current_low.append(item)
        else:
            if current_low:
                low_runs.append(current_low)
            current_low = [item] if low else []
        previous_timestamp = item.timestamp
    if current_low:
        low_runs.append(current_low)

    half_period = nominal_period_sec / 2.0
    vetoes: list[Interval] = []
    for run in low_runs:
        start = max(scope_start, run[0].timestamp - half_period)
        end = min(scope_end, run[-1].timestamp + half_period)
        if end - start >= min_low_activity_sec:
            vetoes.append((start, end))
    return _merge_intervals(vetoes)


def subtract_intervals(intervals: Sequence[Interval], vetoes: Sequence[Interval]) -> list[Interval]:
    """Subtract normalized ``vetoes`` from arbitrary source ``intervals``."""

    sources = _merge_intervals(intervals)
    cuts = _merge_intervals(vetoes)
    output: list[Interval] = []
    for source_start, source_end in sources:
        cursor = source_start
        for cut_start, cut_end in cuts:
            if cut_end <= cursor:
                continue
            if cut_start >= source_end:
                break
            if cut_start > cursor:
                output.append((cursor, min(cut_start, source_end)))
            cursor = max(cursor, cut_end)
            if cursor >= source_end:
                break
        if cursor < source_end:
            output.append((cursor, source_end))
    return output


__all__ = [
    "SparseBurstEvidence",
    "SparseFlowResult",
    "analyze_sparse_flow",
    "sleep_veto_intervals",
    "subtract_intervals",
]
