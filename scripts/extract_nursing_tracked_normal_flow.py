"""Extract track-aware normal reciprocating motion inside the nursing-band ROI.

The signal path deliberately uses three time scales:

* dense optical flow (normally 12.5 Hz) preserves the nursing-motion bandwidth;
* YOLO26 masks (normally about 4 Hz) maintain the dynamic nursing ROI and the
  original global flow features;
* sparse OSNet association keyframes use the sample rate stored in the tracker
  checkpoint (normally 2 Hz).

Full-frame YOLO masks are associated before they are intersected with the
nursing ROI.  Dense per-observation flow measurements are cached during the
video pass and mapped to offline track IDs afterwards.  Outputs are continuous
measurements only; this script does not classify nursing behaviour.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
from scipy.signal import find_peaks, savgol_filter


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import extract_nursing_normal_flow as flow_base  # noqa: E402
from scripts import extract_nursing_normal_flow_features as base  # noqa: E402
from scripts.extract_nursing_band import (  # noqa: E402
    InstanceMask,
    mask_for_shape,
    predict_labeled_instances_batch,
    sow_mask_change_ratio,
)
from scripts.track_yolo26_masks_offline import (  # noqa: E402
    FrameCandidate,
    enforce_frame_identity_cap,
    select_device,
)
from ultralytics.trackers.mask_aware import (  # noqa: E402
    MaskAwareAssociationModel,
    MaskAwareOfflineTracker,
    MaskObservation,
    encode_observation_crops,
    extract_masked_crop,
    load_association_checkpoint,
    tracker_config_from_dict,
)


DEFAULT_OUTPUT = ROOT / "runs" / "nursing_tracked_normal_flow"
DEFAULT_TRACKER_CHECKPOINT = (
    ROOT / "models" / "tracking" / "osnet_x0_25.pt"
)


TRACK_INSTANT_FIELDS = (
    "track_source_contact_pixel_count",
    "track_stable_pixel_count",
    "track_stable_contact_fraction",
    "track_valid",
    "track_normal_signed_mean_px_s",
    "track_normal_signed_median_px_s",
    "track_normal_abs_p75_px_s",
    "track_normal_rms_px_s",
    "track_tangent_signed_mean_px_s",
    "track_tangent_rms_px_s",
    "track_normal_energy_excess_px2_s2",
    "track_tangent_energy_excess_px2_s2",
    "track_normal_excess_rms_px_s",
    "track_tangent_excess_rms_px_s",
    "track_normal_energy_fraction",
)

TRACK_FILTER_FIELDS = (
    "track_hampel_normal_signed_mean_px_s",
    "track_filtered_normal_signed_mean_px_s",
    "track_filter_outlier",
    "track_filter_residual_px_s",
    "track_extremum_kept",
    "track_extremum_type",
    "track_extremum_prominence_px_s",
)

TRACK_TEMPORAL_FIELDS = (
    "track_segment_id",
    "track_window_start_sec",
    "track_window_end_sec",
    "track_feature_valid_fraction",
    "track_window_peak_count",
    "track_window_trough_count",
    "track_window_extrema_cycle_count",
    "track_extrema_frequency_hz",
    "track_window_normal_intensity_median_px_s",
    "track_window_signed_signal_rms_px_s",
    "track_bidirectional_energy_balance",
    "track_cycle_closure",
    "track_dominant_frequency_hz",
    "track_autocorrelation_peak",
    "track_autocorrelation_peak_prominence",
    "track_halfcycle_antisymmetry",
    "track_estimated_cycles_in_window",
    "track_spectral_dominant_frequency_hz",
    "track_periodic_band_power_fraction",
    "track_spectral_peak_concentration",
    "track_periodicity_quality",
    "track_periodic_intensity_px_s",
    "track_quality_gated_frequency_hz",
)

TRACK_AGGREGATE_FIELDS = (
    "tracked_source_track_count",
    "tracked_valid_track_count",
    "tracked_valid_track_fraction",
    "tracked_stable_pixel_count",
    "tracked_normal_signed_mean_px_s",
    "tracked_filtered_normal_signed_mean_px_s",
    "tracked_normal_signed_abs_mean_px_s",
    "tracked_normal_rms_median_px_s",
    "tracked_normal_rms_p75_px_s",
    "tracked_tangent_rms_median_px_s",
    "tracked_tangent_rms_p75_px_s",
    "tracked_phase_cancellation_index",
    "tracked_filtered_phase_cancellation_index",
    "tracked_outward_track_fraction",
    "tracked_normal_dominant_track_fraction",
    "tracked_filter_outlier_fraction",
    "tracked_extremum_count",
    "tracked_peak_count",
    "tracked_trough_count",
    "tracked_temporal_valid_track_count",
    "tracked_temporal_valid_track_fraction",
    "tracked_periodicity_quality_median",
    "tracked_periodicity_quality_p75",
    "tracked_periodic_intensity_median_px_s",
    "tracked_periodic_intensity_p75_px_s",
    "tracked_periodic_intensity_weighted_mean_px_s",
    "tracked_reliable_track_fraction",
    "tracked_reliable_frequency_q25_hz",
    "tracked_reliable_frequency_median_hz",
    "tracked_reliable_frequency_q75_hz",
    "tracked_extrema_frequency_median_hz",
)

TRACK_CSV_FIELDS = (
    "video",
    "sample_index",
    "prev_frame_index",
    "frame_index",
    "interval_start_sec",
    "timestamp_sec",
    "interval_end_sec",
    "dt_sec",
    "actual_flow_fps",
    "roi_segment_id",
    "observation_id",
    "track_id",
    "capped_track_id",
    "tracking_frame_index",
    "tracking_source_frame_index",
    "tracking_timestamp_sec",
    "confidence",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "full_mask_area_px",
    "roi_contact_area_px",
    "roi_contact_fraction",
    *TRACK_INSTANT_FIELDS,
    *TRACK_FILTER_FIELDS,
    *TRACK_TEMPORAL_FIELDS,
)

CSV_FIELDS = (*base.CSV_FIELDS, *TRACK_AGGREGATE_FIELDS)


@dataclass(slots=True)
class TrackerResources:
    model: MaskAwareAssociationModel
    checkpoint: dict[str, Any]
    device: torch.device
    training_sample_fps: float
    checkpoint_sha256: str


@dataclass(frozen=True, slots=True)
class TrackFlowContext:
    """Frame-level arrays shared by every track measurement in one flow interval."""

    stable_destination_support: np.ndarray
    normal_compensated_px_s: np.ndarray | None
    tangent_compensated_px_s: np.ndarray | None
    reference_normal_energy_px2_s2: float
    reference_tangent_energy_px2_s2: float


@dataclass(frozen=True)
class TrackedFeatureFrameState:
    frame_index: int
    timestamp_sec: float
    segment_id: int
    mask_refreshed: bool
    gray_crop: np.ndarray
    contact_mask_crop: np.ndarray
    sow_reference_mask_crop: np.ndarray
    piglet_count: int
    contact_piglet_count: int
    contact_pixel_count: int
    track_contact_masks_crop: dict[int, np.ndarray]


def parse_args() -> argparse.Namespace:
    """Parse tracker-specific flags, then reuse the feature pipeline CLI."""
    custom = argparse.ArgumentParser(add_help=False)
    custom.add_argument("--tracker-checkpoint", type=Path, default=DEFAULT_TRACKER_CHECKPOINT)
    custom.add_argument(
        "--tracking-fps",
        type=float,
        default=0.0,
        help="Association keyframe FPS; 0 uses the checkpoint training rate (normally 2 FPS).",
    )
    custom.add_argument(
        "--yolo-batch-size",
        type=int,
        default=48,
        help="Piglet YOLO batch size; 32 is tuned for a 24 GB RTX 4090 at imgsz=1024.",
    )
    custom.add_argument(
        "--flow-method",
        choices=("farneback", "dis-medium", "dis-fast"),
        default="farneback",
        help="Dense-flow backend. Farneback preserves the original numerical baseline; DIS is faster.",
    )
    custom.add_argument(
        "--flow-scale",
        type=float,
        default=1.0,
        help="Optical-flow resolution scale in (0, 1]; values below 1 trade spatial detail for speed.",
    )
    custom.add_argument(
        "--torch-compile",
        choices=("none", "default", "reduce-overhead", "max-autotune-no-cudagraphs"),
        default="none",
        help="Optional torch.compile mode for long CUDA runs; compilation has a one-time startup cost.",
    )
    custom.add_argument("--tracker-batch-size", type=int, default=256)
    custom.add_argument(
        "--max-frame-instances",
        type=int,
        default=16,
        help="Per-keyframe candidate cap after conservative fragment fusion; 0 disables it.",
    )
    custom.add_argument(
        "--max-track-identities",
        type=int,
        default=0,
        help=(
            "Optional reusable display-ID pool written as capped_track_id. Signal windows always use "
            "uncapped IDs. Default 0 disables the second capped association pass."
        ),
    )
    custom.add_argument("--max-tracker-observations", type=int, default=50000)
    custom.add_argument("--tracker-match-threshold", type=float, default=0.20)
    custom.add_argument("--tracker-max-age", type=int, default=2)
    custom.add_argument("--tracker-max-center-distance", type=float, default=0.30)
    custom.add_argument("--tracker-motion-weight", type=float, default=2.0)
    custom.add_argument("--tracker-no-offline", action="store_true")
    custom.add_argument("--tracker-offline-max-gap", type=int, default=8)
    custom.add_argument("--tracker-offline-match-threshold", type=float, default=0.25)
    custom.add_argument("--tracker-offline-max-center-distance", type=float, default=0.45)
    custom.add_argument("--tracker-offline-max-tracklets", type=int, default=5000)
    custom.add_argument("--min-track-flow-pixels", type=int, default=24)
    custom.add_argument(
        "--track-feature-step-sec",
        type=float,
        default=0.25,
        help="Time spacing between per-track temporal-window evaluations.",
    )
    custom.add_argument(
        "--track-feature-window-sec",
        type=float,
        default=4.0,
        help=(
            "Centered per-track temporal-feature window. The 4 s default leaves about 2 s "
            "of unavailable periodic features on either side of an ROI refresh."
        ),
    )
    custom.add_argument(
        "--track-signal-max-gap-sec",
        type=float,
        default=0.30,
        help="Break a per-track waveform across a longer observation gap.",
    )
    custom.add_argument(
        "--track-hampel-window-sec",
        type=float,
        default=0.56,
        help="Local Hampel/MAD window for isolated signed-flow spikes; 0 disables spike replacement.",
    )
    custom.add_argument(
        "--track-hampel-sigma",
        type=float,
        default=3.5,
        help="Robust-sigma threshold used by the per-track Hampel filter; 0 disables it.",
    )
    custom.add_argument(
        "--track-smooth-method",
        choices=("none", "median", "savgol"),
        default="savgol",
        help="Second-stage smoothing after Hampel replacement (default: shape-preserving Savitzky-Golay).",
    )
    custom.add_argument(
        "--track-smooth-window-sec",
        type=float,
        default=0.32,
        help="Centered smoothing window in seconds; 0 disables the second stage.",
    )
    custom.add_argument(
        "--track-savgol-polyorder",
        type=int,
        default=2,
        help="Savitzky-Golay polynomial order.",
    )
    custom.add_argument(
        "--track-extrema-min-prominence-mad",
        type=float,
        default=0.8,
        help="Minimum peak/trough prominence as a multiple of the run's robust MAD scale.",
    )
    custom.add_argument(
        "--track-extrema-min-distance-sec",
        type=float,
        default=0.0,
        help="Minimum spacing between retained extrema; 0 derives half a cycle from --feature-max-frequency-hz.",
    )
    custom.add_argument(
        "--no-retina-masks",
        action="store_true",
        help="Disable full-resolution YOLO masks. Retina masks are enabled by default for tracking.",
    )
    custom.add_argument(
        "--resume",
        action="store_true",
        help="Skip videos with a complete output set and safely recompute incomplete output sets.",
    )
    custom.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not render per-video PNG plots (recommended for very large rolling runs).",
    )
    custom.add_argument(
        "--no-prefetch",
        action="store_true",
        help="Disable the one-batch decode/YOLO prefetch pipeline.",
    )
    custom.add_argument(
        "--chunk-seconds",
        type=float,
        default=0.0,
        help="Split each source video into independent chunks; 0 keeps one output set per video.",
    )
    custom_args, remaining = custom.parse_known_args()
    if "-h" in remaining or "--help" in remaining:
        print("Track-aware options:")
        custom.print_help()
        print("\nShared normal-flow options:")
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *remaining]
        args = base.parse_args()
    finally:
        sys.argv = original_argv
    if not any(item == "--output" or item.startswith("--output=") for item in remaining):
        args.output = DEFAULT_OUTPUT
    for name, value in vars(custom_args).items():
        setattr(args, name, value)
    if not args.no_retina_masks:
        args.retina_masks = True
    return args


def validate_args(args: argparse.Namespace) -> None:
    base.validate_args(args)
    if args.tracking_fps < 0:
        raise ValueError("--tracking-fps must be non-negative; use 0 for checkpoint FPS.")
    if args.yolo_batch_size <= 0 or args.tracker_batch_size <= 0 or args.max_tracker_observations <= 0:
        raise ValueError("YOLO/tracker batch sizes and observation limit must be positive.")
    if not 0 < args.flow_scale <= 1:
        raise ValueError("--flow-scale must be in (0, 1].")
    if args.max_frame_instances < 0 or args.max_track_identities < 0:
        raise ValueError("Tracking identity limits must be non-negative.")
    if args.max_track_identities and (
        args.max_frame_instances == 0 or args.max_track_identities < args.max_frame_instances
    ):
        raise ValueError(
            "A capped display-ID pool must be at least --max-frame-instances; "
            "for a 15-ID pool also set --max-frame-instances 15."
        )
    if args.min_track_flow_pixels <= 0:
        raise ValueError("--min-track-flow-pixels must be positive.")
    if (
        args.track_feature_step_sec <= 0
        or args.track_feature_window_sec <= 0
        or args.track_signal_max_gap_sec <= 0
    ):
        raise ValueError("Track feature window, step, and signal gap must be positive.")
    if args.track_hampel_window_sec < 0 or args.track_hampel_sigma < 0:
        raise ValueError("Track Hampel window and sigma must be non-negative.")
    if args.track_smooth_window_sec < 0 or args.track_savgol_polyorder < 0:
        raise ValueError("Track smoothing window and Savitzky-Golay order must be non-negative.")
    if args.track_extrema_min_prominence_mad < 0 or args.track_extrema_min_distance_sec < 0:
        raise ValueError("Track extrema prominence and distance must be non-negative.")
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive.")
    if args.chunk_seconds < 0:
        raise ValueError("--chunk-seconds must be non-negative.")
    tracker_config_from_dict(
        {
            "match_threshold": args.tracker_match_threshold,
            "max_age": args.tracker_max_age,
            "max_center_distance": args.tracker_max_center_distance,
            "motion_weight": args.tracker_motion_weight,
            "offline": not args.tracker_no_offline,
            "offline_max_gap": args.tracker_offline_max_gap,
            "offline_match_threshold": args.tracker_offline_match_threshold,
            "offline_max_center_distance": args.tracker_offline_max_center_distance,
            "offline_max_tracklets": args.tracker_offline_max_tracklets,
            "max_identities": args.max_track_identities,
        }
    )


def _as_finite(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def phase_cancellation_index(values: np.ndarray, weights: np.ndarray) -> float | None:
    """Return 0 for aligned signs and 1 for complete weighted signed cancellation."""
    signed = np.asarray(values, dtype=float)
    weight = np.asarray(weights, dtype=float)
    valid = np.isfinite(signed) & np.isfinite(weight) & (weight > 0)
    if not np.any(valid):
        return None
    numerator = abs(float(np.sum(signed[valid] * weight[valid])))
    denominator = float(np.sum(np.abs(signed[valid]) * weight[valid]))
    if denominator <= 1e-12:
        return None
    return float(np.clip(1.0 - numerator / denominator, 0.0, 1.0))


def _empty_track_features(source_pixels: int = 0, stable_pixels: int = 0) -> dict[str, Any]:
    result = {field: None for field in TRACK_INSTANT_FIELDS}
    result.update(
        {
            "track_source_contact_pixel_count": int(source_pixels),
            "track_stable_pixel_count": int(stable_pixels),
            "track_stable_contact_fraction": (
                float(stable_pixels / source_pixels) if source_pixels else 0.0
            ),
            "track_valid": 0,
        }
    )
    return result


def _compensation_values(
    compensation: dict[str, Any] | Sequence[float] | None,
) -> tuple[np.ndarray, float, float, float, float] | None:
    if compensation is None:
        return None
    if not isinstance(compensation, dict):
        values = tuple(float(value) for value in compensation)
        if len(values) != 2 or not np.isfinite(values).all():
            return None
        return np.asarray(values, dtype=np.float64), 0.0, 0.0, 0.0, 0.0
    required = (
        _as_finite(compensation.get("sow_rigid_flow_x_px_s")),
        _as_finite(compensation.get("sow_rigid_flow_y_px_s")),
        _as_finite(compensation.get("sow_residual_normal_median_px_s")),
        _as_finite(compensation.get("sow_residual_tangent_median_px_s")),
        _as_finite(compensation.get("sow_residual_normal_energy_px2_s2")),
        _as_finite(compensation.get("sow_residual_tangent_energy_px2_s2")),
    )
    if any(value is None for value in required):
        return None
    return (
        np.asarray(required[:2], dtype=np.float64),
        float(required[2]),
        float(required[3]),
        float(required[4]),
        float(required[5]),
    )


def track_flow_features(
    flow: np.ndarray,
    normal_field: np.ndarray,
    source_mask: np.ndarray,
    destination_mask: np.ndarray,
    dt_sec: float,
    compensation: dict[str, Any] | Sequence[float] | None,
    min_pixels: int = 24,
    winsor_quantile: float = 0.99,
    residual_alpha: float = 1.0,
    context: TrackFlowContext | None = None,
) -> dict[str, Any]:
    """Measure one source observation's stable, sow-compensated normal flow."""
    if dt_sec <= 0:
        raise ValueError("dt_sec must be positive.")
    if flow.shape != normal_field.shape or flow.shape[:2] != source_mask.shape:
        raise ValueError("Flow, normal field, and source mask shapes do not match.")
    if destination_mask.shape != source_mask.shape:
        raise ValueError("Source and destination masks do not match.")
    source = np.asarray(source_mask, dtype=bool)
    source_count = int(np.count_nonzero(source))
    if context is None:
        context = prepare_track_flow_context(
            flow,
            normal_field,
            destination_mask,
            dt_sec,
            compensation,
        )
    elif context.stable_destination_support.shape != source.shape:
        raise ValueError("Track-flow context and source mask shapes do not match.")
    stable = source & context.stable_destination_support
    stable_count = int(np.count_nonzero(stable))
    result = _empty_track_features(source_count, stable_count)
    if (
        stable_count < max(1, int(min_pixels))
        or context.normal_compensated_px_s is None
        or context.tangent_compensated_px_s is None
    ):
        return result

    normal = context.normal_compensated_px_s[stable]
    tangent = context.tangent_compensated_px_s[stable]
    normal_w = base._winsorized(normal, winsor_quantile)
    tangent_w = base._winsorized(tangent, winsor_quantile)
    normal_energy = float(np.mean(normal_w**2))
    tangent_energy = float(np.mean(tangent_w**2))
    normal_excess = (
        normal_energy - float(residual_alpha) * context.reference_normal_energy_px2_s2
    )
    tangent_excess = (
        tangent_energy - float(residual_alpha) * context.reference_tangent_energy_px2_s2
    )
    positive_normal = max(0.0, normal_excess)
    positive_tangent = max(0.0, tangent_excess)
    energy_sum = positive_normal + positive_tangent
    result.update(
        {
            "track_valid": 1,
            "track_normal_signed_mean_px_s": float(np.mean(normal_w)),
            "track_normal_signed_median_px_s": float(np.median(normal_w)),
            "track_normal_abs_p75_px_s": float(np.percentile(np.abs(normal_w), 75)),
            "track_normal_rms_px_s": float(np.sqrt(normal_energy)),
            "track_tangent_signed_mean_px_s": float(np.mean(tangent_w)),
            "track_tangent_rms_px_s": float(np.sqrt(tangent_energy)),
            "track_normal_energy_excess_px2_s2": normal_excess,
            "track_tangent_energy_excess_px2_s2": tangent_excess,
            "track_normal_excess_rms_px_s": float(np.sqrt(positive_normal)),
            "track_tangent_excess_rms_px_s": float(np.sqrt(positive_tangent)),
            "track_normal_energy_fraction": (
                float(positive_normal / energy_sum) if energy_sum > 1e-12 else None
            ),
        }
    )
    return result


def prepare_track_flow_context(
    flow: np.ndarray,
    normal_field: np.ndarray,
    destination_mask: np.ndarray,
    dt_sec: float,
    compensation: dict[str, Any] | Sequence[float] | None,
) -> TrackFlowContext:
    """Precompute full-frame quantities that are identical for all tracks."""
    if dt_sec <= 0:
        raise ValueError("dt_sec must be positive.")
    if flow.shape != normal_field.shape or flow.shape[:2] != destination_mask.shape:
        raise ValueError("Flow, normal field, and destination mask shapes do not match.")
    finite = np.isfinite(flow).all(axis=2) & np.isfinite(normal_field).all(axis=2)
    field_valid = np.sum(normal_field * normal_field, axis=2) > 1e-12
    supported = base.destination_supported_source_mask(flow, destination_mask)
    stable_support = supported & finite & field_valid
    values = _compensation_values(compensation)
    if values is None:
        return TrackFlowContext(stable_support, None, None, 0.0, 0.0)

    sow_rigid, residual_normal, residual_tangent, reference_normal_energy, reference_tangent_energy = values
    velocity = flow.astype(np.float64) / float(dt_sec)
    normals = normal_field.astype(np.float64)
    relative_x = velocity[..., 0] - sow_rigid[0]
    relative_y = velocity[..., 1] - sow_rigid[1]
    normal = relative_x * normals[..., 0] + relative_y * normals[..., 1] - residual_normal
    tangent = -relative_x * normals[..., 1] + relative_y * normals[..., 0] - residual_tangent
    return TrackFlowContext(
        stable_support,
        normal,
        tangent,
        reference_normal_energy,
        reference_tangent_energy,
    )


def propagate_track_masks(
    flow: np.ndarray,
    source_masks: dict[int, np.ndarray],
    destination_union: np.ndarray,
) -> dict[int, np.ndarray]:
    """Forward-warp track masks one dense-flow step and constrain them to current YOLO support."""
    if flow.ndim != 3 or flow.shape[2] != 2 or flow.shape[:2] != destination_union.shape:
        raise ValueError("Flow and destination mask shapes do not match.")
    height, width = destination_union.shape
    support = np.asarray(destination_union, dtype=bool)
    finite_flow = np.isfinite(flow).all(axis=2)
    candidates: dict[int, np.ndarray] = {}
    for observation_id, source_mask in source_masks.items():
        source = np.asarray(source_mask, dtype=bool)
        if source.shape != support.shape:
            raise ValueError("Source track mask shape does not match flow.")
        y, x = np.nonzero(source & finite_flow)
        if not len(x):
            continue
        destination_x = np.rint(x.astype(np.float32) + flow[y, x, 0]).astype(np.int32)
        destination_y = np.rint(y.astype(np.float32) + flow[y, x, 1]).astype(np.int32)
        valid = (
            (destination_x >= 0)
            & (destination_x < width)
            & (destination_y >= 0)
            & (destination_y < height)
        )
        warped = np.zeros((height, width), dtype=bool)
        warped[destination_y[valid], destination_x[valid]] = True
        warped &= support
        if warped.any():
            candidates[int(observation_id)] = warped

    # Forward splatting can make adjacent masks collide.  Resolve ownership
    # deterministically so a pixel is never counted in two track signals.
    owned = np.zeros((height, width), dtype=bool)
    result: dict[int, np.ndarray] = {}
    for observation_id in sorted(candidates):
        mask = candidates[observation_id] & ~owned
        if mask.any():
            result[observation_id] = mask
            owned |= mask
    return result


def periodicity_quality(features: dict[str, Any]) -> float | None:
    values = (
        _as_finite(features.get("autocorrelation_peak")),
        _as_finite(features.get("periodic_band_power_fraction")),
        _as_finite(features.get("spectral_peak_concentration")),
    )
    if any(value is None for value in values):
        return None
    return float((max(0.0, values[0]) * max(0.0, values[1]) * max(0.0, values[2])) ** (1.0 / 3.0))


def _track_intensity(row: dict[str, Any]) -> float:
    value = _as_finite(row.get("track_normal_excess_rms_px_s"))
    if value is None:
        value = _as_finite(row.get("track_normal_rms_px_s"))
    return float(value) if value is not None else math.nan


def _finite_signal_runs(values: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open finite runs so filters never bridge invalid measurements."""
    finite = np.isfinite(np.asarray(values, dtype=float))
    changes = np.diff(np.concatenate(([False], finite, [False])).astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return [(int(start), int(stop)) for start, stop in zip(starts, stops)]


def _smoothing_window(run_length: int, requested: int, polyorder: int) -> int:
    width = min(int(run_length), max(1, int(requested)))
    if width % 2 == 0:
        width -= 1
    minimum = int(polyorder) + 2
    if minimum % 2 == 0:
        minimum += 1
    return width if width >= minimum else 1


def _filter_extrema(
    values: np.ndarray,
    sample_rate_hz: float,
    prominence_mad: float,
    min_distance_sec: float,
    max_frequency_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect robust, separated, alternating peaks/troughs in finite signal runs."""
    signal = np.asarray(values, dtype=float)
    types = np.full(len(signal), "", dtype=object)
    prominences = np.full(len(signal), np.nan, dtype=float)
    if sample_rate_hz <= 0 or max_frequency_hz <= 0:
        return types, prominences
    distance_sec = (
        float(min_distance_sec)
        if min_distance_sec > 0
        else 0.5 / float(max_frequency_hz)
    )
    distance = max(1, int(round(distance_sec * float(sample_rate_hz))))
    for start, stop in _finite_signal_runs(signal):
        run = signal[start:stop]
        if len(run) < 5:
            continue
        median = float(np.median(run))
        mad = float(np.median(np.abs(run - median)))
        reference_scale = max(float(np.median(np.abs(run))), 0.01 * float(np.ptp(run)), 1e-6)
        robust_scale = max(1.4826 * mad, 0.02 * reference_scale)
        threshold = float(prominence_mad) * robust_scale
        peak_indices, peak_properties = find_peaks(run, prominence=threshold)
        trough_indices, trough_properties = find_peaks(-run, prominence=threshold)
        candidates = [
            (int(index), "peak", float(prominence))
            for index, prominence in zip(peak_indices, peak_properties["prominences"])
        ]
        candidates.extend(
            (int(index), "trough", float(prominence))
            for index, prominence in zip(trough_indices, trough_properties["prominences"])
        )

        # Non-maximum suppression applies the distance across both signs.  A
        # second pass enforces physical peak/trough alternation.
        selected: list[tuple[int, str, float]] = []
        for candidate in sorted(candidates, key=lambda item: item[2], reverse=True):
            if all(abs(candidate[0] - kept[0]) >= distance for kept in selected):
                selected.append(candidate)
        alternating: list[tuple[int, str, float]] = []
        for candidate in sorted(selected, key=lambda item: item[0]):
            if alternating and candidate[1] == alternating[-1][1]:
                if candidate[2] > alternating[-1][2]:
                    alternating[-1] = candidate
            else:
                alternating.append(candidate)
        for index, extremum_type, prominence in alternating:
            global_index = start + index
            types[global_index] = extremum_type
            prominences[global_index] = prominence
    return types, prominences


def preprocess_track_signal(
    values: np.ndarray,
    sample_rate_hz: float,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    """Hampel-clean, smooth, then annotate filtered extrema without overwriting raw values."""
    raw = np.asarray(values, dtype=float)
    hampel = raw.copy()
    filtered = raw.copy()
    outliers = np.zeros(len(raw), dtype=np.uint8)
    hampel_seconds = float(getattr(args, "track_hampel_window_sec", 0.56))
    hampel_sigma = float(getattr(args, "track_hampel_sigma", 3.5))
    hampel_width = (
        base._odd_window(hampel_seconds, sample_rate_hz, minimum=5)
        if hampel_seconds > 0 and hampel_sigma > 0
        else 1
    )
    smooth_method = str(getattr(args, "track_smooth_method", "savgol"))
    smooth_seconds = float(getattr(args, "track_smooth_window_sec", 0.32))
    polyorder = int(getattr(args, "track_savgol_polyorder", 2))
    requested_smooth_width = (
        base._odd_window(smooth_seconds, sample_rate_hz)
        if smooth_seconds > 0 and smooth_method != "none"
        else 1
    )
    for start, stop in _finite_signal_runs(raw):
        run = raw[start:stop]
        cleaned = base._hampel_filter(run, hampel_width, hampel_sigma)
        hampel[start:stop] = cleaned
        outliers[start:stop] = (~np.isclose(cleaned, run, rtol=1e-9, atol=1e-12)).astype(np.uint8)
        if requested_smooth_width <= 1:
            filtered[start:stop] = cleaned
        elif smooth_method == "median":
            filtered[start:stop] = base._centered_nanmedian(
                cleaned,
                requested_smooth_width,
                preserve_nan=True,
            )
        elif smooth_method == "savgol":
            width = _smoothing_window(len(cleaned), requested_smooth_width, polyorder)
            filtered[start:stop] = (
                savgol_filter(cleaned, window_length=width, polyorder=polyorder, mode="mirror")
                if width > 1
                else cleaned
            )
        else:
            filtered[start:stop] = cleaned

    extrema_types, extrema_prominences = _filter_extrema(
        filtered,
        sample_rate_hz,
        float(getattr(args, "track_extrema_min_prominence_mad", 0.8)),
        float(getattr(args, "track_extrema_min_distance_sec", 0.0)),
        float(args.feature_max_frequency_hz),
    )
    return {
        "hampel": hampel,
        "filtered": filtered,
        "outliers": outliers,
        "extrema_types": extrema_types,
        "extrema_prominences": extrema_prominences,
    }


def add_track_temporal_features(track_rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    """Compute waveform features per contiguous ``(track ID, ROI segment)`` run."""
    for row in track_rows:
        row.update({field: None for field in TRACK_FILTER_FIELDS})
        row.update({field: None for field in TRACK_TEMPORAL_FIELDS})
    if not track_rows:
        return

    ordered = sorted(
        range(len(track_rows)),
        key=lambda index: (
            int(track_rows[index].get("track_id", 0)),
            float(track_rows[index]["timestamp_sec"]),
            int(track_rows[index].get("sample_index", 0)),
        ),
    )
    groups: dict[tuple[int, str, int], list[int]] = defaultdict(list)
    sequence_by_identity: dict[tuple[int, str], int] = defaultdict(int)
    last_time: dict[tuple[int, str], float] = {}
    max_gap = float(
        getattr(args, "track_signal_max_gap_sec", getattr(args, "track_max_gap_sec", 0.30))
    )
    for index in ordered:
        row = track_rows[index]
        track_id = int(row.get("track_id", 0))
        roi_segment = str(row.get("roi_segment_id", ""))
        timestamp = float(row["timestamp_sec"])
        identity = (track_id, roi_segment)
        previous_time = last_time.get(identity)
        if previous_time is not None and timestamp - previous_time > max_gap + 1e-9:
            sequence_by_identity[identity] += 1
        sequence = sequence_by_identity[identity]
        segment_id = f"{track_id}:{roi_segment}:{sequence}"
        row["track_segment_id"] = segment_id
        groups[(track_id, roi_segment, sequence)].append(index)
        last_time[identity] = timestamp

    feature_window = float(getattr(args, "track_feature_window_sec", args.feature_window_sec))
    half_window = feature_window / 2.0
    feature_step = float(getattr(args, "track_feature_step_sec", 0.25))
    for indices in groups.values():
        indices.sort(key=lambda index: float(track_rows[index]["timestamp_sec"]))
        times = np.asarray([float(track_rows[index]["timestamp_sec"]) for index in indices])
        signed = np.asarray(
            [
                float(value)
                if (
                    value := _as_finite(
                        track_rows[index].get("track_normal_signed_mean_px_s")
                    )
                )
                is not None
                else math.nan
                for index in indices
            ]
        )
        representative_fps = float(
            np.median([float(track_rows[index]["actual_flow_fps"]) for index in indices])
        )
        processed = preprocess_track_signal(signed, representative_fps, args)
        hampel = processed["hampel"]
        filtered = processed["filtered"]
        outliers = processed["outliers"]
        extrema_types = processed["extrema_types"]
        extrema_prominences = processed["extrema_prominences"]
        for local_index, row_index in enumerate(indices):
            raw_value = signed[local_index]
            hampel_value = hampel[local_index]
            filtered_value = filtered[local_index]
            extremum_type = str(extrema_types[local_index])
            prominence = extrema_prominences[local_index]
            track_rows[row_index].update(
                {
                    "track_hampel_normal_signed_mean_px_s": (
                        float(hampel_value) if np.isfinite(hampel_value) else None
                    ),
                    "track_filtered_normal_signed_mean_px_s": (
                        float(filtered_value) if np.isfinite(filtered_value) else None
                    ),
                    "track_filter_outlier": int(outliers[local_index]),
                    "track_filter_residual_px_s": (
                        float(raw_value - filtered_value)
                        if np.isfinite(raw_value) and np.isfinite(filtered_value)
                        else None
                    ),
                    "track_extremum_kept": int(bool(extremum_type)),
                    "track_extremum_type": extremum_type,
                    "track_extremum_prominence_px_s": (
                        float(prominence) if np.isfinite(prominence) else None
                    ),
                }
            )
        signed = filtered
        intensity = np.asarray([_track_intensity(track_rows[index]) for index in indices])
        last_center: float | None = None
        for local_index, row_index in enumerate(indices):
            center = float(times[local_index])
            if last_center is not None and center - last_center < feature_step - 1e-9:
                continue
            last_center = center
            row = track_rows[row_index]
            actual_fps = float(row["actual_flow_fps"])
            expected = max(1, int(round(feature_window * actual_fps)))
            window_start = center - half_window
            window_end = center + half_window
            left = int(np.searchsorted(times, window_start, side="left"))
            right = int(np.searchsorted(times, window_end, side="right"))
            local = np.arange(left, right)
            valid = np.isfinite(signed[local]) & np.isfinite(intensity[local])
            valid_fraction = float(np.count_nonzero(valid) / expected)
            peak_count = int(np.count_nonzero(extrema_types[local] == "peak"))
            trough_count = int(np.count_nonzero(extrema_types[local] == "trough"))
            extrema_local = local[extrema_types[local] != ""]
            extrema_frequency = None
            if len(extrema_local) >= 3:
                half_periods = np.diff(times[extrema_local])
                finite_half_periods = half_periods[np.isfinite(half_periods) & (half_periods > 0)]
                if len(finite_half_periods):
                    estimate = 1.0 / (2.0 * float(np.median(finite_half_periods)))
                    if args.feature_min_frequency_hz <= estimate <= args.feature_max_frequency_hz:
                        extrema_frequency = estimate
            row["track_window_start_sec"] = round(window_start, 6)
            row["track_window_end_sec"] = round(window_end, 6)
            row["track_feature_valid_fraction"] = min(1.0, valid_fraction)
            row["track_window_peak_count"] = peak_count
            row["track_window_trough_count"] = trough_count
            row["track_window_extrema_cycle_count"] = min(peak_count, trough_count)
            row["track_extrema_frequency_hz"] = None
            if valid_fraction < args.feature_min_valid_fraction or int(np.count_nonzero(valid)) < 8:
                continue
            valid_local = local[valid]
            valid_times = times[valid_local]
            frame_dt = 1.0 / actual_fps
            if valid_times[0] > window_start + 2.0 * frame_dt or valid_times[-1] < window_end - 2.0 * frame_dt:
                continue
            row["track_extrema_frequency_hz"] = extrema_frequency
            grid = np.linspace(window_start, window_end, expected)
            interpolated_signed = np.interp(grid, valid_times, signed[valid_local])
            interpolated_intensity = np.interp(grid, valid_times, intensity[valid_local])
            wave = base.temporal_wave_features(
                interpolated_signed,
                actual_fps,
                args.feature_min_frequency_hz,
                args.feature_max_frequency_hz,
            )
            quality = periodicity_quality(wave)
            signed_rms = _as_finite(wave.get("window_signed_signal_rms_px_s"))
            periodic_intensity = signed_rms * quality if signed_rms is not None and quality is not None else None
            autocorrelation = _as_finite(wave.get("autocorrelation_peak"))
            concentration = _as_finite(wave.get("spectral_peak_concentration"))
            frequency = None
            if (
                autocorrelation is not None
                and autocorrelation >= args.frequency_min_autocorrelation
                and concentration is not None
                and concentration >= args.frequency_min_peak_concentration
            ):
                frequency = _as_finite(wave.get("spectral_dominant_frequency_hz"))
            row.update(
                {
                    "track_window_normal_intensity_median_px_s": float(np.median(interpolated_intensity)),
                    "track_window_signed_signal_rms_px_s": signed_rms,
                    "track_bidirectional_energy_balance": wave.get("bidirectional_energy_balance"),
                    "track_cycle_closure": wave.get("cycle_closure"),
                    "track_dominant_frequency_hz": wave.get("dominant_frequency_hz"),
                    "track_autocorrelation_peak": autocorrelation,
                    "track_autocorrelation_peak_prominence": wave.get("autocorrelation_peak_prominence"),
                    "track_halfcycle_antisymmetry": wave.get("halfcycle_antisymmetry"),
                    "track_estimated_cycles_in_window": wave.get("estimated_cycles_in_window"),
                    "track_spectral_dominant_frequency_hz": wave.get("spectral_dominant_frequency_hz"),
                    "track_periodic_band_power_fraction": wave.get("periodic_band_power_fraction"),
                    "track_spectral_peak_concentration": concentration,
                    "track_periodicity_quality": quality,
                    "track_periodic_intensity_px_s": periodic_intensity,
                    "track_quality_gated_frequency_hz": frequency,
                }
            )


def _empty_track_aggregate() -> dict[str, Any]:
    result = {field: None for field in TRACK_AGGREGATE_FIELDS}
    result.update(
        {
            "tracked_source_track_count": 0,
            "tracked_valid_track_count": 0,
            "tracked_valid_track_fraction": 0.0,
            "tracked_stable_pixel_count": 0,
            "tracked_filter_outlier_fraction": 0.0,
            "tracked_extremum_count": 0,
            "tracked_peak_count": 0,
            "tracked_trough_count": 0,
            "tracked_temporal_valid_track_count": 0,
            "tracked_temporal_valid_track_fraction": 0.0,
            "tracked_reliable_track_fraction": 0.0,
        }
    )
    return result


def aggregate_track_rows(base_rows: list[dict[str, Any]], track_rows: list[dict[str, Any]]) -> None:
    """Attach non-cancelling per-track instant and temporal aggregates to dense rows."""
    by_sample: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for track_row in track_rows:
        by_sample[int(track_row["sample_index"])].append(track_row)
    for row in base_rows:
        aggregate = _empty_track_aggregate()
        local = by_sample.get(int(row["sample_index"]), [])
        aggregate["tracked_source_track_count"] = len(local)
        valid = [
            item
            for item in local
            if int(item.get("track_valid", 0) or 0)
            and _as_finite(item.get("track_normal_signed_mean_px_s")) is not None
        ]
        aggregate["tracked_valid_track_count"] = len(valid)
        aggregate["tracked_valid_track_fraction"] = float(len(valid) / len(local)) if local else 0.0
        aggregate["tracked_stable_pixel_count"] = sum(
            int(item.get("track_stable_pixel_count", 0) or 0) for item in valid
        )
        aggregate["tracked_extremum_count"] = sum(
            int(item.get("track_extremum_kept", 0) or 0) for item in valid
        )
        aggregate["tracked_peak_count"] = sum(
            str(item.get("track_extremum_type", "")) == "peak" for item in valid
        )
        aggregate["tracked_trough_count"] = sum(
            str(item.get("track_extremum_type", "")) == "trough" for item in valid
        )
        if valid:
            weights = np.asarray(
                [max(1, int(item.get("track_stable_pixel_count", 0) or 0)) for item in valid], dtype=float
            )
            signed = np.asarray([float(item["track_normal_signed_mean_px_s"]) for item in valid])
            filtered_signed = np.asarray(
                [
                    float(value)
                    if (value := _as_finite(item.get("track_filtered_normal_signed_mean_px_s")))
                    is not None
                    else float(item["track_normal_signed_mean_px_s"])
                    for item in valid
                ]
            )
            normal_rms = np.asarray([_track_intensity(item) for item in valid])
            tangent_rms = np.asarray(
                [
                    float(value)
                    if (value := _as_finite(item.get("track_tangent_excess_rms_px_s"))) is not None
                    else float(item.get("track_tangent_rms_px_s", math.nan))
                    for item in valid
                ]
            )
            aggregate.update(
                {
                    "tracked_normal_signed_mean_px_s": float(np.average(signed, weights=weights)),
                    "tracked_filtered_normal_signed_mean_px_s": float(
                        np.average(filtered_signed, weights=weights)
                    ),
                    "tracked_normal_signed_abs_mean_px_s": float(np.average(np.abs(signed), weights=weights)),
                    "tracked_normal_rms_median_px_s": float(np.nanmedian(normal_rms)),
                    "tracked_normal_rms_p75_px_s": float(np.nanpercentile(normal_rms, 75)),
                    "tracked_tangent_rms_median_px_s": float(np.nanmedian(tangent_rms)),
                    "tracked_tangent_rms_p75_px_s": float(np.nanpercentile(tangent_rms, 75)),
                    "tracked_phase_cancellation_index": phase_cancellation_index(signed, weights),
                    "tracked_filtered_phase_cancellation_index": phase_cancellation_index(
                        filtered_signed,
                        weights,
                    ),
                    "tracked_outward_track_fraction": float(np.mean(signed > 0)),
                    "tracked_normal_dominant_track_fraction": float(np.mean(normal_rms > tangent_rms)),
                    "tracked_filter_outlier_fraction": float(
                        np.mean([int(item.get("track_filter_outlier", 0) or 0) for item in valid])
                    ),
                }
            )

        temporal = [item for item in valid if _as_finite(item.get("track_periodicity_quality")) is not None]
        aggregate["tracked_temporal_valid_track_count"] = len(temporal)
        aggregate["tracked_temporal_valid_track_fraction"] = (
            float(len(temporal) / len(valid)) if valid else 0.0
        )
        if temporal:
            qualities = np.asarray([float(item["track_periodicity_quality"]) for item in temporal])
            aggregate["tracked_periodicity_quality_median"] = float(np.median(qualities))
            aggregate["tracked_periodicity_quality_p75"] = float(np.percentile(qualities, 75))
            periodic = [
                item for item in temporal if _as_finite(item.get("track_periodic_intensity_px_s")) is not None
            ]
            if periodic:
                intensity = np.asarray([float(item["track_periodic_intensity_px_s"]) for item in periodic])
                weights = np.asarray(
                    [max(1, int(item.get("track_stable_pixel_count", 0) or 0)) for item in periodic], dtype=float
                )
                aggregate["tracked_periodic_intensity_median_px_s"] = float(np.median(intensity))
                aggregate["tracked_periodic_intensity_p75_px_s"] = float(np.percentile(intensity, 75))
                aggregate["tracked_periodic_intensity_weighted_mean_px_s"] = float(
                    np.average(intensity, weights=weights)
                )
            frequencies = [
                float(item["track_quality_gated_frequency_hz"])
                for item in temporal
                if _as_finite(item.get("track_quality_gated_frequency_hz")) is not None
            ]
            aggregate["tracked_reliable_track_fraction"] = float(len(frequencies) / len(temporal))
            if frequencies:
                aggregate["tracked_reliable_frequency_q25_hz"] = float(np.percentile(frequencies, 25))
                aggregate["tracked_reliable_frequency_median_hz"] = float(np.median(frequencies))
                aggregate["tracked_reliable_frequency_q75_hz"] = float(np.percentile(frequencies, 75))
            extrema_frequencies = [
                float(item["track_extrema_frequency_hz"])
                for item in temporal
                if _as_finite(item.get("track_extrema_frequency_hz")) is not None
            ]
            if extrema_frequencies:
                aggregate["tracked_extrema_frequency_median_hz"] = float(
                    np.median(extrema_frequencies)
                )
        row.update(aggregate)


def sampling_schedule(
    source_fps: float,
    requested_flow_fps: float,
    requested_mask_fps: float,
    requested_tracking_fps: float,
) -> dict[str, float | int]:
    """Return nested flow/mask/tracker strides with every tracker keyframe on a mask refresh."""
    if min(source_fps, requested_flow_fps, requested_mask_fps, requested_tracking_fps) <= 0:
        raise ValueError("All sampling rates must be positive.")
    flow_step = max(1, int(round(source_fps / requested_flow_fps)))
    actual_flow_fps = source_fps / flow_step
    mask_stride = max(1, int(round(actual_flow_fps / requested_mask_fps)))
    actual_mask_fps = actual_flow_fps / mask_stride
    tracking_mask_stride = max(1, int(round(actual_mask_fps / requested_tracking_fps)))
    tracking_stride = mask_stride * tracking_mask_stride
    actual_tracking_fps = actual_flow_fps / tracking_stride
    return {
        "flow_step": flow_step,
        "actual_flow_fps": actual_flow_fps,
        "mask_stride": mask_stride,
        "actual_mask_fps": actual_mask_fps,
        "tracking_mask_stride": tracking_mask_stride,
        "tracking_stride": tracking_stride,
        "actual_tracking_fps": actual_tracking_fps,
    }


class DenseFlowEstimator:
    """Reusable dense-flow backend with optional spatial downscaling."""

    def __init__(self, method: str, scale: float = 1.0) -> None:
        self.method = str(method)
        self.scale = float(scale)
        if self.method == "dis-medium":
            self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        elif self.method == "dis-fast":
            self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        elif self.method == "farneback":
            self.dis = None
        else:
            raise ValueError(f"Unsupported dense-flow method: {method}")

    def calculate(self, previous_gray: np.ndarray, current_gray: np.ndarray) -> np.ndarray:
        if previous_gray.shape != current_gray.shape:
            raise ValueError("Dense-flow input frames must have the same shape.")
        height, width = previous_gray.shape
        if self.scale < 1.0:
            scaled_size = (
                max(8, int(round(width * self.scale))),
                max(8, int(round(height * self.scale))),
            )
            previous = cv2.resize(previous_gray, scaled_size, interpolation=cv2.INTER_AREA)
            current = cv2.resize(current_gray, scaled_size, interpolation=cv2.INTER_AREA)
        else:
            previous, current = previous_gray, current_gray
        previous = np.ascontiguousarray(previous)
        current = np.ascontiguousarray(current)
        if self.dis is None:
            flow = flow_base.farneback_flow(previous, current)
        else:
            flow = self.dis.calc(previous, current, None)
        if self.scale < 1.0:
            flow = cv2.resize(flow, (width, height), interpolation=cv2.INTER_LINEAR)
            flow /= self.scale
        return np.asarray(flow, dtype=np.float32)


@dataclass(slots=True)
class SampledFrame:
    frame_index: int
    frame: np.ndarray | None
    raw_piglets: list[InstanceMask] | None = None
    piglet_error: str | None = None


def _is_cuda_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def _infer_raw_piglets_batch(
    model: Any,
    frames: list[np.ndarray],
    args: argparse.Namespace,
) -> list[list[InstanceMask]]:
    """Run batched YOLO masks, halving an oversized batch automatically on CUDA OOM."""
    if not frames:
        return []
    values = vars(args).copy()
    values["batch_size"] = min(args.yolo_batch_size, len(frames))
    batch_args = argparse.Namespace(**values)
    try:
        batches = predict_labeled_instances_batch(
            model,
            frames,
            batch_args,
            args.piglet_label,
            start_index=1,
        )
    except Exception as exc:
        if not _is_cuda_oom(exc) or len(frames) == 1:
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        midpoint = len(frames) // 2
        return [
            *_infer_raw_piglets_batch(model, frames[:midpoint], args),
            *_infer_raw_piglets_batch(model, frames[midpoint:], args),
        ]
    return [
        [instance for instance in instances if instance.area >= args.min_piglet_area]
        for instances in batches
    ]


class BatchedSampleReader:
    """Decode only dense-flow samples and batch the embedded YOLO refresh frames."""

    def __init__(
        self,
        capture: cv2.VideoCapture,
        piglet_model: Any,
        args: argparse.Namespace,
        *,
        start_frame: int,
        end_frame: int,
        flow_step: int,
        mask_stride: int,
    ) -> None:
        self.capture = capture
        self.piglet_model = piglet_model
        self.args = args
        self.end_frame = int(end_frame)
        self.flow_step = int(flow_step)
        self.mask_stride = int(mask_stride)
        self.next_target = int(start_frame)
        self.decode_position = int(start_frame)
        self.next_sample_index = 0
        self.buffer: deque[SampledFrame] = deque()
        self.executor = None if args.no_prefetch else ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="nursing-prefetch",
        )
        self.future: Future[deque[SampledFrame]] | None = (
            self.executor.submit(self._fill) if self.executor is not None else None
        )

    def _read_targets(self, targets: list[int]) -> list[np.ndarray | None]:
        frames: list[np.ndarray | None] = []
        for target in targets:
            while self.decode_position < target:
                if not self.capture.grab():
                    return [*frames, *([None] * (len(targets) - len(frames)))]
                self.decode_position += 1
            ok, frame = self.capture.read()
            if not ok or frame is None:
                return [*frames, *([None] * (len(targets) - len(frames)))]
            frames.append(frame)
            self.decode_position += 1
        return frames

    def _fill(self) -> deque[SampledFrame]:
        result: deque[SampledFrame] = deque()
        if self.next_target >= self.end_frame:
            return result
        sample_capacity = max(1, self.args.yolo_batch_size * self.mask_stride)
        targets: list[int] = []
        while len(targets) < sample_capacity and self.next_target < self.end_frame:
            targets.append(self.next_target)
            self.next_target += self.flow_step
        first_sample_index = self.next_sample_index
        self.next_sample_index += len(targets)
        frames = self._read_targets(targets)
        mask_offsets = [
            offset
            for offset, frame in enumerate(frames)
            if frame is not None and (first_sample_index + offset) % self.mask_stride == 0
        ]
        piglets_by_offset: dict[int, list[InstanceMask]] = {}
        errors_by_offset: dict[int, str] = {}
        if mask_offsets:
            mask_frames = [frames[offset] for offset in mask_offsets]
            try:
                batches = _infer_raw_piglets_batch(self.piglet_model, mask_frames, self.args)
                piglets_by_offset.update(zip(mask_offsets, batches))
            except Exception as batch_exc:
                print(
                    f"[WARN] Piglet batch inference failed ({type(batch_exc).__name__}); "
                    "retrying frame by frame."
                )
                for offset in mask_offsets:
                    try:
                        piglets_by_offset[offset] = flow_base._infer_raw_piglets(
                            self.piglet_model,
                            frames[offset],
                            self.args,
                        )
                    except Exception as frame_exc:
                        errors_by_offset[offset] = type(frame_exc).__name__
        for offset, (target, frame) in enumerate(zip(targets, frames)):
            result.append(
                SampledFrame(
                    target,
                    frame,
                    piglets_by_offset.get(offset),
                    errors_by_offset.get(offset),
                )
            )
        return result

    def read(self) -> SampledFrame | None:
        if not self.buffer:
            if self.future is not None:
                self.buffer = self.future.result()
                self.future = None
            else:
                self.buffer = self._fill()
            if self.buffer and self.executor is not None and self.next_target < self.end_frame:
                self.future = self.executor.submit(self._fill)
        return self.buffer.popleft() if self.buffer else None

    def close(self) -> None:
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=False)
            self.executor = None


def _candidate_from_instance(instance: InstanceMask) -> FrameCandidate:
    mask = np.asarray(instance.mask, dtype=bool)
    rows = np.flatnonzero(np.any(mask, axis=1))
    columns = np.flatnonzero(np.any(mask, axis=0))
    if not len(rows) or not len(columns):
        raise ValueError("cannot compute a bounding box for an empty mask")
    return FrameCandidate(
        detection_indices=[int(instance.index)],
        class_id=int(instance.class_id),
        label=str(instance.label),
        confidence=float(instance.confidence),
        mask=mask,
        bbox_xyxy=np.asarray(
            (columns[0], rows[0], columns[-1] + 1, rows[-1] + 1),
            dtype=np.float32,
        ),
    )


def _observation_from_owned_mask(
    *,
    observation_id: int,
    frame_index: int,
    candidate: FrameCandidate,
    owned_mask: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
    area: int,
    centroid_xy: tuple[float, float],
    payload: dict[str, Any],
) -> MaskObservation:
    """Build the compact tracker observation from already-computed raster statistics."""
    x1, y1, x2, y2 = bbox_xyxy
    spatial = cv2.resize(
        owned_mask.astype(np.uint8),
        (160, 96),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    local = owned_mask[y1:y2, x1:x2]
    shape = cv2.resize(local.astype(np.uint8), (96, 48), interpolation=cv2.INTER_NEAREST).astype(bool)
    bbox_area = max(float((x2 - x1) * (y2 - y1)), 1.0)
    return MaskObservation(
        observation_id=int(observation_id),
        frame_index=int(frame_index),
        class_id=int(candidate.class_id),
        label=str(candidate.label),
        score=float(candidate.confidence),
        bbox_xyxy=np.asarray(bbox_xyxy, dtype=np.float32),
        area=float(area),
        centroid_xy=np.asarray(centroid_xy, dtype=np.float32),
        frame_shape=tuple(int(value) for value in owned_mask.shape),
        spatial_mask=spatial,
        shape_mask=shape,
        fill_ratio=min(float(area) / bbox_area, 1.0),
        payload=payload,
    )


def build_tracking_observations(
    raw_piglets: list[InstanceMask],
    frame: np.ndarray,
    tracking_frame_index: int,
    source_frame_index: int,
    timestamp_sec: float,
    next_observation_id: int,
    max_frame_instances: int,
) -> tuple[
    list[MaskObservation],
    list[torch.Tensor],
    dict[int, InstanceMask],
    dict[int, dict[str, Any]],
    int,
    int,
    int,
]:
    """Create mutually owned full-frame YOLO observations for one sparse ReID keyframe."""
    candidates = [_candidate_from_instance(instance) for instance in raw_piglets if instance.mask.any()]
    candidates, fused, suppressed = enforce_frame_identity_cap(candidates, max_frame_instances)
    owner = np.zeros(frame.shape[:2], dtype=np.uint16)
    for local_index, candidate in enumerate(candidates, 1):
        x1, y1, x2, y2 = (int(value) for value in candidate.bbox_xyxy)
        candidate_local = candidate.mask[y1:y2, x1:x2]
        owner_local = owner[y1:y2, x1:x2]
        owner_local[candidate_local & (owner_local == 0)] = local_index

    observations: list[MaskObservation] = []
    crops: list[torch.Tensor] = []
    active: dict[int, InstanceMask] = {}
    metadata: dict[int, dict[str, Any]] = {}
    observation_id = int(next_observation_id)
    for local_index, candidate in enumerate(candidates, 1):
        candidate_x1, candidate_y1, candidate_x2, candidate_y2 = (
            int(value) for value in candidate.bbox_xyxy
        )
        owned_local = (
            owner[candidate_y1:candidate_y2, candidate_x1:candidate_x2] == local_index
        )
        local_y, local_x = np.nonzero(owned_local)
        if not len(local_x):
            continue
        x1 = candidate_x1 + int(local_x.min())
        y1 = candidate_y1 + int(local_y.min())
        x2 = candidate_x1 + int(local_x.max()) + 1
        y2 = candidate_y1 + int(local_y.max()) + 1
        owned_mask = np.zeros(frame.shape[:2], dtype=bool)
        owned_mask[candidate_y1:candidate_y2, candidate_x1:candidate_x2] = owned_local
        area = int(len(local_x))
        payload = {
            "tracking_source_frame_index": int(source_frame_index),
            "tracking_timestamp_sec": float(timestamp_sec),
            "source_detection_indices": list(candidate.detection_indices),
        }
        observation = _observation_from_owned_mask(
            observation_id=observation_id,
            frame_index=tracking_frame_index,
            candidate=candidate,
            owned_mask=owned_mask,
            bbox_xyxy=(x1, y1, x2, y2),
            area=area,
            centroid_xy=(
                candidate_x1 + float(local_x.mean()),
                candidate_y1 + float(local_y.mean()),
            ),
            payload=payload,
        )
        observations.append(observation)
        crops.append(extract_masked_crop(frame, owned_mask, observation.bbox_xyxy))
        active[observation_id] = InstanceMask(
            index=observation_id,
            class_id=candidate.class_id,
            label=candidate.label,
            confidence=candidate.confidence,
            mask=owned_mask,
        )
        metadata[observation_id] = {
            "observation_id": observation_id,
            "tracking_frame_index": int(tracking_frame_index),
            "tracking_source_frame_index": int(source_frame_index),
            "tracking_timestamp_sec": float(timestamp_sec),
            "confidence": float(candidate.confidence),
            "bbox_x1": float(observation.bbox_xyxy[0]),
            "bbox_y1": float(observation.bbox_xyxy[1]),
            "bbox_x2": float(observation.bbox_xyxy[2]),
            "bbox_y2": float(observation.bbox_xyxy[3]),
            "full_mask_area_px": area,
            "source_detection_indices": list(candidate.detection_indices),
        }
        observation_id += 1
    return observations, crops, active, metadata, observation_id, fused, suppressed


def _track_contact_masks(
    active: dict[int, InstanceMask],
    roi: flow_base.RoiState | None,
    args: argparse.Namespace,
    support: np.ndarray | None = None,
) -> dict[int, np.ndarray]:
    if roi is None:
        return {}
    shape = roi.band.shape[:2]
    if support is None:
        support = _contact_support_mask(roi, args)
    erode_radius = max(0, int(args.contact_erode_px))
    erode_kernel = (
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * erode_radius + 1, 2 * erode_radius + 1),
        )
        if erode_radius
        else None
    )
    result: dict[int, np.ndarray] = {}
    for observation_id, instance in active.items():
        mask = mask_for_shape(instance.mask, shape)
        if erode_kernel is not None:
            mask = cv2.erode(mask.astype(np.uint8), erode_kernel).astype(bool)
        contact = mask & support
        if contact.any():
            result[observation_id] = flow_base._crop(contact, roi.crop_xyxy)
    return result


def _contact_support_mask(roi: flow_base.RoiState, args: argparse.Namespace) -> np.ndarray:
    shape = roi.band.shape[:2]
    band = mask_for_shape(roi.band, shape)
    sow = mask_for_shape(roi.sow.mask, shape)
    radius = max(0, int(args.sow_exclusion_px))
    if radius:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        sow = cv2.dilate(sow.astype(np.uint8), kernel).astype(bool)
    return band & ~sow


def _contact_union_on_support(
    piglets: list[InstanceMask],
    support: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, int]:
    union = np.zeros(support.shape, dtype=bool)
    count = 0
    radius = max(0, int(args.contact_erode_px))
    kernel = (
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        if radius
        else None
    )
    for piglet in piglets:
        mask = mask_for_shape(piglet.mask, support.shape)
        if kernel is not None:
            mask = cv2.erode(mask.astype(np.uint8), kernel).astype(bool)
        contact = mask & support
        if contact.any():
            count += 1
            union |= contact
    return union, count


def _sow_reference_base_mask(
    roi: flow_base.RoiState,
    image_shape: tuple[int, int],
    args: argparse.Namespace,
) -> np.ndarray:
    shape = roi.band.shape[:2]
    scale = image_shape[0] / 1080.0
    minimum = max(0.0, float(args.sow_reference_min_depth_px * scale))
    maximum = max(minimum, float(args.sow_reference_max_depth_px * scale))
    sow = mask_for_shape(roi.sow.mask, shape)
    depth = cv2.distanceTransform(sow.astype(np.uint8), cv2.DIST_L2, 5)
    return mask_for_shape(roi.band, shape) & sow & (depth >= minimum) & (depth <= maximum)


def _exclude_piglets_from_reference(
    reference_base: np.ndarray,
    piglets: list[InstanceMask],
    guard_px: int,
) -> np.ndarray:
    piglets_union = base.piglet_union_mask(piglets, reference_base.shape)
    radius = max(0, int(guard_px))
    if radius:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        piglets_union = cv2.dilate(piglets_union.astype(np.uint8), kernel).astype(bool)
    return reference_base & ~piglets_union


def _track_row(
    *,
    video_name: str,
    sample_index: int,
    current_frame_index: int,
    current_timestamp_sec: float,
    actual_flow_fps: float,
    roi_segment_id: int,
    observation_id: int,
    metadata: dict[str, Any],
    features: dict[str, Any],
    previous: TrackedFeatureFrameState,
) -> dict[str, Any]:
    source_pixels = int(features.get("track_source_contact_pixel_count", 0) or 0)
    full_area = int(metadata.get("full_mask_area_px", 0) or 0)
    row = {
        "video": video_name,
        "sample_index": int(sample_index),
        "prev_frame_index": int(previous.frame_index),
        "frame_index": int(current_frame_index),
        "interval_start_sec": round(float(previous.timestamp_sec), 6),
        "timestamp_sec": round((float(previous.timestamp_sec) + current_timestamp_sec) / 2.0, 6),
        "interval_end_sec": round(float(current_timestamp_sec), 6),
        "dt_sec": round(float(current_timestamp_sec - previous.timestamp_sec), 6),
        "actual_flow_fps": float(actual_flow_fps),
        "roi_segment_id": int(roi_segment_id),
        "observation_id": int(observation_id),
        "track_id": "",
        "capped_track_id": "",
        "roi_contact_area_px": source_pixels,
        "roi_contact_fraction": float(source_pixels / full_area) if full_area else 0.0,
    }
    row.update(metadata)
    row.update(features)
    row.update({field: None for field in TRACK_FILTER_FIELDS})
    row.update({field: None for field in TRACK_TEMPORAL_FIELDS})
    return row


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_rows(
    path: Path,
    rows: list[dict[str, Any]],
    video_name: str,
    dpi: int,
) -> None:
    import matplotlib

    # File-only plots must not create Tk objects that a prefetch thread may collect.
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = np.asarray([float(row["timestamp_sec"]) / 60.0 for row in rows])

    def values(key: str) -> np.ndarray:
        return np.asarray(
            [float(value) if (value := _as_finite(row.get(key))) is not None else math.nan for row in rows]
        )

    figure, axes = plt.subplots(4, 1, figsize=(15, 11), sharex=True)
    axes[0].plot(
        times,
        values("robust_global_normal_rms_px_s"),
        color="#64748B",
        linewidth=0.8,
        label="global normal RMS",
    )
    axes[0].plot(
        times,
        values("tracked_normal_rms_p75_px_s"),
        color="#DC2626",
        linewidth=0.9,
        label="track normal RMS p75",
    )
    axes[0].plot(
        times,
        values("tracked_tangent_rms_p75_px_s"),
        color="#2563EB",
        linewidth=0.9,
        label="track tangent RMS p75",
    )
    axes[0].plot(
        times,
        values("tracked_filtered_normal_signed_mean_px_s"),
        color="#16A34A",
        linewidth=0.8,
        label="filtered signed track mean",
    )
    axes[0].set_ylabel("px/s")
    axes[0].legend(loc="upper right", fontsize=8)

    periodic_intensity = values("tracked_periodic_intensity_p75_px_s")
    periodic_quality = values("tracked_periodicity_quality_p75")
    intensity_valid = np.isfinite(periodic_intensity)
    quality_valid = np.isfinite(periodic_quality)
    axes[1].scatter(
        times[intensity_valid],
        periodic_intensity[intensity_valid],
        color="#7C3AED",
        s=8,
        label="periodic intensity p75",
    )
    quality_axis = axes[1].twinx()
    quality_axis.scatter(
        times[quality_valid],
        periodic_quality[quality_valid],
        color="#0891B2",
        s=8,
        label="periodicity quality p75",
    )
    axes[1].set_ylabel("px/s")
    quality_axis.set_ylabel("quality")
    quality_axis.set_ylim(0.0, 1.0)
    handles_a, labels_a = axes[1].get_legend_handles_labels()
    handles_b, labels_b = quality_axis.get_legend_handles_labels()
    axes[1].legend(handles_a + handles_b, labels_a + labels_b, loc="upper right", fontsize=8)

    axes[2].plot(
        times,
        values("tracked_reliable_track_fraction"),
        color="#059669",
        linewidth=0.8,
        label="reliable-track fraction",
    )
    frequency_axis = axes[2].twinx()
    frequency = values("tracked_reliable_frequency_median_hz")
    frequency_valid = np.isfinite(frequency)
    frequency_axis.scatter(
        times[frequency_valid],
        frequency[frequency_valid],
        color="#EA580C",
        s=8,
        label="median frequency",
    )
    axes[2].set_ylabel("fraction")
    frequency_axis.set_ylabel("Hz")
    handles_a, labels_a = axes[2].get_legend_handles_labels()
    handles_b, labels_b = frequency_axis.get_legend_handles_labels()
    axes[2].legend(handles_a + handles_b, labels_a + labels_b, loc="upper right", fontsize=8)

    axes[3].plot(times, values("tracked_valid_track_count"), color="#1D4ED8", linewidth=0.8, label="valid ROI tracks")
    axes[3].plot(
        times,
        values("tracked_phase_cancellation_index"),
        color="#BE123C",
        linewidth=0.8,
        label="phase cancellation",
    )
    axes[3].plot(
        times,
        values("tracked_filter_outlier_fraction"),
        color="#A16207",
        linewidth=0.7,
        label="Hampel replacement fraction",
    )
    extrema_count = values("tracked_extremum_count")
    extrema_valid = np.isfinite(extrema_count) & (extrema_count > 0)
    axes[3].scatter(
        times[extrema_valid],
        extrema_count[extrema_valid],
        color="#F97316",
        s=8,
        label="retained extrema",
    )
    axes[3].set_ylabel("count / index")
    axes[3].set_xlabel("Source video time (min)")
    axes[3].legend(loc="upper right", fontsize=8)
    for axis in axes:
        axis.grid(True, alpha=0.2)
    figure.suptitle(f"{video_name}\nDense ROI-normal flow grouped by sparse mask-aware OSNet tracks")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _tracker_config(args: argparse.Namespace, *, max_identities: int = 0):
    return tracker_config_from_dict(
        {
            "match_threshold": args.tracker_match_threshold,
            "max_age": args.tracker_max_age,
            "max_center_distance": args.tracker_max_center_distance,
            "motion_weight": args.tracker_motion_weight,
            "offline": not args.tracker_no_offline,
            "offline_max_gap": args.tracker_offline_max_gap,
            "offline_match_threshold": args.tracker_offline_match_threshold,
            "offline_max_center_distance": args.tracker_offline_max_center_distance,
            "offline_max_tracklets": args.tracker_offline_max_tracklets,
            "max_identities": max_identities,
        }
    )


def _video_chunks(video_path: Path, args: argparse.Namespace) -> list[argparse.Namespace]:
    """Return per-chunk argument copies while preserving the user's requested time interval."""
    if args.chunk_seconds <= 0:
        return [args]
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if source_fps <= 0 or frame_count <= 0:
        raise ValueError(f"Invalid video metadata for {video_path}")
    duration = frame_count / source_fps
    interval_start = float(args.start_sec)
    if interval_start >= duration:
        raise ValueError(
            f"--start-sec={interval_start:g} is outside {video_path.name} ({duration:.3f}s)"
        )
    interval_end = (
        duration
        if args.max_seconds is None
        else min(duration, interval_start + float(args.max_seconds))
    )
    chunks: list[argparse.Namespace] = []
    part = 0
    start = interval_start
    while start < interval_end - 1e-9:
        stop = min(interval_end, start + float(args.chunk_seconds))
        values = vars(args).copy()
        values["start_sec"] = start
        values["max_seconds"] = stop - start
        values["output_suffix"] = (
            f"__part_{part + 1:04d}_{int(round(start * 1000)):010d}_{int(round(stop * 1000)):010d}"
        )
        chunks.append(argparse.Namespace(**values))
        start = stop
        part += 1
    return chunks


def process_video(
    piglet_model: Any,
    sow_model: Any,
    resources: TrackerResources,
    video_path: Path,
    args: argparse.Namespace,
) -> tuple[Path, Path, Path | None, Path] | None:
    stem = flow_base.output_base(video_path, args.input, args.output)
    output_suffix = str(getattr(args, "output_suffix", ""))
    if output_suffix:
        stem = stem.with_name(f"{stem.name}{output_suffix}")
    csv_path = stem.with_name(f"{stem.name}_tracked_normal_flow_features.csv")
    tracks_path = stem.with_name(f"{stem.name}_tracked_normal_flow_tracks.csv")
    plot_path = stem.with_name(f"{stem.name}_tracked_normal_flow_plot.png")
    summary_path = stem.with_name(f"{stem.name}_tracked_normal_flow_summary.json")
    required_paths = (csv_path, tracks_path, summary_path) if args.no_plot else (
        csv_path,
        tracks_path,
        plot_path,
        summary_path,
    )
    if args.resume and all(path.is_file() and path.stat().st_size > 0 for path in required_paths):
        print(f"[INFO] complete output exists; skipping {video_path}")
        return None
    for path in required_paths:
        if path.exists() and not args.overwrite:
            if not args.resume:
                raise FileExistsError(f"{path} exists; pass --overwrite or --resume.")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if source_fps <= 0 or frame_count <= 0:
        capture.release()
        raise ValueError(f"Invalid video metadata for {video_path}")
    target_tracking_fps = float(args.tracking_fps or resources.training_sample_fps)
    schedule = sampling_schedule(source_fps, args.flow_fps, args.mask_fps, target_tracking_fps)
    actual_flow_fps = float(schedule["actual_flow_fps"])
    actual_tracking_fps = float(schedule["actual_tracking_fps"])
    relative_fps_error = abs(actual_tracking_fps - resources.training_sample_fps) / resources.training_sample_fps
    if relative_fps_error > 0.10 + 1e-12:
        capture.release()
        raise ValueError(
            f"effective tracking rate {actual_tracking_fps:.3f} FPS differs from checkpoint training rate "
            f"{resources.training_sample_fps:.3f} FPS by more than 10%; adjust --tracking-fps/--flow-fps"
        )
    flow_step = int(schedule["flow_step"])
    mask_stride = int(schedule["mask_stride"])
    tracking_stride = int(schedule["tracking_stride"])
    expected_dt = flow_step / source_fps
    start_frame = min(frame_count - 1, max(0, int(round(args.start_sec * source_fps))))
    end_frame = (
        frame_count
        if args.max_seconds is None
        else min(frame_count, start_frame + max(1, int(math.ceil(args.max_seconds * source_fps))))
    )
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    sample_reader = BatchedSampleReader(
        capture,
        piglet_model,
        args,
        start_frame=start_frame,
        end_frame=end_frame,
        flow_step=flow_step,
        mask_stride=mask_stride,
    )
    flow_estimator = DenseFlowEstimator(args.flow_method, args.flow_scale)

    # Signal IDs are never forcibly merged.  An optional capped ID namespace is
    # computed separately after signal association and is never used for temporal windows.
    tracker = MaskAwareOfflineTracker(
        resources.model,
        resources.device,
        _tracker_config(args, max_identities=0),
    )
    tracker_frames: dict[int, list[MaskObservation]] = {}
    observation_metadata: dict[int, dict[str, Any]] = {}
    pending_observations: list[MaskObservation] = []
    pending_crops: list[torch.Tensor] = []
    active_tracking_instances: dict[int, InstanceMask] = {}
    latest_track_contacts: dict[int, np.ndarray] = {}
    next_observation_id = 1
    tracking_frame_index = 0
    fragment_fusions = 0
    frame_cap_suppressed = 0

    rows: list[dict[str, Any]] = []
    track_rows: list[dict[str, Any]] = []
    roi: flow_base.RoiState | None = None
    outer_band_pixel_count = 0
    latest_sow: InstanceMask | None = None
    latest_piglets: list[InstanceMask] = []
    latest_contact_mask: np.ndarray | None = None
    latest_sow_reference_mask: np.ndarray | None = None
    contact_support_mask: np.ndarray | None = None
    sow_reference_base_mask: np.ndarray | None = None
    latest_contact_count = 0
    previous: TrackedFeatureFrameState | None = None
    next_roi_segment_id = 0
    next_sow_check_sec = args.start_sec
    sample_index = 0
    frame_index = start_frame
    last_progress_sec = args.start_sec - args.progress_sec

    try:
        while True:
            sampled = sample_reader.read()
            if sampled is None or sampled.frame is None:
                break
            frame_index = sampled.frame_index
            frame = sampled.frame
            timestamp_sec = frame_index / source_fps
            mask_refreshed = sample_index % mask_stride == 0
            tracking_refreshed = sample_index % tracking_stride == 0
            if tracking_refreshed and not mask_refreshed:
                raise RuntimeError("internal schedule error: tracking keyframe is not a YOLO mask refresh")
            roi_refreshed = False
            event_status = "ok"
            candidate_sow: InstanceMask | None = None

            if mask_refreshed:
                sow_due = latest_sow is None or timestamp_sec + 1e-9 >= next_sow_check_sec
                if sow_due:
                    try:
                        candidate_sow = flow_base._infer_sow(sow_model, frame, args)
                        if latest_sow is None:
                            latest_sow = candidate_sow
                    except Exception as exc:
                        event_status = f"sow_inference_failed:{type(exc).__name__}"
                    next_sow_check_sec = timestamp_sec + max(expected_dt, args.sow_check_sec)
                raw_piglets = sampled.raw_piglets or []
                if sampled.piglet_error is not None:
                    raw_piglets = []
                    event_status = f"piglet_inference_failed:{sampled.piglet_error}"

                if tracking_refreshed:
                    (
                        observations,
                        crops,
                        active_tracking_instances,
                        metadata,
                        next_observation_id,
                        fused,
                        suppressed,
                    ) = build_tracking_observations(
                        raw_piglets,
                        frame,
                        tracking_frame_index,
                        frame_index,
                        timestamp_sec,
                        next_observation_id,
                        args.max_frame_instances,
                    )
                    if next_observation_id - 1 > args.max_tracker_observations:
                        raise MemoryError(
                            f"video exceeds --max-tracker-observations={args.max_tracker_observations}; "
                            "process a shorter interval"
                        )
                    pending_observations.extend(observations)
                    pending_crops.extend(crops)
                    if len(pending_crops) >= args.tracker_batch_size:
                        encode_observation_crops(
                            resources.model,
                            pending_observations,
                            pending_crops,
                            resources.device,
                            args.tracker_batch_size,
                        )
                        pending_observations.clear()
                        pending_crops.clear()
                    tracker_frames[tracking_frame_index] = observations
                    observation_metadata.update(metadata)
                    tracking_frame_index += 1
                    fragment_fusions += fused
                    frame_cap_suppressed += suppressed

                reference_sow = candidate_sow or (roi.sow if roi is not None else latest_sow)
                filtered_for_candidate = (
                    flow_base._filter_piglets(raw_piglets, reference_sow, args.max_piglet_sow_overlap)
                    if reference_sow is not None
                    else []
                )
                if roi is None and reference_sow is not None:
                    try:
                        roi = flow_base.build_roi_state(
                            reference_sow,
                            filtered_for_candidate,
                            frame.shape[:2],
                            args,
                            segment_id=next_roi_segment_id,
                        )
                        next_roi_segment_id += 1
                        latest_sow = roi.sow
                        latest_piglets = flow_base._filter_piglets(
                            raw_piglets, roi.sow, args.max_piglet_sow_overlap
                        )
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
                                roi = flow_base.build_roi_state(
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
                                latest_sow_reference_mask = None
                                contact_support_mask = None
                                sow_reference_base_mask = None
                                latest_contact_count = 0
                                latest_track_contacts = {}
                                outer_band_pixel_count = 0
                                previous = None
                                roi = None
                    if not refreshed and roi is not None:
                        latest_piglets = (
                            filtered_for_candidate
                            if candidate_sow is None
                            else flow_base._filter_piglets(
                                raw_piglets, roi.sow, args.max_piglet_sow_overlap
                            )
                        )

                if roi is not None:
                    if roi_refreshed or contact_support_mask is None:
                        contact_support_mask = _contact_support_mask(roi, args)
                    latest_contact_mask, latest_contact_count = _contact_union_on_support(
                        latest_piglets,
                        contact_support_mask,
                        args,
                    )
                    scale = frame.shape[0] / 1080.0
                    if roi_refreshed or sow_reference_base_mask is None:
                        sow_reference_base_mask = _sow_reference_base_mask(roi, frame.shape[:2], args)
                        outer_band_pixel_count = int(
                            np.count_nonzero(
                                mask_for_shape(roi.band, frame.shape[:2])
                                & ~mask_for_shape(roi.sow.mask, frame.shape[:2])
                            )
                        )
                    latest_sow_reference_mask = _exclude_piglets_from_reference(
                        sow_reference_base_mask,
                        latest_piglets,
                        max(0, int(round(args.sow_reference_piglet_guard_px * scale))),
                    )
                    if tracking_refreshed or roi_refreshed:
                        latest_track_contacts = _track_contact_masks(
                            active_tracking_instances,
                            roi,
                            args,
                            contact_support_mask,
                        )

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if roi is None or latest_contact_mask is None or latest_sow_reference_mask is None:
                raw_metrics = flow_base._empty_flow_metrics()
                instant_features = base._empty_instant_features()
                rows.append(
                    base._new_row(
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
                        raw_metrics=raw_metrics,
                        instant_features=instant_features,
                        previous=None,
                    )
                )
                previous = None
            else:
                gray_crop = flow_base._crop(gray, roi.crop_xyxy)
                contact_crop = flow_base._crop(latest_contact_mask, roi.crop_xyxy)
                sow_reference_crop = flow_base._crop(latest_sow_reference_mask, roi.crop_xyxy)
                current = TrackedFeatureFrameState(
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                    segment_id=roi.segment_id,
                    mask_refreshed=mask_refreshed,
                    gray_crop=gray_crop,
                    contact_mask_crop=contact_crop,
                    sow_reference_mask_crop=sow_reference_crop,
                    piglet_count=len(latest_piglets),
                    contact_piglet_count=latest_contact_count,
                    contact_pixel_count=int(np.count_nonzero(contact_crop)),
                    track_contact_masks_crop=latest_track_contacts,
                )
                if roi_refreshed or previous is None or previous.segment_id != roi.segment_id:
                    raw_metrics = flow_base._empty_flow_metrics()
                    instant_features = base._empty_instant_features(outer_band_pixels=outer_band_pixel_count)
                    status = event_status if roi_refreshed else "flow_initialized"
                else:
                    dt_sec = timestamp_sec - previous.timestamp_sec
                    if dt_sec <= 0 or dt_sec > expected_dt * 1.75:
                        raw_metrics = flow_base._empty_flow_metrics()
                        instant_features = base._empty_instant_features(outer_band_pixels=outer_band_pixel_count)
                        current = replace(current, track_contact_masks_crop={})
                        latest_track_contacts = {}
                        status = "time_gap_reset"
                    else:
                        flow = flow_estimator.calculate(previous.gray_crop, gray_crop)
                        raw_metrics = flow_base.normal_flow_statistics(
                            flow,
                            roi.normal_field_crop,
                            previous.contact_mask_crop,
                            dt_sec=dt_sec,
                            min_pixels=args.min_flow_pixels,
                        )
                        instant_features = base.global_flow_features(
                            flow,
                            roi.normal_field_crop,
                            previous.contact_mask_crop,
                            current.contact_mask_crop,
                            previous.sow_reference_mask_crop,
                            current.sow_reference_mask_crop,
                            dt_sec,
                            outer_band_pixel_count,
                            min_contact_pixels=args.min_flow_pixels,
                            min_reference_pixels=args.min_sow_reference_pixels,
                            winsor_quantile=args.energy_winsor_quantile,
                            residual_alpha=args.sow_residual_energy_alpha,
                        )
                        track_flow_context = prepare_track_flow_context(
                            flow,
                            roi.normal_field_crop,
                            current.contact_mask_crop,
                            dt_sec,
                            instant_features,
                        )
                        for observation_id, source_track_mask in previous.track_contact_masks_crop.items():
                            features = track_flow_features(
                                flow,
                                roi.normal_field_crop,
                                source_track_mask,
                                current.contact_mask_crop,
                                dt_sec,
                                instant_features,
                                min_pixels=args.min_track_flow_pixels,
                                winsor_quantile=args.energy_winsor_quantile,
                                residual_alpha=args.sow_residual_energy_alpha,
                                context=track_flow_context,
                            )
                            track_rows.append(
                                _track_row(
                                    video_name=video_path.name,
                                    sample_index=sample_index,
                                    current_frame_index=frame_index,
                                    current_timestamp_sec=timestamp_sec,
                                    actual_flow_fps=actual_flow_fps,
                                    roi_segment_id=roi.segment_id,
                                    observation_id=observation_id,
                                    metadata=observation_metadata[observation_id],
                                    features=features,
                                    previous=previous,
                                )
                            )
                        if not tracking_refreshed:
                            propagated = propagate_track_masks(
                                flow,
                                previous.track_contact_masks_crop,
                                current.contact_mask_crop,
                            )
                            current = replace(current, track_contact_masks_crop=propagated)
                            latest_track_contacts = propagated
                        if not raw_metrics["flow_valid"]:
                            status = "insufficient_contact"
                        elif not instant_features["stable_contact_valid"]:
                            status = "insufficient_stable_contact"
                        elif not instant_features["sow_reference_valid"]:
                            status = "insufficient_sow_reference"
                        else:
                            status = "ok"
                        if event_status != "ok":
                            status = f"{status}|{event_status}"
                rows.append(
                    base._new_row(
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
                        raw_metrics=raw_metrics,
                        instant_features=instant_features,
                        previous=None if roi_refreshed else previous,
                    )
                )
                previous = current

            if args.progress_sec > 0 and timestamp_sec - last_progress_sec >= args.progress_sec:
                print(
                    f"[INFO] {video_path.name}: t={timestamp_sec:.1f}s rows={len(rows)} "
                    f"tracking_frames={tracking_frame_index} observations={next_observation_id - 1}"
                )
                last_progress_sec = timestamp_sec
            sample_index += 1
    finally:
        sample_reader.close()
        capture.release()

    if not rows:
        raise RuntimeError(f"No sampled frames were read from {video_path}")
    if pending_crops:
        encode_observation_crops(
            resources.model,
            pending_observations,
            pending_crops,
            resources.device,
            args.tracker_batch_size,
        )
    assignments = tracker.track(tracker_frames)
    capped_tracker: MaskAwareOfflineTracker | None = None
    capped_assignments: dict[int, int] | None = None
    if args.max_track_identities:
        capped_tracker = MaskAwareOfflineTracker(
            resources.model,
            resources.device,
            _tracker_config(args, max_identities=args.max_track_identities),
        )
        capped_assignments = capped_tracker.track(tracker_frames)
    for row in track_rows:
        observation_id = int(row["observation_id"])
        row["track_id"] = int(assignments[observation_id])
        if capped_assignments is not None:
            row["capped_track_id"] = int(capped_assignments[observation_id])
    base.add_temporal_features(rows, args)
    add_track_temporal_features(track_rows, args)
    aggregate_track_rows(rows, track_rows)
    def partial_path(path: Path) -> Path:
        return path.with_name(f"{path.stem}.__partial__{path.suffix}")

    csv_partial = partial_path(csv_path)
    tracks_partial = partial_path(tracks_path)
    plot_partial = partial_path(plot_path)
    summary_partial = partial_path(summary_path)
    write_csv(csv_partial, rows, CSV_FIELDS)
    write_csv(tracks_partial, track_rows, TRACK_CSV_FIELDS)
    if not args.no_plot:
        plot_rows(plot_partial, rows, video_path.name, args.plot_dpi)

    checkpoint_path = args.tracker_checkpoint.expanduser().resolve()
    summary = {
        "schema_version": 2,
        "source": str(video_path.resolve()),
        "source_interval": {
            "start_sec": float(args.start_sec),
            "max_seconds": args.max_seconds,
            "chunk_seconds": float(args.chunk_seconds),
            "output_suffix": output_suffix or None,
        },
        "piglet_yolo_model": str(args.piglet_weights.expanduser().resolve()),
        "sow_yolo_model": str(args.sow_weights.expanduser().resolve()),
        "tracker_checkpoint": str(checkpoint_path),
        "tracker_checkpoint_sha256": resources.checkpoint_sha256,
        "tracker_checkpoint_epoch": resources.checkpoint.get("epoch"),
        "sampling": {
            "source_fps": source_fps,
            "requested_flow_fps": args.flow_fps,
            "requested_mask_fps": args.mask_fps,
            "checkpoint_training_fps": resources.training_sample_fps,
            "requested_tracking_fps": target_tracking_fps,
            **schedule,
        },
        "execution": {
            "device": str(resources.device),
            "half": bool(args.half),
            "imgsz": int(args.imgsz),
            "retina_masks": bool(args.retina_masks),
            "yolo_batch_size": int(args.yolo_batch_size),
            "tracker_batch_size": int(args.tracker_batch_size),
            "prefetch": not args.no_prefetch,
            "torch_compile": args.torch_compile,
            "plots_enabled": not args.no_plot,
        },
        "frame_count": len(rows),
        "tracking_frame_count": len(tracker_frames),
        "observation_count": len(assignments),
        "track_count": len(set(assignments.values())),
        "capped_track_count": (
            len(set(capped_assignments.values())) if capped_assignments is not None else None
        ),
        "max_tracker_observations": args.max_tracker_observations,
        "tracker_config": asdict(tracker.config),
        "identity_cap": capped_tracker.identity_cap_summary if capped_tracker is not None else None,
        "max_frame_instances": args.max_frame_instances,
        "frame_cap_fragment_fusions": fragment_fusions,
        "frame_cap_suppressed_detections": frame_cap_suppressed,
        "cap_semantics": {
            "frame": "tracking branch only; original YOLO masks remain unchanged for ROI/global flow",
            "final_ids": (
                "disabled"
                if args.max_track_identities == 0
                else (
                    "written only as capped_track_id for display/compatibility; all signal windows use the "
                    "uncapped track_id namespace"
                )
            ),
        },
        "mask_ownership": "higher-confidence tracked instance owns overlap pixels; original YOLO union is unchanged",
        "signal": {
            "dense_flow_method": args.flow_method,
            "dense_flow_scale": args.flow_scale,
            "dense_flow_is_not_downsampled_to_tracking_fps": True,
            "track_masks_are_forward_propagated_between_keyframes": True,
            "global_feature_window_sec": args.feature_window_sec,
            "track_feature_window_sec": args.track_feature_window_sec,
            "track_flow_rows": len(track_rows),
            "valid_track_flow_rows": sum(int(row.get("track_valid", 0) or 0) for row in track_rows),
            "filter": {
                "order": "Hampel/MAD spike replacement -> centered smoothing -> extrema filtering",
                "raw_signal_is_preserved": True,
                "temporal_features_use_filtered_signal": True,
                "hampel_window_sec": args.track_hampel_window_sec,
                "hampel_sigma": args.track_hampel_sigma,
                "smooth_method": args.track_smooth_method,
                "smooth_window_sec": args.track_smooth_window_sec,
                "savgol_polyorder": args.track_savgol_polyorder,
                "extrema_min_prominence_mad": args.track_extrema_min_prominence_mad,
                "extrema_min_distance_sec": (
                    args.track_extrema_min_distance_sec
                    if args.track_extrema_min_distance_sec > 0
                    else 0.5 / args.feature_max_frequency_hz
                ),
                "extrema_distance_is_automatic": args.track_extrema_min_distance_sec == 0,
                "hampel_replacement_rows": sum(
                    int(row.get("track_filter_outlier", 0) or 0) for row in track_rows
                ),
                "retained_extrema_rows": sum(
                    int(row.get("track_extremum_kept", 0) or 0) for row in track_rows
                ),
            },
            "temporal_windows_do_not_cross_roi_or_track_gaps": True,
            "behaviour_classification": False,
        },
        "outputs": {
            "aggregate_csv": str(csv_path),
            "track_long_csv": str(tracks_path),
            "plot": None if args.no_plot else str(plot_path),
            "summary": str(summary_path),
        },
    }
    summary_partial.parent.mkdir(parents=True, exist_ok=True)
    summary_partial.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_partial.replace(csv_path)
    tracks_partial.replace(tracks_path)
    if not args.no_plot:
        plot_partial.replace(plot_path)
    # The summary is the completion marker and is therefore committed last.
    summary_partial.replace(summary_path)
    print(
        f"[INFO] wrote {csv_path} ; {tracks_path} ; "
        f"{'plot disabled ; ' if args.no_plot else f'{plot_path} ; '}{summary_path} "
        f"(rows={len(rows)}, track_rows={len(track_rows)}, tracks={summary['track_count']})"
    )
    return csv_path, tracks_path, None if args.no_plot else plot_path, summary_path


def main() -> None:
    args = parse_args()
    validate_args(args)
    videos = flow_base.iter_videos(args.input, args.recursive)
    if args.max_videos is not None:
        videos = videos[: max(0, args.max_videos)]
    if not videos:
        raise ValueError("No videos selected.")
    if args.dry_run:
        print(f"Input: {args.input}")
        print(f"Videos: {len(videos)}")
        print(f"Output: {args.output}")
        print(f"Dense flow requested: {args.flow_fps:g} FPS")
        print(f"Dense flow backend: {args.flow_method} at {args.flow_scale:g}x spatial scale")
        print(f"YOLO masks requested: {args.mask_fps:g} FPS (full-resolution={args.retina_masks})")
        print(
            f"GPU batches: YOLO={args.yolo_batch_size}; ReID={args.tracker_batch_size}; "
            f"prefetch={not args.no_prefetch}; resume={args.resume}; plots={not args.no_plot}"
        )
        print(
            f"Chunking: {'disabled' if args.chunk_seconds <= 0 else f'{args.chunk_seconds:g}s'}"
        )
        print(
            f"Tracker: {args.tracker_checkpoint}; keyframe FPS="
            f"{'checkpoint metadata' if args.tracking_fps == 0 else args.tracking_fps}"
        )
        print(
            f"Caps: per-frame instances={args.max_frame_instances}; "
            f"forced final IDs={args.max_track_identities} (0 means disabled)"
        )
        print(
            f"Track filter: Hampel {args.track_hampel_window_sec:g}s/{args.track_hampel_sigma:g} MAD; "
            f"{args.track_smooth_method} {args.track_smooth_window_sec:g}s; extrema prominence "
            f"{args.track_extrema_min_prominence_mad:g} MAD"
        )
        print(
            f"Track temporal window: centered {args.track_feature_window_sec:g}s "
            f"(about +/-{args.track_feature_window_sec / 2.0:g}s around an ROI refresh)"
        )
        print("Signal: dense sow-compensated normal flow -> sparse observation -> offline track -> temporal aggregate")
        print("Behaviour classification: disabled")
        for video in videos:
            print(video)
        return
    for weights in (args.piglet_weights, args.sow_weights, args.tracker_checkpoint):
        if not weights.expanduser().is_file():
            raise FileNotFoundError(f"Weights not found: {weights}")

    device = select_device(args.device)
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
    resources = TrackerResources(
        association_model,
        checkpoint,
        device,
        training_sample_fps,
        hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
    )

    from ultralytics import YOLO

    args.output.mkdir(parents=True, exist_ok=True)
    piglet_model = YOLO(str(args.piglet_weights), task="segment")
    sow_model = YOLO(str(args.sow_weights), task="segment")
    if args.torch_compile != "none":
        compile_mode: bool | str = True if args.torch_compile == "default" else args.torch_compile
        piglet_model.overrides["compile"] = compile_mode
        sow_model.overrides["compile"] = compile_mode
    failed = 0
    processed = 0
    skipped = 0
    chunk_count = 0
    for index, video_path in enumerate(videos, start=1):
        try:
            chunks = _video_chunks(video_path, args)
        except Exception as exc:
            failed += 1
            if args.fail_on_error:
                raise
            print(f"[WARN] failed {video_path}: {type(exc).__name__}: {exc}")
            continue
        for part_index, chunk_args in enumerate(chunks, start=1):
            chunk_count += 1
            try:
                detail = f" chunk {part_index}/{len(chunks)}" if len(chunks) > 1 else ""
                print(f"[INFO] processing {index}/{len(videos)}{detail}: {video_path}")
                result = process_video(
                    piglet_model,
                    sow_model,
                    resources,
                    video_path,
                    chunk_args,
                )
                if result is None:
                    skipped += 1
                else:
                    processed += 1
            except Exception as exc:
                failed += 1
                if args.fail_on_error:
                    raise
                print(
                    f"[WARN] failed {video_path} chunk {part_index}/{len(chunks)}: "
                    f"{type(exc).__name__}: {exc}"
                )
    print(
        f"Done. videos={len(videos)}, chunks={chunk_count}, processed={processed}, skipped={skipped}, "
        f"failed={failed}, output={args.output}"
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
