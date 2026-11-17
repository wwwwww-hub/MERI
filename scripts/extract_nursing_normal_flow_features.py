"""Extract sow-compensated global nursing-band optical-flow features.

This is an additive companion to ``extract_nursing_normal_flow.py``.  It keeps
the original raw normal-flow measurements, but adds continuous numerical
diagnostics only; it does not classify or label massage behaviour.
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

from scripts import extract_nursing_normal_flow as base  # noqa: E402
from scripts.extract_nursing_band import (  # noqa: E402
    DEFAULT_PIGLET_WEIGHTS,
    DEFAULT_SOW_WEIGHTS,
    InstanceMask,
    mask_for_shape,
    sow_mask_change_ratio,
)


# Directory for input nursing video clips (not bundled in release)
DEFAULT_INPUT = ROOT / "working_data" / "nursing_clips"
DEFAULT_OUTPUT = ROOT / "runs" / "nursing_normal_flow_features"

INSTANT_FEATURE_FIELDS = (
    "outer_band_pixel_count",
    "stable_contact_pixel_count",
    "stable_contact_band_fraction",
    "stable_contact_valid",
    "source_sow_reference_pixel_count",
    "stable_sow_reference_pixel_count",
    "sow_reference_valid",
    "sow_rigid_flow_x_px_s",
    "sow_rigid_flow_y_px_s",
    "sow_rigid_speed_px_s",
    "sow_residual_normal_median_px_s",
    "sow_residual_normal_p75_px_s",
    "sow_residual_normal_rms_px_s",
    "sow_residual_tangent_median_px_s",
    "sow_residual_normal_energy_px2_s2",
    "sow_residual_tangent_energy_px2_s2",
    "after_sow_rigid_normal_signed_mean_px_s",
    "after_sow_rigid_normal_signed_median_px_s",
    "after_sow_rigid_normal_abs_p75_px_s",
    "after_sow_rigid_normal_rms_px_s",
    "sow_compensated_signed_mean_px_s",
    "sow_compensated_signed_median_px_s",
    "normal_contact_energy_px2_s2",
    "tangent_contact_energy_px2_s2",
    "normal_energy_excess_px2_s2",
    "sow_compensated_global_normal_rms_px_s",
    "tangent_energy_excess_px2_s2",
    "sow_compensated_global_tangent_rms_px_s",
    "normal_energy_fraction",
    "contact_common_flow_x_px_s",
    "contact_common_flow_y_px_s",
    "contact_common_speed_px_s",
    "bulk_motion_coherence",
    "common_removed_global_normal_rms_px_s",
)

TEMPORAL_FEATURE_FIELDS = (
    "sow_compensated_global_normal_med3_px_s",
    "common_removed_global_normal_med3_px_s",
    "smoothed_normal_reference_energy_px2_s2",
    "smoothed_tangent_reference_energy_px2_s2",
    "robust_normal_energy_excess_px2_s2",
    "robust_tangent_energy_excess_px2_s2",
    "robust_normal_signed_sqrt_excess_px_s",
    "robust_tangent_signed_sqrt_excess_px_s",
    "robust_global_normal_rms_px_s",
    "robust_global_tangent_rms_px_s",
    "normal_tangent_energy_contrast",
    "feature_window_start_sec",
    "feature_window_end_sec",
    "feature_valid_fraction",
    "window_normal_intensity_median_px_s",
    "window_signed_signal_rms_px_s",
    "bidirectional_energy_balance",
    "cycle_closure",
    "dominant_frequency_hz",
    "autocorrelation_peak",
    "autocorrelation_peak_prominence",
    "halfcycle_antisymmetry",
    "estimated_cycles_in_window",
    "spectral_dominant_frequency_hz",
    "periodic_band_power_fraction",
    "spectral_peak_concentration",
    "quality_gated_frequency_hz",
)

CSV_FIELDS = (*base.CSV_FIELDS, *INSTANT_FEATURE_FIELDS, *TEMPORAL_FEATURE_FIELDS)


@dataclass(frozen=True)
class FeatureFrameState:
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

    # Kept aligned with extract_nursing_band.py because base.build_roi_state() consumes them.
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

    # Continuous feature-estimation parameters.  None of these are behaviour thresholds.
    parser.add_argument("--sow-reference-min-depth-px", type=float, default=6.0)
    parser.add_argument("--sow-reference-max-depth-px", type=float, default=28.0)
    parser.add_argument("--sow-reference-piglet-guard-px", type=int, default=3)
    parser.add_argument("--min-sow-reference-pixels", type=int, default=256)
    parser.add_argument("--energy-winsor-quantile", type=float, default=0.99)
    parser.add_argument("--sow-residual-energy-alpha", type=float, default=1.0)
    parser.add_argument(
        "--reference-energy-smooth-sec",
        type=float,
        default=1.0,
        help="Centered median window for the sow residual-energy baseline.",
    )
    parser.add_argument("--feature-window-sec", type=float, default=6.0)
    parser.add_argument("--feature-min-valid-fraction", type=float, default=0.75)
    parser.add_argument("--feature-min-frequency-hz", type=float, default=0.5)
    parser.add_argument("--feature-max-frequency-hz", type=float, default=3.0)
    parser.add_argument(
        "--frequency-min-autocorrelation",
        type=float,
        default=0.35,
        help="Quality gate for displaying frequency; it is not a behaviour threshold.",
    )
    parser.add_argument(
        "--frequency-min-peak-concentration",
        type=float,
        default=0.20,
        help="Minimum spectral peak concentration for a reportable frequency.",
    )

    parser.add_argument("--plot-dpi", type=int, default=160)
    parser.add_argument(
        "--plot-smooth-sec",
        type=float,
        default=1.0,
        help="Centered median smoothing used only for plots; CSV values remain unchanged.",
    )
    parser.add_argument(
        "--plot-outlier-window-sec",
        type=float,
        default=2.0,
        help="Centered Hampel window used only for plot outlier replacement.",
    )
    parser.add_argument(
        "--plot-outlier-sigma",
        type=float,
        default=4.0,
        help="Hampel outlier threshold used only for plots.",
    )
    parser.add_argument(
        "--plot-sample-sec",
        type=float,
        default=0.5,
        help="Approximate time spacing of plotted points after filtering.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _ellipse(radius: int) -> np.ndarray:
    radius = max(0, int(radius))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def piglet_union_mask(piglets: list[InstanceMask], shape: tuple[int, int]) -> np.ndarray:
    union = np.zeros(shape, dtype=bool)
    for piglet in piglets:
        union |= mask_for_shape(piglet.mask, shape)
    return union


def sow_surface_reference_mask(
    piglets: list[InstanceMask],
    band: np.ndarray,
    sow_mask: np.ndarray,
    min_depth_px: float = 6.0,
    max_depth_px: float = 28.0,
    piglet_guard_px: int = 3,
) -> np.ndarray:
    """Return a protected strip just inside the sow surface within the nursing band."""
    shape = band.shape[:2]
    sow = mask_for_shape(sow_mask, shape)
    band_bool = mask_for_shape(band, shape)
    inside_depth = cv2.distanceTransform(sow.astype(np.uint8), cv2.DIST_L2, 5)
    reference = (
        band_bool
        & sow
        & (inside_depth >= max(0.0, float(min_depth_px)))
        & (inside_depth <= max(float(min_depth_px), float(max_depth_px)))
    )
    piglets_union = piglet_union_mask(piglets, shape)
    guard = max(0, int(piglet_guard_px))
    if guard:
        piglets_union = cv2.dilate(piglets_union.astype(np.uint8), _ellipse(guard)).astype(bool)
    return reference & ~piglets_union


def destination_supported_source_mask(flow: np.ndarray, current_mask: np.ndarray) -> np.ndarray:
    """Sample the current mask at each source pixel's optical-flow destination."""
    height, width = current_mask.shape
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    mapped = cv2.remap(
        current_mask.astype(np.uint8),
        grid_x + flow[..., 0].astype(np.float32),
        grid_y + flow[..., 1].astype(np.float32),
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return mapped.astype(bool)


def stable_source_mask(flow: np.ndarray, previous_mask: np.ndarray, current_mask: np.ndarray) -> np.ndarray:
    return previous_mask.astype(bool) & destination_supported_source_mask(flow, current_mask)


def _winsorized(values: np.ndarray, quantile: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return finite
    cap = float(np.quantile(np.abs(finite), min(1.0, max(0.5, quantile))))
    return np.clip(finite, -cap, cap)


def _safe_corr(first: np.ndarray, second: np.ndarray) -> float | None:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    valid = np.isfinite(first) & np.isfinite(second)
    if int(np.count_nonzero(valid)) < 4:
        return None
    a = first[valid] - float(np.mean(first[valid]))
    b = second[valid] - float(np.mean(second[valid]))
    a_energy = float(np.sum(a * a))
    b_energy = float(np.sum(b * b))
    if a_energy <= 1e-18 or b_energy <= 1e-18:
        return None
    return float(np.sum(a * b) / math.sqrt(a_energy * b_energy))


def _empty_instant_features(
    *,
    outer_band_pixels: int = 0,
    stable_contact_pixels: int = 0,
    source_reference_pixels: int = 0,
    stable_reference_pixels: int = 0,
) -> dict[str, float | int | bool | None]:
    result: dict[str, float | int | bool | None] = {key: None for key in INSTANT_FEATURE_FIELDS}
    result.update(
        {
            "outer_band_pixel_count": int(outer_band_pixels),
            "stable_contact_pixel_count": int(stable_contact_pixels),
            "stable_contact_band_fraction": (
                float(stable_contact_pixels / outer_band_pixels) if outer_band_pixels else 0.0
            ),
            "stable_contact_valid": 0,
            "source_sow_reference_pixel_count": int(source_reference_pixels),
            "stable_sow_reference_pixel_count": int(stable_reference_pixels),
            "sow_reference_valid": 0,
        }
    )
    return result


def global_flow_features(
    flow: np.ndarray,
    normal_field: np.ndarray,
    previous_contact_mask: np.ndarray,
    current_contact_mask: np.ndarray,
    previous_sow_reference_mask: np.ndarray,
    current_sow_reference_mask: np.ndarray,
    dt_sec: float,
    outer_band_pixel_count: int,
    *,
    min_contact_pixels: int = 128,
    min_reference_pixels: int = 256,
    winsor_quantile: float = 0.99,
    residual_alpha: float = 1.0,
) -> dict[str, float | int | bool | None]:
    """Compute continuous global flow features after stable-mask and sow-motion correction."""
    if dt_sec <= 0:
        raise ValueError("dt_sec must be positive.")
    if flow.shape != normal_field.shape or flow.shape[:2] != previous_contact_mask.shape:
        raise ValueError("Flow, normal field, and mask shapes do not match.")

    finite = np.isfinite(flow).all(axis=2) & np.isfinite(normal_field).all(axis=2)
    field_valid = np.linalg.norm(normal_field, axis=2) > 1e-6
    contact = stable_source_mask(flow, previous_contact_mask, current_contact_mask) & finite & field_valid
    reference = (
        stable_source_mask(flow, previous_sow_reference_mask, current_sow_reference_mask) & finite & field_valid
    )
    contact_count = int(np.count_nonzero(contact))
    source_reference_count = int(np.count_nonzero(previous_sow_reference_mask))
    reference_count = int(np.count_nonzero(reference))
    result = _empty_instant_features(
        outer_band_pixels=outer_band_pixel_count,
        stable_contact_pixels=contact_count,
        source_reference_pixels=source_reference_count,
        stable_reference_pixels=reference_count,
    )
    result["stable_contact_valid"] = int(contact_count >= max(1, int(min_contact_pixels)))
    if reference_count < max(1, int(min_reference_pixels)):
        return result

    velocity = flow.astype(np.float64) / float(dt_sec)
    reference_velocity = velocity[reference]
    sow_rigid = np.median(reference_velocity, axis=0)
    result.update(
        {
            "sow_reference_valid": 1,
            "sow_rigid_flow_x_px_s": float(sow_rigid[0]),
            "sow_rigid_flow_y_px_s": float(sow_rigid[1]),
            "sow_rigid_speed_px_s": float(np.linalg.norm(sow_rigid)),
        }
    )

    tangent_field = np.stack((-normal_field[..., 1], normal_field[..., 0]), axis=2).astype(np.float64)
    reference_relative = reference_velocity - sow_rigid
    reference_normal = np.sum(reference_relative * normal_field[reference], axis=1)
    reference_tangent = np.sum(reference_relative * tangent_field[reference], axis=1)
    ref_normal_center = float(np.median(reference_normal))
    ref_tangent_center = float(np.median(reference_tangent))
    reference_normal_centered = reference_normal - ref_normal_center
    reference_tangent_centered = reference_tangent - ref_tangent_center
    result.update(
        {
            "sow_residual_normal_median_px_s": ref_normal_center,
            "sow_residual_normal_p75_px_s": float(np.percentile(np.abs(reference_normal_centered), 75)),
            "sow_residual_normal_rms_px_s": float(np.sqrt(np.mean(reference_normal_centered**2))),
            "sow_residual_tangent_median_px_s": ref_tangent_center,
        }
    )
    if contact_count < max(1, int(min_contact_pixels)) or outer_band_pixel_count <= 0:
        return result

    contact_relative = velocity[contact] - sow_rigid
    contact_normals = normal_field[contact].astype(np.float64)
    contact_tangents = tangent_field[contact]
    normal_after_rigid = np.sum(contact_relative * contact_normals, axis=1)
    tangent_after_rigid = np.sum(contact_relative * contact_tangents, axis=1)
    normal_compensated = normal_after_rigid - ref_normal_center
    tangent_compensated = tangent_after_rigid - ref_tangent_center

    normal_w = _winsorized(normal_compensated, winsor_quantile)
    tangent_w = _winsorized(tangent_compensated, winsor_quantile)
    ref_normal_w = _winsorized(reference_normal_centered, winsor_quantile)
    ref_tangent_w = _winsorized(reference_tangent_centered, winsor_quantile)
    contact_fraction = float(contact_count / outer_band_pixel_count)
    normal_contact_energy = float(np.sum(normal_w**2) / outer_band_pixel_count)
    tangent_contact_energy = float(np.sum(tangent_w**2) / outer_band_pixel_count)
    normal_reference_mean_energy = float(np.mean(ref_normal_w**2))
    tangent_reference_mean_energy = float(np.mean(ref_tangent_w**2))
    normal_reference_energy = contact_fraction * normal_reference_mean_energy
    tangent_reference_energy = contact_fraction * tangent_reference_mean_energy
    normal_excess = normal_contact_energy - float(residual_alpha) * normal_reference_energy
    tangent_excess = tangent_contact_energy - float(residual_alpha) * tangent_reference_energy
    positive_normal = max(0.0, normal_excess)
    positive_tangent = max(0.0, tangent_excess)
    energy_sum = positive_normal + positive_tangent

    contact_common = np.median(contact_relative, axis=0)
    contact_speed = np.linalg.norm(contact_relative, axis=1)
    speed_p75 = float(np.percentile(contact_speed, 75))
    bulk_coherence = float(np.clip(np.linalg.norm(contact_common) / (speed_p75 + 1e-9), 0.0, 1.0))
    common_removed = contact_relative - contact_common
    common_removed_normal = np.sum(common_removed * contact_normals, axis=1) - ref_normal_center
    common_removed_w = _winsorized(common_removed_normal, winsor_quantile)
    common_removed_energy = (
        float(np.sum(common_removed_w**2) / outer_band_pixel_count)
        - float(residual_alpha) * normal_reference_energy
    )

    result.update(
        {
            "after_sow_rigid_normal_signed_mean_px_s": float(np.mean(normal_after_rigid)),
            "after_sow_rigid_normal_signed_median_px_s": float(np.median(normal_after_rigid)),
            "after_sow_rigid_normal_abs_p75_px_s": float(np.percentile(np.abs(normal_after_rigid), 75)),
            "after_sow_rigid_normal_rms_px_s": float(np.sqrt(np.mean(normal_after_rigid**2))),
            "sow_compensated_signed_mean_px_s": float(np.mean(normal_compensated)),
            "sow_compensated_signed_median_px_s": float(np.median(normal_compensated)),
            "normal_contact_energy_px2_s2": normal_contact_energy,
            "tangent_contact_energy_px2_s2": tangent_contact_energy,
            "sow_residual_normal_energy_px2_s2": normal_reference_mean_energy,
            "sow_residual_tangent_energy_px2_s2": tangent_reference_mean_energy,
            "normal_energy_excess_px2_s2": normal_excess,
            "sow_compensated_global_normal_rms_px_s": float(np.sqrt(positive_normal)),
            "tangent_energy_excess_px2_s2": tangent_excess,
            "sow_compensated_global_tangent_rms_px_s": float(np.sqrt(positive_tangent)),
            "normal_energy_fraction": float(positive_normal / energy_sum) if energy_sum > 1e-12 else None,
            "contact_common_flow_x_px_s": float(contact_common[0]),
            "contact_common_flow_y_px_s": float(contact_common[1]),
            "contact_common_speed_px_s": float(np.linalg.norm(contact_common)),
            "bulk_motion_coherence": bulk_coherence,
            "common_removed_global_normal_rms_px_s": float(np.sqrt(max(0.0, common_removed_energy))),
        }
    )
    return result


def _empty_wave_features() -> dict[str, float | None]:
    return {
        "window_signed_signal_rms_px_s": None,
        "bidirectional_energy_balance": None,
        "cycle_closure": None,
        "dominant_frequency_hz": None,
        "autocorrelation_peak": None,
        "autocorrelation_peak_prominence": None,
        "halfcycle_antisymmetry": None,
        "estimated_cycles_in_window": None,
        "spectral_dominant_frequency_hz": None,
        "periodic_band_power_fraction": None,
        "spectral_peak_concentration": None,
        "quality_gated_frequency_hz": None,
    }


def temporal_wave_features(signal: np.ndarray, sample_rate_hz: float, min_hz: float, max_hz: float) -> dict[str, float | None]:
    """Return continuous waveform diagnostics without making a behaviour decision."""
    x = np.asarray(signal, dtype=np.float64)
    if len(x) < 8 or not np.isfinite(x).all() or sample_rate_hz <= 0:
        return _empty_wave_features()
    positive = np.clip(x, 0.0, None)
    negative = np.clip(-x, 0.0, None)
    positive_energy = float(np.sum(positive**2))
    negative_energy = float(np.sum(negative**2))
    larger = max(positive_energy, negative_energy)
    balance = float(min(positive_energy, negative_energy) / larger) if larger > 1e-12 else 0.0
    absolute_sum = float(np.sum(np.abs(x)))
    closure = float(np.clip(1.0 - abs(float(np.sum(x))) / (absolute_sum + 1e-12), 0.0, 1.0))
    if closure < 1e-12:
        closure = 0.0

    index = np.arange(len(x), dtype=np.float64)
    centered_index = index - float(np.mean(index))
    centered_signal = x - float(np.mean(x))
    slope_denominator = float(np.sum(centered_index**2))
    slope = float(np.sum(centered_index * centered_signal) / slope_denominator) if slope_denominator else 0.0
    detrended = centered_signal - slope * centered_index
    detrended -= float(np.median(detrended))
    scale = float(np.median(np.abs(detrended))) * 1.4826
    if scale > 1e-9:
        detrended = np.clip(detrended, -6.0 * scale, 6.0 * scale)
    rms = float(np.sqrt(np.mean(detrended**2)))
    if rms <= 1e-9:
        result = _empty_wave_features()
        result.update(
            {
                "window_signed_signal_rms_px_s": rms,
                "bidirectional_energy_balance": balance,
                "cycle_closure": closure,
            }
        )
        return result

    min_lag = max(1, int(math.ceil(sample_rate_hz / max_hz)))
    max_lag = min(len(x) - 4, int(math.floor(sample_rate_hz / min_hz)))
    candidates: list[tuple[float, int]] = []
    for lag in range(min_lag, max_lag + 1):
        correlation = _safe_corr(detrended[:-lag], detrended[lag:])
        if correlation is not None:
            candidates.append((correlation, lag))
    peak_prominence = None
    if not candidates:
        peak = None
        period_lag = None
    else:
        global_peak = max(item[0] for item in candidates)
        # Autocorrelation repeats at T, 2T, 3T...; prefer the shortest lag
        # whose correlation is effectively tied with the global maximum.
        near_peak = [item for item in candidates if item[0] >= global_peak - 0.03]
        peak, period_lag = min(near_peak, key=lambda item: item[1])
        peak_prominence = float(max(0.0, peak - float(np.median([item[0] for item in candidates]))))

    if period_lag is None:
        dominant_frequency = None
        halfcycle = None
        cycles = None
    else:
        refined_lag = float(period_lag)
        correlations = {lag: correlation for correlation, lag in candidates}
        if period_lag - 1 in correlations and period_lag + 1 in correlations:
            y_minus = correlations[period_lag - 1]
            y_zero = correlations[period_lag]
            y_plus = correlations[period_lag + 1]
            denominator = y_minus - 2.0 * y_zero + y_plus
            if y_zero >= y_minus and y_zero >= y_plus and abs(denominator) > 1e-12:
                offset = 0.5 * (y_minus - y_plus) / denominator
                refined_lag += float(np.clip(offset, -0.5, 0.5))
        dominant_frequency = float(sample_rate_hz / refined_lag)
        half_lag = max(1, int(round(period_lag / 2.0)))
        halfcycle = _safe_corr(detrended[:-half_lag], -detrended[half_lag:])
        cycles = float((len(x) - 1) / refined_lag)

    tapered = detrended * np.hanning(len(detrended))
    spectrum = np.abs(np.fft.rfft(tapered)) ** 2
    frequencies = np.fft.rfftfreq(len(tapered), d=1.0 / sample_rate_hz)
    positive_frequency = frequencies > 0.0
    total_power = float(np.sum(spectrum[positive_frequency]))
    band = positive_frequency & (frequencies >= min_hz) & (frequencies <= max_hz)
    band_power = float(np.sum(spectrum[band]))
    if not np.any(band) or total_power <= 1e-18 or band_power <= 1e-18:
        spectral_frequency = None
        periodic_band_fraction = None
        peak_concentration = None
    else:
        band_indices = np.flatnonzero(band)
        peak_index = int(band_indices[np.argmax(spectrum[band])])
        peak_concentration = float(spectrum[peak_index] / band_power)
        periodic_band_fraction = float(np.clip(band_power / total_power, 0.0, 1.0))
        refined_index = float(peak_index)
        if 0 < peak_index < len(spectrum) - 1:
            log_power = np.log(np.maximum(spectrum[peak_index - 1 : peak_index + 2], 1e-30))
            denominator = log_power[0] - 2.0 * log_power[1] + log_power[2]
            if abs(float(denominator)) > 1e-12:
                offset = 0.5 * float(log_power[0] - log_power[2]) / float(denominator)
                refined_index += float(np.clip(offset, -0.5, 0.5))
        spectral_frequency = float(np.clip(refined_index * sample_rate_hz / len(tapered), min_hz, max_hz))

    return {
        "window_signed_signal_rms_px_s": rms,
        "bidirectional_energy_balance": balance,
        "cycle_closure": closure,
        "dominant_frequency_hz": dominant_frequency,
        "autocorrelation_peak": peak,
        "autocorrelation_peak_prominence": peak_prominence,
        "halfcycle_antisymmetry": halfcycle,
        "estimated_cycles_in_window": cycles,
        "spectral_dominant_frequency_hz": spectral_frequency,
        "periodic_band_power_fraction": periodic_band_fraction,
        "spectral_peak_concentration": peak_concentration,
        "quality_gated_frequency_hz": None,
    }


def _as_float(value: Any) -> float:
    if value in {None, ""}:
        return math.nan
    return float(value)


def _centered_nanmedian(
    values: np.ndarray,
    window: int,
    segments: np.ndarray | None = None,
    *,
    preserve_nan: bool = False,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    output = np.full(len(values), np.nan, dtype=float)
    width = max(1, int(window))
    half_left = (width - 1) // 2
    half_right = width // 2
    for index in range(len(values)):
        if preserve_nan and not np.isfinite(values[index]):
            continue
        start = max(0, index - half_left)
        stop = min(len(values), index + half_right + 1)
        chunk = values[start:stop]
        if segments is not None:
            chunk = chunk[segments[start:stop] == segments[index]]
        finite = chunk[np.isfinite(chunk)]
        if len(finite):
            output[index] = float(np.median(finite))
    return output


def _signed_sqrt(value: float) -> float:
    return float(math.copysign(math.sqrt(abs(value)), value)) if value else 0.0


def add_temporal_features(rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    for row in rows:
        row.update({key: None for key in TEMPORAL_FEATURE_FIELDS})
    if not rows:
        return

    times = np.asarray([float(row["timestamp_sec"]) for row in rows], dtype=float)
    segments = np.asarray([str(row.get("roi_segment_id", "")) for row in rows], dtype=object)
    sample_rates = np.asarray([float(row["actual_flow_fps"]) for row in rows], dtype=float)
    representative_fps = float(np.nanmedian(sample_rates))

    contact_fraction = np.asarray([_as_float(row.get("stable_contact_band_fraction")) for row in rows])
    normal_contact_energy = np.asarray([_as_float(row.get("normal_contact_energy_px2_s2")) for row in rows])
    tangent_contact_energy = np.asarray([_as_float(row.get("tangent_contact_energy_px2_s2")) for row in rows])
    normal_reference_per_pixel = np.asarray(
        [_as_float(row.get("sow_residual_normal_energy_px2_s2")) for row in rows]
    )
    tangent_reference_per_pixel = np.asarray(
        [_as_float(row.get("sow_residual_tangent_energy_px2_s2")) for row in rows]
    )
    reference_width = max(1, int(round(float(args.reference_energy_smooth_sec) * representative_fps)))
    smooth_normal_reference = _centered_nanmedian(normal_reference_per_pixel, reference_width, segments)
    smooth_tangent_reference = _centered_nanmedian(tangent_reference_per_pixel, reference_width, segments)

    for index, row in enumerate(rows):
        values = (
            contact_fraction[index],
            normal_contact_energy[index],
            tangent_contact_energy[index],
            smooth_normal_reference[index],
            smooth_tangent_reference[index],
        )
        if not np.isfinite(values).all():
            continue
        normal_reference = float(contact_fraction[index] * smooth_normal_reference[index])
        tangent_reference = float(contact_fraction[index] * smooth_tangent_reference[index])
        normal_excess = float(normal_contact_energy[index] - args.sow_residual_energy_alpha * normal_reference)
        tangent_excess = float(tangent_contact_energy[index] - args.sow_residual_energy_alpha * tangent_reference)
        denominator = float(
            normal_contact_energy[index]
            + tangent_contact_energy[index]
            + args.sow_residual_energy_alpha * (normal_reference + tangent_reference)
        )
        contrast = (normal_excess - tangent_excess) / (denominator + 1e-12)
        row.update(
            {
                "smoothed_normal_reference_energy_px2_s2": normal_reference,
                "smoothed_tangent_reference_energy_px2_s2": tangent_reference,
                "robust_normal_energy_excess_px2_s2": normal_excess,
                "robust_tangent_energy_excess_px2_s2": tangent_excess,
                "robust_normal_signed_sqrt_excess_px_s": _signed_sqrt(normal_excess),
                "robust_tangent_signed_sqrt_excess_px_s": _signed_sqrt(tangent_excess),
                "robust_global_normal_rms_px_s": float(math.sqrt(max(0.0, normal_excess))),
                "robust_global_tangent_rms_px_s": float(math.sqrt(max(0.0, tangent_excess))),
                "normal_tangent_energy_contrast": float(np.clip(contrast, -1.0, 1.0)),
            }
        )

    # Retain the legacy three-sample series in the CSV for backward comparison.
    for index in range(1, len(rows) - 1):
        group = rows[index - 1 : index + 2]
        segment = rows[index].get("roi_segment_id")
        if segment in {None, ""} or any(item.get("roi_segment_id") != segment for item in group):
            continue
        for source_key, target_key in (
            ("sow_compensated_global_normal_rms_px_s", "sow_compensated_global_normal_med3_px_s"),
            ("common_removed_global_normal_rms_px_s", "common_removed_global_normal_med3_px_s"),
        ):
            values = np.asarray([_as_float(item.get(source_key)) for item in group], dtype=float)
            if np.isfinite(values).all():
                rows[index][target_key] = float(np.median(values))

    signed = np.asarray([_as_float(row.get("sow_compensated_signed_mean_px_s")) for row in rows])
    intensity = np.asarray([_as_float(row.get("robust_global_normal_rms_px_s")) for row in rows])
    half_window = float(args.feature_window_sec) / 2.0

    for index, row in enumerate(rows):
        segment = segments[index]
        if not segment:
            continue
        window_start = times[index] - half_window
        window_end = times[index] + half_window
        left = int(np.searchsorted(times, window_start, side="left"))
        right = int(np.searchsorted(times, window_end, side="right"))
        indices = np.arange(left, right)
        indices = indices[segments[indices] == segment]
        actual_flow_fps = float(row["actual_flow_fps"])
        expected = max(1, int(round(args.feature_window_sec * actual_flow_fps)))
        valid = np.isfinite(signed[indices]) & np.isfinite(intensity[indices])
        valid_fraction = float(np.count_nonzero(valid) / expected)
        row["feature_window_start_sec"] = round(window_start, 6)
        row["feature_window_end_sec"] = round(window_end, 6)
        row["feature_valid_fraction"] = min(1.0, valid_fraction)
        if valid_fraction < args.feature_min_valid_fraction or int(np.count_nonzero(valid)) < 8:
            continue
        valid_indices = indices[valid]
        valid_times = times[valid_indices]
        frame_dt = 1.0 / actual_flow_fps
        if valid_times[0] > window_start + 2.0 * frame_dt or valid_times[-1] < window_end - 2.0 * frame_dt:
            continue
        grid = np.linspace(window_start, window_end, expected)
        interpolated_signed = np.interp(grid, valid_times, signed[valid_indices])
        interpolated_intensity = np.interp(grid, valid_times, intensity[valid_indices])
        row["window_normal_intensity_median_px_s"] = float(np.median(interpolated_intensity))
        wave_features = temporal_wave_features(
            interpolated_signed,
            actual_flow_fps,
            args.feature_min_frequency_hz,
            args.feature_max_frequency_hz,
        )
        autocorrelation = wave_features.get("autocorrelation_peak")
        peak_concentration = wave_features.get("spectral_peak_concentration")
        spectral_frequency = wave_features.get("spectral_dominant_frequency_hz")
        if (
            autocorrelation is not None
            and autocorrelation >= args.frequency_min_autocorrelation
            and peak_concentration is not None
            and peak_concentration >= args.frequency_min_peak_concentration
        ):
            wave_features["quality_gated_frequency_hz"] = spectral_frequency
        row.update(wave_features)


def _new_row(
    *,
    video_name: str,
    sample_index: int,
    current_frame_index: int,
    current_timestamp_sec: float,
    source_fps: float,
    actual_flow_fps: float,
    mask_refreshed: bool,
    roi: base.RoiState | None,
    roi_refreshed: bool,
    status: str,
    raw_metrics: dict[str, float | int | bool | None],
    instant_features: dict[str, float | int | bool | None],
    previous: FeatureFrameState | None,
) -> dict[str, Any]:
    row = base._make_row(
        video_name=video_name,
        sample_index=sample_index,
        current_frame_index=current_frame_index,
        current_timestamp_sec=current_timestamp_sec,
        source_fps=source_fps,
        actual_flow_fps=actual_flow_fps,
        mask_refreshed=mask_refreshed,
        roi=roi,
        roi_refreshed=roi_refreshed,
        status=status,
        metrics=raw_metrics,
        previous=previous,
    )
    row.update(instant_features)
    row.update({key: None for key in TEMPORAL_FEATURE_FIELDS})
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _rolling_nanmedian(values: np.ndarray, window: int, segments: np.ndarray | None = None) -> np.ndarray:
    output = np.full(len(values), np.nan, dtype=float)
    width = max(1, int(window))
    for index in range(len(values)):
        start = max(0, index - width + 1)
        chunk = values[start : index + 1]
        if segments is not None:
            segment = segments[index]
            same_segment = segments[start : index + 1] == segment
            chunk = chunk[same_segment]
        finite = chunk[np.isfinite(chunk)]
        if len(finite):
            output[index] = float(np.median(finite))
    return output


def _odd_window(seconds: float, sample_rate_hz: float, minimum: int = 1) -> int:
    width = max(int(minimum), int(round(float(seconds) * float(sample_rate_hz))))
    if width > 1 and width % 2 == 0:
        width += 1
    return width


def _hampel_filter(
    values: np.ndarray,
    window: int,
    sigma: float,
    segments: np.ndarray | None = None,
) -> np.ndarray:
    """Replace isolated plot outliers with a local median, without crossing ROI segments."""
    values = np.asarray(values, dtype=float)
    output = values.copy()
    if window <= 1 or sigma <= 0:
        return output
    half = max(1, int(window) // 2)
    for index, value in enumerate(values):
        if not np.isfinite(value):
            continue
        start = max(0, index - half)
        stop = min(len(values), index + half + 1)
        chunk = values[start:stop]
        if segments is not None:
            chunk = chunk[segments[start:stop] == segments[index]]
        finite = chunk[np.isfinite(chunk)]
        if len(finite) < 5:
            continue
        median = float(np.median(finite))
        mad = float(np.median(np.abs(finite - median)))
        reference_scale = max(abs(median), float(np.median(np.abs(finite))), 1e-6)
        robust_scale = max(1.4826 * mad, 0.02 * reference_scale)
        if abs(value - median) > float(sigma) * robust_scale:
            output[index] = median
    return output


def _plot_filter(
    values: np.ndarray,
    sample_rate_hz: float,
    segments: np.ndarray,
    *,
    smooth_sec: float,
    outlier_window_sec: float,
    outlier_sigma: float,
) -> np.ndarray:
    outlier_width = _odd_window(outlier_window_sec, sample_rate_hz, minimum=5)
    smooth_width = _odd_window(smooth_sec, sample_rate_hz)
    cleaned = _hampel_filter(values, outlier_width, outlier_sigma, segments)
    return _centered_nanmedian(cleaned, smooth_width, segments, preserve_nan=True)


def _plot_sample_indices(length: int, segments: np.ndarray, stride: int) -> np.ndarray:
    if length <= 0:
        return np.asarray([], dtype=int)
    selected = set(range(0, length, max(1, int(stride))))
    selected.add(length - 1)
    changes = np.flatnonzero(segments[1:] != segments[:-1]) + 1
    for index in changes:
        selected.add(int(index))
        selected.add(int(index - 1))
    return np.asarray(sorted(selected), dtype=int)


def plot_rows(
    path: Path,
    rows: list[dict[str, Any]],
    video_name: str,
    actual_flow_fps: float,
    dpi: int,
    *,
    smooth_sec: float = 1.0,
    outlier_window_sec: float = 2.0,
    outlier_sigma: float = 4.0,
    sample_sec: float = 0.5,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = np.asarray([float(row["timestamp_sec"]) for row in rows]) / 60.0

    def series(key: str) -> np.ndarray:
        return np.asarray([_as_float(row.get(key)) for row in rows], dtype=float)

    segments = np.asarray([str(row.get("roi_segment_id", "")) for row in rows], dtype=object)

    def display_series(key: str) -> np.ndarray:
        return _plot_filter(
            series(key),
            actual_flow_fps,
            segments,
            smooth_sec=smooth_sec,
            outlier_window_sec=outlier_window_sec,
            outlier_sigma=outlier_sigma,
        )

    raw_p75 = display_series("normal_abs_p75_px_s")
    corrected = display_series("robust_global_normal_rms_px_s")
    tangent = display_series("robust_global_tangent_rms_px_s")
    signed_excess = display_series("robust_normal_signed_sqrt_excess_px_s")
    sow_residual = display_series("sow_residual_normal_rms_px_s")
    sow_rigid = display_series("sow_rigid_speed_px_s")
    contrast = display_series("normal_tangent_energy_contrast")
    contact_fraction = display_series("stable_contact_band_fraction")
    autocorrelation = display_series("autocorrelation_peak")
    periodic_band_fraction = display_series("periodic_band_power_fraction")
    frequency = display_series("quality_gated_frequency_hz")
    stride = max(1, int(round(float(sample_sec) * actual_flow_fps)))
    plot_indices = _plot_sample_indices(len(rows), segments, stride)
    plot_times = times[plot_indices]

    figure, axes = plt.subplots(4, 1, figsize=(15, 10), sharex=True)
    axes[0].plot(plot_times, raw_p75[plot_indices], color="#64748B", linewidth=1.0, label="raw normal P75")
    axes[0].plot(plot_times, corrected[plot_indices], color="#1D4ED8", linewidth=1.2, label="robust sow-compensated normal RMS")
    axes[0].plot(plot_times, tangent[plot_indices], color="#EA580C", linewidth=0.9, alpha=0.85, label="robust tangential RMS")
    axes[0].set_ylabel("Flow intensity (px/s)")
    axes[0].legend(loc="upper right", ncol=3, fontsize=8)
    axes[0].grid(alpha=0.2)

    axes[1].plot(plot_times, signed_excess[plot_indices], color="#2563EB", linewidth=1.1, label="signed sqrt normal excess")
    axes[1].plot(plot_times, sow_residual[plot_indices], color="#059669", linewidth=0.9, label="sow residual normal RMS")
    axes[1].plot(plot_times, sow_rigid[plot_indices], color="#7C3AED", linewidth=0.8, alpha=0.85, label="sow rigid speed")
    axes[1].axhline(0.0, color="#475569", linewidth=0.6, alpha=0.6)
    axes[1].set_ylabel("Compensation QC (px/s)")
    axes[1].legend(loc="upper right", ncol=3, fontsize=8)
    axes[1].grid(alpha=0.2)

    axes[2].plot(plot_times, contrast[plot_indices], color="#0F766E", linewidth=1.1, label="normal-tangent energy contrast")
    axes[2].plot(plot_times, contact_fraction[plot_indices], color="#64748B", linewidth=0.9, label="stable contact fraction")
    axes[2].set_ylim(-1.05, 1.05)
    axes[2].set_ylabel("Direction / support")
    axes[2].legend(loc="upper right", ncol=2, fontsize=8)
    axes[2].grid(alpha=0.2)

    axes[3].plot(plot_times, autocorrelation[plot_indices], color="#4338CA", linewidth=1.0, label="autocorrelation peak")
    axes[3].plot(plot_times, periodic_band_fraction[plot_indices], color="#16A34A", linewidth=1.0, label="periodic-band power fraction")
    axes[3].set_ylim(-1.05, 1.05)
    axes[3].set_ylabel("Periodicity quality")
    axes[3].grid(alpha=0.2)
    frequency_axis = axes[3].twinx()
    frequency_axis.scatter(
        plot_times,
        frequency[plot_indices],
        color="#0891B2",
        s=5,
        alpha=0.65,
        label="quality-gated spectral frequency",
    )
    frequency_axis.set_ylabel("Frequency (Hz)")
    handles_a, labels_a = axes[3].get_legend_handles_labels()
    handles_b, labels_b = frequency_axis.get_legend_handles_labels()
    axes[3].legend(handles_a + handles_b, labels_a + labels_b, loc="upper right", fontsize=8)
    axes[3].set_xlabel("Source video time (min)")

    refresh_times = [
        float(row["timestamp_sec"]) / 60.0 for row in rows if int(row.get("roi_refreshed", 0) or 0)
    ]
    for refresh_time in refresh_times:
        axes[2].axvline(refresh_time, color="#9333EA", linestyle=":", linewidth=0.6, alpha=0.45)
    figure.suptitle(
        f"{video_name}\nContinuous numerical features; display only: Hampel {outlier_window_sec:g}s/"
        f"{outlier_sigma:g} MAD, median {smooth_sec:g}s, sample {sample_sec:g}s"
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def process_video(piglet_model: Any, sow_model: Any, video_path: Path, args: argparse.Namespace) -> tuple[Path, Path]:
    stem = base.output_base(video_path, args.input, args.output)
    csv_path = stem.with_name(f"{stem.name}_normal_flow_features.csv")
    plot_path = stem.with_name(f"{stem.name}_normal_flow_features_plot.png")
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
    end_frame = (
        frame_count
        if args.max_seconds is None
        else min(frame_count, start_frame + max(1, int(math.ceil(args.max_seconds * source_fps))))
    )
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    rows: list[dict[str, Any]] = []
    roi: base.RoiState | None = None
    outer_band_pixel_count = 0
    latest_sow: InstanceMask | None = None
    latest_piglets: list[InstanceMask] = []
    latest_contact_mask: np.ndarray | None = None
    latest_sow_reference_mask: np.ndarray | None = None
    latest_contact_count = 0
    previous: FeatureFrameState | None = None
    next_roi_segment_id = 0
    next_sow_check_sec = args.start_sec
    sample_index = 0
    frame_index = start_frame
    last_progress_sec = args.start_sec - args.progress_sec

    try:
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
                        candidate_sow = base._infer_sow(sow_model, frame, args)
                        if latest_sow is None:
                            latest_sow = candidate_sow
                    except Exception as exc:
                        event_status = f"sow_inference_failed:{type(exc).__name__}"
                    next_sow_check_sec = timestamp_sec + max(expected_dt, args.sow_check_sec)
                try:
                    raw_piglets = base._infer_raw_piglets(piglet_model, frame, args)
                except Exception as exc:
                    raw_piglets = []
                    event_status = f"piglet_inference_failed:{type(exc).__name__}"

                reference_sow = candidate_sow or (roi.sow if roi is not None else latest_sow)
                filtered_for_candidate = (
                    base._filter_piglets(raw_piglets, reference_sow, args.max_piglet_sow_overlap)
                    if reference_sow is not None
                    else []
                )
                if roi is None and reference_sow is not None:
                    try:
                        roi = base.build_roi_state(
                            reference_sow,
                            filtered_for_candidate,
                            frame.shape[:2],
                            args,
                            segment_id=next_roi_segment_id,
                        )
                        next_roi_segment_id += 1
                        latest_sow = roi.sow
                        latest_piglets = base._filter_piglets(
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
                                roi = base.build_roi_state(
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
                                latest_contact_count = 0
                                outer_band_pixel_count = 0
                                previous = None
                                roi = None
                    if not refreshed and roi is not None:
                        latest_piglets = base._filter_piglets(
                            raw_piglets, roi.sow, args.max_piglet_sow_overlap
                        )

                if roi is not None:
                    latest_contact_mask, latest_contact_count = base.safe_contact_mask(
                        latest_piglets,
                        roi.band,
                        roi.sow.mask,
                        erode_px=args.contact_erode_px,
                        sow_exclusion_px=args.sow_exclusion_px,
                    )
                    scale = frame.shape[0] / 1080.0
                    latest_sow_reference_mask = sow_surface_reference_mask(
                        latest_piglets,
                        roi.band,
                        roi.sow.mask,
                        min_depth_px=args.sow_reference_min_depth_px * scale,
                        max_depth_px=args.sow_reference_max_depth_px * scale,
                        piglet_guard_px=max(0, int(round(args.sow_reference_piglet_guard_px * scale))),
                    )
                    outer_band_pixel_count = int(
                        np.count_nonzero(mask_for_shape(roi.band, frame.shape[:2]) & ~mask_for_shape(roi.sow.mask, frame.shape[:2]))
                    )

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if roi is None or latest_contact_mask is None or latest_sow_reference_mask is None:
                raw_metrics = base._empty_flow_metrics()
                instant_features = _empty_instant_features()
                rows.append(
                    _new_row(
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
                gray_crop = base._crop(gray, roi.crop_xyxy)
                contact_crop = base._crop(latest_contact_mask, roi.crop_xyxy)
                sow_reference_crop = base._crop(latest_sow_reference_mask, roi.crop_xyxy)
                current = FeatureFrameState(
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
                )
                if roi_refreshed or previous is None or previous.segment_id != roi.segment_id:
                    raw_metrics = base._empty_flow_metrics()
                    instant_features = _empty_instant_features(outer_band_pixels=outer_band_pixel_count)
                    status = event_status if roi_refreshed else "flow_initialized"
                else:
                    dt_sec = timestamp_sec - previous.timestamp_sec
                    if dt_sec <= 0 or dt_sec > expected_dt * 1.75:
                        raw_metrics = base._empty_flow_metrics()
                        instant_features = _empty_instant_features(outer_band_pixels=outer_band_pixel_count)
                        status = "time_gap_reset"
                    else:
                        flow = base.farneback_flow(previous.gray_crop, gray_crop)
                        raw_metrics = base.normal_flow_statistics(
                            flow,
                            roi.normal_field_crop,
                            previous.contact_mask_crop,
                            dt_sec=dt_sec,
                            min_pixels=args.min_flow_pixels,
                        )
                        instant_features = global_flow_features(
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
                    _new_row(
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
                valid_reference = sum(int(bool(row.get("sow_reference_valid"))) for row in rows)
                print(
                    f"[INFO] {video_path.name}: t={timestamp_sec:.1f}s "
                    f"rows={len(rows)} sow_reference_valid={valid_reference}"
                )
                last_progress_sec = timestamp_sec
            sample_index += 1
            frame_index += 1
    finally:
        capture.release()

    if not rows:
        raise RuntimeError(f"No sampled frames were read from {video_path}")
    add_temporal_features(rows, args)
    write_csv(csv_path, rows)
    plot_rows(
        plot_path,
        rows,
        video_path.name,
        actual_flow_fps,
        args.plot_dpi,
        smooth_sec=args.plot_smooth_sec,
        outlier_window_sec=args.plot_outlier_window_sec,
        outlier_sigma=args.plot_outlier_sigma,
        sample_sec=args.plot_sample_sec,
    )
    reference_valid_count = sum(int(bool(row.get("sow_reference_valid"))) for row in rows)
    print(
        f"[INFO] wrote {csv_path} ; {plot_path} "
        f"(rows={len(rows)}, sow_reference_valid={reference_valid_count})"
    )
    return csv_path, plot_path


def validate_args(args: argparse.Namespace) -> None:
    base.validate_args(args)
    if args.sow_reference_min_depth_px < 0 or args.sow_reference_max_depth_px <= args.sow_reference_min_depth_px:
        raise ValueError("Sow reference depth must satisfy 0 <= min < max.")
    if args.sow_reference_piglet_guard_px < 0 or args.min_sow_reference_pixels <= 0:
        raise ValueError("Sow reference guard must be non-negative and minimum pixels positive.")
    if not 0.5 <= args.energy_winsor_quantile <= 1.0:
        raise ValueError("--energy-winsor-quantile must be between 0.5 and 1.0.")
    if args.sow_residual_energy_alpha < 0:
        raise ValueError("--sow-residual-energy-alpha must be non-negative.")
    if args.reference_energy_smooth_sec <= 0:
        raise ValueError("--reference-energy-smooth-sec must be positive.")
    if args.feature_window_sec <= 0 or not 0 < args.feature_min_valid_fraction <= 1:
        raise ValueError("Feature window must be positive and valid fraction in (0, 1].")
    if not 0 < args.feature_min_frequency_hz < args.feature_max_frequency_hz < args.flow_fps / 2.0:
        raise ValueError("Feature frequencies must satisfy 0 < min < max < requested flow Nyquist frequency.")
    if not -1.0 <= args.frequency_min_autocorrelation <= 1.0:
        raise ValueError("--frequency-min-autocorrelation must be in [-1, 1].")
    if not 0.0 <= args.frequency_min_peak_concentration <= 1.0:
        raise ValueError("--frequency-min-peak-concentration must be in [0, 1].")
    if args.plot_smooth_sec <= 0 or args.plot_outlier_window_sec <= 0 or args.plot_sample_sec <= 0:
        raise ValueError("Plot smoothing, outlier window, and sample spacing must be positive.")
    if args.plot_outlier_sigma <= 0:
        raise ValueError("--plot-outlier-sigma must be positive.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    videos = base.iter_videos(args.input, args.recursive)
    if args.max_videos is not None:
        videos = videos[: max(0, args.max_videos)]
    if not videos:
        raise ValueError("No videos selected.")
    if args.dry_run:
        print(f"Input: {args.input}")
        print(f"Videos: {len(videos)}")
        print(f"Output: {args.output}")
        print("Features: robust sow compensation, direction/support energy, quality-gated temporal diagnostics")
        print(
            f"Plot display: Hampel {args.plot_outlier_window_sec:g}s/{args.plot_outlier_sigma:g} MAD, "
            f"median {args.plot_smooth_sec:g}s, sample {args.plot_sample_sec:g}s"
        )
        print("Behaviour classification: disabled")
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
