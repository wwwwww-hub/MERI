"""MERI event measurements, physiological suppression and nursing attempts."""

from __future__ import annotations


import argparse


import json


import math


import re


import sys


from collections import defaultdict


from dataclasses import dataclass


from datetime import datetime, timedelta, timezone


from pathlib import Path


from typing import Any, Iterable, Mapping, Sequence


import numpy as np


import pandas as pd


VIDEO_TIME_RE = re.compile(
    r"^(?P<channel>ch\d+)_(?P<segment_start>\d{14})_(?P<segment_end>\d{14})"
    r"_nursing_\d+_(?P<clip_start>\d{2}-\d{2}-\d{2}-\d{3})_"
    r"(?P<clip_end>\d{2}-\d{2}-\d{2}-\d{3})$",
    flags=re.IGNORECASE,
)


INTENSITY = "tracked_periodic_intensity_p75_px_s__median"


PERIODICITY = "tracked_periodicity_quality_p75__median"


VALID_TRACK_FRACTION = "tracked_valid_track_fraction__median"


TEMPORAL_VALID_FRACTION = "tracked_temporal_valid_track_fraction__median"


NORMAL_DOMINANCE = "tracked_normal_dominant_track_fraction__median"


VALID_TRACK_COUNT = "tracked_valid_track_count__median"


CONTACT_FRACTION = "udder_contact_fraction"


FLOW_VALID_FRACTION = "flow_valid_fraction"


TANGENTIAL_MOTION = "tracked_tangent_rms_p75_px_s__median"


GROUP_MOTION = "contact_common_speed_px_s__median"


SOW_MOTION = "sow_rigid_speed_px_s__median"


DEFAULT_PRE_BURDEN_SEC = 120.0


DEFAULT_PRE_BURDEN_WEIGHT = 0.10


DEFAULT_POST_BURDEN_WEIGHT = 0.15


DEFAULT_FAILURE_WEIGHT = 0.25


DEFAULT_ATTEMPT_PRE_SEC = 240.0


DEFAULT_ATTEMPT_POST_SEC = 180.0


DEFAULT_ATTEMPT_MERGE_GAP_SEC = 30.0


DEFAULT_ATTEMPT_MIN_DURATION_SEC = 60.0


DEFAULT_ATTEMPT_MIN_PROBABILITY = 0.50


DEFAULT_PRE_BURDEN_Q75 = 0.41622215214446556


DEFAULT_PRE_BURDEN_Q95 = 0.5366664078632267


DEFAULT_POST_BURDEN_Q75 = 0.4665808790143308


DEFAULT_POST_BURDEN_Q95 = 0.5942129171667692


DEFAULT_DURATION_P10_SEC = 12.0


DEFAULT_DURATION_P90_SEC = 64.0


EQUAL_COMPONENT_WEIGHT = 0.25


MINIMUM_PHASE_COVERAGE = 0.50


AUDIT_COLUMNS = (
    INTENSITY,
    PERIODICITY,
    VALID_TRACK_FRACTION,
    TEMPORAL_VALID_FRACTION,
    NORMAL_DOMINANCE,
    VALID_TRACK_COUNT,
    CONTACT_FRACTION,
    FLOW_VALID_FRACTION,
    TANGENTIAL_MOTION,
    GROUP_MOTION,
    SOW_MOTION,
    "tracked_reliable_track_fraction__median",
    "tracked_reliable_frequency_median_hz__median",
    "source_piglet_count__median",
    "source_contact_piglet_count__median",
    "missing_fraction",
    "source_row_count",
)


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _clip01(value: Any) -> float | None:
    number = _finite(value)
    return None if number is None else float(np.clip(number, 0.0, 1.0))


def _geometric_mean(values: Iterable[Any]) -> float | None:
    finite = [_clip01(value) for value in values]
    if any(value is None for value in finite):
        return None
    numbers = np.asarray(finite, dtype=float)
    if np.any(numbers <= 0.0):
        return 0.0
    return float(np.exp(np.mean(np.log(numbers))))


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float | None:
    return _weighted_quantile(values, weights, 0.5)


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float | None:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not bool(valid.any()):
        return None
    x = values[valid]
    w = weights[valid]
    order = np.argsort(x, kind="mergesort")
    x = x[order]
    w = w[order]
    cutoff = float(np.clip(quantile, 0.0, 1.0)) * float(np.sum(w))
    index = int(np.searchsorted(np.cumsum(w), cutoff, side="left"))
    return float(x[min(index, len(x) - 1)])


def _parse_offset(value: str) -> float:
    hours, minutes, seconds, milliseconds = (int(part) for part in value.split("-"))
    return hours * 3600.0 + minutes * 60.0 + seconds + milliseconds / 1000.0


def video_absolute_origin(
    video: str, utc_offset_hours: float, source: pd.DataFrame | None = None
) -> datetime:
    if source is not None and "absolute_start" in source.columns and len(source):
        first_start = pd.to_datetime(source["absolute_start"].iloc[0]).to_pydatetime()
        first_bin = float(source["bin_start_sec"].iloc[0]) if "bin_start_sec" in source.columns else 0.0
        return first_start - timedelta(seconds=first_bin)
    match = VIDEO_TIME_RE.fullmatch(video)
    if match is None:
        raise ValueError(f"Cannot parse channel/timestamps from source video stem: {video}")
    zone = timezone(timedelta(hours=float(utc_offset_hours)))
    segment_start = datetime.strptime(match.group("segment_start"), "%Y%m%d%H%M%S").replace(tzinfo=zone)
    return segment_start + timedelta(seconds=_parse_offset(match.group("clip_start")))


def _phase_summary(frame: pd.DataFrame, start: float, end: float) -> dict[str, Any]:
    if end <= start:
        return {"duration_sec": 0.0, "coverage": 0.0}
    left = frame["bin_start_sec"].to_numpy(dtype=float)
    right = frame["bin_end_sec"].to_numpy(dtype=float)
    weights = np.maximum(0.0, np.minimum(right, end) - np.maximum(left, start))
    selected = weights > 1e-9
    result: dict[str, Any] = {
        "duration_sec": float(end - start),
        "observed_sec": float(np.sum(weights[selected])),
        "coverage": float(np.clip(np.sum(weights[selected]) / (end - start), 0.0, 1.0)),
    }
    for name, column in (
        ("intensity_px_s", INTENSITY),
        ("periodicity_quality", PERIODICITY),
        ("valid_track_fraction", VALID_TRACK_FRACTION),
        ("temporal_valid_track_fraction", TEMPORAL_VALID_FRACTION),
        ("normal_dominance", NORMAL_DOMINANCE),
        ("valid_track_count", VALID_TRACK_COUNT),
        ("contact_fraction", CONTACT_FRACTION),
        ("flow_valid_fraction", FLOW_VALID_FRACTION),
        ("tangential_motion_px_s", TANGENTIAL_MOTION),
        ("group_motion_px_s", GROUP_MOTION),
        ("sow_motion_px_s", SOW_MOTION),
        ("milk_probability", "milk_probability_smoothed"),
        ("arching_collective_probability", "arching_probability_collective_smoothed"),
    ):
        if column not in frame:
            result[name] = None
            result[f"{name}_finite_coverage"] = 0.0
            continue
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        finite = selected & np.isfinite(values)
        result[name] = _weighted_median(values[finite], weights[finite])
        result[f"{name}_finite_coverage"] = float(
            np.clip(np.sum(weights[finite]) / (end - start), 0.0, 1.0)
        )
    if CONTACT_FRACTION in frame:
        contact = pd.to_numeric(frame[CONTACT_FRACTION], errors="coerce").to_numpy(dtype=float)
        finite = selected & np.isfinite(contact)
        q25 = _weighted_quantile(contact[finite], weights[finite], 0.25)
        q75 = _weighted_quantile(contact[finite], weights[finite], 0.75)
        result["contact_fraction_iqr"] = (
            None if q25 is None or q75 is None else float(max(0.0, q75 - q25))
        )
    else:
        result["contact_fraction_iqr"] = None
    return result


def _normalized_intensity(value: Any, calibration: Mapping[str, Any]) -> float | None:
    number = _finite(value)
    if number is None or number < 0.0:
        return None
    low = float(calibration["log1p_p10"])
    high = float(calibration["log1p_p90"])
    return float(np.clip((math.log1p(number) - low) / (high - low), 0.0, 1.0))


def _transition_score(before: Any, during: Any, calibration: Mapping[str, Any]) -> float | None:
    left = _finite(before)
    right = _finite(during)
    if left is None or right is None or left < 0.0 or right < 0.0:
        return None
    scale = float(calibration["log1p_p90"]) - float(calibration["log1p_p10"])
    return float(np.clip((math.log1p(left) - math.log1p(right)) / scale, 0.0, 1.0))


def _positive_relative_excess(during: Any, before: Any) -> float | None:
    """Return the scale-free positive increase from before to during."""
    current = _finite(during)
    baseline = _finite(before)
    if current is None or baseline is None or current < 0.0 or baseline < 0.0:
        return None
    denominator = current + baseline
    if denominator <= 1e-12:
        return 0.0
    return float(np.clip((current - baseline) / denominator, 0.0, 1.0))


def _weighted_available_mean(values: Sequence[tuple[Any, float]]) -> float | None:
    usable = [(_clip01(value), float(weight)) for value, weight in values]
    usable = [(value, weight) for value, weight in usable if value is not None and weight > 0.0]
    if not usable:
        return None
    total = sum(weight for _, weight in usable)
    return float(sum(float(value) * weight for value, weight in usable) / total)


def _release_disturbance(
    pre: Mapping[str, Any],
    release: Mapping[str, Any],
    calibration: Mapping[str, Any],
) -> float | None:
    """Quantify gross or udder-disruptive movement during milk ejection."""
    release_arching = _normalized_intensity(release.get("intensity_px_s"), calibration)
    tangential_excess = _positive_relative_excess(
        release.get("tangential_motion_px_s"), pre.get("tangential_motion_px_s")
    )
    group_excess = _positive_relative_excess(
        release.get("group_motion_px_s"), pre.get("group_motion_px_s")
    )
    sow_excess = _positive_relative_excess(
        release.get("sow_motion_px_s"), pre.get("sow_motion_px_s")
    )
    contact_instability = _finite(release.get("contact_fraction_iqr"))
    contact_instability = (
        None
        if contact_instability is None
        else float(np.clip(contact_instability / 0.50, 0.0, 1.0))
    )
    return _weighted_available_mean(
        (
            (release_arching, 0.35),
            (tangential_excess, 0.20),
            (group_excess, 0.20),
            (sow_excess, 0.15),
            (contact_instability, 0.10),
        )
    )


def _organized_massage_load(
    frame: pd.DataFrame,
    start: float,
    end: float,
    calibration: Mapping[str, Any],
) -> dict[str, float | None]:
    """Integrate organized udder-directed massage over an intended window."""
    duration = float(end - start)
    if duration <= 0.0:
        return {"load": None, "coverage": 0.0, "finite_coverage": 0.0}
    left = frame["bin_start_sec"].to_numpy(dtype=float)
    right = frame["bin_end_sec"].to_numpy(dtype=float)
    weights = np.maximum(0.0, np.minimum(right, end) - np.maximum(left, start))
    selected = weights > 1e-9
    intensity = pd.to_numeric(frame.get(INTENSITY), errors="coerce").to_numpy(dtype=float)
    low = float(calibration["log1p_p10"])
    high = float(calibration["log1p_p90"])
    normalized_intensity = np.clip((np.log1p(intensity) - low) / (high - low), 0.0, 1.0)
    quality = np.clip(
        pd.to_numeric(frame.get(PERIODICITY), errors="coerce").to_numpy(dtype=float), 0.0, 1.0
    )
    normal = np.clip(
        pd.to_numeric(frame.get(NORMAL_DOMINANCE), errors="coerce").to_numpy(dtype=float),
        0.0,
        1.0,
    )
    contact = np.clip(
        pd.to_numeric(frame.get(CONTACT_FRACTION), errors="coerce").to_numpy(dtype=float),
        0.0,
        1.0,
    )
    if {
        "arching_probability_partial_smoothed",
        "arching_probability_collective_smoothed",
    }.issubset(frame.columns):
        arching_probability = np.clip(
            pd.to_numeric(
                frame["arching_probability_partial_smoothed"], errors="coerce"
            ).to_numpy(dtype=float)
            + pd.to_numeric(
                frame["arching_probability_collective_smoothed"], errors="coerce"
            ).to_numpy(dtype=float),
            0.0,
            1.0,
        )
    else:
        arching_probability = np.ones(len(frame), dtype=float)
    activity = (
        normalized_intensity * quality * normal * contact
    ) ** 0.25 * arching_probability
    finite = selected & np.isfinite(activity)
    observed = float(np.sum(weights[selected]))
    finite_observed = float(np.sum(weights[finite]))
    load = None
    if finite_observed > 0.0:
        load = float(np.sum(activity[finite] * weights[finite]) / duration)
    return {
        "load": load,
        "coverage": float(np.clip(observed / duration, 0.0, 1.0)),
        "finite_coverage": float(np.clip(finite_observed / duration, 0.0, 1.0)),
    }


def _excess_burden(value: Any, low: float, high: float) -> float | None:
    number = _finite(value)
    if number is None or high <= low:
        return None
    return float(np.clip((number - low) / (high - low), 0.0, 1.0))


def _event_runs(frame: pd.DataFrame, bin_sec: float) -> list[tuple[float, float]]:
    labels = frame["milk_prediction"].to_numpy(dtype=int)
    starts = frame["bin_start_sec"].to_numpy(dtype=float)
    ends = frame["bin_end_sec"].to_numpy(dtype=float)
    runs: list[tuple[float, float]] = []
    begin: int | None = None
    for index, label in enumerate(labels):
        contiguous = index == 0 or abs(starts[index] - starts[index - 1] - bin_sec) <= 1e-6
        if label == 1 and (begin is None or not contiguous):
            if begin is not None:
                runs.append((float(starts[begin]), float(ends[index - 1])))
            begin = index
        elif label != 1 and begin is not None:
            runs.append((float(starts[begin]), float(ends[index - 1])))
            begin = None
    if begin is not None:
        runs.append((float(starts[begin]), float(ends[-1])))
    return runs


def score_events(
    bins: pd.DataFrame,
    calibration: Mapping[str, Any],
    *,
    pre_sec: float,
    pre_burden_sec: float,
    post_sec: float,
    pre_burden_weight: float,
    post_burden_weight: float,
    utc_offset_hours: float,
) -> pd.DataFrame:
    """Calculate milk-ejection benefit minus excess massage burden."""
    rows: list[dict[str, Any]] = []
    bin_sec_values = bins["bin_end_sec"] - bins["bin_start_sec"]
    bin_sec = float(bin_sec_values.quantile(0.9))
    for video, source in bins.groupby("video", sort=True, observed=True):
        source = source.sort_values("bin_start_sec").reset_index(drop=True)
        origin = video_absolute_origin(str(video), utc_offset_hours, source)
        for event_number, (start, end) in enumerate(_event_runs(source, bin_sec), start=1):
            # Use the intended windows even when they cross a clip edge so that
            # phase coverage reflects missing observation instead of silently
            # shortening the denominator.
            pre = _phase_summary(source, start - pre_sec, start)
            release = _phase_summary(source, start, end)
            post = _phase_summary(source, end, end + post_sec)
            pre_load = _organized_massage_load(
                source, start - pre_burden_sec, start, calibration
            )
            post_load = _organized_massage_load(source, end, end + post_sec, calibration)

            pre_intensity = _normalized_intensity(pre.get("intensity_px_s"), calibration)
            pre_stimulation = _geometric_mean(
                (pre_intensity, pre.get("periodicity_quality"), pre.get("normal_dominance"))
            )
            post_intensity = _normalized_intensity(post.get("intensity_px_s"), calibration)
            post_massage = _geometric_mean(
                (
                    post_intensity,
                    post.get("periodicity_quality"),
                    post.get("normal_dominance"),
                    post.get("contact_fraction"),
                )
            )
            release_drop = _transition_score(
                pre.get("intensity_px_s"), release.get("intensity_px_s"), calibration
            )
            release_contact = _clip01(release.get("contact_fraction"))
            release_disturbance = _release_disturbance(pre, release, calibration)
            release_response = _geometric_mean(
                (
                    release_drop,
                    release_contact,
                    None if release_disturbance is None else 1.0 - release_disturbance,
                )
            )
            pre_excess_burden = _excess_burden(
                pre_load.get("load"),
                float(calibration.get("pre_burden_q75", DEFAULT_PRE_BURDEN_Q75)),
                float(calibration.get("pre_burden_q95", DEFAULT_PRE_BURDEN_Q95)),
            )
            post_excess_burden = _excess_burden(
                post_load.get("load"),
                float(calibration.get("post_burden_q75", DEFAULT_POST_BURDEN_Q75)),
                float(calibration.get("post_burden_q95", DEFAULT_POST_BURDEN_Q95)),
            )
            post_rebound = _transition_score(
                post.get("intensity_px_s"), release.get("intensity_px_s"), calibration
            )
            participation = _weighted_median(
                np.asarray(
                    [pre.get("contact_fraction", np.nan), release.get("contact_fraction", np.nan)],
                    dtype=float,
                ),
                np.asarray([pre.get("observed_sec", 0.0), release.get("observed_sec", 0.0)]),
            )
            participation = _clip01(participation)
            if any(value is None for value in (pre_stimulation, release_drop, participation)):
                core_meri = None
                core_score_status = "insufficient_signal"
            elif min(float(pre["coverage"]), float(release["coverage"])) < MINIMUM_PHASE_COVERAGE:
                core_meri = None
                core_score_status = "insufficient_phase_coverage"
            elif min(
                float(pre.get("intensity_px_s_finite_coverage", 0.0)),
                float(release.get("intensity_px_s_finite_coverage", 0.0)),
            ) < MINIMUM_PHASE_COVERAGE:
                core_meri = None
                core_score_status = "insufficient_intensity_coverage"
            else:
                core_meri_fraction = (
                    float(release_drop) ** 0.50
                    * float(pre_stimulation) ** 0.30
                    * float(participation) ** 0.20
                )
                core_meri = 100.0 * core_meri_fraction
                core_score_status = "scored"

            phase_coverages = (
                pre.get("coverage", 0.0),
                release.get("coverage", 0.0),
                post.get("coverage", 0.0),
                pre_load.get("coverage", 0.0),
                post_load.get("coverage", 0.0),
            )
            signal_coverages = (
                pre.get("intensity_px_s_finite_coverage", 0.0),
                release.get("intensity_px_s_finite_coverage", 0.0),
                release.get("contact_fraction_finite_coverage", 0.0),
                pre_load.get("finite_coverage", 0.0),
                post_load.get("finite_coverage", 0.0),
            )
            if any(
                value is None
                for value in (
                    release_response,
                    pre_excess_burden,
                    post_excess_burden,
                )
            ):
                meri = None
                score_status = "insufficient_efficiency_signal"
            elif min(float(value) for value in phase_coverages) < MINIMUM_PHASE_COVERAGE:
                meri = None
                score_status = "insufficient_efficiency_phase_coverage"
            elif min(float(value) for value in signal_coverages) < MINIMUM_PHASE_COVERAGE:
                meri = None
                score_status = "insufficient_efficiency_signal_coverage"
            else:
                meri = 100.0 * float(release_response) * (
                    1.0 - pre_burden_weight * float(pre_excess_burden)
                ) * (
                    1.0 - post_burden_weight * float(post_excess_burden)
                )
                score_status = "scored"

            # Quiet suckling during milk ejection can legitimately drive the
            # temporal-periodicity track fraction to zero.  Confidence therefore
            # measures whether motion/track observations exist, while pre-release
            # periodic organization remains a biological MERI component above.
            core_reliability_values = []
            for phase in (pre, release):
                core_reliability_values.extend(
                    [
                        phase.get("valid_track_fraction"),
                        phase.get("flow_valid_fraction"),
                    ]
                )
            core_reliability = _geometric_mean(core_reliability_values)
            core_coverage = min(float(pre["coverage"]), float(release["coverage"]))
            milk_probability = release.get("milk_probability")
            core_confidence = _geometric_mean(
                (milk_probability, core_reliability, core_coverage)
            )
            core_confidence = None if core_confidence is None else 100.0 * core_confidence
            post_reliability = _geometric_mean(
                (post.get("valid_track_fraction"), post.get("flow_valid_fraction"))
            )
            reliability = _geometric_mean((core_reliability, post_reliability))
            coverage = min(float(value) for value in phase_coverages)
            confidence = _geometric_mean((milk_probability, reliability, coverage))
            confidence = None if confidence is None else 100.0 * confidence

            absolute_start = origin + timedelta(seconds=start)
            absolute_end = origin + timedelta(seconds=end)
            rows.append(
                {
                    "event_id": f"{video}__milk_{event_number:02d}",
                    "video": video,
                    "local_start_sec": start,
                    "local_end_sec": end,
                    "duration_sec": end - start,
                    "absolute_start": absolute_start.isoformat(),
                    "absolute_end": absolute_end.isoformat(),
                    "absolute_midpoint_epoch_sec": (absolute_start + (absolute_end - absolute_start) / 2).timestamp(),
                    "date": absolute_start.date().isoformat(),
                    "meri": meri,
                    "core_meri": core_meri,
                    "meri_confidence": confidence,
                    "core_meri_confidence": core_confidence,
                    "score_status": score_status,
                    "core_score_status": core_score_status,
                    "release_response_score": (
                        None if release_response is None else 100.0 * release_response
                    ),
                    "release_disturbance_score": (
                        None if release_disturbance is None else 100.0 * release_disturbance
                    ),
                    "release_contact_score": (
                        None if release_contact is None else 100.0 * release_contact
                    ),
                    "pre_massage_load_score": (
                        None if pre_load.get("load") is None else 100.0 * float(pre_load["load"])
                    ),
                    "pre_excess_burden_score": (
                        None if pre_excess_burden is None else 100.0 * pre_excess_burden
                    ),
                    "post_massage_load_score": (
                        None if post_load.get("load") is None else 100.0 * float(post_load["load"])
                    ),
                    "post_excess_burden_score": (
                        None if post_excess_burden is None else 100.0 * post_excess_burden
                    ),
                    "pre_stimulation_score": None if pre_stimulation is None else 100.0 * pre_stimulation,
                    "release_drop_score": None if release_drop is None else 100.0 * release_drop,
                    "participation_score": None if participation is None else 100.0 * participation,
                    "post_massage_score": None if post_massage is None else 100.0 * post_massage,
                    "post_rebound_score": None if post_rebound is None else 100.0 * post_rebound,
                    "measurement_reliability": None if reliability is None else 100.0 * reliability,
                    "model_probability": milk_probability,
                    "pre_coverage": pre.get("coverage"),
                    "release_coverage": release.get("coverage"),
                    "post_coverage": post.get("coverage"),
                    "pre_burden_coverage": pre_load.get("coverage"),
                    "post_burden_coverage": post_load.get("coverage"),
                    "pre_burden_finite_coverage": pre_load.get("finite_coverage"),
                    "post_burden_finite_coverage": post_load.get("finite_coverage"),
                    "pre_intensity_px_s": pre.get("intensity_px_s"),
                    "release_intensity_px_s": release.get("intensity_px_s"),
                    "post_intensity_px_s": post.get("intensity_px_s"),
                    "pre_periodicity_quality": pre.get("periodicity_quality"),
                    "pre_normal_dominance": pre.get("normal_dominance"),
                    "post_periodicity_quality": post.get("periodicity_quality"),
                    "post_normal_dominance": post.get("normal_dominance"),
                    "post_contact_fraction": post.get("contact_fraction"),
                    "contact_fraction": participation,
                    "pre_valid_track_count": pre.get("valid_track_count"),
                    "release_valid_track_count": release.get("valid_track_count"),
                    "post_valid_track_count": post.get("valid_track_count"),
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        result.sort_values("absolute_midpoint_epoch_sec", inplace=True)
        result.reset_index(drop=True, inplace=True)
    return result


def physiological_nms(events: pd.DataFrame, refractory_sec: float) -> pd.DataFrame:
    """Suppress lower-probability event duplicates inside a conservative refractory window."""
    result = events.copy()
    if result.empty:
        result["physiology_filter"] = pd.Series(dtype=str)
        return result
    result["physiology_filter"] = "kept"
    result["suppressed_by_event_id"] = ""
    ranking = result.assign(
        _rank_probability=pd.to_numeric(result["model_probability"], errors="coerce").fillna(-1.0),
        _rank_confidence=pd.to_numeric(result["meri_confidence"], errors="coerce").fillna(-1.0),
    ).sort_values(
        ["_rank_probability", "_rank_confidence", "absolute_midpoint_epoch_sec"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    kept: list[int] = []
    for index, row in ranking.iterrows():
        midpoint = float(row["absolute_midpoint_epoch_sec"])
        conflicting = [
            other
            for other in kept
            if abs(midpoint - float(result.at[other, "absolute_midpoint_epoch_sec"])) < refractory_sec
        ]
        if not conflicting:
            kept.append(index)
            continue
        winner = min(
            conflicting,
            key=lambda other: abs(midpoint - float(result.at[other, "absolute_midpoint_epoch_sec"])),
        )
        result.at[index, "physiology_filter"] = "refractory_suppressed"
        result.at[index, "suppressed_by_event_id"] = str(result.at[winner, "event_id"])
    return result.sort_values("absolute_midpoint_epoch_sec").reset_index(drop=True)


def nursing_attempts(
    bins: pd.DataFrame,
    events: pd.DataFrame,
    *,
    pre_association_sec: float,
    post_association_sec: float,
    merge_gap_sec: float,
    minimum_duration_sec: float,
    minimum_probability: float,
    utc_offset_hours: float,
) -> pd.DataFrame:
    """Return successful releases and sustained unassociated collective massage bouts."""
    kept = events[events["physiology_filter"].eq("kept")].copy()
    rows: list[dict[str, Any]] = []
    for row in kept.itertuples():
        rows.append(
            {
                "attempt_id": f"release__{row.event_id}",
                "attempt_status": "release",
                "absolute_start": row.absolute_start,
                "absolute_end": row.absolute_end,
                "duration_sec": float(row.duration_sec),
                "collective_probability": None,
                "linked_event_id": row.event_id,
                "date": row.date,
            }
        )
    if bins.empty or "arching_prediction" not in bins:
        return pd.DataFrame(rows)

    source = bins[
        pd.to_numeric(bins["arching_prediction"], errors="coerce").eq(2)
    ].copy()
    if source.empty:
        return pd.DataFrame(rows)
    source["_start"] = pd.to_datetime(source["absolute_start"], utc=True, format="mixed")
    source["_end"] = pd.to_datetime(source["absolute_end"], utc=True, format="mixed")
    source["_probability"] = pd.to_numeric(
        source["arching_probability_collective_smoothed"], errors="coerce"
    )
    source.sort_values("_probability", ascending=False, inplace=True, kind="mergesort")
    source.drop_duplicates(["_start", "_end"], keep="first", inplace=True)
    source.sort_values("_start", inplace=True, kind="mergesort")

    # Timestamp integer resolution changed from ns to us in newer pandas.
    # Using timestamp() avoids silently shrinking 60 s runs to 0.06 s.
    starts = source["_start"].map(lambda value: float(value.timestamp())).to_numpy()
    ends = source["_end"].map(lambda value: float(value.timestamp())).to_numpy()
    associated = np.zeros(len(source), dtype=bool)
    if not kept.empty:
        release_starts = (
            pd.to_datetime(kept["absolute_start"], utc=True, format="mixed")
            .map(lambda value: float(value.timestamp()))
            .to_numpy()
            - pre_association_sec
        )
        release_ends = (
            pd.to_datetime(kept["absolute_end"], utc=True, format="mixed")
            .map(lambda value: float(value.timestamp()))
            .to_numpy()
            + post_association_sec
        )
        for left, right in zip(release_starts, release_ends):
            associated |= (ends > left) & (starts < right)

    candidates = source.loc[~associated].sort_values("_start", kind="mergesort")
    runs: list[dict[str, Any]] = []
    for start_time, end_time, probability_value in candidates[
        ["_start", "_end", "_probability"]
    ].itertuples(index=False, name=None):
        start = float(start_time.timestamp())
        end = float(end_time.timestamp())
        probability = _finite(probability_value)
        if not runs or start - float(runs[-1]["end"]) > merge_gap_sec:
            runs.append({"start": start, "end": end, "probabilities": [probability]})
        else:
            runs[-1]["end"] = max(float(runs[-1]["end"]), end)
            runs[-1]["probabilities"].append(probability)

    failure_number = 0
    for run in runs:
        probabilities = [value for value in run["probabilities"] if value is not None]
        probability = float(np.mean(probabilities)) if probabilities else None
        duration = float(run["end"] - run["start"])
        if duration < minimum_duration_sec:
            continue
        if probability is None or probability < minimum_probability:
            continue
        failure_number += 1
        local_zone = timezone(timedelta(hours=float(utc_offset_hours)))
        absolute_start = datetime.fromtimestamp(float(run["start"]), local_zone)
        absolute_end = datetime.fromtimestamp(float(run["end"]), local_zone)
        rows.append(
            {
                "attempt_id": f"unassociated_collective__{failure_number:04d}",
                "attempt_status": "unassociated_collective",
                "absolute_start": absolute_start.isoformat(),
                "absolute_end": absolute_end.isoformat(),
                "duration_sec": duration,
                "collective_probability": probability,
                "linked_event_id": "",
                "date": absolute_start.date().isoformat(),
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result.sort_values("absolute_start", inplace=True, kind="mergesort")
        result.reset_index(drop=True, inplace=True)
    return result


def attempt_failure_fraction(attempts: pd.DataFrame) -> float | None:
    if attempts.empty:
        return None
    statuses = attempts["attempt_status"].astype(str)
    successful = int(statuses.eq("release").sum())
    failed = int(statuses.eq("unassociated_collective").sum())
    total = successful + failed
    return None if total == 0 else float(failed / total)


def equal4_eligible_events(
    events: pd.DataFrame,
    minimum_confidence: float,
) -> pd.DataFrame:
    """Select retained releases with sufficient signal confidence and phase coverage.

    K, Dme and P use these event measurements; A uses the full attempt table.
    Missing pre-release burden leaves K undefined in the final summary.
    """
    required = {
        "physiology_filter",
        "meri_confidence",
        "release_drop_score",
        "duration_sec",
        "post_excess_burden_score",
        "pre_coverage",
        "release_coverage",
        "post_burden_coverage",
        "post_burden_finite_coverage",
    }
    missing = required - set(events.columns)
    if missing:
        raise ValueError(
            f"Cannot calculate equal-weight MERI; missing event fields: {sorted(missing)}"
        )
    mask = events["physiology_filter"].astype(str).eq("kept")
    mask &= pd.to_numeric(events["meri_confidence"], errors="coerce").ge(
        minimum_confidence
    )
    for column in (
        "release_drop_score",
        "duration_sec",
        "post_excess_burden_score",
    ):
        mask &= pd.to_numeric(events[column], errors="coerce").notna()
    for column in (
        "pre_coverage",
        "release_coverage",
        "post_burden_coverage",
        "post_burden_finite_coverage",
    ):
        mask &= pd.to_numeric(events[column], errors="coerce").ge(
            MINIMUM_PHASE_COVERAGE
        )
    return events.loc[mask].copy()

