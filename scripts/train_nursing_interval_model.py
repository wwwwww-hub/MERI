"""Train and apply an offline numerical model for nursing behaviour intervals.

The model consumes only the aggregate
``*_tracked_normal_flow_features.csv`` outputs.  Those files already combine
segmentation, tracking, optical-flow and periodicity measurements; the much
larger track-long CSVs are not read.  Feature engineering uses centered rolling
windows, so this is an offline interval-localization model, not an online alarm.

The event-optimized version uses one-second bins and event-balanced sample
weights so short intervals are not dominated by long runs during training.
Two heads preserve the topology of the confirmed labels:

* ``milk`` is binary;
* ``arching_state`` is mutually exclusive ``none / partial / collective``.

The two heads are independent, therefore milk and arching may overlap.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.nursing_interval_annotations import (  # noqa: E402
    build_annotation_report,
    parse_confirmed_annotations,
    rasterize_annotation,
)
from scripts.evaluate_nursing_interval_events import evaluate_events  # noqa: E402


DEFAULT_ANNOTATIONS = ROOT / "data/DS4_intervals/annotations/confirm_milking.md"
DEFAULT_FEATURES = ROOT / "working_data/DS4_intervals/features"
DEFAULT_OUTPUT = ROOT / "reproduced/intervals"
FEATURE_GLOB = "*_tracked_normal_flow_features.csv"
PREPROCESSING_VERSION = "aggregate_features_milk_phase_optimized"
MODEL_NAME = "offline_aggregate_dual_head_milk_phase_optimized_hist_gradient_boosting"

ARCHING_CLASSES = {0: "none", 1: "partial", 2: "collective"}
METADATA_COLUMNS = {
    "video",
    "source_path",
    "bin_start_sec",
    "bin_end_sec",
    "source_duration_sec",
}
LABEL_COLUMNS = {"milk", "milk_known", "arching_state", "arching_known", "fold"}

# These receive distributional aggregation and centered temporal context when
# present in the source schema.  Every other selected numerical signal still
# receives a per-bin median and per-video robust z-score.
KEY_SIGNAL_CANDIDATES = (
    "tracked_periodic_intensity_p75_px_s",
    "tracked_periodicity_quality_p75",
    "tracked_normal_rms_p75_px_s",
    "tracked_tangent_rms_p75_px_s",
    "tracked_phase_cancellation_index",
    "tracked_filtered_phase_cancellation_index",
    "tracked_valid_track_fraction",
    "tracked_temporal_valid_track_fraction",
    "tracked_reliable_track_fraction",
    "tracked_normal_dominant_track_fraction",
    "robust_global_normal_rms_px_s",
    "robust_global_tangent_rms_px_s",
    "normal_energy_fraction",
    "source_piglet_count",
    "source_contact_piglet_count",
    "tracked_valid_track_count",
)

# Offline phase contrasts used primarily by the milk-ejection head.  These
# signals describe the transition from organized massage into the lower-motion
# milk-ejection phase, while retaining sow/body motion as a potential
# confounder.  Post-event contrasts are signed and deliberately unconstrained:
# they may represent either continued quiet sucking or a renewed arching peak.
# These features use past and future context and are not suitable for real-time
# inference.
PHASE_SIGNAL_CANDIDATES = (
    "tracked_periodic_intensity_p75_px_s",
    "tracked_periodicity_quality_p75",
    "tracked_normal_rms_p75_px_s",
    "tracked_tangent_rms_p75_px_s",
    "robust_global_normal_rms_px_s",
    "robust_global_tangent_rms_px_s",
    "normal_energy_fraction",
    "source_contact_piglet_count",
    "tracked_valid_track_fraction",
    "sow_rigid_speed_px_s",
    "contact_common_speed_px_s",
    "udder_contact_fraction",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _runtime_versions() -> dict[str, str]:
    """Return the training/prediction runtime versions needed for reproducibility."""
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }


def _json_safe(value: Any) -> Any:
    """Recursively convert numpy values and non-finite floats for strict JSON."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
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


def discover_feature_files(source: Path | str) -> list[Path]:
    """Return deterministic aggregate-feature CSV inputs from a file or directory."""
    path = Path(source).expanduser().resolve()
    if path.is_file():
        if not path.name.casefold().endswith("_features.csv") or ".__partial__" in path.name:
            raise ValueError(f"Feature input must be a completed *_features.csv file: {path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Feature input does not exist: {path}")
    files = sorted(
        (
            item.resolve()
            for item in path.rglob(FEATURE_GLOB)
            if item.is_file() and ".__partial__" not in item.name
        ),
        key=lambda item: str(item).casefold(),
    )
    if not files:
        raise FileNotFoundError(f"No {FEATURE_GLOB} files found under {path}")
    return files


def _source_schema(path: Path) -> list[str]:
    return list(pd.read_csv(path, nrows=0).columns)


def _is_excluded_raw_column(name: str) -> bool:
    if name in {
        "video",
        "status",
        "sample_index",
        "prev_frame_index",
        "frame_index",
        "roi_segment_id",
        "timestamp_sec",
        "interval_start_sec",
        "interval_end_sec",
        "dt_sec",
    }:
        return True
    if name == "fps" or name.endswith("_fps"):
        return True
    if name in {
        "index",
        "row_index",
        "pair_index",
        "prev_sample_index",
        "next_sample_index",
        "next_frame_index",
    }:
        return True
    if name.endswith("_status"):
        return True
    # Absolute temporal-window coordinates leak position and are bookkeeping,
    # unlike frequencies, durations and cycle counts.
    if re.search(r"(?:^|_)(?:timestamp|start|end)_sec$", name):
        return True
    return False


def _infer_raw_signal_columns(path: Path) -> tuple[list[str], list[str]]:
    """Infer all numerical signal columns from the first unified source schema."""
    schema = _source_schema(path)
    if "timestamp_sec" not in schema or "video" not in schema:
        raise ValueError(f"{path} is not an aggregate feature CSV (missing video/timestamp_sec)")
    candidates = [column for column in schema if not _is_excluded_raw_column(column)]
    sample = pd.read_csv(path, usecols=candidates, nrows=4096, low_memory=False)
    selected: list[str] = []
    for column in candidates:
        series = sample[column]
        if pd.api.types.is_numeric_dtype(series.dtype):
            selected.append(column)
            continue
        nonempty = series.notna() & series.astype(str).str.strip().ne("")
        converted = pd.to_numeric(series, errors="coerce")
        if not bool(nonempty.any()) or bool(converted[nonempty].notna().all()):
            selected.append(column)
    if not selected:
        raise ValueError(f"No numerical signal columns inferred from {path}")
    return schema, selected


def _video_stem_from_path(path: Path) -> str:
    name = path.name
    suffixes = (
        "__mp4_tracked_normal_flow_features.csv",
        "_tracked_normal_flow_features.csv",
    )
    for suffix in suffixes:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _feature_stems(files: Sequence[Path], *, context: str) -> list[str]:
    """Resolve and validate the one-video-per-file identity contract."""
    stems = [_video_stem_from_path(path) for path in files]
    counts: dict[str, int] = {}
    for stem in stems:
        counts[stem] = counts.get(stem, 0) + 1
    duplicates = sorted(stem for stem, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"{context} contains duplicate feature video stems: {duplicates}")
    return stems


def _validate_training_input_contract(
    files: Sequence[Path], annotations: Mapping[str, Any]
) -> list[str]:
    stems = _feature_stems(files, context="Training input")
    feature_set = set(stems)
    annotation_set = {str(key) for key in annotations}
    if feature_set != annotation_set:
        missing_features = sorted(annotation_set - feature_set)
        missing_annotations = sorted(feature_set - annotation_set)
        raise ValueError(
            "Training feature/annotation video sets differ: "
            f"annotations_without_features={missing_features}, "
            f"features_without_annotations={missing_annotations}"
        )
    return stems


def _select_training_feature_files(
    files: Sequence[Path],
    annotations: Mapping[str, Any],
    *,
    exclude_unannotated: bool = False,
) -> tuple[list[Path], list[str]]:
    """Select explicitly annotated videos while preserving strict missing-data checks."""

    stems = _feature_stems(files, context="Training input")
    annotation_set = {str(key) for key in annotations}
    feature_set = set(stems)
    missing_features = sorted(annotation_set - feature_set)
    unannotated_features = sorted(feature_set - annotation_set)
    if missing_features or (unannotated_features and not exclude_unannotated):
        raise ValueError(
            "Training feature/annotation video sets differ: "
            f"annotations_without_features={missing_features}, "
            f"features_without_annotations={unannotated_features}"
        )
    selected = [path for path, stem in zip(files, stems) if stem in annotation_set]
    _validate_training_input_contract(selected, annotations)
    return selected, unannotated_features


def aggregate_feature_csv(
    path: Path | str,
    raw_signal_columns: Sequence[str] | None = None,
    *,
    bin_sec: float = 1.0,
    expected_schema: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Aggregate one frame-pair feature CSV into fixed-width numerical bins."""
    csv_path = Path(path).expanduser().resolve()
    if not math.isfinite(bin_sec) or bin_sec <= 0:
        raise ValueError("bin_sec must be positive")
    schema = _source_schema(csv_path)
    if expected_schema is not None and schema != list(expected_schema):
        raise ValueError(f"Source schema differs from the trained schema: {csv_path}")
    if raw_signal_columns is None:
        _, inferred = _infer_raw_signal_columns(csv_path)
        raw_columns = inferred
    else:
        raw_columns = list(raw_signal_columns)
    required = {"video", "timestamp_sec", *raw_columns}
    missing = required - set(schema)
    if missing:
        raise ValueError(f"{csv_path} is missing trained input columns: {sorted(missing)}")

    read_columns = set(required)
    if "interval_end_sec" in schema:
        read_columns.add("interval_end_sec")
    frame = pd.read_csv(csv_path, usecols=list(read_columns), low_memory=False)
    for column in ("timestamp_sec", *raw_columns):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "interval_end_sec" in frame:
        frame["interval_end_sec"] = pd.to_numeric(frame["interval_end_sec"], errors="coerce")
    frame = frame[np.isfinite(frame["timestamp_sec"].to_numpy(dtype=float))].copy()
    if frame.empty:
        raise ValueError(f"No finite timestamps in {csv_path}")

    source_names = {
        Path(str(value)).stem
        for value in frame["video"].dropna().astype(str)
        if str(value).strip()
    }
    expected_video = _video_stem_from_path(csv_path)
    if len(source_names) != 1:
        raise ValueError(
            f"{csv_path} must contain exactly one non-empty internal video; found {sorted(source_names)}"
        )
    video = next(iter(source_names))
    if video != expected_video:
        raise ValueError(
            f"Internal video {video!r} does not match feature file stem {expected_video!r}: {csv_path}"
        )

    source_duration: float | None = None
    if "interval_end_sec" in frame:
        duration_values = frame["interval_end_sec"].to_numpy(dtype=float)
        finite_durations = duration_values[np.isfinite(duration_values)]
        if finite_durations.size:
            source_duration = float(np.max(finite_durations))
            if source_duration + 1e-6 < float(frame["timestamp_sec"].max()):
                raise ValueError(
                    f"interval_end_sec precedes timestamp_sec in {csv_path}: {source_duration:g}"
                )
    frame["bin_start_sec"] = (
        np.floor((frame["timestamp_sec"].to_numpy(dtype=float) + 1e-9) / bin_sec) * bin_sec
    )
    frame["_row_missing_fraction"] = frame[raw_columns].isna().mean(axis=1)

    grouped = frame.groupby("bin_start_sec", sort=True, observed=True)
    median = grouped[raw_columns].median().add_suffix("__median")
    result = median
    key_raw = [column for column in KEY_SIGNAL_CANDIDATES if column in raw_columns]
    if key_raw:
        result = result.join(grouped[key_raw].quantile(0.25).add_suffix("__q25"))
        result = result.join(grouped[key_raw].quantile(0.75).add_suffix("__q75"))
        result = result.join(grouped[key_raw].std(ddof=0).add_suffix("__std"))
    result["missing_fraction"] = grouped["_row_missing_fraction"].mean()
    result["source_row_count"] = grouped.size().astype(float)
    if "flow_valid" in frame:
        result["flow_valid_fraction"] = grouped["flow_valid"].mean()
    else:
        result["flow_valid_fraction"] = np.nan

    def median_column(raw_name: str) -> pd.Series:
        name = f"{raw_name}__median"
        if name not in result:
            return pd.Series(np.nan, index=result.index, dtype=float)
        return pd.to_numeric(result[name], errors="coerce")

    piglets = median_column("source_piglet_count")
    contacts = median_column("source_contact_piglet_count")
    denominator = piglets.where(piglets > 0)
    result["udder_contact_fraction"] = (contacts / denominator).clip(0.0, 1.0)
    valid_tracks = median_column("tracked_valid_track_count")
    normal_dominant = median_column("tracked_normal_dominant_track_fraction")
    contact_denominator = contacts.where(contacts > 0)
    result["normal_participation_proxy"] = (
        normal_dominant * valid_tracks / contact_denominator
    ).clip(0.0, 1.0)

    result = result.reset_index()
    if source_duration is None:
        source_duration = float(result["bin_start_sec"].max() + bin_sec)
    if not math.isfinite(source_duration) or source_duration < 0.0:
        raise ValueError(f"Invalid source duration for {csv_path}: {source_duration!r}")
    result = result[result["bin_start_sec"] < source_duration - 1e-9].copy()
    if result.empty:
        raise ValueError(f"No positive-duration bins remain in {csv_path}")
    result.insert(0, "video", video)
    result.insert(1, "source_path", str(csv_path))
    result.insert(
        3,
        "bin_end_sec",
        np.minimum(result["bin_start_sec"] + float(bin_sec), source_duration),
    )
    result.insert(4, "source_duration_sec", source_duration)
    result.replace([np.inf, -np.inf], np.nan, inplace=True)
    return result.sort_values("bin_start_sec").reset_index(drop=True)


def _robust_z(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    output = np.full(len(values), np.nan, dtype=float)
    if not finite.size:
        return pd.Series(output, index=series.index)
    center = float(np.median(finite))
    mad = float(np.median(np.abs(finite - center)))
    scale = 1.4826 * mad
    if not math.isfinite(scale) or scale <= 1e-12:
        q25, q75 = np.percentile(finite, [25, 75])
        scale = float((q75 - q25) / 1.349)
    if not math.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    valid = np.isfinite(values)
    output[valid] = np.clip((values[valid] - center) / scale, -30.0, 30.0)
    return pd.Series(output, index=series.index)


def _continuous_segment_ids(starts: Sequence[float], bin_sec: float) -> np.ndarray:
    """Assign segment ids, breaking whenever adjacent bins are not contiguous."""
    values = np.asarray(starts, dtype=float)
    if values.ndim != 1:
        raise ValueError("bin starts must be one-dimensional")
    if not math.isfinite(bin_sec) or bin_sec <= 0:
        raise ValueError("bin_sec must be positive")
    if not np.isfinite(values).all():
        raise ValueError("bin starts must be finite")
    if len(values) == 0:
        return np.empty(0, dtype=np.int64)
    breaks = np.r_[True, np.abs(np.diff(values) - bin_sec) > 1e-6]
    return np.cumsum(breaks, dtype=np.int64) - 1


def _nominal_bin_sec(frame: pd.DataFrame, explicit: float | None) -> float:
    if explicit is not None:
        value = float(explicit)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("bin_sec must be positive")
        return value
    widths = (
        pd.to_numeric(frame["bin_end_sec"], errors="coerce")
        - pd.to_numeric(frame["bin_start_sec"], errors="coerce")
    ).to_numpy(dtype=float)
    finite = widths[np.isfinite(widths) & (widths > 0)]
    if not finite.size:
        raise ValueError("Cannot infer bin_sec from bin_start_sec/bin_end_sec")
    # The final bin may be clipped to the true source duration, so the largest
    # observed width is the best available nominal width.
    return float(np.max(finite))


def engineer_video_features(
    frame: pd.DataFrame,
    key_columns: Sequence[str] | None = None,
    *,
    bin_sec: float | None = None,
) -> pd.DataFrame:
    """Add per-video robust-z and centered offline temporal-context features."""
    required = {"video", "bin_start_sec", "bin_end_sec"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Binned frame is missing metadata columns: {sorted(missing)}")
    if frame.empty:
        return frame.copy()
    requested_keys = list(key_columns or KEY_SIGNAL_CANDIDATES)
    pieces: list[pd.DataFrame] = []
    for _, source in frame.groupby("video", sort=False, observed=True):
        source = source.sort_values("bin_start_sec").copy()
        nominal_bin_sec = _nominal_bin_sec(source, bin_sec)
        segment_ids = pd.Series(
            _continuous_segment_ids(source["bin_start_sec"].to_numpy(dtype=float), nominal_bin_sec),
            index=source.index,
        )
        additions: dict[str, pd.Series] = {}
        numerical = [
            column
            for column in source.columns
            if column not in METADATA_COLUMNS
            and column not in LABEL_COLUMNS
            and pd.api.types.is_numeric_dtype(source[column].dtype)
        ]
        for column in numerical:
            additions[f"{column}__video_robust_z"] = _robust_z(source[column])

        resolved_keys: list[str] = []
        for column in requested_keys:
            if column in source.columns:
                resolved_keys.append(column)
            elif f"{column}__median" in source.columns:
                resolved_keys.append(f"{column}__median")
        for column in dict.fromkeys(resolved_keys):
            values = pd.to_numeric(source[column], errors="coerce")
            grouped = values.groupby(segment_ids, sort=False)
            additions[f"{column}__centered_med3"] = grouped.transform(
                lambda part: part.rolling(3, center=True, min_periods=1).median()
            )
            additions[f"{column}__centered_med7"] = grouped.transform(
                lambda part: part.rolling(7, center=True, min_periods=1).median()
            )
            additions[f"{column}__centered_std5"] = grouped.transform(
                lambda part: part.rolling(5, center=True, min_periods=1).std(ddof=0)
            )
            centered_delta = (grouped.shift(-1) - grouped.shift(1)) / 2.0
            additions[f"{column}__centered_delta"] = centered_delta

        resolved_phase: list[str] = []
        for column in PHASE_SIGNAL_CANDIDATES:
            if column in source.columns:
                resolved_phase.append(column)
            elif f"{column}__median" in source.columns:
                resolved_phase.append(f"{column}__median")
        for column in dict.fromkeys(resolved_phase):
            values = pd.to_numeric(source[column], errors="coerce")
            grouped = values.groupby(segment_ids, sort=False)
            local = grouped.transform(
                lambda part: part.rolling(7, center=True, min_periods=1).median()
            )
            for window in (15, 30):
                past = grouped.transform(
                    lambda part, width=window: part.shift(1).rolling(
                        width, min_periods=1
                    ).median()
                )
                future = grouped.transform(
                    lambda part, width=window: part.iloc[::-1]
                    .shift(1)
                    .rolling(width, min_periods=1)
                    .median()
                    .iloc[::-1]
                )
                additions[f"{column}__phase_pre_minus_local_{window}"] = past - local
                additions[f"{column}__phase_post_minus_local_{window}"] = future - local
        if additions:
            source = pd.concat([source, pd.DataFrame(additions, index=source.index)], axis=1)
        source.replace([np.inf, -np.inf], np.nan, inplace=True)
        pieces.append(source)
    return pd.concat(pieces, ignore_index=True)


def _contiguous_runs(labels: np.ndarray, starts: np.ndarray, bin_sec: float) -> list[tuple[int, int, int]]:
    """Return (value, start_index, stop_index) runs without crossing time gaps."""
    if len(labels) == 0:
        return []
    runs: list[tuple[int, int, int]] = []
    begin = 0
    for index in range(1, len(labels)):
        contiguous = abs(float(starts[index] - starts[index - 1]) - bin_sec) <= 1e-6
        if labels[index] != labels[index - 1] or not contiguous:
            runs.append((int(labels[begin]), begin, index))
            begin = index
    runs.append((int(labels[begin]), begin, len(labels)))
    return runs


def _event_balanced_sample_weights(
    dataset: pd.DataFrame,
    labels: np.ndarray,
    known: np.ndarray,
    selected: np.ndarray,
    positive_states: Sequence[int],
    *,
    bin_sec: float,
    alpha: float,
    boundary_boost: float,
    maximum_weight: float,
) -> np.ndarray:
    """Return per-bin weights that reduce domination by long labelled events.

    Within each positive state, a run of length ``L`` receives a base weight
    proportional to ``(median_event_length / L) ** alpha``.  The weights are
    clipped and then normalized to unit mean within that state, so class
    balancing still controls the total class contribution.  Event boundary
    bins may receive an additional bounded boost.
    """
    if len(dataset) != len(labels) or len(dataset) != len(known) or len(dataset) != len(selected):
        raise ValueError("dataset, labels, known and selected must have equal length")
    if not math.isfinite(bin_sec) or bin_sec <= 0:
        raise ValueError("bin_sec must be positive")
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("event weight alpha cannot be negative")
    if not math.isfinite(boundary_boost) or boundary_boost < 1:
        raise ValueError("event boundary boost must be at least 1")
    if not math.isfinite(maximum_weight) or maximum_weight < 1:
        raise ValueError("maximum event weight must be at least 1")

    labels_array = np.asarray(labels, dtype=int)
    known_array = np.asarray(known, dtype=bool)
    selected_array = np.asarray(selected, dtype=bool) & known_array
    starts = dataset["bin_start_sec"].to_numpy(dtype=float)
    videos = dataset["video"].astype(str).to_numpy()
    weights = np.ones(len(dataset), dtype=np.float64)

    for state in (int(value) for value in positive_states):
        runs: list[np.ndarray] = []
        for video in np.unique(videos[selected_array]):
            indices = np.flatnonzero(selected_array & (videos == video))
            current: list[int] = []
            previous: int | None = None
            for index in indices:
                contiguous = (
                    previous is not None
                    and abs(float(starts[index] - starts[previous]) - float(bin_sec)) <= 1e-6
                )
                if int(labels_array[index]) == state:
                    if current and not contiguous:
                        runs.append(np.asarray(current, dtype=int))
                        current = []
                    current.append(int(index))
                elif current:
                    runs.append(np.asarray(current, dtype=int))
                    current = []
                previous = int(index)
            if current:
                runs.append(np.asarray(current, dtype=int))
        if not runs:
            continue

        reference_length = float(np.median([len(run) for run in runs]))
        for run in runs:
            base = (reference_length / len(run)) ** float(alpha) if alpha > 0 else 1.0
            base = float(np.clip(base, 1.0 / maximum_weight, maximum_weight))
            weights[run] = base
            if boundary_boost > 1:
                weights[run[0]] *= boundary_boost
                if run[-1] != run[0]:
                    weights[run[-1]] *= boundary_boost
        state_mask = selected_array & (labels_array == state)
        state_mean = float(np.mean(weights[state_mask]))
        if state_mean > 0:
            weights[state_mask] /= state_mean

    weights[~selected_array] = 0.0
    return weights


def _remove_short_runs(
    labels: np.ndarray,
    starts: np.ndarray,
    *,
    target: int,
    minimum_duration_sec: float,
    bin_sec: float,
    replacement: int = 0,
    bin_ends: np.ndarray | None = None,
) -> np.ndarray:
    result = labels.copy()
    ends = None if bin_ends is None else np.asarray(bin_ends, dtype=float)
    if ends is not None and len(ends) != len(starts):
        raise ValueError("bin_ends must match starts")
    for value, begin, stop in _contiguous_runs(result, starts, bin_sec):
        end = float(ends[stop - 1]) if ends is not None else float(starts[stop - 1] + bin_sec)
        duration = end - float(starts[begin])
        if value == target and duration + 1e-9 < minimum_duration_sec:
            result[begin:stop] = replacement
    return result


def _merge_short_empty_gaps(
    labels: np.ndarray,
    starts: np.ndarray,
    *,
    target: int,
    maximum_gap_sec: float,
    bin_sec: float,
    empty: int = 0,
) -> np.ndarray:
    result = labels.copy()
    if maximum_gap_sec <= 0:
        return result
    changed = True
    while changed:
        changed = False
        runs = _contiguous_runs(result, starts, bin_sec)
        for index in range(1, len(runs) - 1):
            value, begin, stop = runs[index]
            left, right = runs[index - 1], runs[index + 1]
            if (
                value == empty
                and left[0] == target
                and right[0] == target
                and (stop - begin) * bin_sec <= maximum_gap_sec + 1e-9
            ):
                result[begin:stop] = target
                changed = True
                break
    return result


def labels_to_intervals(
    labels: Sequence[int | bool],
    bin_starts: Sequence[float],
    *,
    target_value: int | bool = 1,
    bin_sec: float = 1.0,
    level: str = "event",
    scores: Sequence[float] | None = None,
    merge_gap_sec: float = 0.0,
    minimum_duration_sec: float = 0.0,
    source_duration_sec: float | None = None,
) -> list[dict[str, Any]]:
    """Convert fixed-bin labels to half-open intervals with optional merging."""
    label_array = np.asarray(labels)
    starts = np.asarray(bin_starts, dtype=float)
    if label_array.ndim != 1 or starts.ndim != 1 or len(label_array) != len(starts):
        raise ValueError("labels and bin_starts must be one-dimensional and equally sized")
    score_array = None if scores is None else np.asarray(scores, dtype=float)
    if score_array is not None and len(score_array) != len(labels):
        raise ValueError("scores must match labels")
    duration_limit = None if source_duration_sec is None else float(source_duration_sec)
    if duration_limit is not None and (not math.isfinite(duration_limit) or duration_limit < 0):
        raise ValueError("source_duration_sec must be finite and non-negative")
    selected = (label_array == target_value).astype(np.int8)
    raw: list[tuple[int, int]] = [
        (begin, stop)
        for value, begin, stop in _contiguous_runs(selected, starts, bin_sec)
        if value == 1
    ]
    merged: list[tuple[int, int]] = []
    for begin, stop in raw:
        if merged:
            previous_begin, previous_stop = merged[-1]
            previous_end = float(starts[previous_stop - 1] + bin_sec)
            gap = float(starts[begin] - previous_end)
            # A run may have been split because the source timeline itself has
            # missing bins.  Only bridge an observed, contiguous background
            # gap; never merge across absent time even when merge_gap_sec is
            # numerically large enough to span it.
            bridge = starts[previous_stop - 1 : begin + 1]
            timeline_contiguous = bool(
                len(bridge) <= 1
                or np.all(np.abs(np.diff(bridge) - bin_sec) <= 1e-6)
            )
            if timeline_contiguous and -1e-9 <= gap <= merge_gap_sec + 1e-9:
                merged[-1] = (previous_begin, stop)
                continue
        merged.append((begin, stop))

    intervals: list[dict[str, Any]] = []
    for begin, stop in merged:
        start = float(starts[begin])
        end = float(starts[stop - 1] + bin_sec)
        if duration_limit is not None:
            end = min(end, duration_limit)
        if end - start + 1e-9 < minimum_duration_sec:
            continue
        record: dict[str, Any] = {
            "level": level,
            "start_sec": start,
            "end_sec": end,
            "duration_sec": end - start,
        }
        if score_array is not None:
            local = score_array[begin:stop]
            finite = local[np.isfinite(local)]
            record["mean_probability"] = float(np.mean(finite)) if finite.size else None
            record["max_probability"] = float(np.max(finite)) if finite.size else None
        intervals.append(record)
    return intervals


def _frame_source_duration(frame: pd.DataFrame) -> float | None:
    if "source_duration_sec" not in frame:
        return None
    values = pd.to_numeric(frame["source_duration_sec"], errors="coerce").to_numpy(dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("source_duration_sec must be finite for every bin")
    duration = float(values[0])
    if not np.all(np.abs(values - duration) <= 1e-6):
        raise ValueError("A video contains inconsistent source_duration_sec values")
    return duration


def _postprocess_video_probabilities(
    frame: pd.DataFrame,
    *,
    milk_threshold: float,
    bin_sec: float,
    probability_rolling_bins: int,
    merge_gap_sec: float,
    milk_minimum_duration_sec: float,
    arching_minimum_duration_sec: float,
    milk_keep_highest: bool = False,
    milk_refractory_sec: float = 1200.0,
    milk_probability_rolling_bins: int | None = None,
    arching_probability_rolling_bins: int | None = None,
    milk_merge_gap_sec: float | None = None,
    arching_merge_gap_sec: float | None = None,
    arching_partial_probability_weight: float = 1.0,
    arching_collective_probability_weight: float = 1.0,
) -> pd.DataFrame:
    source = frame.sort_values("bin_start_sec").copy()
    required = {
        "milk_probability",
        "arching_probability_none",
        "arching_probability_partial",
        "arching_probability_collective",
    }
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"Probability frame is missing columns: {sorted(missing)}")
    fallback_window = max(1, int(probability_rolling_bins))
    milk_window = max(
        1,
        int(
            fallback_window
            if milk_probability_rolling_bins is None
            else milk_probability_rolling_bins
        ),
    )
    arching_window = max(
        1,
        int(
            fallback_window
            if arching_probability_rolling_bins is None
            else arching_probability_rolling_bins
        ),
    )
    resolved_milk_gap = (
        float(merge_gap_sec) if milk_merge_gap_sec is None else float(milk_merge_gap_sec)
    )
    resolved_arching_gap = (
        float(merge_gap_sec) if arching_merge_gap_sec is None else float(arching_merge_gap_sec)
    )
    if arching_partial_probability_weight <= 0 or arching_collective_probability_weight <= 0:
        raise ValueError("Arching probability weights must be positive")
    probability_columns = sorted(required)
    starts = source["bin_start_sec"].to_numpy(dtype=float)
    segment_ids = pd.Series(_continuous_segment_ids(starts, bin_sec), index=source.index)
    for column in probability_columns:
        values = pd.to_numeric(source[column], errors="coerce")
        window = milk_window if column == "milk_probability" else arching_window
        source[f"{column}_smoothed"] = values.groupby(segment_ids, sort=False).transform(
            lambda part: part.rolling(window, center=True, min_periods=1).mean()
        )
    bin_ends = (
        source["bin_end_sec"].to_numpy(dtype=float)
        if "bin_end_sec" in source
        else starts + float(bin_sec)
    )
    source_duration = _frame_source_duration(source)
    milk_scores = source["milk_probability_smoothed"].to_numpy(dtype=float)
    milk = (milk_scores >= milk_threshold).astype(np.int8)
    milk = _merge_short_empty_gaps(
        milk,
        starts,
        target=1,
        maximum_gap_sec=resolved_milk_gap,
        bin_sec=bin_sec,
    )
    milk = _remove_short_runs(
        milk,
        starts,
        target=1,
        minimum_duration_sec=milk_minimum_duration_sec,
        bin_sec=bin_sec,
        bin_ends=bin_ends,
    )
    if milk_refractory_sec > 0:
        candidates = labels_to_intervals(
            milk,
            starts,
            bin_sec=bin_sec,
            scores=milk_scores,
            minimum_duration_sec=milk_minimum_duration_sec,
            level="milk",
            source_duration_sec=source_duration,
        )
        if len(candidates) > 1:
            ranked = sorted(
                candidates,
                key=lambda item: (
                    float(item.get("mean_probability") or -math.inf),
                    float(item.get("max_probability") or -math.inf),
                    float(item["duration_sec"]),
                    -float(item["start_sec"]),
                ),
                reverse=True,
            )
            kept: list[dict[str, Any]] = []
            for item in ranked:
                midpoint = 0.5 * (float(item["start_sec"]) + float(item["end_sec"]))
                if any(
                    abs(midpoint - 0.5 * (float(other["start_sec"]) + float(other["end_sec"])))
                    < milk_refractory_sec
                    for other in kept
                ):
                    continue
                kept.append(item)
            milk[:] = 0
            for item in kept:
                milk[
                    (starts >= float(item["start_sec"]) - 1e-9)
                    & (starts < float(item["end_sec"]) - 1e-9)
                ] = 1

    arch_scores = np.column_stack(
        [
            source[f"arching_probability_{ARCHING_CLASSES[index]}_smoothed"].to_numpy(dtype=float)
            for index in range(3)
        ]
    )
    arch_scores *= np.asarray(
        [1.0, float(arching_partial_probability_weight), float(arching_collective_probability_weight)],
        dtype=float,
    )
    arching = np.nanargmax(np.where(np.isfinite(arch_scores), arch_scores, -math.inf), axis=1).astype(np.int8)
    invalid_rows = ~np.isfinite(arch_scores).any(axis=1)
    arching[invalid_rows] = 0
    for state in (1, 2):
        arching = _remove_short_runs(
            arching,
            starts,
            target=state,
            minimum_duration_sec=arching_minimum_duration_sec,
            bin_sec=bin_sec,
            bin_ends=bin_ends,
        )
    for state in (1, 2):
        arching = _merge_short_empty_gaps(
            arching,
            starts,
            target=state,
            maximum_gap_sec=resolved_arching_gap,
            bin_sec=bin_sec,
        )
    for state in (1, 2):
        arching = _remove_short_runs(
            arching,
            starts,
            target=state,
            minimum_duration_sec=arching_minimum_duration_sec,
            bin_sec=bin_sec,
            bin_ends=bin_ends,
        )
    source["milk_prediction"] = milk
    source["arching_prediction"] = arching
    source["arching_prediction_name"] = [ARCHING_CLASSES[int(value)] for value in arching]
    return source


def _interval_document_from_processed(
    processed: pd.DataFrame,
    *,
    bin_sec: float,
    postprocess: Mapping[str, Any],
) -> dict[str, Any]:
    videos: list[dict[str, Any]] = []
    for video, source in processed.groupby("video", sort=True, observed=True):
        source = source.sort_values("bin_start_sec")
        starts = source["bin_start_sec"].to_numpy(dtype=float)
        source_duration = _frame_source_duration(source)
        milk = labels_to_intervals(
            source["milk_prediction"].to_numpy(dtype=int),
            starts,
            target_value=1,
            bin_sec=bin_sec,
            level="milk",
            scores=source["milk_probability_smoothed"].to_numpy(dtype=float),
            source_duration_sec=source_duration,
        )
        partial = labels_to_intervals(
            source["arching_prediction"].to_numpy(dtype=int),
            starts,
            target_value=1,
            bin_sec=bin_sec,
            level="partial",
            scores=source["arching_probability_partial_smoothed"].to_numpy(dtype=float),
            source_duration_sec=source_duration,
        )
        collective = labels_to_intervals(
            source["arching_prediction"].to_numpy(dtype=int),
            starts,
            target_value=2,
            bin_sec=bin_sec,
            level="collective",
            scores=source["arching_probability_collective_smoothed"].to_numpy(dtype=float),
            source_duration_sec=source_duration,
        )
        videos.append(
            {
                "video": str(video),
                "source_duration_sec": source_duration,
                "milk_intervals": milk,
                "partial_arching_intervals": partial,
                "collective_arching_intervals": collective,
                "intervals": sorted(milk + partial + collective, key=lambda item: (item["start_sec"], item["level"])),
            }
        )
    return {
        "model_kind": MODEL_NAME,
        "time_reference": "seconds relative to each clipped source video; half-open intervals [start, end)",
        "bin_sec": float(bin_sec),
        "postprocess": dict(postprocess),
        "videos": videos,
    }


def decode_intervals(
    probabilities: pd.DataFrame,
    *,
    milk_threshold: float,
    bin_sec: float = 1.0,
    probability_rolling_bins: int = 3,
    merge_gap_sec: float = 2.0,
    milk_minimum_duration_sec: float = 8.0,
    arching_minimum_duration_sec: float = 4.0,
    milk_keep_highest: bool = False,
    milk_refractory_sec: float = 1200.0,
    milk_probability_rolling_bins: int | None = None,
    arching_probability_rolling_bins: int | None = None,
    milk_merge_gap_sec: float | None = None,
    arching_merge_gap_sec: float | None = None,
    arching_partial_probability_weight: float = 1.0,
    arching_collective_probability_weight: float = 1.0,
) -> dict[str, Any]:
    """Smooth probabilities and decode mutually consistent offline intervals."""
    pieces = [
        _postprocess_video_probabilities(
            source,
            milk_threshold=milk_threshold,
            bin_sec=bin_sec,
            probability_rolling_bins=probability_rolling_bins,
            merge_gap_sec=merge_gap_sec,
            milk_minimum_duration_sec=milk_minimum_duration_sec,
            arching_minimum_duration_sec=arching_minimum_duration_sec,
            milk_keep_highest=milk_keep_highest,
            milk_refractory_sec=milk_refractory_sec,
            milk_probability_rolling_bins=milk_probability_rolling_bins,
            arching_probability_rolling_bins=arching_probability_rolling_bins,
            milk_merge_gap_sec=milk_merge_gap_sec,
            arching_merge_gap_sec=arching_merge_gap_sec,
            arching_partial_probability_weight=arching_partial_probability_weight,
            arching_collective_probability_weight=arching_collective_probability_weight,
        )
        for _, source in probabilities.groupby("video", sort=False, observed=True)
    ]
    processed = pd.concat(pieces, ignore_index=True) if pieces else probabilities.copy()
    postprocess = {
        "probability_rolling_bins": int(probability_rolling_bins),
        "milk_probability_rolling_bins": int(
            probability_rolling_bins
            if milk_probability_rolling_bins is None
            else milk_probability_rolling_bins
        ),
        "arching_probability_rolling_bins": int(
            probability_rolling_bins
            if arching_probability_rolling_bins is None
            else arching_probability_rolling_bins
        ),
        "rolling_centered_offline": True,
        "merge_gap_sec": float(merge_gap_sec),
        "milk_merge_gap_sec": float(
            merge_gap_sec if milk_merge_gap_sec is None else milk_merge_gap_sec
        ),
        "arching_merge_gap_sec": float(
            merge_gap_sec if arching_merge_gap_sec is None else arching_merge_gap_sec
        ),
        "milk_minimum_duration_sec": float(milk_minimum_duration_sec),
        "arching_minimum_duration_sec": float(arching_minimum_duration_sec),
        "milk_keep_highest": bool(milk_keep_highest),
        "milk_refractory_sec": float(milk_refractory_sec),
        "milk_threshold": float(milk_threshold),
        "arching_decision": "argmax_none_partial_collective",
        "arching_partial_probability_weight": float(arching_partial_probability_weight),
        "arching_collective_probability_weight": float(arching_collective_probability_weight),
    }
    return _interval_document_from_processed(processed, bin_sec=bin_sec, postprocess=postprocess)


def _processed_probabilities(probabilities: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    pieces = [
        _postprocess_video_probabilities(source, **config)
        for _, source in probabilities.groupby("video", sort=False, observed=True)
    ]
    return pd.concat(pieces, ignore_index=True) if pieces else probabilities.copy()


def _aligned_probabilities(
    model: HistGradientBoostingClassifier,
    features: pd.DataFrame,
    classes: Sequence[int],
) -> np.ndarray:
    raw = model.predict_proba(features)
    aligned = np.zeros((len(features), len(classes)), dtype=float)
    position = {int(value): index for index, value in enumerate(classes)}
    for source_index, class_value in enumerate(model.classes_):
        aligned[:, position[int(class_value)]] = raw[:, source_index]
    return aligned


def _new_model(args: argparse.Namespace, random_state: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=float(args.learning_rate),
        max_iter=int(args.max_iter),
        max_leaf_nodes=int(args.max_leaf_nodes),
        min_samples_leaf=int(args.min_samples_leaf),
        l2_regularization=float(args.l2_regularization),
        class_weight="balanced",
        early_stopping=True,
        random_state=int(random_state),
    )


def _fit_head(
    features: pd.DataFrame,
    labels: np.ndarray,
    expected_classes: Sequence[int],
    args: argparse.Namespace,
    random_state: int,
    sample_weight: np.ndarray | None = None,
) -> HistGradientBoostingClassifier:
    observed = sorted(int(value) for value in np.unique(labels))
    missing = set(expected_classes) - set(observed)
    if missing:
        raise ValueError(f"Training fold lacks required classes {sorted(missing)}; observed={observed}")
    model = _new_model(args, random_state)
    model.fit(features, labels, sample_weight=sample_weight)
    return model


def _best_deployment_milk_threshold(
    oof: pd.DataFrame,
    args: argparse.Namespace,
    annotations: Mapping[str, Any] | None = None,
    source_durations_sec: Mapping[str, float] | None = None,
) -> tuple[float, list[dict[str, float]]]:
    """Tune milk threshold on pooled OOF predictions.

    Training uses original continuous events when annotations and source
    durations are supplied.  The bin-level fallback is retained for small unit
    tests and callers without continuous annotations.
    """
    use_event_objective = annotations is not None and source_durations_sec is not None
    truth: dict[str, list[dict[str, float]]] = {}
    if use_event_objective:
        for video, annotation in annotations.items():
            duration = float(source_durations_sec[str(video)])
            truth[str(video)] = [
                {
                    "start_sec": max(0.0, float(interval.start_sec)),
                    "end_sec": min(duration, float(interval.end_sec)),
                }
                for interval in annotation.milk_intervals
                if min(duration, float(interval.end_sec)) > max(0.0, float(interval.start_sec))
            ]
    search: list[dict[str, float]] = []
    for threshold in np.linspace(0.10, 0.90, 81):
        candidate = round(float(threshold), 2)
        processed = _processed_probabilities(oof, _postprocess_config(args, candidate))
        known = processed["milk_known"].to_numpy(dtype=bool)
        labels = processed.loc[known, "milk"].to_numpy(dtype=int)
        predictions = processed.loc[known, "milk_prediction"].to_numpy(dtype=int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            predictions,
            labels=[0, 1],
            zero_division=0,
        )
        row = {
            "threshold": candidate,
            "decoded_f1": float(f1[1]),
            "decoded_precision": float(precision[1]),
            "decoded_recall": float(recall[1]),
        }
        if use_event_objective:
            document = _interval_document_from_processed(
                processed,
                bin_sec=float(args.bin_sec),
                postprocess=_postprocess_config(args, candidate),
            )
            prediction = {
                str(record["video"]): list(record["milk_intervals"])
                for record in document["videos"]
            }
            event = _event_metric_for_class(truth, prediction, match_iou=0.5)
            row.update(
                event_f1=float(event["f1"]),
                event_precision=float(event["precision"]),
                event_recall=float(event["recall"]),
            )
        search.append(row)
    objective = "event_f1" if use_event_objective else "decoded_f1"
    best = max(
        search,
        key=lambda item: (item[objective], -abs(item["threshold"] - 0.5)),
    )
    return float(best["threshold"]), search


def _binary_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(np.int8)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=[0, 1],
        zero_division=0,
    )
    ap = average_precision_score(labels, probabilities) if len(np.unique(labels)) > 1 else math.nan
    class_probabilities = (1.0 - probabilities, probabilities)
    return {
        "threshold": float(threshold),
        "average_precision_positive": float(ap),
        "f1_positive": float(f1[1]),
        "classes": {
            str(index): {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
                "average_precision": (
                    float(average_precision_score((labels == index).astype(np.int8), class_probabilities[index]))
                    if len(np.unique(labels == index)) > 1
                    else math.nan
                ),
            }
            for index in range(2)
        },
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }


def _multiclass_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    predictions = np.argmax(probabilities, axis=1).astype(np.int8)
    classes = [0, 1, 2]
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=classes,
        zero_division=0,
    )
    class_rows: dict[str, Any] = {}
    for index, name in ARCHING_CLASSES.items():
        binary = (labels == index).astype(np.int8)
        ap = average_precision_score(binary, probabilities[:, index]) if len(np.unique(binary)) > 1 else math.nan
        class_rows[name] = {
            "class_id": index,
            "average_precision": float(ap),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
    return {
        "macro_f1": float(np.mean(f1)),
        "weighted_f1": float(np.average(f1, weights=support)) if int(np.sum(support)) else math.nan,
        "classes": class_rows,
        "confusion_matrix": confusion_matrix(labels, predictions, labels=classes).tolist(),
    }


def _binary_decoded_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=[0, 1],
        zero_division=0,
    )
    return {
        "f1_positive": float(f1[1]),
        "classes": {
            str(index): {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index in range(2)
        },
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }


def _multiclass_decoded_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    classes = [0, 1, 2]
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=classes,
        zero_division=0,
    )
    return {
        "macro_f1": float(np.mean(f1)),
        "weighted_f1": float(np.average(f1, weights=support)) if int(np.sum(support)) else math.nan,
        "classes": {
            ARCHING_CLASSES[index]: {
                "class_id": index,
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index in classes
        },
        "confusion_matrix": confusion_matrix(labels, predictions, labels=classes).tolist(),
    }


def _processed_bin_metrics(processed: pd.DataFrame) -> dict[str, Any]:
    milk_known = processed["milk_known"].to_numpy(dtype=bool)
    arch_known = processed["arching_known"].to_numpy(dtype=bool)
    return {
        "milk": _binary_decoded_metrics(
            processed.loc[milk_known, "milk"].to_numpy(dtype=int),
            processed.loc[milk_known, "milk_prediction"].to_numpy(dtype=int),
        ),
        "arching_state": _multiclass_decoded_metrics(
            processed.loc[arch_known, "arching_state"].to_numpy(dtype=int),
            processed.loc[arch_known, "arching_prediction"].to_numpy(dtype=int),
        ),
    }


def _interval_iou(first: Mapping[str, Any], second: Mapping[str, Any]) -> float:
    intersection = max(
        0.0,
        min(float(first["end_sec"]), float(second["end_sec"]))
        - max(float(first["start_sec"]), float(second["start_sec"])),
    )
    union = (
        float(first["end_sec"])
        - float(first["start_sec"])
        + float(second["end_sec"])
        - float(second["start_sec"])
        - intersection
    )
    return intersection / union if union > 0 else 0.0


def _event_metric_for_class(
    truth: dict[str, list[dict[str, Any]]],
    prediction: dict[str, list[dict[str, Any]]],
    *,
    match_iou: float = 0.5,
) -> dict[str, Any]:
    true_positive = false_positive = false_negative = 0
    matched_ious: list[float] = []
    for video in sorted(set(truth) | set(prediction)):
        expected = truth.get(video, [])
        actual = prediction.get(video, [])
        candidates = sorted(
            (
                (_interval_iou(expected[t], actual[p]), t, p)
                for t in range(len(expected))
                for p in range(len(actual))
            ),
            reverse=True,
        )
        used_true: set[int] = set()
        used_pred: set[int] = set()
        for iou, true_index, pred_index in candidates:
            if iou + 1e-12 < match_iou or true_index in used_true or pred_index in used_pred:
                continue
            used_true.add(true_index)
            used_pred.add(pred_index)
            matched_ious.append(float(iou))
        true_positive += len(used_true)
        false_negative += len(expected) - len(used_true)
        false_positive += len(actual) - len(used_pred)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "match_iou_threshold": float(match_iou),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_matched_iou": float(np.mean(matched_ious)) if matched_ious else None,
    }


def _oof_event_metrics(processed: pd.DataFrame, *, bin_sec: float) -> dict[str, Any]:
    truth: dict[str, dict[str, list[dict[str, Any]]]] = {
        "milk": {},
        "partial": {},
        "collective": {},
    }
    prediction: dict[str, dict[str, list[dict[str, Any]]]] = {
        "milk": {},
        "partial": {},
        "collective": {},
    }
    temporal_iou: dict[str, list[float]] = {name: [] for name in truth}
    for video, source in processed.groupby("video", sort=True, observed=True):
        source = source.sort_values("bin_start_sec")
        starts = source["bin_start_sec"].to_numpy(dtype=float)
        source_duration = _frame_source_duration(source)
        configurations = (
            ("milk", source["milk"].to_numpy(dtype=int), source["milk_prediction"].to_numpy(dtype=int), 1),
            (
                "partial",
                source["arching_state"].to_numpy(dtype=int),
                source["arching_prediction"].to_numpy(dtype=int),
                1,
            ),
            (
                "collective",
                source["arching_state"].to_numpy(dtype=int),
                source["arching_prediction"].to_numpy(dtype=int),
                2,
            ),
        )
        for name, true_values, predicted_values, target in configurations:
            known = (
                source["milk_known"].to_numpy(dtype=bool)
                if name == "milk"
                else source["arching_known"].to_numpy(dtype=bool)
            )
            if not bool(np.any(known)):
                continue
            true_mask = (true_values == target) & known
            pred_mask = (predicted_values == target) & known
            truth[name][str(video)] = labels_to_intervals(
                true_mask.astype(np.int8),
                starts,
                bin_sec=bin_sec,
                level=name,
                source_duration_sec=source_duration,
            )
            prediction[name][str(video)] = labels_to_intervals(
                pred_mask.astype(np.int8),
                starts,
                bin_sec=bin_sec,
                level=name,
                source_duration_sec=source_duration,
            )
            intersection = int(np.count_nonzero(true_mask & pred_mask))
            union = int(np.count_nonzero(true_mask | pred_mask))
            temporal_iou[name].append(intersection / union if union else 1.0)
    return {
        name: {
            **_event_metric_for_class(truth[name], prediction[name]),
            "mean_video_bin_iou": float(np.mean(temporal_iou[name])) if temporal_iou[name] else None,
        }
        for name in truth
    }


def _flatten_intervals(document: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in document.get("videos", []):
        for interval in record.get("intervals", []):
            rows.append({"video": record["video"], **interval})
    return pd.DataFrame(rows)


def _input_sources(annotation_path: Path | None, feature_files: Sequence[Path]) -> dict[str, Any]:
    return {
        "feature_source": "aggregate_features",
        "feature_sources": [
            {
                "name": "aggregate_features",
                "kind": "tracked_normal_flow_aggregate_csv",
                "file_count": len(feature_files),
                "contains": "segmentation + tracking aggregates + optical flow + periodicity",
            }
        ],
        "annotation_source": None if annotation_path is None else str(annotation_path.resolve()),
        "track_long_csv_used": False,
        "summary_json_used": False,
        "extensibility_note": "additional feature-source blocks may be joined by video/bin in later schema versions",
    }


def _assemble_training_dataset(
    files: Sequence[Path],
    annotations: Mapping[str, Any],
    *,
    raw_signal_columns: Sequence[str],
    source_schema: Sequence[str],
    bin_sec: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    pieces: list[pd.DataFrame] = []
    durations: dict[str, float] = {}
    for index, path in enumerate(files, start=1):
        print(f"[features {index:03d}/{len(files):03d}] {path.name}", flush=True)
        binned = aggregate_feature_csv(
            path,
            raw_signal_columns,
            bin_sec=bin_sec,
            expected_schema=source_schema,
        )
        video = str(binned["video"].iloc[0])
        if video not in annotations:
            raise ValueError(f"No confirmed annotation for feature video {video}")
        duration_value = _frame_source_duration(binned)
        if duration_value is None:
            raise ValueError(f"Missing source_duration_sec after aggregation for {video}")
        duration = float(duration_value)
        binned = engineer_video_features(
            binned,
            KEY_SIGNAL_CANDIDATES,
            bin_sec=float(bin_sec),
        )
        durations[video] = duration
        raster = rasterize_annotation(
            annotations[video],
            binned["bin_start_sec"].to_numpy(dtype=float),
            float(bin_sec),
            duration_sec=duration,
        )
        for key in ("milk", "milk_known", "arching_state", "arching_known"):
            if key not in raster or len(raster[key]) != len(binned):
                raise ValueError(f"Invalid rasterized annotation key {key!r} for {video}")
            binned[key] = np.asarray(raster[key])
        pieces.append(binned)
    dataset = pd.concat(pieces, ignore_index=True)
    duplicated = dataset.duplicated(["video", "bin_start_sec"])
    if bool(duplicated.any()):
        examples = dataset.loc[duplicated, ["video", "bin_start_sec"]].head().to_dict("records")
        raise ValueError(f"Duplicate video/bin rows after aggregation: {examples}")
    return _frozen_sample_order(dataset), durations


def _model_feature_columns(dataset: pd.DataFrame) -> list[str]:
    columns = [
        column
        for column in dataset.columns
        if column not in METADATA_COLUMNS
        and column not in LABEL_COLUMNS
        and pd.api.types.is_numeric_dtype(dataset[column].dtype)
    ]
    if not columns:
        raise ValueError("No engineered model features are available")
    return columns



def _frozen_sample_order(dataset):
    path=ROOT/"data/DS4_intervals/folds.json"
    if not path.is_file():return dataset
    meta=json.loads(path.read_text(encoding="utf-8"));order={name:i for i,name in enumerate(meta["training_order"])}
    if not set(dataset.video.astype(str)).issubset(order):return dataset
    return dataset.assign(_sample_order=dataset.video.map(order)).sort_values(["_sample_order","bin_start_sec"],kind="mergesort").drop(columns="_sample_order").reset_index(drop=True)

def _grouped_folds(videos,folds):
    path=ROOT/"data/DS4_intervals/folds.json"
    if path.is_file():
        meta=json.loads(path.read_text(encoding="utf-8"));mapping=meta["fold_by_sample"]
        if folds==meta["fold_count"] and set(videos)==set(mapping):
            return [(np.flatnonzero(np.array([mapping[v] for v in videos])!=fold),np.flatnonzero(np.array([mapping[v] for v in videos])==fold)) for fold in range(1,folds+1)]
    return list(GroupKFold(n_splits=folds).split(np.empty(len(videos)),groups=videos))

def _train_oof(
    dataset: pd.DataFrame,
    model_columns: Sequence[str],
    args: argparse.Namespace,
) -> pd.DataFrame:
    videos = dataset["video"].astype(str).to_numpy()
    unique_videos = np.unique(videos)
    folds = int(args.folds)
    if folds < 2 or folds > len(unique_videos):
        raise ValueError(f"folds must be in [2, {len(unique_videos)}]")
    splitter = GroupKFold(n_splits=folds)
    features = dataset[list(model_columns)].astype(np.float32)
    milk = dataset["milk"].to_numpy(dtype=np.int8)
    milk_known = dataset["milk_known"].to_numpy(dtype=bool)
    arching = dataset["arching_state"].to_numpy(dtype=np.int8)
    arching_known = dataset["arching_known"].to_numpy(dtype=bool)
    milk_probability = np.full(len(dataset), np.nan, dtype=float)
    arch_probability = np.full((len(dataset), 3), np.nan, dtype=float)
    fold_ids = np.full(len(dataset), -1, dtype=int)

    for fold, (train_index, valid_index) in enumerate(_grouped_folds(videos,folds), start=1):
        train_mask = np.zeros(len(dataset), dtype=bool)
        train_mask[train_index] = True
        print(
            f"[fold {fold}/{folds}] train_videos={len(np.unique(videos[train_index]))} "
            f"valid_videos={len(np.unique(videos[valid_index]))} train_bins={len(train_index)} "
            f"valid_bins={len(valid_index)}",
            flush=True,
        )
        milk_train = train_mask & milk_known
        arch_train = train_mask & arching_known
        milk_sample_weight = _event_balanced_sample_weights(
            dataset,
            milk,
            milk_known,
            train_mask,
            (1,),
            bin_sec=float(args.bin_sec),
            alpha=float(args.event_weight_alpha),
            boundary_boost=float(
                args.event_boundary_boost
                if getattr(args, "milk_event_boundary_boost", None) is None
                else args.milk_event_boundary_boost
            ),
            maximum_weight=float(args.maximum_event_weight),
        )
        arch_sample_weight = _event_balanced_sample_weights(
            dataset,
            arching,
            arching_known,
            train_mask,
            (1, 2),
            bin_sec=float(args.bin_sec),
            alpha=float(args.event_weight_alpha),
            boundary_boost=float(args.event_boundary_boost),
            maximum_weight=float(args.maximum_event_weight),
        )
        milk_model = _fit_head(
            features.loc[milk_train],
            milk[milk_train],
            (0, 1),
            args,
            args.random_state + fold * 2,
            milk_sample_weight[milk_train],
        )
        arch_model = _fit_head(
            features.loc[arch_train],
            arching[arch_train],
            (0, 1, 2),
            args,
            args.random_state + fold * 2 + 1,
            arch_sample_weight[arch_train],
        )
        milk_probability[valid_index] = _aligned_probabilities(
            milk_model, features.iloc[valid_index], (0, 1)
        )[:, 1]
        arch_probability[valid_index] = _aligned_probabilities(
            arch_model, features.iloc[valid_index], (0, 1, 2)
        )
        fold_ids[valid_index] = fold

    if np.isnan(milk_probability).any() or np.isnan(arch_probability).any() or np.any(fold_ids < 0):
        raise RuntimeError("OOF prediction did not cover every bin")
    oof = dataset[
        [
            "video",
            "source_path",
            "bin_start_sec",
            "bin_end_sec",
            "source_duration_sec",
            "milk",
            "milk_known",
            "arching_state",
            "arching_known",
        ]
    ].copy()
    oof["fold"] = fold_ids
    oof["milk_probability"] = milk_probability
    oof["arching_probability_none"] = arch_probability[:, 0]
    oof["arching_probability_partial"] = arch_probability[:, 1]
    oof["arching_probability_collective"] = arch_probability[:, 2]
    return oof


def _postprocess_config(args: argparse.Namespace, milk_threshold: float) -> dict[str, Any]:
    shared_rolling = getattr(args, "probability_rolling_bins", None)
    milk_rolling = (
        int(shared_rolling)
        if shared_rolling is not None
        else int(getattr(args, "milk_probability_rolling_bins", 11))
    )
    arching_rolling = (
        int(shared_rolling)
        if shared_rolling is not None
        else int(getattr(args, "arching_probability_rolling_bins", 5))
    )
    shared_gap = getattr(args, "merge_gap_sec", None)
    milk_gap = (
        float(shared_gap)
        if shared_gap is not None
        else float(getattr(args, "milk_merge_gap_sec", 1.0))
    )
    arching_gap = (
        float(shared_gap)
        if shared_gap is not None
        else float(getattr(args, "arching_merge_gap_sec", 0.0))
    )
    return {
        "milk_threshold": float(milk_threshold),
        "bin_sec": float(args.bin_sec),
        # Legacy shared values remain in the contract for direct callers; the
        # head-specific values below take precedence in the optimized decoder.
        "probability_rolling_bins": int(arching_rolling),
        "milk_probability_rolling_bins": int(milk_rolling),
        "arching_probability_rolling_bins": int(arching_rolling),
        "merge_gap_sec": float(arching_gap),
        "milk_merge_gap_sec": float(milk_gap),
        "arching_merge_gap_sec": float(arching_gap),
        "milk_minimum_duration_sec": float(args.milk_minimum_duration_sec),
        "arching_minimum_duration_sec": float(args.arching_minimum_duration_sec),
        "milk_keep_highest": False,
        "milk_refractory_sec": float(getattr(args, "milk_refractory_sec", 1200.0)),
        "arching_partial_probability_weight": float(
            getattr(args, "arching_partial_probability_weight", 1.05)
        ),
        "arching_collective_probability_weight": float(
            getattr(args, "arching_collective_probability_weight", 0.80)
        ),
    }


def _training_model_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "estimator": "sklearn.ensemble.HistGradientBoostingClassifier",
        "learning_rate": float(args.learning_rate),
        "max_iter": int(args.max_iter),
        "max_leaf_nodes": int(args.max_leaf_nodes),
        "min_samples_leaf": int(args.min_samples_leaf),
        "l2_regularization": float(args.l2_regularization),
        "class_weight": "balanced",
        "event_weight_alpha": float(args.event_weight_alpha),
        "event_boundary_boost": float(args.event_boundary_boost),
        "milk_event_boundary_boost": float(
            args.event_boundary_boost
            if getattr(args, "milk_event_boundary_boost", None) is None
            else args.milk_event_boundary_boost
        ),
        "maximum_event_weight": float(args.maximum_event_weight),
        "event_weight_normalization": "unit mean within each positive class",
        "early_stopping": True,
        "random_state": int(args.random_state),
    }


def train_command(args: argparse.Namespace) -> None:
    annotation_path = args.annotations.expanduser().resolve()
    output = args.output.expanduser().resolve()
    files = discover_feature_files(args.features)
    annotations = parse_confirmed_annotations(annotation_path)
    files, excluded_feature_videos = _select_training_feature_files(
        files,
        annotations,
        exclude_unannotated=bool(args.exclude_unannotated_feature_videos),
    )
    if excluded_feature_videos:
        print(
            "[dataset] explicitly excluded unannotated feature videos: "
            + ", ".join(excluded_feature_videos),
            flush=True,
        )
    source_schema, raw_signal_columns = _infer_raw_signal_columns(files[0])
    output.mkdir(parents=True, exist_ok=True)
    dataset, durations = _assemble_training_dataset(
        files,
        annotations,
        raw_signal_columns=raw_signal_columns,
        source_schema=source_schema,
        bin_sec=float(args.bin_sec),
    )
    model_columns = _model_feature_columns(dataset)
    print(
        f"[dataset] videos={dataset['video'].nunique()} bins={len(dataset)} "
        f"raw_signals={len(raw_signal_columns)} model_features={len(model_columns)}",
        flush=True,
    )
    oof = _train_oof(dataset, model_columns, args)
    milk_threshold = float(getattr(args, "milk_threshold", 0.5))
    postprocess = _postprocess_config(args, milk_threshold)
    processed = _processed_probabilities(oof, postprocess)
    fixed_postprocess = postprocess
    fixed_processed = processed
    interval_document = _interval_document_from_processed(
        processed,
        bin_sec=float(args.bin_sec),
        postprocess=postprocess,
    )
    processed.to_csv(output / "oof_bins.csv", index=False, encoding="utf-8-sig")
    _write_json(output / "oof_intervals.json", interval_document)
    _flatten_intervals(interval_document).to_csv(
        output / "oof_intervals.csv", index=False, encoding="utf-8-sig"
    )
    continuous_event_metrics, continuous_tp_matches = evaluate_events(
        annotations,
        interval_document,
        durations,
        iou_threshold=0.5,
    )
    event_output = output / "event_evaluation"
    _write_json(event_output / "event_metrics_deltat.json", continuous_event_metrics)
    continuous_tp_matches.to_csv(
        event_output / "tp_event_matches.csv", index=False, encoding="utf-8-sig"
    )

    milk_known = oof["milk_known"].to_numpy(dtype=bool)
    arch_known = oof["arching_known"].to_numpy(dtype=bool)
    raw_milk_metrics_fixed = _binary_metrics(
        oof.loc[milk_known, "milk"].to_numpy(dtype=int),
        oof.loc[milk_known, "milk_probability"].to_numpy(dtype=float),
        0.5,
    )
    arch_probabilities = oof.loc[
        arch_known,
        ["arching_probability_none", "arching_probability_partial", "arching_probability_collective"],
    ].to_numpy(dtype=float)
    raw_arch_metrics = _multiclass_metrics(
        oof.loc[arch_known, "arching_state"].to_numpy(dtype=int), arch_probabilities
    )
    fixed_decoded_bin_metrics = _processed_bin_metrics(fixed_processed)
    fixed_event_metrics = _oof_event_metrics(fixed_processed, bin_sec=float(args.bin_sec))
    tuned_decoded_bin_metrics = _processed_bin_metrics(processed)
    tuned_event_metrics = _oof_event_metrics(processed, bin_sec=float(args.bin_sec))
    annotation_report = build_annotation_report(annotations, durations=durations)
    input_sources = _input_sources(annotation_path, files)
    report = {
        "created_utc": _utc_now(),
        "model_name": MODEL_NAME,
        "offline_model": True,
        "preprocessing_version": PREPROCESSING_VERSION,
        "feature_source": "aggregate_features",
        "input_sources": input_sources,
        "excluded_feature_videos": excluded_feature_videos,
        "dataset": {
            "video_count": int(dataset["video"].nunique()),
            "bin_count": len(dataset),
            "milk_supervised_bin_count": int(milk_known.sum()),
            "arching_supervised_bin_count": int(arch_known.sum()),
            "source_duration_strategy": (
                "max finite interval_end_sec; fallback to final nominal bin end when absent"
            ),
            "source_durations_sec": durations,
        },
        "cross_validation": {
            "kind": "GroupKFold",
            "group": "video",
            "folds": int(args.folds),
            "random_row_split_used": False,
        },
        "milk_threshold": milk_threshold,
        "evaluation": {
            "oof_fixed_threshold_0_5": {
                "selection_scope": (
                    "pre-specified 0.5 threshold with 1200s refractory period and no per-clip single-interval cap"
                ),
                "raw_probability_bin_metrics": {
                    "milk": raw_milk_metrics_fixed,
                    "arching_state": raw_arch_metrics,
                },
                "decoded_bin_metrics": fixed_decoded_bin_metrics,
                "decoded_interval_event_metrics": fixed_event_metrics,
                "postprocess": fixed_postprocess,
            },
        },
        "bin_metrics": tuned_decoded_bin_metrics,
        "interval_event_metrics": tuned_event_metrics,
        "continuous_interval_event_metrics": continuous_event_metrics,
        "annotation_report": annotation_report,
        "model_config": _training_model_config(args),
        "postprocess": postprocess,
    }
    _write_json(output / "report.json", report)

    features = dataset[model_columns].astype(np.float32)
    milk_mask = dataset["milk_known"].to_numpy(dtype=bool)
    arch_mask = dataset["arching_known"].to_numpy(dtype=bool)
    full_selection = np.ones(len(dataset), dtype=bool)
    milk_labels = dataset["milk"].to_numpy(dtype=np.int8)
    arch_labels = dataset["arching_state"].to_numpy(dtype=np.int8)
    milk_sample_weight = _event_balanced_sample_weights(
        dataset,
        milk_labels,
        milk_mask,
        full_selection,
        (1,),
        bin_sec=float(args.bin_sec),
        alpha=float(args.event_weight_alpha),
        boundary_boost=float(
            args.event_boundary_boost
            if getattr(args, "milk_event_boundary_boost", None) is None
            else args.milk_event_boundary_boost
        ),
        maximum_weight=float(args.maximum_event_weight),
    )
    arch_sample_weight = _event_balanced_sample_weights(
        dataset,
        arch_labels,
        arch_mask,
        full_selection,
        (1, 2),
        bin_sec=float(args.bin_sec),
        alpha=float(args.event_weight_alpha),
        boundary_boost=float(args.event_boundary_boost),
        maximum_weight=float(args.maximum_event_weight),
    )
    print("[final] fitting full-data milk and arching heads", flush=True)
    milk_model = _fit_head(
        features.loc[milk_mask],
        milk_labels[milk_mask],
        (0, 1),
        args,
        args.random_state,
        milk_sample_weight[milk_mask],
    )
    arching_model = _fit_head(
        features.loc[arch_mask],
        arch_labels[arch_mask],
        (0, 1, 2),
        args,
        args.random_state + 1,
        arch_sample_weight[arch_mask],
    )
    artifact = {
        "created_utc": _utc_now(),
        "model_name": MODEL_NAME,
        "offline_model": True,
        "preprocessing_version": PREPROCESSING_VERSION,
        "feature_source": "aggregate_features",
        "input_sources": input_sources,
        "excluded_feature_videos": excluded_feature_videos,
        "source_schema": source_schema,
        "raw_signal_columns": raw_signal_columns,
        "model_feature_columns": model_columns,
        "key_signal_columns": [column for column in KEY_SIGNAL_CANDIDATES if column in raw_signal_columns],
        "bin_sec": float(args.bin_sec),
        "source_duration_strategy": (
            "max finite interval_end_sec; fallback to final nominal bin end when absent"
        ),
        "training_source_durations_sec": durations,
        "milk_threshold": float(milk_threshold),
        "deployment_threshold": {
            "head": "milk",
            "value": float(milk_threshold),
            "selection_scope": "pre-specified fixed 0.5 threshold without threshold search",
        },
        "postprocess": postprocess,
        "model_config": _training_model_config(args),
        "milk_model": milk_model,
        "arching_model": arching_model,
        "arching_classes": ARCHING_CLASSES,
    }
    model_path = output / "model.joblib"
    joblib.dump(artifact, model_path)
    print(f"[done] model={model_path} report={output / 'report.json'}", flush=True)


def _prediction_postprocess_config(args: argparse.Namespace, artifact: Mapping[str, Any]) -> dict[str, Any]:
    stored = dict(artifact["postprocess"])
    stored["milk_threshold"] = (
        float(args.milk_threshold)
        if args.milk_threshold is not None
        else float(artifact["milk_threshold"])
    )
    shared_rolling = getattr(args, "probability_rolling_bins", None)
    if shared_rolling is not None:
        stored["probability_rolling_bins"] = int(shared_rolling)
        stored["milk_probability_rolling_bins"] = int(shared_rolling)
        stored["arching_probability_rolling_bins"] = int(shared_rolling)
    shared_gap = getattr(args, "merge_gap_sec", None)
    if shared_gap is not None:
        stored["merge_gap_sec"] = float(shared_gap)
        stored["milk_merge_gap_sec"] = float(shared_gap)
        stored["arching_merge_gap_sec"] = float(shared_gap)
    for argument, key, cast in (
        (getattr(args, "milk_probability_rolling_bins", None), "milk_probability_rolling_bins", int),
        (
            getattr(args, "arching_probability_rolling_bins", None),
            "arching_probability_rolling_bins",
            int,
        ),
        (getattr(args, "milk_merge_gap_sec", None), "milk_merge_gap_sec", float),
        (getattr(args, "arching_merge_gap_sec", None), "arching_merge_gap_sec", float),
        (args.milk_minimum_duration_sec, "milk_minimum_duration_sec", float),
        (args.arching_minimum_duration_sec, "arching_minimum_duration_sec", float),
        (getattr(args, "milk_refractory_sec", None), "milk_refractory_sec", float),
        (
            getattr(args, "arching_partial_probability_weight", None),
            "arching_partial_probability_weight",
            float,
        ),
        (
            getattr(args, "arching_collective_probability_weight", None),
            "arching_collective_probability_weight",
            float,
        ),
    ):
        if argument is not None:
            stored[key] = cast(argument)
    stored["bin_sec"] = float(artifact["bin_sec"])
    return stored


def predict_command(args: argparse.Namespace) -> None:
    model_path = args.model.expanduser().resolve()
    artifact = joblib.load(model_path)
    if artifact.get("preprocessing_version") not in (None, PREPROCESSING_VERSION):
        raise ValueError(
            "Unsupported preprocessing version: "
            f"{artifact.get('preprocessing_version')!r}; expected {PREPROCESSING_VERSION!r}"
        )
    if artifact.get("feature_source") != "aggregate_features":
        raise ValueError(f"Unsupported feature source: {artifact.get('feature_source')}")
    files = discover_feature_files(args.input)
    _feature_stems(files, context="Prediction input")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw_columns = list(artifact["raw_signal_columns"])
    source_schema = list(artifact["source_schema"])
    model_columns = list(artifact["model_feature_columns"])
    bin_sec = float(artifact["bin_sec"])
    pieces: list[pd.DataFrame] = []
    for index, path in enumerate(files, start=1):
        print(f"[predict {index:03d}/{len(files):03d}] {path.name}", flush=True)
        binned = aggregate_feature_csv(
            path,
            raw_columns,
            bin_sec=bin_sec,
            expected_schema=source_schema,
        )
        engineered = engineer_video_features(
            binned,
            artifact.get("key_signal_columns", KEY_SIGNAL_CANDIDATES),
            bin_sec=bin_sec,
        )
        missing = set(model_columns) - set(engineered.columns)
        if missing:
            raise ValueError(f"Engineered feature contract is incomplete for {path}: {sorted(missing)}")
        features = engineered[model_columns].astype(np.float32)
        milk_probability = _aligned_probabilities(artifact["milk_model"], features, (0, 1))[:, 1]
        arch_probability = _aligned_probabilities(artifact["arching_model"], features, (0, 1, 2))
        probabilities = engineered[
            ["video", "source_path", "bin_start_sec", "bin_end_sec", "source_duration_sec"]
        ].copy()
        probabilities["milk_probability"] = milk_probability
        probabilities["arching_probability_none"] = arch_probability[:, 0]
        probabilities["arching_probability_partial"] = arch_probability[:, 1]
        probabilities["arching_probability_collective"] = arch_probability[:, 2]
        pieces.append(probabilities)
    probabilities = pd.concat(pieces, ignore_index=True)
    duplicated = probabilities.duplicated(["video", "bin_start_sec"])
    if bool(duplicated.any()):
        examples = probabilities.loc[duplicated, ["video", "bin_start_sec"]].head().to_dict("records")
        raise ValueError(f"Prediction inputs contain duplicate video/bin rows: {examples}")
    postprocess = _prediction_postprocess_config(args, artifact)
    processed = _processed_probabilities(probabilities, postprocess)
    intervals = _interval_document_from_processed(
        processed,
        bin_sec=bin_sec,
        postprocess=postprocess,
    )
    processed.to_csv(output / "probabilities.csv", index=False, encoding="utf-8-sig")
    _write_json(output / "intervals.json", intervals)
    print(f"[done] probabilities={output / 'probabilities.csv'} intervals={output / 'intervals.json'}", flush=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train dual heads with grouped OOF evaluation.")
    train.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    train.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    train.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    train.add_argument(
        "--exclude-unannotated-feature-videos",
        action="store_true",
        help="Explicitly ignore feature CSVs whose video is absent from the annotation file.",
    )
    train.add_argument("--bin-sec", type=float, default=1.0)
    train.add_argument("--folds", type=int, default=5)
    train.add_argument("--random-state", type=int, default=20260811)
    train.add_argument("--learning-rate", type=float, default=0.06)
    train.add_argument("--max-iter", type=int, default=160)
    train.add_argument("--max-leaf-nodes", type=int, default=31)
    train.add_argument("--min-samples-leaf", type=int, default=24)
    train.add_argument("--l2-regularization", type=float, default=1.0)
    train.add_argument(
        "--probability-rolling-bins",
        type=int,
        default=None,
        help="Legacy override that applies one rolling window to both heads.",
    )
    train.add_argument("--milk-probability-rolling-bins", type=int, default=11)
    train.add_argument("--arching-probability-rolling-bins", type=int, default=5)
    train.add_argument(
        "--merge-gap-sec",
        type=float,
        default=None,
        help="Legacy override that applies one merge gap to both heads.",
    )
    train.add_argument("--milk-merge-gap-sec", type=float, default=0.0)
    train.add_argument("--arching-merge-gap-sec", type=float, default=0.0)
    train.add_argument("--milk-threshold", type=float, default=0.5)
    train.add_argument("--milk-refractory-sec", type=float, default=1200.0)
    train.add_argument("--milk-minimum-duration-sec", type=float, default=8.0)
    train.add_argument("--arching-minimum-duration-sec", type=float, default=4.0)
    train.add_argument("--arching-partial-probability-weight", type=float, default=1.05)
    train.add_argument("--arching-collective-probability-weight", type=float, default=0.80)
    train.add_argument("--event-weight-alpha", type=float, default=0.5)
    train.add_argument("--event-boundary-boost", type=float, default=1.5)
    train.add_argument(
        "--milk-event-boundary-boost",
        type=float,
        default=3.0,
        help="Milk-only event-boundary weight; use the shared value only when explicitly omitted by callers.",
    )
    train.add_argument("--maximum-event-weight", type=float, default=4.0)
    train.set_defaults(handler=train_command)

    predict = subparsers.add_parser("predict", help="Apply a saved model to one CSV or a directory.")
    predict.add_argument("--input", type=Path, default=DEFAULT_FEATURES)
    predict.add_argument("--model", type=Path, default=ROOT / "models/intervals/model.joblib")
    predict.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    predict.add_argument("--milk-threshold", type=float, default=None)
    predict.add_argument("--milk-refractory-sec", type=float, default=None)
    predict.add_argument("--probability-rolling-bins", type=int, default=None)
    predict.add_argument("--milk-probability-rolling-bins", type=int, default=None)
    predict.add_argument("--arching-probability-rolling-bins", type=int, default=None)
    predict.add_argument("--merge-gap-sec", type=float, default=None)
    predict.add_argument("--milk-merge-gap-sec", type=float, default=None)
    predict.add_argument("--arching-merge-gap-sec", type=float, default=None)
    predict.add_argument("--milk-minimum-duration-sec", type=float, default=None)
    predict.add_argument("--arching-minimum-duration-sec", type=float, default=None)
    predict.add_argument("--arching-partial-probability-weight", type=float, default=None)
    predict.add_argument("--arching-collective-probability-weight", type=float, default=None)
    predict.set_defaults(handler=predict_command)
    return parser.parse_args(argv)


def _validate_cli(args: argparse.Namespace) -> None:
    if getattr(args, "bin_sec", 1.0) <= 0:
        raise ValueError("--bin-sec must be positive")
    for name in (
        "merge_gap_sec",
        "milk_merge_gap_sec",
        "arching_merge_gap_sec",
        "milk_minimum_duration_sec",
        "arching_minimum_duration_sec",
    ):
        value = getattr(args, name, None)
        if value is not None and value < 0:
            raise ValueError(f"--{name.replace('_', '-')} cannot be negative")
    for name in (
        "probability_rolling_bins",
        "milk_probability_rolling_bins",
        "arching_probability_rolling_bins",
    ):
        rolling = getattr(args, name, None)
        if rolling is not None and rolling < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be at least 1")
    for name in (
        "arching_partial_probability_weight",
        "arching_collective_probability_weight",
    ):
        value = getattr(args, name, None)
        if value is not None and value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    alpha = getattr(args, "event_weight_alpha", None)
    if alpha is not None and alpha < 0:
        raise ValueError("--event-weight-alpha cannot be negative")
    for name in ("event_boundary_boost", "milk_event_boundary_boost"):
        boundary_boost = getattr(args, name, None)
        if boundary_boost is not None and boundary_boost < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be at least 1")
    maximum_event_weight = getattr(args, "maximum_event_weight", None)
    if maximum_event_weight is not None and maximum_event_weight < 1:
        raise ValueError("--maximum-event-weight must be at least 1")
    threshold = getattr(args, "milk_threshold", None)
    if threshold is not None and not 0.0 <= threshold <= 1.0:
        raise ValueError("--milk-threshold must be in [0, 1]")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    _validate_cli(args)
    args.handler(args)


if __name__ == "__main__":
    main()
