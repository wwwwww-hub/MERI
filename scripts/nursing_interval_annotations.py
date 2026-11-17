"""Parse and rasterize confirmed nursing interval annotations.

The source Markdown is human-edited and therefore intentionally parsed by
label-token positions rather than by line boundaries.  This matters for the
two entries where the milk and collective-arching fields occur on one line.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


__all__ = [
    "Interval",
    "VideoAnnotation",
    "build_annotation_report",
    "parse_confirmed_annotations",
    "rasterize_annotation",
]


_HEADER_RE = re.compile(
    r"(?m)^[ \t]*(?P<index>\d{3})\\?\.[ \t]+(?P<video>.+?)[ \t]*[：:][ \t]*$"
)
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_LABEL_RE = re.compile(r"(?P<label>建议放乳|集体拱乳|部分拱乳|放乳)[ \t]*[：:]")
_INTERVAL_RE = re.compile(
    r"(?<![\d.])(?P<start>\d+(?:\.\d+)?)[ \t]*s?[ \t]*[-–—−][ \t]*"
    r"(?P<end>\d+(?:\.\d+)?)[ \t]*s?",
    re.IGNORECASE,
)

_MILK = "milk"
_COLLECTIVE = "collective"
_PARTIAL = "partial"
_LABEL_KIND = {
    "放乳": _MILK,
    "建议放乳": _MILK,
    "集体拱乳": _COLLECTIVE,
    "部分拱乳": _PARTIAL,
}


@dataclass(frozen=True)
class Interval:
    """A half-open interval ``[start_sec, end_sec)`` on a video timeline."""

    start_sec: float
    end_sec: float

    def __post_init__(self) -> None:
        start_sec = float(self.start_sec)
        end_sec = float(self.end_sec)
        if not math.isfinite(start_sec) or not math.isfinite(end_sec):
            raise ValueError("Interval endpoints must be finite.")
        if end_sec <= start_sec:
            raise ValueError(
                f"Interval end must be after start, got {start_sec:g}-{end_sec:g}."
            )
        object.__setattr__(self, "start_sec", start_sec)
        object.__setattr__(self, "end_sec", end_sec)

    @property
    def duration_sec(self) -> float:
        """Return the interval duration in seconds."""

        return self.end_sec - self.start_sec


@dataclass(frozen=True)
class VideoAnnotation:
    """All confirmed interval annotations associated with one video stem."""

    index: int
    video: str
    milk_intervals: tuple[Interval, ...]
    collective_intervals: tuple[Interval, ...]
    partial_intervals: tuple[Interval, ...]
    milk_supervised: bool
    milk_note: str
    raw_body: str


def _parse_interval_section(section: str, *, field: str, video: str) -> tuple[Interval, ...]:
    """Extract intervals from one label section while retaining permissive prose."""

    # Markdown strikethrough may split the trailing unit, e.g.
    # ``~~205s-425~~s``.  Removing only the markup from a parsing copy turns
    # both observed strikethrough forms into the normal interval grammar.
    parse_text = section.replace("~~", "")
    intervals = tuple(
        Interval(float(match.group("start")), float(match.group("end")))
        for match in _INTERVAL_RE.finditer(parse_text)
    )
    if intervals:
        return intervals
    if "无" in parse_text:
        return ()
    raise ValueError(f"Could not parse {field} interval(s) for {video!r}: {section!r}")


def _positive_overlap(first: Interval, second: Interval) -> float:
    return max(0.0, min(first.end_sec, second.end_sec) - max(first.start_sec, second.start_sec))


def _find_positive_overlap(
    first: Sequence[Interval], second: Sequence[Interval]
) -> tuple[Interval, Interval, float] | None:
    for first_interval in first:
        for second_interval in second:
            overlap_sec = _positive_overlap(first_interval, second_interval)
            if overlap_sec > 0.0:
                return first_interval, second_interval, overlap_sec
    return None


def _parse_entry(
    index: int,
    video: str,
    raw_body: str,
    *,
    allow_arching_overlap: bool = False,
) -> VideoAnnotation:
    tokens = list(_LABEL_RE.finditer(raw_body))
    sections: dict[str, tuple[str, str]] = {}
    order: list[str] = []
    for token_index, token in enumerate(tokens):
        kind = _LABEL_KIND[token.group("label")]
        if kind in sections:
            raise ValueError(f"Duplicate {kind} label for {video!r}.")
        section_end = tokens[token_index + 1].start() if token_index + 1 < len(tokens) else len(raw_body)
        full_section = raw_body[token.start() : section_end].strip()
        value = raw_body[token.end() : section_end].strip()
        sections[kind] = (full_section, value)
        order.append(kind)

    expected_order = [_MILK, _COLLECTIVE, _PARTIAL]
    if order != expected_order:
        missing = [kind for kind in expected_order if kind not in sections]
        raise ValueError(
            f"Expected milk, collective, and partial labels in order for {video!r}; "
            f"found {order!r}, missing {missing!r}."
        )

    milk_note, milk_value = sections[_MILK]
    _, collective_value = sections[_COLLECTIVE]
    _, partial_value = sections[_PARTIAL]
    milk_intervals = _parse_interval_section(milk_value, field="milk", video=video)
    collective_intervals = _parse_interval_section(
        collective_value, field="collective arching", video=video
    )
    partial_intervals = _parse_interval_section(partial_value, field="partial arching", video=video)

    overlap = _find_positive_overlap(collective_intervals, partial_intervals)
    if overlap is not None and not allow_arching_overlap:
        collective_interval, partial_interval, overlap_sec = overlap
        raise ValueError(
            f"Collective and partial arching overlap for {video!r}: "
            f"{collective_interval!r} vs {partial_interval!r} ({overlap_sec:g}s)."
        )

    # The confirmed file is authoritative.  Human review prose and Markdown
    # strikethrough are retained for audit in milk_note/raw_body but do not mask
    # an interval from supervision.
    return VideoAnnotation(
        index=index,
        video=video,
        milk_intervals=milk_intervals,
        collective_intervals=collective_intervals,
        partial_intervals=partial_intervals,
        milk_supervised=True,
        milk_note=milk_note,
        raw_body=raw_body.strip(),
    )


def parse_confirmed_annotations(
    path: Path,
    *,
    allow_arching_overlap: bool = False,
) -> dict[str, VideoAnnotation]:
    """Parse the confirmed Markdown annotations, keyed by unescaped video stem.

    The parser accepts an escaped list marker such as ``001\\.`` (and the plain
    ``001.`` form), restores Markdown-escaped underscores in video names, and
    extracts the three label sections by token position. ASCII hyphens and
    common Unicode dash characters are accepted between interval endpoints. HTML-commented
    entries are excluded. Interval endpoints
    may be integers or decimals, and the start endpoint's ``s`` suffix is
    optional.  ``建议放乳``, ``保留`` and ``保持`` prose does not change the
    interval semantics; all entries in this confirmed file are supervised.

    Args:
        path: UTF-8 or UTF-8-with-BOM Markdown annotation file.
        allow_arching_overlap: Retain overlapping partial/collective intervals
            for audit. Keep false for model training, whose arching target is a
            single mutually exclusive state.

    Returns:
        An insertion-ordered dictionary from video stem (without an added file
        extension) to :class:`VideoAnnotation`.

    Raises:
        ValueError: If headers are duplicated, a required label is absent, an
            interval is malformed, or collective and partial arching overlap.
    """

    path = Path(path)
    text = _HTML_COMMENT_RE.sub("", path.read_text(encoding="utf-8-sig"))
    headers = list(_HEADER_RE.finditer(text))
    if not headers:
        raise ValueError(f"No annotation headers found in {path}.")

    annotations: dict[str, VideoAnnotation] = {}
    seen_indices: set[int] = set()
    for header_index, header in enumerate(headers):
        index = int(header.group("index"))
        video = header.group("video").replace(r"\_", "_").strip()
        body_end = headers[header_index + 1].start() if header_index + 1 < len(headers) else len(text)
        raw_body = text[header.end() : body_end]
        if index in seen_indices:
            raise ValueError(f"Duplicate annotation index: {index:03d}.")
        if video in annotations:
            raise ValueError(f"Duplicate video annotation: {video!r}.")
        seen_indices.add(index)
        annotations[video] = _parse_entry(
            index,
            video,
            raw_body,
            allow_arching_overlap=allow_arching_overlap,
        )
    return annotations


def _validate_duration(duration_sec: float | None) -> float | None:
    if duration_sec is None:
        return None
    duration = float(duration_sec)
    if not math.isfinite(duration) or duration < 0.0:
        raise ValueError(f"duration_sec must be finite and non-negative, got {duration_sec!r}.")
    return duration


def _clipped_bounds(
    intervals: Sequence[Interval], duration_sec: float | None
) -> list[tuple[float, float]]:
    upper = math.inf if duration_sec is None else duration_sec
    clipped: list[tuple[float, float]] = []
    for interval in intervals:
        start_sec = max(0.0, interval.start_sec)
        end_sec = min(upper, interval.end_sec)
        if end_sec > start_sec:
            clipped.append((start_sec, end_sec))
    return clipped


def _bounds_overlap(
    first: Sequence[tuple[float, float]], second: Sequence[tuple[float, float]]
) -> float:
    return sum(
        max(0.0, min(first_end, second_end) - max(first_start, second_start))
        for first_start, first_end in first
        for second_start, second_end in second
    )


def rasterize_annotation(
    annotation: VideoAnnotation,
    bin_starts: np.ndarray,
    bin_sec: float,
    duration_sec: float | None = None,
) -> dict[str, np.ndarray]:
    """Rasterize one annotation on bin midpoints using half-open intervals.

    ``partial`` maps to arching state 1 and ``collective`` maps to state 2.
    Milk is an independent binary target and may overlap either arching state.
    Intervals are clipped to the video timeline.  Bins whose midpoint lies
    outside ``[0, duration_sec)`` are emitted as zero and unknown when a
    duration is supplied.

    Raises:
        ValueError: If inputs are invalid or clipped collective and partial
            arching intervals have positive-duration overlap.
    """

    starts = np.asarray(bin_starts, dtype=np.float64)
    if starts.ndim != 1:
        raise ValueError(f"bin_starts must be one-dimensional, got shape {starts.shape!r}.")
    if not np.all(np.isfinite(starts)):
        raise ValueError("bin_starts must contain only finite values.")
    width = float(bin_sec)
    if not math.isfinite(width) or width <= 0.0:
        raise ValueError(f"bin_sec must be finite and positive, got {bin_sec!r}.")
    duration = _validate_duration(duration_sec)
    midpoints = starts + width / 2.0

    valid_bins = np.ones(starts.shape, dtype=np.bool_)
    if duration is not None:
        valid_bins &= (midpoints >= 0.0) & (midpoints < duration)

    milk_bounds = _clipped_bounds(annotation.milk_intervals, duration)
    collective_bounds = _clipped_bounds(annotation.collective_intervals, duration)
    partial_bounds = _clipped_bounds(annotation.partial_intervals, duration)
    arching_overlap = _bounds_overlap(collective_bounds, partial_bounds)
    if arching_overlap > 0.0:
        raise ValueError(
            f"Collective and partial arching overlap for {annotation.video!r} "
            f"by {arching_overlap:g}s after clipping."
        )

    milk = np.zeros(starts.shape, dtype=np.int8)
    if annotation.milk_supervised:
        for start_sec, end_sec in milk_bounds:
            milk[(midpoints >= start_sec) & (midpoints < end_sec) & valid_bins] = 1

    arching_state = np.zeros(starts.shape, dtype=np.int8)
    for start_sec, end_sec in partial_bounds:
        arching_state[(midpoints >= start_sec) & (midpoints < end_sec) & valid_bins] = 1
    for start_sec, end_sec in collective_bounds:
        arching_state[(midpoints >= start_sec) & (midpoints < end_sec) & valid_bins] = 2

    milk_known = valid_bins & np.bool_(annotation.milk_supervised)
    arching_known = valid_bins.copy()
    return {
        "milk": milk,
        "milk_known": milk_known.astype(np.bool_, copy=False),
        "arching_state": arching_state,
        "arching_known": arching_known.astype(np.bool_, copy=False),
    }


def _duration_for_video(
    video: str, durations: Mapping[str, float] | None
) -> tuple[float | None, bool]:
    if durations is None:
        return None, False
    candidates = (video, f"{video}.mp4") if not video.lower().endswith(".mp4") else (video, video[:-4])
    for candidate in candidates:
        if candidate in durations:
            return _validate_duration(durations[candidate]), True
    return None, False


def build_annotation_report(
    annotations: Mapping[str, VideoAnnotation],
    durations: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Build aggregate counts, clipped durations, unknown IDs, and overlaps.

    Args:
        annotations: Parsed annotations keyed by video stem.
        durations: Optional video-duration mapping.  Keys may be either the
            annotation stem or that stem plus ``.mp4``.  When present, interval
            durations and overlaps are clipped to ``[0, duration)``.

    Returns:
        A JSON-serializable dictionary.  Frequently used regression fields are
        available as flat keys, while ``class_stats`` and
        ``overlap_duration_sec`` provide structured detail.
    """

    ordered = sorted(annotations.values(), key=lambda annotation: annotation.index)
    unknown_ids = [annotation.index for annotation in ordered if not annotation.milk_supervised]
    interval_counts = {
        _MILK: sum(len(annotation.milk_intervals) for annotation in ordered),
        _COLLECTIVE: sum(len(annotation.collective_intervals) for annotation in ordered),
        _PARTIAL: sum(len(annotation.partial_intervals) for annotation in ordered),
    }
    class_video_counts = {
        _MILK: sum(bool(annotation.milk_intervals) for annotation in ordered),
        _COLLECTIVE: sum(bool(annotation.collective_intervals) for annotation in ordered),
        _PARTIAL: sum(bool(annotation.partial_intervals) for annotation in ordered),
    }
    class_durations = {_MILK: 0.0, _COLLECTIVE: 0.0, _PARTIAL: 0.0}
    overlaps = {
        "milk_collective": 0.0,
        "milk_partial": 0.0,
        "collective_partial": 0.0,
    }
    total_video_duration_sec = 0.0
    duration_video_count = 0
    missing_duration_videos: list[str] = []

    for annotation in ordered:
        duration, has_duration = _duration_for_video(annotation.video, durations)
        if has_duration:
            assert duration is not None
            total_video_duration_sec += duration
            duration_video_count += 1
        elif durations is not None:
            missing_duration_videos.append(annotation.video)

        milk = _clipped_bounds(annotation.milk_intervals, duration)
        collective = _clipped_bounds(annotation.collective_intervals, duration)
        partial = _clipped_bounds(annotation.partial_intervals, duration)
        class_durations[_MILK] += sum(end - start for start, end in milk)
        class_durations[_COLLECTIVE] += sum(end - start for start, end in collective)
        class_durations[_PARTIAL] += sum(end - start for start, end in partial)
        overlaps["milk_collective"] += _bounds_overlap(milk, collective)
        overlaps["milk_partial"] += _bounds_overlap(milk, partial)
        overlaps["collective_partial"] += _bounds_overlap(collective, partial)

    class_stats = {
        kind: {
            "video_count": class_video_counts[kind],
            "interval_count": interval_counts[kind],
            "duration_sec": class_durations[kind],
        }
        for kind in (_MILK, _COLLECTIVE, _PARTIAL)
    }
    return {
        "video_count": len(ordered),
        "milk_supervised_video_count": len(ordered) - len(unknown_ids),
        "milk_unknown_video_count": len(unknown_ids),
        "unknown_ids": unknown_ids,
        "milk_unknown_ids": unknown_ids,
        "milk_interval_count": interval_counts[_MILK],
        "collective_interval_count": interval_counts[_COLLECTIVE],
        "partial_interval_count": interval_counts[_PARTIAL],
        "milk_duration_sec": class_durations[_MILK],
        "collective_duration_sec": class_durations[_COLLECTIVE],
        "partial_duration_sec": class_durations[_PARTIAL],
        "class_stats": class_stats,
        "overlap_duration_sec": overlaps,
        "milk_collective_overlap_sec": overlaps["milk_collective"],
        "milk_partial_overlap_sec": overlaps["milk_partial"],
        "collective_partial_overlap_sec": overlaps["collective_partial"],
        "duration_video_count": duration_video_count,
        "total_video_duration_sec": total_video_duration_sec,
        "missing_duration_videos": missing_duration_videos,
    }
