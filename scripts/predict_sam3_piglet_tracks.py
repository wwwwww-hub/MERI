"""Pre-annotate sampled clips with finetuned SAM3 masks and the official video tracker.

Input is ``selection_manifest.csv`` written by
``sample_finetune_context_clips.py``.  Each clip is sampled to numeric JPEG
frames (required by SAM3's official JPEG-folder loader), initialized on its
anchor frame with the separately finetuned sow and piglet image models, propagated
in both time directions with the official SAM3 tracker, optionally refined on every
frame with the finetuned image models, and exported as native X-AnyLabeling/LabelMe
polygon JSON files.

The finetuned image checkpoints are intentionally *not* passed directly to the
video-model builder: they have image-model keys and no temporal tracker weights.
Instead, ``--final-mask-source hybrid`` (the default) matches per-frame finetuned
masks to the official track IDs and falls back to tracker masks when a match is
missing or ambiguous.  ``tracker`` preserves the original behavior and
``finetuned`` exports only reliable matched finetuned masks.

For the recommended two-stage workflow, run ``--seed-only``, correct the
generated anchor JSON in X-AnyLabeling, then run again with
``--tracking-seed anchor-json`` (without ``--overwrite``).

Example:
    python scripts/predict_sam3_piglet_tracks.py ^
      --input-dir working_data/nursing_clips ^
      --output-dir working_data/tracking_annotations ^
      --track-fps 2 --final-mask-source hybrid --overlap-policy all ^
      --tracking-seed anchor-json ^
      --force-retrack --state-on-gpu
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
from PIL import Image


DEFAULT_SAM3_ROOT = Path(__file__).resolve().parents[1]/"third_party/sam3"
DEFAULT_SOW_CHECKPOINT = Path(__file__).resolve().parents[1]/"models/sam3/sow_finetuned.pt"
DEFAULT_PIGLET_CHECKPOINT = Path(__file__).resolve().parents[1]/"models/sam3/piglet_finetuned.pt"
DEFAULT_VIDEO_CHECKPOINT = Path(__file__).resolve().parents[1]/"models/sam3/sam3_original_video.pt"
DEFAULT_BPE_PATH = DEFAULT_SAM3_ROOT / "assets" / "bpe_simple_vocab_16e6.txt.gz"
MAX_SOW_INSTANCES = 1
NUMERIC_JPEG_RE = re.compile(r"^(?P<index>\d+)\.jpe?g$", re.IGNORECASE)
SAFE_CLIP_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,239}$")


@dataclass(frozen=True)
class ClipSpec:
    clip_id: str
    output_video: Path
    source_video: Path
    anchor_image: Path
    anchor_json: Path | None
    clip_start_sec: float
    clip_duration_sec: float
    anchor_source_sec: float
    anchor_in_clip_sec: float


@dataclass(frozen=True)
class PreparedClip:
    spec: ClipSpec
    clip_dir: Path
    frames_dir: Path
    frame_paths: tuple[Path, ...]
    sample_start_clip_sec: float
    anchor_frame_index: int
    track_fps: float

    def clip_timestamp_sec(self, frame_index: int) -> float:
        return self.sample_start_clip_sec + frame_index / self.track_fps

    def source_timestamp_sec(self, frame_index: int) -> float:
        return self.spec.clip_start_sec + self.clip_timestamp_sec(frame_index)


@dataclass(frozen=True)
class SeedModelSpec:
    label: str
    prompt: str
    checkpoint: Path
    confidence_threshold: float
    max_instances: int


@dataclass(frozen=True)
class SeedInstance:
    track_id: int
    label: str
    mask: np.ndarray
    score: float
    box_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class TrackedInstance:
    track_id: int
    label: str
    mask: np.ndarray
    object_score_logit: float | None
    object_probability: float | None
    mask_source: str
    finetuned_score: float | None = None
    fusion_match_score: float | None = None
    fusion_mask_iou: float | None = None


@dataclass(frozen=True)
class FusionDiagnostics:
    frame_index: int
    label: str
    detection_index: int
    status: str
    finetuned_score: float
    bbox_xyxy: tuple[float, float, float, float]
    area_px: int
    candidate_track_ids: tuple[int, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sam3-root", type=Path, default=DEFAULT_SAM3_ROOT)
    parser.add_argument("--sow-checkpoint", type=Path, default=DEFAULT_SOW_CHECKPOINT)
    parser.add_argument("--piglet-checkpoint", type=Path, default=DEFAULT_PIGLET_CHECKPOINT)
    parser.add_argument("--video-checkpoint", type=Path, default=DEFAULT_VIDEO_CHECKPOINT)
    parser.add_argument("--bpe-path", type=Path, default=DEFAULT_BPE_PATH)
    parser.add_argument("--sow-prompt", default="sow")
    parser.add_argument("--piglet-prompt", default="piglet")
    parser.add_argument("--sow-label", default="sow")
    parser.add_argument("--piglet-label", default="piglet")
    parser.add_argument(
        "--skip-sow",
        action="store_true",
        help="Track piglets only. By default the separately finetuned sow model is also initialized and tracked.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--track-fps", type=float, default=2.0)
    parser.add_argument("--max-clips", type=int, default=0, help="0 processes every manifest row.")
    parser.add_argument(
        "--smoke-radius-frames",
        type=int,
        default=0,
        help="If >0, only sample anchor +/- this many frames for a fast end-to-end smoke test.",
    )
    parser.add_argument("--sow-confidence-threshold", type=float, default=0.5)
    parser.add_argument("--piglet-confidence-threshold", type=float, default=0.5)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--min-seed-mask-pixels", type=int, default=80)
    parser.add_argument(
        "--max-sow-instances",
        type=int,
        choices=(MAX_SOW_INSTANCES,),
        default=MAX_SOW_INSTANCES,
        help="Hard maximum sow seeds; only 1 is supported.",
    )
    parser.add_argument(
        "--max-seed-instances",
        type=int,
        default=0,
        help="Maximum piglet seeds; 0 keeps every accepted seed.",
    )
    parser.add_argument("--dedupe-iou-threshold", type=float, default=0.65)
    parser.add_argument("--containment-threshold", type=float, default=0.85)
    parser.add_argument("--child-max-area-ratio", type=float, default=0.75)
    parser.add_argument("--min-presence-prob", type=float, default=0.10)
    parser.add_argument("--min-track-mask-pixels", type=int, default=40)
    parser.add_argument(
        "--final-mask-source",
        choices=("tracker", "finetuned", "hybrid"),
        default="hybrid",
        help=(
            "tracker exports official tracker masks; finetuned exports only matched per-frame "
            "finetuned masks; hybrid uses matched finetuned masks and falls back to tracker masks."
        ),
    )
    parser.add_argument(
        "--overlap-policy",
        choices=("exclusive", "cross-class", "all"),
        default="all",
        help=(
            "exclusive makes every final instance mutually exclusive; cross-class preserves "
            "sow/piglet overlap while resolving overlaps among piglet instances; all preserves "
            "overlaps between every independently predicted instance."
        ),
    )
    parser.add_argument(
        "--fusion-min-mask-iou",
        type=float,
        default=0.10,
        help="Minimum raw mask IoU for a finetuned detection/track match.",
    )
    parser.add_argument(
        "--fusion-min-match-score",
        type=float,
        default=0.30,
        help="Minimum weighted mask/box/centroid score accepted by finetuned or hybrid mode.",
    )
    parser.add_argument(
        "--fusion-merge-containment-threshold",
        type=float,
        default=0.50,
        help=(
            "Reject a finetuned mask as an ambiguous merge when it contains at least this "
            "fraction of two or more tracker masks of the same class."
        ),
    )
    parser.add_argument("--min-polygon-area", type=float, default=40.0)
    parser.add_argument("--epsilon-ratio", type=float, default=0.002)
    parser.add_argument("--max-polygon-points", type=int, default=300)
    parser.add_argument(
        "--max-polygons-per-mask",
        type=int,
        default=0,
        help="0 keeps all exterior components; positive values keep the largest N.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=2, help="FFmpeg -q:v value, 2 is high quality.")
    parser.add_argument("--ffmpeg", type=Path, default=None)
    parser.add_argument("--anylabeling-version", default="4.0.0-beta.9")
    state_group = parser.add_mutually_exclusive_group()
    state_group.add_argument(
        "--offload-state-to-cpu",
        dest="offload_state_to_cpu",
        action="store_true",
        help="Store tracker state on CPU to reduce VRAM use (default).",
    )
    state_group.add_argument(
        "--state-on-gpu",
        dest="offload_state_to_cpu",
        action="store_false",
        help="Keep tracker state on GPU for speed; may exceed 12 GB VRAM.",
    )
    parser.set_defaults(offload_state_to_cpu=True)
    parser.add_argument("--frames-only", action="store_true")
    parser.add_argument("--seed-only", action="store_true")
    parser.add_argument(
        "--tracking-seed",
        choices=("sam3-cache", "anchor-json"),
        default="sam3-cache",
        help=(
            "Initialize tracking from cached finetuned-SAM3 masks, or from the anchor frame JSON "
            "after correcting it in X-AnyLabeling."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--force-retrack",
        action="store_true",
        help=(
            "Regenerate per-frame tracking outputs while preserving extracted frames, seed caches, "
            "per-frame finetuned caches, and the edited anchor JSON. Other per-frame JSON files are replaced."
        ),
    )
    parser.add_argument("--fail-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    if resolved.stat().st_size <= 0:
        raise ValueError(f"{label} is empty: {resolved}")
    return resolved


def path_fingerprint(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    payload: dict[str, Any] = {"path": str(resolved)}
    if resolved.is_file():
        stat = resolved.stat()
        payload.update({"size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    else:
        payload["missing"] = True
    return payload


def frame_configuration(
    spec: ClipSpec,
    args: argparse.Namespace,
    sample_start_sec: float,
    sample_duration_sec: float,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "input_clip": path_fingerprint(spec.output_video),
        "clip_start_sec": spec.clip_start_sec,
        "clip_duration_sec": spec.clip_duration_sec,
        "anchor_source_sec": spec.anchor_source_sec,
        "anchor_in_clip_sec": spec.anchor_in_clip_sec,
        "track_fps": args.track_fps,
        "sample_start_clip_sec": sample_start_sec,
        "sample_duration_sec": sample_duration_sec,
        "smoke_radius_frames": args.smoke_radius_frames,
        "jpeg_quality": args.jpeg_quality,
    }


def seed_configuration(args: argparse.Namespace, prepared: PreparedClip) -> dict[str, Any]:
    return {
        "schema_version": 4,
        "anchor_frame": path_fingerprint(prepared.frame_paths[prepared.anchor_frame_index]),
        "anchor_frame_index": prepared.anchor_frame_index,
        "bpe_path": path_fingerprint(args.bpe_path),
        "models": [
            {
                "label": model.label,
                "prompt": model.prompt,
                "checkpoint": path_fingerprint(model.checkpoint),
                "confidence_threshold": model.confidence_threshold,
                "max_instances": model.max_instances,
            }
            for model in requested_seed_models(args)
        ],
        "mask_threshold": args.mask_threshold,
        "min_seed_mask_pixels": args.min_seed_mask_pixels,
        "dedupe_iou_threshold": args.dedupe_iou_threshold,
        "containment_threshold": args.containment_threshold,
        "child_max_area_ratio": args.child_max_area_ratio,
        "overlap_policy": args.overlap_policy,
        "sow_mask_policy": "single_instance_largest_connected_component",
    }


def tracking_configuration(args: argparse.Namespace, prepared: PreparedClip) -> dict[str, Any]:
    seed_archive, _ = seed_archive_paths(prepared)
    payload = {
        "schema_version": 7,
        "video_checkpoint": path_fingerprint(args.video_checkpoint),
        "seed_configuration": seed_configuration(args, prepared),
        "seed_archive": path_fingerprint(seed_archive),
        "tracking_seed": args.tracking_seed,
        "frame_count": len(prepared.frame_paths),
        "min_presence_prob": args.min_presence_prob,
        "min_track_mask_pixels": args.min_track_mask_pixels,
        "final_mask_source": args.final_mask_source,
        "overlap_policy": args.overlap_policy,
        "fusion_min_mask_iou": args.fusion_min_mask_iou,
        "fusion_min_match_score": args.fusion_min_match_score,
        "fusion_merge_containment_threshold": args.fusion_merge_containment_threshold,
        "sow_mask_policy": "single_instance_largest_connected_component",
        "min_polygon_area": args.min_polygon_area,
        "epsilon_ratio": args.epsilon_ratio,
        "max_polygon_points": args.max_polygon_points,
        "max_polygons_per_mask": args.max_polygons_per_mask,
        "anylabeling_version": args.anylabeling_version,
    }
    if args.tracking_seed == "anchor-json":
        payload["anchor_annotation"] = path_fingerprint(anchor_annotation_path(prepared))
    return payload


def validate_args(args: argparse.Namespace) -> None:
    args.device = str(args.device).strip().lower()
    if args.track_fps <= 0:
        raise ValueError("--track-fps must be positive")
    if args.max_clips < 0 or args.smoke_radius_frames < 0:
        raise ValueError("--max-clips and --smoke-radius-frames must be non-negative")
    if not 0.0 <= args.sow_confidence_threshold <= 1.0:
        raise ValueError("--sow-confidence-threshold must be in [0, 1]")
    if not 0.0 <= args.piglet_confidence_threshold <= 1.0:
        raise ValueError("--piglet-confidence-threshold must be in [0, 1]")
    if not 0.0 <= args.mask_threshold <= 1.0:
        raise ValueError("--mask-threshold must be in [0, 1]")
    if not 0.0 <= args.min_presence_prob <= 1.0:
        raise ValueError("--min-presence-prob must be in [0, 1]")
    for name in (
        "fusion_min_mask_iou",
        "fusion_min_match_score",
        "fusion_merge_containment_threshold",
    ):
        if not 0.0 <= float(getattr(args, name)) <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1]")
    if args.min_seed_mask_pixels <= 0 or args.min_track_mask_pixels <= 0:
        raise ValueError("minimum mask pixel counts must be positive")
    if args.max_sow_instances != MAX_SOW_INSTANCES:
        raise ValueError(f"--max-sow-instances must be {MAX_SOW_INSTANCES}")
    if args.max_seed_instances < 0 or args.max_polygons_per_mask < 0:
        raise ValueError("instance/polygon limits must be non-negative")
    unit_interval_args = {
        "--dedupe-iou-threshold": args.dedupe_iou_threshold,
        "--containment-threshold": args.containment_threshold,
        "--child-max-area-ratio": args.child_max_area_ratio,
    }
    for name, value in unit_interval_args.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    if args.min_polygon_area <= 0 or args.epsilon_ratio <= 0 or args.max_polygon_points < 3:
        raise ValueError("polygon area/epsilon must be positive and --max-polygon-points must be >= 3")
    if not 1 <= args.jpeg_quality <= 31:
        raise ValueError("--jpeg-quality must be between 1 and 31")
    if args.frames_only and args.seed_only:
        raise ValueError("--frames-only and --seed-only are mutually exclusive")
    if args.overwrite and args.force_retrack:
        raise ValueError("--overwrite and --force-retrack are mutually exclusive")
    if re.fullmatch(r"cuda(?::\d+)?", args.device) is None:
        raise ValueError(
            "--device must be cuda or cuda:<index>; the local official SAM3 tracker uses CUDA-only calls"
        )
    for name in ("sow_prompt", "piglet_prompt", "sow_label", "piglet_label"):
        if not str(getattr(args, name)).strip():
            raise ValueError(f"--{name.replace('_', '-')} cannot be empty")
    if not args.skip_sow and args.sow_label == args.piglet_label:
        raise ValueError("--sow-label and --piglet-label must differ")


def requested_seed_models(args: argparse.Namespace) -> list[SeedModelSpec]:
    models: list[SeedModelSpec] = []
    if not args.skip_sow:
        models.append(
            SeedModelSpec(
                label=args.sow_label,
                prompt=args.sow_prompt,
                checkpoint=args.sow_checkpoint,
                confidence_threshold=args.sow_confidence_threshold,
                max_instances=MAX_SOW_INSTANCES,
            )
        )
    models.append(
        SeedModelSpec(
            label=args.piglet_label,
            prompt=args.piglet_prompt,
            checkpoint=args.piglet_checkpoint,
            confidence_threshold=args.piglet_confidence_threshold,
            max_instances=args.max_seed_instances,
        )
    )
    return models


def activate_cuda_device(device_name: str) -> None:
    import torch

    device = torch.device(device_name)
    torch.cuda.set_device(0 if device.index is None else device.index)


def read_selection_manifest(path: Path) -> list[ClipSpec]:
    required = {
        "clip_id",
        "output_video",
        "source_video",
        "anchor_image",
        "clip_start_sec",
        "clip_duration_sec",
        "anchor_source_sec",
        "anchor_in_clip_sec",
    }
    rows: list[ClipSpec] = []
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"selection manifest is missing fields: {sorted(missing)}")
        for raw in reader:
            clip_id = str(raw["clip_id"]).strip()
            if not SAFE_CLIP_ID_RE.fullmatch(clip_id) or clip_id in {".", ".."}:
                raise ValueError(
                    f"unsafe clip_id {clip_id!r}; expected one filename component containing letters, digits, ._-"
                )
            anchor_json_value = str(raw.get("anchor_json") or "").strip()
            rows.append(
                ClipSpec(
                    clip_id=clip_id,
                    output_video=Path(raw["output_video"]),
                    source_video=Path(raw["source_video"]),
                    anchor_image=Path(raw["anchor_image"]),
                    anchor_json=Path(anchor_json_value) if anchor_json_value else None,
                    clip_start_sec=float(raw["clip_start_sec"]),
                    clip_duration_sec=float(raw["clip_duration_sec"]),
                    anchor_source_sec=float(raw["anchor_source_sec"]),
                    anchor_in_clip_sec=float(raw["anchor_in_clip_sec"]),
                )
            )
    if not rows:
        raise ValueError(f"selection manifest has no rows: {path}")
    if len({row.clip_id for row in rows}) != len(rows):
        raise ValueError("selection manifest contains duplicate clip_id values")
    return rows


def executable_path(name: str, requested: Path | None) -> Path:
    candidates: list[Path] = []
    if requested is not None:
        candidates.append(requested)
    discovered = shutil.which(name)
    if discovered:
        candidates.append(Path(discovered))
    for candidate in candidates:
        if candidate.expanduser().is_file():
            return candidate.expanduser().resolve()
    raise FileNotFoundError(f"Could not find {name}; pass --{name} explicitly")


def numeric_frame_paths(frames_dir: Path) -> tuple[Path, ...]:
    indexed: list[tuple[int, Path]] = []
    if not frames_dir.is_dir():
        return ()
    for path in frames_dir.iterdir():
        if not path.is_file():
            continue
        match = NUMERIC_JPEG_RE.match(path.name)
        if match is not None:
            indexed.append((int(match.group("index")), path))
    indexed.sort()
    indexes = [item[0] for item in indexed]
    if indexes and indexes != list(range(len(indexes))):
        raise ValueError(f"numeric JPEG indexes are not contiguous in {frames_dir}")
    return tuple(path for _, path in indexed)


def remove_generated_annotations(frames_dir: Path) -> None:
    """Remove only files generated by this script inside a verified frames directory."""
    if not frames_dir.is_dir():
        return
    for path in frames_dir.iterdir():
        if path.is_file() and (
            NUMERIC_JPEG_RE.match(path.name)
            or (path.suffix.lower() == ".json" and path.stem.isdigit())
        ):
            path.unlink()


def frame_sampling_window(spec: ClipSpec, track_fps: float, smoke_radius_frames: int) -> tuple[float, float]:
    if smoke_radius_frames <= 0:
        return 0.0, spec.clip_duration_sec
    start = max(0.0, spec.anchor_in_clip_sec - smoke_radius_frames / track_fps)
    desired_count = 2 * smoke_radius_frames + 1
    duration = desired_count / track_fps
    if start + duration > spec.clip_duration_sec:
        start = max(0.0, spec.clip_duration_sec - duration)
    return start, min(duration, spec.clip_duration_sec - start)


def build_extract_frames_command(
    ffmpeg: Path,
    video_path: Path,
    frames_dir: Path,
    sample_start_sec: float,
    sample_duration_sec: float,
    track_fps: float,
    jpeg_quality: int,
) -> list[str]:
    return [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{sample_start_sec:.6f}",
        "-i",
        str(video_path),
        "-t",
        f"{sample_duration_sec:.6f}",
        "-vf",
        f"fps={track_fps:.12g}",
        "-q:v",
        str(jpeg_quality),
        "-start_number",
        "0",
        str(frames_dir / "%06d.jpg"),
    ]


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def finish_failure_report(output_root: Path, failures: Sequence[tuple[str, str]]) -> int:
    failures_path = output_root / "failures.csv"
    if failures:
        write_csv(
            failures_path,
            ("clip_id", "error"),
            ({"clip_id": clip_id, "error": error} for clip_id, error in failures),
        )
        print(f"Completed with {len(failures)} failure(s); see {failures_path}")
        return 1
    if failures_path.is_file():
        failures_path.unlink()
    return 0


def prepare_clip_frames(spec: ClipSpec, output_root: Path, ffmpeg: Path, args: argparse.Namespace) -> PreparedClip:
    clip_dir = output_root / spec.clip_id
    frames_dir = clip_dir / "frames"
    sample_start, sample_duration = frame_sampling_window(spec, args.track_fps, args.smoke_radius_frames)
    metadata_path = clip_dir / "clip_metadata.json"
    expected_configuration = frame_configuration(spec, args, sample_start, sample_duration)

    if args.overwrite:
        remove_generated_annotations(frames_dir)
        for filename in (
            "seed_masks.npz",
            "seed_metadata.json",
            "tracks.jsonl",
            "tracking_complete.json",
        ):
            path = clip_dir / filename
            if path.is_file():
                path.unlink()
        for mask_dir_name in ("id_masks", "sow_masks", "piglet_id_masks"):
            mask_dir = clip_dir / mask_dir_name
            if mask_dir.is_dir():
                for path in mask_dir.glob("*.png"):
                    if path.stem.isdigit():
                        path.unlink()
        instance_mask_dir = clip_dir / "instance_masks"
        if instance_mask_dir.is_dir():
            shutil.rmtree(instance_mask_dir)
        finetuned_cache_dir = clip_dir / "finetuned_frame_cache"
        if finetuned_cache_dir.is_dir():
            shutil.rmtree(finetuned_cache_dir)
    existing = numeric_frame_paths(frames_dir)
    if existing and not args.overwrite:
        if not metadata_path.is_file():
            raise FileExistsError(f"frames exist without clip metadata; use --overwrite: {frames_dir}")
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        if metadata.get("frame_configuration") != expected_configuration:
            raise ValueError(
                f"existing frame configuration differs from this run; use --overwrite: {frames_dir}"
            )
        frame_paths = existing
        anchor_index = int(metadata["anchor_frame_index"])
        if int(metadata.get("frame_count", -1)) != len(frame_paths):
            raise ValueError(f"existing frame count differs from metadata; use --overwrite: {frames_dir}")
        if not 0 <= anchor_index < len(frame_paths):
            raise ValueError(f"invalid anchor frame index in {metadata_path}")
        return PreparedClip(
            spec,
            clip_dir,
            frames_dir,
            frame_paths,
            float(metadata["sample_start_clip_sec"]),
            anchor_index,
            args.track_fps,
        )

    if not spec.output_video.is_file():
        raise FileNotFoundError(f"sampled clip does not exist: {spec.output_video}")
    frames_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        build_extract_frames_command(
            ffmpeg,
            spec.output_video,
            frames_dir,
            sample_start,
            sample_duration,
            args.track_fps,
            args.jpeg_quality,
        ),
        check=True,
    )
    frame_paths = numeric_frame_paths(frames_dir)
    if not frame_paths:
        raise RuntimeError(f"FFmpeg extracted no frames from {spec.output_video}")
    anchor_index = int(round((spec.anchor_in_clip_sec - sample_start) * args.track_fps))
    anchor_index = min(len(frame_paths) - 1, max(0, anchor_index))
    prepared = PreparedClip(spec, clip_dir, frames_dir, frame_paths, sample_start, anchor_index, args.track_fps)
    frame_rows = [
        {
            "frame_index": index,
            "image_path": str(path.resolve()),
            "clip_timestamp_sec": f"{prepared.clip_timestamp_sec(index):.6f}",
            "source_timestamp_sec": f"{prepared.source_timestamp_sec(index):.6f}",
            "is_anchor": int(index == anchor_index),
        }
        for index, path in enumerate(frame_paths)
    ]
    write_csv(
        clip_dir / "frames_manifest.csv",
        ("frame_index", "image_path", "clip_timestamp_sec", "source_timestamp_sec", "is_anchor"),
        frame_rows,
    )
    metadata = {
        "clip_id": spec.clip_id,
        "input_clip": str(spec.output_video.resolve()),
        "source_video": str(spec.source_video.resolve()),
        "anchor_image": str(spec.anchor_image.resolve()),
        "anchor_json": str(spec.anchor_json.resolve()) if spec.anchor_json else None,
        "clip_start_sec": spec.clip_start_sec,
        "clip_duration_sec": spec.clip_duration_sec,
        "anchor_source_sec": spec.anchor_source_sec,
        "anchor_in_clip_sec": spec.anchor_in_clip_sec,
        "track_fps": args.track_fps,
        "sample_start_clip_sec": sample_start,
        "sample_duration_sec": sample_duration,
        "frame_count": len(frame_paths),
        "anchor_frame_index": anchor_index,
        "smoke_radius_frames": args.smoke_radius_frames,
        "frame_configuration": expected_configuration,
        "status": "frames_prepared",
    }
    clip_dir.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return prepared


def tensor_to_numpy(value: Any) -> np.ndarray:
    import torch

    if torch.is_tensor(value):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def normalize_prediction_masks(value: Any, mask_threshold: float) -> np.ndarray:
    masks = tensor_to_numpy(value)
    if masks.ndim == 2:
        masks = masks[None, ...]
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim != 3:
        raise ValueError(f"expected masks [N,H,W] or [N,1,H,W], got {masks.shape}")
    if masks.dtype == np.bool_:
        return masks
    masks = masks.astype(np.float32)
    finite = masks[np.isfinite(masks)]
    if finite.size and (float(finite.min()) < 0.0 or float(finite.max()) > 1.0):
        masks = 1.0 / (1.0 + np.exp(-np.clip(masks, -32.0, 32.0)))
    return masks > mask_threshold


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = int(np.logical_and(first, second).sum())
    if intersection == 0:
        return 0.0
    union = int(np.logical_or(first, second).sum())
    return float(intersection / union) if union else 0.0


def containment_fraction(smaller: np.ndarray, larger: np.ndarray) -> float:
    area = int(smaller.sum())
    return float(np.logical_and(smaller, larger).sum() / area) if area else 0.0


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """Keep exactly the largest 8-connected foreground component of a binary mask."""
    binary = np.asarray(mask, dtype=bool)
    if not binary.any():
        return np.zeros_like(binary, dtype=bool)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8),
        connectivity=8,
    )
    if component_count <= 1:
        return np.zeros_like(binary, dtype=bool)
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest_label


def binary_mask_box(mask: np.ndarray) -> tuple[float, float, float, float]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return (0.0, 0.0, 0.0, 0.0)
    return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))


def constrain_sow_seed_instances(
    instances: Sequence[SeedInstance],
    sow_label: str,
    min_pixels: int,
) -> list[SeedInstance]:
    """Keep at most one sow seed, chosen by largest connected area before score."""
    non_sows = [instance for instance in instances if instance.label != sow_label]
    sow_candidates: list[SeedInstance] = []
    for instance in instances:
        if instance.label != sow_label:
            continue
        mask = largest_connected_component(instance.mask)
        if int(mask.sum()) < min_pixels:
            continue
        sow_candidates.append(
            SeedInstance(
                instance.track_id,
                instance.label,
                mask,
                instance.score,
                binary_mask_box(mask),
            )
        )
    if sow_candidates:
        best_sow = max(
            sow_candidates,
            key=lambda instance: (int(instance.mask.sum()), instance.score),
        )
        non_sows.append(best_sow)
    return sorted(non_sows, key=lambda instance: instance.track_id)


def constrain_sow_tracked_instances(
    instances: Sequence[TrackedInstance],
    args: argparse.Namespace,
) -> list[TrackedInstance]:
    """Keep one sow track and only its largest connected mask component on a frame."""
    non_sows = [instance for instance in instances if instance.label != args.sow_label]
    sow_candidates: list[TrackedInstance] = []
    for instance in instances:
        if instance.label != args.sow_label:
            continue
        mask = largest_connected_component(instance.mask)
        if int(mask.sum()) < args.min_track_mask_pixels:
            mask = np.zeros_like(mask, dtype=bool)
        sow_candidates.append(
            TrackedInstance(
                instance.track_id,
                instance.label,
                mask,
                instance.object_score_logit,
                instance.object_probability,
                instance.mask_source,
                instance.finetuned_score,
                instance.fusion_match_score,
                instance.fusion_mask_iou,
            )
        )
    if sow_candidates:
        best_sow = max(
            sow_candidates,
            key=lambda instance: (
                int(instance.mask.sum()),
                instance.finetuned_score
                if instance.finetuned_score is not None
                else (instance.object_probability or 0.0),
            ),
        )
        non_sows.append(best_sow)
    return sorted(non_sows, key=lambda instance: instance.track_id)


def deduplicate_seed_candidates(
    masks: np.ndarray,
    scores: np.ndarray,
    boxes: np.ndarray,
    args: argparse.Namespace,
    max_instances: int,
) -> list[tuple[np.ndarray, float, tuple[float, float, float, float]]]:
    raw: list[tuple[np.ndarray, float, tuple[float, float, float, float]]] = []
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    boxes = np.asarray(boxes, dtype=np.float32)
    for index, mask in enumerate(masks):
        area = int(mask.sum())
        score = float(scores[index]) if index < len(scores) else 0.0
        if area < args.min_seed_mask_pixels:
            continue
        box = boxes[index] if index < len(boxes) else np.array([0, 0, 0, 0], dtype=np.float32)
        raw.append((mask.astype(bool), score, tuple(float(v) for v in box[:4])))
    raw.sort(key=lambda item: (item[1], int(item[0].sum())), reverse=True)

    duplicate_removed: list[tuple[np.ndarray, float, tuple[float, float, float, float]]] = []
    for candidate in raw:
        if any(mask_iou(candidate[0], kept[0]) >= args.dedupe_iou_threshold for kept in duplicate_removed):
            continue
        duplicate_removed.append(candidate)

    # If a large mask contains multiple independent smaller masks, treat the large one as a merge error.
    removed: set[int] = set()
    for large_index, large in enumerate(duplicate_removed):
        large_area = int(large[0].sum())
        child_indexes: list[int] = []
        for small_index, small in enumerate(duplicate_removed):
            if small_index == large_index:
                continue
            small_area = int(small[0].sum())
            if small_area >= large_area or small_area / max(large_area, 1) > args.child_max_area_ratio:
                continue
            if containment_fraction(small[0], large[0]) >= args.containment_threshold:
                child_indexes.append(small_index)
        independent = all(
            mask_iou(duplicate_removed[a][0], duplicate_removed[b][0]) <= 0.25
            for offset, a in enumerate(child_indexes)
            for b in child_indexes[offset + 1 :]
        )
        if len(child_indexes) >= 2 and independent:
            removed.add(large_index)

    kept = [item for index, item in enumerate(duplicate_removed) if index not in removed]
    # In exclusive/cross-class modes, same-class detections must have one pixel owner.
    # ``all`` deliberately keeps the independent finetuned instance masks unchanged.
    if kept and args.overlap_policy != "all":
        owner = np.full(kept[0][0].shape, -1, dtype=np.int16)
        best = np.full(kept[0][0].shape, -np.inf, dtype=np.float32)
        for index, (mask, score, _) in enumerate(kept):
            update = mask & (score > best)
            owner[update] = index
            best[update] = score
        non_overlapping = []
        for index, (_, score, box) in enumerate(kept):
            mask = owner == index
            if int(mask.sum()) >= args.min_seed_mask_pixels:
                non_overlapping.append((mask, score, box))
        kept = non_overlapping

    if max_instances > 0:
        kept = kept[:max_instances]
    return kept


def deterministic_seed_instances(
    candidates: Sequence[tuple[np.ndarray, float, tuple[float, float, float, float]]],
    label: str,
    start_track_id: int = 1,
) -> list[SeedInstance]:
    sortable = []
    for mask, score, box in candidates:
        ys, xs = np.nonzero(mask)
        if not len(xs):
            continue
        sortable.append((float(ys.mean()), float(xs.mean()), mask, score, box))
    sortable.sort(key=lambda item: (item[0], item[1]))
    return [
        SeedInstance(track_id=index, label=label, mask=item[2], score=item[3], box_xyxy=item[4])
        for index, item in enumerate(sortable, start=start_track_id)
    ]


def load_seed_processor(args: argparse.Namespace, model_spec: SeedModelSpec):
    activate_cuda_device(args.device)
    sam3_root = args.sam3_root.resolve()
    scripts_dir = sam3_root / "scripts"
    for path in (sam3_root, scripts_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3_predictions_to_labelme import load_finetuned_model

    model = load_finetuned_model(model_spec.checkpoint, args.bpe_path, args.device)
    processor = Sam3Processor(
        model,
        resolution=1008,
        device=args.device,
        confidence_threshold=model_spec.confidence_threshold,
    )
    return model, processor


def infer_seed_instances(
    processor,
    image_path: Path,
    model_spec: SeedModelSpec,
    args: argparse.Namespace,
) -> list[SeedInstance]:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    state = processor.set_image(image)
    output = processor.set_text_prompt(prompt=model_spec.prompt, state=state)
    masks = normalize_prediction_masks(output.get("masks_logits", output["masks"]), args.mask_threshold)
    scores = tensor_to_numpy(output["scores"])
    boxes = tensor_to_numpy(output["boxes"])
    if model_spec.label == args.sow_label:
        flat_scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        raw_sows = [
            SeedInstance(
                track_id=index + 1,
                label=model_spec.label,
                mask=mask.astype(bool, copy=False),
                score=float(flat_scores[index]) if index < len(flat_scores) else 0.0,
                box_xyxy=(0.0, 0.0, 0.0, 0.0),
            )
            for index, mask in enumerate(masks)
        ]
        return constrain_sow_seed_instances(
            raw_sows,
            args.sow_label,
            args.min_seed_mask_pixels,
        )
    candidates = deduplicate_seed_candidates(masks, scores, boxes, args, model_spec.max_instances)
    return deterministic_seed_instances(candidates, model_spec.label)


def finetuned_cache_label_dir(prepared: PreparedClip, label: str) -> Path:
    safe_label = re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("._")
    if not safe_label:
        raise ValueError(f"label cannot be converted to a safe cache directory name: {label!r}")
    return prepared.clip_dir / "finetuned_frame_cache" / safe_label


def finetuned_cache_configuration(
    prepared: PreparedClip,
    model_spec: SeedModelSpec,
    args: argparse.Namespace,
) -> dict[str, Any]:
    payload = {
        "schema_version": 3 if model_spec.label == args.sow_label else 2,
        "clip_id": prepared.spec.clip_id,
        "frame_count": len(prepared.frame_paths),
        "clip_metadata": path_fingerprint(prepared.clip_dir / "clip_metadata.json"),
        "model": {
            "label": model_spec.label,
            "prompt": model_spec.prompt,
            "checkpoint": path_fingerprint(model_spec.checkpoint),
            "confidence_threshold": model_spec.confidence_threshold,
            "max_instances": model_spec.max_instances,
        },
        "bpe_path": path_fingerprint(args.bpe_path),
        "mask_threshold": args.mask_threshold,
        "min_seed_mask_pixels": args.min_seed_mask_pixels,
        "dedupe_iou_threshold": args.dedupe_iou_threshold,
        "containment_threshold": args.containment_threshold,
        "child_max_area_ratio": args.child_max_area_ratio,
        "overlap_policy": args.overlap_policy,
    }
    if model_spec.label == args.sow_label:
        payload["sow_mask_policy"] = "single_instance_largest_connected_component"
    return payload


def finetuned_frame_cache_path(prepared: PreparedClip, label: str, frame_index: int) -> Path:
    return finetuned_cache_label_dir(prepared, label) / f"{frame_index:06d}.npz"


def finetuned_cache_is_reusable(
    prepared: PreparedClip,
    model_spec: SeedModelSpec,
    args: argparse.Namespace,
) -> bool:
    cache_dir = finetuned_cache_label_dir(prepared, model_spec.label)
    completion_path = cache_dir / "complete.json"
    if not completion_path.is_file():
        return False
    with completion_path.open("r", encoding="utf-8") as file:
        completion = json.load(file)
    if completion.get("configuration") != finetuned_cache_configuration(prepared, model_spec, args):
        return False
    return all(
        finetuned_frame_cache_path(prepared, model_spec.label, frame_index).is_file()
        for frame_index in range(len(prepared.frame_paths))
    )


def save_finetuned_frame_cache(
    prepared: PreparedClip,
    frame_index: int,
    model_spec: SeedModelSpec,
    instances: Sequence[SeedInstance],
) -> None:
    cache_path = finetuned_frame_cache_path(prepared, model_spec.label, frame_index)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if instances:
        masks = np.stack([instance.mask.astype(bool, copy=False) for instance in instances])
        scores = np.asarray([instance.score for instance in instances], dtype=np.float32)
        boxes = np.asarray([instance.box_xyxy for instance in instances], dtype=np.float32)
    else:
        with Image.open(prepared.frame_paths[frame_index]) as image:
            width, height = image.size
        masks = np.zeros((0, height, width), dtype=bool)
        scores = np.zeros((0,), dtype=np.float32)
        boxes = np.zeros((0, 4), dtype=np.float32)
    temporary_path = cache_path.with_suffix(".tmp")
    with temporary_path.open("wb") as file:
        np.savez_compressed(file, masks=masks, scores=scores, boxes=boxes)
    os.replace(temporary_path, cache_path)


def load_finetuned_frame_cache(
    prepared: PreparedClip,
    frame_index: int,
    model_spec: SeedModelSpec,
) -> list[SeedInstance]:
    cache_path = finetuned_frame_cache_path(prepared, model_spec.label, frame_index)
    if not cache_path.is_file():
        raise FileNotFoundError(f"finetuned frame cache is missing: {cache_path}")
    with np.load(cache_path, allow_pickle=False) as archive:
        masks = np.asarray(archive["masks"], dtype=bool)
        scores = np.asarray(archive["scores"], dtype=np.float32).reshape(-1)
        boxes = np.asarray(archive["boxes"], dtype=np.float32).reshape(-1, 4)
    if masks.ndim != 3 or len(masks) != len(scores) or len(masks) != len(boxes):
        raise ValueError(f"invalid finetuned frame cache arrays: {cache_path}")
    return [
        SeedInstance(
            track_id=index,
            label=model_spec.label,
            mask=masks[index],
            score=float(scores[index]),
            box_xyxy=tuple(float(value) for value in boxes[index]),
        )
        for index in range(len(masks))
    ]


def prepare_finetuned_frame_caches(
    prepared_clips: Sequence[PreparedClip],
    model_specs: Sequence[SeedModelSpec],
    args: argparse.Namespace,
    failures: list[tuple[str, str]],
) -> list[PreparedClip]:
    """Run each image model once over all required clips and cache variable-count masks per frame."""
    import torch

    failed_clip_ids: set[str] = set()
    for model_spec in model_specs:
        targets = [
            prepared
            for prepared in prepared_clips
            if prepared.spec.clip_id not in failed_clip_ids
            and not finetuned_cache_is_reusable(prepared, model_spec, args)
        ]
        if not targets:
            print(f"[finetuned-cache-reuse] {model_spec.label}: all frames")
            continue
        print(f"[load-frame-model] {model_spec.label}: {model_spec.checkpoint}")
        model = None
        processor = None
        try:
            model, processor = load_seed_processor(args, model_spec)
            with torch.inference_mode():
                for prepared in targets:
                    cache_dir = finetuned_cache_label_dir(prepared, model_spec.label)
                    if cache_dir.is_dir():
                        shutil.rmtree(cache_dir)
                    try:
                        for frame_index, frame_path in enumerate(prepared.frame_paths):
                            instances = infer_seed_instances(processor, frame_path, model_spec, args)
                            save_finetuned_frame_cache(
                                prepared,
                                frame_index,
                                model_spec,
                                instances,
                            )
                        completion = {
                            "configuration": finetuned_cache_configuration(prepared, model_spec, args),
                            "frame_count": len(prepared.frame_paths),
                        }
                        with (cache_dir / "complete.json").open("w", encoding="utf-8") as file:
                            json.dump(completion, file, ensure_ascii=False, indent=2)
                            file.write("\n")
                        print(
                            f"[finetuned-cache-{model_spec.label}] {prepared.spec.clip_id}: "
                            f"{len(prepared.frame_paths)} frames"
                        )
                    except Exception as exc:
                        failed_clip_ids.add(prepared.spec.clip_id)
                        failures.append(
                            (
                                prepared.spec.clip_id,
                                f"{model_spec.label} per-frame finetuned inference: {exc}",
                            )
                        )
                        print(f"[failed] {prepared.spec.clip_id}: {exc}", file=sys.stderr)
                        if args.fail_on_error:
                            raise
        finally:
            if processor is not None:
                del processor
            if model is not None:
                del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return [prepared for prepared in prepared_clips if prepared.spec.clip_id not in failed_clip_ids]


def finalize_seed_instances(
    seeds_by_label: dict[str, list[SeedInstance]],
    model_specs: Sequence[SeedModelSpec],
    args: argparse.Namespace,
) -> list[SeedInstance]:
    """Assign clip-wide IDs and optionally make sow/piglet seed masks mutually exclusive."""
    piglets = list(seeds_by_label.get(args.piglet_label, []))
    if not piglets:
        raise RuntimeError("finetuned SAM3 found no usable piglet seed masks")

    ordered: list[SeedInstance] = []
    if not args.skip_sow:
        sows = constrain_sow_seed_instances(
            seeds_by_label.get(args.sow_label, []),
            args.sow_label,
            args.min_seed_mask_pixels,
        )
        if not sows:
            raise RuntimeError("finetuned SAM3 found no usable sow seed mask")
        sows.sort(key=lambda seed: (int(seed.mask.sum()), seed.score), reverse=True)
        sows = sows[:MAX_SOW_INSTANCES]
        for seed in sows:
            sow_mask = seed.mask
            if args.overlap_policy == "exclusive":
                piglet_union = np.logical_or.reduce([piglet.mask for piglet in piglets])
                sow_mask = sow_mask & ~piglet_union
            sow_mask = largest_connected_component(sow_mask)
            if int(sow_mask.sum()) < args.min_seed_mask_pixels:
                raise RuntimeError("sow seed became empty after resolving sow/piglet overlap")
            ordered.append(
                SeedInstance(0, seed.label, sow_mask, seed.score, binary_mask_box(sow_mask))
            )

    # Model specifications define the class order; currently sow first, then piglets.
    for model_spec in model_specs:
        if model_spec.label == args.piglet_label:
            ordered.extend(piglets)
    return [
        SeedInstance(index, seed.label, seed.mask, seed.score, seed.box_xyxy)
        for index, seed in enumerate(ordered, start=1)
    ]


def seed_archive_paths(prepared: PreparedClip) -> tuple[Path, Path]:
    return prepared.clip_dir / "seed_masks.npz", prepared.clip_dir / "seed_metadata.json"


def anchor_annotation_path(prepared: PreparedClip) -> Path:
    return prepared.frame_paths[prepared.anchor_frame_index].with_suffix(".json")


def seed_archive_is_reusable(prepared: PreparedClip, args: argparse.Namespace) -> bool:
    archive_path, metadata_path = seed_archive_paths(prepared)
    if not archive_path.exists() and not metadata_path.exists():
        return False
    if not archive_path.is_file() or not metadata_path.is_file():
        raise ValueError(
            f"incomplete seed cache for {prepared.spec.clip_id}; use --overwrite to regenerate it"
        )
    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    if metadata.get("seed_configuration") != seed_configuration(args, prepared):
        # Seed masks are derived data and can be regenerated without deleting an
        # edited anchor JSON.  This is required when overlap policy changes.
        return False
    return True


def save_seed_instances(prepared: PreparedClip, seeds: Sequence[SeedInstance], args: argparse.Namespace) -> None:
    sow_count = sum(seed.label == args.sow_label for seed in seeds)
    if sow_count > MAX_SOW_INSTANCES:
        raise ValueError(f"refusing to cache {sow_count} sow instances; maximum is {MAX_SOW_INSTANCES}")
    archive_path, metadata_path = seed_archive_paths(prepared)
    masks = np.stack([seed.mask.astype(np.uint8) for seed in seeds], axis=0) if seeds else np.zeros((0, 0, 0), dtype=np.uint8)
    np.savez_compressed(
        archive_path,
        masks=masks,
        track_ids=np.asarray([seed.track_id for seed in seeds], dtype=np.int32),
        labels=np.asarray([seed.label for seed in seeds], dtype=np.str_),
        scores=np.asarray([seed.score for seed in seeds], dtype=np.float32),
        boxes=np.asarray([seed.box_xyxy for seed in seeds], dtype=np.float32).reshape(-1, 4),
        anchor_frame_index=np.asarray([prepared.anchor_frame_index], dtype=np.int32),
    )
    metadata = {
        "clip_id": prepared.spec.clip_id,
        "anchor_frame_index": prepared.anchor_frame_index,
        "anchor_frame": prepared.frame_paths[prepared.anchor_frame_index].name,
        "models": [
            {
                "label": model.label,
                "prompt": model.prompt,
                "checkpoint": str(model.checkpoint.resolve()),
                "confidence_threshold": model.confidence_threshold,
                "max_instances": model.max_instances,
            }
            for model in requested_seed_models(args)
        ],
        "seed_configuration": seed_configuration(args, prepared),
        "mask_threshold": args.mask_threshold,
        "seed_count": len(seeds),
        "seeds": [
            {
                "track_id": seed.track_id,
                "label": seed.label,
                "score": seed.score,
                "area_px": int(seed.mask.sum()),
                "box_xyxy": list(seed.box_xyxy),
            }
            for seed in seeds
        ],
    }
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
        file.write("\n")


def load_seed_instances(prepared: PreparedClip, args: argparse.Namespace) -> list[SeedInstance]:
    archive_path, _ = seed_archive_paths(prepared)
    with np.load(archive_path) as archive:
        anchor_index = int(np.asarray(archive["anchor_frame_index"]).reshape(-1)[0])
        if anchor_index != prepared.anchor_frame_index:
            raise ValueError(f"seed anchor index mismatch for {prepared.spec.clip_id}")
        masks = np.asarray(archive["masks"]).astype(bool)
        track_ids = np.asarray(archive["track_ids"]).astype(int)
        labels = np.asarray(archive["labels"]).astype(str)
        scores = np.asarray(archive["scores"]).astype(float)
        boxes = np.asarray(archive["boxes"]).astype(float)
    count = len(track_ids)
    if not (len(labels) == len(masks) == len(scores) == len(boxes) == count):
        raise ValueError(f"seed archive arrays have inconsistent lengths for {prepared.spec.clip_id}")
    if count == 0 or masks.ndim != 3:
        raise ValueError(f"seed archive contains no valid masks for {prepared.spec.clip_id}")
    with Image.open(prepared.frame_paths[prepared.anchor_frame_index]) as image:
        expected_shape = (image.height, image.width)
    if tuple(masks.shape[1:]) != expected_shape:
        raise ValueError(
            f"seed mask size {tuple(masks.shape[1:])} does not match anchor frame {expected_shape}"
        )
    if len(set(int(value) for value in track_ids)) != count:
        raise ValueError(f"seed archive has duplicate track IDs for {prepared.spec.clip_id}")
    seeds = [
        SeedInstance(
            int(track_ids[i]),
            str(labels[i]),
            masks[i],
            float(scores[i]),
            tuple(float(v) for v in boxes[i]),
        )
        for i in range(len(track_ids))
    ]
    seeds = constrain_sow_seed_instances(
        seeds,
        args.sow_label,
        args.min_seed_mask_pixels,
    )
    if not args.skip_sow and not any(seed.label == args.sow_label for seed in seeds):
        raise ValueError("seed archive contains no usable sow connected component")
    return seeds


def load_anchor_json_seed_instances(
    prepared: PreparedClip,
    args: argparse.Namespace,
) -> list[SeedInstance]:
    annotation_path = validate_file(anchor_annotation_path(prepared), "anchor-frame annotation")
    with annotation_path.open("r", encoding="utf-8-sig") as file:
        document = json.load(file)
    with Image.open(prepared.frame_paths[prepared.anchor_frame_index]) as image:
        width, height = image.size
    if int(document.get("imageWidth", -1)) != width or int(document.get("imageHeight", -1)) != height:
        raise ValueError(
            f"anchor JSON size does not match its frame ({width}x{height}): {annotation_path}"
        )

    allowed_labels = {args.piglet_label}
    if not args.skip_sow:
        allowed_labels.add(args.sow_label)
    grouped_masks: dict[tuple[str, int], np.ndarray] = {}
    grouped_scores: dict[tuple[str, int], float] = {}
    id_to_label: dict[int, str] = {}
    for shape_index, shape in enumerate(document.get("shapes", [])):
        label = str(shape.get("label") or "").strip()
        if label not in allowed_labels:
            raise ValueError(
                f"unsupported label {label!r} in anchor JSON shape {shape_index}; "
                f"expected {sorted(allowed_labels)}"
            )
        raw_group_id = shape.get("group_id")
        if raw_group_id is None:
            raise ValueError(
                f"anchor JSON shape {shape_index} has no group_id; assign one stable ID in X-AnyLabeling"
            )
        try:
            group_id = int(raw_group_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid group_id {raw_group_id!r} in anchor JSON") from exc
        if not 1 <= group_id <= np.iinfo(np.uint16).max:
            raise ValueError(f"group_id must be in [1, 65535], got {group_id}")
        previous_label = id_to_label.setdefault(group_id, label)
        if previous_label != label:
            raise ValueError(
                f"group_id {group_id} is shared by labels {previous_label!r} and {label!r}"
            )

        points = np.asarray(shape.get("points", []), dtype=np.float32)
        shape_type = str(shape.get("shape_type") or "polygon").lower()
        if shape_type == "rectangle":
            if points.shape != (2, 2):
                raise ValueError(f"rectangle shape {shape_index} must contain exactly two points")
            x0, y0 = points.min(axis=0)
            x1, y1 = points.max(axis=0)
            points = np.asarray([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
        elif shape_type != "polygon":
            raise ValueError(
                f"unsupported shape_type {shape_type!r} in anchor JSON; use polygon or rectangle"
            )
        if points.ndim != 2 or points.shape[1:] != (2,) or len(points) < 3:
            raise ValueError(f"anchor JSON shape {shape_index} has fewer than three valid points")
        if not np.isfinite(points).all():
            raise ValueError(f"anchor JSON shape {shape_index} contains non-finite coordinates")
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        polygon = np.rint(points).astype(np.int32)
        key = (label, group_id)
        mask = grouped_masks.setdefault(key, np.zeros((height, width), dtype=np.uint8))
        cv2.fillPoly(mask, [polygon], color=1)
        raw_score = shape.get("score")
        score = 1.0 if raw_score in (None, "") else float(raw_score)
        grouped_scores[key] = max(grouped_scores.get(key, 0.0), score)

    seeds: list[SeedInstance] = []
    for (label, group_id), raw_mask in sorted(grouped_masks.items(), key=lambda item: item[0][1]):
        mask = raw_mask.astype(bool)
        if int(mask.sum()) < args.min_seed_mask_pixels:
            raise ValueError(f"anchor JSON group_id {group_id} is smaller than --min-seed-mask-pixels")
        ys, xs = np.nonzero(mask)
        box = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
        seeds.append(SeedInstance(group_id, label, mask, grouped_scores[(label, group_id)], box))

    seeds = constrain_sow_seed_instances(
        seeds,
        args.sow_label,
        args.min_seed_mask_pixels,
    )
    piglets = [seed for seed in seeds if seed.label == args.piglet_label]
    if not piglets:
        raise ValueError("anchor JSON contains no piglet instances")
    if not args.skip_sow:
        sows = [seed for seed in seeds if seed.label == args.sow_label]
        if not sows:
            raise ValueError("anchor JSON contains no sow instance")
        if args.overlap_policy == "exclusive":
            piglet_union = np.logical_or.reduce([seed.mask for seed in piglets])
            seeds = [
                SeedInstance(
                    seed.track_id,
                    seed.label,
                    seed.mask & ~piglet_union if seed.label == args.sow_label else seed.mask,
                    seed.score,
                    seed.box_xyxy,
                )
                for seed in seeds
            ]
            seeds = constrain_sow_seed_instances(
                seeds,
                args.sow_label,
                args.min_seed_mask_pixels,
            )
        for seed in seeds:
            if int(seed.mask.sum()) < args.min_seed_mask_pixels:
                raise ValueError(
                    f"anchor JSON group_id {seed.track_id} became too small after overlap resolution"
                )
    return seeds


def load_tracking_seed_instances(
    prepared: PreparedClip,
    args: argparse.Namespace,
) -> list[SeedInstance]:
    if args.tracking_seed == "anchor-json":
        return load_anchor_json_seed_instances(prepared, args)
    return load_seed_instances(prepared, args)


def load_official_tracker(args: argparse.Namespace):
    import torch

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("The official SAM3 tracker path used here requires a CUDA device")
    activate_cuda_device(args.device)
    sam3_root = args.sam3_root.resolve()
    if str(sam3_root) not in sys.path:
        sys.path.insert(0, str(sam3_root))
    from sam3.model_builder import build_sam3_video_model

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    video_model = build_sam3_video_model(
        checkpoint_path=str(args.video_checkpoint),
        load_from_HF=False,
        bpe_path=str(args.bpe_path),
        strict_state_dict_loading=True,
        # Build and load on CPU first. Moving the full detector to a 12 GB GPU
        # before discarding its unused heads creates an avoidable VRAM peak.
        device="cpu",
    )
    predictor = video_model.tracker
    predictor.backbone = video_model.detector.backbone
    # The tracker only calls forward_image; release the detector's language encoder.
    predictor.backbone.language_backbone = None
    # Keep the official tracker internally exclusive for stable identity matching.
    # In hybrid/finetuned mode, cross-class overlap is restored from the independent
    # per-frame finetuned sow and piglet masks after track-ID association.
    predictor.non_overlap_masks_for_output = True
    del video_model
    gc.collect()
    predictor.to(args.device)
    predictor.eval()
    torch.cuda.empty_cache()
    return predictor


def sigmoid_scalar(value: float) -> float:
    if value >= 0:
        return float(1.0 / (1.0 + math.exp(-value)))
    exp_value = math.exp(value)
    return float(exp_value / (1.0 + exp_value))


def output_masks_from_tracker(
    obj_ids: Sequence[Any],
    video_res_masks: Any,
    obj_scores: Any,
    labels_by_id: dict[int, str],
    source: str,
    args: argparse.Namespace,
) -> list[TrackedInstance]:
    masks = tensor_to_numpy(video_res_masks)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    scores = tensor_to_numpy(obj_scores).reshape(len(obj_ids), -1)
    instances: list[TrackedInstance] = []
    for index, obj_id in enumerate(obj_ids):
        track_id = int(obj_id.item()) if hasattr(obj_id, "item") else int(obj_id)
        logit = float(scores[index, 0]) if scores.size else None
        probability = sigmoid_scalar(logit) if logit is not None else None
        mask = masks[index] > 0.0
        if probability is not None and probability < args.min_presence_prob:
            mask = np.zeros_like(mask, dtype=bool)
        if int(mask.sum()) < args.min_track_mask_pixels:
            mask = np.zeros_like(mask, dtype=bool)
        if track_id not in labels_by_id:
            raise KeyError(f"tracker returned unknown object ID {track_id}")
        instances.append(
            TrackedInstance(track_id, labels_by_id[track_id], mask, logit, probability, source)
        )
    return instances


def iter_propagated_frames(
    predictor,
    prepared: PreparedClip,
    seeds: Sequence[SeedInstance],
    args: argparse.Namespace,
) -> Iterable[tuple[int, list[TrackedInstance], list[FusionDiagnostics]]]:
    import torch

    if not seeds:
        raise RuntimeError(f"no seed piglets detected for {prepared.spec.clip_id}")
    state = predictor.init_state(
        video_path=str(prepared.frames_dir),
        offload_video_to_cpu=True,
        offload_state_to_cpu=args.offload_state_to_cpu,
        async_loading_frames=False,
    )
    try:
        labels_by_id = {seed.track_id: seed.label for seed in seeds}
        # Masks are explicitly disjoint, but class priority and low-to-high score
        # ordering are extra guards because add_new_mask gives later masks priority.
        for seed in sorted(
            seeds,
            key=lambda item: (item.label != args.sow_label, item.score),
        ):
            predictor.add_new_mask(
                state,
                frame_idx=prepared.anchor_frame_index,
                obj_id=seed.track_id,
                mask=torch.from_numpy(seed.mask),
            )
        predictor.propagate_in_video_preflight(state)

        anchor_source = (
            "init_anchor_json" if args.tracking_seed == "anchor-json" else "init_finetuned_sam3"
        )
        yield (
            prepared.anchor_frame_index,
            [
                TrackedInstance(
                    seed.track_id,
                    seed.label,
                    seed.mask,
                    None,
                    seed.score,
                    anchor_source,
                )
                for seed in seeds
            ],
            [],
        )
        forward_count = len(prepared.frame_paths) - 1 - prepared.anchor_frame_index
        for frame_index, obj_ids, _, masks, scores in predictor.propagate_in_video(
            state,
            start_frame_idx=prepared.anchor_frame_index,
            max_frame_num_to_track=forward_count,
            reverse=False,
            propagate_preflight=False,
            tqdm_disable=False,
        ):
            if int(frame_index) == prepared.anchor_frame_index:
                continue
            yield (
                int(frame_index),
                output_masks_from_tracker(
                    obj_ids, masks, scores, labels_by_id, "tracker_forward", args
                ),
                [],
            )

        backward_count = prepared.anchor_frame_index
        for frame_index, obj_ids, _, masks, scores in predictor.propagate_in_video(
            state,
            start_frame_idx=prepared.anchor_frame_index,
            max_frame_num_to_track=backward_count,
            reverse=True,
            propagate_preflight=False,
            tqdm_disable=False,
        ):
            if int(frame_index) == prepared.anchor_frame_index:
                continue
            yield (
                int(frame_index),
                output_masks_from_tracker(
                    obj_ids, masks, scores, labels_by_id, "tracker_backward", args
                ),
                [],
            )
    finally:
        del state
        gc.collect()
        torch.cuda.empty_cache()


def bbox_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left + 1.0) * max(0.0, bottom - top + 1.0)
    if intersection <= 0.0:
        return 0.0
    first_area = max(0.0, first[2] - first[0] + 1.0) * max(0.0, first[3] - first[1] + 1.0)
    second_area = max(0.0, second[2] - second[0] + 1.0) * max(0.0, second[3] - second[1] + 1.0)
    union = first_area + second_area - intersection
    return float(intersection / union) if union > 0.0 else 0.0


def mask_geometry(mask: np.ndarray) -> tuple[tuple[float, float, float, float], tuple[float, float]]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return (0.0, 0.0, 0.0, 0.0), (0.0, 0.0)
    return (
        (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())),
        (float(xs.mean()), float(ys.mean())),
    )


def fusion_pair_metrics(track_mask: np.ndarray, detection_mask: np.ndarray) -> tuple[float, float]:
    raw_mask_iou = mask_iou(track_mask, detection_mask)
    track_box, track_center = mask_geometry(track_mask)
    detection_box, detection_center = mask_geometry(detection_mask)
    raw_box_iou = bbox_iou(track_box, detection_box)
    union_box = (
        min(track_box[0], detection_box[0]),
        min(track_box[1], detection_box[1]),
        max(track_box[2], detection_box[2]),
        max(track_box[3], detection_box[3]),
    )
    diagonal = math.hypot(
        union_box[2] - union_box[0] + 1.0,
        union_box[3] - union_box[1] + 1.0,
    )
    center_distance = math.hypot(
        track_center[0] - detection_center[0],
        track_center[1] - detection_center[1],
    )
    center_similarity = max(0.0, 1.0 - center_distance / max(diagonal, 1.0))
    match_score = 0.70 * raw_mask_iou + 0.20 * raw_box_iou + 0.10 * center_similarity
    return raw_mask_iou, float(match_score)


def minimum_cost_assignment(cost_matrix: np.ndarray) -> list[tuple[int, int]]:
    """Rectangular O(n^3) Hungarian assignment without adding a SciPy dependency."""
    costs = np.asarray(cost_matrix, dtype=np.float64)
    if costs.ndim != 2:
        raise ValueError(f"assignment cost matrix must be 2-D, got {costs.shape}")
    if not costs.size:
        return []
    transposed = False
    if costs.shape[0] > costs.shape[1]:
        costs = costs.T
        transposed = True
    row_count, column_count = costs.shape
    row_potential = np.zeros(row_count + 1, dtype=np.float64)
    column_potential = np.zeros(column_count + 1, dtype=np.float64)
    column_to_row = np.zeros(column_count + 1, dtype=np.int32)
    predecessor = np.zeros(column_count + 1, dtype=np.int32)

    for row in range(1, row_count + 1):
        column_to_row[0] = row
        current_column = 0
        minimum = np.full(column_count + 1, np.inf, dtype=np.float64)
        used = np.zeros(column_count + 1, dtype=bool)
        while True:
            used[current_column] = True
            current_row = int(column_to_row[current_column])
            delta = np.inf
            next_column = 0
            for column in range(1, column_count + 1):
                if used[column]:
                    continue
                reduced = (
                    costs[current_row - 1, column - 1]
                    - row_potential[current_row]
                    - column_potential[column]
                )
                if reduced < minimum[column]:
                    minimum[column] = reduced
                    predecessor[column] = current_column
                if minimum[column] < delta:
                    delta = minimum[column]
                    next_column = column
            for column in range(column_count + 1):
                if used[column]:
                    row_potential[column_to_row[column]] += delta
                    column_potential[column] -= delta
                else:
                    minimum[column] -= delta
            current_column = next_column
            if column_to_row[current_column] == 0:
                break
        while True:
            previous_column = int(predecessor[current_column])
            column_to_row[current_column] = column_to_row[previous_column]
            current_column = previous_column
            if current_column == 0:
                break

    pairs: list[tuple[int, int]] = []
    for column in range(1, column_count + 1):
        if column_to_row[column] == 0:
            continue
        row_index = int(column_to_row[column] - 1)
        column_index = column - 1
        pairs.append(
            (column_index, row_index) if transposed else (row_index, column_index)
        )
    return sorted(pairs)


def load_frame_finetuned_instances(
    prepared: PreparedClip,
    frame_index: int,
    model_specs: Sequence[SeedModelSpec],
    args: argparse.Namespace,
) -> dict[str, list[SeedInstance]]:
    by_label = {
        model_spec.label: load_finetuned_frame_cache(prepared, frame_index, model_spec)
        for model_spec in model_specs
    }
    piglets = by_label.get(args.piglet_label, [])
    sows = constrain_sow_seed_instances(
        by_label.get(args.sow_label, []),
        args.sow_label,
        args.min_track_mask_pixels,
    )
    if args.sow_label in by_label:
        by_label[args.sow_label] = sows
    if piglets and sows and args.overlap_policy == "exclusive":
        piglet_union = np.logical_or.reduce([instance.mask for instance in piglets])
        resolved_sows: list[SeedInstance] = []
        for instance in sows[:MAX_SOW_INSTANCES]:
            mask = instance.mask & ~piglet_union
            if int(mask.sum()) >= args.min_track_mask_pixels:
                resolved_sows.append(
                    SeedInstance(
                        instance.track_id,
                        instance.label,
                        mask,
                        instance.score,
                        instance.box_xyxy,
                    )
                )
        by_label[args.sow_label] = constrain_sow_seed_instances(
            resolved_sows,
            args.sow_label,
            args.min_track_mask_pixels,
        )
    return by_label


def resolve_fused_overlaps(
    instances: Sequence[TrackedInstance],
    args: argparse.Namespace,
) -> list[TrackedInstance]:
    if not instances:
        return []
    shape = instances[0].mask.shape
    claimed_by_group: dict[str, np.ndarray] = {}
    ordered = sorted(
        instances,
        key=lambda instance: (
            instance.label == args.piglet_label,
            instance.mask_source.startswith("finetuned_matched"),
            instance.finetuned_score
            if instance.finetuned_score is not None
            else (instance.object_probability or 0.0),
            -instance.track_id,
        ),
        reverse=True,
    )
    resolved: list[TrackedInstance] = []
    for instance in ordered:
        if args.overlap_policy == "exclusive":
            overlap_group = "__all_instances__"
        elif args.overlap_policy == "cross-class":
            overlap_group = instance.label
        else:
            # One ownership group per track means no mask is clipped by another
            # instance, while the minimum-area validation below still applies.
            overlap_group = f"track:{instance.track_id}"
        claimed = claimed_by_group.setdefault(overlap_group, np.zeros(shape, dtype=bool))
        mask = instance.mask & ~claimed
        if int(mask.sum()) < args.min_track_mask_pixels:
            mask = np.zeros_like(mask, dtype=bool)
        else:
            claimed |= mask
        resolved.append(
            TrackedInstance(
                instance.track_id,
                instance.label,
                mask,
                instance.object_score_logit,
                instance.object_probability,
                instance.mask_source,
                instance.finetuned_score,
                instance.fusion_match_score,
                instance.fusion_mask_iou,
            )
        )
    return sorted(resolved, key=lambda instance: instance.track_id)


def fuse_frame_instances(
    prepared: PreparedClip,
    frame_index: int,
    tracker_instances: Sequence[TrackedInstance],
    model_specs: Sequence[SeedModelSpec],
    args: argparse.Namespace,
) -> tuple[list[TrackedInstance], list[FusionDiagnostics]]:
    if args.final_mask_source == "tracker":
        return list(tracker_instances), []
    if frame_index == prepared.anchor_frame_index and args.tracking_seed == "anchor-json":
        return list(tracker_instances), []

    finetuned_by_label = load_frame_finetuned_instances(
        prepared,
        frame_index,
        model_specs,
        args,
    )
    final_instances: list[TrackedInstance] = []
    diagnostics: list[FusionDiagnostics] = []
    for model_spec in model_specs:
        label = model_spec.label
        label_tracks = [instance for instance in tracker_instances if instance.label == label]
        active_track_indexes = [
            index
            for index, instance in enumerate(label_tracks)
            if int(instance.mask.sum()) >= args.min_track_mask_pixels
        ]
        detections = finetuned_by_label.get(label, [])

        ambiguous_detections: set[int] = set()
        candidate_ids_by_detection: dict[int, tuple[int, ...]] = {}
        for detection_index, detection in enumerate(detections):
            contained_track_ids = tuple(
                track.track_id
                for track in label_tracks
                if int(track.mask.sum()) >= args.min_track_mask_pixels
                and containment_fraction(track.mask, detection.mask)
                >= args.fusion_merge_containment_threshold
            )
            candidate_ids_by_detection[detection_index] = contained_track_ids
            if len(contained_track_ids) >= 2:
                ambiguous_detections.add(detection_index)

        eligible_detection_indexes = [
            index for index in range(len(detections)) if index not in ambiguous_detections
        ]
        score_matrix = np.zeros(
            (len(active_track_indexes), len(eligible_detection_indexes)),
            dtype=np.float64,
        )
        iou_matrix = np.zeros_like(score_matrix)
        for row, track_index in enumerate(active_track_indexes):
            for column, detection_index in enumerate(eligible_detection_indexes):
                raw_iou, match_score = fusion_pair_metrics(
                    label_tracks[track_index].mask,
                    detections[detection_index].mask,
                )
                iou_matrix[row, column] = raw_iou
                score_matrix[row, column] = match_score

        matched_by_track: dict[int, tuple[int, float, float]] = {}
        matched_detection_indexes: set[int] = set()
        for row, column in minimum_cost_assignment(1.0 - score_matrix):
            raw_iou = float(iou_matrix[row, column])
            match_score = float(score_matrix[row, column])
            if raw_iou < args.fusion_min_mask_iou or match_score < args.fusion_min_match_score:
                continue
            track_index = active_track_indexes[row]
            detection_index = eligible_detection_indexes[column]
            matched_by_track[track_index] = (detection_index, match_score, raw_iou)
            matched_detection_indexes.add(detection_index)

        for track_index, track in enumerate(label_tracks):
            match = matched_by_track.get(track_index)
            if match is not None:
                detection_index, match_score, raw_iou = match
                detection = detections[detection_index]
                final_instances.append(
                    TrackedInstance(
                        track.track_id,
                        track.label,
                        detection.mask,
                        track.object_score_logit,
                        track.object_probability,
                        "finetuned_matched",
                        detection.score,
                        match_score,
                        raw_iou,
                    )
                )
            elif args.final_mask_source == "hybrid":
                final_instances.append(
                    TrackedInstance(
                        track.track_id,
                        track.label,
                        track.mask,
                        track.object_score_logit,
                        track.object_probability,
                        "hybrid_tracker_fallback",
                    )
                )
            else:
                final_instances.append(
                    TrackedInstance(
                        track.track_id,
                        track.label,
                        np.zeros_like(track.mask, dtype=bool),
                        track.object_score_logit,
                        track.object_probability,
                        "finetuned_unmatched",
                    )
                )

        for detection_index, detection in enumerate(detections):
            if detection_index in matched_detection_indexes:
                continue
            status = (
                "ambiguous_merge"
                if detection_index in ambiguous_detections
                else "unmatched_new_object_candidate"
            )
            diagnostics.append(
                FusionDiagnostics(
                    frame_index=frame_index,
                    label=label,
                    detection_index=detection_index,
                    status=status,
                    finetuned_score=detection.score,
                    bbox_xyxy=detection.box_xyxy,
                    area_px=int(detection.mask.sum()),
                    candidate_track_ids=candidate_ids_by_detection.get(detection_index, ()),
                )
            )

    known_labels = {model_spec.label for model_spec in model_specs}
    final_instances.extend(
        instance for instance in tracker_instances if instance.label not in known_labels
    )
    return resolve_fused_overlaps(final_instances, args), diagnostics


def iter_final_frame_results(
    frame_results: Iterable[tuple[int, list[TrackedInstance], list[FusionDiagnostics]]],
    prepared: PreparedClip,
    model_specs: Sequence[SeedModelSpec],
    args: argparse.Namespace,
) -> Iterable[tuple[int, list[TrackedInstance], list[FusionDiagnostics]]]:
    for frame_index, tracker_instances, diagnostics in frame_results:
        tracker_instances = constrain_sow_tracked_instances(tracker_instances, args)
        if args.final_mask_source == "tracker":
            yield frame_index, tracker_instances, diagnostics
            continue
        fused_instances, fusion_diagnostics = fuse_frame_instances(
            prepared,
            frame_index,
            tracker_instances,
            model_specs,
            args,
        )
        yield (
            frame_index,
            constrain_sow_tracked_instances(fused_instances, args),
            fusion_diagnostics,
        )


def mask_to_polygons(mask: np.ndarray, args: argparse.Namespace) -> list[list[list[float]]]:
    contours, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[tuple[float, list[list[float]]]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < args.min_polygon_area:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        epsilon = max(1.0, args.epsilon_ratio * perimeter)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        while len(approx) > args.max_polygon_points and epsilon < max(1.0, 0.1 * perimeter):
            epsilon *= 1.5
            approx = cv2.approxPolyDP(contour, epsilon, True)
        points = approx.reshape(-1, 2)
        if len(points) > args.max_polygon_points:
            indexes = np.linspace(
                0,
                len(points),
                num=args.max_polygon_points,
                endpoint=False,
                dtype=int,
            )
            points = points[indexes]
        if len(points) < 3:
            continue
        polygons.append((area, [[float(x), float(y)] for x, y in points]))
    polygons.sort(key=lambda item: item[0], reverse=True)
    if args.max_polygons_per_mask > 0:
        polygons = polygons[: args.max_polygons_per_mask]
    return [points for _, points in polygons]


def anylabeling_shape(
    label: str,
    track_id: int,
    points: Sequence[Sequence[float]],
    score: float | None,
    source: str,
) -> dict[str, Any]:
    human_seed = source in {"human", "init_anchor_json"}
    if human_seed:
        annotation_source = "human"
    elif source.startswith("finetuned"):
        annotation_source = "finetuned_sam3"
    elif source.startswith("hybrid"):
        annotation_source = "sam3_tracker_fallback"
    else:
        annotation_source = "sam3_tracker"
    return {
        "label": label,
        "score": None if score is None else round(float(score), 6),
        "points": [[float(x), float(y)] for x, y in points],
        "group_id": int(track_id),
        "description": f"auto_source={source}",
        "difficult": False,
        "shape_type": "polygon",
        "flags": {},
        "attributes": {
            "visibility": "visible",
            "annotation_source": annotation_source,
        },
        "kie_linking": [],
    }


def anylabeling_document(
    image_path: Path,
    width: int,
    height: int,
    shapes: Sequence[dict[str, Any]],
    version: str,
) -> dict[str, Any]:
    return {
        "version": version,
        "flags": {},
        "shapes": list(shapes),
        "description": "SAM3 video-tracker pre-annotation; requires human review",
        "checked": False,
        "imagePath": image_path.name,
        "imageData": None,
        "imageHeight": int(height),
        "imageWidth": int(width),
    }


def mask_bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def write_seed_anchor_annotation(
    prepared: PreparedClip,
    seeds: Sequence[SeedInstance],
    args: argparse.Namespace,
) -> Path:
    image_path = prepared.frame_paths[prepared.anchor_frame_index]
    annotation_path = anchor_annotation_path(prepared)
    if annotation_path.exists() and not args.overwrite:
        return annotation_path
    with Image.open(image_path) as image:
        width, height = image.size
    shapes: list[dict[str, Any]] = []
    for seed in seeds:
        for points in mask_to_polygons(seed.mask, args):
            shapes.append(
                anylabeling_shape(
                    seed.label,
                    seed.track_id,
                    points,
                    seed.score,
                    "init_finetuned_sam3",
                )
            )
    with annotation_path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(
            anylabeling_document(
                image_path,
                width,
                height,
                shapes,
                args.anylabeling_version,
            ),
            file,
            ensure_ascii=False,
            indent=2,
        )
        file.write("\n")
    return annotation_path


def write_tracking_outputs(
    prepared: PreparedClip,
    frame_results: Iterable[tuple[int, list[TrackedInstance], list[FusionDiagnostics]]],
    args: argparse.Namespace,
) -> None:
    id_mask_dir = prepared.clip_dir / "id_masks"
    sow_mask_dir = prepared.clip_dir / "sow_masks"
    piglet_id_mask_dir = prepared.clip_dir / "piglet_id_masks"
    instance_mask_dir = prepared.clip_dir / "instance_masks"
    for mask_dir in (id_mask_dir, sow_mask_dir, piglet_id_mask_dir, instance_mask_dir):
        mask_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for mask_dir in (id_mask_dir, sow_mask_dir, piglet_id_mask_dir):
            for path in mask_dir.glob("*.png"):
                if path.stem.isdigit():
                    path.unlink()
        if instance_mask_dir.is_dir():
            shutil.rmtree(instance_mask_dir)
            instance_mask_dir.mkdir(parents=True, exist_ok=True)

    sidecar_path = prepared.clip_dir / "tracks.jsonl"
    diagnostics_path = prepared.clip_dir / "fusion_diagnostics.jsonl"
    sidecar_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    seen_frames: set[int] = set()
    for frame_index, instances, frame_diagnostics in frame_results:
        if not 0 <= frame_index < len(prepared.frame_paths):
            raise IndexError(f"tracker returned out-of-range frame index {frame_index}")
        if frame_index in seen_frames:
            raise RuntimeError(f"tracker returned frame {frame_index} more than once")
        seen_frames.add(frame_index)
        image_path = prepared.frame_paths[frame_index]
        with Image.open(image_path) as image:
            width, height = image.size
        shapes: list[dict[str, Any]] = []
        id_mask = np.zeros((height, width), dtype=np.uint16)
        sow_mask = np.zeros((height, width), dtype=bool)
        piglet_id_mask = np.zeros((height, width), dtype=np.uint16)
        present_instances: list[TrackedInstance] = []
        frame_instance_mask_dir = instance_mask_dir / f"{frame_index:06d}"
        if frame_instance_mask_dir.is_dir():
            for path in frame_instance_mask_dir.glob("*.png"):
                if path.stem.isdigit():
                    path.unlink()
        frame_instance_mask_dir.mkdir(parents=True, exist_ok=True)
        for instance in sorted(instances, key=lambda item: item.track_id):
            area = int(instance.mask.sum())
            present = area >= args.min_track_mask_pixels
            polygons = mask_to_polygons(instance.mask, args) if present else []
            for points in polygons:
                shapes.append(
                    anylabeling_shape(
                        instance.label,
                        instance.track_id,
                        points,
                        instance.object_probability,
                        instance.mask_source,
                    )
                )
            if present:
                present_instances.append(instance)
            instance_mask_path = frame_instance_mask_dir / f"{instance.track_id:05d}.png"
            if not cv2.imwrite(
                str(instance_mask_path),
                instance.mask.astype(np.uint8) * 255,
            ):
                raise OSError(
                    f"could not write instance mask for frame {frame_index}, "
                    f"track {instance.track_id}"
                )
            sidecar_rows.append(
                {
                    "clip_id": prepared.spec.clip_id,
                    "frame_index": frame_index,
                    "image": image_path.name,
                    "clip_timestamp_sec": round(prepared.clip_timestamp_sec(frame_index), 6),
                    "source_timestamp_sec": round(prepared.source_timestamp_sec(frame_index), 6),
                    "anchor_frame_index": prepared.anchor_frame_index,
                    "track_id": instance.track_id,
                    "label": instance.label,
                    "mask_source": instance.mask_source,
                    "object_score_logit": instance.object_score_logit,
                    "object_probability": instance.object_probability,
                    "finetuned_score": instance.finetuned_score,
                    "fusion_match_score": instance.fusion_match_score,
                    "fusion_mask_iou": instance.fusion_mask_iou,
                    "bbox_xyxy": mask_bbox(instance.mask),
                    "area_px": area,
                    "present": present,
                }
            )
        # The legacy single-channel ID mask cannot encode sow/piglet overlap.
        # Fill sow first and piglets second so its visible ownership is deterministic,
        # while the class-layered masks below preserve the full overlapping geometry.
        for instance in sorted(
            present_instances,
            key=lambda item: (item.label != args.sow_label, item.track_id),
        ):
            id_mask[instance.mask] = np.uint16(instance.track_id)
            if instance.label == args.sow_label:
                sow_mask |= instance.mask
            elif instance.label == args.piglet_label:
                piglet_id_mask[instance.mask] = np.uint16(instance.track_id)
        for diagnostic in frame_diagnostics:
            diagnostic_rows.append(
                {
                    "clip_id": prepared.spec.clip_id,
                    "frame_index": frame_index,
                    "image": image_path.name,
                    "clip_timestamp_sec": round(prepared.clip_timestamp_sec(frame_index), 6),
                    "source_timestamp_sec": round(prepared.source_timestamp_sec(frame_index), 6),
                    "label": diagnostic.label,
                    "detection_index": diagnostic.detection_index,
                    "status": diagnostic.status,
                    "finetuned_score": diagnostic.finetuned_score,
                    "bbox_xyxy": list(diagnostic.bbox_xyxy),
                    "area_px": diagnostic.area_px,
                    "candidate_track_ids": list(diagnostic.candidate_track_ids),
                }
            )
        json_path = image_path.with_suffix(".json")
        with json_path.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(anylabeling_document(image_path, width, height, shapes, args.anylabeling_version), file, ensure_ascii=False, indent=2)
            file.write("\n")
        if not cv2.imwrite(str(id_mask_dir / f"{frame_index:06d}.png"), id_mask):
            raise OSError(f"could not write ID mask for frame {frame_index}")
        if not cv2.imwrite(
            str(sow_mask_dir / f"{frame_index:06d}.png"),
            sow_mask.astype(np.uint8) * 255,
        ):
            raise OSError(f"could not write sow mask for frame {frame_index}")
        if not cv2.imwrite(
            str(piglet_id_mask_dir / f"{frame_index:06d}.png"),
            piglet_id_mask,
        ):
            raise OSError(f"could not write piglet ID mask for frame {frame_index}")

    missing_frames = set(range(len(prepared.frame_paths))) - seen_frames
    if missing_frames:
        raise RuntimeError(f"tracker returned no result for frames: {sorted(missing_frames)}")
    sidecar_rows.sort(key=lambda row: (int(row["frame_index"]), int(row["track_id"])))
    with sidecar_path.open("w", encoding="utf-8", newline="\n") as file:
        for row in sidecar_rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    diagnostic_rows.sort(
        key=lambda row: (
            int(row["frame_index"]),
            str(row["label"]),
            int(row["detection_index"]),
        )
    )
    if diagnostic_rows:
        with diagnostics_path.open("w", encoding="utf-8", newline="\n") as file:
            for row in diagnostic_rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
    elif diagnostics_path.is_file():
        diagnostics_path.unlink()
    source_counts: dict[str, int] = {}
    for row in sidecar_rows:
        source = str(row["mask_source"])
        source_counts[source] = source_counts.get(source, 0) + 1
    diagnostic_counts: dict[str, int] = {}
    for row in diagnostic_rows:
        status = str(row["status"])
        diagnostic_counts[status] = diagnostic_counts.get(status, 0) + 1
    completion = {
        "clip_id": prepared.spec.clip_id,
        "frame_count": len(prepared.frame_paths),
        "anchor_frame_index": prepared.anchor_frame_index,
        "track_ids": sorted({row["track_id"] for row in sidecar_rows}),
        "track_labels": {
            str(row["track_id"]): row["label"]
            for row in sidecar_rows
        },
        "annotation_format": "X-AnyLabeling native per-frame polygon JSON",
        "identity_field": "shape.group_id",
        "final_mask_source": args.final_mask_source,
        "overlap_policy": args.overlap_policy,
        "sow_mask_policy": {
            "max_instances_per_frame": MAX_SOW_INSTANCES,
            "connected_components_per_instance": 1,
            "selection": "largest_connected_area_then_score",
        },
        "mask_layers": {
            "sow_masks": "uint8 binary amodal sow mask; may overlap piglet_id_masks",
            "piglet_id_masks": (
                "legacy uint16 flattened piglet track-ID raster; cannot encode piglet/piglet overlap"
            ),
            "id_masks": "legacy uint16 flattened raster with piglet priority in overlap pixels",
            "instance_masks": (
                "uint8 binary mask per frame/track ID; authoritative representation for all overlaps"
            ),
        },
        "mask_source_counts": source_counts,
        "fusion_diagnostic_counts": diagnostic_counts,
        "human_review_required": True,
        "new_objects_after_anchor": (
            "not auto-initialized; persistent unmatched finetuned detections are listed in "
            "fusion_diagnostics.jsonl as review candidates"
        ),
        "tracking_configuration": tracking_configuration(args, prepared),
    }
    with (prepared.clip_dir / "tracking_complete.json").open("w", encoding="utf-8") as file:
        json.dump(completion, file, ensure_ascii=False, indent=2)
        file.write("\n")


def tracking_output_is_reusable(prepared: PreparedClip, args: argparse.Namespace) -> bool:
    completion_path = prepared.clip_dir / "tracking_complete.json"
    if not completion_path.exists():
        return False
    if not completion_path.is_file():
        raise ValueError(f"tracking completion path is not a file: {completion_path}")
    with completion_path.open("r", encoding="utf-8") as file:
        completion = json.load(file)
    if completion.get("tracking_configuration") != tracking_configuration(args, prepared):
        raise ValueError(
            f"tracking/export configuration changed for {prepared.spec.clip_id}; use --overwrite"
        )
    sidecar_path = prepared.clip_dir / "tracks.jsonl"
    id_mask_dir = prepared.clip_dir / "id_masks"
    sow_mask_dir = prepared.clip_dir / "sow_masks"
    piglet_id_mask_dir = prepared.clip_dir / "piglet_id_masks"
    instance_mask_dir = prepared.clip_dir / "instance_masks"
    expected_track_ids = [int(value) for value in completion.get("track_ids", [])]
    if not sidecar_path.is_file() or sidecar_path.stat().st_size == 0:
        raise ValueError(
            f"completion marker exists but tracks.jsonl is missing/empty for {prepared.spec.clip_id}; "
            "use --overwrite"
        )
    for frame_index, frame_path in enumerate(prepared.frame_paths):
        if not frame_path.with_suffix(".json").is_file():
            raise ValueError(
                f"completion marker exists but an annotation is missing for {prepared.spec.clip_id}; "
                "use --overwrite"
            )
        if not (id_mask_dir / f"{frame_index:06d}.png").is_file():
            raise ValueError(
                f"completion marker exists but an ID mask is missing for {prepared.spec.clip_id}; "
                "use --overwrite"
            )
        if not (sow_mask_dir / f"{frame_index:06d}.png").is_file():
            raise ValueError(
                f"completion marker exists but a sow mask is missing for {prepared.spec.clip_id}; "
                "use --overwrite"
            )
        if not (piglet_id_mask_dir / f"{frame_index:06d}.png").is_file():
            raise ValueError(
                f"completion marker exists but a piglet ID mask is missing for {prepared.spec.clip_id}; "
                "use --overwrite"
            )
        frame_instance_mask_dir = instance_mask_dir / f"{frame_index:06d}"
        for track_id in expected_track_ids:
            if not (frame_instance_mask_dir / f"{track_id:05d}.png").is_file():
                raise ValueError(
                    f"completion marker exists but instance mask {track_id} is missing at frame "
                    f"{frame_index} for {prepared.spec.clip_id}; use --overwrite"
                )
    return True


def write_attributes_file(output_root: Path, labels: Sequence[str], overwrite: bool = False) -> None:
    label_payload = {
        label: {
            "visibility": [
                "visible",
                "partial_occlusion",
                "full_occlusion",
                "truncated",
                "uncertain",
            ],
            "annotation_source": [
                "finetuned_sam3",
                "sam3_tracker",
                "sam3_tracker_fallback",
                "human",
            ],
        }
        for label in labels
    }
    widget_payload = {
        label: {
            "visibility": "combobox",
            "annotation_source": "radiobutton",
        }
        for label in labels
    }
    payload = {
        **label_payload,
        "__widget_types__": widget_payload,
    }
    output_path = output_root / "attributes.json"
    if output_path.exists() and not overwrite:
        return
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def main() -> int:
    args = parse_args()
    validate_args(args)
    model_specs = requested_seed_models(args)
    labels = [model.label for model in model_specs]
    manifest = validate_file(args.manifest, "selection manifest")
    clips = read_selection_manifest(manifest)
    if args.max_clips > 0:
        clips = clips[: args.max_clips]
    output_root = args.output_dir.resolve()
    ffmpeg = executable_path("ffmpeg", args.ffmpeg)

    print(
        f"Clips: {len(clips)}; track FPS: {args.track_fps:g}; "
        f"smoke radius: {args.smoke_radius_frames}; final masks: {args.final_mask_source}; "
        f"overlap: {args.overlap_policy}; "
        f"output: {output_root}"
    )
    for spec in clips:
        print(
            f"  {spec.clip_id}: {spec.output_video} -> "
            f"anchor {spec.anchor_in_clip_sec:.3f}s"
        )
    if args.dry_run:
        print("Dry run: no frames, model outputs, or annotations were written.")
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    write_attributes_file(output_root, labels, overwrite=args.overwrite)
    prepared_clips: list[PreparedClip] = []
    failures: list[tuple[str, str]] = []
    for spec in clips:
        try:
            prepared = prepare_clip_frames(spec, output_root, ffmpeg, args)
            prepared_clips.append(prepared)
            write_attributes_file(prepared.clip_dir, labels, overwrite=args.overwrite)
            print(
                f"[frames] {spec.clip_id}: {len(prepared.frame_paths)} frames; "
                f"anchor index {prepared.anchor_frame_index}"
            )
        except Exception as exc:
            failures.append((spec.clip_id, f"frame preparation: {exc}"))
            print(f"[failed] {spec.clip_id}: {exc}", file=sys.stderr)
            if args.fail_on_error:
                raise
    if args.frames_only:
        return finish_failure_report(output_root, failures)
    if not prepared_clips:
        raise RuntimeError("no clips were prepared successfully")

    for model_spec in model_specs:
        validate_file(model_spec.checkpoint, f"{model_spec.label} checkpoint")
    validate_file(args.bpe_path, "BPE vocabulary")
    missing_seed_clips: list[PreparedClip] = []
    for prepared in prepared_clips:
        if args.overwrite or not seed_archive_is_reusable(prepared, args):
            missing_seed_clips.append(prepared)
    if missing_seed_clips:
        import torch

        seeds_by_clip: dict[str, dict[str, list[SeedInstance]]] = {
            prepared.spec.clip_id: {} for prepared in missing_seed_clips
        }
        seed_failed_clips: set[str] = set()
        for model_spec in model_specs:
            print(f"[load-seed-model] {model_spec.label}: {model_spec.checkpoint}")
            model = None
            processor = None
            try:
                model, processor = load_seed_processor(args, model_spec)
                with torch.inference_mode():
                    for prepared in missing_seed_clips:
                        if prepared.spec.clip_id in seed_failed_clips:
                            continue
                        try:
                            seeds = infer_seed_instances(
                                processor,
                                prepared.frame_paths[prepared.anchor_frame_index],
                                model_spec,
                                args,
                            )
                            if not seeds:
                                raise RuntimeError(
                                    f"finetuned SAM3 found no usable {model_spec.label} seed masks"
                                )
                            seeds_by_clip[prepared.spec.clip_id][model_spec.label] = seeds
                            print(
                                f"[seed-{model_spec.label}] {prepared.spec.clip_id}: "
                                f"{len(seeds)} instance(s)"
                            )
                        except Exception as exc:
                            seed_failed_clips.add(prepared.spec.clip_id)
                            failures.append(
                                (prepared.spec.clip_id, f"{model_spec.label} seed inference: {exc}")
                            )
                            print(f"[failed] {prepared.spec.clip_id}: {exc}", file=sys.stderr)
                            if args.fail_on_error:
                                raise
            finally:
                if processor is not None:
                    del processor
                if model is not None:
                    del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        for prepared in missing_seed_clips:
            clip_id = prepared.spec.clip_id
            if clip_id in seed_failed_clips:
                continue
            try:
                seeds = finalize_seed_instances(seeds_by_clip[clip_id], model_specs, args)
                save_seed_instances(prepared, seeds, args)
                seed_json_path = write_seed_anchor_annotation(prepared, seeds, args)
                counts = {
                    label: sum(seed.label == label for seed in seeds)
                    for label in labels
                }
                print(f"[seed-combined] {clip_id}: {counts}; editable anchor: {seed_json_path.name}")
            except Exception as exc:
                failures.append((clip_id, f"seed finalization: {exc}"))
                print(f"[failed] {clip_id}: {exc}", file=sys.stderr)
                if args.fail_on_error:
                    raise
    if args.seed_only:
        return finish_failure_report(output_root, failures)

    trackable = [prepared for prepared in prepared_clips if seed_archive_paths(prepared)[0].is_file()]
    if not trackable:
        raise RuntimeError("no clips have seed masks to track")
    validate_file(args.video_checkpoint, "official SAM3 video checkpoint")
    pending_tracking: list[PreparedClip] = []
    for prepared in trackable:
        if not args.overwrite and not args.force_retrack and tracking_output_is_reusable(prepared, args):
            print(f"[skip-complete] {prepared.spec.clip_id}")
        else:
            pending_tracking.append(prepared)
    if not pending_tracking:
        status = finish_failure_report(output_root, failures)
        if status:
            return status
        print("Done. Every clip already has matching tracking outputs.")
        return 0
    if args.final_mask_source != "tracker":
        pending_tracking = prepare_finetuned_frame_caches(
            pending_tracking,
            model_specs,
            args,
            failures,
        )
        if not pending_tracking:
            return finish_failure_report(output_root, failures)
    predictor = load_official_tracker(args)
    try:
        for prepared in pending_tracking:
            completion_path = prepared.clip_dir / "tracking_complete.json"
            if args.force_retrack and completion_path.is_file():
                completion_path.unlink()
            try:
                seeds = load_tracking_seed_instances(prepared, args)
                tracker_results = iter_propagated_frames(predictor, prepared, seeds, args)
                results = iter_final_frame_results(
                    tracker_results,
                    prepared,
                    model_specs,
                    args,
                )
                try:
                    write_tracking_outputs(prepared, results, args)
                finally:
                    close = getattr(results, "close", None)
                    if close is not None:
                        close()
                    close_tracker = getattr(tracker_results, "close", None)
                    if close_tracker is not None:
                        close_tracker()
                print(
                    f"[tracked] {prepared.spec.clip_id}: {len(prepared.frame_paths)} frames, "
                    f"{len(seeds)} track IDs, final masks={args.final_mask_source}"
                )
            except Exception as exc:
                failures.append((prepared.spec.clip_id, f"tracking/export: {exc}"))
                print(f"[failed] {prepared.spec.clip_id}: {exc}", file=sys.stderr)
                if args.fail_on_error:
                    raise
    finally:
        del predictor
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    status = finish_failure_report(output_root, failures)
    if status:
        return status
    print(f"Done. Open each <clip_id>/frames directory in AnyLabeling (Ctrl+U).")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
