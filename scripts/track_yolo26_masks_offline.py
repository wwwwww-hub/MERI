"""Run YOLO26 segmentation followed by the trained mask-aware offline tracker.

The script never changes YOLO masks.  It writes a uint16 ID raster per frame and
``tracks.jsonl`` with stable IDs assigned by the student association model.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.trackers.mask_aware import (
    MaskAwareOfflineTracker,
    MaskObservation,
    TrackerConfig,
    encode_observation_crops,
    extract_masked_crop,
    load_association_checkpoint,
    observation_from_mask,
    tracker_config_from_dict,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_YOLO_MODEL = ROOT / "models" / "segmentation" / "piglet.pt"
GENERATED_NAMES = {"_local_masks", "id_masks", "tracks.jsonl", "tracking_summary.json"}
VIDEO_SUFFIXES = {".asf", ".avi", ".gif", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".ts", ".wmv"}
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
FRAGMENT_MAX_AREA_RATIO = 0.35
FRAGMENT_MAX_GAP_SCALE = 0.35
FRAGMENT_MAX_UNION_BOX_RATIO = 2.50


@dataclass(slots=True)
class FrameCandidate:
    """One per-frame instance, optionally formed by fusing raw YOLO fragments."""

    detection_indices: list[int]
    class_id: int
    label: str
    confidence: float
    mask: np.ndarray
    bbox_xyxy: np.ndarray


def mask_bbox(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("cannot compute a bounding box for an empty mask")
    return np.asarray((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1), dtype=np.float32)


def candidate_from_detection(
    detection_index: int,
    masks: np.ndarray,
    classes: np.ndarray,
    confidences: np.ndarray,
    names: dict[int, str],
) -> FrameCandidate:
    mask = np.asarray(masks[detection_index], dtype=bool)
    class_id = int(classes[detection_index])
    return FrameCandidate(
        detection_indices=[int(detection_index)],
        class_id=class_id,
        label=str(names[class_id]),
        confidence=float(confidences[detection_index]),
        mask=mask,
        bbox_xyxy=mask_bbox(mask),
    )


def fragment_merge_score(first: FrameCandidate, second: FrameCandidate) -> float | None:
    """Return a conservative lower-is-better score for a likely small adjacent fragment."""
    if first.class_id != second.class_id:
        return None
    first_area = int(first.mask.sum())
    second_area = int(second.mask.sum())
    smaller, larger = sorted((first_area, second_area))
    area_ratio = smaller / max(larger, 1)
    if area_ratio > FRAGMENT_MAX_AREA_RATIO:
        return None
    first_box, second_box = first.bbox_xyxy, second.bbox_xyxy
    gap_x = max(float(first_box[0] - second_box[2]), float(second_box[0] - first_box[2]), 0.0)
    gap_y = max(float(first_box[1] - second_box[3]), float(second_box[1] - first_box[3]), 0.0)
    gap_scale = float(np.hypot(gap_x, gap_y)) / max(float(np.sqrt(larger)), 1.0)
    if gap_scale > FRAGMENT_MAX_GAP_SCALE:
        return None
    union_box = np.asarray(
        (
            min(first_box[0], second_box[0]),
            min(first_box[1], second_box[1]),
            max(first_box[2], second_box[2]),
            max(first_box[3], second_box[3]),
        ),
        dtype=np.float32,
    )
    union_box_area = max(float(union_box[2] - union_box[0]), 1.0) * max(
        float(union_box[3] - union_box[1]), 1.0
    )
    union_box_ratio = union_box_area / max(first_area + second_area, 1)
    if union_box_ratio > FRAGMENT_MAX_UNION_BOX_RATIO:
        return None
    return gap_scale + area_ratio + 0.05 * union_box_ratio


def merge_frame_candidates(first: FrameCandidate, second: FrameCandidate) -> FrameCandidate:
    mask = np.logical_or(first.mask, second.mask)
    return FrameCandidate(
        detection_indices=sorted(first.detection_indices + second.detection_indices),
        class_id=first.class_id,
        label=first.label,
        confidence=max(first.confidence, second.confidence),
        mask=mask,
        bbox_xyxy=mask_bbox(mask),
    )


def enforce_frame_identity_cap(
    candidates: list[FrameCandidate], max_identities: int
) -> tuple[list[FrameCandidate], int, int]:
    """Fuse plausible fragments, then suppress weakest overflow so a frame never exceeds the hard cap."""
    values = list(candidates)
    fused = 0
    while max_identities > 0 and len(values) > max_identities:
        merge_options = [
            (score, first_index, second_index)
            for first_index, first in enumerate(values)
            for second_index, second in enumerate(values[first_index + 1 :], first_index + 1)
            if (score := fragment_merge_score(first, second)) is not None
        ]
        if not merge_options:
            break
        _, first_index, second_index = min(merge_options)
        merged = merge_frame_candidates(values[first_index], values[second_index])
        values = [
            candidate
            for index, candidate in enumerate(values)
            if index not in (first_index, second_index)
        ]
        values.append(merged)
        fused += 1
    values.sort(key=lambda item: (-item.confidence, item.detection_indices))
    suppressed = 0
    if max_identities > 0 and len(values) > max_identities:
        suppressed = sum(len(item.detection_indices) for item in values[max_identities:])
        values = values[:max_identities]
    return values, fused, suppressed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yolo-model", type=Path, default=DEFAULT_YOLO_MODEL)
    parser.add_argument("--tracker-checkpoint", type=Path, default=ROOT / "models/tracking/osnet_x0_25.pt")
    parser.add_argument(
        "--source", required=True, help="One local video, or one ordered image directory already sampled at training FPS."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", action="append", default=None, help="YOLO class name to track; default: piglet.")
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64, help="Student embedding batch size per frame.")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=0.0,
        help="Target video FPS; 0 uses the rate saved in the student checkpoint (normally 2 FPS).",
    )
    parser.add_argument(
        "--vid-stride",
        type=int,
        default=0,
        help="Explicit video frame stride. Normally inferred from source FPS and --sample-fps.",
    )
    parser.add_argument("--match-threshold", type=float, default=0.20)
    parser.add_argument("--max-age", type=int, default=2)
    parser.add_argument("--max-center-distance", type=float, default=0.30)
    parser.add_argument("--motion-weight", type=float, default=2.0)
    parser.add_argument("--no-offline", action="store_true")
    parser.add_argument("--offline-max-gap", type=int, default=8)
    parser.add_argument("--offline-match-threshold", type=float, default=0.25)
    parser.add_argument("--offline-max-center-distance", type=float, default=0.45)
    parser.add_argument("--offline-max-tracklets", type=int, default=5000)
    parser.add_argument(
        "--max-identities",
        type=int,
        default=14,
        help=(
            "Hard reusable-ID pool. Lower-confidence detections beyond the per-frame cap are suppressed, and "
            "non-overlapping tracklets reuse IDs. Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--max-observations",
        type=int,
        default=50000,
        help="Safety limit for one offline clip; split longer recordings into clips.",
    )
    parser.add_argument("--keep-local-masks", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def select_device(value: str) -> torch.device:
    value = value.strip().lower()
    if value in {"", "cuda"}:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if value.isdigit():
        return torch.device(f"cuda:{value}")
    return torch.device(value)


def prepare_output(path: Path, overwrite: bool) -> Path:
    output = path.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    existing = [child for child in output.iterdir() if child.name in GENERATED_NAMES]
    if existing and not overwrite:
        raise FileExistsError(f"generated tracker outputs already exist in {output}; use --overwrite")
    if overwrite:
        for child in existing:
            resolved = child.resolve()
            if resolved.parent != output:
                raise RuntimeError(f"refusing to remove path outside output directory: {resolved}")
            if resolved.is_dir():
                shutil.rmtree(resolved)
            else:
                resolved.unlink()
    (output / "_local_masks").mkdir()
    (output / "id_masks").mkdir()
    return output


def validate_source(source: str) -> tuple[Path, str]:
    path = Path(source).expanduser().resolve()
    if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
        return path, "video"
    if path.is_dir():
        images = [item for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES]
        if not images:
            raise ValueError(f"image sequence directory contains no supported images: {path}")
        return path, "image_sequence"
    raise ValueError("--source must be one local video or one ordered image-sequence directory")


def video_sampling(
    source: Path, source_kind: str, target_fps: float, explicit_stride: int
) -> tuple[int, float | None, float | None]:
    """Return Ultralytics video stride plus source and effective FPS when knowable."""
    if target_fps <= 0 or explicit_stride < 0:
        raise ValueError("target FPS must be positive and --vid-stride non-negative")
    if source_kind == "image_sequence":
        if explicit_stride not in (0, 1):
            raise ValueError("--vid-stride must be 1 for an already sampled image sequence")
        return 1, None, target_fps
    capture = cv2.VideoCapture(str(source))
    try:
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if not np.isfinite(source_fps) or source_fps <= 0:
        if not explicit_stride:
            raise ValueError(f"could not read source FPS for {source}; pass --vid-stride explicitly")
        return explicit_stride, None, None
    stride = explicit_stride or max(int(round(source_fps / target_fps)), 1)
    return stride, source_fps, source_fps / stride


def resize_masks(mask_tensor: torch.Tensor, image_shape: tuple[int, int]) -> np.ndarray:
    masks = mask_tensor.detach().float().cpu().numpy()
    if masks.shape[-2:] == image_shape:
        return masks > 0.5
    return np.stack(
        [
            cv2.resize(mask, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST) > 0.5
            for mask in masks
        ]
    )


def main() -> None:
    args = parse_args()
    if not 0 <= args.conf <= 1 or not 0 <= args.iou <= 1:
        raise ValueError("--conf and --iou must be in [0, 1]")
    if (
        args.imgsz <= 0
        or args.batch_size <= 0
        or args.max_frames < 0
        or args.vid_stride < 0
        or args.max_observations <= 0
        or args.max_identities < 0
    ):
        raise ValueError("image/batch sizes must be positive and limit values non-negative")
    labels = tuple(args.label or ["piglet"])
    source, source_kind = validate_source(args.source)
    if not args.yolo_model.expanduser().is_file():
        raise FileNotFoundError(f"YOLO model does not exist: {args.yolo_model}")
    if not args.tracker_checkpoint.expanduser().is_file():
        raise FileNotFoundError(f"tracker checkpoint does not exist: {args.tracker_checkpoint}")
    device = select_device(args.device)
    association_model, checkpoint = load_association_checkpoint(args.tracker_checkpoint, device)
    training_args = checkpoint.get("training_args", {})
    checkpoint_fps = float(training_args.get("sample_fps", 2.0))
    target_fps = float(args.sample_fps or checkpoint_fps)
    vid_stride, source_fps, effective_fps = video_sampling(source, source_kind, target_fps, args.vid_stride)
    if effective_fps is not None and abs(effective_fps - checkpoint_fps) / checkpoint_fps > 0.10:
        raise ValueError(
            f"effective sampling rate {effective_fps:.3f} FPS differs from the student training rate "
            f"{checkpoint_fps:.3f} FPS by more than 10%; adjust --vid-stride"
        )
    print(
        f"temporal sampling: training={checkpoint_fps:.3f} FPS, target={target_fps:.3f} FPS, "
        f"source={source_fps if source_fps is not None else 'pre-sampled image sequence'}, stride={vid_stride}, "
        f"effective={effective_fps if effective_fps is not None else target_fps}"
    )
    yolo = YOLO(str(args.yolo_model))
    config = tracker_config_from_dict(
        {
            "match_threshold": args.match_threshold,
            "max_age": args.max_age,
            "max_center_distance": args.max_center_distance,
            "motion_weight": args.motion_weight,
            "offline": not args.no_offline,
            "offline_max_gap": args.offline_max_gap,
            "offline_match_threshold": args.offline_match_threshold,
            "offline_max_center_distance": args.offline_max_center_distance,
            "offline_max_tracklets": args.offline_max_tracklets,
            "max_identities": args.max_identities,
        }
    )
    tracker = MaskAwareOfflineTracker(association_model, device, config)
    output = prepare_output(args.output_dir, args.overwrite)
    prediction_device = str(device.index) if device.type == "cuda" else "cpu"
    stream = yolo.predict(
        source=str(source),
        stream=True,
        save=False,
        verbose=False,
        retina_masks=True,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=prediction_device,
        half=args.half,
        vid_stride=vid_stride,
    )
    frames: dict[int, list[MaskObservation]] = {}
    frame_local_observations: dict[int, list[tuple[int, int]]] = {}
    observation_rows: dict[int, dict] = {}
    next_observation_id = 1
    processed_frames = 0
    identity_cap_suppressed = 0
    fragment_fusions = 0
    for frame_index, result in enumerate(stream):
        if args.max_frames and frame_index >= args.max_frames:
            break
        image = result.orig_img
        image_shape = image.shape[:2]
        frame_observations: list[MaskObservation] = []
        crops: list[torch.Tensor] = []
        local_ids: list[tuple[int, int]] = []
        local_mask = np.zeros(image_shape, dtype=np.uint16)
        suppressed_this_frame = 0
        fused_this_frame = 0
        if result.boxes is not None and result.masks is not None and len(result.boxes):
            masks = resize_masks(result.masks.data, image_shape)
            boxes = result.boxes.xyxy.detach().cpu().numpy()
            classes = result.boxes.cls.detach().cpu().numpy().astype(int)
            confidences = result.boxes.conf.detach().cpu().numpy()
            selected_indices = [
                index
                for index, class_id in enumerate(classes)
                if str(result.names[int(class_id)]) in labels and masks[index].any()
            ]
            candidates = [
                candidate_from_detection(index, masks, classes, confidences, result.names)
                for index in selected_indices
            ]
            candidates, fused_this_frame, suppressed_this_frame = enforce_frame_identity_cap(
                candidates, args.max_identities
            )
            fragment_fusions += fused_this_frame
            identity_cap_suppressed += suppressed_this_frame
            for local_index, candidate in enumerate(candidates, 1):
                local_mask[np.logical_and(candidate.mask, local_mask == 0)] = local_index
            for local_index, candidate in enumerate(candidates, 1):
                raw_mask = candidate.mask
                mask = local_mask == local_index
                if not mask.any():
                    continue
                observation = observation_from_mask(
                    observation_id=next_observation_id,
                    frame_index=frame_index,
                    class_id=candidate.class_id,
                    label=candidate.label,
                    score=candidate.confidence,
                    bbox_xyxy=candidate.bbox_xyxy,
                    mask=mask,
                    payload={
                        "local_index": local_index,
                        "source_detection_indices": candidate.detection_indices,
                    },
                )
                if next_observation_id > args.max_observations:
                    raise MemoryError(
                        f"offline clip exceeds --max-observations={args.max_observations}; "
                        "split the recording into shorter clips"
                    )
                next_observation_id += 1
                frame_observations.append(observation)
                crops.append(extract_masked_crop(image, mask, observation.bbox_xyxy))
                local_ids.append((local_index, observation.observation_id))
                observation_rows[observation.observation_id] = {
                    "frame_index": frame_index,
                    "source_frame_index_approx": frame_index * vid_stride,
                    "source_path": str(result.path),
                    "label": candidate.label,
                    "class_id": candidate.class_id,
                    "confidence": candidate.confidence,
                    "bbox_xyxy": [float(x) for x in observation.bbox_xyxy],
                    "area_px": int(mask.sum()),
                    "raw_yolo_bbox_xyxy": [float(x) for x in candidate.bbox_xyxy],
                    "raw_yolo_mask_area_px": int(raw_mask.sum()),
                    "source_detection_indices": candidate.detection_indices,
                    "source_yolo_bbox_xyxy": [
                        [float(x) for x in boxes[index]] for index in candidate.detection_indices
                    ],
                    "fused_fragment_count": len(candidate.detection_indices),
                    "local_index": local_index,
                }
        encode_observation_crops(association_model, frame_observations, crops, device, args.batch_size)
        frames[frame_index] = frame_observations
        frame_local_observations[frame_index] = local_ids
        if not cv2.imwrite(str(output / "_local_masks" / f"{frame_index:06d}.png"), local_mask):
            raise OSError(f"could not write local mask for frame {frame_index}")
        processed_frames += 1
        print(
            f"[segment] frame={frame_index} instances={len(frame_observations)} "
            f"fragment_fusions={fused_this_frame} identity_cap_suppressed={suppressed_this_frame}"
        )
    if not processed_frames:
        raise RuntimeError("YOLO produced no frames")
    assignments = tracker.track(frames)
    if tracker.identity_cap_summary is not None:
        cap = tracker.identity_cap_summary
        minimum = cap["minimum_merge_score"]
        minimum_text = f"{minimum:.4f}" if minimum is not None else "n/a"
        print(
            f"identity cap: {cap['tracks_before']} -> {cap['tracks_after']} "
            f"(max={cap['max_identities']}, forced_merges={cap['forced_merges']}, "
            f"minimum_merge_score={minimum_text})"
        )
    if max(assignments.values(), default=0) > np.iinfo(np.uint16).max:
        raise OverflowError("track ID exceeds uint16 output capacity")
    tracks_path = output / "tracks.jsonl"
    with tracks_path.open("w", encoding="utf-8") as tracks_file:
        for frame_index in range(processed_frames):
            local_mask = cv2.imread(
                str(output / "_local_masks" / f"{frame_index:06d}.png"), cv2.IMREAD_UNCHANGED
            )
            if local_mask is None:
                raise OSError(f"could not reload local mask for frame {frame_index}")
            id_mask = np.zeros_like(local_mask, dtype=np.uint16)
            for local_index, observation_id in frame_local_observations[frame_index]:
                track_id = int(assignments[observation_id])
                id_mask[local_mask == local_index] = track_id
                row = observation_rows[observation_id]
                row.update({"observation_id": observation_id, "track_id": track_id})
                tracks_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            if not cv2.imwrite(str(output / "id_masks" / f"{frame_index:06d}.png"), id_mask):
                raise OSError(f"could not write final ID mask for frame {frame_index}")
    if not args.keep_local_masks:
        local_dir = (output / "_local_masks").resolve()
        if local_dir.parent != output:
            raise RuntimeError(f"refusing to remove unsafe temporary directory: {local_dir}")
        shutil.rmtree(local_dir)
    summary = {
        "schema_version": 1,
        "source": str(source),
        "source_kind": source_kind,
        "yolo_model": str(args.yolo_model.resolve()),
        "tracker_checkpoint": str(args.tracker_checkpoint.resolve()),
        "tracker_checkpoint_epoch": checkpoint.get("epoch"),
        "labels": labels,
        "frame_count": processed_frames,
        "training_sample_fps": checkpoint_fps,
        "target_sample_fps": target_fps,
        "source_fps": source_fps,
        "vid_stride": vid_stride,
        "effective_sample_fps": effective_fps,
        "observation_count": len(assignments),
        "max_observations": args.max_observations,
        "track_count": len(set(assignments.values())),
        "tracker_config": asdict(config),
        "identity_cap": tracker.identity_cap_summary,
        "identity_cap_semantics": "IDs are reusable only across observations that never share a frame",
        "identity_cap_suppressed_detections": identity_cap_suppressed,
        "identity_cap_fragment_fusions": fragment_fusions,
        "identity_cap_fragment_policy": {
            "trigger": "only while a frame exceeds max_identities",
            "max_small_to_large_area_ratio": FRAGMENT_MAX_AREA_RATIO,
            "max_mask_gap_over_large_sqrt_area": FRAGMENT_MAX_GAP_SCALE,
            "max_union_bbox_area_over_mask_area": FRAGMENT_MAX_UNION_BOX_RATIO,
        },
        "mask_ownership": "higher-confidence instance owns flattened overlap pixels",
    }
    (output / "tracking_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"done: {output}; tracks={summary['track_count']}, observations={summary['observation_count']}, "
        f"fragment_fusions={fragment_fusions}, identity_cap_suppressed={identity_cap_suppressed}"
    )


if __name__ == "__main__":
    main()
