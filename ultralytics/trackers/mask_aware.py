"""Mask-aware association student and offline multi-object tracker.

This module deliberately decouples per-frame instance segmentation from identity
association.  A segmentation model (for example YOLO26-seg) owns every output
mask; the student only learns whether two masked instances belong to the same
object.  Compact full-frame and object-normalized mask representations make the
mask part of the association signal without retaining full-resolution masks for
an entire video.
"""

from __future__ import annotations

import json
from ultralytics.trackers.dataset_paths import (
    clip_video, extract_video_frames, frame_file, instance_file, manual_mask_file, mask_file,
    frame_index as dataset_frame_index,
)
import math
from collections import OrderedDict, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.utils.data import Dataset

from .osnet import OSNetX025FeatureExtractor


PAIR_FEATURE_NAMES = (
    "spatial_mask_iou",
    "shape_mask_iou",
    "bbox_iou",
    "center_distance_frame",
    "center_distance_object",
    "log_area_ratio",
    "log_aspect_ratio",
    "fill_ratio_difference",
    "frame_gap",
)
DEFAULT_CROP_SIZE = (96, 192)
CANONICAL_MTIME_TOLERANCE_SECONDS = 1.0


@dataclass(slots=True)
class AnnotatedInstance:
    """One teacher-labelled instance in one video frame."""

    frame_index: int
    track_id: int
    label: str
    score: float
    bbox_xyxy: tuple[float, float, float, float]
    area_px: int
    image_name: str


@dataclass(slots=True)
class ClipScan:
    """Result of scanning one dataset split."""

    clips: list["AnnotatedClip"]
    skipped: list[tuple[str, str]]


@dataclass(slots=True)
class PairReference:
    """A positive or negative pair used to train the association student."""

    clip_index: int
    first: AnnotatedInstance
    second: AnnotatedInstance
    target: float


@dataclass(slots=True)
class MaskObservation:
    """Compact representation consumed by the mask-aware tracker."""

    observation_id: int
    frame_index: int
    class_id: int
    label: str
    score: float
    bbox_xyxy: np.ndarray
    area: float
    centroid_xy: np.ndarray
    frame_shape: tuple[int, int]
    spatial_mask: np.ndarray
    shape_mask: np.ndarray
    fill_ratio: float
    embedding: np.ndarray | None = None
    gt_track_id: int | None = None
    payload: Any = None


@dataclass(slots=True)
class TrackerConfig:
    """Association and offline-linking thresholds."""

    match_threshold: float = 0.20
    max_age: int = 2
    max_center_distance: float = 0.30
    motion_weight: float = 2.0
    embedding_momentum: float = 0.90
    offline: bool = True
    offline_max_gap: int = 8
    offline_match_threshold: float = 0.25
    offline_max_center_distance: float = 0.45
    offline_max_tracklets: int = 5000
    max_identities: int = 0


@dataclass(slots=True)
class IdentityMetrics:
    """Detection-fixed identity metrics for association-only validation."""

    idtp: int
    idfp: int
    idfn: int
    idf1: float
    id_precision: float
    id_recall: float
    id_switches: int
    fragmentations: int
    gt_tracks: int
    predicted_tracks: int
    observations: int
    track_purity: float


class AnnotatedClip:
    """Lazy reader for one SAM3/AnyLabeling tracking export directory."""

    def __init__(
        self,
        root: Path,
        clip_id: str,
        frame_count: int,
        instances: Sequence[AnnotatedInstance],
        cache_frames: int = 4,
        decoded_frames: Path | None = None,
    ) -> None:
        self.root = Path(root)
        self.clip_id = clip_id
        self.frame_count = int(frame_count)
        self.instances = list(instances)
        self._decoded_frames = decoded_frames
        self.by_frame: dict[int, list[AnnotatedInstance]] = defaultdict(list)
        self.by_track: dict[tuple[str, int], list[AnnotatedInstance]] = defaultdict(list)
        for instance in self.instances:
            self.by_frame[instance.frame_index].append(instance)
            self.by_track[(instance.label, instance.track_id)].append(instance)
        for values in self.by_frame.values():
            values.sort(key=lambda item: (item.label, item.track_id))
        for values in self.by_track.values():
            values.sort(key=lambda item: item.frame_index)
        self._cache_frames = max(int(cache_frames), 0)
        self._image_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._mask_cache: OrderedDict[tuple[str, int], np.ndarray] = OrderedDict()

    @property
    def frames_dir(self) -> Path:
        return self._decoded_frames if self._decoded_frames is not None else self.root / "frames"

    def image_path(self, frame_index: int) -> Path:
        if self._decoded_frames is not None:
            return self.frames_dir / f"{frame_index:06d}.jpg"
        instances = self.by_frame.get(frame_index)
        name = instances[0].image_name if instances else frame_file(self.root,frame_index).name
        return self.frames_dir / name

    def _remember(self, cache: OrderedDict, key: Any, value: np.ndarray) -> np.ndarray:
        if self._cache_frames <= 0:
            return value
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > self._cache_frames:
            cache.popitem(last=False)
        return value

    def load_image(self, frame_index: int) -> np.ndarray:
        if frame_index in self._image_cache:
            self._image_cache.move_to_end(frame_index)
            return self._image_cache[frame_index]
        path = self.image_path(frame_index)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"could not read frame image: {path}")
        return self._remember(self._image_cache, frame_index, image)

    def load_label_mask(self, label: str, frame_index: int) -> np.ndarray:
        key = (label, frame_index)
        if key in self._mask_cache:
            self._mask_cache.move_to_end(key)
            return self._mask_cache[key]
        if label == "piglet":
            path = mask_file(self.root,"piglet_id_masks",frame_index)
            mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        elif label == "sow":
            path = mask_file(self.root,"sow_masks",frame_index)
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        else:
            path = mask_file(self.root,"id_masks",frame_index)
            mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise OSError(f"could not read {label!r} mask: {path}")
        if mask.ndim == 3 and mask.shape[2] == 1:
            mask = mask[..., 0]
        if mask.ndim != 2:
            raise ValueError(f"expected a single-channel ID mask, got {mask.shape}: {path}")
        return self._remember(self._mask_cache, key, mask)

    def load_instance_mask(self, instance: AnnotatedInstance) -> np.ndarray:
        annotation = (manual_mask_file(self.root, instance.frame_index) if self._decoded_frames is not None
                      else frame_file(self.root, instance.frame_index, '.json'))
        if annotation.is_file():
            _,masks=_read_polygon_masks(annotation)
            key=(instance.label,instance.track_id)
            if key not in masks:raise ValueError(f"instance {key} missing from {annotation}")
            return masks[key].astype(bool)
        path=instance_file(self.root,instance.frame_index,instance.track_id)
        if path.is_file():
            mask=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
            if mask is None:raise OSError(f"could not read instance mask: {path}")
            if mask.ndim==3:mask=mask[...,0]
            return mask>0
        labels = self.load_label_mask(instance.label, instance.frame_index)
        if instance.label == "sow":
            return labels > 0
        return labels == instance.track_id

    def clear_frame_caches(self) -> None:
        """Release full-resolution image and label rasters after compact crops are built."""
        self._image_cache.clear()
        self._mask_cache.clear()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _read_polygon_masks(path):
    document=json.loads(Path(path).read_text(encoding="utf-8-sig"))
    masks={}
    for shape in document["shapes"]:
        label=str(shape["label"])
        if label not in {"piglet","sow"}:continue
        key=(label,int(shape["group_id"]))
        mask=masks.setdefault(key,np.zeros((document["imageHeight"],document["imageWidth"]),np.uint8))
        cv2.fillPoly(mask,[np.rint(np.asarray(shape["points"])).astype(np.int32)],1)
    return document,masks


def _scan_polygon_reference(clip_dir,requested):
    images=sorted((clip_dir/"frames").glob("*.jpg"))
    annotations=sorted((clip_dir/"frames").glob("*.json"))
    if {p.stem for p in images}!={p.stem for p in annotations}:
        raise ValueError("reference image/annotation pairs are incomplete")
    if [dataset_frame_index(p) for p in images]!=list(range(len(images))):
        raise ValueError("reference frame indices must be contiguous")
    instances=[]
    for path in annotations:
        document,masks=_read_polygon_masks(path)
        for (label,identity),mask in sorted(masks.items()):
            if label not in requested or not mask.any():continue
            ys,xs=np.nonzero(mask)
            instances.append(AnnotatedInstance(dataset_frame_index(path),identity,label,1.0,
                (float(xs.min()),float(ys.min()),float(xs.max()+1),float(ys.max()+1)),
                int(mask.sum()),path.with_suffix(".jpg").name))
    if not instances:raise ValueError("reference contains no requested instances")
    return AnnotatedClip(clip_dir,clip_dir.name,len(images),instances)


def _scan_video_reference(clip_dir, decoded_frames, requested, ffmpeg):
    meta = extract_video_frames(clip_dir, decoded_frames, ffmpeg)
    count = int(meta['frame_count'])
    instances = []
    if (clip_dir / 'manual_masks').is_dir():
        for fi in range(count):
            path = manual_mask_file(clip_dir, fi)
            document, masks = _read_polygon_masks(path)
            for (label, identity), mask in sorted(masks.items()):
                if label not in requested or not mask.any():
                    continue
                x, y, w, h = cv2.boundingRect(mask)
                instances.append(AnnotatedInstance(fi, identity, label, 1.0, (x, y, x+w, y+h),
                    cv2.countNonZero(mask), f'{fi:06d}.jpg'))
    elif (clip_dir / 'instance_masks').is_dir():
        if 'piglet' in requested:
            expected_shape = (int(meta['height']), int(meta['width']))
            for folder in sorted(p for p in (clip_dir / 'instance_masks').iterdir() if p.is_dir()):
                fi = dataset_frame_index(folder)
                if not 0 <= fi < count:
                    raise ValueError(f'mask frame index outside clip: {folder.name}')
                for path in sorted(folder.glob('*.png')):
                    identity = dataset_frame_index(path)
                    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                    if mask is not None and mask.ndim == 3 and mask.shape[2] == 1:
                        mask = mask[..., 0]
                    if mask is None or mask.shape != expected_shape:
                        raise ValueError(f'invalid SAM3 mask: {path}')
                    x, y, w, h = cv2.boundingRect(mask)
                    if w == 0 or h == 0:
                        raise ValueError(f'empty SAM3 mask: {path}')
                    instances.append(AnnotatedInstance(fi, identity, 'piglet', 1.0, (x, y, x+w, y+h),
                        cv2.countNonZero(mask), f'{fi:06d}.jpg'))
    if not instances:
        raise ValueError(f'no instances for labels {sorted(requested)}')
    if len({(i.frame_index, i.label, i.track_id) for i in instances}) != len(instances):
        raise ValueError('duplicate mask instance in a frame')
    return AnnotatedClip(clip_dir, clip_dir.name, count, instances, decoded_frames=decoded_frames)


def _reject_stale_canonical_outputs(clip_dir: Path, completion_path: Path) -> None:
    """Reject compiled masks/tracks after a human annotation JSON has changed."""
    if not completion_path.is_file():
        return
    frame_json_paths = list((clip_dir / "frames").glob("*.json"))
    if not frame_json_paths:
        return
    latest_json_path = max(frame_json_paths, key=lambda path: path.stat().st_mtime_ns)
    latest_json_mtime_ns = latest_json_path.stat().st_mtime_ns
    completion_mtime_ns = completion_path.stat().st_mtime_ns
    tolerance_ns = int(CANONICAL_MTIME_TOLERANCE_SECONDS * 1_000_000_000)
    if latest_json_mtime_ns > completion_mtime_ns + tolerance_ns:
        age_seconds = (latest_json_mtime_ns - completion_mtime_ns) / 1_000_000_000
        raise ValueError(
            "canonical tracks/masks are stale: "
            f"frames/{latest_json_path.name} is {age_seconds:.3f}s newer than tracking_complete.json; "
            "update the dataset annotations first"
        )


def scan_anylabeling_split(
    dataset_root: str | Path,
    split: str,
    labels: Sequence[str] = ("piglet",),
    require_complete: bool = True,
    *,
    decoded_root: Path | None = None,
    ffmpeg: str = 'ffmpeg',
) -> ClipScan:
    """Scan complete teacher exports and reject malformed clips with explicit reasons."""
    split_root = Path(dataset_root).expanduser().resolve() / split
    if not split_root.is_dir():
        raise FileNotFoundError(f"dataset split does not exist: {split_root}")
    requested = {str(label) for label in labels}
    clips: list[AnnotatedClip] = []
    skipped: list[tuple[str, str]] = []
    for clip_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
        completion_path = clip_dir / "annotation_info.json"
        tracks_path = clip_dir / "tracks.jsonl"
        try:
            if clip_video(clip_dir).is_file():
                if decoded_root is None:
                    raise ValueError('decoded_root is required for a video-based DS3 split')
                clips.append(_scan_video_reference(clip_dir, decoded_root / split / clip_dir.name,
                                                   requested, ffmpeg))
                continue
            if not tracks_path.is_file() and any((clip_dir/"frames").glob("*.json")):
                clips.append(_scan_polygon_reference(clip_dir,requested))
                continue
            if require_complete and not completion_path.is_file():
                raise ValueError("annotation_info.json is missing")
            if not tracks_path.is_file() or tracks_path.stat().st_size == 0:
                raise ValueError("tracks.jsonl is missing or empty")
            completion = {}
            if completion_path.is_file():
                completion = json.loads(completion_path.read_text(encoding="utf-8"))
            _reject_stale_canonical_outputs(clip_dir, completion_path)
            frame_count = int(completion.get("frame_count", 0))
            instances: list[AnnotatedInstance] = []
            with tracks_path.open(encoding="utf-8") as file:
                for line_number, line in enumerate(file, 1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    label = str(row.get("label", ""))
                    if label not in requested or not bool(row.get("present", True)):
                        continue
                    bbox = row.get("bbox_xyxy")
                    if not isinstance(bbox, list) or len(bbox) != 4:
                        raise ValueError(f"tracks.jsonl line {line_number} has invalid bbox_xyxy")
                    frame_index = int(row["frame_index"])
                    frame_count = max(frame_count, frame_index + 1)
                    instances.append(
                        AnnotatedInstance(
                            frame_index=frame_index,
                            track_id=int(row["track_id"]),
                            label=label,
                            score=_safe_float(
                                row.get("finetuned_score"), _safe_float(row.get("object_probability"), 1.0)
                            ),
                            bbox_xyxy=tuple(float(x) for x in bbox),
                            area_px=int(row.get("area_px", 0)),
                            image_name=str(row.get("image", f"{frame_index:06d}.jpg")),
                        )
                    )
            if not instances:
                raise ValueError(f"no instances for labels {sorted(requested)}")
            missing_frames = sorted(
                {
                    item.frame_index
                    for item in instances
                    if not (clip_dir / "frames" / item.image_name).is_file()
                }
            )
            if missing_frames:
                raise ValueError(f"frame images are missing for indexes {missing_frames[:5]}")
            required_mask_dirs = {
                "piglet": "piglet_id_masks",
                "sow": "sow_masks",
            }
            for label in {item.label for item in instances}:
                mask_dir = clip_dir / required_mask_dirs.get(label, "id_masks")
                if not mask_dir.is_dir() and not (clip_dir/"instance_masks").is_dir():
                    raise ValueError(f"mask directory is missing: {mask_dir.name}")
            clips.append(AnnotatedClip(clip_dir, clip_dir.name, frame_count, instances))
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            skipped.append((clip_dir.name, str(exc)))
    return ClipScan(clips, skipped)


def _expanded_box(
    bbox_xyxy: Sequence[float], image_shape: tuple[int, int], padding: float
) -> tuple[int, int, int, int]:
    height, width = image_shape
    x1, y1, x2, y2 = (float(x) for x in bbox_xyxy)
    box_width = max(x2 - x1, 1.0)
    box_height = max(y2 - y1, 1.0)
    x1 = max(0, int(math.floor(x1 - padding * box_width)))
    y1 = max(0, int(math.floor(y1 - padding * box_height)))
    x2 = min(width, int(math.ceil(x2 + padding * box_width)))
    y2 = min(height, int(math.ceil(y2 + padding * box_height)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"empty bbox after clipping: {bbox_xyxy}")
    return x1, y1, x2, y2


def extract_masked_crop(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    bbox_xyxy: Sequence[float],
    output_size: tuple[int, int] = DEFAULT_CROP_SIZE,
    padding: float = 0.10,
) -> torch.Tensor:
    """Create a normalized RGB+mask crop, shaped ``(4, height, width)``."""
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"expected BGR image with three channels, got {image_bgr.shape}")
    binary = np.asarray(mask, dtype=bool)
    if binary.shape != image_bgr.shape[:2]:
        raise ValueError(f"mask/image shape mismatch: {binary.shape} vs {image_bgr.shape[:2]}")
    x1, y1, x2, y2 = _expanded_box(bbox_xyxy, binary.shape, padding)
    crop_mask = binary[y1:y2, x1:x2]
    crop_rgb = cv2.cvtColor(image_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
    height, width = output_size
    crop_rgb = cv2.resize(crop_rgb, (width, height), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    crop_mask = cv2.resize(
        crop_mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
    ).astype(np.float32)
    # Zeroing the background prevents the encoder from identifying a piglet by pen location.
    crop_rgb *= crop_mask[..., None]
    mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
    std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
    crop_rgb = (crop_rgb - mean) / std
    crop_rgb *= crop_mask[..., None]
    tensor = np.concatenate((crop_rgb.transpose(2, 0, 1), crop_mask[None]), axis=0)
    return torch.from_numpy(np.ascontiguousarray(tensor, dtype=np.float32))


def observation_from_mask(
    observation_id: int,
    frame_index: int,
    class_id: int,
    label: str,
    score: float,
    bbox_xyxy: Sequence[float],
    mask: np.ndarray,
    gt_track_id: int | None = None,
    payload: Any = None,
    spatial_size: tuple[int, int] = (96, 160),
    shape_size: tuple[int, int] = (48, 96),
) -> MaskObservation:
    """Convert a full-resolution instance mask to a compact tracking observation."""
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2 or not binary.any():
        raise ValueError("instance mask must be a non-empty 2D array")
    ys, xs = np.nonzero(binary)
    area = float(len(xs))
    centroid = np.asarray((float(xs.mean()), float(ys.mean())), dtype=np.float32)
    supplied_bbox = np.asarray(tuple(float(x) for x in bbox_xyxy), dtype=np.float32)
    if supplied_bbox.shape != (4,):
        raise ValueError("bbox_xyxy must contain four values")
    # Raster bounds are authoritative. Teacher JSON uses inclusive maxima while
    # YOLO boxes are continuous/exclusive, so deriving an exclusive box here
    # removes that format ambiguity and keeps training/inference identical.
    bbox = np.asarray((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1), dtype=np.float32)
    x1, y1, x2, y2 = _expanded_box(bbox, binary.shape, 0.0)
    bbox_area = max(float((x2 - x1) * (y2 - y1)), 1.0)
    spatial = cv2.resize(
        binary.astype(np.uint8), (spatial_size[1], spatial_size[0]), interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    local = binary[y1:y2, x1:x2]
    shape = cv2.resize(
        local.astype(np.uint8), (shape_size[1], shape_size[0]), interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    return MaskObservation(
        observation_id=int(observation_id),
        frame_index=int(frame_index),
        class_id=int(class_id),
        label=str(label),
        score=float(score),
        bbox_xyxy=bbox,
        area=area,
        centroid_xy=centroid,
        frame_shape=tuple(int(x) for x in binary.shape),
        spatial_mask=spatial,
        shape_mask=shape,
        fill_ratio=min(area / bbox_area, 1.0),
        gt_track_id=gt_track_id,
        payload=payload,
    )


def _binary_iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = np.logical_and(first, second).sum(dtype=np.int64)
    union = np.logical_or(first, second).sum(dtype=np.int64)
    return float(intersection / union) if union else 0.0


def _bbox_iou(first: np.ndarray, second: np.ndarray) -> float:
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
    first_area = max(float(first[2] - first[0]), 0.0) * max(float(first[3] - first[1]), 0.0)
    second_area = max(float(second[2] - second[0]), 0.0) * max(float(second[3] - second[1]), 0.0)
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def pair_geometry_features(first: MaskObservation, second: MaskObservation) -> np.ndarray:
    """Return bounded, symmetric mask/geometry features for an observation pair."""
    if first.frame_shape != second.frame_shape:
        raise ValueError(f"frame shape changed inside a sequence: {first.frame_shape} vs {second.frame_shape}")
    height, width = first.frame_shape
    diagonal = max(math.hypot(width, height), 1.0)
    center_distance = float(np.linalg.norm(first.centroid_xy - second.centroid_xy))
    object_scale = max(math.sqrt((first.area + second.area) / 2.0), 1.0)
    first_width = max(float(first.bbox_xyxy[2] - first.bbox_xyxy[0]), 1.0)
    first_height = max(float(first.bbox_xyxy[3] - first.bbox_xyxy[1]), 1.0)
    second_width = max(float(second.bbox_xyxy[2] - second.bbox_xyxy[0]), 1.0)
    second_height = max(float(second.bbox_xyxy[3] - second.bbox_xyxy[1]), 1.0)
    return np.asarray(
        (
            _binary_iou(first.spatial_mask, second.spatial_mask),
            _binary_iou(first.shape_mask, second.shape_mask),
            _bbox_iou(first.bbox_xyxy, second.bbox_xyxy),
            min(center_distance / diagonal, 1.0),
            min(center_distance / object_scale / 10.0, 1.0),
            min(abs(math.log(max(first.area, 1.0) / max(second.area, 1.0))) / 4.0, 1.0),
            min(abs(math.log((first_width / first_height) / (second_width / second_height))) / 3.0, 1.0),
            min(abs(first.fill_ratio - second.fill_ratio), 1.0),
            min(abs(second.frame_index - first.frame_index) / 10.0, 1.0),
        ),
        dtype=np.float32,
    )


class MaskAppearanceEncoder(nn.Module):
    """RGB+mask appearance encoder using OSNet-x0.25."""

    def __init__(
        self, embedding_dim: int = 128, width: int = 24, architecture: str = "osnet_x0_25"
    ) -> None:
        super().__init__()
        self.architecture = architecture
        if architecture != "osnet_x0_25":
            raise ValueError(f"unsupported appearance encoder: {architecture}")
        self.features=OSNetX025FeatureExtractor(in_channels=4)
        feature_dim=self.features.output_dim
        self.pool=nn.Identity()
        self.projection = nn.Sequential(nn.Linear(feature_dim, embedding_dim, bias=False), nn.LayerNorm(embedding_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(self.pool(self.features(x))), dim=1)


class MaskAwareAssociationModel(nn.Module):
    """Siamese masked-appearance encoder plus a symmetric edge classifier."""

    def __init__(
        self,
        embedding_dim: int = 128,
        encoder_width: int = 24,
        hidden_dim: int = 192,
        encoder_arch: str = "osnet_x0_25",
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.encoder_width = int(encoder_width)
        self.hidden_dim = int(hidden_dim)
        self.encoder_arch = str(encoder_arch)
        self.encoder = MaskAppearanceEncoder(self.embedding_dim, self.encoder_width, self.encoder_arch)
        pair_dim = 2 * self.embedding_dim + len(PAIR_FEATURE_NAMES)
        self.edge_head = nn.Sequential(
            nn.Linear(pair_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.SiLU(inplace=True),
            nn.Linear(self.hidden_dim // 2, 1),
        )

    @property
    def model_config(self) -> dict[str, int]:
        return {
            "embedding_dim": self.embedding_dim,
            "encoder_width": self.encoder_width,
            "hidden_dim": self.hidden_dim,
            "encoder_arch": self.encoder_arch,
        }

    def encode(self, crops: torch.Tensor) -> torch.Tensor:
        return self.encoder(crops)

    def score_embeddings(
        self, first_embeddings: torch.Tensor, second_embeddings: torch.Tensor, geometry: torch.Tensor
    ) -> torch.Tensor:
        pair = torch.cat(
            (torch.abs(first_embeddings - second_embeddings), first_embeddings * second_embeddings, geometry), dim=1
        )
        return self.edge_head(pair).squeeze(1)

    def forward(
        self, first_crops: torch.Tensor, second_crops: torch.Tensor, geometry: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        first_embeddings = self.encode(first_crops)
        second_embeddings = self.encode(second_crops)
        logits = self.score_embeddings(first_embeddings, second_embeddings, geometry)
        return logits, first_embeddings, second_embeddings


def save_association_checkpoint(
    path: str | Path,
    model: MaskAwareAssociationModel,
    epoch: int,
    metrics: dict[str, float] | None = None,
    training_args: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_config": model.model_config,
            "model_state": model.state_dict(),
            "feature_names": PAIR_FEATURE_NAMES,
            "epoch": int(epoch),
            "metrics": metrics or {},
            "training_args": training_args or {},
        },
        path,
    )


def load_association_checkpoint(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[MaskAwareAssociationModel, dict[str, Any]]:
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    if tuple(checkpoint.get("feature_names", ())) != PAIR_FEATURE_NAMES:
        raise ValueError("checkpoint pair features do not match this code version")
    model_config = dict(checkpoint["model_config"])
    model = MaskAwareAssociationModel(**model_config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model, checkpoint


class AnyLabelingPairDataset(Dataset):
    """Balanced same/different pairs built from teacher track IDs."""

    def __init__(
        self,
        clips: Sequence[AnnotatedClip],
        max_positive_gap: int = 5,
        positives_per_anchor: int = 4,
        negatives_per_positive: int = 2,
        crop_size: tuple[int, int] = DEFAULT_CROP_SIZE,
        augment: bool = False,
        seed: int = 0,
        max_pairs: int = 0,
        instance_cache: int = 512,
    ) -> None:
        if max_positive_gap < 1 or positives_per_anchor < 1 or negatives_per_positive < 0:
            raise ValueError("pair sampling counts are invalid")
        self.clips = list(clips)
        self.crop_size = crop_size
        self.augment = bool(augment)
        self.seed = int(seed)
        self.epoch = 0
        self.instance_cache = max(int(instance_cache), 0)
        self._instance_cache: OrderedDict[
            tuple[int, int, int, str], tuple[MaskObservation, torch.Tensor]
        ] = OrderedDict()
        self.pairs = self._build_pairs(max_positive_gap, positives_per_anchor, negatives_per_positive)
        if max_pairs > 0 and len(self.pairs) > max_pairs:
            rng = np.random.default_rng(self.seed)
            selected = np.sort(rng.choice(len(self.pairs), max_pairs, replace=False))
            self.pairs = [self.pairs[int(index)] for index in selected]

    def _build_pairs(self, max_gap: int, positives_per_anchor: int, negatives_per_positive: int) -> list[PairReference]:
        pairs: list[PairReference] = []
        for clip_index, clip in enumerate(self.clips):
            for (label, track_id), track in sorted(clip.by_track.items()):
                for first_index, first in enumerate(track):
                    candidates = [
                        item for item in track[first_index + 1 :] if 0 < item.frame_index - first.frame_index <= max_gap
                    ]
                    if not candidates:
                        continue
                    # Prefer 0.5/1/2/4-second edges at the current 2 FPS while
                    # remaining valid for clips with occasional absent frames.
                    if len(candidates) > positives_per_anchor:
                        desired_gaps = np.geomspace(1, max_gap, positives_per_anchor)
                        selected: list[AnnotatedInstance] = []
                        for desired in desired_gaps:
                            candidate = min(
                                candidates,
                                key=lambda item: abs((item.frame_index - first.frame_index) - desired),
                            )
                            if candidate not in selected:
                                selected.append(candidate)
                        candidates = selected
                    for second in candidates:
                        pairs.append(PairReference(clip_index, first, second, 1.0))
                        negatives = [
                            item
                            for item in clip.by_frame[second.frame_index]
                            if item.label == label and item.track_id != track_id
                        ]
                        negatives.sort(
                            key=lambda item: _bbox_center_distance(first.bbox_xyxy, item.bbox_xyxy)
                        )
                        for negative in negatives[:negatives_per_positive]:
                            pairs.append(PairReference(clip_index, first, negative, 0.0))
        if not pairs:
            raise ValueError("no trainable observation pairs were found")
        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def set_epoch(self, epoch: int) -> None:
        """Vary deterministic augmentations between epochs."""
        self.epoch = int(epoch)

    def preload_instances(self) -> int:
        """Decode every instance used by this dataset once into the compact float16 LRU."""
        return self.preload_instances_limited(0)

    def unique_instance_count(self) -> int:
        """Return the number of distinct crops referenced by the sampled pairs."""
        return len(self._unique_instances())

    def estimated_cached_instance_bytes(self) -> int:
        """Conservative memory estimate for one cached crop and its compact masks."""
        height, width = self.crop_size
        crop_bytes = 4 * height * width * np.dtype(np.float16).itemsize
        compact_mask_bytes = 96 * 160 + 48 * 96
        return crop_bytes + compact_mask_bytes + 2048

    def _unique_instances(self) -> dict[tuple[int, int, int, str], tuple[int, AnnotatedInstance]]:
        unique: dict[tuple[int, int, int, str], tuple[int, AnnotatedInstance]] = {}
        for pair in self.pairs:
            for instance in (pair.first, pair.second):
                key = (pair.clip_index, instance.frame_index, instance.track_id, instance.label)
                unique[key] = (pair.clip_index, instance)
        return unique

    def preload_instances_limited(self, max_instances: int = 0) -> int:
        """Preload at most ``max_instances`` crops; zero means every referenced crop."""
        if max_instances < 0:
            raise ValueError("max_instances must be non-negative")
        unique = self._unique_instances()
        limit = min(max_instances or len(unique), len(unique))
        self.instance_cache = max(self.instance_cache, limit)
        for key in sorted(unique)[:limit]:
            clip_index, instance = unique[key]
            self._observation(clip_index, self.clips[clip_index], instance)
        for clip in self.clips:
            clip.clear_frame_caches()
        return limit

    def _observation(
        self, clip_index: int, clip: AnnotatedClip, instance: AnnotatedInstance
    ) -> tuple[MaskObservation, torch.Tensor]:
        key = (clip_index, instance.frame_index, instance.track_id, instance.label)
        if key in self._instance_cache:
            self._instance_cache.move_to_end(key)
            observation, crop = self._instance_cache[key]
            return observation, crop.float()
        image = clip.load_image(instance.frame_index)
        loaded_mask = clip.load_instance_mask(instance)
        mask = np.asarray(loaded_mask, dtype=bool)
        if mask.ndim != 2 or not mask.any():
            raise ValueError(
                f"empty teacher mask for {clip.clip_id} frame={instance.frame_index} track={instance.track_id}"
            )
        observation = observation_from_mask(
            observation_id=instance.frame_index * 100000 + instance.track_id,
            frame_index=instance.frame_index,
            class_id=0,
            label=instance.label,
            score=instance.score,
            bbox_xyxy=instance.bbox_xyxy,
            mask=mask,
            gt_track_id=instance.track_id,
        )
        crop = extract_masked_crop(image, mask, observation.bbox_xyxy, self.crop_size)
        if self.instance_cache:
            self._instance_cache[key] = (observation, crop.half())
            while len(self._instance_cache) > self.instance_cache:
                self._instance_cache.popitem(last=False)
        return observation, crop

    def _augment_crop(self, crop: torch.Tensor, index: int, side: int) -> torch.Tensor:
        if not self.augment:
            return crop
        rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + index * 2 + side)
        result = crop.clone()
        if rng.random() < 0.5:
            result = result.flip(-1)
        gain = float(rng.uniform(0.80, 1.20))
        bias = float(rng.uniform(-0.12, 0.12))
        mask = result[3:4]
        result[:3] = (result[:3] * gain + bias) * mask
        if rng.random() < 0.25:
            height, width = result.shape[-2:]
            erase_h = max(1, int(height * rng.uniform(0.05, 0.20)))
            erase_w = max(1, int(width * rng.uniform(0.05, 0.20)))
            top = int(rng.integers(0, max(height - erase_h + 1, 1)))
            left = int(rng.integers(0, max(width - erase_w + 1, 1)))
            result[:, top : top + erase_h, left : left + erase_w] = 0
        return result

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pair = self.pairs[index]
        clip = self.clips[pair.clip_index]
        first_observation, first_crop = self._observation(pair.clip_index, clip, pair.first)
        second_observation, second_crop = self._observation(pair.clip_index, clip, pair.second)
        geometry = torch.from_numpy(pair_geometry_features(first_observation, second_observation))
        return (
            self._augment_crop(first_crop, index, 0),
            self._augment_crop(second_crop, index, 1),
            geometry,
            torch.tensor(pair.target, dtype=torch.float32),
        )


def _bbox_center_distance(first: Sequence[float], second: Sequence[float]) -> float:
    first_center = np.asarray(((first[0] + first[2]) / 2.0, (first[1] + first[3]) / 2.0))
    second_center = np.asarray(((second[0] + second[2]) / 2.0, (second[1] + second[3]) / 2.0))
    return float(np.linalg.norm(first_center - second_center))


def encode_observation_crops(
    model: MaskAwareAssociationModel,
    observations: Sequence[MaskObservation],
    crops: Sequence[torch.Tensor],
    device: torch.device,
    batch_size: int = 64,
) -> None:
    """Attach embeddings to observations in place."""
    if len(observations) != len(crops):
        raise ValueError("observations and crops have different lengths")
    model.eval()
    with torch.inference_mode():
        offset = 0
        while offset < len(crops):
            batch = torch.stack(list(crops[offset : offset + batch_size])).to(device)
            embeddings = model.encode(batch).cpu().numpy()
            for observation, embedding in zip(observations[offset : offset + batch_size], embeddings):
                observation.embedding = np.asarray(embedding, dtype=np.float32)
            offset += batch_size


@dataclass(slots=True)
class _Tracklet:
    tracklet_id: int
    observations: list[MaskObservation]
    embedding: np.ndarray

    @property
    def first(self) -> MaskObservation:
        return self.observations[0]

    @property
    def last(self) -> MaskObservation:
        return self.observations[-1]

    def predicted_center(self, frame_index: int) -> np.ndarray:
        if len(self.observations) < 2:
            return self.last.centroid_xy
        previous, last = self.observations[-2:]
        elapsed = max(last.frame_index - previous.frame_index, 1)
        velocity = (last.centroid_xy - previous.centroid_xy) / elapsed
        return last.centroid_xy + velocity * (frame_index - last.frame_index)


@dataclass(slots=True)
class _IdentityGroup:
    """One or more non-overlapping tracklets that share a final identity."""

    tracklet_ids: set[int]
    observations: list[MaskObservation]
    embedding: np.ndarray

    @property
    def class_id(self) -> int:
        return self.observations[0].class_id

    @property
    def sort_key(self) -> tuple[int, int]:
        return self.observations[0].frame_index, min(self.tracklet_ids)


class MaskAwareOfflineTracker:
    """Online Hungarian association followed by optional global tracklet linking."""

    def __init__(
        self,
        model: MaskAwareAssociationModel,
        device: torch.device | str = "cpu",
        config: TrackerConfig | None = None,
    ) -> None:
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.config = config or TrackerConfig()
        self.identity_cap_summary: dict[str, float | int | None] | None = None

    def _model_probabilities(
        self,
        pairs: Sequence[tuple[MaskObservation, MaskObservation]],
        first_embeddings: Sequence[np.ndarray] | None = None,
        second_embeddings: Sequence[np.ndarray] | None = None,
    ) -> np.ndarray:
        if not pairs:
            return np.empty(0, dtype=np.float32)
        if any(first.embedding is None or second.embedding is None for first, second in pairs):
            raise ValueError("every observation must have an embedding before association")
        if first_embeddings is not None and len(first_embeddings) != len(pairs):
            raise ValueError("first_embeddings and pairs have different lengths")
        if second_embeddings is not None and len(second_embeddings) != len(pairs):
            raise ValueError("second_embeddings and pairs have different lengths")
        first_values = (
            first_embeddings if first_embeddings is not None else [np.asarray(item[0].embedding) for item in pairs]
        )
        second_values = (
            second_embeddings
            if second_embeddings is not None
            else [np.asarray(item[1].embedding) for item in pairs]
        )
        first = torch.from_numpy(np.stack(first_values).astype(np.float32, copy=False)).to(self.device)
        second = torch.from_numpy(np.stack(second_values).astype(np.float32, copy=False)).to(self.device)
        geometry = torch.from_numpy(np.stack([pair_geometry_features(*item) for item in pairs])).to(self.device)
        with torch.inference_mode():
            return self.model.score_embeddings(first, second, geometry).sigmoid().cpu().numpy()

    @staticmethod
    def _normalized_motion_distance(tracklet: _Tracklet, observation: MaskObservation) -> float:
        predicted = tracklet.predicted_center(observation.frame_index)
        height, width = observation.frame_shape
        return float(np.linalg.norm(predicted - observation.centroid_xy) / max(math.hypot(width, height), 1.0))

    def _online_tracklets(self, frames: dict[int, list[MaskObservation]]) -> list[_Tracklet]:
        tracklets: list[_Tracklet] = []
        active: list[_Tracklet] = []
        next_id = 1
        for frame_index in sorted(frames):
            detections = frames[frame_index]
            active = [item for item in active if frame_index - item.last.frame_index <= self.config.max_age]
            candidates: list[tuple[int, int, MaskObservation, MaskObservation]] = []
            for track_index, tracklet in enumerate(active):
                for detection_index, detection in enumerate(detections):
                    if tracklet.last.class_id != detection.class_id:
                        continue
                    motion = self._normalized_motion_distance(tracklet, detection)
                    if motion <= self.config.max_center_distance:
                        candidates.append((track_index, detection_index, tracklet.last, detection))
            score_matrix = np.full((len(active), len(detections)), -1.0, dtype=np.float32)
            probabilities = self._model_probabilities(
                [(item[2], item[3]) for item in candidates],
                first_embeddings=[active[item[0]].embedding for item in candidates],
            )
            for candidate, probability in zip(candidates, probabilities):
                track_index, detection_index = candidate[:2]
                motion = self._normalized_motion_distance(active[track_index], detections[detection_index])
                score_matrix[track_index, detection_index] = float(probability) * math.exp(
                    -self.config.motion_weight * motion
                )
            matched_tracks: set[int] = set()
            matched_detections: set[int] = set()
            if score_matrix.size:
                rows, cols = linear_sum_assignment(1.0 - score_matrix)
                for row, col in zip(rows, cols):
                    if score_matrix[row, col] < self.config.match_threshold:
                        continue
                    tracklet = active[int(row)]
                    detection = detections[int(col)]
                    tracklet.observations.append(detection)
                    embedding = np.asarray(detection.embedding, dtype=np.float32)
                    tracklet.embedding = self.config.embedding_momentum * tracklet.embedding + (
                        1.0 - self.config.embedding_momentum
                    ) * embedding
                    tracklet.embedding /= max(float(np.linalg.norm(tracklet.embedding)), 1e-12)
                    matched_tracks.add(int(row))
                    matched_detections.add(int(col))
            for detection_index, detection in enumerate(detections):
                if detection_index in matched_detections:
                    continue
                embedding = np.asarray(detection.embedding, dtype=np.float32).copy()
                tracklet = _Tracklet(next_id, [detection], embedding)
                next_id += 1
                active.append(tracklet)
                tracklets.append(tracklet)
        return tracklets

    def _offline_links(self, tracklets: Sequence[_Tracklet]) -> dict[int, int]:
        if not self.config.offline or len(tracklets) < 2:
            return {item.tracklet_id: item.tracklet_id for item in tracklets}
        count = len(tracklets)
        if self.config.offline_max_tracklets and count > self.config.offline_max_tracklets:
            raise MemoryError(
                f"offline linking produced {count} tracklets, exceeding the safety limit "
                f"{self.config.offline_max_tracklets}; split the source into shorter clips"
            )
        no_link_cost = 1.0 - self.config.offline_match_threshold
        costs = np.full((count, count + count), 1e6, dtype=np.float32)
        scores = np.full((count, count), -1.0, dtype=np.float32)
        pairs: list[tuple[int, int, MaskObservation, MaskObservation]] = []
        for left_index, left in enumerate(tracklets):
            costs[left_index, count + left_index] = no_link_cost
            for right_index, right in enumerate(tracklets):
                gap = right.first.frame_index - left.last.frame_index
                if not 0 < gap <= self.config.offline_max_gap:
                    continue
                if left.last.class_id != right.first.class_id:
                    continue
                motion = self._normalized_motion_distance(left, right.first)
                if motion <= self.config.offline_max_center_distance:
                    pairs.append((left_index, right_index, left.last, right.first))
        probabilities = self._model_probabilities(
            [(item[2], item[3]) for item in pairs],
            first_embeddings=[tracklets[item[0]].embedding for item in pairs],
            second_embeddings=[tracklets[item[1]].embedding for item in pairs],
        )
        for candidate, probability in zip(pairs, probabilities):
            left_index, right_index = candidate[:2]
            motion = self._normalized_motion_distance(tracklets[left_index], tracklets[right_index].first)
            score = float(probability) * math.exp(-self.config.motion_weight * motion)
            scores[left_index, right_index] = score
            costs[left_index, right_index] = 1.0 - score
        rows, cols = linear_sum_assignment(costs)
        successor: dict[int, int] = {}
        predecessor: dict[int, int] = {}
        for row, col in zip(rows, cols):
            if col >= count or scores[row, col] < self.config.offline_match_threshold:
                continue
            left_id = tracklets[int(row)].tracklet_id
            right_id = tracklets[int(col)].tracklet_id
            successor[left_id] = right_id
            predecessor[right_id] = left_id
        final: dict[int, int] = {}
        next_final_id = 1
        for tracklet in sorted(tracklets, key=lambda item: (item.first.frame_index, item.tracklet_id)):
            if tracklet.tracklet_id in predecessor:
                continue
            current = tracklet.tracklet_id
            while current not in final:
                final[current] = next_final_id
                if current not in successor:
                    break
                current = successor[current]
            next_final_id += 1
        for tracklet in tracklets:
            if tracklet.tracklet_id not in final:
                final[tracklet.tracklet_id] = next_final_id
                next_final_id += 1
        return final

    @staticmethod
    def _identity_group(tracklets: Sequence[_Tracklet]) -> _IdentityGroup:
        observations = sorted(
            (observation for tracklet in tracklets for observation in tracklet.observations),
            key=lambda item: (item.frame_index, item.observation_id),
        )
        if not observations:
            raise ValueError("an identity group cannot be empty")
        frame_indices = [item.frame_index for item in observations]
        if len(frame_indices) != len(set(frame_indices)):
            raise RuntimeError("one identity was assigned to multiple observations in the same frame")
        values = np.stack([np.asarray(item.embedding, dtype=np.float32) for item in observations])
        embedding = values.mean(axis=0)
        embedding /= max(float(np.linalg.norm(embedding)), 1e-12)
        return _IdentityGroup(
            tracklet_ids={item.tracklet_id for item in tracklets},
            observations=observations,
            embedding=embedding,
        )

    @staticmethod
    def _cap_merge_candidate(
        first: _IdentityGroup, second: _IdentityGroup
    ) -> tuple[MaskObservation, MaskObservation, np.ndarray, np.ndarray] | None:
        """Return the closest chronological boundary when two groups can safely share an ID."""
        if first.class_id != second.class_id:
            return None
        first_frames = {item.frame_index for item in first.observations}
        if any(item.frame_index in first_frames for item in second.observations):
            return None
        ordered = sorted(
            [(item.frame_index, item.observation_id, 0, item) for item in first.observations]
            + [(item.frame_index, item.observation_id, 1, item) for item in second.observations]
        )
        boundary = min(
            (
                (right[0] - left[0], left[1], right[1], left, right)
                for left, right in zip(ordered, ordered[1:])
                if left[2] != right[2]
            ),
            default=None,
        )
        if boundary is None:
            return None
        left, right = boundary[3], boundary[4]
        embeddings = (first.embedding, second.embedding)
        return left[3], right[3], embeddings[left[2]], embeddings[right[2]]

    @staticmethod
    def _merge_identity_groups(first: _IdentityGroup, second: _IdentityGroup) -> _IdentityGroup:
        observations = sorted(
            first.observations + second.observations,
            key=lambda item: (item.frame_index, item.observation_id),
        )
        total = len(observations)
        embedding = (len(first.observations) * first.embedding + len(second.observations) * second.embedding) / total
        embedding /= max(float(np.linalg.norm(embedding)), 1e-12)
        return _IdentityGroup(first.tracklet_ids | second.tracklet_ids, observations, embedding)

    def _cap_final_identities(
        self, tracklets: Sequence[_Tracklet], final_ids: dict[int, int]
    ) -> dict[int, int]:
        """Merge the most compatible non-overlapping groups until the hard identity cap is met."""
        limit = int(self.config.max_identities)
        if limit <= 0:
            self.identity_cap_summary = None
            return final_ids
        grouped: dict[int, list[_Tracklet]] = defaultdict(list)
        for tracklet in tracklets:
            grouped[final_ids[tracklet.tracklet_id]].append(tracklet)
        groups = [self._identity_group(values) for _, values in sorted(grouped.items())]
        groups.sort(key=lambda item: item.sort_key)
        before = len(groups)
        merge_scores: list[float] = []
        while len(groups) > limit:
            candidates: list[
                tuple[int, int, tuple[MaskObservation, MaskObservation, np.ndarray, np.ndarray]]
            ] = []
            for first_index, first in enumerate(groups):
                for second_index in range(first_index + 1, len(groups)):
                    candidate = self._cap_merge_candidate(first, groups[second_index])
                    if candidate is not None:
                        candidates.append((first_index, second_index, candidate))
            if not candidates:
                frame_counts: dict[int, int] = defaultdict(int)
                for group in groups:
                    for observation in group.observations:
                        frame_counts[observation.frame_index] += 1
                simultaneous = max(frame_counts.values(), default=0)
                raise RuntimeError(
                    f"cannot enforce max_identities={limit} without assigning the same ID to multiple "
                    f"observations in one frame; current tracks={len(groups)}, max simultaneous={simultaneous}. "
                    "Increase --conf to remove false positives or raise --max-identities."
                )
            probabilities = self._model_probabilities(
                [(item[2][0], item[2][1]) for item in candidates],
                first_embeddings=[item[2][2] for item in candidates],
                second_embeddings=[item[2][3] for item in candidates],
            )
            best_index = int(np.argmax(probabilities))
            first_index, second_index, _ = candidates[best_index]
            merge_scores.append(float(probabilities[best_index]))
            merged = self._merge_identity_groups(groups[first_index], groups[second_index])
            groups = [
                group
                for index, group in enumerate(groups)
                if index not in (first_index, second_index)
            ]
            groups.append(merged)
            groups.sort(key=lambda item: item.sort_key)
        capped: dict[int, int] = {}
        for identity, group in enumerate(groups, 1):
            for tracklet_id in group.tracklet_ids:
                capped[tracklet_id] = identity
        self.identity_cap_summary = {
            "max_identities": limit,
            "tracks_before": before,
            "tracks_after": len(groups),
            "forced_merges": len(merge_scores),
            "minimum_merge_score": min(merge_scores) if merge_scores else None,
            "mean_merge_score": float(np.mean(merge_scores)) if merge_scores else None,
        }
        return capped

    def track(self, frames: dict[int, list[MaskObservation]]) -> dict[int, int]:
        """Return ``observation_id -> final_track_id`` for one video."""
        seen_ids: set[int] = set()
        for detections in frames.values():
            for detection in detections:
                if detection.observation_id in seen_ids:
                    raise ValueError(f"duplicate observation ID: {detection.observation_id}")
                if detection.embedding is None:
                    raise ValueError(f"observation {detection.observation_id} has no appearance embedding")
                seen_ids.add(detection.observation_id)
        tracklets = self._online_tracklets(frames)
        final_ids = self._offline_links(tracklets)
        final_ids = self._cap_final_identities(tracklets, final_ids)
        assignments: dict[int, int] = {}
        for tracklet in tracklets:
            final_id = final_ids[tracklet.tracklet_id]
            for observation in tracklet.observations:
                assignments[observation.observation_id] = final_id
        if assignments.keys() != seen_ids:
            raise RuntimeError("tracker did not assign every observation exactly once")
        return assignments


def identity_metrics(
    observations: Iterable[MaskObservation], assignments: dict[int, int]
) -> IdentityMetrics:
    """Compute global-ID matching metrics while holding detections and masks fixed."""
    values = [item for item in observations if item.gt_track_id is not None]
    if not values:
        raise ValueError("identity metrics require gt_track_id values")
    gt_ids = sorted({int(item.gt_track_id) for item in values})
    pred_ids = sorted({int(assignments[item.observation_id]) for item in values})
    gt_index = {value: index for index, value in enumerate(gt_ids)}
    pred_index = {value: index for index, value in enumerate(pred_ids)}
    counts = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.int64)
    by_gt: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for item in values:
        predicted = int(assignments[item.observation_id])
        counts[gt_index[int(item.gt_track_id)], pred_index[predicted]] += 1
        by_gt[int(item.gt_track_id)].append((item.frame_index, predicted))
    rows, cols = linear_sum_assignment(-counts)
    idtp = int(counts[rows, cols].sum())
    total = len(values)
    idfp = total - idtp
    idfn = total - idtp
    precision = idtp / (idtp + idfp) if idtp + idfp else 0.0
    recall = idtp / (idtp + idfn) if idtp + idfn else 0.0
    idf1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    switches = 0
    fragmentations = 0
    for sequence in by_gt.values():
        predicted_sequence = [predicted for _, predicted in sorted(sequence)]
        switches += sum(first != second for first, second in zip(predicted_sequence, predicted_sequence[1:]))
        fragmentations += max(len(set(predicted_sequence)) - 1, 0)
    dominant = counts.max(axis=0, initial=0).sum()
    purity = float(dominant / total) if total else 0.0
    return IdentityMetrics(
        idtp=idtp,
        idfp=idfp,
        idfn=idfn,
        idf1=idf1,
        id_precision=precision,
        id_recall=recall,
        id_switches=switches,
        fragmentations=fragmentations,
        gt_tracks=len(gt_ids),
        predicted_tracks=len(pred_ids),
        observations=total,
        track_purity=purity,
    )


def tracker_config_from_dict(values: dict[str, Any] | None) -> TrackerConfig:
    """Construct a checked tracker configuration from serialized CLI values."""
    config = TrackerConfig(**(values or {}))
    for name in (
        "match_threshold",
        "offline_match_threshold",
        "max_center_distance",
        "offline_max_center_distance",
        "embedding_momentum",
    ):
        value = float(getattr(config, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}")
    if (
        config.max_age < 0
        or config.offline_max_gap < 1
        or config.motion_weight < 0
        or config.offline_max_tracklets < 0
        or config.max_identities < 0
    ):
        raise ValueError("tracker age/gap/weight/identity-limit values are invalid")
    return config


def config_as_dict(config: TrackerConfig) -> dict[str, Any]:
    return asdict(config)
