#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run separate finetuned SAM3 sow and piglet models, then merge predictions into
LabelMe JSON files.

Example:
  python third_party/sam3/scripts/predict_sow_piglet_mask.py \
    --input data/DS2_segmentation/test \
    --output-dir working_data/two_model_labelme \
    --overwrite
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for import_path in (REPO_ROOT, SCRIPT_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from sam3_predictions_to_labelme import (  # noqa: E402
    Shape,
    compact_group_ids_by_label,
    detect_default_device,
    iter_images,
    load_finetuned_model,
    make_labelme_json,
    masks_to_labelme_shapes,
    tensor_to_numpy,
    validate_checkpoint_file,
    write_json,
)


@dataclass(frozen=True)
class ModelSpec:
    label: str
    prompt: str
    checkpoint: Path
    confidence_threshold: float


@dataclass
class InstanceCandidate:
    label: str
    mask: np.ndarray
    box: np.ndarray
    score: float
    source_index: int

    @property
    def area(self) -> int:
        return int(self.mask.sum())


def normalize_masks(masks: np.ndarray, mask_threshold: float) -> np.ndarray:
    masks = np.asarray(masks)
    if masks.ndim == 2:
        masks = masks[None, ...]
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0, :, :]
    if masks.ndim != 3:
        raise ValueError(f"Expected masks with shape [N,H,W] or [N,1,H,W], got {masks.shape}")

    if masks.dtype == np.bool_:
        return masks

    masks = masks.astype(np.float32)
    finite_values = masks[np.isfinite(masks)]
    if finite_values.size and (float(finite_values.min()) < 0.0 or float(finite_values.max()) > 1.0):
        masks = 1.0 / (1.0 + np.exp(-masks))
    return masks > mask_threshold


def build_instance_candidates(
    masks: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    label: str,
    args: argparse.Namespace,
) -> List[InstanceCandidate]:
    masks_bin = normalize_masks(masks, args.mask_threshold)
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)

    candidates: List[InstanceCandidate] = []
    for idx, mask in enumerate(masks_bin):
        score = 0.0 if idx >= len(scores) else float(scores[idx])
        if score < args.score_threshold:
            continue
        area = int(mask.sum())
        if area < args.min_mask_pixels:
            continue
        box = boxes[idx] if idx < len(boxes) else np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        candidates.append(
            InstanceCandidate(
                label=label,
                mask=mask,
                box=np.asarray(box, dtype=np.float32),
                score=score,
                source_index=idx,
            )
        )
    return candidates


def pairwise_mask_stats(a: InstanceCandidate, b: InstanceCandidate) -> tuple[int, float, float]:
    intersection = int(np.logical_and(a.mask, b.mask).sum())
    if intersection == 0:
        return 0, 0.0, 0.0
    union = a.area + b.area - intersection
    iou = intersection / union if union > 0 else 0.0
    containment = intersection / min(a.area, b.area)
    return intersection, iou, containment


def candidate_rank(candidate: InstanceCandidate) -> tuple[float, int]:
    return candidate.score, candidate.area


def sort_instance_candidates(candidates: Sequence[InstanceCandidate]) -> List[InstanceCandidate]:
    return sorted(candidates, key=candidate_rank, reverse=True)


def mark_lower_ranked_duplicate(
    keep_removed: set[int],
    candidates: Sequence[InstanceCandidate],
    idx_a: int,
    idx_b: int,
) -> None:
    if candidate_rank(candidates[idx_a]) >= candidate_rank(candidates[idx_b]):
        keep_removed.add(idx_b)
    else:
        keep_removed.add(idx_a)


def find_contained_children(
    candidates: Sequence[InstanceCandidate],
    large_idx: int,
    args: argparse.Namespace,
) -> List[int]:
    large = candidates[large_idx]
    children = []
    for idx, candidate in enumerate(candidates):
        if idx == large_idx or candidate.area >= large.area:
            continue
        if candidate.area / max(large.area, 1) > args.child_max_area_ratio:
            continue
        intersection = int(np.logical_and(large.mask, candidate.mask).sum())
        if intersection / max(candidate.area, 1) >= args.child_containment_threshold:
            children.append(idx)
    return children


def children_are_independent(candidates: Sequence[InstanceCandidate], child_indexes: Sequence[int], args: argparse.Namespace) -> bool:
    for pos, idx_a in enumerate(child_indexes):
        for idx_b in child_indexes[pos + 1 :]:
            _, iou, _ = pairwise_mask_stats(candidates[idx_a], candidates[idx_b])
            if iou > args.merge_child_max_iou:
                return False
    return True


def deduplicate_instance_candidates(
    candidates: Sequence[InstanceCandidate],
    args: argparse.Namespace,
) -> List[InstanceCandidate]:
    if not args.enable_mask_dedup or len(candidates) <= 1:
        return sort_instance_candidates(candidates)

    removed: set[int] = set()

    for idx_a, candidate_a in enumerate(candidates):
        if idx_a in removed:
            continue
        for idx_b in range(idx_a + 1, len(candidates)):
            if idx_b in removed:
                continue
            candidate_b = candidates[idx_b]
            _, iou, containment = pairwise_mask_stats(candidate_a, candidate_b)
            if iou >= args.dedupe_iou_threshold:
                mark_lower_ranked_duplicate(removed, candidates, idx_a, idx_b)
                if idx_a in removed:
                    break
                continue

            smaller_idx, larger_idx = (idx_a, idx_b)
            if candidate_a.area > candidate_b.area:
                smaller_idx, larger_idx = idx_b, idx_a
            smaller = candidates[smaller_idx]
            larger = candidates[larger_idx]
            area_ratio = smaller.area / max(larger.area, 1)
            if containment >= args.child_containment_threshold and area_ratio <= args.child_max_area_ratio:
                children = [idx for idx in find_contained_children(candidates, larger_idx, args) if idx not in removed]
                if len(children) >= args.merge_min_children and children_are_independent(candidates, children, args):
                    removed.add(larger_idx)
                else:
                    removed.add(smaller_idx)
                if idx_a in removed:
                    break

    kept = [candidate for idx, candidate in enumerate(candidates) if idx not in removed]
    return sort_instance_candidates(kept)


def candidates_to_labelme_shapes(
    candidates: Sequence[InstanceCandidate],
    width: int,
    height: int,
    args: argparse.Namespace,
) -> List[Shape]:
    if not candidates:
        return []
    masks = np.stack([candidate.mask for candidate in candidates], axis=0)
    boxes = np.stack([candidate.box for candidate in candidates], axis=0)
    scores = np.asarray([candidate.score for candidate in candidates], dtype=np.float32)
    labels = [candidate.label for candidate in candidates]
    max_polygons_per_mask = max_polygons_per_mask_for_label(labels[0], args)
    return masks_to_labelme_shapes(
        masks=masks,
        boxes_xyxy=boxes,
        scores=scores,
        labels=labels,
        width=width,
        height=height,
        score_threshold=args.score_threshold,
        mask_threshold=0.5,
        min_area=args.min_area,
        epsilon_ratio=args.epsilon_ratio,
        max_points=args.max_points,
        max_polygons_per_mask=max_polygons_per_mask,
    )


def count_shape_groups(shapes: Sequence[Shape]) -> int:
    return len({shape.get("group_id") for shape in shapes})


def count_groups_by_label(shapes: Sequence[Shape]) -> dict[str, int]:
    grouped: dict[str, set] = {}
    for shape in shapes:
        grouped.setdefault(str(shape.get("label", "")), set()).add(shape.get("group_id"))
    return {label: len(group_ids) for label, group_ids in grouped.items()}


def run_model_on_image(
    processor,
    image: Image.Image,
    label: str,
    prompt: str,
    width: int,
    height: int,
    args: argparse.Namespace,
) -> List[Shape]:
    state = processor.set_image(image)
    output = processor.set_text_prompt(prompt=prompt, state=state)

    masks = tensor_to_numpy(output.get("masks_logits", output["masks"]))
    boxes = tensor_to_numpy(output["boxes"])
    scores = tensor_to_numpy(output["scores"])
    raw_candidates = build_instance_candidates(masks, boxes, scores, label, args)
    deduped_candidates = deduplicate_instance_candidates(raw_candidates, args)
    if len(raw_candidates) != len(deduped_candidates):
        print(f"  {label}: deduped {len(raw_candidates)} -> {len(deduped_candidates)} instance(s)")
    label_limit = object_count_limit_for_label(label, args)
    if label_limit is not None and len(deduped_candidates) > label_limit:
        print(f"  {label}: capped {len(deduped_candidates)} -> {label_limit} instance(s)")
        deduped_candidates = deduped_candidates[:label_limit]
    return candidates_to_labelme_shapes(deduped_candidates, width, height, args)


def object_count_limits_from_args(args: argparse.Namespace) -> dict:
    limits = {"sow": 1}
    if not args.no_object_count_limits:
        limits["piglet"] = 13
    if args.max_sow is not None:
        if args.max_sow >= 0:
            limits["sow"] = args.max_sow
        else:
            limits.pop("sow", None)
    if args.max_piglet is not None:
        if args.max_piglet >= 0:
            limits["piglet"] = args.max_piglet
        else:
            limits.pop("piglet", None)
    return limits


def object_count_limit_for_label(label: str, args: argparse.Namespace) -> Optional[int]:
    return object_count_limits_from_args(args).get(label)


def max_polygons_per_mask_for_label(label: str, args: argparse.Namespace) -> int:
    if label == "sow":
        return args.sow_max_polygons_per_mask
    if label == "piglet" and args.piglet_max_polygons_per_mask is not None:
        return args.piglet_max_polygons_per_mask
    return args.max_polygons_per_mask


def resolve_optional_threshold(value: Optional[float], fallback: float) -> float:
    return fallback if value is None else value


def run_spec_on_images(
    spec: ModelSpec,
    image_paths: Sequence[Path],
    args: argparse.Namespace,
    shapes_by_image: dict[Path, List[Shape]],
) -> None:
    import torch
    from sam3.model.sam3_image_processor import Sam3Processor

    validate_checkpoint_file(spec.checkpoint)
    print(f"Loading {spec.label} model: {spec.checkpoint}")
    model = load_finetuned_model(spec.checkpoint, Path(args.bpe_path), args.device)
    processor = Sam3Processor(
        model,
        resolution=args.resolution,
        device=args.device,
        confidence_threshold=spec.confidence_threshold,
    )

    with torch.inference_mode():
        for image_path in image_paths:
            with Image.open(image_path) as src_image:
                image = src_image.convert("RGB")
            width, height = image.size
            model_shapes = run_model_on_image(
                processor=processor,
                image=image,
                label=spec.label,
                prompt=spec.prompt,
                width=width,
                height=height,
                args=args,
            )
            shapes_by_image[image_path].extend(model_shapes)
            print(
                f"{image_path.name}: {spec.label} -> "
                f"{count_shape_groups(model_shapes)} instance(s), {len(model_shapes)} polygon shape(s)"
            )

    del processor
    del model
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict with separate sow/piglet SAM3 checkpoints and write merged LabelMe JSON files."
    )
    parser.add_argument("--input", required=True, help="Image file or image directory.")
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "sam3_out" / "two_model_labelme_json"),
        help="Output directory for merged LabelMe JSON files.",
    )
    parser.add_argument(
        "--sow-checkpoint",
        default=str(REPO_ROOT.parents[1] / "models" / "sam3" / "sow_finetuned.pt"),
    )
    parser.add_argument(
        "--piglet-checkpoint",
        default=str(REPO_ROOT.parents[1] / "models" / "sam3" / "piglet_finetuned.pt"),
    )
    parser.add_argument("--sow-prompt", default="sow", help="Text prompt for the sow model.")
    parser.add_argument("--piglet-prompt", default="piglet", help="Text prompt for the piglet model.")
    parser.add_argument("--bpe-path", default=str(REPO_ROOT / "assets" / "bpe_simple_vocab_16e6.txt.gz"))
    parser.add_argument("--device", default=detect_default_device())
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="SAM3 processor confidence threshold for both models unless overridden below.",
    )
    parser.add_argument("--sow-confidence-threshold", type=float, default=None)
    parser.add_argument("--piglet-confidence-threshold", type=float, default=None)

    parser.add_argument("--score-threshold", type=float, default=0.0, help="Drop converted predictions below this score.")
    parser.add_argument("--mask-threshold", type=float, default=0.5, help="Threshold for probability/logit masks.")
    parser.add_argument("--min-area", type=float, default=40.0, help="Minimum contour area in pixels.")
    parser.add_argument(
        "--min-mask-pixels",
        type=int,
        default=40,
        help="Drop raw candidate masks with fewer foreground pixels before deduplication.",
    )
    parser.add_argument("--epsilon-ratio", type=float, default=0.002, help="Contour simplification ratio.")
    parser.add_argument("--max-points", type=int, default=300, help="Soft maximum polygon points per contour.")
    parser.add_argument(
        "--max-polygons-per-mask",
        type=int,
        default=1,
        help="Keep at most this many polygons from each mask. Use 0 to keep all contours.",
    )
    parser.add_argument(
        "--sow-max-polygons-per-mask",
        type=int,
        default=1,
        help="Keep at most this many polygons from each sow mask. Default is 1 for stricter sow output.",
    )
    parser.add_argument(
        "--piglet-max-polygons-per-mask",
        type=int,
        default=None,
        help="Optional piglet-specific polygon limit. Defaults to --max-polygons-per-mask.",
    )
    parser.add_argument(
        "--enable-mask-dedup",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Remove duplicate same-class masks before converting them to LabelMe polygons.",
    )
    parser.add_argument(
        "--dedupe-iou-threshold",
        type=float,
        default=0.65,
        help="If two candidate masks overlap this much by IoU, keep the higher-ranked one.",
    )
    parser.add_argument(
        "--child-containment-threshold",
        type=float,
        default=0.85,
        help="Treat a smaller mask as a duplicate body fragment when this fraction is inside a larger mask.",
    )
    parser.add_argument(
        "--child-max-area-ratio",
        type=float,
        default=0.75,
        help="Only apply child-mask removal when the smaller mask is at most this fraction of the larger mask.",
    )
    parser.add_argument(
        "--merge-min-children",
        type=int,
        default=2,
        help="If a large mask contains at least this many independent smaller masks, drop the large merged mask instead.",
    )
    parser.add_argument(
        "--merge-child-max-iou",
        type=float,
        default=0.25,
        help="Contained child masks are considered independent only when their pairwise IoU is no larger than this.",
    )
    parser.add_argument(
        "--max-sow",
        type=int,
        default=None,
        help="Maximum sow instances kept after conversion. Default is always 1; use -1 for no sow cap.",
    )
    parser.add_argument(
        "--max-piglet",
        type=int,
        default=-1,
        help="Maximum piglet instances kept after conversion. Default is 13; use -1 for no piglet cap.",
    )
    parser.add_argument(
        "--no-object-count-limits",
        action="store_true",
        help="Disable the default piglet cap. The sow cap remains 1 unless --max-sow -1 is used.",
    )
    parser.add_argument("--include-image-data", action="store_true", help="Embed base64 imageData in LabelMe JSON.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing LabelMe JSON files.")
    parser.add_argument("--max-images", type=int, default=0, help="Limit number of images.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    image_paths = list(iter_images(input_path))
    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        raise FileNotFoundError(f"No images found under {input_path}")

    specs = [
        ModelSpec(
            label="sow",
            prompt=args.sow_prompt,
            checkpoint=Path(args.sow_checkpoint),
            confidence_threshold=resolve_optional_threshold(args.sow_confidence_threshold, args.confidence_threshold),
        ),
        ModelSpec(
            label="piglet",
            prompt=args.piglet_prompt,
            checkpoint=Path(args.piglet_checkpoint),
            confidence_threshold=resolve_optional_threshold(args.piglet_confidence_threshold, args.confidence_threshold),
        ),
    ]

    shapes_by_image: dict[Path, List[Shape]] = {image_path: [] for image_path in image_paths}

    for spec in specs:
        run_spec_on_images(spec, image_paths, args, shapes_by_image)

    written = 0
    for image_path in image_paths:
        with Image.open(image_path) as image:
            width, height = image.size

        shapes = shapes_by_image[image_path]
        raw_shape_count = len(shapes)
        shapes = compact_group_ids_by_label(shapes)

        out_path = output_dir / f"{image_path.stem}.json"
        if out_path.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists, use --overwrite to replace it: {out_path}")
        write_json(out_path, make_labelme_json(image_path, width, height, shapes, args.include_image_data))
        written += 1

        suffix = "" if raw_shape_count == len(shapes) else f" (postprocessed from {raw_shape_count})"
        print(
            f"{image_path.name}: merged {count_groups_by_label(shapes)} instance(s), "
            f"{len(shapes)} polygon shape(s){suffix} -> {out_path}"
        )

    print(f"Done. Wrote {written} merged LabelMe JSON file(s).")


if __name__ == "__main__":
    main()
