#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert SAM3 predictions into LabelMe JSON files.
Postprocessing defaults to one sow and at most 13 piglet shapes per image.

Supported inputs:
  1. Run SAM3 image inference directly from images:
       python scripts/sam3_predictions_to_labelme.py --input images --checkpoint ckpt.pt --prompts sow piglet

  2. Convert an agent-style SAM3 prediction JSON with pred_masks/pred_boxes:
       python scripts/sam3_predictions_to_labelme.py --prediction-json pred.json --label piglet

  3. Convert saved mask PNGs such as sam3_out/mask_00.png plus an image:
       python scripts/sam3_predictions_to_labelme.py --mask-dir sam3_out --image image.jpg --label piglet
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_LABELME_VERSION = "5.11.2"
DEFAULT_MAX_SOWS = 1
DEFAULT_MAX_PIGLETS = 13
LABEL_ALIASES = {
    "sow": "sow",
    "sows": "sow",
    "mother": "sow",
    "mother_pig": "sow",
    "母猪": "sow",
    "piglet": "piglet",
    "piglets": "piglet",
    "pig": "piglet",
    "pigs": "piglet",
    "仔猪": "piglet",
}


Point = List[float]
Shape = Dict[str, Any]


def iter_images(path: Path) -> Iterable[Path]:
    if path.is_file():
        if path.suffix.lower() in IMAGE_EXTS:
            yield path
        return
    for image_path in sorted(path.rglob("*")):
        if image_path.suffix.lower() in IMAGE_EXTS:
            yield image_path


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def validate_checkpoint_file(checkpoint_path: Path) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not checkpoint_path.is_file():
        raise ValueError(f"Checkpoint path is not a file: {checkpoint_path}")
    size = checkpoint_path.stat().st_size
    if size == 0:
        raise ValueError(
            "Checkpoint is empty (0 bytes): "
            f"{checkpoint_path}. Re-run training or pass a non-empty checkpoint."
        )


def labelme_shape(
    label: str,
    points: Sequence[Sequence[float]],
    shape_type: str,
    group_id: Optional[int] = None,
    score: Optional[float] = None,
) -> Shape:
    description = "" if score is None else f"score={score:.6f}"
    return {
        "label": label,
        "points": [[float(x), float(y)] for x, y in points],
        "group_id": group_id,
        "description": description,
        "shape_type": shape_type,
        "flags": {},
        "mask": None,
    }


def make_labelme_json(
    image_path: Path,
    width: int,
    height: int,
    shapes: Sequence[Shape],
    include_image_data: bool = False,
) -> Dict[str, Any]:
    image_data = None
    if include_image_data:
        import base64

        image_data = base64.b64encode(image_path.read_bytes()).decode("utf-8")

    return {
        "version": DEFAULT_LABELME_VERSION,
        "flags": {},
        "shapes": list(shapes),
        "imagePath": image_path.name,
        "imageData": image_data,
        "imageHeight": int(height),
        "imageWidth": int(width),
    }


def clamp_box_xyxy(box: Sequence[float], width: int, height: int) -> List[List[float]]:
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    x1 = min(max(x1, 0.0), float(width - 1))
    x2 = min(max(x2, 0.0), float(width - 1))
    y1 = min(max(y1, 0.0), float(height - 1))
    y2 = min(max(y2, 0.0), float(height - 1))
    return [[min(x1, x2), min(y1, y2)], [max(x1, x2), max(y1, y2)]]


def box_xywh_to_xyxy(box: Sequence[float], width: int, height: int, normalized: bool) -> List[float]:
    x, y, w, h = [float(v) for v in box[:4]]
    if normalized:
        x *= width
        w *= width
        y *= height
        h *= height
    return [x, y, x + w, y + h]


def resize_mask_if_needed(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    if mask.shape == (height, width):
        return mask
    pil_mask = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    pil_mask = pil_mask.resize((width, height), resample=Image.Resampling.NEAREST)
    return np.asarray(pil_mask) > 0


def mask_to_polygons(
    mask: np.ndarray,
    width: int,
    height: int,
    min_area: float,
    epsilon_ratio: float,
    max_points: int,
) -> List[List[Point]]:
    mask = resize_mask_if_needed(mask.astype(bool), width, height)
    mask_u8 = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons: List[List[Point]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue

        perimeter = float(cv2.arcLength(contour, True))
        epsilon = max(1.0, epsilon_ratio * perimeter)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        while len(approx) > max_points and epsilon < 0.1 * perimeter:
            epsilon *= 1.5
            approx = cv2.approxPolyDP(contour, epsilon, True)

        points = approx.reshape(-1, 2).astype(float)
        if len(points) < 3:
            continue

        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        polygons.append(points.tolist())

    polygons.sort(key=lambda pts: cv2.contourArea(np.asarray(pts, dtype=np.float32)), reverse=True)
    return polygons


def canonical_label(label: str) -> str:
    return LABEL_ALIASES.get(str(label).strip().lower(), str(label).strip().lower())


def parse_shape_score(shape: Shape) -> float:
    description = str(shape.get("description") or "")
    if not description.startswith("score="):
        return 0.0
    try:
        return float(description.split("=", 1)[1])
    except ValueError:
        return 0.0


def shape_area(shape: Shape) -> float:
    points = np.asarray(shape.get("points", []), dtype=np.float32)
    if len(points) < 2:
        return 0.0
    if shape.get("shape_type") == "rectangle" and len(points) >= 2:
        x1, y1 = points[0]
        x2, y2 = points[1]
        return float(abs(x2 - x1) * abs(y2 - y1))
    if len(points) < 3:
        return 0.0
    return float(abs(cv2.contourArea(points)))


def limit_shapes_by_label(shapes: Sequence[Shape], label_limits: Dict[str, int]) -> List[Shape]:
    """Keep the highest-confidence, largest shapes while preserving final image order."""
    if not label_limits:
        return list(shapes)

    keep_indexes = set(range(len(shapes)))
    grouped_indexes: Dict[str, List[int]] = {}
    for idx, shape in enumerate(shapes):
        label = canonical_label(str(shape.get("label", "")))
        if label in label_limits and label_limits[label] >= 0:
            grouped_indexes.setdefault(label, []).append(idx)

    for label, indexes in grouped_indexes.items():
        limit = label_limits[label]
        if len(indexes) <= limit:
            continue
        ranked = sorted(
            indexes,
            key=lambda idx: (parse_shape_score(shapes[idx]), shape_area(shapes[idx])),
            reverse=True,
        )
        keep_indexes.difference_update(ranked[limit:])

    return [shape for idx, shape in enumerate(shapes) if idx in keep_indexes]


def compact_group_ids_by_label(shapes: Sequence[Shape]) -> List[Shape]:
    """Renumber kept shapes so LabelMe's sidebar count is easy to read."""
    next_group_ids: Dict[str, int] = {}
    group_id_map: Dict[Tuple[str, int], int] = {}
    compacted: List[Shape] = []

    for idx, shape in enumerate(shapes):
        copied = dict(shape)
        label = canonical_label(str(copied.get("label", "")))
        old_group_id = copied.get("group_id")
        if old_group_id is None:
            old_group_id = -(idx + 1)
        key = (label, int(old_group_id))

        if key not in group_id_map:
            next_group_ids[label] = next_group_ids.get(label, 0) + 1
            group_id_map[key] = next_group_ids[label]
        copied["group_id"] = group_id_map[key]
        compacted.append(copied)

    return compacted


def object_count_limits_from_args(args: argparse.Namespace) -> Dict[str, int]:
    limits: Dict[str, int] = {}

    max_sow = args.max_sow
    if max_sow is None and not args.no_object_count_limits:
        max_sow = DEFAULT_MAX_SOWS
    if max_sow is not None and max_sow >= 0:
        limits["sow"] = max_sow

    max_piglet = args.max_piglet
    if max_piglet is None and not args.no_object_count_limits:
        max_piglet = DEFAULT_MAX_PIGLETS
    if max_piglet is not None and max_piglet >= 0:
        limits["piglet"] = max_piglet

    return limits


def masks_to_labelme_shapes(
    masks: np.ndarray,
    boxes_xyxy: Optional[np.ndarray],
    scores: Optional[np.ndarray],
    labels: Sequence[str],
    width: int,
    height: int,
    score_threshold: float,
    mask_threshold: float,
    min_area: float,
    epsilon_ratio: float,
    max_points: int,
    max_polygons_per_mask: int,
) -> List[Shape]:
    masks = np.asarray(masks)
    if masks.ndim == 2:
        masks = masks[None, ...]
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0, :, :]
    if masks.ndim != 3:
        raise ValueError(f"Expected masks with shape [N,H,W] or [N,1,H,W], got {masks.shape}")

    if boxes_xyxy is not None:
        boxes_xyxy = np.asarray(boxes_xyxy, dtype=np.float32)
    if scores is not None:
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)

    shapes: List[Shape] = []
    for idx, raw_mask in enumerate(masks):
        score = None if scores is None or idx >= len(scores) else float(scores[idx])
        if score is not None and score < score_threshold:
            continue

        label = labels[idx] if idx < len(labels) else labels[-1]
        mask = raw_mask.astype(np.float32) > mask_threshold
        group_id = idx + 1
        polygons = mask_to_polygons(mask, width, height, min_area, epsilon_ratio, max_points)
        if max_polygons_per_mask > 0:
            polygons = polygons[:max_polygons_per_mask]

        if polygons:
            for polygon in polygons:
                shapes.append(labelme_shape(label, polygon, "polygon", group_id=group_id, score=score))
            continue

        if boxes_xyxy is not None and idx < len(boxes_xyxy):
            rect = clamp_box_xyxy(boxes_xyxy[idx], width, height)
            shapes.append(labelme_shape(label, rect, "rectangle", group_id=group_id, score=score))

    return shapes


def boxes_to_labelme_shapes(
    boxes_xyxy: np.ndarray,
    scores: Optional[np.ndarray],
    labels: Sequence[str],
    width: int,
    height: int,
    score_threshold: float,
) -> List[Shape]:
    shapes: List[Shape] = []
    for idx, box in enumerate(np.asarray(boxes_xyxy, dtype=np.float32)):
        score = None if scores is None or idx >= len(scores) else float(scores[idx])
        if score is not None and score < score_threshold:
            continue
        label = labels[idx] if idx < len(labels) else labels[-1]
        rect = clamp_box_xyxy(box, width, height)
        shapes.append(labelme_shape(label, rect, "rectangle", group_id=idx + 1, score=score))
    return shapes


def load_finetuned_model(checkpoint_path: Path, bpe_path: Path, device: str):
    validate_checkpoint_file(checkpoint_path)

    import torch

    from sam3.model_builder import build_sam3_image_model

    model = build_sam3_image_model(
        bpe_path=str(bpe_path),
        device="cpu",
        eval_mode=True,
        checkpoint_path=None,
        load_from_HF=False,
        enable_segmentation=True,
    )
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except EOFError as exc:
        raise ValueError(
            "Checkpoint could not be read and appears truncated or corrupt: "
            f"{checkpoint_path}"
        ) from exc
    state_dict = checkpoint.get("model", checkpoint)
    if any(key.startswith("detector.") for key in state_dict):
        state_dict = {
            key.replace("detector.", ""): value
            for key, value in state_dict.items()
            if key.startswith("detector.")
        }
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: missing keys: {len(missing)}")
    if unexpected:
        print(f"Warning: unexpected keys: {len(unexpected)}")
    model.to(device)
    model.eval()
    return model


def tensor_to_numpy(value: Any) -> np.ndarray:
    import torch

    if torch.is_tensor(value):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def run_inference_to_labelme(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    image_paths = list(iter_images(input_path))
    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        raise FileNotFoundError(f"No images found under {input_path}")

    checkpoint_path = Path(args.checkpoint)
    validate_checkpoint_file(checkpoint_path)

    import torch

    from sam3.model.sam3_image_processor import Sam3Processor

    model = load_finetuned_model(checkpoint_path, Path(args.bpe_path), args.device)
    processor = Sam3Processor(
        model,
        resolution=args.resolution,
        device=args.device,
        confidence_threshold=args.confidence_threshold,
    )

    written = 0
    with torch.inference_mode():
        for image_path in image_paths:
            image = Image.open(image_path).convert("RGB")
            width, height = image.size
            shapes: List[Shape] = []

            for prompt in args.prompts:
                state = processor.set_image(image)
                output = processor.set_text_prompt(prompt=prompt, state=state)
                masks = tensor_to_numpy(output.get("masks_logits", output["masks"]))
                boxes = tensor_to_numpy(output["boxes"])
                scores = tensor_to_numpy(output["scores"])
                labels = [prompt] * max(1, len(scores))
                shapes.extend(
                    masks_to_labelme_shapes(
                        masks=masks,
                        boxes_xyxy=boxes,
                        scores=scores,
                        labels=labels,
                        width=width,
                        height=height,
                        score_threshold=args.score_threshold,
                        mask_threshold=args.mask_threshold,
                        min_area=args.min_area,
                        epsilon_ratio=args.epsilon_ratio,
                        max_points=args.max_points,
                        max_polygons_per_mask=args.max_polygons_per_mask,
                    )
                )

            raw_shape_count = len(shapes)
            shapes = limit_shapes_by_label(shapes, object_count_limits_from_args(args))
            shapes = compact_group_ids_by_label(shapes)
            labelme = make_labelme_json(image_path, width, height, shapes, args.include_image_data)
            out_path = output_dir / f"{image_path.stem}.json"
            if out_path.exists() and not args.overwrite:
                raise FileExistsError(f"Output exists, use --overwrite to replace it: {out_path}")
            write_json(out_path, labelme)
            written += 1
            suffix = "" if raw_shape_count == len(shapes) else f" (postprocessed from {raw_shape_count})"
            print(f"{image_path.name}: {len(shapes)} shapes{suffix} -> {out_path}")

    print(f"Done. Wrote {written} LabelMe JSON file(s).")


def decode_rle_masks(prediction: Dict[str, Any]) -> Optional[np.ndarray]:
    pred_masks = prediction.get("pred_masks")
    if not pred_masks:
        return None

    try:
        from pycocotools import mask as mask_utils
    except ImportError as exc:
        raise ImportError("pycocotools is required to decode pred_masks RLE data.") from exc

    height = int(prediction.get("orig_img_h") or prediction.get("imageHeight") or prediction.get("height"))
    width = int(prediction.get("orig_img_w") or prediction.get("imageWidth") or prediction.get("width"))
    decoded = []
    for rle in pred_masks:
        if isinstance(rle, str):
            rle_obj = {"size": [height, width], "counts": rle}
        elif isinstance(rle, dict):
            rle_obj = dict(rle)
            rle_obj.setdefault("size", [height, width])
        else:
            raise ValueError(f"Unsupported RLE mask entry: {type(rle)}")
        decoded.append(mask_utils.decode(rle_obj).astype(bool))
    return np.stack(decoded, axis=0) if decoded else None


def labels_from_prediction(prediction: Dict[str, Any], fallback_label: str, count: int) -> List[str]:
    candidates = prediction.get("pred_labels") or prediction.get("labels") or prediction.get("class_names")
    if isinstance(candidates, list) and candidates:
        labels = [str(item.get("label", item.get("name", item))) if isinstance(item, dict) else str(item) for item in candidates]
        if len(labels) >= count:
            return labels[:count]
    return [fallback_label] * count


def convert_prediction_json_to_labelme(args: argparse.Namespace) -> None:
    prediction_path = Path(args.prediction_json)
    prediction = load_json(prediction_path)
    image_path_value = args.image or prediction.get("original_image_path") or prediction.get("imagePath")
    if not image_path_value:
        raise ValueError("Use --image, or provide original_image_path/imagePath in the prediction JSON.")
    image_path = Path(str(image_path_value))

    if image_path.exists():
        with Image.open(image_path) as image:
            width, height = image.size
    else:
        width = int(prediction.get("orig_img_w") or prediction.get("imageWidth") or prediction.get("width"))
        height = int(prediction.get("orig_img_h") or prediction.get("imageHeight") or prediction.get("height"))

    masks = decode_rle_masks(prediction)
    scores = np.asarray(prediction.get("pred_scores", prediction.get("scores", [])), dtype=np.float32)
    labels = labels_from_prediction(prediction, args.label or prediction_path.stem, len(scores) if len(scores) else 1)

    boxes_xyxy = None
    boxes = prediction.get("pred_boxes") or prediction.get("boxes")
    if boxes:
        normalized = bool(args.normalized_boxes)
        boxes_xyxy = np.asarray([box_xywh_to_xyxy(box, width, height, normalized) for box in boxes], dtype=np.float32)

    if masks is not None:
        shapes = masks_to_labelme_shapes(
            masks=masks,
            boxes_xyxy=boxes_xyxy,
            scores=scores if len(scores) else None,
            labels=labels,
            width=width,
            height=height,
            score_threshold=args.score_threshold,
            mask_threshold=args.mask_threshold,
            min_area=args.min_area,
            epsilon_ratio=args.epsilon_ratio,
            max_points=args.max_points,
            max_polygons_per_mask=args.max_polygons_per_mask,
        )
    elif boxes_xyxy is not None:
        shapes = boxes_to_labelme_shapes(
            boxes_xyxy=boxes_xyxy,
            scores=scores if len(scores) else None,
            labels=labels,
            width=width,
            height=height,
            score_threshold=args.score_threshold,
        )
    else:
        raise ValueError(f"No pred_masks/pred_boxes found in {prediction_path}")

    raw_shape_count = len(shapes)
    shapes = limit_shapes_by_label(shapes, object_count_limits_from_args(args))
    shapes = compact_group_ids_by_label(shapes)
    out_path = Path(args.output_dir) / f"{image_path.stem}.json"
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists, use --overwrite to replace it: {out_path}")
    write_json(out_path, make_labelme_json(image_path, width, height, shapes, args.include_image_data))
    suffix = "" if raw_shape_count == len(shapes) else f" (postprocessed from {raw_shape_count})"
    print(f"{prediction_path.name}: {len(shapes)} shapes{suffix} -> {out_path}")


def read_boxes_scores_csv(path: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if not path.exists():
        return None, None
    boxes: List[List[float]] = []
    scores: List[float] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            boxes.append([float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])])
            scores.append(float(row["score"]))
    return np.asarray(boxes, dtype=np.float32), np.asarray(scores, dtype=np.float32)


def convert_mask_dir_to_labelme(args: argparse.Namespace) -> None:
    if not args.image:
        raise ValueError("--image is required with --mask-dir")

    image_path = Path(args.image)
    with Image.open(image_path) as image:
        width, height = image.size

    mask_dir = Path(args.mask_dir)
    mask_paths = sorted(mask_dir.glob(args.mask_glob))
    if not mask_paths:
        raise FileNotFoundError(f"No masks matched {args.mask_glob!r} under {mask_dir}")

    masks = []
    for mask_path in mask_paths:
        mask = Image.open(mask_path).convert("L")
        masks.append(np.asarray(mask) > 0)

    boxes, scores = read_boxes_scores_csv(Path(args.boxes_csv) if args.boxes_csv else mask_dir / "boxes_scores.csv")
    labels = [args.label or image_path.stem] * len(masks)
    shapes = masks_to_labelme_shapes(
        masks=np.stack(masks, axis=0),
        boxes_xyxy=boxes,
        scores=scores,
        labels=labels,
        width=width,
        height=height,
        score_threshold=args.score_threshold,
        mask_threshold=args.mask_threshold,
        min_area=args.min_area,
        epsilon_ratio=args.epsilon_ratio,
        max_points=args.max_points,
        max_polygons_per_mask=args.max_polygons_per_mask,
    )

    raw_shape_count = len(shapes)
    shapes = limit_shapes_by_label(shapes, object_count_limits_from_args(args))
    shapes = compact_group_ids_by_label(shapes)
    out_path = Path(args.output_dir) / f"{image_path.stem}.json"
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists, use --overwrite to replace it: {out_path}")
    write_json(out_path, make_labelme_json(image_path, width, height, shapes, args.include_image_data))
    suffix = "" if raw_shape_count == len(shapes) else f" (postprocessed from {raw_shape_count})"
    print(f"{len(mask_paths)} masks: {len(shapes)} shapes{suffix} -> {out_path}")


def detect_default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert SAM3 predictions to editable LabelMe JSON files.")

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", help="Image file or image directory. Runs SAM3 inference before conversion.")
    input_group.add_argument("--prediction-json", help="SAM3 prediction JSON containing pred_masks/pred_boxes.")
    input_group.add_argument("--mask-dir", help="Directory containing saved binary mask PNGs such as mask_00.png.")

    parser.add_argument("--image", help="Original image path for --prediction-json or --mask-dir.")
    parser.add_argument("--output-dir", default="sam3_out/labelme_json", help="Output directory for LabelMe JSON files.")
    parser.add_argument("--label", help="Label to use when converting saved predictions.")
    parser.add_argument("--prompts", nargs="+", default=["sow"], help="Text prompts/labels for direct SAM3 inference.")

    parser.add_argument("--checkpoint", default="models/sam3/piglet_finetuned.pt")
    parser.add_argument("--bpe-path", default="third_party/sam3/assets/bpe_simple_vocab_16e6.txt.gz")
    parser.add_argument("--device", default=detect_default_device())
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)

    parser.add_argument("--score-threshold", type=float, default=0.0, help="Drop predictions below this score.")
    parser.add_argument("--mask-threshold", type=float, default=0.5, help="Threshold for probability/logit masks.")
    parser.add_argument("--min-area", type=float, default=20.0, help="Minimum contour area in pixels.")
    parser.add_argument("--epsilon-ratio", type=float, default=0.002, help="Contour simplification ratio.")
    parser.add_argument("--max-points", type=int, default=300, help="Soft maximum polygon points per contour.")
    parser.add_argument(
        "--max-polygons-per-mask",
        type=int,
        default=1,
        help="Keep at most this many polygons from each mask. Use 0 to keep all contours.",
    )
    parser.add_argument(
        "--max-sow",
        type=int,
        default=None,
        help="Maximum sow shapes kept after conversion. Default is 1 unless --no-object-count-limits is used; use -1 for no sow cap.",
    )
    parser.add_argument(
        "--max-piglet",
        type=int,
        default=None,
        help="Maximum piglet shapes kept after conversion. Default is the pen upper bound 13 unless --no-object-count-limits is used; use -1 for no piglet cap.",
    )
    parser.add_argument(
        "--no-object-count-limits",
        action="store_true",
        help="Disable default postprocessing caps. Explicit --max-sow/--max-piglet values still apply.",
    )

    parser.add_argument("--mask-glob", default="mask_*.png", help="Mask filename glob used with --mask-dir.")
    parser.add_argument("--boxes-csv", help="Optional boxes_scores.csv path used with --mask-dir.")
    parser.add_argument(
        "--normalized-boxes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Treat prediction JSON boxes as normalized xywh. Use --no-normalized-boxes for absolute xywh.",
    )
    parser.add_argument("--include-image-data", action="store_true", help="Embed base64 imageData in the LabelMe JSON.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing LabelMe JSON files.")
    parser.add_argument("--max-images", type=int, default=0, help="Limit number of images for direct inference.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input:
        run_inference_to_labelme(args)
    elif args.prediction_json:
        convert_prediction_json_to_labelme(args)
    elif args.mask_dir:
        convert_mask_dir_to_labelme(args)


if __name__ == "__main__":
    main()
