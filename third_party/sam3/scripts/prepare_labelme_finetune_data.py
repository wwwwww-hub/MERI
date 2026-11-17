#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert a LabelMe polygon dataset into a COCO-style dataset split for SAM3 fine-tuning.

Equivalent Python version of the PowerShell script `prepare_labelme_finetune_data.ps1`.

Default input/output:
    input:  finetundata/
    output: finetundata_coco/sam3_labelme/

Single-label fine-tuning examples:
    --dataset-name sam3_sow --include-label sow
    --dataset-name sam3_piglet --include-label piglet

Output structure:
    finetundata_coco/sam3_labelme/
        train/
            images copied here
            _annotations.coco.json
        test/
            images copied here
            _annotations.coco.json
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


Point = Sequence[float]


def polygon_area(points: Sequence[Point]) -> float:
    """Calculate polygon area using the shoelace formula."""
    if len(points) < 3:
        return 0.0

    total = 0.0
    for i in range(len(points)):
        j = (i + 1) % len(points)
        x1, y1 = float(points[i][0]), float(points[i][1])
        x2, y2 = float(points[j][0]), float(points[j][1])
        total += (x1 * y2) - (x2 * y1)
    return abs(total) / 2.0


def load_json(json_path: Path) -> Dict[str, Any]:
    """Load a UTF-8 JSON file."""
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid LabelMe JSON format, expected object: {json_path}")
    return data


def write_json(json_path: Path, data: Dict[str, Any], indent: int = 2) -> None:
    """Write JSON using UTF-8 without BOM."""
    with json_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
        f.write("\n")


def infer_image_name(labelme: Dict[str, Any], json_path: Path) -> str:
    """Infer image filename from LabelMe imagePath; fallback to json basename + .jpg."""
    image_path = labelme.get("imagePath")
    if image_path:
        return Path(str(image_path)).name
    return f"{json_path.stem}.jpg"


def normalize_label(label: Any) -> str:
    """Normalize LabelMe labels before category lookup/filtering."""
    return str(label or "").strip()


def collect_category_ids(
    json_files: Iterable[Path],
    include_labels: Optional[Set[str]] = None,
) -> Dict[str, int]:
    """Collect all non-empty shape labels and assign category ids sorted by label name."""
    labels = set()
    for json_file in json_files:
        labelme = load_json(json_file)
        for shape in labelme.get("shapes", []):
            label = normalize_label(shape.get("label"))
            if label and (include_labels is None or label in include_labels):
                labels.add(label)

    return {label: idx for idx, label in enumerate(sorted(labels), start=1)}


def convert_split(
    json_files: Sequence[Path],
    split_name: str,
    split_dir: Path,
    category_ids: Dict[str, int],
) -> Dict[str, Any]:
    """Convert one split into COCO annotation format and copy images."""
    split_dir.mkdir(parents=True, exist_ok=True)

    images: List[Dict[str, Any]] = []
    annotations: List[Dict[str, Any]] = []
    ann_id = 1
    img_id = 1

    for json_file in json_files:
        labelme = load_json(json_file)
        image_name = infer_image_name(labelme, json_file)
        source_image = json_file.parent / image_name

        if not source_image.exists():
            raise FileNotFoundError(f"Image not found for {json_file.name}: {source_image}")

        shutil.copy2(source_image, split_dir / image_name)

        try:
            width = int(labelme["imageWidth"])
            height = int(labelme["imageHeight"])
        except KeyError as exc:
            raise KeyError(f"Missing {exc} in LabelMe JSON: {json_file}") from exc

        images.append(
            {
                "id": img_id,
                "file_name": image_name,
                "width": width,
                "height": height,
            }
        )

        for shape in labelme.get("shapes", []):
            shape_type = shape.get("shape_type")
            if shape_type and shape_type != "polygon":
                continue

            label = normalize_label(shape.get("label"))
            if label not in category_ids:
                continue

            points = shape.get("points", [])
            if not isinstance(points, list) or len(points) < 3:
                continue

            # Keep only valid 2D points.
            clean_points: List[Tuple[float, float]] = []
            for point in points:
                if not isinstance(point, (list, tuple)) or len(point) < 2:
                    continue
                clean_points.append((float(point[0]), float(point[1])))

            if len(clean_points) < 3:
                continue

            xs = [p[0] for p in clean_points]
            ys = [p[1] for p in clean_points]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            bbox_w = max(0.0, max_x - min_x)
            bbox_h = max(0.0, max_y - min_y)

            if bbox_w <= 0 or bbox_h <= 0:
                continue

            segmentation: List[float] = []
            for x, y in clean_points:
                segmentation.extend([x, y])

            area = polygon_area(clean_points)
            if area <= 0:
                area = bbox_w * bbox_h

            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": int(category_ids[label]),
                    "segmentation": [segmentation],
                    "bbox": [min_x, min_y, bbox_w, bbox_h],
                    "area": area,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

        img_id += 1

    categories = [
        {
            "id": int(cat_id),
            "name": str(label),
            "supercategory": "object",
        }
        for label, cat_id in sorted(category_ids.items(), key=lambda item: item[1])
    ]

    coco = {
        "info": {
            "description": f"SAM3 LabelMe finetune dataset ({split_name})",
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    out_file = split_dir / "_annotations.coco.json"
    write_json(out_file, coco, indent=2)

    return {
        "Split": split_name,
        "Images": len(images),
        "Annotations": len(annotations),
        "AnnotationFile": str(out_file),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert LabelMe polygon annotations to COCO format and split into train/test."
    )
    parser.add_argument("--input-dir", default="finetundata", help="Input directory containing LabelMe JSON files and images.")
    parser.add_argument("--output-root", default="finetundata_coco", help="Root output directory.")
    parser.add_argument("--dataset-name", default="sam3_labelme", help="Dataset subdirectory name under output root.")
    parser.add_argument(
        "--include-label",
        action="append",
        default=None,
        metavar="LABEL",
        help="Only include this LabelMe label. Repeat the option to include multiple labels.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation/test split ratio. Default: 0.2")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for splitting. Default: 123")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_dir).expanduser().resolve()
    output_path = Path(args.output_root).expanduser().resolve()
    dataset_path = output_path / args.dataset_name

    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_path}")
    if not input_path.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_path}")
    if not (0 < args.val_ratio < 1):
        raise ValueError("--val-ratio must be between 0 and 1, for example 0.2")

    json_files = sorted(input_path.glob("*.json"), key=lambda p: p.name)
    if not json_files:
        raise FileNotFoundError(f"No LabelMe JSON files found in {input_path}")

    include_labels = None
    if args.include_label:
        include_labels = {normalize_label(label) for label in args.include_label}
        include_labels = {label for label in include_labels if label}
        if not include_labels:
            raise ValueError("--include-label was provided, but all labels were empty")

    category_ids = collect_category_ids(json_files, include_labels=include_labels)
    if not category_ids:
        requested = "" if include_labels is None else f" matching {sorted(include_labels)}"
        raise ValueError(f"No valid labels{requested} found in LabelMe JSON files under {input_path}")

    rng = random.Random(args.seed)
    shuffled = list(json_files)
    rng.shuffle(shuffled)

    val_count = max(1, round(len(json_files) * args.val_ratio))
    train_count = len(json_files) - val_count
    if train_count < 1:
        raise ValueError("Split would leave no training images. Lower --val-ratio.")

    train_files = shuffled[:train_count]
    val_files = shuffled[train_count:]

    dataset_path.mkdir(parents=True, exist_ok=True)

    train_stats = convert_split(
        json_files=train_files,
        split_name="train",
        split_dir=dataset_path / "train",
        category_ids=category_ids,
    )
    val_stats = convert_split(
        json_files=val_files,
        split_name="test",
        split_dir=dataset_path / "test",
        category_ids=category_ids,
    )

    print(json.dumps(train_stats, ensure_ascii=False, indent=2))
    print(json.dumps(val_stats, ensure_ascii=False, indent=2))
    if include_labels is not None:
        print(f"Included labels: {', '.join(sorted(include_labels))}")
    print(f"Categories: {', '.join(category_ids.keys())}")
    print(f"Dataset written to: {dataset_path}")


if __name__ == "__main__":
    main()
