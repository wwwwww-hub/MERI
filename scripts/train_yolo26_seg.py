#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert Labelme polygon annotations to Ultralytics YOLO segmentation format,
then train a YOLO*-seg model, e.g. YOLO26-seg.

This version is for the common case where raw images and Labelme JSON labels are
stored in two different directories. It creates only train/val splits; no test
split is generated.

Example:
python scripts/train_yolo26_seg.py \
  --img-dir data/DS2_segmentation/pseudo_train \
  --label-dir data/DS2_segmentation/pseudo_train \
  --out working_data/segmentation_piglet \
  --model yolo26s-seg.pt \
  --names sow,piglet \
  --train-labels piglet \
  --imgsz 1024 --batch 8 --epochs 150 --device 0 --overwrite
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SCORE_RE = re.compile(r"score\s*=\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Labelme polygon annotations to Ultralytics YOLO-seg format "
            "and train YOLO*-seg. Images and labels may be in separate folders."
        )
    )
    parser.add_argument(
        "--img-dir",
        "--image-dir",
        dest="img_dir",
        type=Path,
        required=True,
        help="Directory containing raw images. The script searches recursively.",
    )
    parser.add_argument(
        "--label-dir",
        "--json-dir",
        dest="label_dir",
        type=Path,
        required=True,
        help="Directory containing Labelme .json files. The script searches recursively.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output YOLO dataset root")
    parser.add_argument(
        "--model",
        type=str,
        default="yolo26s-seg.pt",
        help="Ultralytics model yaml/pt, e.g. yolo26n-seg.pt, yolo26s-seg.pt, yolo11n-seg.pt",
    )
    parser.add_argument(
        "--names",
        type=str,
        default="sow,piglet",
        help="All possible class names in fixed order, comma-separated. Use empty string for auto-detect.",
    )
    parser.add_argument(
        "--train-labels",
        "--include-labels",
        dest="train_labels",
        type=str,
        default="",
        help=(
            "Which Labelme labels to train, comma-separated. "
            "Default empty means all labels from --names. Examples: sow, piglet, sow,piglet. "
            "When a subset is selected, class ids are re-indexed from 0 in data.yaml."
        ),
    )
    parser.add_argument(
        "--ignore-labels",
        type=str,
        default="_ignore,ignore,background",
        help="Labels to ignore, comma-separated",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Filter pseudo labels by Labelme shape description like score=0.73. Shapes without score are kept.",
    )
    parser.add_argument(
        "--instance-mode",
        choices=["group", "shape"],
        default="group",
        help="group: merge shapes with the same (label, group_id) as one instance; shape: every polygon is one instance.",
    )
    parser.add_argument(
        "--multi-contour-policy",
        choices=["largest", "split"],
        default="largest",
        help="When grouped masks still have multiple disconnected contours: largest keeps one instance; split writes multiple instances.",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=20.0,
        help="Ignore polygons/contours with pixel area smaller than this value.",
    )
    parser.add_argument(
        "--approx-eps-ratio",
        type=float,
        default=0.0015,
        help="Polygon simplification epsilon ratio relative to contour perimeter. Set 0 to disable.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio; train ratio is 1 - val_ratio")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy-mode", choices=["copy", "link"], default="copy")
    parser.add_argument("--convert-only", action="store_true", help="Only convert dataset and write data.yaml; do not train")
    parser.add_argument("--overwrite", action="store_true", help="Remove existing output directory before conversion")

    # Ultralytics training arguments
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", type=str, default="0", help="GPU id like 0, or cpu")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", type=str, default="runs/segment")
    parser.add_argument("--name", type=str, default="yolo26_seg_labelme")
    parser.add_argument("--optimizer", type=str, default="SGD", help="SGD, AdamW, auto, etc.")
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--cache", action="store_true", help="Enable Ultralytics image cache")
    parser.add_argument("--resume", action="store_true", help="Resume an interrupted Ultralytics run")
    return parser.parse_args()


def split_csv(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_image_size(img_path: Optional[Path], data: dict) -> Tuple[int, int]:
    w = data.get("imageWidth")
    h = data.get("imageHeight")
    if isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0:
        return w, h
    if img_path is None:
        raise ValueError("imageWidth/imageHeight missing and image file was not found")
    if Image is None:
        raise RuntimeError("Pillow is required when imageWidth/imageHeight are missing in Labelme JSON")
    with Image.open(img_path) as im:
        return im.size


def build_image_indexes(img_dir: Path) -> Tuple[Dict[str, List[Path]], Dict[str, List[Path]]]:
    """Return indexes by filename and by stem for image lookup."""
    by_name: Dict[str, List[Path]] = defaultdict(list)
    by_stem: Dict[str, List[Path]] = defaultdict(list)
    for p in sorted(img_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            by_name[p.name].append(p)
            by_stem[p.stem].append(p)
    return by_name, by_stem


def first_existing(candidates: Iterable[Path]) -> Optional[Path]:
    seen = set()
    for c in candidates:
        try:
            key = c.resolve()
        except Exception:
            key = c
        if key in seen:
            continue
        seen.add(key)
        if c.exists() and c.is_file():
            return c
    return None


def find_image_for_json(
    json_path: Path,
    data: dict,
    img_dir: Path,
    image_by_name: Dict[str, List[Path]],
    image_by_stem: Dict[str, List[Path]],
) -> Optional[Path]:
    """
    Match a Labelme JSON to an image. Priority:
    1. absolute imagePath in JSON
    2. imagePath relative to label folder
    3. imagePath relative to image root
    4. exact filename lookup under image root
    5. JSON stem lookup under image root
    """
    candidates: List[Path] = []
    image_path = data.get("imagePath")
    if image_path:
        image_path_obj = Path(str(image_path))
        if image_path_obj.is_absolute():
            candidates.append(image_path_obj)
        else:
            candidates.append(json_path.parent / image_path_obj)
            candidates.append(json_path.parent / image_path_obj.name)
            candidates.append(img_dir / image_path_obj)
            candidates.append(img_dir / image_path_obj.name)
            candidates.extend(image_by_name.get(image_path_obj.name, []))
            candidates.extend(image_by_stem.get(image_path_obj.stem, []))

    # Fallback: match by JSON basename, regardless of image suffix.
    for ext in sorted(IMG_EXTS):
        candidates.append(img_dir / f"{json_path.stem}{ext}")
    candidates.extend(image_by_name.get(json_path.with_suffix(".jpg").name, []))
    candidates.extend(image_by_stem.get(json_path.stem, []))

    return first_existing(candidates)


def parse_score(desc: object) -> Optional[float]:
    if not isinstance(desc, str):
        return None
    m = SCORE_RE.search(desc)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def rectangle_to_polygon(points: Sequence[Sequence[float]]) -> List[Tuple[float, float]]:
    if len(points) < 2:
        return []
    x1, y1 = points[0]
    x2, y2 = points[1]
    xmin, xmax = sorted([float(x1), float(x2)])
    ymin, ymax = sorted([float(y1), float(y2)])
    return [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]


def shape_to_polygon(shape: dict) -> List[Tuple[float, float]]:
    points = shape.get("points") or []
    shape_type = shape.get("shape_type", "polygon")
    if shape_type == "polygon":
        out = []
        for pt in points:
            if len(pt) < 2:
                continue
            x, y = float(pt[0]), float(pt[1])
            if math.isfinite(x) and math.isfinite(y):
                out.append((x, y))
        return out
    if shape_type == "rectangle":
        return rectangle_to_polygon(points)
    # Lines, points, circles and masks are not suitable for YOLO polygon segmentation here.
    return []


def clamp_polygon(poly: np.ndarray, w: int, h: int) -> np.ndarray:
    poly = poly.astype(np.float32, copy=False)
    poly[:, 0] = np.clip(poly[:, 0], 0, max(w - 1, 0))
    poly[:, 1] = np.clip(poly[:, 1], 0, max(h - 1, 0))
    return poly


def polygon_area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def contour_to_poly(contour: np.ndarray, eps_ratio: float) -> np.ndarray:
    contour = contour.reshape(-1, 2).astype(np.float32)
    if eps_ratio > 0 and len(contour) >= 4:
        peri = cv2.arcLength(contour.reshape(-1, 1, 2), True)
        eps = max(peri * eps_ratio, 0.5)
        contour = cv2.approxPolyDP(contour.reshape(-1, 1, 2), eps, True).reshape(-1, 2).astype(np.float32)
    return contour


def grouped_polygons_to_output_polys(
    polygons: List[np.ndarray],
    w: int,
    h: int,
    min_area: float,
    eps_ratio: float,
    multi_contour_policy: str,
) -> List[np.ndarray]:
    """Return one or more polygons for a grouped instance."""
    if not polygons:
        return []
    if len(polygons) == 1:
        poly = clamp_polygon(polygons[0], w, h)
        return [poly] if len(poly) >= 3 and polygon_area(poly) >= min_area else []
    if cv2 is None:
        # Fallback: use the largest raw polygon when OpenCV is unavailable.
        polys = [clamp_polygon(p, w, h) for p in polygons]
        polys = [p for p in polys if len(p) >= 3 and polygon_area(p) >= min_area]
        if multi_contour_policy == "split":
            return polys
        return [max(polys, key=polygon_area)] if polys else []

    mask = np.zeros((h, w), dtype=np.uint8)
    for poly in polygons:
        poly = clamp_polygon(poly, w, h)
        if len(poly) >= 3 and polygon_area(poly) >= min_area:
            cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], 255)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: List[np.ndarray] = []
    for c in contours:
        area = float(cv2.contourArea(c))
        if area < min_area:
            continue
        poly = contour_to_poly(c, eps_ratio)
        if len(poly) >= 3:
            out.append(poly)
    if not out:
        return []
    out.sort(key=polygon_area, reverse=True)
    if multi_contour_policy == "largest":
        return [out[0]]
    return out


def normalize_poly(poly: np.ndarray, w: int, h: int) -> List[float]:
    poly = clamp_polygon(poly, w, h)
    norm = poly.copy().astype(np.float64)
    norm[:, 0] /= float(w)
    norm[:, 1] /= float(h)
    norm = np.clip(norm, 0.0, 1.0)
    return norm.reshape(-1).tolist()


def format_yolo_line(cls_id: int, coords: Sequence[float]) -> str:
    values = [str(cls_id)] + [f"{v:.6f}" for v in coords]
    return " ".join(values)


def collect_class_names(json_paths: Iterable[Path], ignore: set, min_score: float) -> List[str]:
    names: List[str] = []
    seen = set()
    for jp in json_paths:
        try:
            data = read_json(jp)
        except Exception:
            continue
        for shape in data.get("shapes", []):
            label = str(shape.get("label", "")).strip()
            if not label or label in ignore:
                continue
            score = parse_score(shape.get("description"))
            if score is not None and score < min_score:
                continue
            if label not in seen:
                names.append(label)
                seen.add(label)
    return names


def select_train_class_names(all_class_names: List[str], train_labels_arg: str) -> List[str]:
    """Return the class list that will actually be written to YOLO labels/data.yaml.

    Empty/"all" keeps all classes. A subset such as "piglet" produces a
    one-class YOLO dataset where piglet is class 0.
    """
    requested = split_csv(train_labels_arg)
    if not requested:
        return all_class_names
    if len(requested) == 1 and requested[0].lower() == "all":
        return all_class_names

    available = set(all_class_names)
    missing = [name for name in requested if name not in available]
    if missing:
        raise ValueError(
            f"--train-labels contains labels not present in --names/auto-detected labels: {missing}. "
            f"Available labels: {all_class_names}"
        )

    # Keep the global --names order, not the user typed order, for deterministic class ids.
    selected = [name for name in all_class_names if name in set(requested)]
    if not selected:
        raise ValueError("--train-labels did not select any class")
    return selected


def make_splits(items: List[Tuple[Path, Path]], val_ratio: float, seed: int) -> Dict[str, List[Tuple[Path, Path]]]:
    if val_ratio < 0 or val_ratio >= 1:
        raise ValueError("Require 0 <= val_ratio < 1")
    items = items[:]
    random.Random(seed).shuffle(items)
    n = len(items)
    n_val = int(round(n * val_ratio))
    # Avoid empty training split for tiny debug datasets.
    if n > 0 and n_val >= n:
        n_val = max(0, n - 1)
    val_items = items[:n_val]
    train_items = items[n_val:]
    return {"train": train_items, "val": val_items}


def safe_copy_or_link(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "link":
        try:
            dst.symlink_to(src.resolve())
            return
        except Exception:
            pass
    shutil.copy2(src, dst)


def unique_image_name(img: Path, img_dir: Path, used: set[str]) -> str:
    """Create a deterministic output image filename and avoid basename collisions."""
    try:
        rel = img.resolve().relative_to(img_dir.resolve())
        base = "__".join(rel.with_suffix("").parts) + img.suffix.lower()
    except Exception:
        base = img.name
    name = base
    i = 1
    while name in used:
        name = f"{Path(base).stem}__dup{i}{Path(base).suffix}"
        i += 1
    used.add(name)
    return name


def convert_one_json(
    json_path: Path,
    img_path: Optional[Path],
    label_out: Path,
    names_to_id: Dict[str, int],
    ignore_labels: set,
    min_score: float,
    instance_mode: str,
    multi_contour_policy: str,
    min_area: float,
    eps_ratio: float,
) -> Tuple[int, Counter]:
    data = read_json(json_path)
    w, h = get_image_size(img_path, data)
    grouped: Dict[Tuple[str, object], List[np.ndarray]] = defaultdict(list)
    class_counter: Counter = Counter()

    for idx, shape in enumerate(data.get("shapes", [])):
        label = str(shape.get("label", "")).strip()
        if not label or label in ignore_labels or label not in names_to_id:
            continue
        score = parse_score(shape.get("description"))
        if score is not None and score < min_score:
            continue
        poly_pts = shape_to_polygon(shape)
        if len(poly_pts) < 3:
            continue
        poly = np.array(poly_pts, dtype=np.float32).reshape(-1, 2)
        poly = clamp_polygon(poly, w, h)
        if polygon_area(poly) < min_area:
            continue
        if instance_mode == "group":
            gid = shape.get("group_id", None)
            key = (label, gid if gid is not None else f"shape_{idx}")
        else:
            key = (label, f"shape_{idx}")
        grouped[key].append(poly)

    lines: List[str] = []
    for (label, _gid), polygons in grouped.items():
        out_polys = grouped_polygons_to_output_polys(
            polygons, w, h, min_area, eps_ratio, multi_contour_policy
        )
        for poly in out_polys:
            if len(poly) < 3 or polygon_area(poly) < min_area:
                continue
            coords = normalize_poly(poly, w, h)
            if len(coords) < 6:
                continue
            cls_id = names_to_id[label]
            lines.append(format_yolo_line(cls_id, coords))
            class_counter[label] += 1

    label_out.parent.mkdir(parents=True, exist_ok=True)
    label_out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines), class_counter


def write_data_yaml(out_root: Path, class_names: List[str]) -> Path:
    data = {
        "path": str(out_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(class_names)},
    }
    yaml_path = out_root / "data.yaml"
    if yaml is not None:
        yaml_path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    else:
        lines = [f"path: {data['path']}", "train: images/train", "val: images/val", "names:"]
        for i, name in enumerate(class_names):
            lines.append(f"  {i}: {name}")
        yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path


def convert_dataset(args: argparse.Namespace) -> Path:
    img_dir = args.img_dir.resolve()
    label_dir = args.label_dir.resolve()
    out = args.out.resolve()

    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {img_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Label directory does not exist: {label_dir}")

    if args.overwrite and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    json_paths = sorted([p for p in label_dir.rglob("*.json") if p.is_file()])
    if not json_paths:
        raise FileNotFoundError(f"No Labelme JSON files found under {label_dir}")

    ignore_labels = set(split_csv(args.ignore_labels))
    all_class_names = split_csv(args.names)
    if not all_class_names:
        all_class_names = collect_class_names(json_paths, ignore_labels, args.min_score)
    if not all_class_names:
        raise RuntimeError("No class names found. Pass --names sow,piglet or check your labels.")

    class_names = select_train_class_names(all_class_names, args.train_labels)
    names_to_id = {name: i for i, name in enumerate(class_names)}

    image_by_name, image_by_stem = build_image_indexes(img_dir)
    if not image_by_name:
        raise FileNotFoundError(f"No image files found under {img_dir}")

    items: List[Tuple[Path, Path]] = []
    missing_images: List[Path] = []
    ambiguous_images: List[str] = []
    for jp in json_paths:
        data = read_json(jp)
        img = find_image_for_json(jp, data, img_dir, image_by_name, image_by_stem)
        if img is None:
            missing_images.append(jp)
            continue
        # Record potential ambiguity for manual checking if many images share the same basename/stem.
        image_path = data.get("imagePath")
        if image_path:
            image_name = Path(str(image_path)).name
            image_stem = Path(str(image_path)).stem
        else:
            image_name = jp.with_suffix(".jpg").name
            image_stem = jp.stem
        if len(image_by_name.get(image_name, [])) > 1 or len(image_by_stem.get(image_stem, [])) > 1:
            ambiguous_images.append(f"{jp} -> {img}")
        items.append((jp, img))

    if missing_images:
        miss_path = out / "missing_images.txt"
        miss_path.write_text("\n".join(str(p) for p in missing_images) + "\n", encoding="utf-8")
        print(f"[WARN] {len(missing_images)} JSON files skipped because images were not found. See: {miss_path}")
    if ambiguous_images:
        amb_path = out / "ambiguous_image_matches.txt"
        amb_path.write_text("\n".join(ambiguous_images) + "\n", encoding="utf-8")
        print(f"[WARN] {len(ambiguous_images)} JSON files had non-unique image name/stem matches. Check: {amb_path}")
    if not items:
        raise RuntimeError(
            "No JSON-image pairs were found. Check Labelme imagePath, image filenames, or --img-dir/--label-dir."
        )

    splits = make_splits(items, args.val_ratio, args.seed)
    total_counter: Counter = Counter()
    split_counter: Dict[str, Counter] = {}
    n_images = 0
    n_instances = 0
    used_names_by_split: Dict[str, set[str]] = {"train": set(), "val": set()}

    for split, split_items in splits.items():
        split_counter[split] = Counter()
        for jp, img in split_items:
            dst_img_name = unique_image_name(img, img_dir, used_names_by_split[split])
            dst_img = out / "images" / split / dst_img_name
            safe_copy_or_link(img, dst_img, args.copy_mode)
            dst_label = out / "labels" / split / f"{Path(dst_img_name).stem}.txt"
            n, c = convert_one_json(
                jp,
                img,
                dst_label,
                names_to_id,
                ignore_labels,
                args.min_score,
                args.instance_mode,
                args.multi_contour_policy,
                args.min_area,
                args.approx_eps_ratio,
            )
            n_images += 1
            n_instances += n
            total_counter.update(c)
            split_counter[split].update(c)

    yaml_path = write_data_yaml(out, class_names)

    print("\n[OK] Dataset converted")
    print(f"  image dir    : {img_dir}")
    print(f"  label dir    : {label_dir}")
    print(f"  output       : {out}")
    print(f"  data.yaml    : {yaml_path}")
    print(f"  all labels   : {all_class_names}")
    print(f"  train labels : {names_to_id}")
    print(f"  images       : {n_images}")
    print(f"  instances    : {n_instances}")
    for split, c in split_counter.items():
        print(f"  {split:<5}       : {len(splits[split])} images, {dict(c)}")
    print(f"  total classes: {dict(total_counter)}\n")
    return yaml_path


def train_yolo(args: argparse.Namespace, data_yaml: Path) -> None:
    from ultralytics import YOLO

    model = YOLO(args.model)
    train_kwargs = dict(
        data=str(data_yaml),
        task="segment",
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        optimizer=args.optimizer,
        lr0=args.lr0,
        weight_decay=args.weight_decay,
        patience=args.patience,
        cache=args.cache,
        amp=True,
        plots=True,
        close_mosaic=20,
        overlap_mask=False,
        mask_ratio=4,
        resume=args.resume,
    )
    print("[INFO] Start training with Ultralytics:")
    print(train_kwargs)
    model.train(**train_kwargs)


def main() -> None:
    args = parse_args()
    data_yaml = convert_dataset(args)
    if args.convert_only:
        print("[OK] --convert-only enabled; training skipped.")
        return
    train_yolo(args, data_yaml)


if __name__ == "__main__":
    main()
