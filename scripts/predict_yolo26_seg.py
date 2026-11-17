"""Predict images with a YOLO26 segmentation model and export LabelMe JSON files."""

from __future__ import annotations

import argparse
import base64
import gc
import json
import shutil
from pathlib import Path
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = ROOT / "models" / "segmentation" / "piglet.pt"
DEFAULT_OUTPUT = ROOT / "runs" / "labelme_predictions"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="Trained YOLO segmentation .pt weights.")
    parser.add_argument("--source", type=Path, required=True, help="Image file or image directory to predict.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Directory for LabelMe JSON outputs.")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default="", help="CUDA device like 0, cpu, or empty for auto.")
    parser.add_argument("--max-det", type=int, default=100, help="Maximum detections per image. Lower values use less memory.")
    parser.add_argument("--half", action="store_true", help="Use FP16 inference on CUDA to reduce memory.")
    parser.add_argument("--retina-masks", action="store_true", help="Use native-resolution masks. More accurate polygons but much higher memory.")
    parser.add_argument("--empty-cache-every", type=int, default=1, help="Call torch.cuda.empty_cache() every N images; 0 disables it.")
    parser.add_argument("--oom-retry-min-imgsz", type=int, default=512, help="Lowest image size to retry after CUDA OOM.")
    parser.add_argument("--oom-retry-min-max-det", type=int, default=1, help="Lowest max_det to retry after CUDA OOM.")
    parser.add_argument("--cpu-on-oom", action="store_true", help="Retry an image on CPU if CUDA OOM still happens.")
    parser.add_argument("--recursive", action="store_true", help="Search images recursively when source is a directory.")
    parser.add_argument("--copy-images", action="store_true", help="Copy predicted images into the output directory.")
    parser.add_argument("--embed-image", action="store_true", help="Embed image bytes in imageData for standalone JSON files.")
    parser.add_argument("--save-preview", action="store_true", help="Also save Ultralytics visual prediction images.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing LabelMe JSON files.")
    parser.add_argument("--dry-run", action="store_true", help="Print what would run without loading the model.")
    return parser.parse_args()


def iter_images(source: Path, recursive: bool) -> list[Path]:
    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTS:
            raise ValueError(f"Unsupported image extension: {source}")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"Source not found: {source}")

    pattern = "**/*" if recursive else "*"
    images = sorted(path for path in source.glob(pattern) if path.is_file() and path.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise FileNotFoundError(f"No images found in {source}.")
    return images


def encode_image_data(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def result_to_labelme(result: Any, image_path: Path, image_path_value: str, embed_image: bool) -> dict[str, Any]:
    height, width = result.orig_shape
    shapes: list[dict[str, Any]] = []

    masks = result.masks.xy if result.masks is not None else []
    boxes = result.boxes
    classes = boxes.cls.tolist() if boxes is not None and boxes.cls is not None else []
    confs = boxes.conf.tolist() if boxes is not None and boxes.conf is not None else []
    names = result.names or {}

    for index, polygon in enumerate(masks):
        if len(polygon) < 3:
            continue

        class_id = int(classes[index]) if index < len(classes) else 0
        confidence = float(confs[index]) if index < len(confs) else None
        label = names.get(class_id, str(class_id)) if isinstance(names, dict) else str(class_id)
        points = [[round(float(x), 2), round(float(y), 2)] for x, y in polygon.tolist()]

        shapes.append(
            {
                "label": label,
                "points": points,
                "group_id": None,
                "description": "" if confidence is None else f"confidence={confidence:.4f}",
                "shape_type": "polygon",
                "flags": {},
                "mask": None,
            }
        )

    return {
        "version": "5.11.2",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path_value,
        "imageData": encode_image_data(image_path) if embed_image else None,
        "imageHeight": int(height),
        "imageWidth": int(width),
    }


def output_json_path(image_path: Path, source: Path, output: Path, recursive: bool) -> Path:
    if source.is_dir() and recursive:
        rel_parent = image_path.parent.relative_to(source)
        return output / rel_parent / f"{image_path.stem}.json"
    return output / f"{image_path.stem}.json"


def image_path_for_json(image_path: Path, json_path: Path, copy_images: bool) -> str:
    if copy_images:
        return image_path.name
    try:
        return str(image_path.relative_to(json_path.parent))
    except ValueError:
        return str(image_path.resolve())


def clear_cuda_cache_if_needed(count: int, every: int) -> None:
    if every <= 0 or count % every != 0:
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


def is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text and ("cuda" in text or "torch" in text)


def predict_one_image(model: Any, image_path: Path, args: argparse.Namespace, imgsz: int, max_det: int, device: str | None) -> Iterator[Any]:
    return model.predict(
        source=str(image_path),
        imgsz=imgsz,
        batch=1,
        conf=args.conf,
        iou=args.iou,
        device=device,
        max_det=max_det,
        half=args.half and device != "cpu",
        retina_masks=args.retina_masks,
        save=args.save_preview,
        stream=True,
        verbose=False,
    )

def export_predictions(args: argparse.Namespace) -> None:
    images = iter_images(args.source, recursive=args.recursive)
    if args.dry_run:
        print(f"Weights: {args.weights}")
        print(f"Images: {len(images)}")
        print(f"Output: {args.output}")
        print(
            "Predict: "
            f"imgsz={args.imgsz}, conf={args.conf}, iou={args.iou}, "
            f"max_det={args.max_det}, device={args.device or None}, half={args.half}, "
            f"retina_masks={args.retina_masks}"
        )
        return

    if not args.weights.exists():
        raise FileNotFoundError(f"Trained weights not found: {args.weights}")

    from ultralytics import YOLO

    args.output.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.weights), task="segment")

    written = 0
    min_imgsz = max(32, int(args.oom_retry_min_imgsz))
    min_max_det = max(1, int(args.oom_retry_min_max_det))

    for index, input_path in enumerate(images, start=1):
        retry_imgsz = max(min_imgsz, int(args.imgsz))
        retry_max_det = max(1, int(args.max_det))
        while True:
            try:
                results = predict_one_image(
                    model=model,
                    image_path=input_path,
                    args=args,
                    imgsz=retry_imgsz,
                    max_det=retry_max_det,
                    device=args.device or None,
                )
                for result in results:
                    image_path = Path(result.path)
                    json_path = output_json_path(image_path, args.source, args.output, args.recursive)
                    json_path.parent.mkdir(parents=True, exist_ok=True)
                    if json_path.exists() and not args.overwrite:
                        raise FileExistsError(f"{json_path} already exists. Use --overwrite to replace it.")

                    if args.copy_images:
                        copied_image = json_path.parent / image_path.name
                        if not copied_image.exists() or args.overwrite:
                            shutil.copy2(image_path, copied_image)
                        image_path_value = image_path.name
                    else:
                        image_path_value = image_path_for_json(image_path, json_path, copy_images=False)

                    labelme = result_to_labelme(
                        result=result,
                        image_path=image_path,
                        image_path_value=image_path_value,
                        embed_image=args.embed_image,
                    )
                    json_path.write_text(json.dumps(labelme, ensure_ascii=False, indent=2), encoding="utf-8")
                    written += 1
                    del result
                del results
                break
            except RuntimeError as exc:
                if not is_cuda_oom(exc):
                    raise
                clear_cuda_cache_if_needed(1, 1)
                next_max_det = max(min_max_det, retry_max_det // 2)
                next_imgsz = max(min_imgsz, retry_imgsz // 2)
                if next_max_det < retry_max_det or next_imgsz < retry_imgsz:
                    retry_max_det = next_max_det
                    retry_imgsz = next_imgsz
                    print(
                        f"[WARN] CUDA OOM on image {index}/{len(images)}; "
                        f"retrying with imgsz={retry_imgsz}, max_det={retry_max_det}: {input_path}"
                    )
                    continue
                if args.cpu_on_oom:
                    print(f"[WARN] CUDA OOM; retrying this image on CPU: {input_path}")
                    cpu_model = YOLO(str(args.weights), task="segment")
                    results = predict_one_image(cpu_model, input_path, args, retry_imgsz, retry_max_det, "cpu")
                    for result in results:
                        image_path = Path(result.path)
                        json_path = output_json_path(image_path, args.source, args.output, args.recursive)
                        json_path.parent.mkdir(parents=True, exist_ok=True)
                        if json_path.exists() and not args.overwrite:
                            raise FileExistsError(f"{json_path} already exists. Use --overwrite to replace it.")
                        image_path_value = (
                            image_path.name
                            if args.copy_images
                            else image_path_for_json(image_path, json_path, copy_images=False)
                        )
                        if args.copy_images:
                            copied_image = json_path.parent / image_path.name
                            if not copied_image.exists() or args.overwrite:
                                shutil.copy2(image_path, copied_image)
                        labelme = result_to_labelme(result, image_path, image_path_value, args.embed_image)
                        json_path.write_text(json.dumps(labelme, ensure_ascii=False, indent=2), encoding="utf-8")
                        written += 1
                        del result
                    del results, cpu_model
                    clear_cuda_cache_if_needed(1, 1)
                    break
                raise RuntimeError(
                    f"CUDA OOM while predicting {input_path} even with imgsz={retry_imgsz}, max_det={retry_max_det}. "
                    "Try --half, --cpu-on-oom, lower --imgsz, lower --max-det, or raise --conf."
                ) from exc

        if index % 10 == 0 or index == len(images):
            print(f"[INFO] Predicted {index}/{len(images)} images; wrote={written}")
        clear_cuda_cache_if_needed(index, args.empty_cache_every)

    del model
    clear_cuda_cache_if_needed(1, 1)

    print(f"Wrote {written} LabelMe JSON files to {args.output}.")


def main() -> None:
    export_predictions(parse_args())


if __name__ == "__main__":
    main()
