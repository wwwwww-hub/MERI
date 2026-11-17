"""Train the RGB+mask association student on identity-labelled tracklets.

Examples:
    python scripts/train_mask_aware_tracker.py --dry-run
    python scripts/train_mask_aware_tracker.py --epochs 30 --device 0
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from tempfile import TemporaryDirectory
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from ultralytics.trackers.mask_aware import (
    AnyLabelingPairDataset,
    MaskAwareAssociationModel,
    save_association_checkpoint,
    scan_anylabeling_split,
)
from ultralytics.trackers.osnet import OSNetX025FeatureExtractor, load_osnet_x0_25_imagenet
from ultralytics.utils.files import increment_path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data/DS3_tracking"
DEFAULT_OSNET_WEIGHTS = ROOT / "working_data/pretrained/osnet_x0_25_imagenet.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--ffmpeg", default="ffmpeg", help="FFmpeg executable used to read DS3 videos.")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default=None, help="Optional separate validation directory; test is reserved for evaluation.")
    parser.add_argument("--label", action="append", default=None, help="Track label; repeat for multiple labels.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0, help="0 is the safest setting on Windows.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Optimizer fallback learning rate.")
    parser.add_argument("--backbone-lr", type=float, default=1e-4, help="OSNet pretrained-layer learning rate.")
    parser.add_argument("--head-lr", type=float, default=1e-3, help="OSNet conv1, projection, and Edge-MLP LR.")
    parser.add_argument(
        "--freeze-backbone-epochs",
        type=int,
        default=2,
        help="Initially freeze OSNet conv2-conv5 and its official FC; conv1 remains trainable for the mask channel.",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-loss-weight", type=float, default=0.20)
    parser.add_argument("--negative-margin", type=float, default=0.20)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument(
        "--encoder-arch", choices=("osnet_x0_25",), default="osnet_x0_25"
    )
    parser.add_argument("--encoder-width", type=int, default=24)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--osnet-pretrained", type=Path, default=DEFAULT_OSNET_WEIGHTS)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--max-positive-gap", type=int, default=8)
    parser.add_argument("--positives-per-anchor", type=int, default=4)
    parser.add_argument("--negatives-per-positive", type=int, default=2)
    parser.add_argument("--max-train-pairs", type=int, default=0)
    parser.add_argument("--max-val-pairs", type=int, default=12000)
    parser.add_argument("--instance-cache", type=int, default=512, help="Per-dataset compact crop LRU size.")
    parser.add_argument(
        "--preload-max-gb",
        type=float,
        default=2.0,
        help="Maximum estimated RAM used by train+val crop preload; ignored with --no-preload.",
    )
    parser.add_argument(
        "--no-preload",
        action="store_true",
        help="Disable one-time compact crop preload (uses less RAM but repeatedly decodes 1080p files).",
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=2.0,
        help="Temporal sampling rate represented by adjacent dataset frames; saved in the checkpoint.",
    )
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="Print train/val progress every N batches; 0 prints only the first and last batch.",
    )
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--device", default="", help="Empty selects CUDA when available, otherwise CPU.")
    parser.add_argument("--project", type=Path, default=ROOT / "reproduced/tracking")
    parser.add_argument("--name", default="train")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Scan data and execute one forward pass without writes.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_ints = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_positive_gap": args.max_positive_gap,
        "positives_per_anchor": args.positives_per_anchor,
        "embedding_dim": args.embedding_dim,
        "encoder_width": args.encoder_width,
        "hidden_dim": args.hidden_dim,
    }
    invalid = [name for name, value in positive_ints.items() if value <= 0]
    if invalid:
        raise ValueError(f"arguments must be positive: {invalid}")
    if (
        args.workers < 0
        or args.negatives_per_positive < 0
        or args.max_train_pairs < 0
        or args.max_val_pairs < 0
        or args.instance_cache < 0
        or args.log_interval < 0
    ):
        raise ValueError("worker, negative, maximum-pair, cache, and log-interval values must be non-negative")
    if min(args.lr, args.backbone_lr, args.head_lr) <= 0 or args.weight_decay < 0 or args.embedding_loss_weight < 0:
        raise ValueError("optimizer/loss values are invalid")
    if args.batch_size < 2:
        raise ValueError("--batch-size must be at least 2 because OSNet contains BatchNorm1d")
    if args.freeze_backbone_epochs < 0:
        raise ValueError("--freeze-backbone-epochs must be non-negative")
    if args.preload_max_gb <= 0 or args.sample_fps <= 0:
        raise ValueError("--preload-max-gb and --sample-fps must be positive")
    if not args.no_preload and args.workers:
        raise ValueError("preloaded crops with Windows workers can multiply RAM; use --workers 0 or --no-preload")
    if not -1.0 < args.negative_margin < 1.0:
        raise ValueError("--negative-margin must be between -1 and 1")


def select_device(value: str) -> torch.device:
    value = value.strip().lower()
    if value in {"", "cuda"}:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if value.isdigit():
        return torch.device(f"cuda:{value}")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {value}")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_duration(seconds: float) -> str:
    """Format an elapsed/ETA duration compactly for fixed PowerShell progress logs."""
    if not math.isfinite(seconds) or seconds < 0:
        return "--"
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def should_log_batch(batch_index: int, total_batches: int, interval: int) -> bool:
    """Always log the first/last batch and optionally every fixed interval."""
    return (
        batch_index == 1
        or batch_index == total_batches
        or (interval > 0 and batch_index % interval == 0)
    )


def device_memory_text(device: torch.device) -> str:
    """Return current/peak CUDA allocation without affecting CPU training."""
    if device.type != "cuda":
        return ""
    gib = 1024**3
    allocated = torch.cuda.memory_allocated(device) / gib
    peak = torch.cuda.max_memory_allocated(device) / gib
    return f" gpu_mem={allocated:.2f}GiB peak={peak:.2f}GiB"


def progress_text(
    phase: str,
    batch_index: int,
    total_batches: int,
    running_loss: float,
    samples: int,
    batch_loss: float,
    data_seconds: float,
    step_seconds: float,
    started: float,
    finished: float,
    device: torch.device,
) -> str:
    """Build one stable, newline-based progress record (no ANSI terminal control)."""
    elapsed = max(finished - started, 1e-9)
    batches_per_second = batch_index / elapsed
    remaining = max(total_batches - batch_index, 0)
    eta = remaining / batches_per_second if batches_per_second > 0 else math.inf
    percent = 100.0 * batch_index / max(total_batches, 1)
    return (
        f"{phase} batch {batch_index}/{total_batches} ({percent:5.1f}%) "
        f"loss={running_loss / max(samples, 1):.4f} batch_loss={batch_loss:.4f} "
        f"data_ms={data_seconds * 1000:.0f} step_ms={step_seconds * 1000:.0f} "
        f"rate={batches_per_second:.2f} batch/s elapsed={format_duration(elapsed)} "
        f"eta={format_duration(eta)}{device_memory_text(device)}"
    )


def print_scan(name: str, scan) -> None:
    observations = sum(len(clip.instances) for clip in scan.clips)
    tracks = sum(len(clip.by_track) for clip in scan.clips)
    print(f"{name}: {len(scan.clips)} complete clips, {tracks} tracklets, {observations} observations")
    for clip_id, reason in scan.skipped:
        print(f"  [skip] {clip_id}: {reason}")


def binary_auc(targets: np.ndarray, scores: np.ndarray) -> float:
    positives = int((targets == 1).sum())
    negatives = int((targets == 0).sum())
    if not positives or not negatives:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positive_rank_sum = ranks[targets == 1].sum()
    return float((positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))


def pair_metrics(targets: list[float], scores: list[float]) -> dict[str, float]:
    target = np.asarray(targets, dtype=np.int64)
    score = np.asarray(scores, dtype=np.float64)
    predicted = score >= 0.5
    tp = int(np.logical_and(predicted, target == 1).sum())
    fp = int(np.logical_and(predicted, target == 0).sum())
    fn = int(np.logical_and(~predicted, target == 1).sum())
    tn = int(np.logical_and(~predicted, target == 0).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "accuracy": (tp + tn) / max(len(target), 1),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "auc": binary_auc(target, score),
    }


def contrastive_embedding_loss(
    first: torch.Tensor, second: torch.Tensor, target: torch.Tensor, negative_margin: float
) -> torch.Tensor:
    similarity = F.cosine_similarity(first, second)
    positive = target * (1.0 - similarity)
    negative = (1.0 - target) * F.relu(similarity - negative_margin)
    return (positive + negative).mean()


def set_osnet_backbone_frozen(model: MaskAwareAssociationModel, frozen: bool) -> None:
    """Freeze/unfreeze pretrained OSNet stages while leaving the RGB+mask conv1 trainable."""
    if model.encoder_arch != "osnet_x0_25":
        return
    backbone = model.encoder.features
    if not isinstance(backbone, OSNetX025FeatureExtractor):
        raise TypeError("OSNet encoder did not construct the expected backbone")
    for name in ("conv2", "conv3", "conv4", "conv5", "fc"):
        module = getattr(backbone, name)
        module.requires_grad_(not frozen)
        if frozen:
            module.eval()


def build_optimizer(model: MaskAwareAssociationModel, args: argparse.Namespace) -> torch.optim.Optimizer:
    if model.encoder_arch != "osnet_x0_25":
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    backbone = model.encoder.features
    if not isinstance(backbone, OSNetX025FeatureExtractor):
        raise TypeError("OSNet encoder did not construct the expected backbone")
    pretrained_parameters = []
    for name in ("conv2", "conv3", "conv4", "conv5", "fc"):
        pretrained_parameters.extend(getattr(backbone, name).parameters())
    new_parameters = [
        *backbone.conv1.parameters(),
        *model.encoder.projection.parameters(),
        *model.edge_head.parameters(),
    ]
    return torch.optim.AdamW(
        (
            {"params": pretrained_parameters, "lr": args.backbone_lr, "name": "osnet_backbone"},
            {"params": new_parameters, "lr": args.head_lr, "name": "mask_and_association_head"},
        ),
        weight_decay=args.weight_decay,
    )


@torch.inference_mode()
def evaluate(
    model: MaskAwareAssociationModel,
    loader: DataLoader,
    device: torch.device,
    progress_label: str = "val",
    log_interval: int = 50,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    targets: list[float] = []
    scores: list[float] = []
    total_batches = len(loader)
    started = time.perf_counter()
    previous_finished = started
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    print(f"{progress_label} start: batches={total_batches}, pairs={len(loader.dataset)}", flush=True)
    for batch_index, (first, second, geometry, target) in enumerate(loader, 1):
        batch_ready = time.perf_counter()
        data_seconds = batch_ready - previous_finished
        first, second = first.to(device), second.to(device)
        geometry, target = geometry.to(device), target.to(device)
        logits, _, _ = model(first, second, geometry)
        loss = F.binary_cross_entropy_with_logits(logits, target)
        batch_loss = float(loss)
        total_loss += batch_loss * len(target)
        total += len(target)
        targets.extend(target.cpu().tolist())
        scores.extend(logits.sigmoid().cpu().tolist())
        finished = time.perf_counter()
        if should_log_batch(batch_index, total_batches, log_interval):
            print(
                progress_text(
                    progress_label,
                    batch_index,
                    total_batches,
                    total_loss,
                    total,
                    batch_loss,
                    data_seconds,
                    finished - batch_ready,
                    started,
                    finished,
                    device,
                ),
                flush=True,
            )
        previous_finished = time.perf_counter()
    metrics = pair_metrics(targets, scores)
    metrics["loss"] = total_loss / max(total, 1)
    metrics["seconds"] = time.perf_counter() - started
    return metrics


def main() -> None:
    args = parse_args()
    validate_args(args)
    with TemporaryDirectory(prefix='meri_ds3_train_') as temporary:
        train(args, Path(temporary))


def train(args, decoded_root):
    seed_everything(args.seed)
    labels = tuple(args.label or ["piglet"])
    train_scan = scan_anylabeling_split(
        args.dataset, args.train_split, labels=labels,
        decoded_root=decoded_root, ffmpeg=args.ffmpeg,
    )
    if args.val_split == "test":
        raise ValueError("Use the test split with evaluate_tracking.py, not for training decisions")
    val_scan = scan_anylabeling_split(
        args.dataset, args.val_split, labels=labels,
        decoded_root=decoded_root, ffmpeg=args.ffmpeg,
    ) if args.val_split else None
    print_scan("train", train_scan)
    if val_scan is not None:
        print_scan("val", val_scan)
    if not train_scan.clips or train_scan.skipped:
        raise RuntimeError(f"training data are incomplete: {train_scan.skipped}")
    if val_scan is not None and (not val_scan.clips or val_scan.skipped):
        raise RuntimeError(f"validation data are incomplete: {val_scan.skipped}")
    train_dataset = AnyLabelingPairDataset(
        train_scan.clips,
        max_positive_gap=args.max_positive_gap,
        positives_per_anchor=args.positives_per_anchor,
        negatives_per_positive=args.negatives_per_positive,
        augment=True,
        seed=args.seed,
        max_pairs=args.max_train_pairs,
        instance_cache=args.instance_cache,
    )
    val_dataset = AnyLabelingPairDataset(
        val_scan.clips,
        max_positive_gap=args.max_positive_gap,
        positives_per_anchor=args.positives_per_anchor,
        negatives_per_positive=args.negatives_per_positive,
        augment=False,
        seed=args.seed + 1,
        max_pairs=args.max_val_pairs,
        instance_cache=args.instance_cache,
    ) if val_scan is not None else None
    print(f"pairs: train={len(train_dataset)}, val={len(val_dataset) if val_dataset is not None else 0}")
    if not args.no_preload and not args.dry_run:
        datasets=[d for d in [train_dataset,val_dataset] if d is not None]
        total_unique=sum(d.unique_instance_count() for d in datasets)
        per_instance=max(d.estimated_cached_instance_bytes() for d in datasets)
        budget=max(int(args.preload_max_gb * 1024**3 / per_instance),1)
        for d in datasets:
            limit=min(d.unique_instance_count(),max(1,int(budget*d.unique_instance_count()/total_unique)))
            d.preload_instances_limited(limit)
    device = select_device(args.device)
    print(f"device: {device}")
    model = MaskAwareAssociationModel(
        args.embedding_dim, args.encoder_width, args.hidden_dim, args.encoder_arch
    ).to(device)
    pretrained_report = None
    if args.encoder_arch == "osnet_x0_25" and not args.no_pretrained:
        if not args.osnet_pretrained.is_file():
            raise FileNotFoundError(
                f"official OSNet-x0.25 ImageNet weights are missing: {args.osnet_pretrained.resolve()}"
            )
        backbone = model.encoder.features
        if not isinstance(backbone, OSNetX025FeatureExtractor):
            raise TypeError("OSNet encoder did not construct the expected backbone")
        pretrained_report = load_osnet_x0_25_imagenet(backbone, args.osnet_pretrained)
        print(
            f"OSNet pretrained: matched={pretrained_report['matched_layers']}, "
            f"discarded={pretrained_report['discarded_layers']}, mask_channel_zero=True"
        )
    if args.dry_run:
        first, second, geometry, target = next(iter(DataLoader(train_dataset, batch_size=min(args.batch_size, 4))))
        model.eval()
        with torch.inference_mode():
            logits, first_embedding, _ = model(first.to(device), second.to(device), geometry.to(device))
        print(
            f"dry-run OK: crops={tuple(first.shape)}, geometry={tuple(geometry.shape)}, "
            f"embedding={tuple(first_embedding.shape)}, logits={tuple(logits.shape)}, positives={int(target.sum())}"
        )
        return
    run_dir = increment_path(args.project / args.name, exist_ok=args.exist_ok, mkdir=True)
    print(f"run directory: {run_dir.resolve()}", flush=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=len(train_dataset) >= args.batch_size,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    ) if val_dataset is not None else None
    optimizer = build_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_auc = -math.inf
    stale_epochs = 0
    history: list[dict[str, float]] = []
    serializable_args = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    serializable_args["pretrained_report"] = pretrained_report
    training_started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_dataset.set_epoch(epoch)
        model.train()
        backbone_frozen = args.encoder_arch == "osnet_x0_25" and epoch <= args.freeze_backbone_epochs
        set_osnet_backbone_frozen(model, backbone_frozen)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        running_loss = 0.0
        samples = 0
        total_batches = len(train_loader)
        epoch_started = time.perf_counter()
        previous_finished = epoch_started
        phase = f"epoch {epoch:03d}/{args.epochs:03d} train"
        print(
            f"{phase} start: batches={total_batches}, pairs={len(train_dataset)}, "
            f"backbone_frozen={backbone_frozen}",
            flush=True,
        )
        for batch_index, (first, second, geometry, target) in enumerate(train_loader, 1):
            batch_ready = time.perf_counter()
            data_seconds = batch_ready - previous_finished
            first, second = first.to(device, non_blocking=True), second.to(device, non_blocking=True)
            geometry, target = geometry.to(device, non_blocking=True), target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits, first_embedding, second_embedding = model(first, second, geometry)
                classification = F.binary_cross_entropy_with_logits(logits, target)
                embedding = contrastive_embedding_loss(
                    first_embedding, second_embedding, target, args.negative_margin
                )
                loss = classification + args.embedding_loss_weight * embedding
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            batch_loss = float(loss.detach())
            running_loss += batch_loss * len(target)
            samples += len(target)
            finished = time.perf_counter()
            if should_log_batch(batch_index, total_batches, args.log_interval):
                print(
                    progress_text(
                        phase,
                        batch_index,
                        total_batches,
                        running_loss,
                        samples,
                        batch_loss,
                        data_seconds,
                        finished - batch_ready,
                        epoch_started,
                        finished,
                        device,
                    ),
                    flush=True,
                )
            previous_finished = time.perf_counter()
        scheduler.step()
        metrics = evaluate(
            model, val_loader, device,
            progress_label=f"epoch {epoch:03d}/{args.epochs:03d} val",
            log_interval=args.log_interval,
        ) if val_loader is not None else {}
        epoch_seconds = time.perf_counter() - epoch_started
        average_epoch_seconds = (time.perf_counter() - training_started) / epoch
        training_eta = average_epoch_seconds * (args.epochs - epoch)
        metrics.update(
            {
                "epoch": float(epoch),
                "train_loss": running_loss / max(samples, 1),
                "lr": optimizer.param_groups[0]["lr"],
                "lr_groups": {group.get("name", str(index)): group["lr"] for index, group in enumerate(optimizer.param_groups)},
                "backbone_frozen": backbone_frozen,
                "epoch_seconds": epoch_seconds,
            }
        )
        history.append(metrics)
        print(f"epoch {epoch:03d}: train_loss={metrics['train_loss']:.4f} "
              f"elapsed={format_duration(epoch_seconds)} training_eta={format_duration(training_eta)}",flush=True)
        save_association_checkpoint(run_dir / "final.pt", model, epoch, metrics, serializable_args)
        if val_loader is not None:
            score=metrics["auc"] if math.isfinite(metrics["auc"]) else metrics["f1"]
            if score>best_auc:
                best_auc=score
                stale_epochs=0
                save_association_checkpoint(run_dir / "best.pt",model,epoch,metrics,serializable_args)
            else:
                stale_epochs+=1
        (run_dir / "metrics.json").write_text(json.dumps(history,indent=2),encoding="utf-8")
        if val_loader is not None and args.patience>0 and stale_epochs>=args.patience:
            break
    summary = {
        "dataset": str(args.dataset.resolve()),
        "labels": labels,
        "train_clips": len(train_scan.clips),
        "val_clips": (len(val_scan.clips) if val_scan is not None else 0),
        "train_pairs": len(train_dataset),
        "val_pairs": (len(val_dataset) if val_dataset is not None else 0),
        "best_auc": best_auc if val_scan is not None else None,
        "model_config": model.model_config,
        "pretrained": pretrained_report,
        "skipped_train": train_scan.skipped,
        "skipped_val": val_scan.skipped if val_scan is not None else [],
    }
    (run_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved: {run_dir.resolve()}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise
