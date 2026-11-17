import argparse
import json
import random
import re
from pathlib import Path


CLASSES = (("nursing", 0), ("other", 1))
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")
PART_SUFFIX_RE = re.compile(r"_part\d+$", re.IGNORECASE)
TIME_RANGE_SUFFIX_RE = re.compile(r"_\d{6}-\d{6}$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create MMAAction2 annotation files for the nursing dataset."
    )
    parser.add_argument(
        "--data-root",
        # Root of 30s screening dataset; annotation lists are in data/DS1_screening/annotations (raw videos not bundled)
        default="data/DS1_screening",
        help="Dataset root with class subdirectories.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory for generated annotations. Defaults to DATA_ROOT/annotations.",
    )
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio", type=float, default=0.20)
    return parser.parse_args()


def collect_videos(data_root):
    samples = []
    for class_name, label in CLASSES:
        class_dir = data_root / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")
        for path in sorted(class_dir.iterdir(), key=lambda item: item.name.lower()):
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
                rel_path = path.relative_to(data_root).as_posix()
                samples.append((class_name, rel_path, label))
    return samples


def get_group_id(rel_path):
    rel = Path(rel_path)
    class_name = rel.parts[0]
    source_stem = PART_SUFFIX_RE.sub("", rel.stem)
    source_stem = TIME_RANGE_SUFFIX_RE.sub("", source_stem)
    return class_name, f"{class_name}/{source_stem}"


def collect_groups_from_report(data_root, samples):
    report_path = data_root / "split_30s_report.json"
    if not report_path.is_file():
        return None

    existing = {rel_path for _, rel_path, _ in samples}
    reported = set()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    groups = []
    for video in report["videos"]:
        group_items = []
        group_class = None
        for segment in video["segments"]:
            segment_path = Path(segment["path"])
            rel_path = segment_path.relative_to(data_root).as_posix()
            reported.add(rel_path)
            if rel_path not in existing:
                continue
            class_name = Path(rel_path).parts[0]
            label = dict(CLASSES)[class_name]
            group_class = class_name
            group_items.append((rel_path, label))
        if not group_items:
            continue
        source = Path(video["source"])
        group_id = f"{group_class}/{source.stem}"
        groups.append((group_class, group_id, group_items))

    extra_groups = {}
    for class_name, rel_path, label in samples:
        if rel_path in reported:
            continue
        _, group_id = get_group_id(rel_path)
        extra_groups.setdefault((class_name, group_id), []).append((rel_path, label))

    for (class_name, group_id), items in sorted(extra_groups.items()):
        groups.append((class_name, group_id, sorted(items)))
    return groups


def make_single_video_groups(class_samples):
    return [
        (class_name, rel_path, [(rel_path, label)])
        for class_name, rel_path, label in class_samples
    ]


def split_class_groups(groups, seed, train_ratio, val_ratio):
    ratio_sum = train_ratio + val_ratio
    if abs(ratio_sum - 1.0) > 1e-8:
        raise ValueError("train/val ratios must sum to 1.0")

    by_class = {}
    for class_name, group_id, items in groups:
        by_class.setdefault(class_name, []).append((group_id, items))

    split = {"train": [], "val": []}
    split_groups = {"train": [], "val": []}
    rng = random.Random(seed)
    for class_name, class_groups in by_class.items():
        class_groups = list(class_groups)
        rng.shuffle(class_groups)
        n_total = len(class_groups)
        n_train = int(round(n_total * train_ratio))
        train_groups = class_groups[:n_train]
        val_groups = class_groups[n_train:]
        for group_id, items in train_groups:
            split["train"].extend(items)
            split_groups["train"].append((class_name, group_id, len(items)))
        for group_id, items in val_groups:
            split["val"].extend(items)
            split_groups["val"].append((class_name, group_id, len(items)))

    for items in split.values():
        items.sort(key=lambda item: item[0].lower())
    for groups_for_split in split_groups.values():
        groups_for_split.sort(key=lambda item: item[1].lower())
    return split, split_groups


def write_list(path, items):
    path.write_text(
        "".join(f"{rel_path} {label}\n" for rel_path, label in items),
        encoding="utf-8",
    )


def summarize(split, split_groups=None):
    summary = {}
    for split_name, items in split.items():
        class_counts = {class_name: 0 for class_name, _ in CLASSES}
        labels_to_names = {label: class_name for class_name, label in CLASSES}
        for _, label in items:
            class_counts[labels_to_names[label]] += 1
        summary[split_name] = {
            "total": len(items),
            "classes": class_counts,
        }
        if split_groups is not None:
            group_counts = {class_name: 0 for class_name, _ in CLASSES}
            for class_name, _, _ in split_groups[split_name]:
                group_counts[class_name] += 1
            summary[split_name]["groups"] = group_counts
    summary["all"] = {
        "total": sum(part["total"] for part in summary.values()),
        "classes": {
            class_name: sum(part["classes"][class_name] for part in summary.values())
            for class_name, _ in CLASSES
        },
    }
    return summary


def main():
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else data_root / "annotations"
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = collect_videos(data_root)
    groups = collect_groups_from_report(data_root, samples)
    group_source = "split_30s_report.json"
    if groups is None:
        groups = make_single_video_groups(samples)
        group_source = "individual videos"
    split, split_groups = split_class_groups(
        groups,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    write_list(out_dir / "train.txt", split["train"])
    write_list(out_dir / "val.txt", split["val"])
    write_list(out_dir / "trainval.txt", split["train"] + split["val"])
    stale_test_file = out_dir / "test.txt"
    if stale_test_file.exists():
        stale_test_file.unlink()
    write_list(
        out_dir / "all.txt",
        sorted(
            [(rel_path, label) for _, rel_path, label in samples],
            key=lambda item: item[0].lower(),
        ),
    )

    (out_dir / "classes.txt").write_text(
        "".join(f"{class_name}\n" for class_name, _ in CLASSES),
        encoding="utf-8",
    )
    (out_dir / "label_map.txt").write_text(
        "".join(f"{label} {class_name}\n" for class_name, label in CLASSES),
        encoding="utf-8",
    )

    summary = summarize(split, split_groups=split_groups)
    summary["seed"] = args.seed
    summary["ratios"] = {
        "train": args.train_ratio,
        "val": args.val_ratio,
    }
    summary["data_root"] = str(data_root)
    summary["group_source"] = group_source
    summary["groups"] = {
        "total": len(groups),
        "train": len(split_groups["train"]),
        "val": len(split_groups["val"]),
    }
    (out_dir / "split_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
