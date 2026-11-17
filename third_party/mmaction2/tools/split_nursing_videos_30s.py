import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path


CLASSES = ("nursing", "other")
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split nursing videos into 30-second units.")
    # Source directory of raw class videos and destination 30s split directory (raw videos not bundled in release)
    parser.add_argument("--src-root", default="working_data/raw_nursing_videos")
    parser.add_argument("--dst-root", default="data/DS1_screening")
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing split videos in the destination directory.")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    return parser.parse_args()


def run_command(command):
    subprocess.run(command, check=True)


def get_duration(path, ffprobe):
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(
        command, check=True, capture_output=True, text=True)
    return float(result.stdout.strip())


def get_segments(duration, seconds):
    if duration <= seconds:
        return [(0.0, duration)]

    full_units = int(duration // seconds)
    remainder = duration - full_units * seconds
    if remainder > 0.001:
        segment_count = full_units
    else:
        segment_count = full_units

    segments = []
    for index in range(segment_count):
        start = index * seconds
        if index == segment_count - 1:
            end = duration
        else:
            end = (index + 1) * seconds
        segments.append((start, end - start))
    return segments


def split_video(src_path, dst_dir, seconds, ffmpeg, ffprobe, overwrite):
    duration = get_duration(src_path, ffprobe)
    segments = get_segments(duration, seconds)
    dst_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    if len(segments) == 1 and duration <= seconds:
        dst_path = dst_dir / src_path.name
        if dst_path.exists() and not overwrite:
            raise FileExistsError(f"Destination exists: {dst_path}")
        shutil.copy2(src_path, dst_path)
        outputs.append(dst_path)
        return {
            "source": str(src_path),
            "duration": duration,
            "segments": [{"path": str(dst_path), "start": 0.0, "duration": duration}],
        }

    width = max(3, int(math.log10(len(segments))) + 1)
    for index, (start, segment_duration) in enumerate(segments, start=1):
        dst_name = f"{src_path.stem}_part{index:0{width}d}{src_path.suffix}"
        dst_path = dst_dir / dst_name
        if dst_path.exists() and not overwrite:
            raise FileExistsError(f"Destination exists: {dst_path}")
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y" if overwrite else "-n",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(src_path),
            "-t",
            f"{segment_duration:.3f}",
            "-map",
            "0",
            "-c",
            "copy",
            "-avoid_negative_ts",
            "make_zero",
            str(dst_path),
        ]
        run_command(command)
        outputs.append(dst_path)

    return {
        "source": str(src_path),
        "duration": duration,
        "segments": [
            {"path": str(path), "start": start, "duration": seg_duration}
            for path, (start, seg_duration) in zip(outputs, segments)
        ],
    }


def main():
    args = parse_args()
    src_root = Path(args.src_root).resolve()
    dst_root = Path(args.dst_root).resolve()

    reports = []
    for class_name in CLASSES:
        src_dir = src_root / class_name
        if not src_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {src_dir}")
        dst_dir = dst_root / class_name
        videos = [
            path for path in sorted(src_dir.iterdir(), key=lambda item: item.name.lower())
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        ]
        for src_path in videos:
            report = split_video(
                src_path,
                dst_dir,
                seconds=args.seconds,
                ffmpeg=args.ffmpeg,
                ffprobe=args.ffprobe,
                overwrite=args.overwrite,
            )
            reports.append(report)
            print(
                f"{class_name}: {src_path.name} -> "
                f"{len(report['segments'])} segment(s)")

    dst_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "src_root": str(src_root),
        "dst_root": str(dst_root),
        "seconds": args.seconds,
        "source_videos": len(reports),
        "output_videos": sum(len(item["segments"]) for item in reports),
        "classes": {},
    }
    for class_name in CLASSES:
        class_dir = dst_root / class_name
        summary["classes"][class_name] = len([
            path for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        ])

    (dst_root / "split_30s_report.json").write_text(
        json.dumps({"summary": summary, "videos": reports}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
