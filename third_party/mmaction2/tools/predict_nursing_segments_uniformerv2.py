# Copyright (c) OpenMMLab. All rights reserved.
"""Find nursing intervals with a trained UniFormerV2 recognizer.

This script reuses the Swin long-video prediction workflow, but swaps the
model defaults and frame sampler for UniFormerV2's ``UniformSample`` test
pipeline. Positive windows are decided by the nursing class score threshold
only; the nursing class does not need to be the model top-1 prediction.
"""

import argparse
import gc
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Set, Tuple

import numpy as np
from mmengine import Config, DictAction
from mmengine.dataset import Compose, pseudo_collate
import torch

from mmaction.apis import init_recognizer

import predict_nursing_segments_swin as runner

_ORIGINAL_ITER_VIDEOS = runner.base.iter_videos


DEFAULT_CONFIG = 'configs/uniformerv2.py'
DEFAULT_CHECKPOINT = 'models/screening/uniformerv2.pt'
DEFAULT_LABELS = None
DEFAULT_OUTPUT = 'reproduced/screening/nursing_segments_uniformerv2.csv'
DEFAULT_DETAIL_OUTPUT = 'reproduced/screening/nursing_windows_uniformerv2.csv'
# Optional output directory for clipped nursing segments (external working directory, not bundled in release)
DEFAULT_CLIP_OUTPUT_DIR = 'working_data/nursing_clips'


_CURRENT_ARGS: Optional[argparse.Namespace] = None
_RESUME_COMPLETED_VIDEOS: Optional[Set[str]] = None
_RESUME_REMAINING_VIDEOS: Optional[List[Path]] = None


@dataclass(frozen=True)
class UniformSamplingConfig:
    clip_len: int
    num_clips: int
    frame_interval = None

    @property
    def frames_per_window(self) -> int:
        return self.clip_len * self.num_clips


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Predict nursing intervals from long videos with '
        'UniFormerV2.')
    parser.add_argument(
        '--config', default=DEFAULT_CONFIG, help='MMAction2 config file.')
    parser.add_argument(
        '--checkpoint',
        default=DEFAULT_CHECKPOINT,
        help='Model checkpoint. Wildcards are allowed.')
    parser.add_argument(
        '--input-dir',
        # Directory for raw long NVR recordings to screen (not bundled in release)
        default='working_data/raw_videos',
        help='Directory containing videos to process.')
    parser.add_argument(
        '--labels',
        default=DEFAULT_LABELS,
        help='Class label file. Defaults to classes in the config.')
    parser.add_argument(
        '--class-name',
        default='nursing',
        help='Positive class name to summarize.')
    parser.add_argument(
        '--device', default='cuda:0', help='Inference device, e.g. cuda:0/cpu.')
    parser.add_argument(
        '--output',
        default=DEFAULT_OUTPUT,
        help='Merged segment CSV output path.')
    parser.add_argument(
        '--detail-output',
        default=DEFAULT_DETAIL_OUTPUT,
        help='Per-window CSV output path.')
    parser.add_argument(
        '--status-output',
        default=None,
        help='Per-video status CSV output path. Defaults to output stem + '
        '_status.csv.')
    parser.add_argument(
        '--clip-output-dir',
        default=DEFAULT_CLIP_OUTPUT_DIR,
        help='If set, export predicted positive segments into this directory.')
    parser.add_argument(
        '--clip-method',
        choices=['copy', 'reencode'],
        default='copy',
        help='How to export positive clips. copy is fast but keyframe-aligned; '
        'reencode is slower but usually more accurate.')
    parser.add_argument(
        '--ffmpeg-path',
        default='ffmpeg',
        help='Path to ffmpeg executable used for clip export.')
    parser.add_argument(
        '--overwrite-clips',
        action='store_true',
        help='Overwrite existing exported positive clips.')
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Skip source videos that already have exported clips in '
        'clip-output-dir and append new results.')
    parser.add_argument(
        '--stop-on-error',
        action='store_true',
        help='Stop immediately when one video fails. By default errors are '
        'recorded and remaining videos continue.')
    parser.add_argument(
        '--save-full-scores',
        action='store_true',
        help='Write full model score vectors to detail CSV. This can be large.')
    parser.add_argument(
        '--window-sec',
        type=float,
        default=30.0,
        help='Sliding window length in seconds.')
    parser.add_argument(
        '--stride-sec',
        type=float,
        default=15.0,
        help='Sliding window stride in seconds.')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Number of windows inferred in one model batch.')
    parser.add_argument(
        '--min-score',
        type=float,
        default=0.8,
        help='Minimum positive class probability for a nursing window.')
    parser.add_argument(
        '--merge-gap-sec',
        type=float,
        default=None,
        help='Merge positive windows if the gap is not larger than this. '
        'Defaults to stride-sec.')
    parser.add_argument(
        '--clip-len',
        type=int,
        default=None,
        help='Override UniformSample clip_len. Defaults to the config value.')
    parser.add_argument(
        '--num-clips',
        type=int,
        default=None,
        help='Override UniformSample num_clips. Defaults to the config value.')
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Search videos recursively under input-dir.')
    parser.add_argument(
        '--max-videos',
        type=int,
        default=None,
        help='Only process the first N videos.')
    parser.add_argument(
        '--start-index',
        type=int,
        default=1,
        help='1-based index of the first video to process after sorting.')
    parser.add_argument(
        '--end-index',
        type=int,
        default=None,
        help='1-based inclusive index of the last video to process after '
        'sorting.')
    parser.add_argument(
        '--limit-windows',
        type=int,
        default=None,
        help='Only process the first N windows in each video.')
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='List videos and exit without loading the model.')
    parser.add_argument(
        '--decord-threads',
        type=int,
        default=4,
        help='Number of decord reader threads per video.')
    parser.add_argument(
        '--decode-device',
        default='cpu',
        help='Video decode device for Decord: cpu or cuda:0. '
        'Requires a CUDA-enabled Decord build for cuda; falls back to cpu.')
    parser.add_argument(
        '--progress-interval',
        type=float,
        default=10.0,
        help='Seconds between progress messages. Set 0 to print every batch.')
    parser.add_argument(
        '--profile',
        action='store_true',
        help='Print time spent in frame reading, preprocessing and inference.')
    parser.add_argument(
        '--no-tqdm',
        action='store_true',
        help='Use plain text progress messages instead of tqdm progress bars.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override config options, e.g. model.cls_head.dropout_ratio=0.0')
    args = parser.parse_args()
    global _CURRENT_ARGS, _RESUME_COMPLETED_VIDEOS, _RESUME_REMAINING_VIDEOS
    _CURRENT_ARGS = args
    if args.resume:
        _RESUME_COMPLETED_VIDEOS = completed_videos_from_clip_output_dir(
            log=True)
        _RESUME_REMAINING_VIDEOS = [
            video_path for video_path in selected_input_videos(args)
            if video_path.name not in _RESUME_COMPLETED_VIDEOS
        ]
        runner.base.log_message(
            f'[INFO] Resume precheck: {len(_RESUME_REMAINING_VIDEOS)} '
            'video(s) remain after filtering existing outputs.')
        if not _RESUME_REMAINING_VIDEOS:
            raise SystemExit(0)
        args.start_index = 1
        args.end_index = None
        args.max_videos = None
    else:
        _RESUME_COMPLETED_VIDEOS = set()
        _RESUME_REMAINING_VIDEOS = None
    return args


def resolve_checkpoint(path_text: str) -> Path:
    path = runner.base.resolve_path(path_text)
    if any(char in str(path) for char in '*?['):
        matches = sorted(path.parent.glob(path.name))
        if not matches:
            raise FileNotFoundError(
                f'Checkpoint pattern did not match any file: {path}. '
                'Please pass --checkpoint explicitly.')
        return max(matches, key=lambda item: item.stat().st_mtime)
    return path


def extract_sampling_config(cfg: Config,
                            args: argparse.Namespace) -> UniformSamplingConfig:
    uniform_cfg = None
    for transform in cfg.test_pipeline:
        if transform['type'] == 'UniformSample':
            uniform_cfg = transform
            break
    if uniform_cfg is None:
        raise ValueError('The UniFormerV2 test_pipeline must contain '
                         'UniformSample.')

    clip_len = args.clip_len or int(uniform_cfg['clip_len'])
    num_clips = args.num_clips or int(uniform_cfg.get('num_clips', 1))
    if clip_len <= 0 or num_clips <= 0:
        raise ValueError('--clip-len and --num-clips must be positive.')
    return UniformSamplingConfig(clip_len=clip_len, num_clips=num_clips)


def build_array_pipeline(cfg: Config) -> Compose:
    pipeline_cfg = []
    skip_types = {
        'DecordInit', 'OpenCVInit', 'PyAVInit', 'UniformSample',
        'SampleFrames', 'DenseSampleFrames', 'DecordDecode', 'OpenCVDecode',
        'PyAVDecode'
    }
    decode_inserted = False

    for transform in cfg.test_pipeline:
        transform_type = transform['type']
        if transform_type in {'DecordDecode', 'OpenCVDecode', 'PyAVDecode'}:
            if not decode_inserted:
                pipeline_cfg.append(dict(type='ArrayDecode'))
                decode_inserted = True
            continue
        if transform_type in skip_types:
            continue
        pipeline_cfg.append(transform.copy())

    if not decode_inserted:
        pipeline_cfg.insert(0, dict(type='ArrayDecode'))

    return Compose(pipeline_cfg)


def prepare_model(args: argparse.Namespace, labels: List[str]):
    config_path = runner.base.resolve_path(args.config)
    checkpoint_path = resolve_checkpoint(args.checkpoint)

    if not config_path.is_file():
        raise FileNotFoundError(f'Config file not found: {config_path}')
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'Checkpoint file not found: {checkpoint_path}')

    cfg = Config.fromfile(str(config_path))
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    sampling = extract_sampling_config(cfg, args)
    checkpoint_num_classes = runner.base.infer_checkpoint_num_classes(
        checkpoint_path)
    if checkpoint_num_classes is not None:
        cfg.model.cls_head.num_classes = checkpoint_num_classes
        if checkpoint_num_classes != len(labels):
            print('[WARN] Checkpoint has '
                  f'{checkpoint_num_classes} classes, label file has '
                  f'{len(labels)} labels.')

    model = init_recognizer(cfg, str(checkpoint_path), device=args.device)
    pipeline = build_array_pipeline(cfg)
    print('[INFO] UniFormerV2 sampling: '
          f'clip_len={sampling.clip_len}, '
          f'num_clips={sampling.num_clips}, '
          f'frames/window={sampling.frames_per_window}')
    return model, pipeline, sampling


class RobustFrameReader(runner.base.FrameReader):
    """FrameReader with OpenCV fallback when Decord fails during reading."""

    _warned_runtime_fallbacks = set()

    def _switch_decord_to_opencv(self, exc: Exception) -> None:
        if not self.is_decord:
            raise exc

        warning_key = str(exc).splitlines()[0]
        if warning_key not in self._warned_runtime_fallbacks:
            print('[WARN] Decord read failed; retry this video with OpenCV: '
                  f'{exc}')
            self._warned_runtime_fallbacks.add(warning_key)

        self.vr = None
        self.is_decord = False
        self.backend = 'opencv'
        gc.collect()

        self.cap = runner.base.cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f'Cannot open video with OpenCV: '
                               f'{self.video_path}') from exc

        total_frames = int(self.cap.get(runner.base.cv2.CAP_PROP_FRAME_COUNT))
        fps = float(self.cap.get(runner.base.cv2.CAP_PROP_FPS))
        if total_frames > 0:
            self.total_frames = total_frames
        if fps > 0:
            self.fps = fps

    def read(self, frame_indices: Sequence[int]) -> np.ndarray:
        try:
            return super().read(frame_indices)
        except Exception as exc:
            self._switch_decord_to_opencv(exc)
            return super().read(frame_indices)

    def read_windows(self,
                     window_frame_indices: Sequence[np.ndarray]
                     ) -> Tuple[List[np.ndarray], int, int]:
        try:
            return super().read_windows(window_frame_indices)
        except Exception as exc:
            self._switch_decord_to_opencv(exc)
            return super().read_windows(window_frame_indices)

    def close(self) -> None:
        super().close()
        self.vr = None
        gc.collect()


def predict_batch(model, pipeline: Compose, sampling: UniformSamplingConfig,
                  frame_batches: Sequence[np.ndarray],
                  device: str = 'cuda:0') -> Tuple[List[List[float]], float,
                                                   float]:
    preprocess_start = time.time()
    data_list = [
        pipeline(runner.make_data(frames, sampling))
        for frames in frame_batches
    ]
    data = pseudo_collate(data_list)
    preprocess_time = time.time() - preprocess_start

    if str(device).startswith('cuda') and torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_start = time.time()
    with torch.no_grad():
        results = model.test_step(data)
    if str(device).startswith('cuda') and torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_time = time.time() - inference_start
    scores = [result.pred_score.detach().cpu().tolist() for result in results]

    del results, data, data_list
    if str(device).startswith('cuda') and torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return scores, preprocess_time, inference_time


def sample_window_indices(start_frame: int, end_frame: int,
                          sampling: UniformSamplingConfig):
    local_total = max(1, end_frame - start_frame)
    seg_size = float(local_total - 1) / sampling.clip_len
    duration = seg_size / (sampling.num_clips + 1)

    frame_indices = []
    for clip_idx in range(sampling.num_clips):
        for frame_idx in range(sampling.clip_len):
            seg_start = int(round(seg_size * frame_idx))
            local_index = seg_start + int(duration * (clip_idx + 1))
            local_index = max(0, min(local_total - 1, local_index))
            frame_indices.append(local_index + start_frame)

    return np.asarray(frame_indices, dtype=np.int64)




def selected_input_videos(args: argparse.Namespace) -> List[Path]:
    input_dir = runner.base.resolve_path(args.input_dir)
    videos = _ORIGINAL_ITER_VIDEOS(input_dir, args.recursive)
    start_offset = args.start_index - 1
    end_offset = args.end_index
    videos = videos[start_offset:end_offset]
    if args.max_videos is not None:
        videos = videos[:args.max_videos]
    return videos


def completed_videos_from_clip_output_dir(
        _status_path: Optional[Path] = None,
        log: bool = False) -> Set[str]:
    """Return input video names that already have clips in clip-output-dir."""
    args = _CURRENT_ARGS
    if args is None or not getattr(args, 'resume', False):
        return set()
    if not args.clip_output_dir:
        if log:
            runner.base.log_message(
                '[INFO] Resume precheck: --clip-output-dir is empty; '
                '0 video(s) will be skipped because of existing clips.')
        return set()

    clip_output_dir = runner.base.resolve_path(args.clip_output_dir)
    selected_videos = selected_input_videos(args)
    if not clip_output_dir.exists():
        if log:
            runner.base.log_message(
                f'[INFO] Resume precheck: clip-output-dir does not exist: '
                f'{clip_output_dir.resolve()}; 0/{len(selected_videos)} '
                'video(s) will be skipped because of existing clips.')
        return set()

    safe_to_names = {
        runner.base.safe_filename(video_path.stem): video_path.name
        for video_path in selected_videos
    }

    completed = set()
    suffixes = runner.base.VIDEO_SUFFIXES
    for output_path in clip_output_dir.rglob('*'):
        if not output_path.is_file():
            continue
        if output_path.suffix.lower() not in suffixes:
            continue
        output_stem = output_path.stem
        parent_stem = output_path.parent.name
        for safe_stem, video_name in safe_to_names.items():
            if (parent_stem == safe_stem or output_stem == safe_stem or
                    output_stem.startswith(safe_stem + '_')):
                completed.add(video_name)
                break

    if log:
        runner.base.log_message(
            f'[INFO] Resume precheck: {len(completed)}/{len(selected_videos)} '
            f'video(s) will be skipped because exported clips already exist '
            f'in {clip_output_dir.resolve()}')
    return completed


def completed_videos_from_cached_clip_output_dir(_status_path: Path) -> Set[str]:
    if _RESUME_COMPLETED_VIDEOS is not None:
        return set(_RESUME_COMPLETED_VIDEOS)
    return completed_videos_from_clip_output_dir(_status_path, log=False)


def iter_videos_after_resume_filter(input_dir: Path, recursive: bool) -> List[Path]:
    if _RESUME_REMAINING_VIDEOS is not None:
        args = _CURRENT_ARGS
        requested_dir = runner.base.resolve_path(args.input_dir)
        if Path(input_dir).resolve() == requested_dir.resolve():
            return list(_RESUME_REMAINING_VIDEOS)
    return _ORIGINAL_ITER_VIDEOS(input_dir, recursive)

runner.parse_args = parse_args
runner.extract_sampling_config = extract_sampling_config
runner.build_array_pipeline = build_array_pipeline
runner.prepare_model = prepare_model
runner.predict_batch = predict_batch
runner.sample_window_indices = sample_window_indices

runner.base.FrameReader = RobustFrameReader
runner.base.iter_videos = iter_videos_after_resume_filter
runner.base.completed_videos_from_status = completed_videos_from_cached_clip_output_dir


if __name__ == '__main__':
    runner.main()