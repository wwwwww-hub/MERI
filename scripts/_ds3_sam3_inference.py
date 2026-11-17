"""Run the SAM3 part of DS3 evaluation in its own Python environment."""
import argparse
import gc
import json
from pathlib import Path
import shutil

import numpy as np
from pycocotools import mask as mask_utils
import torch

import predict_sam3_piglet_tracks as sam3


def encode(mask):
    rle = mask_utils.encode(np.asfortranarray(mask, dtype=np.uint8))
    rle['counts'] = rle['counts'].decode('ascii')
    return rle


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding='utf-8')


def prepare_clip(request, root):
    paths = tuple(Path(p) for p in request['paths'])
    anchor = request['anchor']
    directory = root / 'sam3_clips' / request['name']
    if paths != tuple(paths[0].parent / f'{i:06d}.jpg' for i in range(len(paths))):
        frames = directory / 'frames'
        frames.mkdir(parents=True)
        for i, path in enumerate(paths):
            shutil.copyfile(path, frames / f'{i:06d}.jpg')
        paths = tuple(frames / f'{i:06d}.jpg' for i in range(len(paths)))
    anchor_time = anchor / 2
    clip = sam3.ClipSpec(request['name'], paths[anchor], paths[anchor], paths[anchor], None,
                         0., len(paths) / 2, anchor_time, anchor_time)
    return sam3.PreparedClip(clip, directory, paths[0].parent, paths, 0., anchor, 2.)


def manual_seeds(masks):
    ordered = []
    for raster in masks:
        mask = mask_utils.decode(raster).astype(bool)
        ys, xs = np.nonzero(mask)
        ordered.append((float(xs.mean()), float(ys.mean()), mask,
                        (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))))
    ordered.sort(key=lambda item: (item[0], item[1]))
    return [sam3.SeedInstance(i, 'piglet', mask, 1., box)
            for i, (_, _, mask, box) in enumerate(ordered, 1)]


def detect(clips, args, root):
    model_spec = sam3.requested_seed_models(args)[0]
    model, processor = sam3.load_seed_processor(args, model_spec)
    anchors = {}
    for clip in clips:
        frames = []
        for i, path in enumerate(clip.frame_paths):
            detections = sam3.infer_seed_instances(processor, path, model_spec, args)
            frames.append([(encode(x.mask), x.score) for x in detections])
            if args.tracker == 'sam3' and i == clip.anchor_frame_index:
                anchors[clip.spec.clip_id] = sam3.finalize_seed_instances(
                    {'piglet': detections}, [model_spec], args) if detections else []
        save(root / 'sam3_detections' / f'{clip.spec.clip_id}.json', frames)
        print(f'SAM3 image masks: {clip.spec.clip_id}', flush=True)
    del processor, model
    gc.collect()
    torch.cuda.empty_cache()
    return anchors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('request', type=Path)
    parser.add_argument('--device', required=True)
    parser.add_argument('--mask-source', choices=('manual', 'sam3'), required=True)
    parser.add_argument('--tracker', choices=('osnet', 'sam3'), required=True)
    args = parser.parse_args()
    args.sam3_root = sam3.DEFAULT_SAM3_ROOT
    args.bpe_path = sam3.DEFAULT_BPE_PATH
    args.piglet_checkpoint = sam3.DEFAULT_PIGLET_CHECKPOINT
    args.video_checkpoint = sam3.DEFAULT_VIDEO_CHECKPOINT
    args.piglet_label = args.piglet_prompt = 'piglet'
    args.sow_label = 'sow'
    args.skip_sow = True
    args.offload_state_to_cpu = True
    args.overlap_policy = 'all'
    args.tracking_seed = 'anchor-json' if args.mask_source == 'manual' else 'sam3-cache'
    args.min_presence_prob, args.min_track_mask_pixels = .10, 40
    args.piglet_confidence_threshold, args.mask_threshold = .5, .5
    args.min_seed_mask_pixels, args.max_seed_instances = 80, 0
    args.dedupe_iou_threshold = .65
    args.containment_threshold, args.child_max_area_ratio = .85, .75

    sam3.activate_cuda_device(args.device)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    root = args.request.parent
    requests = json.loads(args.request.read_text(encoding='utf-8'))
    clips = [prepare_clip(request, root) for request in requests]
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        if args.mask_source == 'sam3':
            anchors = detect(clips, args, root)
        else:
            anchors = {request['name']: manual_seeds(request['anchor_masks']) for request in requests}
        if args.tracker == 'sam3':
            predictor = sam3.load_official_tracker(args)
            for clip in clips:
                name = clip.spec.clip_id
                frames = {i: [] for i in range(len(clip.frame_paths))}
                seeds = anchors.pop(name)
                if seeds:
                    for i, tracks, _ in sam3.iter_propagated_frames(predictor, clip, seeds, args):
                        frames[i] = [(x.track_id, encode(x.mask)) for x in tracks if x.mask.any()]
                save(root / 'sam3_tracks' / f'{name}.json', frames)
                print(f'SAM3 video tracks: {name}', flush=True)


if __name__ == '__main__':
    main()
