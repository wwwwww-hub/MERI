"""Evaluate DS3 identity association with mask-based HOTA and MOT identity metrics."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'third_party/motmetrics'),
                str(ROOT / 'third_party/sam3/sam3/eval/hota_eval_toolkit')]

import cv2
import motmetrics as mm
import numpy as np
from pycocotools import mask as mask_utils
from scipy.optimize import linear_sum_assignment
import torch
from trackeval.metrics import HOTA
from ultralytics.trackers.dataset_paths import clip_video, extract_video_frames, frame_file, manual_mask_file
from ultralytics.trackers.mask_aware import (
    MaskAwareOfflineTracker, TrackerConfig, _read_polygon_masks, encode_observation_crops,
    extract_masked_crop, load_association_checkpoint, observation_from_mask,
)


MOT_METRICS = ['idf1', 'idp', 'idr', 'idtp', 'idfp', 'idfn', 'precision', 'recall',
               'num_switches', 'num_fragmentations', 'num_objects', 'num_predictions',
               'num_unique_objects', 'mostly_tracked', 'partially_tracked', 'mostly_lost', 'num_frames',
               'num_false_positives']
HOTA_METRICS = ['HOTA', 'DetA', 'AssA', 'LocA']
PROJECTION_IOU = .05
COMMON_MATCH_METRICS = ['common_matched_instances', 'idsw_on_common_matches', 'yolo_idsw_on_common_matches']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'data/DS3_tracking')
    parser.add_argument('--ffmpeg', default='ffmpeg', help='FFmpeg executable used to read DS3 videos.')
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'models/tracking/osnet_x0_25.pt')
    parser.add_argument('--config', type=Path, default=ROOT / 'models/tracking/config.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'reproduced/tracking')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--tracker', choices=('osnet', 'sam3'), default='osnet')
    parser.add_argument('--mask-source', choices=('manual', 'sam3'), default='manual',
                        help='Input masks for association. Scoring uses the manual reference masks.')
    parser.add_argument('--sam3-python', default=sys.executable,
                        help='Python executable for SAM3 inference in a separate environment.')
    parser.add_argument('--common-matched-idsw', action='store_true',
                        help='For OSNet with manual masks, also count ID changes on reference instances '
                             'matched by both the manual-mask and YOLO26-mask OSNet runs.')
    args = parser.parse_args()
    if (args.tracker == 'sam3' or args.mask_source == 'sam3') and not args.device.startswith('cuda'):
        parser.error('SAM3 inference requires a CUDA device')
    if args.common_matched_idsw and (args.tracker, args.mask_source) != ('osnet', 'manual'):
        parser.error('--common-matched-idsw requires --tracker osnet --mask-source manual')
    with TemporaryDirectory(prefix='meri_ds3_test_') as temporary:
        evaluate(args, Path(temporary))


def project_tracks(propagated, masks):
    """Transfer track IDs only to matched masks. Unassigned masks have no prediction."""
    if not propagated or not masks:
        return []
    iou = mask_utils.iou([mask for _, mask in propagated], masks, [0] * len(masks))
    rows, columns = linear_sum_assignment(-iou)
    assigned = {j: propagated[i][0] for i, j in zip(rows, columns) if iou[i, j] >= PROJECTION_IOU}
    return [(assigned[j], mask) for j, mask in enumerate(masks) if j in assigned]


def score_clip(ground_truth, predictions, anchor_frame, hota=None):
    """Give TrackEval the full IoU matrices, while MOT uses IoU >= 0.50."""
    acc = mm.MOTAccumulator(auto_id=False)
    gt_indices, track_indices = {}, {}
    gt_ids, tracker_ids, similarities = [], [], []
    for fi, gt in ground_truth.items():
        if fi == anchor_frame:
            continue
        predicted = predictions[fi]
        reference_ids = [identity for identity, _ in gt]
        predicted_ids = [identity for identity, _ in predicted]
        iou = np.zeros((len(gt), len(predicted)), dtype=float)
        if gt and predicted:
            iou = mask_utils.iou([mask for _, mask in gt], [mask for _, mask in predicted], [0] * len(predicted))
        distance = 1 - iou
        distance[iou < .5] = np.nan
        acc.update(reference_ids, predicted_ids, distance, frameid=fi)
        if hota is not None:
            gt_ids.append(np.array([gt_indices.setdefault(i, len(gt_indices)) for i in reference_ids], dtype=int))
            tracker_ids.append(np.array([track_indices.setdefault(i, len(track_indices)) for i in predicted_ids], dtype=int))
            similarities.append(iou)

    if hota is None:
        return acc, None

    result = hota.eval_sequence(dict(
        gt_ids=gt_ids, tracker_ids=tracker_ids, similarity_scores=similarities,
        num_gt_ids=len(gt_indices), num_tracker_ids=len(track_indices),
        num_gt_dets=sum(map(len, gt_ids)), num_tracker_dets=sum(map(len, tracker_ids)),
        num_timesteps=len(similarities),
    ))
    return acc, result


def matched_ids(accumulator):
    events = accumulator.mot_events.reset_index()
    matches = events[events['Type'].isin(['MATCH', 'SWITCH'])]
    return {(int(frame), int(reference)): int(track) for frame, reference, track in
            matches[['FrameId', 'OId', 'HId']].itertuples(index=False, name=None)}


def common_matched_changes(manual_acc, yolo_acc):
    """Count ID changes between consecutive common observations of each reference trajectory."""
    manual, yolo = matched_ids(manual_acc), matched_ids(yolo_acc)
    common = sorted(manual.keys() & yolo.keys())
    previous = {}
    manual_changes = yolo_changes = 0
    for key in common:
        _, reference = key
        current = manual[key], yolo[key]
        if reference in previous:
            old_manual, old_yolo = previous[reference]
            manual_changes += current[0] != old_manual
            yolo_changes += current[1] != old_yolo
        previous[reference] = current
    return len(common), manual_changes, yolo_changes


def prepare_clips(test_dirs, decoded_root, ffmpeg):
    clips = {}
    for directory in test_dirs:
        video_input = clip_video(directory).is_file()
        decoded = decoded_root / directory.name
        if video_input:
            meta = extract_video_frames(directory, decoded, ffmpeg)
        else:
            meta = json.loads((directory / 'clip_metadata.json').read_text(encoding='utf-8'))
        ground_truth, frame_paths = {}, []
        for fi in range(int(meta['frame_count'])):
            image_path = decoded / f'{fi:06d}.jpg' if video_input else frame_file(directory, fi)
            mask_path = manual_mask_file(directory, fi) if video_input else frame_file(directory, fi, '.json')
            _, masks = _read_polygon_masks(mask_path)
            ground_truth[fi] = [
                (identity, mask_utils.encode(np.asfortranarray(mask, dtype=np.uint8)))
                for (label, identity), mask in sorted(masks.items()) if label == 'piglet' and mask.any()
            ]
            frame_paths.append(image_path)
        clips[directory.name] = (frame_paths, int(meta['anchor_frame_index']), ground_truth)
    return clips


def track_osnet(frame_paths, masks_by_frame, model, device, config):
    observations_by_frame = {}
    next_id = 1
    for fi, image_path in enumerate(frame_paths):
        image = cv2.imread(str(image_path))
        if image is None:
            raise OSError(image_path)
        observations, crops = [], []
        for raster, score in masks_by_frame[fi]:
            mask = mask_utils.decode(raster).astype(bool)
            ys, xs = np.nonzero(mask)
            observation = observation_from_mask(
                next_id, fi, 0, 'piglet', score, [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], mask,
                payload={'mask': raster},
            )
            next_id += 1
            observations.append(observation)
            crops.append(extract_masked_crop(image, mask, observation.bbox_xyxy))
        encode_observation_crops(model, observations, crops, device, 64)
        observations_by_frame[fi] = observations
    assignments = MaskAwareOfflineTracker(model, device, config).track(observations_by_frame)
    return {fi: [(assignments[o.observation_id], o.payload['mask']) for o in observations]
            for fi, observations in observations_by_frame.items()}


def yolo_masks(model, image_path, class_ids, device):
    image = cv2.imread(str(image_path))
    if image is None:
        raise OSError(image_path)
    result = model.predict(image, imgsz=1024, conf=.15, iou=.70, rect=True, retina_masks=True,
                           classes=class_ids, device=str(device), half=False, verbose=False)[0]
    if result.masks is None:
        return []
    masks = result.masks.data.cpu().numpy() > .5
    scores = result.boxes.conf.cpu().numpy()
    occupied = np.zeros(image.shape[:2], dtype=bool)
    detections = []
    for i in np.argsort(-scores):
        mask = masks[i] & ~occupied
        occupied |= mask
        if mask.any():
            raster = mask_utils.encode(np.asfortranarray(mask, dtype=np.uint8))
            detections.append((raster, float(scores[i])))
    return detections


def run_sam3(args, clips, decoded_root):
    request_file = decoded_root / 'sam3_request.json'
    requests = []
    for name, (paths, anchor, truth) in clips.items():
        anchor_masks = []
        if args.mask_source == 'manual':
            anchor_masks = [{'size': mask['size'], 'counts': mask['counts'].decode('ascii')}
                            for _, mask in truth[anchor]]
        requests.append(dict(name=name, paths=[str(p.resolve()) for p in paths], anchor=anchor,
                             anchor_masks=anchor_masks))
    request_file.write_text(json.dumps(requests), encoding='utf-8')
    subprocess.run([args.sam3_python, '-B', str(ROOT / 'scripts/_ds3_sam3_inference.py'), str(request_file),
                    '--device', args.device, '--mask-source', args.mask_source, '--tracker', args.tracker], check=True)


def evaluate(args, decoded_root):
    test_dirs = sorted(d for d in (args.dataset / 'test').iterdir() if d.is_dir())
    if len(test_dirs) != 15:
        raise ValueError('DS3 evaluation requires 15 test clips')
    clips = prepare_clips(test_dirs, decoded_root, args.ffmpeg)
    if args.tracker == 'sam3' or args.mask_source == 'sam3':
        run_sam3(args, clips, decoded_root)
    if args.tracker == 'osnet':
        device = torch.device(args.device)
        if args.common_matched_idsw:
            from ultralytics import YOLO

            mm.lap.default_solver = 'scipy'
            torch.manual_seed(20260914)
            torch.backends.cudnn.benchmark = False
            yolo = YOLO(str(ROOT / 'models/segmentation/piglet.pt'))
            class_ids = [i for i, label in yolo.names.items() if label == 'piglet']
            if not class_ids:
                raise ValueError('YOLO checkpoint does not predict piglet masks')
        model, _ = load_association_checkpoint(args.checkpoint, device)
        settings = json.loads(args.config.read_text(encoding='utf-8'))
        config = TrackerConfig(**{key: value for key, value in settings.items() if key != 'sample_fps'})
        if args.common_matched_idsw:
            for _ in range(5):
                yolo_masks(yolo, clips[test_dirs[0].name][0][0], class_ids, device)

    hota = HOTA()
    accumulators, hota_results = {}, {}
    common_results = {}
    for name, (frame_paths, anchor_frame, ground_truth) in clips.items():
        if args.mask_source == 'manual':
            masks = {fi: [(raster, 1.) for _, raster in gt] for fi, gt in ground_truth.items()}
        else:
            path = decoded_root / 'sam3_detections' / f'{name}.json'
            masks = dict(enumerate(json.loads(path.read_text(encoding='utf-8'))))
        if args.tracker == 'osnet':
            predictions = track_osnet(frame_paths, masks, model, device, config)
        else:
            path = decoded_root / 'sam3_tracks' / f'{name}.json'
            tracks = json.loads(path.read_text(encoding='utf-8'))
            predictions = {fi: project_tracks(tracks[str(fi)], [mask for mask, _ in entries])
                           for fi, entries in masks.items()}
        accumulators[name], hota_results[name] = score_clip(ground_truth, predictions, anchor_frame, hota)
        if args.common_matched_idsw:
            yolo_input = {i: yolo_masks(yolo, path, class_ids, device) for i, path in enumerate(frame_paths)}
            yolo_predictions = track_osnet(frame_paths, yolo_input, model, device, config)
            yolo_acc, _ = score_clip(ground_truth, yolo_predictions, anchor_frame)
            common_results[name] = common_matched_changes(accumulators[name], yolo_acc)
        print(name, flush=True)

    table = mm.metrics.create().compute_many(list(accumulators.values()), names=list(accumulators),
                                             metrics=MOT_METRICS, generate_overall=True)
    hota_results['OVERALL'] = hota.combine_sequences(hota_results)
    for key in HOTA_METRICS:
        table[key] = [float(hota_results[name][key].mean()) for name in table.index]
    if args.common_matched_idsw:
        common_results['OVERALL'] = tuple(map(sum, zip(*common_results.values())))
        for i, key in enumerate(COMMON_MATCH_METRICS):
            table[key] = [common_results[name][i] for name in table.index]
    args.output.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output / 'metrics.csv')
    (args.output / 'metrics.json').write_text(table.to_json(orient='index', indent=2), encoding='utf-8')
    print(table.loc[['OVERALL']].to_string())


if __name__ == '__main__':
    main()
