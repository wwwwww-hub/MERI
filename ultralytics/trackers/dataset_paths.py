"""Resolve numbered dataset samples without relying on acquisition filenames."""
from pathlib import Path
import json,re
import shutil
import subprocess

def frame_index(path):
    return int(re.search(r'(\d+)$',Path(path).stem).group(1))

def indexed_file(directory,index,width,suffix,prefix):
    directory=Path(directory)
    named=directory/f'{prefix}_{index:0{width}d}{suffix}'
    return named if named.is_file() else directory/f'{index:0{width}d}{suffix}'

def frame_file(root,index,suffix='.jpg'):
    root=Path(root);return indexed_file(root/'frames',index,6,suffix,root.name)

def mask_file(root,kind,index):
    root=Path(root);return indexed_file(root/kind,index,6,'.png',root.name)

def instance_file(root,frame,identity):
    root=Path(root);folder=root/'instance_masks'/f'{root.name}_{frame:06d}'
    if not folder.is_dir():folder=root/'instance_masks'/f'{frame:06d}'
    return indexed_file(folder,identity,5,'.png',root.name)


def clip_video(root):
    root = Path(root)
    return root / f'{root.name}.mp4'


def extract_video_frames(root, frame_dir, ffmpeg='ffmpeg'):
    """Regenerate the JPEG inputs that the published masks were aligned to."""
    root, frame_dir = Path(root), Path(frame_dir)
    video = clip_video(root)
    if not video.is_file():
        raise FileNotFoundError(video)
    meta = json.loads((root / 'clip_metadata.json').read_text(encoding='utf-8'))
    count = int(meta['frame_count'])
    fps = float(meta['track_fps'])
    start = float(meta['sample_start_clip_sec'])
    duration = float(meta['sample_duration_sec'])
    quality = int(meta['jpeg_quality'])
    if count <= 0 or fps <= 0 or start < 0 or duration <= 0 or not 1 <= quality <= 31:
        raise ValueError(f'Invalid video sampling metadata in {root.name}')
    exe = shutil.which(str(ffmpeg))
    if exe is None:
        raise FileNotFoundError(f'FFmpeg is required to read DS3 videos: {ffmpeg}')
    frame_dir.mkdir(parents=True, exist_ok=False)
    subprocess.run([exe, '-hide_banner', '-loglevel', 'error', '-threads', '2',
                    '-ss', f'{start:.6f}', '-i', str(video), '-t', f'{duration:.6f}',
                    '-vf', f'fps={fps:.12g}', '-q:v', str(quality), '-threads', '1',
                    '-start_number', '0', str(frame_dir / '%06d.jpg')], check=True)
    if sorted(p.name for p in frame_dir.glob('*.jpg')) != [f'{i:06d}.jpg' for i in range(count)]:
        raise ValueError(f'FFmpeg did not extract the expected {count} frames from {video.name}')
    return meta


def manual_mask_file(root, index):
    root = Path(root)
    return indexed_file(root / 'manual_masks', index, 6, '.json', root.name)


def tracking_protocol(root):
    root=Path(root)
    base=root/'data/DS3_tracking'
    clips=[]
    for split,meta_name in [('train','annotation_info.json'),('test','clip_metadata.json')]:
        for d in sorted(p for p in (base/split).iterdir() if p.is_dir()):
            if clip_video(d).is_file():
                meta_name_for_clip = 'clip_metadata.json'
            else:
                meta_name_for_clip = meta_name
            meta=json.loads((d/meta_name_for_clip).read_text(encoding='utf-8'))
            clips.append({'clip_id':d.name,'split':split,'frames':int(meta['frame_count']),'root':str(d),'metadata':meta})
    return {'counts':{'train':sum(c['split']=='train' for c in clips),'test':sum(c['split']=='test' for c in clips)},'clips':clips}
