from __future__ import annotations
from pathlib import Path
import pickle
from mmengine.registry import TRANSFORMS

@TRANSFORMS.register_module()
class LoadTeacherAttn:
    """给每段视频加载 teacher attention (.pkl)。
    缺失文件时返回 None，训练时自动忽略。
    """
    def __init__(self, attn_root: str, key: str = 'attn', strict: bool = False):
        self.attn_root = Path(attn_root)
        self.key = key
        self.strict = strict

    def __call__(self, results):
        vid = results['video_name']
        pkl_path = self.attn_root / f'{vid}.pkl'
        if pkl_path.exists():
            with open(pkl_path, 'rb') as f:
                results[self.key] = pickle.load(f)['attention_weights']  # (N,N)
        elif self.strict:
            raise FileNotFoundError(f'{pkl_path} not found')
        else:
            results[self.key] = None          # 用 None 占位
        return results
