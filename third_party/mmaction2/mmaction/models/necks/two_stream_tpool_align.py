import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Union, List
from mmengine.registry import MODELS

Tensor = torch.Tensor
FeatLike = Union[Tensor, List, Tuple, Dict]

def _flatten_to_levels(x: FeatLike) -> List[Tensor]:
    out: List[Tensor] = []
    if isinstance(x, torch.Tensor):
        out.append(x)
    elif isinstance(x, (list, tuple)):
        for xi in x:
            out.extend(_flatten_to_levels(xi))
    elif isinstance(x, dict):
        for v in x.values():
            out.extend(_flatten_to_levels(v))
    else:
        raise TypeError(f'Unsupported feature container: {type(x)}')
    return out

@MODELS.register_module()
class TwoStreamTPoolAlign(nn.Module):
    """Temporal-only pool + spatial align for (RGB, Flow) **before** RoI, keep multi-levels."""
    def __init__(self, t_pool: str = 'avg', align_to: str = 'rgb'):
        super().__init__()
        assert t_pool in ('avg', 'max')
        assert align_to in ('rgb', 'flow')
        self.t_pool = t_pool
        self.align_to = align_to
        self._logged = False  # 只在第一次 forward 打印一次形状

    def _tpool(self, x: Tensor) -> Tensor:
        assert x.dim() == 5, f'Expect 5D (N,C,T,H,W), got {x.shape}'
        if x.size(2) == 1:
            return x
        return x.mean(dim=2, keepdim=True) if self.t_pool == 'avg' else x.amax(dim=2, keepdim=True)

    def forward(self, feats: Union[Dict[str, FeatLike], Tuple[FeatLike, FeatLike], List[FeatLike]]):
        if isinstance(feats, dict):
            rgb_raw, flow_raw = feats['rgb'], feats['flow']
        elif isinstance(feats, (tuple, list)) and len(feats) == 2:
            rgb_raw, flow_raw = feats
        else:
            raise TypeError('TwoStreamTPoolAlign expects dict{rgb,flow} or tuple/list(rgb, flow)')

        rgb_levels  = _flatten_to_levels(rgb_raw)
        flow_levels = _flatten_to_levels(flow_raw)
        assert len(rgb_levels) > 0 and len(flow_levels) > 0, 'Empty stream features.'

        # 参考对齐尺寸（取 RGB 的最后一层）
        ref = rgb_levels[-1] if self.align_to == 'rgb' else flow_levels[-1]
        T, H, W = ref.shape[2:]

        def _proc(levels: List[Tensor]) -> Tuple[Tensor, ...]:
            outs = []
            for x in levels:
                x = self._tpool(x)                                # (N,C,1,h,w)
                x = F.interpolate(x, size=(T, H, W),
                                  mode='trilinear', align_corners=False)
                outs.append(x)
            return tuple(outs)  # ☆ SingleRoIExtractor3D 偏好 tuple

        rgb_out  = _proc(rgb_levels)
        flow_out = _proc(flow_levels)

        if not self._logged:
            print('[TwoStreamTPoolAlign] rgb levels:', [t.shape for t in rgb_out])
            print('[TwoStreamTPoolAlign] flow levels:', [t.shape for t in flow_out])
            self._logged = True

        return {'rgb': rgb_out, 'flow': flow_out}
