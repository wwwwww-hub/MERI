import torch
import torch.nn as nn
from typing import Dict, Tuple, Union, List, Optional
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

def _as_arg(levels: List[Tensor]):
    # SingleRoIExtractor3D 接口：Tensor 或 Tuple[Tensor]
    if len(levels) == 1:
        return levels[0]
    return tuple(levels)

@MODELS.register_module()
class ConcatTwoStreamRoIExtractor3D(nn.Module):
    """Run SingleRoIExtractor3D on RGB & Flow separately, then concat along C."""
    def __init__(self, single_extractor: dict, with_global: Optional[bool] = None, **kwargs):
        super().__init__()
        self.rgb_extractor  = MODELS.build(single_extractor)
        self.flow_extractor = MODELS.build(single_extractor)
        self.with_global = with_global if with_global is not None \
            else bool(getattr(self.rgb_extractor, 'with_global', False))
        self._logged = False

    def forward(self,
                feats: Union[Dict[str, FeatLike], Tuple[FeatLike, FeatLike], List[FeatLike]],
                proposals,
                **kwargs):
        if isinstance(feats, dict):
            rgb_feat, flow_feat = feats['rgb'], feats['flow']
        elif isinstance(feats, (tuple, list)) and len(feats) == 2:
            rgb_feat, flow_feat = feats
        else:
            raise TypeError('Expect dict{rgb,flow} or tuple/list(rgb, flow)')

        rgb_levels  = _flatten_to_levels(rgb_feat)
        flow_levels = _flatten_to_levels(flow_feat)
        assert len(rgb_levels) > 0 and len(flow_levels) > 0, 'Empty feature list after flatten.'
        for t in rgb_levels + flow_levels:
            assert isinstance(t, torch.Tensor) and t.dim() == 5, \
                f'Expect 5D tensor, got {type(t)} {getattr(t, "shape", None)}'

        out_rgb  = self.rgb_extractor(_as_arg(rgb_levels),  proposals, **kwargs)
        out_flow = self.flow_extractor(_as_arg(flow_levels), proposals, **kwargs)

        if self.with_global:
            roi_rgb,  glob_rgb  = out_rgb
            roi_flow, glob_flow = out_flow

            if not self._logged:
                print('[ConcatTwoStreamRoIExtractor3D] roi_rgb C=', roi_rgb.size(1),
                      ' roi_flow C=', roi_flow.size(1),
                      ' glob_rgb C=', glob_rgb.size(1),
                      ' glob_flow C=', glob_flow.size(1))
                self._logged = True

            # ☆ 这里应该分别是 4608 与 4096（参考官方日志）
            # 如果断言失败，你就知道是哪一路没进来
            assert roi_rgb.size(1)  in (2304, 4608, 2048), 'RGB roi channels unexpected'
            assert roi_flow.size(1) in (2048, 4096),       'Flow roi channels unexpected'

            return torch.cat([roi_rgb,  roi_flow],  dim=1), \
                   torch.cat([glob_rgb, glob_flow], dim=1)
        else:
            roi_rgb  = out_rgb[0]  if isinstance(out_rgb,  (tuple, list)) else out_rgb
            roi_flow = out_flow[0] if isinstance(out_flow, (tuple, list)) else out_flow

            if not self._logged:
                print('[ConcatTwoStreamRoIExtractor3D] (no-global) roi_rgb C=', roi_rgb.size(1),
                      ' roi_flow C=', roi_flow.size(1))
                self._logged = True

            return torch.cat([roi_rgb, roi_flow], dim=1)
