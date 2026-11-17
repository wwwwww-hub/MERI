from typing import Dict, Tuple, Union, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.registry import MODELS

Tensor = torch.Tensor
RGBFeat = Union[Tuple[Tensor, Tensor], List[Tensor], Tensor]  # (slow, fast) | list | tensor


@MODELS.register_module()
class GlobalPoolConcatNeck(nn.Module):
    """
    仅在时间维做平均池化，保留 H/W 给 RoIAlign；
    在通道拼接前，按 RGB 的 H/W 将 Flow 特征图做空间对齐（5D trilinear）。

    输入:
      feats: dict{'rgb': (x_slow, x_fast), 'flow': x_flow_last} 或 (rgb, flow)
        - x_* 形状均为 (N, C, T, H, W)
    输出:
      (out,)  # 单尺度特征，形状 (N, C_total, 1, H_rgb, W_rgb)
    """

    def __init__(self):
        super().__init__()
        # 仅时间维做均值池化：等价 x.mean(dim=2, keepdim=True)
        self._temporal_reduce = lambda x: x.mean(dim=2, keepdim=True)

    @staticmethod
    def _ensure_5d(x: Tensor) -> Tensor:
        if x.dim() == 5:
            return x
        if x.dim() == 4:
            return x.unsqueeze(2)  # 补 T 维
        raise ValueError(f"Expect 5D/4D tensor, got {x.dim()}D")

    def _pool_T(self, x: Tensor) -> Tensor:
        x = self._ensure_5d(x)
        return self._temporal_reduce(x)  # (N,C,1,H,W)

    def _pool_rgb(self, rgb_feat: RGBFeat) -> Tensor:
        """处理 RGB 的 (slow, fast) 或单张特征。"""
        if isinstance(rgb_feat, (list, tuple)):
            if len(rgb_feat) == 2:
                f_slow = self._pool_T(rgb_feat[0])    # (N,Cs,1,H,W)
                f_fast = self._pool_T(rgb_feat[1])    # (N,Cf,1,H,W)
                return torch.cat([f_slow, f_fast], dim=1)  # (N,Cs+Cf,1,H,W)
            else:
                return self._pool_T(rgb_feat[-1])
        else:
            return self._pool_T(rgb_feat)

    def _pool_flow(self, flow_feat: Optional[Union[List[Tensor], Tensor]]) -> Optional[Tensor]:
        if flow_feat is None:
            return None
        if isinstance(flow_feat, (list, tuple)):
            flow_feat = flow_feat[-1]
        return self._pool_T(flow_feat)

    @staticmethod
    def _spatial_align_to(x: Tensor, ref: Tensor) -> Tensor:
        """
        将 x 的 H/W 对齐到 ref 的 H/W（保持 N,C,T 不变；T 此时应为 1）。
        使用 5D 三线性插值：mode='trilinear', align_corners=False
        """
        assert x.dim() == 5 and ref.dim() == 5
        H_ref, W_ref = ref.shape[-2], ref.shape[-1]
        H_x, W_x = x.shape[-2], x.shape[-1]
        if (H_x, W_x) == (H_ref, W_ref):
            return x
        return F.interpolate(x, size=(x.shape[2], H_ref, W_ref),
                             mode='trilinear', align_corners=False)

    def forward(self, feats: Union[Dict[str, RGBFeat], Tuple[RGBFeat, Optional[Tensor]], List[RGBFeat]]):
        # 解析输入
        if isinstance(feats, dict):
            rgb_feat, flow_feat = feats.get('rgb'), feats.get('flow')
        elif isinstance(feats, (tuple, list)) and len(feats) == 2:
            rgb_feat, flow_feat = feats
        else:
            raise TypeError("GlobalPoolConcatNeck expects dict{'rgb','flow'} or tuple/list (rgb, flow).")

        # 1) 时间池化
        f_rgb = self._pool_rgb(rgb_feat)              # (N,Crgb,1,Hr,Wr)
        f_flow = self._pool_flow(flow_feat)           # (N,Cflow,1,Hf,Wf) or None

        # 2) 空间对齐到 RGB 的 H/W（确保可拼接）
        if f_flow is not None:
            f_flow = self._spatial_align_to(f_flow, f_rgb)

        # 3) 通道拼接
        out = torch.cat([f_rgb, f_flow], dim=1) if f_flow is not None else f_rgb  # (N,Crgb+Cflow,1,Hr,Wr)

        # Debug（需要时可打开）
        # print('[NECK] f_rgb', tuple(f_rgb.shape), 'f_flow', None if f_flow is None else tuple(f_flow.shape),
        #       'out', tuple(out.shape))

        return (out,)
