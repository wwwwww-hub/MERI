from typing import Any, Dict, Tuple, Union, List, Optional

import torch
import torch.nn as nn
from mmengine.registry import MODELS
from mmaction.models.backbones import ResNet3dSlowFast, ResNet3dSlowOnly

# 可选：ViT 增强版 SlowFast（RGB）
try:
    from mmaction.models.backbones.resnet3d_slowfast_vit import ResNet3dSlowFastViT
except Exception:
    ResNet3dSlowFastViT = None

# 可选：ViT 增强版 SlowOnly（Flow）
try:
    from mmaction.models.backbones.resnet3d_slowonly_vit import ResNet3dSlowOnlyViT
except Exception:
    ResNet3dSlowOnlyViT = None

Tensor = torch.Tensor
RGBFeat = Union[Tensor, Tuple[Tensor, Tensor], List[Tensor]]
FlowFeat = Union[Tensor, List[Tensor], Tuple[Tensor, ...]]


@MODELS.register_module()
class TwoStreamSlowFast(nn.Module):
    """双流 SlowFast + SlowOnly，用于 RGB+Flow 检测.

    Args:
        slowfast_cfg (dict): RGB SlowFast 主干的构造参数。
        flow_cfg (dict): Flow SlowOnly 主干的构造参数。
        rgb_type (str): 'ResNet3dSlowFast' 或 'ResNet3dSlowFastViT'。
        flow_type (str): 'ResNet3dSlowOnly' 或 'ResNet3dSlowOnlyViT'。
    """

    def __init__(
        self,
        slowfast_cfg: Dict[str, Any],
        flow_cfg: Dict[str, Any],
        rgb_type: str = "ResNet3dSlowFast",
        flow_type: str = "ResNet3dSlowOnly",
    ) -> None:
        super().__init__()

        self.rgb_type = rgb_type
        self.flow_type = flow_type

        # ---- 构建 RGB 主干 ----
        self.rgb_net = self._build_rgb_net(slowfast_cfg)

        # ---- 构建 Flow 主干 ----
        self.flow_net = self._build_flow_net(flow_cfg)

        # 给 AttentionAlignHook 用
        self.feature_map_for_distill: Optional[Tensor] = None

    # ------------------------------------------------------------------ #
    #  构建子网络
    # ------------------------------------------------------------------ #
    def _build_rgb_net(self, cfg: Dict[str, Any]) -> nn.Module:
        cfg = cfg.copy()
        if self.rgb_type == "ResNet3dSlowFastViT":
            if ResNet3dSlowFastViT is None:
                raise RuntimeError(
                    'ResNet3dSlowFastViT not available, '
                    'but rgb_type="ResNet3dSlowFastViT" requested.'
                )
            return ResNet3dSlowFastViT(**cfg)
        else:
            return ResNet3dSlowFast(**cfg)

    def _build_flow_net(self, cfg: Dict[str, Any]) -> nn.Module:
        cfg = cfg.copy()
        if self.flow_type == "ResNet3dSlowOnlyViT":
            if ResNet3dSlowOnlyViT is None:
                raise RuntimeError(
                    'ResNet3dSlowOnlyViT not available, '
                    'but flow_type="ResNet3dSlowOnlyViT" requested.'
                )
            return ResNet3dSlowOnlyViT(**cfg)
        else:
            return ResNet3dSlowOnly(**cfg)

    # ------------------------------------------------------------------ #
    #  工具函数：从各种结构中取“最后一层”Flow 特征
    # ------------------------------------------------------------------ #
    @staticmethod
    def _select_last(feat: Optional[FlowFeat]) -> Optional[Tensor]:
        if feat is None:
            return None
        if isinstance(feat, torch.Tensor):
            return feat
        if isinstance(feat, (list, tuple)):
            return feat[-1]
        if isinstance(feat, dict):
            last_val = None
            for v in feat.values():
                last_val = v
            if last_val is None:
                return None
            return TwoStreamSlowFast._select_last(last_val)
        raise TypeError(f"Unsupported flow feature type: {type(feat)}")

    # ------------------------------------------------------------------ #
    #  输入拆分：支持 TwoStreamDataPreprocessor 返回的 (rgb, flow)
    #           以及 dict / 单路 RGB 等情况
    # ------------------------------------------------------------------ #
    def _split_inputs(
        self, x: Union[Tensor, Tuple, List, Dict[str, Tensor]]
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """解析 batch_inputs，返回 (x_rgb, x_flow)."""

        # 1) TwoStreamDataPreprocessor 的输出： (rgb, flow)
        #    rgb, flow 均为 (N,C,T,H,W) tensor
        if isinstance(x, (list, tuple)):
            if len(x) == 2 and torch.is_tensor(x[0]):
                return x[0], x[1]
            if len(x) == 1 and torch.is_tensor(x[0]):
                return x[0], None
            raise ValueError(
                f"Expect list/tuple of 1 or 2 tensors as inputs, got len={len(x)}"
            )

        # 2) dict 形式（防止后面你自己扩展）
        if isinstance(x, dict):
            rgb = (
                x.get("imgs_rgb", None)
                or x.get("rgb", None)
                or x.get("inputs_rgb", None)
                or x.get("imgs", None)
            )
            flow = (
                x.get("imgs_flow", None)
                or x.get("flow", None)
                or x.get("inputs_flow", None)
            )
            if rgb is None:
                raise ValueError(f"Cannot find RGB input in dict keys: {list(x.keys())}")
            return rgb, flow

        # 3) 单路 RGB：兼容旧逻辑
        if torch.is_tensor(x):
            return x, None

        raise TypeError(
            f"Unexpected input type for TwoStreamSlowFast: {type(x)}. "
            "Expected Tensor, (rgb,flow) or dict."
        )

    # ------------------------------------------------------------------ #
    #  前向：给 mmdet 的 backbone 用
    # ------------------------------------------------------------------ #
    def forward(
        self, x: Union[Tensor, Tuple, List, Dict[str, Tensor]], *args, **kwargs
    ) -> Dict[str, Any]:
        # 1) 拆输入
        x_rgb, x_flow = self._split_inputs(x)
        assert x_rgb is not None, "TwoStreamSlowFast expects at least RGB input."

        # 2) 主干前向
        #    RGB SlowFastViT: 输入是单个 RGB clip (N,3,T,H,W)
        feat_rgb: RGBFeat = self.rgb_net(x_rgb)  # (x_slow, x_fast) or tensor

        #    Flow SlowOnlyViT: 输入是 (N,2,T,H,W)，输出可能是 (x,) 或 tensor
        feat_flow: Optional[FlowFeat] = (
            self.flow_net(x_flow) if x_flow is not None else None
        )

        # 3) 暴露蒸馏用特征（RGB slow 路）
        slow_feat = getattr(self.rgb_net, "feature_map_for_distill_slow", None)
        fast_feat = getattr(self.rgb_net, "feature_map_for_distill_fast", None)

        # 方案一：打包成 dict 交给 Hook
        self.feature_map_for_distill = dict(
            slow=slow_feat,
            fast=fast_feat,
        )

        # 其余保持不变
        out = {
            "rgb": feat_rgb,
            "flow": self._select_last(feat_flow) if feat_flow is not None else None,
        }
        return out

        # 4) 整理输出给 neck（GlobalPoolConcatNeck 期待 dict{'rgb','flow'}）:contentReference[oaicite:2]{index=2}
        out: Dict[str, Any] = dict(
            rgb=feat_rgb,
            flow=self._select_last(feat_flow) if feat_flow is not None else None,
        )
        return out
