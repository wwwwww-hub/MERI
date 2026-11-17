# ------------------------------------------------------------
# ResNet3dSlowFastViT  (updated for attention distillation)
# ------------------------------------------------------------
# 关键更新：
#   • 在 forward() 中暴露 self.feature_map_for_distill = slow_feat
#     供 AttentionAlignHook 访问。
#   • 其它逻辑保持不变，确保与原 MHSA 版本兼容。
# ------------------------------------------------------------

import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmengine.registry import MODELS
from mmaction.models.backbones.resnet3d import ResNet3d


class ConservativeTransformerInsert(nn.Module):
    """极保守的 Transformer 通道注意力"""

    def __init__(self, channels: int):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.channel_att = nn.Sequential(
            nn.Linear(channels, channels // 16),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 16, channels),
            nn.Sigmoid(),
        )
        self.alpha = nn.Parameter(torch.tensor(0.01))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        nn.init.constant_(self.alpha, 0.01)

    def forward(self, x):
        identity = x
        b, c = x.shape[:2]
        att = self.global_pool(x).view(b, c)
        att = self.channel_att(att).view(b, c, 1, 1, 1)
        out = identity * (1 - self.alpha) + x * att * self.alpha
        return out


class StageTrainingTransformer(nn.Module):
    """分阶段通道+时间注意力"""

    def __init__(self, channels: int):
        super().__init__()
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // 8, channels, 1),
            nn.Sigmoid(),
        )
        self.temporal_att = nn.Sequential(
            nn.Conv3d(channels, channels, (3, 1, 1), padding=(1, 0, 0), groups=channels),
            nn.Conv3d(channels, channels, 1),
            nn.Sigmoid(),
        )
        self.enable_temporal = nn.Parameter(torch.tensor(0.0))
        self.alpha = nn.Parameter(torch.tensor(0.05))
        self.training_stage = 1  # default stage‑1

    def set_training_stage(self, stage: int):
        self.training_stage = stage
        if stage >= 2:
            self.enable_temporal.data.fill_(1.0)

    def forward(self, x):
        out = x * self.channel_att(x)
        if self.training_stage >= 2:
            out = out * self.temporal_att(out) * self.enable_temporal
        return x + self.alpha * (out - x)


@MODELS.register_module()
class ResNet3dSlowFastViT(BaseModule):
    """SlowFast backbone with lightweight ViT inserts + distillation hook support"""

    def __init__(
        self,
        slow_pathway: dict,
        fast_pathway: dict,
        resample_rate: int = 8,
        transformer_layers: list = (4,5),
        transformer_mode: str = "conservative",  # 'conservative' | 'staged'
        **kwargs,
    ):
        kwargs.pop("pretrained", None)
        super().__init__(**kwargs)
        self.alpha = resample_rate
        tcls = ConservativeTransformerInsert if transformer_mode == "conservative" else StageTrainingTransformer

        self.slow_path = self._make_pathway(slow_pathway, transformer_layers, tcls)
        self.fast_path = self._make_pathway(fast_pathway, transformer_layers, tcls)

        self.out_channels = self._path_out_dim(self.slow_path) + self._path_out_dim(self.fast_path)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _make_pathway(self, cfg: dict, transformer_layers, tcls):
        cfg = cfg.copy(); cfg.pop("type", None)
        path = ResNet3d(**cfg)
        if 4 in transformer_layers:
            ch = path.layer4[-1].conv3.out_channels if hasattr(path.layer4[-1], "conv3") else path.layer4[-1].conv2.out_channels
            path.layer4 = nn.Sequential(path.layer4, tcls(ch))
        return path

    def _path_out_dim(self, path):
        layer4 = path.layer4[0][-1] if isinstance(path.layer4, nn.Sequential) else path.layer4[-1]
        return layer4.conv3.out_channels if hasattr(layer4, "conv3") else layer4.conv2.out_channels

    def set_training_stage(self, stage: int):
        for p in [self.slow_path, self.fast_path]:
            if isinstance(p.layer4, nn.Sequential) and len(p.layer4) > 1 and hasattr(p.layer4[1], "set_training_stage"):
                p.layer4[1].set_training_stage(stage)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, x):
        # handle input list / tensor as in standard SlowFast
        if isinstance(x, torch.Tensor):
            if x.dim() == 6 and x.shape[1] == 1:
                x = x.squeeze(1)
            if x.dim() != 5:
                raise ValueError("SlowFast expects 5D input (N,C,T,H,W)")
            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1, 1)
            fast = x
            slow = x[:, :, :: self.alpha, :, :]
            x = [slow, fast]
        else:
            x = [t if t.dim() == 5 else t.squeeze(1) for t in x]
            x = [t.repeat(1, 3, 1, 1, 1) if t.shape[1] == 1 else t for t in x]

        # pathway forward
        slow_feat = self.slow_path(x[0])  # (B, Cs, Ts, Hs, Ws)
        fast_feat = self.fast_path(x[1])  # (B, Cf, Tf, Hf, Wf)

        # --- expose feature map for distillation ------------------------
        self.feature_map_for_distill = slow_feat  # 供 AttentionAlignHook 使用
        # ----------------------------------------------------------------

        # align temporal dim then concat
        if slow_feat.shape[2] != fast_feat.shape[2]:
            fast_feat = nn.functional.interpolate(
                fast_feat,
                size=(slow_feat.shape[2], slow_feat.shape[3], slow_feat.shape[4]),
                mode="trilinear",
                align_corners=False,
            )
        return torch.cat([slow_feat, fast_feat], dim=1)
