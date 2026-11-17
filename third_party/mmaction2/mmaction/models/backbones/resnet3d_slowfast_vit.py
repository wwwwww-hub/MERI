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
from .resnet3d_slowfast import ResNet3dPathway


def _insert_after_block_with_mhsa(path: nn.Module, layer_name: str, tcls):
    """将 path.layer{layer_idx} 包装为 nn.Sequential(layer, MHSA)"""
    blk = getattr(path, layer_name)
    # 取该 stage 的最后一个 bottleneck，拿到通道数
    last = blk[-1]
    ch = last.conv3.out_channels if hasattr(last, "conv3") else last.conv2.out_channels
    setattr(path, layer_name, nn.Sequential(blk, tcls(ch)))

def _stage_out_channels(stage: nn.Module) -> int:
    # 支持 layerX 已经被 nn.Sequential(layerX, MHSA) 包过的情况
    core = stage[0] if (isinstance(stage, nn.Sequential) and isinstance(stage[0], nn.Sequential)) else stage
    last = core[-1]
    ch = last.conv3.out_channels if hasattr(last, 'conv3') else last.conv2.out_channels
    return int(ch)

def _unwrap_to_block_sequential(stage: nn.Module) -> nn.Module:
    """剥开多层 Sequential 封装，拿到真正存放 bottleneck block 的那层 Sequential。"""
    m = stage
    while isinstance(m, nn.Sequential):
        children = list(m.children())
        if not children:
            break
        # 优先往第一个是 Sequential 的孩子里面钻（典型形态：Sequential(layerX, MHSA, MHSA2, ...))
        if isinstance(children[0], nn.Sequential):
            m = children[0]
            continue
        # 否则在孩子里找一个“看起来像 block 序列”的 Sequential
        found = None
        for ch in children:
            if isinstance(ch, nn.Sequential) and len(list(ch.children())) > 0:
                # 这个 ch 可能就是 block 的序列
                found = ch
                break
        if found is not None:
            m = found
            continue
        break
    return m

def _stage_out_channels(stage: nn.Module) -> int:
    """拿到某个 stage（可能被多层 Sequential 包裹）的输出通道数。"""
    core = _unwrap_to_block_sequential(stage)
    last = core[-1] if isinstance(core, nn.Sequential) else core
    if hasattr(last, 'conv3') and isinstance(last.conv3, nn.Conv3d):
        return int(last.conv3.out_channels)
    if hasattr(last, 'conv2') and isinstance(last.conv2, nn.Conv3d):
        return int(last.conv2.out_channels)
    # 兜底：从最后往前找一个 Conv3d
    for sub in reversed(list(last.modules())):
        if isinstance(sub, nn.Conv3d):
            return int(sub.out_channels)
    raise RuntimeError('Cannot infer out_channels for this stage (no conv2/conv3 found).')


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

class TemporalShiftAttention3D(nn.Module):
    """
    TSA: 部分通道做时间平移 (TSM-style) + 轻量时序注意力
    输入 / 输出: (B, C, T, H, W)
    只用于 Fast 分支，强调快速细粒度运动。
    """

    def __init__(self, channels: int, shift_ratio: float = 1 / 8):
        super().__init__()
        self.channels = channels
        self.shift_ratio = float(shift_ratio)

        # 需要做 shift 的通道数，前后各一半
        n_shift = max(1, int(channels * self.shift_ratio))
        n_shift = max(2, n_shift)
        if n_shift % 2 != 0:
            n_shift -= 1
        self.n_shift = n_shift

        # 轻量时序注意力（深度可分离 3D Conv，只在 T 上卷积）
        self.temporal_att = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(3, 1, 1),
                      padding=(1, 0, 0), groups=channels),
            nn.Conv3d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

        # 残差缩放，初始很小，避免一开始破坏预训练特征
        self.alpha = nn.Parameter(torch.tensor(0.05))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape

        # T 非常短或通道太少时，退化为普通 temporal attention
        if T <= 1 or C < 2:
            att = self.temporal_att(x)
            y = x * att
            return x + self.alpha * (y - x)

        n_shift = min(self.n_shift, C)
        fold = n_shift // 2
        if fold == 0:
            att = self.temporal_att(x)
            y = x * att
            return x + self.alpha * (y - x)

        # ---- TSM-style temporal shift ----
        x_shift = x.clone()

        # 前移：前 fold 个通道向“未来”移动一帧
        x_shift[:, :fold, 1:, :, :] = x[:, :fold, :-1, :, :]
        x_shift[:, :fold, 0, :, :] = 0

        # 后移：中间 fold 个通道向“过去”移动一帧
        x_shift[:, fold:2*fold, :-1, :, :] = x[:, fold:2*fold, 1:, :, :]
        x_shift[:, fold:2*fold, -1, :, :] = 0

        # 其余通道保持不变（clone 已经拷贝了原值）
        # x_shift[:, 2*fold:, ...] = x[:, 2*fold:, ...]  # 实际上已经相同，可不写

        # ---- temporal attention on shifted features ----
        att = self.temporal_att(x_shift)   # (B,C,T,H,W)
        y = x_shift * att

        return x + self.alpha * (y - x)

# === add: Global MHSA for 3D feature maps (drop-in) ===
import math

class GlobalSelfAttention3D(nn.Module):
    """
    3D Global Multi-Head Self-Attention (MHSA).
      * 输入/输出: (B, C, T, H, W)
      * 实现: 展平 T×H×W 为 N 个 token，对 token 维做多头注意力
      * QKV/Proj 用 1x1 Conv1d（等价于 per-token Linear）
      * 残差 + 可学习缩放 gamma
    自定义 FLOPs（乘加=2 FLOPs）:
      QKV:     2 * N * C * (3C)
      QK^T:    2 * N^2 * C
      Attn·V:  2 * N^2 * C
      Proj:    2 * N * C * C
    """
    def __init__(self, channels: int,
                 num_heads: int = 8,
                 qkv_bias: bool = True,
                 attn_drop: float = 0.0,
                 proj_drop: float = 0.0):
        super().__init__()
        assert channels % num_heads == 0, 'channels must be divisible by num_heads'
        self.dim = int(channels)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads

        self.qkv = nn.Conv1d(self.dim, self.dim * 3, kernel_size=1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv1d(self.dim, self.dim, kernel_size=1, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)

        # 残差缩放参数
        self.gamma = nn.Parameter(torch.tensor(1e-2))
        # 给外部 FLOPs 汇总器的标记
        self._is_custom_flops = True

        # init
        nn.init.xavier_uniform_(self.qkv.weight)
        if self.qkv.bias is not None:
            nn.init.zeros_(self.qkv.bias)
        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        N = T * H * W
        x_flat = x.view(B, C, N)                         # (B, C, N)

        qkv = self.qkv(x_flat)                            # (B, 3C, N)
        q, k, v = qkv.chunk(3, dim=1)                     # (B, C, N) ×3

        def _split_heads(t):
            t = t.view(B, self.num_heads, self.head_dim, N)  # (B, h, d, N)
            return t.transpose(-2, -1).contiguous()          # (B, h, N, d)

        q, k, v = map(_split_heads, (q, k, v))
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B,h,N,N)
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)                      # (B,h,N,d)
        out = out.transpose(-2, -1).contiguous().view(B, C, N)  # (B,C,N)
        out = self.proj(out)
        out = self.proj_drop(out)
        out = out.view(B, C, T, H, W)

        return x + self.gamma * out

    # 供外部 FLOPs 汇总器调用
    def flops(self, T: int, H: int, W: int) -> int:
        N, C = T * H * W, self.dim
        qkv   = 2 * N * C * (3 * C)
        qkt   = 2 * N * N * C
        av    = 2 * N * N * C
        proj  = 2 * N * C * C
        return int(qkv + qkt + av + proj)

@MODELS.register_module()
class ResNet3dSlowFastViT(BaseModule):
    """SlowFast backbone with lightweight ViT inserts + distillation hook support + LATERAL connections"""

    def __init__(
        self,
        slow_pathway: dict,
        fast_pathway: dict,
        resample_rate: int = 8,
        transformer_layers: list = (4,5),
        transformer_mode: str = "conservative",
        speed_ratio: int = 8,                 # 对应官方 alpha（Fast:Slow 时间尺度比）
        channel_ratio: int = 8,               # 对应官方 beta（Fast 通道缩放）
        fusion_kernel: int = 5,
        lateral_infl: int = 2,
        lateral_norm: bool = False,
        lateral_inv: bool = False,
        lateral_activate: tuple = (1, 1, 1, 1),
        **kwargs,
    ):
        kwargs.pop("pretrained", None)
        super().__init__(**kwargs)
        self.alpha = resample_rate
        # 若用户没改 speed_ratio，就让它与 α 一致（更稳妥）
        if speed_ratio is None:
            speed_ratio = resample_rate

        # 新增 'mhsa' 分支；默认 8 头，保证与常见 2048/1024/512/256 通道都整除
        # ---- 注意力插入策略：为 Slow / Fast 各准备一个构造函数 ----

        def _build_mhsa(ch: int) -> nn.Module:
            return GlobalSelfAttention3D(
                channels=ch,
                num_heads=8,
                qkv_bias=True,
                attn_drop=0.0,
                proj_drop=0.0,
            )

        def _build_conservative(ch: int) -> nn.Module:
            return ConservativeTransformerInsert(ch)

        def _build_stage(ch: int) -> nn.Module:
            return StageTrainingTransformer(ch)

        def _build_tsa(ch: int) -> nn.Module:
            # 专门给 Fast 分支用的 TSA
            return TemporalShiftAttention3D(channels=ch, shift_ratio=1 / 8)

        # 默认：Slow / Fast 用同一套（保持向后兼容）
        tcls_slow = _build_conservative
        tcls_fast = _build_conservative

        if transformer_mode == "mhsa":
            # 原始行为：Slow/Fast 都是 MHSA
            tcls_slow = _build_mhsa
            tcls_fast = _build_mhsa
        elif transformer_mode == "conservative":
            tcls_slow = _build_conservative
            tcls_fast = _build_conservative
        elif transformer_mode == "stage":
            tcls_slow = _build_stage
            tcls_fast = _build_stage
        elif transformer_mode == "mhsa_slow_tsa_fast":
            # 新增模式：Slow = MHSA, Fast = TSA
            tcls_slow = _build_mhsa
            tcls_fast = _build_tsa
        else:
            # 兜底：当作 stage 处理，避免旧配置崩
            tcls_slow = _build_stage
            tcls_fast = _build_stage

        # ---- Slow 用带 lateral 的官方 Pathway；Fast 用原 ResNet3d ----
        def _make_pathway(cfg: dict, is_slow: bool):
            cfg = cfg.copy(); cfg.pop("type", None)
            if is_slow:
                path = ResNet3dPathway(
                    lateral=True,
                    speed_ratio=speed_ratio,
                    channel_ratio=channel_ratio,
                    fusion_kernel=fusion_kernel,
                    lateral_infl=lateral_infl,
                    lateral_norm=lateral_norm,
                    lateral_inv=lateral_inv,
                    lateral_activate=list(lateral_activate),
                    **cfg
                )
                tcls = tcls_slow
            else:
                path = ResNet3d(**cfg)
                tcls = tcls_fast

            # 在指定层插入注意力模块
            for lv in sorted(set(transformer_layers)):
                if lv == 3:
                    ch = _stage_out_channels(path.layer3)
                    path.layer3 = nn.Sequential(path.layer3, tcls(ch))
                elif lv == 4:
                    ch = _stage_out_channels(path.layer4)
                    path.layer4 = nn.Sequential(path.layer4, tcls(ch))
                elif lv == 5:
                    # 约定：5 表示在 layer4 后再叠一个模块
                    ch = _stage_out_channels(path.layer4)
                    path.layer4 = nn.Sequential(path.layer4, tcls(ch))
                else:
                    pass
            return path



        self.slow_path = _make_pathway(slow_pathway, is_slow=True)
        self.fast_path = _make_pathway(fast_pathway, is_slow=False)

        # 与原接口保持一致：给出 slow+fast 的总通道数（供上层参考）
        def _path_out_dim(path: nn.Module) -> int:
            core = _unwrap_to_block_sequential(path.layer4)
            last = core[-1] if isinstance(core, nn.Sequential) else core
            if hasattr(last, 'conv3') and isinstance(last.conv3, nn.Conv3d):
                return int(last.conv3.out_channels)
            if hasattr(last, 'conv2') and isinstance(last.conv2, nn.Conv3d):
                return int(last.conv2.out_channels)
            for sub in reversed(list(last.modules())):
                if isinstance(sub, nn.Conv3d):
                    return int(sub.out_channels)
            raise RuntimeError('Cannot infer out_channels for layer4.')

    def set_training_stage(self, stage: int):
        # 保持你原本对 MHSA 的逐阶段控制
        for p in [self.slow_path, self.fast_path]:
            if isinstance(p.layer4, nn.Sequential) and len(p.layer4) > 1 and hasattr(p.layer4[1], "set_training_stage"):
                p.layer4[1].set_training_stage(stage)

    def forward(self, x):
        # 与原版一致的输入处理（支持 NCTHW / list），参考你现有实现
        if isinstance(x, torch.Tensor):
            if x.dim() == 6 and x.shape[1] == 1:
                x = x.squeeze(1)
            if x.dim() != 5:
                raise ValueError("SlowFast expects 5D input (N,C,T,H,W)")
            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1, 1)
            x_fast = x
            x_slow = x[:, :, :: self.alpha, :, :]
        else:
            x_slow, x_fast = x
            x_slow = x_slow if x_slow.dim() == 5 else x_slow.squeeze(1)
            x_fast = x_fast if x_fast.dim() == 5 else x_fast.squeeze(1)
            x_slow = x_slow.repeat(1, 3, 1, 1, 1) if x_slow.shape[1] == 1 else x_slow
            x_fast = x_fast.repeat(1, 3, 1, 1, 1) if x_fast.shape[1] == 1 else x_fast

        # ---------------- conv1 / pool1 ----------------
        s = self.slow_path.conv1(x_slow); s = self.slow_path.maxpool(s)
        f = self.fast_path.conv1(x_fast); f = self.fast_path.maxpool(f)

        # conv1 侧向：Fast → Slow
        if hasattr(self.slow_path, 'conv1_lateral'):
            f_lat = self.slow_path.conv1_lateral(f)   # 对齐时间尺度与通道
            s = torch.cat((s, f_lat), dim=1)

        # ---------------- res2 / res3 / res4 / res5 （逐层侧连） ----------------
        for i, layer_name in enumerate(self.slow_path.res_layers):
            # 对应层前向
            s = getattr(self.slow_path, layer_name)(s)
            f = getattr(self.fast_path, layer_name)(f)

            # 非最后一层才做 lateral（与官方一致）
            if (i != len(self.slow_path.res_layers) - 1) and hasattr(self.slow_path, 'lateral_connections'):
                lateral_name = self.slow_path.lateral_connections[i]
                if hasattr(self.slow_path, lateral_name):
                    f_lat = getattr(self.slow_path, lateral_name)(f)
                    s = torch.cat((s, f_lat), dim=1)

        # ---------- 暴露蒸馏特征：同时保存 slow 和 fast ----------
        # 方便调试的话，也可以顺手把单独分支存一下
        self.feature_map_for_distill_slow = s
        self.feature_map_for_distill_fast = f

        # AttentionAlignHook 现在支持 dict，会对每个分支单独算一个损失再平均
        self.feature_map_for_distill = dict(
            slow=s,
            fast=f,
        )

        # 与你现有接口保持一致：时间/空间对齐后做 concat 输出
        if (s.shape[2] != f.shape[2]) or (s.shape[3] != f.shape[3]) or (s.shape[4] != f.shape[4]):
            f = nn.functional.interpolate(
                f,
                size=(s.shape[2], s.shape[3], s.shape[4]),
                mode="trilinear",
                align_corners=False
            )
        return torch.cat([s, f], dim=1)
