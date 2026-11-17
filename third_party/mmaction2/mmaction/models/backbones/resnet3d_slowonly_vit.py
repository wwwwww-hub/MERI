# -----------------------------------------------------------------------------
# ResNet3dSlowOnlyViT
# -----------------------------------------------------------------------------
#   • Builds on official ResNet3dSlowOnly backbone.
#   • Inserts a lightweight MHSA‑style Transformer block *after layer4*.
#   • Exposes `self.feature_map_for_distill` for AttentionAlignHook.
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn
from mmaction.models.backbones.resnet3d_slowonly import ResNet3dSlowOnly
from mmengine.registry import MODELS

# ---------------------------- helper MHSA block ------------------------------
class LiteMHSA(nn.Module):
    """3D Multi‑head self‑attention operating on (B,C,T,H,W).
    Token = flatten(T,H,W). No positional embed; lightweight feed‑forward.
    """
    def __init__(self, embed_dim: int = 2048, num_heads: int = 8,
                 ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * ff_mult),
            nn.GELU(),
            nn.Linear(embed_dim * ff_mult, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, C, T, H, W) → (B, N, C)
        B, C, T, H, W = x.shape
        x = x.view(B, C, T * H * W).transpose(1, 2)  # (B,N,C)

        qkv = self.qkv(self.norm1(x)).reshape(B, -1, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.unbind(2)  # each (B, N, heads, head_dim)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, -1, C)
        x = self.proj_drop(self.proj(x))
        x = x + self.ffn(self.norm2(x))            # residual FFN

        x = x.transpose(1, 2).reshape(B, C, T, H, W)
        return x

class FlowMagnitudeAttention3D(nn.Module):

    def __init__(self, channels: int, kernel_size=(3, 3, 3)):
        super().__init__()
        kt, kh, kw = kernel_size
        self.spatial_conv = nn.Conv3d(
                1, 1,
                kernel_size=(kt, kh, kw),
                padding=(kt // 2, kh // 2, kw // 2),
                bias=True,
            )

        self.alpha = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        B, C, T, H, W = x.shape

        # 通道维平方 + 平均，做“运动幅值”近似 (B,1,T,H,W)
        mag = x.pow(2).mean(dim=1, keepdim=True)

        # 小 3D 卷积 + Sigmoid 得到时空注意力图
        attn = torch.sigmoid(self.spatial_conv(mag))  # (B,1,T,H,W)

        # 按位置缩放特征，然后 residual
        y = x * attn
        return x + self.alpha * (y - x)


# ------------------------- backbone with inserts -------------------------
@MODELS.register_module()
class ResNet3dSlowOnlyViT(ResNet3dSlowOnly):
    """SlowOnly backbone + FlowMagnitude@layer2 + LiteMHSA@layer4.

    - 继承官方 ResNet3dSlowOnly；
    - 在 layer2 末尾插入 FlowMagnitudeAttention3D（利用光流“幅值”信息）；
    - 在 layer4 末尾插入 LiteMHSA（全局时空 MHSA）；
    - 最后把输出存在 self.feature_map_for_distill，供注意力蒸馏 Hook 使用。
    """

    def __init__(self, num_heads: int = 8, **kwargs):
        super().__init__(**kwargs)

        # ---- 统计 layer2 / layer4 输出通道数 ----
        last_block2 = self.layer2[-1]
        c2 = last_block2.conv3.out_channels if hasattr(last_block2, 'conv3') \
            else last_block2.conv2.out_channels

        last_block4 = self.layer4[-1]
        c4 = last_block4.conv3.out_channels if hasattr(last_block4, 'conv3') \
            else last_block4.conv2.out_channels

        # ---- 构造模块（注意：这里只用局部变量，避免重复注册）----
        flow_mag2 = FlowMagnitudeAttention3D(channels=c2)
        mhsa4 = LiteMHSA(embed_dim=c4, num_heads=num_heads)

        # 把模块“缝”到对应 stage 后面：
        # layer2: 原来的 layer2 → FlowMagnitudeAttention3D
        self.layer2 = nn.Sequential(self.layer2, flow_mag2)
        # layer4: 原来的 layer4 → LiteMHSA
        self.layer4 = nn.Sequential(self.layer4, mhsa4)

    def forward(self, x: torch.Tensor):
        # (B, 2, T, H, W) optical-flow 输入
        # super().forward 会依次跑 conv1 → layer1 → layer2(含 FlowMag) →
        #                 layer3 → layer4(含 MHSA)
        x = super().forward(x)   # (B, C, T', H', W')

        # 供注意力蒸馏 Hook 使用
        self.feature_map_for_distill = x

        # 与 SlowFast 保持接口一致：返回 tuple
        return (x,)

