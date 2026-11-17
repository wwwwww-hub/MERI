# models/backbones/modules/convvit_block.py
import torch, torch.nn as nn
from mmaction.models.backbones.swin import PatchEmbed3D

class ConvViTStage3D(nn.Module):
    """一个 Stage = 先 3×3×3 Conv Downsample (可选)，再若干个 SwinBlockSequence"""
    def __init__(self,
                 in_channels,
                 embed_dim,
                 depth,
                 num_heads,
                 stride=(1,2,2),
                 window_size=(8,7,7),
                 use_conv_downsample=True,
                 drop_path=0.1):
        super().__init__()
        if use_conv_downsample:
            self.downsample = nn.Conv3d(in_channels, embed_dim,
                                        kernel_size=3, stride=stride,
                                        padding=1, bias=False)
            self.norm = nn.BatchNorm3d(embed_dim)
        else:
            self.downsample = self.norm = nn.Identity()

        self.patch_embed = PatchEmbed3D(
            in_channels=embed_dim,
            embed_dims=embed_dim,
            kernel_size=(1,1,1),
            stride=(1,1,1))
        self.blocks = SwinBlockSequence(
            embed_dims=embed_dim,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            drop_path_rate=drop_path,
            use_checkpoint=False)

    def forward(self, x):
        x = self.norm(self.downsample(x))
        # (B, C, T, H, W) -> (B, N, C)
        x, _ = self.patch_embed(x)
        x = self.blocks(x)            # 经过 Swin Block
        # 返回 (B, C, T, H, W) 供后续 AdaptiveAvgPool3d / RoIAlign3D 使用
        B, N, C = x.shape
        T = H = W = int(round(N ** (1/3)))  # 简化：假设 cubic
        x = x.transpose(1, 2).reshape(B, C, T, H, W)
        return x
