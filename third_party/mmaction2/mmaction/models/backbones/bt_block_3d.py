import torch, torch.nn as nn
from einops import rearrange

class MHSA3D(nn.Module):                     # 3D Multi-Head Self-Attention
    def __init__(self, dim, heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Conv3d(dim, dim * 3, 1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv3d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):                    # x: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        qkv = self.qkv(x).reshape(B, 3, self.heads, C // self.heads, T * H * W)
        q, k, v = qkv[:,0], qkv[:,1], qkv[:,2]        # (B, h, C/h, N)
        attn = (q.transpose(-2,-1) @ k) * self.scale  # (B,h,N,N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out  = (v @ attn.transpose(-2,-1)).reshape(B, C, T, H, W)
        out  = self.proj(out)
        return self.proj_drop(out)

class BTBlock3d(nn.Module):                  # BoTNet-style bottleneck
    def __init__(self, inplanes, planes, stride=1,
                 heads=8, reduction=4):
        super().__init__()
        width = planes // reduction
        self.conv1 = nn.Conv3d(inplanes, width, 1, bias=False)
        self.bn1   = nn.BatchNorm3d(width)
        self.mhsa  = MHSA3D(width, heads=heads)
        self.conv3 = nn.Conv3d(width, planes, 1, bias=False)
        self.bn3   = nn.BatchNorm3d(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = (None if stride==1 and inplanes==planes else
                           nn.Sequential(nn.Conv3d(inplanes, planes, 1, stride, bias=False),
                                         nn.BatchNorm3d(planes)))
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.mhsa(out)                          # ✨ attention 替代 3×3 conv
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = self.relu(out + identity)
        return out
