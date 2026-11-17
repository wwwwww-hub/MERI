# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from mmengine.registry import MODELS


@MODELS.register_module()
class SingleRoIExtractor3D(nn.Module):
    """Extract RoI features from a single‑level 3D feature map.

    **Key changes (2025‑06‑16)**
    1. *FP16 safety*: `rois` is now cast to the same `dtype` and `device` as
       the feature map before calling RoIAlign. This prevents the common
       RuntimeError: "expected scalar type Half but found Float" when AMP is on.
    2. *Minor clean‑ups*: kept original logic intact; only added two small
       helper lines (`_align_rois_dtype`).
    """

    def __init__(
        self,
        roi_layer_type: str = "RoIAlign",
        featmap_stride: int = 16,
        output_size: int | Tuple[int, int] = 16,
        sampling_ratio: int = 0,
        pool_mode: str = "avg",
        aligned: bool = True,
        with_temporal_pool: bool = True,
        temporal_pool_mode: str = "avg",
        with_global: bool = False,
    ) -> None:
        super().__init__()
        self.roi_layer_type = roi_layer_type
        assert self.roi_layer_type in {"RoIPool", "RoIAlign"}
        self.featmap_stride = featmap_stride
        self.spatial_scale = 1.0 / self.featmap_stride

        self.output_size = output_size
        self.sampling_ratio = sampling_ratio
        self.pool_mode = pool_mode
        self.aligned = aligned

        self.with_temporal_pool = with_temporal_pool
        self.temporal_pool_mode = temporal_pool_mode

        self.with_global = with_global

        try:
            from mmcv.ops import RoIAlign, RoIPool  # noqa: F401
        except (ImportError, ModuleNotFoundError):
            raise ImportError(
                "Failed to import `RoIAlign` and `RoIPool` from `mmcv.ops`. "
                "The two modules will be used in `SingleRoIExtractor3D`!"
            )

        if self.roi_layer_type == "RoIPool":
            self.roi_layer = RoIPool(self.output_size, self.spatial_scale)
        else:
            self.roi_layer = RoIAlign(
                self.output_size,
                self.spatial_scale,
                sampling_ratio=self.sampling_ratio,
                pool_mode=self.pool_mode,
                aligned=self.aligned,
            )
        self.global_pool = nn.AdaptiveAvgPool2d(self.output_size)

    # ---------------------------------------------------------------------
    # Helper
    # ---------------------------------------------------------------------
    @staticmethod
    def _align_rois_dtype(rois: Tensor, ref_tensor: Tensor) -> Tensor:
        """Ensure `rois` has the same dtype & device as `ref_tensor`."""
        if rois.dtype != ref_tensor.dtype or rois.device != ref_tensor.device:
            rois = rois.to(dtype=ref_tensor.dtype, device=ref_tensor.device)
        return rois

    # ---------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------
    def forward(self, feat: Union[Tensor, Tuple[Tensor]], rois: Tensor) -> tuple:
        """Forward function for extracting RoI features.

        Args:
            feat (Tensor | Tuple[Tensor]): Video features of shape (N, C, T, H, W).
            rois (Tensor): RoIs, shape (K, 5) with (batch_idx, x1, y1, x2, y2).

        Returns:
            tuple: (roi_feats, global_feats)
                * roi_feats (Tensor): Extracted RoI features (N_rois, C, T', H', W').
                * global_feats (Tensor): Global spatio‑temporal features (N, C, T, H, W).
        """
        # ---- 1. Normalise input container --------------------------------
        if not isinstance(feat, tuple):
            feat = (feat,)

        # ---- 2. Resize all paths to the largest temporal length ------------
        if len(feat) >= 2:
            max_T = max(x.shape[2] for x in feat)
            target_shape = (max_T,) + feat[0].shape[3:]  # (T, H, W)
            feat = [F.interpolate(x, target_shape).contiguous() for x in feat]

        # ---- 3. Optional temporal pooling ----------------------------------
        if self.with_temporal_pool:
            if self.temporal_pool_mode == "avg":
                feat = [torch.mean(x, dim=2, keepdim=True) for x in feat]
            elif self.temporal_pool_mode == "max":
                feat = [torch.max(x, dim=2, keepdim=True)[0] for x in feat]
            else:
                raise NotImplementedError(
                    f"Unsupported temporal_pool_mode: {self.temporal_pool_mode}"
                )

        # ---- 4. Concatenate along channel dim -----------------------------
        feat = torch.cat(feat, dim=1).contiguous()  # (N, C_cat, T', H, W)

        # Align dtype/device once for all frames
        rois = self._align_rois_dtype(rois, feat)

        # ---- 5. Per‑frame RoIAlign ----------------------------------------
        roi_feats = []
        for t in range(feat.size(2)):
            frame_feat = feat[:, :, t].contiguous()  # (N, C, H, W)

            roi_feat = self.roi_layer(frame_feat, rois)

            if self.with_global:
                global_feat = self.global_pool(frame_feat)  # (N, C, H', W')
                inds = rois[:, 0].long()  # batch indices
                global_feat = global_feat[inds]
                roi_feat = torch.cat([roi_feat, global_feat], dim=1).contiguous()

            roi_feats.append(roi_feat)

        roi_feats = torch.stack(roi_feats, dim=2)  # (N_rois, C', T', H', W')
        return roi_feats, feat
