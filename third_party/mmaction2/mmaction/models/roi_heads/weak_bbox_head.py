# mmaction/models/roi_heads/weak_bbox_head.py
import torch
import torch.nn as nn
from mmaction.registry import MODELS as MMACTION_MODELS
from mmdet.registry import MODELS as MMDET_MODELS
from mmaction.models.roi_heads.bbox_heads.bbox_head import BBoxHeadAVA


@MMACTION_MODELS.register_module()
@MMDET_MODELS.register_module()
class WeakBBoxHeadAVA(BBoxHeadAVA):
    """BBox head for weakly‑supervised, multi‑label action detection.

    * Fixes the original BBoxHeadAVA parameter order bug (nn.Linear created as
      ``nn.Linear(num_classes, in_channels)``) by re‑initialising ``fc_cls``
      with the correct (``in_channels``, ``num_classes``) order.
    * Uses BCEWithLogits loss for soft labels.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # ------------------------------------------------------------------
        # Replace the incorrectly shaped classifier with a correct one
        # Input feature dim  == self.in_channels (C)
        # Output dim (classes) == self.num_classes
        # ------------------------------------------------------------------
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)

        # Loss for weak (soft) labels
        self.loss_cls = nn.BCEWithLogitsLoss()

    # ----------------------------------------------------------------------
    # Forward (train)
    # ----------------------------------------------------------------------
    def forward_train(
        self,
        x,                     # (N, C, 1, 1, 1)
        data_samples,
        gt_bboxes,
        gt_scores,             # (N, num_classes) – soft labels in [0,1]
        **kwargs,
    ):
        cls_score = self(x)
        return dict(loss_cls=self.loss_cls(cls_score, gt_scores))
