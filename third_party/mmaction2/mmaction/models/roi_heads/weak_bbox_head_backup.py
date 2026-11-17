import torch
import torch.nn as nn
from mmaction.registry import MODELS as MMACTION_MODELS
from mmdet.registry import MODELS as MMDET_MODELS
from mmaction.models.roi_heads.bbox_heads.bbox_head import BBoxHeadAVA

@MMACTION_MODELS.register_module()
@MMDET_MODELS.register_module()
class WeakBBoxHeadAVA(BBoxHeadAVA):
    """
    Weak-label aware AVA bbox head.

    使用自定义 BCEWithLogitsLoss 对软标签进行训练。
    """
    def __init__(self, *args, **kwargs):
        # 保持与父类 __init__ 签名一致，自动接收所有参数
        super().__init__(*args, **kwargs)
        # 将分类损失替换为 BCEWithLogitsLoss
        self.loss_cls = nn.BCEWithLogitsLoss()

    def forward_train(self, x, data_samples, gt_bboxes, gt_scores, **kwargs):
        """
        Args:
            x (Tensor): RoI 特征
            data_samples (List[ActionDataSample]): 样本元信息列表
            gt_bboxes (Tensor[N,4]): 真实框
            gt_scores (Tensor[N,C]): 软标签分数
        Returns:
            dict: 包含 'loss_cls' 的分类损失
        """
        # 计算分类得分
        cls_score = self(x)
        # 计算软标签损失
        loss = self.loss_cls(cls_score, gt_scores)
        return {'loss_cls': loss}
