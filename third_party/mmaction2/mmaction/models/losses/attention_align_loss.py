import torch.nn as nn
from mmengine.registry import MODELS
import torch.nn.functional as F

@MODELS.register_module()
class AttentionAlignLoss(nn.Module):
    """KL or MSE between teacher & student attention."""
    def __init__(self, mode='mse', loss_weight=0.2):
        super().__init__()
        self.mode = mode
        self.w = loss_weight
        self.kl = nn.KLDivLoss(reduction='batchmean')
        self.mse = nn.MSELoss()

    def forward(self, student_attn, teacher_attn):
        if teacher_attn.shape[-1] != student_attn.shape[-1]:
            teacher_attn = F.interpolate(
                teacher_attn.unsqueeze(1),            # (B,1,N,N) teacher_attn.unsqueeze(1),            # (B,1,N,N)
                size=student_attn.shape[-1],
                mode='bilinear', align_corners=False
               ).squeeze(1)
        if self.mode == 'kl':
            loss = self.kl(student_attn.log(), teacher_attn)
        else:
            loss = self.mse(student_attn, teacher_attn)
        return self.w * loss
