# 文件路径: mmaction/models/losses/attention_guidance_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmaction.registry import MODELS
import warnings


@MODELS.register_module()
class AttentionGuidanceLoss(nn.Module):
    """注意力引导损失模块"""

    def __init__(self,
                 loss_weight=1.0,
                 teacher_checkpoint=None,
                 teacher_config=None,
                 guidance_type='center',  # 'center', 'teacher', 'adaptive'
                 temperature=4.0,
                 start_epoch=0,
                 end_epoch=120):
        super().__init__()

        self.loss_weight = loss_weight
        self.teacher_checkpoint = teacher_checkpoint
        self.guidance_type = guidance_type
        self.temperature = temperature
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch

        # 当前epoch（需要外部设置）
        self.current_epoch = 0

        # 教师模型（可选）
        self.teacher_model = None
        if teacher_checkpoint and teacher_config:
            self._init_teacher_model(teacher_config)

        print(f"AttentionGuidanceLoss initialized: weight={loss_weight}, type={guidance_type}")

    def _init_teacher_model(self, teacher_config):
        """初始化教师模型"""
        try:
            self.teacher_model = MODELS.build(teacher_config)

            if self.teacher_checkpoint:
                checkpoint = torch.load(self.teacher_checkpoint, map_location='cpu')
                state_dict = checkpoint.get('state_dict', checkpoint)
                self.teacher_model.load_state_dict(state_dict, strict=False)

            # 冻结教师模型
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            self.teacher_model.eval()

            print(f"✓ Teacher model loaded from {self.teacher_checkpoint}")

        except Exception as e:
            print(f"✗ Failed to load teacher model: {e}")
            self.teacher_model = None

    def set_epoch(self, epoch):
        """设置当前训练轮次"""
        self.current_epoch = epoch

    def get_current_weight(self):
        """计算当前损失权重"""
        if self.current_epoch < self.start_epoch:
            return 0.0
        elif self.current_epoch > self.end_epoch:
            return self.loss_weight * 0.1  # 保持10%的引导
        else:
            # 线性衰减
            progress = (self.current_epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)
            return self.loss_weight * (1 - progress * 0.9)

    def forward(self, student_features, input_data=None, **kwargs):
        """计算注意力引导损失"""
        current_weight = self.get_current_weight()

        if current_weight == 0.0:
            return torch.tensor(0.0, device=student_features.device, requires_grad=True)

        # 根据引导类型计算损失
        if self.guidance_type == 'center':
            return self._center_guidance_loss(student_features, current_weight)
        elif self.guidance_type == 'teacher' and self.teacher_model is not None:
            return self._teacher_guidance_loss(student_features, input_data, current_weight)
        elif self.guidance_type == 'adaptive':
            return self._adaptive_guidance_loss(student_features, current_weight)
        else:
            return self._center_guidance_loss(student_features, current_weight)

    def _center_guidance_loss(self, student_features, weight):
        """中心注意力引导损失"""
        if not isinstance(student_features, (list, tuple)):
            student_features = [student_features]

        total_loss = 0.0
        valid_features = 0

        for feat in student_features:
            if feat is not None and feat.numel() > 0:
                B, C, T, H, W = feat.shape

                # 创建中心注意力目标
                center_attention = self._create_center_attention(B, T, H, W, feat.device)

                # 将特征转换为注意力图
                feat_attention = self._feature_to_attention(feat)

                # 计算KL散度损失
                loss = self._compute_attention_loss(feat_attention, center_attention)
                total_loss += loss
                valid_features += 1

        if valid_features > 0:
            return weight * total_loss / valid_features
        else:
            return torch.tensor(0.0, device=student_features[0].device, requires_grad=True)

    def _teacher_guidance_loss(self, student_features, input_data, weight):
        """教师模型引导损失"""
        if self.teacher_model is None or input_data is None:
            return self._center_guidance_loss(student_features, weight)

        try:
            with torch.no_grad():
                # 从教师模型提取注意力
                teacher_attention = self._extract_teacher_attention(input_data)

            if teacher_attention is None:
                return self._center_guidance_loss(student_features, weight)

            # 计算学生特征注意力
            if not isinstance(student_features, (list, tuple)):
                student_features = [student_features]

            total_loss = 0.0
            valid_features = 0

            for feat in student_features:
                if feat is not None and feat.numel() > 0:
                    student_attention = self._feature_to_attention(feat)

                    # 调整教师注意力尺寸
                    adapted_teacher_attention = self._adapt_attention_size(
                        teacher_attention, student_attention.shape
                    )

                    # 计算损失
                    loss = self._compute_attention_loss(student_attention, adapted_teacher_attention)
                    total_loss += loss
                    valid_features += 1

            if valid_features > 0:
                return weight * total_loss / valid_features
            else:
                return torch.tensor(0.0, device=student_features[0].device, requires_grad=True)

        except Exception as e:
            print(f"Teacher guidance failed: {e}")
            return self._center_guidance_loss(student_features, weight)

    def _adaptive_guidance_loss(self, student_features, weight):
        """自适应注意力引导损失"""
        if not isinstance(student_features, (list, tuple)):
            student_features = [student_features]

        total_loss = 0.0
        valid_features = 0

        for feat in student_features:
            if feat is not None and feat.numel() > 0:
                # 基于特征分布创建自适应目标
                adaptive_attention = self._create_adaptive_attention(feat)

                # 将特征转换为注意力图
                feat_attention = self._feature_to_attention(feat)

                # 计算损失
                loss = self._compute_attention_loss(feat_attention, adaptive_attention)
                total_loss += loss
                valid_features += 1

        if valid_features > 0:
            return weight * total_loss / valid_features
        else:
            return torch.tensor(0.0, device=student_features[0].device, requires_grad=True)

    def _feature_to_attention(self, features):
        """将特征图转换为注意力图"""
        # 对通道维度求平均得到空间注意力
        attention = features.mean(dim=1)  # [B, T, H, W]

        # 归一化到[0,1]
        attention = self._normalize_attention(attention)

        return attention

    def _normalize_attention(self, attention):
        """归一化注意力图"""
        B = attention.shape[0]
        attention_flat = attention.view(B, -1)

        # 最小-最大归一化
        min_val = attention_flat.min(dim=1, keepdim=True)[0]
        max_val = attention_flat.max(dim=1, keepdim=True)[0]

        normalized = (attention_flat - min_val) / (max_val - min_val + 1e-8)

        return normalized.view_as(attention)

    def _create_center_attention(self, B, T, H, W, device):
        """创建中心注意力图"""
        attention = torch.zeros(B, T, H, W, device=device)
        center_h, center_w = H // 2, W // 2
        sigma = min(H, W) / 6.0  # 控制注意力分布的宽度

        for i in range(H):
            for j in range(W):
                dist_sq = (i - center_h) ** 2 + (j - center_w) ** 2
                attention[:, :, i, j] = torch.exp(-dist_sq / (2 * sigma ** 2))

        # 归一化
        attention = self._normalize_attention(attention)

        return attention

    def _create_adaptive_attention(self, features):
        """基于特征创建自适应注意力"""
        B, C, T, H, W = features.shape

        # 计算每个空间位置的特征强度
        feature_magnitude = torch.norm(features, dim=1)  # [B, T, H, W]

        # 应用软最大值增强显著区域
        attention = F.softmax(feature_magnitude.view(B, -1) / self.temperature, dim=1)
        attention = attention.view(B, T, H, W)

        return attention

    def _compute_attention_loss(self, student_attention, target_attention):
        """计算注意力损失"""
        # 使用KL散度 + MSE损失的组合

        # KL散度损失
        student_prob = F.softmax(student_attention.view(student_attention.shape[0], -1), dim=1)
        target_prob = F.softmax(target_attention.view(target_attention.shape[0], -1), dim=1)

        kl_loss = F.kl_div(
            F.log_softmax(student_attention.view(student_attention.shape[0], -1), dim=1),
            target_prob,
            reduction='batchmean'
        )

        # MSE损失
        mse_loss = F.mse_loss(student_attention, target_attention)

        # 组合损失
        total_loss = 0.7 * kl_loss + 0.3 * mse_loss

        return total_loss

    def _extract_teacher_attention(self, input_data):
        """从教师模型提取注意力（简化版本）"""
        try:
            # 这里应该实现真正的教师模型注意力提取
            # 目前返回None，使用中心注意力作为后备
            return None
        except Exception:
            return None

    def _adapt_attention_size(self, attention, target_shape):
        """调整注意力图尺寸"""
        B, T, H, W = target_shape

        if attention.shape != target_shape:
            attention = F.interpolate(
                attention.unsqueeze(1),  # 添加channel维度
                size=(T, H, W),
                mode='trilinear',
                align_corners=False
            ).squeeze(1)

        return attention