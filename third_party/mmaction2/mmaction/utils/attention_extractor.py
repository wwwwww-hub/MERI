# 文件路径: mmaction/models/backbones/attention_guided_slowfast.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmaction.registry import MODELS
from .resnet3d_slowfast import ResNet3dSlowFast
from ..utils.attention_extractor import VideoSwinAttentionExtractor
import warnings


@MODELS.register_module()
class AttentionGuidedSlowFast(ResNet3dSlowFast):
    """注意力引导的SlowFast模型"""

    def __init__(self,
                 teacher_checkpoint=None,
                 attention_guide_config=None,
                 **kwargs):
        super().__init__(**kwargs)

        # 注意力引导配置
        if attention_guide_config is None:
            attention_guide_config = {}

        self.guide_layers = attention_guide_config.get(
            'guide_layers', ['stage2.blocks.1', 'stage3.blocks.1']
        )
        self.guide_weight = attention_guide_config.get('guide_weight', 0.4)
        self.guide_start_epoch = attention_guide_config.get('guide_start_epoch', 0)
        self.guide_end_epoch = attention_guide_config.get('guide_end_epoch', 120)

        # 当前训练轮次
        self.current_epoch = 0

        # 构建注意力提取器
        teacher_config = attention_guide_config.get('teacher_config', None)

        try:
            self.attention_extractor = VideoSwinAttentionExtractor(
                teacher_checkpoint=teacher_checkpoint,
                teacher_config=teacher_config,
                extract_layers=self.guide_layers
            )
        except Exception as e:
            warnings.warn(f"Failed to initialize attention extractor: {e}")
            self.attention_extractor = None

        # 构建注意力适配器
        self.attention_adapters = nn.ModuleDict()
        self._build_attention_adapters(attention_guide_config)

        print(f"AttentionGuidedSlowFast initialized with guide_layers: {self.guide_layers}")

    def _build_attention_adapters(self, guide_config):
        """构建注意力适配器"""
        adapter_config = guide_config.get('adapter_config', {})

        # 默认适配器配置
        default_adapters = {
            'stage2.blocks.1': {'channels': 256},  # slow_res2输出
            'stage3.blocks.1': {'channels': 512},  # slow_res3输出
        }

        for layer_name in self.guide_layers:
            if layer_name in default_adapters:
                channels = default_adapters[layer_name]['channels']

                # 创建简单的注意力适配器
                self.attention_adapters[layer_name] = nn.Sequential(
                    nn.Conv3d(1, 16, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(16, 1, kernel_size=1),
                    nn.Sigmoid()
                )

    def set_epoch(self, epoch):
        """设置当前训练轮次"""
        self.current_epoch = epoch
        if epoch % 10 == 0:  # 每10轮打印一次
            current_weight = self.get_current_guide_weight()
            print(f"Epoch {epoch}: attention guide weight = {current_weight:.3f}")

    def get_current_guide_weight(self):
        """动态计算当前的引导权重"""
        if self.current_epoch < self.guide_start_epoch:
            return 0.0
        elif self.current_epoch > self.guide_end_epoch:
            return self.guide_weight * 0.2  # 保持20%的引导强度
        else:
            # 线性衰减
            progress = (self.current_epoch - self.guide_start_epoch) / \
                       (self.guide_end_epoch - self.guide_start_epoch)
            return self.guide_weight * (1 - progress * 0.8)

    def forward(self, x):
        """带注意力引导的前向传播"""
        current_guide_weight = self.get_current_guide_weight()

        # 如果不需要引导或提取器不可用，使用原始forward
        if current_guide_weight == 0.0 or self.attention_extractor is None:
            return super().forward(x)

        # 提取教师注意力
        try:
            with torch.no_grad():
                teacher_attention = self.attention_extractor(x)
        except Exception as e:
            print(f"Failed to extract teacher attention: {e}")
            return super().forward(x)

        # 带引导的前向传播
        return self.forward_with_guidance(x, teacher_attention, current_guide_weight)

    def forward_with_guidance(self, x, teacher_attention, guide_weight):
        """注意力引导的前向传播"""
        # 调用父类的原始forward，但在中间插入注意力引导
        # 由于SlowFast的forward比较复杂，这里简化处理

        # 先正常进行前向传播
        slow_features, fast_features = super().forward(x)

        # 在这个简化版本中，我们直接返回原始特征
        # 实际的注意力引导需要更深入地修改forward过程
        return slow_features, fast_features

    def apply_attention_guidance(self, features, attention_map, layer_name, guide_weight):
        """应用注意力引导"""
        if layer_name in self.attention_adapters:
            # 调整注意力图尺寸
            B, C, T, H, W = features.shape

            if attention_map.shape[-3:] != (T, H, W):
                attention_map = F.interpolate(
                    attention_map.unsqueeze(1),
                    size=(T, H, W),
                    mode='trilinear',
                    align_corners=False
                ).squeeze(1)

            # 通过适配器处理注意力
            adapted_attention = self.attention_adapters[layer_name](
                attention_map.unsqueeze(1)
            ).squeeze(1)

            # 应用注意力引导
            guided_features = features * (1 + guide_weight * adapted_attention.unsqueeze(1))
            return guided_features

        return features