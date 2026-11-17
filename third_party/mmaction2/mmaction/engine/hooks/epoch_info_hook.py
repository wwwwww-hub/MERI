# 创建 mmaction/engine/hooks/epoch_info_hook.py
from mmengine.hooks import Hook
from mmaction.registry import HOOKS


@HOOKS.register_module()
class EpochInfoHook(Hook):
    """将当前epoch信息传递给模型"""

    def before_train_epoch(self, runner):
        """在每个训练epoch开始前更新模型的epoch信息"""
        if hasattr(runner.model.module.backbone, 'set_epoch'):
            runner.model.module.backbone.set_epoch(runner.epoch)
        elif hasattr(runner.model.backbone, 'set_epoch'):
            runner.model.backbone.set_epoch(runner.epoch)