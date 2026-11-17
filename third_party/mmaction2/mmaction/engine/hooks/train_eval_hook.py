# 创建文件：mmaction/engine/hooks/simple_train_eval_hook.py

from typing import Dict, Optional
import torch
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.evaluator import Evaluator
from mmaction.registry import HOOKS


@HOOKS.register_module()
class SimpleTrainEvalHook(Hook):
    """简化版训练集评估钩子"""

    priority = 75

    def __init__(self,
                 train_dataloader_cfg: Dict,
                 train_evaluator_cfg: Dict,
                 interval: int = 1):
        super().__init__()
        self.train_dataloader_cfg = train_dataloader_cfg
        self.train_evaluator_cfg = train_evaluator_cfg
        self.interval = interval
        self._train_dataloader = None
        self._train_evaluator = None

    def _build_train_evaluator(self, runner: Runner):
        """构建训练集数据加载器和评估器"""
        if self._train_dataloader is None:
            self._train_dataloader = runner.build_dataloader(
                self.train_dataloader_cfg
            )

        if self._train_evaluator is None:
            self._train_evaluator = Evaluator([self.train_evaluator_cfg])

    def after_val_epoch(self, runner: Runner, **kwargs) -> None:
        """在验证epoch后执行训练集评估"""
        # 检查是否需要在这个epoch进行评估
        if not self.every_n_epochs(runner, self.interval):
            return

        try:
            # 构建评估器
            self._build_train_evaluator(runner)

            # 设置模型为评估模式
            runner.model.eval()

            runner.logger.info("开始评估训练集...")

            # 使用测试循环评估训练集
            from mmengine.runner import TestLoop

            # 创建临时测试循环
            test_loop = TestLoop(
                runner=runner,
                dataloader=self._train_dataloader,
                evaluator=self._train_evaluator
            )

            # 运行评估
            train_metrics = test_loop.run()

            # 获取验证集mAP（从kwargs中的metrics参数）
            val_map = 0.0
            if 'metrics' in kwargs and kwargs['metrics']:
                metrics = kwargs['metrics']
                if isinstance(metrics, dict):
                    val_map = metrics.get('mAP_overall', 0.0)

            # 获取训练集mAP
            train_map = 0.0
            if isinstance(train_metrics, dict):
                train_map = train_metrics.get('mAP_overall', 0.0)

            # 打印对比结果
            runner.logger.info(f"训练集 mAP_overall: {train_map:.4f}")
            runner.logger.info(
                f"Epoch {runner.epoch} - 验证集 mAP: {val_map:.4f}, "
                f"训练集 mAP: {train_map:.4f}, "
                f"差异: {val_map - train_map:.4f}"
            )

        except Exception as e:
            runner.logger.error(f"训练集评估失败: {e}")
            import traceback
            runner.logger.error(traceback.format_exc())