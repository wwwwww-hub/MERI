import os, sys
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)
import argparse
import logging
import os.path as osp

os.environ['MMENGINE_FORKIMPORT'] = '1'
logging.getLogger().setLevel(logging.DEBUG)  # 全局 DEBUG，Traceback 更完整
import torch
from copy import deepcopy

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from mmengine.evaluator import Evaluator
from mmengine.hooks import Hook

from mmaction.registry import DATASETS
from mmaction.utils import register_all_modules


# class SimpleTrainEvalHook(Hook):
#     """最简化的训练集评估钩子"""
#
#     def __init__(self):
#         super().__init__()
#         self.priority = 80  # 在验证后执行
#         self._train_eval_dataloader = None
#         self._train_evaluator = None
#         self._initialized = False
#         self._evaluating_train = False  # 防止递归调用的标志
#
#     def _lazy_init(self, runner):
#         """延迟初始化"""
#         if self._initialized:
#             return
#
#         try:
#             cfg = runner.cfg
#
#             # 构建训练集评估数据加载器
#             if hasattr(cfg, 'train_eval_dataloader'):
#                 self._train_eval_dataloader = runner.build_dataloader(cfg.train_eval_dataloader)
#
#             # 构建训练集评估器
#             if hasattr(cfg, 'train_evaluator'):
#                 self._train_evaluator = Evaluator([cfg.train_evaluator])
#
#             self._initialized = True
#             runner.logger.info("训练集评估器初始化成功")
#
#         except Exception as e:
#             runner.logger.warning(f"训练集评估器初始化失败: {e}")
#
#     def after_val_epoch(self, runner, **kwargs):
#         """在验证后评估训练集"""
#         # 防止递归调用
#         if self._evaluating_train:
#             return
#
#         try:
#             # 延迟初始化
#             self._lazy_init(runner)
#
#             if not (self._train_eval_dataloader and self._train_evaluator):
#                 return
#
#             # 设置标志防止递归
#             self._evaluating_train = True
#
#             runner.logger.info("开始评估训练集...")
#             runner.logger.info(f"训练集总样本数: {len(self._train_eval_dataloader.dataset)}")
#
#             # 临时替换验证循环的数据加载器和评估器
#             original_val_dataloader = runner.val_loop.dataloader
#             original_val_evaluator = runner.val_loop.evaluator
#
#             # 用训练集的数据加载器和评估器替换
#             runner.val_loop.dataloader = self._train_eval_dataloader
#             runner.val_loop.evaluator = self._train_evaluator
#
#             # 重要：临时禁用钩子以防止递归
#             original_hooks = runner._hooks
#             runner._hooks = [hook for hook in runner._hooks if not isinstance(hook, SimpleTrainEvalHook)]
#
#             try:
#                 # 运行验证循环（实际上是在训练集上）
#                 train_metrics = runner.val_loop.run()
#             finally:
#                 # 恢复钩子
#                 runner._hooks = original_hooks
#
#             # 恢复原始的验证循环配置
#             runner.val_loop.dataloader = original_val_dataloader
#             runner.val_loop.evaluator = original_val_evaluator
#
#             # 获取验证集结果
#             val_map = 0.0
#             if 'metrics' in kwargs:
#                 metrics = kwargs['metrics']
#                 if isinstance(metrics, dict):
#                     val_map = metrics.get('mAP_overall', 0.0)
#
#             # 获取训练集mAP（修复提取逻辑）
#             train_map = 0.0
#             if isinstance(train_metrics, dict):
#                 # 尝试多种可能的键名
#                 train_map = train_metrics.get('mAP_overall',
#                                               train_metrics.get('mAP/overall',
#                                                                 train_metrics.get('overall_mAP', 0.0)))
#
#             # 如果还是0，尝试从日志中提取（备用方案）
#             if train_map == 0.0:
#                 runner.logger.warning("无法从train_metrics中提取mAP，跳过数值对比")
#                 runner.logger.info("训练集评估完成")
#                 runner.logger.info(f"训练集评估结果: {train_metrics}")
#             else:
#                 # 打印对比结果
#                 runner.logger.info(f"训练集 mAP_overall: {train_map:.4f}")
#                 runner.logger.info(
#                     f"Epoch {runner.epoch} - 验证集 mAP: {val_map:.4f}, "
#                     f"训练集 mAP: {train_map:.4f}, "
#                     f"差异: {val_map - train_map:.4f}"
#                 )
#
#         except Exception as e:
#             runner.logger.error(f"训练集评估失败: {e}")
#             import traceback
#             runner.logger.error(traceback.format_exc())
#
#             # 确保恢复原始配置
#             try:
#                 runner.val_loop.dataloader = original_val_dataloader
#                 runner.val_loop.evaluator = original_val_evaluator
#                 runner._hooks = original_hooks
#             except:
#                 pass
#
#         finally:
#             # 确保清除标志
#             self._evaluating_train = False


def parse_args():
    parser = argparse.ArgumentParser(description='Train a recognizer')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
             'specify, try to auto resume from the latest checkpoint '
             'in the work directory.')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    # register all modules in mmaction into the registries
    register_all_modules()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # enable automatically scaling LR
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    # resume is determined in this priority: resume from > auto_resume
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    # build the runner from config
    if 'runner_type' not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)

    # 注册自定义钩子
    # train_eval_hook = SimpleTrainEvalHook()
    # runner.register_hook(train_eval_hook)

    runner.train()


if __name__ == '__main__':
    main()