from mmengine.registry import HOOKS, LOOPS
from mmengine.hooks import Hook
import torch
import torch.nn.functional as F


def get_student_attention(f_map: torch.Tensor) -> torch.Tensor:
    """把特征图转成自注意力矩阵 (B, N, N).

    Args:
        f_map: (B, C, T, H, W)

    Returns:
        attn: (B, N, N) where N = T * H * W
    """
    B, C, T, H, W = f_map.shape
    # 展平空间+时间成 token 维度
    x = f_map.flatten(2).permute(0, 2, 1)  # (B, N, C)
    x = F.normalize(x, dim=-1)
    attn = x @ x.transpose(-2, -1)
    return attn.softmax(-1)  # (B, N, N)


@HOOKS.register_module()
class AttentionAlignHook(Hook):
    """Compare student & teacher attention each iter.

    支持两种 feature_map_for_distill 形式：
    1) Tensor: (B,C,T,H,W)，例如单路 SlowOnly / SlowFast slow 分支
    2) dict:   {'slow': (B,C,T,H,W), 'fast': (B,C,T,H,W), ...}
       会对每个分支单独算一个蒸馏损失，然后求平均。
    """

    def after_train_iter(self, runner, batch_idx, data_batch, outputs):

        # 1. 从 backbone 取缓存的特征图：
        #    可能是 Tensor，也可能是 dict(slow=..., fast=...)
        feat_map = getattr(runner.model.backbone, 'feature_map_for_distill', None)
        if feat_map is None:
            return

        # ----------------- 收集 teacher attention -----------------
        teacher_list = []
        for ds in data_batch['data_samples']:  # ← 逐样本读取
            teacher_list.append(getattr(ds, 'teacher_attn', None))

        # 整个 batch 都没有 teacher，就直接返回
        if all(t is None for t in teacher_list):
            return

        # 把有 teacher 的样本堆成一个 (B_valid, N, N) 的张量
        teacher_stack = []
        valid_idx = []
        for i, t in enumerate(teacher_list):
            if t is None:
                continue
            if t.dim() == 2:     # (N,N) → (1,N,N)
                t = t.unsqueeze(0)
            teacher_stack.append(t)
            valid_idx.append(i)

        if len(teacher_stack) == 0:
            return

        teacher = torch.stack(teacher_stack, dim=0)  # (B_valid, N, N)

        # ----------------- 定义一个对单个分支对齐的函数 -----------------
        def _branch_loss(f_map_branch: torch.Tensor) -> torch.Tensor:
            """给一个特征图分支计算对齐损失."""
            if f_map_branch is None:
                return None  # type: ignore

            # 学生注意力 (B, N, N)
            student = get_student_attention(f_map_branch)

            # 只取有 teacher 的样本
            if student.size(0) != len(teacher_list):
                # 防御性：batch 大小不一致时直接截断/对齐到较小的那个
                min_b = min(student.size(0), len(teacher_list))
                idx = valid_idx[:min_b]
                student = student[idx]
                teach = teacher[:min_b]
            else:
                student = student[valid_idx]
                teach = teacher

            # 具体的损失形式仍由模型里的 attn_loss 决定
            return runner.model.attn_loss(student, teach)

        # ----------------- 兼容 Tensor / dict 两种情况 -----------------
        if isinstance(feat_map, dict):
            # 多分支，例如 {'slow': feat_slow, 'fast': feat_fast}
            branch_losses = []
            for name, fmap_b in feat_map.items():
                if fmap_b is None:
                    continue
                loss_b = _branch_loss(fmap_b)
                if loss_b is not None:
                    branch_losses.append(loss_b)

            if len(branch_losses) == 0:
                return

            # 这里采用简单的平均，你也可以改成加权和（比如 slow+0.5*fast）
            loss = sum(branch_losses) / len(branch_losses)

        else:
            # 单分支 Tensor：保持原有行为
            loss = _branch_loss(feat_map)
            if loss is None:
                return

        # ----------------- 反向 + 日志 -----------------
        runner.optim_wrapper.backward(loss)
        runner.message_hub.update_scalar('loss_attn', loss.item())
