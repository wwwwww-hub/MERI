from mmaction.datasets.transforms import PackActionInputs
from mmengine.registry import TRANSFORMS
import torch


@TRANSFORMS.register_module()
class PackWeakActionInputs(PackActionInputs):
    """
    在原有 PackActionInputs 基础上，额外打包 gt_scores 到 DataSample 的 field 'label_scores'
    """
    def __call__(self, results):
        # 先调用父类打包，可能返回 dict 或 ActionDataSample
        data = super().__call__(results)
        gt_scores = results.get('gt_scores', None)
        if gt_scores is None:
            return data
        scores_tensor = torch.tensor(gt_scores, dtype=torch.float32)
        # 如果返回的是 dict，里面包含 'data_samples'
        if isinstance(data, dict) and 'data_samples' in data:
            samples = data['data_samples']
            # samples 可能是 list，也可能是单个 DataSample
            if isinstance(samples, (list, tuple)):
                for sample in samples:
                    sample.set_field(scores_tensor, 'label_scores')
            else:
                # 单个 DataSample
                samples.set_field(scores_tensor, 'label_scores')
        else:
            # 直接是单个 DataSample
            data.set_field(scores_tensor, 'label_scores')
        return data
        scores_tensor = torch.tensor(gt_scores, dtype=torch.float32)
        # 如果是 dict，处理其中的 data_samples 列表
        if isinstance(data, dict) and 'data_samples' in data:
            for sample in data['data_samples']:
                sample.set_field(scores_tensor, 'label_scores')
        else:
            # 否则，假定 data 本身是单个 DataSample 对象
            data.set_field(scores_tensor, 'label_scores')
        return data
