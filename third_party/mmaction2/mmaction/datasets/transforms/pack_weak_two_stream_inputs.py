import numpy as np
import torch
from mmaction.datasets.transforms import PackActionInputs
from mmengine.registry import TRANSFORMS


@TRANSFORMS.register_module()
class PackWeakTwoStreamInputs(PackActionInputs):
    """把 RGB + Flow + 软标签打包成 model 输入."""

    def __call__(self, results):
        # ---- 1. 先保证 imgs / imgs_flow 都是 ndarray ----
        if isinstance(results['imgs'], list):
            results['imgs'] = np.stack(results['imgs'])        # (T,H,W,C)
        if isinstance(results['imgs_flow'], list):
            results['imgs_flow'] = np.stack(results['imgs_flow'])

        # ---- 2. 利用 PackActionInputs 打包 RGB ----
        packed = super().__call__(results)

        # ---- 3. 再把 flow 添加进来 ----
        flow_tensor = torch.from_numpy(
            results['imgs_flow']).permute(3, 0, 1, 2)  # (C,T,H,W)
        # inputs 是 list，索引 0 = RGB，1 = Flow
        packed['inputs'] = [packed['inputs'], flow_tensor]

        return packed
