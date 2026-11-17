from __future__ import annotations
from typing import Dict, Any
import torch
from mmengine.registry import TRANSFORMS
from mmaction.datasets.transforms import PackWeakActionInputs
from mmaction.structures import ActionDataSample

@TRANSFORMS.register_module()
class PackWeakActionInputsAttn(PackWeakActionInputs):
    """在原 PackWeakActionInputs 基础上，附加 teacher_attn (可缺失)。"""

    def transform(self, results: Dict[str, Any]) -> Dict[str, Any]:
        packed = super().transform(results)
        data_sample: ActionDataSample = packed['data_samples']

        attn = results.get('attn', None)
        if attn is not None:
            data_sample.teacher_attn = (
                torch.tensor(attn, dtype=torch.float32)
                if not torch.is_tensor(attn) else attn)
        else:
            data_sample.teacher_attn = None
        return packed
