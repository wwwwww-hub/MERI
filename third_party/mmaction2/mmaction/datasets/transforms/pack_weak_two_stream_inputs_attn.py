from __future__ import annotations
from typing import Dict, Any
import torch
from mmengine.registry import TRANSFORMS
from mmaction.structures import ActionDataSample
from mmaction.datasets.transforms.pack_weak_two_stream_inputs import PackWeakTwoStreamInputs

@TRANSFORMS.register_module()
class PackWeakTwoStreamInputsAttn(PackWeakTwoStreamInputs):
    """Pack (RGB, Flow) inputs and attach optional teacher attention to data_samples."""
    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        packed = super().__call__(results)
        ds: ActionDataSample = packed['data_samples']
        attn = results.get('attn', None)
        if attn is not None:
            ds.teacher_attn = torch.tensor(attn, dtype=torch.float32) if not torch.is_tensor(attn) else attn
        else:
            ds.teacher_attn = None
        return packed
