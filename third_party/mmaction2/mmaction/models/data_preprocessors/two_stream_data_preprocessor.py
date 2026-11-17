from mmaction.models.data_preprocessors.data_preprocessor import ActionDataPreprocessor
from mmengine.registry import MODELS
import torch

@MODELS.register_module()
class TwoStreamDataPreprocessor(ActionDataPreprocessor):
    """把 (RGB, Flow) 堆成 (N,C,T,H,W)×2，并在此处完成归一化。"""

    def __init__(self,
                 rgb_mean=(123.675, 116.28, 103.53),
                 rgb_std=(58.395, 57.12, 57.375),
                 flow_div255=True,   # 你的光流是 img_x/img_y 0~255 编码，保持 /255
                 **kwargs):
        super().__init__(**kwargs)
        # 注册到 buffer，便于与模型一起搬到 device
        self.register_buffer('rgb_mean',
                             torch.tensor(rgb_mean, dtype=torch.float32).view(1, 3, 1, 1, 1),
                             persistent=False)
        self.register_buffer('rgb_std',
                             torch.tensor(rgb_std, dtype=torch.float32).view(1, 3, 1, 1, 1),
                             persistent=False)
        self.flow_div255 = flow_div255

    def preprocess(self, inputs, data_samples=None, training: bool = False):
        # ------- 1) 拆解为张量 -------
        if (isinstance(inputs, (list, tuple)) and len(inputs) == 2 and
                isinstance(inputs[0], (list, tuple))):
            rgb, flow = torch.stack(inputs[0], 0), torch.stack(inputs[1], 0)
        elif (isinstance(inputs, (list, tuple)) and len(inputs) == 2 and
              torch.is_tensor(inputs[0])):
            rgb, flow = inputs
        else:
            raise TypeError(f'Unexpected inputs structure: {type(inputs)}')

        # ------- 2) 转 float32 -------
        rgb = rgb.to(torch.float32)
        flow = flow.to(torch.float32)

        # ------- 3) 归一化 -------
        # RGB: (x-mean)/std   （Kinetics 习惯）
        rgb = rgb.div_(255.0)
        rgb = (rgb * 255.0 - self.rgb_mean) / self.rgb_std

        # Flow: 维持 0~1（或改成减均值/除方差取决于你的光流编码）
        if self.flow_div255:
            flow = flow.div_(255.0)

        return (rgb, flow), data_samples
