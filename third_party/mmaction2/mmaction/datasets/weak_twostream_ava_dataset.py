# mmaction/datasets/weak_twostream_ava_dataset.py

import os.path as osp
from collections import defaultdict
from typing import List, Optional, Union

import numpy as np
from mmengine.fileio import exists, list_from_file
from mmaction.registry import DATASETS
from mmaction.utils import ConfigType
from mmaction.datasets.weak_ava_dataset import WeakAVADataset


@DATASETS.register_module()
class WeakTwoStreamAVADataset(WeakAVADataset):
    """
    Two-Stream 弱标签 AVA 数据集。

    继承自 WeakAVADataset，因此自动支持：
        - exclude_file
        - label_file
        - proposal_file
        - fps
        - multilabel...
    并在每条数据里注入 flow_frame_dir。
    """

    def __init__(self,
                 ann_file: str,
                 pipeline: List[Union[ConfigType, callable]],
                 data_prefix: ConfigType = dict(img='', flow_img=''),
                 exclude_file: Optional[str] = None,
                 label_file: Optional[str] = None,
                 filename_tmpl: str = '{:06d}.jpg',
                 start_index: int = 1,
                 proposal_file: Optional[str] = None,
                 num_classes: int = 13,
                 custom_classes: Optional[List[int]] = None,
                 modality: str = 'RGB',
                 test_mode: bool = False,
                 fps: int = 30,
                 multilabel: bool = True,
                 **kwargs) -> None:
        # 拆出 RGB / Flow 根目录
        rgb_root  = data_prefix.get('img', '')
        flow_root = data_prefix.get('flow_img', rgb_root + '/flows')
        self.flow_root = flow_root

        # 调用父类（WeakAVADataset），只传 RGB 相关的 data_prefix
        super().__init__(
            ann_file=ann_file,
            pipeline=pipeline,
            data_prefix=dict(img=rgb_root),
            exclude_file=exclude_file,
            label_file=label_file,
            filename_tmpl=filename_tmpl,
            start_index=start_index,
            proposal_file=proposal_file,
            num_classes=num_classes,
            custom_classes=custom_classes,
            modality=modality,
            test_mode=test_mode,
            fps=fps,
            multilabel=multilabel,
            **kwargs)

    def load_data_list(self) -> List[dict]:
        """
        完整重写，不再调用 MMEngine 的 load(self.ann_file)，
        而是用我们自己的 list_from_file + parse_img_record。
        """
        exists(self.ann_file)
        raw_lines = list_from_file(self.ann_file)
        records = defaultdict(list)
        for line in raw_lines:
            parts = line.strip().split(',')
            if len(parts) < 9:
                continue
            vid, ts = parts[0], int(parts[1])
            box = np.array(list(map(float, parts[2:6])), dtype=np.float32)
            lbl = int(parts[6])
            pid = int(parts[7])
            score = float(parts[8])
            key = f'{vid},{ts:04d}'
            records[key].append({
                'entity_box': box,
                'label': lbl,
                'entity_id': pid,
                'score': score
            })

        data_list = []
        # 复用 WeakAVADataset.parse_img_record
        for key, recs in records.items():
            vid, ts_str = key.split(',')
            ts = int(ts_str)
            # 这里 self.parse_img_record 一定返回 4 个元素
            bboxes, labels, entity_ids, scores = self.parse_img_record(recs)

            ann = dict(
                gt_bboxes=bboxes,
                gt_labels=labels,
                entity_ids=entity_ids,
                gt_scores=scores
            )
            rgb_dir  = osp.join(self.data_prefix['img'], vid)
            flow_dir = osp.join(self.flow_root, vid)

            info = dict(
                frame_dir=rgb_dir,
                flow_frame_dir=flow_dir,
                video_id=vid,
                timestamp=ts,
                fps=self._FPS,
                filename_tmpl=self.filename_tmpl,
                ann=ann
            )
            # proposals 已在父类里加载
            if self.proposals is None:
                info.pop('proposals', None)
            data_list.append(info)
        return data_list
