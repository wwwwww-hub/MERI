import os.path as osp
from collections import defaultdict
from typing import List

import numpy as np
from mmengine.fileio import list_from_file
from mmaction.registry import DATASETS
from mmaction.datasets.ava_dataset import AVADataset  # 官方 AVADataset:contentReference[oaicite:2]{index=2}


@DATASETS.register_module()
class WeakAVADataset(AVADataset):
    """
    AVA 数据集的弱标签版本，在官方 AVADataset 基础上注入软标签分数 gt_scores。

    原始 CSV 格式：
        video_id, timestamp, x1, y1, x2, y2, action_id, person_id, score

    最终每条 data_info 会有：
        - gt_bboxes  (N,4)
        - gt_labels  (N,num_classes)
        - entity_ids (N,)
        - gt_scores  (N,num_classes)  ← 新增项，弱标签分数
        - proposals, fps, filename_tmpl, shot_info… 等 100% 复用父类
    """

    def load_data_list(self) -> list:
        # 1. 先调用父类，获取包含 'ann'、'proposals'、'fps'、'filename_tmpl'、
        #    'timestamp_start'/'timestamp_end'、'shot_info' 等完整字段的 data_list
        data_list = super().load_data_list()

        # 2. 读取弱标签 CSV，把第 9 列 score 收集到字典
        #    key 格式与父类相同："{video_id},{timestamp:04d}"
        raw = list_from_file(self.ann_file)
        score_dict = defaultdict(list)  # img_key -> List[ (label, score) ]
        for line in raw:
            parts = line.strip().split(',')
            if len(parts) < 9:
                continue
            vid, ts, *_, lbl, _, sc = parts
            key = f'{vid},{int(ts):04d}'
            score_dict[key].append((int(lbl), float(sc)))

        # 3. 遍历 data_list，将软标签注入到 ann['gt_scores']
        for data_info in data_list:
            key = data_info['img_key']
            recs = score_dict.get(key, [])

            # ann 阶段：已经有 gt_bboxes, gt_labels, entity_ids
            ann = data_info['ann']
            bboxes = ann['gt_bboxes']
            num_boxes = bboxes.shape[0]
            num_classes = self.num_classes

            # 默认全 1.0（硬标签置信度），再对 recs 中的 (label,score) 覆盖
            scores = np.ones((num_boxes, num_classes), dtype=np.float32)
            for lbl, sc in recs:
                # 找到所有匹配 lbl 的行，并赋值
                # 通常 proposals 与 gt_bboxes 一一对应，行序一致
                # 这里简单按行号匹配：gt_labels 行中哪一列为 1，对应 box
                # 但 gt_labels shape (num_boxes,num_classes)
                # 我们认为 recs 中的标签正好出现在第 i 行第 lbl 列
                # 因此直接 scores[i, lbl] = sc
                for i in range(num_boxes):
                    if ann['gt_labels'][i, lbl] == 1.0:
                        scores[i, lbl] = sc

            # 最后，替换 ann 中的 gt_scores
            ann['gt_scores'] = scores
            data_info['ann'] = ann

        return data_list
