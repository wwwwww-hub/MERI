# MERI

[English](README.md)


## 下载数据和模型

| 压缩包 | 百度网盘 | 提取码 | 解压位置 |
| --- | --- | --- | --- |
| `data.zip` | [下载数据](https://pan.baidu.com/s/1_s3fWbbEtwAIiJcn7p3JKw?pwd=bd5i) | `bd5i` | 在仓库根目录新建 `data/`，将压缩包内容解压到该目录。 |
| `models.zip` | [下载模型](https://pan.baidu.com/s/1U7qG_xA7aTukvK1AAjpG8w?pwd=cucd) | `cucd` | 直接解压到仓库根目录。压缩包自带 `models/` 目录。 |


## 仓库和压缩包内容

| 路径 | 内容 |
| --- | --- |
| `data/DS1_screening/` | 哺乳识别视频、标签及数据划分 |
| `data/DS2_segmentation/` | 分割图像、人工标注和伪标签 |
| `data/DS3_tracking/` | 30 段带 SAM3 实例掩膜的训练视频，15 段带人工参考掩膜的测试视频 |
| `data/DS4_intervals/` | 101 段源视频、数值特征、逐轨迹信号、区间标注及按视频分组的五折划分 |
| `data/MERI_15pens/` | 15 个栏位的数值时间序列及固定校准参数 |
| `models/` | 哺乳识别、分割、跟踪和区间估计模型，包含 SAM3 权重和对照模型 |
| `configs/`、`scripts/` | 模型配置、训练、评估、特征提取和 MERI 计算 |
| `review_tools/` | 复现哺乳区间评估和最终 MERI 得分的脚本 |
| `ultralytics/`、`third_party/` | 包含掩膜关联跟踪实现的 Ultralytics 8.4.17 修改版，以及随附的第三方代码和许可证 |

## 复现数值结果

```bash
python -m pip install -r requirements-meri.txt
python review_tools/reproduce_interval_result.py
python review_tools/reproduce_meri.py
```

## 许可证

源码许可见 [LICENSE](LICENSE)，数据的非商业使用条款见 [DATA_LICENSE.md](DATA_LICENSE.md)。第三方代码和模型权重适用各自的许可，其中 SAM3 适用 [SAM License](third_party/sam3/LICENSE)。
