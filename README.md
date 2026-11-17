# MERI

[简体中文](README.zh-CN.md)


## Download the data and models

| Archive | Baidu Netdisk | Access code | Where to extract |
| --- | --- | --- | --- |
| `data.zip` | [Download data](https://pan.baidu.com/s/1_s3fWbbEtwAIiJcn7p3JKw?pwd=bd5i) | `bd5i` | Create `data/` at the repository root and extract the archive into it. |
| `models.zip` | [Download models](https://pan.baidu.com/s/1U7qG_xA7aTukvK1AAjpG8w?pwd=cucd) | `cucd` | Extract to the repository root. The archive contains the `models/` directory. |

## Repository and archive contents

| Path | Contents |
| --- | --- |
| `data/DS1_screening/` | Nursing-recognition videos, labels, and split definitions |
| `data/DS2_segmentation/` | Segmentation images, human annotations, and pseudo-labels |
| `data/DS3_tracking/` | 30 training clips with SAM3 instance masks and 15 test clips with manual reference masks |
| `data/DS4_intervals/` | 101 source clips, numerical features, track signals, interval annotations, and five video-grouped folds |
| `data/MERI_15pens/` | Numerical timelines for 15 pens and fixed calibration parameters |
| `models/` | Screening, segmentation, tracking, and interval models, including SAM3 weights and comparison models |
| `configs/`, `scripts/` | Model configurations, training, evaluation, feature extraction, and MERI computation |
| `review_tools/` | Scripts to reproduce the interval evaluation and final MERI scores |
| `ultralytics/`, `third_party/` | Modified Ultralytics 8.4.17 with mask-aware association tracking, plus bundled third-party code and licenses |

## Reproduce the numerical results

```bash
python -m pip install -r requirements-meri.txt
python review_tools/reproduce_interval_result.py
python review_tools/reproduce_meri.py
```

The scripts write the interval evaluation to `reproduced/intervals/evaluation/` and the pen-level MERI scores to `reproduced/meri/pen_scores.csv`.

## Licenses

See [LICENSE](LICENSE) for the source code license and [DATA_LICENSE.md](DATA_LICENSE.md) for the non-commercial data usage terms. Third-party code and model weights are subject to their respective terms, including the [SAM License](third_party/sam3/LICENSE) for SAM3.
