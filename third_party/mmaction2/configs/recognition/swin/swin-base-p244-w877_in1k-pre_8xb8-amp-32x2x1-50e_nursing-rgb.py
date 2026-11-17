_base_ = 'swin-base-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb.py'

dataset_type = 'VideoDataset'
# Annotation lists are in data/DS1_screening/annotations; raw 30s videos are placed under data/DS1_screening (not bundled in release)
data_root = 'data/DS1_screening'
ann_root = 'data/DS1_screening/annotations'
ann_file_train = ann_root + '/train.txt'
ann_file_val = ann_root + '/val.txt'
ann_file_test = ann_file_val
classes = ('nursing', 'other')

model = dict(
    # Pretrained Swin-Base backbone weights for training from scratch (not bundled in release)
    backbone=dict(pretrained='working_data/pretrained/swin_base_patch4_window7_224.pth'),
    cls_head=dict(num_classes=2))

train_dataloader = dict(
    batch_size=4,
    num_workers=1,
    dataset=dict(
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root),
        metainfo=dict(classes=classes)))
val_dataloader = dict(
    batch_size=4,
    num_workers=1,
    dataset=dict(
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root),
        metainfo=dict(classes=classes)))
test_dataloader = dict(
    batch_size=2,
    num_workers=1,
    dataset=dict(
        ann_file=ann_file_test,
        data_prefix=dict(video=data_root),
        metainfo=dict(classes=classes)))

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=50, val_begin=1, val_interval=1)
default_hooks = dict(
    checkpoint=dict(interval=3, max_keep_ckpts=5, save_best='auto'),
    logger=dict(interval=10))
auto_scale_lr = dict(enable=False, base_batch_size=32)
work_dir = './work_dirs/swin-base_nursing'
