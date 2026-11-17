_base_ = 'vit-base-p16_videomaev2-vit-g-dist-k710-pre_16x4x1_kinetics-400.py'

dataset_type = 'VideoDataset'
# Annotation lists are in data/DS1_screening/annotations; raw 30s videos are placed under data/DS1_screening (not bundled in release)
data_root = 'data/DS1_screening'
ann_root = 'data/DS1_screening/annotations'
ann_file_train = ann_root + '/train.txt'
ann_file_val = ann_root + '/val.txt'
ann_file_test = ann_file_val
classes = ('nursing', 'other')

model = dict(cls_head=dict(num_classes=2))
# Pretrained VideoMAEv2 backbone checkpoint for training from scratch (not bundled in release)
load_from = 'working_data/pretrained/vit-base-p16_videomae-k400-pre_16x4x1_kinetics-400.pth'

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05),
    clip_grad=dict(max_norm=5, norm_type=2))

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
    num_workers=4,
    dataset=dict(
        ann_file=ann_file_test,
        data_prefix=dict(video=data_root),
        metainfo=dict(classes=classes)))

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=50, val_begin=1, val_interval=1)
default_hooks = dict(
    checkpoint=dict(interval=3, max_keep_ckpts=5, save_best='auto'),
    logger=dict(interval=10))
auto_scale_lr = dict(enable=False, base_batch_size=16)
work_dir = './work_dirs/videomaev2-base_nursing'
