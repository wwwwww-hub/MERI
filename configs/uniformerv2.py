ann_file_test = 'data/DS1_screening/annotations/val.txt'
ann_file_train = 'data/DS1_screening/annotations/train.txt'
ann_file_val = 'data/DS1_screening/annotations/val.txt'
ann_root = 'data/DS1_screening/annotations'
auto_scale_lr = dict(base_batch_size=4, enable=False)
base_lr = 1e-05
classes = (
    'nursing',
    'other',
)
data_root = 'data/DS1_screening'
dataset_type = 'VideoDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=3, max_keep_ckpts=5, save_best='auto', type='CheckpointHook'),
    logger=dict(ignore_last=False, interval=20, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    runtime_info=dict(type='RuntimeInfoHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'),
    timer=dict(type='IterTimerHook'))
default_scope = 'mmaction'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
file_client_args = dict(io_backend='disk')
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=20)
optim_wrapper = dict(
    clip_grad=dict(max_norm=20, norm_type=2),
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=1e-05, type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(bias_decay_mult=0.0, norm_decay_mult=0.0),
    type='AmpOptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=5,
        start_factor=0.1,
        type='LinearLR'),
    dict(
        T_max=45,
        begin=5,
        by_epoch=True,
        convert_to_iter_based=True,
        end=50,
        eta_min_ratio=0.1,
        type='CosineAnnealingLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='data/DS1_screening/annotations/val.txt',
        data_prefix=dict(video='data/DS1_screening'),
        metainfo=dict(classes=(
            'nursing',
            'other',
        )),
        pipeline=[
            dict(io_backend='disk', type='DecordInit'),
            dict(
                clip_len=8, num_clips=4, test_mode=True, type='UniformSample'),
            dict(type='DecordDecode'),
            dict(scale=(
                -1,
                224,
            ), type='Resize'),
            dict(crop_size=224, type='ThreeCrop'),
            dict(input_format='NCTHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        test_mode=True,
        type='VideoDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(type='AccMetric')
test_pipeline = [
    dict(io_backend='disk', type='DecordInit'),
    dict(clip_len=8, num_clips=4, test_mode=True, type='UniformSample'),
    dict(type='DecordDecode'),
    dict(scale=(
        -1,
        224,
    ), type='Resize'),
    dict(crop_size=224, type='ThreeCrop'),
    dict(input_format='NCTHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]
train_cfg = dict(
    max_epochs=50, type='EpochBasedTrainLoop', val_begin=1, val_interval=1)
train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file='data/DS1_screening/annotations/train.txt',
        data_prefix=dict(video='data/DS1_screening'),
        metainfo=dict(classes=(
            'nursing',
            'other',
        )),
        pipeline=[
            dict(io_backend='disk', type='DecordInit'),
            dict(clip_len=8, num_clips=1, type='UniformSample'),
            dict(type='DecordDecode'),
            dict(scale=(
                -1,
                256,
            ), type='Resize'),
            dict(type='RandomResizedCrop'),
            dict(keep_ratio=False, scale=(
                224,
                224,
            ), type='Resize'),
            dict(flip_ratio=0.5, type='Flip'),
            dict(input_format='NCTHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        type='VideoDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(io_backend='disk', type='DecordInit'),
    dict(clip_len=8, num_clips=1, type='UniformSample'),
    dict(type='DecordDecode'),
    dict(scale=(
        -1,
        256,
    ), type='Resize'),
    dict(type='RandomResizedCrop'),
    dict(keep_ratio=False, scale=(
        224,
        224,
    ), type='Resize'),
    dict(flip_ratio=0.5, type='Flip'),
    dict(input_format='NCTHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file='data/DS1_screening/annotations/val.txt',
        data_prefix=dict(video='data/DS1_screening'),
        metainfo=dict(classes=(
            'nursing',
            'other',
        )),
        pipeline=[
            dict(io_backend='disk', type='DecordInit'),
            dict(
                clip_len=8, num_clips=1, test_mode=True, type='UniformSample'),
            dict(type='DecordDecode'),
            dict(scale=(
                -1,
                224,
            ), type='Resize'),
            dict(crop_size=224, type='CenterCrop'),
            dict(input_format='NCTHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        test_mode=True,
        type='VideoDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(type='AccMetric')
val_pipeline = [
    dict(io_backend='disk', type='DecordInit'),
    dict(clip_len=8, num_clips=1, test_mode=True, type='UniformSample'),
    dict(type='DecordDecode'),
    dict(scale=(
        -1,
        224,
    ), type='Resize'),
    dict(crop_size=224, type='CenterCrop'),
    dict(input_format='NCTHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='ActionVisualizer', vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = 'reproduced/screening'
