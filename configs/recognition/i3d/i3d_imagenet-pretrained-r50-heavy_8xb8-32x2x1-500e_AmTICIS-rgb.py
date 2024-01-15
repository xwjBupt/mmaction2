_base_ = [
    "../../_base_/models/i3d_r50.py",
    "../../_base_/default_runtime.py",
]

dataset_type = "VideoDataset"
data_root_train = "/ai/mnt/data/erase_renamed_pair_relabel_RS/train"
data_root_val = "/ai/mnt/data/erase_renamed_pair_relabel_RS/val"
ann_file_train = "/ai/mnt/code/mmaction2/tools/data/AmTICIS/train.txt"
ann_file_val = "/ai/mnt/code/mmaction2/tools/data/AmTICIS/val.txt"
ann_file_test = "/ai/mnt/code/mmaction2/tools/data/AmTICIS/val.txt"

# model settings
model = dict(
    backbone=dict(
        inflate=(1, 1, 1, 1),
        conv1_stride_t=1,
        pool1_stride_t=1,
        with_pool2=True,
        pretrained2d=None,
        pretrained=None,
    ),
    data_preprocessor=dict(
        mean=[160, 160, 160],
        std=[44, 44, 44],
    ),
    cls_head=dict(num_classes=4),
)


file_client_args = dict(io_backend="disk")
train_pipeline = [
    dict(type="DecordInit", **file_client_args),
    dict(type="SampleFrames", clip_len=32, frame_interval=2, num_clips=1),
    dict(type="DecordDecode"),
    dict(type="Resize", scale=(-1, 256)),
    dict(
        type="MultiScaleCrop",
        input_size=224,
        scales=(1, 0.8),
        random_crop=False,
        max_wh_scale_gap=0,
    ),
    dict(type="Resize", scale=(224, 224), keep_ratio=False),
    dict(type="Flip", flip_ratio=0.5),
    dict(type="FormatShape", input_format="NCTHW"),
    dict(type="PackActionInputs"),
]
val_pipeline = [
    dict(type="DecordInit", **file_client_args),
    dict(
        type="SampleFrames", clip_len=32, frame_interval=2, num_clips=1, test_mode=True
    ),
    dict(type="DecordDecode"),
    dict(type="Resize", scale=(-1, 256)),
    dict(type="CenterCrop", crop_size=224),
    dict(type="FormatShape", input_format="NCTHW"),
    dict(type="PackActionInputs"),
]
test_pipeline = [
    dict(type="DecordInit", **file_client_args),
    dict(
        type="SampleFrames", clip_len=32, frame_interval=2, num_clips=10, test_mode=True
    ),
    dict(type="DecordDecode"),
    dict(type="Resize", scale=(-1, 256)),
    dict(type="ThreeCrop", crop_size=256),
    dict(type="FormatShape", input_format="NCTHW"),
    dict(type="PackActionInputs"),
]

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root_train),
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root_val),
        pipeline=val_pipeline,
        test_mode=True,
    ),
)
test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(video=data_root_val),
        pipeline=test_pipeline,
        test_mode=True,
    ),
)

val_evaluator = dict(type="AccMetric")
test_evaluator = val_evaluator

train_cfg = dict(
    type="EpochBasedTrainLoop", max_epochs=500, val_begin=50, val_interval=5
)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

param_scheduler = [
    dict(
        type="MultiStepLR",
        begin=0,
        end=500,
        by_epoch=True,
        milestones=[200, 400],
        gamma=0.1,
    )
]

optim_wrapper = dict(
    optimizer=dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=40, norm_type=2),
)


default_hooks = dict(checkpoint=dict(interval=5, max_keep_ckpts=5))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (8 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=64)
load_from = "/ai/mnt/code/mmaction2/configs/recognition/i3d/i3d_imagenet-pretrained-r50-heavy_8xb8-32x2x1-100e_kinetics400-rgb_20220812-ed501b31.pth"
