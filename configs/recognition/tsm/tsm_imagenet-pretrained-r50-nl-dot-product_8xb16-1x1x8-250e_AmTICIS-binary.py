_base_ = ["../../_base_/models/tsm_r50.py", "../../_base_/default_runtime.py"]

# dataset settings
dataset_type = "VideoDataset"
data_root_train = "/ai/mnt/data/erase_renamed_pair_relabel_RS/train"
data_root_val = "/ai/mnt/data/erase_renamed_pair_relabel_RS/val"
ann_file_train = "/ai/mnt/code/mmaction2/tools/data/AmTICIS/train_binary.txt"
ann_file_val = "/ai/mnt/code/mmaction2/tools/data/AmTICIS/val_binary.txt"
ann_file_test = "/ai/mnt/code/mmaction2/tools/data/AmTICIS/val_binary.txt"


file_client_args = dict(io_backend="disk")

train_pipeline = [
    dict(type="DecordInit", **file_client_args),
    dict(type="SampleFrames", clip_len=1, frame_interval=1, num_clips=8),
    dict(type="DecordDecode"),
    dict(type="Resize", scale=(-1, 256)),
    dict(
        type="MultiScaleCrop",
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1,
        num_fixed_crops=13,
    ),
    dict(type="Resize", scale=(224, 224), keep_ratio=False),
    dict(type="Flip", flip_ratio=0.5),
    dict(type="FormatShape", input_format="NCHW"),
    dict(type="PackActionInputs"),
]
val_pipeline = [
    dict(type="DecordInit", **file_client_args),
    dict(
        type="SampleFrames", clip_len=1, frame_interval=1, num_clips=8, test_mode=True
    ),
    dict(type="DecordDecode"),
    dict(type="Resize", scale=(-1, 256)),
    dict(type="CenterCrop", crop_size=224),
    dict(type="FormatShape", input_format="NCHW"),
    dict(type="PackActionInputs"),
]
test_pipeline = [
    dict(type="DecordInit", **file_client_args),
    dict(
        type="SampleFrames", clip_len=1, frame_interval=1, num_clips=8, test_mode=True
    ),
    dict(type="DecordDecode"),
    dict(type="Resize", scale=(-1, 256)),
    dict(type="TenCrop", crop_size=224),
    dict(type="FormatShape", input_format="NCHW"),
    dict(type="PackActionInputs"),
]

train_dataloader = dict(
    batch_size=16,
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
    batch_size=16,
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
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root_val),
        pipeline=test_pipeline,
        test_mode=True,
    ),
)

val_evaluator = dict(type="AccMetric")
test_evaluator = val_evaluator

default_hooks = dict(checkpoint=dict(interval=3, max_keep_ckpts=3))

train_cfg = dict(
    type="EpochBasedTrainLoop", max_epochs=250, val_begin=10, val_interval=5
)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

param_scheduler = [
    dict(type="LinearLR", start_factor=0.1, by_epoch=True, begin=0, end=5),
    dict(
        type="MultiStepLR",
        begin=0,
        end=250,
        by_epoch=True,
        milestones=[125, 225],
        gamma=0.1,
    ),
]

optim_wrapper = dict(
    constructor="TSMOptimWrapperConstructor",
    paramwise_cfg=dict(fc_lr5=True),
    optimizer=dict(type="SGD", lr=0.02, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=20, norm_type=2),
)

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (16 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=128)


# model settings
model = dict(
    backbone=dict(
        non_local=((0, 0, 0), (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), (0, 0, 0)),
        non_local_cfg=dict(
            sub_sample=True,
            use_scale=False,
            norm_cfg=dict(type="BN3d", requires_grad=True),
            mode="dot_product",
        ),
    ),
    cls_head=dict(num_classes=2),
)
load_from = "/ai/mnt/code/mmaction2/configs/recognition/tsm/tsm_imagenet-pretrained-r50-nl-dot-product_8xb16-1x1x8-50e_kinetics400-rgb_20220831-108bfde5.pth"
