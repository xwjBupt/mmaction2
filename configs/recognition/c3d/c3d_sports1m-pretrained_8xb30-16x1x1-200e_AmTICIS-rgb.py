_base_ = [
    "../../_base_/models/c3d_sports1m_pretrained.py",
    "../../_base_/default_runtime.py",
]

# dataset settings
dataset_type = "VideoDataset"
data_root_train = "/ai/mnt/data/erase_renamed_pair_relabel_RS/train"
data_root_val = "/ai/mnt/data/erase_renamed_pair_relabel_RS/val"
ann_file_train = f"/ai/mnt/code/mmaction2/tools/data/AmTICIS/train.txt"
ann_file_val = f"/ai/mnt/code/mmaction2/tools/data/AmTICIS/val.txt"
ann_file_test = f"/ai/mnt/code/mmaction2/tools/data/AmTICIS/val.txt"

model = dict(
    backbone=dict(
        pretrained="/ai/mnt/code/mmaction2/configs/recognition/c3d/c3d_sports1m-pretrained_8xb30-16x1x1-45e_ucf101-rgb_20220811-31723200.pth"
    ),
    cls_head=dict(num_classes=4),
    data_preprocessor=dict(
        mean=[160, 160, 160],
        std=[44, 44, 44],
    ),
)
file_client_args = dict(io_backend="disk")
train_pipeline = [
    dict(type="DecordInit", **file_client_args),
    dict(type="SampleFrames", clip_len=16, frame_interval=1, num_clips=1),
    dict(type="DecordDecode"),
    dict(type="Resize", scale=(-1, 128)),
    dict(type="RandomCrop", size=112),
    dict(type="Flip", flip_ratio=0.5),
    dict(type="FormatShape", input_format="NCTHW"),
    dict(type="PackActionInputs"),
]
val_pipeline = [
    dict(type="DecordInit", **file_client_args),
    dict(
        type="SampleFrames", clip_len=16, frame_interval=1, num_clips=1, test_mode=True
    ),
    dict(type="DecordDecode"),
    dict(type="Resize", scale=(-1, 128)),
    dict(type="CenterCrop", crop_size=112),
    dict(type="FormatShape", input_format="NCTHW"),
    dict(type="PackActionInputs"),
]
test_pipeline = [
    dict(type="DecordInit", **file_client_args),
    dict(
        type="SampleFrames", clip_len=16, frame_interval=1, num_clips=10, test_mode=True
    ),
    dict(type="DecordDecode"),
    dict(type="Resize", scale=(-1, 128)),
    dict(type="CenterCrop", crop_size=112),
    dict(type="FormatShape", input_format="NCTHW"),
    dict(type="PackActionInputs"),
]

train_dataloader = dict(
    batch_size=128,
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
    batch_size=128,
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
    type="EpochBasedTrainLoop", max_epochs=200, val_begin=1, val_interval=10
)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

param_scheduler = [
    dict(
        type="MultiStepLR",
        begin=0,
        end=200,
        by_epoch=True,
        milestones=[80, 140, 180],
        gamma=0.1,
    )
]

optim_wrapper = dict(
    optimizer=dict(type="SGD", lr=0.001, momentum=0.9, weight_decay=0.0005),
    clip_grad=dict(max_norm=40, norm_type=2),
)

default_hooks = dict(checkpoint=dict(interval=5, save_best="auto", max_keep_ckpts=3))


# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (30 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=240)
