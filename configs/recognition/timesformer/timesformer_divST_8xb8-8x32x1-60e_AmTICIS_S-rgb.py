_base_ = ["../../_base_/default_runtime.py"]

# model settings
model = dict(
    type="Recognizer3D",
    backbone=dict(
        type="TimeSformer",
        pretrained=None,  # noqa: E251  # noqa: E501
        num_frames=8,
        img_size=224,
        patch_size=16,
        embed_dims=768,
        in_channels=3,
        dropout_ratio=0.0,
        transformer_layers=None,
        attention_type="divided_space_time",
        norm_cfg=dict(type="LN", eps=1e-6),
    ),
    cls_head=dict(
        type="TimeSformerHead", num_classes=4, in_channels=768, average_clips="prob"
    ),
    data_preprocessor=dict(
        type="ActionDataPreprocessor",
        mean=[160, 160, 160],
        std=[44, 44, 44],
        format_shape="NCTHW",
    ),
)

# dataset settings
dataset_type = "VideoDataset"
data_root_train = "/ai/mnt/data/erase_renamed_pair_relabel_RS/train"
data_root_val = "/ai/mnt/data/erase_renamed_pair_relabel_RS/val"
ann_file_train = "/ai/mnt/code/mmaction2/tools/data/AmTICIS/train_S.txt"
ann_file_val = "/ai/mnt/code/mmaction2/tools/data/AmTICIS/val_S.txt"
ann_file_test = "/ai/mnt/code/mmaction2/tools/data/AmTICIS/val_S.txt"

file_client_args = dict(io_backend="disk")

train_pipeline = [
    dict(type="DecordInit", **file_client_args),
    dict(type="SampleFrames", clip_len=8, frame_interval=32, num_clips=1),
    dict(type="DecordDecode"),
    dict(type="RandomRescale", scale_range=(256, 320)),
    dict(type="RandomCrop", size=224),
    dict(type="Flip", flip_ratio=0.5),
    dict(type="FormatShape", input_format="NCTHW"),
    dict(type="PackActionInputs"),
]
val_pipeline = [
    dict(type="DecordInit", **file_client_args),
    dict(
        type="SampleFrames", clip_len=8, frame_interval=32, num_clips=1, test_mode=True
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
        type="SampleFrames", clip_len=8, frame_interval=32, num_clips=1, test_mode=True
    ),
    dict(type="DecordDecode"),
    dict(type="Resize", scale=(-1, 224)),
    dict(type="ThreeCrop", crop_size=224),
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

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=60, val_begin=1, val_interval=4)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

optim_wrapper = dict(
    optimizer=dict(
        type="SGD", lr=0.003, momentum=0.9, weight_decay=1e-4, nesterov=True
    ),
    paramwise_cfg=dict(
        custom_keys={
            ".backbone.cls_token": dict(decay_mult=0.0),
            ".backbone.pos_embed": dict(decay_mult=0.0),
            ".backbone.time_embed": dict(decay_mult=0.0),
        }
    ),
    clip_grad=dict(max_norm=40, norm_type=2),
)

param_scheduler = [
    dict(
        type="MultiStepLR",
        begin=0,
        end=60,
        by_epoch=True,
        milestones=[30, 50],
        gamma=0.1,
    )
]

default_hooks = dict(checkpoint=dict(interval=5, save_best="auto", max_keep_ckpts=3))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (8 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=64)
load_from = "/ai/mnt/code/mmaction2/configs/recognition/timesformer/timesformer_divST_8xb8-8x32x1-15e_kinetics400-rgb_20220815-a4d0d01f.pth"
