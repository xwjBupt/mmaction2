_base_ = ["../../_base_/models/mvit_small.py", "../../_base_/default_runtime.py"]

model = dict(
    backbone=dict(
        arch="base",
        temporal_size=32,
        drop_path_rate=0.3,
        pretrained="/ai/mnt/code/mmaction2/configs/recognition/mvit/mvit-base-p244_32x3x1_kinetics400-rgb_20221021-f392cd2d#revised.pth",
    ),
    data_preprocessor=dict(
        type="ActionDataPreprocessor",
        mean=[160.75, 160.75, 160.75],
        std=[44.375, 44.375, 44.375],
        blending=dict(
            type="RandomBatchAugment",
            augments=[
                dict(type="MixupBlending", alpha=0.8, num_classes=4),
                dict(type="CutmixBlending", alpha=1, num_classes=4),
            ],
        ),
        format_shape="NCTHW",
    ),
    cls_head=dict(num_classes=4),
)

# dataset settings
# dataset_type = "VideoDataset"
# data_root = "data/kinetics400/videos_train"
# data_root_val = "data/kinetics400/videos_val"
# ann_file_train = "data/kinetics400/kinetics400_train_list_videos.txt"
# ann_file_val = "data/kinetics400/kinetics400_val_list_videos.txt"
# ann_file_test = "data/kinetics400/kinetics400_val_list_videos.txt"

dataset_type = "VideoDataset"
data_root_train = "/ai/mnt/data/erase_renamed_pair_relabel_RS/train"
data_root_val = "/ai/mnt/data/erase_renamed_pair_relabel_RS/val"
ann_file_train = "/ai/mnt/code/mmaction2/tools/data/AmTICIS/train.txt"
ann_file_val = "/ai/mnt/code/mmaction2/tools/data/AmTICIS/val.txt"
ann_file_test = "/ai/mnt/code/mmaction2/tools/data/AmTICIS/val.txt"

file_client_args = dict(io_backend="disk")
train_pipeline = [
    dict(type="DecordInit", **file_client_args),
    dict(type="SampleFrames", clip_len=32, frame_interval=3, num_clips=1),
    dict(type="DecordDecode"),
    dict(type="Resize", scale=(-1, 256)),
    dict(type="PytorchVideoWrapper", op="RandAugment", magnitude=7, num_layers=4),
    dict(type="RandomResizedCrop"),
    dict(type="Resize", scale=(224, 224), keep_ratio=False),
    dict(type="Flip", flip_ratio=0.5),
    dict(type="RandomErasing", erase_prob=0.25, mode="rand"),
    dict(type="FormatShape", input_format="NCTHW"),
    dict(type="PackActionInputs"),
]
val_pipeline = [
    dict(type="DecordInit", **file_client_args),
    dict(
        type="SampleFrames", clip_len=32, frame_interval=3, num_clips=1, test_mode=True
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
        type="SampleFrames", clip_len=32, frame_interval=3, num_clips=5, test_mode=True
    ),
    dict(type="DecordDecode"),
    dict(type="Resize", scale=(-1, 224)),
    dict(type="CenterCrop", crop_size=224),
    dict(type="FormatShape", input_format="NCTHW"),
    dict(type="PackActionInputs"),
]

repeat_sample = 2
train_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    collate_fn=dict(type="repeat_pseudo_collate"),
    dataset=dict(
        type="RepeatAugDataset",
        num_repeats=repeat_sample,
        sample_once=True,
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
    type="EpochBasedTrainLoop", max_epochs=200, val_begin=1, val_interval=4
)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

base_lr = 1.6e-3
optim_wrapper = dict(
    optimizer=dict(type="AdamW", lr=base_lr, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0),
    clip_grad=dict(max_norm=1, norm_type=2),
)

param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=0.01,
        by_epoch=True,
        begin=0,
        end=30,
        convert_to_iter_based=True,
    ),
    dict(
        type="CosineAnnealingLR",
        T_max=200,
        eta_min=base_lr / 100,
        by_epoch=True,
        begin=30,
        end=200,
        convert_to_iter_based=True,
    ),
]

default_hooks = dict(
    checkpoint=dict(interval=3, max_keep_ckpts=5, save_best="auto"),
    logger=dict(interval=100),
)

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (8 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=512 // repeat_sample)
