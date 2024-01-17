_base_ = ["./r2plus1d_r34_8xb8-8x8x1-180e_kinetics400-rgb.py"]

# dataset settings
dataset_type = "VideoDataset"
data_root_train = "/ai/mnt/data/erase_renamed_pair_relabel_RS/train"
data_root_val = "/ai/mnt/data/erase_renamed_pair_relabel_RS/val"
ann_file_train = "/ai/mnt/code/mmaction2/tools/data/AmTICIS/train_S.txt"
ann_file_val = "/ai/mnt/code/mmaction2/tools/data/AmTICIS/val_S.txt"
ann_file_test = "/ai/mnt/code/mmaction2/tools/data/AmTICIS/val_S.txt"

model = dict(
    cls_head=dict(num_classes=4),
    data_preprocessor=dict(
        mean=[160, 160, 160],
        std=[44, 44, 44],
    ),
)
file_client_args = dict(io_backend="disk")
train_pipeline = [
    dict(type="DecordInit", **file_client_args),
    dict(type="SampleFrames", clip_len=32, frame_interval=2, num_clips=1),
    dict(type="DecordDecode"),
    dict(type="Resize", scale=(-1, 256)),
    dict(type="RandomResizedCrop"),
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
train_cfg = dict(
    type="EpochBasedTrainLoop", max_epochs=900, val_begin=50, val_interval=20
)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

optim_wrapper = dict(
    optimizer=dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=1e-4),
    clip_grad=dict(max_norm=40, norm_type=2),
)

param_scheduler = [
    dict(
        type="CosineAnnealingLR",
        T_max=900,
        eta_min=0,
        by_epoch=True,
    )
]

default_hooks = dict(checkpoint=dict(interval=5, save_best="auto", max_keep_ckpts=3))
load_from = "/ai/mnt/code/mmaction2/configs/recognition/r2plus1d/r2plus1d_r34_8xb8-32x2x1-180e_kinetics400-rgb_20220812-4270588c.pth"
