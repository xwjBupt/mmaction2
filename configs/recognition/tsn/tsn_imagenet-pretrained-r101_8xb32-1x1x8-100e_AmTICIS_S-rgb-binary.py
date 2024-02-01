_base_ = ["tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py"]

# dataset settings
dataset_type = "VideoDataset"
data_root_train = "/ai/mnt/data/erase_renamed_pair_relabel_RS/train"
data_root_val = "/ai/mnt/data/erase_renamed_pair_relabel_RS/val"
ann_file_train = "/ai/mnt/code/mmaction2/tools/data/AmTICIS/train_S_binary.txt"
ann_file_val = "/ai/mnt/code/mmaction2/tools/data/AmTICIS/val_S_binary.txt"
ann_file_test = "/ai/mnt/code/mmaction2/tools/data/AmTICIS/val_S_binary.txt"

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
    dataset=dict(
        type="VideoDataset",
        ann_file="/ai/mnt/code/mmaction2/tools/data/AmTICIS/val_S_binary.txt",
        data_prefix=dict(video="/ai/mnt/data/erase_renamed_pair_relabel_RS/val"),
    ),
)
model = dict(
    backbone=dict(
        pretrained=None,
        depth=101,
    ),
    cls_head=dict(num_classes=2),
    data_preprocessor=dict(
        mean=[160, 160, 160],
        std=[44, 44, 44],
    ),
)
default_hooks = dict(checkpoint=dict(save_best="auto", max_keep_ckpts=3))
load_from = "/ai/mnt/code/mmaction2/configs/recognition/tsn/tsn_imagenet-pretrained-r101_8xb32-1x1x8-100e_kinetics400-rgb_20220906-23cff032.pth"
