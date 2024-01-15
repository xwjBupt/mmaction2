_base_ = ["slowfast_r50_8xb8-8x8x1-256e_kinetics400-rgb.py"]

dataset_type = "VideoDataset"
data_root = "/ai/mnt/data/erase_renamed_pair_relabel_RS/train"
data_root_val = "/ai/mnt/data/erase_renamed_pair_relabel_RS/val"
ann_file_train = "/ai/mnt/code/mmaction2/tools/data/AmTICIS/train.txt"
ann_file_val = "/ai/mnt/code/mmaction2/tools/data/AmTICIS/val.txt"
ann_file_test = "/ai/mnt/code/mmaction2/tools/data/AmTICIS/val.txt"

model = dict(
    backbone=dict(
        slow_pathway=dict(depth=101),
        fast_pathway=dict(depth=101),
    ),
    cls_head=dict(
        num_classes=4,
    ),
    data_preprocessor=dict(
        mean=[160.675, 160.28, 160.53],
        std=[44.395, 44.12, 44.375],
    ),
)

train_dataloader = dict(
    batch_size=8,
    dataset=dict(ann_file=ann_file_train, data_prefix=dict(video=data_root)),
)
val_dataloader = dict(
    batch_size=8,
    dataset=dict(ann_file=ann_file_val, data_prefix=dict(video=data_root_val)),
)
test_dataloader = dict(
    dataset=dict(ann_file=ann_file_val, data_prefix=dict(video=data_root_val))
)
train_cfg = dict(
    type="EpochBasedTrainLoop", max_epochs=1280, val_begin=50, val_interval=25
)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

optim_wrapper = dict(
    optimizer=dict(type="SGD", lr=0.1, momentum=0.9, weight_decay=1e-4),
    clip_grad=dict(max_norm=40, norm_type=2),
)

param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=0.1,
        by_epoch=True,
        begin=0,
        end=170,
        convert_to_iter_based=True,
    ),
    dict(
        type="CosineAnnealingLR",
        T_max=1280,
        eta_min=0,
        by_epoch=True,
        begin=0,
        end=1280,
    ),
]

load_from = "/ai/mnt/code/mmaction2/configs/recognition/slowfast/slowfast_r101_8xb8-8x8x1-256e_kinetics400-rgb_20220818-9c0e09bd.pth"
