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
        pretrained=None,
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
    batch_size=2,
    dataset=dict(ann_file=ann_file_train, data_prefix=dict(video=data_root)),
)
val_dataloader = dict(
    batch_size=4,
    dataset=dict(ann_file=ann_file_val, data_prefix=dict(video=data_root_val)),
)
test_dataloader = dict(
    dataset=dict(ann_file=ann_file_val, data_prefix=dict(video=data_root_val))
)
load_from = "/ai/mnt/code/mmaction2/configs/recognition/slowfast/slowfast_r101_8xb8-8x8x1-256e_kinetics400-rgb_20220818-9c0e09bd.pth"
