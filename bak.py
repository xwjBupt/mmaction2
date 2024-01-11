dataset_type = "VideoDataset"
data_root_train = "/ai/mnt/data/erase_renamed_pair_relabel_RS/train"
data_root_val = "/ai/mnt/data/erase_renamed_pair_relabel_RS/val"
ann_file_train = "/ai/mnt/code/mmaction2/tools/data/AmTICIS/train.txt"
ann_file_val = "/ai/mnt/code/mmaction2/tools/data/AmTICIS/val.txt"
ann_file_test = "/ai/mnt/code/mmaction2/tools/data/AmTICIS/val.txt"

model = dict(
    data_preprocessor=dict(
        mean=[160, 160, 160],
        std=[44, 44, 44],
    ),
    backbone=dict(pretrained=None),
    cls_head=dict(num_classes=4),
)

default_hooks = dict(checkpoint=dict(interval=5, save_best="auto", max_keep_ckpts=3))
val_evaluator = dict(type="AccMetric", metric_list=("mean_class_accuracy"))
load_from = "https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb_20220906-cd10898e.pth"
