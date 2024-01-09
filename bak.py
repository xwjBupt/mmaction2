dataset_type = "VideoDataset"
data_root_train = "/ai/mnt/data/erase_renamed_pair_relabel_RS/train"
data_root_val = "/ai/mnt/data/erase_renamed_pair_relabel_RS/val"
ann_file_train = f"/ai/mnt/code/mmaction2/tools/data/AmTICIS/train.txt"
ann_file_val = f"/ai/mnt/code/mmaction2/tools/data/AmTICIS/val.txt"
ann_file_test = f"/ai/mnt/code/mmaction2/tools/data/AmTICIS/val.txt"

data_preprocessor = dict(
    mean=[160, 160, 160],
    std=[44, 44, 44],
)

default_hooks = dict(checkpoint=dict(interval=5, save_best="auto", max_keep_ckpts=3))
val_evaluator = dict(type="AccMetric", metric_list=("mean_class_accuracy"))
