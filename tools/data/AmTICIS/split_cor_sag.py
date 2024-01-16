import os
import glob
from tqdm import tqdm


def get_label(file_dir, binary=False):
    label_str = file_dir.split("/")[-1].split("_")[1]
    if label_str == "T0" or label_str == "T1":
        label = 0
    elif label_str == "T2a" or label_str == "T2A":
        label = 1
    elif label_str == "T2b" or label_str == "T2B":
        label = 2
    elif label_str == "T3":
        label = 3
    else:
        assert False, "%s not support" % label_str
    if binary:
        if label == 0 or label == 1:
            label = 0
        else:
            label = 1
    return label


binary = False
index = "_S"
dataroot = "/ai/mnt/data/erase_renamed_pair_relabel_RS/"
trains = glob.glob(dataroot + "/train/*%s*fps1.mp4" % index)
vals = glob.glob(dataroot + "/val/*%s*fps1.mp4" % index)
lines = []
for train in tqdm(trains):
    label = get_label(train, binary)
    lines.append("%s %s\n" % (train, label))
if binary:
    file = open(
        "/ai/mnt/code/mmaction2/tools/data/AmTICIS/train_binary%s.txt" % index,
        mode="w",
        encoding="utf-8",
    )
else:
    file = open(
        "/ai/mnt/code/mmaction2/tools/data/AmTICIS/train%s.txt" % index,
        mode="w",
        encoding="utf-8",
    )
file.writelines(lines)
file.close()

lines = []
for val in tqdm(vals):
    label = get_label(val, binary)
    lines.append("%s %s\n" % (val, label))
if binary:
    file = open(
        "/ai/mnt/code/mmaction2/tools/data/AmTICIS/val_binary%s.txt" % index,
        mode="w",
        encoding="utf-8",
    )
else:
    file = open(
        "/ai/mnt/code/mmaction2/tools/data/AmTICIS/val%s.txt" % index,
        mode="w",
        encoding="utf-8",
    )
file.writelines(lines)
file.close()
