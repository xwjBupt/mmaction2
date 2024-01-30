#!/bin/bash

bash /ai/mnt/code/mmaction2/tools/dist_train.sh /ai/mnt/code/mmaction2/configs/recognition/i3d/i3d_imagenet-pretrained-r50-heavy_8xb8-32x2x1-100e_AmTICIS_C-rgb.py 2 --seed 222 --work_dir /ai/mnt/code/mmaction2/WORK_DIRS/work_dirs_update_samples_cor_try2/
bash /ai/mnt/code/mmaction2/tools/dist_train.sh /ai/mnt/code/mmaction2/configs/recognition/i3d/i3d_imagenet-pretrained-r50-heavy_8xb8-32x2x1-100e_AmTICIS_C-rgb.py 2 --seed 333 --work_dir /ai/mnt/code/mmaction2/WORK_DIRS/work_dirs_update_samples_cor_try3/
bash /ai/mnt/code/mmaction2/tools/dist_train.sh /ai/mnt/code/mmaction2/configs/recognition/i3d/i3d_imagenet-pretrained-r50-heavy_8xb8-32x2x1-100e_AmTICIS_S-rgb.py 2 --seed 222 --work_dir /ai/mnt/code/mmaction2/WORK_DIRS/work_dirs_update_samples_sag_try2/
bash /ai/mnt/code/mmaction2/tools/dist_train.sh /ai/mnt/code/mmaction2/configs/recognition/i3d/i3d_imagenet-pretrained-r50-heavy_8xb8-32x2x1-100e_AmTICIS_S-rgb.py 2 --seed 333 --work_dir /ai/mnt/code/mmaction2/WORK_DIRS/work_dirs_update_samples_sag_try3

bash /ai/mnt/code/mmaction2/tools/dist_train.sh /ai/mnt/code/mmaction2/configs/recognition/tsn/tsn_imagenet-pretrained-r101_8xb32-1x1x8-100e_AmTICIS_C-rgb.py 2 --seed 222 --work_dir /ai/mnt/code/mmaction2/WORK_DIRS/work_dirs_update_samples_cor_try2/
bash /ai/mnt/code/mmaction2/tools/dist_train.sh /ai/mnt/code/mmaction2/configs/recognition/tsn/tsn_imagenet-pretrained-r101_8xb32-1x1x8-100e_AmTICIS_C-rgb.py 2 --seed 333 --work_dir /ai/mnt/code/mmaction2/WORK_DIRS/work_dirs_update_samples_cor_try3/
bash /ai/mnt/code/mmaction2/tools/dist_train.sh /ai/mnt/code/mmaction2/configs/recognition/tsn/tsn_imagenet-pretrained-r101_8xb32-1x1x8-100e_AmTICIS_S-rgb.py 2 --seed 222 --work_dir /ai/mnt/code/mmaction2/WORK_DIRS/work_dirs_update_samples_sag_try2/
bash /ai/mnt/code/mmaction2/tools/dist_train.sh /ai/mnt/code/mmaction2/configs/recognition/tsn/tsn_imagenet-pretrained-r101_8xb32-1x1x8-100e_AmTICIS_S-rgb.py 2 --seed 333 --work_dir /ai/mnt/code/mmaction2/WORK_DIRS/work_dirs_update_samples_sag_try3

bash /ai/mnt/code/mmaction2/tools/dist_train.sh /ai/mnt/code/mmaction2/configs/recognition/uniformerv2/uniformerv2-base-p16-res224_clip-kinetics710-pre_8xb32-u8_AmTICIS_C-rgb-100e.py 2 --seed 222 --work_dir /ai/mnt/code/mmaction2/WORK_DIRS/work_dirs_update_samples_cor_try2/
bash /ai/mnt/code/mmaction2/tools/dist_train.sh /ai/mnt/code/mmaction2/configs/recognition/uniformerv2/uniformerv2-base-p16-res224_clip-kinetics710-pre_8xb32-u8_AmTICIS_C-rgb-100e.py 2 --seed 333 --work_dir /ai/mnt/code/mmaction2/WORK_DIRS/work_dirs_update_samples_cor_try3/
bash /ai/mnt/code/mmaction2/tools/dist_train.sh /ai/mnt/code/mmaction2/configs/recognition/uniformerv2/uniformerv2-base-p16-res224_clip-kinetics710-pre_8xb32-u8_AmTICIS_S-rgb-100e.py 2 --seed 222 --work_dir /ai/mnt/code/mmaction2/WORK_DIRS/work_dirs_update_samples_sag_try2/
bash /ai/mnt/code/mmaction2/tools/dist_train.sh /ai/mnt/code/mmaction2/configs/recognition/uniformerv2/uniformerv2-base-p16-res224_clip-kinetics710-pre_8xb32-u8_AmTICIS_S-rgb-100e.py 2 --seed 333 --work_dir /ai/mnt/code/mmaction2/WORK_DIRS/work_dirs_update_samples_sag_try3


# bash /ai/mnt/code/mmaction2/tools/dist_train.sh /ai/mnt/code/mmaction2/configs/recognition/tanet/tanet_imagenet-pretrained-r50_8xb8-dense-1x1x8-500e_AmTICIS-rgb.py 2 --seed 222 --work_dir /ai/mnt/code/mmaction2/work_dirs_update_samples_try2
# bash /ai/mnt/code/mmaction2/tools/dist_train.sh /ai/mnt/code/mmaction2/configs/recognition/tanet/tanet_imagenet-pretrained-r50_8xb8-dense-1x1x8-500e_AmTICIS-rgb.py 2 --seed 333 --work_dir /ai/mnt/code/mmaction2/work_dirs_update_samples_try3

# bash /ai/mnt/code/mmaction2/tools/dist_train.sh /ai/mnt/code/mmaction2/configs/recognition/r2plus1d/r2plus1d_r34_8xb8-32x2x1-900e_AmTICIS-rgb.py 2 --seed 222 --work_dir /ai/mnt/code/mmaction2/work_dirs_update_samples_try2
# bash /ai/mnt/code/mmaction2/tools/dist_train.sh /ai/mnt/code/mmaction2/configs/recognition/r2plus1d/r2plus1d_r34_8xb8-32x2x1-900e_AmTICIS-rgb.py 2 --seed 333 --work_dir /ai/mnt/code/mmaction2/work_dirs_update_samples_try3

# bash /ai/mnt/code/mmaction2/tools/dist_train.sh /ai/mnt/code/mmaction2/configs/recognition/tsn/tsn_imagenet-pretrained-r101_8xb32-1x1x8-100e_AmTICIS-rgb.py 2 --seed 222 --work_dir /ai/mnt/code/mmaction2/work_dirs_update_samples_try2
# bash /ai/mnt/code/mmaction2/tools/dist_train.sh /ai/mnt/code/mmaction2/configs/recognition/tsn/tsn_imagenet-pretrained-r101_8xb32-1x1x8-100e_AmTICIS-rgb.py 2 --seed 333 --work_dir /ai/mnt/code/mmaction2/work_dirs_update_samples_try3

# bash /ai/mnt/code/mmaction2/tools/dist_train.sh /ai/mnt/code/mmaction2/configs/recognition/tsm/tsm_imagenet-pretrained-r50-nl-dot-product_8xb16-1x1x8-250e_AmTICIS-rgb.py 2 --seed 222 --work_dir /ai/mnt/code/mmaction2/work_dirs_update_samples_try2
# bash /ai/mnt/code/mmaction2/tools/dist_train.sh /ai/mnt/code/mmaction2/configs/recognition/tsm/tsm_imagenet-pretrained-r50-nl-dot-product_8xb16-1x1x8-250e_AmTICIS-rgb.py 2 --seed 333 --work_dir /ai/mnt/code/mmaction2/work_dirs_update_samples_try3

# bash /ai/mnt/code/mmaction2/tools/dist_train.sh /ai/mnt/code/mmaction2/configs/recognition/uniformerv2/uniformerv2-base-p16-res224_clip-kinetics710-pre_8xb32-u8_AmTICIS-rgb-100e.py 2 --seed 222 --work_dir /ai/mnt/code/mmaction2/work_dirs_update_samples_try2
# bash /ai/mnt/code/mmaction2/tools/dist_train.sh /ai/mnt/code/mmaction2/configs/recognition/uniformerv2/uniformerv2-base-p16-res224_clip-kinetics710-pre_8xb32-u8_AmTICIS-rgb-100e.py 2 --seed 333 --work_dir /ai/mnt/code/mmaction2/work_dirs_update_samples_try3


# bash /ai/mnt/code/mmaction2/tools/dist_train.sh /ai/mnt/code/mmaction2/configs/recognition/uniformerv2/uniformerv2-base-p16-res224_clip-kinetics710-pre_8xb32-u8_AmTICIS-binary-100e.py 2 --seed 111 --work_dir /ai/mnt/code/mmaction2/work_dirs_update_samples_binary_try1
# bash /ai/mnt/code/mmaction2/tools/dist_train.sh /ai/mnt/code/mmaction2/configs/recognition/uniformerv2/uniformerv2-base-p16-res224_clip-kinetics710-pre_8xb32-u8_AmTICIS-binary-100e.py 2 --seed 222 --work_dir /ai/mnt/code/mmaction2/work_dirs_update_samples_binary_try2
# bash /ai/mnt/code/mmaction2/tools/dist_train.sh /ai/mnt/code/mmaction2/configs/recognition/uniformerv2/uniformerv2-base-p16-res224_clip-kinetics710-pre_8xb32-u8_AmTICIS-binary-100e.py 2 --seed 333 --work_dir /ai/mnt/code/mmaction2/work_dirs_update_samples_binary_try3

# bash /ai/mnt/code/mmaction2/tools/dist_train.sh /ai/mnt/code/mmaction2/configs/recognition/i3d/i3d_imagenet-pretrained-r50-heavy_8xb8-32x2x1-100e_AmTICIS-binary-rgb.py 2 --seed 111 --work_dir /ai/mnt/code/mmaction2/work_dirs_update_samples_binary_try1
# bash /ai/mnt/code/mmaction2/tools/dist_train.sh /ai/mnt/code/mmaction2/configs/recognition/i3d/i3d_imagenet-pretrained-r50-heavy_8xb8-32x2x1-100e_AmTICIS-binary-rgb.py 2 --seed 222 --work_dir /ai/mnt/code/mmaction2/work_dirs_update_samples_binary_try2
# bash /ai/mnt/code/mmaction2/tools/dist_train.sh /ai/mnt/code/mmaction2/configs/recognition/i3d/i3d_imagenet-pretrained-r50-heavy_8xb8-32x2x1-100e_AmTICIS-binary-rgb.py 2 --seed 333 --work_dir /ai/mnt/code/mmaction2/work_dirs_update_samples_binary_try3

# bash /ai/mnt/code/mmaction2/tools/dist_train.sh /ai/mnt/code/mmaction2/configs/recognition/tsn/tsn_imagenet-pretrained-r101_8xb32-1x1x8-100e_AmTICIS-binary.py 2 --seed 111 --work_dir /ai/mnt/code/mmaction2/work_dirs_update_samples_binary_try1
# bash /ai/mnt/code/mmaction2/tools/dist_train.sh /ai/mnt/code/mmaction2/configs/recognition/tsn/tsn_imagenet-pretrained-r101_8xb32-1x1x8-100e_AmTICIS-binary.py 2 --seed 222 --work_dir /ai/mnt/code/mmaction2/work_dirs_update_samples_binary_try2
# bash /ai/mnt/code/mmaction2/tools/dist_train.sh /ai/mnt/code/mmaction2/configs/recognition/tsn/tsn_imagenet-pretrained-r101_8xb32-1x1x8-100e_AmTICIS-binary.py 2 --seed 333 --work_dir /ai/mnt/code/mmaction2/work_dirs_update_samples_binary_try3
