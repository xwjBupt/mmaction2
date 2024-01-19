#!/bin/bash

bash /ai/mnt/code/mmaction2/tools/dist_train.sh /ai/mnt/code/mmaction2/configs/recognition/timesformer/timesformer_divST_8xb8-8x32x1-60e_AmTICIS-rgb.py 2 --seed 222 --work_dir /ai/mnt/code/mmaction2/work_dirs_update_samples_try2/
bash /ai/mnt/code/mmaction2/tools/dist_train.sh /ai/mnt/code/mmaction2/configs/recognition/timesformer/timesformer_divST_8xb8-8x32x1-60e_AmTICIS-rgb.py 2 --seed 333 --work_dir /ai/mnt/code/mmaction2/work_dirs_update_samples_try3/

bash /ai/mnt/code/mmaction2/tools/dist_train.sh /ai/mnt/code/mmaction2/configs/recognition/i3d/i3d_imagenet-pretrained-r50-heavy_8xb8-32x2x1-100e_AmTICIS-rgb.py 2 --seed 222 --work_dir /ai/mnt/code/mmaction2/work_dirs_update_samples_try2/
bash /ai/mnt/code/mmaction2/tools/dist_train.sh /ai/mnt/code/mmaction2/configs/recognition/i3d/i3d_imagenet-pretrained-r50-heavy_8xb8-32x2x1-100e_AmTICIS-rgb.py 2 --seed 333 --work_dir /ai/mnt/code/mmaction2/work_dirs_update_samples_try3/

bash /ai/mnt/code/mmaction2/tools/dist_train.sh /ai/mnt/code/mmaction2/configs/recognition/c3d/c3d_sports1m-pretrained_8xb30-16x1x1-200e_AmTICIS-rgb.py 2 --seed 222 --work_dir /ai/mnt/code/mmaction2/work_dirs_update_samples_try2/
bash /ai/mnt/code/mmaction2/tools/dist_train.sh /ai/mnt/code/mmaction2/configs/recognition/c3d/c3d_sports1m-pretrained_8xb30-16x1x1-200e_AmTICIS-rgb.py 2 --seed 333 --work_dir /ai/mnt/code/mmaction2/work_dirs_update_samples_try3/

bash /ai/mnt/code/mmaction2/tools/dist_train.sh /ai/mnt/code/mmaction2/configs/recognition/slowfast/slowfast_r101_8xb8-8x8x1-1280e_AmTICIS_C-rgb.py 2 --seed 222 --work_dir /ai/mnt/code/mmaction2/work_dirs_update_samples_try2/
bash /ai/mnt/code/mmaction2/tools/dist_train.sh /ai/mnt/code/mmaction2/configs/recognition/slowfast/slowfast_r101_8xb8-8x8x1-1280e_AmTICIS_C-rgb.py 2 --seed 333 --work_dir /ai/mnt/code/mmaction2/work_dirs_update_samples_try3/

bash /ai/mnt/code/mmaction2/tools/dist_train.sh /ai/mnt/code/mmaction2/configs/recognition/swin/swin-base-p244-w877_in1k-pre_8xb8-amp-32x2x1-300e_AmTICIS-rgb.py 2 --seed 222 --work_dir /ai/mnt/code/mmaction2/work_dirs_update_samples_try2/
bash /ai/mnt/code/mmaction2/tools/dist_train.sh /ai/mnt/code/mmaction2/configs/recognition/swin/swin-base-p244-w877_in1k-pre_8xb8-amp-32x2x1-300e_AmTICIS-rgb.py 2 --seed 333 --work_dir /ai/mnt/code/mmaction2/work_dirs_update_samples_try3/

bash /ai/mnt/code/mmaction2/tools/dist_train.sh /ai/mnt/code/mmaction2/configs/recognition/tin/tin_kinetics400-pretrained-tsm-r50_1x1x8-250e_AmTICIS-rgb.py 2 --seed 222 --work_dir /ai/mnt/code/mmaction2/work_dirs_update_samples_try2/
bash /ai/mnt/code/mmaction2/tools/dist_train.sh /ai/mnt/code/mmaction2/configs/recognition/tin/tin_kinetics400-pretrained-tsm-r50_1x1x8-250e_AmTICIS-rgb.py 2 --seed 333 --work_dir /ai/mnt/code/mmaction2/work_dirs_update_samples_try3/

