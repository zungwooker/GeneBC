#!/bin/sh
# glee gpu

python preprocess.py \
    -gpu_num 3 \
    -dataset bffhq \
    -conflict_ratio '0.5' \
    -n_bias 1 \
    -tag2text_thres 0.68 \
    -sim_thres 0.95 \
    -root_path '/mnt/sdc/Debiasing' \
    -pretrained_path '/mnt/sdc/Debiasing/pretrained' \
    -random_seed 0