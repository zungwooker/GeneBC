#!/bin/sh
# minjung gpu

python preprocess.py \
    -gpu_num 5 \
    -dataset cmnist \
    -conflict_ratio '0.5' \
    -n_bias 1 \
    -tag2text_thres 0.68 \
    -sim_thres 0.95 \
    -root_path '/mnt/sdc/Debiasing' \
    -pretrained_path '/mnt/sdc/Debiasing/pretrained' \
    -random_seed 0