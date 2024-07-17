#!/bin/sh

python preprocess.py \
    -gpu_num 4 \
    -dataset cmnist \
    -conflict_ratio '0.5' \
    -n_bias 1 \
    -root_path '/mnt/sdc/Debiasing' \
    -pretrained_path '/mnt/sdc/Debiasing/pretrained' \
    -preproc 'preproc_digit_igs1.0' \
    -random_seed 0 \
    -tag2text_thres 0.68 \
    -sim_thres 0.95 \
    -image_guidance_scale 1.0 \
    -edit_class_idx '3,4'