#!/bin/sh

# python preprocess.py \
#     -gpu_num 0 \
#     -dataset cmnist \
#     -conflict_ratio '0.5' \
#     -n_bias 1 \
#     -root_path '/home/zungwooker/Debiasing' \
#     -pretrained_path '/home/zungwooker/Debiasing/pretrained' \
#     -preproc 'preproc' \
#     -random_seed 0 \
#     -tag2text_thres 0.68 \
#     -sim_thres 0.95 \
#     -compute_tag_stats

# python preprocess.py \
#     -gpu_num 0 \
#     -dataset cmnist \
#     -conflict_ratio '1' \
#     -n_bias 1 \
#     -root_path '/home/zungwooker/Debiasing' \
#     -pretrained_path '/home/zungwooker/Debiasing/pretrained' \
#     -preproc 'preproc' \
#     -random_seed 0 \
#     -tag2text_thres 0.68 \
#     -sim_thres 0.95 \
#     -compute_tag_stats

# python preprocess.py \
#     -gpu_num 0 \
#     -dataset cmnist \
#     -conflict_ratio '2' \
#     -n_bias 1 \
#     -root_path '/home/zungwooker/Debiasing' \
#     -pretrained_path '/home/zungwooker/Debiasing/pretrained' \
#     -preproc 'preproc' \
#     -random_seed 0 \
#     -tag2text_thres 0.68 \
#     -sim_thres 0.95 \
#     -compute_tag_stats

python preprocess.py \
    -gpu_num 0 \
    -dataset cmnist \
    -conflict_ratio '5' \
    -n_bias 1 \
    -root_path '/home/zungwooker/Debiasing' \
    -pretrained_path '/home/zungwooker/Debiasing/pretrained' \
    -preproc 'preproc' \
    -random_seed 0 \
    -tag2text_thres 0.68 \
    -sim_thres 0.95 \
    -extract_tags \
    -compute_tag_stats