#!/bin/sh

# python preprocess.py \
#     -gpu_num 0 \
#     -dataset cmnist \
#     -conflict_ratio '0.5' \
#     -n_bias 1 \
#     -root_path '/mnt/sdd/Debiasing' \
#     -pretrained_path '/mnt/sdd/Debiasing/pretrained' \
#     -preproc 'preproc' \
#     -random_seed 0 \
#     -tag2text_thres 0.68 \
#     -sim_thres 0.93 \
#     -compute_tag_stats \
#     -email "KM 0: CMNIST Tag stats generated. 0.5pct"

# python preprocess.py \
#     -gpu_num 0 \
#     -dataset cmnist \
#     -conflict_ratio '1' \
#     -n_bias 1 \
#     -root_path '/mnt/sdd/Debiasing' \
#     -pretrained_path '/mnt/sdd/Debiasing/pretrained' \
#     -preproc 'preproc' \
#     -random_seed 0 \
#     -tag2text_thres 0.68 \
#     -sim_thres 0.93 \
#     -compute_tag_stats \
#     -email "KM 0: CMNIST Tag stats generated. 1pct"

# python preprocess.py \
#     -gpu_num 0 \
#     -dataset cmnist \
#     -conflict_ratio '2' \
#     -n_bias 1 \
#     -root_path '/mnt/sdd/Debiasing' \
#     -pretrained_path '/mnt/sdd/Debiasing/pretrained' \
#     -preproc 'preproc' \
#     -random_seed 0 \
#     -tag2text_thres 0.68 \
#     -sim_thres 0.93 \
#     -compute_tag_stats \
#     -email "KM 0: CMNIST Tag stats generated. 2pct"

python preprocess.py \
    -gpu_num 6 \
    -dataset cmnist \
    -conflict_ratio '5' \
    -n_bias 1 \
    -root_path '/mnt/sdd/Debiasing' \
    -pretrained_path '/mnt/sdd/Debiasing/pretrained' \
    -preproc 'preproc' \
    -random_seed 0 \
    -tag2text_thres 0.68 \
    -sim_thres 0.93 \
    -compute_tag_stats \
    -email "KM 0: CMNIST Tag stats generated. 5pct"