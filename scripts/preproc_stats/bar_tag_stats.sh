#!/bin/sh

python preprocess.py \
    -gpu_num 2 \
    -dataset bar \
    -conflict_ratio '1' \
    -n_bias 1 \
    -root_path '/mnt/sdd//Debiasing' \
    -pretrained_path '/mnt/sdd//Debiasing/pretrained' \
    -preproc 'preproc' \
    -random_seed 0 \
    -tag2text_thres 0.68 \
    -sim_thres 0.95 \
    -compute_tag_stats

python preprocess.py \
    -gpu_num 2 \
    -dataset bar \
    -conflict_ratio '5' \
    -n_bias 1 \
    -root_path '/mnt/sdd//Debiasing' \
    -pretrained_path '/mnt/sdd//Debiasing/pretrained' \
    -preproc 'preproc' \
    -random_seed 0 \
    -tag2text_thres 0.68 \
    -sim_thres 0.95 \
    -compute_tag_stats