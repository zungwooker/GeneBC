#!/bin/sh

python preprocess.py \
    -gpu_num 2 \
    -dataset dogs_and_cats \
    -conflict_ratio '1' \
    -n_bias 1 \
    -root_path '/home/zungwooker/Debiasing' \
    -pretrained_path '/home/zungwooker/Debiasing/pretrained' \
    -preproc 'preproc' \
    -random_seed 0 \
    -tag2text_thres 0.68 \
    -sim_thres 0.93 \
    -compute_tag_stats \
    -email "KM 2: DNC Tag stats generated. 1pct"

python preprocess.py \
    -gpu_num 2 \
    -dataset dogs_and_cats \
    -conflict_ratio '5' \
    -n_bias 1 \
    -root_path '/home/zungwooker/Debiasing' \
    -pretrained_path '/home/zungwooker/Debiasing/pretrained' \
    -preproc 'preproc' \
    -random_seed 0 \
    -tag2text_thres 0.68 \
    -sim_thres 0.93 \
    -compute_tag_stats \
    -email "KM 2: DNC Tag stats generated. 5pct"

python preprocess.py \
    -gpu_num 2 \
    -dataset bar \
    -conflict_ratio '1' \
    -n_bias 1 \
    -root_path '/home/zungwooker/Debiasing' \
    -pretrained_path '/home/zungwooker/Debiasing/pretrained' \
    -preproc 'preproc' \
    -random_seed 0 \
    -tag2text_thres 0.68 \
    -sim_thres 0.93 \
    -compute_tag_stats \
    -email "KM 2: BAR Tag stats generated. 1pct"

python preprocess.py \
    -gpu_num 2 \
    -dataset bar \
    -conflict_ratio '5' \
    -n_bias 1 \
    -root_path '/home/zungwooker/Debiasing' \
    -pretrained_path '/home/zungwooker/Debiasing/pretrained' \
    -preproc 'preproc' \
    -random_seed 0 \
    -tag2text_thres 0.68 \
    -sim_thres 0.93 \
    -compute_tag_stats \
    -email "KM 2: BAR Tag stats generated. 5pct"