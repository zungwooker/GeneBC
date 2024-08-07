#!/bin/sh

python preprocess.py \
    -gpu_num 3 \
    -dataset cifar10c \
    -conflict_ratio '0.5' \
    -n_bias 1 \
    -root_path '/home/zungwooker/Debiasing' \
    -pretrained_path '/home/zungwooker/Debiasing/pretrained' \
    -preproc 'preproc' \
    -random_seed 0 \
    -tag2text_thres 0.68 \
    -sim_thres 0.93 \
    -compute_tag_stats \
    -email "KM 3: CIFAR10C Tag stats generated. 0.5pct"

python preprocess.py \
    -gpu_num 3 \
    -dataset cifar10c \
    -conflict_ratio '1' \
    -n_bias 1 \
    -root_path '/home/zungwooker/Debiasing' \
    -pretrained_path '/home/zungwooker/Debiasing/pretrained' \
    -preproc 'preproc' \
    -random_seed 0 \
    -tag2text_thres 0.68 \
    -sim_thres 0.93 \
    -compute_tag_stats \
    -email "KM 3: CIFAR10C Tag stats generated. 1pct"

python preprocess.py \
    -gpu_num 3 \
    -dataset cifar10c \
    -conflict_ratio '2' \
    -n_bias 1 \
    -root_path '/home/zungwooker/Debiasing' \
    -pretrained_path '/home/zungwooker/Debiasing/pretrained' \
    -preproc 'preproc' \
    -random_seed 0 \
    -tag2text_thres 0.68 \
    -sim_thres 0.93 \
    -compute_tag_stats \
    -email "KM 3: CIFAR10C Tag stats generated. 2pct"

python preprocess.py \
    -gpu_num 3 \
    -dataset cifar10c \
    -conflict_ratio '5' \
    -n_bias 1 \
    -root_path '/home/zungwooker/Debiasing' \
    -pretrained_path '/home/zungwooker/Debiasing/pretrained' \
    -preproc 'preproc' \
    -random_seed 0 \
    -tag2text_thres 0.68 \
    -sim_thres 0.93 \
    -compute_tag_stats \
    -email "KM 3: CIFAR10C Tag stats generated. 5pct"