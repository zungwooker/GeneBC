#!/bin/sh

python preprocess.py \
    -gpu_num 0 \
    -dataset cifar10c \
    -conflict_ratio '0.5' \
    -n_bias 1 \
    -root_path '/home/zungwooker/Debiasing' \
    -pretrained_path '/home/zungwooker/Debiasing/pretrained' \
    -preproc 'preproc' \
    -random_seed 0 \
    -tag2text_thres 0.68 \
    -sim_thres 0.93 \
    -image_guidance_scale 1.0 \
    -generate_gate \
    -email "KM 3: CIFAR 0.5pct CLEAN."

python preprocess.py \
    -gpu_num 0 \
    -dataset cifar10c \
    -conflict_ratio '1' \
    -n_bias 1 \
    -root_path '/home/zungwooker/Debiasing' \
    -pretrained_path '/home/zungwooker/Debiasing/pretrained' \
    -preproc 'preproc' \
    -random_seed 0 \
    -tag2text_thres 0.68 \
    -sim_thres 0.93 \
    -image_guidance_scale 1.0 \
    -generate_gate \
    -email "KM 3: CIFAR 1pct CLEAN."

python preprocess.py \
    -gpu_num 0 \
    -dataset cifar10c \
    -conflict_ratio '2' \
    -n_bias 1 \
    -root_path '/home/zungwooker/Debiasing' \
    -pretrained_path '/home/zungwooker/Debiasing/pretrained' \
    -preproc 'preproc' \
    -random_seed 0 \
    -tag2text_thres 0.68 \
    -sim_thres 0.93 \
    -image_guidance_scale 1.0 \
    -generate_gate \
    -email "KM 3: CIFAR 2pct CLEAN."

python preprocess.py \
    -gpu_num 0 \
    -dataset cifar10c \
    -conflict_ratio '5' \
    -n_bias 1 \
    -root_path '/home/zungwooker/Debiasing' \
    -pretrained_path '/home/zungwooker/Debiasing/pretrained' \
    -preproc 'preproc' \
    -random_seed 0 \
    -tag2text_thres 0.68 \
    -sim_thres 0.93 \
    -image_guidance_scale 1.0 \
    -generate_gate \
    -email "KM 3: CIFAR 5pct ALL CLEAN."