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
    -image_guidance_scale 1.0 \
    -generate_gate \
    -edit_class_idx '7,8,9' \
    -email "KM 3: CIFAR 7,8,9 generated. | 0.5 "

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
    -image_guidance_scale 1.0 \
    -generate_gate \
    -edit_class_idx '7,8,9' \
    -email "KM 3: CIFAR 7,8,9 generated. | 1"

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
    -image_guidance_scale 1.0 \
    -generate_gate \
    -edit_class_idx '7,8,9' \
    -email "KM 3: CIFAR 7,8,9 generated. | 2"

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
    -image_guidance_scale 1.0 \
    -generate_gate \
    -edit_class_idx '7,8,9' \
    -email "KM 3: CIFAR 7,8,9 generated. | 5"