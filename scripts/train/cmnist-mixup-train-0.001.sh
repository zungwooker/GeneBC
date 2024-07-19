#!/bin/sh

python train.py \
    -root_path '/home/zungwooker/Debiasing' \
    -preproc 'preproc_digit_igs1.0' \
    -dataset cmnist \
    -conflict_ratio '0.5' \
    -train_method 'mixup' \
    -with_edited \
    -lr 0.001 \
    -epochs 200 \
    -batch_size 256 \
    -seed 0 \
    -gpu_num 2 \
    -projcode='GeneBC-CMNIST' \
    -run_name='mixup-0.1' \
    -wandb