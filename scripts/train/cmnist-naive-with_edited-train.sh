#!/bin/sh

python train.py \
    -root_path '/mnt/sdc/Debiasing' \
    -dataset cmnist \
    -conflict_ratio '0.5' \
    -train_method 'naive' \
    -with_edited \
    -lr 0.01 \
    -epochs 100 \
    -batch_size 256 \
    -seed 0 \
    -gpu_num 6 \
    -wandb \
    -projcode='GeneBC-CMNIST' \
    -run_name='naive-with_edited'