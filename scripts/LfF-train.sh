#!/bin/sh

python LfF-train.py \
    -model_type mlp \
    -dataset cmnist \
    -lr 0.001 \
    -epochs 10 \
    -batch_size 256 \
    -seed 0 \
    -conflict_ratio 0.5 \
    -projcode='LfF-Tutorial' \
    -gpu_num 6 \
    -run_name='LfF: Base' \
    -wandb