#!/bin/sh

python train.py \
    -root_path '/mnt/sdc/Debiasing' \
    -dataset bffhq \
    -conflict_ratio '0.5' \
    -train_method 'pairing' \
    -with_edited \
    -lr 0.0001 \
    -epochs 100 \
    -batch_size 64 \
    -seed 0 \
    -gpu_num 3 \
    -wandb \
    -projcode='GeneBC-bFFHQ' \
    -run_name='pairing'