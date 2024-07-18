#!/bin/sh

python train.py \
    -root_path '/home/zungwooker/Debiasing' \
    -preproc 'preproc_youngoldperson_igs1.0' \
    -dataset bffhq \
    -conflict_ratio '0.5' \
    -train_method 'mixup' \
    -with_edited \
    -lr 0.0001 \
    -epochs 100 \
    -batch_size 64 \
    -seed 0 \
    -gpu_num 3 \
    -projcode='GeneBC-bFFHQ' \
    -run_name='mixup-youngoldperson' \
    -wandb