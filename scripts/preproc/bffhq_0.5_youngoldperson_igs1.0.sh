#!/bin/sh

python preprocess.py \
    -gpu_num 3 \
    -dataset bffhq \
    -conflict_ratio '0.5' \
    -n_bias 1 \
    -root_path '/home/zungwooker/Debiasing' \
    -pretrained_path '/home/zungwooker/Debiasing/pretrained' \
    -preproc 'preproc_youngoldperson_igs1.0' \
    -random_seed 0 \
    -tag2text_thres 0.68 \
    -sim_thres 0.95 \
    -image_guidance_scale 1.0