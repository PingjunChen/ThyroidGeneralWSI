#!/bin/bash

FUSION_MODE="pooling"
Input_Fea_Num=2048

CUDA_VISIBLE_DEVICES=7 \
python train.py \
    --pre_load \
    --mode ${FUSION_MODE} \
    --input_fea_num ${Input_Fea_Num} \
    --maxepoch 100 \
    --batch_size 24 \
    --verbose
