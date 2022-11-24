#!/bin/bash

dataset_dir="/NAS/dataset/imagenet_100cls"
[[ ! -d $dataset_dir ]] && echo "Dataset dir ${dataset_dir} not found" && exit 1

output_dir="/data/work/convnext_output"
[[ ! -d $output_dir ]] && mkdir -p $output_dir

/usr/bin/env python3 -m torch.distributed.launch \
        --nproc_per_node=1 \
        main.py \
        --model convnext_base \
        --drop_path 0.5 \
        --batch_size 128 \
        --lr 4e-3 \
        --update_freq 1 \
        --model_ema true --model_ema_eval true \
        --data_path $dataset_dir \
        --output_dir $output_dir \
        --epochs 25
