#!/bin/bash

dataset_dir="/NAS/dataset/imagenet_100cls"
[[ ! -d $dataset_dir ]] && echo "Dataset dir ${dataset_dir} not found" && exit 1

output_dir="/data/work/convnext_output"
[[ ! -d $output_dir ]] && echo "No output dir ${output_dir} found" && exit 1

/usr/bin/env python3 -m torch.distributed.launch \
        --nproc_per_node=1 \
        main.py \
        --model convnext_base \
        --drop_path 0.8 \
        --input_size 348 \
        --batch_size 32 \
        --lr 5e-5 \
        --update_freq 2 \
        --head_init_scale 0.001 --cutmix 0 --mixup 0 \
        --finetune $output_dir/checkpoint-best-ema.pth \
        --data_path $dataset_dir \
        --output_dir $output_dir \
        --warmup_epochs 0 --epochs 28
