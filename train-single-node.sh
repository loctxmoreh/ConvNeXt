#!/bin/bash

python -m torch.distributed.launch \
        --nproc_per_node=1 \
        main.py \
        --model convnext_base \
        --drop_path 0.5 \
        --batch_size 128 \
        --lr 4e-3 \
        --update_freq 1 \
        --model_ema true --model_ema_eval true \
        --data_path /data/work/dataset/imagenet_100cls \
        --output_dir /data/work/convnext_output \
        --epochs 25
