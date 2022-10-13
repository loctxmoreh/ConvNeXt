#!/bin/bash

python -m torch.distributed.launch \
        --nproc_per_node=1 \
        main.py \
        --model convnext_base \
        --drop_path 0.8 \
        --input_size 348 \
        --batch_size 32 \
        --lr 5e-5 \
        --update_freq 2 \
        --head_init_scale 0.001 --cutmix 0 --mixup 0 \
        --finetune /data/work/convnext_output/checkpoint-best-ema.pth \
        --data_path /data/work/dataset/imagenet_100cls \
        --output_dir /data/work/convnext_output \
        --warmup_epochs 0 --epochs 28
