#!/bin/bash

# Training script for CBraMod sentiment classifier
# Uses GLIM dataloader with weighted sampling
# Resamples EEG from 128Hz to 200Hz for CBraMod

python train_sentiment_cls_cbramod.py \
    --data_path /nfs/usrhome2/yguoco/checkpoints_sentiment_cls_with_mlp/tmp/zuco_merged.df \
    --pretrained_weights ./checkpoints/cbramod_pretrained_weights.pth \
    --num_channels 104 \
    --patch_size 200 \
    --num_patches 8 \
    --d_model 200 \
    --dim_feedforward 800 \
    --n_layer 12 \
    --nhead 8 \
    --src_sample_rate 128 \
    --tgt_sample_rate 200 \
    --hidden_dims 512 256 \
    --dropout 0.3 \
    --batch_size 64 \
    --val_batch_size 24 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --max_epochs 50 \
    --log_dir ./logs_sentiment_cls_cbramod \
    --experiment_name sentiment_cls_cbramod \
    --checkpoint_dir ./checkpoints_sentiment_cls_cbramod \
    --accelerator auto \
    --device 3 \
    --precision 32 \
    --seed 42 \
    --num_workers 4 \
    "$@"
