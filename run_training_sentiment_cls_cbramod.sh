#!/bin/bash

# Training script for CBraMod sentiment classifier
# Uses GLIM dataloader with weighted sampling
# Resamples EEG from 128Hz to 200Hz for CBraMod

python train_sentiment_cls_cbramod.py \
    --data_path ./data/tmp/zuco_merged.df \
    --num_channels 105 \
    --patch_size 200 \
    --num_patches 1 \
    --d_model 200 \
    --dim_feedforward 800 \
    --n_layer 12 \
    --nhead 8 \
    --src_sample_rate 128 \
    --tgt_sample_rate 200 \
    --hidden_dims 512 256 \
    --dropout 0.3 \
    --batch_size 24 \
    --val_batch_size 24 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --max_epochs 50 \
    --log_dir ./logs_sentiment_cls_cbramod \
    --experiment_name sentiment_cls_cbramod \
    --checkpoint_dir ./checkpoints_sentiment_cls_cbramod \
    --accelerator auto \
    --device 0 \
    --precision 32 \
    --seed 42 \
    --num_workers 4 \
    "$@"
