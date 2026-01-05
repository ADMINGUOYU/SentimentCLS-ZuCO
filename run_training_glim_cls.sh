#!/bin/bash

# Training script for GLIM_CLS model
# Combined GLIM encoder + MLP classifier for classification tasks

# Example 1: Train with default sentiment classification
python train_glim_cls.py \
    --data_path ./data/tmp/zuco_merged.df \
    --classification_label_key "sentiment label" \
    --classification_labels negative neutral positive \
    --batch_size 24 \
    --max_epochs 50 \
    --lr 1e-4 \
    --min_lr 1e-6 \
    --warmup_steps 100 \
    --mlp_hidden_dims 512 256 \
    --mlp_dropout 0.3 \
    --clip_loss_weight 0.5 \
    --mlp_loss_weight 0.5 \
    --device 0

# Example 2: Train with frozen encoder (only MLP training)
# python train_glim_cls.py \
#     --data_path ./data/tmp/zuco_merged.df \
#     --classification_label_key "sentiment label" \
#     --classification_labels negative neutral positive \
#     --batch_size 24 \
#     --max_epochs 50 \
#     --freeze_encoder \
#     --device 0

# Example 3: Resume training from checkpoint
# python train_glim_cls.py \
#     --data_path ./data/tmp/zuco_merged.df \
#     --checkpoint ./checkpoints_glim_cls/glim_cls/last.ckpt \
#     --batch_size 24 \
#     --max_epochs 100 \
#     --device 0
