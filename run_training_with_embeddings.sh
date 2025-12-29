#!/bin/bash
# Training script with sentence embedding alignment
# This uses precomputed sentence embeddings for alignment

python train.py \
  --data_path ./tmp/zuco_merged.df \
  --embeddings_path ./tmp/embeddings.pickle \
  --use_sentence_embeddings \
  --device 1 \
  --batch_size 48 \
  --lr 1e-5 \
  --max_epochs 100 \
  --early_stopping \
  --patience 10 \
  --experiment_name sentence_embedding_alignment
