#!/bin/bash
# Training script with sentence embedding alignment
# This uses precomputed sentence embeddings for alignment

python train_with_embeddings.py \
  --data_path /nfs/usrhome2/yguoco/checkpoints_sentiment_cls_with_mlp/tmp/zuco_merged.df \
  --embeddings_path /nfs/usrhome2/yguoco/checkpoints_sentiment_cls_with_mlp/tmp/embeddings.pickle \
  --use_sentence_embeddings \
  --eeg_emb_to_sentence_emb_hidden_dims 1024 512 \
  --device 3 \
  --batch_size 64 \
  --lr 2e-4 \
  --warm_up_step 10 \
  --max_epochs 100 \
  --early_stopping \
  --patience 100 \
  --experiment_name sentence_embedding_alignment
