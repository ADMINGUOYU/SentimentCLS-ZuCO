# Train MLP on GLIM embeddings
python train_mlp.py \
  --device 1 \
  --data_path /nfs/usrhome2/yguoco/checkpoints_sentiment_cls_with_mlp/tmp/zuco_merged.df \
  --embeddings_cache_dir /nfs/usrhome2/yguoco/checkpoints_sentiment_cls_with_mlp/tmp/embeddings_cache \
  --glim_checkpoint ./checkpoints/glim-zuco-epoch=199-step=49600.ckpt \
  --hidden_dims 512 256 \
  --patience 20 \
  --max_epochs 100 \
  --batch_size 128 \
  --early_stopping 