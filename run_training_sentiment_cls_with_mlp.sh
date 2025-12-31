# Train end-to-end sentiment classifier (GLIM + MLP) with gradients
python train_sentiment_cls_with_mlp.py \
  --device 2 \
  --data_path /nfs/usrhome2/yguoco/checkpoints_sentiment_cls_with_mlp/tmp/zuco_merged.df \
  --glim_checkpoint ./checkpoints/glim-zuco-epoch=199-step=49600.ckpt \
  --checkpoint_dir ./checkpoints_sentiment_cls_with_mlp \
  --experiment_name yes_freeze_with_prompt \
  --hidden_dims 512 256 128 \
  --patience 50 \
  --max_epochs 50 \
  --batch_size 64 \
  --lr 2e-4 \
  --early_stopping \
  --freeze_encoder \
  # --do_not_use_prompt \
 
