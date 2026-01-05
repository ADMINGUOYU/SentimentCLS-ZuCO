# Train end-to-end sentiment classifier (GLIM + MLP) with gradients
# 
# Option 1: Load from a pre-trained GLIM checkpoint
# python train_sentiment_cls_with_mlp.py \
#   --device 2 \
#   --data_path /nfs/usrhome2/yguoco/checkpoints_sentiment_cls_with_mlp/tmp/zuco_merged.df \
#   --glim_checkpoint ./checkpoints/glim-zuco-epoch=199-step=49600.ckpt \
#   --checkpoint_dir ./checkpoints_sentiment_cls_with_mlp \
#   --experiment_name yes_freeze_with_prompt \
#   --hidden_dims 512 256 128 \
#   --patience 50 \
#   --max_epochs 50 \
#   --batch_size 64 \
#   --lr 2e-4 \
#   --early_stopping \
#   --freeze_encoder \
#   --clip_loss_weight 0.5 \
#   --lm_loss_weight 0.5 \
#   --commitment_loss_weight 0.0 \
#   --mlp_loss_weight 0.5
#   # Optional: --do_not_use_prompt

# Option 2: Create a new GLIM model with semantic embedding alignment
python train_sentiment_cls_with_mlp.py \
  --device 2 \
  --data_path /nfs/usrhome2/yguoco/checkpoints_sentiment_cls_with_mlp/tmp/zuco_merged.df \
  --embeddings_path /nfs/usrhome2/yguoco/checkpoints_sentiment_cls_with_mlp/tmp/zuco_embeddings.pkl \
  --checkpoint_dir ./checkpoints_sentiment_cls_with_mlp \
  --experiment_name new_glim_semantic_alignment \
  --hidden_dims 512 256 128 \
  --patience 50 \
  --max_epochs 50 \
  --batch_size 64 \
  --lr 2e-4 \
  --early_stopping \
  --text_model google/flan-t5-large \
  --hidden_dim 128 \
  --embed_dim 1024 \
  --n_in_blocks 6 \
  --n_out_blocks 6 \
  --num_heads 8 \
  --glim_dropout 0.1 \
  --use_sentence_embeddings \
  --eeg_emb_to_sentence_emb_hidden_dims 1024 512 \
  --clip_loss_weight 0.9 \
  --lm_loss_weight 0.0 \
  --commitment_loss_weight 0.0 \
  --mlp_loss_weight 0.5
  # Optional: --do_not_use_prompt --freeze_encoder

