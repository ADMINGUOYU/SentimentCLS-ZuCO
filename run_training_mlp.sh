# Train MLP on GLIM embeddings
python train_mlp.py \
  --device 1 \
  --glim_checkpoint ./checkpoints/glim-zuco-epoch=199-step=49600.ckpt \
  --hidden_dims 512 256 128 \
  --max_epochs 50 \
  --early_stopping