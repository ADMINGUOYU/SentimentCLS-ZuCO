# Train MLP on GLIM embeddings
python train_mlp.py \
  --device 3 \
  --glim_checkpoint ./checkpoints/glim-zuco-epoch=199-step=49600.ckpt \
  --hidden_dims 512 256 \
  --patience 20 \
  --max_epochs 100 \
  --batch_size 128 \
  --early_stopping 