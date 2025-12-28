# Train end-to-end sentiment classifier (GLIM + MLP) with gradients
python train_sentiment_cls_with_mlp.py \
  --device 3 \
  --glim_checkpoint ./checkpoints/glim-zuco-epoch=199-step=49600.ckpt \
  --hidden_dims 512 256 128 \
  --patience 15 \
  --max_epochs 50 \
  --batch_size 24 \
  --lr 1e-4 \
  --early_stopping
