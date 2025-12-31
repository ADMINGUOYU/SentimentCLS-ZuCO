#!/bin/bash
# Launch TensorBoard to view training logs

LOG_DIR="${1:-./logs_sentiment_cls_cbramod}"

echo "Launching TensorBoard..."
echo "Log directory: $LOG_DIR"
echo ""
echo "Open your browser and navigate to: http://localhost:6006"
echo "Press Ctrl+C to stop TensorBoard"
echo ""

tensorboard --logdir "$LOG_DIR" --port 6006 --host localhost
