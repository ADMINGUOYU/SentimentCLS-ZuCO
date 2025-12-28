# Sentiment Classification with MLP - Implementation Guide

## Overview

This implementation adds a new end-to-end sentiment classifier that combines the GLIM encoder with an MLP classifier head. Unlike the existing `SentimentMLP` model which uses pre-extracted, cached embeddings, this new model trains both the encoder and classifier together with gradients flowing through both components.

## Key Components

### 1. New Model: `sentiment_cls_with_mlp.py`

Located at: `model/sentiment_cls_with_mlp.py`

**Features:**
- Combines GLIM encoder with MLP classifier
- Supports loading pretrained GLIM checkpoint
- Trains both encoder and classifier together (gradients enabled)
- Option to freeze encoder and only train classifier
- Uses differential learning rates (encoder: 10x lower, classifier: normal)

**Usage:**
```python
from model.sentiment_cls_with_mlp import SentimentCLSWithMLP

# Create model with pretrained GLIM
model = SentimentCLSWithMLP(
    glim_checkpoint='path/to/glim.ckpt',
    hidden_dims=[512, 256, 128],
    num_classes=3,
    dropout=0.3,
    lr=1e-4,
    freeze_encoder=False  # Set to True to only train MLP
)
```

### 2. Training Script: `train_sentiment_cls_with_mlp.py`

**Features:**
- Trains end-to-end with raw EEG data (not cached embeddings)
- Weighted sampling enabled by default for class balance
- Supports command-line arguments
- Includes proper logging and checkpointing
- Displays confusion matrix after testing

**Run Training:**
```bash
python train_sentiment_cls_with_mlp.py \
  --glim_checkpoint ./checkpoints/glim-zuco-epoch=199-step=49600.ckpt \
  --hidden_dims 512 256 128 \
  --batch_size 24 \
  --lr 1e-4 \
  --max_epochs 50 \
  --early_stopping \
  --device 0
```

Or use the provided shell script:
```bash
bash run_training_sentiment_cls_with_mlp.sh
```

### 3. GLIM Model Enhancement

**New Method: `extract_embeddings_with_grad()`**

Added to `model/glim.py` to support gradient-enabled embedding extraction:

```python
# For end-to-end training (with gradients)
embeddings = glim_model.extract_embeddings_with_grad(eeg, prompts=prompts)

# For cached embedding extraction (without gradients) - OLD METHOD
embeddings = glim_model.extract_embeddings(eeg, prompts=prompts)
```

**Important:** The existing `extract_embeddings()` method retains its `@torch.no_grad()` decorator to ensure backward compatibility with the existing `train_mlp.py` script.

## Comparison: Old vs New Approach

### Old Approach (`SentimentMLP` + `train_mlp.py`)

1. **Two-stage process:**
   - Stage 1: Extract embeddings from frozen GLIM encoder
   - Stage 2: Train MLP on cached embeddings

2. **Characteristics:**
   - No gradients through GLIM encoder
   - Embeddings cached to disk for efficiency
   - Faster training (no encoder forward pass)
   - Cannot fine-tune encoder

3. **Use case:** When you have a well-trained GLIM encoder and want to quickly train a classifier

### New Approach (`SentimentCLSWithMLP` + `train_sentiment_cls_with_mlp.py`)

1. **End-to-end training:**
   - Single-stage: Train encoder and classifier together
   - Gradients flow through both components

2. **Characteristics:**
   - Gradients through GLIM encoder (fine-tuning)
   - No embedding caching (computed on-the-fly)
   - Slower training (full forward/backward pass)
   - Can adapt encoder to sentiment task

3. **Use case:** When you want to fine-tune the encoder for better sentiment classification performance

## Key Differences in Implementation

### Data Flow

**Old approach (train_mlp.py):**
```
Raw EEG → GLIM.extract_embeddings() [no_grad] → Cache → MLP → Loss
                                                               ↓
                                                          Backprop (MLP only)
```

**New approach (train_sentiment_cls_with_mlp.py):**
```
Raw EEG → GLIM.extract_embeddings_with_grad() → MLP → Loss
    ↑                                                    ↓
    └────────────────── Backprop (Both) ────────────────┘
```

### Learning Rates

The new model uses differential learning rates:
- **Encoder (GLIM):** `lr * 0.1` (lower for fine-tuning)
- **Classifier (MLP):** `lr` (full learning rate)

This prevents catastrophic forgetting of the pretrained encoder weights.

## Command-Line Arguments

### Common Arguments (both scripts)

- `--data_path`: Path to merged dataset pickle file
- `--glim_checkpoint`: Path to pretrained GLIM checkpoint
- `--hidden_dims`: MLP hidden layer dimensions (e.g., `512 256 128`)
- `--dropout`: Dropout rate (default: 0.3)
- `--batch_size`: Training batch size
- `--lr`: Learning rate
- `--max_epochs`: Maximum training epochs
- `--early_stopping`: Enable early stopping
- `--patience`: Early stopping patience
- `--device`: GPU device index

### New Model Specific Arguments

- `--freeze_encoder`: Freeze GLIM encoder (only train MLP)
  - If set, behaves similar to old approach but without caching

## Weighted Sampling

Both scripts support weighted sampling to handle class imbalance:

- **train_mlp.py:** Optional, controlled by `use_weighted_sampler` parameter
- **train_sentiment_cls_with_mlp.py:** **Enabled by default**

The weighted sampler ensures balanced representation of sentiment classes during training, which is crucial for imbalanced datasets.

## Testing

A validation script is provided to verify the implementation:

```bash
python test_implementation.py
```

This tests:
1. Old method still has `@torch.no_grad()` decorator
2. New method exists without `@torch.no_grad()`
3. Old script uses non-gradient method
4. New model uses gradient-enabled method
5. Weighted sampling is enabled
6. All files exist and are correctly structured

## Files Modified/Created

### New Files
- `model/sentiment_cls_with_mlp.py` - End-to-end model
- `train_sentiment_cls_with_mlp.py` - Training script
- `run_training_sentiment_cls_with_mlp.sh` - Shell script
- `test_implementation.py` - Validation tests
- `IMPLEMENTATION_GUIDE.md` - This documentation

### Modified Files
- `model/glim.py` - Added `extract_embeddings_with_grad()` method

### Unchanged Files
- `model/sentiment_mlp.py` - Original MLP-only model
- `train_mlp.py` - Original training script (still works as before)
- `data/datamodule.py` - Data module with weighted sampling support

## Backward Compatibility

✅ **The existing `train_mlp.py` script continues to work exactly as before.**

The old script still uses `extract_embeddings()` with `@torch.no_grad()`, ensuring:
- Embeddings are extracted without gradients
- Embeddings are cached to disk for efficiency
- Training behavior is unchanged

## Performance Considerations

### Old Approach (Cached Embeddings)
- **Pros:**
  - Faster training (no encoder forward pass)
  - Lower memory usage
  - Can train multiple MLP variants quickly
- **Cons:**
  - Encoder frozen (no adaptation)
  - Requires storage space for cached embeddings

### New Approach (End-to-End)
- **Pros:**
  - Encoder can adapt to sentiment task
  - Potentially better performance through fine-tuning
  - No caching required
- **Cons:**
  - Slower training (full model forward pass)
  - Higher memory usage (gradients for entire model)
  - Risk of overfitting without proper regularization

## Recommendations

### Use Old Approach When:
- You have limited GPU memory
- You want to experiment with different MLP architectures quickly
- Your GLIM encoder is already well-trained for sentiment
- You need faster iteration during hyperparameter tuning

### Use New Approach When:
- You have sufficient GPU memory
- You want to fine-tune encoder for better performance
- Your dataset differs significantly from GLIM's training data
- You're willing to invest more training time for better results

## Example Workflows

### Workflow 1: Quick Experimentation (Old Approach)
```bash
# Extract embeddings once
python train_mlp.py \
  --glim_checkpoint ./checkpoints/glim.ckpt \
  --embeddings_cache_dir ./cache \
  --batch_size 64 \
  --max_epochs 50

# Embeddings are cached, subsequent runs are faster
```

### Workflow 2: Fine-Tuning (New Approach)
```bash
# End-to-end training with fine-tuning
python train_sentiment_cls_with_mlp.py \
  --glim_checkpoint ./checkpoints/glim.ckpt \
  --batch_size 24 \
  --lr 1e-4 \
  --max_epochs 50 \
  --early_stopping
```

### Workflow 3: Hybrid Approach
```bash
# Step 1: Quick baseline with old approach
python train_mlp.py --glim_checkpoint ./checkpoints/glim.ckpt

# Step 2: Fine-tune with new approach if baseline is promising
python train_sentiment_cls_with_mlp.py \
  --glim_checkpoint ./checkpoints/glim.ckpt \
  --lr 5e-5  # Lower LR for fine-tuning
```

## Troubleshooting

### Issue: Out of Memory (OOM)
**Solution:** 
- Reduce batch size: `--batch_size 16` or lower
- Use mixed precision: `--precision bf16-mixed`
- Consider freezing encoder: `--freeze_encoder`

### Issue: Training is too slow
**Solution:**
- Use old approach with cached embeddings
- Or freeze encoder: `--freeze_encoder`
- Reduce model size: fewer/smaller hidden layers

### Issue: Overfitting
**Solution:**
- Enable early stopping: `--early_stopping --patience 10`
- Increase dropout: `--dropout 0.5`
- Reduce learning rate: `--lr 5e-5`
- Consider freezing encoder initially

## Future Enhancements

Potential improvements to consider:
1. Support for different encoder architectures
2. Multi-task learning (sentiment + other tasks)
3. Layer-wise learning rate decay
4. Gradual unfreezing strategy
5. Knowledge distillation from fine-tuned to cached model

## References

- Original GLIM paper: [Link to paper if available]
- PyTorch Lightning documentation: https://lightning.ai/docs/pytorch/
- Weighted sampling: https://pytorch.org/docs/stable/data.html#torch.utils.data.WeightedRandomSampler
