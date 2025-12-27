"""
Example demonstrating the embedding caching feature for MLP training.

This shows how embeddings are automatically cached to avoid reprocessing.
"""

import os
import time
from pathlib import Path

# Example 1: First run - extracts and caches embeddings
print("=" * 80)
print("Example 1: First Run (Extract and Cache)")
print("=" * 80)
print("""
On the first run, the script will:
1. Load the GLIM model
2. Extract embeddings from all data splits (train/val/test)
3. Save embeddings to cache directory (default: ./data/embeddings_cache/)

Command:
python train_mlp.py \\
  --data_path ./data/tmp/zuco_merged.df \\
  --glim_checkpoint ./checkpoints/sentiment_classification/last.ckpt \\
  --max_epochs 10

Expected output:
  Extracting train embeddings from GLIM (no masking applied for MLP classifier)...
    Processed 10 batches...
    Processed 20 batches...
  Extracted 1000 embeddings of dimension 1024
  Saving train embeddings to cache: ./data/embeddings_cache/embeddings_train.pt
    Successfully cached 1000 embeddings
  
  [Similar output for val and test splits]

Total time for embedding extraction: ~2-5 minutes (depending on dataset size)
""")

# Example 2: Subsequent runs - loads from cache
print("\n" + "=" * 80)
print("Example 2: Subsequent Runs (Load from Cache)")
print("=" * 80)
print("""
On subsequent runs with the same GLIM checkpoint, the script will:
1. Detect existing cached embeddings
2. Load embeddings instantly from cache
3. Skip the extraction step entirely

Command (same as before):
python train_mlp.py \\
  --data_path ./data/tmp/zuco_merged.df \\
  --glim_checkpoint ./checkpoints/sentiment_classification/last.ckpt \\
  --max_epochs 10

Expected output:
  Loading cached train embeddings from ./data/embeddings_cache/embeddings_train.pt
    Loaded 1000 cached embeddings
  Loading cached val embeddings from ./data/embeddings_cache/embeddings_val.pt
    Loaded 200 cached embeddings
  Loading cached test embeddings from ./data/embeddings_cache/embeddings_test.pt
    Loaded 300 cached embeddings

Total time for loading: ~1-2 seconds!

💡 Benefits:
  - 100x+ faster setup for training
  - Great for hyperparameter tuning
  - Great for experimenting with MLP architectures
  - No need to reprocess embeddings every time
""")

# Example 3: Force recomputation
print("\n" + "=" * 80)
print("Example 3: Force Recompute (Ignore Cache)")
print("=" * 80)
print("""
If you need to recompute embeddings (e.g., after updating GLIM model):

Command:
python train_mlp.py \\
  --data_path ./data/tmp/zuco_merged.df \\
  --glim_checkpoint ./checkpoints/sentiment_classification/last.ckpt \\
  --max_epochs 10 \\
  --force_recompute

This will:
1. Ignore existing cached embeddings
2. Recompute all embeddings from scratch
3. Overwrite the cache with new embeddings
""")

# Example 4: Custom cache directory
print("\n" + "=" * 80)
print("Example 4: Custom Cache Directory")
print("=" * 80)
print("""
You can specify a custom cache directory:

Command:
python train_mlp.py \\
  --data_path ./data/tmp/zuco_merged.df \\
  --glim_checkpoint ./checkpoints/sentiment_classification/last.ckpt \\
  --embeddings_cache_dir ./my_custom_cache \\
  --max_epochs 10

This is useful for:
- Organizing caches for different experiments
- Sharing caches across team members
- Using faster storage locations (e.g., SSD)
""")

# Example 5: Cache management
print("\n" + "=" * 80)
print("Example 5: Cache Management")
print("=" * 80)
print("""
Cache files location: ./data/embeddings_cache/

Files created:
- embeddings_train.pt  (~100-500 MB depending on dataset size)
- embeddings_val.pt
- embeddings_test.pt

To clear cache:
  rm -rf ./data/embeddings_cache/

To check cache size:
  du -sh ./data/embeddings_cache/

💡 Tips:
  - Cache is automatically excluded from git (.gitignore)
  - Each GLIM checkpoint should have its own cache directory
  - Cache format: PyTorch .pt files (embeddings + sentiment_ids)
""")

# Summary
print("\n" + "=" * 80)
print("Summary")
print("=" * 80)
print("""
The embedding caching feature makes MLP training much more efficient:

⏱️  First run: ~2-5 minutes (extract + cache)
⚡ Subsequent runs: ~1-2 seconds (load from cache)

Perfect for:
✅ Hyperparameter tuning (try different learning rates, dropouts, etc.)
✅ Architecture experiments (different hidden layer sizes)
✅ Quick iterations during development
✅ Reproducible experiments

Just run once, then experiment freely! 🚀
""")
