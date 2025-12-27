"""
Example script demonstrating how to use GLIM's extract_embeddings method.

This shows the simplified way to extract embeddings with minimal setup.
Note: No mask is applied when extracting embeddings for MLP classifier.
"""

import torch
from model.glim import GLIM

# Load pre-trained GLIM model
print("Loading GLIM model...")
model = GLIM.load_from_checkpoint('./checkpoints/glim-zuco-epoch=199-step=49600.ckpt', map_location = 'cuda:0')
model.eval()

# Example 1: Extract embeddings without mask (simplest usage)
print("\n=== Example 1: Minimal usage (no masking) ===")

# Your EEG data
eeg_data = torch.randn(12, 1280, 128).to(torch.device('cuda:0'))  # (batch_size, seq_len, channels)

# Extract embeddings - no mask needed!
# All timesteps are used for embedding extraction
embeddings = model.extract_embeddings(eeg_data)
print(f"Extracted embeddings shape: {embeddings.shape}")  # (12, 1024)


# Example 2: Extract embeddings with custom prompts
print("\n=== Example 2: With custom prompts (no masking) ===")

# Define prompts as list of tuples (task, dataset, subject)
prompts = [
    ['<NR>','<NR>', '<NR>', '<TSR>', '<NR>','<NR>', '<NR>', '<TSR>', '<NR>','<NR>', '<NR>', '<TSR>'],
    ['ZuCo1', 'ZuCo2', 'ZuCo1', 'ZuCo2', 'ZuCo1', 'ZuCo2', 'ZuCo1', 'ZuCo2', 'ZuCo1', 'ZuCo2', 'ZuCo1', 'ZuCo2'],
    ['ZAB', 'ZDM', 'ZGW', 'ZKB', 'ZAB', 'ZDM', 'ZGW', 'ZKB', 'ZAB', 'ZDM', 'ZGW', 'ZKB']
]

embeddings = model.extract_embeddings(eeg_data, prompts=prompts)
print(f"Extracted embeddings shape: {embeddings.shape}")
print(f"Extracted embeddings [ : ,  :  20]:\n{embeddings[ : ,  :  20]}")

# Example 3: Use embeddings with MLP classifier
print("\n=== Example 3: Use with MLP classifier (no masking) ===")

from model.sentiment_mlp import SentimentMLP

# Load MLP classifier
mlp = SentimentMLP(input_dim=1024, hidden_dims=[512, 256], num_classes=3).to(torch.device('cuda:0'))
mlp.eval()

# Extract embeddings - no mask needed
embeddings = model.extract_embeddings(eeg_data)

# Get predictions
with torch.no_grad():
    logits = mlp(embeddings)
    predictions = torch.argmax(logits, dim=1)
    
print(f"Predictions: {predictions}")
print(f"Predicted labels: {[mlp.sentiment_labels[p] for p in predictions]}")


# Example 4: Batch processing
print("\n=== Example 4: Process multiple batches (no masking) ===")

from torch.utils.data import DataLoader, TensorDataset

# Create a simple dataset (only EEG data, no masks needed)
dataset = TensorDataset(torch.randn(100, 1280, 128))  # EEG data only
dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

all_embeddings = []
for eeg_batch, in dataloader:
    # Extract embeddings without mask - all timesteps are used
    embeddings = model.extract_embeddings(eeg_batch)
    all_embeddings.append(embeddings)

all_embeddings = torch.cat(all_embeddings, dim=0)
print(f"Total embeddings extracted: {all_embeddings.shape}")


print("\n=== Summary ===")
print("The extract_embeddings method provides a simple way to get embeddings")
print("without needing to:")
print("  - Call setup() method")
print("  - Provide target text or labels")
print("  - Handle complex batch dictionaries")
print("  - Deal with tokenization or text encoding")
print("  - Apply masking (all timesteps are used)")
print("\nJust provide EEG data and optionally prompts!")
