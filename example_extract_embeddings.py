"""
Example script demonstrating how to use GLIM's extract_embeddings method.

This shows the simplified way to extract embeddings with minimal setup.
"""

import torch
from model.glim import GLIM

# Load pre-trained GLIM model
print("Loading GLIM model...")
model = GLIM.load_from_checkpoint('path/to/checkpoint.ckpt')
model.eval()

# Example 1: Extract embeddings with default prompts (simplest usage)
print("\n=== Example 1: Minimal usage with default prompts ===")

# Your EEG data
eeg_data = torch.randn(4, 1280, 128)  # (batch_size, seq_len, channels)
eeg_mask = torch.ones(4, 1280)  # (batch_size, seq_len)

# Extract embeddings - that's it!
embeddings = model.extract_embeddings(eeg_data, eeg_mask)
print(f"Extracted embeddings shape: {embeddings.shape}")  # (4, 1024)


# Example 2: Extract embeddings with custom prompts
print("\n=== Example 2: With custom prompts ===")

# Define prompts as list of tuples (task, dataset, subject)
prompts = [
    ('task1', 'ZuCo1', 'ZAB'),
    ('task2', 'ZuCo2', 'ZDM'),
    ('task1', 'ZuCo1', 'ZGW'),
    ('task3', 'ZuCo2', 'ZKB')
]

embeddings = model.extract_embeddings(eeg_data, eeg_mask, prompts)
print(f"Extracted embeddings shape: {embeddings.shape}")


# Example 3: Use embeddings with MLP classifier
print("\n=== Example 3: Use with MLP classifier ===")

from model.sentiment_mlp import SentimentMLP

# Load MLP classifier
mlp = SentimentMLP(input_dim=1024, hidden_dims=[512, 256], num_classes=3)
mlp.eval()

# Extract embeddings
embeddings = model.extract_embeddings(eeg_data, eeg_mask)

# Get predictions
with torch.no_grad():
    logits = mlp(embeddings)
    predictions = torch.argmax(logits, dim=1)
    
print(f"Predictions: {predictions}")
print(f"Predicted labels: {[mlp.sentiment_labels[p] for p in predictions]}")


# Example 4: Batch processing
print("\n=== Example 4: Process multiple batches ===")

from torch.utils.data import DataLoader, TensorDataset

# Create a simple dataset
dataset = TensorDataset(
    torch.randn(100, 1280, 128),  # EEG data
    torch.ones(100, 1280)  # masks
)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

all_embeddings = []
for eeg_batch, mask_batch in dataloader:
    embeddings = model.extract_embeddings(eeg_batch, mask_batch)
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
print("\nJust provide EEG data, mask, and optionally prompts!")
