"""
Training script for the MLP sentiment classifier using GLIM embeddings.

This script:
1. Loads a pre-trained GLIM model
2. Extracts EEG embeddings from GLIM
3. Trains an MLP classifier on these embeddings for sentiment classification
"""

import os
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter

import torch
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar
)

from data.datamodule import GLIMDataModule
from model.glim import GLIM
from model.sentiment_mlp import SentimentMLP


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train MLP sentiment classifier using GLIM embeddings'
    )
    
    # Data arguments
    parser.add_argument(
        '--data_path',
        type=str,
        default='./data/tmp/zuco_merged.df',
        help='Path to the merged dataset pickle file'
    )
    parser.add_argument(
        '--glim_checkpoint',
        type=str,
        required=True,
        help='Path to pre-trained GLIM checkpoint'
    )
    parser.add_argument(
        '--embeddings_cache_dir',
        type=str,
        default='./data/embeddings_cache',
        help='Directory to cache extracted embeddings'
    )
    parser.add_argument(
        '--force_recompute',
        action='store_true',
        help='Force recomputation of embeddings even if cache exists'
    )
    
    # MLP model arguments
    parser.add_argument(
        '--hidden_dims',
        type=int,
        nargs='+',
        default=[512, 256],
        help='Hidden layer dimensions for MLP'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.3,
        help='Dropout rate'
    )
    
    # Training arguments
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Training batch size'
    )
    parser.add_argument(
        '--val_batch_size',
        type=int,
        default=32,
        help='Validation batch size'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='Learning rate'
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=1e-4,
        help='Weight decay for optimizer'
    )
    parser.add_argument(
        '--max_epochs',
        type=int,
        default=50,
        help='Maximum number of training epochs'
    )
    
    # Logging and checkpoint arguments
    parser.add_argument(
        '--log_dir',
        type=str,
        default='./logs_mlp',
        help='Directory for TensorBoard logs'
    )
    parser.add_argument(
        '--experiment_name',
        type=str,
        default='sentiment_mlp',
        help='Name of the experiment'
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='./checkpoints_mlp',
        help='Directory to save model checkpoints'
    )
    
    # Hardware arguments
    parser.add_argument(
        '--accelerator',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'gpu', 'tpu'],
        help='Accelerator type'
    )
    parser.add_argument(
        '--device',
        type=int,
        default=1,
        help='GPU Index'
    )
    parser.add_argument(
        '--precision',
        type=str,
        default='32',
        choices=['32', '16-mixed', 'bf16-mixed'],
        help='Training precision'
    )
    
    # Early stopping arguments
    parser.add_argument(
        '--early_stopping',
        action='store_true',
        help='Enable early stopping'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=10,
        help='Early stopping patience'
    )
    
    # Misc
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of data loading workers'
    )
    
    return parser.parse_args()


class EmbeddingDataModule(L.LightningDataModule):
    """
    DataModule that extracts embeddings from GLIM and provides them for MLP training.
    Supports caching embeddings to disk to avoid recomputation.
    """
    
    def __init__(self, glim_model, base_datamodule, batch_size=64, num_workers=4, 
                 cache_dir=None, force_recompute=False):
        super().__init__()
        self.glim_model = glim_model
        self.base_datamodule = base_datamodule
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_dir = cache_dir
        self.force_recompute = force_recompute
        
        # Create cache directory if specified
        if self.cache_dir:
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Set GLIM to eval mode and freeze it
        self.glim_model.eval()
        for param in self.glim_model.parameters():
            param.requires_grad = False
    
    def setup(self, stage=None):
        """Setup datasets with pre-computed embeddings."""
        # Setup base datamodule first
        self.base_datamodule.setup(stage)
        
        if stage == 'fit' or stage is None:
            self.train_embeddings = self._extract_embeddings(
                self.base_datamodule.train_dataloader(), 
                split_name='train'
            )
            self.val_embeddings = self._extract_embeddings(
                self.base_datamodule.val_dataloader(),
                split_name='val'
            )
        
        if stage == 'test' or stage is None:
            self.test_embeddings = self._extract_embeddings(
                self.base_datamodule.test_dataloader(),
                split_name='test'
            )
    
    def _get_cache_path(self, split_name):
        """Get the cache file path for a specific split."""
        if not self.cache_dir:
            return None
        
        # Create a cache filename based on the GLIM checkpoint and split
        cache_filename = f"embeddings_{split_name}.pt"
        return Path(self.cache_dir) / cache_filename
    
    def _load_cached_embeddings(self, split_name):
        """Load cached embeddings from disk if available."""
        cache_path = self._get_cache_path(split_name)
        
        if not cache_path or self.force_recompute:
            return None
        
        if cache_path.exists():
            print(f"Loading cached {split_name} embeddings from {cache_path}")
            try:
                cached_data = torch.load(cache_path)
                print(f"  Loaded {cached_data['embeddings'].shape[0]} cached embeddings")
                
                # Create TensorDataset
                dataset = torch.utils.data.TensorDataset(
                    cached_data['embeddings'],
                    cached_data['sentiment_ids']
                )
                return dataset
            except Exception as e:
                print(f"  Warning: Failed to load cache ({e}), will recompute embeddings")
                return None
        
        return None
    
    def _save_embeddings_to_cache(self, embeddings, sentiment_ids, split_name):
        """Save embeddings to disk cache."""
        cache_path = self._get_cache_path(split_name)
        
        if not cache_path:
            return
        
        print(f"Saving {split_name} embeddings to cache: {cache_path}")
        try:
            torch.save({
                'embeddings': embeddings,
                'sentiment_ids': sentiment_ids,
            }, cache_path)
            print(f"  Successfully cached {embeddings.shape[0]} embeddings")
        except Exception as e:
            print(f"  Warning: Failed to save cache ({e})")
    
    @torch.no_grad()
    def _extract_embeddings(self, dataloader, split_name=''):
        """Extract embeddings from GLIM for the entire dataset using simplified method."""
        # Try to load from cache first
        cached_dataset = self._load_cached_embeddings(split_name)
        if cached_dataset is not None:
            return cached_dataset
        
        # Cache not available or force recompute, extract embeddings
        print(f"Extracting {split_name} embeddings from GLIM (no masking applied for MLP classifier)...")
        embeddings_list = []
        sentiment_ids_list = []
        
        self.glim_model.eval()
        device = next(self.glim_model.parameters()).device
        
        for batch_idx, batch in enumerate(dataloader):
            # Extract required inputs with minimal processing
            eeg = batch['eeg'].to(device)  # (n, l, c)
            prompts = batch['prompt']  # list of lists of three prompts
            sentiment_label = batch['sentiment label']  # list of str
            
            # Use the simplified extract_embeddings method
            # Note: No mask is applied - all timesteps are used for embedding extraction
            eeg_emb_vector = self.glim_model.extract_embeddings(eeg, prompts=prompts)
            
            # Encode sentiment labels
            sentiment_ids = self.glim_model.encode_labels(sentiment_label)
            
            embeddings_list.append(eeg_emb_vector.cpu())
            sentiment_ids_list.append(sentiment_ids.cpu())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1} batches...")
        
        # Concatenate all embeddings
        all_embeddings = torch.cat(embeddings_list, dim=0)
        all_sentiment_ids = torch.cat(sentiment_ids_list, dim=0)
        
        print(f"Extracted {all_embeddings.shape[0]} embeddings of dimension {all_embeddings.shape[1]}")
        
        # Save to cache
        self._save_embeddings_to_cache(all_embeddings, all_sentiment_ids, split_name)
        
        # Create TensorDataset
        dataset = torch.utils.data.TensorDataset(all_embeddings, all_sentiment_ids)
        return dataset
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_embeddings,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=custom_collate_fn
        )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_embeddings,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=custom_collate_fn
        )
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_embeddings,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=custom_collate_fn
        )


def custom_collate_fn(batch):
    """Custom collate function to create proper batch format."""
    embeddings, sentiment_ids = zip(*batch)
    embeddings = torch.stack(embeddings)
    sentiment_ids = torch.stack(sentiment_ids)
    
    return {
        'eeg_emb_vector': embeddings,
        'sentiment_ids': sentiment_ids
    }


def display_label_distribution(datamodule, glim_model):
    """Display sentiment label distribution in the training dataset."""
    print("\n" + "=" * 80)
    print("Sentiment Label Distribution in Training Data")
    print("=" * 80)
    
    # Setup the base datamodule to access training data
    datamodule.base_datamodule.setup('fit')
    
    # Get training dataset
    train_dataset = datamodule.base_datamodule.train_set
    
    # Collect all sentiment labels
    sentiment_labels = []
    for i in range(len(train_dataset)):
        sample = train_dataset[i]
        sentiment_label = sample['sentiment label']
        sentiment_labels.append(sentiment_label)
    
    # Encode labels to IDs using GLIM's encoder
    sentiment_ids = glim_model.encode_labels(sentiment_labels).cpu().numpy()
    
    # Count the distribution
    label_counts = Counter(sentiment_ids)
    
    # Map IDs to label names
    id_to_label = {0: 'negative', 1: 'neutral', 2: 'positive'}
    
    print(f"\nTotal samples: {len(sentiment_ids)}")
    print("\nLabel distribution:")
    for label_id in sorted(label_counts.keys()):
        label_name = id_to_label.get(label_id, f'unknown({label_id})')
        count = label_counts[label_id]
        percentage = (count / len(sentiment_ids)) * 100
        print(f"  {label_name} (ID: {label_id}): {count} samples ({percentage:.2f}%)")
    
    # Display class weights if weighted sampler is enabled
    if datamodule.base_datamodule.use_weighted_sampler:
        print("\nWeighted sampling enabled - computed class weights:")
        total_samples = len(sentiment_ids)
        num_classes = len(label_counts)
        for label_id in sorted(label_counts.keys()):
            label_name = id_to_label.get(label_id, f'unknown({label_id})')
            count = label_counts[label_id]
            class_weight = total_samples / (num_classes * count)
            print(f"  {label_name} (ID: {label_id}): weight = {class_weight:.4f}")
    
    print("=" * 80)


def display_confusion_matrix(mlp_model):
    """Display the confusion matrix after testing."""
    if not hasattr(mlp_model, 'confusion_matrix'):
        print("\nWarning: No confusion matrix found. Make sure test() was called.")
        return
    
    cm = mlp_model.confusion_matrix
    
    print("\n" + "=" * 80)
    print("Confusion Matrix (Test Set)")
    print("=" * 80)
    print("\nRows represent true labels, columns represent predictions")
    print(f"Label mapping: 0=negative, 1=neutral, 2=positive\n")
    
    # Print column headers
    print("           Predicted")
    print("         ", end="")
    for i in range(3):
        print(f"{i:>8}", end="")
    print()
    
    # Print matrix with row labels
    print("Actual")
    for i, row in enumerate(cm):
        print(f"  {i}   ", end="")
        for val in row:
            print(f"{val:>8}", end="")
        print()
    
    # Calculate per-class metrics
    print("\nPer-class metrics:")
    label_names = ['negative', 'neutral', 'positive']
    for i, label_name in enumerate(label_names):
        # True positives, false positives, false negatives
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        
        # Precision, recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"  {label_name}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    
    print("=" * 80)


def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seed
    L.seed_everything(args.seed, workers=True)
    
    # Create directories
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("MLP Sentiment Classification Training using GLIM Embeddings")
    print("=" * 80)
    print(f"Data path: {args.data_path}")
    print(f"GLIM checkpoint: {args.glim_checkpoint}")
    print(f"Hidden dims: {args.hidden_dims}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Max epochs: {args.max_epochs}")
    print("=" * 80)
    
    # Check if files exist
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found: {args.data_path}")
    
    if not os.path.exists(args.glim_checkpoint):
        raise FileNotFoundError(f"GLIM checkpoint not found: {args.glim_checkpoint}")
    
    # Load pre-trained GLIM model
    print("\nLoading pre-trained GLIM model...")
    if torch.cuda.is_available():
        # make sure we put this model somewhere right
        device = f"cuda:{args.device}"
    else:
        device = "cpu"
    glim_model = GLIM.load_from_checkpoint(args.glim_checkpoint, map_location = device)
    glim_model.eval()
    print("GLIM model loaded successfully!")
    
    # Get embed_dim from GLIM
    embed_dim = glim_model.embed_dim
    print(f"GLIM embedding dimension: {embed_dim}")
    
    # Initialize base data module with weighted sampling enabled for MLP training
    print("\nInitializing base data module...")
    base_datamodule = GLIMDataModule(
        data_path=args.data_path,
        eval_noise_input=False,
        bsz_train=args.batch_size,
        bsz_val=args.val_batch_size,
        bsz_test=args.val_batch_size,
        num_workers=args.num_workers,
        use_weighted_sampler=True  # Enable weighted sampling for balanced batches
    )
    print("Weighted sampling enabled in GLIMDataModule to handle class imbalance")
    
    # Create embedding data module
    print("Creating embedding data module...")
    print(f"Embeddings cache directory: {args.embeddings_cache_dir}")
    if args.force_recompute:
        print("Force recompute enabled - will ignore cached embeddings")
    
    embedding_datamodule = EmbeddingDataModule(
        glim_model=glim_model,
        base_datamodule=base_datamodule,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_dir=args.embeddings_cache_dir,
        force_recompute=args.force_recompute
    )
    
    # Display sentiment label distribution before training
    display_label_distribution(embedding_datamodule, glim_model)
    
    # Initialize MLP model
    print("\nInitializing MLP sentiment classifier...")
    mlp_model = SentimentMLP(
        input_dim=embed_dim,
        hidden_dims=args.hidden_dims,
        num_classes=3,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    print(f"MLP model created with architecture: {embed_dim} -> {' -> '.join(map(str, args.hidden_dims))} -> 3")
    
    # Setup logger
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    version_name = f'{args.experiment_name}_{timestamp}'
    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name=args.experiment_name,
        version=version_name,
        default_hp_metric=False
    )
    print(f"\nTensorBoard logs: {logger.log_dir}")
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(args.checkpoint_dir, args.experiment_name),
            filename='mlp-{epoch:02d}-{val/accuracy:.4f}',
            monitor='val/accuracy',
            mode='max',
            save_top_k=3,
            save_last=True,
            verbose=True
        ),
        LearningRateMonitor(logging_interval='epoch'),
        RichProgressBar()
    ]
    
    if args.early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor='val/loss',
                min_delta=0.001,
                patience=args.patience,
                mode='min',
                verbose=True
            )
        )
        print(f"Early stopping enabled with patience={args.patience}")
    
    # Initialize trainer
    print("\nInitializing trainer...")
    if torch.cuda.is_available():
        device = [args.device]
    else:
        device = 1
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=device,
        precision=args.precision,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=10,
        val_check_interval=1.0,
        deterministic=True
    )
    
    # Log hyperparameters
    trainer.logger.log_hyperparams(vars(args))
    
    # Train the model
    print("\nStarting training...")
    print("=" * 80)
    trainer.fit(mlp_model, embedding_datamodule)
    
    # Print best model info
    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Best model checkpoint: {trainer.checkpoint_callback.best_model_path}")
    print(f"Best validation accuracy: {trainer.checkpoint_callback.best_model_score:.4f}")
    print("=" * 80)
    
    # Test the best model
    if trainer.checkpoint_callback.best_model_path:
        print("\nRunning test evaluation on best model...")
        trainer.test(mlp_model, embedding_datamodule, 
                    ckpt_path=trainer.checkpoint_callback.best_model_path)
        
        # Display confusion matrix after testing
        display_confusion_matrix(mlp_model)


if __name__ == '__main__':
    main()