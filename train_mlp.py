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
        '--devices',
        type=int,
        default=1,
        help='Number of devices to use'
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
    """
    
    def __init__(self, glim_model, base_datamodule, batch_size=64, num_workers=4):
        super().__init__()
        self.glim_model = glim_model
        self.base_datamodule = base_datamodule
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Set GLIM to eval mode and freeze it
        self.glim_model.eval()
        for param in self.glim_model.parameters():
            param.requires_grad = False
    
    def setup(self, stage=None):
        """Setup datasets with pre-computed embeddings."""
        # Setup base datamodule first
        self.base_datamodule.setup(stage)
        
        if stage == 'fit' or stage is None:
            self.train_embeddings = self._extract_embeddings(self.base_datamodule.train_dataloader())
            self.val_embeddings = self._extract_embeddings(self.base_datamodule.val_dataloader())
        
        if stage == 'test' or stage is None:
            self.test_embeddings = self._extract_embeddings(self.base_datamodule.test_dataloader())
    
    @torch.no_grad()
    def _extract_embeddings(self, dataloader):
        """Extract embeddings from GLIM for the entire dataset."""
        print(f"Extracting embeddings from GLIM...")
        embeddings_list = []
        sentiment_ids_list = []
        
        self.glim_model.eval()
        device = next(self.glim_model.parameters()).device
        
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Get embeddings from GLIM
            shared_outputs = self.glim_model.shared_forward(batch)
            eeg_emb_vector = shared_outputs['eeg_emb_vector']  # (n, embed_dim)
            sentiment_ids = shared_outputs['sentiment_ids']  # (n,)
            
            embeddings_list.append(eeg_emb_vector.cpu())
            sentiment_ids_list.append(sentiment_ids.cpu())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1} batches...")
        
        # Concatenate all embeddings
        all_embeddings = torch.cat(embeddings_list, dim=0)
        all_sentiment_ids = torch.cat(sentiment_ids_list, dim=0)
        
        print(f"Extracted {all_embeddings.shape[0]} embeddings of dimension {all_embeddings.shape[1]}")
        
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
    glim_model = GLIM.load_from_checkpoint(args.glim_checkpoint)
    glim_model.eval()
    print("GLIM model loaded successfully!")
    
    # Get embed_dim from GLIM
    embed_dim = glim_model.embed_dim
    print(f"GLIM embedding dimension: {embed_dim}")
    
    # Initialize base data module
    print("\nInitializing base data module...")
    base_datamodule = GLIMDataModule(
        data_path=args.data_path,
        eval_noise_input=False,
        bsz_train=args.batch_size,
        bsz_val=args.val_batch_size,
        bsz_test=args.val_batch_size,
        num_workers=args.num_workers
    )
    
    # Create embedding data module
    print("Creating embedding data module...")
    embedding_datamodule = EmbeddingDataModule(
        glim_model=glim_model,
        base_datamodule=base_datamodule,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
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
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
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


if __name__ == '__main__':
    main()
