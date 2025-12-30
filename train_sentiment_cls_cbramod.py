"""
Training script for CBraMod-based sentiment classifier using GLIM dataloader.

This script trains the CBraMod backbone with MLP classifier for sentiment classification:
1. Uses GLIM dataloader to retrieve EEG signals
2. Processes EEG through CBraMod backbone
3. Uses weighted sampling for class imbalance
4. Uses CosineLR scheduler as in GLIM
5. Displays label distribution before training
6. Displays confusion matrix after testing
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
from model.sentiment_cls_cbramod import SentimentCLSCBraMod


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train CBraMod-based sentiment classifier using GLIM dataloader'
    )
    
    # Data arguments
    parser.add_argument(
        '--data_path',
        type=str,
        default='./data/tmp/zuco_merged.df',
        help='Path to the merged dataset pickle file'
    )
    
    # CBraMod arguments
    parser.add_argument(
        '--num_channels',
        type=int,
        default=105,
        help='Number of EEG channels for CBraMod'
    )
    parser.add_argument(
        '--patch_size',
        type=int,
        default=200,
        help='Size of each EEG patch (typically sampling rate)'
    )
    parser.add_argument(
        '--num_patches',
        type=int,
        default=1,
        help='Number of patches per channel'
    )
    parser.add_argument(
        '--d_model',
        type=int,
        default=200,
        help='Hidden dimension of CBraMod'
    )
    parser.add_argument(
        '--dim_feedforward',
        type=int,
        default=800,
        help='FFN hidden dimension in CBraMod'
    )
    parser.add_argument(
        '--n_layer',
        type=int,
        default=12,
        help='Number of transformer layers in CBraMod'
    )
    parser.add_argument(
        '--nhead',
        type=int,
        default=8,
        help='Number of attention heads in CBraMod'
    )
    parser.add_argument(
        '--pretrained_weights',
        type=str,
        default=None,
        help='Path to pretrained CBraMod weights'
    )
    
    # Sample rate arguments
    parser.add_argument(
        '--src_sample_rate',
        type=int,
        default=128,
        help='Source EEG sample rate from ZuCo preprocessing (default: 128Hz)'
    )
    parser.add_argument(
        '--tgt_sample_rate',
        type=int,
        default=200,
        help='Target sample rate expected by CBraMod (default: 200Hz)'
    )
    
    # MLP classifier arguments
    parser.add_argument(
        '--hidden_dims',
        type=int,
        nargs='+',
        default=[512, 256],
        help='Hidden layer dimensions for MLP classifier'
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
        default=24,
        help='Training batch size'
    )
    parser.add_argument(
        '--val_batch_size',
        type=int,
        default=24,
        help='Validation batch size'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=1e-4,
        help='Weight decay for optimizer'
    )
    parser.add_argument(
        '--warm_up_step',
        type=int,
        default=None,
        help='Number of warmup steps for CosineLR scheduler'
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
        default='./logs_sentiment_cls_cbramod',
        help='Directory for TensorBoard logs'
    )
    parser.add_argument(
        '--experiment_name',
        type=str,
        default='sentiment_cls_cbramod',
        help='Name of the experiment'
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='./checkpoints_sentiment_cls_cbramod',
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


def display_label_distribution(datamodule):
    """Display sentiment label distribution in the training dataset."""
    print("\n" + "=" * 80)
    print("Sentiment Label Distribution in Training Data")
    print("=" * 80)
    
    # Setup the datamodule to access training data
    datamodule.setup('fit')
    
    # Get training dataset
    train_dataset = datamodule.train_set
    
    # Collect all sentiment labels
    sentiment_labels = []
    for i in range(len(train_dataset)):
        sample = train_dataset[i]
        sentiment_label = sample['sentiment label']
        sentiment_labels.append(sentiment_label)
    
    # Map labels to IDs
    label_to_id = {'negative': 0, 'neutral': 1, 'positive': 2}
    sentiment_ids = [label_to_id.get(label, -1) for label in sentiment_labels]
    
    # Count the distribution
    label_counts = Counter(sentiment_ids)
    
    # Map IDs to label names
    id_to_label = {0: 'negative', 1: 'neutral', 2: 'positive'}
    
    print(f"\nTotal samples: {len(sentiment_ids)}")
    print("\nLabel distribution:")
    for label_id in sorted(label_counts.keys()):
        if label_id == -1:
            continue
        label_name = id_to_label.get(label_id, f'unknown({label_id})')
        count = label_counts[label_id]
        percentage = (count / len(sentiment_ids)) * 100
        print(f"  {label_name} (ID: {label_id}): {count} samples ({percentage:.2f}%)")
    
    # Display class weights if weighted sampler is enabled
    if getattr(datamodule, 'use_weighted_sampler', False):
        print("\nWeighted sampling enabled - computed class weights:")
        total_samples = len(sentiment_ids)
        num_classes = len([k for k in label_counts.keys() if k != -1])
        for label_id in sorted(label_counts.keys()):
            if label_id == -1:
                continue
            label_name = id_to_label.get(label_id, f'unknown({label_id})')
            count = label_counts[label_id]
            class_weight = total_samples / (num_classes * count)
            print(f"  {label_name} (ID: {label_id}): weight = {class_weight:.4f}")
    
    print("=" * 80)


def display_confusion_matrix(model):
    """Display the confusion matrix after testing."""
    if not hasattr(model, 'confusion_matrix'):
        print("\nWarning: No confusion matrix found. Make sure test() was called.")
        return
    
    cm = model.confusion_matrix
    
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
    print("CBraMod Sentiment Classification Training")
    print("=" * 80)
    print(f"Data path: {args.data_path}")
    print(f"CBraMod config:")
    print(f"  - num_channels: {args.num_channels}")
    print(f"  - patch_size: {args.patch_size}")
    print(f"  - num_patches: {args.num_patches}")
    print(f"  - d_model: {args.d_model}")
    print(f"  - n_layer: {args.n_layer}")
    print(f"  - nhead: {args.nhead}")
    if args.pretrained_weights:
        print(f"  - pretrained_weights: {args.pretrained_weights}")
    print(f"Sample rate: {args.src_sample_rate}Hz -> {args.tgt_sample_rate}Hz (resampling)")
    print(f"MLP hidden dims: {args.hidden_dims}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Max epochs: {args.max_epochs}")
    if args.warm_up_step:
        print(f"Warmup steps: {args.warm_up_step} (using CosineLR)")
    print("=" * 80)
    
    # Check if data file exists
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found: {args.data_path}")
    
    # Initialize data module with weighted sampling enabled by default
    print("\nInitializing data module with weighted sampling...")
    datamodule = GLIMDataModule(
        data_path=args.data_path,
        eval_noise_input=False,
        bsz_train=args.batch_size,
        bsz_val=args.val_batch_size,
        bsz_test=args.val_batch_size,
        num_workers=args.num_workers,
        use_weighted_sampler=True  # Enable weighted sampling by default
    )
    print("Weighted sampling enabled to handle class imbalance")
    
    # Display sentiment label distribution
    display_label_distribution(datamodule)
    
    # Initialize model
    print("\nInitializing CBraMod sentiment classifier...")
    
    model = SentimentCLSCBraMod(
        num_channels=args.num_channels,
        patch_size=args.patch_size,
        num_patches=args.num_patches,
        d_model=args.d_model,
        dim_feedforward=args.dim_feedforward,
        n_layer=args.n_layer,
        nhead=args.nhead,
        hidden_dims=args.hidden_dims,
        num_classes=3,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warm_up_step=args.warm_up_step,
        pretrained_weights=args.pretrained_weights,
        batch_size=args.batch_size,
        src_sample_rate=args.src_sample_rate,
        tgt_sample_rate=args.tgt_sample_rate
    )
    
    print(f"Model created with:")
    print(f"  CBraMod output dim: {args.d_model}")
    print(f"  MLP architecture: {args.d_model} -> {' -> '.join(map(str, args.hidden_dims))} -> 3")
    
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
            filename='model-{epoch:02d}-{val/accuracy:.4f}',
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
        device = 'auto'
    
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
    trainer.fit(model, datamodule)
    
    # Print best model info
    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Best model checkpoint: {trainer.checkpoint_callback.best_model_path}")
    print(f"Best validation accuracy: {trainer.checkpoint_callback.best_model_score:.4f}")
    print("=" * 80)
    
    # Test the best model
    if trainer.checkpoint_callback.best_model_path:
        print("\nRunning test evaluation on best model...")
        trainer.test(model, datamodule,
                    ckpt_path=trainer.checkpoint_callback.best_model_path)
        
        # Display confusion matrix after testing
        display_confusion_matrix(model)


if __name__ == '__main__':
    main()
