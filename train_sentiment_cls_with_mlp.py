"""
Training script for end-to-end sentiment classifier (GLIM encoder + MLP).

This script trains both the GLIM encoder and MLP classifier together with:
1. Raw EEG data (not pre-extracted embeddings)
2. Gradients flowing through both encoder and classifier
3. Weighted sampling by default for class balance
4. Option to load pretrained GLIM checkpoint for fine-tuning
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
from model.sentiment_cls_with_mlp import SentimentCLSWithMLP
from model.glim import GLIM

# Configuration constants (same as train_with_embeddings.py)
SUPPORTED_TEXT_MODELS = [
    'google/flan-t5-xl',
    'google/flan-t5-large',
    'facebook/bart-large-cnn',
    'jbochi/madlad400-3b-mt'
]

# Model defaults
DEFAULT_INPUT_EEG_LEN = 1280
DEFAULT_HIDDEN_EEG_LEN = 96
DEFAULT_INPUT_TEXT_LEN = 96
DEFAULT_TGT_TEXT_LEN = 64
DEFAULT_INPUT_DIM = 128


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train end-to-end sentiment classifier (GLIM + MLP)'
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
        default=None,
        help='Path to pre-trained GLIM checkpoint for initialization (optional)'
    )
    
    # GLIM model arguments (used when glim_checkpoint is not provided)
    parser.add_argument(
        '--text_model',
        type=str,
        default='google/flan-t5-large',
        choices=SUPPORTED_TEXT_MODELS,
        help='Pre-trained text model to use'
    )
    parser.add_argument(
        '--hidden_dim',
        type=int,
        default=128,
        help='Hidden dimension size for GLIM'
    )
    parser.add_argument(
        '--embed_dim',
        type=int,
        default=1024,
        help='Embedding dimension size for GLIM'
    )
    parser.add_argument(
        '--n_in_blocks',
        type=int,
        default=6,
        help='Number of encoder blocks'
    )
    parser.add_argument(
        '--n_out_blocks',
        type=int,
        default=6,
        help='Number of decoder blocks'
    )
    parser.add_argument(
        '--num_heads',
        type=int,
        default=8,
        help='Number of attention heads'
    )
    parser.add_argument(
        '--glim_dropout',
        type=float,
        default=0.1,
        help='Dropout rate for GLIM'
    )
    parser.add_argument(
        '--warm_up_step',
        type=int,
        default=0,
        help='Warm up step for Cosine loss (0 means no warm-up)'
    )
    parser.add_argument(
        '--full_val_interval',
        type=int,
        default=10,
        help='Interval for full validation with generation'
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
        help='Dropout rate for MLP classifier'
    )
    parser.add_argument(
        '--freeze_encoder',
        action='store_true',
        help='Freeze GLIM encoder weights (only train MLP)'
    )
    parser.add_argument(
        '--do_not_use_prompt',
        action='store_true',
        help='Whether or not to use prompt embeddings'
    )
    
    # Loss weight arguments
    parser.add_argument(
        '--clip_loss_weight',
        type=float,
        default=0.5,
        help='Weight for contrastive (CLIP) loss'
    )
    parser.add_argument(
        '--lm_loss_weight',
        type=float,
        default=0.5,
        help='Weight for language model loss'
    )
    parser.add_argument(
        '--commitment_loss_weight',
        type=float,
        default=0.0,
        help='Weight for commitment loss'
    )
    parser.add_argument(
        '--mlp_loss_weight',
        type=float,
        default=0.5,
        help='Weight for MLP classification loss'
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
        help='Learning rate for classifier (encoder uses 10x lower)'
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
        default='./logs_sentiment_cls_with_mlp',
        help='Directory for TensorBoard logs'
    )
    parser.add_argument(
        '--experiment_name',
        type=str,
        default='sentiment_cls_with_mlp',
        help='Name of the experiment'
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='./checkpoints_sentiment_cls_with_mlp',
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
    if datamodule.use_weighted_sampler:
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
    print("End-to-End Sentiment Classification Training (GLIM + MLP)")
    print("=" * 80)
    print(f"Data path: {args.data_path}")
    print(f"GLIM checkpoint: {args.glim_checkpoint}")
    print(f"Hidden dims: {args.hidden_dims}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Freeze encoder: {args.freeze_encoder}")
    print(f"Use Prompt: {not args.do_not_use_prompt}")
    print(f"Loss weights - clip: {args.clip_loss_weight}, lm: {args.lm_loss_weight}, commitment: {args.commitment_loss_weight}, mlp: {args.mlp_loss_weight}")
    print("=" * 80)
    
    # Check if files exist
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found: {args.data_path}")
    
    if args.glim_checkpoint is not None and not os.path.exists(args.glim_checkpoint):
        raise FileNotFoundError(f"GLIM checkpoint not found: {args.glim_checkpoint}")
    
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
    print("\nInitializing end-to-end sentiment classifier...")
    
    # Either load from checkpoint or create a new GLIM model
    glim_model = None
    if args.glim_checkpoint is not None:
        print(f"Loading GLIM encoder from checkpoint: {args.glim_checkpoint}")
    else:
        print("Creating new GLIM model (using text encoder for alignment)...")
        print(f"  Text model: {args.text_model}")
        print(f"  Hidden dim: {args.hidden_dim}")
        print(f"  Embed dim: {args.embed_dim}")
        print(f"  N in blocks: {args.n_in_blocks}")
        print(f"  N out blocks: {args.n_out_blocks}")
        print(f"  Num heads: {args.num_heads}")
        print(f"  GLIM dropout: {args.glim_dropout}")
        print(f"  Use prompt: {not args.do_not_use_prompt}")
        
        glim_model = GLIM(
            input_eeg_len=DEFAULT_INPUT_EEG_LEN,
            hidden_eeg_len=DEFAULT_HIDDEN_EEG_LEN,
            input_text_len=DEFAULT_INPUT_TEXT_LEN,
            tgt_text_len=DEFAULT_TGT_TEXT_LEN,
            input_dim=DEFAULT_INPUT_DIM,
            hidden_dim=args.hidden_dim,
            embed_dim=args.embed_dim,
            text_model_id=args.text_model,
            n_in_blocks=args.n_in_blocks,
            n_out_blocks=args.n_out_blocks,
            num_heads=args.num_heads,
            dropout=args.glim_dropout,
            clip_loss_weight=args.clip_loss_weight,
            bsz_train=args.batch_size,
            bsz_val=args.val_batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            warm_up_step=args.warm_up_step if args.warm_up_step > 0 else None,
            full_val_interval=args.full_val_interval,
            use_prompt=(not args.do_not_use_prompt)
        )

    glim_model.setup(None)
    
    model = SentimentCLSWithMLP(
        glim_checkpoint=args.glim_checkpoint,
        glim_model=glim_model,
        hidden_dims=args.hidden_dims,
        num_classes=3,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        freeze_encoder=args.freeze_encoder,
        clip_loss_weight=args.clip_loss_weight,
        lm_loss_weight=args.lm_loss_weight,
        commitment_loss_weight=args.commitment_loss_weight,
        mlp_loss_weight=args.mlp_loss_weight,
        batch_size=args.batch_size
    )
    
    embed_dim = model.embed_dim
    print(f"Model created with:")
    print(f"  GLIM embedding dim: {embed_dim}")
    print(f"  MLP architecture: {embed_dim} -> {' -> '.join(map(str, args.hidden_dims))} -> 3")
    
    if args.freeze_encoder:
        print(f"  Encoder: FROZEN (only training MLP classifier)")
    else:
        print(f"  Encoder: TRAINABLE (fine-tuning with 10x lower LR)")
    
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