"""
Training script for GLIM_CLS model.

This script trains the combined GLIM encoder + MLP classifier model with:
1. Configurable classification labels and label keys
2. Cosine LR scheduler with warmup
3. Support for loading from checkpoint
4. bfloat16 precision for T5 model
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
from model.glim_cls import GLIM_CLS

# Configuration constants
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
DEFAULT_INPUT_DIM = 128


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train GLIM_CLS model (combined GLIM encoder + MLP classifier)'
    )
    
    # Data arguments
    parser.add_argument(
        '--data_path',
        type=str,
        default='./data/tmp/zuco_merged.df',
        help='Path to the merged dataset pickle file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint for resuming training or fine-tuning (optional)'
    )
    
    # Classification arguments
    parser.add_argument(
        '--classification_label_key',
        type=str,
        default='sentiment label',
        help='Key in batch dict for classification labels'
    )
    parser.add_argument(
        '--classification_labels',
        type=str,
        nargs='+',
        default=['negative', 'neutral', 'positive'],
        help='List of classification label names'
    )
    
    # GLIM encoder arguments
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
        '--encoder_dropout',
        type=float,
        default=0.1,
        help='Dropout rate for encoder'
    )
    parser.add_argument(
        '--do_not_use_prompt',
        action='store_true',
        help='Whether or not to use prompt embeddings'
    )
    
    # MLP classifier arguments
    parser.add_argument(
        '--mlp_hidden_dims',
        type=int,
        nargs='+',
        default=[512, 256],
        help='Hidden layer dimensions for MLP classifier'
    )
    parser.add_argument(
        '--mlp_dropout',
        type=float,
        default=0.3,
        help='Dropout rate for MLP classifier'
    )
    parser.add_argument(
        '--freeze_encoder',
        action='store_true',
        help='Freeze encoder weights (only train MLP)'
    )
    
    # Loss weight arguments
    parser.add_argument(
        '--clip_loss_weight',
        type=float,
        default=0.5,
        help='Weight for contrastive (CLIP) loss'
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
        '--max_epochs',
        type=int,
        default=50,
        help='Maximum number of training epochs'
    )
    
    # Optimizer arguments (Cosine LR with warmup)
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Maximum learning rate (classifier uses this, encoder uses 10x lower)'
    )
    parser.add_argument(
        '--min_lr',
        type=float,
        default=1e-6,
        help='Minimum learning rate for cosine schedule'
    )
    parser.add_argument(
        '--warmup_steps',
        type=int,
        default=100,
        help='Number of warmup steps for learning rate scheduler'
    )
    
    # Logging and checkpoint arguments
    parser.add_argument(
        '--log_dir',
        type=str,
        default='./logs_glim_cls',
        help='Directory for TensorBoard logs'
    )
    parser.add_argument(
        '--experiment_name',
        type=str,
        default='glim_cls',
        help='Name of the experiment'
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='./checkpoints_glim_cls',
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
        default=0,
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


def display_label_distribution(datamodule, classification_label_key, classification_labels):
    """Display classification label distribution in the training dataset."""
    print("\n" + "=" * 80)
    print(f"Classification Label Distribution ({classification_label_key})")
    print("=" * 80)
    
    datamodule.setup('fit')
    train_dataset = datamodule.train_set
    
    labels = []
    for i in range(len(train_dataset)):
        sample = train_dataset[i]
        label = sample.get(classification_label_key, None)
        if label is not None:
            labels.append(label)
    
    label_to_id = {label: i for i, label in enumerate(classification_labels)}
    label_ids = [label_to_id.get(label, -1) for label in labels]
    
    label_counts = Counter(label_ids)
    id_to_label = {i: label for i, label in enumerate(classification_labels)}
    
    print(f"\nTotal samples: {len(label_ids)}")
    print("\nLabel distribution:")
    for label_id in sorted(label_counts.keys()):
        if label_id == -1:
            print(f"  Unknown labels: {label_counts[label_id]} samples")
            continue
        label_name = id_to_label.get(label_id, f'unknown({label_id})')
        count = label_counts[label_id]
        percentage = (count / len(label_ids)) * 100
        print(f"  {label_name} (ID: {label_id}): {count} samples ({percentage:.2f}%)")
    
    print("=" * 80)


def display_confusion_matrix(model, classification_labels):
    """Display the confusion matrix after testing."""
    if not hasattr(model, 'confusion_matrix'):
        print("\nWarning: No confusion matrix found. Make sure test() was called.")
        return
    
    cm = model.confusion_matrix
    num_classes = len(classification_labels)
    
    print("\n" + "=" * 80)
    print("Confusion Matrix (Test Set)")
    print("=" * 80)
    print("\nRows represent true labels, columns represent predictions")
    print(f"Label mapping: {', '.join([f'{i}={label}' for i, label in enumerate(classification_labels)])}\n")
    
    print("           Predicted")
    print("         ", end="")
    for i in range(num_classes):
        print(f"{i:>8}", end="")
    print()
    
    print("Actual")
    for i, row in enumerate(cm):
        print(f"  {i}   ", end="")
        for val in row:
            print(f"{val:>8}", end="")
        print()
    
    print("\nPer-class metrics:")
    for i, label_name in enumerate(classification_labels):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"  {label_name}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    
    print("=" * 80)


def main():
    """Main training function."""
    args = parse_args()
    
    L.seed_everything(args.seed, workers=True)
    
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("GLIM_CLS Training (Combined GLIM Encoder + MLP Classifier)")
    print("=" * 80)
    print(f"Data path: {args.data_path}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Classification label key: {args.classification_label_key}")
    print(f"Classification labels: {args.classification_labels}")
    print(f"MLP hidden dims: {args.mlp_hidden_dims}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr} (max), {args.min_lr} (min)")
    print(f"Warmup steps: {args.warmup_steps}")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Freeze encoder: {args.freeze_encoder}")
    print(f"Use Prompt: {not args.do_not_use_prompt}")
    print(f"Loss weights - clip: {args.clip_loss_weight}, commitment: {args.commitment_loss_weight}, mlp: {args.mlp_loss_weight}")
    print("=" * 80)
    
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found: {args.data_path}")
    
    if args.checkpoint is not None and not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    print("\nInitializing data module with weighted sampling...")
    datamodule = GLIMDataModule(
        data_path=args.data_path,
        eval_noise_input=False,
        bsz_train=args.batch_size,
        bsz_val=args.val_batch_size,
        bsz_test=args.val_batch_size,
        num_workers=args.num_workers,
        use_weighted_sampler=True
    )
    print("Weighted sampling enabled to handle class imbalance")
    
    display_label_distribution(datamodule, args.classification_label_key, args.classification_labels)
    
    print("\nInitializing GLIM_CLS model...")
    
    if args.checkpoint is not None:
        print(f"Loading model from checkpoint: {args.checkpoint}")
        model = GLIM_CLS.load_from_checkpoint(args.checkpoint)
    else:
        print("Creating new GLIM_CLS model...")
        model = GLIM_CLS(
            input_eeg_len=DEFAULT_INPUT_EEG_LEN,
            hidden_eeg_len=DEFAULT_HIDDEN_EEG_LEN,
            input_text_len=DEFAULT_INPUT_TEXT_LEN,
            input_dim=DEFAULT_INPUT_DIM,
            hidden_dim=args.hidden_dim,
            embed_dim=args.embed_dim,
            text_model_id=args.text_model,
            n_in_blocks=args.n_in_blocks,
            n_out_blocks=args.n_out_blocks,
            num_heads=args.num_heads,
            dropout=args.encoder_dropout,
            use_prompt=(not args.do_not_use_prompt),
            batch_size=args.batch_size,
            
            # Classification arguments
            classification_label_key=args.classification_label_key,
            classification_labels=args.classification_labels,
            mlp_hidden_dims=args.mlp_hidden_dims,
            mlp_dropout=args.mlp_dropout,
            freeze_encoder=args.freeze_encoder,
            
            # Loss weights
            clip_loss_weight=args.clip_loss_weight,
            commitment_loss_weight=args.commitment_loss_weight,
            mlp_loss_weight=args.mlp_loss_weight,
            
            # Optimizer arguments
            lr=args.lr,
            min_lr=args.min_lr,
            warmup_steps=args.warmup_steps,
        )
    
    print(f"Model created with:")
    print(f"  Embedding dim: {model.embed_dim}")
    print(f"  MLP architecture: {model.embed_dim} -> {' -> '.join(map(str, args.mlp_hidden_dims))} -> {len(args.classification_labels)}")
    print(f"  Classification labels: {args.classification_labels}")
    
    if args.freeze_encoder:
        print(f"  Encoder: FROZEN (only training MLP classifier)")
    else:
        print(f"  Encoder: TRAINABLE (fine-tuning with 10x lower LR)")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    version_name = f'{args.experiment_name}_{timestamp}'
    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name=args.experiment_name,
        version=version_name,
        default_hp_metric=False
    )
    print(f"\nTensorBoard logs: {logger.log_dir}")
    
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
        LearningRateMonitor(logging_interval='step'),
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
    
    trainer.logger.log_hyperparams(vars(args))
    
    print("\nStarting training...")
    print("=" * 80)
    trainer.fit(model, datamodule)
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Best model checkpoint: {trainer.checkpoint_callback.best_model_path}")
    print(f"Best validation accuracy: {trainer.checkpoint_callback.best_model_score:.4f}")
    print("=" * 80)
    
    if trainer.checkpoint_callback.best_model_path:
        print("\nRunning test evaluation on best model...")
        trainer.test(model, datamodule,
                    ckpt_path=trainer.checkpoint_callback.best_model_path)
        
        display_confusion_matrix(model, args.classification_labels)


if __name__ == '__main__':
    main()
