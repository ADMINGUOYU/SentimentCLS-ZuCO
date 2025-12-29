#!/usr/bin/env python
"""
Training script for sentiment classification on ZuCo 1.0 tasks.
Uses PyTorch Lightning with TensorBoard logging.
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
DEFAULT_TGT_TEXT_LEN = 64
DEFAULT_INPUT_DIM = 128


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train GLIM model for sentiment classification on ZuCo 1.0 tasks'
    )
    
    # Data arguments
    parser.add_argument(
        '--data_path',
        type=str,
        default='./tmp/zuco_merged.df',
        help='Path to the merged dataset pickle file'
    )
    parser.add_argument(
        '--embeddings_path',
        type=str,
        default=None,
        help='Path to the embeddings pickle file (optional, for sentence embedding alignment)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of data loading workers'
    )
    
    # Model arguments
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
        help='Hidden dimension size'
    )
    parser.add_argument(
        '--embed_dim',
        type=int,
        default=1024,
        help='Embedding dimension size'
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
        '--dropout',
        type=float,
        default=0.1,
        help='Dropout rate'
    )
    parser.add_argument(
        '--use_sentence_embeddings',
        action='store_true',
        help='Use precomputed sentence embeddings for alignment instead of text encoder'
    )
    parser.add_argument(
        '--eeg_emb_to_sentence_emb_hidden_dims',
        type=int,
        nargs='+',
        default=[1024, 512],
        help='Hidden layer dimensions for converting eeg embeddings to sentence embeddings\' dimension'
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
        default=1e-5,
        help='Learning rate'
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.01,
        help='Weight decay for optimizer (not used if --warm_up_step > 0)'
    )
    parser.add_argument(
        '--warm_up_step',
        type=int,
        default=0,
        help='Warm up step for Cosine loss'
    )
    parser.add_argument(
        '--clip_loss_weight',
        type=float,
        default=0.9,
        help='Weight for contrastive (CLIP) loss'
    )
    parser.add_argument(
        '--max_epochs',
        type=int,
        default=100,
        help='Maximum number of training epochs'
    )
    parser.add_argument(
        '--full_val_interval',
        type=int,
        default=10,
        help='Interval for full validation with generation'
    )
    
    # Logging and checkpoint arguments
    parser.add_argument(
        '--log_dir',
        type=str,
        default='./logs',
        help='Directory for TensorBoard logs'
    )
    parser.add_argument(
        '--experiment_name',
        type=str,
        default='sentiment_classification',
        help='Name of the experiment'
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='./checkpoints',
        help='Directory to save model checkpoints'
    )
    parser.add_argument(
        '--save_top_k',
        type=int,
        default=3,
        help='Save top k checkpoints'
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
        default='bf16-mixed',
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
    
    # Resume training
    parser.add_argument(
        '--resume_from_checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )
    
    # Misc
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    return parser.parse_args()


def setup_logger(log_dir, experiment_name):
    """Setup TensorBoard logger."""
    # Create timestamped version name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    version_name = f'{experiment_name}_{timestamp}'
    
    logger = TensorBoardLogger(
        save_dir=log_dir,
        name=experiment_name,
        version=version_name,
        default_hp_metric=False
    )
    
    print(f"TensorBoard logs will be saved to: {logger.log_dir}")
    print(f"To view logs, run: tensorboard --logdir {log_dir}")
    
    return logger


def setup_callbacks(args):
    """Setup training callbacks."""
    callbacks = []
    
    # Model checkpoint callback - save best models based on validation sentiment accuracy
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.checkpoint_dir, args.experiment_name),
        filename='glim-{epoch:02d}-{val/loss:.4f}',
        monitor='val/loss',
        mode='min',
        save_top_k=args.save_top_k,
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # Rich progress bar
    progress_bar = RichProgressBar()
    callbacks.append(progress_bar)
    
    # Early stopping (optional)
    if args.early_stopping:
        early_stop_callback = EarlyStopping(
            monitor='val/loss',
            min_delta=0.001,
            patience=args.patience,
            mode='max',
            verbose=True
        )
        callbacks.append(early_stop_callback)
        print(f"Early stopping enabled with patience={args.patience}")
    
    return callbacks


def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seed for reproducibility
    L.seed_everything(args.seed, workers=True)
    
    # Create directories
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Sentiment Classification Training on ZuCo 1.0 Tasks")
    print("=" * 80)
    print(f"Data path: {args.data_path}")
    print(f"Text model: {args.text_model}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Warming up epochs: {args.warm_up_step}")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Accelerator: {args.accelerator}")
    print(f"Device / GPU index: {args.device}")
    print(f"Precision: {args.precision}")
    print(f"EEG emb to Sentence emb hidden layers: {args.eeg_emb_to_sentence_emb_hidden_dims}")
    print("=" * 80)
    
    # Check if data file exists
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(
            f"Data file not found: {args.data_path}\n"
            "Please run the preprocessing pipeline first:\n"
            "  cd data\n"
            "  python preprocess_mat.py\n"
            "  python preprocess_gen_lbl.py\n"
            "  python preprocess_merge.py"
        )
    
    # Initialize data module
    print("\nInitializing data module...")
    datamodule = GLIMDataModule(
        data_path=args.data_path,
        embeddings_path=args.embeddings_path,
        eval_noise_input=False,
        bsz_train=args.batch_size,
        bsz_val=args.val_batch_size,
        bsz_test=args.val_batch_size,
        num_workers=args.num_workers
    )
    
    # Initialize model
    print("Initializing model...")
    model = GLIM(
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
        dropout=args.dropout,
        clip_loss_weight=args.clip_loss_weight,
        bsz_train=args.batch_size,
        bsz_val=args.val_batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warm_up_step = args.warm_up_step if args.warm_up_step > 0 else None,
        full_val_interval=args.full_val_interval,
        use_sentence_embeddings=args.use_sentence_embeddings,
        eeg_emb_to_sentence_emb_proj_hidden = args.eeg_emb_to_sentence_emb_hidden_dims
    )
    
    # Setup logger
    logger = setup_logger(args.log_dir, args.experiment_name)
    
    # Setup callbacks
    callbacks = setup_callbacks(args)
    
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
        deterministic=True,
        gradient_clip_val=1.0,
    )
    
    # Log hyperparameters
    trainer.logger.log_hyperparams(vars(args))
    
    # Train the model
    print("\nStarting training...")
    print("=" * 80)
    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=args.resume_from_checkpoint
    )
    
    # Print best model path
    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Best model checkpoint: {trainer.checkpoint_callback.best_model_path}")
    print(f"Best validation sentiment accuracy: {trainer.checkpoint_callback.best_model_score:.4f}")
    print("=" * 80)
    
    # Test the best model
    if trainer.checkpoint_callback.best_model_path:
        print("\nRunning test evaluation on best model...")
        trainer.test(
            model,
            datamodule=datamodule,
            ckpt_path=trainer.checkpoint_callback.best_model_path
        )


if __name__ == '__main__':
    main()
