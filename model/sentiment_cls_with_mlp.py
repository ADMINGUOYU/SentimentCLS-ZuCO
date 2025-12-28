"""
End-to-end sentiment classifier that combines GLIM encoder with MLP classifier.

This model differs from SentimentMLP in that:
1. It trains both the GLIM encoder and MLP classifier together (gradients enabled)
2. It processes raw EEG data directly (not pre-extracted embeddings)
3. It can load a pretrained GLIM checkpoint and fine-tune it

The model uses the GLIM encoder to extract embeddings from EEG data, then
passes these embeddings through an MLP classifier for sentiment classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics.functional.classification import multiclass_accuracy
from sklearn.metrics import confusion_matrix
import numpy as np

from .glim import GLIM


class SentimentCLSWithMLP(L.LightningModule):
    """
    End-to-end sentiment classifier combining GLIM encoder with MLP.
    
    This model:
    - Loads a pretrained GLIM model
    - Extracts embeddings using GLIM encoder (with gradients)
    - Classifies sentiment using an MLP head
    - Trains both encoder and classifier together
    
    Args:
        glim_checkpoint: Path to pretrained GLIM checkpoint or GLIM model instance
        hidden_dims: List of hidden layer dimensions for MLP (default: [512, 256])
        num_classes: Number of sentiment classes (default: 3)
        dropout: Dropout rate for regularization (default: 0.3)
        lr: Learning rate (default: 1e-4)
        weight_decay: Weight decay for optimizer (default: 1e-4)
        freeze_encoder: If True, freeze GLIM encoder weights (default: False)
    """
    
    def __init__(
        self,
        glim_checkpoint: str = None,
        glim_model: GLIM = None,
        hidden_dims: list = None,
        num_classes: int = 3,
        dropout: float = 0.3,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        freeze_encoder: bool = False,
        batch_size: int = 24
    ):
        super().__init__()

        # batch_size record
        self.batch_size = batch_size
        
        if hidden_dims is None:
            hidden_dims = [512, 256]
        
        # Load or use provided GLIM model
        if glim_model is not None:
            self.glim_encoder = glim_model
        elif glim_checkpoint is not None:
            self.glim_encoder = GLIM.load_from_checkpoint(glim_checkpoint)
        else:
            raise ValueError("Either glim_checkpoint or glim_model must be provided")
        
        # Get embedding dimension from GLIM
        self.embed_dim = self.glim_encoder.embed_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.freeze_encoder = freeze_encoder
        
        # Freeze encoder if requested
        if self.freeze_encoder:
            for param in self.glim_encoder.parameters():
                param.requires_grad = False
        else:
            # Enable gradients for encoder
            for param in self.glim_encoder.parameters():
                param.requires_grad = True
        
        # Define sentiment labels
        self.sentiment_labels = ['negative', 'neutral', 'positive']
        
        # Initialize lists to accumulate training metrics per epoch
        self.training_step_outputs = []
        
        # Initialize lists to accumulate test predictions and targets for confusion matrix
        self.test_step_outputs = []
        
        # Build MLP classifier head
        layers = []
        in_dim = self.embed_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(in_dim, num_classes))
        
        self.mlp_classifier = nn.Sequential(*layers)
        
        # Save hyperparameters
        self.save_hyperparameters(ignore=['glim_model', 'glim_checkpoint'])
    
    def forward(self, eeg, eeg_mask=None, prompts=None):
        """
        Forward pass through GLIM encoder and MLP classifier.
        
        Args:
            eeg: EEG tensor of shape (batch_size, seq_len, channels)
            eeg_mask: Optional mask tensor
            prompts: Optional prompts tuple
            
        Returns:
            logits: Classification logits of shape (batch_size, num_classes)
        """
        # Extract embeddings using GLIM encoder (with gradients)
        embeddings = self.glim_encoder.extract_embeddings_with_grad(eeg, eeg_mask, prompts)
        
        # Classify using MLP head
        logits = self.mlp_classifier(embeddings)
        
        return logits
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        eeg = batch['eeg']  # (batch_size, seq_len, channels)
        eeg_mask = batch['mask']  # (batch_size, seq_len)
        prompts = batch['prompt']  # list of tuples
        sentiment_label = batch['sentiment label']  # list of strings
        
        # Encode sentiment labels
        sentiment_ids = self.glim_encoder.encode_labels(sentiment_label)
        sentiment_ids = sentiment_ids.to(torch.int64)
        
        # Forward pass
        logits = self(eeg, eeg_mask, prompts)
        
        # Calculate loss
        loss = F.cross_entropy(logits, sentiment_ids, ignore_index=-1)
        
        # Calculate accuracy
        acc = multiclass_accuracy(
            logits, sentiment_ids, 
            average='micro', 
            num_classes=self.num_classes, 
            ignore_index=-1,
            top_k=1
        )
        
        # Store metrics for epoch-level logging
        self.training_step_outputs.append({
            'loss': loss.detach(),
            'accuracy': acc.detach()
        })
        
        return loss
    
    def on_train_epoch_end(self):
        """Compute and log average metrics at the end of each training epoch."""
        if len(self.training_step_outputs) == 0:
            return
        
        # Compute average loss and accuracy
        avg_loss = torch.stack([x['loss'] for x in self.training_step_outputs]).mean()
        avg_acc = torch.stack([x['accuracy'] for x in self.training_step_outputs]).mean()
        
        # Log to tensorboard
        self.log('train/loss', avg_loss, prog_bar=True, sync_dist=True, batch_size=self.batch_size)
        self.log('train/accuracy', avg_acc, prog_bar=True, sync_dist=True, batch_size=self.batch_size)
        
        # Clear for next epoch
        self.training_step_outputs.clear()
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        eeg = batch['eeg']
        eeg_mask = batch['mask']
        prompts = batch['prompt']
        sentiment_label = batch['sentiment label']
        
        # Encode sentiment labels
        sentiment_ids = self.glim_encoder.encode_labels(sentiment_label)
        sentiment_ids = sentiment_ids.to(torch.int64)
        
        # Forward pass
        logits = self(eeg, eeg_mask, prompts)
        
        # Calculate loss
        loss = F.cross_entropy(logits, sentiment_ids, ignore_index=-1)
        
        # Calculate accuracy
        acc = multiclass_accuracy(
            logits, sentiment_ids,
            average='micro',
            num_classes=self.num_classes,
            ignore_index=-1,
            top_k=1
        )
        
        # Log metrics
        self.log('val/loss', loss, prog_bar=True, sync_dist=True, batch_size=self.batch_size)
        self.log('val/accuracy', acc, prog_bar=True, sync_dist=True, batch_size=self.batch_size)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        eeg = batch['eeg']
        eeg_mask = batch['mask']
        prompts = batch['prompt']
        sentiment_label = batch['sentiment label']
        
        # Encode sentiment labels
        sentiment_ids = self.glim_encoder.encode_labels(sentiment_label)
        sentiment_ids = sentiment_ids.to(torch.int64)
        
        # Forward pass
        logits = self(eeg, eeg_mask, prompts)
        
        # Calculate loss
        loss = F.cross_entropy(logits, sentiment_ids, ignore_index=-1)
        
        # Calculate accuracy
        acc = multiclass_accuracy(
            logits, sentiment_ids,
            average='micro',
            num_classes=self.num_classes,
            ignore_index=-1,
            top_k=1
        )
        
        # Get predictions
        preds = torch.argmax(logits, dim=1)
        
        # Log metrics
        self.log('test/loss', loss, sync_dist=True, batch_size=self.batch_size)
        self.log('test/accuracy', acc, sync_dist=True, batch_size=self.batch_size)
        
        # Store predictions and targets for confusion matrix
        self.test_step_outputs.append({
            'predictions': preds.detach().cpu(),
            'targets': sentiment_ids.detach().cpu()
        })
        
        return {'loss': loss, 'accuracy': acc, 'predictions': preds, 'targets': sentiment_ids}
    
    def on_test_epoch_end(self):
        """Compute confusion matrix at the end of testing."""
        if len(self.test_step_outputs) == 0:
            return
        
        # Concatenate all predictions and targets
        all_preds = torch.cat([x['predictions'] for x in self.test_step_outputs])
        all_targets = torch.cat([x['targets'] for x in self.test_step_outputs])
        
        # Compute confusion matrix
        cm = confusion_matrix(all_targets.numpy(), all_preds.numpy(), labels=[0, 1, 2])
        
        # Store confusion matrix for later display
        self.confusion_matrix = cm
        
        # Clear for next test run
        self.test_step_outputs.clear()
    
    def predict_step(self, batch, batch_idx):
        """Prediction step."""
        eeg = batch['eeg']
        eeg_mask = batch['mask']
        prompts = batch['prompt']
        
        # Forward pass
        logits = self(eeg, eeg_mask, prompts)
        probs = F.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=1)
        
        # Convert predictions to labels
        pred_labels = [self.sentiment_labels[pred.item()] for pred in preds]
        
        return {'predictions': preds, 'probabilities': probs, 'labels': pred_labels}
    
    def configure_optimizers(self):
        """Configure optimizer with separate learning rates for encoder and classifier."""
        # Separate parameters for encoder and classifier
        encoder_params = []
        classifier_params = []
        
        if not self.freeze_encoder:
            encoder_params = [p for p in self.glim_encoder.parameters() if p.requires_grad]
        
        classifier_params = [p for p in self.mlp_classifier.parameters() if p.requires_grad]
        
        # Use lower learning rate for encoder (fine-tuning) and higher for classifier
        optimizer = torch.optim.Adam([
            {'params': encoder_params, 'lr': self.lr * 0.1},  # 10x lower LR for encoder
            {'params': classifier_params, 'lr': self.lr}
        ], weight_decay=self.weight_decay)
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }
