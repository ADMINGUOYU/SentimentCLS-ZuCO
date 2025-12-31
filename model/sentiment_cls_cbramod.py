"""
CBraMod-based sentiment classifier using GLIM dataloader.

This model uses the CBraMod backbone for EEG processing combined with an MLP classifier
for sentiment classification. It uses the GLIM dataloader to retrieve EEG signals.

Key differences from GLIM-based sentiment classifier:
1. Uses CBraMod backbone instead of GLIM encoder
2. No subject/task/dataset embeddings - processes raw EEG signals
3. Uses CosineLR scheduler as in GLIM
4. Resamples EEG from 128Hz (ZuCo preprocessing) to 200Hz (CBraMod expected)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics.functional.classification import multiclass_accuracy
from sklearn.metrics import confusion_matrix
from transformers import get_cosine_schedule_with_warmup
from einops.layers.torch import Rearrange

from .CBraMod import CBraMod


class SentimentCLSCBraMod(L.LightningModule):
    """
    CBraMod-based sentiment classifier.
    
    This model:
    - Uses CBraMod backbone for EEG feature extraction
    - Resamples EEG from source sample rate (128Hz) to target rate (200Hz)
    - Applies an MLP classifier for sentiment classification
    - Uses CosineLR scheduler
    - Uses weighted sampling for class imbalance
    
    Args:
        num_channels: Number of EEG channels (default: 105)
        patch_size: Size of each EEG patch in samples (default: 200, 1 second at 200Hz)
        num_patches: Number of patches/segments per channel (default: 1)
        d_model: Hidden dimension of CBraMod (default: 200)
        dim_feedforward: FFN hidden dimension (default: 800)
        n_layer: Number of transformer layers (default: 12)
        nhead: Number of attention heads (default: 8)
        hidden_dims: List of hidden layer dimensions for MLP (default: [512, 256])
        num_classes: Number of sentiment classes (default: 3)
        dropout: Dropout rate (default: 0.3)
        lr: Learning rate (default: 1e-4)
        weight_decay: Weight decay for optimizer (default: 1e-4)
        warm_up_step: Number of warmup steps for CosineLR scheduler (default: None)
        pretrained_weights: Path to pretrained CBraMod weights (default: None)
        src_sample_rate: Source EEG sample rate from ZuCo preprocessing (default: 128)
        tgt_sample_rate: Target sample rate expected by CBraMod (default: 200)
    """
    
    def __init__(
        self,
        num_channels: int = 105,
        patch_size: int = 200,
        num_patches: int = 1,
        d_model: int = 200,
        dim_feedforward: int = 800,
        n_layer: int = 12,
        nhead: int = 8,
        hidden_dims: list = None,
        num_classes: int = 3,
        dropout: float = 0.3,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        warm_up_step: int = None,
        pretrained_weights: str = None,
        batch_size: int = 24,
        src_sample_rate: int = 128,
        tgt_sample_rate: int = 200
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256]
        
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.d_model = d_model
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.warm_up_step = warm_up_step
        self.batch_size = batch_size
        self.src_sample_rate = src_sample_rate
        self.tgt_sample_rate = tgt_sample_rate
        
        # CBraMod backbone
        self.backbone = CBraMod(
            in_dim=patch_size,
            out_dim=d_model,
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            seq_len=num_patches,
            n_layer=n_layer,
            nhead=nhead
        )
        
        # Load pretrained weights if provided
        if pretrained_weights is not None:
            self._load_pretrained_weights(pretrained_weights)
        
        # Remove the proj_out layer from backbone (use Identity)
        self.backbone.proj_out = nn.Identity()
        
        # Define sentiment labels
        self.sentiment_labels = ['negative', 'neutral', 'positive']
        
        # Initialize lists for metrics
        self.training_step_outputs = []
        self.test_step_outputs = []
        
        # MLP classifier with pooling
        # CBraMod output shape: (batch_size, num_channels, num_patches, d_model)
        # We pool over channels and patches to get (batch_size, d_model)
        classifier_layers = [
            Rearrange('b c s d -> b d c s'),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        ]
        
        # Add MLP layers
        in_dim = d_model
        for hidden_dim in hidden_dims:
            classifier_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        # Output layer
        classifier_layers.append(nn.Linear(in_dim, num_classes))
        
        self.classifier = nn.Sequential(*classifier_layers)
        
        # Save hyperparameters
        self.save_hyperparameters()
    
    def _load_pretrained_weights(self, pretrained_weights: str):
        """
        Load pretrained CBraMod weights from checkpoint file.
        
        Handles different checkpoint formats:
        - Direct state_dict
        - Checkpoint with 'state_dict' key
        - Checkpoint with 'model_state_dict' key
        
        Args:
            pretrained_weights: Path to the pretrained weights file (.pth)
        """
        print(f"Loading pretrained CBraMod weights from: {pretrained_weights}")
        # Note: weights_only=False is required to load complex checkpoint formats
        # Only load checkpoints from trusted sources
        checkpoint = torch.load(pretrained_weights, map_location='cpu', weights_only=False)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                # Assume the checkpoint is a direct state_dict
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Remove 'backbone.' prefix if present (from Lightning checkpoints)
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('backbone.'):
                new_key = key[len('backbone.'):]
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        
        # Load weights (strict=False allows partial loading)
        missing_keys, unexpected_keys = self.backbone.load_state_dict(new_state_dict, strict=False)
        
        if missing_keys:
            print(f"  Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"  Unexpected keys (ignored): {unexpected_keys}")
        print("  Pretrained weights loaded successfully!")
    
    def _resample_eeg(self, eeg, src_len, tgt_len):
        """
        Resample EEG signal from source length to target length using linear interpolation.
        
        Args:
            eeg: EEG tensor of shape (batch_size, channels, seq_len)
            src_len: Current sequence length
            tgt_len: Target sequence length
            
        Returns:
            Resampled EEG tensor of shape (batch_size, channels, tgt_len)
        """
        if src_len == tgt_len:
            return eeg
        
        # Use interpolate for resampling (linear interpolation)
        # Input shape for interpolate: (N, C, L) -> (N, C, L_out)
        eeg_resampled = F.interpolate(
            eeg, 
            size=tgt_len, 
            mode='linear', 
            align_corners=False
        )
        return eeg_resampled
    
    def _reshape_eeg_for_cbramod(self, eeg):
        """
        Reshape and optionally resample EEG data from GLIM format to CBraMod format.
        
        GLIM format: (batch_size, seq_len, channels)
        CBraMod format: (batch_size, num_channels, num_patches, patch_size)
        
        Steps:
        1. Transpose to (batch_size, channels, seq_len)
        2. Resample from src_sample_rate to tgt_sample_rate (only if they differ)
        3. Adjust channels (pad or truncate to num_channels)
        4. Reshape time points into patches
        
        Note: Resampling is skipped if src_sample_rate == tgt_sample_rate (e.g., both 200Hz)
        
        Args:
            eeg: EEG tensor from GLIM dataloader (batch_size, seq_len, channels)
            
        Returns:
            Reshaped EEG tensor (batch_size, num_channels, num_patches, patch_size)
        """
        batch_size, seq_len, channels = eeg.shape
        
        # Transpose to (batch_size, channels, seq_len)
        eeg = eeg.transpose(1, 2)  # (batch_size, channels, seq_len)
        
        # Resample only if source and target sample rates differ
        if self.src_sample_rate != self.tgt_sample_rate:
            # Calculate target length after resampling (use round for precision)
            tgt_seq_len = round(seq_len * self.tgt_sample_rate / self.src_sample_rate)
            eeg = self._resample_eeg(eeg, seq_len, tgt_seq_len)
        else:
            # No resampling needed
            tgt_seq_len = seq_len
        
        # Adjust channels if needed
        if channels > self.num_channels:
            eeg = eeg[:, :self.num_channels, :]
        elif channels < self.num_channels:
            # Pad with zeros if we have fewer channels
            padding = torch.zeros(batch_size, self.num_channels - channels, tgt_seq_len, 
                                device=eeg.device, dtype=eeg.dtype)
            eeg = torch.cat([eeg, padding], dim=1)
        
        # Reshape time points into patches
        # Calculate the total points we need: num_patches * patch_size
        target_length = self.num_patches * self.patch_size
        
        if tgt_seq_len >= target_length:
            # Truncate to target length
            eeg = eeg[:, :, :target_length]
        else:
            # Pad with zeros
            padding = torch.zeros(batch_size, self.num_channels, target_length - tgt_seq_len, 
                                device=eeg.device, dtype=eeg.dtype)
            eeg = torch.cat([eeg, padding], dim=2)
        
        # Reshape to (batch_size, num_channels, num_patches, patch_size)
        eeg = eeg.reshape(batch_size, self.num_channels, self.num_patches, self.patch_size)
        
        return eeg
    
    def encode_labels(self, labels: list, ignore_idx=-1):
        """Encode sentiment labels to tensor indices."""
        label_ids = []
        for label in labels:
            if label in self.sentiment_labels:
                label_id = self.sentiment_labels.index(label)
            else:
                label_id = ignore_idx
            label_ids.append(label_id)
        label_ids = torch.tensor(label_ids, dtype=torch.int64, device=self.device)
        return label_ids
    
    def forward(self, eeg):
        """
        Forward pass through CBraMod backbone and MLP classifier.
        
        Args:
            eeg: EEG tensor of shape (batch_size, seq_len, channels) from GLIM dataloader
            
        Returns:
            logits: Classification logits of shape (batch_size, num_classes)
        """
        # Reshape EEG for CBraMod
        eeg = self._reshape_eeg_for_cbramod(eeg)
        
        # Pass through CBraMod backbone
        features = self.backbone(eeg)  # (batch_size, num_channels, num_patches, d_model)
        
        # Classify using MLP head
        logits = self.classifier(features)
        
        return logits
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        eeg = batch['eeg']  # (batch_size, seq_len, channels)
        sentiment_label = batch['sentiment label']  # list of strings
        
        # Encode sentiment labels
        sentiment_ids = self.encode_labels(sentiment_label)
        
        # Forward pass
        logits = self(eeg)
        
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
        sentiment_label = batch['sentiment label']
        
        # Encode sentiment labels
        sentiment_ids = self.encode_labels(sentiment_label)
        
        # Forward pass
        logits = self(eeg)
        
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
        sentiment_label = batch['sentiment label']
        
        # Encode sentiment labels
        sentiment_ids = self.encode_labels(sentiment_label)
        
        # Forward pass
        logits = self(eeg)
        
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
        
        # Forward pass
        logits = self(eeg)
        probs = F.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=1)
        
        # Convert predictions to labels
        pred_labels = [self.sentiment_labels[pred.item()] for pred in preds]
        
        return {'predictions': preds, 'probabilities': probs, 'labels': pred_labels}
    
    def configure_optimizers(self):
        """Configure optimizer with CosineLR scheduler as in GLIM."""
        params = [p for p in self.parameters() if p.requires_grad]
        
        if self.warm_up_step is not None:
            opt = torch.optim.Adam(
                params,
                lr=self.lr,
                weight_decay=self.weight_decay
            )
            # Get total training steps (accounts for dataloader size and epochs)
            estimated_steps = self.trainer.max_epochs # !!!! [WARNING] self.trainer.estimated_stepping_batches
            lr_scheduler = get_cosine_schedule_with_warmup(
                opt,
                num_warmup_steps=self.warm_up_step,
                num_training_steps=estimated_steps
            )
            return {
                "optimizer": opt,
                "lr_scheduler": lr_scheduler
            }
        else:
            opt = torch.optim.Adam(
                params,
                lr=self.lr,
                weight_decay=self.weight_decay
            )
            return opt
