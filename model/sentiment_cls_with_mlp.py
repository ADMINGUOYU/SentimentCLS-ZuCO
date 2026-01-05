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
        clip_loss_weight: contrast loss weight
        lm_loss_weight: lm loss weight
        commitment_loss_weight: mse loss weight
        mlp_loss_weight: classification mlp loss weight
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
        clip_loss_weight = 0.5,
        lm_loss_weight = 0.5,
        commitment_loss_weight = 0.0,
        mlp_loss_weight = 0.5,
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

        # loss weights
        self.clip_loss_weight = clip_loss_weight
        self.lm_loss_weight = lm_loss_weight
        self.commitment_loss_weight = commitment_loss_weight
        self.mlp_loss_weight = mlp_loss_weight
        
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
    
    def encode_labels(self, labels:list[str], ignore_idx=-1):
        label_ids = []
        for label in labels:
            if label in self.sentiment_labels:
                label_id = self.sentiment_labels.index(label)
            else:
                # Handle any unexpected labels
                label_id = ignore_idx
            label_ids.append(label_id)
        label_ids = torch.tensor(label_ids, dtype=torch.int, device=self.device)
        return label_ids

    def shared_forward(self, batch):
        # get sentiment labels for classification
        sentiment_label = batch['sentiment label']  # list of strings
        # !!! set whether to use prompt in glim model
        
        # Encode sentiment labels
        sentiment_ids = self.encode_labels(sentiment_label)
        sentiment_ids = sentiment_ids.to(torch.int64)
        
        # Forward pass GLIM
        shared_outputs = self.glim_encoder.shared_forward(batch)
        loss_commitment = shared_outputs['loss_commitment']     # (1)
        loss_clip = shared_outputs['loss_clip']                 # (1)
        loss_lm = shared_outputs['loss_lm']                     # (1)
        eeg_emb = shared_outputs['eeg_emb_vector']

        # pass mlp
        logits = self.mlp_classifier(eeg_emb)
        
        # Calculate mlp loss
        loss_mlp = F.cross_entropy(logits, sentiment_ids, ignore_index=-1)
        
        # calculate total loss
        loss = \
        loss_clip * self.clip_loss_weight + \
        loss_lm * self.lm_loss_weight + \
        loss_commitment * self.commitment_loss_weight + \
        loss_mlp * self.mlp_loss_weight

        # Calculate accuracy
        acc = multiclass_accuracy(
            logits, sentiment_ids, 
            average='micro', 
            num_classes=self.num_classes, 
            ignore_index=-1,
            top_k=1
        )

        # return dict
        output = {
            'total_loss' : loss,
            'loss_clip' : loss_clip,
            'loss_lm' : loss_lm,
            'loss_commitment' : loss_commitment,
            'loss_mlp' : loss_mlp,
            'acc' : acc,
            'logits' : logits
        }
        
        return output

    def training_step(self, batch, batch_idx):
        """Training step."""

        # forward
        output  = self.shared_forward(batch)

        # get metrics
        loss = output['total_loss']
        loss_clip = output['loss_clip']
        loss_lm = output['loss_lm']
        loss_commitment = output['loss_commitment']
        loss_mlp = output['loss_mlp']
        acc = output['acc']

        # Log to metrics
        self.log('train/loss', loss, prog_bar=True, sync_dist=True, batch_size=self.batch_size)
        self.log('train/accuracy', acc, prog_bar=True, sync_dist=True, batch_size=self.batch_size)
        self.log('train/loss_clip', loss_clip, prog_bar=True, sync_dist=True, batch_size=self.batch_size)
        self.log('train/loss_lm', loss_lm, prog_bar=True, sync_dist=True, batch_size=self.batch_size)
        self.log('train/loss_commitment', loss_commitment, prog_bar=True, sync_dist=True, batch_size=self.batch_size)
        self.log('train/loss_mlp', loss_mlp, prog_bar=True, sync_dist=True, batch_size=self.batch_size)

        # return loss
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        
        with torch.no_grad():
            # forward
            output  = self.shared_forward(batch)

        # get metrics
        loss = output['total_loss']
        loss_clip = output['loss_clip']
        loss_lm = output['loss_lm']
        loss_commitment = output['loss_commitment']
        loss_mlp = output['loss_mlp']
        acc = output['acc']

        # Log to metrics
        self.log('val/loss', loss, prog_bar=True, sync_dist=True, batch_size=self.batch_size)
        self.log('val/accuracy', acc, prog_bar=True, sync_dist=True, batch_size=self.batch_size)
        self.log('val/loss_clip', loss_clip, prog_bar=True, sync_dist=True, batch_size=self.batch_size)
        self.log('val/loss_lm', loss_lm, prog_bar=True, sync_dist=True, batch_size=self.batch_size)
        self.log('val/loss_commitment', loss_commitment, prog_bar=True, sync_dist=True, batch_size=self.batch_size)
        self.log('val/loss_mlp', loss_mlp, prog_bar=True, sync_dist=True, batch_size=self.batch_size)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        with torch.no_grad():
            # forward
            output  = self.shared_forward(batch)

        # get metrics
        loss = output['total_loss']
        acc = output['acc']
        logits = output['logits']
    
        # Get predictions
        preds = torch.argmax(logits, dim=1)
        # get sentiment labels for classification
        sentiment_label = batch['sentiment label']  # list of strings
        # Encode sentiment labels
        sentiment_ids = self.encode_labels(sentiment_label)
        sentiment_ids = sentiment_ids.to(torch.int64)
        
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
        if self.glim_encoder.use_prompt:
            # assert
            assert prompts is not None, "[ERROR] prompt should not be None"
            logits = self(eeg, eeg_mask, prompts)
        else:
            logits = self(eeg, eeg_mask, None)
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
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.lr,
            eta_min=1e-5
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
