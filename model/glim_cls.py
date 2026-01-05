"""
GLIM_CLS: Combined GLIM encoder with MLP classifier for end-to-end classification.

This model combines the GLIM encoder with an MLP classifier for generalized
classification tasks. It differs from the original SentimentCLSWithMLP in that:
1. It's a standalone model without dependency on the separate GLIM class
2. The classification label key is configurable (not hardcoded to 'sentiment label')
3. The classification labels are configurable (not hardcoded to sentiment labels)
4. Uses Cosine LR scheduler with warmup (configurable max_lr, min_lr, warmup_steps)
5. Supports loading from checkpoint
6. Uses bfloat16 as default for T5 model
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from typing import Literal, Dict, Any, Optional
from copy import deepcopy
from torchmetrics.functional.classification import multiclass_accuracy
from sklearn.metrics import confusion_matrix
from transformers import AutoTokenizer, T5ForConditionalGeneration, get_cosine_schedule_with_warmup
from transformers.modeling_outputs import BaseModelOutput

from .modules import PromptEmbedder, EEGEncoder, Aligner


class GLIM_CLS(L.LightningModule):
    """
    Combined GLIM encoder with MLP classifier for end-to-end classification.
    
    This model:
    - Contains the GLIM encoder architecture
    - Extracts embeddings using the EEG encoder (with gradients)
    - Classifies using an MLP head
    - Supports configurable classification labels and label keys
    - Uses Cosine LR scheduler with warmup
    
    Args:
        input_eeg_len: Length of input EEG sequence (default: 1280)
        hidden_eeg_len: Length of hidden EEG representation (default: 96)
        input_text_len: Length of input text sequence (default: 96)
        input_dim: Input dimension (default: 128)
        hidden_dim: Hidden dimension (default: 128)
        embed_dim: Embedding dimension (default: 1024)
        text_model_id: Pre-trained text model to use (default: "google/flan-t5-large")
        prompt_nums: Tuple of prompt numbers for (task, dataset, subject)
        prompt_dropout_probs: Tuple of dropout probabilities for prompts
        evaluate_prompt_embed: Evaluation method for prompt embeddings
        n_in_blocks: Number of encoder blocks (default: 6)
        n_out_blocks: Number of decoder blocks (default: 6)
        in_temporal_modulate: Whether to use temporal modulation in encoder
        out_is_causal: Whether decoder attention is causal
        use_prompt: Whether to use prompt embeddings (default: True)
        num_heads: Number of attention heads (default: 8)
        mlp_ratio: MLP expansion ratio (default: 4)
        dropout: Dropout rate (default: 0.0)
        clip_loss_weight: Weight for contrastive loss (default: 0.5)
        lm_loss_weight: Weight for language model loss (default: 0.0)
        commitment_loss_weight: Weight for commitment loss (default: 0.0)
        commitment_loss_key: Type of commitment loss ('mse' or 'kl_div')
        use_y_mask: Whether to use mask for alignment (default: False)
        batch_size: Training batch size (default: 24)
        
        # Classification-specific arguments
        classification_label_key: Key in batch dict for classification labels (default: 'sentiment label')
        classification_labels: List of classification label names (default: ['negative', 'neutral', 'positive'])
        mlp_hidden_dims: List of hidden dimensions for MLP classifier (default: [512, 256])
        mlp_dropout: Dropout rate for MLP classifier (default: 0.3)
        mlp_loss_weight: Weight for MLP classification loss (default: 0.5)
        freeze_encoder: If True, freeze encoder weights (default: False)
        
        # Optimizer arguments (Cosine LR with warmup only)
        lr: Maximum learning rate (default: 1e-4)
        min_lr: Minimum learning rate (default: 1e-6)
        warmup_steps: Number of warmup steps (default: 100)
    """

    SUPPORTED_TEXT_MODELS = Literal["google/flan-t5-xl", "google/flan-t5-large", 
                                    "facebook/bart-large-cnn", "jbochi/madlad400-3b-mt"]

    def __init__(self,
                 # EEG Encoder arguments
                 input_eeg_len: int = 1280,
                 hidden_eeg_len: int = 96,
                 input_text_len: int = 96,
                 tgt_text_len: int = 64,
                 input_dim: int = 128,
                 hidden_dim: int = 128,
                 embed_dim: int = 1024,
                 text_model_id: SUPPORTED_TEXT_MODELS = "google/flan-t5-large",
                 prompt_nums: tuple = (3, 3, 31),
                 prompt_dropout_probs: tuple = (0.0, 0.0, 0.0),
                 evaluate_prompt_embed: Literal['zero', 'sum', 'mean', 'src'] = 'src',
                 n_in_blocks: int = 6,
                 n_out_blocks: int = 6,
                 in_temporal_modulate: bool = True,
                 out_is_causal: bool = True,
                 use_prompt: bool = True,
                 num_heads: int = 8,
                 mlp_ratio: int = 4,
                 dropout: float = 0.0,
                 clip_loss_weight: float = 0.5,
                 lm_loss_weight: float = 0.0,
                 commitment_loss_weight: float = 0.0,
                 commitment_loss_key: Literal['mse', 'kl_div'] = 'mse',
                 use_y_mask: bool = False,
                 batch_size: int = 24,
                 
                 # Classification-specific arguments
                 classification_label_key: str = 'sentiment label',
                 classification_labels: list = None,
                 mlp_hidden_dims: list = None,
                 mlp_dropout: float = 0.3,
                 mlp_loss_weight: float = 0.5,
                 freeze_encoder: bool = False,
                 
                 # Optimizer arguments (Cosine LR with warmup)
                 lr: float = 1e-4,
                 min_lr: float = 1e-6,
                 warmup_steps: int = 100,
                 ):
        super().__init__()
        
        # Set default values for mutable arguments
        if classification_labels is None:
            classification_labels = ['negative', 'neutral', 'positive']
        if mlp_hidden_dims is None:
            mlp_hidden_dims = [512, 256]
        
        # Store core parameters
        self.input_text_len = input_text_len
        self.tgt_text_len = tgt_text_len
        self.eval_pembed = evaluate_prompt_embed
        self.use_prompt = use_prompt
        self.embed_dim = embed_dim
        self.text_model_id = text_model_id
        self.batch_size = batch_size
        
        # Loss weights
        self.clip_loss_weight = clip_loss_weight
        self.lm_loss_weight = lm_loss_weight
        self.commitment_loss_weight = commitment_loss_weight
        self.mlp_loss_weight = mlp_loss_weight
        
        # Optimizer parameters
        self.lr = lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        
        # Classification parameters
        self.classification_label_key = classification_label_key
        self.classification_labels = classification_labels
        self.num_classes = len(classification_labels)
        self.freeze_encoder = freeze_encoder
        
        # Prompt configuration
        self.prompt_keys = {
            'task': ['<UNK>'] + ['<NR>', '<TSR>'],
            'dataset': ['<UNK>'] + ['ZuCo1', 'ZuCo2'],
            'subject': ['<UNK>'] + ['ZAB', 'ZDM', 'ZDN', 'ZGW', 'ZJM', 'ZJN',
                                    'ZJS', 'ZKB', 'ZKH', 'ZKW', 'ZMG', 'ZPH',
                                    'YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR',
                                    'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YMS',
                                    'YRH', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL'],
        }
        
        # Build prompt embedder
        self.p_embedder = PromptEmbedder(input_dim, prompt_nums, prompt_dropout_probs, self.prompt_keys)
        
        # Build EEG encoder
        self.eeg_encoder = EEGEncoder(input_eeg_len, hidden_eeg_len, input_dim, hidden_dim,
                                      0,  # prompt_tuning_len not used for classification
                                      n_in_blocks, n_out_blocks,
                                      in_temporal_modulate, out_is_causal,
                                      num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
        
        # Build aligner
        self.aligner = Aligner(hidden_dim, embed_dim, num_heads, dropout, commitment_loss_key, use_y_mask)
        self.use_y_mask = use_y_mask
        
        # Build MLP classifier
        layers = []
        in_dim = embed_dim
        for hidden_dim_mlp in mlp_hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim_mlp),
                nn.BatchNorm1d(hidden_dim_mlp),
                nn.ReLU(),
                nn.Dropout(mlp_dropout)
            ])
            in_dim = hidden_dim_mlp
        layers.append(nn.Linear(in_dim, self.num_classes))
        self.mlp_classifier = nn.Sequential(*layers)
        
        # Initialize test outputs list for confusion matrix
        self.test_step_outputs = []
        
        # Save hyperparameters
        self.save_hyperparameters(logger=True)
    
    def setup(self, stage):
        """Setup the text model (T5) using bfloat16 by default."""
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.tokenizer = AutoTokenizer.from_pretrained(self.text_model_id)
        self.text_model = T5ForConditionalGeneration.from_pretrained(
            self.text_model_id,
            device_map=self.device,
            torch_dtype=torch.bfloat16,  # Use bfloat16 as default
        ).requires_grad_(False)
        assert self.embed_dim == self.text_model.config.d_model
        
        # Freeze encoder if requested
        if self.freeze_encoder:
            for param in self.p_embedder.parameters():
                param.requires_grad = False
            for param in self.eeg_encoder.parameters():
                param.requires_grad = False
            for param in self.aligner.parameters():
                param.requires_grad = False
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Remove text model weights from checkpoint to save space."""
        for key in deepcopy(list(checkpoint['state_dict'].keys())):
            if 'text_model' in key:
                checkpoint['state_dict'].pop(key)
    
    def configure_optimizers(self):
        """Configure optimizer with Cosine LR scheduler with warmup."""
        # Separate parameters for encoder and classifier
        encoder_params = []
        classifier_params = list(self.mlp_classifier.parameters())
        
        if not self.freeze_encoder:
            encoder_params.extend(list(self.p_embedder.parameters()))
            encoder_params.extend(list(self.eeg_encoder.parameters()))
            encoder_params.extend(list(self.aligner.parameters()))
        
        # Use lower learning rate for encoder (fine-tuning) and higher for classifier
        param_groups = []
        if encoder_params:
            param_groups.append({'params': encoder_params, 'lr': self.lr * 0.1})
        param_groups.append({'params': classifier_params, 'lr': self.lr})
        
        optimizer = torch.optim.Adam(param_groups)
        
        # Cosine LR scheduler with warmup
        # Calculate total training steps
        total_steps = self.trainer.estimated_stepping_batches
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps,
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }
    
    def tokenize(self, texts: list, max_length: int):
        """Tokenize text inputs."""
        inputs = self.tokenizer(texts, max_length=max_length, padding='max_length',
                                truncation=True, return_tensors="pt")
        ids = inputs['input_ids'].to(self.device)
        mask = inputs['attention_mask'].to(self.device)
        return ids, mask
    
    def encode_labels(self, labels: list, ignore_idx: int = -1):
        """Encode string labels to integer IDs."""
        label_ids = []
        for label in labels:
            if label in self.classification_labels:
                label_id = self.classification_labels.index(label)
            else:
                label_id = ignore_idx
            label_ids.append(label_id)
        label_ids = torch.tensor(label_ids, dtype=torch.int64, device=self.device)
        return label_ids
    
    def encode_text(self, src_ids, src_mask):
        """Encode text using the T5 encoder."""
        text_encoder = self.text_model.get_encoder()
        with torch.no_grad():
            outputs = text_encoder(input_ids=src_ids,
                                   attention_mask=src_mask,
                                   return_dict=True)
        hidden_states = outputs['last_hidden_state']
        return hidden_states, src_mask
    
    def text_decoder_forward(self, src_embeds, src_mask, tgt_ids):
        """Compute language model loss using the T5 decoder."""
        labels = tgt_ids.detach().clone()
        labels.masked_fill_(labels == self.text_model.config.pad_token_id, -100)
        mask = src_mask if (self.use_y_mask and self.training) else None
        outputs = self.text_model(encoder_outputs=BaseModelOutput(src_embeds),
                                  attention_mask=mask,
                                  labels=labels)
        loss = outputs['loss']
        return loss
    
    def extract_embeddings(self, eeg, eeg_mask=None, prompts=None):
        """
        Extract EEG embeddings with gradients enabled for end-to-end training.
        
        Args:
            eeg: EEG tensor of shape (batch_size, seq_len, channels)
            eeg_mask: Optional mask tensor (will be ignored - no masking applied)
            prompts: Optional tuple of (task, dataset, subject) prompts
        
        Returns:
            eeg_emb_vector: EEG embedding vectors of shape (batch_size, embed_dim)
        """
        eeg = eeg.to(self.device)
        
        batch_size, seq_len, _ = eeg.shape
        eeg_mask_unmasked = torch.ones(batch_size, seq_len, dtype=torch.float32, device=self.device)
        
        if prompts is None:
            prompts = [['<UNK>'] * batch_size, ['<UNK>'] * batch_size, ['<UNK>'] * batch_size]
        
        prompt_ids = self.p_embedder.encode(prompts, device=self.device)
        prompt_embed = self.p_embedder(prompt_ids, self.eval_pembed)
        
        eeg_hiddens, _ = self.eeg_encoder(eeg, eeg_mask_unmasked, prompt_embed)
        _, eeg_emb_vector = self.aligner.embed_eeg(eeg_hiddens, mask=None)
        
        return eeg_emb_vector
    
    def forward(self, eeg, eeg_mask=None, prompts=None):
        """
        Forward pass through encoder and MLP classifier.
        
        Args:
            eeg: EEG tensor of shape (batch_size, seq_len, channels)
            eeg_mask: Optional mask tensor
            prompts: Optional prompts tuple
            
        Returns:
            logits: Classification logits of shape (batch_size, num_classes)
        """
        embeddings = self.extract_embeddings(eeg, eeg_mask, prompts)
        logits = self.mlp_classifier(embeddings)
        return logits
    
    def shared_forward(self, batch):
        """Shared forward pass for training, validation, and testing."""
        eeg = batch['eeg']
        eeg_mask = batch['mask']
        prompts = batch['prompt']
        input_text = batch['input text']
        target_text = batch['target text']
        
        # Get classification labels using configurable key
        classification_label = batch[self.classification_label_key]
        classification_ids = self.encode_labels(classification_label)
        
        # Handle prompts
        if self.use_prompt is False:
            batch_size = eeg.shape[0]
            prompts = [['<UNK>'] * batch_size, ['<UNK>'] * batch_size, ['<UNK>'] * batch_size]
        
        # Encode prompts and EEG
        prompt_ids = self.p_embedder.encode(prompts, device=self.device)
        prompt_embed = self.p_embedder(prompt_ids, self.eval_pembed)
        
        eeg_hiddens, _ = self.eeg_encoder(eeg, eeg_mask, prompt_embed)
        
        # Encode text for alignment loss
        input_ids, input_mask = self.tokenize(input_text, self.input_text_len)
        input_text_embeds, hidden_text_mask = self.encode_text(input_ids, input_mask)
        
        # Alignment losses
        (loss_clip, logits_clip, loss_commitment,
         eeg_embeds, eeg_emb, input_text_emb) = self.aligner.forward(eeg_hiddens, input_text_embeds, hidden_text_mask)
        
        # Language model loss (if weight > 0)
        if self.lm_loss_weight > 0:
            tgt_ids, _ = self.tokenize(target_text, self.tgt_text_len)
            loss_lm = self.text_decoder_forward(eeg_embeds, hidden_text_mask, tgt_ids)
        else:
            loss_lm = torch.tensor(0.0, device=self.device)
        
        # MLP classification
        logits = self.mlp_classifier(eeg_emb)
        loss_mlp = F.cross_entropy(logits, classification_ids, ignore_index=-1)
        
        # Total loss
        loss = (loss_clip * self.clip_loss_weight +
                loss_lm * self.lm_loss_weight +
                loss_commitment * self.commitment_loss_weight +
                loss_mlp * self.mlp_loss_weight)
        
        # Calculate accuracy
        acc = multiclass_accuracy(
            logits, classification_ids,
            average='micro',
            num_classes=self.num_classes,
            ignore_index=-1,
            top_k=1
        )
        
        return {
            'total_loss': loss,
            'loss_clip': loss_clip,
            'loss_lm': loss_lm,
            'loss_commitment': loss_commitment,
            'loss_mlp': loss_mlp,
            'acc': acc,
            'logits': logits,
            'classification_ids': classification_ids,
        }
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        output = self.shared_forward(batch)
        
        loss = output['total_loss']
        loss_clip = output['loss_clip']
        loss_lm = output['loss_lm']
        loss_commitment = output['loss_commitment']
        loss_mlp = output['loss_mlp']
        acc = output['acc']
        
        self.log('train/loss', loss, prog_bar=True, sync_dist=True, batch_size=self.batch_size)
        self.log('train/accuracy', acc, prog_bar=True, sync_dist=True, batch_size=self.batch_size)
        self.log('train/loss_clip', loss_clip, sync_dist=True, batch_size=self.batch_size)
        self.log('train/loss_lm', loss_lm, sync_dist=True, batch_size=self.batch_size)
        self.log('train/loss_commitment', loss_commitment, sync_dist=True, batch_size=self.batch_size)
        self.log('train/loss_mlp', loss_mlp, sync_dist=True, batch_size=self.batch_size)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        with torch.no_grad():
            output = self.shared_forward(batch)
        
        loss = output['total_loss']
        loss_clip = output['loss_clip']
        loss_lm = output['loss_lm']
        loss_commitment = output['loss_commitment']
        loss_mlp = output['loss_mlp']
        acc = output['acc']
        
        self.log('val/loss', loss, prog_bar=True, sync_dist=True, batch_size=self.batch_size)
        self.log('val/accuracy', acc, prog_bar=True, sync_dist=True, batch_size=self.batch_size)
        self.log('val/loss_clip', loss_clip, sync_dist=True, batch_size=self.batch_size)
        self.log('val/loss_lm', loss_lm, sync_dist=True, batch_size=self.batch_size)
        self.log('val/loss_commitment', loss_commitment, sync_dist=True, batch_size=self.batch_size)
        self.log('val/loss_mlp', loss_mlp, sync_dist=True, batch_size=self.batch_size)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        with torch.no_grad():
            output = self.shared_forward(batch)
        
        loss = output['total_loss']
        acc = output['acc']
        logits = output['logits']
        classification_ids = output['classification_ids']
        
        preds = torch.argmax(logits, dim=1)
        
        self.log('test/loss', loss, sync_dist=True, batch_size=self.batch_size)
        self.log('test/accuracy', acc, sync_dist=True, batch_size=self.batch_size)
        
        self.test_step_outputs.append({
            'predictions': preds.detach().cpu(),
            'targets': classification_ids.detach().cpu()
        })
        
        return {'loss': loss, 'accuracy': acc, 'predictions': preds, 'targets': classification_ids}
    
    def on_test_epoch_end(self):
        """Compute confusion matrix at the end of testing."""
        if len(self.test_step_outputs) == 0:
            return
        
        all_preds = torch.cat([x['predictions'] for x in self.test_step_outputs])
        all_targets = torch.cat([x['targets'] for x in self.test_step_outputs])
        
        labels = list(range(self.num_classes))
        cm = confusion_matrix(all_targets.numpy(), all_preds.numpy(), labels=labels)
        self.confusion_matrix = cm
        
        self.test_step_outputs.clear()
    
    def predict_step(self, batch, batch_idx):
        """Prediction step."""
        eeg = batch['eeg']
        eeg_mask = batch['mask']
        prompts = batch['prompt']
        
        if not self.use_prompt:
            prompts = None
        
        logits = self(eeg, eeg_mask, prompts)
        probs = F.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=1)
        
        pred_labels = [self.classification_labels[pred.item()] for pred in preds]
        
        return {'predictions': preds, 'probabilities': probs, 'labels': pred_labels}
