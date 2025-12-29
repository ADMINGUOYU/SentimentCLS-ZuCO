# Sentence Embedding Alignment Changes

This document describes the changes made to enable EEG encoder alignment with sentence embeddings instead of text encoder outputs.

## Overview

The modifications allow the GLIM model to:
1. Use precomputed sentence embeddings from SBERT (all-mpnet-base-v2) 
2. Align EEG encoder outputs with these embeddings during training
3. Maintain backward compatibility with existing models and workflows

## Changes Made

### 1. Preprocessing Scripts

**Location**: `data/preprocess_gen_lbl.py`, `data/preprocess_mat.py`, `data/preprocess_merge.py`

- Changed `tmp_path` from hardcoded NFS path to `./tmp` for local storage
- Added automatic directory creation with `os.makedirs(tmp_path, exist_ok=True)`
- Added top 3 keywords columns (`keyword_1`, `keyword_2`, `keyword_3`) to merged dataframe

**Output Files**:
- `./tmp/embeddings.pickle`: Dictionary mapping text_uid to sentence/keyword embeddings
- `./tmp/zuco_label_input_text.df`: Labels with keywords
- `./tmp/zuco_merged.df`: Merged EEG data with labels and keywords

### 2. Data Module

**Location**: `data/datamodule.py`

**New Parameters**:
- `embeddings_path` (optional): Path to embeddings.pickle file

**Changes**:
- `GLIMDataModule.setup()`: Loads embeddings from pickle file if path provided
- `ZuCoDataset.__init__()`: Accepts `embeddings_dict` parameter
- `ZuCoDataset.__fetch_from_df()`: Extracts embeddings by text_uid
- `ZuCoDataset.__getitem__()`: Returns embeddings in batch dict

**Backward Compatibility**: 
- When `embeddings_path=None`, dataloader works exactly as before
- Embeddings are optional fields in batch

### 3. GLIM Model

**Location**: `model/glim.py`

**New Parameters**:
- `use_sentence_embeddings` (bool, default=False): Enable sentence embedding alignment

**New Components**:
- `sentence_emb_proj`: Linear layer to project 768-dim embeddings to model's embed_dim
- `encode_sentence_embeddings()`: Method to process sentence embeddings

**Modified Methods**:
- `get_inputs()`: Extracts sentence embeddings from batch
- `shared_forward()`: Uses sentence embeddings when flag is enabled

**Backward Compatibility**:
- When `use_sentence_embeddings=False`, model uses text encoder as before
- All existing functionality preserved

### 4. Training Script

**Location**: `train.py`

**New Arguments**:
- `--embeddings_path`: Path to embeddings.pickle (optional)
- `--use_sentence_embeddings`: Enable sentence embedding alignment (flag)

## Usage

### Option 1: Traditional Training (Text Encoder)

```bash
python train.py \
  --data_path ./tmp/zuco_merged.df \
  --batch_size 48 \
  --lr 1e-5 \
  --max_epochs 100
```

This uses the text encoder for alignment (original behavior).

### Option 2: Sentence Embedding Alignment

```bash
python train.py \
  --data_path ./tmp/zuco_merged.df \
  --embeddings_path ./tmp/embeddings.pickle \
  --use_sentence_embeddings \
  --batch_size 48 \
  --lr 1e-5 \
  --max_epochs 100
```

This aligns EEG encoder with precomputed sentence embeddings.

## Preprocessing Pipeline

Run preprocessing in this order:

```bash
cd data

# Step 1: Process EEG mat files
python preprocess_mat.py

# Step 2: Generate labels and embeddings
python preprocess_gen_lbl.py

# Step 3: Merge EEG data with labels/embeddings
python preprocess_merge.py
```

**Output**: `./tmp/` directory with all preprocessed files

## Architecture Details

### Sentence Embedding Flow

1. **Preprocessing**: 
   - SBERT generates 768-dim sentence embeddings
   - Stored in embeddings.pickle by text_uid

2. **Data Loading**:
   - Dataloader loads embeddings.pickle
   - Maps embeddings to samples via text_uid
   - Returns embeddings in batch

3. **Model Forward Pass**:
   - Sentence embeddings: (batch, 768)
   - Project to embed_dim: (batch, 768) → (batch, embed_dim)
   - Expand to sequence: (batch, embed_dim) → (batch, 1, embed_dim)
   - Feed to aligner for CLIP-style alignment

### Alignment Comparison

**Traditional (Text Encoder)**:
```
Input Text → Tokenizer → Text Encoder → (batch, seq_len, embed_dim)
                                              ↓
EEG → EEG Encoder → Aligner ← ───────────────┘
```

**Sentence Embeddings**:
```
Precomputed Embeddings (batch, 768) → Projection → (batch, 1, embed_dim)
                                                          ↓
EEG → EEG Encoder → Aligner ← ────────────────────────────┘
```

## Top 3 Keywords

Keywords are extracted using KeyBERT during preprocessing:
- Stored in merged dataframe as `keyword_1`, `keyword_2`, `keyword_3`
- Embeddings stored in embeddings.pickle: `keyword_embeddings` (3, 768)
- Available in dataloader for future use

## Backward Compatibility

All changes maintain backward compatibility:

1. **No Embeddings**: Set `embeddings_path=None` - works as before
2. **No Sentence Embeddings**: Set `use_sentence_embeddings=False` - uses text encoder
3. **Other Models**: MLP and sentiment_cls_with_mlp models unaffected

## Testing

Syntax checks pass for all modified files:
```bash
python -m py_compile data/datamodule.py
python -m py_compile model/glim.py  
python -m py_compile train.py
python -m py_compile data/preprocess_*.py
```

## Future Enhancements

Potential extensions:
1. Use keyword embeddings for attention or additional supervision
2. Multi-task learning with both text encoder and sentence embeddings
3. Fine-tune sentence embedding projection layer separately
4. Experiment with different sentence embedding models
