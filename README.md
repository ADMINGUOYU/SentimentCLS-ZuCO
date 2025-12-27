# SentimentCLS-ZuCO

This repository is built on [GLIM](https://github.com/justin-xzliu/GLIM) and adapted to perform **sentiment classification** on all ZuCo 1.0 tasks.

## Key Changes from Original GLIM

1. **Sentiment labels** are generated for all ZuCo 1.0 tasks (task1, task2, task3)
2. **Relation labels** are preserved for task2 and task3 (set to 'nan' for task1)
3. Labels are stored as **text strings** (e.g., "negative", "neutral", "positive") to match the model's expectations

## Data Preprocessing Pipeline

The preprocessing consists of three main steps that must be run in order:

### 1. Process EEG Data (`data/preprocess_mat.py`)

Loads the raw EEG data from MATLAB files and processes it:
- Resamples EEG signals to 128Hz
- Pads sequences to fixed length (1280 timesteps, 128 channels)
- Filters out invalid samples
- Creates masks for valid timesteps

**Output**: `./data/tmp/zuco_eeg_128ch_1280len.df`

**Columns**: 'eeg', 'mask', 'text', 'dataset', 'task', 'subject'

### 2. Generate Labels (`data/preprocess_gen_lbl.py`)

Reads task material CSV files and generates sentiment labels:
- Loads sentiment labels for task1 from CSV (though these are regenerated)
- Loads relation labels for task2 and task3 from CSV
- **Generates sentiment labels** for all tasks using `cardiffnlp/twitter-roberta-base-sentiment-latest` model
- Assigns relation labels: 'nan' for task1, actual labels for task2/task3
- Applies typo corrections to text

**Output**: `./data/tmp/zuco_label_input_text.df`

**Columns**: 'raw text', 'dataset', 'task', 'raw label', 'input text', 'sentiment label', 'relation label', 'text uid'

### 3. Merge Data (`data/preprocess_merge.py`)

Combines EEG data with labels:
- Merges EEG dataframe with labels dataframe on ('text', 'dataset', 'task')
- Adds placeholder target text columns (uses input text as target for classification tasks)
- **Splits data** into train/val/test sets (70%/10%/20%) by text_uid
- Creates the final dataframe ready for training

**Output**: `./data/tmp/zuco_merged.df`

**Required columns**:
- EEG data: 'eeg', 'mask'
- Text: 'input text'
- Metadata: 'dataset', 'task', 'subject', 'text uid'
- Labels: 'sentiment label', 'relation label'
- Phase: 'phase' (train/val/test)
- Target texts: 'lexical simplification (v0)', 'lexical simplification (v1)', etc.

## Running Preprocessing

```bash
cd data
python preprocess_mat.py
python preprocess_gen_lbl.py
python preprocess_merge.py
```

## Label Format

### Sentiment Labels (all tasks)
Generated automatically using sentiment analysis pipeline:
- **"negative"** - negative sentiment
- **"neutral"** - neutral sentiment  
- **"positive"** - positive sentiment

### Relation Labels (task2 and task3 only)
From original ZuCo dataset:
- "awarding", "education", "employment", "foundation", "job title", "nationality", "political affiliation", "visit", "marriage"
- Set to **"nan"** for task1 (which has no relation labels)

## Model

The model (`model/glim.py`) is adapted for sentiment classification:
- `encode_labels()` method converts text labels to integer IDs
- `run_sentiment_cls()` performs sentiment classification using contrastive learning
- `run_relation_cls()` performs relation classification (for task2/task3)
- Both classification tasks use the aligned EEG-text embeddings

## Dataset and Dataloader

The dataloader (`data/datamodule.py`):
- Reads the merged pickle file
- Creates train/val/test datasets based on the 'phase' column
- Returns batches with EEG signals, text, and labels (both sentiment and relation)
- Uses custom `GLIMSampler` to ensure unique texts per batch for contrastive learning

## Training

### Quick Start

Train the model with default settings:

```bash
python train.py --data_path ./data/tmp/zuco_merged.df
```

### Training Options

```bash
python train.py \
  --data_path ./data/tmp/zuco_merged.df \
  --text_model google/flan-t5-large \
  --batch_size 48 \
  --val_batch_size 24 \
  --lr 1e-5 \
  --max_epochs 100 \
  --accelerator auto \
  --devices 1 \
  --precision bf16-mixed \
  --experiment_name sentiment_cls_experiment \
  --early_stopping \
  --patience 10
```

**Key Arguments:**

- `--data_path`: Path to merged dataset (default: `./data/tmp/zuco_merged.df`)
- `--text_model`: Pre-trained text model (default: `google/flan-t5-large`)
- `--batch_size`: Training batch size (default: 48)
- `--lr`: Learning rate (default: 1e-5)
- `--max_epochs`: Maximum training epochs (default: 100)
- `--accelerator`: Hardware accelerator (auto/cpu/gpu/tpu, default: auto)
- `--devices`: Number of devices to use (default: 1)
- `--precision`: Training precision (32/16-mixed/bf16-mixed, default: bf16-mixed)
- `--early_stopping`: Enable early stopping
- `--patience`: Early stopping patience (default: 10)
- `--resume_from_checkpoint`: Resume from checkpoint path

Run `python train.py --help` for full list of options.

### Viewing Training Progress

The training script logs all metrics to TensorBoard. To view logs:

```bash
# Option 1: Use the provided script
./view_logs.sh

# Option 2: Launch TensorBoard manually
tensorboard --logdir ./logs --port 6006
```

Then open your browser and navigate to: `http://localhost:6006`

**TensorBoard Features:**

- **Scalars**: Loss curves (total, CLIP, LM, commitment), accuracy metrics (sentiment/relation classification), retrieval accuracy
- **Hparams**: Hyperparameter comparison across experiments
- **Text**: Generated text samples and predictions during validation
- **Graphs**: Model architecture visualization

### Checkpoints

Model checkpoints are saved to `./checkpoints/` directory:
- Best models based on validation sentiment accuracy
- Last checkpoint for resuming training
- Top-k models (default: 3)

To resume training from a checkpoint:

```bash
python train.py --resume_from_checkpoint ./checkpoints/sentiment_classification/last.ckpt
```

## Evaluation

The model is evaluated on:
1. **Sentiment Classification**: Accuracy on all tasks (task1, task2, task3)
2. **Relation Classification**: Accuracy on task2 and task3
3. **Text Generation**: BLEU, ROUGE scores for paraphrase generation
4. **Retrieval**: EEG-text retrieval accuracy

Metrics are logged per-task and averaged across all tasks.

