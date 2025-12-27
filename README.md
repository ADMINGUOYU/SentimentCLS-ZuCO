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
