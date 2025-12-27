import numpy as np
import pandas as pd
import sys
import os
from sklearn.model_selection import train_test_split

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))
from preprocess_gen_lbl import revise_typo

# Configuration constants
TARGET_KEYS = [
    'lexical simplification (v0)', 'lexical simplification (v1)', 
    'semantic clarity (v0)', 'semantic clarity (v1)', 
    'syntax simplification (v0)', 'syntax simplification (v1)',
    'naive rewritten', 'naive simplified'
]
TEST_SIZE = 0.2
VAL_SIZE = 0.1  # 10% of total data
RANDOM_SEED = 42

"""
Merge the EEG data with the sentiment and relation labels.
This script assumes that:
1. preprocess_mat.py has been run to generate zuco_eeg_128ch_1280len.df
2. preprocess_gen_lbl.py has been run to generate zuco_label_input_text.df
"""

# Load the EEG data
df_eeg = pd.read_pickle('./data/tmp/zuco_eeg_128ch_1280len.df')
print(f"Loaded EEG data: {df_eeg.shape[0]} rows")
print(f"EEG columns: {df_eeg.columns.tolist()}")

# Load the label data
df_labels = pd.read_pickle('./data/tmp/zuco_label_input_text.df')
print(f"Loaded label data: {df_labels.shape[0]} rows")
print(f"Label columns: {df_labels.columns.tolist()}")

# Apply the same typo corrections to the EEG text column
df_eeg['text'] = df_eeg['text'].apply(revise_typo)

# Merge the dataframes on 'text', 'dataset', 'task', and 'subject'
# The EEG data uses 'text' column, while labels use 'input text'
df_merged = pd.merge(df_eeg, 
                     df_labels[['input text', 'sentiment label', 'relation label', 'text uid', 'dataset', 'task']], 
                     left_on=['text', 'dataset', 'task'], 
                     right_on=['input text', 'dataset', 'task'], 
                     how='inner')

print(f"Merged data: {df_merged.shape[0]} rows")
print(f"Merged columns: {df_merged.columns.tolist()}")

# Drop the redundant 'text' column
df_merged = df_merged.drop(['text'], axis=1)

# Add target text columns - for sentiment classification, we use input text as target
# These columns are needed by the dataloader for paraphrasing evaluation compatibility
# Note: In this sentiment classification task, we're not doing paraphrasing, but the 
# dataloader expects these columns. Using input text as target is appropriate since
# we're focused on classification accuracy, not text generation quality.
for key in TARGET_KEYS:
    df_merged[key] = df_merged['input text']

# Assign train/val/test split
# Split by text uid to ensure no data leakage
unique_text_uids = df_merged['text uid'].unique()
train_uids, test_uids = train_test_split(unique_text_uids, test_size=TEST_SIZE, random_state=RANDOM_SEED)
# Calculate correct validation split size from remaining training data
val_split_ratio = VAL_SIZE / (1 - TEST_SIZE)
train_uids, val_uids = train_test_split(train_uids, test_size=val_split_ratio, random_state=RANDOM_SEED)

def assign_phase(text_uid):
    if text_uid in train_uids:
        return 'train'
    elif text_uid in val_uids:
        return 'val'
    else:
        return 'test'

df_merged['phase'] = df_merged['text uid'].apply(assign_phase)

print(f"Final merged data: {df_merged.shape[0]} rows")
print(f"Train: {(df_merged['phase'] == 'train').sum()}, Val: {(df_merged['phase'] == 'val').sum()}, Test: {(df_merged['phase'] == 'test').sum()}")
print(f"Columns: {df_merged.columns.tolist()}")

# Save the merged dataframe
pd.to_pickle(df_merged, './data/tmp/zuco_merged.df')
print("Saved merged data to ./data/tmp/zuco_merged.df")
