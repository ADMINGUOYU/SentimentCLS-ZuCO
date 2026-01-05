import numpy as np
import pandas as pd
import sys
import os
from sklearn.model_selection import train_test_split

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))
# from preprocess_gen_lbl import revise_typo

# tmp path (saving path)
# User can modify this path to save outputs to a different location
# Default: './tmp' for local storage
tmp_path = '/nfs/usrhome2/yguoco/checkpoints_sentiment_cls_with_mlp/tmp'
# Create tmp directory if it doesn't exist
os.makedirs(tmp_path, exist_ok=True)

"""
Please make sure you have zuco_merged.df and zuco_label_8variants.df ready
!!!!!! Important !!!!!!
"""

# load dataframes
merged_df: pd.DataFrame = pd.read_pickle(tmp_path + '/zuco_merged.df')
variants: pd.DataFrame = pd.read_pickle(tmp_path + '/zuco_label_8variants.df')
# NOTE: merged_df shall have more rows than label variants
# assert len(merged_df) == len(variants), f"[ERROR] zuco_merged.df({len(merged_df)}) and zuco_label_8variants.df({len(variants)}) shall at least have same length"

# Configuration constants
TARGET_KEYS = [
    'lexical simplification (v0)', 'lexical simplification (v1)', 
    'semantic clarity (v0)', 'semantic clarity (v1)', 
    'syntax simplification (v0)', 'syntax simplification (v1)',
    'naive rewritten', 'naive simplified'
]

# iterate all rows of zuco_merged, check alignment
for idx in range(len(merged_df)):
    # the text is:
    text: str = merged_df.iloc[idx]['input text']
    # get row index of 8variants
    index = variants.index[variants['input text'] == text].to_list()
    assert len(index) >= 1, "[ERROR] cannot locate input text, merge error"
    index = index[0]
    # copy variants
    for key in TARGET_KEYS:
        merged_df.loc[idx, key] = variants.iloc[index][key]

print(merged_df.iloc[99])
raise "STOP"
# Save the merged dataframe
save_location = tmp_path + '/zuco_merged_with_variants.df'
pd.to_pickle(merged_df, save_location)
print(f"Saved merged data to {save_location}")
