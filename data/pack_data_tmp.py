import numpy as np
import pandas as pd
import sys
import os
import pickle

tmp_path = '/nfs/usrhome2/yguoco/checkpoints_sentiment_cls_with_mlp/tmp'

# load embeddings
with open(tmp_path + "/embeddings.pickle", "rb") as f:
    embedding_dict = pickle.load(f)
# load dataframe
input_text = pd.read_pickle(tmp_path + '/zuco_label_input_text.df')

# save lists
task_1 = [ ]
task_2 = [ ]
task_3 = [ ]

# loop through all entries
print(len(embedding_dict), len(input_text))
for i, entry in input_text.iterrows():
    uid = entry['text uid']
    emb = embedding_dict[uid]['sentence']
    text = entry['input text']
    task = entry['task']
    if task == 'task1':
        idx = len(task_1)
        task_1.append((idx, text, emb))
    elif task == 'task2':
        idx = len(task_2)
        task_2.append((idx, text, emb))
    elif task == 'task3':
        idx = len(task_3)
        task_3.append((idx, text, emb))
    else:
        raise

# display
print(len(task_1), len(task_2), len(task_3))

# save results
with open("./data/tmp/task_1.pickle", "wb") as f:
    pickle.dump(task_1, f, protocol = pickle.HIGHEST_PROTOCOL)
with open("./data/tmp/task_2.pickle", "wb") as f:
    pickle.dump(task_2, f, protocol = pickle.HIGHEST_PROTOCOL)
with open("./data/tmp/task_3.pickle", "wb") as f:
    pickle.dump(task_3, f, protocol = pickle.HIGHEST_PROTOCOL)
