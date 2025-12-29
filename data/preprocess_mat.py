import numpy as np
import pandas as pd
from load_mat import mat2df_zuco

data_dir = './datasets/ZuCo'
zuco1_task1_mats_path = data_dir

# tmp path (saving path)
tmp_path = '/nfs/usrhome2/yguoco/checkpoints_sentiment_cls_with_mlp/tmp'

########################################
""" Process mat: ZuCO 1.0 Task 1 - 3 """
########################################
df_zuco1 = mat2df_zuco(dataset_name='ZuCo1',
                       eeg_src_dir = zuco1_task1_mats_path,
                       task_dir_names = ['task1-SR', 'task2-NR', 'task3-TSR'],
                       task_keys = ['task1', 'task2', 'task3'],
                       subject_keys = ['ZAB', 'ZDM', 'ZDN', 'ZGW', 'ZJM', 'ZJN', \
                                       'ZJS', 'ZKB', 'ZKH', 'ZKW', 'ZMG', 'ZPH'],
                       n_sentences = [400, 300, 407])

#########################
""" Concat dataframes """
#########################
df = df_zuco1

#######################
""" Save to pickle """
#######################
pd.to_pickle(df, tmp_path + '/zuco_eeg_128ch_1280len.df')