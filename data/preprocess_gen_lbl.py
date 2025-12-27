import numpy as np
import pandas as pd
from generate_sentiment import generate as gen_senti_lbl
from load_mat import mat2df_zuco

data_dir = './datasets/ZuCo'
zuco1_task1_lbl_path = data_dir + '/task_materials/sentiment_labels_task1.csv'
zuco1_task2_lbl_path = data_dir + '/task_materials/relations_labels_task2.csv'
zuco1_task3_lbl_path = data_dir + '/task_materials/relations_labels_task3.csv'
zuco1_task1_mats_path = data_dir

########################
""" ZuCO 1.0 task 1 """
########################
df11_raw = pd.read_csv(zuco1_task1_lbl_path, 
                       sep=';', header=0,  skiprows=[1], encoding='utf-8',
                       dtype={'sentence': str, 'control': str, 'sentiment_label':str})
# print(df1_raw)
# n_row, n_column = df11_raw.shape
df11 = df11_raw.rename(columns={'sentence': 'raw text', 
                            'sentiment_label': 'raw label'})
df11 = df11.reindex(columns=['raw text', 'dataset', 'task', 'control', 'raw label',])
                      
df11['dataset'] =  ['ZuCo1'] * df11.shape[0]  # each item is init as a tuple with len==1 for easy extension
df11['task'] =  ['task1'] * df11.shape[0]
# drop unused column
df11 = df11.drop(['control'], axis = 1)

# print(df11.shape, df11.columns)
# print(df11['raw text'].nunique())

########################
""" ZuCO 1.0 task 2 """
########################
df12_raw = pd.read_csv(zuco1_task2_lbl_path, 
                       sep=',', header=0, encoding='utf-8',
                       dtype={'sentence': str,'control': str,'relation_types':str})
# n_row, n_column = df12_raw.shape
df12 = df12_raw.rename(columns={'sentence': 'raw text', 
                                'relation_types': 'raw label'})
df12 = df12.reindex(columns=['raw text', 'dataset', 'task', 'control', 'raw label',])
df12['dataset'] =  ['ZuCo1'] * df12.shape[0]
df12['task'] =  ['task2'] * df12.shape[0]
# drop unused column
df12 = df12.drop(['control'], axis = 1)

# print(df12.shape, df12.columns)
# print(df12['raw text'].nunique())

########################
""" ZuCO 1.0 task 3 """
########################
df13_raw = pd.read_csv(zuco1_task3_lbl_path, 
                       sep=';', header=0, encoding='utf-8', 
                       dtype={'sentence': str, 'relation-type':str})
df13 = df13_raw.rename(columns={'sentence': 'raw text', 
                            'relation-type': 'raw label'})
df13 = df13.reindex(columns=['raw text', 'dataset', 'task', 'control', 'raw label',])
df13['dataset'] =  ['ZuCo1'] * df13.shape[0]
df13['task'] =  ['task3'] * df13.shape[0]
# drop unused column
df13 = df13.drop(['control'], axis = 1)

# print(df13.shape, df13.columns)
# print(df13['raw text'].nunique())

#########################
""" Concat dataframes """
#########################
df = pd.concat([df11, df12, df13], ignore_index = True,)
# print(df.shape, df.columns)

####################
""" Revise typo """
####################
typobook = {"emp11111ty":   "empty",
            "film.1":       "film.",
            "–":            "-",
            "’s":           "'s",
            "�s":           "'s",
            "`s":           "'s",
            "Maria":        "Marić",
            "1Universidad": "Universidad",
            "1902—19":      "1902 - 19",
            "Wuerttemberg": "Württemberg",
            "long -time":   "long-time",
            "Jose":         "José",
            "Bucher":       "Bôcher",
            "1839 ? May":   "1839 - May",
            "G�n�ration":  "Generation",
            "Bragança":     "Bragana",
            "1837?October": "1837 - October",
            "nVera-Ellen":  "Vera-Ellen",
            "write Ethics": "wrote Ethics",
            "Adams-Onis":   "Adams-Onís",
            "(40 km?)":     "(40 km²)",
            "(40 km˝)":     "(40 km²)",
            " (IPA: /?g?nz?b?g/) ": " ",
            '""Canes""':    '"Canes"',

            }

def revise_typo(text):
    # the typo book 
    book = typobook
    for src, tgt in book.items():
        if src in text:
            text = text.replace(src, tgt)
    return text

df['input text'] = df['raw text'].apply(revise_typo)

# print(df.columns)
# print(df['raw text'].nunique(), df['input text'].nunique())

#################################
""" Generate sentiment label """
#################################
df['sentiment label'] = df['input text'].apply(gen_senti_lbl)

##################################
""" Assign relation label """
##################################
# Task 1 has sentiment labels in 'raw label', but we use generated sentiment labels
# Task 2 and 3 have relation labels in 'raw label'
# Set relation label to 'nan' for task1, and use 'raw label' for task2 and task3
df['relation label'] = df.apply(lambda row: 'nan' if row['task'] == 'task1' else str(row['raw label']), axis=1)

#########################
""" Assign Unique IDs """
#########################
uids, unique_texts = pd.factorize(df['input text'])
df['text uid'] = uids.tolist()

#######################
""" Save to pickle """
#######################
pd.to_pickle(df, './data/tmp/zuco_label_input_text.df')
df.to_csv('./data/tmp/zuco_label_input_text.csv')