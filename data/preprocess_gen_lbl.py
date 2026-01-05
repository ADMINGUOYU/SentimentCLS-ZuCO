import numpy as np
import pandas as pd
from generate_sentiment import generate as gen_senti_lbl
import generate_embedding
import pickle
import os

data_dir = './datasets/ZuCo'
zuco1_task1_lbl_path = data_dir + '/revised_csv/sentiment_labels_task1.csv'
zuco1_task2_lbl_path = data_dir + '/revised_csv/relations_labels_task2.csv'
zuco1_task3_lbl_path = data_dir + '/revised_csv/relations_labels_task3.csv'

# tmp path (saving path)
# User can modify this path to save outputs to a different location
# Default: './tmp' for local storage
tmp_path = '/nfs/usrhome2/yguoco/checkpoints_sentiment_cls_with_mlp/tmp'
# Create tmp directory if it doesn't exist
os.makedirs(tmp_path, exist_ok=True)

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

#########################################
""" Generate keywords and embeddings """
#########################################
# generate full sentence embeddings
sentence_embeddings = generate_embedding.generate_embedding(df['input text'].to_list())
assert sentence_embeddings.shape[1] == 768, "[ERROR] sentence embeddings is expected in shape (N, 768)"
# get top 3 key words
keywords_list = []
for text in df['input text']:
    keywords = generate_embedding.generate_top_3_keywords(text)
    keywords_list.append(keywords)
# flatten all keywords
flat_keywords = [kw for sublist in keywords_list for kw in sublist]
# encode all key words
kw_embeddings_flat = generate_embedding.generate_embedding(flat_keywords)
# reshape (N*3, 768) to (N, 3, 768)
num_sentences = len(df)
keyword_embeddings = kw_embeddings_flat.reshape(num_sentences, 3, 768)
assert (keyword_embeddings.shape[1] == 3) and (keyword_embeddings.shape[2] == 768), "[ERROR] top 3 words' embeddings is expected in shape (N, 3, 768)"
# Prepare metadata DataFrame (text only)
keywords_df = pd.DataFrame(keywords_list, columns = ['keyword_1', 'keyword_2', 'keyword_3'])
df = pd.concat([df, keywords_df], axis = 1)

#########################
""" Assign Unique IDs """
#########################
uids, unique_texts = pd.factorize(df['input text'])
df['text uid'] = uids.tolist()

#######################################
""" Assign embeddings to Unique IDs """
#######################################
assert (sentence_embeddings.shape[0] == len(df)) and (keyword_embeddings.shape[0] == len(df)), "[ERROR] Expected number of sentences to be the same"
embedding_dict = { }
for row, uid in enumerate(df['text uid']):
    embedding_dict[uid] = { 'sentence' : sentence_embeddings[row, : ], 'keyword' : keyword_embeddings[row, : , : ] }

#######################
""" Save to pickle """
#######################
# # Save sentence embeddings (N, 768)
# sent_npy_path = tmp_path + '/zuco_sentence_embeddings.npy' # shape (N, 768)
# np.save(sent_npy_path, sentence_embeddings)
# # Save keyword embeddings (N, 3, 768)
# kw_npy_path = tmp_path + '/zuco_keyword_embeddings.npy'   # shape (N, 3, 768)
# np.save(kw_npy_path, keyword_embeddings)
# save embeddings
with open(tmp_path + "/embeddings.pickle", "wb") as f:
    pickle.dump(embedding_dict, f, protocol = pickle.HIGHEST_PROTOCOL)
# save dataframe
pd.to_pickle(df, tmp_path + '/zuco_label_input_text.df')
df.to_csv(tmp_path + '/zuco_label_input_text.csv')