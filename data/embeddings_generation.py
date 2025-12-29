import os
os.environ['HF_HOME'] = '/mnt/afs/250010218/hf_cache'
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT

# 1. Load raw data
csv_input_path = 'zuco_label_input_text/zuco_label_input_text.csv'
data = pd.read_csv(csv_input_path)

# 2. Generate sentence embeddings (Shape: N, 768)
sbert_model = SentenceTransformer('all-mpnet-base-v2')
sentence_embeddings = sbert_model.encode(data['raw text'].tolist(), show_progress_bar=True)

# 3. Extract keyword text (Top 3)
kw_model = KeyBERT(model=sbert_model) # Reuse model to save memory
keywords_list = []
for text in data['raw text']:
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 1), stop_words='english', top_n=3)
    # Pad with empty strings if less than 3 keywords found
    keywords_list.append([k[0] for k in keywords] + [''] * (3 - len(keywords)))

# 4. Generate keyword embeddings
# Flatten for batch encoding efficiency
flat_keywords = [kw for sublist in keywords_list for kw in sublist]

print("Encoding keywords...")
kw_embeddings_flat = sbert_model.encode(flat_keywords, show_progress_bar=True)

# Reshape from (N*3, 768) to (N, 3, 768)
# Index [i, 0, :] corresponds to Top 1 keyword
num_sentences = len(data)
keyword_embeddings = kw_embeddings_flat.reshape(num_sentences, 3, 768)

# 5. Prepare metadata DataFrame (text only)
keywords_df = pd.DataFrame(keywords_list, columns=['keyword_1', 'keyword_2', 'keyword_3'])
data_new = pd.concat([data, keywords_df], axis=1)

# 6. Save outputs
# Save text metadata
csv_output_path = 'zuco_label_input_text/zuco_label_input_text_new.csv'
data_new.to_csv(csv_output_path, index=False)

# Save sentence embeddings (N, 768)
sent_npy_path = 'zuco_label_input_text/zuco_sentence_embeddings.npy'
np.save(sent_npy_path, sentence_embeddings)

# Save keyword embeddings (N, 3, 768)
kw_npy_path = 'zuco_label_input_text/zuco_keyword_embeddings.npy'
np.save(kw_npy_path, keyword_embeddings)

print("-" * 30)
print(f"Processing Complete!")
print(f"1. CSV metadata saved: {csv_output_path}")
print(f"2. Sentence embeddings shape: {sentence_embeddings.shape}")
print(f"3. Keyword embeddings shape: {keyword_embeddings.shape}")
print("-" * 30)