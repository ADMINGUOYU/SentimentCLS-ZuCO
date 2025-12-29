import torch
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
import numpy
import typing

# Use CUDA if available, otherwise fallback to CPU
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# set up model
sbert_model = SentenceTransformer('all-mpnet-base-v2', device = device)
kw_model = KeyBERT(model = sbert_model) # Reuse model to save memory

def generate_embedding(text: typing.Union[str, typing.List[str]]) -> numpy.ndarray:
    return sbert_model.encode(text)

def generate_top_3_keywords(text: str) -> typing.List[str]:
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 1), stop_words='english', top_n=3)
    return [k[0] for k in keywords] + [''] * (3 - len(keywords))
