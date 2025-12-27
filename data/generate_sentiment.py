import torch
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from transformers import pipeline

sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest", device = 'cuda:0')
sentiment_label_map = {'negative' : 0,
                       'neutral' : 1,
                       'positive' : 2}

def generate(text:str) -> str:
    # Returns the sentiment label as string: 'negative', 'neutral', or 'positive'
    return sentiment_pipeline(text)[0]['label']
