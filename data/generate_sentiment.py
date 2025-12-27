import torch
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from transformers import pipeline

sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest", device = 'cuda:0')
sentiment_label = {'negative' : 0,
                   'neutral' : 1,
                   'positive' : 2}

def generate(text:str) -> int:
    # 0:Negative, 1:Neutral, 2:Positive
    return sentiment_label[sentiment_pipeline(text)[0]['label']]
