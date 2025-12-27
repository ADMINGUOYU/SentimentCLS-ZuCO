import torch
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from transformers import pipeline

sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest", device = 'cuda:0')

def generate(text:str) -> str:
    """
    Generate sentiment label for the given text.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Sentiment label as string: 'negative', 'neutral', or 'positive'
        
    Raises:
        ValueError: If the sentiment pipeline returns an unexpected label
    """
    try:
        result = sentiment_pipeline(text)
        label = result[0]['label']
        # Validate the label is one of the expected values
        if label not in ['negative', 'neutral', 'positive']:
            raise ValueError(f"Unexpected sentiment label: {label}")
        return label
    except Exception as e:
        print(f"Error analyzing sentiment for text: {text[:50]}...")
        print(f"Error: {e}")
        # Return neutral as default for failed cases
        return 'neutral'
