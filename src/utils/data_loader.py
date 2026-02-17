import pandas as pd
import numpy as np

EMOTION_COLUMNS = [
    'neutral', 'approval', 'annoyance', 'realization', 'admiration',
    'disappointment', 'disapproval', 'excitement', 'sadness', 'anger',
    'disgust', 'amusement', 'joy', 'confusion', 'fear', 'optimism',
    'curiosity', 'love', 'surprise', 'desire', 'gratitude', 'caring',
    'embarrassment', 'grief', 'pride', 'nervousness', 'relief', 'remorse'
]

def load_books(csv_path: str):
    df = pd.read_csv(csv_path)
    df_books = df.drop(columns={'Description', 'full_txt'}, errors='ignore')
    vectors = df_books[EMOTION_COLUMNS].values.astype("float32")

    # normalize once
    vectors /= (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9)

    return df_books, vectors
