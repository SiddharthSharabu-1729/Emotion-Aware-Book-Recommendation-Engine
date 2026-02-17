# import pandas as pd
# import numpy as np

# EMOTION_COLUMNS = [
#     'neutral', 'approval', 'annoyance', 'realization', 'admiration',
#     'disappointment', 'disapproval', 'excitement', 'sadness', 'anger',
#     'disgust', 'amusement', 'joy', 'confusion', 'fear', 'optimism',
#     'curiosity', 'love', 'surprise', 'desire', 'gratitude', 'caring',
#     'embarrassment', 'grief', 'pride', 'nervousness', 'relief', 'remorse'
# ]

# def load_books(csv_path: str):
#     df = pd.read_csv(csv_path)
#     df_books = df.drop(columns={'Description', 'full_txt'}, errors='ignore')
#     vectors = df_books[EMOTION_COLUMNS].values.astype("float32")

#     # normalize once
#     vectors /= (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9)

#     return df_books, vectors

from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]

def load_books(path: str):
    full_path = BASE_DIR / path

    if not full_path.exists():
        raise FileNotFoundError(f"{full_path} not found")

    if full_path.suffix == ".parquet":
        df = pd.read_parquet(full_path)
    else:
        df = pd.read_csv(full_path)

    df_books = df.drop(columns={'Description', 'full_txt'}, errors='ignore')

    vectors = df_books.select_dtypes(include=[np.number]).values.astype("float32")
    vectors /= (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9)

    return df_books, vectors
