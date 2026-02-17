from fastapi import FastAPI
from pydantic import BaseModel

from src.models.emotion_model import load_emotion_classifier
from src.utils.data_loader import load_books
from src.recommender.engine import recommend_books

app = FastAPI(title="Mood-Based Book Recommender")

classifier = load_emotion_classifier(device=-1)
df_books, book_vectors = load_books("data/final_df_books.csv")

class UserInput(BaseModel):
    text: str

@app.post("/recommend")
def recommend(input: UserInput):
    return recommend_books(
        text=input.text,
        df_books=df_books,
        book_vectors=book_vectors,
        classifier=classifier
    )
