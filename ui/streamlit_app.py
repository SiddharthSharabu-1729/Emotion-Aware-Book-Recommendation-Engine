import streamlit as st
import requests
import sys
from pathlib import Path

# ------------------------------------------------------------------
# Fix Python path so `src` is importable
# ------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.models.emotion_model import load_emotion_classifier
from src.utils.data_loader import load_books
from src.recommender.engine import recommend_books

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Mood-Based Book Recommender",
    page_icon="📚",
    layout="centered"
)

# ------------------------------------------------------------------
# Load heavy resources ONCE
# ------------------------------------------------------------------
@st.cache_resource
def load_resources():
    classifier = load_emotion_classifier(device=-1)  # CPU only
    df_books, book_vectors = load_books("data/final_df_books.parquet")
    return classifier, df_books, book_vectors

classifier, df_books, book_vectors = load_resources()

# ------------------------------------------------------------------

# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
st.title("📚 Mood-Based Book Recommender")
st.write(
    "Describe how you're feeling, and get book recommendations "
    "based on emotional similarity."
)

# ---- FORM (prevents scroll jump) ----
with st.form("recommendation_form"):
    user_text = st.text_area(
        "How are you feeling right now?",
        placeholder="Example: I feel lonely and disconnected lately...",
        height=120
    )
    submitted = st.form_submit_button(
        "Recommend Books",
        width="stretch"
    )

# ---- Results container (clears old UI) ----
results_container = st.empty()

# ------------------------------------------------------------------
# Handle submission
# ------------------------------------------------------------------
if submitted:
    results_container.empty()  # clear previous results

    if not user_text.strip():
        st.warning("Please enter some text describing your mood.")
        st.stop()

    with st.spinner("Analyzing mood and finding books..."):
        result = recommend_books(
            text=user_text,
            df_books=df_books,
            book_vectors=book_vectors,
            classifier=classifier
        )

    mood = result.get("detected_mood", "unknown")
    is_confident = result.get("is_confident", True)
    recommendations = result.get("recommendations", [])

    with results_container.container():
        if not is_confident:
            st.warning("I couldn't detect a strong emotion from your text. Tell me a bit more about how you feel!")
            st.stop()
            
        st.success(f"Detected Mood: **{mood.capitalize()}**")

        if not recommendations:
            st.info("No recommendations found.")
            st.stop()

        st.subheader("📖 Recommended Books")
        
        import pandas as pd
        import urllib.parse
        
        # Add a Goodreads search link column
        data_for_df = []
        for book in recommendations:
            goodreads_link = f"https://www.goodreads.com/search?q={urllib.parse.quote_plus(book['title'])}"
            data_for_df.append({
                "Title": book['title'],
                "Recommendation Reason": book['reason'],
                "Match Score": book['score'],
                "Goodreads": goodreads_link
            })
        df_results = pd.DataFrame(data_for_df)

        # Display as a dataframe (interactive table)
        st.dataframe(
            df_results,
            width='stretch',
            hide_index=True,
            column_config={
                "Title": st.column_config.TextColumn("Title"),
                "Recommendation Reason": st.column_config.TextColumn(
                    "Recommendation Reason",
                    width="small"
                ),
                "Match Score": st.column_config.NumberColumn(
                    "Match Score",
                    format="%.4f",
                    width="small"
                ),
                "Goodreads": st.column_config.LinkColumn(
                    "Goodreads Link",
                    width="medium",
                    display_text="Search Here"
                )
            }
        )
        
        st.divider()
        st.write("How were these recommendations?")
        st.feedback("thumbs")
