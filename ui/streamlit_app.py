import streamlit as st
import requests
from PIL import Image
from io import BytesIO
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
# Skeleton CSS
# ------------------------------------------------------------------
st.markdown("""
<style>
.skeleton {
  width: 100%;
  height: 350px;
  background: linear-gradient(
    90deg,
    #eeeeee 25%,
    #dddddd 37%,
    #eeeeee 63%
  );
  animation: shimmer 1.4s infinite;
  border-radius: 8px;
  margin-bottom: 10px;
}

@keyframes shimmer {
  0% { background-position: -450px 0; }
  100% { background-position: 450px 0; }
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Load heavy resources ONCE
# ------------------------------------------------------------------
@st.cache_resource
def load_resources():
    classifier = load_emotion_classifier(device=-1)  # CPU only
    df_books, book_vectors = load_books("data/final_df_books.csv")
    default_image = Image.open("ui/assets/no_cover.png")
    return classifier, df_books, book_vectors, default_image


classifier, df_books, book_vectors, DEFAULT_IMAGE = load_resources()

# ------------------------------------------------------------------
# Open Library cover fetcher
# ------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def fetch_cover_image(title: str, timeout: float = 2.5):
    try:
        search_url = (
            "https://openlibrary.org/search.json"
            f"?title={title.replace(' ', '+')}&limit=1"
        )
        search_res = requests.get(search_url, timeout=timeout)
        search_res.raise_for_status()

        docs = search_res.json().get("docs", [])
        if not docs or "cover_i" not in docs[0]:
            return None

        cover_id = docs[0]["cover_i"]
        img_url = f"https://covers.openlibrary.org/b/id/{cover_id}-M.jpg"

        img_res = requests.get(img_url, timeout=timeout)
        img_res.raise_for_status()

        return Image.open(BytesIO(img_res.content))

    except Exception:
        return None

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
        use_container_width=True
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
    recommendations = result.get("recommendations", [])

    with results_container.container():
        st.success(f"Detected Mood: **{mood.capitalize()}**")

        if not recommendations:
            st.info("No recommendations found.")
            st.stop()

        st.subheader("📖 Recommended Books")
        cols = st.columns(2)

        for idx, book in enumerate(recommendations):
            with cols[idx % 2]:

                # Skeleton placeholder
                placeholder = st.empty()
                placeholder.markdown(
                    '<div class="skeleton"></div>',
                    unsafe_allow_html=True
                )

                image = fetch_cover_image(book["title"])
                placeholder.empty()

                if image:
                    st.image(image, use_container_width=True)
                else:
                    st.image(DEFAULT_IMAGE, use_container_width=True)

                st.markdown(f"**{book['title']}**")
                st.caption(f"Why: {book['reason']}")
                st.caption(f"Match score: {book['score']}")
                st.divider()
