import numpy as np
import pandas as pd

from .taxonomy import mood_categories, recommendation_ratios, get_main_category
from .similarity import compute_similarity

def recommend_books(
    text: str,
    df_books: pd.DataFrame,
    book_vectors: np.ndarray,
    classifier,
    max_books: int = 8
):
    output = classifier(text)
    output = output[0] if isinstance(output[0], list) else output

    output = sorted(output, key=lambda x: x["score"], reverse=True)
    top_emotion = output[0]["label"]
    intensity = output[0]["score"]

    total = int(np.clip(intensity * 12, 3, max_books))

    user_vec = np.array([o["score"] for o in output], dtype="float32")
    similarity = compute_similarity(book_vectors, user_vec)

    local_df = df_books.copy()
    local_df["match_score"] = similarity

    main_cat = get_main_category(top_emotion)
    ratios = recommendation_ratios.get(main_cat, recommendation_ratios["neutral"])

    results = []
    picked = set()

    for cat, ratio in ratios.items():
        count = max(1, round(total * ratio))
        emotions = mood_categories.get(cat)

        if not emotions:
            continue

        local_df["cat_strength"] = local_df[emotions].sum(axis=1)
        local_df["final_rank"] = local_df["match_score"] * local_df["cat_strength"]

        candidates = (
            local_df[~local_df["Title"].isin(picked)]
            .sort_values("final_rank", ascending=False)
            .head(count)
        )

        for _, row in candidates.iterrows():
            results.append({
                "title": row["Title"],
                "reason": cat,
                "score": round(float(row["final_rank"]), 4)
            })
            picked.add(row["Title"])

    return {
        "detected_mood": top_emotion,
        "count": len(results),
        "recommendations": results
    }
