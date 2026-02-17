# 📚 Mood-Based Book Recommendation System

A production-ready **emotion-aware book recommender system** that suggests books based on a user’s emotional state extracted from raw natural-language input.

The system uses **precomputed emotion profiles for books** and matches them against the **emotion distribution inferred from user text**, enabling fast, scalable, and explainable recommendations.

---

## 🚀 Live Demo

👉 **Streamlit App**: *(add your deployed URL here)*  
⏱️ **Note**: First load may take ~30–60 seconds due to model cold start on free tier.

---

## 🧠 Problem Statement

Traditional book recommendation systems rely on:
- ratings,
- popularity,
- collaborative filtering.

These approaches fail when:
- a user is new (cold start),
- the user wants recommendations based on **how they feel**, not past behavior.

**Goal**:  
Recommend books that align with or appropriately respond to the user’s **emotional state**, inferred directly from free-form text.

---

## 🧩 Solution Overview

This project implements a **content-based emotional recommender system** with the following design:

```
User text
   ↓
Emotion inference (GO-Emotions)
   ↓
Emotion vector (28-dimensional)
   ↓
Cosine similarity with book emotion vectors
   ↓
Emotion-aware re-ranking
   ↓
Final book recommendations
```

### Key design choice
> **All book emotion inference is done offline.**  
> Online inference is limited to user input only.

This ensures:
- low latency,
- scalability,
- predictable performance.

---

## 🏗️ Architecture

### High-Level Architecture

![Architecture]("ui\assets\architecture.png")

## 📁 Project Structure

```
mood-book-recommender/
│
├── data/
│   └── final_df_books.csv        # Books + emotion scores
│
├── src/
│   ├── models/
│   │   └── emotion_model.py      # GO-Emotions loader
│   │
│   ├── recommender/
│   │   ├── taxonomy.py           # Emotion groupings
│   │   ├── similarity.py         # Cosine similarity
│   │   └── engine.py             # Recommendation logic
│   │
│   └── utils/
│       └── data_loader.py        # CSV loading & normalization
│
├── ui/
│   ├── streamlit_app.py          # Streamlit UI
│   └── assets/
│       └── no_cover.png          # Image fallback
│
├── requirements.txt
└── README.md
```

---

## 🔬 Emotion Modeling

### Emotion Classifier
- **Model**: `SamLowe/roberta-base-go_emotions`
- **Dataset**: Google GO-Emotions
- **Output**: 28 emotion probabilities per input text

---

## 📊 Book Representation

Each book is represented as a **28-dimensional emotion vector** derived from:
- title
- subtitle
- description

Emotion scores are:
- inferred offline,
- normalized,
- stored in CSV.

---

## 📐 Recommendation Logic

1. Emotion inference on user text  
2. Cosine similarity against all book vectors  
3. Emotion-aware re-ranking using curated heuristics  

---

## 🎨 User Interface

Built using **Streamlit** with:
- natural-language input
- emotion detection feedback
- book cards with Open Library covers
- skeleton loaders and fallbacks

---

## ⚙️ Tech Stack

- Python
- Hugging Face Transformers
- RoBERTa (GO-Emotions)
- Pandas / NumPy
- Streamlit

---

## 🚀 Running Locally

```bash
pip install -r requirements.txt
streamlit run ui/streamlit_app.py
```

---

## ⚠️ Known Limitations

- Emotion inference from short text is noisy
- Title-based cover lookup may be inaccurate
- Cultural bias from training data

---

## 👤 Author

**Sid**  
Software Engineer | AI/ML Enthusiast
