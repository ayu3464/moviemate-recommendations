from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.data_cleaning import clean_movie_data
from src.utils import MODEL_DIR, PROCESSED_DATA_PATH


def normalize_tokens(value):
    if pd.isna(value):
        return ""
    text = str(value)
    text = text.replace("[", " ").replace("]", " ")
    text = text.replace("'", " ").replace('"', " ")
    text = text.replace("{", " ").replace("}", " ")
    return " ".join(text.lower().split())


def parse_genre_field(value):
    if pd.isna(value):
        return ""
    text = str(value)
    if text.startswith("["):
        try:
            import ast
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return " ".join(str(item).strip() for item in parsed if str(item).strip())
        except Exception:
            pass
    return " ".join(part.strip() for part in text.split(",") if part.strip())


def build_combined_feature(df):
    working = df.copy()
    working["genres"] = working["genres"].fillna("Unknown").apply(parse_genre_field)
    working["overview"] = working["overview"].fillna("").astype(str)
    working["keywords"] = working["keywords"].fillna("").apply(normalize_tokens)
    working["original_language"] = working["original_language"].fillna("").astype(str).str.lower()
    working["combined_feature"] = (
        working["genres"].fillna("")
        + " "
        + working["overview"].fillna("")
        + " "
        + working["keywords"].fillna("")
        + " "
        + working["original_language"].fillna("")
    )
    return working


def build_similarity_model(df=None, save_dir=MODEL_DIR):
    if df is None:
        if not Path(PROCESSED_DATA_PATH).exists():
            df = clean_movie_data()
        else:
            df = pd.read_csv(PROCESSED_DATA_PATH)

    working = build_combined_feature(df)
    working["combined_feature"] = working["combined_feature"].fillna("").apply(lambda x: " ".join(str(x).split()).lower())
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)
    matrix = vectorizer.fit_transform(working["combined_feature"])
    similarity = cosine_similarity(matrix)

    save_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, save_dir / "tfidf_vectorizer.pkl")
    np.save(save_dir / "cosine_similarity.npy", similarity)

    return working, vectorizer, similarity


if __name__ == "__main__":
    df, vectorizer, similarity = build_similarity_model()
    print(f"Prepared {len(df)} movies with TF-IDF matrix size {similarity.shape}")
