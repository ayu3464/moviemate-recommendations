from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_cleaning import clean_movie_data
from src.feature_engineering import build_similarity_model
from src.utils import PROCESSED_DATA_PATH


@lru_cache(maxsize=1)
def get_dataset():
    if not Path(PROCESSED_DATA_PATH).exists():
        clean_movie_data()
    return pd.read_csv(PROCESSED_DATA_PATH)


@lru_cache(maxsize=1)
def get_model_bundle():
    df = get_dataset()
    working, vectorizer, similarity = build_similarity_model(df, save_dir=Path(__file__).resolve().parents[1] / "models")
    return working, vectorizer, similarity


def _normalize_genres(value):
    if pd.isna(value):
        return []
    text = str(value)
    return [item.strip() for item in text.replace("[", "").replace("]", "").split(",") if item.strip()]


def _score_range(value, min_value, max_value):
    if pd.isna(value):
        return 0.0
    if value < min_value:
        return max(0.0, 1 - ((min_value - value) / max(1, min_value)))
    if value > max_value:
        return max(0.0, 1 - ((value - max_value) / max(1, max_value)))
    return 1.0


def _genre_match_score(candidate_genres, preferred_genres):
    if not preferred_genres:
        return 0.5
    candidate_set = set(_normalize_genres(candidate_genres))
    if not candidate_set:
        return 0.0
    overlap = len(set(preferred_genres) & candidate_set)
    return overlap / max(1, len(set(preferred_genres)))


def _build_reason_list(movie, favorite_titles, preferences):
    reasons = []
    favorite_names = [title.strip() for title in favorite_titles if title and title.strip()]
    if favorite_names:
        reasons.append(f"Similar to your favorite movie(s): {', '.join(favorite_names[:2])}")
    if preferences.get("genres"):
        reasons.append("Matches your preferred genre")
    if movie["vote_average"] >= preferences.get("min_rating", 0):
        reasons.append("Meets your minimum rating threshold")
    if preferences.get("language") and movie["original_language"].lower() == str(preferences.get("language")).lower():
        reasons.append("Matches your preferred language")
    if preferences.get("year_from") and preferences.get("year_to"):
        reasons.append("Released within your selected year range")
    if preferences.get("mood"):
        reasons.append(f"Aligned with your mood: {preferences['mood']}")
    if not reasons:
        reasons.append("Strong match based on content similarity")
    return reasons[:4]


def recommend_movies(preferences, favorite_movies=None, top_n=8):
    if favorite_movies is None:
        favorite_movies = []

    df = get_dataset()
    working, vectorizer, similarity = get_model_bundle()

    preferred_genres = preferences.get("genres", [])
    min_rating = float(preferences.get("min_rating", 0.0))
    year_from = int(preferences.get("year_from", 1900))
    year_to = int(preferences.get("year_to", 2035))
    preferred_language = preferences.get("language")
    min_runtime = float(preferences.get("min_runtime", 0))
    max_runtime = float(preferences.get("max_runtime", 500))
    mood = preferences.get("mood", "")

    filtered = df.copy()
    if preferred_genres:
        filtered = filtered[filtered["genres"].fillna("").str.lower().str.contains("|".join([g.lower() for g in preferred_genres]), case=False, na=False)]
    if preferred_language:
        filtered = filtered[filtered["original_language"].fillna("").str.lower() == str(preferred_language).lower()]
    filtered = filtered[(filtered["vote_average"] >= min_rating) & (filtered["release_year"].between(year_from, year_to))]
    filtered = filtered[(filtered["runtime"] >= min_runtime) & (filtered["runtime"] <= max_runtime)]

    if filtered.empty:
        filtered = df.copy()

    favorites = []
    for title in favorite_movies:
        match = df[df["title"].str.lower() == str(title).lower()]
        if not match.empty:
            favorites.append(match.iloc[0])

    if not favorites:
        favorites = df[df["title"].isin(filtered["title"].head(3).tolist())].head(3).to_dict("records") if not filtered.empty else []

    if not favorites:
        favorites = df.head(1).to_dict("records")

    favorite_indices = []
    for favorite in favorites:
        idx = working[working["title"].str.lower() == str(favorite["title"]).lower()].index
        if len(idx):
            favorite_indices.append(int(idx[0]))

    if not favorite_indices:
        favorite_indices = [0]

    ranked = []
    for _, movie in filtered.iterrows():
        title = movie["title"]
        if title.lower() in {favorite["title"].lower() for favorite in favorites}:
            continue

        candidate_idx = working[working["title"].str.lower() == str(title).lower()].index
        if len(candidate_idx) == 0:
            continue
        candidate_idx = int(candidate_idx[0])

        content_scores = []
        for fav_idx in favorite_indices:
            content_scores.append(float(similarity[fav_idx, candidate_idx]))
        content_score = float(np.mean(content_scores)) if content_scores else 0.0

        genre_score = _genre_match_score(movie.get("genres", ""), preferred_genres)
        rating_score = min(1.0, float(movie["vote_average"]) / 10.0)
        language_score = 1.0 if (preferred_language and str(movie["original_language"]).lower() == str(preferred_language).lower()) else 0.5
        year_score = 1.0 if (year_from <= int(movie["release_year"]) <= year_to) else 0.0
        popularity_score = min(1.0, float(movie["popularity"]) / max(1.0, df["popularity"].max()))
        mood_score = 1.0 if (not mood or mood.lower() in str(movie.get("genres", "")).lower()) else 0.5

        final_score = (
            0.50 * content_score
            + 0.20 * genre_score
            + 0.10 * rating_score
            + 0.10 * language_score
            + 0.05 * year_score
            + 0.05 * popularity_score
            + 0.10 * mood_score
        )

        reasons = _build_reason_list(movie, [fav["title"] for fav in favorites], preferences)
        ranked.append({
            "title": movie["title"],
            "id": int(movie["id"]),
            "release_year": int(movie["release_year"]),
            "vote_average": float(movie["vote_average"]),
            "runtime": float(movie["runtime"]),
            "genres": movie["genres"],
            "original_language": movie["original_language"],
            "overview": movie["overview"],
            "poster_path": movie.get("poster_path", ""),
            "popularity": float(movie["popularity"]),
            "final_score": float(final_score),
            "reasons": reasons,
            "content_similarity": float(content_score),
        })

    ranked = sorted(ranked, key=lambda x: x["final_score"], reverse=True)
    return ranked[:top_n]


if __name__ == "__main__":
    prefs = {
        "genres": ["Science Fiction", "Adventure"],
        "min_rating": 7.5,
        "year_from": 2015,
        "year_to": 2026,
        "language": "English",
        "mood": "Sci-Fi",
        "min_runtime": 90,
        "max_runtime": 180,
    }
    movies = recommend_movies(prefs, favorite_movies=["Interstellar"], top_n=5)
    print(movies[0]["title"] if movies else "No results")
