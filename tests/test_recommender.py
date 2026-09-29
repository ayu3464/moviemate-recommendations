from src.recommender import recommend_movies


def test_recommend_movies_returns_valid_results():
    preferences = {
        "genres": ["Science Fiction", "Adventure"],
        "language": "English",
        "min_rating": 7.0,
        "year_from": 2015,
        "year_to": 2026,
        "min_runtime": 90,
        "max_runtime": 180,
        "mood": "Sci-Fi",
    }

    results = recommend_movies(preferences, favorite_movies=["Interstellar"], top_n=5)

    assert len(results) > 0
    assert all("title" in item for item in results)
    assert all("reasons" in item for item in results)
    assert all("content_similarity" in item for item in results)
    assert all(item["title"] != "Interstellar" for item in results)
