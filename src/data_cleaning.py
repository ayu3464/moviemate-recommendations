import pandas as pd

from src.utils import RAW_DATA_PATH, PROCESSED_DATA_PATH, ensure_dirs


def clean_movie_data(raw_path=RAW_DATA_PATH, output_path=PROCESSED_DATA_PATH):
    ensure_dirs()
    df = pd.read_csv(raw_path)

    if df.empty:
        raise ValueError("The movie dataset is empty.")

    df = df.copy()
    df.columns = [col.strip().lower() for col in df.columns]

    required_cols = {
        "id": "id",
        "title": "title",
        "release_date": "release_date",
        "genres": "genres",
        "overview": "overview",
        "original_language": "original_language",
        "runtime": "runtime",
        "vote_average": "vote_average",
        "vote_count": "vote_count",
        "popularity": "popularity",
        "keywords": "keywords",
    }

    for original, normalized in required_cols.items():
        if original not in df.columns:
            if normalized in df.columns:
                df.rename(columns={normalized: original}, inplace=True)
            else:
                continue

    df["title"] = df["title"].fillna("Unknown Title").astype(str)
    df["overview"] = df["overview"].fillna("").astype(str)
    df["genres"] = df["genres"].fillna("Unknown").astype(str)
    df["original_language"] = df["original_language"].fillna("Unknown").astype(str)

    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_year"] = df["release_date"].dt.year
    df = df.dropna(subset=["release_year"]).copy()

    df["runtime"] = pd.to_numeric(df["runtime"], errors="coerce")
    runtime_median = df["runtime"].median()
    if pd.notna(runtime_median):
        df["runtime"] = df["runtime"].fillna(runtime_median)

    df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce")
    rating_median = df["vote_average"].median()
    if pd.notna(rating_median):
        df["vote_average"] = df["vote_average"].fillna(rating_median)

    df["vote_count"] = pd.to_numeric(df["vote_count"], errors="coerce")
    vote_count_median = df["vote_count"].median()
    if pd.notna(vote_count_median):
        df["vote_count"] = df["vote_count"].fillna(vote_count_median)

    df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce")
    popularity_median = df["popularity"].median()
    if pd.notna(popularity_median):
        df["popularity"] = df["popularity"].fillna(popularity_median)

    df["keywords"] = df["keywords"].fillna("").astype(str)
    df["genres"] = df["genres"].apply(lambda x: x if x not in ["nan", "None", "Unknown"] else "Unknown")

    df = df.drop_duplicates(subset=["id", "title"], keep="first").reset_index(drop=True)
    df = df[(df["title"].str.len() > 0) & (df["release_year"].between(1900, 2035))].copy()
    df = df[df["vote_count"] >= 1000].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    cleaned = clean_movie_data()
    print(f"Saved {len(cleaned)} cleaned movies to {PROCESSED_DATA_PATH}")
