from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "TMDB_movie_dataset_v11.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "cleaned_movies.csv"
MODEL_DIR = ROOT_DIR / "models"


def normalize_text(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    return " ".join(text.split())


def get_genre_list(value):
    if value is None or value == "":
        return []
    text = str(value)
    if text.startswith("["):
        try:
            import ast
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except Exception:
            pass
    return [part.strip() for part in text.split(",") if part.strip()]


def ensure_dirs():
    DATA_DIR.mkdir(exist_ok=True)
    (DATA_DIR / "raw").mkdir(exist_ok=True)
    (DATA_DIR / "processed").mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)


# Keep a lightweight import for pd without creating circular references.
import pandas as pd
