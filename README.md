# MovieMate

A personalized movie recommendation system built with Python, pandas, scikit-learn, and Streamlit.

## Features

- Personalized movie recommendations based on genre, rating, language, release year, runtime, and favorite movie
- TF-IDF + cosine similarity content-based recommendation engine
- Search for a movie and find similar titles
- Explainable results with reasoning for each recommendation
- EDA dashboard for dataset analysis

## Project structure

```text
Movie- prediction/
├── app.py
├── data/
│   ├── raw/
│   │   └── TMDB_movie_dataset_v11.csv
│   └── processed/
│       └── cleaned_movies.csv
├── models/
├── notebooks/
├── src/
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── recommender.py
│   └── utils.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Run the app

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Data source

This project uses the TMDB movie dataset from the downloaded archive, with a cleaned and processed subset built for recommendation.
