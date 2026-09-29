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
MovieMate/
├── app.py
├── assets/
│   └── ayush_tiwari.jpg
├── data/
│   └── processed/
│       └── cleaned_movies.csv
├── notebooks/
├── src/
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── recommender.py
│   └── utils.py
├── tests/
│   └── test_recommender.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Run the app

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

The cleaned dataset is included in the repository. The original raw dataset is not included because of its size. Recommendation model files are generated locally from the cleaned dataset when recommendations are requested.

## Deploy

This app can be deployed from this repository with Streamlit Community Cloud by selecting `app.py` as the app entry point. The dependencies are listed in `requirements.txt`.

## Data source

This project uses the TMDB movie dataset from the downloaded archive, with a cleaned and processed subset built for recommendation.
