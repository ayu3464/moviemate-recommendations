from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_cleaning import clean_movie_data
from src.recommender import get_dataset, recommend_movies

st.set_page_config(page_title="MovieMate", page_icon="🎬", layout="wide")

COMMON_GENRES = [
    "Action",
    "Adventure",
    "Comedy",
    "Drama",
    "Horror",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "Animation",
    "Fantasy",
    "Mystery",
    "Documentary",
]

GENRE_ALIASES = {
    "Sci-Fi": "Science Fiction",
    "Action": "Action",
    "Adventure": "Adventure",
    "Comedy": "Comedy",
    "Drama": "Drama",
    "Horror": "Horror",
    "Romance": "Romance",
    "Thriller": "Thriller",
    "Animation": "Animation",
    "Fantasy": "Fantasy",
    "Mystery": "Mystery",
    "Documentary": "Documentary",
}


@st.cache_data
def load_cleaned_dataset():
    path = Path("data/processed/cleaned_movies.csv")
    if not path.exists():
        clean_movie_data()
    df = pd.read_csv(path)
    return df


@st.cache_data
def get_movie_options():
    df = load_cleaned_dataset()
    return sorted(df["title"].dropna().unique().tolist())


@st.cache_data
def get_genre_options():
    df = load_cleaned_dataset()
    genres = []
    for value in df["genres"].fillna(""):
        for item in str(value).replace("[", "").replace("]", "").split(","):
            genre = item.strip().title()
            if genre:
                genres.append(genre)
    return sorted(set(genres))


@st.cache_data
def get_language_options():
    df = load_cleaned_dataset()
    languages = [str(value).strip() for value in df["original_language"].dropna().unique().tolist() if str(value).strip()]
    return sorted(set(languages))


def render_home():
    st.markdown(
        """
        <style>
        .topbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.8rem 0 1.2rem 0;
            border-bottom: 1px solid rgba(255,255,255,0.08);
            margin-bottom: 1.2rem;
        }
        .brand {
            font-size: 1.7rem;
            font-weight: 900;
            color: #e50914;
            letter-spacing: 0.08rem;
            text-shadow: 0 0 12px rgba(229,9,20,0.5), 0 0 28px rgba(229,9,20,0.22);
        }
        .nav-links {
            display: flex;
            gap: 1.1rem;
            flex-wrap: wrap;
            color: #eaeaea;
            font-size: 0.9rem;
        }
        .nav-links span {
            opacity: 0.8;
        }
        .nav-links .active {
            color: white;
            font-weight: 700;
        }
        .movie-hero {
            background: linear-gradient(90deg, rgba(10,10,10,0.9), rgba(17,17,17,0.6)),
                        url('https://images.unsplash.com/photo-1517604931442-7e0c8ed2963c?auto=format&fit=crop&w=1600&q=80') center/cover no-repeat;
            border-radius: 26px;
            padding: 2.8rem 2rem;
            min-height: 360px;
            display: flex;
            align-items: end;
            box-shadow: 0 30px 80px rgba(0,0,0,0.5);
            border: 1px solid rgba(255,255,255,0.08);
            margin-bottom: 1.3rem;
        }
        .movie-hero .content {
            max-width: 700px;
        }
        .movie-kicker {
            letter-spacing: 0.18rem;
            color: #ff6b6b;
            font-size: 0.82rem;
            text-transform: uppercase;
            font-weight: 800;
        }
        .movie-title {
            font-size: 3.1rem;
            font-weight: 900;
            line-height: 1.0;
            margin: 0.5rem 0;
            color: white;
        }
        .movie-sub {
            font-size: 1.12rem;
            color: #e5e5e5;
            margin-bottom: 1.1rem;
            max-width: 560px;
        }
        .movie-button-row {
            display: flex;
            gap: 0.8rem;
            flex-wrap: wrap;
        }
        .movie-button {
            border-radius: 999px;
            padding: 0.8rem 1.5rem;
            font-weight: 700;
            border: none;
            cursor: pointer;
            background: linear-gradient(135deg, #e50914, #b00610);
            color: white;
            box-shadow: 0 15px 30px rgba(229,9,20,0.35);
        }
        .movie-button.secondary {
            background: rgba(255,255,255,0.1);
            color: white;
            border: 1px solid rgba(255,255,255,0.22);
            box-shadow: none;
        }
        .stButton > button {
            border-radius: 999px;
            padding: 0.8rem 1.5rem;
            font-weight: 700;
            border: none;
            background: linear-gradient(135deg, #e50914, #b00610);
            color: white;
            box-shadow: 0 15px 30px rgba(229,9,20,0.35);
        }
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 18px 35px rgba(229,9,20,0.45);
        }
        .feature-card {
            background: linear-gradient(180deg, rgba(30,30,30,0.96), rgba(18,18,18,0.96));
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            padding: 1rem;
            box-shadow: 0 18px 35px rgba(0,0,0,0.25);
            min-height: 150px;
        }
        .mini-tag {
            display: inline-block;
            background: rgba(229,9,20,0.12);
            color: #ffd6d8;
            padding: 0.32rem 0.7rem;
            border-radius: 999px;
            font-size: 0.75rem;
            margin: 0.3rem 0.4rem 0.3rem 0;
        }
        .genre-pill {
            display: inline-block;
            padding: 0.35rem 0.75rem;
            margin: 0.2rem 0.35rem 0.2rem 0;
            border-radius: 999px;
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.1);
            color: #f2f2f2;
            font-size: 0.76rem;
        }
        .poster-row {
            display: flex;
            gap: 0.8rem;
            overflow-x: auto;
            padding-bottom: 0.7rem;
            scrollbar-width: thin;
        }
        .poster-card {
            min-width: 140px;
            background: linear-gradient(180deg, rgba(44,44,44,0.85), rgba(18,18,18,0.9));
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.06);
            box-shadow: 0 12px 28px rgba(0,0,0,0.32);
            transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease;
        }
        .poster-card:hover {
            transform: translateY(-6px) scale(1.025);
            border-color: rgba(229,9,20,0.55);
            box-shadow: 0 18px 36px rgba(0,0,0,0.48), 0 0 18px rgba(229,9,20,0.18);
        }
        .poster-card img {
            width: 100%;
            height: 210px;
            object-fit: cover;
            display: block;
        }
        .poster-label {
            padding: 0.55rem 0.6rem 0.7rem;
            font-size: 0.78rem;
            color: white;
        }
        .row-title {
            font-weight: 800;
            font-size: 1.2rem;
            color: white;
            margin: 0.8rem 0 0.45rem 0;
        }
        .glow-box {
            background: linear-gradient(135deg, rgba(229,9,20,0.12), rgba(255,255,255,0.03));
            border: 1px solid rgba(229,9,20,0.18);
            border-radius: 18px;
            padding: 1rem;
        }
        @media (max-width: 640px) {
            .topbar {
                align-items: flex-start;
                flex-direction: column;
                gap: 0.7rem;
            }
            .nav-links {
                gap: 0.8rem;
            }
            .movie-title {
                font-size: 2.2rem;
            }
        }
        </style>
        <div class="topbar">
            <div class="brand">MovieMate</div>
            <div class="nav-links">
                <span class="active">Home</span>
                <span>Movies</span>
                <span>My List</span>
                <span>Trending</span>
            </div>
        </div>
        <div class="movie-hero">
            <div class="content">
                <div class="movie-kicker">MovieMate</div>
                <h1 class="movie-title">Find your next movie night</h1>
                <div class="movie-sub">Smart recommendations based on your taste, your mood, and the kind of stories you love to watch.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    action_col_1, action_col_2 = st.columns(2)
    with action_col_1:
        if st.button("Get My Movie", key="home_get_movie", use_container_width=True):
            st.session_state.page = "Movie Recommendations"
            st.rerun()
    with action_col_2:
        if st.button("Explore Picks", key="home_explore", use_container_width=True):
            st.session_state.page = "Movie Recommendations"
            st.rerun()

    st.write("")
    st.subheader("Popular mood picks")
    st.markdown('<div class="row-title">Trending now</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="poster-row">
            <div class="poster-card">
                <img src="https://image.tmdb.org/t/p/w500/gEU2QniE6E77NI6lCU6MxlNBvIx.jpg" alt="Interstellar" />
                <div class="poster-label">Interstellar</div>
            </div>
            <div class="poster-card">
                <img src="https://image.tmdb.org/t/p/w500/oYuLEt3zVCKq57qu2F8dT7NIa6f.jpg" alt="Inception" />
                <div class="poster-label">Inception</div>
            </div>
            <div class="poster-card">
                <img src="https://image.tmdb.org/t/p/w500/5BHuvQ6p9kfc091Z8RiFNhCwL4b.jpg" alt="The Martian" />
                <div class="poster-label">The Martian</div>
            </div>
            <div class="poster-card">
                <img src="https://image.tmdb.org/t/p/w500/x2FJsf1ElAgr63Y3PNPtJrcmpoe.jpg" alt="Arrival" />
                <div class="poster-label">Arrival</div>
            </div>
            <div class="poster-card">
                <img src="https://image.tmdb.org/t/p/w500/d5NXSklXo0qyIYkgV94XAgMIckC.jpg" alt="Dune" />
                <div class="poster-label">Dune</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="row-title">Mood-based picks</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="glow-box">
            <span class="genre-pill">Sci‑Fi</span>
            <span class="genre-pill">Mind-bending</span>
            <span class="genre-pill">Adventure</span>
            <span class="genre-pill">Drama</span>
            <span class="genre-pill">Action</span>
            <span class="genre-pill">Feel good</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Why people love MovieMate")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown(
            """
            <div class="feature-card">
                <div class="mini-tag">Smart</div>
                <h4>Tailored picks</h4>
                <p>Recommendations adapt to your favorite movies, preferred genres, and mood.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            """
            <div class="feature-card">
                <div class="mini-tag">Fast</div>
                <h4>Quick filters</h4>
                <p>Choose a few simple preferences instead of complicated technical filters.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_c:
        st.markdown(
            """
            <div class="feature-card">
                <div class="mini-tag">Clear</div>
                <h4>Explainable results</h4>
                <p>Every recommendation tells you why it matches your taste and viewing style.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

def render_recommendations():
    df = load_cleaned_dataset()
    languages = get_language_options()
    movie_titles = get_movie_options()

    st.title("Movie Recommendations")
    st.caption("Simple picks, fewer words, better choices.")

    with st.form("recommendation_form"):
        selected_genres = st.multiselect(
            "Pick the genres you like",
            options=COMMON_GENRES,
            default=["Sci-Fi", "Adventure"],
            help="Choose up to 3 genres to keep it simple.",
        )
        selected_language = st.selectbox("Language", options=["Any language"] + languages)
        min_rating = st.slider("Minimum rating", 0.0, 10.0, 7.5, step=0.1)
        year_from, year_to = st.slider("Release year", 1900, 2026, (2015, 2026))
        min_runtime, max_runtime = st.slider("Runtime", 30, 300, (90, 180))
        favorite_movies = st.multiselect("Your favorite movie(s)", options=movie_titles, default=["Interstellar"])
        mood = st.selectbox(
            "What is your mood?",
            ["Any", "Feel Good", "Comedy", "Romance", "Action", "Horror", "Mind-Bending", "Sci-Fi", "Mystery", "Drama"],
        )

        submitted = st.form_submit_button("Get My Movie")

    if submitted:
        preferences = {
            "genres": [GENRE_ALIASES.get(g, g) for g in selected_genres],
            "language": None if selected_language == "Any language" else selected_language,
            "min_rating": min_rating,
            "year_from": year_from,
            "year_to": year_to,
            "min_runtime": min_runtime,
            "max_runtime": max_runtime,
            "mood": None if mood == "Any" else mood,
        }
        recommendations = recommend_movies(preferences, favorite_movies=favorite_movies, top_n=8)

        if not recommendations:
            st.warning("No movies match this selection. Try choosing a wider range or different genre vibe.")
            return

        st.subheader(f"Top {len(recommendations)} movies for you")
        for idx, movie in enumerate(recommendations, start=1):
            with st.container():
                col_image, col_text = st.columns([1, 4])
                with col_image:
                    poster = movie.get("poster_path")
                    if poster:
                        st.image(f"https://image.tmdb.org/t/p/w500/{poster}", use_container_width=True)
                    else:
                        st.image("https://via.placeholder.com/300x450?text=No+Poster", use_container_width=True)
                with col_text:
                    st.markdown(f"### {idx}. {movie['title']} ({movie['release_year']})")
                    st.write(f"⭐ {movie['vote_average']:.1f}  •  {movie['runtime']:.0f} min  •  {movie['original_language']}  •  Match: {movie['content_similarity']:.2f}")
                    st.write(f"Genres: {movie['genres']}")
                    st.write(movie.get('overview', 'No overview available.')[:250] + ('...' if len(str(movie.get('overview', ''))) > 250 else ''))
                    st.write("Why it fits you:")
                    for reason in movie["reasons"]:
                        st.write(f"✓ {reason}")
                    st.markdown("---")


def render_search():
    st.title("Search Movies")
    df = load_cleaned_dataset()
    title = st.selectbox("Search movie", options=get_movie_options())
    movie = df[df["title"] == title].iloc[0]

    st.subheader(movie["title"])
    col1, col2 = st.columns([2, 4])
    with col1:
        poster = movie.get("poster_path")
        if pd.notna(poster):
            st.image(f"https://image.tmdb.org/t/p/w500/{poster}", use_container_width=True)
        else:
            st.image("https://via.placeholder.com/300x450?text=Poster", use_container_width=True)
    with col2:
        st.write(f"Release Year: {int(movie['release_year'])}")
        st.write(f"Rating: {movie['vote_average']:.1f}")
        st.write(f"Genres: {movie['genres']}")
        st.write(f"Runtime: {int(movie['runtime'])} minutes")
        st.write(f"Language: {movie['original_language']}")
        st.write(f"Popularity: {movie['popularity']:.1f}")
        st.write(f"Vote Count: {int(movie['vote_count'])}")
        st.write(f"Overview: {movie['overview']}")
        homepage = movie.get("homepage")
        if pd.notna(homepage) and str(homepage).strip():
            st.markdown(f"[Official page]({homepage})")
        imdb_id = movie.get("imdb_id")
        if pd.notna(imdb_id) and str(imdb_id).strip():
            st.markdown(f"[IMDb page](https://www.imdb.com/title/{imdb_id}/)")

    with st.expander("Movie details"):
        st.write(f"Status: {movie.get('status', 'Unknown')}")
        st.write(f"Budget: ${movie.get('budget', 0):,.0f}")
        st.write(f"Revenue: ${movie.get('revenue', 0):,.0f}")
        st.write(f"Tagline: {movie.get('tagline', 'N/A')}")

    if st.button("Find Similar Movies"):
        prefs = {
            "genres": [part.strip() for part in str(movie["genres"]).replace("[", "").replace("]", "").split(",") if part.strip()],
            "language": movie["original_language"],
            "min_rating": max(0.0, float(movie["vote_average"]) - 0.5),
            "year_from": max(1900, int(movie["release_year"]) - 10),
            "year_to": 2026,
            "min_runtime": max(30, int(movie["runtime"]) - 30),
            "max_runtime": 300,
            "mood": None,
        }
        result = recommend_movies(prefs, favorite_movies=[movie["title"]], top_n=6)
        st.subheader("Similar movies")
        for rec in result:
            st.write(f"- {rec['title']} ({rec['release_year']}) — {rec['vote_average']:.1f}★")


def render_eda():
    st.title("Dataset Analysis")
    df = load_cleaned_dataset()

    if df.empty:
        st.info("Dataset is empty.")
        return

    st.subheader("Movies by Year")
    yearly = df["release_year"].value_counts().sort_index()
    st.bar_chart(yearly)

    st.subheader("Genre Distribution")
    genre_rows = []
    for row in df["genres"].fillna("").tolist():
        for item in str(row).replace("[", "").replace("]", "").split(","):
            item = item.strip().title()
            if item:
                genre_rows.append(item)
    genre_df = pd.DataFrame({"genre": genre_rows}).value_counts().reset_index(name="count")
    genre_df.columns = ["genre", "count"]
    st.bar_chart(genre_df.set_index("genre")["count"])

    st.subheader("Rating Distribution")
    st.histogram(df["vote_average"].dropna())

    st.subheader("Rating vs Popularity")
    fig = px.scatter(df, x="vote_average", y="popularity", title="Rating vs Popularity", hover_name="title")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Language Distribution")
    lang_df = df["original_language"].value_counts().head(10).reset_index()
    lang_df.columns = ["language", "count"]
    st.bar_chart(lang_df.set_index("language")["count"])


def render_about():
    st.title("About Project")

    col_left, col_right = st.columns([1.2, 3])
    with col_left:
        st.image("assets/ayush_tiwari.jpg", width=240, clamp=True)
    with col_right:
        st.markdown(
            """
            ## Ayush Tiwari

            Data Science and Machine Learning Enthusiast

            I am a student passionate about building intelligent systems, especially recommendation engines, data analysis dashboards, and AI-driven products that solve real-world problems.

            This project reflects my interest in combining data science, machine learning, and user-centric product design to create a practical recommendation system that helps people discover movies they are likely to enjoy.
            """
        )

    st.markdown("---")
    st.markdown(
        """
        ### Machine learning workflow

        User Preferences → Data Filtering → Feature Extraction → TF-IDF → Cosine Similarity → Preference Scoring → Final Ranking

        ### Recommendation approach

        This project uses a content-based recommendation system. It compares movie metadata such as genre, overview, keywords, and language to identify the most relevant titles and then ranks them according to user preferences.

        ### Evaluation

        A practical recommendation system is evaluated with relevance-focused metrics like Precision@K and Recall@K, and this project also uses a manual example: if a user likes Interstellar, similar titles like The Martian, Arrival, Gravity, and Dune are expected to surface because they share themes of space, survival, and scientific exploration.
        """
    )


page_names = ["Home", "Movie Recommendations", "Search Movies", "Dataset Analysis", "About Project"]
if "page" not in st.session_state:
    st.session_state.page = "Home"

page = st.sidebar.radio("Navigation", page_names, index=page_names.index(st.session_state.page) if st.session_state.page in page_names else 0)
st.session_state.page = page

if page == "Home":
    render_home()
elif page == "Movie Recommendations":
    render_recommendations()
elif page == "Search Movies":
    render_search()
elif page == "Dataset Analysis":
    render_eda()
else:
    render_about()
