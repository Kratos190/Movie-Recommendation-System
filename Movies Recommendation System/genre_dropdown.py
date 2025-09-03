import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process
from imdb import IMDb

# ========= STREAMLIT CONFIG ========= #
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")

# ========= CUSTOM DARK THEME STYLING ========= #
def add_custom_css():
    st.markdown("""
        <style>
        /* Background & Text */
        .stApp {
            background-color: #0f1117;
            color: #f5f6fa;
            font-family: 'Segoe UI', sans-serif;
        }
        /* Title Styling */
        h1 {
            text-align: center;
            color: #f5f6fa;
            font-weight: 700;
            letter-spacing: 1px;
            margin-bottom: 20px;
        }
        /* Radio Buttons */
        .stRadio > div {
            flex-direction: row !important;
            gap: 15px;
        }
        /* Buttons */
        div.stButton > button {
            background: linear-gradient(90deg, #ff4b2b, #ff416c);
            color: white;
            border-radius: 25px;
            padding: 0.6em 1.2em;
            border: none;
            font-weight: bold;
            transition: 0.3s ease;
        }
        div.stButton > button:hover {
            transform: scale(1.05);
            background: linear-gradient(90deg, #ff416c, #ff4b2b);
        }
        /* Movie Cards */
        .movie-card {
            background-color: #1c1f26;
            border-radius: 10px;
            padding: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.5);
            transition: transform 0.2s ease;
            text-align: center;
             margin-bottom: 10px;
        }
        .movie-card:hover {
            transform: translateY(-5px);
        }
        .movie-card img {
            border-radius: 10px;
            margin-bottom: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

add_custom_css()

# ========= IMDbPY INIT ========= #
ia = IMDb()

# ========= DATA LOADING ========= #
@st.cache_data
def load_data():
    movies = pd.read_csv(r"C:\Users\Vivek\OneDrive\Desktop\Movies Recommendation System\ml-32m\movies.csv")
    movies = movies.head(10000)
    movies.drop_duplicates(inplace=True)
    movies.dropna(subset=['title', 'genres'], inplace=True)
    movies['genres'] = movies['genres'].str.replace('|', ' ', regex=False)
    movies['title'] = movies['title'].str.strip()
    movies['title_lower'] = movies['title'].str.lower()
    movies['tags'] = movies['title_lower'] + ' ' + movies['genres'].str.lower()
    movies.reset_index(drop=True, inplace=True)
    return movies

# ========= SIMILARITY MATRIX ========= #
@st.cache_data
def compute_similarity(movies):
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(movies['tags']).toarray()
    return cosine_similarity(vectors)

# ========= POSTER FETCHING ========= #
@st.cache_data
def get_poster_url(movie_title):
    try:
        search_results = ia.search_movie(movie_title)
        if search_results:
            movie = search_results[0]
            ia.update(movie)
            url = movie.get('full-size cover url', None)
            if isinstance(url, str) and url.startswith("http"):
                return url
    except:
        pass
    return "https://via.placeholder.com/150?text=No+Image"

# ========= RECOMMEND FUNCTION ========= #
def recommend(movie_name, movies, similarity):
    movie_name = movie_name.lower().strip()
    matches = process.extractOne(movie_name, movies['title_lower'])
    if not matches or matches[1] < 60:
        st.error("❌ Movie not found in dataset.")
        return
    best_match = matches[0]
    index = movies[movies['title_lower'] == best_match].index[0]
    distances = list(enumerate(similarity[index]))
    similar_movies = sorted(distances, key=lambda x: x[1], reverse=True)[1:10]
    st.subheader(f"🎯 Recommendations for '{movies.loc[index, 'title']}':")
    cols = st.columns(3)
    for idx, (i, _) in enumerate(similar_movies):
        movie_title = movies.loc[i, 'title']
        poster_url = get_poster_url(movie_title)
        with cols[idx % 3]:
            st.markdown(f"""
                <div class="movie-card">
                    <img src="{poster_url}" width="150">
                    <p>{movie_title}</p>
                </div>
            """, unsafe_allow_html=True)

# ========= GENRE SEARCH FUNCTION ========= #
def search_by_genre(selected_genre, movies):
    filtered_movies = movies[movies['genres'].str.contains(selected_genre, case=False, na=False)]
    if filtered_movies.empty:
        st.error("❌ No movies found in this genre.")
        return
    st.subheader(f"🎯 Movies in '{selected_genre}' genre:")
    cols = st.columns(3)
    for idx, row in enumerate(filtered_movies.head(9).itertuples()):
        poster_url = get_poster_url(row.title)
        with cols[idx % 3]:
            st.markdown(f"""
                <div class="movie-card">
                    <img src="{poster_url}" width="150">
                    <p>{row.title}</p>
                </div>
            """, unsafe_allow_html=True)

# ========= MAIN APP ========= #
movies = load_data()
similarity = compute_similarity(movies)

st.title("🎬 Movie Recommender System")

search_type = st.radio("Choose search type:", ("Search by Title", "Search by Genre"))

if search_type == "Search by Title":
    movie_input = st.text_input("Enter a movie name:")
    if st.button("Recommend"):
        if movie_input.strip():
            recommend(movie_input, movies, similarity)
        else:
            st.warning("Please enter a movie name.")

elif search_type == "Search by Genre":
    genre_input = st.selectbox("Select a genre:", sorted(set(g for genre in movies['genres'] for g in genre.split())))
    if st.button("Find Movies"):
        search_by_genre(genre_input, movies)
