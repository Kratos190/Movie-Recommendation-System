import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process
from imdb import IMDb

# ========= STREAMLIT CONFIG ========= #
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")

# ========= IMDbPY INIT ========= #
ia = IMDb()

# ========= DATA LOADING & PREPROCESSING (CACHED) ========= #
@st.cache_data
def load_data():
    movies = pd.read_csv(r"C:/Users/Vivek/OneDrive/Desktop/Movies Recommendation System/ml-32m/movies.csv")
    movies = movies.head(10000)
    movies.drop_duplicates(inplace=True)
    movies.dropna(subset=['title', 'genres'], inplace=True)
    movies['genres'] = movies['genres'].str.replace('|', ' ', regex=False)
    movies['title'] = movies['title'].str.strip()
    movies['title_lower'] = movies['title'].str.lower()
    movies['tags'] = movies['title_lower'] + ' ' + movies['genres'].str.lower()
    movies.reset_index(drop=True, inplace=True)
    return movies

# ========= SIMILARITY MATRIX (CACHED) ========= #
@st.cache_data
def compute_similarity(movies):
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(movies['tags']).toarray()
    return cosine_similarity(vectors)

# ========= POSTER FETCHING (CACHED) ========= #
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

# ========= TITLE-BASED RECOMMENDATION ========= #
def recommend(movie_name, movies, similarity):
    movie_name = movie_name.lower().strip()
    matches = process.extractOne(movie_name, movies['title_lower'])

    if not matches or matches[1] < 60:
        st.error("❌ Movie not found in dataset.")
        return

    best_match = matches[0]
    index = movies[movies['title_lower'] == best_match].index[0]
    distances = list(enumerate(similarity[index]))
    similar_movies = sorted(distances, key=lambda x: x[1], reverse=True)[1:10]  # Top 9

    st.subheader(f"🎯 Recommendations for '{movies.loc[index, 'title']}':")

    cols = st.columns(3)
    for idx, (i, _) in enumerate(similar_movies):
        movie_title = movies.loc[i, 'title']
        poster_url = get_poster_url(movie_title)
        col = cols[idx % 3]
        with col:
            st.image(poster_url, width=150)
            st.caption(movie_title)

# ========= GENRE-BASED SEARCH ========= #
def recommend_by_genre(genre, movies):
    genre = genre.lower().strip()
    filtered = movies[movies['genres'].str.lower().str.contains(genre)]

    if filtered.empty:
        st.error("❌ No movies found for this genre.")
        return

    st.subheader(f"🎯 Movies in '{genre.title()}' genre:")
    cols = st.columns(3)
    for idx, (_, row) in enumerate(filtered.head(15).iterrows()):  # Limit to first 15
        movie_title = row['title']
        poster_url = get_poster_url(movie_title)
        col = cols[idx % 3]
        with col:
            st.image(poster_url, width=150)
            st.caption(movie_title)

# ========= MAIN APP ========= #
movies = load_data()
similarity = compute_similarity(movies)

st.title("🎬 Movie Recommender System")

option = st.radio("Choose search type:", ["Search by Title", "Search by Genre"])

if option == "Search by Title":
    movie_input = st.text_input("Enter a movie name:")
    if st.button("Recommend"):
        if movie_input.strip():
            recommend(movie_input, movies, similarity)
        else:
            st.warning("Please enter a movie name.")

elif option == "Search by Genre":
    genre_input = st.text_input("Enter a genre (e.g., Action, Comedy, Drama):")
    if st.button("Find Movies"):
        if genre_input.strip():
            recommend_by_genre(genre_input, movies)
        else:
            st.warning("Please enter a genre.")
