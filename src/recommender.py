import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# 1. Load modeling dataset
df = pd.read_csv("data/movies_model.csv")

df["overview"] = df["overview"].fillna("")

# 2. Create TF-IDF vectors
tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=10000
)

tfidf_matrix = tfidf.fit_transform(df["overview"])


# 3. Recommendation function
def recommend_movies(movie_title, top_n=10):

    # Find movies matching the title
    matches = df[
        df["title"].str.lower() == movie_title.lower()
    ]

    if matches.empty:
        print(f"Movie '{movie_title}' not found.")
        return

    # Get the row position of the selected movie
    movie_index = df.index.get_loc(matches.index[0])

    # Calculate similarity to every other movie
    similarity_scores = cosine_similarity(
        tfidf_matrix[movie_index],
        tfidf_matrix
    ).flatten()

    # Sort movie positions by similarity
    similar_indices = similarity_scores.argsort()[::-1]

    # Exclude the selected movie itself
    similar_indices = [
        i for i in similar_indices
        if i != movie_index
    ]

    # Select top recommendations
    top_indices = similar_indices[:top_n]

    recommendations = df.iloc[top_indices][
        ["title", "release_date", "genres", "vote_average"]
    ].copy()

    recommendations["similarity"] = (
        similarity_scores[top_indices]
    )

    return recommendations


# 4. Test the recommender
results = recommend_movies("Inception", top_n=10)

if results is not None:
    print("\nRecommendations for Inception:\n")
    print(results.to_string(index=False))