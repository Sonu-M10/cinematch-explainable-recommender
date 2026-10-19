import time
import tracemalloc

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DATASET_PATH = "data/movies_cleaned.csv"

FEATURES = [
    "overview",
    "genres",
    "keywords",
    "directors",
    "cast"
]

WEIGHTS = {
    "overview": 0.40,
    "genres": 0.25,
    "keywords": 0.20,
    "directors": 0.10,
    "cast": 0.05
}

TEST_MOVIES = {
    "Inception": 27205,
    "Interstellar": 157336,
    "The Conjuring": 138843,
    "La La Land": 313369,
    "Get Out": 419430,
    "Knives Out": 546554
}


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def prepare_metadata(text):

    if not text:
        return ""

    items = text.split(",")

    tokens = [
        item.strip().lower().replace(" ", "_")
        for item in items
        if item.strip()
    ]

    return " ".join(tokens)


def format_memory(num_bytes):

    return num_bytes / (1024 ** 2)


# --------------------------------------------------
# LOAD AND PREPARE DATA
# --------------------------------------------------

print("\nLoading expanded movie dataset...")

start_time = time.perf_counter()

tracemalloc.start()

df = pd.read_csv(DATASET_PATH)

df = df.dropna(
    subset=["overview", "genres"]
).copy()

df = df.reset_index(drop=True)

for feature in FEATURES:
    df[feature] = df[feature].fillna("")

for feature in [
    "genres",
    "keywords",
    "directors",
    "cast"
]:
    df[f"{feature}_processed"] = df[feature].apply(
        prepare_metadata
    )

load_time = time.perf_counter() - start_time

print(f"Movies loaded: {len(df):,}")
print(f"Data preparation time: {load_time:.2f} seconds")


# --------------------------------------------------
# BUILD TF-IDF MATRICES
# --------------------------------------------------

print("\nBuilding TF-IDF matrices...")

feature_matrices = {}

build_start = time.perf_counter()

for feature in FEATURES:

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=10000,
        token_pattern=r"(?u)\b[\w-]+\b"
    )

    if feature == "overview":
        text_data = df[feature]
    else:
        text_data = df[f"{feature}_processed"]

    matrix = vectorizer.fit_transform(text_data)

    feature_matrices[feature] = matrix

    matrix_memory = (
        matrix.data.nbytes
        + matrix.indices.nbytes
        + matrix.indptr.nbytes
    )

    print(
        f"{feature}: "
        f"shape={matrix.shape}, "
        f"memory={format_memory(matrix_memory):.2f} MB"
    )

build_time = time.perf_counter() - build_start

print(f"\nTF-IDF build time: {build_time:.2f} seconds")


# --------------------------------------------------
# BAYESIAN RATINGS
# --------------------------------------------------

C = df["vote_average"].mean()
m = df["vote_count"].quantile(0.75)

v = df["vote_count"]
R = df["vote_average"]

df["weighted_rating"] = (
    (v / (v + m)) * R
    + (m / (v + m)) * C
)

print(f"\nDataset average rating: {C:.2f}")
print(f"75th percentile vote count: {m:.0f}")


# --------------------------------------------------
# RECOMMENDATION FUNCTION
# --------------------------------------------------

def recommend_movies(movie_id, top_n=5, quality_weight=0.15):

    matches = df.index[df["id"] == movie_id]

    if len(matches) == 0:
        return None

    movie_index = matches[0]

    combined_scores = np.zeros(len(df))

    for feature in FEATURES:

        matrix = feature_matrices[feature]

        scores = cosine_similarity(
            matrix[movie_index],
            matrix
        ).flatten()

        combined_scores += scores * WEIGHTS[feature]

    combined_scores[movie_index] = -1

    candidate_indices = (
        combined_scores.argsort()[::-1][:50]
    )

    candidates = df.iloc[candidate_indices].copy()

    candidates["similarity"] = (
        combined_scores[candidate_indices]
    )

    candidates["quality_score"] = (
        candidates["weighted_rating"] / 10
    )

    candidates["final_score"] = (
        (1 - quality_weight) * candidates["similarity"]
        + quality_weight * candidates["quality_score"]
    )

    return candidates.sort_values(
        "final_score",
        ascending=False
    ).head(top_n)[
        [
            "title",
            "release_date",
            "vote_count",
            "vote_average",
            "similarity",
            "final_score"
        ]
    ]


# --------------------------------------------------
# BENCHMARK RECOMMENDATIONS
# --------------------------------------------------

print("\nTesting recommendation performance...")

query_times = []

for movie_title, movie_id in TEST_MOVIES.items():

    query_start = time.perf_counter()

    recommendations = recommend_movies(movie_id)

    query_time = time.perf_counter() - query_start

    query_times.append(query_time)

    print("\n" + "=" * 60)
    print(f"RECOMMENDATIONS FOR: {movie_title}")
    print(f"Query time: {query_time:.3f} seconds")
    print("=" * 60)

    if recommendations is not None:
        print(recommendations.to_string(index=False))


# --------------------------------------------------
# FINAL METRICS
# --------------------------------------------------

current_memory, peak_memory = tracemalloc.get_traced_memory()

tracemalloc.stop()

print("\n" + "=" * 60)
print("BENCHMARK SUMMARY")
print("=" * 60)

print(f"Total movies: {len(df):,}")
print(f"Data preparation: {load_time:.2f} seconds")
print(f"TF-IDF construction: {build_time:.2f} seconds")
print(f"Average query time: {np.mean(query_times):.3f} seconds")

print(
    f"Python-tracked peak memory: "
    f"{format_memory(peak_memory):.2f} MB"
)

print(
    "\nNote: tracemalloc does not measure all memory "
    "allocated by NumPy, SciPy, and other native libraries."
)