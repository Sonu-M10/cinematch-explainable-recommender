
from pathlib import Path
import time

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_ROOT / "data" / "movies_cleaned.csv"

FEATURE_WEIGHTS = {
    "overview": 0.40,
    "genres": 0.25,
    "keywords": 0.20,
    "directors": 0.10,
    "cast": 0.05,
}

TEST_MOVIES = {
    "Inception": 27205,
    "Interstellar": 157336,
    "The Conjuring": 138843,
    "La La Land": 313369,
    "Get Out": 419430,
    "Knives Out": 546554,
}

QUALITY_WEIGHT = 0.15
CANDIDATE_POOL_SIZE = 50
TOP_N = 5


# --------------------------------------------------
# METADATA PREPARATION
# --------------------------------------------------

def prepare_metadata(text):
    if not text:
        return ""

    return " ".join(
        item.strip().lower().replace(" ", "_")
        for item in str(text).split(",")
        if item.strip()
    )


print("Loading expanded dataset...")

df = pd.read_csv(DATASET_PATH)

df = df.dropna(
    subset=["overview", "genres"]
).copy()

df = df.reset_index(drop=True)

for feature in FEATURE_WEIGHTS:
    df[feature] = df[feature].fillna("")

for feature in ["genres", "keywords", "directors", "cast"]:
    df[f"{feature}_processed"] = df[feature].apply(
        prepare_metadata
    )

print(f"Movies available: {len(df):,}")


# --------------------------------------------------
# BUILD THE SAME TF-IDF MODEL
# --------------------------------------------------

print("\nBuilding TF-IDF matrices...")

start = time.perf_counter()

feature_matrices = {}

for feature in FEATURE_WEIGHTS:

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=10000,
        token_pattern=r"(?u)\b[\w-]+\b",
    )

    column = (
        feature
        if feature == "overview"
        else f"{feature}_processed"
    )

    feature_matrices[feature] = vectorizer.fit_transform(
        df[column]
    )

print(
    f"Model built in {time.perf_counter() - start:.2f} seconds"
)


# --------------------------------------------------
# RATING STRATEGIES
# --------------------------------------------------

# Strategy A: All movies, including unrated movies.
C_a = df["vote_average"].mean()
m_a = df["vote_count"].quantile(0.75)

# Strategy B: Original model parameters.
C_b = 6.37
m_b = 456

# Strategy C: Derive parameters from movies with >=50 votes.
qualified = df[df["vote_count"] >= 50]

C_c = qualified["vote_average"].mean()
m_c = qualified["vote_count"].quantile(0.75)

STRATEGIES = {
    "A - Expanded baseline": (C_a, m_a),
    "B - Original baseline": (C_b, m_b),
    "C - Qualified baseline": (C_c, m_c),
}

print("\nRating parameters:")

for name, (C, m) in STRATEGIES.items():
    print(f"{name}: C={C:.3f}, m={m:.0f}")

print(f"Qualified movies: {len(qualified):,}")


# --------------------------------------------------
# BAYESIAN RATING FUNCTION
# --------------------------------------------------

def weighted_rating(votes, rating, C, m):
    return (
        (votes / (votes + m)) * rating
        + (m / (votes + m)) * C
    )


# --------------------------------------------------
# CALCULATE SIMILARITY ONCE PER MOVIE
# --------------------------------------------------

def get_candidates(movie_id):

    matches = df.index[df["id"] == movie_id]

    if len(matches) == 0:
        return None

    movie_index = matches[0]

    combined_scores = np.zeros(len(df))

    for feature, weight in FEATURE_WEIGHTS.items():

        matrix = feature_matrices[feature]

        similarities = cosine_similarity(
            matrix[movie_index],
            matrix
        ).ravel()

        combined_scores += similarities * weight

    combined_scores[movie_index] = -1

    candidate_indices = np.argsort(
        combined_scores
    )[::-1][:CANDIDATE_POOL_SIZE]

    candidates = df.iloc[candidate_indices].copy()

    candidates["similarity"] = (
        combined_scores[candidate_indices]
    )

    return candidates


# --------------------------------------------------
# COMPARE RANKING STRATEGIES
# --------------------------------------------------

def rank_candidates(candidates, C, m):

    results = candidates.copy()

    results["weighted_rating"] = weighted_rating(
        results["vote_count"],
        results["vote_average"],
        C,
        m,
    )

    results["quality_score"] = (
        results["weighted_rating"] / 10
    )

    results["final_score"] = (
        (1 - QUALITY_WEIGHT) * results["similarity"]
        + QUALITY_WEIGHT * results["quality_score"]
    )

    return results.sort_values(
        "final_score",
        ascending=False
    ).head(TOP_N)


# --------------------------------------------------
# RUN COMPARISON
# --------------------------------------------------

for movie_title, movie_id in TEST_MOVIES.items():

    candidates = get_candidates(movie_id)

    if candidates is None:
        print(f"\nMovie not found: {movie_title}")
        continue

    print("\n" + "=" * 75)
    print(f"RECOMMENDATIONS FOR: {movie_title}")
    print("=" * 75)

    for strategy_name, (C, m) in STRATEGIES.items():

        results = rank_candidates(
            candidates,
            C,
            m,
        )

        print(f"\n{strategy_name}")

        print(
            results[
                [
                    "title",
                    "vote_count",
                    "vote_average",
                    "weighted_rating",
                    "similarity",
                    "final_score",
                ]
            ].to_string(index=False)
        )


print("\nComparison complete.")
