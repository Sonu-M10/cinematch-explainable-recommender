
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# PROJECT PATHS AND CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "movies_expanded.csv"

FEATURE_WEIGHTS = {
    "overview": 0.40,
    "genres": 0.25,
    "keywords": 0.20,
    "directors": 0.10,
    "cast": 0.05,
}

GENRE_DISCOUNT = 0.15
ADAPTIVE_QUALITY_WEIGHT = 0.30
FIXED_QUALITY_WEIGHT = 0.15

CANDIDATE_POOL_SIZE = 100


# --------------------------------------------------
# METADATA PREPARATION
# --------------------------------------------------

def normalize_entities(value):
    if pd.isna(value):
        return ""

    return " ".join(
        part.strip().lower().replace(" ", "_")
        for part in str(value).split(",")
        if part.strip()
    )


def load_data():
    data = pd.read_csv(
        DATA_PATH,
        low_memory=False
    ).reset_index(drop=True)

    data["id"] = pd.to_numeric(
        data["id"],
        errors="coerce"
    )

    data = data.dropna(subset=["id"]).copy()
    data["id"] = data["id"].astype("int64")

    data = data.drop_duplicates(
        subset="id",
        keep="first"
    ).reset_index(drop=True)

    for field in FEATURE_WEIGHTS:
        data[field] = (
            data[field].fillna("").astype(str)
        )

    for field in [
        "genres",
        "keywords",
        "directors",
        "cast",
    ]:
        data[field + "_processed"] = (
            data[field].map(normalize_entities)
        )

    for field in ["vote_count", "vote_average"]:
        data[field] = (
            pd.to_numeric(
                data[field],
                errors="coerce"
            )
            .fillna(0)
            .clip(lower=0)
        )

    data["vote_average"] = (
        data["vote_average"].clip(upper=10)
    )

    return data


# --------------------------------------------------
# LOAD DATA AND BUILD TF-IDF MATRICES
# --------------------------------------------------

print("Loading CineMatch dataset...")

df = load_data()

print(f"Movies loaded: {len(df):,}")

feature_matrices = {}

for feature in FEATURE_WEIGHTS:

    column = (
        feature
        if feature == "overview"
        else feature + "_processed"
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=10000,
        token_pattern=r"(?u)\b[\w-]+\b",
    )

    feature_matrices[feature] = (
        vectorizer.fit_transform(df[column])
    )

print("TF-IDF matrices ready.")


# --------------------------------------------------
# BAYESIAN RATING CALCULATION
# --------------------------------------------------

qualified = df.loc[
    df["vote_count"] >= 50
]

if qualified.empty:
    raise ValueError(
        "No movies with 50+ votes to establish "
        "the Bayesian reference."
    )

C = float(
    qualified["vote_average"].mean()
)

m = float(
    qualified["vote_count"].quantile(0.75)
)

v = df["vote_count"]
R = df["vote_average"]

df["weighted_rating"] = (
    (v * R + m * C) / (v + m)
)

df["quality_bonus"] = (
    (df["weighted_rating"] - C) / 10
).clip(lower=0)

df["rating_confidence"] = (
    v / (v + m)
)

movie_id_to_index = dict(
    zip(df["id"], df.index)
)


# --------------------------------------------------
# MAIN RECOMMENDATION FUNCTION
# --------------------------------------------------

def recommend_movies(
    movie_id,
    top_n=10,
    ranking_mode="evidence_aware",
    quality_weight=None,
    candidate_pool_size=CANDIDATE_POOL_SIZE,
):
    """
    Return movie recommendations with explainable scores.

    ranking_mode:
        evidence_aware
        pure_similarity
        fixed_quality

    Legacy behavior:
        quality_weight=0 selects pure similarity.
        Positive quality_weight selects fixed quality.
    """

    if quality_weight is not None:
        ranking_mode = (
            "pure_similarity"
            if quality_weight == 0
            else "fixed_quality"
        )

    if ranking_mode not in {
        "evidence_aware",
        "pure_similarity",
        "fixed_quality",
    }:
        raise ValueError("Unknown ranking_mode")

    if top_n < 1 or candidate_pool_size < 1:
        raise ValueError(
            "top_n and candidate_pool_size "
            "must be positive"
        )

    if movie_id not in movie_id_to_index:
        return None

    index = movie_id_to_index[movie_id]

    scores = np.zeros(
        len(df),
        dtype=np.float64
    )

    contributions = {}

    # Calculate weighted similarity for each feature.
    for feature, weight in FEATURE_WEIGHTS.items():

        matrix = feature_matrices[feature]

        values = cosine_similarity(
            matrix[index],
            matrix
        ).ravel() * weight

        scores += values
        contributions[feature] = values

    # Exclude the selected movie.
    scores[index] = -1

    pool_size = min(
        len(df) - 1,
        max(top_n, candidate_pool_size)
    )

    if pool_size == 0:
        return df.iloc[:0].copy()

    # Find top candidates without sorting the entire
    # movie catalog.
    indices = np.argpartition(
        scores,
        -pool_size
    )[-pool_size:]

    indices = indices[
        np.argsort(
            -scores[indices],
            kind="stable"
        )
    ]

    results = df.iloc[indices].copy()

    results["similarity"] = scores[indices]

    for feature in FEATURE_WEIGHTS:
        results[
            feature + "_contribution"
        ] = contributions[feature][indices]

    # Measure similarity from non-genre metadata.
    results["specific_similarity"] = sum(
        results[f + "_contribution"]
        for f in [
            "overview",
            "keywords",
            "directors",
            "cast",
        ]
    )

    results["specificity"] = (
        results["specific_similarity"]
        / results["similarity"].clip(lower=1e-10)
    ).clip(0, 1)

    results["genre_dependence"] = (
        1 - results["specificity"]
    )

    # Discount matches relying heavily on genres.
    results["adjusted_similarity"] = (
        results["similarity"]
        * (
            1
            - GENRE_DISCOUNT
            * results["genre_dependence"]
        )
    )

    # Rating contribution is stronger when:
    # 1. Ratings have more evidence.
    # 2. The content match is genre-dominated.
    results["adaptive_quality_bonus"] = (
        ADAPTIVE_QUALITY_WEIGHT
        * results["rating_confidence"]
        * results["genre_dependence"]
        * results["quality_bonus"]
    )

    # Choose ranking policy.
    if ranking_mode == "evidence_aware":

        results["final_score"] = (
            results["adjusted_similarity"]
            + results["adaptive_quality_bonus"]
        )

    elif ranking_mode == "fixed_quality":

        weight = (
            FIXED_QUALITY_WEIGHT
            if quality_weight is None
            else quality_weight
        )

        results["final_score"] = (
            results["similarity"]
            + weight * results["quality_bonus"]
        )

    else:
        results["final_score"] = (
            results["similarity"]
        )

    return (
        results.sort_values(
            ["final_score", "similarity", "id"],
            ascending=[False, False, True]
        )
        .head(top_n)
        .reset_index(drop=True)
    )


# --------------------------------------------------
# LOCAL TEST
# --------------------------------------------------

if __name__ == "__main__":

    test_movies = {
        "Inception": 27205,
        "Interstellar": 157336,
        "The Conjuring": 138843,
        "La La Land": 313369,
        "Get Out": 419430,
        "Knives Out": 546554,
    }

    for name, movie_id in test_movies.items():

        print("\n" + name)

        output = recommend_movies(
            movie_id,
            top_n=5
        )

        if output is None:
            print("Not in catalog")
        else:
            print(
                output[
                    [
                        "title",
                        "similarity",
                        "specificity",
                        "final_score",
                    ]
                ].to_string(index=False)
            )
