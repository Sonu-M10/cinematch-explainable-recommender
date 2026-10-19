
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


# ==================================================
# CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parents[1]

ML_DIR = ROOT / "data" / "movielens"

TRAIN_PATH = ML_DIR / "ranking_train.csv.gz"

CATALOG_PATH = ROOT / "data" / "movies_with_imdb.csv"

OUTPUT_PATH = ML_DIR / "audience_affinity.csv"

POSITIVE_THRESHOLD = 4.0
TOP_MOVIES = 1500
MIN_SHARED_USERS = 10
MIN_POSITIVE_RATINGS = 20
SUPPORT_SHRINKAGE = 25
TOP_NEIGHBORS = 30


# ==================================================
# LOAD POSITIVE INTERACTIONS
# ==================================================

def load_positive_interactions():
    print("Loading training interactions...")

    ratings = pd.read_csv(
        TRAIN_PATH,
        usecols=["userId", "cinematch_id", "rating"],
        dtype={
            "userId": "int32",
            "cinematch_id": "int32",
            "rating": "float32"
        }
    )

    positives = ratings.loc[
        ratings["rating"] >= POSITIVE_THRESHOLD,
        ["userId", "cinematch_id"]
    ].copy()

    positives = positives.drop_duplicates()

    print(f"Positive interactions: {len(positives):,}")

    counts = positives["cinematch_id"].value_counts()

    eligible = counts[
        counts >= MIN_POSITIVE_RATINGS
    ].head(TOP_MOVIES)

    positives = positives[
        positives["cinematch_id"].isin(eligible.index)
    ].copy()

    print(f"Selected movies: {len(eligible):,}")
    print(f"Selected interactions: {len(positives):,}")

    return positives


# ==================================================
# CREATE SPARSE USER-MOVIE MATRIX
# ==================================================

def create_matrix(positives):
    print("\nBuilding sparse user-movie matrix...")

    user_codes, user_ids = pd.factorize(
        positives["userId"],
        sort=True
    )

    movie_codes, movie_ids = pd.factorize(
        positives["cinematch_id"],
        sort=True
    )

    matrix = csr_matrix(
        (
            np.ones(len(positives), dtype=np.float32),
            (user_codes, movie_codes)
        ),
        shape=(len(user_ids), len(movie_ids))
    )

    matrix.sum_duplicates()

    print(f"Users: {matrix.shape[0]:,}")
    print(f"Movies: {matrix.shape[1]:,}")
    print(f"Nonzero interactions: {matrix.nnz:,}")

    return matrix, np.asarray(movie_ids)


# ==================================================
# CALCULATE AUDIENCE AFFINITY
# ==================================================

def calculate_affinity(matrix, movie_ids):
    print("\nCalculating shared positive audiences...")

    # Movie x movie co-occurrence matrix.
    cooccurrence = (matrix.T @ matrix).tocoo()

    audience_sizes = np.asarray(
        matrix.sum(axis=0)
    ).ravel()

    rows = cooccurrence.row
    cols = cooccurrence.col
    shared = cooccurrence.data

    valid = (
        (rows != cols)
        & (shared >= MIN_SHARED_USERS)
    )

    rows = rows[valid]
    cols = cols[valid]
    shared = shared[valid]

    if len(shared) == 0:
        raise RuntimeError(
            "No movie pairs meet the support threshold."
        )

    denominator = np.sqrt(
        audience_sizes[rows] * audience_sizes[cols]
    )

    cosine = shared / np.maximum(
        denominator, 1.0
    )

    # Reduce confidence in low-support pairs.
    shrinkage = shared / (
        shared + SUPPORT_SHRINKAGE
    )

    adjusted = cosine * shrinkage

    results = pd.DataFrame({
        "anchor_id": movie_ids[rows],
        "candidate_id": movie_ids[cols],
        "shared_positive_users": shared.astype(int),
        "anchor_positive_users": audience_sizes[rows].astype(int),
        "candidate_positive_users": audience_sizes[cols].astype(int),
        "cosine_affinity": cosine,
        "adjusted_affinity": adjusted
    })

    results = results.sort_values(
        ["anchor_id", "adjusted_affinity"],
        ascending=[True, False]
    )

    results = (
        results.groupby("anchor_id", sort=False)
        .head(TOP_NEIGHBORS)
        .reset_index(drop=True)
    )

    return results


# ==================================================
# ADD MOVIE TITLES
# ==================================================

def attach_titles(results):
    catalog = pd.read_csv(
        CATALOG_PATH,
        usecols=["id", "title"]
    ).drop_duplicates(subset=["id"])

    titles = catalog.set_index("id")["title"]

    results["anchor_title"] = (
        results["anchor_id"].map(titles)
    )

    results["candidate_title"] = (
        results["candidate_id"].map(titles)
    )

    return results


# ==================================================
# REPORT
# ==================================================

def print_examples(results):
    print("\n" + "=" * 60)
    print("CINEMATCH AUDIENCE AFFINITY BASELINE")
    print("=" * 60)

    print(f"Saved recommendation pairs: {len(results):,}")

    print(
        "Anchors with recommendations: "
        f"{results['anchor_id'].nunique():,}"
    )

    example_titles = [
        "Inception",
        "Interstellar",
        "The Dark Knight Rises",
        "The Martian"
    ]

    for title in example_titles:
        subset = results[
            results["anchor_title"] == title
        ].head(10)

        if subset.empty:
            continue

        print(f"\nRecommendations for {title}:")

        print(
            subset[
                [
                    "candidate_title",
                    "shared_positive_users",
                    "cosine_affinity",
                    "adjusted_affinity"
                ]
            ].to_string(
                index=False,
                float_format=lambda value: f"{value:.4f}"
            )
        )

    print(f"\nOutput: {OUTPUT_PATH}")


# ==================================================
# MAIN
# ==================================================

def main():
    positives = load_positive_interactions()

    matrix, movie_ids = create_matrix(
        positives
    )

    results = calculate_affinity(
        matrix,
        movie_ids
    )

    results = attach_titles(results)

    results.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print_examples(results)


if __name__ == "__main__":
    main()
