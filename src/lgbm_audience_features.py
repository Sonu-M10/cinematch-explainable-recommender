
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


# ==================================================
# CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parents[1]
ML_DIR = ROOT / "data" / "movielens"

TRAIN_PATH = ML_DIR / "lgbm_train.csv.gz"
CATALOG_PATH = ROOT / "data" / "movies_with_imdb.csv"

AFFINITY_PATH = ML_DIR / "lgbm_audience_affinity.csv"
UNIVERSE_PATH = ML_DIR / "lgbm_movie_universe.csv"

POSITIVE_THRESHOLD = 4.0
TOP_MOVIES = 1500
MIN_POSITIVE_RATINGS = 20
MIN_SHARED_USERS = 10
SUPPORT_SHRINKAGE = 25
TOP_NEIGHBORS = 100


# ==================================================
# LOAD TRAINING INTERACTIONS
# ==================================================

def load_training_data():
    print("Loading LightGBM training interactions...")

    df = pd.read_csv(
        TRAIN_PATH,
        usecols=["userId", "cinematch_id", "rating"],
        dtype={
            "userId": "int32",
            "cinematch_id": "int32",
            "rating": "float32"
        }
    )

    if df.duplicated(
        ["userId", "cinematch_id"]
    ).any():
        raise ValueError(
            "Duplicate user-movie interactions detected."
        )

    print(f"Training interactions: {len(df):,}")

    return df


# ==================================================
# SELECT MOVIE UNIVERSE
# ==================================================

def select_universe(df):
    print("\nSelecting supported movies...")

    positives = df.loc[
        df["rating"] >= POSITIVE_THRESHOLD,
        ["userId", "cinematch_id"]
    ].copy()

    counts = (
        positives["cinematch_id"]
        .value_counts()
    )

    eligible = (
        counts[counts >= MIN_POSITIVE_RATINGS]
        .head(TOP_MOVIES)
    )

    universe = pd.DataFrame({
        "cinematch_id": eligible.index.to_numpy(),
        "positive_rating_count": eligible.to_numpy()
    })

    catalog = pd.read_csv(
        CATALOG_PATH,
        usecols=["id", "title"]
    ).drop_duplicates("id")

    universe = universe.merge(
        catalog,
        left_on="cinematch_id",
        right_on="id",
        how="left"
    ).drop(columns=["id"])

    universe.to_csv(
        UNIVERSE_PATH,
        index=False
    )

    positives = positives[
        positives["cinematch_id"].isin(
            universe["cinematch_id"]
        )
    ].copy()

    print(f"Selected movies: {len(universe):,}")
    print(f"Positive interactions: {len(positives):,}")

    return positives, universe


# ==================================================
# BUILD SPARSE MATRIX
# ==================================================

def build_matrix(positives):
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
    print(f"Interactions: {matrix.nnz:,}")

    return matrix, np.asarray(movie_ids)


# ==================================================
# COMPUTE AFFINITY
# ==================================================

def calculate_affinity(matrix, movie_ids):
    print("\nCalculating audience relationships...")

    shared_matrix = (matrix.T @ matrix).tocoo()

    audience_sizes = np.asarray(
        matrix.sum(axis=0)
    ).ravel()

    rows = shared_matrix.row
    cols = shared_matrix.col
    shared = shared_matrix.data

    valid = (
        (rows != cols)
        & (shared >= MIN_SHARED_USERS)
    )

    rows = rows[valid]
    cols = cols[valid]
    shared = shared[valid]

    denominator = np.sqrt(
        audience_sizes[rows] * audience_sizes[cols]
    )

    cosine = shared / np.maximum(
        denominator, 1.0
    )

    adjusted = cosine * (
        shared / (shared + SUPPORT_SHRINKAGE)
    )

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
# ADD TITLES AND REPORT
# ==================================================

def attach_titles(results):
    catalog = pd.read_csv(
        CATALOG_PATH,
        usecols=["id", "title"]
    ).drop_duplicates("id")

    titles = catalog.set_index("id")["title"]

    results["anchor_title"] = (
        results["anchor_id"].map(titles)
    )

    results["candidate_title"] = (
        results["candidate_id"].map(titles)
    )

    return results


def print_report(results, universe):
    print("\n" + "=" * 60)
    print("CINEMATCH — LIGHTGBM AUDIENCE FEATURES")
    print("=" * 60)

    print(f"Eligible movies: {len(universe):,}")
    print(f"Audience pairs saved: {len(results):,}")
    print(
        "Anchors with recommendations: "
        f"{results['anchor_id'].nunique():,}"
    )

    for title in ["Inception", "Interstellar", "The Martian"]:
        subset = results[
            results["anchor_title"] == title
        ].head(5)

        if subset.empty:
            continue

        print(f"\nTop audience neighbors for {title}:")

        print(
            subset[
                [
                    "candidate_title",
                    "shared_positive_users",
                    "adjusted_affinity"
                ]
            ].to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}"
            )
        )

    print(f"\nAffinity file: {AFFINITY_PATH}")
    print(f"Movie universe: {UNIVERSE_PATH}")


# ==================================================
# MAIN
# ==================================================

def main():
    df = load_training_data()

    positives, universe = select_universe(df)

    matrix, movie_ids = build_matrix(positives)

    results = calculate_affinity(
        matrix,
        movie_ids
    )

    results = attach_titles(results)

    results.to_csv(
        AFFINITY_PATH,
        index=False
    )

    print_report(results, universe)


if __name__ == "__main__":
    main()
