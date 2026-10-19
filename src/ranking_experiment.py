
"""CineMatch Step 38: Full-catalog ranking experiment.

Compares:
1. Original hybrid
2. Hybrid with a modest Bayesian quality adjustment
3. Percentile-calibrated hybrid

Run from the project root:
    python3 src/ranking_experiment.py

Requires the full semantic embedding cache generated in Step 37.
Does not regenerate embeddings or modify the Streamlit app.
"""

from pathlib import Path
import hashlib
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "movies_expanded.csv"

CACHE_DIR = ROOT / "data" / "semantic_cache"

EMBEDDINGS_PATH = CACHE_DIR / "full_embeddings.npy"

CACHE_METADATA_PATH = (
    CACHE_DIR / "full_embeddings_metadata.json"
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TOP_N = 10

SEMANTIC_WEIGHT = 0.50
METADATA_WEIGHT = 0.50

# Maximum possible quality adjustment is intentionally small.
QUALITY_STRENGTH = 0.05

FEATURE_WEIGHTS = {
    "overview": 0.40,
    "genres": 0.25,
    "keywords": 0.20,
    "directors": 0.10,
    "cast": 0.05,
}

TEST_MOVIES = {
    "The Batman (2022)": 414906,
    "Inception": 27205,
    "Interstellar": 157336,
    "The Conjuring": 138843,
    "La La Land": 313369,
    "Get Out": 419430,
    "Knives Out": 546554,
}


# ============================================================
# 2. LOAD THE CATALOG
# ============================================================

def load_movies():

    columns = [
        "id",
        "title",
        "overview",
        "genres",
        "keywords",
        "directors",
        "cast",
        "vote_average",
        "vote_count",
    ]

    df = pd.read_csv(
        DATA_PATH,
        usecols=columns,
        low_memory=False,
    )

    df["id"] = pd.to_numeric(
        df["id"],
        errors="coerce",
    )

    df = df.dropna(subset=["id"]).copy()

    df["id"] = df["id"].astype("int64")

    df = df.drop_duplicates(
        subset="id"
    ).reset_index(drop=True)

    for column in [
        "title",
        "overview",
        "genres",
        "keywords",
        "directors",
        "cast",
    ]:
        df[column] = (
            df[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    for column in [
        "vote_average",
        "vote_count",
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        ).fillna(0)

    print(f"Loaded {len(df):,} movies.")

    return df


# ============================================================
# 3. LOAD EXISTING SEMANTIC EMBEDDINGS
# ============================================================

def catalog_fingerprint(df):

    digest = hashlib.sha256()

    for movie_id, overview in zip(
        df["id"],
        df["overview"],
    ):
        digest.update(
            str(movie_id).encode("utf-8")
        )

        digest.update(b"\0")

        digest.update(
            overview.encode("utf-8")
        )

        digest.update(b"\0")

    return digest.hexdigest()


def load_embeddings(df):

    if not (
        EMBEDDINGS_PATH.exists()
        and CACHE_METADATA_PATH.exists()
    ):
        raise FileNotFoundError(
            "Full embeddings cache not found. "
            "Run src/hybrid_recommender.py first."
        )

    metadata = json.loads(
        CACHE_METADATA_PATH.read_text()
    )

    expected_fingerprint = catalog_fingerprint(df)

    if (
        metadata.get("model") != MODEL_NAME
        or metadata.get("fingerprint") != expected_fingerprint
        or metadata.get("rows") != len(df)
    ):
        raise ValueError(
            "Embedding cache does not match the "
            "current catalog or model. "
            "Regenerate it using hybrid_recommender.py."
        )

    embeddings = np.load(
        EMBEDDINGS_PATH,
        mmap_mode="r",
    )

    expected_shape = (
        len(df),
        metadata.get("dimensions"),
    )

    if embeddings.shape != expected_shape:
        raise ValueError(
            "Embedding dimensions do not match "
            "the saved cache metadata."
        )

    print(
        f"Loaded cached embeddings: {embeddings.shape}"
    )

    return embeddings


# ============================================================
# 4. BUILD TF-IDF MATRICES
# ============================================================

def prepare_metadata(series):

    def clean(value):

        items = [
            item.strip().lower().replace(" ", "_")
            for item in value.split(",")
            if item.strip()
        ]

        return " ".join(items)

    return series.map(clean)


def build_metadata_matrices(df):

    matrices = {}

    for feature in FEATURE_WEIGHTS:

        print(f"Building TF-IDF: {feature}")

        if feature == "overview":
            texts = df[feature].str.lower()
        else:
            texts = prepare_metadata(
                df[feature]
            )

        if not texts.str.strip().any():

            matrices[feature] = csr_matrix(
                (len(df), 1),
                dtype=np.float32,
            )

            continue

        vectorizer = TfidfVectorizer(
            max_features=10_000,
            stop_words="english",
            token_pattern=r"(?u)\b[\w-]+\b",
            dtype=np.float32,
        )

        matrices[feature] = (
            vectorizer.fit_transform(texts)
        )

    return matrices


# ============================================================
# 5. CALCULATE RAW SIMILARITY
# ============================================================

def semantic_scores(index, embeddings):

    query = np.asarray(
        embeddings[index],
        dtype=np.float32,
    )

    scores = embeddings @ query

    return np.asarray(
        scores,
        dtype=np.float32,
    ).ravel()


def metadata_scores(index, matrices):

    count = next(
        iter(matrices.values())
    ).shape[0]

    scores = np.zeros(
        count,
        dtype=np.float32,
    )

    for feature, weight in FEATURE_WEIGHTS.items():

        matrix = matrices[feature]

        query = matrix.getrow(index)

        if query.nnz == 0:
            continue

        feature_scores = (
            matrix @ query.T
        ).toarray().ravel()

        scores += weight * feature_scores

    return scores


# ============================================================
# 6. BAYESIAN QUALITY
# ============================================================

def calculate_quality(df):

    votes = df["vote_count"].to_numpy(
        dtype=np.float64
    )

    ratings = df["vote_average"].to_numpy(
        dtype=np.float64
    )

    # Reference ratings are drawn from movies
    # with at least 50 votes.
    qualified = (
        (votes >= 50)
        & (ratings > 0)
        & (ratings <= 10)
    )

    if not np.any(qualified):
        raise ValueError(
            "No qualified movies for Bayesian rating."
        )

    C = float(
        np.mean(ratings[qualified])
    )

    m = float(
        np.percentile(
            votes[qualified],
            75,
        )
    )

    weighted_rating = (
        (votes * ratings + m * C)
        / (votes + m)
    )

    confidence = (
        votes / (votes + m)
    )

    # A positive-only quality bonus:
    # Below-average movies get no extra reward.
    # Ratings are on a 0–10 scale.
    positive_quality = np.clip(
        (weighted_rating - C) / 10.0,
        0.0,
        1.0,
    )

    quality_bonus = (
        QUALITY_STRENGTH
        * confidence
        * positive_quality
    )

    print(
        f"Bayesian reference average C: {C:.4f}"
    )

    print(
        f"Bayesian vote threshold m: {m:.1f}"
    )

    return (
        quality_bonus.astype(np.float32),
        weighted_rating.astype(np.float32),
        confidence.astype(np.float32),
    )


# ============================================================
# 7. THREE RANKING STRATEGIES
# ============================================================

def calculate_rankings(
    semantic,
    metadata,
    quality_bonus,
):

    # Strategy A: original 50/50 hybrid.
    baseline = (
        SEMANTIC_WEIGHT * semantic
        + METADATA_WEIGHT * metadata
    )

    # Strategy B: preserve similarity, add a
    # modest confidence-aware quality bonus.
    quality_aware = (
        baseline + quality_bonus
    )

    # Strategy C: convert each signal to its
    # percentile rank across the entire catalog.
    #
    # This equalizes ranking distributions.
    # It does NOT guarantee better relevance.
    semantic_percentile = (
        rankdata(
            semantic,
            method="average",
        ) / len(semantic)
    )

    metadata_percentile = (
        rankdata(
            metadata,
            method="average",
        ) / len(metadata)
    )

    calibrated = (
        SEMANTIC_WEIGHT * semantic_percentile
        + METADATA_WEIGHT * metadata_percentile
    )

    return {
        "Original hybrid": baseline,
        "Quality-aware hybrid": quality_aware,
        "Calibrated hybrid": calibrated,
    }


# ============================================================
# 8. DISPLAY RESULTS
# ============================================================

def top_indices(scores, query_index, n):

    working = scores.copy()

    working[query_index] = -np.inf

    count = min(
        n,
        len(working) - 1,
    )

    if count <= 0:
        return np.array([], dtype=int)

    indices = np.argpartition(
        working,
        -count,
    )[-count:]

    # Deterministic ordering by score, then index.
    return np.array(
        sorted(
            indices,
            key=lambda i: (
                -float(working[i]),
                int(i),
            ),
        ),
        dtype=int,
    )


def print_ranking(
    name,
    scores,
    query_index,
    df,
    semantic,
    metadata,
    quality_bonus,
):

    indices = top_indices(
        scores,
        query_index,
        TOP_N,
    )

    results = df.iloc[indices][
        [
            "title",
            "vote_count",
        ]
    ].copy()

    results.insert(
        0,
        "rank",
        range(1, len(results) + 1),
    )

    results["semantic"] = semantic[indices]
    results["metadata"] = metadata[indices]
    results["quality_bonus"] = quality_bonus[indices]
    results["final_score"] = scores[indices]

    print(f"\n{name}")
    print("-" * 100)

    print(
        results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


# ============================================================
# 9. RUN EXPERIMENT
# ============================================================

def main():

    start = time.time()

    print("=" * 70)
    print("CINEMATCH — STEP 38 RANKING EXPERIMENT")
    print("=" * 70)

    df = load_movies()

    embeddings = load_embeddings(df)

    matrices = build_metadata_matrices(df)

    (
        quality_bonus,
        weighted_rating,
        confidence,
    ) = calculate_quality(df)

    movie_ids = df["id"].to_numpy()

    for name, movie_id in TEST_MOVIES.items():

        matches = np.flatnonzero(
            movie_ids == movie_id
        )

        print("\n" + "=" * 110)
        print(name)
        print("=" * 110)

        if len(matches) == 0:
            print("Movie not found.")
            continue

        query_index = int(matches[0])

        semantic = semantic_scores(
            query_index,
            embeddings,
        )

        metadata = metadata_scores(
            query_index,
            matrices,
        )

        rankings = calculate_rankings(
            semantic,
            metadata,
            quality_bonus,
        )

        for strategy_name, scores in rankings.items():

            print_ranking(
                strategy_name,
                scores,
                query_index,
                df,
                semantic,
                metadata,
                quality_bonus,
            )

    elapsed = time.time() - start

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)

    print(
        f"Runtime: {elapsed:.1f} seconds"
    )

    print(
        "No semantic embeddings were regenerated."
    )

    print(
        "Scores are ranking signals, not probabilities."
    )

    print(
        "Calibrated scores use percentile ranks "
        "and are not directly comparable in magnitude "
        "with raw hybrid scores."
    )


if __name__ == "__main__":
    main()
