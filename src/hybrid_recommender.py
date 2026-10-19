
"""CineMatch: full-catalog semantic + metadata hybrid experiment.

Run:
    python3 src/hybrid_recommender.py

The first run generates and caches semantic embeddings.
Subsequent runs reuse them if the movie IDs and overviews match.
"""

from pathlib import Path
import hashlib
import json
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "movies_expanded.csv"
CACHE_DIR = ROOT / "data" / "semantic_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

BATCH_SIZE = 32
TOP_N = 10
CANDIDATES_PER_METHOD = 300

# Initial experiment: semantic and metadata contribute equally.
SEMANTIC_WEIGHT = 0.50
METADATA_WEIGHT = 0.50

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
# 2. LOAD MOVIES
# ============================================================

def load_movies():
    print("\nLoading movie catalog...")

    columns = [
        "id",
        "title",
        "overview",
        "genres",
        "keywords",
        "directors",
        "cast",
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

    df["vote_count"] = pd.to_numeric(
        df["vote_count"],
        errors="coerce",
    ).fillna(0)

    print(f"Loaded {len(df):,} movies.")

    return df


# ============================================================
# 3. SEMANTIC EMBEDDINGS
# ============================================================

def catalog_fingerprint(df):
    """Identify the exact movie IDs and overviews in this catalog."""

    digest = hashlib.sha256()

    for movie_id, overview in zip(
        df["id"],
        df["overview"],
    ):
        digest.update(str(movie_id).encode("utf-8"))
        digest.update(b"\0")
        digest.update(overview.encode("utf-8"))
        digest.update(b"\0")

    return digest.hexdigest()


def get_semantic_embeddings(df):
    print("\nLoading semantic model...")

    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu",
    )

    embedding_path = (
        CACHE_DIR / "full_embeddings.npy"
    )

    metadata_path = (
        CACHE_DIR / "full_embeddings_metadata.json"
    )

    fingerprint = catalog_fingerprint(df)

    expected_metadata = {
        "model": MODEL_NAME,
        "fingerprint": fingerprint,
        "rows": len(df),
        "dimensions": model.get_sentence_embedding_dimension(),
    }

    if (
        embedding_path.exists()
        and metadata_path.exists()
    ):
        try:
            saved_metadata = json.loads(
                metadata_path.read_text()
            )

            if saved_metadata == expected_metadata:
                embeddings = np.load(
                    embedding_path,
                    mmap_mode="r",
                )

                expected_shape = (
                    len(df),
                    expected_metadata["dimensions"],
                )

                if embeddings.shape == expected_shape:
                    print(
                        "Loaded full-catalog cached embeddings."
                    )
                    return embeddings

        except (OSError, ValueError, json.JSONDecodeError):
            print("Existing cache invalid; regenerating.")

    print(
        f"Encoding all {len(df):,} movie overviews..."
    )

    print(
        "The first run may take several minutes."
    )

    start = time.time()

    # A memory-mapped file avoids accumulating every
    # embedding in RAM while encoding the catalog.
    temp_path = CACHE_DIR / "full_embeddings.tmp.npy"

    dimensions = model.get_sentence_embedding_dimension()

    output = np.lib.format.open_memmap(
        temp_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(df), dimensions),
    )

    for start_index in range(0, len(df), BATCH_SIZE):
        end_index = min(
            start_index + BATCH_SIZE,
            len(df),
        )

        texts = df["overview"].iloc[
            start_index:end_index
        ].tolist()

        batch_embeddings = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        output[start_index:end_index] = (
            batch_embeddings.astype(np.float32)
        )

        if (
            end_index % 5000 < BATCH_SIZE
            or end_index == len(df)
        ):
            print(
                f"Encoded {end_index:,}/{len(df):,} movies"
            )

    output.flush()
    del output

    temp_path.replace(embedding_path)

    metadata_path.write_text(
        json.dumps(expected_metadata, indent=2)
    )

    elapsed = time.time() - start

    print(
        f"Finished encoding in {elapsed / 60:.1f} minutes."
    )

    return np.load(
        embedding_path,
        mmap_mode="r",
    )


# ============================================================
# 4. METADATA PREPROCESSING
# ============================================================

def prepare_metadata(text_series):
    """Convert comma-separated metadata into searchable tokens."""

    def clean(value):
        items = [
            item.strip().lower().replace(" ", "_")
            for item in value.split(",")
            if item.strip()
        ]
        return " ".join(items)

    return text_series.map(clean)


def build_metadata_matrices(df):
    print("\nBuilding TF-IDF metadata matrices...")

    matrices = {}

    for feature in FEATURE_WEIGHTS:
        print(f"Processing: {feature}")

        if feature == "overview":
            texts = df[feature].str.lower()
        else:
            texts = prepare_metadata(df[feature])

        vectorizer = TfidfVectorizer(
            max_features=10_000,
            stop_words="english",
            token_pattern=r"(?u)\b[\w-]+\b",
            dtype=np.float32,
        )

        # Avoid errors if an entire feature column is empty.
        if not texts.str.strip().any():
            matrices[feature] = csr_matrix(
                (len(df), 1),
                dtype=np.float32,
            )
            continue

        matrices[feature] = vectorizer.fit_transform(
            texts
        )

        print(
            f"  Shape: {matrices[feature].shape}"
        )

    return matrices


# ============================================================
# 5. SIMILARITY CALCULATIONS
# ============================================================

def semantic_scores(query_index, embeddings):
    """Cosine similarity for normalized semantic vectors."""

    query = np.asarray(
        embeddings[query_index],
        dtype=np.float32,
    )

    scores = embeddings @ query

    return np.asarray(
        scores,
        dtype=np.float32,
    ).ravel()


def metadata_scores(query_index, matrices):
    """Weighted cosine similarity across the five TF-IDF features."""

    n_movies = next(
        iter(matrices.values())
    ).shape[0]

    scores = np.zeros(
        n_movies,
        dtype=np.float32,
    )

    for feature, weight in FEATURE_WEIGHTS.items():
        matrix = matrices[feature]

        query_vector = matrix.getrow(query_index)

        if query_vector.nnz == 0:
            continue

        feature_scores = (
            matrix @ query_vector.T
        ).toarray().ravel()

        scores += weight * feature_scores

    return scores


def top_indices(scores, n, exclude_index):
    """Get top candidates without sorting the whole catalog."""

    working = scores.copy()
    working[exclude_index] = -np.inf

    count = min(n, len(working) - 1)

    if count <= 0:
        return np.array([], dtype=int)

    indices = np.argpartition(
        working,
        -count,
    )[-count:]

    return indices[
        np.argsort(working[indices])[::-1]
    ]


# ============================================================
# 6. HYBRID CANDIDATE RETRIEVAL
# ============================================================

def recommend(
    movie_id,
    df,
    embeddings,
    matrices,
):
    matches = np.flatnonzero(
        df["id"].to_numpy() == movie_id
    )

    if len(matches) == 0:
        return None

    query_index = int(matches[0])

    semantic = semantic_scores(
        query_index,
        embeddings,
    )

    metadata = metadata_scores(
        query_index,
        matrices,
    )

    semantic_candidates = top_indices(
        semantic,
        CANDIDATES_PER_METHOD,
        query_index,
    )

    metadata_candidates = top_indices(
        metadata,
        CANDIDATES_PER_METHOD,
        query_index,
    )

    # Include movies found by either retrieval method.
    candidates = np.union1d(
        semantic_candidates,
        metadata_candidates,
    )

    # Both component scores are cosine-based.
    # No per-query min-max scaling is applied.
    hybrid = (
        SEMANTIC_WEIGHT * semantic[candidates]
        + METADATA_WEIGHT * metadata[candidates]
    )

    results = df.iloc[candidates][
        ["id", "title", "vote_count"]
    ].copy()

    results["semantic"] = semantic[candidates]
    results["metadata"] = metadata[candidates]
    results["hybrid"] = hybrid

    results = results.sort_values(
        ["hybrid", "id"],
        ascending=[False, True],
    ).head(TOP_N)

    return results.reset_index(drop=True)


# ============================================================
# 7. RUN FULL-CATALOG EXPERIMENT
# ============================================================

def main():
    print("=" * 65)
    print("CINEMATCH — FULL-CATALOG HYBRID EXPERIMENT")
    print("=" * 65)

    start = time.time()

    df = load_movies()

    embeddings = get_semantic_embeddings(df)

    matrices = build_metadata_matrices(df)

    print("\nRunning recommendation tests...")

    for movie_name, movie_id in TEST_MOVIES.items():
        print("\n" + "=" * 80)
        print(movie_name)
        print("=" * 80)

        results = recommend(
            movie_id,
            df,
            embeddings,
            matrices,
        )

        if results is None:
            print("Movie not found in catalog.")
            continue

        print(
            results[
                [
                    "title",
                    "vote_count",
                    "semantic",
                    "metadata",
                    "hybrid",
                ]
            ].to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )

    elapsed = time.time() - start

    print("\n" + "=" * 65)
    print("EXPERIMENT COMPLETE")
    print("=" * 65)

    print(
        f"Total runtime: {elapsed / 60:.1f} minutes"
    )

    print(
        "Similarity values are scores, not probabilities."
    )

    print(
        "This experiment does not apply quality bonuses "
        "or learned adaptive weights."
    )


if __name__ == "__main__":
    main()
