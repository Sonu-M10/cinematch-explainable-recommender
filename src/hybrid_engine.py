
"""CineMatch experimental hybrid recommendation engine.

Uses:
- Existing TF-IDF feature matrices from weighted_recommender.py
- Cached full-catalog semantic embeddings from Step 37

No model download or embedding generation occurs here.
"""

from pathlib import Path
import hashlib
import json

import numpy as np
from scipy.sparse import issparse

import weighted_recommender as classic


ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "semantic_cache"

EMBEDDINGS_PATH = CACHE_DIR / "full_embeddings.npy"
METADATA_PATH = CACHE_DIR / "full_embeddings_metadata.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

SEMANTIC_WEIGHT = 0.50
METADATA_WEIGHT = 0.50

# Reuse the existing catalog and TF-IDF matrices.
df = classic.df
FEATURE_WEIGHTS = classic.FEATURE_WEIGHTS
feature_matrices = classic.feature_matrices
movie_id_to_index = classic.movie_id_to_index


def catalog_fingerprint():
    """Verify that embeddings correspond to the current catalog."""

    digest = hashlib.sha256()

    for movie_id, overview in zip(
        df["id"],
        df["overview"],
    ):
        digest.update(str(movie_id).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(overview).strip().encode("utf-8"))
        digest.update(b"\0")

    return digest.hexdigest()


def load_embeddings():
    """Load the existing full-catalog embedding cache."""

    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            "Full-catalog embeddings not found. "
            "Run python3 src/hybrid_recommender.py first."
        )

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            "Semantic cache metadata not found. "
            "Run python3 src/hybrid_recommender.py first."
        )

    metadata = json.loads(METADATA_PATH.read_text())

    if metadata.get("model") != MODEL_NAME:
        raise ValueError(
            "The cached embeddings use a different model."
        )

    if metadata.get("rows") != len(df):
        raise ValueError(
            "The cached embeddings do not match "
            "the current movie catalog size."
        )

    if metadata.get("fingerprint") != catalog_fingerprint():
        raise ValueError(
            "The movie catalog has changed since "
            "the semantic embeddings were generated."
        )

    embeddings = np.load(
        EMBEDDINGS_PATH,
        mmap_mode="r",
    )

    if embeddings.shape != (
        len(df),
        metadata.get("dimensions"),
    ):
        raise ValueError(
            "The embedding cache has unexpected dimensions."
        )

    return embeddings


def calculate_feature_scores(query_index):
    """Calculate metadata cosine similarities.

    TF-IDF vectors are L2-normalized, so their dot
    products equal cosine similarity.
    """

    scores = np.zeros(len(df), dtype=np.float64)
    contributions = {}

    for feature, weight in FEATURE_WEIGHTS.items():
        matrix = feature_matrices[feature]

        query_vector = matrix.getrow(query_index)

        if query_vector.nnz == 0:
            values = np.zeros(len(df), dtype=np.float64)
        else:
            product = matrix @ query_vector.T

            if issparse(product):
                values = product.toarray().ravel()
            else:
                values = np.asarray(product).ravel()

            values = values.astype(np.float64) * weight

        scores += values
        contributions[feature] = values

    return scores, contributions


def recommend_movies(
    movie_id,
    top_n=10,
    embeddings=None,
):
    """Return recommendations using the full-catalog hybrid."""

    if movie_id not in movie_id_to_index:
        return None

    if top_n < 1:
        raise ValueError("top_n must be positive.")

    if embeddings is None:
        embeddings = load_embeddings()

    if embeddings.shape[0] != len(df):
        raise ValueError(
            "Embedding and catalog row counts differ."
        )

    query_index = movie_id_to_index[movie_id]

    query_embedding = np.asarray(
        embeddings[query_index],
        dtype=np.float32,
    )

    # Embeddings were normalized when generated.
    semantic = np.asarray(
        embeddings @ query_embedding,
        dtype=np.float64,
    ).ravel()

    metadata, contributions = calculate_feature_scores(
        query_index
    )

    final_scores = (
        SEMANTIC_WEIGHT * semantic
        + METADATA_WEIGHT * metadata
    )

    # Never recommend the movie itself.
    final_scores[query_index] = -np.inf

    count = min(top_n, len(df) - 1)

    if count <= 0:
        return df.iloc[:0].copy()

    indices = np.argpartition(
        final_scores,
        -count,
    )[-count:]

    indices = np.array(
        sorted(
            indices,
            key=lambda i: (
                -float(final_scores[i]),
                int(df.iloc[i]["id"]),
            ),
        ),
        dtype=int,
    )

    results = df.iloc[indices].copy()

    results["semantic_similarity"] = semantic[indices]
    results["metadata_similarity"] = metadata[indices]
    results["similarity"] = final_scores[indices]
    results["final_score"] = final_scores[indices]

    for feature, values in contributions.items():
        results[f"{feature}_contribution"] = values[indices]

    # Share of metadata similarity supported by
    # non-genre features, for explanation only.
    non_genre = sum(
        contributions[feature][indices]
        for feature in [
            "overview",
            "keywords",
            "directors",
            "cast",
        ]
    )

    results["specificity"] = np.clip(
        non_genre / np.maximum(metadata[indices], 1e-10),
        0,
        1,
    )

    return results.reset_index(drop=True)
