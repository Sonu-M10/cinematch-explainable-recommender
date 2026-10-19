
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


# ==================================================
# CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ML_DIR = DATA_DIR / "movielens"
CACHE_DIR = DATA_DIR / "semantic_cache"

CATALOG_PATH = DATA_DIR / "movies_expanded.csv"
INPUT_PATH = ML_DIR / "lgbm_training_pairs_v2.csv.gz"
OUTPUT_PATH = ML_DIR / "lgbm_training_pairs_v3.csv.gz"

EMBEDDINGS_PATH = CACHE_DIR / "full_embeddings.npy"
METADATA_PATH = (
    CACHE_DIR / "full_embeddings_metadata.json"
)


# ==================================================
# REPRODUCE HYBRID CATALOG PREPARATION
# ==================================================

def load_catalog():
    print("Loading original movie catalog...")

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
        CATALOG_PATH,
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

    print(f"Catalog movies: {len(df):,}")

    return df


# ==================================================
# VERIFY EMBEDDING ALIGNMENT
# ==================================================

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


def load_verified_embeddings(df):
    print("\nVerifying semantic embedding cache...")

    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            f"Missing embeddings: {EMBEDDINGS_PATH}"
        )

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing metadata: {METADATA_PATH}"
        )

    metadata = json.loads(
        METADATA_PATH.read_text()
    )

    fingerprint = catalog_fingerprint(df)

    if metadata.get("fingerprint") != fingerprint:
        raise ValueError(
            "Embedding fingerprint mismatch. "
            "The cache does not correspond to "
            "this catalog ordering and content."
        )

    if metadata.get("rows") != len(df):
        raise ValueError(
            "Embedding row count mismatch."
        )

    embeddings = np.load(
        EMBEDDINGS_PATH,
        mmap_mode="r",
        allow_pickle=False,
    )

    expected_shape = (
        len(df),
        metadata.get("dimensions"),
    )

    if embeddings.shape != expected_shape:
        raise ValueError(
            "Embedding dimensions do not match metadata."
        )

    if embeddings.shape[1] != 384:
        raise ValueError(
            "Unexpected semantic embedding dimension."
        )

    print("Catalog fingerprint: MATCH")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Model: {metadata.get('model')}")
    print("Embedding alignment verified.")

    return embeddings


# ==================================================
# CALCULATE SEMANTIC SIMILARITY
# ==================================================

def add_semantic_similarity(pairs, df, embeddings):
    print("\nMapping movie IDs to embedding rows...")

    id_to_position = {
        int(movie_id): index
        for index, movie_id in enumerate(df["id"])
    }

    anchor_positions = pairs[
        "anchor_id"
    ].map(id_to_position)

    candidate_positions = pairs[
        "candidate_id"
    ].map(id_to_position)

    if (
        anchor_positions.isna().any()
        or candidate_positions.isna().any()
    ):
        raise ValueError(
            "Some training movies are missing "
            "from the embedding catalog."
        )

    anchor_positions = (
        anchor_positions.to_numpy(dtype=np.int64)
    )

    candidate_positions = (
        candidate_positions.to_numpy(dtype=np.int64)
    )

    print("Calculating semantic cosine similarity...")

    anchor_vectors = np.asarray(
        embeddings[anchor_positions],
        dtype=np.float32,
    )

    candidate_vectors = np.asarray(
        embeddings[candidate_positions],
        dtype=np.float32,
    )

    dot_products = np.einsum(
        "ij,ij->i",
        anchor_vectors,
        candidate_vectors,
    )

    anchor_norms = np.linalg.norm(
        anchor_vectors,
        axis=1,
    )

    candidate_norms = np.linalg.norm(
        candidate_vectors,
        axis=1,
    )

    denominators = (
        anchor_norms * candidate_norms
    )

    similarities = np.divide(
        dot_products,
        denominators,
        out=np.full(
            len(pairs),
            np.nan,
            dtype=np.float32,
        ),
        where=denominators > 0,
    )

    pairs["semantic_similarity"] = similarities

    return pairs


# ==================================================
# MAIN
# ==================================================

def main():
    print("=" * 60)
    print("CINEMATCH — LIGHTGBM V3 FEATURES")
    print("=" * 60)

    pairs = pd.read_csv(INPUT_PATH)

    print(f"\nTraining rows: {len(pairs):,}")

    catalog = load_catalog()

    embeddings = load_verified_embeddings(
        catalog
    )

    result = add_semantic_similarity(
        pairs,
        catalog,
        embeddings,
    )

    if len(result) != len(pairs):
        raise RuntimeError(
            "Training row count changed."
        )

    if not np.isfinite(
        result["semantic_similarity"]
        .dropna()
        .to_numpy()
    ).all():
        raise RuntimeError(
            "Non-finite semantic similarities detected."
        )

    result.to_csv(
        OUTPUT_PATH,
        index=False,
        compression="gzip",
    )

    scores = result["semantic_similarity"]

    print("\n" + "=" * 60)
    print("SEMANTIC FEATURE REPORT")
    print("=" * 60)

    print(f"Training rows: {len(result):,}")
    print(
        "Ranking groups: "
        f"{result['userId'].nunique():,}"
    )

    print(
        "Semantic coverage: "
        f"{scores.notna().mean():.2%}"
    )

    print(
        f"Mean similarity: {scores.mean():.4f}"
    )

    print(
        f"Minimum similarity: {scores.min():.4f}"
    )

    print(
        f"Maximum similarity: {scores.max():.4f}"
    )

    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
