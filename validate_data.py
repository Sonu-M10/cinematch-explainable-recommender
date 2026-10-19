
"""Validate the data artifacts required to run CineMatch.

Examples:
    python3 setup_data.py
    python3 setup_data.py --engine classic
    python3 setup_data.py --engine hybrid
    python3 setup_data.py --engine experimental
    python3 setup_data.py --engine ensemble

This script validates existing files only.
It does not download, modify, or regenerate datasets.
"""

import argparse
import hashlib
import json
from pathlib import Path
from src.artifact_integrity import verify_manifest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent

ARTIFACTS = {
    "catalog": Path("data/movies_expanded.csv"),
    "imdb_catalog": Path("data/movies_with_imdb.csv"),
    "embeddings": Path("data/semantic_cache/full_embeddings.npy"),
    "embedding_metadata": Path(
        "data/semantic_cache/full_embeddings_metadata.json"
    ),
    "affinity": Path("data/movielens/lgbm_history_affinity.csv"),
    "universe": Path("data/movielens/lgbm_history_universe.csv"),
    "training_pairs": Path(
        "data/movielens/lgbm_training_pairs_v3.csv.gz"
    ),
    "ranker": Path("models/lgbm_ranker_v4_experimental.txt"),
}

ENGINE_REQUIREMENTS = {
    "classic": ["catalog"],
    "hybrid": [
        "catalog",
        "embeddings",
        "embedding_metadata",
    ],
    "experimental": [
        "catalog",
        "imdb_catalog",
        "embeddings",
        "embedding_metadata",
        "affinity",
        "universe",
        "training_pairs",
        "ranker",
    ],
}

ENGINE_REQUIREMENTS["ensemble"] = list(
    dict.fromkeys(
        requirement
        for engine in ("classic", "hybrid", "experimental")
        for requirement in ENGINE_REQUIREMENTS[engine]
    )
)


def check_artifact(name):
    """Check that an artifact exists and is not empty."""
    path = ROOT / ARTIFACTS[name]

    if not path.is_file():
        print(f"  [MISSING] {ARTIFACTS[name]}")
        return False

    size = path.stat().st_size

    if size == 0:
        print(f"  [EMPTY]   {ARTIFACTS[name]}")
        return False

    print(
        f"  [OK]      {ARTIFACTS[name]} "
        f"({size / (1024 ** 2):.2f} MB)"
    )
    return True


def catalog_fingerprint(df):
    """Hash catalog IDs and overviews in their existing row order.

    This matches the fingerprint algorithm in
    src/hybrid_recommender.py.
    """
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


def validate_catalog():
    """Validate catalog IDs and overview data."""
    print("\nValidating movie catalog")
    print("-" * 60)

    path = ROOT / ARTIFACTS["catalog"]

    try:
        df = pd.read_csv(
            path,
            usecols=["id", "title", "overview"],
            low_memory=False,
        )

        if df.empty:
            raise ValueError("Catalog has no rows.")

        if df["id"].isna().any():
            raise ValueError("Catalog contains missing movie IDs.")

        if df["id"].duplicated().any():
            raise ValueError("Catalog contains duplicate movie IDs.")

        if df["overview"].isna().any():
            raise ValueError("Catalog contains missing overviews.")

        print(f"  [OK] Movie rows: {len(df):,}")
        print(f"  [OK] Unique IDs: {df['id'].nunique():,}")
        print("  [OK] No missing overviews")

        return df

    except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
        print(f"  [FAIL] Catalog validation: {exc}")
        return None


def validate_semantic_cache(catalog):
    """Validate embedding shape, metadata and catalog fingerprint."""
    print("\nValidating semantic cache")
    print("-" * 60)

    embeddings_path = ROOT / ARTIFACTS["embeddings"]
    metadata_path = ROOT / ARTIFACTS["embedding_metadata"]

    try:
        embeddings = np.load(
            embeddings_path,
            mmap_mode="r",
            allow_pickle=False,
        )

        metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )

        if embeddings.ndim != 2:
            raise ValueError(
                f"Expected a 2D embedding matrix, got {embeddings.ndim}D."
            )

        expected_rows = len(catalog)
        actual_rows, actual_dimensions = embeddings.shape

        if actual_rows != expected_rows:
            raise ValueError(
                f"Embedding rows ({actual_rows:,}) do not match "
                f"catalog rows ({expected_rows:,})."
            )

        if metadata.get("rows") != actual_rows:
            raise ValueError(
                "Metadata row count does not match embeddings."
            )

        if metadata.get("dimensions") != actual_dimensions:
            raise ValueError(
                "Metadata dimensions do not match embeddings."
            )

        if metadata.get("model") != (
            "sentence-transformers/all-MiniLM-L6-v2"
        ):
            raise ValueError(
                "Unexpected embedding model in cache metadata."
            )

        actual_fingerprint = catalog_fingerprint(catalog)
        expected_fingerprint = metadata.get("fingerprint")

        if actual_fingerprint != expected_fingerprint:
            raise ValueError(
                "Catalog fingerprint does not match the semantic cache."
            )

        print(f"  [OK] Embedding rows: {actual_rows:,}")
        print(f"  [OK] Dimensions: {actual_dimensions}")
        print(f"  [OK] Data type: {embeddings.dtype}")
        print(f"  [OK] Model: {metadata['model']}")
        print("  [OK] Catalog fingerprint matches")

        return True

    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ) as exc:
        print(f"  [FAIL] Semantic cache validation: {exc}")
        return False


def check_engine(engine):
    """Check required files and validate applicable data structures."""
    print(f"\n{'=' * 60}")
    print(f"Checking {engine.upper()} engine")
    print("=" * 60)

    requirements = ENGINE_REQUIREMENTS[engine]

    file_results = {
        name: check_artifact(name)
        for name in requirements
    }

    if not all(file_results.values()):
        print(f"\n{engine.upper()}: Missing or empty artifacts.")
        return False

    catalog = validate_catalog()

    if catalog is None:
        return False

    if "embeddings" in requirements:
        if not validate_semantic_cache(catalog):
            return False

    print(f"\n{engine.upper()}: Validation passed.")
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Validate CineMatch runtime artifacts."
    )

    parser.add_argument(
        "--engine",
        choices=[
            "all",
            "classic",
            "hybrid",
            "experimental",
            "ensemble",
        ],
        default="all",
        help="Engine to validate (default: all).",
    )

    parser.add_argument(
        "--verify-checksums",
        action="store_true",
        help="Verify all artifacts against data_manifest.json.",
    )

    args = parser.parse_args()

    engines = (
        ["classic", "hybrid", "experimental", "ensemble"]
        if args.engine == "all"
        else [args.engine]
    )

    print("=" * 60)
    print("CineMatch Data Validation")
    print("=" * 60)

    results = [check_engine(engine) for engine in engines]

    if args.verify_checksums:
        results.append(verify_manifest())

    if all(results):
        print("\nSUCCESS: All selected validations passed.")
        return 0

    print("\nFAILED: One or more validations did not pass.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
