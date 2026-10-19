
from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "semantic_cache"
CATALOG_PATH = ROOT / "data" / "movies_with_imdb.csv"

EMBEDDINGS_PATH = CACHE_DIR / "full_embeddings.npy"


def inspect_cache():
    print("=" * 60)
    print("CINEMATCH — SEMANTIC CACHE AUDIT")
    print("=" * 60)

    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            f"Embedding file missing: {EMBEDDINGS_PATH}"
        )

    embeddings = np.load(
        EMBEDDINGS_PATH,
        mmap_mode="r",
        allow_pickle=False
    )

    print("\nEmbedding matrix:")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Data type: {embeddings.dtype}")

    if embeddings.ndim != 2:
        raise ValueError(
            "Expected a two-dimensional embedding matrix."
        )

    print("\nCache directory contents:")

    for path in sorted(CACHE_DIR.iterdir()):
        print(
            f"  {path.name} "
            f"({path.stat().st_size / 1024**2:.2f} MB)"
        )

    return embeddings


def inspect_catalog():
    print("\nLoading IMDb-enriched catalog...")

    catalog = pd.read_csv(
        CATALOG_PATH,
        usecols=["id", "title"],
        low_memory=False
    )

    print(f"Catalog rows: {len(catalog):,}")
    print(f"Unique movie IDs: {catalog['id'].nunique():,}")

    print("\nFirst five catalog movies:")

    print(
        catalog.head(5).to_string(
            index=False
        )
    )

    return catalog


def inspect_mapping_files():
    print("\nLooking for embedding ID mappings...")

    candidates = []

    for path in CACHE_DIR.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() in {
            ".csv", ".json", ".npy", ".npz", ".parquet"
        } and path.name != EMBEDDINGS_PATH.name:
            candidates.append(path)

    if not candidates:
        print("No separate mapping files found.")
        return

    for path in candidates:
        print(f"\nFile: {path.name}")

        try:
            suffix = path.suffix.lower()

            if suffix == ".csv":
                sample = pd.read_csv(
                    path,
                    nrows=5
                )

                print("Columns:", list(sample.columns))
                print(sample.to_string(index=False))

            elif suffix == ".json":
                with open(path) as file:
                    data = json.load(file)

                if isinstance(data, dict):
                    print("Keys:", list(data)[:15])
                elif isinstance(data, list):
                    print("List length:", len(data))
                    print("First entries:", data[:5])

            elif suffix == ".npy":
                array = np.load(
                    path,
                    mmap_mode="r",
                    allow_pickle=False
                )

                print("Shape:", array.shape)
                print("Dtype:", array.dtype)
                print("First entries:", array[:5])

            elif suffix == ".npz":
                with np.load(
                    path,
                    allow_pickle=False
                ) as archive:
                    print("Arrays:", archive.files)

                    for name in archive.files:
                        array = archive[name]
                        print(
                            name,
                            array.shape,
                            array.dtype
                        )

            elif suffix == ".parquet":
                sample = pd.read_parquet(path).head(5)
                print("Columns:", list(sample.columns))
                print(sample.to_string(index=False))

        except Exception as error:
            print(
                f"Could not inspect: "
                f"{type(error).__name__}: {error}"
            )


def main():
    embeddings = inspect_cache()
    catalog = inspect_catalog()

    print("\nEmbedding/catalog row comparison:")

    if embeddings.shape[0] == len(catalog):
        print(
            "Row counts match, but row ordering "
            "is NOT yet verified."
        )
    else:
        print(
            "Row counts differ. An explicit "
            "movie-ID mapping is required."
        )

    inspect_mapping_files()

    print("\nAudit complete.")
    print(
        "Do not join embeddings to movies "
        "by row position until alignment "
        "has been established."
    )


if __name__ == "__main__":
    main()
