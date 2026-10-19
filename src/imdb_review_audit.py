
from pathlib import Path
import re
import unicodedata

import pandas as pd
from huggingface_hub import HfApi


# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

CATALOG_PATH = ROOT / "data" / "movies_expanded.csv"

IMDB_METADATA_PATH = (
    ROOT / "data" / "review_audit" / "title.basics.tsv.gz"
)

OUTPUT_DIR = ROOT / "data" / "review_audit"

DATASET_NAME = "Daksh0505/IMDB-Reviews"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Title normalization
# --------------------------------------------------

def normalize_title(value):
    if pd.isna(value):
        return ""

    value = unicodedata.normalize("NFKD", str(value))

    value = "".join(
        char for char in value
        if not unicodedata.combining(char)
    )

    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)

    return " ".join(value.split())


# --------------------------------------------------
# Inspect dataset movie files
# --------------------------------------------------

def get_dataset_movie_ids():
    print("Inspecting Hugging Face dataset files...")

    api = HfApi()

    files = api.list_repo_files(
        repo_id=DATASET_NAME,
        repo_type="dataset"
    )

    print(f"Repository files: {len(files):,}")

    movie_ids = set()
    movie_paths = {}

    for path in files:
        filename = Path(path).name

        # Actual format: tt1234567_reviews.json
        match = re.fullmatch(
            r"(tt\d+)_reviews\.json",
            filename,
            flags=re.IGNORECASE
        )

        if match:
            movie_id = match.group(1).lower()

            movie_ids.add(movie_id)
            movie_paths[movie_id] = path

    print(f"Movie JSON files identified: {len(movie_ids):,}")

    if not movie_ids:
        print("\nExample repository files:")

        for path in files[:15]:
            print(" ", path)

        raise RuntimeError(
            "Could not identify per-movie review files."
        )

    return movie_ids, movie_paths


# --------------------------------------------------
# Load CineMatch catalog
# --------------------------------------------------

def load_catalog():
    print("\nLoading CineMatch catalog...")

    catalog = pd.read_csv(
        CATALOG_PATH,
        usecols=["id", "title", "release_date"]
    )

    catalog["year"] = pd.to_numeric(
        catalog["release_date"].astype(str).str[:4],
        errors="coerce"
    )

    catalog["normalized_title"] = (
        catalog["title"].apply(normalize_title)
    )

    catalog = catalog.dropna(subset=["year"])

    catalog["year"] = catalog["year"].astype(int)

    catalog = catalog.drop_duplicates(subset=["id"])

    print(f"CineMatch movies: {len(catalog):,}")

    return catalog


# --------------------------------------------------
# Read IMDb title metadata
# --------------------------------------------------

def load_imdb_titles(movie_ids):
    print("\nReading IMDb title metadata...")

    if not IMDB_METADATA_PATH.exists():
        raise FileNotFoundError(
            "Missing IMDb title.basics.tsv.gz. "
            "Download it from https://datasets.imdbws.com/"
        )

    matched_chunks = []

    for chunk in pd.read_csv(
        IMDB_METADATA_PATH,
        sep="\t",
        compression="gzip",
        usecols=[
            "tconst",
            "titleType",
            "primaryTitle",
            "originalTitle",
            "startYear"
        ],
        dtype=str,
        na_values=["\\N"],
        chunksize=250_000
    ):
        selected = chunk[
            chunk["tconst"].isin(movie_ids)
            & chunk["titleType"].eq("movie")
        ]

        if not selected.empty:
            matched_chunks.append(selected.copy())

    if not matched_chunks:
        raise RuntimeError(
            "No dataset movie IDs matched IMDb metadata."
        )

    imdb = pd.concat(
        matched_chunks,
        ignore_index=True
    )

    imdb["year"] = pd.to_numeric(
        imdb["startYear"],
        errors="coerce"
    )

    imdb = imdb.dropna(subset=["year"])

    imdb["year"] = imdb["year"].astype(int)

    print(f"IMDb movie records found: {len(imdb):,}")

    return imdb


# --------------------------------------------------
# Match IMDb movies to CineMatch
# --------------------------------------------------

def match_movies(catalog, imdb, movie_paths):
    print("\nMatching movies to CineMatch...")

    catalog_keys = catalog[
        ["id", "title", "year", "normalized_title"]
    ].copy()

    candidate_matches = []

    for title_column in ["primaryTitle", "originalTitle"]:
        imdb_keys = imdb[
            ["tconst", "year", title_column]
        ].copy()

        imdb_keys["normalized_title"] = (
            imdb_keys[title_column].apply(normalize_title)
        )

        joined = imdb_keys.merge(
            catalog_keys,
            on=["normalized_title", "year"],
            how="inner"
        )

        candidate_matches.append(
            joined[["tconst", "id", "title", "year"]]
        )

    matches = pd.concat(
        candidate_matches,
        ignore_index=True
    ).drop_duplicates()

    # Exclude ambiguous title matches.
    unique_counts = (
        matches.groupby("tconst")["id"].nunique()
    )

    unambiguous_ids = unique_counts[
        unique_counts == 1
    ].index

    matches = matches[
        matches["tconst"].isin(unambiguous_ids)
    ].drop_duplicates(subset=["tconst"])

    matches["dataset_file"] = (
        matches["tconst"].map(movie_paths)
    )

    matches.to_csv(
        OUTPUT_DIR / "imdb_dataset_matches.csv",
        index=False
    )

    return matches


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    movie_ids, movie_paths = get_dataset_movie_ids()

    catalog = load_catalog()

    imdb = load_imdb_titles(movie_ids)

    matches = match_movies(
        catalog,
        imdb,
        movie_paths
    )

    print("\n" + "=" * 50)
    print("IMDb DATASET COVERAGE — STAGE 1")
    print("=" * 50)

    print(f"Dataset movie IDs: {len(movie_ids):,}")

    print(f"CineMatch catalog movies: {len(catalog):,}")

    print(f"Matched movies: {len(matches):,}")

    print(
        "Catalog coverage: "
        f"{len(matches) / len(catalog):.3%}"
    )

    if not matches.empty:
        print("\nMatched titles:")

        print(
            matches[
                ["title", "year", "tconst"]
            ].sort_values(
                ["year", "title"]
            ).to_string(index=False)
        )

    print(
        "\nResults saved to:",
        OUTPUT_DIR / "imdb_dataset_matches.csv"
    )

    print(
        "\nNext stage: download reviews only for "
        "matched movies and apply the 50-word / "
        "5-review thresholds."
    )


if __name__ == "__main__":
    main()
