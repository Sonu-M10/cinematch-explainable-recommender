
from pathlib import Path
import re
import unicodedata

import pandas as pd
import requests


# ==================================================
# CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parents[1]

CATALOG_PATH = ROOT / "data" / "movies_expanded.csv"

IMDB_DIR = ROOT / "data" / "imdb"
IMDB_DIR.mkdir(parents=True, exist_ok=True)

BASICS_PATH = (
    ROOT / "data" / "review_audit" / "title.basics.tsv.gz"
)

RATINGS_PATH = IMDB_DIR / "title.ratings.tsv.gz"

OUTPUT_PATH = ROOT / "data" / "movies_with_imdb.csv"
MATCH_REPORT_PATH = IMDB_DIR / "imdb_matches.csv"

RATINGS_URL = (
    "https://datasets.imdbws.com/title.ratings.tsv.gz"
)


# ==================================================
# HELPERS
# ==================================================

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


def download_ratings():
    if RATINGS_PATH.exists():
        print("Using existing IMDb ratings file.")
        return

    print("Downloading IMDb ratings...")

    temporary_path = RATINGS_PATH.with_suffix(
        RATINGS_PATH.suffix + ".part"
    )

    try:
        with requests.get(
            RATINGS_URL,
            stream=True,
            timeout=60
        ) as response:
            response.raise_for_status()

            with open(temporary_path, "wb") as file:
                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):
                    if chunk:
                        file.write(chunk)

        temporary_path.replace(RATINGS_PATH)

    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    print("IMDb ratings downloaded.")


# ==================================================
# LOAD CINEMATCH CATALOG
# ==================================================

def load_catalog():
    print("\nLoading CineMatch catalog...")

    catalog = pd.read_csv(CATALOG_PATH)

    required_columns = {
        "id", "title", "release_date"
    }

    missing = required_columns - set(catalog.columns)

    if missing:
        raise ValueError(
            f"Missing catalog columns: {missing}"
        )

    catalog["year"] = pd.to_numeric(
        catalog["release_date"].astype(str).str[:4],
        errors="coerce"
    )

    catalog["normalized_title"] = (
        catalog["title"].apply(normalize_title)
    )

    print(f"Catalog movies: {len(catalog):,}")

    return catalog


# ==================================================
# LOAD IMDb METADATA
# ==================================================

def load_imdb_metadata():
    print("\nLoading IMDb movie metadata...")

    if not BASICS_PATH.exists():
        raise FileNotFoundError(
            f"IMDb metadata not found: {BASICS_PATH}"
        )

    chunks = []

    for chunk in pd.read_csv(
        BASICS_PATH,
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
        movies = chunk[
            chunk["titleType"] == "movie"
        ].copy()

        if not movies.empty:
            chunks.append(movies)

    imdb = pd.concat(
        chunks,
        ignore_index=True
    )

    imdb["year"] = pd.to_numeric(
        imdb["startYear"],
        errors="coerce"
    )

    imdb = imdb.dropna(subset=["year"])
    imdb["year"] = imdb["year"].astype(int)

    print(f"IMDb movies: {len(imdb):,}")

    return imdb


# ==================================================
# LOAD IMDb RATINGS
# ==================================================

def load_imdb_ratings():
    print("\nLoading IMDb ratings...")

    ratings = pd.read_csv(
        RATINGS_PATH,
        sep="\t",
        compression="gzip",
        dtype={"tconst": str}
    )

    ratings["averageRating"] = pd.to_numeric(
        ratings["averageRating"],
        errors="coerce"
    )

    ratings["numVotes"] = pd.to_numeric(
        ratings["numVotes"],
        errors="coerce"
    )

    ratings = ratings.dropna(
        subset=["averageRating", "numVotes"]
    )

    print(f"Rated IMDb titles: {len(ratings):,}")

    return ratings


# ==================================================
# MATCH MOVIES
# ==================================================

def match_movies(catalog, imdb, ratings):
    print("\nMatching IMDb titles to CineMatch...")

    # Restrict to rated movies first.
    imdb = imdb.merge(
        ratings,
        on="tconst",
        how="inner",
        validate="one_to_one"
    )

    # Catalog keys used for matching.
    catalog_keys = catalog[
        ["id", "title", "year", "normalized_title"]
    ].dropna(subset=["year"]).copy()

    catalog_keys = catalog_keys[
        catalog_keys["normalized_title"] != ""
    ]

    candidate_matches = []

    for title_column in [
        "primaryTitle",
        "originalTitle"
    ]:
        imdb_keys = imdb[
            [
                "tconst",
                "year",
                title_column,
                "averageRating",
                "numVotes"
            ]
        ].copy()

        imdb_keys["normalized_title"] = (
            imdb_keys[title_column].apply(
                normalize_title
            )
        )

        imdb_keys = imdb_keys[
            imdb_keys["normalized_title"] != ""
        ]

        joined = catalog_keys.merge(
            imdb_keys,
            on=["normalized_title", "year"],
            how="inner"
        )

        candidate_matches.append(
            joined[
                [
                    "id",
                    "title",
                    "year",
                    "tconst",
                    "averageRating",
                    "numVotes"
                ]
            ]
        )

    matches = pd.concat(
        candidate_matches,
        ignore_index=True
    ).drop_duplicates(
        subset=["id", "tconst"]
    )

    print(
        "Candidate movie pairs: "
        f"{len(matches):,}"
    )

    # A CineMatch ID must map to exactly one IMDb ID.
    tmdb_counts = matches.groupby(
        "id"
    )["tconst"].nunique()

    valid_tmdb_ids = tmdb_counts[
        tmdb_counts == 1
    ].index

    # An IMDb ID must also map to exactly one TMDB ID.
    imdb_counts = matches.groupby(
        "tconst"
    )["id"].nunique()

    valid_imdb_ids = imdb_counts[
        imdb_counts == 1
    ].index

    matches = matches[
        matches["id"].isin(valid_tmdb_ids)
        & matches["tconst"].isin(valid_imdb_ids)
    ].copy()

    matches = matches.drop_duplicates(
        subset=["id"]
    )

    matches = matches.rename(
        columns={
            "tconst": "imdb_id",
            "averageRating": "imdb_rating",
            "numVotes": "imdb_votes"
        }
    )

    print(
        "Unambiguous rated matches: "
        f"{len(matches):,}"
    )

    return matches


# ==================================================
# ENRICH CATALOG
# ==================================================

def enrich_catalog(catalog, matches):
    print("\nCreating enriched catalog...")

    rating_columns = matches[
        [
            "id",
            "imdb_id",
            "imdb_rating",
            "imdb_votes"
        ]
    ]

    enriched = catalog.merge(
        rating_columns,
        on="id",
        how="left",
        validate="many_to_one"
    )

    # Remove temporary matching columns.
    enriched = enriched.drop(
        columns=["year", "normalized_title"]
    )

    if len(enriched) != len(catalog):
        raise RuntimeError(
            "Catalog row count changed during enrichment."
        )

    enriched.to_csv(
        OUTPUT_PATH,
        index=False
    )

    matches.to_csv(
        MATCH_REPORT_PATH,
        index=False
    )

    return enriched


# ==================================================
# REPORT
# ==================================================

def print_report(enriched, matches):
    total = len(enriched)

    matched = enriched["imdb_rating"].notna().sum()

    print("\n" + "=" * 55)
    print("CINEMATCH IMDb RATINGS ENRICHMENT")
    print("=" * 55)

    print(f"Total catalog movies: {total:,}")
    print(f"Movies with IMDb ratings: {matched:,}")

    print(
        f"Coverage: {matched / total:.2%}"
    )

    if matched:
        print(
            "Average IMDb rating: "
            f"{enriched['imdb_rating'].mean():.2f}"
        )

        print(
            "Median IMDb votes: "
            f"{enriched['imdb_votes'].median():,.0f}"
        )

        print("\nExample matched movies:")

        examples = matches.sort_values(
            "imdb_votes",
            ascending=False
        ).head(15)

        print(
            examples[
                [
                    "title",
                    "year",
                    "imdb_rating",
                    "imdb_votes"
                ]
            ].to_string(index=False)
        )

    print(f"\nEnriched catalog: {OUTPUT_PATH}")
    print(f"Match report: {MATCH_REPORT_PATH}")

    print("\nOriginal catalog was not modified.")


# ==================================================
# MAIN
# ==================================================

def main():
    download_ratings()

    catalog = load_catalog()
    imdb = load_imdb_metadata()
    ratings = load_imdb_ratings()

    matches = match_movies(
        catalog,
        imdb,
        ratings
    )

    enriched = enrich_catalog(
        catalog,
        matches
    )

    print_report(
        enriched,
        matches
    )


if __name__ == "__main__":
    main()
