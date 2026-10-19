
from pathlib import Path
from collections import defaultdict
import hashlib
import re
import unicodedata

import pandas as pd
from datasets import load_dataset


# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

CATALOG_PATH = ROOT / "data" / "movies_expanded.csv"
OUTPUT_DIR = ROOT / "data" / "review_audit"

IMDB_TITLES_PATH = OUTPUT_DIR / "title.basics.tsv.gz"

DATASET_NAME = "fgiobergia/imdb-id"

MIN_REVIEW_WORDS = 50
MIN_REVIEWS_PER_MOVIE = 5

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def normalize_title(value):
    if pd.isna(value):
        return ""

    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(
        char for char in text
        if not unicodedata.combining(char)
    )

    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)

    return " ".join(text.split())


def count_words(text):
    return len(re.findall(r"\b[\w]+(?:['’-][\w]+)*\b", text))


def review_hash(text):
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


def load_catalog():
    print("Loading CineMatch catalog...")

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

    print(f"Catalog movies: {len(catalog):,}")

    return catalog


# --------------------------------------------------
# Load review dataset
# --------------------------------------------------

def load_reviews():
    print("\nLoading external IMDb reviews...")

    dataset = load_dataset(DATASET_NAME)

    frames = []

    for split_name, split in dataset.items():
        frame = split.to_pandas()
        frame["source_split"] = split_name
        frames.append(frame)

    reviews = pd.concat(frames, ignore_index=True)

    required = {"text", "movie_id"}

    missing = required - set(reviews.columns)

    if missing:
        raise ValueError(
            f"Missing required review columns: {missing}"
        )

    reviews["text"] = reviews["text"].fillna("").astype(str)
    reviews["movie_id"] = reviews["movie_id"].astype(str)

    reviews["word_count"] = reviews["text"].apply(count_words)
    reviews["review_hash"] = reviews["text"].apply(review_hash)

    return reviews


# --------------------------------------------------
# Review quality
# --------------------------------------------------

def audit_review_quality(reviews):
    print("\nReview quality analysis")

    total = len(reviews)

    long_enough = reviews[
        reviews["word_count"] >= MIN_REVIEW_WORDS
    ].copy()

    deduplicated = long_enough.drop_duplicates(
        subset=["movie_id", "review_hash"]
    ).copy()

    print(f"Total reviews: {total:,}")
    print(f"Reviews >= 50 words: {len(long_enough):,}")
    print(
        "Percentage >= 50 words: "
        f"{len(long_enough) / max(total, 1):.1%}"
    )

    print(
        "Duplicates among length-qualified reviews: "
        f"{len(long_enough) - len(deduplicated):,}"
    )

    print(
        "Median review length: "
        f"{reviews['word_count'].median():.0f} words"
    )

    print(
        "Mean review length: "
        f"{reviews['word_count'].mean():.1f} words"
    )

    return deduplicated


# --------------------------------------------------
# Match IMDb IDs to CineMatch
# --------------------------------------------------

def load_imdb_movie_metadata(target_ids):
    print("\nMatching IMDb movie identifiers...")

    if not IMDB_TITLES_PATH.exists():
        raise FileNotFoundError(
            "IMDb title metadata not found.\n"
            f"Expected: {IMDB_TITLES_PATH}\n"
            "Download title.basics.tsv.gz from "
            "https://datasets.imdbws.com/"
        )

    matched_chunks = []

    columns = [
        "tconst",
        "titleType",
        "primaryTitle",
        "originalTitle",
        "startYear"
    ]

    for chunk in pd.read_csv(
        IMDB_TITLES_PATH,
        sep="\t",
        compression="gzip",
        usecols=columns,
        dtype=str,
        chunksize=250_000,
        na_values=["\\N"]
    ):
        selected = chunk[
            chunk["tconst"].isin(target_ids)
            & chunk["titleType"].eq("movie")
        ]

        if not selected.empty:
            matched_chunks.append(selected.copy())

    if not matched_chunks:
        raise ValueError(
            "No matching IMDb movie IDs found in title metadata."
        )

    return pd.concat(matched_chunks, ignore_index=True)


def build_catalog_matches(imdb_movies, catalog):
    catalog_lookup = defaultdict(list)

    for row in catalog.itertuples(index=False):
        key = (row.normalized_title, row.year)
        catalog_lookup[key].append(row.id)

    matches = []

    for movie in imdb_movies.itertuples(index=False):
        if pd.isna(movie.startYear):
            continue

        try:
            year = int(movie.startYear)
        except ValueError:
            continue

        candidate_ids = set()

        for title in [movie.primaryTitle, movie.originalTitle]:
            key = (normalize_title(title), year)
            candidate_ids.update(catalog_lookup.get(key, []))

        # Keep only unambiguous matches.
        if len(candidate_ids) == 1:
            matches.append({
                "movie_id": movie.tconst,
                "id": next(iter(candidate_ids))
            })

    return pd.DataFrame(
        matches,
        columns=["movie_id", "id"]
    ).drop_duplicates(subset=["movie_id"])


# --------------------------------------------------
# Coverage analysis
# --------------------------------------------------

def audit_coverage(reviews, catalog, matches):
    print("\nCineMatch coverage analysis")

    matched_reviews = reviews.merge(
        matches,
        on="movie_id",
        how="inner"
    )

    movie_counts = (
        matched_reviews
        .groupby("id")
        .size()
        .rename("qualifying_review_count")
        .reset_index()
    )

    coverage = catalog.merge(
        movie_counts,
        on="id",
        how="left"
    )

    coverage["qualifying_review_count"] = (
        coverage["qualifying_review_count"]
        .fillna(0)
        .astype(int)
    )

    eligible = coverage[
        coverage["qualifying_review_count"]
        >= MIN_REVIEWS_PER_MOVIE
    ].copy()

    total_movies = len(catalog)

    print(f"Catalog movies: {total_movies:,}")
    print(
        "Movies with >= 1 qualifying review: "
        f"{(coverage['qualifying_review_count'] > 0).sum():,}"
    )

    print(
        f"Movies with >= {MIN_REVIEWS_PER_MOVIE} "
        f"qualifying reviews: {len(eligible):,}"
    )

    print(
        "Eligible catalog coverage: "
        f"{len(eligible) / max(total_movies, 1):.2%}"
    )

    print(
        "Qualifying reviews attached to CineMatch movies: "
        f"{len(matched_reviews):,}"
    )

    coverage.sort_values(
        "qualifying_review_count",
        ascending=False
    ).to_csv(
        OUTPUT_DIR / "movie_review_coverage.csv",
        index=False
    )

    eligible.sort_values(
        "qualifying_review_count",
        ascending=False
    ).to_csv(
        OUTPUT_DIR / "eligible_movies.csv",
        index=False
    )

    matched_reviews.to_csv(
        OUTPUT_DIR / "matched_reviews.csv",
        index=False
    )

    return coverage, eligible


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    catalog = load_catalog()
    reviews = load_reviews()

    qualified_reviews = audit_review_quality(reviews)

    target_ids = set(
        qualified_reviews["movie_id"].dropna()
    )

    imdb_movies = load_imdb_movie_metadata(target_ids)

    matches = build_catalog_matches(
        imdb_movies,
        catalog
    )

    print(
        f"Unambiguous IMDb-to-CineMatch matches: "
        f"{len(matches):,}"
    )

    coverage, eligible = audit_coverage(
        qualified_reviews,
        catalog,
        matches
    )

    print("\nAudit complete.")
    print(f"Results saved to: {OUTPUT_DIR}")

    print("\nTop 10 covered movies:")

    print(
        eligible[
            ["title", "year", "qualifying_review_count"]
        ].head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()
