
from pathlib import Path
import ast
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ML_DIR = DATA_DIR / "movielens"

PAIRS_PATH = ML_DIR / "lgbm_training_pairs.csv.gz"
CATALOG_PATH = DATA_DIR / "movies_with_imdb.csv"
OUTPUT_PATH = ML_DIR / "lgbm_training_pairs_v2.csv.gz"


def parse_genres(value):
    if pd.isna(value):
        return set()

    text = str(value).strip()

    try:
        parsed = ast.literal_eval(text)

        if isinstance(parsed, list):
            genres = []

            for item in parsed:
                if isinstance(item, dict):
                    genres.append(str(item.get("name", "")))
                else:
                    genres.append(str(item))

            return {
                genre.strip().lower()
                for genre in genres
                if genre.strip()
            }

    except (ValueError, SyntaxError):
        pass

    return {
        part.strip().lower()
        for part in re.split(r"[,|]", text)
        if part.strip()
    }


def genre_jaccard(first, second):
    if not first or not second:
        return np.nan

    union = first | second

    if not union:
        return np.nan

    return len(first & second) / len(union)


def load_data():
    print("Loading supervised ranking pairs...")

    pairs = pd.read_csv(PAIRS_PATH)

    print(f"Ranking rows: {len(pairs):,}")

    print("Loading IMDb-enriched movie catalog...")

    catalog = pd.read_csv(
        CATALOG_PATH,
        low_memory=False
    )

    required = [
        "id",
        "title",
        "overview",
        "genres",
        "release_date",
        "imdb_rating",
        "imdb_votes"
    ]

    missing = [
        column for column in required
        if column not in catalog.columns
    ]

    if missing:
        raise ValueError(
            f"Missing catalog columns: {missing}"
        )

    catalog = catalog[required].copy()

    catalog["id"] = pd.to_numeric(
        catalog["id"],
        errors="coerce"
    )

    catalog = catalog.dropna(subset=["id"])
    catalog["id"] = catalog["id"].astype("int64")

    if catalog["id"].duplicated().any():
        raise ValueError(
            "Duplicate movie IDs in catalog."
        )

    return pairs, catalog


def prepare_catalog(catalog, movie_ids):
    selected = catalog[
        catalog["id"].isin(movie_ids)
    ].copy()

    selected["overview"] = (
        selected["overview"]
        .fillna("")
        .astype(str)
    )

    selected["genre_set"] = (
        selected["genres"].apply(parse_genres)
    )

    selected["year"] = pd.to_datetime(
        selected["release_date"],
        errors="coerce"
    ).dt.year

    selected["imdb_rating"] = pd.to_numeric(
        selected["imdb_rating"],
        errors="coerce"
    )

    selected["imdb_votes"] = pd.to_numeric(
        selected["imdb_votes"],
        errors="coerce"
    )

    selected["imdb_log_votes"] = np.log1p(
        selected["imdb_votes"].clip(lower=0)
    )

    return selected.set_index("id")


def build_features(pairs, catalog):
    print("\nPreparing movie metadata...")

    movie_ids = set(pairs["anchor_id"]) | set(
        pairs["candidate_id"]
    )

    movies = prepare_catalog(
        catalog,
        movie_ids
    )

    missing_ids = movie_ids - set(movies.index)

    if missing_ids:
        raise ValueError(
            f"{len(missing_ids)} movie IDs missing "
            "from the catalog."
        )

    print(
        f"Unique movies represented: {len(movie_ids):,}"
    )

    # Fit TF-IDF on the movie descriptions.
    # No supervised labels are used in fitting.
    print("\nBuilding TF-IDF description vectors...")

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=30000,
        min_df=2,
        ngram_range=(1, 2),
        dtype=np.float32
    )

    descriptions = movies["overview"]

    tfidf = vectorizer.fit_transform(
        descriptions
    )

    movie_positions = {
        movie_id: position
        for position, movie_id in enumerate(
            movies.index
        )
    }

    anchor_positions = (
        pairs["anchor_id"]
        .map(movie_positions)
        .to_numpy()
    )

    candidate_positions = (
        pairs["candidate_id"]
        .map(movie_positions)
        .to_numpy()
    )

    print("Calculating pairwise TF-IDF similarity...")

    # TF-IDF vectors are L2-normalized by default.
    # Their dot product is cosine similarity.
    anchor_vectors = tfidf[anchor_positions]
    candidate_vectors = tfidf[candidate_positions]

    pairs["tfidf_similarity"] = np.asarray(
        anchor_vectors.multiply(
            candidate_vectors
        ).sum(axis=1)
    ).ravel()

    print("Calculating genre similarity...")

    genres = movies["genre_set"].to_dict()

    pairs["genre_overlap"] = [
        genre_jaccard(
            genres[anchor],
            genres[candidate]
        )
        for anchor, candidate in zip(
            pairs["anchor_id"],
            pairs["candidate_id"]
        )
    ]

    print("Adding release-year features...")

    years = movies["year"]

    anchor_years = (
        pairs["anchor_id"].map(years)
    )

    candidate_years = (
        pairs["candidate_id"].map(years)
    )

    pairs["year_difference"] = (
        anchor_years - candidate_years
    ).abs()

    pairs["candidate_year"] = candidate_years

    print("Adding IMDb features...")

    pairs["candidate_imdb_rating"] = (
        pairs["candidate_id"].map(
            movies["imdb_rating"]
        )
    )

    pairs["candidate_imdb_log_votes"] = (
        pairs["candidate_id"].map(
            movies["imdb_log_votes"]
        )
    )

    pairs["anchor_imdb_rating"] = (
        pairs["anchor_id"].map(
            movies["imdb_rating"]
        )
    )

    return pairs


def main():
    pairs, catalog = load_data()

    result = build_features(
        pairs,
        catalog
    )

    if len(result) != len(pairs):
        raise RuntimeError(
            "Feature generation changed row count."
        )

    if result.duplicated(
        ["userId", "candidate_id"]
    ).any():
        raise RuntimeError(
            "Duplicate ranking examples detected."
        )

    result.to_csv(
        OUTPUT_PATH,
        index=False,
        compression="gzip"
    )

    new_features = [
        "tfidf_similarity",
        "genre_overlap",
        "year_difference",
        "candidate_year",
        "candidate_imdb_rating",
        "candidate_imdb_log_votes",
        "anchor_imdb_rating"
    ]

    print("\n" + "=" * 60)
    print("CINEMATCH — LIGHTGBM V2 FEATURES")
    print("=" * 60)

    print(f"Rows: {len(result):,}")
    print(
        f"Ranking groups: "
        f"{result['userId'].nunique():,}"
    )

    print("\nFeature availability:")

    for feature in new_features:
        coverage = result[feature].notna().mean()

        print(
            f"  {feature}: {coverage:.2%}"
        )

    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
