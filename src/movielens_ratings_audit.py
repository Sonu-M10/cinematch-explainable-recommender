
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd


# ==================================================
# CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
ML_DIR = DATA_DIR / "movielens"

ZIP_PATH = ML_DIR / "ml-32m.zip"

MATCHES_PATH = (
    ML_DIR / "movielens_cinematch_matches.csv"
)

CATALOG_PATH = DATA_DIR / "movies_with_imdb.csv"

MOVIE_REPORT_PATH = (
    ML_DIR / "matched_movie_rating_counts.csv"
)

SUMMARY_PATH = ML_DIR / "ratings_audit_summary.csv"

CHUNK_SIZE = 500_000

# These are exploratory thresholds, not final
# model-training requirements.
MIN_MOVIE_RATINGS = 20
MIN_USER_RATINGS = 5


# ==================================================
# LOAD MATCHED MOVIES
# ==================================================

def load_matches():
    print("Loading MovieLens-to-CineMatch matches...")

    matches = pd.read_csv(
        MATCHES_PATH,
        usecols=[
            "movieId",
            "cinematch_id",
            "cinematch_title"
        ]
    )

    if matches["movieId"].duplicated().any():
        raise ValueError(
            "Duplicate MovieLens IDs in match report."
        )

    if matches["cinematch_id"].duplicated().any():
        raise ValueError(
            "Duplicate CineMatch IDs in match report."
        )

    print(f"Matched movies: {len(matches):,}")

    return matches


# ==================================================
# STREAM RATINGS
# ==================================================

def audit_ratings(matches):
    print("\nStreaming MovieLens ratings...")

    matched_ids = set(matches["movieId"])

    movie_counts = {}
    user_counts = {}

    total_ratings = 0
    matched_ratings = 0
    rating_sum = 0.0

    rating_distribution = {
        value: 0
        for value in np.arange(0.5, 5.1, 0.5)
    }

    with ZipFile(ZIP_PATH) as archive:
        with archive.open("ml-32m/ratings.csv") as file:
            chunks = pd.read_csv(
                file,
                usecols=[
                    "userId",
                    "movieId",
                    "rating"
                ],
                dtype={
                    "userId": "int32",
                    "movieId": "int32",
                    "rating": "float32"
                },
                chunksize=CHUNK_SIZE
            )

            for chunk_number, chunk in enumerate(
                chunks,
                start=1
            ):
                total_ratings += len(chunk)

                selected = chunk[
                    chunk["movieId"].isin(matched_ids)
                ]

                matched_ratings += len(selected)

                rating_sum += float(
                    selected["rating"].sum()
                )

                # Movie-level interaction counts.
                counts_by_movie = (
                    selected.groupby("movieId").size()
                )

                for movie_id, count in counts_by_movie.items():
                    movie_counts[movie_id] = (
                        movie_counts.get(movie_id, 0)
                        + int(count)
                    )

                # User-level interaction counts.
                counts_by_user = (
                    selected.groupby("userId").size()
                )

                for user_id, count in counts_by_user.items():
                    user_counts[user_id] = (
                        user_counts.get(user_id, 0)
                        + int(count)
                    )

                # Rating distribution.
                distribution = (
                    selected["rating"].value_counts()
                )

                for rating, count in distribution.items():
                    rating_distribution[float(rating)] += (
                        int(count)
                    )

                print(
                    f"Chunk {chunk_number}: "
                    f"{total_ratings:,} total ratings, "
                    f"{matched_ratings:,} matched ratings"
                )

    return {
        "total_ratings": total_ratings,
        "matched_ratings": matched_ratings,
        "rating_sum": rating_sum,
        "movie_counts": movie_counts,
        "user_counts": user_counts,
        "rating_distribution": rating_distribution
    }


# ==================================================
# MOVIE REPORT
# ==================================================

def create_movie_report(matches, results):
    counts = pd.Series(
        results["movie_counts"],
        name="rating_count",
        dtype="int64"
    )

    counts.index.name = "movieId"

    report = matches.merge(
        counts.reset_index(),
        on="movieId",
        how="left"
    )

    report["rating_count"] = (
        report["rating_count"]
        .fillna(0)
        .astype(int)
    )

    report = report.sort_values(
        "rating_count",
        ascending=False
    )

    report.to_csv(
        MOVIE_REPORT_PATH,
        index=False
    )

    return report


# ==================================================
# YEAR DISTRIBUTION
# ==================================================

def audit_years(matches):
    catalog = pd.read_csv(
        CATALOG_PATH,
        usecols=["id", "release_date"]
    )

    catalog["year"] = pd.to_numeric(
        catalog["release_date"].astype(str).str[:4],
        errors="coerce"
    )

    years = matches.merge(
        catalog[["id", "year"]],
        left_on="cinematch_id",
        right_on="id",
        how="left"
    )

    before_2010 = (
        years["year"].notna()
        & (years["year"] < 2010)
    ).sum()

    from_2010 = (
        years["year"].notna()
        & (years["year"] >= 2010)
    ).sum()

    print("\nMatched catalog release years:")
    print(f"Before 2010: {before_2010:,}")
    print(f"2010 onward: {from_2010:,}")
    print(
        f"Missing release year: "
        f"{years['year'].isna().sum():,}"
    )


# ==================================================
# REPORT
# ==================================================

def print_report(matches, results, movie_report):
    total = results["total_ratings"]
    matched = results["matched_ratings"]

    user_counts = pd.Series(
        results["user_counts"],
        dtype="int64"
    )

    movie_counts = movie_report["rating_count"]

    movies_with_ratings = int(
        (movie_counts > 0).sum()
    )

    movies_with_20 = int(
        (movie_counts >= MIN_MOVIE_RATINGS).sum()
    )

    users_with_5 = int(
        (user_counts >= MIN_USER_RATINGS).sum()
    )

    users_with_2 = int(
        (user_counts >= 2).sum()
    )

    average_rating = (
        results["rating_sum"] / matched
        if matched else float("nan")
    )

    summary = {
        "total_ratings": total,
        "matched_ratings": matched,
        "matched_rating_percentage": (
            matched / total * 100 if total else 0
        ),
        "matched_movies": len(matches),
        "movies_with_ratings": movies_with_ratings,
        "movies_with_20_ratings": movies_with_20,
        "users_with_matched_ratings": len(user_counts),
        "users_with_2_matched_ratings": users_with_2,
        "users_with_5_matched_ratings": users_with_5,
        "average_matched_rating": average_rating
    }

    pd.DataFrame(
        [summary]
    ).to_csv(
        SUMMARY_PATH,
        index=False
    )

    print("\n" + "=" * 60)
    print("MOVIELENS 32M — RATINGS COVERAGE AUDIT")
    print("=" * 60)

    print(f"Total MovieLens ratings: {total:,}")
    print(f"Ratings on matched movies: {matched:,}")

    print(
        "Ratings retained: "
        f"{matched / total:.2%}"
    )

    print(
        f"\nMatched movies: {len(matches):,}"
    )

    print(
        f"Matched movies with ratings: "
        f"{movies_with_ratings:,}"
    )

    print(
        f"Movies with >= {MIN_MOVIE_RATINGS} ratings: "
        f"{movies_with_20:,}"
    )

    print(
        f"\nUsers with matched ratings: "
        f"{len(user_counts):,}"
    )

    print(
        "Users with >= 2 matched ratings: "
        f"{users_with_2:,}"
    )

    print(
        f"Users with >= {MIN_USER_RATINGS} "
        f"matched ratings: {users_with_5:,}"
    )

    print(
        f"\nAverage matched rating: "
        f"{average_rating:.3f}"
    )

    print("\nRating distribution:")

    for rating, count in sorted(
        results["rating_distribution"].items()
    ):
        print(f"  {rating:.1f}: {count:,}")

    print("\nTop 15 most-rated matched movies:")

    print(
        movie_report[
            ["cinematch_title", "rating_count"]
        ].head(15).to_string(index=False)
    )

    print(f"\nMovie report: {MOVIE_REPORT_PATH}")
    print(f"Summary: {SUMMARY_PATH}")


# ==================================================
# MAIN
# ==================================================

def main():
    if not ZIP_PATH.exists():
        raise FileNotFoundError(
            "MovieLens ZIP missing. "
            "Run movielens_coverage_audit.py first."
        )

    matches = load_matches()

    results = audit_ratings(matches)

    movie_report = create_movie_report(
        matches,
        results
    )

    print_report(
        matches,
        results,
        movie_report
    )

    audit_years(matches)


if __name__ == "__main__":
    main()
