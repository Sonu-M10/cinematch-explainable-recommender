
from pathlib import Path
import hashlib
import os
import re
import time

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "data" / "movies_expanded.csv"
OUTPUT_DIR = ROOT / "data" / "review_audit"

SAMPLE_SIZE = 200
MIN_WORDS = 50
MIN_REVIEWS = 5
MAX_PAGES = 5

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def count_words(text):
    return len(re.findall(r"\b[\w]+(?:['’-][\w]+)*\b", text))


def normalize_text(text):
    return " ".join(str(text).lower().split())


def load_sample():
    catalog = pd.read_csv(
        CATALOG_PATH,
        usecols=["id", "title", "vote_count", "release_date"]
    )

    catalog["id"] = pd.to_numeric(
        catalog["id"], errors="coerce"
    )
    catalog["vote_count"] = pd.to_numeric(
        catalog["vote_count"], errors="coerce"
    ).fillna(0)

    catalog = catalog.dropna(subset=["id"])
    catalog["id"] = catalog["id"].astype(int)
    catalog = catalog.drop_duplicates("id")

    # Four popularity groups with 50 movies each.
    catalog["popularity_group"] = pd.qcut(
        catalog["vote_count"].rank(method="first"),
        q=4,
        labels=["low", "medium_low", "medium_high", "high"]
    )

    samples = []

    for group, movies in catalog.groupby(
        "popularity_group", observed=True
    ):
        sample = movies.sample(
            n=min(SAMPLE_SIZE // 4, len(movies)),
            random_state=42
        ).copy()
        samples.append(sample)

    result = pd.concat(samples, ignore_index=True)

    print(f"Sampled {len(result)} CineMatch movies")
    return result


def fetch_reviews(session, movie_id):
    reviews = []
    page = 1
    total_pages = 1

    while page <= min(total_pages, MAX_PAGES):
        url = (
            f"https://api.themoviedb.org/3/movie/"
            f"{movie_id}/reviews"
        )

        response = None

        for attempt in range(4):
            try:
                response = session.get(
                    url,
                    params={"page": page},
                    timeout=20
                )

                if response.status_code == 429:
                    wait = response.headers.get(
                        "Retry-After", "3"
                    )
                    time.sleep(min(float(wait), 30))
                    continue

                response.raise_for_status()
                break

            except requests.RequestException:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)

        if response is None or not response.ok:
            raise RuntimeError(
                f"Unable to fetch movie {movie_id}"
            )

        payload = response.json()
        total_pages = payload.get("total_pages", 1)

        reviews.extend(payload.get("results", []))
        page += 1

        time.sleep(0.15)

    return reviews, total_pages


def main():
    token = os.environ.get("TMDB_READ_TOKEN")

    if not token:
        raise RuntimeError(
            "TMDB_READ_TOKEN is missing. "
            "Set it in your terminal first."
        )

    sample = load_sample()

    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {token}",
        "Accept": "application/json"
    })

    review_records = []
    movie_records = []

    for index, movie in enumerate(
        sample.itertuples(index=False), start=1
    ):
        print(
            f"[{index}/{len(sample)}] "
            f"{movie.title}",
            flush=True
        )

        try:
            reviews, total_pages = fetch_reviews(
                session, movie.id
            )
        except Exception as error:
            print(f"  Request failed: {error}")
            movie_records.append({
                "id": movie.id,
                "title": movie.title,
                "popularity_group": str(movie.popularity_group),
                "status": "failed",
                "total_reviews": 0,
                "qualifying_reviews": 0,
                "total_pages": None
            })
            continue

        seen = set()
        qualifying_count = 0

        for review in reviews:
            content = review.get("content") or ""
            word_count = count_words(content)

            normalized = normalize_text(content)
            fingerprint = hashlib.sha256(
                normalized.encode("utf-8")
            ).hexdigest()

            duplicate = fingerprint in seen
            seen.add(fingerprint)

            qualifies = (
                word_count >= MIN_WORDS
                and not duplicate
            )

            if qualifies:
                qualifying_count += 1

            review_records.append({
                "movie_id": movie.id,
                "review_id": review.get("id"),
                "word_count": word_count,
                "duplicate": duplicate,
                "qualifies": qualifies
            })

        movie_records.append({
            "id": movie.id,
            "title": movie.title,
            "popularity_group": str(movie.popularity_group),
            "status": "ok",
            "total_reviews": len(reviews),
            "qualifying_reviews": qualifying_count,
            "total_pages": total_pages
        })

    review_df = pd.DataFrame(review_records)
    movie_df = pd.DataFrame(movie_records)

    movie_df.to_csv(
        OUTPUT_DIR / "tmdb_movie_coverage.csv",
        index=False
    )
    review_df.to_csv(
        OUTPUT_DIR / "tmdb_review_quality.csv",
        index=False
    )

    successful = movie_df[movie_df["status"] == "ok"]
    failed = len(movie_df) - len(successful)

    eligible = successful[
        successful["qualifying_reviews"] >= MIN_REVIEWS
    ]

    print("\n" + "=" * 48)
    print("TMDB REVIEW AUDIT")
    print("=" * 48)

    print(f"Movies sampled: {len(movie_df)}")
    print(f"Successful requests: {len(successful)}")
    print(f"Failed requests: {failed}")

    print(
        "Movies with at least one qualifying review: "
        f"{(successful['qualifying_reviews'] > 0).sum()}"
    )

    print(
        f"Movies with at least {MIN_REVIEWS} "
        f"qualifying reviews: {len(eligible)}"
    )

    if len(successful):
        print(
            "Eligible coverage in successful sample: "
            f"{len(eligible) / len(successful):.2%}"
        )

    if len(review_df):
        print(
            "Reviews with >= 50 words: "
            f"{review_df['qualifies'].sum():,} / "
            f"{len(review_df):,}"
        )
        print(
            "Median review length: "
            f"{review_df['word_count'].median():.0f} words"
        )

    print("\nCoverage by popularity group:")

    print(
        successful.groupby(
            "popularity_group"
        )["qualifying_reviews"].agg(
            movies="count",
            average="mean",
            median="median",
            eligible=lambda x: (x >= MIN_REVIEWS).sum()
        ).to_string()
    )

    print("\nResults saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
