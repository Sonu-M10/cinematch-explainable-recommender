
from pathlib import Path
from zipfile import ZipFile

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

ML_DIR = ROOT / "data" / "movielens"

ZIP_PATH = ML_DIR / "ml-32m.zip"

MATCHES_PATH = (
    ML_DIR / "movielens_cinematch_matches.csv"
)

OUTPUT_PATH = (
    ML_DIR / "matched_interactions.csv.gz"
)

CHUNK_SIZE = 500_000


def load_movie_mapping():
    print("Loading MovieLens movie mapping...")

    matches = pd.read_csv(
        MATCHES_PATH,
        usecols=["movieId", "cinematch_id"],
        dtype={
            "movieId": "int32",
            "cinematch_id": "int32"
        }
    )

    if matches["movieId"].duplicated().any():
        raise ValueError(
            "Duplicate MovieLens movie IDs detected."
        )

    if matches["cinematch_id"].duplicated().any():
        raise ValueError(
            "Duplicate CineMatch movie IDs detected."
        )

    print(f"Matched movie mappings: {len(matches):,}")

    return matches


def extract_interactions(matches):
    print("\nExtracting matched interactions...")

    matched_ids = set(matches["movieId"])

    movie_mapping = matches.set_index(
        "movieId"
    )["cinematch_id"]

    total_rows = 0
    retained_rows = 0
    first_chunk = True

    # Remove previous output to avoid appending
    # duplicate rows from an earlier run.
    OUTPUT_PATH.unlink(missing_ok=True)

    with ZipFile(ZIP_PATH) as archive:
        with archive.open("ml-32m/ratings.csv") as file:

            chunks = pd.read_csv(
                file,
                usecols=[
                    "userId",
                    "movieId",
                    "rating",
                    "timestamp"
                ],
                dtype={
                    "userId": "int32",
                    "movieId": "int32",
                    "rating": "float32",
                    "timestamp": "int64"
                },
                chunksize=CHUNK_SIZE
            )

            for number, chunk in enumerate(
                chunks,
                start=1
            ):
                total_rows += len(chunk)

                selected = chunk.loc[
                    chunk["movieId"].isin(matched_ids)
                ].copy()

                selected["cinematch_id"] = (
                    selected["movieId"].map(
                        movie_mapping
                    )
                )

                if selected["cinematch_id"].isna().any():
                    raise RuntimeError(
                        "Movie ID mapping failed."
                    )

                selected["cinematch_id"] = (
                    selected["cinematch_id"].astype("int32")
                )

                selected = selected[
                    [
                        "userId",
                        "cinematch_id",
                        "rating",
                        "timestamp"
                    ]
                ]

                selected.to_csv(
                    OUTPUT_PATH,
                    mode="wt" if first_chunk else "at",
                    header=first_chunk,
                    index=False,
                    compression="gzip"
                )

                first_chunk = False
                retained_rows += len(selected)

                if number % 10 == 0:
                    print(
                        f"Processed {total_rows:,} ratings; "
                        f"retained {retained_rows:,}"
                    )

    if retained_rows == 0:
        raise RuntimeError(
            "No matched interactions were extracted."
        )

    print("\n" + "=" * 50)
    print("MATCHED INTERACTIONS EXTRACTION")
    print("=" * 50)

    print(f"Original ratings: {total_rows:,}")
    print(f"Retained ratings: {retained_rows:,}")
    print(
        f"Retention rate: "
        f"{retained_rows / total_rows:.2%}"
    )
    print(f"Output: {OUTPUT_PATH}")


def main():
    if not ZIP_PATH.exists():
        raise FileNotFoundError(ZIP_PATH)

    matches = load_movie_mapping()
    extract_interactions(matches)


if __name__ == "__main__":
    main()
