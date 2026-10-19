
from pathlib import Path
from zipfile import ZipFile
import hashlib
import requests
import pandas as pd


# ==================================================
# CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
ML_DIR = DATA_DIR / "movielens"

ML_DIR.mkdir(parents=True, exist_ok=True)

CATALOG_PATH = DATA_DIR / "movies_with_imdb.csv"
ZIP_PATH = ML_DIR / "ml-32m.zip"

OUTPUT_PATH = ML_DIR / "movielens_cinematch_matches.csv"
CONFLICT_PATH = ML_DIR / "identifier_conflicts.csv"

DATASET_URL = (
    "https://files.grouplens.org/"
    "datasets/movielens/ml-32m.zip"
)

CHECKSUM_URL = DATASET_URL + ".md5"


# ==================================================
# DOWNLOAD AND VERIFY ARCHIVE
# ==================================================

def download_archive():
    if ZIP_PATH.exists():
        print("Using existing MovieLens archive.")
        return

    print("Downloading MovieLens 32M (~239 MB)...")

    temp_path = ZIP_PATH.with_suffix(".zip.part")

    try:
        with requests.get(
            DATASET_URL,
            stream=True,
            timeout=(20, 120)
        ) as response:
            response.raise_for_status()

            with open(temp_path, "wb") as file:
                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):
                    if chunk:
                        file.write(chunk)

        temp_path.replace(ZIP_PATH)

    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    print("Download complete.")


def verify_archive():
    print("\nVerifying archive checksum...")

    response = requests.get(
        CHECKSUM_URL,
        timeout=30
    )
    response.raise_for_status()

    expected = response.text.strip().split()[0].lower()

    digest = hashlib.md5()

    with open(ZIP_PATH, "rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b""
        ):
            digest.update(chunk)

    actual = digest.hexdigest()

    if actual != expected:
        raise RuntimeError(
            "MovieLens checksum mismatch. "
            "Delete the incomplete/corrupted ZIP "
            "and download it again."
        )

    print("Checksum verified.")


# ==================================================
# LOAD DATA
# ==================================================

def load_catalog():
    print("\nLoading CineMatch catalog...")

    catalog = pd.read_csv(
        CATALOG_PATH,
        usecols=[
            "id",
            "title",
            "release_date",
            "imdb_id"
        ]
    )

    catalog["id"] = pd.to_numeric(
        catalog["id"],
        errors="coerce"
    ).astype("Int64")

    catalog["imdb_id"] = (
        catalog["imdb_id"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    catalog = catalog.drop_duplicates(subset=["id"])

    print(f"CineMatch movies: {len(catalog):,}")

    return catalog


def load_movielens():
    print("\nReading MovieLens metadata from ZIP...")

    with ZipFile(ZIP_PATH) as archive:
        with archive.open("ml-32m/links.csv") as file:
            links = pd.read_csv(
                file,
                dtype={
                    "movieId": "int64",
                    "imdbId": "string",
                    "tmdbId": "string"
                }
            )

        with archive.open("ml-32m/movies.csv") as file:
            movies = pd.read_csv(file)

    links["tmdbId"] = pd.to_numeric(
        links["tmdbId"],
        errors="coerce"
    ).astype("Int64")

    # MovieLens stores IMDb IDs without the "tt" prefix.
    links["imdb_id"] = links["imdbId"].apply(
        lambda value: (
            "tt" + str(value).strip().zfill(7)
            if pd.notna(value)
            else pd.NA
        )
    )

    links = links.merge(
        movies[["movieId", "title"]],
        on="movieId",
        how="left",
        validate="one_to_one"
    )

    links = links.rename(
        columns={"title": "movielens_title"}
    )

    print(f"MovieLens movies: {len(links):,}")

    return links


# ==================================================
# MATCH IDENTIFIERS
# ==================================================

def match_movies(catalog, links):
    print("\nMatching MovieLens to CineMatch...")

    catalog_ids = catalog[
        ["id", "title", "release_date", "imdb_id"]
    ].rename(
        columns={
            "id": "cinematch_id",
            "title": "cinematch_title"
        }
    )

    # Primary match: TMDB identifiers.
    tmdb_matches = links.merge(
        catalog_ids,
        left_on="tmdbId",
        right_on="cinematch_id",
        how="inner"
    )

    tmdb_pairs = tmdb_matches[
        ["movieId", "cinematch_id"]
    ].drop_duplicates()

    # Fallback match: IMDb identifiers.
    imdb_catalog = catalog_ids.dropna(
        subset=["imdb_id"]
    )

    imdb_links = links.dropna(
        subset=["imdb_id"]
    )

    imdb_matches = imdb_links.merge(
        imdb_catalog,
        on="imdb_id",
        how="inner"
    )

    imdb_pairs = imdb_matches[
        ["movieId", "cinematch_id"]
    ].drop_duplicates()

    print(f"TMDB identifier matches: {len(tmdb_pairs):,}")
    print(f"IMDb identifier matches: {len(imdb_pairs):,}")

    # Collect every candidate match with its source.
    candidates = pd.concat(
        [
            tmdb_pairs.assign(source="tmdb"),
            imdb_pairs.assign(source="imdb")
        ],
        ignore_index=True
    )

    candidates = candidates.drop_duplicates()

    # Detect disagreements across identifier sources.
    movie_counts = candidates.groupby(
        "movieId"
    )["cinematch_id"].nunique()

    conflict_movie_ids = set(
        movie_counts[movie_counts > 1].index
    )

    conflicts = candidates[
        candidates["movieId"].isin(conflict_movie_ids)
    ].copy()

    conflicts.to_csv(
        CONFLICT_PATH,
        index=False
    )

    candidates = candidates[
        ~candidates["movieId"].isin(conflict_movie_ids)
    ]

    # One MovieLens movie must map to one CineMatch movie.
    matches = candidates.drop_duplicates(
        subset=["movieId", "cinematch_id"]
    ).copy()

    # Also prevent multiple MovieLens IDs from
    # silently mapping to the same CineMatch movie.
    cinematch_counts = matches.groupby(
        "cinematch_id"
    )["movieId"].nunique()

    unique_ids = set(
        cinematch_counts[cinematch_counts == 1].index
    )

    duplicate_targets = matches[
        ~matches["cinematch_id"].isin(unique_ids)
    ]

    if not duplicate_targets.empty:
        print(
            "Additional duplicate-target matches excluded:",
            len(duplicate_targets)
        )

    matches = matches[
        matches["cinematch_id"].isin(unique_ids)
    ]

    matches = matches.merge(
        links[["movieId", "movielens_title"]],
        on="movieId",
        how="left"
    )

    matches = matches.merge(
        catalog_ids[
            ["cinematch_id", "cinematch_title", "release_date"]
        ],
        on="cinematch_id",
        how="left"
    )

    matches = matches.sort_values("movieId")

    matches.to_csv(
        OUTPUT_PATH,
        index=False
    )

    return matches, conflicts, tmdb_pairs, imdb_pairs


# ==================================================
# REPORT
# ==================================================

def print_report(
    catalog,
    links,
    matches,
    conflicts,
    tmdb_pairs,
    imdb_pairs
):
    print("\n" + "=" * 55)
    print("MOVIELENS 32M — CINEMATCH COVERAGE AUDIT")
    print("=" * 55)

    total_ml = len(links)
    total_catalog = len(catalog)
    matched = len(matches)

    print(f"MovieLens movies: {total_ml:,}")
    print(f"CineMatch movies: {total_catalog:,}")

    print(f"\nTMDB identifier pairs: {len(tmdb_pairs):,}")
    print(f"IMDb identifier pairs: {len(imdb_pairs):,}")

    print(f"Identifier conflicts: {conflicts['movieId'].nunique():,}")
    print(f"Final unique matches: {matched:,}")

    print(
        "MovieLens movie coverage: "
        f"{matched / total_ml:.2%}"
    )

    print(
        "CineMatch catalog coverage: "
        f"{matched / total_catalog:.2%}"
    )

    if matched:
        print("\nSample matched movies:")

        print(
            matches[
                [
                    "movielens_title",
                    "cinematch_title",
                    "cinematch_id"
                ]
            ].head(20).to_string(index=False)
        )

    print(f"\nMatches saved to: {OUTPUT_PATH}")
    print(f"Conflicts saved to: {CONFLICT_PATH}")

    print(
        "\nNext step: count actual ratings associated "
        "with matched movies using chunked processing."
    )


# ==================================================
# MAIN
# ==================================================

def main():
    download_archive()
    verify_archive()

    catalog = load_catalog()
    links = load_movielens()

    matches, conflicts, tmdb_pairs, imdb_pairs = (
        match_movies(catalog, links)
    )

    print_report(
        catalog,
        links,
        matches,
        conflicts,
        tmdb_pairs,
        imdb_pairs
    )


if __name__ == "__main__":
    main()
