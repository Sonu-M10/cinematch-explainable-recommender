
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = PROJECT_ROOT / "data" / "movies_cleaned.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "movies_expanded.csv"

MIN_OVERVIEW_WORDS = 8

ADDITIONAL_METADATA = [
    "keywords",
    "directors",
    "cast",
]


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv(INPUT_PATH)

print(f"\nOriginal cleaned movies: {len(df):,}")


# --------------------------------------------------
# CLEAN METADATA
# --------------------------------------------------

metadata_columns = [
    "overview",
    "genres",
    "keywords",
    "directors",
    "cast",
]

for column in metadata_columns:
    df[column] = df[column].fillna("").astype(str).str.strip()


# --------------------------------------------------
# METADATA QUALITY CHECKS
# --------------------------------------------------

df["overview_word_count"] = (
    df["overview"].str.split().str.len()
)

df["has_valid_genres"] = (
    df["genres"].str.strip().ne("")
)

df["has_additional_metadata"] = (
    df[ADDITIONAL_METADATA]
    .ne("")
    .any(axis=1)
)

df["passes_metadata_quality"] = (
    (df["overview_word_count"] >= MIN_OVERVIEW_WORDS)
    & df["has_valid_genres"]
    & df["has_additional_metadata"]
)


# --------------------------------------------------
# ANALYZE EXCLUSIONS
# --------------------------------------------------

print("\nMetadata quality report")
print("-" * 50)

print(
    "Overview shorter than threshold:",
    f"{(df['overview_word_count'] < MIN_OVERVIEW_WORDS).sum():,}"
)

print(
    "Missing genres:",
    f"{(~df['has_valid_genres']).sum():,}"
)

print(
    "Missing additional metadata:",
    f"{(~df['has_additional_metadata']).sum():,}"
)

print(
    "Movies passing all checks:",
    f"{df['passes_metadata_quality'].sum():,}"
)

print(
    "Movies excluded:",
    f"{(~df['passes_metadata_quality']).sum():,}"
)


# --------------------------------------------------
# INSPECT EXCLUDED MOVIES
# --------------------------------------------------

excluded = df[
    ~df["passes_metadata_quality"]
]

print("\nSample excluded movies:")
print("-" * 50)

print(
    excluded[
        [
            "title",
            "release_date",
            "overview_word_count",
            "genres",
            "vote_count",
        ]
    ].head(15).to_string(index=False)
)


# --------------------------------------------------
# CREATE EXPANDED DATASET
# --------------------------------------------------

expanded = df[
    df["passes_metadata_quality"]
].copy()

expanded = expanded.drop(
    columns=[
        "overview_word_count",
        "has_valid_genres",
        "has_additional_metadata",
        "passes_metadata_quality",
    ]
)

expanded.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nExpanded dataset saved:")
print(OUTPUT_PATH)

print(
    f"Final movies available: {len(expanded):,}"
)
