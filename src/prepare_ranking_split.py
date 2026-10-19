
from pathlib import Path

import numpy as np
import pandas as pd


# ==================================================
# CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parents[1]
ML_DIR = ROOT / "data" / "movielens"

INPUT_PATH = ML_DIR / "matched_interactions.csv.gz"
TRAIN_PATH = ML_DIR / "ranking_train.csv.gz"
TEST_PATH = ML_DIR / "ranking_test.csv.gz"

MIN_USER_RATINGS = 5
RANDOM_SEED = 42


# ==================================================
# LOAD AND VALIDATE
# ==================================================

def load_interactions():
    print("Loading matched interactions...")

    df = pd.read_csv(
        INPUT_PATH,
        dtype={
            "userId": "int32",
            "cinematch_id": "int32",
            "rating": "float32",
            "timestamp": "int64"
        }
    )

    print(f"Original interactions: {len(df):,}")

    missing = df.isna().sum()

    if missing.any():
        raise ValueError(
            f"Missing values detected:\n{missing}"
        )

    if not df["rating"].between(0.5, 5.0).all():
        raise ValueError(
            "Ratings outside the expected 0.5–5 range."
        )

    valid_half_stars = np.isclose(
        df["rating"].to_numpy() * 2,
        np.round(df["rating"].to_numpy() * 2)
    )

    if not valid_half_stars.all():
        raise ValueError(
            "Unexpected rating increments detected."
        )

    if (df["timestamp"] <= 0).any():
        raise ValueError(
            "Invalid timestamps detected."
        )

    return df


# ==================================================
# DEDUPLICATION
# ==================================================

def remove_duplicates(df):
    print("\nChecking duplicate user–movie ratings...")

    duplicate_count = df.duplicated(
        subset=["userId", "cinematch_id"]
    ).sum()

    print(f"Duplicate interactions: {duplicate_count:,}")

    if duplicate_count:
        # Keep the most recent rating.
        df = df.sort_values(
            ["timestamp"],
            kind="stable"
        )

        df = df.drop_duplicates(
            subset=["userId", "cinematch_id"],
            keep="last"
        )

    print(f"Unique interactions: {len(df):,}")

    return df


# ==================================================
# TRAIN / TEST SPLIT
# ==================================================

def create_split(df):
    print("\nCreating chronological user-level split...")

    # Count how many matched movies each user rated.
    user_counts = df.groupby("userId")[
        "cinematch_id"
    ].transform("size")

    # Only users with sufficient history contribute
    # a held-out rating.
    eligible = user_counts >= MIN_USER_RATINGS

    # Sort each user's interactions chronologically.
    # Movie ID makes ties deterministic.
    df = df.sort_values(
        ["userId", "timestamp", "cinematch_id"],
        kind="stable"
    ).reset_index(drop=True)

    eligible = eligible.loc[df.index] if False else (
        df.groupby("userId")["cinematch_id"]
        .transform("size") >= MIN_USER_RATINGS
    )

    # Mark each eligible user's last interaction.
    last_in_user_group = ~df.duplicated(
        subset=["userId"],
        keep="last"
    )

    test_mask = eligible & last_in_user_group

    train = df.loc[~test_mask].copy()
    test = df.loc[test_mask].copy()

    print(f"Training interactions: {len(train):,}")
    print(f"Held-out interactions: {len(test):,}")

    return train, test


# ==================================================
# VERIFY AND SAVE
# ==================================================

def verify_split(train, test):
    print("\nVerifying split integrity...")

    train_pairs = pd.MultiIndex.from_frame(
        train[["userId", "cinematch_id"]]
    )

    test_pairs = pd.MultiIndex.from_frame(
        test[["userId", "cinematch_id"]]
    )

    overlapping_pairs = test_pairs.isin(train_pairs).sum()

    if overlapping_pairs:
        raise RuntimeError(
            f"Train/test interaction overlap: "
            f"{overlapping_pairs}"
        )

    train_users = set(train["userId"].unique())
    test_users = set(test["userId"].unique())

    missing_users = test_users - train_users

    if missing_users:
        raise RuntimeError(
            "Some test users have no training history."
        )

    print("No duplicate user–movie pairs across splits.")
    print("Every test user has training history.")

    return True


def save_split(train, test):
    print("\nSaving datasets...")

    train.to_csv(
        TRAIN_PATH,
        index=False,
        compression="gzip"
    )

    test.to_csv(
        TEST_PATH,
        index=False,
        compression="gzip"
    )

    print(f"Train: {TRAIN_PATH}")
    print(f"Test: {TEST_PATH}")


# ==================================================
# SUMMARY
# ==================================================

def print_summary(train, test):
    print("\n" + "=" * 55)
    print("CINEMATCH RANKING DATASET PREPARATION")
    print("=" * 55)

    print(f"Training ratings: {len(train):,}")
    print(f"Test ratings: {len(test):,}")

    print(
        f"Training users: "
        f"{train['userId'].nunique():,}"
    )

    print(
        f"Test users: "
        f"{test['userId'].nunique():,}"
    )

    print(
        f"Training movies: "
        f"{train['cinematch_id'].nunique():,}"
    )

    print(
        f"Test movies: "
        f"{test['cinematch_id'].nunique():,}"
    )

    print(
        "Average training rating: "
        f"{train['rating'].mean():.3f}"
    )

    print(
        "Average test rating: "
        f"{test['rating'].mean():.3f}"
    )

    print("\nRating thresholds in training:")

    for threshold in [3.0, 3.5, 4.0, 4.5]:
        count = int(
            (train["rating"] >= threshold).sum()
        )

        print(
            f"  >= {threshold:.1f}: "
            f"{count:,} ({count / len(train):.1%})"
        )

    print("\nNext: construct movie-to-movie relevance")
    print("signals using training interactions only.")


# ==================================================
# MAIN
# ==================================================

def main():
    np.random.seed(RANDOM_SEED)

    df = load_interactions()
    df = remove_duplicates(df)

    train, test = create_split(df)

    verify_split(train, test)
    save_split(train, test)

    print_summary(train, test)


if __name__ == "__main__":
    main()
