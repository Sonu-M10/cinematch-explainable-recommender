
from pathlib import Path
import pandas as pd


# ==================================================
# CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parents[1]
ML_DIR = ROOT / "data" / "movielens"

SOURCE_PATH = ML_DIR / "ranking_train.csv.gz"
TEST_PATH = ML_DIR / "ranking_test.csv.gz"

OUTPUT_TRAIN = ML_DIR / "lgbm_train.csv.gz"
OUTPUT_VALID = ML_DIR / "lgbm_validation.csv.gz"

MIN_HISTORY = 5


# ==================================================
# LOAD
# ==================================================

def load_data():
    print("Loading existing training interactions...")

    df = pd.read_csv(
        SOURCE_PATH,
        dtype={
            "userId": "int32",
            "cinematch_id": "int32",
            "rating": "float32",
            "timestamp": "int64"
        }
    )

    required = {
        "userId",
        "cinematch_id",
        "rating",
        "timestamp"
    }

    if not required.issubset(df.columns):
        raise ValueError("Missing required columns.")

    if df[list(required)].isna().any().any():
        raise ValueError("Missing values detected.")

    if df.duplicated(
        ["userId", "cinematch_id"]
    ).any():
        raise ValueError(
            "Duplicate user-movie interactions detected."
        )

    print(f"Source interactions: {len(df):,}")

    return df


# ==================================================
# SPLIT
# ==================================================

def create_validation_split(df):
    print("\nCreating chronological validation split...")

    df = df.sort_values(
        ["userId", "timestamp", "cinematch_id"],
        kind="stable"
    ).reset_index(drop=True)

    history_counts = (
        df.groupby("userId")["cinematch_id"]
        .transform("size")
    )

    eligible = history_counts >= MIN_HISTORY

    # Last chronological interaction per user.
    last_interaction = ~df.duplicated(
        "userId",
        keep="last"
    )

    validation_mask = eligible & last_interaction

    train = df.loc[~validation_mask].copy()
    validation = df.loc[validation_mask].copy()

    return train, validation


# ==================================================
# VALIDATE
# ==================================================

def verify_splits(train, validation):
    print("\nChecking split integrity...")

    if validation.empty:
        raise RuntimeError(
            "No validation interactions created."
        )

    train_pairs = pd.MultiIndex.from_frame(
        train[["userId", "cinematch_id"]]
    )

    validation_pairs = pd.MultiIndex.from_frame(
        validation[["userId", "cinematch_id"]]
    )

    if validation_pairs.isin(train_pairs).any():
        raise RuntimeError(
            "Training/validation interaction overlap."
        )

    train_users = set(train["userId"].unique())
    validation_users = set(
        validation["userId"].unique()
    )

    if not validation_users.issubset(train_users):
        raise RuntimeError(
            "Validation users missing training history."
        )

    # Every validation interaction must be at
    # least as recent as that user's training ratings.
    latest_training = train.groupby(
        "userId"
    )["timestamp"].max()

    validation_times = validation.set_index(
        "userId"
    )["timestamp"]

    if (
        validation_times
        < latest_training.loc[validation_times.index]
    ).any():
        raise RuntimeError(
            "Chronological split violation."
        )

    # Check that the previous test set contains
    # no interactions in either new split.
    print("Checking against existing test set...")

    test = pd.read_csv(
        TEST_PATH,
        usecols=["userId", "cinematch_id"]
    )

    test_pairs = pd.MultiIndex.from_frame(test)

    if test_pairs.isin(train_pairs).any():
        raise RuntimeError(
            "Test/training interaction overlap."
        )

    if test_pairs.isin(validation_pairs).any():
        raise RuntimeError(
            "Test/validation interaction overlap."
        )

    print("All split checks passed.")


# ==================================================
# REPORT
# ==================================================

def print_report(train, validation):
    print("\n" + "=" * 60)
    print("CINEMATCH LIGHTGBM DATA SPLIT")
    print("=" * 60)

    print(f"Training interactions: {len(train):,}")
    print(
        f"Validation interactions: "
        f"{len(validation):,}"
    )

    print(
        f"\nTraining users: "
        f"{train['userId'].nunique():,}"
    )

    print(
        f"Validation users: "
        f"{validation['userId'].nunique():,}"
    )

    print(
        f"\nTraining movies: "
        f"{train['cinematch_id'].nunique():,}"
    )

    print(
        f"Validation movies: "
        f"{validation['cinematch_id'].nunique():,}"
    )

    print(
        f"\nAverage training rating: "
        f"{train['rating'].mean():.3f}"
    )

    print(
        f"Average validation rating: "
        f"{validation['rating'].mean():.3f}"
    )

    print(
        "\nPositive validation ratings (>= 4.0): "
        f"{(validation['rating'] >= 4.0).sum():,}"
    )

    print(f"\nTraining file: {OUTPUT_TRAIN}")
    print(f"Validation file: {OUTPUT_VALID}")

    print("\nExisting test file remains unchanged.")


# ==================================================
# MAIN
# ==================================================

def main():
    df = load_data()

    train, validation = create_validation_split(df)

    verify_splits(train, validation)

    print("\nSaving datasets...")

    train.to_csv(
        OUTPUT_TRAIN,
        index=False,
        compression="gzip"
    )

    validation.to_csv(
        OUTPUT_VALID,
        index=False,
        compression="gzip"
    )

    print_report(train, validation)


if __name__ == "__main__":
    main()
