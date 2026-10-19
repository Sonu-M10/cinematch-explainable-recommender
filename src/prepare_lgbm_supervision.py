
from pathlib import Path

import numpy as np
import pandas as pd


# ==================================================
# CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parents[1]
ML_DIR = ROOT / "data" / "movielens"

SOURCE_PATH = ML_DIR / "lgbm_train.csv.gz"

FEATURE_PATH = (
    ML_DIR / "lgbm_feature_history.csv.gz"
)

SUPERVISION_PATH = (
    ML_DIR / "lgbm_supervision.csv.gz"
)

MIN_USER_HISTORY = 7
HOLDOUT_COUNT = 2
POSITIVE_THRESHOLD = 4.0


# ==================================================
# LOAD DATA
# ==================================================

def load_data():
    print("Loading LightGBM training interactions...")

    df = pd.read_csv(
        SOURCE_PATH,
        dtype={
            "userId": "int32",
            "cinematch_id": "int32",
            "rating": "float32",
            "timestamp": "int64"
        }
    )

    required = [
        "userId",
        "cinematch_id",
        "rating",
        "timestamp"
    ]

    if df[required].isna().any().any():
        raise ValueError("Missing values detected.")

    if df.duplicated(
        ["userId", "cinematch_id"]
    ).any():
        raise ValueError(
            "Duplicate user-movie ratings detected."
        )

    print(f"Source interactions: {len(df):,}")

    return df


# ==================================================
# CREATE SUPERVISION SPLIT
# ==================================================

def create_supervision_split(df):
    print("\nCreating chronological supervision split...")

    df = df.sort_values(
        ["userId", "timestamp", "cinematch_id"],
        kind="stable"
    ).reset_index(drop=True)

    user_sizes = df.groupby(
        "userId"
    )["cinematch_id"].transform("size")

    # Only users with enough ratings contribute
    # supervision examples.
    eligible = user_sizes >= MIN_USER_HISTORY

    # Position of each interaction within its
    # user's chronological history.
    position = df.groupby("userId").cumcount()

    remaining = user_sizes - position

    # Hold out the last two interactions.
    supervision_mask = (
        eligible
        & (remaining <= HOLDOUT_COUNT)
    )

    feature_history = df.loc[
        ~supervision_mask
    ].copy()

    supervision = df.loc[
        supervision_mask
    ].copy()

    supervision["relevance"] = (
        supervision["rating"] >= POSITIVE_THRESHOLD
    ).astype("int8")

    return feature_history, supervision


# ==================================================
# VERIFY SPLIT
# ==================================================

def verify_split(feature_history, supervision):
    print("\nVerifying separation...")

    if supervision.empty:
        raise RuntimeError(
            "No supervision interactions created."
        )

    feature_pairs = pd.MultiIndex.from_frame(
        feature_history[
            ["userId", "cinematch_id"]
        ]
    )

    supervision_pairs = pd.MultiIndex.from_frame(
        supervision[
            ["userId", "cinematch_id"]
        ]
    )

    if supervision_pairs.isin(feature_pairs).any():
        raise RuntimeError(
            "Feature/supervision interaction overlap."
        )

    latest_feature_times = (
        feature_history.groupby("userId")[
            "timestamp"
        ].max()
    )

    first_supervision_times = (
        supervision.groupby("userId")[
            "timestamp"
        ].min()
    )

    if (
        first_supervision_times
        < latest_feature_times.loc[
            first_supervision_times.index
        ]
    ).any():
        raise RuntimeError(
            "Chronological separation failed."
        )

    feature_counts = (
        feature_history.groupby("userId").size()
    )

    supervision_users = supervision[
        "userId"
    ].unique()

    if (
        feature_counts.loc[supervision_users] < 5
    ).any():
        raise RuntimeError(
            "Insufficient feature history."
        )

    print("No overlapping user-movie interactions.")
    print("Supervision follows feature history.")
    print("All supervised users retain 5+ ratings.")


# ==================================================
# REPORT
# ==================================================

def print_report(feature_history, supervision):
    positive = int(
        supervision["relevance"].sum()
    )

    negative = len(supervision) - positive

    print("\n" + "=" * 60)
    print("CINEMATCH — LIGHTGBM SUPERVISION SPLIT")
    print("=" * 60)

    print(
        "Feature-history interactions: "
        f"{len(feature_history):,}"
    )

    print(
        "Supervision interactions: "
        f"{len(supervision):,}"
    )

    print(
        "Supervised users: "
        f"{supervision['userId'].nunique():,}"
    )

    print(
        "Feature-history movies: "
        f"{feature_history['cinematch_id'].nunique():,}"
    )

    print(
        "Supervision movies: "
        f"{supervision['cinematch_id'].nunique():,}"
    )

    print(
        f"\nPositive labels (>= 4.0): {positive:,}"
    )

    print(
        f"Other observed ratings (< 4.0): "
        f"{negative:,}"
    )

    print(
        f"Positive share: "
        f"{positive / len(supervision):.2%}"
    )

    print(
        "\nFeature history: "
        f"{FEATURE_PATH}"
    )

    print(
        "Supervision: "
        f"{SUPERVISION_PATH}"
    )


# ==================================================
# MAIN
# ==================================================

def main():
    df = load_data()

    feature_history, supervision = (
        create_supervision_split(df)
    )

    verify_split(
        feature_history,
        supervision
    )

    print("\nSaving datasets...")

    feature_history.to_csv(
        FEATURE_PATH,
        index=False,
        compression="gzip"
    )

    supervision.to_csv(
        SUPERVISION_PATH,
        index=False,
        compression="gzip"
    )

    print_report(
        feature_history,
        supervision
    )


if __name__ == "__main__":
    main()
