
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ML_DIR = ROOT / "data" / "movielens"

HISTORY_PATH = ML_DIR / "lgbm_feature_history.csv.gz"
SUPERVISION_PATH = ML_DIR / "lgbm_supervision.csv.gz"
UNIVERSE_PATH = ML_DIR / "lgbm_history_universe.csv"
AFFINITY_PATH = ML_DIR / "lgbm_history_affinity.csv"

OUTPUT_PATH = ML_DIR / "lgbm_training_pairs.csv.gz"

POSITIVE_THRESHOLD = 4.0


def main():
    print("Loading training datasets...")

    history = pd.read_csv(
        HISTORY_PATH,
        usecols=[
            "userId",
            "cinematch_id",
            "rating",
            "timestamp"
        ]
    )

    supervision = pd.read_csv(
        SUPERVISION_PATH,
        usecols=[
            "userId",
            "cinematch_id",
            "rating",
            "relevance"
        ]
    )

    universe = pd.read_csv(
        UNIVERSE_PATH
    )

    affinity = pd.read_csv(
        AFFINITY_PATH
    )

    movie_ids = set(universe["cinematch_id"])

    # Only historical positive movies in our
    # collaborative candidate universe can be anchors.
    positive_history = history.loc[
        (history["rating"] >= POSITIVE_THRESHOLD)
        & history["cinematch_id"].isin(movie_ids)
    ].copy()

    positive_history = positive_history.sort_values(
        ["userId", "timestamp", "cinematch_id"]
    )

    anchors = (
        positive_history.groupby("userId")
        .tail(1)[["userId", "cinematch_id"]]
        .rename(
            columns={"cinematch_id": "anchor_id"}
        )
    )

    print(f"Users with valid anchors: {len(anchors):,}")

    # Candidate movies must also belong to
    # the supported movie universe.
    supervision = supervision[
        supervision["cinematch_id"].isin(movie_ids)
    ].copy()

    supervision = supervision.rename(
        columns={"cinematch_id": "candidate_id"}
    )

    pairs = supervision.merge(
        anchors,
        on="userId",
        how="inner"
    )

    pairs = pairs[
        pairs["anchor_id"] != pairs["candidate_id"]
    ].copy()

    print(
        "Eligible supervised candidate ratings: "
        f"{len(pairs):,}"
    )

    # Each user should have two held-out ratings.
    # We require both to be in the supported universe.
    group_sizes = pairs.groupby("userId").size()

    valid_users = group_sizes[
        group_sizes == 2
    ].index

    pairs = pairs[
        pairs["userId"].isin(valid_users)
    ].copy()

    # Require one positive and one non-positive
    # observed rating within each ranking group.
    label_counts = (
        pairs.groupby("userId")["relevance"]
        .nunique()
    )

    mixed_users = label_counts[
        label_counts == 2
    ].index

    pairs = pairs[
        pairs["userId"].isin(mixed_users)
    ].copy()

    print(
        "Users with contrasting observed outcomes: "
        f"{len(mixed_users):,}"
    )

    # Join affinity as a historical-only feature.
    affinity_features = affinity[
        [
            "anchor_id",
            "candidate_id",
            "adjusted_affinity",
            "cosine_affinity",
            "shared_positive_users"
        ]
    ].copy()

    pairs = pairs.merge(
        affinity_features,
        on=["anchor_id", "candidate_id"],
        how="left",
        validate="many_to_one"
    )

    # A missing pair is not evidence of zero
    # audience overlap: it may be outside the
    # stored top-100 neighbors.
    pairs["affinity_available"] = (
        pairs["adjusted_affinity"].notna()
        .astype("int8")
    )

    # Add historical candidate popularity.
    popularity = universe[
        ["cinematch_id", "positive_rating_count"]
    ].rename(
        columns={
            "cinematch_id": "candidate_id",
            "positive_rating_count":
                "candidate_positive_count"
        }
    )

    pairs = pairs.merge(
        popularity,
        on="candidate_id",
        how="left",
        validate="many_to_one"
    )

    pairs = pairs.sort_values(
        ["userId", "candidate_id"]
    ).reset_index(drop=True)

    # Final safety checks.
    sizes = pairs.groupby("userId").size()

    assert sizes.eq(2).all()
    assert pairs.groupby(
        "userId"
    )["relevance"].nunique().eq(2).all()

    if pairs.duplicated(
        ["userId", "candidate_id"]
    ).any():
        raise RuntimeError(
            "Duplicate ranking candidates detected."
        )

    pairs.to_csv(
        OUTPUT_PATH,
        index=False,
        compression="gzip"
    )

    print("\n" + "=" * 60)
    print("CINEMATCH — SUPERVISED RANKING PAIRS")
    print("=" * 60)

    print(
        f"Ranking groups: "
        f"{pairs['userId'].nunique():,}"
    )

    print(
        f"Candidate rows: {len(pairs):,}"
    )

    print(
        "Unique anchors: "
        f"{pairs['anchor_id'].nunique():,}"
    )

    print(
        "Unique candidates: "
        f"{pairs['candidate_id'].nunique():,}"
    )

    print(
        "Affinity feature coverage: "
        f"{pairs['affinity_available'].mean():.2%}"
    )

    print(
        "Positive candidate rows: "
        f"{pairs['relevance'].sum():,}"
    )

    print(
        "\nNote: Missing affinity values are "
        "retained as missing, not set to zero."
    )

    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
