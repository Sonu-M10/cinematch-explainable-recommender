
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ML_DIR = ROOT / "data" / "movielens"

TRAIN_PATH = ML_DIR / "ranking_train.csv.gz"
TEST_PATH = ML_DIR / "ranking_test.csv.gz"
AFFINITY_PATH = ML_DIR / "audience_affinity.csv"

K = 10
POSITIVE_THRESHOLD = 4.0


def load_data():
    print("Loading evaluation data...")

    train = pd.read_csv(
        TRAIN_PATH,
        usecols=[
            "userId", "cinematch_id",
            "rating", "timestamp"
        ]
    )

    test = pd.read_csv(
        TEST_PATH,
        usecols=["userId", "cinematch_id", "rating"]
    )

    affinity = pd.read_csv(AFFINITY_PATH)

    return train, test, affinity


def build_evaluation_cases(train, test, affinity):
    # Evaluate only held-out movies within the
    # 1,500-movie audience-affinity universe.
    universe = set(affinity["anchor_id"])

    positive_test = test[
        (test["rating"] >= POSITIVE_THRESHOLD)
        & (test["cinematch_id"].isin(universe))
    ].copy()

    positive_train = train[
        (train["rating"] >= POSITIVE_THRESHOLD)
        & (train["cinematch_id"].isin(universe))
    ].copy()

    # Latest positive training movie is the anchor.
    anchors = (
        positive_train.sort_values(
            ["userId", "timestamp", "cinematch_id"]
        )
        .drop_duplicates("userId", keep="last")
        [["userId", "cinematch_id"]]
        .rename(columns={"cinematch_id": "anchor_id"})
    )

    cases = positive_test.merge(
        anchors,
        on="userId",
        how="inner"
    )

    # Exclude held-out movies that already appeared
    # in the user's training history (defensive check).
    train_pairs = pd.MultiIndex.from_frame(
        train[["userId", "cinematch_id"]]
    )

    case_pairs = pd.MultiIndex.from_frame(
        cases[["userId", "cinematch_id"]]
    )

    cases = cases.loc[
        ~case_pairs.isin(train_pairs)
    ].copy()

    return cases, universe


def evaluate(train, cases, affinity, universe):
    print("\nPreparing recommendation lookups...")

    # Training popularity baseline: number of
    # positive ratings per movie.
    popularity = (
        train[
            (train["rating"] >= POSITIVE_THRESHOLD)
            & (train["cinematch_id"].isin(universe))
        ]["cinematch_id"]
        .value_counts()
        .index.tolist()
    )

    affinity = affinity.sort_values(
        ["anchor_id", "adjusted_affinity"],
        ascending=[True, False]
    )

    neighbors = {
        int(anchor): group["candidate_id"].astype(int).tolist()
        for anchor, group in affinity.groupby("anchor_id")
    }

    # All previously rated movies must be excluded,
    # regardless of whether the user liked them.
    histories = (
        train.groupby("userId")["cinematch_id"]
        .agg(set)
        .to_dict()
    )

    affinity_hits = 0
    popularity_hits = 0
    evaluated = 0
    affinity_short_lists = 0

    for row in cases.itertuples(index=False):
        user = int(row.userId)
        anchor = int(row.anchor_id)
        target = int(row.cinematch_id)

        seen = histories[user]

        affinity_recs = [
            movie
            for movie in neighbors.get(anchor, [])
            if movie not in seen
        ][:K]

        popularity_recs = [
            movie
            for movie in popularity
            if movie not in seen
        ][:K]

        if len(affinity_recs) < K:
            affinity_short_lists += 1

        affinity_hits += target in affinity_recs
        popularity_hits += target in popularity_recs
        evaluated += 1

    return {
        "evaluated": evaluated,
        "affinity_hits": affinity_hits,
        "popularity_hits": popularity_hits,
        "affinity_short_lists": affinity_short_lists
    }


def main():
    train, test, affinity = load_data()

    cases, universe = build_evaluation_cases(
        train, test, affinity
    )

    print(f"Affinity universe: {len(universe):,} movies")
    print(f"Eligible evaluation users: {len(cases):,}")

    if cases.empty:
        raise RuntimeError("No eligible evaluation cases.")

    results = evaluate(
        train, cases, affinity, universe
    )

    n = results["evaluated"]

    affinity_rate = results["affinity_hits"] / n
    popularity_rate = results["popularity_hits"] / n

    print("\n" + "=" * 55)
    print("CINEMATCH — OFFLINE RECOMMENDATION EVALUATION")
    print("=" * 55)

    print(f"Evaluated users: {n:,}")
    print(f"Recommendation cutoff: {K}")

    print(
        f"\nAudience affinity HitRate@{K}: "
        f"{affinity_rate:.2%}"
    )

    print(
        f"Popularity HitRate@{K}: "
        f"{popularity_rate:.2%}"
    )

    print(
        f"\nDifference: "
        f"{(affinity_rate - popularity_rate) * 100:+.2f} "
        "percentage points"
    )

    print(
        "Affinity lists with fewer than 10 unseen "
        f"movies: {results['affinity_short_lists']:,}"
    )

    print(
        "\nNote: This evaluates only positive held-out "
        "movies within the affinity model's "
        "1,500-movie universe."
    )


if __name__ == "__main__":
    main()
