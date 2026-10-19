
from pathlib import Path

import numpy as np
import pandas as pd

import train_lgbm_ranker as trainer


# ==================================================
# CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parents[1]
ML_DIR = ROOT / "data" / "movielens"
MODEL_DIR = ROOT / "models"

trainer.DATA_PATH = (
    ML_DIR / "lgbm_training_pairs_v3.csv.gz"
)

trainer.MODEL_PATH = (
    MODEL_DIR / "lgbm_ranker_v3.txt"
)

trainer.REPORT_PATH = (
    MODEL_DIR / "lgbm_training_report_v3.json"
)

trainer.FEATURES = [
    "adjusted_affinity",
    "cosine_affinity",
    "shared_positive_users",
    "affinity_available",
    "candidate_positive_count",
    "tfidf_similarity",
    "genre_overlap",
    "year_difference",
    "candidate_year",
    "candidate_imdb_rating",
    "candidate_imdb_log_votes",
    "anchor_imdb_rating",
    "semantic_similarity",
]


# ==================================================
# IMDb-ONLY BASELINE
# ==================================================

def evaluate_imdb_baseline():
    print("\nEvaluating IMDb-only baseline...")

    df = trainer.load_data()

    _, dev = trainer.split_groups(df)

    # IMDb-only baseline:
    # Rank candidates by IMDb rating.
    #
    # Missing IMDb ratings receive -1, below
    # the valid IMDb rating range.
    scores = (
        dev["candidate_imdb_rating"]
        .fillna(-1)
        .to_numpy()
    )

    accuracy = trainer.pairwise_accuracy(
        dev,
        scores,
    )

    # Check ties, because idxmax resolves them
    # according to row order.
    scored = dev[
        ["userId", "relevance"]
    ].copy()

    scored["score"] = scores

    score_ranges = scored.groupby(
        "userId"
    )["score"].agg(
        lambda values: (
            values.max() - values.min()
        )
    )

    tie_count = int(
        np.isclose(
            score_ranges.to_numpy(),
            0,
        ).sum()
    )

    print("\nIMDb-only development results:")
    print(f"Pairwise accuracy: {accuracy:.2%}")
    print(f"Tied rating groups: {tie_count:,}")

    print(
        "\nNote: IMDb ties are resolved by "
        "candidate row order in this baseline."
    )

    return accuracy


# ==================================================
# MAIN
# ==================================================

def main():
    print("=" * 60)
    print("CINEMATCH — LIGHTGBM V3")
    print("=" * 60)

    print(
        "\nTraining with semantic, metadata, "
        "IMDb, and collaborative features.\n"
    )

    trainer.main()

    imdb_accuracy = evaluate_imdb_baseline()

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print("LightGBM v1:       57.11%")
    print("LightGBM v2:       61.44%")
    print(
        "IMDb-only:        "
        f"{imdb_accuracy:.2%}"
    )

    print(
        "\nLightGBM v3 accuracy is shown "
        "in the training report above."
    )

    print(
        "\nOriginal v1 and v2 model files "
        "remain unchanged."
    )


if __name__ == "__main__":
    main()
