
from pathlib import Path

import train_lgbm_ranker as trainer


ROOT = Path(__file__).resolve().parents[1]
ML_DIR = ROOT / "data" / "movielens"
MODEL_DIR = ROOT / "models"


# Use the enriched training table.
trainer.DATA_PATH = (
    ML_DIR / "lgbm_training_pairs_v2.csv.gz"
)

# Preserve the original baseline model.
trainer.MODEL_PATH = (
    MODEL_DIR / "lgbm_ranker_v2.txt"
)

trainer.REPORT_PATH = (
    MODEL_DIR / "lgbm_training_report_v2.json"
)


# Original features plus new content features.
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
    "anchor_imdb_rating"
]


def main():
    print("=" * 60)
    print("CINEMATCH — LIGHTGBM V2")
    print("=" * 60)

    print(
        "\nTraining with audience, content, "
        "and IMDb features.\n"
    )

    trainer.main()

    print(
        "\nOriginal LightGBM v1 model "
        "has been preserved."
    )


if __name__ == "__main__":
    main()
