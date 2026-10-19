
from pathlib import Path

import lgbm_audience_features as affinity


# ==================================================
# CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parents[1]
ML_DIR = ROOT / "data" / "movielens"

affinity.TRAIN_PATH = (
    ML_DIR / "lgbm_feature_history.csv.gz"
)

affinity.AFFINITY_PATH = (
    ML_DIR / "lgbm_history_affinity.csv"
)

affinity.UNIVERSE_PATH = (
    ML_DIR / "lgbm_history_universe.csv"
)


# ==================================================
# REBUILD FEATURES
# ==================================================

def main():
    print("=" * 60)
    print("REBUILDING LEAKAGE-CONTROLLED FEATURES")
    print("=" * 60)

    print("\nSource:", affinity.TRAIN_PATH)

    if not affinity.TRAIN_PATH.exists():
        raise FileNotFoundError(
            "Feature-history dataset is missing."
        )

    df = affinity.load_training_data()

    positives, universe = affinity.select_universe(df)

    matrix, movie_ids = affinity.build_matrix(
        positives
    )

    results = affinity.calculate_affinity(
        matrix,
        movie_ids
    )

    results = affinity.attach_titles(results)

    results.to_csv(
        affinity.AFFINITY_PATH,
        index=False
    )

    affinity.print_report(results, universe)

    print("\nFeature rebuild complete.")
    print(
        "These features exclude all reserved "
        "supervision interactions."
    )


if __name__ == "__main__":
    main()
