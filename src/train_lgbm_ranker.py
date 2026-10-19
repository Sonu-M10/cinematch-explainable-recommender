
from pathlib import Path
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ==================================================
# CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parents[1]
ML_DIR = ROOT / "data" / "movielens"
MODEL_DIR = ROOT / "models"

DATA_PATH = ML_DIR / "lgbm_training_pairs.csv.gz"
MODEL_PATH = MODEL_DIR / "lgbm_ranker.txt"
REPORT_PATH = MODEL_DIR / "lgbm_training_report.json"

RANDOM_STATE = 42

FEATURES = [
    "adjusted_affinity",
    "cosine_affinity",
    "shared_positive_users",
    "affinity_available",
    "candidate_positive_count",
]


# ==================================================
# LOAD DATA
# ==================================================

def load_data():
    print("Loading supervised ranking pairs...")

    df = pd.read_csv(DATA_PATH)

    required = [
        "userId",
        "anchor_id",
        "candidate_id",
        "relevance",
        *FEATURES,
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    if df[required[:4]].isna().any().any():
        raise ValueError(
            "Missing identifiers or labels."
        )

    if df.duplicated(
        ["userId", "candidate_id"]
    ).any():
        raise ValueError(
            "Duplicate candidates in ranking groups."
        )

    sizes = df.groupby("userId").size()
    label_counts = (
        df.groupby("userId")["relevance"].nunique()
    )

    if not sizes.eq(2).all():
        raise ValueError(
            "Expected exactly two candidates per user."
        )

    if not label_counts.eq(2).all():
        raise ValueError(
            "Each group must have contrasting labels."
        )

    if not df["relevance"].isin([0, 1]).all():
        raise ValueError(
            "Expected binary relevance labels."
        )

    print(f"Candidate rows: {len(df):,}")
    print(f"Ranking groups: {df['userId'].nunique():,}")

    return df


# ==================================================
# DEVELOPMENT SPLIT
# ==================================================

def split_groups(df):
    print("\nSplitting users for model development...")

    users = df["userId"].unique()

    train_users, dev_users = train_test_split(
        users,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    train = df[
        df["userId"].isin(train_users)
    ].copy()

    dev = df[
        df["userId"].isin(dev_users)
    ].copy()

    train = train.sort_values(
        ["userId", "candidate_id"]
    ).reset_index(drop=True)

    dev = dev.sort_values(
        ["userId", "candidate_id"]
    ).reset_index(drop=True)

    print(f"Training groups: {len(train_users):,}")
    print(f"Development groups: {len(dev_users):,}")

    return train, dev


# ==================================================
# PREPARE LIGHTGBM DATA
# ==================================================

def prepare_dataset(df):
    x = df[FEATURES].copy()

    for column in FEATURES:
        x[column] = pd.to_numeric(
            x[column],
            errors="coerce"
        )

    y = df["relevance"].astype("int32")

    group_sizes = (
        df.groupby("userId", sort=False)
        .size()
        .to_numpy()
    )

    return x, y, group_sizes


# ==================================================
# EVALUATION
# ==================================================

def pairwise_accuracy(df, scores):
    evaluation = df[
        ["userId", "relevance"]
    ].copy()

    evaluation["score"] = scores

    # Every group has exactly two candidates.
    # A correct ranking places the positive first.
    predicted = (
        evaluation.groupby("userId")["score"]
        .idxmax()
    )

    winners = evaluation.loc[predicted]

    return float(
        winners["relevance"].mean()
    )


def evaluate_baseline(df, column):
    # Missing affinity values are treated as
    # unavailable for this simple comparison.
    # This is a baseline convention, not a
    # claim that missing affinity equals zero.
    scores = df[column].fillna(-1).to_numpy()

    return pairwise_accuracy(df, scores)


# ==================================================
# TRAIN MODEL
# ==================================================

def train_model(train, dev):
    x_train, y_train, groups_train = (
        prepare_dataset(train)
    )

    x_dev, y_dev, groups_dev = (
        prepare_dataset(dev)
    )

    train_data = lgb.Dataset(
        x_train,
        label=y_train,
        group=groups_train,
        feature_name=FEATURES,
    )

    dev_data = lgb.Dataset(
        x_dev,
        label=y_dev,
        group=groups_dev,
        reference=train_data,
    )

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [1, 2],
        "learning_rate": 0.05,
        "num_leaves": 15,
        "min_data_in_leaf": 50,
        "feature_fraction": 1.0,
        "verbosity": -1,
        "seed": RANDOM_STATE,
        "num_threads": 4,
    }

    print("\nTraining LightGBM...")

    model = lgb.train(
        params,
        train_data,
        num_boost_round=400,
        valid_sets=[dev_data],
        valid_names=["development"],
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=30,
                verbose=False,
            ),
            lgb.log_evaluation(period=50),
        ],
    )

    return model, x_dev


# ==================================================
# REPORT
# ==================================================

def main():
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = load_data()

    train, dev = split_groups(df)

    model, x_dev = train_model(train, dev)

    predictions = model.predict(
        x_dev,
        num_iteration=model.best_iteration,
    )

    model_accuracy = pairwise_accuracy(
        dev,
        predictions
    )

    affinity_accuracy = evaluate_baseline(
        dev,
        "adjusted_affinity"
    )

    popularity_accuracy = evaluate_baseline(
        dev,
        "candidate_positive_count"
    )

    importance = sorted(
        zip(
            FEATURES,
            model.feature_importance(
                importance_type="gain"
            ),
        ),
        key=lambda item: item[1],
        reverse=True,
    )

    report = {
        "training_groups": int(
            train["userId"].nunique()
        ),
        "development_groups": int(
            dev["userId"].nunique()
        ),
        "best_iteration": int(
            model.best_iteration
        ),
        "lightgbm_pairwise_accuracy": model_accuracy,
        "affinity_pairwise_accuracy": affinity_accuracy,
        "popularity_pairwise_accuracy": popularity_accuracy,
        "features": FEATURES,
        "feature_importance_gain": {
            name: float(gain)
            for name, gain in importance
        },
        "evaluation_note": (
            "Two observed candidates per user, "
            "one positive and one non-positive. "
            "Random user-disjoint development split. "
            "Not equivalent to full-list HitRate@10."
        ),
    }

    model.save_model(str(MODEL_PATH))

    REPORT_PATH.write_text(
        json.dumps(report, indent=2)
    )

    print("\n" + "=" * 60)
    print("CINEMATCH — FIRST LIGHTGBM RANKER")
    print("=" * 60)

    print(
        f"Best iteration: {model.best_iteration}"
    )

    print("\nDevelopment pairwise accuracy:")
    print(
        f"LightGBM:          {model_accuracy:.2%}"
    )
    print(
        f"Audience affinity: {affinity_accuracy:.2%}"
    )
    print(
        f"Popularity:        {popularity_accuracy:.2%}"
    )

    print("\nFeature importance (gain):")

    for name, gain in importance:
        print(f"  {name}: {gain:,.2f}")

    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Saved report: {REPORT_PATH}")

    print(
        "\nThese results measure two-choice "
        "ranking, not HitRate@10."
    )


if __name__ == "__main__":
    main()
