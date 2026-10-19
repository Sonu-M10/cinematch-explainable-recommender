
from pathlib import Path
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from evaluate_lgbm_recommendations import (
    load_inputs,
    build_candidate_lists,
    add_content_features,
    add_semantic_features,
    FEATURES,
)


ROOT = Path(__file__).resolve().parents[1]
ML = ROOT / "data" / "movielens"
MODELS = ROOT / "models"

SEED = 42
K = 10
MAX_TRAIN_USERS = 4000
MAX_EVAL_USERS = 1500

MODEL_PATH = MODELS / "lgbm_ranker_v4_experimental.txt"
REPORT_PATH = MODELS / "lgbm_v4_experiment.json"


def make_cases(
    history,
    outcomes,
    universe,
    max_users,
):
    supported = set(
        universe["cinematch_id"].astype(int)
    )

    positives = history[
        (history["rating"] >= 4.0)
        & (history["cinematch_id"].isin(supported))
    ]

    anchors = (
        positives.sort_values(
            ["userId", "timestamp", "cinematch_id"]
        )
        .drop_duplicates("userId", keep="last")
        [["userId", "cinematch_id"]]
        .rename(columns={"cinematch_id": "anchor_id"})
    )

    targets = outcomes[
        (outcomes["rating"] >= 4.0)
        & (outcomes["cinematch_id"].isin(supported))
    ].copy()

    # For training, a user may have multiple
    # subsequent positive interactions.
    # Select the earliest observed positive target.
    targets = (
        targets.sort_values(
            ["userId", "timestamp", "cinematch_id"]
        )
        .drop_duplicates("userId", keep="first")
    )

    cases = targets.merge(
        anchors,
        on="userId",
        how="inner",
    )

    seen = pd.MultiIndex.from_frame(
        history[["userId", "cinematch_id"]]
    )

    targets_index = pd.MultiIndex.from_frame(
        cases[["userId", "cinematch_id"]]
    )

    cases = cases.loc[
        ~targets_index.isin(seen)
    ].copy()

    if len(cases) > max_users:
        cases = cases.sample(
            n=max_users,
            random_state=SEED,
        )

    return cases.sort_values("userId")


def prepare_candidates(
    cases,
    history,
    affinity,
    universe,
    catalog,
):
    print(
        f"\nBuilding candidate lists for "
        f"{len(cases):,} users..."
    )

    candidates = build_candidate_lists(
        cases,
        history,
        affinity,
        universe,
    )

    if candidates.empty:
        raise RuntimeError(
            "No recommendation candidates generated."
        )

    candidates = add_content_features(
        candidates,
        catalog,
    )

    candidates = add_semantic_features(
        candidates
    )

    return candidates


def evaluate(pairs, score_column):
    total_users = pairs["userId"].nunique()

    ranked = pairs.sort_values(
        ["userId", score_column, "candidate_id"],
        ascending=[True, False, True],
        kind="stable",
    )

    top = ranked.groupby(
        "userId",
        sort=False,
    ).head(K).copy()

    top["rank"] = (
        top.groupby("userId")
        .cumcount() + 1
    )

    hits = int(top["relevance"].sum())

    ndcg = float(
        (
            top["relevance"]
            / np.log2(top["rank"] + 1)
        ).sum() / total_users
    )

    coverage = (
        top["candidate_id"].nunique()
        / 1500
    )

    return {
        "hit_rate_at_10": hits / total_users,
        "ndcg_at_10": ndcg,
        "coverage_at_10": coverage,
    }


def main():
    print("=" * 64)
    print("CINEMATCH — LIGHTGBM V4 LISTWISE EXPERIMENT")
    print("=" * 64)

    MODELS.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        lgbm_train,
        validation,
        affinity,
        universe,
        catalog,
    ) = load_inputs()

    # Feature history predates the supervision
    # interactions used to train this experiment.
    feature_history = pd.read_csv(
        ML / "lgbm_feature_history.csv.gz"
    )

    supervision = pd.read_csv(
        ML / "lgbm_supervision.csv.gz"
    )

    print("\nPreparing training cases...")

    train_cases = make_cases(
        feature_history,
        supervision,
        universe,
        MAX_TRAIN_USERS,
    )

    print(
        f"Sampled training users: "
        f"{len(train_cases):,}"
    )

    train_pairs = prepare_candidates(
        train_cases,
        feature_history,
        affinity,
        universe,
        catalog,
    )

    # Train only on lists where the observed
    # positive target was actually retrieved.
    retrieved = (
        train_pairs.groupby("userId")["relevance"]
        .max()
    )

    retrievable_users = set(
        retrieved[retrieved == 1].index
    )

    print(
        "Training candidate recall: "
        f"{len(retrievable_users) / len(train_cases):.2%}"
    )

    train_pairs = train_pairs[
        train_pairs["userId"].isin(
            retrievable_users
        )
    ].copy()

    # All non-target candidates are unobserved,
    # not necessarily disliked.
    #
    # This is a sampled-exposure ranking
    # experiment, not explicit negative feedback.
    train_pairs = train_pairs.sort_values(
        ["userId", "candidate_id"]
    ).reset_index(drop=True)

    print(
        f"Training candidate rows: "
        f"{len(train_pairs):,}"
    )

    print(
        f"Training groups: "
        f"{train_pairs['userId'].nunique():,}"
    )

    x_train = train_pairs[FEATURES].apply(
        pd.to_numeric,
        errors="coerce",
    )

    y_train = (
        train_pairs["relevance"]
        .astype("int32")
    )

    group_sizes = (
        train_pairs.groupby(
            "userId",
            sort=False,
        ).size().to_numpy()
    )

    training_data = lgb.Dataset(
        x_train,
        label=y_train,
        group=group_sizes,
        feature_name=FEATURES,
    )

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [10],
        "learning_rate": 0.05,
        "num_leaves": 15,
        "min_data_in_leaf": 100,
        "feature_fraction": 1.0,
        "verbosity": -1,
        "seed": SEED,
        "num_threads": 4,
    }

    print("\nTraining LightGBM v4...")

    model = lgb.train(
        params,
        training_data,
        num_boost_round=100,
        callbacks=[
            lgb.log_evaluation(period=25)
        ],
    )

    model.save_model(
        str(MODEL_PATH)
    )

    # ------------------------------------------
    # HELD-OUT EVALUATION
    # ------------------------------------------

    print("\nPreparing later validation cases...")

    eval_cases = make_cases(
        lgbm_train,
        validation,
        universe,
        MAX_EVAL_USERS,
    )

    print(
        f"Sampled evaluation users: "
        f"{len(eval_cases):,}"
    )

    # Exclusions and anchor selection use
    # the later pre-validation history.
    #
    # Candidate affinity and candidate
    # popularity features remain based on
    # the earlier feature-history cutoff.
    eval_pairs = prepare_candidates(
        eval_cases,
        lgbm_train,
        affinity,
        universe,
        catalog,
    )

    x_eval = eval_pairs[FEATURES].apply(
        pd.to_numeric,
        errors="coerce",
    )

    eval_pairs["v4_score"] = model.predict(
        x_eval,
        num_threads=4,
    )

    v3 = lgb.Booster(
        model_file=str(
            MODELS / "lgbm_ranker_v3.txt"
        )
    )

    if v3.feature_name() != FEATURES:
        raise ValueError(
            "v3 feature order mismatch."
        )

    eval_pairs["v3_score"] = v3.predict(
        x_eval,
        num_threads=4,
    )

    eval_pairs["affinity_score"] = (
        eval_pairs["adjusted_affinity"]
        .fillna(-1)
    )

    eval_pairs["popularity_score"] = (
        eval_pairs["candidate_positive_count"]
    )

    eval_pairs["imdb_score"] = (
        eval_pairs["candidate_imdb_rating"]
        .fillna(-1)
    )

    candidate_recall = float(
        eval_pairs.groupby("userId")["relevance"]
        .max()
        .mean()
    )

    policies = {
        "Audience affinity": "affinity_score",
        "Popularity": "popularity_score",
        "IMDb-only": "imdb_score",
        "LightGBM v3": "v3_score",
        "LightGBM v4": "v4_score",
    }

    results = {
        name: evaluate(eval_pairs, column)
        for name, column in policies.items()
    }

    print("\n" + "=" * 64)
    print("LIGHTGBM V4 — TOP-10 COMPARISON")
    print("=" * 64)

    print(
        f"Evaluation users: "
        f"{eval_pairs['userId'].nunique():,}"
    )

    print(
        f"Candidate recall: "
        f"{candidate_recall:.2%}"
    )

    print(
        "\nPolicy               Hit@10    NDCG@10   Coverage"
    )
    print("-" * 59)

    for name, result in results.items():
        print(
            f"{name:<20}"
            f"{result['hit_rate_at_10']:>7.2%}   "
            f"{result['ndcg_at_10']:>7.4f}   "
            f"{result['coverage_at_10']:>7.2%}"
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

    print("\nFeature importance:")

    for feature, gain in importance:
        print(f"  {feature}: {gain:,.2f}")

    report = {
        "experiment": "LightGBM v4 listwise",
        "training_groups": int(
            train_pairs["userId"].nunique()
        ),
        "training_rows": int(
            len(train_pairs)
        ),
        "evaluation_users": int(
            eval_pairs["userId"].nunique()
        ),
        "candidate_recall": candidate_recall,
        "results": results,
        "feature_importance": {
            name: float(gain)
            for name, gain in importance
        },
        "limitations": [
            "Unobserved candidates are uncertain negatives.",
            "Training retains only retrievable positives.",
            "Evaluation is restricted to 1500 movies.",
            "The validation period has been examined in earlier experiments.",
            "Popularity and affinity features use the earlier feature-history cutoff.",
        ],
    }

    REPORT_PATH.write_text(
        json.dumps(report, indent=2)
    )

    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Saved report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
