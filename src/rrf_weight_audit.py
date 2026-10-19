
"""CineMatch RRF weight audit.

Compare ensemble weights without modifying the live engines.

Run:
    python3 src/rrf_weight_audit.py

Outputs:
    data/evaluation/rrf_weight_audit.csv
    data/evaluation/rrf_weight_summary.csv
"""

from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

import weighted_recommender as classic
import hybrid_engine as hybrid
import rrf_ensemble as rrf

from experimental_lgbm_engine import load_ranker
from evaluation_benchmark import resolve_anchors


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "data" / "evaluation"

TOP_N = 10
POOL_SIZE = 100

WEIGHT_CONFIGS = {
    "Equal baseline": {
        "classic": 1.0,
        "hybrid": 1.0,
        "experimental": 1.0,
    },
    "Mild audience boost": {
        "classic": 1.0,
        "hybrid": 1.0,
        "experimental": 1.25,
    },
    "Moderate audience boost": {
        "classic": 1.0,
        "hybrid": 1.0,
        "experimental": 1.5,
    },
    "Strong audience boost": {
        "classic": 1.0,
        "hybrid": 1.0,
        "experimental": 2.0,
    },
    "Content emphasis": {
        "classic": 1.5,
        "hybrid": 1.5,
        "experimental": 1.0,
    },
}


def jaccard(a, b):
    a = set(a)
    b = set(b)

    if not (a or b):
        return np.nan

    return len(a & b) / len(a | b)


def get_rankings(movie_id, embeddings, ranker):
    """Retrieve the same candidate pools used by RRF."""

    classic_results = classic.recommend_movies(
        movie_id,
        top_n=POOL_SIZE,
        ranking_mode="evidence_aware",
    )

    hybrid_results = hybrid.recommend_movies(
        movie_id,
        top_n=POOL_SIZE,
        embeddings=embeddings,
    )

    rankings = {
        "classic": rrf.normalize_recommendations(
            classic_results,
            movie_id,
        ),
        "hybrid": rrf.normalize_recommendations(
            hybrid_results,
            movie_id,
        ),
    }

    if ranker.supports(movie_id):
        experimental_results = ranker.recommend_movies(
            movie_id,
            top_n=POOL_SIZE,
            embeddings=embeddings,
        )

        if (
            experimental_results is not None
            and not experimental_results.empty
            and "ranking_source" in experimental_results.columns
        ):
            native = experimental_results[
                experimental_results["ranking_source"]
                == "LightGBM v4 Experimental"
            ]

            ids = rrf.normalize_recommendations(
                native,
                movie_id,
            )

            if ids:
                rankings["experimental"] = ids

    return rankings


def audit_movie(anchor_name, anchor_id, rankings):
    """Evaluate all weight configurations for one seed."""

    rows = []

    baseline_ids = rrf.fuse_rankings(
        rankings,
        top_n=TOP_N,
        weights=WEIGHT_CONFIGS["Equal baseline"],
    )["id"].astype(int).tolist()

    for config_name, weights in WEIGHT_CONFIGS.items():
        fused = rrf.fuse_rankings(
            rankings,
            top_n=TOP_N,
            weights=weights,
        )

        final_ids = fused["id"].astype(int).tolist()

        # Which input models independently retrieved
        # the final recommendations?
        source_counts = {
            model: sum(
                movie_id in set(movie_ids)
                for movie_id in final_ids
            )
            for model, movie_ids in rankings.items()
        }

        # Count how many recommendations were
        # supported by more than one model.
        multi_model_count = int(
            (fused["model_support"] >= 2).sum()
        )

        # Popularity proxy: TMDB vote_count.
        catalog_indices = [
            classic.movie_id_to_index[movie_id]
            for movie_id in final_ids
        ]

        selected_movies = classic.df.iloc[
            catalog_indices
        ]

        vote_counts = pd.to_numeric(
            selected_movies["vote_count"],
            errors="coerce",
        ).fillna(0)

        rows.append({
            "anchor_title": anchor_name,
            "anchor_id": anchor_id,
            "configuration": config_name,
            "model_count": len(rankings),
            "experimental_active": (
                "experimental" in rankings
            ),
            "classic_supported": source_counts.get(
                "classic", 0
            ),
            "hybrid_supported": source_counts.get(
                "hybrid", 0
            ),
            "experimental_supported": source_counts.get(
                "experimental", 0
            ),
            "multi_model_count": multi_model_count,
            "median_vote_count": float(
                vote_counts.median()
            ),
            "baseline_jaccard": jaccard(
                baseline_ids,
                final_ids,
            ),
            "baseline_shared_count": len(
                set(baseline_ids) & set(final_ids)
            ),
            "recommendation_ids": ",".join(
                str(movie_id) for movie_id in final_ids
            ),
        })

    return rows


def main():
    print("\nCineMatch — RRF Weight Audit")
    print("=" * 60)

    print("\nLoading embeddings...")
    embeddings = hybrid.load_embeddings()

    print("Loading LightGBM...")
    ranker = load_ranker()

    anchors = resolve_anchors()

    rows = []

    for index, (title, movie_id) in enumerate(
        anchors,
        start=1,
    ):
        print(
            f"[{index}/{len(anchors)}] {title}"
        )

        rankings = get_rankings(
            movie_id,
            embeddings,
            ranker,
        )

        rows.extend(
            audit_movie(
                title,
                movie_id,
                rankings,
            )
        )

    results = pd.DataFrame(rows)

    summary = (
        results.groupby("configuration")
        .agg(
            mean_classic_supported=(
                "classic_supported", "mean"
            ),
            mean_hybrid_supported=(
                "hybrid_supported", "mean"
            ),
            mean_experimental_supported=(
                "experimental_supported", "mean"
            ),
            mean_multi_model_count=(
                "multi_model_count", "mean"
            ),
            median_vote_count=(
                "median_vote_count", "median"
            ),
            mean_baseline_jaccard=(
                "baseline_jaccard", "mean"
            ),
            mean_baseline_shared_count=(
                "baseline_shared_count", "mean"
            ),
        )
        .round(3)
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_DIR / "rrf_weight_audit.csv",
        index=False,
    )

    summary.to_csv(
        OUTPUT_DIR / "rrf_weight_summary.csv"
    )

    print("\n" + "=" * 60)
    print("RRF WEIGHT AUDIT SUMMARY")
    print("=" * 60)

    print(summary.to_string())

    print(
        "\nDetailed results saved to:",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
