
"""CineMatch Reciprocal Rank Fusion ensemble.

Combines ranked recommendations from:
1. CineMatch Classic
2. CineMatch Hybrid
3. LightGBM v4 (Experimental), when supported

RRF uses ranking positions, not raw model scores.

Default weights:
- Classic: 1.00
- Hybrid: 1.00
- LightGBM: 1.25

The weights are provisional, selected using an
eight-seed AI-assisted relevance evaluation.

No model training occurs in this module.
"""

from collections import defaultdict

import numpy as np
import pandas as pd

import weighted_recommender as classic
import hybrid_engine as hybrid


RRF_K = 60
DEFAULT_POOL_SIZE = 100

MODEL_WEIGHTS = {
    "classic": 1.0,
    "hybrid": 1.0,
    "experimental": 1.25,
}

MODEL_NAMES = {
    "classic": "Classic",
    "hybrid": "Hybrid",
    "experimental": "CineMatch AI",
}


def reciprocal_rank(rank, k=RRF_K):
    """Calculate the RRF contribution for one rank."""

    if rank < 1:
        raise ValueError("Rank must be positive.")

    return 1.0 / (k + rank)


def normalize_recommendations(results, movie_id):
    """Extract unique movie IDs in recommendation order."""

    if results is None or results.empty:
        return []

    if "id" not in results.columns:
        raise ValueError(
            "Recommendation results must contain an id column."
        )

    movie_ids = []
    seen = set()

    for value in results["id"]:
        if pd.isna(value):
            continue

        candidate_id = int(value)

        if candidate_id == movie_id:
            continue

        if candidate_id in seen:
            continue

        if candidate_id not in classic.movie_id_to_index:
            continue

        seen.add(candidate_id)
        movie_ids.append(candidate_id)

    return movie_ids


def fuse_rankings(
    rankings,
    top_n=10,
    k=RRF_K,
    weights=None,
):
    """Fuse named, ordered movie-ID lists using RRF.

    Returns a DataFrame with:
    id, rrf_score, model_support, supporting_models,
    and each model's rank.
    """

    if top_n < 1:
        raise ValueError("top_n must be positive.")

    if k <= 0:
        raise ValueError("RRF k must be positive.")

    if weights is None:
        weights = MODEL_WEIGHTS

    scores = defaultdict(float)
    support = defaultdict(list)
    positions = defaultdict(dict)

    for model_name, movie_ids in rankings.items():

        weight = float(
            weights.get(model_name, 1.0)
        )

        if weight <= 0:
            continue

        seen = set()

        for rank, movie_id in enumerate(
            movie_ids,
            start=1,
        ):
            movie_id = int(movie_id)

            if movie_id in seen:
                continue

            seen.add(movie_id)

            scores[movie_id] += (
                weight * reciprocal_rank(rank, k)
            )

            support[movie_id].append(
                MODEL_NAMES.get(
                    model_name,
                    model_name,
                )
            )

            positions[movie_id][model_name] = rank

    rows = []

    for movie_id, score in scores.items():

        row = {
            "id": movie_id,
            "rrf_score": score,
            "model_support": len(
                support[movie_id]
            ),
            "supporting_models": ", ".join(
                support[movie_id]
            ),
        }

        for model_name in MODEL_NAMES:
            row[f"{model_name}_rank"] = (
                positions[movie_id].get(
                    model_name,
                    np.nan,
                )
            )

        rows.append(row)

    columns = [
        "id",
        "rrf_score",
        "model_support",
        "supporting_models",
        "classic_rank",
        "hybrid_rank",
        "experimental_rank",
    ]

    if not rows:
        return pd.DataFrame(columns=columns)

    fused = pd.DataFrame(rows)

    fused = fused.sort_values(
        [
            "rrf_score",
            "model_support",
            "id",
        ],
        ascending=[
            False,
            False,
            True,
        ],
        kind="stable",
    )

    return (
        fused.head(top_n)
        .reset_index(drop=True)
    )


def recommend_movies(
    movie_id,
    top_n=10,
    pool_size=DEFAULT_POOL_SIZE,
    embeddings=None,
    experimental_ranker=None,
    ranking_mode="evidence_aware",
):
    """Return RRF recommendations for one selected movie."""

    movie_id = int(movie_id)

    if movie_id not in classic.movie_id_to_index:
        return None

    if top_n < 1:
        raise ValueError(
            "top_n must be positive."
        )

    if pool_size < 1:
        raise ValueError(
            "pool_size must be positive."
        )

    retrieval_count = max(
        int(top_n),
        int(pool_size),
    )

    # ------------------------------------------
    # CLASSIC RANKING
    # ------------------------------------------

    classic_results = classic.recommend_movies(
        movie_id,
        top_n=retrieval_count,
        ranking_mode=ranking_mode,
    )

    classic_ids = normalize_recommendations(
        classic_results,
        movie_id,
    )

    # ------------------------------------------
    # HYBRID RANKING
    # ------------------------------------------

    if embeddings is None:
        embeddings = hybrid.load_embeddings()

    hybrid_results = hybrid.recommend_movies(
        movie_id,
        top_n=retrieval_count,
        embeddings=embeddings,
    )

    hybrid_ids = normalize_recommendations(
        hybrid_results,
        movie_id,
    )

    rankings = {
        "classic": classic_ids,
        "hybrid": hybrid_ids,
    }

    # ------------------------------------------
    # OPTIONAL EXPERIMENTAL RANKING
    # ------------------------------------------

    if (
        experimental_ranker is not None
        and experimental_ranker.supports(movie_id)
    ):

        experimental_results = (
            experimental_ranker.recommend_movies(
                movie_id,
                top_n=retrieval_count,
                embeddings=embeddings,
            )
        )

        if (
            experimental_results is not None
            and not experimental_results.empty
        ):

            # Do not include a Hybrid fallback
            # as an independent third vote.
            if "ranking_source" in experimental_results.columns:

                valid_results = experimental_results[
                    experimental_results["ranking_source"]
                    == "LightGBM v4 Experimental"
                ]

            else:
                valid_results = (
                    experimental_results.iloc[:0]
                )

            experimental_ids = (
                normalize_recommendations(
                    valid_results,
                    movie_id,
                )
            )

            if experimental_ids:
                rankings["experimental"] = (
                    experimental_ids
                )

    # ------------------------------------------
    # RECIPROCAL RANK FUSION
    # ------------------------------------------

    fused = fuse_rankings(
        rankings,
        top_n=top_n,
    )

    if fused.empty:
        return classic.df.iloc[:0].copy()

    # ------------------------------------------
    # ATTACH MOVIE INFORMATION
    # ------------------------------------------

    catalog = classic.df.set_index(
        "id",
        drop=False,
    )

    movie_details = catalog.loc[
        fused["id"].tolist()
    ].reset_index(drop=True)

    # Keep the fused ranking order.
    results = movie_details.copy()

    for column in fused.columns:

        if column == "id":
            continue

        results[column] = (
            fused[column].to_numpy()
        )

    # These are compatibility fields for
    # the Streamlit presentation layer.
    #
    # The values are RRF scores,
    # not cosine similarities.
    results["similarity"] = (
        results["rrf_score"]
    )

    results["final_score"] = (
        results["rrf_score"]
    )

    results["ranking_source"] = (
        "RRF Ensemble"
    )

    results["ensemble_models"] = (
        ", ".join(
            MODEL_NAMES[name]
            for name in rankings
        )
    )

    return results.reset_index(drop=True)
