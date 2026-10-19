
"""Experimental query-adaptive movie similarity for CineMatch.

Run from project root: python3 src/adaptive_similarity.py
Does not modify the production recommender.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from weighted_recommender import (
    df,
    feature_matrices,
    FEATURE_WEIGHTS,
    movie_id_to_index,
    recommend_movies,
)

TEST_MOVIES = {
    "The Batman (2022)": 414906,
    "Inception": 27205,
    "Interstellar": 157336,
    "The Conjuring": 138843,
    "La La Land": 313369,
    "Get Out": 419430,
    "Knives Out": 546554,
}

TOP_N = 10
CANDIDATE_POOL_SIZE = 200

# Controls how far adaptive weights may move from the baseline.
ADAPTATION_STRENGTH = 0.60

# Limits the influence of unusually rare or common metadata.
RELATIVE_INFO_MIN = 0.5
RELATIVE_INFO_MAX = 2.0


def feature_informativeness(movie_index):
    """Estimate informativeness using term rarity and metadata richness."""

    raw = {}

    for feature, matrix in feature_matrices.items():

        row = matrix.getrow(movie_index)

        if row.nnz == 0:
            raw[feature] = 0.0
            continue

        df_values = DOCUMENT_FREQUENCIES[feature][row.indices]

        idf = (
            np.log(
                (1.0 + matrix.shape[0])
                / (1.0 + df_values)
            )
            + 1.0
        )

        completeness = 1.0 - np.exp(-row.nnz / 5.0)

        raw[feature] = float(
            np.mean(idf) * completeness
        )

    return raw


def build_reference_informativeness():
    """Build typical informativeness values for each feature."""

    sample_indices = np.linspace(
        0,
        len(df) - 1,
        num=min(1200, len(df)),
        dtype=int,
    )

    references = {}

    for feature, matrix in feature_matrices.items():

        counts = DOCUMENT_FREQUENCIES[feature]
        scores = []

        for i in sample_indices:

            row = matrix.getrow(int(i))

            if row.nnz == 0:
                continue

            idf = (
                np.log(
                    (1.0 + matrix.shape[0])
                    / (1.0 + counts[row.indices])
                )
                + 1.0
            )

            completeness = (
                1.0 - np.exp(-row.nnz / 5.0)
            )

            scores.append(
                float(np.mean(idf) * completeness)
            )

        references[feature] = (
            float(np.median(scores))
            if scores
            else 1.0
        )

    return references


# Count how many movies contain each vocabulary term.
DOCUMENT_FREQUENCIES = {
    feature: np.diff(matrix.tocsc().indptr)
    for feature, matrix in feature_matrices.items()
}

REFERENCE_INFORMATIVENESS = (
    build_reference_informativeness()
)


def adaptive_weights(movie_index):
    """Generate five movie-specific weights summing to 100%."""

    information = feature_informativeness(movie_index)

    features = list(FEATURE_WEIGHTS)

    baseline = np.array(
        list(FEATURE_WEIGHTS.values()),
        dtype=float,
    )

    relative = np.array([
        information[feature]
        / max(
            REFERENCE_INFORMATIVENESS[feature],
            1e-10,
        )
        for feature in features
    ])

    relative = np.clip(
        relative,
        RELATIVE_INFO_MIN,
        RELATIVE_INFO_MAX,
    )

    # Missing metadata has no adaptive contribution.
    for j, feature in enumerate(features):
        if information[feature] == 0:
            relative[j] = 0.0

    adapted = baseline * relative

    if adapted.sum() <= 0:
        adapted = baseline.copy()
    else:
        adapted /= adapted.sum()

    # Blend with the original weights for stability.
    blended = (
        (1 - ADAPTATION_STRENGTH) * baseline
        + ADAPTATION_STRENGTH * adapted
    )

    blended /= blended.sum()

    return dict(zip(features, blended)), information


def adaptive_recommendations(
    movie_id,
    top_n=TOP_N,
    pool_size=CANDIDATE_POOL_SIZE,
):
    """Rank the catalog using query-specific feature weights."""

    if movie_id not in movie_id_to_index:
        return None, None, None

    if top_n < 1 or pool_size < 1:
        raise ValueError(
            "top_n and pool_size must be positive"
        )

    idx = movie_id_to_index[movie_id]

    weights, information = adaptive_weights(idx)

    total = np.zeros(
        len(df),
        dtype=np.float64,
    )

    for feature, weight in weights.items():

        matrix = feature_matrices[feature]

        total += (
            weight
            * cosine_similarity(
                matrix[idx],
                matrix,
            ).ravel()
        )

    # Exclude the selected movie itself.
    total[idx] = -1.0

    count = min(
        len(df) - 1,
        max(top_n, pool_size),
    )

    if count <= 0:
        return (
            df.iloc[:0].copy(),
            weights,
            information,
        )

    indices = np.argpartition(
        total,
        -count,
    )[-count:]

    results = df.iloc[indices].copy()

    results["adaptive_similarity"] = total[indices]

    results = (
        results.sort_values(
            ["adaptive_similarity", "id"],
            ascending=[False, True],
        )
        .head(top_n)
    )

    return results, weights, information


def display(title, results, score_column):

    print(f"\n{title}")

    if results is None or results.empty:
        print("No results")
        return

    print(
        results[
            ["title", "vote_count", score_column]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )


def main():

    print(
        "\nCINEMATCH: QUERY-ADAPTIVE "
        "SIMILARITY EXPERIMENT"
    )

    print(
        "This experiment compares rankings; "
        "it does not measure relevance accuracy."
    )

    for name, movie_id in TEST_MOVIES.items():

        print("\n" + "=" * 75)
        print(name)
        print("=" * 75)

        adaptive, weights, info = (
            adaptive_recommendations(movie_id)
        )

        if adaptive is None:
            print("Movie not in catalog")
            continue

        print("\nQuery-specific feature weights:")

        for feature in FEATURE_WEIGHTS:

            print(
                f"  {feature:10s} "
                f"baseline={FEATURE_WEIGHTS[feature]:.1%} "
                f"adaptive={weights[feature]:.1%} "
                f"informativeness={info[feature]:.3f}"
            )

        fixed = recommend_movies(
            movie_id,
            top_n=TOP_N,
            ranking_mode="pure_similarity",
        )

        evidence = recommend_movies(
            movie_id,
            top_n=TOP_N,
            ranking_mode="evidence_aware",
        )

        display(
            "FIXED CONTENT SIMILARITY",
            fixed,
            "similarity",
        )

        display(
            "ADAPTIVE CONTENT SIMILARITY",
            adaptive,
            "adaptive_similarity",
        )

        display(
            "EVIDENCE-AWARE RERANKING (EXISTING)",
            evidence,
            "final_score",
        )

    print(
        "\nCompare rankings manually. "
        "Do not treat these scores as match probabilities."
    )


if __name__ == "__main__":
    main()
