
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

from weighted_recommender import (
    df,
    feature_matrices,
    FEATURE_WEIGHTS,
    movie_id_to_index,
    C,
    m,
)


# --------------------------------------------------
# EXPERIMENT CONFIGURATION
# --------------------------------------------------

TEST_MOVIES = {
    "Inception": 27205,
    "Interstellar": 157336,
    "The Conjuring": 138843,
    "La La Land": 313369,
    "Get Out": 419430,
    "Knives Out": 546554,
}

CANDIDATE_POOL_SIZE = 100
TOP_N = 5

# These are experimental values, not optimized parameters.
SIMILARITY_TOLERANCES = [0.01, 0.03, 0.05]

QUALITY_WEIGHT = 0.15


# --------------------------------------------------
# GENERATE CANDIDATES AND FEATURE CONTRIBUTIONS
# --------------------------------------------------

def get_candidates(movie_id):

    if movie_id not in movie_id_to_index:
        return None

    movie_index = movie_id_to_index[movie_id]

    total_similarity = np.zeros(
        len(df),
        dtype=np.float64
    )

    contributions = {}

    for feature, weight in FEATURE_WEIGHTS.items():

        matrix = feature_matrices[feature]

        scores = cosine_similarity(
            matrix[movie_index],
            matrix
        ).ravel()

        weighted_scores = scores * weight

        contributions[feature] = weighted_scores

        total_similarity += weighted_scores

    # Never recommend the selected movie itself.
    total_similarity[movie_index] = -1

    pool_size = min(
        CANDIDATE_POOL_SIZE,
        len(df) - 1
    )

    candidate_indices = np.argsort(
        total_similarity
    )[::-1][:pool_size]

    candidates = df.iloc[
        candidate_indices
    ].copy()

    candidates["similarity"] = (
        total_similarity[candidate_indices]
    )

    for feature in FEATURE_WEIGHTS:
        candidates[f"{feature}_contribution"] = (
            contributions[feature][candidate_indices]
        )

    return candidates


# --------------------------------------------------
# RANKING STRATEGIES
# --------------------------------------------------

def pure_similarity_ranking(candidates):

    return candidates.sort_values(
        ["similarity", "id"],
        ascending=[False, True]
    ).copy()


def fixed_quality_ranking(candidates):

    results = candidates.copy()

    results["final_score"] = (
        results["similarity"]
        + QUALITY_WEIGHT * results["quality_bonus"]
    )

    return results.sort_values(
        ["final_score", "similarity", "id"],
        ascending=[False, False, True]
    )


def adaptive_ranking(candidates, tolerance):

    if tolerance < 0:
        raise ValueError(
            "Similarity tolerance cannot be negative."
        )

    remaining = pure_similarity_ranking(
        candidates
    ).copy()

    ranked_bands = []
    band_number = 1

    while not remaining.empty:

        # Anchor the band to the highest remaining
        # similarity, avoiding inconsistent pairwise
        # comparisons.
        anchor = remaining["similarity"].iloc[0]

        in_band = (
            remaining["similarity"]
            >= anchor - tolerance
        )

        band = remaining.loc[in_band].copy()

        remaining = remaining.loc[
            ~in_band
        ].copy()

        # Quality bonus comes from our Bayesian model.
        # Unrated movies receive zero bonus.
        band["adaptive_score"] = (
            band["similarity"]
            + QUALITY_WEIGHT * band["quality_bonus"]
        )

        band["similarity_band"] = band_number

        band = band.sort_values(
            ["adaptive_score", "similarity", "id"],
            ascending=[False, False, True]
        )

        ranked_bands.append(band)

        band_number += 1

    return pd.concat(
        ranked_bands,
        ignore_index=True
    )


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

def print_results(title, results):

    print(f"\n{title}")
    print("-" * 75)

    columns = [
        "title",
        "vote_count",
        "similarity",
        "quality_bonus",
    ]

    if "similarity_band" in results.columns:
        columns.append("similarity_band")

    print(
        results.head(TOP_N)[columns].to_string(
            index=False
        )
    )


def print_feature_explanation(results):

    print("\nFeature contributions for top recommendation:")

    if results.empty:
        return

    movie = results.iloc[0]

    print(f"Movie: {movie['title']}")

    for feature in FEATURE_WEIGHTS:

        contribution = movie[
            f"{feature}_contribution"
        ]

        print(
            f"  {feature:10s}: {contribution:.4f}"
        )

    print(
        f"  {'TOTAL':10s}: "
        f"{movie['similarity']:.4f}"
    )


# --------------------------------------------------
# RUN EXPERIMENT
# --------------------------------------------------

print("\n" + "=" * 75)
print("CINEMATCH ADAPTIVE RANKING EXPERIMENT")
print("=" * 75)

print(f"Movies in catalog: {len(df):,}")
print(f"Bayesian reference average: {C:.3f}")
print(f"Bayesian confidence parameter: {m:.0f}")
print(f"Candidate pool: {CANDIDATE_POOL_SIZE}")


for movie_title, movie_id in TEST_MOVIES.items():

    candidates = get_candidates(movie_id)

    if candidates is None:
        print(f"\nMovie not found: {movie_title}")
        continue

    print("\n" + "=" * 75)
    print(f"RECOMMENDATIONS FOR: {movie_title}")
    print("=" * 75)

    pure = pure_similarity_ranking(
        candidates
    )

    fixed = fixed_quality_ranking(
        candidates
    )

    print_results(
        "PURE SIMILARITY",
        pure
    )

    print_results(
        "CURRENT FIXED QUALITY BONUS",
        fixed
    )

    for tolerance in SIMILARITY_TOLERANCES:

        adaptive = adaptive_ranking(
            candidates,
            tolerance=tolerance
        )

        print_results(
            f"ADAPTIVE RANKING (tolerance={tolerance})",
            adaptive
        )

        # Show the explanation for the default
        # experimental tolerance.
        if tolerance == 0.03:
            print_feature_explanation(
                adaptive
            )


print("\nExperiment complete.")
