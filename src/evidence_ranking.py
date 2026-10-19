
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

FIXED_QUALITY_WEIGHT = 0.15

# Initial experimental values, not optimized.
GENRE_DISCOUNT = 0.15
ADAPTIVE_QUALITY_WEIGHT = 0.30


# --------------------------------------------------
# GENERATE CANDIDATES
# --------------------------------------------------

def get_candidates(movie_id):

    if movie_id not in movie_id_to_index:
        return None

    movie_index = movie_id_to_index[movie_id]

    total_similarity = np.zeros(
        len(df),
        dtype=np.float64
    )

    feature_scores = {}

    for feature, weight in FEATURE_WEIGHTS.items():

        matrix = feature_matrices[feature]

        similarities = cosine_similarity(
            matrix[movie_index],
            matrix
        ).ravel()

        contribution = similarities * weight

        feature_scores[feature] = contribution

        total_similarity += contribution

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
            feature_scores[feature][candidate_indices]
        )

    return candidates


# --------------------------------------------------
# RANKING STRATEGIES
# --------------------------------------------------

def rank_pure_similarity(candidates):

    results = candidates.copy()

    results["final_score"] = results["similarity"]

    return results.sort_values(
        ["final_score", "id"],
        ascending=[False, True]
    )


def rank_fixed_quality(candidates):

    results = candidates.copy()

    results["final_score"] = (
        results["similarity"]
        + FIXED_QUALITY_WEIGHT * results["quality_bonus"]
    )

    return results.sort_values(
        ["final_score", "similarity", "id"],
        ascending=[False, False, True]
    )


def rank_evidence_aware(
    candidates,
    genre_discount=GENRE_DISCOUNT,
    quality_weight=ADAPTIVE_QUALITY_WEIGHT,
):

    results = candidates.copy()

    non_genre_features = [
        "overview",
        "keywords",
        "directors",
        "cast",
    ]

    results["specific_similarity"] = sum(
        results[f"{feature}_contribution"]
        for feature in non_genre_features
    )

    results["specificity"] = (
        results["specific_similarity"]
        / results["similarity"].clip(lower=1e-10)
    ).clip(lower=0, upper=1)

    results["genre_dependence"] = (
        1 - results["specificity"]
    )

    # Bayesian confidence based on vote count.
    results["rating_confidence"] = (
        results["vote_count"]
        / (results["vote_count"] + m)
    )

    # Discount similarity that relies heavily on genre.
    results["adjusted_similarity"] = (
        results["similarity"]
        * (
            1
            - genre_discount
            * results["genre_dependence"]
        )
    )

    # Increase quality influence when the similarity
    # is primarily based on genres, but only when
    # the audience rating has substantial evidence.
    results["adaptive_quality_bonus"] = (
        quality_weight
        * results["rating_confidence"]
        * results["genre_dependence"]
        * results["quality_bonus"]
    )

    results["final_score"] = (
        results["adjusted_similarity"]
        + results["adaptive_quality_bonus"]
    )

    return results.sort_values(
        ["final_score", "similarity", "id"],
        ascending=[False, False, True]
    )


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

def print_ranking(label, results):

    print(f"\n{label}")
    print("-" * 95)

    columns = [
        "title",
        "vote_count",
        "similarity",
        "final_score",
    ]

    if "specificity" in results.columns:
        columns.extend(
            [
                "specificity",
                "rating_confidence",
            ]
        )

    print(
        results.head(TOP_N)[columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}"
        )
    )


def explain_top_recommendation(results):

    if results.empty:
        return

    movie = results.iloc[0]

    print("\nWhy this recommendation ranks first:")
    print(f"Movie: {movie['title']}")

    print(
        f"Similarity: {movie['similarity']:.4f}"
    )

    print(
        f"Specificity: {movie['specificity']:.2%}"
    )

    print(
        f"Genre dependence: "
        f"{movie['genre_dependence']:.2%}"
    )

    print(
        f"Rating confidence: "
        f"{movie['rating_confidence']:.2%}"
    )

    print(
        f"Adjusted similarity: "
        f"{movie['adjusted_similarity']:.4f}"
    )

    print(
        f"Adaptive quality bonus: "
        f"{movie['adaptive_quality_bonus']:.4f}"
    )

    print(
        f"Final score: {movie['final_score']:.4f}"
    )


# --------------------------------------------------
# RUN EXPERIMENT
# --------------------------------------------------

print("\n" + "=" * 95)
print("CINEMATCH EVIDENCE-AWARE RANKING EXPERIMENT")
print("=" * 95)

print(f"Movies in catalog: {len(df):,}")
print(f"Reference rating (C): {C:.3f}")
print(f"Confidence parameter (m): {m:.0f}")

print(
    f"Genre discount: {GENRE_DISCOUNT}"
)

print(
    f"Adaptive quality weight: "
    f"{ADAPTIVE_QUALITY_WEIGHT}"
)


for movie_title, movie_id in TEST_MOVIES.items():

    candidates = get_candidates(movie_id)

    if candidates is None:
        print(f"Movie not found: {movie_title}")
        continue

    print("\n" + "=" * 95)
    print(f"RECOMMENDATIONS FOR: {movie_title}")
    print("=" * 95)

    pure = rank_pure_similarity(candidates)

    fixed = rank_fixed_quality(candidates)

    evidence = rank_evidence_aware(candidates)

    print_ranking(
        "PURE SIMILARITY",
        pure
    )

    print_ranking(
        "FIXED QUALITY BONUS",
        fixed
    )

    print_ranking(
        "EVIDENCE-AWARE RANKING",
        evidence
    )

    explain_top_recommendation(evidence)


print("\nExperiment complete.")
