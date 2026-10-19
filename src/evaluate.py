
"""
Reproducible qualitative comparison.

This is NOT a statistical accuracy test.
"""

from pathlib import Path

import pandas as pd

from weighted_recommender import (
    df,
    recommend_movies,
)


# --------------------------------------------------
# EVALUATION CONFIGURATION
# --------------------------------------------------

TEST_MOVIES = {
    "Inception": 27205,
    "Interstellar": 157336,
    "The Conjuring": 138843,
    "La La Land": 313369,
    "Get Out": 419430,
    "Knives Out": 546554,
}

MODES = [
    "pure_similarity",
    "fixed_quality",
    "evidence_aware",
]


# --------------------------------------------------
# EVALUATION FUNCTION
# --------------------------------------------------

def run():

    print(
        f"Catalog: {len(df):,} movies"
    )

    records = []

    for title, movie_id in TEST_MOVIES.items():

        for mode in MODES:

            result = recommend_movies(
                movie_id,
                top_n=5,
                ranking_mode=mode
            )

            if result is None:

                print(
                    f"Skipping unavailable title: {title}"
                )

                continue

            for rank, row in enumerate(
                result.itertuples(index=False),
                start=1
            ):

                records.append(
                    {
                        "seed_title": title,
                        "ranking_mode": mode,
                        "rank": rank,
                        "movie_id": row.id,
                        "recommendation": row.title,
                        "similarity": row.similarity,
                        "specificity": row.specificity,
                        "rating_confidence": row.rating_confidence,
                        "final_score": row.final_score,
                    }
                )

    output = pd.DataFrame(records)

    path = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "evaluation_comparison.csv"
    )

    output.to_csv(
        path,
        index=False
    )

    print(
        output[
            output["rank"] == 1
        ][
            [
                "seed_title",
                "ranking_mode",
                "recommendation",
            ]
        ].to_string(index=False)
    )

    print(
        f"\nSaved comparison to {path}"
    )

    print(
        "Interpret rankings manually. "
        "No relevance labels exist, so do not "
        "report precision or accuracy."
    )


# --------------------------------------------------
# RUN
# --------------------------------------------------

if __name__ == "__main__":
    run()
