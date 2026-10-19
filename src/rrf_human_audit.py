
"""CineMatch — blinded RRF human weight audit.

Run from the project root:

    python3 src/rrf_human_audit.py generate
    python3 src/rrf_human_audit.py score

This is an exploratory validation experiment, not a
fully independent final test.

It does not modify the live recommendation engines or
overwrite existing evaluation files.
"""

from pathlib import Path
from time import perf_counter
import argparse
import hashlib
import random

import numpy as np
import pandas as pd

import weighted_recommender as classic
import hybrid_engine as hybrid
import rrf_ensemble as rrf
from experimental_lgbm_engine import load_ranker
from evaluation_benchmark import resolve_anchors


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "evaluation" / "rrf_human_audit"

RANKINGS_PATH = OUT / "rankings.csv"
REVIEWS_PATH = OUT / "reviews.csv"
SEEDS_PATH = OUT / "seeds.csv"
METRICS_PATH = OUT / "metrics.csv"

TOP_K = 10
POOL_SIZE = 100
RANDOM_SEED = 42
SUPPORTED_SEEDS = 8
UNSUPPORTED_SEEDS = 4

CONFIGS = {
    "Equal": {
        "classic": 1.0,
        "hybrid": 1.0,
        "experimental": 1.0,
    },
    "Mild": {
        "classic": 1.0,
        "hybrid": 1.0,
        "experimental": 1.25,
    },
    "Moderate": {
        "classic": 1.0,
        "hybrid": 1.0,
        "experimental": 1.5,
    },
}

REVIEW_COLUMNS = [
    "narrative_relevance",
    "experience_relevance",
    "review_notes",
]


def pair_id(anchor_id, candidate_id):
    raw = f"{anchor_id}:{candidate_id}"

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()[:16]


def year_of(value):
    if pd.isna(value):
        return ""

    text = str(value)
    return text[:4] if text[:4].isdigit() else ""


def select_seeds(ranker):
    """Select new seeds, excluding previous benchmark anchors."""

    excluded = {
        movie_id
        for _, movie_id in resolve_anchors()
    }

    catalog_ids = set(
        classic.df["id"].astype(int)
    ) - excluded

    supported = sorted(
        catalog_ids & set(ranker.supported_ids)
    )

    unsupported = sorted(
        catalog_ids - set(ranker.supported_ids)
    )

    rng = random.Random(RANDOM_SEED)

    if len(supported) < SUPPORTED_SEEDS:
        raise ValueError(
            "Not enough supported evaluation seeds."
        )

    if len(unsupported) < UNSUPPORTED_SEEDS:
        raise ValueError(
            "Not enough unsupported evaluation seeds."
        )

    chosen = [
        (movie_id, "supported")
        for movie_id in rng.sample(
            supported,
            SUPPORTED_SEEDS,
        )
    ]

    chosen += [
        (movie_id, "unsupported")
        for movie_id in rng.sample(
            unsupported,
            UNSUPPORTED_SEEDS,
        )
    ]

    rng.shuffle(chosen)

    catalog = classic.df.set_index("id")

    return [
        {
            "anchor_id": movie_id,
            "anchor_title": str(
                catalog.loc[movie_id, "title"]
            ),
            "anchor_year": year_of(
                catalog.loc[movie_id, "release_date"]
            ),
            "seed_group": group,
        }
        for movie_id, group in chosen
    ]


def get_rankings(movie_id, embeddings, ranker):
    """Reproduce the live RRF candidate retrieval."""

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
        experimental = ranker.recommend_movies(
            movie_id,
            top_n=POOL_SIZE,
            embeddings=embeddings,
        )

        if (
            experimental is not None
            and not experimental.empty
            and "ranking_source" in experimental.columns
        ):
            native = experimental[
                experimental["ranking_source"]
                == "LightGBM v4 Experimental"
            ]

            experimental_ids = (
                rrf.normalize_recommendations(
                    native,
                    movie_id,
                )
            )

            if experimental_ids:
                rankings["experimental"] = (
                    experimental_ids
                )

    return rankings


def generate():
    OUT.mkdir(parents=True, exist_ok=True)

    protected = [
        RANKINGS_PATH,
        REVIEWS_PATH,
        SEEDS_PATH,
    ]

    existing = [
        path for path in protected if path.exists()
    ]

    if existing:
        raise FileExistsError(
            "Existing audit files found. "
            "Move or back them up before regenerating:\n"
            + "\n".join(map(str, existing))
        )

    print("Loading semantic embeddings...")
    embeddings = hybrid.load_embeddings()

    print("Loading LightGBM...")
    ranker = load_ranker()

    seeds = select_seeds(ranker)

    catalog = classic.df.set_index("id")

    ranking_rows = []
    review_rows = {}

    for index, seed in enumerate(seeds, start=1):
        anchor_id = seed["anchor_id"]

        print(
            f"[{index}/{len(seeds)}] "
            f"{seed['anchor_title']} "
            f"({seed['seed_group']})"
        )

        start = perf_counter()

        rankings = get_rankings(
            anchor_id,
            embeddings,
            ranker,
        )

        print(
            "  Participating models:",
            ", ".join(rankings),
        )

        for config_name, weights in CONFIGS.items():
            fused = rrf.fuse_rankings(
                rankings,
                top_n=TOP_K,
                weights=weights,
            )

            if len(fused) != TOP_K:
                raise ValueError(
                    f"{config_name} returned "
                    f"{len(fused)} recommendations "
                    f"for {seed['anchor_title']}."
                )

            for rank, row in enumerate(
                fused.itertuples(index=False),
                start=1,
            ):
                candidate_id = int(row.id)

                identifier = pair_id(
                    anchor_id,
                    candidate_id,
                )

                movie = catalog.loc[candidate_id]

                ranking_rows.append({
                    "pair_id": identifier,
                    "anchor_id": anchor_id,
                    "anchor_title": seed["anchor_title"],
                    "seed_group": seed["seed_group"],
                    "configuration": config_name,
                    "rank": rank,
                    "candidate_id": candidate_id,
                    "rrf_score": float(row.rrf_score),
                    "model_support": int(
                        row.model_support
                    ),
                })

                if identifier not in review_rows:
                    overview = movie.get(
                        "overview", ""
                    )

                    if pd.isna(overview):
                        overview = ""

                    review_rows[identifier] = {
                        "pair_id": identifier,
                        "anchor_id": anchor_id,
                        "anchor_title": seed["anchor_title"],
                        "anchor_year": seed["anchor_year"],
                        "candidate_id": candidate_id,
                        "candidate_title": str(
                            movie["title"]
                        ),
                        "candidate_year": year_of(
                            movie.get("release_date")
                        ),
                        "candidate_genres": movie.get(
                            "genres", ""
                        ),
                        "candidate_overview": str(
                            overview
                        )[:800],
                        "narrative_relevance": "",
                        "experience_relevance": "",
                        "review_notes": "",
                    }

        print(
            f"  Finished in "
            f"{perf_counter() - start:.2f}s"
        )

    rankings_df = pd.DataFrame(ranking_rows)

    reviews_df = pd.DataFrame(
        list(review_rows.values())
    ).sort_values(
        ["anchor_title", "candidate_title", "candidate_id"]
    )

    seeds_df = pd.DataFrame(seeds)

    # Exclusive creation: never silently overwrite files.
    rankings_df.to_csv(
        RANKINGS_PATH,
        index=False,
        mode="x",
    )

    reviews_df.to_csv(
        REVIEWS_PATH,
        index=False,
        mode="x",
    )

    seeds_df.to_csv(
        SEEDS_PATH,
        index=False,
        mode="x",
    )

    print("\nAUDIT GENERATED")
    print("=" * 60)
    print("Seeds:", len(seeds_df))
    print("Ranking rows:", len(rankings_df))
    print("Unique pairs to review:", len(reviews_df))

    # Show how much the three configurations overlap.
    print("\nUnique review pairs by seed:")
    print(
        reviews_df.groupby("anchor_title")
        .size()
        .to_string()
    )

    print("\nReview sheet:")
    print(REVIEWS_PATH)


def dcg(values):
    values = np.asarray(values, dtype=float)

    gains = np.power(2.0, values) - 1.0

    discounts = np.log2(
        np.arange(2, len(values) + 2)
    )

    return float(np.sum(gains / discounts))


def score():
    if not RANKINGS_PATH.exists():
        raise FileNotFoundError(
            "Generate the audit first."
        )

    if not REVIEWS_PATH.exists():
        raise FileNotFoundError(
            "Review sheet is missing."
        )

    rankings = pd.read_csv(
        RANKINGS_PATH,
        dtype={"pair_id": str},
    )

    reviews = pd.read_csv(
        REVIEWS_PATH,
        dtype={"pair_id": str},
    )

    if reviews["pair_id"].duplicated().any():
        raise ValueError(
            "Duplicate review pair IDs."
        )

    for column in [
        "narrative_relevance",
        "experience_relevance",
    ]:
        values = pd.to_numeric(
            reviews[column],
            errors="coerce",
        )

        if values.isna().any():
            raise ValueError(
                f"{column} has "
                f"{values.isna().sum()} missing ratings."
            )

        if not values.isin([0, 1, 2, 3]).all():
            raise ValueError(
                f"{column} must contain integers 0–3."
            )

        reviews[column] = values.astype(int)

    reviews["relevance"] = (
        reviews["narrative_relevance"]
        + reviews["experience_relevance"]
    ) / 2.0

    merged = rankings.merge(
        reviews[["pair_id", "relevance"]],
        on="pair_id",
        how="left",
        validate="many_to_one",
    )

    if merged["relevance"].isna().any():
        raise ValueError(
            "Some recommendations lack judgments."
        )

    rows = []

    for anchor_id, anchor_data in merged.groupby(
        "anchor_id"
    ):
        # Shared judged candidate pool across configurations.
        ideal_values = (
            anchor_data.drop_duplicates("pair_id")[
                "relevance"
            ]
            .sort_values(ascending=False)
            .head(TOP_K)
            .to_numpy()
        )

        ideal_dcg = dcg(ideal_values)

        for config_name, group in anchor_data.groupby(
            "configuration"
        ):
            group = group.sort_values("rank")

            if len(group) != TOP_K:
                raise ValueError(
                    "Incomplete ranking detected."
                )

            relevance = group[
                "relevance"
            ].to_numpy()

            rows.append({
                "anchor_id": anchor_id,
                "anchor_title": group.iloc[0][
                    "anchor_title"
                ],
                "seed_group": group.iloc[0][
                    "seed_group"
                ],
                "configuration": config_name,
                "precision_at_10": float(
                    np.mean(relevance >= 2)
                ),
                "mean_relevance": float(
                    np.mean(relevance)
                ),
                "ndcg_shared_at_10": (
                    dcg(relevance) / ideal_dcg
                    if ideal_dcg > 0 else 0.0
                ),
            })

    metrics = pd.DataFrame(rows)

    if METRICS_PATH.exists():
        raise FileExistsError(
            f"{METRICS_PATH} already exists. "
            "Move it before rescoring."
        )

    metrics.to_csv(
        METRICS_PATH,
        index=False,
        mode="x",
    )

    print("\nRRF HUMAN WEIGHT AUDIT")
    print("=" * 60)

    for group_name, group in metrics.groupby(
        "seed_group"
    ):
        print(f"\n{group_name.upper()} SEEDS")

        summary = (
            group.groupby("configuration")[
                [
                    "precision_at_10",
                    "mean_relevance",
                    "ndcg_shared_at_10",
                ]
            ]
            .mean()
            .round(3)
        )

        print(summary.to_string())

    print("\nResults saved to:", METRICS_PATH)
    print(
        "\nShared NDCG uses the union of judged "
        "recommendations, not the full catalog."
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "command",
        choices=["generate", "score"],
    )

    args = parser.parse_args()

    if args.command == "generate":
        generate()
    else:
        score()


if __name__ == "__main__":
    main()
