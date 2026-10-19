
"""CineMatch evaluation benchmark.

Commands:
    python3 src/evaluation_benchmark.py generate
    python3 src/evaluation_benchmark.py score

Generate:
    Runs four recommendation engines on fixed movie anchors.
    Produces rankings, a blinded human-review sheet,
    run diagnostics, and model-overlap statistics.

Score:
    Calculates human-rated Precision@10, mean relevance,
    and two versions of NDCG@10.

Existing output files are never overwritten by generate.
"""

from pathlib import Path
from itertools import combinations
from time import perf_counter
import argparse
import hashlib

import numpy as np
import pandas as pd

import weighted_recommender as classic


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "data" / "evaluation"

RANKINGS_PATH = OUTPUT_DIR / "model_rankings.csv"
REVIEWS_PATH = OUTPUT_DIR / "human_reviews.csv"
METRICS_PATH = OUTPUT_DIR / "model_metrics.csv"
RUNS_PATH = OUTPUT_DIR / "model_runs.csv"
OVERLAP_PATH = OUTPUT_DIR / "model_overlap.csv"

TOP_K = 10

ANCHORS = [
    ("The Batman", 414906),
    ("Inception", 27205),
    ("Interstellar", 157336),
    ("The Conjuring", 138843),
    ("La La Land", 313369),
    ("Get Out", 419430),
    ("Knives Out", 546554),
]

ADDITIONAL_TITLES = [
    ("Arrival", 2016),
    ("Whiplash", 2014),
    ("Parasite", 2019),
    ("Gone Girl", 2014),
    ("The Grand Budapest Hotel", 2014),
    ("Inside Out", 2015),
    ("Her", 2013),
    ("Mad Max: Fury Road", 2015),
]

MODELS = [
    "Classic",
    "Hybrid",
    "LightGBM",
    "RRF Ensemble",
]

REVIEW_COLUMNS = [
    "narrative_relevance",
    "experience_relevance",
    "reliability",
    "review_notes",
]


# --------------------------------------------------
# MOVIE IDENTIFICATION
# --------------------------------------------------

def movie_year(value):
    if pd.isna(value):
        return None

    text = str(value)

    if len(text) >= 4 and text[:4].isdigit():
        return int(text[:4])

    return None


def resolve_anchors():
    """Find the fixed evaluation movies in the catalog."""

    catalog = classic.df
    resolved = list(ANCHORS)

    for title, year in ADDITIONAL_TITLES:
        matches = catalog[
            catalog["title"].fillna("").str.casefold()
            == title.casefold()
        ].copy()

        matches = matches[
            matches["release_date"].map(movie_year)
            == year
        ]

        if matches.empty:
            raise ValueError(
                f"Evaluation anchor not found: "
                f"{title} ({year})"
            )

        matches = matches.sort_values(
            ["vote_count", "id"],
            ascending=[False, True],
        )

        resolved.append(
            (title, int(matches.iloc[0]["id"]))
        )

    missing = [
        (title, movie_id)
        for title, movie_id in resolved
        if movie_id not in classic.movie_id_to_index
    ]

    if missing:
        raise ValueError(
            f"Missing evaluation anchors: {missing}"
        )

    return resolved


def pair_id(anchor_id, candidate_id):
    """Stable identifier for a seed-recommendation pair."""

    raw = f"{anchor_id}:{candidate_id}"

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()[:16]


# --------------------------------------------------
# MODEL EXECUTION
# --------------------------------------------------

def run_model(
    model_name,
    anchor_id,
    embeddings,
    ranker,
    hybrid,
    rrf,
):
    """Execute one existing recommendation engine."""

    if model_name == "Classic":
        return classic.recommend_movies(
            anchor_id,
            top_n=TOP_K,
            ranking_mode="evidence_aware",
        )

    if model_name == "Hybrid":
        return hybrid.recommend_movies(
            anchor_id,
            top_n=TOP_K,
            embeddings=embeddings,
        )

    if model_name == "LightGBM":
        return ranker.recommend_movies(
            anchor_id,
            top_n=TOP_K,
            embeddings=embeddings,
        )

    if model_name == "RRF Ensemble":
        return rrf.recommend_movies(
            anchor_id,
            top_n=TOP_K,
            embeddings=embeddings,
            experimental_ranker=ranker,
            ranking_mode="evidence_aware",
        )

    raise ValueError(
        f"Unknown model: {model_name}"
    )


def ranking_source(model_name, results):
    """Identify the actual ranking mechanism used."""

    if (
        results is not None
        and not results.empty
        and "ranking_source" in results.columns
    ):
        return str(results.iloc[0]["ranking_source"])

    return model_name


# --------------------------------------------------
# GENERATE BENCHMARK
# --------------------------------------------------

def generate():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    protected_paths = [
        RANKINGS_PATH,
        REVIEWS_PATH,
        RUNS_PATH,
        OVERLAP_PATH,
    ]

    existing = [
        path
        for path in protected_paths
        if path.exists()
    ]

    if existing:
        raise FileExistsError(
            "Existing evaluation files detected:\n"
            + "\n".join(str(path) for path in existing)
            + "\nBack them up or move them before "
            "generating a new benchmark."
        )

    import hybrid_engine as hybrid
    import rrf_ensemble as rrf
    from experimental_lgbm_engine import load_ranker

    print("Loading semantic embeddings...")
    embeddings = hybrid.load_embeddings()

    print("Loading experimental LightGBM ranker...")
    ranker = load_ranker()

    anchors = resolve_anchors()

    ranking_rows = []
    review_rows = {}
    run_rows = []
    overlap_rows = []

    for anchor_name, anchor_id in anchors:
        print(f"\nEvaluating: {anchor_name}")

        outputs = {}

        for model_name in MODELS:
            start = perf_counter()

            try:
                results = run_model(
                    model_name,
                    anchor_id,
                    embeddings,
                    ranker,
                    hybrid,
                    rrf,
                )

                elapsed = perf_counter() - start

                if results is None or results.empty:
                    raise ValueError(
                        "No recommendations returned."
                    )

                if len(results) < TOP_K:
                    raise ValueError(
                        f"Only {len(results)} recommendations "
                        f"returned; expected {TOP_K}."
                    )

                source = ranking_source(
                    model_name,
                    results,
                )

                outputs[model_name] = results.head(
                    TOP_K
                ).copy()

                run_rows.append({
                    "anchor_id": anchor_id,
                    "anchor_title": anchor_name,
                    "model": model_name,
                    "status": "success",
                    "ranking_source": source,
                    "native_lightgbm": (
                        source == "LightGBM v4 Experimental"
                        if model_name == "LightGBM"
                        else np.nan
                    ),
                    "latency_seconds": elapsed,
                    "error": "",
                })

                print(
                    f"  {model_name}: success "
                    f"({elapsed:.3f}s) [{source}]"
                )

            except Exception as exc:
                elapsed = perf_counter() - start

                run_rows.append({
                    "anchor_id": anchor_id,
                    "anchor_title": anchor_name,
                    "model": model_name,
                    "status": "error",
                    "ranking_source": "",
                    "native_lightgbm": np.nan,
                    "latency_seconds": elapsed,
                    "error": (
                        f"{type(exc).__name__}: {exc}"
                    ),
                })

                print(
                    f"  {model_name}: ERROR — {exc}"
                )

        # Save each model's ranking.
        for model_name, results in outputs.items():
            source = ranking_source(
                model_name,
                results,
            )

            for rank, (_, movie) in enumerate(
                results.iterrows(),
                start=1,
            ):
                candidate_id = int(movie["id"])

                identifier = pair_id(
                    anchor_id,
                    candidate_id,
                )

                ranking_rows.append({
                    "pair_id": identifier,
                    "anchor_id": anchor_id,
                    "anchor_title": anchor_name,
                    "model": model_name,
                    "ranking_source": source,
                    "rank": rank,
                    "candidate_id": candidate_id,
                    "candidate_title": movie["title"],
                    "model_score": float(
                        movie["final_score"]
                    ),
                })

                # Only one human judgment per movie pair,
                # even if multiple models recommend it.
                if identifier not in review_rows:
                    overview = movie.get(
                        "overview", ""
                    )

                    if pd.isna(overview):
                        overview = ""

                    review_rows[identifier] = {
                        "pair_id": identifier,
                        "anchor_id": anchor_id,
                        "anchor_title": anchor_name,
                        "candidate_id": candidate_id,
                        "candidate_title": movie["title"],
                        "candidate_year": movie_year(
                            movie.get("release_date")
                        ),
                        "candidate_genres": movie.get(
                            "genres", ""
                        ),
                        "candidate_overview": str(
                            overview
                        )[:800],
                        "vote_count": movie.get(
                            "vote_count", 0
                        ),
                        "narrative_relevance": "",
                        "experience_relevance": "",
                        "reliability": "",
                        "review_notes": "",
                    }

        # Compare recommendation sets for this anchor.
        for model_a, model_b in combinations(
            outputs,
            2,
        ):
            ids_a = set(
                outputs[model_a]["id"].astype(int)
            )

            ids_b = set(
                outputs[model_b]["id"].astype(int)
            )

            intersection = ids_a & ids_b
            union = ids_a | ids_b

            overlap_rows.append({
                "anchor_id": anchor_id,
                "anchor_title": anchor_name,
                "model_a": model_a,
                "model_b": model_b,
                "shared_at_10": len(intersection),
                "jaccard_at_10": (
                    len(intersection) / len(union)
                    if union else np.nan
                ),
            })

    rankings = pd.DataFrame(ranking_rows)

    reviews = pd.DataFrame(
        list(review_rows.values())
    )

    runs = pd.DataFrame(run_rows)
    overlaps = pd.DataFrame(overlap_rows)

    if not reviews.empty:
        reviews = reviews.sort_values(
            [
                "anchor_title",
                "candidate_title",
                "candidate_id",
            ]
        ).reset_index(drop=True)

    # Write the four outputs only after all evaluation
    # attempts have finished.
    rankings.to_csv(
        RANKINGS_PATH,
        index=False,
    )

    reviews.to_csv(
        REVIEWS_PATH,
        index=False,
    )

    runs.to_csv(
        RUNS_PATH,
        index=False,
    )

    overlaps.to_csv(
        OVERLAP_PATH,
        index=False,
    )

    print("\n" + "=" * 60)
    print("BENCHMARK GENERATION COMPLETE")
    print("=" * 60)

    print(f"Anchors: {len(anchors)}")
    print(f"Successful runs: {len(outputs) if False else (runs['status'] == 'success').sum()}")
    print(f"Failed runs: {(runs['status'] == 'error').sum()}")
    print(f"Ranking rows: {len(rankings)}")
    print(f"Unique review pairs: {len(reviews)}")

    if not overlaps.empty:
        print("\nAverage Jaccard overlap@10:")

        print(
            overlaps.groupby(
                ["model_a", "model_b"]
            )["jaccard_at_10"]
            .mean()
            .round(3)
            .to_string()
        )

    print(f"\nResults saved to: {OUTPUT_DIR}")

    print(
        "\nNext: Complete narrative_relevance and "
        "experience_relevance in human_reviews.csv."
    )


# --------------------------------------------------
# RANKING METRICS
# --------------------------------------------------

def dcg_at_k(relevances):
    relevances = np.asarray(
        relevances,
        dtype=float,
    )

    gains = np.power(
        2.0,
        relevances,
    ) - 1.0

    discounts = np.log2(
        np.arange(2, len(relevances) + 2)
    )

    return float(
        np.sum(gains / discounts)
    )


def ndcg_at_k(relevances, ideal_relevances=None):
    """NDCG with an optional shared judged ideal."""

    actual = dcg_at_k(relevances)

    if ideal_relevances is None:
        ideal_relevances = relevances

    ideal = dcg_at_k(
        sorted(
            ideal_relevances,
            reverse=True,
        )[:len(relevances)]
    )

    if ideal <= 0:
        return 0.0

    return actual / ideal


def validate_ratings(reviews):
    required = [
        "narrative_relevance",
        "experience_relevance",
    ]

    for column in required:
        values = pd.to_numeric(
            reviews[column],
            errors="coerce",
        )

        if values.isna().any():
            raise ValueError(
                f"{column} contains missing or "
                "invalid judgments."
            )

        if not values.isin([0, 1, 2, 3]).all():
            raise ValueError(
                f"{column} must contain integers 0–3."
            )

        reviews[column] = values.astype(int)

    reliability = pd.to_numeric(
        reviews["reliability"],
        errors="coerce",
    )

    if not (
        reliability.isna()
        | reliability.isin([0, 1, 2])
    ).all():
        raise ValueError(
            "Reliability must be blank or 0–2."
        )

    reviews["reliability"] = reliability

    return reviews


# --------------------------------------------------
# HUMAN EVALUATION SCORING
# --------------------------------------------------

def score():
    if not RANKINGS_PATH.exists():
        raise FileNotFoundError(
            "Generate the benchmark first."
        )

    if not REVIEWS_PATH.exists():
        raise FileNotFoundError(
            "Human review sheet not found."
        )

    rankings = pd.read_csv(
        RANKINGS_PATH,
        dtype={"pair_id": str},
    )

    reviews = pd.read_csv(
        REVIEWS_PATH,
        dtype={"pair_id": str},
    )

    if rankings.empty or reviews.empty:
        raise ValueError(
            "No rankings or reviews available."
        )

    if reviews["pair_id"].duplicated().any():
        raise ValueError(
            "Duplicate pair IDs in review sheet."
        )

    reviews = validate_ratings(reviews)

    merged = rankings.merge(
        reviews[
            [
                "pair_id",
                "narrative_relevance",
                "experience_relevance",
                "reliability",
            ]
        ],
        on="pair_id",
        how="left",
        validate="many_to_one",
    )

    if merged[
        [
            "narrative_relevance",
            "experience_relevance",
        ]
    ].isna().any().any():
        raise ValueError(
            "Some ranked pairs lack human judgments."
        )

    merged["combined_relevance"] = (
        merged["narrative_relevance"]
        + merged["experience_relevance"]
    ) / 2.0

    # The shared ideal is based on all judged movies
    # retrieved by any model for the same anchor.
    # It is not an ideal ranking of the full catalog.
    judged_ideals = {}

    for anchor_id, group in merged.groupby(
        "anchor_id"
    ):
        unique_pairs = group.drop_duplicates(
            "pair_id"
        )

        judged_ideals[anchor_id] = (
            unique_pairs["combined_relevance"]
            .sort_values(ascending=False)
            .head(TOP_K)
            .to_numpy()
        )

    rows = []

    for (model, anchor_id), group in merged.groupby(
        ["model", "anchor_id"]
    ):
        group = group.sort_values("rank")

        if len(group) != TOP_K:
            raise ValueError(
                "Incomplete model ranking detected."
            )

        relevance = group[
            "combined_relevance"
        ].to_numpy(dtype=float)

        rows.append({
            "model": model,
            "anchor_id": anchor_id,
            "anchor_title": group.iloc[0][
                "anchor_title"
            ],
            "ranking_source": group.iloc[0].get(
                "ranking_source",
                model,
            ),
            "precision_at_10": float(
                np.mean(relevance >= 2.0)
            ),
            "ndcg_local_at_10": ndcg_at_k(
                relevance
            ),
            "ndcg_shared_at_10": ndcg_at_k(
                relevance,
                judged_ideals[anchor_id],
            ),
            "mean_relevance": float(
                np.mean(relevance)
            ),
            "mean_reliability": (
                float(group["reliability"].mean())
                if group["reliability"].notna().any()
                else np.nan
            ),
        })

    metrics = pd.DataFrame(rows)

    metrics.to_csv(
        METRICS_PATH,
        index=False,
    )

    summary = (
        metrics.groupby("model")[
            [
                "precision_at_10",
                "ndcg_local_at_10",
                "ndcg_shared_at_10",
                "mean_relevance",
                "mean_reliability",
            ]
        ]
        .mean()
        .round(3)
    )

    print("\nCINEMATCH HUMAN EVALUATION")
    print("=" * 60)

    print(summary.to_string())

    print(f"\nResults saved to: {METRICS_PATH}")

    print(
        "\nNote: Shared NDCG uses the union of "
        "judged recommendations for each anchor, "
        "not the entire movie catalog."
    )

    print(
        "LightGBM fallback results must be interpreted "
        "separately from native LightGBM rankings."
    )


# --------------------------------------------------
# COMMAND-LINE INTERFACE
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CineMatch evaluation benchmark"
    )

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
