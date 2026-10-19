
from pathlib import Path
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from build_lgbm_content_features import parse_genres
from build_lgbm_semantic_features import (
    load_catalog,
    load_verified_embeddings,
)


ROOT = Path(__file__).resolve().parents[1]
ML = ROOT / "data" / "movielens"

K = 10
MAX_USERS = 2000
SEED = 42
POSITIVE = 4.0
POPULAR_CANDIDATES = 100

FEATURES = [
    "adjusted_affinity",
    "cosine_affinity",
    "shared_positive_users",
    "affinity_available",
    "candidate_positive_count",
    "tfidf_similarity",
    "genre_overlap",
    "year_difference",
    "candidate_year",
    "candidate_imdb_rating",
    "candidate_imdb_log_votes",
    "anchor_imdb_rating",
    "semantic_similarity",
]


def load_inputs():
    print("Loading MovieLens history and validation...")

    history = pd.read_csv(
        ML / "lgbm_train.csv.gz"
    )

    validation = pd.read_csv(
        ML / "lgbm_validation.csv.gz"
    )

    affinity = pd.read_csv(
        ML / "lgbm_history_affinity.csv"
    )

    universe = pd.read_csv(
        ML / "lgbm_history_universe.csv"
    )

    movie_catalog = pd.read_csv(
        ROOT / "data" / "movies_with_imdb.csv",
        low_memory=False,
    )

    return history, validation, affinity, universe, movie_catalog


def build_cases(history, validation, universe):
    supported = set(
        universe["cinematch_id"].astype(int)
    )

    positive_history = history[
        (history["rating"] >= POSITIVE)
        & (history["cinematch_id"].isin(supported))
    ]

    anchors = (
        positive_history
        .sort_values(
            ["userId", "timestamp", "cinematch_id"]
        )
        .drop_duplicates("userId", keep="last")
        [["userId", "cinematch_id"]]
        .rename(columns={"cinematch_id": "anchor_id"})
    )

    cases = validation[
        (validation["rating"] >= POSITIVE)
        & (validation["cinematch_id"].isin(supported))
    ].merge(
        anchors,
        on="userId",
        how="inner",
    )

    seen_pairs = pd.MultiIndex.from_frame(
        history[["userId", "cinematch_id"]]
    )

    case_pairs = pd.MultiIndex.from_frame(
        cases[["userId", "cinematch_id"]]
    )

    cases = cases.loc[
        ~case_pairs.isin(seen_pairs)
    ].copy()

    if cases["userId"].duplicated().any():
        raise ValueError(
            "Expected one held-out interaction per user."
        )

    if len(cases) > MAX_USERS:
        cases = cases.sample(
            n=MAX_USERS,
            random_state=SEED,
        )

    return cases.sort_values("userId")


def build_candidate_lists(
    cases, history, affinity, universe
):
    supported = set(
        universe["cinematch_id"].astype(int)
    )

    # Historical popularity is computed from the
    # same pre-validation interaction history.
    popularity_counts = (
        history[
            (history["rating"] >= POSITIVE)
            & (history["cinematch_id"].isin(supported))
        ]["cinematch_id"]
        .value_counts()
    )

    popular = [
        int(movie)
        for movie in popularity_counts.index
    ]

    # These affinity features were constructed from
    # feature-history interactions, not validation.
    affinity_lookup = {
        (int(row.anchor_id), int(row.candidate_id)): row
        for row in affinity.itertuples(index=False)
    }

    neighbors = {
        int(anchor): group.sort_values(
            "adjusted_affinity",
            ascending=False,
        )["candidate_id"].astype(int).tolist()
        for anchor, group in affinity.groupby("anchor_id")
    }

    users = set(cases["userId"])

    histories = (
        history[history["userId"].isin(users)]
        .groupby("userId")["cinematch_id"]
        .agg(set)
        .to_dict()
    )

    rows = []

    for case in cases.itertuples(index=False):
        user = int(case.userId)
        anchor = int(case.anchor_id)
        target = int(case.cinematch_id)

        seen = histories.get(user, set())

        affinity_candidates = [
            movie
            for movie in neighbors.get(anchor, [])
            if movie not in seen and movie != anchor
        ][:100]

        popular_candidates = [
            movie
            for movie in popular
            if movie not in seen and movie != anchor
        ][:POPULAR_CANDIDATES]

        # Same candidate pool for all ranking methods.
        candidates = list(dict.fromkeys(
            affinity_candidates + popular_candidates
        ))

        for candidate in candidates:
            affinity_row = affinity_lookup.get(
                (anchor, candidate)
            )

            rows.append({
                "userId": user,
                "anchor_id": anchor,
                "candidate_id": candidate,
                "target_id": target,
                "relevance": int(candidate == target),
                "adjusted_affinity": (
                    affinity_row.adjusted_affinity
                    if affinity_row is not None else np.nan
                ),
                "cosine_affinity": (
                    affinity_row.cosine_affinity
                    if affinity_row is not None else np.nan
                ),
                "shared_positive_users": (
                    affinity_row.shared_positive_users
                    if affinity_row is not None else np.nan
                ),
                "affinity_available": int(
                    affinity_row is not None
                ),
                "candidate_positive_count": (
                    popularity_counts.get(candidate, 0)
                ),
            })

    return pd.DataFrame(rows)


def add_content_features(pairs, movie_catalog):
    print("Reconstructing v3 content features...")

    # Reproduce the original TF-IDF fitting universe
    # from the v3 training-pair movie IDs.
    training_pairs = pd.read_csv(
        ML / "lgbm_training_pairs_v3.csv.gz",
        usecols=["anchor_id", "candidate_id"],
    )

    training_ids = set(
        training_pairs["anchor_id"]
    ) | set(
        training_pairs["candidate_id"]
    )

    catalog = movie_catalog.copy()
    catalog["id"] = pd.to_numeric(
        catalog["id"], errors="coerce"
    )
    catalog = catalog.dropna(subset=["id"])
    catalog["id"] = catalog["id"].astype("int64")
    catalog = catalog.drop_duplicates("id")

    # Preserve the original catalog ordering,
    # matching v2's TF-IDF construction.
    fitting_movies = catalog[
        catalog["id"].isin(training_ids)
    ].copy()

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=30000,
        min_df=2,
        ngram_range=(1, 2),
        dtype=np.float32,
    )

    fitting_movies["overview"] = (
        fitting_movies["overview"]
        .fillna("")
        .astype(str)
    )

    vectorizer.fit(
        fitting_movies["overview"]
    )

    catalog["overview"] = (
        catalog["overview"]
        .fillna("")
        .astype(str)
    )

    # Transform only movies needed for this evaluation.
    needed_ids = set(
        pairs["anchor_id"]
    ) | set(
        pairs["candidate_id"]
    )

    movies = catalog[
        catalog["id"].isin(needed_ids)
    ].copy()

    movie_ids = movies["id"].tolist()

    matrix = vectorizer.transform(
        movies["overview"]
    )

    positions = {
        movie_id: index
        for index, movie_id in enumerate(movie_ids)
    }

    a = pairs["anchor_id"].map(positions)
    c = pairs["candidate_id"].map(positions)

    if a.isna().any() or c.isna().any():
        raise ValueError(
            "Missing movie metadata for evaluation."
        )

    a = a.to_numpy(dtype=int)
    c = c.to_numpy(dtype=int)

    pairs["tfidf_similarity"] = np.asarray(
        matrix[a].multiply(matrix[c]).sum(axis=1)
    ).ravel()

    movies = movies.set_index("id")

    genres = (
        movies["genres"]
        .apply(parse_genres)
        .to_dict()
    )

    pairs["genre_overlap"] = [
        (
            len(genres[x] & genres[y])
            / len(genres[x] | genres[y])
            if genres[x] and genres[y] else np.nan
        )
        for x, y in zip(
            pairs["anchor_id"],
            pairs["candidate_id"],
        )
    ]

    years = pd.to_datetime(
        movies["release_date"],
        errors="coerce",
    ).dt.year

    pairs["candidate_year"] = (
        pairs["candidate_id"].map(years)
    )

    pairs["year_difference"] = (
        pairs["anchor_id"].map(years)
        - pairs["candidate_year"]
    ).abs()

    imdb = pd.to_numeric(
        movies["imdb_rating"],
        errors="coerce",
    )

    votes = pd.to_numeric(
        movies["imdb_votes"],
        errors="coerce",
    )

    pairs["candidate_imdb_rating"] = (
        pairs["candidate_id"].map(imdb)
    )

    pairs["anchor_imdb_rating"] = (
        pairs["anchor_id"].map(imdb)
    )

    pairs["candidate_imdb_log_votes"] = np.log1p(
        pairs["candidate_id"]
        .map(votes)
        .clip(lower=0)
    )

    return pairs


def add_semantic_features(pairs):
    print("Loading verified semantic embeddings...")

    catalog = load_catalog()
    embeddings = load_verified_embeddings(catalog)

    positions = {
        int(movie_id): index
        for index, movie_id in enumerate(catalog["id"])
    }

    anchor_idx = pairs["anchor_id"].map(positions)
    candidate_idx = pairs["candidate_id"].map(positions)

    if anchor_idx.isna().any() or candidate_idx.isna().any():
        raise ValueError(
            "Missing semantic embeddings for candidates."
        )

    a = np.asarray(
        embeddings[anchor_idx.to_numpy(dtype=int)],
        dtype=np.float32,
    )

    b = np.asarray(
        embeddings[candidate_idx.to_numpy(dtype=int)],
        dtype=np.float32,
    )

    dot = np.einsum("ij,ij->i", a, b)
    denominator = (
        np.linalg.norm(a, axis=1)
        * np.linalg.norm(b, axis=1)
    )

    pairs["semantic_similarity"] = np.divide(
        dot,
        denominator,
        out=np.full(len(pairs), np.nan),
        where=denominator > 0,
    )

    return pairs


def evaluate_rankings(pairs, universe_size):
    model = lgb.Booster(
        model_file=str(
            ROOT / "models" / "lgbm_ranker_v3.txt"
        )
    )

    if model.feature_name() != FEATURES:
        raise ValueError(
            "Saved LightGBM feature order differs "
            "from the evaluation feature order."
        )

    x = pairs[FEATURES].apply(
        pd.to_numeric,
        errors="coerce",
    )

    pairs["lgbm_score"] = model.predict(
        x,
        num_threads=4,
    )

    # Missing scores are placed last in the simple
    # ranking baselines.
    pairs["popularity_score"] = (
        pairs["candidate_positive_count"]
    )

    pairs["affinity_score"] = (
        pairs["adjusted_affinity"].fillna(-1)
    )

    pairs["imdb_score"] = (
        pairs["candidate_imdb_rating"].fillna(-1)
    )

    policies = {
        "Popularity": "popularity_score",
        "Audience affinity": "affinity_score",
        "IMDb-only": "imdb_score",
        "LightGBM v3": "lgbm_score",
    }

    total_users = pairs["userId"].nunique()

    print("\n" + "=" * 64)
    print("CINEMATCH — OFFLINE TOP-10 EVALUATION")
    print("=" * 64)

    print(f"Evaluated users: {total_users:,}")
    print(f"Candidate universe: {universe_size:,}")
    print(f"Candidate rows: {len(pairs):,}")

    recall = (
        pairs.groupby("userId")["relevance"]
        .max()
        .mean()
    )

    print(f"Candidate recall: {recall:.2%}")

    results = {}

    for name, score_column in policies.items():
        ranked = pairs.sort_values(
            ["userId", score_column, "candidate_id"],
            ascending=[True, False, True],
            kind="stable",
        )

        top = ranked.groupby(
            "userId", sort=False
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
            / universe_size
        )

        results[name] = {
            "hit_rate_at_10": hits / total_users,
            "ndcg_at_10": ndcg,
            "catalog_coverage_at_10": coverage,
            "hits": hits,
        }

    report = {
        "evaluated_users": int(total_users),
        "candidate_universe": int(universe_size),
        "candidate_recall": float(recall),
        "policies": results,
    }

    print(
        "\nPolicy               Hit@10    NDCG@10   Coverage"
    )
    print("-" * 58)

    for name, values in results.items():
        print(
            f"{name:<20}"
            f"{values['hit_rate_at_10']:>7.2%}   "
            f"{values['ndcg_at_10']:>7.4f}   "
            f"{values['catalog_coverage_at_10']:>7.2%}"
        )

    output = (
        ROOT / "models"
        / "lgbm_offline_evaluation.json"
    )

    output.write_text(
        json.dumps(report, indent=2)
    )

    print(f"\nSaved report: {output}")


def main():
    history, validation, affinity, universe, catalog = (
        load_inputs()
    )

    cases = build_cases(
        history, validation, universe
    )

    print(f"Eligible sampled users: {len(cases):,}")

    if cases.empty:
        raise RuntimeError(
            "No eligible validation users."
        )

    pairs = build_candidate_lists(
        cases, history, affinity, universe
    )

    if pairs.empty:
        raise RuntimeError(
            "No candidate lists generated."
        )

    if pairs["userId"].nunique() != len(cases):
        raise RuntimeError(
            "Some users have no candidates."
        )

    pairs = add_content_features(
        pairs, catalog
    )

    pairs = add_semantic_features(pairs)

    evaluate_rankings(
        pairs,
        universe_size=len(universe),
    )


if __name__ == "__main__":
    main()
