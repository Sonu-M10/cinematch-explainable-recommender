
"""CineMatch LightGBM v4 experimental recommendation engine.

Uses a selected movie as the recommendation anchor.
The ranker was trained with user histories, so this
single-anchor application is an experimental adaptation.

No model training or embedding generation occurs here.
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import hybrid_engine as hybrid
from build_lgbm_content_features import (
    parse_genres,
    genre_jaccard,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ML_DIR = DATA_DIR / "movielens"

MODEL_PATH = (
    ROOT / "models" / "lgbm_ranker_v4_experimental.txt"
)

AFFINITY_PATH = (
    ML_DIR / "lgbm_history_affinity.csv"
)

UNIVERSE_PATH = (
    ML_DIR / "lgbm_history_universe.csv"
)

TRAINING_PAIRS_PATH = (
    ML_DIR / "lgbm_training_pairs_v3.csv.gz"
)

CATALOG_PATH = (
    DATA_DIR / "movies_with_imdb.csv"
)

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


class ExperimentalRanker:
    def __init__(self):
        print("Loading CineMatch LightGBM v4...")

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Experimental model missing: {MODEL_PATH}"
            )

        self.model = lgb.Booster(
            model_file=str(MODEL_PATH)
        )

        if self.model.feature_name() != FEATURES:
            raise ValueError(
                "LightGBM feature order does not match "
                "the experimental inference pipeline."
            )

        self.affinity = pd.read_csv(
            AFFINITY_PATH
        )

        self.universe = pd.read_csv(
            UNIVERSE_PATH
        )

        self.supported_ids = set(
            self.universe["cinematch_id"].astype(int)
        )

        self.popularity = (
            self.universe.set_index(
                "cinematch_id"
            )["positive_rating_count"]
            .to_dict()
        )

        self.neighbors = {
            int(anchor): group.sort_values(
                "adjusted_affinity",
                ascending=False,
            ).copy()
            for anchor, group in self.affinity.groupby(
                "anchor_id"
            )
        }

        self.catalog = pd.read_csv(
            CATALOG_PATH,
            low_memory=False,
        )

        self.catalog["id"] = pd.to_numeric(
            self.catalog["id"],
            errors="coerce",
        )

        self.catalog = (
            self.catalog.dropna(subset=["id"])
            .drop_duplicates("id")
            .copy()
        )

        self.catalog["id"] = (
            self.catalog["id"].astype("int64")
        )

        self.catalog["overview"] = (
            self.catalog["overview"]
            .fillna("")
            .astype(str)
        )

        self.catalog["genre_set"] = (
            self.catalog["genres"].apply(parse_genres)
        )

        self.catalog["year"] = pd.to_datetime(
            self.catalog["release_date"],
            errors="coerce",
        ).dt.year

        self.catalog["imdb_rating"] = pd.to_numeric(
            self.catalog["imdb_rating"],
            errors="coerce",
        )

        self.catalog["imdb_votes"] = pd.to_numeric(
            self.catalog["imdb_votes"],
            errors="coerce",
        )

        self.catalog["imdb_log_votes"] = np.log1p(
            self.catalog["imdb_votes"].clip(lower=0)
        )

        self.catalog = self.catalog.set_index("id")

        # Reconstruct the exact training vocabulary.
        training_pairs = pd.read_csv(
            TRAINING_PAIRS_PATH,
            usecols=["anchor_id", "candidate_id"],
        )

        training_ids = (
            set(training_pairs["anchor_id"])
            | set(training_pairs["candidate_id"])
        )

        fitting_catalog = self.catalog.loc[
            self.catalog.index.isin(training_ids)
        ]

        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=30000,
            min_df=2,
            ngram_range=(1, 2),
            dtype=np.float32,
        )

        self.vectorizer = vectorizer

        self.vectorizer.fit(
            fitting_catalog["overview"]
        )

        # Transform the supported catalog once.
        supported_catalog = self.catalog.loc[
            self.catalog.index.isin(self.supported_ids)
        ]

        self.tfidf_ids = (
            supported_catalog.index.astype(int).tolist()
        )

        self.tfidf_positions = {
            movie_id: position
            for position, movie_id in enumerate(
                self.tfidf_ids
            )
        }

        self.tfidf = self.vectorizer.transform(
            supported_catalog["overview"]
        )

        # Hybrid already verifies semantic cache
        # fingerprint and catalog alignment.
        self.embeddings = hybrid.load_embeddings()

        self.semantic_positions = (
            hybrid.movie_id_to_index
        )

        print(
            "Experimental ranker ready: "
            f"{len(self.supported_ids):,} supported movies."
        )

    def supports(self, movie_id):
        return int(movie_id) in self.supported_ids

    def _content_features(self, anchor, candidates):
        anchor_metadata = self.catalog.loc[anchor]

        candidate_metadata = self.catalog.reindex(
            candidates
        )

        a_pos = self.tfidf_positions[anchor]

        c_pos = [
            self.tfidf_positions[candidate]
            for candidate in candidates
        ]

        similarities = np.asarray(
            self.tfidf[c_pos].multiply(
                self.tfidf[a_pos]
            ).sum(axis=1)
        ).ravel()

        genres = [
            genre_jaccard(
                anchor_metadata["genre_set"],
                value,
            )
            for value in candidate_metadata["genre_set"]
        ]

        anchor_year = anchor_metadata["year"]
        candidate_years = (
            candidate_metadata["year"].to_numpy()
        )

        year_difference = np.abs(
            candidate_years - anchor_year
        )

        anchor_semantic = np.asarray(
            self.embeddings[
                self.semantic_positions[anchor]
            ],
            dtype=np.float32,
        )

        candidate_semantic = np.asarray(
            self.embeddings[
                [
                    self.semantic_positions[movie]
                    for movie in candidates
                ]
            ],
            dtype=np.float32,
        )

        dot = candidate_semantic @ anchor_semantic

        denominator = (
            np.linalg.norm(
                candidate_semantic,
                axis=1,
            )
            * np.linalg.norm(anchor_semantic)
        )

        semantic_similarity = np.divide(
            dot,
            denominator,
            out=np.full(
                len(candidates),
                np.nan,
                dtype=np.float32,
            ),
            where=denominator > 0,
        )

        return {
            "tfidf_similarity": similarities,
            "genre_overlap": genres,
            "year_difference": year_difference,
            "candidate_year": candidate_years,
            "candidate_imdb_rating": (
                candidate_metadata[
                    "imdb_rating"
                ].to_numpy()
            ),
            "candidate_imdb_log_votes": (
                candidate_metadata[
                    "imdb_log_votes"
                ].to_numpy()
            ),
            "anchor_imdb_rating": (
                anchor_metadata["imdb_rating"]
            ),
            "semantic_similarity": semantic_similarity,
        }

    def recommend_movies(
        self,
        movie_id,
        top_n=10,
        embeddings=None,
    ):
        movie_id = int(movie_id)

        if top_n < 1:
            raise ValueError(
                "top_n must be positive."
            )

        if movie_id not in hybrid.movie_id_to_index:
            return None

        if not self.supports(movie_id):
            fallback = hybrid.recommend_movies(
                movie_id,
                top_n=top_n,
                embeddings=(
                    embeddings
                    if embeddings is not None
                    else self.embeddings
                ),
            )

            if fallback is not None:
                fallback["ranking_source"] = (
                    "Hybrid fallback"
                )

            return fallback

        neighbor_rows = self.neighbors.get(
            movie_id
        )

        if neighbor_rows is None or neighbor_rows.empty:
            return hybrid.recommend_movies(
                movie_id,
                top_n=top_n,
                embeddings=self.embeddings,
            ).assign(
                ranking_source="Hybrid fallback"
            )

        neighbor_rows = neighbor_rows[
            neighbor_rows["candidate_id"].isin(
                self.supported_ids
            )
            & (
                neighbor_rows["candidate_id"]
                != movie_id
            )
        ].copy()

        candidates = (
            neighbor_rows["candidate_id"]
            .astype(int)
            .tolist()
        )

        if not candidates:
            return hybrid.recommend_movies(
                movie_id,
                top_n=top_n,
                embeddings=self.embeddings,
            ).assign(
                ranking_source="Hybrid fallback"
            )

        features = neighbor_rows[
            [
                "adjusted_affinity",
                "cosine_affinity",
                "shared_positive_users",
            ]
        ].reset_index(drop=True).copy()

        features["affinity_available"] = 1

        features["candidate_positive_count"] = [
            self.popularity.get(candidate, 0)
            for candidate in candidates
        ]

        content = self._content_features(
            movie_id,
            candidates,
        )

        for name, values in content.items():
            features[name] = values

        x = features[FEATURES].apply(
            pd.to_numeric,
            errors="coerce",
        )

        scores = self.model.predict(
            x,
            num_threads=4,
        )

        ranking = pd.DataFrame({
            "id": candidates,
            "experimental_score": scores,
            "audience_affinity": (
                features["adjusted_affinity"]
            ),
            "semantic_similarity": (
                features["semantic_similarity"]
            ),
        })

        ranking = ranking.sort_values(
            ["experimental_score", "id"],
            ascending=[False, True],
        ).head(top_n)

        # Use the same display catalog as Classic/Hybrid.
        results = (
            hybrid.df.set_index("id")
            .reindex(ranking["id"])
            .reset_index()
        )

        results["experimental_score"] = (
            ranking["experimental_score"].to_numpy()
        )

        results["audience_affinity"] = (
            ranking["audience_affinity"].to_numpy()
        )

        results["semantic_similarity"] = (
            ranking["semantic_similarity"].to_numpy()
        )

        # Existing app expects a similarity column.
        # This is a ranking score, NOT a similarity.
        results["similarity"] = (
            results["experimental_score"]
        )

        results["final_score"] = (
            results["experimental_score"]
        )

        results["ranking_source"] = (
            "LightGBM v4 Experimental"
        )

        return results.reset_index(drop=True)


def load_ranker():
    return ExperimentalRanker()
