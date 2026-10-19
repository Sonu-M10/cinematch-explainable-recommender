
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "movies_expanded.csv"
CACHE_DIR = ROOT / "data" / "semantic_cache"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

SAMPLE_SIZE = 10_000
SEED = 42
TOP_N = 10

TEST_MOVIES = {
    "The Batman (2022)": 414906,
    "Inception": 27205,
    "Interstellar": 157336,
    "The Conjuring": 138843,
    "La La Land": 313369,
    "Get Out": 419430,
    "Knives Out": 546554,
}


# --------------------------------------------------
# 2. LOAD A REPRODUCIBLE MOVIE SAMPLE
# --------------------------------------------------

def load_sample():

    data = pd.read_csv(
        DATA_PATH,
        usecols=[
            "id",
            "title",
            "overview",
            "vote_count",
        ],
        low_memory=False,
    )

    data["id"] = pd.to_numeric(
        data["id"],
        errors="coerce",
    )

    data = data.dropna(subset=["id"]).copy()

    data["id"] = data["id"].astype("int64")

    data = data.drop_duplicates(
        "id"
    ).reset_index(drop=True)

    data["overview"] = (
        data["overview"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    data["vote_count"] = pd.to_numeric(
        data["vote_count"],
        errors="coerce",
    ).fillna(0)

    # Remove movies with extremely short descriptions.
    data = data[
        data["overview"].str.len() >= 40
    ].copy()

    # Ensure our evaluation movies are included.
    test_ids = set(TEST_MOVIES.values())

    anchors = data[
        data["id"].isin(test_ids)
    ]

    missing = test_ids - set(anchors["id"])

    if missing:
        print(
            "Warning: evaluation IDs absent "
            f"from dataset: {sorted(missing)}"
        )

    # Randomly sample other movies.
    others = data[
        ~data["id"].isin(test_ids)
    ]

    n_other = max(
        0,
        SAMPLE_SIZE - len(anchors),
    )

    sample = others.sample(
        n=min(n_other, len(others)),
        random_state=SEED,
    )

    selected = pd.concat(
        [anchors, sample],
        ignore_index=True,
    )

    return (
        selected
        .sort_values("id")
        .reset_index(drop=True)
    )


# --------------------------------------------------
# 3. GENERATE OR LOAD SEMANTIC EMBEDDINGS
# --------------------------------------------------

def load_or_encode(data, model):

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    embedding_path = (
        CACHE_DIR / "sample_embeddings.npy"
    )

    ids_path = (
        CACHE_DIR / "sample_ids.npy"
    )

    config_path = (
        CACHE_DIR / "sample_config.json"
    )

    ids = data["id"].to_numpy(
        dtype=np.int64
    )

    config = {
        "model": MODEL_NAME,
        "sample_size": SAMPLE_SIZE,
        "seed": SEED,
    }

    # Reuse saved embeddings if the sample matches.
    if (
        embedding_path.exists()
        and ids_path.exists()
        and config_path.exists()
    ):

        saved_config = json.loads(
            config_path.read_text()
        )

        saved_ids = np.load(ids_path)

        if (
            saved_config == config
            and np.array_equal(ids, saved_ids)
        ):

            embeddings = np.load(
                embedding_path
            )

            expected_shape = (
                len(data),
                model.get_sentence_embedding_dimension(),
            )

            if embeddings.shape == expected_shape:

                print("Loaded cached embeddings.")

                return embeddings

    print(
        f"Encoding {len(data):,} movie overviews..."
    )

    embeddings = model.encode(
        data["overview"].tolist(),
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    # Save results for future runs.
    np.save(
        embedding_path,
        embeddings,
    )

    np.save(
        ids_path,
        ids,
    )

    config_path.write_text(
        json.dumps(config, indent=2)
    )

    print(
        f"Saved embeddings to {embedding_path}"
    )

    return embeddings


# --------------------------------------------------
# 4. CALCULATE SEMANTIC RECOMMENDATIONS
# --------------------------------------------------

def recommendations(
    movie_id,
    data,
    embeddings,
    top_n=TOP_N,
):

    matches = np.flatnonzero(
        data["id"].to_numpy() == movie_id
    )

    if len(matches) == 0:
        return None

    query_index = int(matches[0])

    # Embeddings are normalized, so the dot product
    # is equivalent to cosine similarity.
    scores = (
        embeddings @ embeddings[query_index]
    )

    # Exclude the selected movie.
    scores[query_index] = -np.inf

    n = min(
        top_n,
        len(data) - 1,
    )

    if n <= 0:
        return data.iloc[:0].copy()

    # Find the highest similarity scores.
    indices = np.argpartition(
        scores,
        -n,
    )[-n:]

    results = data.iloc[
        indices
    ].copy()

    results["semantic_similarity"] = (
        scores[indices]
    )

    results = results.sort_values(
        [
            "semantic_similarity",
            "id",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(drop=True)

    return results


# --------------------------------------------------
# 5. RUN OUR EXPERIMENT
# --------------------------------------------------

def main():

    print(
        "CineMatch — semantic plot experiment"
    )

    data = load_sample()

    print(
        f"Sample size: {len(data):,} movies"
    )

    print(
        f"Loading model: {MODEL_NAME}"
    )

    # Use CPU for our initial experiment.
    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu",
    )

    embeddings = load_or_encode(
        data,
        model,
    )

    print(
        f"Embeddings shape: {embeddings.shape}"
    )

    for name, movie_id in TEST_MOVIES.items():

        print("\n" + "=" * 70)

        print(name)

        print("=" * 70)

        result = recommendations(
            movie_id,
            data,
            embeddings,
        )

        if result is None:

            print(
                "Movie not in sample."
            )

        else:

            print(
                result[
                    [
                        "title",
                        "vote_count",
                        "semantic_similarity",
                    ]
                ].to_string(
                    index=False,
                    float_format=lambda x: f"{x:.4f}",
                )
            )

    print(
        "\nThese are cosine similarity scores, "
        "not match probabilities."
    )

    print(
        "Compare qualitatively; this "
        "10,000-movie sample is not "
        "the full catalog."
    )


if __name__ == "__main__":
    main()
