# CineMatch — Dataset Setup Guide

CineMatch does **not** redistribute the original movie datasets or the processed data artifacts. Obtain source data from its providers, review the applicable terms, and generate artifacts locally.

## 1. Install the project

```bash
git clone https://github.com/Sonu-M10/modern-movie-recommender.git
cd modern-movie-recommender
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Download the source movie catalog

The original catalog used for this research project is hosted at:

- [HenryWaltson/TMDB-IMDB-Movies-Dataset on Hugging Face](https://huggingface.co/datasets/HenryWaltson/TMDB-IMDB-Movies-Dataset)
- File: `TMDB  IMDB Movies Dataset.csv` (two spaces between `TMDB` and `IMDB`)
- Pinned revision: `620ea877e884bfc96eb56a988ffc57f4586cc9b0`
- Expected SHA-256: `da825d814096de90374cc7e10c4c8ad0911baee0b01e420d5f57ff2184e48a25`

If `download_source.py` is included in your checkout, run:

```bash
python3 download_source.py
```

The downloader saves/verifies the source at `data/movies.csv`. If obtaining it manually, place the downloaded CSV at that path. The checksum above identifies the precise source version used to produce the reference artifacts.

## 3. Prepare the movie catalog

The catalog preprocessing pipeline is:

```text
data/movies.csv
  -> src/clean.py
  -> data/movies_cleaned.csv
  -> src/metadata_quality.py
  -> data/movies_expanded.csv
```

On checkouts containing the setup wrapper and reproducibility builder:

```bash
python3 setup_data.py --build
```

**Important:** In the currently tested version, `--build` performs an isolated reproducibility check. It does **not** install its outputs into the main `data/` folder. To generate the catalog in place, use:

```bash
python3 src/clean.py
python3 src/metadata_quality.py
```

These direct scripts write their output files, so avoid running them over datasets you wish to preserve.

The reference catalog contains 133,424 movies. A different source revision or dependency version may produce different outputs.

## 4. IMDb enrichment (for the experimental engine)

IMDb source files are available from [IMDb Non-Commercial Datasets](https://developer.imdb.com/non-commercial-datasets/) and its [dataset downloads](https://datasets.imdbws.com/). Review IMDb's usage terms before downloading or using these files.

The existing enrichment workflow expects:

```text
data/review_audit/title.basics.tsv.gz
data/imdb/title.ratings.tsv.gz
```

With these inputs in place, the project uses `src/imdb_ratings_enrichment.py` to produce `data/movies_with_imdb.csv` and `data/imdb/imdb_matches.csv`. Source snapshots change over time, so a current download may not match the original project's checksums.

## 5. Semantic embeddings (for Hybrid and Ensemble)

The Hybrid engine uses `sentence-transformers/all-MiniLM-L6-v2` and a local embedding cache:

```text
data/semantic_cache/full_embeddings.npy
data/semantic_cache/full_embeddings_metadata.json
```

The generation code is in `src/hybrid_recommender.py`. The reference cache has 133,424 rows and 384 dimensions. Generating embeddings can take substantial time and memory. Ensure the cache and metadata correspond to the exact catalog version; the validator checks their fingerprint.

## 6. MovieLens features (for experimental LightGBM)

Obtain the applicable MovieLens release from [GroupLens](https://grouplens.org/datasets/movielens/) and follow its specific license and usage requirements. **The precise MovieLens release and complete end-to-end feature regeneration instructions are not yet documented**, so the experimental LightGBM engine is not currently reproducible from a fresh clone using this guide alone.

The experimental engine expects:

```text
data/movielens/lgbm_history_affinity.csv
data/movielens/lgbm_history_universe.csv
data/movielens/lgbm_training_pairs_v3.csv.gz
models/lgbm_ranker_v4_experimental.txt
```

The trained model file is small and is tracked in the repository, but it does not replace the MovieLens-derived features.

## 7. Validate and run

The setup interface can validate engine-specific artifacts:

```bash
python3 setup_data.py --engine classic
python3 setup_data.py --engine hybrid
python3 setup_data.py --engine ensemble
```

For an exact match to the original project's artifact snapshots, run:

```bash
python3 setup_data.py --engine ensemble --verify-checksums
```

**Checksum verification is only expected to pass for the original reference artifacts.** Independently rebuilt artifacts may be valid without having identical checksums, especially if upstream datasets have changed.

Start the app after the relevant engine's files are available:

```bash
streamlit run app.py
```

## Data provenance and permissions

This repository distributes implementation code, not the underlying third-party datasets. Each user is responsible for obtaining data through authorized sources and following the providers' applicable licenses and terms. Public download availability alone does not imply permission to republish a dataset.
