# CineMatch — Explainable Movie Recommendations

CineMatch is a movie-discovery application built with Python, Streamlit, and machine learning. It combines traditional content-based filtering, semantic similarity, and experimental learning-to-rank methods to generate and explain movie recommendations.

The application currently uses a catalog of approximately **133,424 films**.

## Features

- Search movies by title and explore recommendations.
- Compare multiple recommendation engines.
- Discover movies using plot descriptions, genres, keywords, directors, and cast.
- Explore semantic similarities between movies.
- View available movie metadata, ratings, and recommendation explanations.
- Combine recommendations from multiple engines using Reciprocal Rank Fusion (RRF).

## Recommendation Engines

### 1. CineMatch Classic

A content-based recommendation engine using weighted TF-IDF cosine similarities.

| Feature | Weight |
|---|---:|
| Overview | 40% |
| Genres | 25% |
| Keywords | 20% |
| Directors | 10% |
| Cast | 5% |

The engine supports several ranking policies, including an experimental evidence-aware approach that considers similarity quality and rating confidence.

### 2. CineMatch Hybrid

Combines semantic embeddings with structured movie metadata to capture similarities that may not be apparent from shared keywords alone.

### 3. CineMatch AI — LightGBM v4

An experimental learning-to-rank engine designed to model audience-affinity signals.

The current LightGBM model supports approximately **1,500 seed movies**. For unsupported movies, the application relies on other recommendation engines rather than treating a fallback as an independent LightGBM prediction.

### 4. RRF Ensemble

The ensemble combines recommendation rankings using Reciprocal Rank Fusion.

Rather than directly comparing raw scores from different models, RRF assigns contributions based on each model's ranking positions.

The provisional model weights are:

| Model | Weight |
|---|---:|
| Classic | 1.00 |
| Hybrid | 1.00 |
| LightGBM v4 | 1.25 |

For movies unsupported by LightGBM, the ensemble combines Classic and Hybrid without double-counting a fallback ranking.

## Ensemble Evaluation

Three weight configurations were compared using eight LightGBM-supported seed movies and AI-assisted relevance judgments.

| Configuration | Precision@10 | Mean relevance | NDCG@10 |
|---|---:|---:|---:|
| Equal (1 / 1 / 1) | 0.6125 | 1.9562 | 0.8402 |
| Mild (1 / 1 / 1.25) | **0.6500** | **2.0125** | **0.8680** |
| Moderate (1 / 1 / 1.5) | 0.6250 | 1.9812 | 0.8630 |

The mild configuration achieved the highest average scores across the three metrics.

These findings are **exploratory**, not evidence of general superiority. The evaluation used only eight supported seed movies, AI-assisted rather than independent human judgments, and configurations selected during iterative experimentation.

NDCG was calculated against the ideal ordering within each seed's judged recommendation union, not against all possible relevant movies in the catalog.

## Installation

### Dataset Setup

CineMatch does not redistribute third-party movie datasets.

For instructions on obtaining the original datasets,
preprocessing them, and validating local artifacts, see
[DATA_SETUP.md](DATA_SETUP.md).

Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
## Required Data Artifacts

CineMatch uses locally prepared movie datasets, semantic embeddings, and audience-affinity features. These artifacts are excluded from Git because of their size and data provenance considerations.

The application expects the following files:

| Component | Required artifact |
|---|---|
| Classic | `data/movies_expanded.csv` |
| Hybrid | `data/semantic_cache/full_embeddings.npy` |
| Hybrid | `data/semantic_cache/full_embeddings_metadata.json` |
| LightGBM v4 | `models/lgbm_ranker_v4_experimental.txt` |
| LightGBM v4 | `data/ml/lgbm_history_affinity.csv` |
| LightGBM v4 | `data/ml/lgbm_history_universe.csv` |
| LightGBM v4 | `data/ml/lgbm_training_pairs_v3.csv.gz` |
| LightGBM v4 | `data/movies_with_imdb.csv` |

The small LightGBM model artifacts are included in the repository. The larger data artifacts must be prepared separately.

**Current reproducibility limitation:** A fresh clone cannot run the complete application without these additional data files. The project does not yet provide a fully automated data acquisition and preparation workflow.

### Model Availability

LightGBM v4 supports approximately 1,500 seed movies.

When a selected movie is unsupported, CineMatch's RRF ensemble uses Classic and Hybrid without counting a Hybrid fallback as an independent LightGBM vote.

### Data and model artifacts

The prepared movie catalog, semantic embeddings, and generated evaluation datasets are not included in this repository.

The project requires its corresponding local data artifacts to run. The repository includes small LightGBM model files, but these do not replace the underlying movie catalog or embedding data.

A fully reproducible data acquisition and preparation workflow remains future work.

## Running the Application

Start the main Streamlit interface:

```bash
streamlit run app.py
```

An additional evaluation interface is available:

```bash
streamlit run evaluation_app.py
```

## Project Structure

- `app.py` — Main Streamlit application.
- `evaluation_app.py` — Evaluation interface.
- `src/weighted_recommender.py` — Classic TF-IDF engine.
- `src/hybrid_engine.py` — Hybrid recommendation engine.
- `src/experimental_lgbm_engine.py` — Experimental LightGBM ranker.
- `src/rrf_ensemble.py` — Reciprocal Rank Fusion ensemble.
- `src/rrf_weight_audit.py` — Ensemble weight comparison.
- `src/rrf_human_audit.py` — Relevance evaluation workflow.
- `models/` — Small trained model artifacts and experiment reports.
- `tests/` — Automated tests.
- `data/` — Local datasets and generated artifacts (excluded from Git).
- `.streamlit/config.toml` — Streamlit configuration.

## Limitations

- LightGBM supports only a subset of the movie catalog.
- Movie metadata quality and completeness vary.
- Popularity and audience ratings can introduce selection biases.
- Semantic and content similarity do not guarantee narrative or experiential relevance.
- Ensemble weights have not been validated on a large, independently judged test set.
- The repository does not yet provide a complete reproducible dataset-building pipeline.

## Future Work

- Broader human relevance evaluations and held-out benchmarks.
- Better metadata and entity deduplication.
- Improved coverage for experimental ranking.
- Adaptive ensemble weighting experiments.
- Reproducible data preparation and model training.
- Further investigation of recommendation diversity and self-reinforcing exposure patterns.
