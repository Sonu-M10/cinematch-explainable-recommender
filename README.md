
# CineMatch — Explainable Movie Recommendations

**Discover movies. Understand why they were recommended.**

CineMatch is a movie-discovery application built with Python, Streamlit, and machine learning. It combines traditional content-based filtering, semantic similarity, experimental learning-to-rank methods, and ensemble techniques to generate and explain movie recommendations.

Rather than relying on a single recommendation algorithm, CineMatch explores how different approaches identify relevant movies, how their rankings differ, and how their decisions can be made more transparent.

The application currently uses a catalog of approximately **133,424 films**.

**🌐 Try CineMatch Online — No Installation Required**

You can use CineMatch directly in your browser through the **[live Streamlit application](https://cinematch-moresh.streamlit.app/)**. No downloads, Python installation, or local setup are required.

The installation instructions below are intended for developers who want to run the project locally, explore its source code, or experiment with the recommendation engines.

**Note:** The public version currently provides CineMatch Classic. The full local application supports all four recommendation engines when the required data and model artifacts are available.

## Features

### Movie Discovery

- Search movies by title and generate recommendations based on a selected film.
- Explore movies through mood-based discovery.
- Discover similarities using plot descriptions, genres, keywords, directors, and cast.
- Refine results using genre, runtime, language, and other available movie metadata.
- View movie descriptions, ratings, and other available information.

### Multiple Recommendation Engines

- **CineMatch Classic:** Weighted TF-IDF content-based filtering.
- **CineMatch Hybrid:** Semantic embeddings combined with metadata similarity.
- **CineMatch AI:** Experimental LightGBM learning-to-rank model.
- **RRF Ensemble:** Combines rankings from multiple recommendation engines.

### Explainable Recommendations

- Understand why individual movies were recommended.
- Inspect the features and signals contributing to a recommendation.
- Expand mathematical explanations showing actual scoring calculations.
- Compare different ranking strategies and their underlying methods.
- Distinguish between experimental model predictions and fallback recommendations.

### Personal Movie Management

- Maintain a personal watchlist.
- Track watched movies.
- Dismiss unwanted recommendations.
- Preserve movie lists using browser-based storage on supported deployments.
- Access external JustWatch searches to explore where movies may be available to stream.

CineMatch does not directly integrate with streaming-provider APIs or guarantee real-time streaming availability.

---

## Recommendation Engines

### 1. CineMatch Classic

CineMatch Classic is a content-based recommendation engine using weighted TF-IDF cosine similarity.

Each movie is represented through five groups of features:

| Feature | Weight |
|---|---:|
| Overview | 40% |
| Genres | 25% |
| Keywords | 20% |
| Directors | 10% |
| Cast | 5% |

For each feature, CineMatch calculates cosine similarity between the selected movie and a candidate movie, then multiplies that similarity by the corresponding feature weight.

The combined content-similarity score is:

$$
S = \sum_f w_f \cos(\mathbf{x}_f,\mathbf{y}_f)
$$

Where:

- \(w_f\) is the weight assigned to feature \(f\).
- \(\mathbf{x}_f\) and \(\mathbf{y}_f\) represent the movies' TF-IDF feature vectors.
- \(S\) is the combined content-similarity score.

#### Ranking Strategies

Classic supports three ranking strategies:

**Pure Similarity**

Ranks movies entirely according to their weighted content similarity.

$$
F = S
$$

**Evidence-Aware Ranking**

An experimental strategy that reduces the influence of matches dominated by shared genres while incorporating a rating-quality adjustment scaled by rating confidence.

$$
F = S(1 - 0.15D) + 0.30CDQ
$$

Where:

- \(D\) represents genre dependence.
- \(C\) represents rating confidence.
- \(Q\) represents the rating-quality bonus.

**Fixed Quality**

Adds a fixed-weight rating-quality adjustment to content similarity.

$$
F = S + 0.15Q
$$

These strategies allow CineMatch to explore the trade-off between matching movie content and incorporating additional evidence about movie quality.

### 2. CineMatch Hybrid

CineMatch Hybrid combines semantic embeddings with structured movie metadata.

Traditional TF-IDF methods identify similarities based on word usage and feature overlap. Semantic embeddings provide an additional representation of textual meaning, allowing CineMatch to identify relationships that may not be apparent from shared keywords alone.

The engine uses the **all-MiniLM-L6-v2** sentence-transformer model to represent movie descriptions as dense vectors.

It then combines two signals:

1. **Semantic similarity:** Cosine similarity between normalized movie embeddings.
2. **Metadata similarity:** Weighted TF-IDF cosine similarity across overview, genres, keywords, directors, and cast.

The final Hybrid score is:

$$
F_{\text{Hybrid}} = 0.50S_{\text{semantic}} + 0.50S_{\text{metadata}}
$$

Both signals receive equal weight.

This approach allows CineMatch to combine broader semantic relationships with explicit movie attributes.

### 3. CineMatch AI — LightGBM v4

CineMatch AI is an experimental learning-to-rank engine built using **LightGBM**.

Unlike Classic and Hybrid, which use explicitly defined similarity formulas, the LightGBM model learns ranking relationships from training data.

The model evaluates 13 numerical features covering:

- Audience affinity between movies.
- Shared positive-rating users.
- Candidate movie popularity.
- TF-IDF content similarity.
- Semantic similarity.
- Genre overlap.
- Release-year information.
- IMDb ratings and voting signals.

The model combines learned decision-tree outputs to produce a ranking score:

$$
F(\mathbf{x}) = b + \sum_{t=1}^{T} f_t(\mathbf{x})
$$

Where each \(f_t\) represents a trained tree and \(b\) represents the model's initial score.

The current LightGBM model supports approximately **1,500 seed movies**.

#### Experimental Model Fallback

When a selected movie is unsupported, CineMatch uses the Hybrid engine instead.

The interface explicitly identifies the result as a **Hybrid fallback**, rather than presenting it as an independent LightGBM prediction.

The experimental model was trained using user-history information and is adapted to the application's selected-movie recommendation workflow. Its scores should therefore be interpreted as experimental ranking signals, not calibrated probabilities of user enjoyment.

### 4. RRF Ensemble

The RRF Ensemble combines recommendation rankings from Classic, Hybrid, and LightGBM when supported.

Because the engines produce different types of scores, directly averaging their raw outputs would not be appropriate.

Instead, CineMatch uses **Reciprocal Rank Fusion (RRF)**, which combines each model's ranking positions.

The formula is:

$$
\text{RRF}(d) = \sum_m \frac{w_m}{60 + r_m(d)}
$$

Where:

- \(w_m\) is the weight assigned to model \(m\).
- \(r_m(d)\) is movie \(d\)'s position in that model's ranking.
- 60 is the RRF smoothing constant.

The provisional model weights are:

| Model | Weight |
|---|---:|
| Classic | 1.00 |
| Hybrid | 1.00 |
| LightGBM v4 | 1.25 |

Movies ranked highly by multiple engines receive contributions from each participating model.

For movies unsupported by LightGBM, the ensemble combines Classic and Hybrid without double-counting a Hybrid fallback as an independent experimental ranking.

---

## Explainable Recommendations

One of CineMatch's central objectives is to make recommendation systems more interpretable.

Rather than simply presenting a ranked list of movies, the application includes a **"Why CineMatch recommended this"** section for individual recommendations.

This section provides a short, human-readable description of the recommendation and identifies the engine responsible for generating it.

Users can expand **"How did CineMatch calculate this?"** to explore the mathematical process behind the result.

### Engine-Specific Explanations

| Engine | Explanation Provided |
|---|---|
| Classic | Feature contributions, content similarity, specificity, and ranking adjustments |
| Hybrid | Semantic similarity, metadata similarity, and their weighted combination |
| CineMatch AI | Available model signals, LightGBM ranking score, and learned ranking process |
| RRF Ensemble | Participating model ranks, weighted reciprocal-rank contributions, and final score |

#### Mathematical Transparency

Where applicable, CineMatch displays the actual numerical values used in the recommendation calculations.

For example, a Hybrid recommendation might have:

- Semantic similarity: 0.5244
- Metadata similarity: 0.2087

The combined score is approximately:

$$
F = 0.50(0.5244) + 0.50(0.2087) = 0.3666
$$

Displayed values are rounded, while the underlying calculations use full precision.

For LightGBM, CineMatch explains the model's inputs and ranking process without incorrectly presenting available input values as exact feature attributions.

**Important:** Recommendation scores are ranking signals, not probabilities that a user will enjoy a movie.

---

## Ensemble Evaluation

Three ensemble weight configurations were compared using eight LightGBM-supported seed movies and AI-assisted relevance judgments.

| Configuration | Precision@10 | Mean relevance | NDCG@10 |
|---|---:|---:|---:|
| Equal (1 / 1 / 1) | 0.6125 | 1.9562 | 0.8402 |
| Mild (1 / 1 / 1.25) | **0.6500** | **2.0125** | **0.8680** |
| Moderate (1 / 1 / 1.5) | 0.6250 | 1.9812 | 0.8630 |

The mild configuration achieved the highest average scores across the three metrics.

### Evaluation Metrics

**Precision@10**

Measures the proportion of the top ten recommendations judged relevant.

**Mean Relevance**

Measures the average relevance judgment assigned to recommended movies.

**NDCG@10**

Normalized Discounted Cumulative Gain measures ranking quality while placing greater importance on relevant movies appearing near the top of the recommendation list.

### Interpretation and Limitations

These findings are **exploratory**, not evidence of general superiority.

The evaluation used only eight supported seed movies, AI-assisted rather than independent human judgments, and configurations selected during iterative experimentation.

NDCG was calculated against the ideal ordering within each seed's judged recommendation union, not against all possible relevant movies in the catalog.

The results provide an initial comparison of ensemble weighting strategies rather than a definitive benchmark of recommendation quality.

---

## Technology Stack

| Category | Technologies |
|---|---|
| Programming language | Python |
| Web application | Streamlit |
| Data processing | pandas, NumPy |
| Machine learning | scikit-learn, LightGBM |
| Text representation | TF-IDF |
| Semantic representation | Sentence Transformers |
| Similarity methods | Cosine similarity |
| Recommendation methods | Content-based filtering, semantic similarity, learning-to-rank, Reciprocal Rank Fusion |
| Deployment | Streamlit Community Cloud |
| Version control | Git, GitHub |

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Sonu-M10/modern-movie-recommender.git
cd modern-movie-recommender
```

### 2. Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the Datasets

CineMatch does not redistribute third-party movie datasets.

For instructions on obtaining the original datasets, preprocessing them, and validating local artifacts, see [DATA_SETUP.md](DATA_SETUP.md).

### Required Data Artifacts

CineMatch uses locally prepared movie datasets, semantic embeddings, and audience-affinity features.

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

The small LightGBM model artifacts are included in the repository. Larger data artifacts must be prepared separately.

**Current reproducibility limitation:** A fresh clone cannot run the complete application without these additional data files. The project does not yet provide a fully automated data acquisition and preparation workflow.

### Model Availability

LightGBM v4 supports approximately 1,500 seed movies.

When a selected movie is unsupported, CineMatch's RRF ensemble uses Classic and Hybrid without counting a Hybrid fallback as an independent LightGBM vote.

---

## Running the Application

Start the main Streamlit interface:

```bash
streamlit run app.py
```

An additional evaluation interface is available:

```bash
streamlit run evaluation_app.py
```

### Live Deployment

**[CineMatch on Streamlit Cloud](https://cinematch-moresh.streamlit.app/)**

The public deployment currently exposes the Classic recommendation engine.

The full local application supports Classic, Hybrid, CineMatch AI, and RRF Ensemble when the required data artifacts are available.

---

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

---

## Limitations

- LightGBM supports only a subset of the movie catalog.
- Movie metadata quality and completeness vary.
- Popularity and audience ratings can introduce selection biases.
- Semantic and content similarity do not guarantee narrative or experiential relevance.
- LightGBM ranking scores are not calibrated user-preference probabilities.
- The current LightGBM explanations do not include exact feature-level attributions.
- Ensemble weights have not been validated on a large, independently judged test set.
- The repository does not yet provide a complete reproducible dataset-building pipeline.
- The public Streamlit Cloud deployment currently exposes only the Classic engine.
- Streaming-provider availability is accessed through external searches rather than a live integrated API.

---

## Future Work

- Broader human relevance evaluations and held-out benchmarks.
- Better metadata quality and entity deduplication.
- Improved coverage for experimental ranking.
- Feature-attribution methods for the LightGBM model.
- Adaptive ensemble weighting experiments.
- Reproducible data preparation and model training.
- More extensive analysis of recommendation diversity, novelty, and coverage.
- Investigation of self-reinforcing exposure patterns and filter-bubble formation.
- Controlled simulations of user feedback and changing preferences under different recommendation policies.

---

## Project Status

CineMatch is an ongoing machine-learning portfolio and research project focused on building, comparing, evaluating, and explaining recommendation systems.

The project currently includes four locally supported recommendation engines, mathematical explanations, an interactive Streamlit interface, and exploratory ensemble evaluations.

**Developed by [Moresh Shukla](https://github.com/Sonu-M10).**
