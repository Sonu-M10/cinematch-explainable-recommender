"""
CineMatch — Ocean Depths

A streamlined movie discovery application.

Features:
- Mood-based discovery
- Classic recommendation engine with three ranking modes
- Hybrid semantic + metadata engine
- CineMatch AI experimental LightGBM engine
- RRF Ensemble
- Four-column movie grid
- Advanced filters
- Movie details and model insights
- Watchlist and reversible feedback
"""

from pathlib import Path
import html
import json
import os
import sys
import uuid

import pandas as pd
import streamlit as st
from src import user_storage
import browser_storage


# ============================================================
# SETUP
# ============================================================

ROOT = Path(__file__).resolve().parent
IS_CLOUD = Path("/mount/src").exists()
USE_BROWSER_STORAGE = (
    IS_CLOUD
    or os.getenv("CINEMATCH_TEST_BROWSER_STORAGE") == "1"
)
sys.path.insert(0, str(ROOT / "src"))

st.set_page_config(
    page_title="CineMatch | Discover Movies",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

from discovery import (
    MOODS,
    available_fields,
    apply_hard_filters,
    discover,
)


# ============================================================
# OCEAN DEPTHS THEME
# ============================================================

st.markdown(
    """
    <style>
    :root {
        --ocean-bg: #0B1720;
        --ocean-surface: #172B36;
        --ocean-raised: #203744;
        --ocean-accent: #48C6C1;
        --ocean-text: #EDF7F7;
        --ocean-muted: #A3BBC2;
        --ocean-border: #35505C;
    }

    .stApp {
        background: var(--ocean-bg);
        color: var(--ocean-text);
    }

    .block-container {
        max-width: 1440px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] {
        background: #101F29;
        border-right: 1px solid var(--ocean-border);
    }

    h1, h2, h3 {
        color: var(--ocean-text);
        letter-spacing: -0.025em;
    }

    .stButton > button {
        border-radius: 9px;
        min-height: 2.5rem;
    }

    .stButton > button[kind="primary"] {
        background: var(--ocean-accent);
        color: var(--ocean-bg);
        border-color: var(--ocean-accent);
        font-weight: 700;
    }

    .stButton > button[kind="primary"]:hover {
        background: #71D8D3;
        color: var(--ocean-bg);
        border-color: #71D8D3;
    }

    .stButton > button[kind="secondary"] {
        background: var(--ocean-surface);
        color: var(--ocean-text);
        border: 1px solid var(--ocean-border);
    }

    .stButton > button[kind="secondary"]:hover {
        background: var(--ocean-raised);
        border-color: var(--ocean-accent);
    }

    button:focus-visible,
    input:focus-visible {
        outline: 3px solid var(--ocean-accent) !important;
        outline-offset: 2px;
    }

    [data-testid="stImage"] img {
        border-radius: 9px;
    }

    [data-testid="stExpander"] {
        border-color: var(--ocean-border);
        border-radius: 10px;
    }

    .cm-brand {
        color: var(--ocean-text);
        font-size: 1.5rem;
        font-weight: 800;
        letter-spacing: -0.04em;
    }

    .cm-brand span {
        color: var(--ocean-accent);
    }

    .cm-hero {
        padding: 1rem 0 1.5rem;
    }

    .cm-hero h1 {
        font-size: clamp(2rem, 4vw, 3rem);
        margin-bottom: 0.5rem;
    }

    .cm-hero p {
        color: var(--ocean-muted);
        font-size: 1.05rem;
    }

    .cm-card-title {
        color: var(--ocean-text);
        font-size: 1rem;
        font-weight: 700;
        line-height: 1.35;
        min-height: 2.7rem;
    }

    .cm-card-meta {
        color: var(--ocean-muted);
        font-size: 0.8rem;
        min-height: 1.3rem;
    }

    .cm-card-reason {
        color: var(--ocean-accent);
        font-size: 0.83rem;
        line-height: 1.4;
        min-height: 2.5rem;
    }

    .cm-eyebrow {
        color: var(--ocean-accent);
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }

    @media (max-width: 768px) {
        .block-container {
            padding-top: 1rem;
        }

        .cm-hero h1 {
            font-size: 2rem;
        }
    }

    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation: none !important;
            transition: none !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD EXISTING MODELS
# ============================================================

@st.cache_resource(show_spinner="Loading movie catalog...")
def load_classic():
    import weighted_recommender
    return weighted_recommender


@st.cache_resource(show_spinner="Loading semantic model...")
def load_hybrid():
    import hybrid_engine

    return (
        hybrid_engine,
        hybrid_engine.load_embeddings(),
    )


@st.cache_resource(show_spinner="Loading CineMatch AI...")
def load_experimental():
    from experimental_lgbm_engine import load_ranker
    return load_ranker()


@st.cache_resource(show_spinner="Preparing ensemble...")
def load_ensemble():
    import rrf_ensemble
    return rrf_ensemble


from cloud_data import ensure_classic_data

with st.spinner("Preparing CineMatch movie catalog..."):
    ensure_classic_data()

classic = load_classic()
catalog = classic.df
fields = available_fields(catalog)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULTS = {
    "page": "Discover",
    "mode": "Explore by mood",
    "mood": "Surprise me",
    "genre": "Any genre",
    "runtime": "Any length",
    "language": "Any language",
    "age": "Any age rating",
    "excluded": [],
    "years": (2010, 2026),
    "query": "",
    "selected_movie": None,
    "model": "Classic" if IS_CLOUD else "Hybrid",
    "classic_mode": "Evidence-aware",
    "results": None,
    "shown": 8,
    "details_id": None,
    "details_from": "Discover",
    "search_description": None,
    **(
        {bucket: [] for bucket in user_storage.BUCKETS}
        if USE_BROWSER_STORAGE
        else user_storage.load_preferences()
    ),
}

for key, default in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = (
            default.copy()
            if isinstance(default, list)
            else default
        )


# ============================================================
# BROWSER PREFERENCE PERSISTENCE
# ============================================================

def preference_snapshot():
    return {
        bucket: list(st.session_state[bucket])
        for bucket in browser_storage.BUCKETS
    }


def queue_preference_save():
    """Queue a unique write for the next Streamlit render."""
    if USE_BROWSER_STORAGE:
        st.session_state["_browser_write"] = {
            "id": uuid.uuid4().hex,
            "preferences": preference_snapshot(),
        }


if USE_BROWSER_STORAGE:
    result = browser_storage.render_storage(
        st.session_state.get("_browser_write")
    )

    storage_error = getattr(result, "error", None)
    browser_snapshot = getattr(result, "snapshot", None)
    acknowledged = getattr(result, "ack", None)

    if storage_error:
        st.warning(
            "Browser storage is unavailable. "
            "Preferences will only last for this session."
        )
        st.session_state["_browser_loaded"] = True

    if (
        not st.session_state.get("_browser_loaded", False)
        and browser_snapshot is not None
    ):
        restored = browser_storage.normalize_preferences(
            browser_snapshot
        )

        for bucket, values in restored.items():
            st.session_state[bucket] = values

        st.session_state["_browser_loaded"] = True
        st.rerun()

    pending = st.session_state.get("_browser_write")

    if (
        pending is not None
        and acknowledged == pending["id"]
    ):
        st.session_state["_browser_write"] = None

    if not st.session_state.get("_browser_loaded", False):
        st.info("Loading your saved movies from this browser...")
        st.stop()


# ============================================================
# DATA HELPERS
# ============================================================

def safe_text(value, fallback="Not available"):
    if value is None:
        return fallback

    try:
        if pd.isna(value):
            return fallback
    except (TypeError, ValueError):
        pass

    value = str(value).strip()
    return value if value else fallback


def number(row, column):
    return pd.to_numeric(
        row.get(column),
        errors="coerce",
    )


def movie_year(row):
    return safe_text(
        row.get("release_date"),
        "",
    )[:4]


@st.cache_data(show_spinner=False)
def movie_index():
    return (
        catalog
        .drop_duplicates("id")
        .set_index("id", drop=False)
    )


def get_movie(movie_id):
    try:
        return movie_index().loc[int(movie_id)]
    except (KeyError, TypeError, ValueError):
        return None


def poster_url(row):
    path = row.get("poster_path")

    if (
        isinstance(path, str)
        and path.startswith("/")
        and len(path) > 1
        and not any(c.isspace() for c in path)
    ):
        return "https://image.tmdb.org/t/p/w500" + path

    return None


def display_poster(row):
    url = poster_url(row)

    if url:
        st.image(
            url,
            use_container_width=True,
        )
    else:
        st.info("🎞️ Poster unavailable")


def toggle(bucket, movie_id):
    movie_id = int(movie_id)
    values = list(st.session_state[bucket])

    if movie_id in values:
        values.remove(movie_id)
    else:
        values.append(movie_id)

    if not USE_BROWSER_STORAGE:
        user_storage.toggle_movie(bucket, movie_id)
    st.session_state[bucket] = values
    queue_preference_save()


def open_details(movie_id, source="Discover"):
    st.session_state.details_id = int(movie_id)
    st.session_state.details_from = source


def close_details():
    st.session_state.details_id = None


# ============================================================
# FILTERS
# ============================================================

@st.cache_data(show_spinner=False)
def genre_options():
    genres = set()

    for value in catalog["genres"].dropna().astype(str):
        for part in value.split(","):
            genre = part.strip().strip("[]'\" ")

            if genre and len(genre) < 35:
                genres.add(genre)

    return sorted(genres)


@st.cache_data(show_spinner=False)
def language_options():
    if not fields.get("original_language", False):
        return []

    return sorted(
        catalog["original_language"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )


@st.cache_data(show_spinner=False)
def age_options():
    column = (
        "age_rating"
        if fields.get("age_rating", False)
        else (
            "certification"
            if fields.get("certification", False)
            else None
        )
    )

    if column is None:
        return []

    return sorted(
        catalog[column]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )


def selected_filters():
    return {
        "genre": (
            None
            if st.session_state.genre == "Any genre"
            else st.session_state.genre
        ),
        "max_runtime": {
            "Under 90 minutes": 90,
            "Under 120 minutes": 120,
            "Under 150 minutes": 150,
        }.get(st.session_state.runtime),
        "language": (
            None
            if st.session_state.language == "Any language"
            else st.session_state.language
        ),
        "age_rating": (
            None
            if st.session_state.age == "Any age rating"
            else st.session_state.age
        ),
        "excluded_genres": st.session_state.excluded,
        "year_min": (
            st.session_state.years[0]
            if st.session_state.years[0] != 2010
            else None
        ),
        "year_max": (
            st.session_state.years[1]
            if st.session_state.years[1] != 2026
            else None
        ),
    }


def reset_filters():
    for key in (
        "mood",
        "genre",
        "runtime",
        "language",
        "age",
        "excluded",
        "years",
        "query",
        "selected_movie",
        "model",
        "classic_mode",
    ):
        default = DEFAULTS[key]

        st.session_state[key] = (
            default.copy()
            if isinstance(default, list)
            else default
        )

    st.session_state.results = None
    st.session_state.shown = 8
    st.session_state.search_description = None


# ============================================================
# RECOMMENDATION ENGINES
# ============================================================

MODEL_OPTIONS = (
    ["Classic"] if IS_CLOUD else [
        "Hybrid",
        "Classic",
        "CineMatch AI (Experimental)",
        "RRF Ensemble",
    ]
)

CLASSIC_MODES = {
    "Evidence-aware": "evidence_aware",
    "Pure similarity": "pure_similarity",
    "Fixed quality": "fixed_quality",
}


def recommend(movie_id, model, ranking_mode, pool=300):
    movie_id = int(movie_id)

    if model == "Classic":
        return classic.recommend_movies(
            movie_id,
            top_n=pool,
            ranking_mode=CLASSIC_MODES[ranking_mode],
        )

    if model == "Hybrid":
        hybrid, embeddings = load_hybrid()

        return hybrid.recommend_movies(
            movie_id,
            top_n=pool,
            embeddings=embeddings,
        )

    if model == "CineMatch AI (Experimental)":
        return load_experimental().recommend_movies(
            movie_id,
            top_n=pool,
        )

    if model == "RRF Ensemble":
        ensemble = load_ensemble()
        _, embeddings = load_hybrid()

        return ensemble.recommend_movies(
            movie_id,
            top_n=pool,
            pool_size=pool,
            embeddings=embeddings,
            experimental_ranker=load_experimental(),
            ranking_mode=CLASSIC_MODES[ranking_mode],
        )

    raise ValueError("Unknown recommendation engine.")


def enrich_results(results):
    if results is None or results.empty:
        return catalog.iloc[:0].copy()

    if "id" not in results.columns:
        raise ValueError(
            "Recommendation results are missing movie IDs."
        )

    results = results.copy()

    results["id"] = pd.to_numeric(
        results["id"],
        errors="coerce",
    )

    results = (
        results
        .dropna(subset=["id"])
        .copy()
    )

    results["id"] = results["id"].astype(int)

    results = (
        results
        .drop_duplicates("id")
        .reset_index(drop=True)
    )

    missing_columns = [
        column
        for column in catalog.columns
        if (
            column not in results.columns
            and column != "id"
        )
    ]

    if missing_columns:
        metadata = (
            catalog[["id"] + missing_columns]
            .drop_duplicates("id")
        )

        results = results.merge(
            metadata,
            on="id",
            how="left",
            sort=False,
        )

    return results


# ============================================================
# MODEL EXPLANATIONS
# ============================================================

def recommendation_reason(row):
    return safe_text(
        row.get("_reason"),
        "Selected using your discovery preferences.",
    )


def model_insights(row):
    source = safe_text(
        row.get("ranking_source"),
        "",
    )

    if source:
        st.write(f"**Ranking source:** {source}")

    if source == "Hybrid fallback":
        st.info(
            "The experimental model used Hybrid "
            "because audience history was not supported."
        )

    supporters = row.get("supporting_models")

    if isinstance(supporters, (list, tuple, set)):
        supporters = ", ".join(map(str, supporters))

    supporters = safe_text(supporters, "")

    if supporters:
        st.write(f"**Supporting models:** {supporters}")

    metrics = (
        ("classic_rank", "Classic rank"),
        ("hybrid_rank", "Hybrid rank"),
        ("experimental_rank", "Experimental rank"),
        ("model_support", "Number of supporting models"),
        ("semantic_similarity", "Semantic similarity"),
        ("metadata_similarity", "Metadata similarity"),
        ("audience_affinity", "Audience affinity"),
        ("specificity", "Specificity"),
        ("rrf_score", "RRF score"),
        ("experimental_score", "Experimental score"),
        ("similarity", "Similarity / ranking score"),
    )

    displayed = False

    for column, label in metrics:
        value = number(row, column)

        if pd.notna(value):
            displayed = True

            if column.endswith("_rank") or column == "model_support":
                formatted = str(int(value))
            else:
                formatted = f"{value:.4f}"

            st.write(f"**{label}:** {formatted}")

    for feature in getattr(classic, "FEATURE_WEIGHTS", {}):
        value = number(
            row,
            f"{feature}_contribution",
        )

        if pd.notna(value):
            displayed = True
            st.write(
                f"**{feature.title()} contribution:** "
                f"{value:.4f}"
            )

    if not displayed:
        st.caption(
            "No additional numerical model diagnostics "
            "were returned for this recommendation."
        )

    st.caption(
        "These values are ranking signals, not "
        "probabilities of enjoyment. Scores from "
        "different engines may not be comparable."
    )


# ============================================================
# COMPACT MOVIE CARD
# ============================================================

def movie_card(row):
    movie_id = int(row["id"])

    with st.container(border=True):
        display_poster(row)

        title = html.escape(
            safe_text(row.get("title"))
        )

        st.markdown(
            f'<div class="cm-card-title">{title}</div>',
            unsafe_allow_html=True,
        )

        year = movie_year(row)

        st.markdown(
            f'<div class="cm-card-meta">'
            f'{html.escape(year)}'
            f'</div>',
            unsafe_allow_html=True,
        )

        reason = recommendation_reason(row)

        if len(reason) > 85:
            reason = reason[:82].rstrip() + "…"

        st.markdown(
            '<div class="cm-card-reason">'
            '✦ '
            + html.escape(reason)
            + '</div>',
            unsafe_allow_html=True,
        )

        left, right = st.columns(2, gap="small")

        left.button(
            "Details",
            key=f"details_{movie_id}",
            on_click=open_details,
            args=(movie_id, "Discover"),
            use_container_width=True,
        )

        right.button(
            (
                "✓ Saved"
                if movie_id in st.session_state.saved
                else "＋ Save"
            ),
            key=f"save_{movie_id}",
            on_click=toggle,
            args=("saved", movie_id),
            use_container_width=True,
        )


# ============================================================
# MOVIE DETAILS
# ============================================================

def movie_details(movie_id):
    st.button(
        "← Back",
        on_click=close_details,
    )

    row = get_movie(movie_id)

    if row is None:
        st.warning("Movie not found.")
        return

    left, right = st.columns(
        [1, 2],
        gap="large",
    )

    with left:
        display_poster(row)

    with right:
        st.title(
            safe_text(row.get("title"))
        )

        metadata = [
            movie_year(row),
            safe_text(row.get("genres"), ""),
        ]

        runtime = number(row, "runtime")

        if pd.notna(runtime) and runtime > 0:
            metadata.append(
                f"{int(runtime)} min"
            )

        st.caption(
            " · ".join(x for x in metadata if x)
        )

        rating = number(row, "vote_average")

        if pd.notna(rating):
            st.write(
                f"**TMDB rating:** {rating:.1f}/10"
            )

        for column, label in (
            ("directors", "Director"),
            ("cast", "Cast"),
            ("age_rating", "Age rating"),
            ("certification", "Certification"),
        ):
            value = safe_text(
                row.get(column),
                "",
            )

            if value:
                st.write(
                    f"**{label}:** {value}"
                )

        st.subheader("Synopsis")

        st.write(
            safe_text(
                row.get("overview"),
                "Synopsis unavailable.",
            )
        )

        st.button(
            (
                "✓ Remove from watchlist"
                if movie_id in st.session_state.saved
                else "＋ Add to watchlist"
            ),
            key=f"detail_save_{movie_id}",
            on_click=toggle,
            args=("saved", movie_id),
            type="primary",
        )

        st.caption(
            "Streaming availability and verified "
            "trailer links are not connected."
        )

    st.divider()

    st.subheader("Your preferences")

    for bucket, label in (
        ("liked", "More like this"),
        ("dismissed", "Not interested"),
        ("watched", "Already watched"),
    ):
        active = movie_id in st.session_state[bucket]

        st.button(
            (
                f"Undo {label.lower()}"
                if active
                else label
            ),
            key=f"detail_{bucket}_{movie_id}",
            on_click=toggle,
            args=(bucket, movie_id),
        )

    st.caption(
        "Feedback is reversible. Not interested and "
        "Already watched exclude movies from future "
        "searches. More like this is saved as feedback "
        "but does not automatically rerank results."
    )

    results = st.session_state.results

    if (
        results is not None
        and not results.empty
        and "id" in results.columns
    ):
        matching = results.loc[
            results["id"] == movie_id
        ]

        if not matching.empty:
            result_row = matching.iloc[0]

            st.divider()

            st.subheader(
                "Why CineMatch recommended this"
            )

            st.write(
                recommendation_reason(result_row)
            )

            with st.expander("Model insights"):
                model_insights(result_row)


# ============================================================
# SELECTED MOVIE PREVIEW
# ============================================================

def selected_movie_preview(movie_id):
    """
    Show the selected movie's poster and details
    before generating recommendations.
    """

    row = get_movie(movie_id)

    if row is None:
        st.warning("Movie details are unavailable.")
        return

    st.markdown("### Your selected movie")

    with st.container(border=True):
        poster_col, info_col = st.columns(
            [1, 3],
            gap="large",
            vertical_alignment="top",
        )

        with poster_col:
            display_poster(row)

        with info_col:
            st.subheader(
                safe_text(row.get("title"))
            )

            metadata = []

            year = movie_year(row)
            if year:
                metadata.append(year)

            genres = safe_text(
                row.get("genres"),
                "",
            )
            if genres:
                metadata.append(genres)

            runtime = number(row, "runtime")
            if pd.notna(runtime) and runtime > 0:
                metadata.append(f"{int(runtime)} min")

            if metadata:
                st.caption(" · ".join(metadata))

            rating = number(row, "vote_average")
            if pd.notna(rating):
                st.write(f"⭐ **TMDB:** {rating:.1f}/10")

            director = safe_text(
                row.get("directors"),
                "",
            )
            if director:
                st.write(f"**Director:** {director}")

            st.markdown("**Synopsis**")
            st.write(
                safe_text(
                    row.get("overview"),
                    "Synopsis unavailable.",
                )
            )

            st.caption(
                "This is the movie CineMatch will use "
                "as the starting point for recommendations."
            )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        '<div class="cm-brand">'
        'Cine<span>Match</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.caption("Your next movie starts here.")

    st.divider()

    st.radio(
        "Navigation",
        ["Discover", "Watchlist"],
        key="page",
        label_visibility="collapsed",
    )

    st.divider()

    st.caption(
        f"🎬 {len(catalog):,} movies"
    )

    st.caption(
        f"🔖 {len(st.session_state.saved)} saved"
    )

    with st.expander("Your data"):
        st.caption(
            "Watchlist and feedback are stored in "
            + (
                "this browser. Export them for a backup."
                if USE_BROWSER_STORAGE
                else "your local database. Export them for a backup."
            )
        )

        buckets = (
            "saved",
            "watched",
            "dismissed",
            "liked",
        )

        snapshot = {
            key: st.session_state[key]
            for key in buckets
        }

        st.download_button(
            "Export lists",
            data=json.dumps(
                snapshot,
                indent=2,
            ),
            file_name="cinematch_lists.json",
            mime="application/json",
            use_container_width=True,
        )

        upload = st.file_uploader(
            "Import lists",
            type=["json"],
        )

        if upload is not None and st.button("Restore lists"):
            try:
                incoming = json.load(upload)

                if not isinstance(incoming, dict):
                    raise ValueError(
                        "Expected a JSON object."
                    )

                restored = {}

                for key in buckets:
                    values = incoming.get(key, [])

                    if not isinstance(values, list):
                        raise ValueError(
                            f"{key} must be a list."
                        )

                    restored[key] = [
                        int(value)
                        for value in values
                        if str(value).isdigit()
                    ][:1000]

                for key, values in restored.items():
                    st.session_state[key] = values

                if not USE_BROWSER_STORAGE:
                    user_storage.save_preferences(restored)
                queue_preference_save()
                st.success("Lists restored.")
                st.rerun()

            except (
                ValueError,
                TypeError,
                json.JSONDecodeError,
            ) as exc:
                st.error(
                    f"Import failed: {exc}"
                )

        if st.button("Clear saved data"):
            if not USE_BROWSER_STORAGE:
                user_storage.clear_preferences()
            for key in buckets:
                st.session_state[key] = []

            queue_preference_save()
            st.rerun()


# ============================================================
# WATCHLIST PAGE
# ============================================================

if st.session_state.page == "Watchlist":
    if st.session_state.details_id is not None:
        movie_details(
            st.session_state.details_id
        )
        st.stop()

    st.title("Your watchlist")

    st.caption("Movies you've saved for later.")

    if not st.session_state.saved:
        st.info(
            "Your watchlist is empty. "
            "Discover a movie and save it here."
        )

    for movie_id in list(st.session_state.saved):
        row = get_movie(movie_id)

        if row is None:
            continue

        with st.container(border=True):
            a, b, c = st.columns(
                [1, 4, 1],
                gap="medium",
            )

            with a:
                display_poster(row)

            with b:
                st.markdown(
                    f"**{safe_text(row.get('title'))}**"
                )

                st.caption(
                    movie_year(row)
                )

                st.write(
                    safe_text(
                        row.get("genres"),
                        "",
                    )
                )

            with c:
                st.button(
                    "Details",
                    key=f"watch_details_{movie_id}",
                    on_click=open_details,
                    args=(movie_id, "Watchlist"),
                )

                st.button(
                    "Remove",
                    key=f"watch_remove_{movie_id}",
                    on_click=toggle,
                    args=("saved", movie_id),
                )

    st.stop()


# ============================================================
# DETAILS VIEW
# ============================================================

if st.session_state.details_id is not None:
    movie_details(
        st.session_state.details_id
    )
    st.stop()


# ============================================================
# DISCOVER PAGE
# ============================================================

st.markdown(
    """
    <div class="cm-hero">
        <div class="cm-eyebrow">Discover with CineMatch</div>
        <h1>Find your next favorite movie.</h1>
        <p>
            Less searching. More watching.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.radio(
    "How would you like to discover?",
    [
        "Explore by mood",
        "Find similar movies",
    ],
    key="mode",
    horizontal=True,
)

st.divider()

selected_movie_id = None


# ============================================================
# MOOD DISCOVERY
# ============================================================

if st.session_state.mode == "Explore by mood":
    st.subheader("What are you in the mood for?")

    st.pills(
        "Mood",
        list(MOODS),
        key="mood",
        selection_mode="single",
        label_visibility="collapsed",
    )

    st.caption(
        "Mood suggestions use catalog genres, "
        "not natural-language emotion analysis."
    )


# ============================================================
# MOVIE-BASED DISCOVERY
# ============================================================

else:
    st.subheader("Start with a movie you love")

    st.text_input(
        "Search for a movie",
        key="query",
        placeholder="Inception, Interstellar, The Dark Knight...",
    )

    query = st.session_state.query.strip()

    if len(query) >= 2:
        matches = catalog.loc[
            catalog["title"]
            .fillna("")
            .astype(str)
            .str.contains(
                query,
                case=False,
                regex=False,
            )
        ].copy()

        if not matches.empty:
            if "vote_count" in matches.columns:
                matches["_votes"] = pd.to_numeric(
                    matches["vote_count"],
                    errors="coerce",
                ).fillna(0)
            else:
                matches["_votes"] = 0

            matches = (
                matches
                .sort_values(
                    "_votes",
                    ascending=False,
                )
                .drop_duplicates("id")
                .head(40)
            )

            options = (
                matches["id"]
                .astype(int)
                .tolist()
            )

            by_id = matches.set_index("id")

            if st.session_state.selected_movie not in options:
                st.session_state.selected_movie = options[0]

            st.selectbox(
                "Choose a movie",
                options,
                key="selected_movie",
                format_func=lambda mid: (
                    f"{safe_text(by_id.loc[mid, 'title'])} "
                    f"({safe_text(by_id.loc[mid, 'release_date'], '')[:4]})"
                ),
            )

            selected_movie_id = (
                st.session_state.selected_movie
            )

        else:
            st.warning(
                "No matching movies. Try a shorter title."
            )

            st.session_state.selected_movie = None

    # Display the selected movie BEFORE the engine selector.
    # This is the important fix.
    if selected_movie_id is not None:
        selected_movie_preview(selected_movie_id)
        st.write("")

    # Keep this outside the search-result condition.
    # It should always appear in Find similar movies mode.
    st.selectbox(
        "Recommendation engine",
        MODEL_OPTIONS,
        key="model",
    )

    descriptions = {
        "Hybrid": (
            "Combines semantic meaning with movie metadata."
        ),
        "Classic": (
            "Uses weighted movie features and optional "
            "quality-aware ranking."
        ),
        "CineMatch AI (Experimental)": (
            "Uses audience affinity where supported, "
            "with Hybrid fallback."
        ),
        "RRF Ensemble": (
            "Combines Classic, Hybrid, and supported "
            "experimental rankings."
        ),
    }

    st.caption(
        descriptions[st.session_state.model]
    )

    if st.session_state.model == "Classic":
        st.radio(
            "Classic ranking strategy",
            list(CLASSIC_MODES),
            key="classic_mode",
            horizontal=True,
        )

    elif st.session_state.model == "RRF Ensemble":
        with st.expander("Ensemble settings"):
            st.radio(
                "Classic component ranking",
                list(CLASSIC_MODES),
                key="classic_mode",
                horizontal=True,
            )


# ============================================================
# SIMPLE FILTERS
# ============================================================

st.subheader("Refine your search")

a, b = st.columns(2)

with a:
    st.selectbox(
        "Genre",
        ["Any genre"] + genre_options(),
        key="genre",
    )

with b:
    st.selectbox(
        "Runtime",
        [
            "Any length",
            "Under 90 minutes",
            "Under 120 minutes",
            "Under 150 minutes",
        ],
        key="runtime",
        disabled=not fields.get("runtime", False),
    )


# ============================================================
# ADVANCED FILTERS
# ============================================================

with st.expander("Advanced filters"):
    languages = language_options()

    st.selectbox(
        "Original language",
        ["Any language"] + languages,
        key="language",
        disabled=not languages,
    )

    st.slider(
        "Release years",
        2010,
        2026,
        key="years",
    )

    ages = age_options()

    st.selectbox(
        "Age rating",
        ["Any age rating"] + ages,
        key="age",
        disabled=not ages,
    )

    st.multiselect(
        "Exclude genres",
        genre_options(),
        key="excluded",
    )

    st.caption(
        "These filters are strict. Movies with "
        "missing required metadata may be excluded."
    )


# ============================================================
# SEARCH ACTIONS
# ============================================================

st.write("")

a, b = st.columns([4, 1])

find_clicked = a.button(
    "🔎 Find my movies",
    type="primary",
    use_container_width=True,
)

b.button(
    "Reset",
    on_click=reset_filters,
    use_container_width=True,
)


# ============================================================
# RUN RECOMMENDATIONS
# ============================================================

if find_clicked:
    try:
        with st.spinner("Finding movies..."):
            requirements = selected_filters()

            hidden = set(
                st.session_state.dismissed
                + st.session_state.watched
            )

            if st.session_state.mode == "Explore by mood":
                results = discover(
                    catalog,
                    mood=st.session_state.mood,
                    hidden_ids=hidden,
                    **requirements,
                )

                description = "Mood-based discovery"

            else:
                if selected_movie_id is None:
                    raise ValueError(
                        "Search for and select a movie first."
                    )

                model = st.session_state.model

                results = recommend(
                    selected_movie_id,
                    model,
                    st.session_state.classic_mode,
                    pool=300,
                )

                results = enrich_results(results)

                if not results.empty:
                    results = apply_hard_filters(
                        results,
                        **requirements,
                    )

                    results = results.loc[
                        ~results["id"].isin(hidden)
                    ].copy()

                    results = results.loc[
                        results["id"] != selected_movie_id
                    ].copy()

                    seed = get_movie(selected_movie_id)

                    seed_title = (
                        safe_text(seed.get("title"))
                        if seed is not None
                        else "your selected movie"
                    )

                    results["_reason"] = (
                        f"Recommended by {model} "
                        f"based on {seed_title}."
                    )

                description = f"{model} recommendations"

            st.session_state.results = (
                results.reset_index(drop=True)
            )

            st.session_state.shown = 8

            st.session_state.search_description = (
                description
            )

    except (
        ValueError,
        KeyError,
        OSError,
        ImportError,
        AttributeError,
        TypeError,
    ) as exc:
        st.error(
            f"Could not generate recommendations: {exc}"
        )

        st.info(
            "Check your selected movie, filters, "
            "and model files, then try again."
        )


# ============================================================
# FOUR-COLUMN RECOMMENDATION GRID
# ============================================================

st.divider()

if st.session_state.results is None:
    st.info(
        "Choose a mood or a movie, then click "
        "**Find my movies** to get started."
    )

else:
    results = st.session_state.results

    st.subheader("Picked for you")

    if st.session_state.search_description:
        st.caption(
            st.session_state.search_description
        )

    if results.empty:
        st.warning(
            "No movies matched all your requirements."
        )

        st.write(
            "Try broadening your filters, "
            "removing an excluded genre, "
            "or choosing a different movie."
        )

    else:
        st.caption(
            f"{len(results)} matching recommendations"
        )

        visible = results.head(
            st.session_state.shown
        )

        # Four movie cards per row.
        for start in range(0, len(visible), 4):
            columns = st.columns(
                4,
                gap="small",
            )

            batch = visible.iloc[
                start:start + 4
            ]

            for column, (_, row) in zip(
                columns,
                batch.iterrows(),
            ):
                with column:
                    movie_card(row)

        if len(results) > st.session_state.shown:
            st.button(
                "Show more movies",
                use_container_width=True,
                on_click=lambda: st.session_state.update(
                    shown=st.session_state.shown + 8
                ),
            )

        st.caption(
            "Recommendations are based on catalog "
            "metadata and model ranking signals, "
            "not predicted probabilities of enjoyment."
        )