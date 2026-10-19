
"""
CineMatch Human Evaluation Studio

Run from the project root:

    streamlit run evaluation_app.py

Reviews are saved to:
    data/evaluation/human_reviews.csv

This app does not modify the recommendation models.
"""

from pathlib import Path
import sys

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

REVIEWS_PATH = (
    ROOT / "data" / "evaluation" / "human_reviews.csv"
)

st.set_page_config(
    page_title="CineMatch | Review Studio",
    page_icon="🎬",
    layout="wide",
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_resource(show_spinner="Loading movie catalog...")
def get_catalog():
    import weighted_recommender

    return weighted_recommender.df.set_index(
        "id",
        drop=False,
    )


@st.cache_data(show_spinner=False)
def load_reviews():
    return pd.read_csv(
        REVIEWS_PATH,
        dtype={
            "pair_id": str,
            "review_notes": "string",
        },
        keep_default_na=True,
    )


def reload_reviews():
    load_reviews.clear()
    return load_reviews()


if not REVIEWS_PATH.exists():
    st.error(
        "Evaluation review sheet not found. "
        "Run python3 src/evaluation_benchmark.py generate first."
    )
    st.stop()

catalog = get_catalog()
reviews = load_reviews()


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def clean_text(value, fallback=""):
    if value is None or pd.isna(value):
        return fallback

    text = str(value).strip()

    return text if text else fallback


def poster_url(value):
    if not isinstance(value, str):
        return None

    value = value.strip()

    if (
        not value.startswith("/")
        or value == "/"
        or " " in value
    ):
        return None

    return "https://image.tmdb.org/t/p/w342" + value


def movie_details(movie_id):
    movie_id = int(movie_id)

    if movie_id not in catalog.index:
        return None

    return catalog.loc[movie_id]


def is_reviewed(row):
    return (
        pd.notna(row["narrative_relevance"])
        and pd.notna(row["experience_relevance"])
    )


def rating_value(value):
    if pd.isna(value):
        return None

    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def display_movie(movie_id, label):
    movie = movie_details(movie_id)

    if movie is None:
        st.warning("Movie not found in catalog.")
        return

    st.caption(label)

    st.subheader(
        clean_text(movie["title"], "Untitled")
    )

    poster_col, info_col = st.columns(
        [1, 1.6],
        gap="medium",
    )

    with poster_col:
        url = poster_url(
            movie.get("poster_path")
        )

        if url:
            st.image(
                url,
                use_container_width=True,
            )
        else:
            st.info("Poster unavailable")

    with info_col:
        date = clean_text(
            movie.get("release_date"),
            "Unknown release date",
        )

        st.caption(
            f"Released: {date[:10]}"
        )

        st.caption(
            "Genres: "
            + clean_text(
                movie.get("genres"),
                "Unknown",
            )
        )

        votes_value = movie.get("vote_count", 0)
        rating_value_raw = movie.get(
            "vote_average", 0
        )

        votes = (
            int(votes_value)
            if pd.notna(votes_value)
            else 0
        )

        rating = (
            float(rating_value_raw)
            if pd.notna(rating_value_raw)
            else 0.0
        )

        if votes > 0:
            st.caption(
                f"TMDB rating: {rating:.1f}/10 "
                f"({votes:,} votes)"
            )
        else:
            st.caption("Not yet rated")

    st.write(
        clean_text(
            movie.get("overview"),
            "No synopsis available.",
        )
    )


def save_review(
    pair_id,
    narrative,
    experience,
    reliability,
    notes,
):
    """
    Update one existing review row safely.

    The review_notes column is explicitly loaded
    as text to prevent pandas from inferring
    float64 when the column is initially empty.
    """

    current = pd.read_csv(
        REVIEWS_PATH,
        dtype={
            "pair_id": str,
            "review_notes": "string",
        },
    )

    matches = current["pair_id"] == pair_id

    if matches.sum() != 1:
        raise ValueError(
            "Could not uniquely identify review row."
        )

    current.loc[
        matches,
        "narrative_relevance",
    ] = narrative

    current.loc[
        matches,
        "experience_relevance",
    ] = experience

    current.loc[
        matches,
        "reliability",
    ] = reliability

    current.loc[
        matches,
        "review_notes",
    ] = str(notes)

    # Save to a temporary file first.
    # This avoids leaving a partially written CSV.
    temporary = REVIEWS_PATH.with_suffix(".tmp")

    current.to_csv(
        temporary,
        index=False,
    )

    temporary.replace(REVIEWS_PATH)

    # Clear cached review data so the app
    # loads the latest saved values.
    reload_reviews()


# --------------------------------------------------
# STYLING
# --------------------------------------------------

st.markdown(
    """
    <style>
    .stApp {
        background: #0B101B;
        color: #F8F3E9;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
    }

    h1, h2, h3 {
        letter-spacing: -0.035em;
    }

    [data-testid="stSidebar"] {
        background: #111827;
    }

    div.stButton > button[kind="primary"] {
        background: #C74256;
        border: none;
        color: white;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# HEADER AND PROGRESS
# --------------------------------------------------

st.title("🎬 CineMatch Review Studio")

st.caption(
    "Evaluate movie recommendations using human "
    "judgments. Model identities are hidden "
    "during review."
)

completed = reviews.apply(
    is_reviewed,
    axis=1,
)

total = len(reviews)
done = int(completed.sum())

st.progress(
    done / total if total else 0,
    text=f"{done} of {total} movie pairs reviewed",
)


# --------------------------------------------------
# SIDEBAR FILTERS
# --------------------------------------------------

with st.sidebar:
    st.header("Review controls")

    anchors = sorted(
        reviews["anchor_title"]
        .dropna()
        .unique()
    )

    preferred = [
        "Inception",
        "The Batman",
        "La La Land",
    ]

    default_anchor = next(
        (
            title
            for title in preferred
            if title in anchors
        ),
        anchors[0],
    )

    selected_anchor = st.selectbox(
        "Reference movie",
        anchors,
        index=anchors.index(default_anchor),
    )

    show_completed = st.checkbox(
        "Include completed reviews",
        value=False,
    )

    st.divider()

    st.caption(
        "Narrative and experience scores are "
        "required. Reliability is optional."
    )


# --------------------------------------------------
# CHOOSE NEXT REVIEW
# --------------------------------------------------

subset = reviews[
    reviews["anchor_title"] == selected_anchor
].copy()

if not show_completed:
    subset = subset[
        ~subset.apply(is_reviewed, axis=1)
    ]

subset = subset.reset_index(drop=True)

if subset.empty:
    st.success(
        "No remaining reviews for this movie "
        "under the current filter."
    )
    st.stop()

candidate_options = subset[
    "pair_id"
].tolist()

selected_pair = st.selectbox(
    "Recommendation to review",
    candidate_options,
    format_func=lambda identifier: (
        subset.loc[
            subset["pair_id"] == identifier,
            "candidate_title",
        ].iloc[0]
    ),
)

review = subset[
    subset["pair_id"] == selected_pair
].iloc[0]


# --------------------------------------------------
# MOVIE COMPARISON
# --------------------------------------------------

left, right = st.columns(
    2,
    gap="large",
)

with left:
    display_movie(
        review["anchor_id"],
        "REFERENCE MOVIE",
    )

with right:
    display_movie(
        review["candidate_id"],
        "RECOMMENDED MOVIE",
    )

st.divider()


# --------------------------------------------------
# HUMAN JUDGMENTS
# --------------------------------------------------

st.subheader(
    "Evaluate this recommendation"
)

st.caption(
    "Base your judgment on the relationship "
    "between the two movies, not their popularity."
)

rating_options = {
    "Not rated": None,
    "0 — No meaningful connection": 0,
    "1 — Weak connection": 1,
    "2 — Reasonably relevant": 2,
    "3 — Strongly relevant": 3,
}

reverse_options = {
    value: label
    for label, value in rating_options.items()
}


def rating_index(value):
    current = rating_value(value)

    label = reverse_options.get(
        current,
        "Not rated",
    )

    return list(
        rating_options
    ).index(label)


with st.form(
    key=f"review_form_{selected_pair}",
):
    narrative_label = st.selectbox(
        "Narrative relevance",
        list(rating_options),
        index=rating_index(
            review["narrative_relevance"]
        ),
        help=(
            "Compare story, characters, conflict, "
            "themes and narrative structure."
        ),
    )

    experience_label = st.selectbox(
        "Experiential relevance",
        list(rating_options),
        index=rating_index(
            review["experience_relevance"]
        ),
        help=(
            "Compare mood, atmosphere, pacing "
            "and overall viewing experience."
        ),
    )

    reliability_options = {
        "Not assessed": None,
        "0 — Insufficient evidence": 0,
        "1 — Moderate confidence": 1,
        "2 — Strong confidence": 2,
    }

    previous_reliability = rating_value(
        review["reliability"]
    )

    reliability_default = next(
        (
            label
            for label, value
            in reliability_options.items()
            if value == previous_reliability
        ),
        "Not assessed",
    )

    reliability_label = st.selectbox(
        "Reliability (optional)",
        list(reliability_options),
        index=list(
            reliability_options
        ).index(reliability_default),
        help=(
            "How much evidence supports your "
            "judgment? Consider familiarity, "
            "metadata completeness and rating evidence."
        ),
    )

    notes = st.text_area(
        "Review notes (optional)",
        value=clean_text(
            review["review_notes"]
        ),
        placeholder=(
            "What makes these movies similar "
            "or different?"
        ),
    )

    submitted = st.form_submit_button(
        "Save review",
        type="primary",
        use_container_width=True,
    )


# --------------------------------------------------
# SAVE
# --------------------------------------------------

if submitted:
    narrative = rating_options[
        narrative_label
    ]

    experience = rating_options[
        experience_label
    ]

    if (
        narrative is None
        or experience is None
    ):
        st.error(
            "Please rate both narrative and "
            "experiential relevance."
        )

    else:
        save_review(
            pair_id=selected_pair,
            narrative=narrative,
            experience=experience,
            reliability=reliability_options[
                reliability_label
            ],
            notes=notes,
        )

        st.rerun()


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "CineMatch human evaluation benchmark. "
    "Judgments are saved locally to "
    "data/evaluation/human_reviews.csv."
)
