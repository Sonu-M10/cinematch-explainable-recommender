
"""Preference-based discovery using the existing CineMatch catalog.

No streaming availability, age certification, or content-warning inference.
Hard filters apply only when the corresponding metadata exists.
"""
import re
import pandas as pd

MOODS = {
    'Funny': ('Comedy',),
    'Suspenseful': ('Thriller', 'Mystery'),
    'Comforting': ('Comedy', 'Family', 'Romance'),
    'Thought-provoking': ('Drama', 'Science Fiction', 'Documentary'),
    'Surprise me': (),
}


def available_fields(df):
    return {
        key: key in df.columns and df[key].notna().any()
        for key in (
            'runtime',
            'original_language',
            'release_date',
            'age_rating',
            'certification',
            'streaming_services',
            'trailer_url',
        )
    }


def apply_hard_filters(
    df,
    genre=None,
    max_runtime=None,
    language=None,
    year_min=None,
    year_max=None,
    age_rating=None,
    excluded_genres=(),
):
    """Return filtered catalog; reject requirements that cannot be verified."""
    out = df.copy()

    if genre:
        out = out[
            out['genres'].fillna('').str.contains(
                re.escape(genre),
                case=False,
                regex=True,
            )
        ]

    if excluded_genres:
        for excluded in excluded_genres:
            out = out[
                ~out['genres'].fillna('').str.contains(
                    re.escape(excluded),
                    case=False,
                    regex=True,
                )
            ]

    if max_runtime is not None:
        if 'runtime' not in out:
            raise ValueError(
                'Runtime data is unavailable; the runtime limit cannot be verified.'
            )

        runtime = pd.to_numeric(
            out['runtime'],
            errors='coerce',
        )

        out = out[
            runtime.notna()
            & (runtime > 0)
            & (runtime <= max_runtime)
        ]

    if language:
        if 'original_language' not in out:
            raise ValueError(
                'Language data is unavailable; the language filter cannot be verified.'
            )

        out = out[
            out['original_language']
            .fillna('')
            .astype(str)
            .str.casefold()
            == language.casefold()
        ]

    if year_min is not None or year_max is not None:
        if 'release_date' not in out:
            raise ValueError(
                'Release dates are unavailable; the year range cannot be verified.'
            )

        year = pd.to_numeric(
            out['release_date'].astype(str).str[:4],
            errors='coerce',
        )

        out = out[year.notna()]

        if year_min is not None:
            out = out[year >= year_min]

        if year_max is not None:
            out = out[year <= year_max]

    if age_rating:
        field = next(
            (
                f
                for f in ('age_rating', 'certification')
                if f in out
            ),
            None,
        )

        if field is None:
            raise ValueError(
                'Age-rating data is unavailable; this filter cannot be verified.'
            )

        out = out[
            out[field]
            .fillna('')
            .astype(str)
            .str.casefold()
            == age_rating.casefold()
        ]

    return out


def discover(
    df,
    *,
    genre=None,
    mood=None,
    max_runtime=None,
    language=None,
    year_min=None,
    year_max=None,
    age_rating=None,
    excluded_genres=(),
    hidden_ids=(),
    seed_id=None,
    recommender=None,
    limit=120,
):
    """Create a deterministic, varied shortlist while keeping hard constraints."""

    candidates = apply_hard_filters(
        df,
        genre,
        max_runtime,
        language,
        year_min,
        year_max,
        age_rating,
        excluded_genres,
    )

    if hidden_ids:
        candidates = candidates[
            ~candidates['id'].isin(hidden_ids)
        ]

    if candidates.empty:
        return candidates.copy()

    candidates = candidates.drop_duplicates('id').copy()

    candidates['_reason'] = (
        'Selected from the catalog using your filters and available ratings.'
    )

    candidates['_relevance'] = 0.0

    # Mood is a flexible ranking signal.
    if mood and mood != 'Surprise me':
        matches = MOODS.get(mood, ())

        match = candidates['genres'].fillna('').apply(
            lambda s: any(
                re.search(
                    r'\b' + re.escape(g) + r'\b',
                    str(s),
                    re.I,
                )
                for g in matches
            )
        )

        candidates['_relevance'] += (
            match.astype(float) * 2.0
        )

        candidates.loc[match, '_reason'] = (
            f'Its catalog genres align with your {mood.lower()} mood.'
        )

    # An example movie can contribute existing model rankings.
    if seed_id is not None and recommender is not None:
        recs = recommender(
            int(seed_id),
            top_n=300,
        )

        if recs is not None and not recs.empty:
            ranks = {
                int(mid): i
                for i, mid in enumerate(recs['id'], 1)
            }

            ranks_score = candidates['id'].map(
                lambda x: (
                    1.0 / (1 + ranks[int(x)])
                    if int(x) in ranks
                    else 0.0
                )
            )

            candidates['_relevance'] += (
                ranks_score * 90.0
            )

            match = candidates['id'].isin(ranks)

            candidates.loc[match, '_reason'] = (
                'Recommended by CineMatch based on your selected movie.'
            )

    # Rating evidence is used for ordering, not as
    # a probability of personal enjoyment.
    quality = pd.to_numeric(
        candidates.get(
            'weighted_rating',
            pd.Series(0, index=candidates.index),
        ),
        errors='coerce',
    ).fillna(0)

    votes = pd.to_numeric(
        candidates.get(
            'vote_count',
            pd.Series(0, index=candidates.index),
        ),
        errors='coerce',
    ).fillna(0)

    candidates['_quality'] = (
        quality
        + votes.clip(upper=1000) / 10000
    )

    candidates = candidates.sort_values(
        ['_relevance', '_quality', 'id'],
        ascending=[False, False, True],
    )

    # Avoid more than two consecutive movies
    # with the same first listed genre.
    pool = candidates.head(
        max(limit * 5, 250)
    )

    ordered = []
    remaining = list(pool.index)

    while remaining and len(ordered) < limit:
        pick = None

        for idx in remaining:
            genres = str(
                pool.at[idx, 'genres']
            ).split(',')

            first = (
                genres[0].strip().lower()
                if genres
                else ''
            )

            recent = [
                str(
                    pool.at[j, 'genres']
                ).split(',')[0].strip().lower()
                for j in ordered[-2:]
            ]

            if not (
                len(recent) == 2
                and recent[0] == recent[1] == first
                and first
            ):
                pick = idx
                break

        if pick is None:
            pick = remaining[0]

        ordered.append(pick)
        remaining.remove(pick)

    return (
        pool.loc[ordered]
        .reset_index(drop=True)
    )
