"""Same-browser preference persistence for CineMatch."""

import streamlit as st

STORAGE_KEY = "cinematch.preferences.v1"
BUCKETS = ("saved", "watched", "dismissed", "liked")


def empty_preferences():
    return {bucket: [] for bucket in BUCKETS}


def normalize_preferences(value):
    """Accept valid movie IDs from a browser snapshot."""
    if not isinstance(value, dict):
        return empty_preferences()

    cleaned = empty_preferences()

    for bucket in BUCKETS:
        values = value.get(bucket, [])

        if not isinstance(values, list):
            continue

        seen = set()

        for item in values[:1000]:
            if isinstance(item, bool):
                continue

            try:
                movie_id = int(item)
            except (TypeError, ValueError, OverflowError):
                continue

            if movie_id >= 0 and movie_id not in seen:
                cleaned[bucket].append(movie_id)
                seen.add(movie_id)

    return cleaned


_COMPONENT = st.components.v2.component(
    name="cinematch_browser_preferences",
    html="<span class='storage-status'></span>",
    js="""
    export default function({
        data,
        parentElement,
        setStateValue
    }) {
        const status = parentElement.querySelector(
            ".storage-status"
        );

        const key = data.storageKey;
        const command = data.command;

        try {
            if (command && command.id) {
                const ackKey = key + ".command." + command.id;

                if (sessionStorage.getItem(ackKey) !== "done") {
                    localStorage.setItem(
                        key,
                        JSON.stringify(command.preferences)
                    );

                    sessionStorage.setItem(ackKey, "done");
                }

                setStateValue("ack", command.id);
            }

            const raw = localStorage.getItem(key);

            const snapshot = raw === null
                ? {
                    saved: [],
                    watched: [],
                    dismissed: [],
                    liked: []
                }
                : JSON.parse(raw);

            setStateValue("snapshot", snapshot);

            status.textContent = "";
        } catch (error) {
            status.textContent =
                "Browser storage unavailable";

            setStateValue("error", String(error.message));
        }
    }
    """,
)


def render_storage(command=None):
    """Render the browser bridge and return its state."""
    return _COMPONENT(
        data={
            "storageKey": STORAGE_KEY,
            "command": command,
        },
        key="cinematch_browser_storage",
    )
