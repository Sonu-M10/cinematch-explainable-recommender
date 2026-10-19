
"""Prepare the CineMatch Classic catalog for cloud deployment."""

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CATALOG = DATA_DIR / "movies_expanded.csv"


def run_script(script):
    """Run an existing project script and stop if it fails."""
    print(f"[CineMatch setup] Running {script}", flush=True)

    subprocess.run(
        [sys.executable, str(ROOT / script)],
        cwd=ROOT,
        check=True,
    )


def validate_catalog():
    """Check the generated catalog before loading the engine."""
    if not CATALOG.is_file() or CATALOG.stat().st_size == 0:
        return False

    try:
        data = pd.read_csv(
            CATALOG,
            usecols=["id", "title", "overview"],
            low_memory=False,
        )

        valid = (
            not data.empty
            and data["id"].notna().all()
            and data["id"].is_unique
            and data["title"].notna().all()
            and data["overview"].notna().all()
        )

        if valid:
            print(
                f"[CineMatch setup] Catalog ready: "
                f"{len(data):,} movies",
                flush=True,
            )

        return bool(valid)

    except (OSError, ValueError, KeyError, pd.errors.ParserError):
        return False


def ensure_classic_data():
    """Download and build the catalog only when needed."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if validate_catalog():
        return

    if CATALOG.exists():
        raise RuntimeError(
            "An existing catalog failed validation. "
            "Refusing to overwrite it automatically."
        )

    print(
        "[CineMatch setup] Catalog missing. "
        "Preparing it from the original source.",
        flush=True,
    )

    run_script("download_source.py")
    run_script("src/clean.py")
    run_script("src/metadata_quality.py")

    if not validate_catalog():
        raise RuntimeError(
            "Catalog generation completed, but validation failed."
        )

    print(
        "[CineMatch setup] Classic catalog setup complete.",
        flush=True,
    )


if __name__ == "__main__":
    ensure_classic_data()
