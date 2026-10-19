
"""Prepare the CineMatch Classic catalog safely for cloud deployment."""

import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CATALOG = DATA_DIR / "movies_expanded.csv"
LOCK_FILE = DATA_DIR / ".catalog_setup.lock"

LOCK_TIMEOUT = 900
POLL_INTERVAL = 2


def run_script(script):
    """Run a preprocessing script and fail if it exits unsuccessfully."""
    print(f"[CineMatch setup] Running {script}", flush=True)

    subprocess.run(
        [sys.executable, str(ROOT / script)],
        cwd=ROOT,
        check=True,
    )


def validate_catalog():
    """Check that the generated catalog is usable."""
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
                f"[CineMatch setup] Catalog ready: {len(data):,} movies",
                flush=True,
            )

        return bool(valid)

    except (OSError, ValueError, KeyError, pd.errors.ParserError):
        return False


def build_catalog():
    """Build the catalog after acquiring exclusive setup ownership."""
    if validate_catalog():
        return

    if CATALOG.exists():
        raise RuntimeError(
            "An existing catalog failed validation. "
            "Refusing to overwrite it automatically."
        )

    print(
        "[CineMatch setup] Preparing movie catalog.",
        flush=True,
    )

    run_script("download_source.py")
    run_script("src/clean.py")
    run_script("src/metadata_quality.py")

    if not validate_catalog():
        raise RuntimeError(
            "Catalog generation completed but validation failed."
        )

    print(
        "[CineMatch setup] Catalog setup complete.",
        flush=True,
    )


def ensure_classic_data():
    """Prevent concurrent executions from building the same catalog."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if validate_catalog():
        return

    start = time.monotonic()

    while True:
        try:
            with LOCK_FILE.open("x") as lock:
                lock.write(str(time.time()))
            break

        except FileExistsError:
            if validate_catalog():
                return

            if time.monotonic() - start > LOCK_TIMEOUT:
                raise TimeoutError(
                    "Timed out waiting for catalog setup. "
                    "Check for an interrupted deployment."
                )

            print(
                "[CineMatch setup] Another execution is preparing "
                "the catalog. Waiting...",
                flush=True,
            )
            time.sleep(POLL_INTERVAL)

    try:
        build_catalog()
    finally:
        LOCK_FILE.unlink(missing_ok=True)


if __name__ == "__main__":
    ensure_classic_data()
