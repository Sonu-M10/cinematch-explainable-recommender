
"""Download and verify CineMatch's original movie dataset.

Usage:
    python3 download_source.py

The source is pinned to a specific Hugging Face revision.
Existing files are never overwritten.
"""

import hashlib
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parent

REPO_ID = "HenryWaltson/TMDB-IMDB-Movies-Dataset"
REVISION = "620ea877e884bfc96eb56a988ffc57f4586cc9b0"
SOURCE_FILENAME = "TMDB  IMDB Movies Dataset.csv"

EXPECTED_SHA256 = (
    "da825d814096de90374cc7e10c4c8ad0911baee0b01e420d5f57ff2184e48a25"
)

DESTINATION = ROOT / "data" / "movies.csv"


def calculate_sha256(path):
    """Calculate a file's SHA-256 checksum in chunks."""
    digest = hashlib.sha256()

    with path.open("rb") as source:
        for chunk in iter(
            lambda: source.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def download_source():
    """Download a pinned source dataset without overwriting files."""
    print("CineMatch Source Dataset Downloader")
    print("=" * 60)

    if DESTINATION.exists():
        print("Existing source dataset found.")

        actual_hash = calculate_sha256(DESTINATION)

        if actual_hash == EXPECTED_SHA256:
            print("[OK] Existing dataset matches expected SHA-256.")
            return True

        print("[FAIL] Existing dataset has an unexpected checksum.")
        print("No files were modified.")
        return False

    print(f"Repository: {REPO_ID}")
    print(f"Revision: {REVISION}")
    print(f"Source: {SOURCE_FILENAME}")

    try:
        cached_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=SOURCE_FILENAME,
            repo_type="dataset",
            revision=REVISION,
        )

        actual_hash = calculate_sha256(cached_path)

        if actual_hash != EXPECTED_SHA256:
            print("[FAIL] Downloaded file checksum mismatch.")
            return False

        DESTINATION.parent.mkdir(parents=True, exist_ok=True)

        temporary_path = DESTINATION.with_name(
            DESTINATION.name + ".partial"
        )

        if temporary_path.exists():
            print("[FAIL] A partial download file already exists.")
            return False

        try:
            with cached_path.open("rb") as source:
                with temporary_path.open("xb") as target:
                    shutil.copyfileobj(source, target)

            if calculate_sha256(temporary_path) != EXPECTED_SHA256:
                print("[FAIL] Copied file checksum mismatch.")
                return False

            if DESTINATION.exists():
                print("[FAIL] Destination appeared during download.")
                return False

            temporary_path.rename(DESTINATION)

        finally:
            if temporary_path.exists():
                temporary_path.unlink()

        print("[OK] Download completed and verified.")
        print(f"Saved: {DESTINATION.relative_to(ROOT)}")
        return True

    except (OSError, ValueError, RuntimeError) as exc:
        print(f"[FAIL] {exc}")
        return False


if __name__ == "__main__":
    raise SystemExit(0 if download_source() else 1)
