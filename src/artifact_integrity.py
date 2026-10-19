
"""Verify CineMatch artifacts against the versioned data manifest."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "data_manifest.json"


def calculate_sha256(path):
    """Calculate a file's SHA-256 hash without loading it into memory."""
    digest = hashlib.sha256()

    with path.open("rb") as source:
        for chunk in iter(
            lambda: source.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def verify_manifest():
    """Check all files listed in data_manifest.json."""
    print("\nVerifying CineMatch artifact manifest")
    print("-" * 60)

    if not MANIFEST_PATH.is_file():
        print("[FAIL] data_manifest.json is missing.")
        return False

    try:
        manifest = json.loads(
            MANIFEST_PATH.read_text(encoding="utf-8")
        )

        if manifest.get("schema_version") != 1:
            raise ValueError("Unsupported manifest schema version.")

        artifacts = manifest.get("artifacts")

        if not isinstance(artifacts, dict) or not artifacts:
            raise ValueError("Manifest has no valid artifacts.")

        all_valid = True

        for filename, expected in artifacts.items():
            relative = Path(filename)

            if (
                relative.is_absolute()
                or ".." in relative.parts
                or not relative.parts
            ):
                print(f"[FAIL] Invalid artifact path: {filename}")
                all_valid = False
                continue

            path = ROOT / relative

            if not path.is_file():
                print(f"[MISSING] {filename}")
                all_valid = False
                continue

            expected_size = expected["size_bytes"]
            actual_size = path.stat().st_size

            if actual_size != expected_size:
                print(f"[FAIL] Size mismatch: {filename}")
                all_valid = False
                continue

            expected_hash = expected["sha256"]
            actual_hash = calculate_sha256(path)

            if actual_hash != expected_hash:
                print(f"[FAIL] SHA-256 mismatch: {filename}")
                all_valid = False
                continue

            print(f"[OK] {filename}")

        return all_valid

    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"[FAIL] Manifest validation: {exc}")
        return False
