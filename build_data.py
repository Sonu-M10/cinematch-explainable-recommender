
"""Reproduce CineMatch preprocessing in an isolated workspace.

Usage:
    python3 build_data.py
    python3 build_data.py --stage imdb
    python3 build_data.py --stage imdb --keep-output

Existing project datasets are never overwritten.
"""

import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent

EXPECTED_HASHES = {
    "data/movies.csv": (
        "da825d814096de90374cc7e10c4c8ad0911baee0b01e420d5f57ff2184e48a25"
    ),
    "data/review_audit/title.basics.tsv.gz": (
        "f4cdbc4a13c6d5e96356c6b1f44de89e715bb2e2eb9cfb500ca4381af227df56"
    ),
    "data/imdb/title.ratings.tsv.gz": (
        "e3ee78c5367c7f9db435d9d900f0f92e0aebcaa14ec85147c7a3ade5bd3bb2ab"
    ),
}

STAGES = {
    "catalog": [
        "src/clean.py",
        "src/metadata_quality.py",
    ],
    "imdb": [
        "src/clean.py",
        "src/metadata_quality.py",
        "src/imdb_ratings_enrichment.py",
    ],
}

OUTPUTS = {
    "catalog": [
        "data/movies_cleaned.csv",
        "data/movies_expanded.csv",
    ],
    "imdb": [
        "data/movies_cleaned.csv",
        "data/movies_expanded.csv",
        "data/movies_with_imdb.csv",
        "data/imdb/imdb_matches.csv",
    ],
}


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def copy_verified_file(relative_path, workspace):
    source = ROOT / relative_path
    destination = workspace / relative_path

    if not source.is_file():
        raise FileNotFoundError(
            f"Required input is missing: {relative_path}"
        )

    print(f"Verifying {relative_path}...", flush=True)

    if sha256_file(source) != EXPECTED_HASHES[relative_path]:
        raise ValueError(
            f"Checksum mismatch: {relative_path}"
        )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(source, destination)


def prepare_workspace(workspace, stage):
    inputs = ["data/movies.csv"]

    if stage == "imdb":
        inputs.extend([
            "data/review_audit/title.basics.tsv.gz",
            "data/imdb/title.ratings.tsv.gz",
        ])

    for filename in inputs:
        copy_verified_file(filename, workspace)

    for script in STAGES[stage]:
        source = ROOT / script
        destination = workspace / script

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copy2(source, destination)


def run_pipeline(workspace, stage):
    for script in STAGES[stage]:
        print(f"\nRunning {script}", flush=True)
        print("-" * 60, flush=True)

        subprocess.run(
            [sys.executable, script],
            cwd=workspace,
            check=True,
        )

    for filename in OUTPUTS[stage]:
        path = workspace / filename

        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(
                f"Missing generated output: {filename}"
            )

        print(
            f"[OK] {filename}: "
            f"{path.stat().st_size / 1024**2:.2f} MiB"
        )


def compare_outputs(workspace, stage):
    print("\nReproducibility report")
    print("=" * 60)

    all_match = True

    for filename in OUTPUTS[stage]:
        rebuilt = workspace / filename
        existing = ROOT / filename

        if not existing.is_file():
            print(f"[NO BASELINE] {filename}")
            continue

        match = sha256_file(rebuilt) == sha256_file(existing)

        print(
            f"[{'MATCH' if match else 'DIFFERENT'}] {filename}"
        )

        if not match:
            all_match = False

    return all_match


def main():
    parser = argparse.ArgumentParser(
        description="Safely reproduce CineMatch datasets."
    )

    parser.add_argument(
        "--stage",
        choices=["catalog", "imdb"],
        default="catalog",
    )

    parser.add_argument(
        "--keep-output",
        action="store_true",
    )

    args = parser.parse_args()

    print("CineMatch Reproducibility Build")
    print("=" * 60)
    print(f"Stage: {args.stage}", flush=True)

    try:
        with tempfile.TemporaryDirectory(
            prefix="cinematch-build-"
        ) as temporary:
            workspace = Path(temporary)

            prepare_workspace(workspace, args.stage)
            run_pipeline(workspace, args.stage)

            all_match = compare_outputs(
                workspace,
                args.stage,
            )

            if args.keep_output:
                destination = ROOT / "build_output"

                if destination.exists():
                    raise FileExistsError(
                        "build_output already exists."
                    )

                shutil.copytree(
                    workspace / "data",
                    destination,
                )

                print(
                    f"\nSaved rebuilt data to {destination}"
                )

        print("\nOriginal project datasets were not modified.")

        if all_match:
            print("[SUCCESS] All available baselines match.")
            return 0

        print("[WARNING] Some rebuilt files differ.")
        return 1

    except (
        OSError,
        ValueError,
        RuntimeError,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"\n[FAILED] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
