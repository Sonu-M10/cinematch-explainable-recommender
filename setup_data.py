"""CineMatch data setup and validation CLI."""

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run_script(filename, arguments=None):
    """Run another project script using the active Python environment."""
    command = [
        sys.executable,
        str(ROOT / filename),
        *(arguments or []),
    ]

    result = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
    )

    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Download, build, or validate CineMatch data."
    )

    actions = parser.add_mutually_exclusive_group()

    actions.add_argument(
        "--download",
        action="store_true",
        help="Download and verify the original movie dataset.",
    )

    actions.add_argument(
        "--build",
        action="store_true",
        help="Rebuild the catalog in an isolated workspace.",
    )

    parser.add_argument(
        "--stage",
        choices=["catalog", "imdb"],
        default="catalog",
        help="Build stage (used with --build).",
    )

    parser.add_argument(
        "--keep-output",
        action="store_true",
        help="Preserve rebuilt datasets in build_output/.",
    )

    parser.add_argument(
        "--engine",
        choices=[
            "all",
            "classic",
            "hybrid",
            "experimental",
            "ensemble",
        ],
        default="all",
        help="Engine whose runtime artifacts should be validated.",
    )

    parser.add_argument(
        "--verify-checksums",
        action="store_true",
        help="Verify runtime artifacts against data_manifest.json.",
    )

    args = parser.parse_args()

    if args.download:
        if args.keep_output or args.stage != "catalog" or args.verify_checksums:
            parser.error(
                "--download cannot be combined with build or checksum options."
            )

        return run_script("download_source.py")

    if args.build:
        arguments = ["--stage", args.stage]

        if args.keep_output:
            arguments.append("--keep-output")

        return run_script("build_data.py", arguments)

    if args.keep_output or args.stage != "catalog":
        parser.error("--stage and --keep-output require --build.")

    arguments = ["--engine", args.engine]

    if args.verify_checksums:
        arguments.append("--verify-checksums")

    return run_script("validate_data.py", arguments)


if __name__ == "__main__":
    raise SystemExit(main())
