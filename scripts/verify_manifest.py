#!/usr/bin/env python3
"""Verify every file recorded in MANIFEST.sha256."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATED_SUFFIXES = {".pyc", ".pyo"}
GENERATED_DIRECTORIES = {"__pycache__", ".pycache"}


def main() -> None:
    manifest = ROOT / "MANIFEST.sha256"
    entries = 0
    for line in manifest.read_text().splitlines():
        expected, relative = line.split("  ", 1)
        relative_path = Path(relative.removeprefix("./"))
        if (
            relative_path.suffix in GENERATED_SUFFIXES
            or any(part in GENERATED_DIRECTORIES for part in relative_path.parts)
        ):
            raise AssertionError(f"generated artifact recorded in manifest: {relative}")
        path = ROOT / relative_path
        if not path.is_file():
            raise AssertionError(f"manifest entry is missing: {relative}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise AssertionError(f"digest mismatch: {relative}")
        entries += 1
    print(f"manifest_ok entries={entries}")


if __name__ == "__main__":
    main()
