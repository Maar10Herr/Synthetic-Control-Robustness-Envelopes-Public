#!/usr/bin/env python3
"""Verify every file recorded in MANIFEST.sha256."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATED_SUFFIXES = {".pyc", ".pyo"}
GENERATED_DIRECTORIES = {"__pycache__", ".pycache"}
NON_RELEASE_DIRECTORIES = {".git", ".venv", ".uv-cache", "build", "dist"}
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _release_files() -> set[str]:
    files = set()
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if any(part in NON_RELEASE_DIRECTORIES for part in relative.parts):
            continue
        if (
            relative.name == "MANIFEST.sha256"
            or relative.suffix in GENERATED_SUFFIXES
            or any(part in GENERATED_DIRECTORIES for part in relative.parts)
        ):
            continue
        if path.is_symlink():
            raise AssertionError(f"symbolic link is not a release file: {relative}")
        if path.is_file():
            files.add(str(relative))
    return files


def _manifest_entry(raw_path: str) -> tuple[Path, str]:
    """Return a safe, normalized manifest path and its release name."""

    relative = Path(raw_path.removeprefix("./"))
    if (
        not raw_path
        or relative.is_absolute()
        or relative == Path(".")
        or ".." in relative.parts
        or any(part in NON_RELEASE_DIRECTORIES for part in relative.parts)
    ):
        raise AssertionError(f"manifest path is not a safe relative path: {raw_path!r}")
    path = ROOT / relative
    current = ROOT
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise AssertionError(f"symbolic link is not a release file: {raw_path}")
    try:
        path.resolve().relative_to(ROOT.resolve())
    except ValueError as error:
        raise AssertionError(f"manifest path escapes repository: {raw_path}") from error
    return path, str(relative)


def main() -> None:
    manifest = ROOT / "MANIFEST.sha256"
    entries = 0
    listed = set()
    for line in manifest.read_text().splitlines():
        try:
            expected, raw_path = line.split("  ", 1)
        except ValueError as error:
            raise AssertionError(f"malformed manifest line: {line!r}") from error
        if not SHA256.fullmatch(expected):
            raise AssertionError(f"invalid SHA-256 digest: {expected!r}")
        path, relative = _manifest_entry(raw_path)
        relative_path = Path(relative)
        if (
            relative_path.suffix in GENERATED_SUFFIXES
            or any(part in GENERATED_DIRECTORIES for part in relative_path.parts)
        ):
            raise AssertionError(f"generated artifact recorded in manifest: {raw_path}")
        if not path.is_file():
            raise AssertionError(f"manifest entry is missing: {raw_path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise AssertionError(f"digest mismatch: {raw_path}")
        if relative in listed:
            raise AssertionError(f"duplicate manifest entry: {raw_path}")
        listed.add(relative)
        entries += 1
    unlisted = _release_files() - listed
    if unlisted:
        raise AssertionError(f"unlisted release files: {sorted(unlisted)}")
    stale = listed - _release_files()
    if stale:
        raise AssertionError(f"stale manifest entries: {sorted(stale)}")
    print(f"manifest_ok entries={entries}")


if __name__ == "__main__":
    main()
