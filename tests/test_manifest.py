from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts import verify_manifest


class ManifestVerifierTests(unittest.TestCase):
    def _run_manifest(self, line: str, files: dict[str, bytes] | None = None) -> None:
        with tempfile.TemporaryDirectory(prefix="manifest-test-") as temporary:
            root = Path(temporary)
            previous_root = verify_manifest.ROOT
            verify_manifest.ROOT = root
            try:
                for relative, content in (files or {}).items():
                    (root / relative).write_bytes(content)
                (root / "MANIFEST.sha256").write_text(line, encoding="utf-8")
                verify_manifest.main()
            finally:
                verify_manifest.ROOT = previous_root

    def test_valid_relative_entry_is_accepted(self) -> None:
        content = b"fixture\n"
        digest = hashlib.sha256(content).hexdigest()
        self._run_manifest(
            f"{digest}  ./fixture.txt\n",
            {"fixture.txt": content},
        )

    def test_parent_and_absolute_entries_are_rejected(self) -> None:
        digest = "0" * 64
        for path in ("../outside.txt", "/etc/passwd", ".git/config"):
            with self.assertRaises(AssertionError):
                self._run_manifest(f"{digest}  {path}\n")

    def test_symlink_component_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="manifest-test-") as temporary:
            root = Path(temporary)
            (root / "outside").mkdir()
            (root / "outside" / "fixture.txt").write_text("fixture\n")
            (root / "link").symlink_to(root / "outside", target_is_directory=True)
            previous_root = verify_manifest.ROOT
            verify_manifest.ROOT = root
            try:
                (root / "MANIFEST.sha256").write_text(
                    f"{'0' * 64}  link/fixture.txt\n",
                    encoding="utf-8",
                )
                with self.assertRaises(AssertionError):
                    verify_manifest.main()
            finally:
                verify_manifest.ROOT = previous_root

    def test_malformed_line_and_digest_are_rejected(self) -> None:
        with self.assertRaises(AssertionError):
            self._run_manifest("malformed\n")
        with self.assertRaises(AssertionError):
            self._run_manifest(f"{'x' * 64}  missing.txt\n")


if __name__ == "__main__":
    unittest.main()
