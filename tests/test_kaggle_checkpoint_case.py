"""Synthetic Linux filename regressions; no planner execution or real runtime data."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import kaggle_a3_bundle as archive_tools
import kaggle_terrain_compare_bundle as checkpoint


class CheckpointCaseTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="synthetic-checkpoint-case-", dir=ROOT / "tests/fixtures")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.upper = self.root / "first-fixture.txt"
        self.lower = self.root / "second-fixture.txt"
        self.upper.write_bytes(b"synthetic uppercase terminfo")
        self.lower.write_bytes(b"synthetic lowercase terminfo")
        # ZIP names are case-sensitive even when the host filesystem is Windows.
        base = "repo/.managed-python/cpython-3.10.19-linux-x86_64-gnu/share/terminfo/2/"
        self.entries = [(base + "2621A", self.upper), (base + "2621a", self.lower)]

    def make_linux_archive(self, name="linux.zip"):
        output = self.root / name
        archive_tools.archive(self.entries, output, checkpoint.CHECKPOINT_KIND,
                              path_case_policy=archive_tools.LINUX_CASE_POLICY)
        return output

    def rewrite(self, source, name, change):
        destination = self.root / name
        with zipfile.ZipFile(source) as original, zipfile.ZipFile(destination, "w") as rewritten:
            manifest = json.loads(original.read("bundle.json"))
            change(manifest)
            for entry in original.infolist():
                data = json.dumps(manifest) if entry.filename == "bundle.json" else original.read(entry.filename)
                rewritten.writestr(entry, data)
        return destination

    def test_linux_zip_keeps_distinct_terminfo_bytes_and_hashes(self):
        output = self.make_linux_archive()
        manifest = archive_tools.verify(output, archive_tools.sha256(output))
        self.assertEqual(manifest["path_case_policy"], archive_tools.LINUX_CASE_POLICY)
        self.assertEqual([item["path"] for item in manifest["files"]], [name for name, _ in self.entries])
        self.assertNotEqual(manifest["files"][0]["sha256"], manifest["files"][1]["sha256"])
        with zipfile.ZipFile(output) as content:
            for name, source in self.entries:
                self.assertEqual(content.read(name), source.read_bytes())

    def test_source_and_legacy_archives_keep_portable_casefold_contract(self):
        with self.assertRaisesRegex(ValueError, "duplicate archive member"):
            archive_tools.archive(self.entries, self.root / "portable.zip", checkpoint.KIND)
        linux = self.make_linux_archive()
        legacy = self.rewrite(linux, "legacy.zip", lambda manifest: manifest.pop("path_case_policy"))
        with self.assertRaisesRegex(ValueError, "duplicate archive member"):
            archive_tools.verify(legacy)
        source = self.root / "source.zip"
        manifest = archive_tools.archive([("source.txt", self.upper)], source, checkpoint.KIND)
        self.assertNotIn("path_case_policy", manifest)
        archive_tools.extract(source, self.root / "portable-destination")

    def test_exact_duplicates_forbidden_in_every_policy(self):
        for policy in (archive_tools.PORTABLE_CASE_POLICY, archive_tools.LINUX_CASE_POLICY):
            with self.subTest(policy=policy), self.assertRaisesRegex(ValueError, "duplicate"):
                archive_tools.archive([("same", self.upper), ("same", self.lower)],
                                      self.root / (policy + ".zip"), "SYNTHETIC", path_case_policy=policy)
        linux = self.make_linux_archive()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(linux, "a") as content:
                content.writestr(self.entries[0][0], b"duplicate")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            archive_tools.verify(linux)

    def test_unknown_policy_rejected_before_destination_writes(self):
        with self.assertRaisesRegex(ValueError, "unsupported"):
            archive_tools.archive([], self.root / "unknown.zip", "SYNTHETIC", path_case_policy="guess")
        self.assertFalse((self.root / "unknown.zip.partial").exists())
        linux = self.make_linux_archive()
        unknown = self.rewrite(linux, "unknown-manifest.zip", lambda manifest: manifest.update(path_case_policy="guess"))
        target = self.root / "not-created"
        with self.assertRaisesRegex(ValueError, "unsupported"):
            archive_tools.extract(unknown, target)
        self.assertFalse(target.exists())

    def test_case_sensitive_policy_keeps_traversal_and_checksum_checks(self):
        with self.assertRaisesRegex(ValueError, "unsafe"):
            archive_tools.archive([("../escape", self.upper)], self.root / "escape.zip", "SYNTHETIC",
                                  path_case_policy=archive_tools.LINUX_CASE_POLICY)
        linux = self.make_linux_archive()
        corrupt = self.root / "corrupt.zip"
        with zipfile.ZipFile(linux) as original, zipfile.ZipFile(corrupt, "w") as output:
            for entry in original.infolist():
                data = original.read(entry.filename)
                if entry.filename == self.entries[0][0]:
                    data = b"x" * len(data)
                output.writestr(entry, data)
        with self.assertRaisesRegex(ValueError, "checksum"):
            archive_tools.verify(corrupt)

    def test_non_linux_extraction_refuses_before_creating_target(self):
        linux = self.make_linux_archive()
        target = self.root / "not-created"
        with patch.object(archive_tools.sys, "platform", "win32"):
            with self.assertRaisesRegex(ValueError, "requires a Linux destination"):
                archive_tools.extract(linux, target)
        self.assertFalse(target.exists())

    def test_linux_export_explicitly_selects_case_sensitive_policy(self):
        repo, workspace = self.root / "repo", self.root / "campaign"
        runtime = repo / ".managed-python/native"
        runtime.parent.mkdir(parents=True)
        runtime.write_bytes(b"synthetic native runtime")
        workspace.mkdir()
        with patch.object(checkpoint.sys, "platform", "linux"):
            manifest = checkpoint.export_checkpoint(workspace, self.root / "export.zip", root=repo)
        self.assertEqual(manifest["path_case_policy"], archive_tools.LINUX_CASE_POLICY)
        self.assertIn("repo/.managed-python/native", [item["path"] for item in manifest["files"]])

    def link_fixture(self, policy, *, duplicate=False):
        repo, workspace = self.root / "repo", self.root / "campaign"
        repo.mkdir(exist_ok=True)
        workspace.mkdir(exist_ok=True)
        for name in ("target-one", "target-two"):
            (repo / name).mkdir(exist_ok=True)
        links = [{"path": "repo/venv/" + name, "target": "repo/" + destination,
                  "target_is_directory": True, "target_sha256": None}
                 for name, destination in (("Alias", "target-one"),
                                            ("Alias" if duplicate else "alias", "target-two"))]
        output = self.root / (policy + ("-duplicate" if duplicate else "-links") + ".zip")
        archive_tools.archive([("campaign/fixture", self.upper)], output, checkpoint.CHECKPOINT_KIND,
                              {"root_path": str(repo), "workspace_path": str(workspace),
                               "source_input_sha256": "synthetic", "internal_runtime_symlinks": links},
                              path_case_policy=policy)
        return output, repo, workspace

    def test_restore_alias_names_follow_manifest_policy(self):
        # Test name comparison and restore control flow on every host. This is
        # deliberately not a claim of a Linux filesystem roundtrip on Windows.
        linux, repo, workspace = self.link_fixture(archive_tools.LINUX_CASE_POLICY)
        aliases = []
        with patch.object(checkpoint, "validate_extraction_case_policy"), \
             patch.object(checkpoint, "extract", return_value=archive_tools.verify(linux)), \
             patch.object(Path, "symlink_to", lambda path, *args, **kwargs: aliases.append(path.name)):
            checkpoint.restore_checkpoint(linux, self.root, archive_tools.sha256(linux),
                                          source_sha256="synthetic", root=repo, workspace=workspace)
        self.assertEqual(aliases, ["Alias", "alias"])
        portable, repo, workspace = self.link_fixture(archive_tools.PORTABLE_CASE_POLICY)
        with self.assertRaisesRegex(ValueError, "duplicate internal runtime link"):
            checkpoint.restore_checkpoint(portable, self.root, archive_tools.sha256(portable),
                                          source_sha256="synthetic", root=repo, workspace=workspace)
        duplicate, repo, workspace = self.link_fixture(archive_tools.LINUX_CASE_POLICY, duplicate=True)
        with patch.object(checkpoint, "validate_extraction_case_policy"):
            with self.assertRaisesRegex(ValueError, "duplicate internal runtime link"):
                checkpoint.restore_checkpoint(duplicate, self.root, archive_tools.sha256(duplicate),
                                              source_sha256="synthetic", root=repo, workspace=workspace)

    def test_restore_checks_host_before_removing_any_alias(self):
        linux, repo, workspace = self.link_fixture(archive_tools.LINUX_CASE_POLICY)
        with patch.object(archive_tools.sys, "platform", "win32"), patch.object(Path, "unlink") as unlink:
            with self.assertRaisesRegex(ValueError, "requires a Linux destination"):
                checkpoint.restore_checkpoint(linux, self.root, archive_tools.sha256(linux),
                                              source_sha256="synthetic", root=repo, workspace=workspace)
            unlink.assert_not_called()

    @unittest.skipUnless(sys.platform.startswith("linux") and os.name != "nt", "actual Linux filesystem roundtrip required")
    def test_real_linux_export_extract_and_alias_roundtrip(self):
        repo, workspace = self.root / "repo", self.root / "campaign"
        runtime = repo / ".managed-python/native/share/terminfo/2"
        runtime.mkdir(parents=True)
        workspace.mkdir()
        first, second = runtime / "2621A", runtime / "2621a"
        first.write_bytes(self.upper.read_bytes())
        second.write_bytes(self.lower.read_bytes())
        self.assertNotEqual(first.read_bytes(), second.read_bytes())
        aliases = repo / ".venv-terrain-sgp/bin"
        aliases.mkdir(parents=True)
        (aliases / "PYTHON").symlink_to(first)
        (aliases / "python").symlink_to(second)
        source, output = self.root / "source.zip", self.root / "actual-linux.zip"
        source.write_bytes(b"synthetic source identity")
        checkpoint.export_checkpoint(workspace, output, root=repo, input_archive=source)
        for owned in (aliases / "PYTHON", aliases / "python", first, second):
            owned.unlink()
        for _ in range(2):
            checkpoint.restore_checkpoint(output, self.root, archive_tools.sha256(output),
                                          source_sha256=archive_tools.sha256(source), root=repo, workspace=workspace)
            self.assertEqual(first.read_bytes(), self.upper.read_bytes())
            self.assertEqual(second.read_bytes(), self.lower.read_bytes())
            self.assertTrue((aliases / "PYTHON").is_symlink())
            self.assertEqual((aliases / "PYTHON").resolve(), first)
            self.assertEqual((aliases / "python").resolve(), second)


if __name__ == "__main__":
    unittest.main()
