"""Synthetic storage/fault fixtures; no planner or benchmark executions."""
from collections import namedtuple
import errno
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tools"))
from native_paper_v4_storage import ArtifactStore, DiskBudgetPolicy, GIB, MANIFEST, file_hash, preflight


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="v4_storage_synthetic_")
        self.workspace = Path(self.temporary.name)
        self.store = ArtifactStore(self.workspace)
        self.directory = self.workspace/"active/w000/map_101"
        self.directory.mkdir(parents=True)
        (self.directory/"trial_one/snapshots/s0").mkdir(parents=True)
        (self.directory/"empty_directory").mkdir()
        self.files = {"trial_one/request.json": b'{"planner_seed":11}\n',
                      "trial_one/worker.json": b'{"status":"EXACT_SOLUTION"}\n',
                      "trial_one/snapshots/s0/path_xy.csv": b"x_m,y_m\n1.25,2.5\n3.5,4.75\n",
                      "trial_one/binary.npz": bytes(range(256))*50,
                      "trial_one/empty.log": b""}
        for name, payload in self.files.items():
            (self.directory/name).write_bytes(payload)
        (self.workspace/"data").mkdir()
        (self.workspace/"data/immutable_input.txt").write_text("retained input")

    def tearDown(self):
        self.temporary.cleanup()

    def test_lossless_pack_materialize_and_idempotent_index(self):
        receipt = self.store.pack_map("w000", "map_101")
        self.assertFalse(self.directory.exists())
        self.assertEqual(receipt["file_count"], len(self.files))
        self.assertEqual(receipt["uncompressed_bytes"], sum(map(len, self.files.values())))
        archive = self.workspace/receipt["archive"]
        self.assertEqual(file_hash(archive), receipt["sha256"])
        original_index = self.store.index_path.read_bytes()
        with self.store.materialize_map("w000", "map_101") as restored:
            self.assertEqual(restored, self.directory)
            self.assertTrue((restored/"empty_directory").is_dir())
            for name, payload in self.files.items():
                self.assertEqual((restored/name).read_bytes(), payload)
                self.assertEqual(file_hash(restored/name), receipt["inventory"][name]["sha256"])
        self.assertFalse(self.directory.exists())
        self.assertEqual(self.store.pack_map("w000", "map_101"), receipt)
        self.assertEqual(self.store.index_path.read_bytes(), original_index)
        self.assertEqual((self.workspace/"data/immutable_input.txt").read_text(), "retained input")

    def test_unsealed_active_validation_can_write_and_is_not_removed(self):
        with self.store.materialize_map("w000", "map_101") as active:
            (active/"validation.json").write_text('{"valid":true}')
        self.assertTrue(self.directory.exists())
        receipt = self.store.pack_map("w000", "map_101", cleanup=False)
        self.assertIn("validation.json", receipt["inventory"])
        with self.store.materialize_map("w000", "map_101") as active:
            self.assertTrue((active/"validation.json").exists())
        self.assertTrue(self.directory.exists())

    def test_materialized_mutation_is_preserved_and_reported(self):
        self.store.pack_map("w000", "map_101")
        with self.assertRaisesRegex(ValueError, "changed; preserving"):
            with self.store.materialize_map("w000", "map_101") as active:
                (active/"new_scientific_data.txt").write_text("must not delete")
        self.assertEqual((self.directory/"new_scientific_data.txt").read_text(), "must not delete")
        with self.assertRaisesRegex(ValueError, "refusing cleanup"):
            self.store.pack_map("w000", "map_101")
        self.assertTrue((self.directory/"trial_one/request.json").exists())

    def test_disk_full_during_pack_keeps_raw_and_quarantines_bad_partial_on_resume(self):
        error = OSError(errno.ENOSPC, "synthetic disk full")
        with patch("native_paper_v4_storage.shutil.copyfileobj", side_effect=error):
            with self.assertRaises(OSError):
                self.store.pack_map("w000", "map_101")
        self.assertTrue(self.directory.exists())
        self.assertFalse(self.store.index_path.exists())
        self.assertTrue((self.workspace/"shards/w000__map_101.zip.partial").exists())
        receipt = self.store.pack_map("w000", "map_101")
        self.assertEqual(receipt["file_count"], len(self.files))
        quarantined = list((self.workspace/"shards/quarantine").iterdir())
        self.assertEqual(len(quarantined), 2)  # Failed bytes and a diagnostic receipt.
        self.assertFalse(self.directory.exists())

    def test_archive_rename_before_index_commit_resumes_without_repacking(self):
        with patch.object(self.store, "_commit", side_effect=RuntimeError("synthetic crash before index")):
            with self.assertRaises(RuntimeError):
                self.store.pack_map("w000", "map_101")
        archive = self.workspace/"shards/w000__map_101.zip"
        digest = file_hash(archive)
        self.assertTrue(self.directory.exists())
        self.assertFalse(self.store.index_path.exists())
        receipt = self.store.pack_map("w000", "map_101")
        self.assertEqual(receipt["sha256"], digest)
        self.assertFalse(self.directory.exists())

    def test_verified_partial_adopted_without_repacking(self):
        with patch.object(self.store, "_commit", side_effect=RuntimeError("synthetic index interruption")):
            with self.assertRaises(RuntimeError):
                self.store.pack_map("w000", "map_101")
        archive = self.workspace/"shards/w000__map_101.zip"
        digest = file_hash(archive)
        archive.rename(archive.with_name(archive.name+".partial"))
        with patch("native_paper_v4_storage.shutil.copyfileobj", side_effect=AssertionError("must adopt existing bytes")):
            receipt = self.store.pack_map("w000", "map_101")
        self.assertEqual(receipt["sha256"], digest)
        self.assertFalse(self.directory.exists())

    def test_interrupted_cleanup_resumes_only_verified_subset(self):
        original = Path.unlink
        deletions = 0
        def interrupted(path, *args, **kwargs):
            nonlocal deletions
            if path.is_relative_to(self.directory):
                deletions += 1
                if deletions == 2:
                    raise OSError("synthetic crash during cleanup")
            return original(path, *args, **kwargs)
        with patch.object(Path, "unlink", interrupted):
            with self.assertRaises(OSError):
                self.store.pack_map("w000", "map_101")
        self.assertTrue(self.store.index_path.exists())
        self.assertTrue(self.directory.exists())
        self.assertLess(sum(p.is_file() for p in self.directory.rglob("*")), len(self.files))
        self.store.pack_map("w000", "map_101")
        self.assertFalse(self.directory.exists())
        with self.store.materialize_map("w000", "map_101") as restored:
            for name, payload in self.files.items():
                self.assertEqual((restored/name).read_bytes(), payload)

    def test_archive_corruption_never_removes_active_files(self):
        receipt = self.store.pack_map("w000", "map_101", cleanup=False)
        archive = self.workspace/receipt["archive"]
        with archive.open("r+b") as stream:
            stream.seek(70);value = stream.read(1);stream.seek(70);stream.write(bytes([value[0]^1]))
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.store.pack_map("w000", "map_101")
        for name, payload in self.files.items():
            self.assertEqual((self.directory/name).read_bytes(), payload)

    def test_unsafe_ids_extra_artifacts_and_hardlinks_are_rejected(self):
        for wave, map_id in (("../data", "map_101"), ("w000", "/outside"), ("w__0", "map"), ("w", "a\\b")):
            with self.assertRaises(ValueError):
                self.store.pack_map(wave, map_id)
        outside = self.workspace/"data/immutable_input.txt"
        hardlink = self.directory/"hardlink.txt"
        try:
            hardlink.hardlink_to(outside)
        except OSError:
            hardlink = None
        if hardlink is not None:
            with self.assertRaisesRegex(ValueError, "hardlinked"):
                self.store.pack_map("w000", "map_101")
            hardlink.unlink()
        self.store.pack_map("w000", "map_101", cleanup=False)
        (self.directory/"new_data.txt").write_text("preserve")
        with self.assertRaisesRegex(ValueError, "refusing cleanup"):
            self.store.pack_map("w000", "map_101")
        self.assertEqual(outside.read_text(), "retained input")
        self.assertTrue((self.directory/"new_data.txt").exists())

    def test_archive_traversal_and_symlink_members_are_quarantined(self):
        for badname, mode in (("../escape", stat.S_IFREG|0o644), ("link", stat.S_IFLNK|0o777)):
            pending = self.workspace/"shards/w000__map_101.zip.partial"
            with zipfile.ZipFile(pending, "w") as archive:
                item = zipfile.ZipInfo(badname);item.create_system = 3;item.external_attr = mode << 16
                archive.writestr(item, b"unsafe")
            with self.assertRaises((ValueError, zipfile.BadZipFile)):
                self.store._verify(pending, "w000", "map_101")
            # pack_map quarantines the untrusted bytes and uses retained raw.
            self.store.pack_map("w000", "map_101", cleanup=False)
            (self.workspace/"shards/w000__map_101.zip").unlink()
            self.store.index_path.unlink()
        self.assertFalse((self.workspace/"escape").exists())
        self.assertTrue(self.directory.exists())

    def test_real_symlink_is_never_traversed_or_deleted(self):
        outside = self.workspace/"data/immutable_input.txt"
        link = self.directory/"outside_link"
        try:
            link.symlink_to(outside)
        except OSError:
            self.skipTest("host does not grant creation of real symlinks")
        with self.assertRaisesRegex(ValueError, "links are forbidden"):
            self.store.pack_map("w000", "map_101")
        self.assertTrue(link.is_symlink())
        self.assertEqual(outside.read_text(), "retained input")


class DiskBudgetTests(unittest.TestCase):
    def policy(self, free, used=0, output=0):
        usage = namedtuple("usage", "total used free")
        return DiskBudgetPolicy(disk_usage_fn=lambda _:usage(20*GIB, 20*GIB-free, free),
                                usage_fn=lambda p:output if Path(p).name == "shards" else used)

    def test_first_later_wave_guards_and_reserve(self):
        self.assertFalse(preflight("synthetic", policy=self.policy(6*GIB-1), first_wave=True)["allowed"])
        self.assertTrue(preflight("synthetic", policy=self.policy(6*GIB), first_wave=True)["allowed"])
        later = preflight("synthetic", policy=self.policy(4*GIB), raw_budget_bytes=GIB//2, pack_budget_bytes=GIB//2)
        self.assertTrue(later["allowed"])
        self.assertEqual(later["required_free_bytes"], 4*GIB)
        self.assertEqual(later["reserve_bytes"], 2*GIB)
        larger = preflight("synthetic", policy=self.policy(7*GIB), raw_budget_bytes=4*GIB, pack_budget_bytes=2*GIB)
        self.assertFalse(larger["allowed"])
        self.assertEqual(larger["required_free_bytes"], 8*GIB)

    def test_output_and_workspace_soft_limits_are_independent(self):
        value = preflight("synthetic", policy=self.policy(10*GIB), projected_output_bytes=16*GIB+1)
        self.assertEqual(value["reason"], "OUTPUT_SOFT_LIMIT")
        value = preflight("synthetic", policy=self.policy(10*GIB, used=14*GIB))
        self.assertEqual(value["reason"], "WORKSPACE_SOFT_LIMIT")
        for bad in (-1, 1., True):
            with self.assertRaises(ValueError):
                preflight("synthetic", policy=self.policy(10*GIB), raw_budget_bytes=bad)


if __name__ == "__main__":
    unittest.main()
