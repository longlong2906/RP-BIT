import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest import mock
import warnings
import zipfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import run_cloud_posthoc_validation as cloud
import apply_cloud_validation_patch as patch_apply


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


class SafeZipTests(unittest.TestCase):
    def _archive(self, root, entries):
        path = Path(root) / "fixture.zip"
        with zipfile.ZipFile(path, "w") as archive:
            for item in entries:
                if isinstance(item, zipfile.ZipInfo):
                    archive.writestr(item, b"target")
                else:
                    archive.writestr(item, b"data")
        return path

    def test_rejects_traversal_absolute_backslash_and_symlink(self):
        cases = []
        with tempfile.TemporaryDirectory() as temporary:
            for index, name in enumerate(("../escape", "/absolute")):
                folder = Path(temporary) / str(index)
                folder.mkdir()
                cases.append(self._archive(folder, [name]))
            symlink = zipfile.ZipInfo("link")
            symlink.create_system = 3
            symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
            folder = Path(temporary) / "symlink"
            folder.mkdir()
            cases.append(self._archive(folder, [symlink]))
            for path in cases:
                with self.subTest(path=path), zipfile.ZipFile(path) as archive:
                    with self.assertRaises(cloud.IntegrityError):
                        cloud._zip_members(archive, "fixture")
        # ZipInfo normalizes the native separator on Windows before writing;
        # exercise the canonical member-name gate directly for this case.
        with self.assertRaises(cloud.IntegrityError):
            cloud._safe_member_name("a\\b")

    def test_rejects_duplicate_and_case_colliding_members(self):
        with tempfile.TemporaryDirectory() as temporary:
            for names in (("same", "same"), ("Name", "name")):
                path = Path(temporary) / (names[0] + ".zip")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    with zipfile.ZipFile(path, "w") as archive:
                        for name in names:
                            archive.writestr(name, b"x")
                with zipfile.ZipFile(path) as archive, self.assertRaises(cloud.IntegrityError):
                    cloud._zip_members(archive, "fixture")

    def test_deterministic_zip_and_exclusive_publish(self):
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first.zip"
            second = Path(temporary) / "second.zip"
            members = {"b/file.txt": b"two", "a.json": b"one"}
            cloud._write_deterministic_zip(first, members)
            cloud._write_deterministic_zip(second, members)
            self.assertEqual(cloud._sha256_file(first), cloud._sha256_file(second))
            with self.assertRaises(FileExistsError):
                cloud._write_deterministic_zip(first, members)

    def test_persisted_tree_rejects_nested_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target.txt"
            target.write_text("immutable", encoding="utf-8")
            nested = root / "nested"
            nested.mkdir()
            link = nested / "link.txt"
            try:
                link.symlink_to(target)
            except OSError as error:
                del error
                link.write_text("simulated link", encoding="utf-8")
                original_lstat = Path.lstat

                def fake_lstat(path):
                    if path == link:
                        result = original_lstat(path)
                        return os.stat_result((stat.S_IFLNK | 0o777, *result[1:]))
                    return original_lstat(path)

                with mock.patch.object(Path, "lstat", fake_lstat), self.assertRaisesRegex(
                    cloud.IntegrityError, "symlink"
                ):
                    cloud._regular_tree_files(root, "fixture")
            else:
                with self.assertRaisesRegex(cloud.IntegrityError, "symlink"):
                    cloud._regular_tree_files(root, "fixture")


class ControlParityTests(unittest.TestCase):
    def _write_control(self, directory, *, numeric_delta=0.0, hash_value="a" * 64, wall=1.0):
        directory = Path(directory)
        record = {
            "valid": True,
            "errors": [],
            "cost_terrain_m": 10.0 + numeric_delta,
            "length_xy_m": 9.0,
            "cost_abs_error_m": 0.0,
            "minimum_footprint_clearance_m": 1.0,
            "maximum_slope_deg": 2.0,
            "validation_wall_s": wall,
            "method": "explicit_triangle_halfplane_clipping_and_quadratic_capsule_distance",
            "snapshot_id": "snapshot_0",
            "xy_sha256": hash_value,
            "surface_sha256": "b" * 64,
        }
        final = dict(record)
        final.pop("snapshot_id")
        final.pop("xy_sha256")
        final.pop("surface_sha256")
        final["exact_goal_reached"] = True
        final["snapshots"] = [record]
        event = {
            "event_type": "improvement",
            "snapshot_id": "snapshot_0",
            "elapsed_s": 0.2,
            "best_cost_terrain_m": 10.0,
            "independent_valid": True,
        }
        trial = {
            "independent_valid": True,
            "independently_recomputed_cost_m": 10.0 + numeric_delta,
            "length_xy_m": 9.0,
            "cost_abs_error_m": 0.0,
            "minimum_footprint_clearance_m": 1.0,
            "maximum_slope_deg": 2.0,
            "success": True,
            "snapshots_valid": True,
            "validation_wall_s": wall,
        }
        dump(directory / "validation.json", final)
        dump(directory / "validated_events.json", [event])
        dump(directory / "trial.json", trial)
        dump(directory / "snapshots/snapshot_0/validation.json", record)

    def test_ignores_wall_time_and_accepts_numeric_tolerance(self):
        with tempfile.TemporaryDirectory() as temporary:
            actual = Path(temporary) / "actual"
            expected = Path(temporary) / "expected"
            self._write_control(actual, numeric_delta=5e-9, wall=99.0)
            self._write_control(expected, numeric_delta=0.0, wall=0.01)
            result = cloud.compare_control_outputs(actual, expected, "control")
            self.assertEqual(result["status"], "PASS")
            self.assertTrue(result["validation_wall_s_ignored"])

    def test_rejects_numeric_hash_and_event_order_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for mode in ("numeric", "hash", "events"):
                actual = root / mode / "actual"
                expected = root / mode / "expected"
                self._write_control(actual)
                self._write_control(expected)
                if mode == "numeric":
                    value = json.loads((actual / "validation.json").read_text())
                    value["cost_terrain_m"] += 2e-8
                    dump(actual / "validation.json", value)
                elif mode == "hash":
                    value = json.loads((actual / "snapshots/snapshot_0/validation.json").read_text())
                    value["xy_sha256"] = "c" * 64
                    dump(actual / "snapshots/snapshot_0/validation.json", value)
                else:
                    value = json.loads((actual / "validated_events.json").read_text())
                    value[0]["snapshot_id"] = "snapshot_1"
                    dump(actual / "validated_events.json", value)
                with self.subTest(mode=mode), self.assertRaises(cloud.ControlMismatch):
                    cloud.compare_control_outputs(actual, expected, "control")


class CheckpointTests(unittest.TestCase):
    def test_registered_python_dependencies_are_present(self):
        cloud._verify_runtime_packages()

    def test_checkpoint_inventory_detects_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / "nested/data.json"
            path.parent.mkdir()
            path.write_bytes(b"original")
            records = [
                {
                    "path": "nested/data.json",
                    "size_bytes": len(b"original"),
                    "sha256": cloud._sha256_bytes(b"original"),
                }
            ]
            cloud._verify_inventory(directory, records, "checkpoint")
            path.write_bytes(b"tampered")
            with self.assertRaises(cloud.IntegrityError):
                cloud._verify_inventory(directory, records, "checkpoint")

    def test_primary_inventory_uses_digest_of_entries_not_json_file_hash(self):
        entries = [
            {
                "path": "trial/file.txt",
                "size_bytes": 3,
                "sha256": "a" * 64,
            }
        ]
        digest = cloud.hashlib.sha256(
            json.dumps(entries, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
        payload = cloud._json_bytes(
            {
                "schema_version": 1,
                "policy": "all_regular_files_exact",
                "entries": entries,
                "inventory_sha256": digest,
            }
        )
        self.assertNotEqual(cloud._sha256_bytes(payload), digest)
        checked = cloud._verify_primary_inventory(payload, digest)
        self.assertEqual(checked["inventory_sha256"], digest)

    def test_resume_layout_allows_only_completed_prefix_and_current_trial(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "done").mkdir()
            (root / "next").mkdir()
            checkpoint = {
                "completed_trial_ids": ["done"],
                "validation_order": ["done", "next", "later"],
            }
            cloud._verify_resume_runs_layout(root, checkpoint)
            (root / "unrelated").mkdir()
            with self.assertRaises(cloud.IntegrityError):
                cloud._verify_resume_runs_layout(root, checkpoint)

    def test_jobs_other_than_one_fail_before_capsule_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(cloud.IntegrityError, "jobs must equal 1"):
                cloud.run_cloud_validation(
                    root / "missing.zip",
                    root / "work",
                    root / "patch.zip",
                    jobs=4,
                )

    def test_windows_map_path_is_exact_and_fail_closed(self):
        accepted = r"D:\BIT_2.5D\data\a3_vs_all_v1\rolling_terrain_5007.t25d"
        match = cloud.WINDOWS_MAP_PATH.fullmatch(accepted)
        self.assertIsNotNone(match)
        self.assertEqual(match.group("map_id"), "rolling_terrain_5007")
        for rejected in (
            r"C:\BIT_2.5D\data\a3_vs_all_v1\rolling_terrain_5007.t25d",
            r"D:\BIT_2.5D\data\other\rolling_terrain_5007.t25d",
            "D:/BIT_2.5D/data/a3_vs_all_v1/rolling_terrain_5007.t25d",
            r"D:\BIT_2.5D\data\a3_vs_all_v1\..\secret.t25d",
        ):
            self.assertIsNone(cloud.WINDOWS_MAP_PATH.fullmatch(rejected))


class PatchCompatibilityTests(unittest.TestCase):
    def test_runner_patch_is_accepted_by_local_applier_inspector(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            extracted = root / "extracted"
            runs = root / "runs"
            extracted.mkdir()
            runs.mkdir()
            trial_ids = [f"trial_{index:03d}" for index in range(cloud.EXPECTED_PENDING_COUNT)]
            inventory = {
                "schema_version": 1,
                "policy": "all_regular_files_exact",
                "entries": [],
                "inventory_sha256": cloud.hashlib.sha256(b"[]").hexdigest(),
            }
            dump(extracted / "primary_inventory.json", inventory)
            checkpoint_trials = {}
            for trial_id in trial_ids:
                directory = runs / trial_id
                dump(directory / "trial.json", {"trial_id": trial_id, "independent_valid": True})
                dump(directory / "validation.json", {"valid": True, "snapshots": []})
                dump(directory / "validated_events.json", [])
                checkpoint_trials[trial_id] = {
                    "input_entries": [],
                    "output_entries": cloud._validation_output_inventory(directory),
                    "invalid": False,
                }
            campaign_counts = {
                "planned": 250,
                "executed": 250,
                "not_run": 0,
                "offline_validated": 115,
                "validation_pending": 135,
                "validated_success": 115,
                "geometric_invalid": 0,
            }
            capsule = cloud.VerifiedCapsule(
                root / "unused.zip",
                "c" * 64,
                "d" * 64,
                {
                    "pending_trial_ids": trial_ids,
                    "validation_order": trial_ids,
                    "primary": {
                        "primary_inventory_sha256": inventory["inventory_sha256"],
                        "campaign_counts": campaign_counts,
                    },
                },
            )
            checkpoint = {
                "completed_trial_ids": trial_ids,
                "trials": checkpoint_trials,
                "control_results": [
                    {
                        "trial_id": trial_id,
                        "status": "PASS",
                        "snapshots_compared": 1,
                        "validation_wall_s_ignored": True,
                        "numeric_absolute_tolerance": 1e-8,
                    }
                    for trial_id in cloud.EXPECTED_CONTROL_TRIAL_IDS
                ],
                "cloud_environment": {
                    "schema_version": 1,
                    "execution_role": "posthoc_independent_validation_only",
                    "system": "Linux",
                    "jobs": 1,
                    "planner_trials_launched": 0,
                    "seed_changes": 0,
                },
            }
            patch = root / "patch.zip"
            manifest, patch_hash = cloud._build_patch(
                capsule, extracted, runs, checkpoint, patch
            )
            inspected = patch_apply.inspect_patch(patch)
            self.assertEqual(inspected["archive_sha256"], patch_hash)
            self.assertEqual(inspected["manifest"], manifest)
            self.assertEqual(manifest["source_primary"]["counts"], campaign_counts)
            self.assertEqual(manifest["status"], "COMPLETE")


if __name__ == "__main__":
    unittest.main()
