"""Safety regressions for the local Cloud validation patch importer."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from apply_cloud_validation_patch import (  # noqa: E402
    CLASSIFICATION,
    EXPECTED_CONTROL_TRIAL_IDS,
    EXPECTED_SOURCE_AT_RUN_SHA256,
    PATCH_TYPE,
    _allowed_trial_member,
    _require_pending_order,
    _validate_patched_trial,
    _verify_frozen_local_toolchain,
    apply_trial_files,
    inspect_patch,
    verify_primary_inventory,
)
from experiment_common import digest, file_hash  # noqa: E402


class CloudValidationPatchApplyTests(unittest.TestCase):
    @staticmethod
    def _json_bytes(value):
        return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")

    def _write_patch(self, root: Path, extra_members=None):
        members = {
            "primary_inventory.json": self._json_bytes({
                "schema_version": 1,
                "policy": "all_regular_files_exact",
                "entries": [],
                "inventory_sha256": digest([]),
            }),
            "cloud_environment.json": self._json_bytes({
                "schema_version": 1,
                "execution_role": "posthoc_independent_validation_only",
                "system": "Linux",
                "jobs": 1,
                "planner_trials_launched": 0,
                "seed_changes": 0,
            }),
            "validation_summary.json": self._json_bytes({
                "schema_version": 1,
                "status": "COMPLETE",
                "planned_pending": 135,
                "validated_pending": 135,
                "valid_pending": 135,
                "geometric_or_snapshot_invalid": 0,
                "invalid_trial_ids": [],
                "controls_passed": 5,
                "jobs": 1,
                "planner_trials_launched": 0,
                "seed_changes": 0,
                "classification": CLASSIFICATION,
            }),
        }
        members.update(extra_members or {})
        entries = [
            {"path": name, "size_bytes": len(payload), "sha256": digest(payload)}
            for name, payload in sorted(members.items())
        ]
        manifest = {
            "schema_version": 1,
            "artifact_type": PATCH_TYPE,
            "entries": entries,
            "trial_ids": [f"trial_{index:03d}" for index in range(135)],
            "validation_order": [f"trial_{index:03d}" for index in range(135)],
            "capsule_sha256": "0" * 64,
            "capsule_manifest_sha256": "1" * 64,
            "source_at_run_sha256": EXPECTED_SOURCE_AT_RUN_SHA256,
            "jobs": 1,
            "planner_trials_launched": 0,
            "seed_changes": 0,
            "source_primary": {
                "manifest_sha256": "2" * 64,
                "campaign_state_sha256": "3" * 64,
                "primary_inventory_sha256": "4" * 64,
                "counts": {
                    "planned": 250,
                    "executed": 250,
                    "not_run": 0,
                    "offline_validated": 115,
                    "validation_pending": 135,
                    "validated_success": 115,
                    "geometric_invalid": 0,
                },
            },
            "control_results": [{
                "trial_id": trial_id,
                "status": "PASS",
                "snapshots_compared": 1,
                "validation_wall_s_ignored": True,
                "numeric_absolute_tolerance": 1e-8,
            } for trial_id in EXPECTED_CONTROL_TRIAL_IDS],
            "counts": {"trial_ids": 135, "valid": 135, "invalid": 0, "controls": 5},
            "status": "COMPLETE",
        }
        patch = root / "patch.zip"
        with zipfile.ZipFile(patch, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("patch_manifest.json", self._json_bytes(manifest))
            for name, payload in members.items():
                archive.writestr(name, payload)
        Path(str(patch) + ".sha256").write_text(file_hash(patch) + "  patch.zip\n", encoding="ascii")
        return patch

    def test_inspect_patch_accepts_hashed_minimal_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            patch = self._write_patch(Path(directory))
            result = inspect_patch(patch)
            self.assertEqual(result["manifest"]["artifact_type"], PATCH_TYPE)
            self.assertEqual(len(result["manifest"]["trial_ids"]), 135)
            self.assertEqual(result["archive_sha256"], file_hash(patch))

    def test_inspect_patch_rejects_path_traversal_before_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            patch = self._write_patch(Path(directory), {"../escape.json": b"{}\n"})
            with self.assertRaisesRegex(ValueError, "unsafe ZIP member"):
                inspect_patch(patch)
            self.assertFalse((Path(directory).parent / "escape.json").exists())

    def test_primary_inventory_is_exact_and_detects_extra_file(self):
        with tempfile.TemporaryDirectory() as directory:
            primary = Path(directory)
            payload = primary / "manifest.json"
            payload.write_text("{}\n", encoding="utf-8")
            entries = [{
                "path": "manifest.json",
                "size_bytes": payload.stat().st_size,
                "sha256": file_hash(payload),
            }]
            inventory = {
                "schema_version": 1,
                "policy": "all_regular_files_exact",
                "entries": entries,
                "inventory_sha256": digest(entries),
            }
            self.assertEqual(verify_primary_inventory(primary, inventory), digest(entries))
            (primary / "unexpected.txt").write_text("x", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "primary differs"):
                verify_primary_inventory(primary, inventory)

    def test_trial_patch_may_only_change_validation_fields(self):
        primary = {
            "trial_id": "trial_001",
            "planner_id": "bitstar_a3",
            "independent_valid": None,
            "success": False,
            "length_xy_m": 12.0,
        }
        patched = dict(
            primary,
            independent_valid=True,
            success=True,
            length_xy_m=12.5,
            validation_wall_s=1.0,
            snapshots_valid=True,
        )
        _validate_patched_trial(primary, patched, "trial_001")
        corrupted = dict(patched, planner_id="abitstar")
        with self.assertRaisesRegex(ValueError, "non-validation"):
            _validate_patched_trial(primary, corrupted, "trial_001")

    def test_trial_member_allowlist_is_narrow(self):
        trial_ids = {"trial_001"}
        self.assertTrue(_allowed_trial_member("trials/trial_001/validation.json", trial_ids))
        self.assertTrue(_allowed_trial_member(
            "trials/trial_001/snapshots/snapshot_12/validation.json", trial_ids))
        self.assertFalse(_allowed_trial_member("trials/trial_001/request.json", trial_ids))
        self.assertFalse(_allowed_trial_member("trials/trial_002/validation.json", trial_ids))
        self.assertFalse(_allowed_trial_member(
            "trials/trial_001/snapshots/not_a_snapshot/validation.json", trial_ids))

    def test_pending_trial_order_is_exact_not_set_only(self):
        _require_pending_order(["trial_001", "trial_002"], ["trial_001", "trial_002"])
        with self.assertRaisesRegex(ValueError, "IDs/order"):
            _require_pending_order(["trial_002", "trial_001"], ["trial_001", "trial_002"])

    def test_local_integration_toolchain_matches_frozen_campaign_source(self):
        verified = _verify_frozen_local_toolchain(ROOT / "results" / "a3_vs_all_v1_20260930")
        self.assertIn("tools/analyze_results.py", verified)
        self.assertIn("tools/run_advanced_campaign.py", verified)

    def test_apply_replaces_only_uncommitted_snapshot_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primary = root / "primary"
            derivative = root / "derivative"
            extracted = root / "patch"
            original = {
                "trial_id": "trial_001",
                "planner_id": "bitstar_a3",
                "execution_status": "COMPLETED",
                "planner_status": "EXACT_SOLUTION",
                "time_budget_s": 20.0,
                "independent_valid": None,
                "success": False,
            }
            patched = dict(
                original,
                independent_valid=True,
                success=True,
                independently_recomputed_cost_m=12.5,
                length_xy_m=12.0,
                cost_abs_error_m=0.0,
                minimum_footprint_clearance_m=0.5,
                maximum_slope_deg=5.0,
                validation_wall_s=2.0,
                snapshots_valid=True,
            )
            raw_event = {"event_type": "improvement", "snapshot_id": "snapshot_0", "elapsed_s": 1.0}
            for base in (primary, derivative):
                trial = base / "trial_001"
                (trial / "snapshots" / "snapshot_0").mkdir(parents=True)
                (trial / "trial.json").write_text(json.dumps(original), encoding="utf-8")
                (trial / "progress.jsonl").write_text(json.dumps(raw_event) + "\n", encoding="utf-8")
                (trial / "snapshots" / "snapshot_0" / "path_xy.csv").write_text("x,y\n0,0\n", encoding="utf-8")
                (trial / "snapshots" / "snapshot_0" / "path_surface.csv").write_text("x,y,z\n0,0,0\n", encoding="utf-8")
            stale = derivative / "trial_001" / "snapshots" / "snapshot_0" / "validation.json"
            stale.write_text('{"stale": true}\n', encoding="utf-8")
            source = extracted / "trials" / "trial_001"
            (source / "snapshots" / "snapshot_0").mkdir(parents=True)
            (source / "trial.json").write_text(json.dumps(patched), encoding="utf-8")
            snapshot_validation = {
                "valid": True,
                "snapshot_id": "snapshot_0",
                "xy_sha256": file_hash(primary / "trial_001" / "snapshots" / "snapshot_0" / "path_xy.csv"),
                "surface_sha256": file_hash(primary / "trial_001" / "snapshots" / "snapshot_0" / "path_surface.csv"),
            }
            final_validation = {
                "valid": True,
                "exact_goal_reached": True,
                "cost_terrain_m": 12.5,
                "length_xy_m": 12.0,
                "cost_abs_error_m": 0.0,
                "minimum_footprint_clearance_m": 0.5,
                "maximum_slope_deg": 5.0,
                "snapshots": [snapshot_validation],
            }
            events = [dict(raw_event, independent_valid=True)]
            (source / "validation.json").write_text(json.dumps(final_validation), encoding="utf-8")
            (source / "validated_events.json").write_text(json.dumps(events), encoding="utf-8")
            replacement = source / "snapshots" / "snapshot_0" / "validation.json"
            replacement.write_text(json.dumps(snapshot_validation), encoding="utf-8")
            apply_trial_files(primary, derivative, extracted, ["trial_001"])
            self.assertEqual(stale.read_text(encoding="utf-8"), replacement.read_text(encoding="utf-8"))
            self.assertTrue((derivative / "trial_001" / "validation.json").is_file())

            forged_events = [dict(events[0], elapsed_s=2.0)]
            (source / "validated_events.json").write_text(json.dumps(forged_events), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "immutable progress data"):
                apply_trial_files(primary, derivative, extracted, ["trial_001"])
            (source / "validated_events.json").write_text(json.dumps(events), encoding="utf-8")

            # A final validation file is a committed-output collision and must
            # never be silently replaced on a second application.
            with self.assertRaisesRegex(ValueError, "committed patch output"):
                apply_trial_files(primary, derivative, extracted, ["trial_001"])


if __name__ == "__main__":
    unittest.main()
