"""Portable Cloud validation capsule regressions; no planner is executed."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import build_cloud_validation_capsule as capsule
from experiment_common import digest, file_hash


PLANNERS = (
    "bitstar_a3",
    "abitstar",
    "aitstar",
    "lazyprmstar_hauser2015",
    "sem_rrtstar_common_v1",
)
FAMILIES = (
    "traversable_hill",
    "ridge_pass",
    "rolling_terrain",
    "narrow_passage",
    "mixed_obstacles",
)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def suffix(planner_id: str) -> str:
    return planner_id


def trial_id(map_id: str, planner_id: str) -> str:
    return f"{map_id}_q0_{suffix(planner_id)}_s93001"


class LockedFixture:
    def __init__(self, root: Path):
        self.root = root
        self.campaign = root / capsule.CAMPAIGN_BASENAME
        self.maps = root / "maps"
        self.campaign.mkdir()
        self.maps.mkdir()
        self.map_entries = []
        self.trials = []

        control_maps = {value.split("_q0_")[0] for value in capsule.CONTROL_TRIAL_IDS}
        all_map_ids = [f"{family}_{seed}" for family in FAMILIES for seed in range(5001, 5011)]
        pending_maps = set([value for value in all_map_ids if value not in control_maps][:27])

        for map_id in all_map_ids:
            family, seed = map_id.rsplit("_", 1)
            map_path = self.maps / f"{map_id}.t25d"
            map_path.write_bytes(("fixture-map:" + map_id + "\n").encode("ascii"))
            map_sha = file_hash(map_path)
            original = rf"D:\BIT_2.5D\data\a3_vs_all_v1\{map_id}.t25d"
            self.map_entries.append(
                {
                    "map_id": map_id,
                    "family": family,
                    "map_seed": int(seed),
                    "path": original,
                    "sha256": map_sha,
                    "queries": [{"query_id": "q0"}],
                }
            )
            for planner_id in PLANNERS:
                identifier = trial_id(map_id, planner_id)
                entry = {
                    "trial_id": identifier,
                    "map_id": map_id,
                    "map_path": original,
                    "map_hash": map_sha,
                    "planner_id": planner_id,
                    "problem_slot_id": f"{map_id}_q0_s93001",
                }
                self.trials.append(entry)
                self._write_trial(entry, pending=map_id in pending_maps)

        self.source = self.campaign / "source_at_run.zip"
        self.source.write_bytes(b"frozen fixture source\n")
        write_json(
            self.campaign / "source_manifest.json",
            {"archive_sha256": file_hash(self.source), "files": {}, "code_sha256": "fixture"},
        )
        write_json(self.campaign / "resolved_config.json", {"fixture": True})
        write_json(
            self.campaign / "map_manifest.json",
            {"schema_version": 5, "maps": self.map_entries},
        )
        manifest = {
            "schema_version": 3,
            "experiment_id": capsule.CAMPAIGN_BASENAME,
            "profile": "combined",
            "planned_trials": 250,
            "trials": self.trials,
            "identity": {"map_hashes": {entry["map_id"]: entry["sha256"] for entry in self.map_entries}},
            "counts": {"planned": 250, "validated_success": 115},
        }
        write_json(self.campaign / "manifest.json", manifest)
        write_json(
            self.campaign / "campaign_state.json",
            {
                "schema_version": 1,
                "protocol_id": capsule.PROTOCOL_ID,
                "status": "PARTIAL",
                "phase": "INTERRUPTED",
                "counts": dict(capsule.EXPECTED_COUNTS),
            },
        )
        self.baseline_hashes = {
            name: file_hash(self.campaign / name) for name in capsule.BASELINE_HASHES
        }

        # Simulate an interrupted validator that wrote partial outputs but did
        # not atomically commit validation_wall_s to the pending trial record.
        pending_id = next(entry["trial_id"] for entry in self.trials if entry["map_id"] in pending_maps)
        write_json(self.campaign / pending_id / "validation.json", {"partial": True})
        write_json(
            self.campaign / pending_id / "snapshots" / "snapshot_0" / "validation.json",
            {"partial": True},
        )
        self.partial_trial_id = pending_id

    def _write_trial(self, entry, pending: bool) -> None:
        directory = self.campaign / entry["trial_id"]
        directory.mkdir()
        request = {
            "map_path": entry["map_path"],
            "planner_id": entry["planner_id"],
            "planner_seed": 93001,
            "start": [0.0, 0.0],
            "goal": [1.0, 0.0],
        }
        trial = {
            **entry,
            "execution_status": "COMPLETED",
            "planner_status": "EXACT_SOLUTION",
            "path_available": True,
            "success": not pending,
            "independent_valid": None if pending else True,
            "snapshots_valid": None if pending else True,
        }
        if not pending:
            trial["validation_wall_s"] = 0.25
        write_json(directory / "request.json", request)
        write_json(directory / "trial.json", trial)
        (directory / "path_xy.csv").write_text("x,y\n0,0\n1,0\n", encoding="utf-8")
        (directory / "path_surface.csv").write_text("x,y,z\n0,0,0\n1,0,0\n", encoding="utf-8")
        event = {
            "event_type": "improvement",
            "snapshot_id": "snapshot_0",
            "elapsed_s": 0.1,
            "best_cost_terrain_m": 1.0,
        }
        (directory / "progress.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")
        snapshot = directory / "snapshots" / "snapshot_0"
        snapshot.mkdir(parents=True)
        (snapshot / "path_xy.csv").write_text("x,y\n0,0\n1,0\n", encoding="utf-8")
        (snapshot / "path_surface.csv").write_text(
            "x,y,z\n0,0,0\n1,0,0\n", encoding="utf-8"
        )
        if not pending:
            expected = {"valid": True, "errors": [], "cost_terrain_m": 1.0}
            write_json(directory / "validation.json", expected)
            write_json(directory / "validated_events.json", [{**event, "independent_valid": True}])
            write_json(snapshot / "validation.json", {**expected, "snapshot_id": "snapshot_0"})


class CloudValidationCapsuleTests(unittest.TestCase):
    def test_build_is_deterministic_complete_and_excludes_partial_outputs(self):
        with tempfile.TemporaryDirectory(prefix="capsule_fixture_") as directory:
            fixture = LockedFixture(Path(directory))
            first = Path(directory) / "first.zip"
            second = Path(directory) / "second.zip"
            with patch.object(capsule, "BASELINE_HASHES", fixture.baseline_hashes):
                first_result = capsule.build_capsule(fixture.campaign, fixture.maps, first)
                second_result = capsule.build_capsule(fixture.campaign, fixture.maps, second)

            self.assertEqual(first_result["capsule_sha256"], second_result["capsule_sha256"])
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(first_result["pending_trials"], 135)
            self.assertEqual(first_result["control_trials"], 5)
            self.assertEqual(first_result["maps"], 32)
            self.assertEqual(
                first.with_name(first.name + ".sha256").read_text(encoding="ascii"),
                f"{file_hash(first)}  {first.name}\n",
            )

            with zipfile.ZipFile(first) as archive:
                names = archive.namelist()
                manifest = json.loads(archive.read("capsule_manifest.json"))
                inventory = json.loads(archive.read("primary_inventory.json"))
            self.assertEqual(names[0], "capsule_manifest.json")
            self.assertEqual(names[1:], [entry["path"] for entry in manifest["entries"]])
            self.assertEqual(manifest["artifact_type"], capsule.ARTIFACT_TYPE)
            self.assertEqual(manifest["validation_order"], manifest["pending_trial_ids"])
            self.assertEqual(len(manifest["pending_trial_ids"]), 135)
            self.assertEqual(len(manifest["maps"]), 32)
            self.assertTrue(all(set(record) == capsule.MAP_RECORD_KEYS for record in manifest["maps"]))
            self.assertEqual(inventory["policy"], capsule.PRIMARY_INVENTORY_POLICY)
            self.assertEqual(inventory["inventory_sha256"], digest(inventory["entries"]))
            primary_paths = {entry["path"] for entry in inventory["entries"]}
            self.assertIn(f"{fixture.partial_trial_id}/validation.json", primary_paths)
            self.assertNotIn(f"pending/{fixture.partial_trial_id}/validation.json", names)
            self.assertNotIn(
                f"pending/{fixture.partial_trial_id}/snapshots/snapshot_0/validation.json", names
            )

    def test_existing_output_or_sidecar_is_never_overwritten(self):
        with tempfile.TemporaryDirectory(prefix="capsule_no_overwrite_") as directory:
            root = Path(directory)
            campaign = root / capsule.CAMPAIGN_BASENAME
            maps = root / "maps"
            campaign.mkdir()
            maps.mkdir()
            output = root / "capsule.zip"
            output.write_bytes(b"keep-me")
            with self.assertRaises(FileExistsError):
                capsule.build_capsule(campaign, maps, output)
            self.assertEqual(output.read_bytes(), b"keep-me")
            output.unlink()
            sidecar = output.with_name(output.name + ".sha256")
            sidecar.write_bytes(b"keep-sidecar")
            with self.assertRaises(FileExistsError):
                capsule.build_capsule(campaign, maps, output)
            self.assertEqual(sidecar.read_bytes(), b"keep-sidecar")
            self.assertFalse(output.exists())

    def test_campaign_count_drift_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="capsule_count_drift_") as directory:
            fixture = LockedFixture(Path(directory))
            state_path = fixture.campaign / "campaign_state.json"
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state["counts"]["validation_pending"] = 134
            write_json(state_path, state)
            with patch.object(capsule, "BASELINE_HASHES", fixture.baseline_hashes), \
                 self.assertRaisesRegex(capsule.CapsuleError, "campaign_state counts"):
                capsule.build_capsule(
                    fixture.campaign, fixture.maps, Path(directory) / "invalid.zip"
                )

    def test_archive_paths_are_strict_relative_posix_paths(self):
        invalid = (
            "", "/absolute", "../escape", "a/../b", r"a\b", "C:/escape", "./a",
            "pending/a:stream/path_xy.csv",
        )
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(capsule.CapsuleError):
                capsule._safe_archive_path(value)
        self.assertEqual(capsule._safe_archive_path("pending/trial/path_xy.csv"),
                         "pending/trial/path_xy.csv")


if __name__ == "__main__":
    unittest.main()
