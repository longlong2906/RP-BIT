"""Schema/order/generator tests; synthetic metadata remains in memory or fixtures."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from a3_pilot_protocol import (EXPECTED_TRIALS, PLANNERS, PROTOCOL_ID, audit_trial_plan,
    comparison_pairs, expected_config, plan_trials, relative_map_path, validate_config)
from experiment_common import digest, load_config, read_json, resolved_planner_options
from generate_scenarios import generate

CONFIG = ROOT / "configs/a3_improvements_pilot_v1.yaml"


def fixture_manifest(config, profile):
    """Synthetic metadata, never experimental observations or benchmark output."""
    settings = config["profiles"][profile]
    maps = []
    for family in settings["families"]:
        for seed in settings["map_seeds"]:
            map_id = f"{family}_{seed}"
            map_hash = digest({"synthetic_test_fixture": map_id})
            queries = []
            for index in range(settings["queries_per_map"]):
                start, goal = [1., 1.], [9., 9.]
                checked = {"valid": True, "cost_terrain_m": 12.}
                queries.append({"query_id": f"q{index}", "start": start, "goal": goal,
                    "query_hash": digest({"start": start, "goal": goal,
                        "map_hash": map_hash, "model": config["model"]}),
                    "feasibility": "feasible_witness_validated", "reference": {
                        "reference_type": "validated_witness", "validation": checked,
                        "map_sha256": map_hash, "cost_terrain_m": 12.}})
            maps.append({"map_id": map_id, "family": family, "map_seed": seed,
                         "path": map_id + ".t25d", "sha256": map_hash, "queries": queries})
    return {"protocol_id": PROTOCOL_ID, "profile": profile, "config_hash": digest(config),
            "maps": maps, "seed_replacements": [], "seed_replacement_policy": "forbidden"}


class A3PilotProtocolTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(CONFIG)

    def test_design_cardinality_and_original_controls(self):
        self.assertEqual(self.config, expected_config())
        self.assertEqual(len(PLANNERS), 13)
        self.assertEqual(len(comparison_pairs()), 19)
        self.assertEqual(len(set(comparison_pairs())), 19)
        for profile, expected in EXPECTED_TRIALS.items():
            cfg = load_config(CONFIG, profile)
            p = cfg["profiles"][profile]
            self.assertEqual(len(p["families"]) * p["maps_per_family"] * p["queries_per_map"] *
                             len(p["planner_seeds"]) * len(p["planners"]), expected)
        for label in ("bitstar_a3", *PLANNERS[10:]):
            options = resolved_planner_options(self.config, label)
            self.assertEqual(options, {"planner_id": label, "state_cache": False,
                "scalar_cost": False, "collision_index": False, "samples_per_batch": 200,
                "batch_after_first_solution": 0})

    def test_noop_renamed_or_unlocked_design_is_rejected(self):
        mutations = [
            ("schema_version", 6), ("language", "en"),
            ("model.robot_radius_m", .31), ("algorithm.bitstar_samples_per_batch", 100),
            ("execution.jobs", 2), ("execution.jobs", True),
            ("execution.default_profile", "a3_pilot"), ("execution.shuffle_trial_order_seed", 10),
            ("profiles.a3_pilot.planning_budget_s", 9.),
            ("profiles.a3_pilot.map_seeds", [201, 203]),
            ("profiles.a3_pilot.planner_seeds", [11, 22]),
            ("profiles.smoke.planners", list(PLANNERS) + ["abitstar_optimized"]),
            ("planner_variants.a3_cache.state_cache", False),
            ("a3_pilot_benchmark.bootstrap_seed", 73129),
            ("a3_pilot_benchmark.primary_metrics", ["cost_terrain_m_at_budget"]),
            ("a3_pilot_benchmark.comparison_pairs", []),
            ("execution.watchdog_grace_s", float("nan")),
        ]
        for name, value in mutations:
            with self.subTest(name=name, value=value):
                changed = copy.deepcopy(self.config)
                target = changed
                parts = name.split(".")
                for key in parts[:-1]:
                    target = target[key]
                target[parts[-1]] = value
                with self.assertRaises(ValueError):
                    validate_config(changed)
        for profile in ("full", "pilot", "combined"):
            with self.assertRaises(ValueError):
                load_config(CONFIG, profile)
        for field in ("unexpected", "full"):
            changed = copy.deepcopy(self.config)
            changed["profiles"][field] = changed["profiles"]["a3_pilot"]
            with self.assertRaises(ValueError):
                validate_config(changed)

    def test_legacy_contracts_unchanged(self):
        for filename, profile, version in (
                ("experiments.yaml", "smoke", 1), ("experiments.yaml", "pilot", 1),
                ("a3_vs_advanced_v1.yaml", "advanced", 2),
                ("a3_vs_sem_rrtstar_common_v1.yaml", "sem_comparison", 3),
                ("a3_vs_all_v1.yaml", "combined", 4),
                ("a3_vs_all_v2.yaml", "combined", 5),
                ("a3_improvements_smoke.yaml", "smoke", 6)):
            with self.subTest(filename=filename):
                path = ROOT / "configs" / filename
                parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
                loaded = load_config(path, profile)
                self.assertEqual(loaded, parsed)
                self.assertEqual(loaded["schema_version"], version)
                self.assertNotIn("a3_pilot_benchmark", loaded)

    def test_block_order_balanced_before_shuffle_and_input_order_independent(self):
        for profile in EXPECTED_TRIALS:
            with self.subTest(profile=profile):
                manifest = fixture_manifest(self.config, profile)
                trials = plan_trials(self.config, profile, manifest)
                audit = audit_trial_plan(trials, profile)
                self.assertEqual(audit["planned_trials"], EXPECTED_TRIALS[profile])
                self.assertEqual(audit["problem_slots"], 60 if profile == "a3_pilot" else 4)
                for scope in [audit["position_counts"], *audit["family_position_counts"].values()]:
                    for counts in scope.values():
                        self.assertLessEqual(max(counts) - min(counts), 1)
                manifest["maps"].reverse()
                for item in manifest["maps"]:
                    item["queries"].reverse()
                self.assertEqual(plan_trials(self.config, profile, manifest), trials)
                self.assertNotEqual([trials[n]["block_index"] for n in range(0, len(trials), 13)],
                                    list(range(len(trials) // 13)))
                for entry in trials:
                    self.assertEqual(entry["resolved_optimizations_sha256"],
                                     digest(entry["resolved_optimizations"]))

    def test_order_audit_rejects_tampering(self):
        trials = plan_trials(self.config, "a3_pilot", fixture_manifest(self.config, "a3_pilot"))
        for field, value in (("planner_position", 5), ("block_index", -1),
                             ("map_hash", "bad"), ("problem_slot_order", -1),
                             ("query_id", "q99"), ("planner_id", "abitstar_optimized")):
            with self.subTest(field=field):
                changed = copy.deepcopy(trials)
                changed[0][field] = value
                with self.assertRaises(ValueError):
                    audit_trial_plan(changed)
        with self.assertRaises(ValueError):
            audit_trial_plan(trials[:-1])

    def test_missing_replaced_or_unqualified_input_is_rejected(self):
        manifest = fixture_manifest(self.config, "a3_pilot")
        for change in (lambda m: m["maps"].pop(),
                       lambda m: m.update(seed_replacements=[201]),
                       lambda m: m.update(config_hash="stale"),
                       lambda m: m["maps"][0].update(map_seed=203),
                       lambda m: m["maps"][0]["queries"].pop(),
                       lambda m: m["maps"][0]["queries"][0].update(query_hash="forged"),
                       lambda m: m["maps"][0]["queries"][0]["reference"]["validation"].update(valid=False)):
            changed = copy.deepcopy(manifest)
            change(changed)
            with self.assertRaises(ValueError):
                plan_trials(self.config, "a3_pilot", changed)

    def test_paths_are_portable_and_do_not_escape_dataset(self):
        self.assertEqual(relative_map_path("maps/hill.t25d"), "maps/hill.t25d")
        for path in ("../map.t25d", "/tmp/map.t25d", "C:/map.t25d", "C:\\map.t25d",
                     "maps/../map.t25d", "maps//map.t25d", "./map.t25d", "map.json", ""):
            with self.subTest(path=path), self.assertRaises(ValueError):
                relative_map_path(path)

    def test_actual_smoke_generator_portability_and_witness_contract(self):
        with tempfile.TemporaryDirectory(prefix="synthetic-a3-protocol-", dir=ROOT / "tests/fixtures") as root:
            folder = Path(root) / "maps"
            manifest = generate(self.config, "smoke", folder)
            self.assertEqual(manifest, read_json(folder / "manifest.json"))
            self.assertFalse(manifest["held_out"])
            self.assertEqual(manifest["seed_replacements"], [])
            self.assertEqual(len(plan_trials(self.config, "smoke", manifest)), 52)
            for item in manifest["maps"]:
                self.assertEqual(item["path"], item["map_id"] + ".t25d")
                metadata = read_json((folder / item["path"]).with_suffix(".json"))
                self.assertEqual(metadata["queries"], item["queries"])
                self.assertTrue(item["queries"][0]["reference"]["validation"]["valid"])
            with self.assertRaises(ValueError):
                generate(self.config, "smoke", folder)

    def test_generator_qualification_failure_stops_without_reseed(self):
        with tempfile.TemporaryDirectory(prefix="synthetic-a3-protocol-", dir=ROOT / "tests/fixtures") as root:
            folder = Path(root) / "maps"
            with patch("generate_scenarios.pilot_geometry", side_effect=ValueError("qualification fixture")) as geometry:
                with self.assertRaisesRegex(ValueError, "qualification fixture"):
                    generate(self.config, "a3_pilot", folder)
                geometry.assert_called_once_with("traversable_hill", 201, 40., 40.)
            evidence = read_json(folder / "qualification_failure.json")
            self.assertEqual(evidence["status"], "FAILED")
            self.assertEqual(evidence["seed_replacements"], [])
            self.assertFalse((folder / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
