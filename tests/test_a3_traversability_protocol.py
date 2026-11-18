"""Schema/transport/fixture tests only; this module never launches a planner."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tools"))
from a3_traversability_protocol import (FAMILIES, PLANNERS, PROTOCOL_ID,
    objective_identity, resolved_traversability, validate_config)
from experiment_common import digest, load_config, resolved_planner_options
from generate_scenarios import generate
from run_experiments import plan, request_for, verify_variant_result


class TraversabilityProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(ROOT/"configs/a3_traversability_smoke.yaml")
        cls.temporary = tempfile.TemporaryDirectory()
        cls.manifest = generate(cls.config, output=Path(cls.temporary.name)/"maps")
        cls.entries = plan(cls.config, "smoke", cls.manifest)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_config_and_manifest_have_exact_bounded_design(self):
        self.assertEqual(tuple(self.config["profiles"]["smoke"]["families"]), FAMILIES)
        self.assertEqual(tuple(self.config["profiles"]["smoke"]["planners"]), PLANNERS)
        self.assertEqual(self.manifest["protocol_id"], PROTOCOL_ID)
        self.assertEqual(len(self.entries), 36)
        self.assertEqual(len({entry["trial_id"] for entry in self.entries}), 36)
        self.assertEqual(self.entries, plan(self.config, "smoke", self.manifest))

    def test_geometric_witnesses_and_direct_path_qualification(self):
        for item in self.manifest["maps"]:
            query = item["queries"][0]
            self.assertEqual(query["query_id"], "q0")
            self.assertTrue(all(w["validation"]["valid"] for w in query["witnesses"]))
            blocked = item["family"] in {"unsafe_short_safe_detour", "two_corridors_slope", "nodata_thin_obstacle"}
            self.assertEqual(query["direct_edge_validation"]["valid"], not blocked)
            self.assertEqual(query["reference"]["reference_type"], "validated_witness")
            if item["family"].startswith("two_corridors"):
                short, long = query["witnesses"]
                self.assertLess(short["validation"]["cost_terrain_m"], long["validation"]["cost_terrain_m"])
                self.assertGreater(short["traversability_validation"]["risk_integral_m"],
                                   long["traversability_validation"]["risk_integral_m"])
                self.assertEqual(query["tradeoff_qualification"]["status"], "VALIDATED_WITNESS_TRADEOFF")

    def test_every_request_has_separate_objective_identity(self):
        for entry in self.entries:
            request = request_for(entry, self.config, "smoke")
            expected = objective_identity(self.config, entry["planner_id"])
            self.assertEqual(request["traversability"], expected["traversability"])
            self.assertEqual(request["objective_type"], expected["objective_type"])
            self.assertEqual(entry["objective_config_sha256"], digest(expected))
            self.assertNotIn("lambda", entry["resolved_optimizations"])
            self.assertNotIn("traversability", entry["resolved_optimizations"])
            self.assertEqual(request["variant_id"], entry["planner_id"])
        self.assertEqual([resolved_traversability(self.config, p)["lambda"] for p in PLANNERS], [0., .25, 1., 4., 1., 1.])

    def result(self, entry):
        options = resolved_planner_options(self.config, entry["planner_id"])
        result = {"planner_id": entry["planner_id"], "variant_id": entry["planner_id"],
                  "backend_planner_id": options["planner_id"], "effective_policy": dict(options, batch_policy="fixed")}
        result.update(objective_identity(self.config, entry["planner_id"]))
        result["requested_traversability"] = result.pop("traversability")
        result["effective_traversability"] = copy.deepcopy(result["requested_traversability"])
        return result

    def test_effective_objective_mismatch_fails_instead_of_relabelling(self):
        entry = self.entries[0]
        verify_variant_result(entry, self.config, self.result(entry))
        for key in ("objective_type", "requested_traversability", "effective_traversability"):
            result = self.result(entry)
            result.pop(key)
            with self.assertRaisesRegex(ValueError, "traversability"):
                verify_variant_result(entry, self.config, result)
        result = self.result(entry)
        result["effective_traversability"]["lambda"] += 1.
        with self.assertRaisesRegex(ValueError, "traversability"):
            verify_variant_result(entry, self.config, result)

    def test_configuration_change_invalidates_planned_objective_hash(self):
        entry = next(e for e in self.entries if e["planner_id"] == "a3_t_lambda1")
        changed = copy.deepcopy(self.config)
        changed["planner_variants"]["a3_t_lambda1"]["lambda"] = 2.
        self.assertNotEqual(digest(changed), digest(self.config))
        with self.assertRaisesRegex(ValueError, "changed after trial planning"):
            request_for(entry, changed, "smoke")
        changed = copy.deepcopy(self.config)
        changed["traversability"]["clearance_comfort_m"] = .8
        with self.assertRaisesRegex(ValueError, "changed after trial planning"):
            request_for(entry, changed, "smoke")

    def test_invalid_features_limits_profiles_and_backends_rejected(self):
        for name, value in (("lambda", -1.), ("lambda", True), ("lambda", float("nan")),
                ("lambda", float("inf")), ("slope_weight", -.1), ("slope_weight", .8),
                ("clearance_comfort_m", 0.), ("feature_version", "unknown"), ("roughness_weight", .2)):
            config = copy.deepcopy(self.config)
            config["traversability"][name] = value
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                validate_config(config)
        for profile in ("pilot", "full", "a3_pilot"):
            with self.assertRaisesRegex(ValueError, "smoke only"):
                load_config(ROOT/"configs/a3_traversability_smoke.yaml", profile)
        for backend in ("rrtstar", "abitstar", "sem_rrtstar_common_v1"):
            config = copy.deepcopy(self.config)
            config["planner_variants"]["a0_t_lambda1"]["planner_id"] = backend
            with self.assertRaises(ValueError):
                validate_config(config)
        config = copy.deepcopy(self.config)
        config["planner_variants"]["a3_t_lambda1"]["lambda"] = True
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_bad_map_manifest_identity_rejected(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["config_hash"] = "mismatch"
        with self.assertRaisesRegex(ValueError, "manifest"):
            plan(self.config, "smoke", manifest)

    def test_legacy_requests_remain_surface_only(self):
        entry = {"map_path": "test-fixture.t25d", "planner_id": "bitstar_a3", "planner_seed": 11,
                 "start": [1., 1.], "goal": [9., 9.]}
        for filename, profile in (("experiments.yaml", "smoke"),
                ("a3_vs_advanced_v1.yaml", "advanced"), ("a3_vs_sem_rrtstar_common_v1.yaml", "sem_comparison"),
                ("a3_vs_all_v1.yaml", "combined"), ("a3_vs_all_v2.yaml", "combined"),
                ("a3_improvements_smoke.yaml", "smoke"), ("a3_improvements_pilot_v1.yaml", "smoke")):
            config = load_config(ROOT/"configs"/filename, profile)
            self.assertEqual(config["model"]["objective"], "surface_length")
            request = request_for(entry, config, profile)
            self.assertNotIn("traversability", request)
            self.assertNotIn("objective_type", request)


if __name__ == "__main__":
    unittest.main()
