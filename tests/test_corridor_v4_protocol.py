"""Regression tests for the v4 paired schedule, policies, and selection."""
import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from analyze_corridor_v4 import _selection
from corridor_v4_protocol import (ALPHAS, COMMON, LEGACY, NATIVE, PRIMARY, RAW, WIDE,
                                  default_config, inventory, load_config, request_for)


class V4ProtocolTests(unittest.TestCase):
    def test_registered_blocks_are_fully_paired_and_independently_rotated(self):
        config = load_config()
        self.assertEqual(config["execution"]["max_workers"], 4)
        self.assertEqual(config["execution"]["validation_workers"], 4)
        smoke = inventory(config)
        self.assertEqual(smoke["planned_trials"], 42)
        self.assertEqual(inventory(config, "pilot")["planned_trials"], 300)
        full = inventory(config, "full", 1.3)
        self.assertEqual(full["planned_trials"], 2000)
        self.assertFalse(set(config["profiles"]["pilot"]["map_seeds"]) &
                         set(config["profiles"]["full"]["map_seeds"]))
        entries = smoke["entries"]
        self.assertEqual(len({entry["trial_id"] for entry in entries}), 42)
        for i in range(7):
            block = entries[6*i:6*(i+1)]
            self.assertEqual({entry["block_index"] for entry in block}, {i})
            self.assertEqual({entry["method"] for entry in block}, {LEGACY, RAW, WIDE, PRIMARY, COMMON, NATIVE})
        self.assertNotEqual(entries[0]["method"], entries[6]["method"])
        self.assertEqual(inventory(default_config(1, 1))["entries"], entries)

    def test_ablation_requests_change_only_the_registered_policy(self):
        config = default_config()
        entries = {entry["method"]: entry for entry in inventory(config)["entries"][:6]}
        info = dict(start=[1., 1.], goal=[5., 5.], map_path=str(ROOT / "map.t25d"))
        field = dict(field_sha256="field", cloud_sha256="cloud", builder_source_sha256="builder")
        requests = {name: request_for(entries[name], info, config, "smoke", ROOT / "field.txt", field)
                    for name in (LEGACY, RAW, WIDE, PRIMARY, COMMON)}
        safe = {name: req["safe_length"] for name, req in requests.items()}
        self.assertEqual(safe[LEGACY]["policy_id"], LEGACY)
        self.assertEqual(safe[RAW]["corridor_policy"]["selector"], "raw_pri")
        self.assertEqual(safe[WIDE]["corridor_policy"]["dilation_cells"], 1)
        self.assertEqual(safe[PRIMARY]["corridor_policy"]["dilation_cells"], 0)
        self.assertEqual(safe[PRIMARY]["corridor_policy"], safe[COMMON]["corridor_policy"])
        self.assertEqual(requests[COMMON]["planner_id"], "sem_rrtstar_common_v1")
        self.assertTrue(all(req["planning_budget_s"] == 2. for req in requests.values()))
        bad = dict(entries[RAW], policy="preview_d0")
        with self.assertRaises(ValueError):
            request_for(bad, info, config, "smoke")

    def test_optimized_beam_config_keeps_the_paired_smoke_and_cap(self):
        config = load_config(ROOT / "configs/a3t_corridor_v4_label_churn.json")
        slots = inventory(config)["entries"]
        self.assertEqual(slots, inventory(default_config())["entries"])
        info = dict(start=[1., 1.], goal=[5., 5.], map_path=str(ROOT / "map.t25d"))
        field = dict(field_sha256="field", cloud_sha256="cloud", builder_source_sha256="builder")
        for entry in slots[:6]:
            if entry["method"] == NATIVE:
                continue
            request = request_for(entry, info, config, "smoke", ROOT / "field.txt", field)
            policy = request["safe_length"]["corridor_policy"]
            self.assertEqual(policy["max_labels"], 2_000_000)
            self.assertEqual(policy.get("beam_scheduler"),
                             None if entry["method"] == LEGACY else "length_first_stable")

    def test_selection_requires_all_three_gates_and_complete_data(self):
        def contrast(alpha, psr, length, pri):
            return dict(scope="all", baseline=NATIVE, method=f"{PRIMARY}_a{alpha}", alpha=alpha,
                planned_pairs=30, safe_pairs=30, metrics={
                    name: dict(difference=dict(mean=value), left_mean=20+value)
                    for name, value in zip(("psr", "final_cost_terrain_m", "final_pri"), (psr, length, pri))})

        report = dict(profile="pilot", status="DONE", paired_comparisons=[
            contrast(ALPHAS[0], 0., -3., .1),
            contrast(ALPHAS[1], -.1, -2., -.5),
            contrast(ALPHAS[2], 0., -1., -.1)])
        self.assertEqual(_selection(report)["selected_alpha"], 1.4)
        invalid = copy.deepcopy(report)
        invalid["paired_comparisons"][2]["metrics"]["final_pri"]["difference"]["mean"] = 0.
        self.assertEqual(_selection(invalid)["status"], "NO_CANDIDATE")
        self.assertEqual(_selection(dict(report, status="INCOMPLETE"))["status"], "BLOCKED")


if __name__ == "__main__":
    unittest.main()
