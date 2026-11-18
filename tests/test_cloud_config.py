"""Regression gates for the separately versioned Cloud design."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from experiment_common import ROOT, load_config
import experiment_common as common


class CloudConfigTests(unittest.TestCase):
    def test_cloud_code_hash_is_independent_of_source_path_enumeration_order(self):
        # Real checked-in bytes; no synthetic experiment artifacts are created.
        paths = list(common.source_files())
        with patch.object(common, "source_files", return_value=paths):
            original = common.cloud_code_hash()
        with patch.object(common, "source_files", return_value=list(reversed(paths))):
            reversed_hash = common.cloud_code_hash()
        self.assertEqual(original, reversed_hash)
    def test_locked_cardinality_and_prior_protocol(self):
        old = load_config(ROOT / "configs/a3_vs_all_v1.yaml", "combined")
        new = load_config(ROOT / "configs/a3_vs_all_v2.yaml", "combined")
        p = new["profiles"]["combined"]
        self.assertEqual(5 * len(p["map_seeds"]) * len(p["planner_seeds"]) * len(p["planners"]), 1500)
        self.assertEqual(old["profiles"]["combined"]["planning_budget_s"], 20)
        self.assertIsNone(new["combined_benchmark"]["hard_finalize_s"])
        self.assertEqual(new["analysis"]["bootstrap_unit_order"], ["family", "map", "seed"])

    def test_reject_drift_and_unknown_fields(self):
        base = load_config(ROOT / "configs/a3_vs_all_v2.yaml", "combined")
        cases = [("model", "state", "xyz"), ("algorithm", "nearest_neighbor_metric", "terrain"),
                 ("combined_benchmark", "hard_finalize_s", 10800),
                 ("combined_benchmark", "max_cloud_tasks", 5), ("execution", "jobs", 4),
                 ("analysis", "bootstrap_unit_order", ["map", "query", "seed"]),
                 ("logging", "invented_flag", True)]
        with tempfile.TemporaryDirectory(prefix="cloud_config_test_") as folder:
            for section, field, value in cases:
                with self.subTest(section=section, field=field):
                    cfg = copy.deepcopy(base); cfg[section][field] = value
                    path = Path(folder) / "fixture.yaml"
                    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
                    with self.assertRaises(ValueError): load_config(path, "combined")

    def test_reject_wrong_seed_budget_and_schema(self):
        base = load_config(ROOT / "configs/a3_vs_all_v2.yaml", "combined")
        with tempfile.TemporaryDirectory(prefix="cloud_config_test_") as folder:
            for field, value in [("map_seeds", list(range(5001, 5021))),
                                 ("planner_seeds", [94001]), ("planning_budget_s", 20)]:
                cfg = copy.deepcopy(base); cfg["profiles"]["combined"][field] = value
                path = Path(folder) / "fixture.yaml"
                path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
                with self.assertRaises(ValueError): load_config(path, "combined")


if __name__ == "__main__": unittest.main()

