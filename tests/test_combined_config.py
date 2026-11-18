import copy
from pathlib import Path
import sys
import tempfile
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from experiment_common import load_config


CONFIG_PATH = ROOT / "configs" / "a3_vs_all_v1.yaml"
PLANNERS = [
    "bitstar_a3",
    "abitstar",
    "aitstar",
    "lazyprmstar_hauser2015",
    "sem_rrtstar_common_v1",
]


class CombinedConfigTests(unittest.TestCase):
    def load_mutation(self, config):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "combined.yaml"
            path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            return load_config(path, "combined")

    def test_locked_five_planner_campaign(self):
        config = load_config(CONFIG_PATH, "combined")
        profile = config["profiles"]["combined"]
        protocol = config["combined_benchmark"]

        self.assertEqual(config["schema_version"], 4)
        self.assertEqual(config["execution"]["default_profile"], "combined")
        self.assertEqual(config["execution"]["jobs"], 1)
        self.assertEqual(config["execution"]["shuffle_trial_order_seed"], 73136)
        self.assertEqual(config["analysis"]["bootstrap_unit_order"],
                         ["map", "query", "seed"])
        self.assertEqual(profile["families"], [
            "traversable_hill", "ridge_pass", "rolling_terrain",
            "narrow_passage", "mixed_obstacles",
        ])
        self.assertEqual(profile["map_seeds"], list(range(5001, 5011)))
        self.assertEqual(profile["planner_seeds"], [93001])
        self.assertEqual(profile["planners"], PLANNERS)
        self.assertEqual(profile["planning_budget_s"], 20.0)
        self.assertEqual(
            len(profile["families"])
            * profile["maps_per_family"]
            * profile["queries_per_map"]
            * len(profile["planner_seeds"])
            * len(profile["planners"]),
            250,
        )

        self.assertEqual(protocol["protocol_id"], "a3_vs_all_v1")
        self.assertEqual(protocol["planner_ids"], PLANNERS)
        self.assertEqual(protocol["primary_planner_id"], "bitstar_a3")
        self.assertEqual(protocol["comparator_planner_ids"], PLANNERS[1:])
        self.assertEqual(protocol["required_query_ids"], ["q0"])
        self.assertEqual(protocol["order_policy"], "latin_square_five_planners_v1")
        self.assertEqual(protocol["planner_block_size"], 5)
        self.assertEqual(protocol["expected_trials"], 250)
        self.assertEqual(protocol["required_allow_flag"], "--allow-combined-benchmark")
        self.assertEqual(
            (protocol["collection_soft_stop_s"], protocol["validation_checkpoint_s"],
             protocol["hard_finalize_s"]),
            (6300.0, 9600.0, 10800.0),
        )
        self.assertEqual(
            (protocol["bootstrap_replicates"], protocol["bootstrap_seed"]),
            (2000, 73137),
        )

    def test_sem_comparator_is_explicitly_adapted_to_common_cost(self):
        protocol = load_config(CONFIG_PATH, "combined")["combined_benchmark"]
        self.assertEqual((protocol["sem_range_m"], protocol["sem_neighbor_radius_m"]),
                         (1.0, 3.0))
        self.assertEqual(protocol["search_direction"], "backward_goal_to_start")
        self.assertEqual(protocol["lower_bound_policy"], "d_xy")
        self.assertEqual(protocol["informed_domain_policy"], "E_xy")
        self.assertFalse(protocol["terrain_risk_cost"])
        self.assertFalse(protocol["turning_angle_cost"])
        self.assertFalse(protocol["rolling_optimization"])
        self.assertFalse(protocol["official_sem_rrtstar_implementation"])
        self.assertEqual(protocol["sem_comparator_label"],
                         "SEM-RRT*-adapted common-objective")
        self.assertEqual(load_config(CONFIG_PATH, "combined")["model"]["objective"],
                         "surface_length")

    def test_report_metrics_cover_quality_timing_resources_and_effort(self):
        metrics = load_config(CONFIG_PATH, "combined")["combined_benchmark"]["report_metrics"]
        self.assertEqual(metrics, [
            "validated_success", "first_solution_time_s", "cost_terrain_m_at_budget",
            "length_xy_m_final", "actual_solve_wall_s", "worker_wall_s",
            "validation_wall_s", "peak_rss_mb", "motion_validity_calls",
            "state_validity_calls", "full_surface_cost_calls", "triangle_visits",
            "height_queries", "sample_proposals", "coverage",
        ])

    def test_protocol_mutations_are_rejected(self):
        source = load_config(CONFIG_PATH, "combined")
        mutations = {
            "planner order": ("profiles", "combined", "planners", list(reversed(PLANNERS))),
            "budget": ("profiles", "combined", "planning_budget_s", 19.0),
            "map seed": ("profiles", "combined", "map_seeds", list(range(5002, 5012))),
            "planner seed": ("profiles", "combined", "planner_seeds", [93002]),
            "expected trials": ("combined_benchmark", "expected_trials", 249),
            "allow flag": ("combined_benchmark", "required_allow_flag", "--allow-full"),
            "order policy": ("combined_benchmark", "order_policy", "unbalanced"),
            "SEM official label": ("combined_benchmark", "official_sem_rrtstar_implementation", True),
            "jobs": ("execution", "jobs", 2),
        }
        for label, mutation in mutations.items():
            with self.subTest(label=label):
                config = copy.deepcopy(source)
                *parents, key, value = mutation
                target = config
                for parent in parents:
                    target = target[parent]
                target[key] = value
                with self.assertRaises(ValueError):
                    self.load_mutation(config)

    def test_schema_and_profile_are_not_implicitly_reinterpreted(self):
        with self.assertRaisesRegex(ValueError, "profile combined"):
            load_config(CONFIG_PATH, "advanced")
        source = load_config(CONFIG_PATH, "combined")
        source["combined_benchmark"]["unregistered_field"] = True
        with self.assertRaisesRegex(ValueError, "missing/unknown fields"):
            self.load_mutation(source)


if __name__ == "__main__":
    unittest.main()
