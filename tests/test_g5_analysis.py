"""Synthetic in-memory G5 analysis fixtures; no experimental results are written."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
from g5_analysis import (WORK_METRICS, checkpoints, choose_budget, geometry_association,
                         paired_bootstrap, prepare_observations, tail_record)


def event(t, c, n=1):
    return {"elapsed_s": t, "best_cost_terrain_m": c,
            "counters": {k: n for k in WORK_METRICS}}


def row(events, *, success=True, actual=10.):
    return {"events": events, "executed": True, "success": success, "time_budget_s": 10.,
            "metrics": {"actual_solve_wall_s": actual}, "termination_reason": "FIXTURE_STOP",
            "cost_terrain_m": events[-1]["best_cost_terrain_m"] if events and success else None,
            "first_solution_time_s": events[0]["elapsed_s"] if events and success else None}


class G5AnalysisTests(unittest.TestCase):
    def test_tail_boundary_and_future_not_backfilled(self):
        result = tail_record([event(1., 100.), event(4., 90.), event(4.5, 81.), event(5.1, 70.)])
        self.assertAlmostEqual(result["relative_tail_gain"], .1)
        self.assertTrue(result["strict_late_improvement"])
        first_late = tail_record([event(4.5, 81.)])
        self.assertIsNone(first_late["relative_tail_gain"])
        self.assertFalse(first_late["strict_late_improvement"])
        boundary = tail_record([event(1., 100.), event(4., 90.)])
        self.assertEqual(boundary["relative_tail_gain"], 0.)
        self.assertFalse(boundary["strict_late_improvement"])

    def test_budget_rule_is_joint_and_missing_gain_never_zero(self):
        small = {"gain_null": 0, "median": .001, "p90": .002, "strict_late_improvement_fraction": .2}
        self.assertEqual(choose_budget([small]*20), 5.)
        for bad in (dict(small, gain_null=1), dict(small, median=.006),
                    dict(small, p90=.021), dict(small, strict_late_improvement_fraction=.3)):
            self.assertEqual(choose_budget([small]*19+[bad]), 10.)

    def test_anytime_target_and_early_stop_keep_real_event_work(self):
        record = row([event(.3, 12., 10), event(.8, 9., 30)], actual=1.)
        reference = {"status": "FROZEN_VALIDATED_TARGET", "cost_terrain_m": 10.}
        prepare_observations(record, reference, [.1, .5, 1., 10.])
        self.assertIsNone(record["checkpoint_observations"][0]["cost_terrain_m"])
        self.assertEqual(record["checkpoint_observations"][1]["cost_terrain_m"], 12.)
        self.assertEqual(record["checkpoint_observations"][-1]["cost_terrain_m"], 9.)
        self.assertTrue(record["checkpoint_observations"][-1]["retained_incumbent_after_termination"])
        self.assertEqual(record["frozen_target"]["event_time_s"], .8)
        self.assertEqual(record["frozen_target"]["observation_end_s"], 1.)
        self.assertTrue(all(v == 30 for v in record["frozen_target"]["counters_at_attainment"].values()))
        missed = row([event(.3, 12.)])
        prepare_observations(missed, reference, checkpoints(10.))
        self.assertIsNone(missed["frozen_target"]["event_time_s"])
        self.assertTrue(all(v is None for v in missed["frozen_target"]["counters_at_attainment"].values()))

    def pairs(self, values):
        rows = []
        for map_id, seeds in values.items():
            for seed, value in enumerate(seeds):
                for planner, metric in (("bitstar_a0", value), ("bitstar_a3", 0.)):
                    rows.append({"map_id": map_id, "query_id": "q0", "planner_seed": seed,
                                 "planner_id": planner, "executed": True, "analysis_values": {"cost": metric}})
        return rows

    def test_map_weight_is_independent_of_seed_replication(self):
        rows = self.pairs({"m0": [10.], "m1": [-2.]*10})
        result = paired_bootstrap(rows, "cost", replicates=200, seed=21)
        self.assertEqual(result["estimate"], 4.)
        self.assertEqual(result["units"]["maps"], 2)
        self.assertEqual(result["units"]["paired_observations"], 11)
        self.assertEqual(result, paired_bootstrap(rows, "cost", replicates=200, seed=21))
        constant = paired_bootstrap(self.pairs({"m0": [3.]*2, "m1": [3.]*10}), "cost", replicates=100)
        self.assertEqual(constant["ci"], [3., 3.])

    def test_missing_pairs_excluded_with_counts_and_single_map_no_ci(self):
        rows = self.pairs({"m0": [1., 2.]})
        rows[-1]["analysis_values"]["cost"] = None
        result = paired_bootstrap(rows, "cost", replicates=10)
        self.assertEqual(result["units"]["unavailable_complete_pairs"], 1)
        self.assertEqual(result["units"]["paired_observations"], 1)
        self.assertIsNone(result["ci"])

    def test_geometry_direction_and_degenerate_covariate(self):
        rows = self.pairs({"m0": [-1.]*2, "m1": [-2.]*2, "m2": [-3.]*2})
        geometry = {(f"m{i}", "q0"): {"x": float(i+1)} for i in range(3)}
        result = geometry_association(rows, geometry, "cost", "x", replicates=100)
        self.assertAlmostEqual(result["slope"], 1.)
        self.assertAlmostEqual(result["pearson_r"], 1.)
        for g in geometry.values():
            g["x"] = 1.
        result = geometry_association(rows, geometry, "cost", "x", replicates=20)
        self.assertIsNone(result["slope"])
        self.assertIsNone(result["pearson_r"])


if __name__ == "__main__":
    unittest.main()
