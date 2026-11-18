"""Labeled parser/statistical fixtures; never benchmark observations."""
import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tools"))
from native_paper_v3_analysis import *
from native_paper_v3_analysis import _validate_group
from terrain_io import write_map


def event(time=1., length=10., turns=1, geometry=True, paper=True, sid="fixture_s0", pri=5.):
    return dict(snapshot_id=sid, elapsed_s=time, independent_valid=geometry, paper_valid=paper,
                metrics=dict(cost_terrain_m=length, length_xy_m=length, sharp_turn_count=turns,
                             maximum_swept_footprint_slope_deg=0., pri=pri))


def row(backend="a3", seed=1, status="COMPLETED", geometry=True, paper=True, length=10., turns=1):
    return dict(trial_id=f"synthetic_{backend}_{seed}", backend=backend, family="fixture", map_id="fixture", planner_seed=seed,
        execution_status=status, success_geometry=geometry, success_paper=paper, success=paper,
        attained_geometry=geometry, attained_paper=paper, budget_s=20., timing_stratum="fixture_machine",
        solve_time_s=20., first_geometry_s=1. if geometry else None, first_paper_s=1. if paper else None,
        first_geometry_latency_s=2. if geometry else None, first_paper_latency_s=2. if paper else None,
        events=[event(length=length, turns=turns, geometry=geometry, paper=paper)] if geometry else [])


class V3AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.fixtures = ROOT/"tests/fixtures/native_paper_v3"
        self.fixtures.mkdir(parents=True, exist_ok=True)

    def test_final_selection_never_falls_back_and_excludes_late(self):
        r = row()
        r["events"] += [event(time=19., length=20., paper=False, sid="s1"), event(time=20.001, length=1., sid="late")]
        self.assertEqual(final_event(r["events"], 20.)["snapshot_id"], "s1")
        self.assertEqual(final_metrics(r)["cost_terrain_m"], 20.)
        self.assertEqual(first_paper_metrics(r)["cost_terrain_m"], 10.)
        r["events"][1]["independent_valid"] = False
        self.assertEqual(final_event(r["events"], 20.)["snapshot_id"], "s1")

    def test_psr_attempts_sti_ratio_and_same_success_set(self):
        rows = [row(length=10., turns=1), row(seed=2, length=30., turns=2),
                row(seed=3, status="CRASH", geometry=False, paper=False),
                row(seed=4, status="NOT_RUN", geometry=False, paper=False), row(seed=5, paper=False)]
        summary = group_summary(rows)
        self.assertEqual((summary["planned"], summary["attempted"], summary["success"]), (5, 4, 2))
        self.assertEqual(summary["psr_paper"], .5)
        self.assertEqual(summary["psr_geometry"], .75)
        self.assertAlmostEqual(summary["sti"], 3/40)
        self.assertEqual(summary["pri"]["n"], summary["cost_terrain_m"]["n"])
        self.assertEqual(summary["solve_time_s"]["n"], 2)

    def test_first_safe_attainment_is_distinct_from_final_psr(self):
        r = row()
        r["events"].append(event(time=19., paper=False, sid="last"))
        r["success_paper"] = r["success"] = False
        self.assertEqual(group_summary([r])["success"], 0)
        self.assertEqual(group_summary([r], "first_paper")["success"], 1)
        attained = attainment_rows([r], [0., .5, 1., 20.])
        self.assertEqual([value["fraction"] for value in attained if value["backend"] == "a3" and value["criterion"] == "paper"], [0., 0., 1., 1.])

    def test_paired_bootstrap_and_stratum_coverage(self):
        rows = [row("a3", length=8.), row("sem", length=10.),
                row("a3", seed=2, status="CRASH", geometry=False, paper=False), row("sem", seed=2)]
        result = paired_bootstrap(rows, "sem", "cost_terrain_m", 50)
        self.assertEqual(result["usable_pairs"], 1)
        self.assertEqual(result["ci95"], [-2., -2.])
        self.assertEqual(paired_bootstrap(rows, "sem", "psr_paper", 50)["usable_pairs"], 2)
        rows[1]["timing_stratum"] = "different_machine"
        result = paired_bootstrap(rows, "sem", "solve_time_s", 50)
        self.assertEqual(result["excluded_stratum_mismatch"], 1)
        self.assertEqual(result["usable_pairs"], 0)

    def test_turns_duplicates_exact_threshold_and_zero_length(self):
        self.assertEqual(sharp_turn_count([[0, 0], [1, 0], [1, 0], [1, 1]]), 1)
        self.assertEqual(sharp_turn_count([[0, 0], [1, 0], [1+math.cos(math.pi/6), .5]]), 0)
        self.assertIsNone(group_summary([row(length=0., turns=0)])["sti"])

    def test_validation_cache_cost_tamper_and_no_clearance(self):
        with tempfile.TemporaryDirectory(prefix="synthetic_v3_validation_", dir=self.fixtures) as directory:
            root = Path(directory)
            mp = root/"fixture.t25d"
            write_map(mp, [[0.]*5 for _ in range(5)])
            axis = np.arange(.025, 4., .05)
            x, y = np.meshgrid(axis, axis)
            cloud = root/"fixture.cloud.npy"
            np.save(cloud, np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size))), allow_pickle=False)
            request = dict(map_path=str(mp), cloud_path=str(cloud), cloud_sha256=sha256(cloud),
                           start=[1., 1.], goal=[3., 1.], planning_budget_s=20.)
            entry = dict(trial_id="synthetic_a3_fixture", backend="a3", track="native", family="fixture", map_id="fixture",
                         query_id="q0", planner_seed=1, execution_status="COMPLETED", map_sha256=sha256(mp), request=request)
            trial = root/entry["trial_id"]
            trial.mkdir()
            dump(trial/"request.json", request)
            snapshot = dict(snapshot_id="s0", path_xy=[[1., 1.], [3., 1.]], elapsed_s=.13, native_objective_value=2.)
            worker = dict(status="EXACT_SOLUTION", snapshots=[snapshot], phase_timings=dict(solve_s=20., setup_s=.1))
            dump(trial/"worker.json", worker)
            (trial/"path_xy.csv").write_text("x,y\n1,1\n3,1\n", encoding="utf-8")
            task = (root, [entry], validation_code_identity())
            first = _validate_group(task)[0]
            self.assertTrue(first["success_paper"], first["errors"])
            self.assertTrue(first["success_geometry"])
            self.assertFalse(first["validation_cache_hit"])
            self.assertEqual(first["first_paper_s"], .13)
            self.assertTrue(_validate_group(task)[0]["validation_cache_hit"])
            serialized = json.dumps(first)
            self.assertNotIn("minimum_footprint_clearance", serialized)
            self.assertNotIn("minimum_obstacle_footprint_clearance", serialized)
            dump(root/"manifest.json", dict(protocol_id=PROTOCOL_ID, entries=[entry], synthetic_fixture=True))
            dump(root/"validation_summary.json", dict(manifest_sha256=sha256(root/"manifest.json"), raw_hashes=raw_hashes(root),
                 validation_code_identity=validation_code_identity(), rows=[first], correctness_failures=0,
                 shared_evaluator_inventory=evaluator_inventory(root, [first])))
            summary = analyze_campaign(root, root/"report")
            self.assertEqual(summary["completion"], "COMPLETE")
            self.assertFalse(summary["plots"])
            self.assertTrue((root/"report/attainment.csv").exists())
            self.assertNotIn("minimum_footprint_clearance", (root/"report/trials.csv").read_text())
            self.assertTrue(any(value["elapsed_s"] == .13 for value in summary["attainment"]))
            field_path = root/"shared_sem_evaluator/fixture/field.npz"
            field_bytes = field_path.read_bytes()
            field_path.write_bytes(field_bytes+b"tampered_evaluator_fixture")
            with self.assertRaisesRegex(ValueError, "hash"):
                _validate_group(task)
            with self.assertRaisesRegex(ValueError, "evaluator"):
                analyze_campaign(root, root/"report")
            field_path.write_bytes(field_bytes)
            worker["snapshots"][0]["native_objective_value"] = 1.
            dump(trial/"worker.json", worker)
            changed = _validate_group(task)[0]
            self.assertFalse(changed["success_paper"])
            self.assertIn("native_objective_mismatch", changed["errors"])
            self.assertFalse(changed["validation_cache_hit"])


if __name__ == "__main__":
    unittest.main()
