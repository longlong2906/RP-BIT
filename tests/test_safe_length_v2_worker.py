"""Real worker integration on explicitly synthetic correctness fixtures.

These are not benchmark observations. The independent Python supercover and
raw-map geometry oracle do not reuse the C++ calibration/validity routines.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import csv
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from independent_verifier import Oracle
from native_paper_v3_sem_evaluator import SharedSEMField
from terrain_io import write_map
from test_safe_length_oracle import lattice, write_exchange

WORKER = SAFETY_CLI = None


def read_path(path):
    with Path(path).open(newline="") as stream:
        reader = csv.reader(stream)
        next(reader)
        return [[float(row[0]), float(row[1])] for row in reader]


class V2WorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="synthetic_v2_worker_", dir=ROOT/"tests/fixtures")
        self.folder = Path(self.temp.name)
        self.map = self.folder/"synthetic.t25d"
        write_map(self.map, [[0.]*31 for _ in range(31)], spacing=(.2, .2))
        self.field = self.folder/"synthetic_field.txt"
        self.scorer = SharedSEMField(lattice(), (0., 0., 6., 6.))
        self.scorer.risk = self.scorer.risk.copy()
        self.scorer.valid = self.scorer.valid.copy()
        self.scorer.risk[:] = 1.
        self.scorer.risk[20:40, 26:34] = 12.
        self.save_field()
        self.oracle = Oracle(self.map)
        self.request = dict(planner_id="bitstar_a3", variant_id="synthetic_a3t_v2", map_path=str(self.map),
            start=[1., 3.], goal=[5., 3.], planner_seed=11, planning_budget_s=15., work_iterations=1200,
            samples_per_batch=100, batch_after_first_solution=200, rewire_factor=1.1,
            sample_attempts=100, progress_interval_s=.05, collision_index=True, state_cache=True,
            scalar_cost=True, sampler_query_cache=True, exact_flat_certificate=False,
            objective_type="surface_length", safe_length=dict(policy_id="a3t_safe_length_v2",
            field_path=str(self.field), risk_slack=.5, slope_cap_deg=35., atlas_sampling=True))

    def tearDown(self):
        self.temp.cleanup()

    def save_field(self):
        write_exchange(self.field, self.scorer.valid, self.scorer.risk)

    def launch(self, name, request=None, expected=0):
        directory = self.folder/name
        directory.mkdir()
        rp = directory/"request.json"
        rp.write_text(json.dumps(request or self.request), encoding="utf-8")
        result = subprocess.run([WORKER, str(rp), str(directory)], capture_output=True, text=True, timeout=90)
        (directory/"stdout.log").write_text(result.stdout, encoding="utf-8")
        (directory/"stderr.log").write_text(result.stderr, encoding="utf-8")
        self.assertEqual(result.returncode, expected, result.stderr)
        return directory, json.loads((directory/"worker.json").read_text())

    def assert_calibration(self, directory, worker):
        calibration = worker["calibration"]
        self.assertEqual(calibration["status"], "CALIBRATED")
        self.assertEqual(calibration["grid_spacing_m"], .1)
        self.assertEqual(calibration["endpoint_radius_m"], .2)
        self.assertFalse(calibration["witness_used_as_incumbent"])
        path = read_path(directory/calibration["witness_file"])
        self.assertEqual(path, calibration["witness_xy"])
        self.assertEqual(path[0], self.request["start"])
        self.assertEqual(path[-1], self.request["goal"])
        self.assertTrue(self.oracle.validate_path(path, path[0], path[-1])["valid"])
        risk = self.scorer.score_path(path)
        self.assertTrue(risk["valid"])
        self.assertAlmostEqual(risk["risk_max"], calibration["b_graph"], delta=1e-10)
        self.assertEqual(calibration["effective_risk_cap"], min(40., calibration["b_graph"]+.5))
        self.assertEqual(worker["effective_safe_length"]["risk_cap"], calibration["effective_risk_cap"])
        self.assertTrue(worker["effective_safe_length"]["thresholds_immutable_during_solve"])

    def test_real_bitstar_callbacks_cost_hooks_and_fixed_work_parallel(self):
        directory, worker = self.launch("serial")
        self.assert_calibration(directory, worker)
        self.assertEqual(worker["ompl_version"], "1.7.0")
        self.assertTrue(worker["planner_solve_called"])
        self.assertEqual(worker["planner_status"], "EXACT_SOLUTION")
        self.assertEqual(worker["termination_reason"], "WORK_BUDGET")
        policy = worker["effective_policy"]
        self.assertEqual(policy["nearest_neighbor_metric"], "euclidean_xy")
        self.assertEqual(policy["motion_objective"], "surface_length")
        self.assertTrue(policy["k_nearest"])
        self.assertFalse(policy["jit"])
        for counter in ("motion_cost_calls", "motion_heuristic_calls", "cost_to_go_calls"):
            self.assertGreater(worker["counters"][counter], 0, counter)
        self.assertGreater(worker["setup_policy_counters"]["sampler_allocations"] +
                           worker["counters"]["sampler_allocations"], 0)
        self.assertEqual(worker["batch_transition_count"], 1)
        events = [json.loads(line) for line in (directory/"progress.jsonl").read_text().splitlines()]
        improvements = [event for event in events if event["event_type"] == "improvement"]
        self.assertGreater(len(improvements), 0)
        previous = math.inf
        for event in improvements:
            path = read_path(directory/"snapshots"/event["snapshot_id"]/"path_xy.csv")
            verdict = self.oracle.validate_path(path, self.request["start"], self.request["goal"],
                                                claimed_cost=event["best_cost_terrain_m"])
            self.assertTrue(verdict["valid"], verdict)
            self.assertLessEqual(self.scorer.score_path(path)["risk_max"], 1.5)
            self.assertLessEqual(event["best_cost_terrain_m"], previous+1e-8)
            previous = event["best_cost_terrain_m"]
        with ThreadPoolExecutor(max_workers=2) as pool:
            repetitions = list(pool.map(self.launch, ("parallel_a", "parallel_b")))
        for other_dir, other in repetitions:
            self.assertEqual(other["termination_reason"], "WORK_BUDGET")
            for key in ("counters", "iterations", "batches", "best_cost_terrain_m", "batch_transition_count"):
                self.assertEqual(other[key], worker[key], key)
            self.assertEqual(read_path(other_dir/"path_xy.csv"), read_path(directory/"path_xy.csv"))
            self.assertEqual(other["calibration"]["witness_xy"], worker["calibration"]["witness_xy"])

    def test_zero_budget_does_not_turn_calibration_witness_into_solution(self):
        request = copy.deepcopy(self.request)
        request.update(planning_budget_s=0., work_iterations=0)
        directory, worker = self.launch("zero", request)
        self.assert_calibration(directory, worker)
        self.assertFalse(worker["path_available"])
        self.assertIsNone(worker["best_cost_terrain_m"])
        self.assertFalse((directory/"path_xy.csv").exists())
        self.assertEqual(worker["improvement_count"], 0)

    def test_disconnection_is_recorded_without_fallback_or_solve(self):
        self.scorer.valid[:, 29:31] = False
        self.save_field()
        directory, worker = self.launch("disconnected")
        self.assertEqual(worker["calibration"]["status"], "CALIBRATION_NO_CONNECTION")
        self.assertEqual(worker["termination_reason"], "CALIBRATION_NO_CONNECTION")
        self.assertEqual(worker["execution_status"], "COMPLETED")
        self.assertFalse(worker["planner_solve_called"])
        self.assertIsNone(worker["effective_safe_length"]["risk_cap"])
        self.assertFalse(worker["path_available"])
        self.assertFalse((directory/"calibration_witness_xy.csv").exists())

    def test_v2_rejects_fixed_cap_and_unknown_policy_fields(self):
        for index, extra in enumerate(({"risk_cap": 40.}, {"unregistered_risk_weight": .1},
                                       {"slope_cap_deg": 20.}, {"risk_slack": -1.})):
            request = copy.deepcopy(self.request)
            request["safe_length"].update(extra)
            directory = self.folder/("rejected_"+str(index))
            directory.mkdir()
            path = directory/"request.json"
            path.write_text(json.dumps(request), encoding="utf-8")
            process = subprocess.run([WORKER, str(path), str(directory)], capture_output=True,
                                     text=True, timeout=30)
            self.assertEqual(process.returncode, 2, process.stderr)
            self.assertFalse((directory/"path_xy.csv").exists())

    def test_edge_summary_maximum_matches_independent_supercover(self):
        rng = random.Random(421)
        self.scorer.risk = np.random.default_rng(427).uniform(0., 40., (60, 60))
        self.scorer.valid[30, 30] = False
        self.save_field()
        paths = [[[rng.uniform(.5, 5.5), rng.uniform(.5, 5.5)] for _ in range(2)] for _ in range(16)]
        paths += [[[1., 3.], [5., 3.]], [[1., 1.], [5., 5.]], [[3., 3.]]]
        for index, path in enumerate(paths):
            variants = [path, path[::-1]]
            if len(path) == 2:
                variants.append([path[0], [(path[0][i]+path[1][i])/2 for i in range(2)], path[1]])
            for variant in variants:
                csv_path = self.folder/"edge.csv"
                with csv_path.open("w", newline="") as stream:
                    writer = csv.writer(stream);writer.writerow(("x", "y"));writer.writerows(variant)
                result = subprocess.run([SAFETY_CLI, "--map", str(self.map), "--field", str(self.field),
                    "--risk-cap", "40", "--slope-cap", "35", "--path", str(csv_path)],
                    capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                actual, expected = json.loads(result.stdout), self.scorer.score_path(variant)
                self.assertEqual(actual["supported"], expected["invalid_cell_count"] == 0, index)
                self.assertEqual(actual["maximum_risk"], expected["risk_max"], index)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True)
    parser.add_argument("--safety-cli", required=True)
    args, remaining = parser.parse_known_args()
    WORKER, SAFETY_CLI = str(Path(args.worker).resolve()), str(Path(args.safety_cli).resolve())
    os.environ["PATH"] = os.pathsep.join((str(Path(WORKER).parent), str(ROOT/".deps/install/lib"),
                                         "C:/mingw64/bin", os.environ.get("PATH", "")))
    unittest.main(argv=[sys.argv[0], *remaining])
