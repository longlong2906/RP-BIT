"""Synthetic independent-oracle fixtures; never experimental observations."""
import csv
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tools"))
from corridor_v3_validation import (canonical_sha256, path_in_mask, reconstruct_mask,
    record_corridor_receipt, risk_exposure, risk_metrics, shortest_graph_length,
    validate_trial, witness_on_registered_graph)
from native_paper_oracle import NativePaperOracle
from native_paper_v3_sem_evaluator import SharedSEMField
from run_safe_length import dump, read
from safe_length_field import file_hash
from terrain_io import write_map


def write_path(path, points):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["x", "y"])
        writer.writerows(points)


def synthetic_field(risk):
    result = SharedSEMField.__new__(SharedSEMField)
    result.risk = np.asarray(risk, dtype=float)
    result.valid = np.isfinite(result.risk)
    result.resolution = .1
    ny, nx = result.risk.shape
    result.bounds = (0., 0., nx*.1, ny*.1)
    return result


class CorridorValidationTests(unittest.TestCase):
    def fixture(self, directory, failure=False):
        terrain = directory/"synthetic.t25d"
        write_map(terrain, np.zeros((7, 7)).tolist(), spacing=(.5, .5))
        x, y = np.meshgrid((np.arange(30)+.5)*.1, (np.arange(30)+.5)*.1)
        cloud = directory/"synthetic.cloud.npy"
        np.save(cloud, np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size))), allow_pickle=False)
        info = dict(map_path=str(terrain), map_sha256=file_hash(terrain), cloud_path=str(cloud),
                    cloud_sha256=file_hash(cloud), bounds=[0., 0., 3., 3.])
        field = SharedSEMField(np.load(cloud, allow_pickle=False), info["bounds"])
        oracle = NativePaperOracle(terrain)
        corridor_policy = dict(alpha=1.1, label_bins=16, dilation_cells=1, max_labels=2000000,
                               max_memory_bytes=536870912)
        policy = dict(policy_id="a3t_corridor_v3", risk_cap=40., slope_cap_deg=35.,
                      atlas_sampling=False, corridor_policy=corridor_policy)
        request = dict(start=[.7, 1.2], goal=[2.2, 1.2], planning_budget_s=2.,
                       planner_id="sem_rrtstar_common_v1", samples_per_batch=200,
                       batch_after_first_solution=0, safe_length=policy)
        path = [request["start"], request["goal"]]
        metrics = risk_metrics(field, path)
        corridor = dict(status="GUIDANCE_NO_CONNECTION" if failure else "GUIDANCE_READY",
                        alpha=1.1, label_bins=16, dilation_cells=1, grid_spacing_m=.1,
                        endpoint_radius_m=.2, max_labels=2000000, max_memory_bytes=536870912,
                        guide_used_as_incumbent=False, graph_s=.01, guidance_s=.02, mask_s=.01, total_s=.04,
                        guide_file=None if failure else "corridor_guide_xy.csv",
                        L0=None if failure else 1.5, guide_length=None if failure else 1.5,
                        guide_pri=None if failure else metrics["pri"], guide_exposure=None if failure else 0.,
                        risk_sum=None if failure else metrics["risk_sum"],
                        risk_count=0 if failure else metrics["risk_count"],
                        risk_quantiles=[] if failure else metrics["risk_quantiles"],
                        guide_xy=[] if failure else path,
                        mask=[] if failure else reconstruct_mask(field, path))
        effective = dict(policy_id=policy["policy_id"], risk_cap=None if failure else 40., slope_cap_deg=35.,
                         atlas_sampling=False, thresholds_immutable_during_solve=True,
                         domain="immutable_centerline_corridor", guide_used_as_incumbent=False)
        worker = dict(planner_status="NO_SOLUTION" if failure else "EXACT_SOLUTION",
                      corridor=corridor, requested_safe_length=policy, effective_safe_length=effective,
                      snapshots=[], planner_solve_called=not failure)
        if not failure:
            write_path(directory/"corridor_guide_xy.csv", path)
            write_path(directory/"snapshots/synthetic_0/path_xy.csv", path)
            write_path(directory/"path_xy.csv", path)
            worker["snapshots"] = [dict(snapshot_id="synthetic_0", elapsed_s=.1,
                                        native_objective_value=1.5, cost_terrain_m=1.5, path_xy=path)]
        dump(directory/"worker.json", worker)
        return request, info, worker, oracle, field

    def check(self, directory, request, info, oracle, field, status="COMPLETED"):
        record_corridor_receipt(directory)
        return validate_trial(directory, request, info, dict(status=status, field_build_s=0.),
                              oracle=oracle, field=field)

    def test_raw_map_cloud_and_all_snapshot_validation(self):
        with tempfile.TemporaryDirectory(prefix="corridor_raw_synthetic_", dir=ROOT/"tests/fixtures") as tmp:
            directory = Path(tmp)
            request, info, worker, oracle, field = self.fixture(directory)
            result = self.check(directory, request, info, oracle, field)
            self.assertEqual(result["correctness_errors"], [])
            self.assertTrue(result["success"])
            self.assertTrue(result["corridor"]["independent_valid"])
            self.assertEqual(result["final_pri"], 0.)
            self.assertEqual(result["final_risk_count"], worker["corridor"]["risk_count"])
            self.assertEqual(result["corridor"]["L0_certificate"]["settled_nodes"], 0)

    def test_receipt_tampering_and_wrong_mask_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix="corridor_hash_synthetic_", dir=ROOT/"tests/fixtures") as tmp:
            directory = Path(tmp)
            request, info, worker, oracle, field = self.fixture(directory)
            record_corridor_receipt(directory)
            worker["corridor"]["mask"] = [1]*len(worker["corridor"]["mask"])
            dump(directory/"worker.json", worker)
            result = validate_trial(directory, request, info, dict(status="COMPLETED"), oracle=oracle, field=field)
            self.assertFalse(result["success"])
            self.assertIn("corridor_receipt_mismatch:worker_sha256", result["correctness_errors"])
            self.assertIn("corridor_mask_not_exact_guide_dilation", result["correctness_errors"])

    def test_inflated_L0_cannot_claim_graph_shortest(self):
        with tempfile.TemporaryDirectory(prefix="corridor_L0_synthetic_", dir=ROOT/"tests/fixtures") as tmp:
            directory = Path(tmp)
            request, info, worker, oracle, field = self.fixture(directory)
            worker["corridor"]["L0"] = 1.6
            dump(directory/"worker.json", worker)
            result = self.check(directory, request, info, oracle, field)
            self.assertIn("L0_not_independent_graph_shortest_length", result["correctness_errors"])

    def test_non_graph_guide_and_cycle_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix="corridor_cycle_synthetic_", dir=ROOT/"tests/fixtures") as tmp:
            directory = Path(tmp)
            request, info, worker, oracle, field = self.fixture(directory)
            path = [request["start"], [.8, 1.2], [.9, 1.2], [.8, 1.2], request["goal"]]
            worker["corridor"]["guide_xy"] = path
            write_path(directory/"corridor_guide_xy.csv", path)
            dump(directory/"worker.json", worker)
            result = self.check(directory, request, info, oracle, field)
            self.assertIn("guide_has_cycle", result["correctness_errors"])
            self.assertIn("guide_not_on_registered_graph", result["correctness_errors"])

    def test_guidance_failure_and_crash_are_not_success(self):
        for failure, status in ((True, "COMPLETED"), (False, "CRASH")):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory(
                    prefix="corridor_failure_synthetic_", dir=ROOT/"tests/fixtures") as tmp:
                directory = Path(tmp)
                request, info, _, oracle, field = self.fixture(directory, failure=failure)
                result = self.check(directory, request, info, oracle, field, status=status)
                self.assertEqual(result["correctness_errors"], [])
                self.assertFalse(result["success"])
                self.assertFalse(result["attained_safe"])
                if not failure:
                    self.assertTrue(result["safe_snapshot_observed_in_failed_attempt"])

    def test_closed_cells_mask_checks_entire_edge(self):
        field = synthetic_field(np.ones((10, 10)))
        start, goal = [.15, .25], [.85, .25]
        mask = np.zeros(100, dtype=int).tolist()
        for i, j in field.path_cells([start])+field.path_cells([goal]):
            mask[j*10+i] = 1
        self.assertTrue(path_in_mask(field, [start], mask))
        self.assertTrue(path_in_mask(field, [goal], mask))
        self.assertFalse(path_in_mask(field, [start, goal], mask))
        # A corner contact belongs to all four closed incident cells.
        self.assertEqual(set(field.path_cells([[.2, .2]])), {(1, 1), (1, 2), (2, 1), (2, 2)})
        self.assertEqual(sum(reconstruct_mask(field, [[.2, .2]], 1)), 16)

    def test_unique_cell_PRI_split_reverse_and_duplicate_invariants(self):
        field = synthetic_field(np.arange(100).reshape(10, 10)/10)
        path = [[.15, .25], [.85, .25]]
        expected = risk_metrics(field, path)
        for variant in (path[::-1], [path[0], [.5, .25], path[1]],
                        [path[0], path[0], [.5, .25], path[1], path[1]]):
            self.assertEqual(risk_metrics(field, variant), expected)
        # Same maximum does not imply the same unique-cell mean.
        low = synthetic_field(np.ones((10, 10)))
        low.risk[2, 1] = 10.
        high = synthetic_field(low.risk.copy())
        high.risk[2, 2:8] = 8.
        self.assertEqual(low.score_path(path)["risk_max"], high.score_path(path)["risk_max"])
        self.assertLess(risk_metrics(low, path)["pri"], risk_metrics(high, path)["pri"])

    def test_exposure_surface_integral_and_boundary_maximum(self):
        with tempfile.TemporaryDirectory(prefix="corridor_exposure_synthetic_", dir=ROOT/"tests/fixtures") as tmp:
            terrain = Path(tmp)/"synthetic_tilted.t25d"
            write_map(terrain, [[.2*i for i in range(4)] for _ in range(4)], spacing=(1., 1.), radius=.01, margin=0.)
            oracle = NativePaperOracle(terrain)
            field = synthetic_field(np.ones((30, 30))*2.)
            path = [[.55, 1.2], [2.45, 1.2]]
            expected = 2.*math.dist(*path)*math.sqrt(1.+.2**2)
            self.assertAlmostEqual(risk_exposure(oracle, field, path), expected, places=11)
            for variant in (path[::-1], [path[0], [1.5, 1.2], path[1]], [path[0], path[0], path[1]]):
                self.assertAlmostEqual(risk_exposure(oracle, field, variant), expected, places=11)
            # Along y=1.2 both incident rows count; use the upper risk.
            field.risk[11, :] = 7.
            self.assertAlmostEqual(risk_exposure(oracle, field, path), expected*3.5, places=10)
            field.valid[11, 10] = False
            with self.assertRaisesRegex(ValueError, "missing_sem_support"):
                risk_exposure(oracle, field, path)

    def test_independent_graph_search_goes_around_thin_obstacle(self):
        with tempfile.TemporaryDirectory(prefix="corridor_graph_synthetic_", dir=ROOT/"tests/fixtures") as tmp:
            terrain = Path(tmp)/"synthetic_obstacle.t25d"
            write_map(terrain, np.zeros((5, 5)).tolist(), spacing=(.25, .25), radius=.01, margin=0.,
                      rectangles=[(.49, .3, .51, .7)])
            oracle = NativePaperOracle(terrain)
            field = synthetic_field(np.ones((10, 10)))
            result = shortest_graph_length(oracle, field, [.2, .5], [.8, .5], 1.0)
            self.assertIsNotNone(result["length"])
            self.assertGreater(result["length"], .6)
            self.assertLessEqual(result["length"], 1.0)
            self.assertGreater(result["settled_nodes"], 0)
            self.assertTrue(shortest_graph_length(oracle, field, [.2, .5], [.8, .5], 1.0)["cached"])

    def test_metadata_and_exposure_claims_are_checked(self):
        with tempfile.TemporaryDirectory(prefix="corridor_metadata_synthetic_", dir=ROOT/"tests/fixtures") as tmp:
            directory = Path(tmp)
            request, info, worker, oracle, field = self.fixture(directory)
            worker["corridor"].update(alpha=1.2, guide_exposure=1.)
            dump(directory/"worker.json", worker)
            result = self.check(directory, request, info, oracle, field)
            self.assertIn("corridor_metadata_mismatch:alpha", result["correctness_errors"])
            self.assertIn("guide_metric_mismatch:guide_exposure", result["correctness_errors"])


if __name__ == "__main__":
    unittest.main()
