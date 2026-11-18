"""Labeled synthetic objective/oracle fixtures, never experimental run data."""
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
sys.path.insert(0, str(ROOT/"tests"))

from native_paper_oracle import NativePaperOracle
from native_paper_v3_sem_evaluator import SharedSEMField
from run_safe_length import dump
from safe_length_field import file_hash
from sem_exposure_validation import METRICS, OBJECTIVE, exact_sem_exposure, objective_metrics, validate_geometry, validate_trial
from terrain_io import write_map
from test_corridor_v3_validation import synthetic_field, write_path


class SEMExposureValidationTests(unittest.TestCase):
    def workspace(self):
        return tempfile.TemporaryDirectory(prefix="synthetic_sem_exposure_", dir=ROOT/"tests/fixtures")

    def fixture(self, directory, *, detour=False, risk=0., weight=4.):
        terrain = directory/"synthetic_map.t25d"
        write_map(terrain, np.zeros((7, 7)).tolist(), spacing=(.5, .5))
        x, y = np.meshgrid(np.arange(.025, 3., .05), np.arange(.025, 3., .05))
        cloud = directory/"synthetic_cloud.npy"
        np.save(cloud, np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size))), allow_pickle=False)
        info = dict(map_path=str(terrain), map_sha256=file_hash(terrain), cloud_path=str(cloud),
                    cloud_sha256=file_hash(cloud), bounds=[0., 0., 3., 3.])
        field = SharedSEMField(np.load(cloud, allow_pickle=False), info["bounds"])
        field.risk, field.valid = field.risk.copy(), field.valid.copy()
        field.risk[:] = risk
        if detour:
            field.risk[12:14, 10:20] = 20.
        oracle = NativePaperOracle(terrain)
        policy = dict(policy_id="a3t_safe_length_v1", risk_cap=40., slope_cap_deg=35., atlas_sampling=True)
        request = dict(start=[.7, 1.25], goal=[2.2, 1.25], planning_budget_s=2., planner_id="bitstar_a3",
                       samples_per_batch=100, batch_after_first_solution=200, safe_length=policy,
                       objective_type=OBJECTIVE, sem_exposure=dict(lambda_=weight, risk_scale=40.))
        request["sem_exposure"]["lambda"] = request["sem_exposure"].pop("lambda_")
        paths = [[request["start"], request["goal"]]]
        if detour:
            paths.append([request["start"], [.7, 1.55], [2.2, 1.55], request["goal"]])
        worker = dict(objective_type=OBJECTIVE, requested_sem_exposure=copy.deepcopy(request["sem_exposure"]),
                      effective_sem_exposure=dict(request["sem_exposure"], density_version="exact_canonical_raster_v1"),
                      requested_safe_length=copy.deepcopy(policy),
                      effective_safe_length=dict(policy, thresholds_immutable_during_solve=True),
                      effective_policy=dict(batch_policy="adaptive", samples_per_batch=100,
                          batch_after_first_solution=200, safety_atlas_sampling=True, informed_cost_bound="objective_value_J",
                          motion_objective=OBJECTIVE, k_nearest=True, jit=False, planner_class="ompl::geometric::BITstar"),
                      planner_status="EXACT_SOLUTION", execution_status="COMPLETED", batch_transition_count=1,
                      actual_solve_wall_s=.5, exact_certificate=dict(enabled=False))
        events = []
        for i, path in enumerate(paths):
            sid = f"synthetic_{i}"
            values = objective_metrics(oracle, field, path, weight)
            write_path(directory/"snapshots"/sid/"path_xy.csv", path)
            events.append(dict(event_type="improvement", snapshot_id=sid, elapsed_s=.1*(i+1),
                               feasible=True, **values))
        write_path(directory/"path_xy.csv", paths[-1])
        worker.update({key: events[-1][key] for key in METRICS})
        dump(directory/"worker.json", worker)
        self.save_progress(directory, events)
        return request, info, worker, oracle, field, events

    def save_progress(self, directory, events):
        rows = [dict(event_type="poll", elapsed_s=.05, feasible=False, snapshot_id=None,
                     **{key: None for key in METRICS})]
        for event in events:
            rows.extend((event, dict(event_type="poll", elapsed_s=event["elapsed_s"]+.01,
                                    feasible=True, snapshot_id=event["snapshot_id"],
                                    **{key: event[key] for key in METRICS})))
        (directory/"progress.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows), encoding="utf-8")

    def check(self, directory, fixture, status="COMPLETED", independent_rebuild=False):
        request, info, _, oracle, field, _ = fixture
        kw = {} if independent_rebuild else dict(oracle=oracle, field=field)
        return validate_trial(directory, request, info, dict(status=status, field_build_s=0.), **kw)

    def test_raw_cloud_reconstruction_and_all_metrics(self):
        with self.workspace() as tmp:
            directory = Path(tmp)
            fixture = self.fixture(directory)
            row = self.check(directory, fixture, independent_rebuild=True)
            self.assertEqual(row["correctness_errors"], [])
            self.assertTrue(row["success"])
            self.assertEqual(row["final_pri"], 0.)
            self.assertEqual(row["final_exposure"], 0.)
            self.assertAlmostEqual(row["final_cost_terrain_m"], 1.5)
            self.assertEqual(len(row["events"]), 1)

    def test_J_decreases_while_length_increases(self):
        with self.workspace() as tmp:
            directory = Path(tmp)
            fixture = self.fixture(directory, detour=True)
            first, last = fixture[-1]
            self.assertLess(last["objective_value"], first["objective_value"])
            self.assertGreater(last["cost_terrain_m"], first["cost_terrain_m"])
            row = self.check(directory, fixture)
            self.assertEqual(row["correctness_errors"], [])
            self.assertTrue(row["success"])
            self.assertEqual(row["selected_snapshot_id"], "synthetic_1")

    def test_default_objective_parameters_and_missing_effective_policy(self):
        with self.workspace() as tmp:
            directory = Path(tmp)
            fixture = self.fixture(directory, weight=1.)
            fixture[0]["sem_exposure"] = {}
            fixture[2]["requested_sem_exposure"] = {}
            dump(directory/"worker.json", fixture[2])
            self.assertTrue(self.check(directory, fixture)["success"])
            fixture[2]["effective_policy"] = {}
            dump(directory/"worker.json", fixture[2])
            self.assertFalse(self.check(directory, fixture)["success"])

    def test_legacy_density_version_is_rejected_even_on_constant_risk(self):
        with self.workspace() as tmp:
            directory = Path(tmp)
            fixture = self.fixture(directory, risk=5.)
            fixture[2]["effective_sem_exposure"]["density_version"] = "legacy_midpoint_ulp_band"
            dump(directory/"worker.json", fixture[2])
            row = self.check(directory, fixture)
            self.assertFalse(row["success"])
            self.assertIn("effective_objective_mismatch:density_version", row["correctness_errors"])

    def test_tampered_objective_or_terrain_length_cannot_be_accepted(self):
        for key in METRICS:
            with self.subTest(key=key), self.workspace() as tmp:
                directory = Path(tmp)
                fixture = self.fixture(directory, risk=5.)
                fixture[-1][0][key] += .1
                self.save_progress(directory, fixture[-1])
                row = self.check(directory, fixture)
                self.assertFalse(row["success"])
                self.assertTrue(any("metric_mismatch:"+key in error for error in row["correctness_errors"]))

    def test_poll_no_future_backfill_and_same_snapshot_tuple(self):
        for future in (False, True):
            with self.subTest(future=future), self.workspace() as tmp:
                directory = Path(tmp)
                fixture = self.fixture(directory, detour=True)
                progress = [json.loads(line) for line in (directory/"progress.jsonl").read_text().splitlines()]
                if future:
                    progress[0].update(fixture[-1][1], event_type="poll", elapsed_s=.05)
                else:
                    progress[2]["sem_exposure_risk_m"] += 1.
                (directory/"progress.jsonl").write_text("".join(json.dumps(row)+"\n" for row in progress))
                row = self.check(directory, fixture)
                self.assertFalse(row["success"])
                self.assertTrue(any("future_incumbent_backfill" in error if future else "metric_mismatch" in error
                                    for error in row["correctness_errors"]))

    def test_late_snapshot_is_kept_but_not_selected(self):
        with self.workspace() as tmp:
            directory = Path(tmp)
            fixture = self.fixture(directory, detour=True)
            fixture[-1][1]["elapsed_s"] = 2.1
            self.save_progress(directory, fixture[-1])
            row = self.check(directory, fixture)
            self.assertEqual(row["correctness_errors"], [])
            self.assertEqual(row["selected_snapshot_id"], "synthetic_0")
            self.assertEqual(row["late_snapshot_count"], 1)

    def test_failed_attempt_is_retained_and_never_success(self):
        with self.workspace() as tmp:
            directory = Path(tmp)
            row = self.check(directory, self.fixture(directory), status="WATCHDOG")
            self.assertFalse(row["success"])
            self.assertTrue(row["safe_snapshot_observed_in_failed_attempt"])
            self.assertIsNone(row["first_safe_s"])

    def test_empty_no_solution_early_invalid_and_missing_worker_contracts(self):
        for status in ("NO_SOLUTION", "INVALID_START", "INVALID_GOAL"):
            with self.subTest(status=status), self.workspace() as tmp:
                directory = Path(tmp)
                fixture = self.fixture(directory)
                worker = fixture[2]
                worker.update(planner_status=status, path_available=False,
                              **{key: None for key in METRICS})
                if status != "NO_SOLUTION":
                    worker["effective_policy"] = {}
                dump(directory/"worker.json", worker)
                self.save_progress(directory, [])
                row = self.check(directory, fixture)
                self.assertEqual(row["correctness_errors"], [])
                self.assertFalse(row["success"])
                self.assertIsNone(row["final_cost_terrain_m"])
                worker["objective_value"] = 0.
                dump(directory/"worker.json", worker)
                self.assertIn("no_solution_has_fabricated_final_metrics", self.check(directory, fixture)["correctness_errors"])
        with self.workspace() as tmp:
            directory = Path(tmp)
            fixture = self.fixture(directory)
            (directory/"worker.json").unlink()
            self.assertIn("completed_process_missing_worker_output", self.check(directory, fixture)["correctness_errors"])
            crash = self.check(directory, fixture, status="CRASH")
            self.assertFalse(crash["success"])
            self.assertEqual(crash["correctness_errors"], [])

    def test_invalid_last_incumbent_cannot_be_rescued_by_earlier_safe_path(self):
        with self.workspace() as tmp:
            directory = Path(tmp)
            fixture = self.fixture(directory, detour=True)
            bad = [[.7, 1.25], [.7, 1.55], [2.2, 1.55], [2.1999, 1.25]]
            values = objective_metrics(fixture[3], fixture[4], bad, 4.)
            fixture[-1][1].update(values)
            fixture[2].update(values)
            write_path(directory/"snapshots/synthetic_1/path_xy.csv", bad)
            write_path(directory/"path_xy.csv", bad)
            dump(directory/"worker.json", fixture[2])
            self.save_progress(directory, fixture[-1])
            row = self.check(directory, fixture)
            self.assertFalse(row["success"])
            self.assertEqual(row["selected_snapshot_id"], "synthetic_1")
            self.assertIn("synthetic_1:independent_geometry_invalid", row["correctness_errors"])
            self.assertIsNone(row["first_safe_s"])

    def test_missing_support_and_hard_risk_ceiling_fail(self):
        for missing in (False, True):
            with self.subTest(missing=missing), self.workspace() as tmp:
                directory = Path(tmp)
                fixture = self.fixture(directory)
                if missing:
                    fixture[4].valid[12, 15] = False
                else:
                    fixture[4].risk[12, 15] = 41.
                row = self.check(directory, fixture)
                self.assertFalse(row["success"])
                self.assertIn("synthetic_0:hard_safety_contract_failed", row["correctness_errors"])

    def test_raw_hash_mismatch_fails_before_validation(self):
        with self.workspace() as tmp:
            directory = Path(tmp)
            fixture = self.fixture(directory)
            fixture[1]["cloud_sha256"] = "tampered"
            with self.assertRaisesRegex(ValueError, "raw input hash mismatch"):
                self.check(directory, fixture)

    def test_constant_risk_split_reverse_boundary_and_singleton(self):
        with self.workspace() as tmp:
            directory = Path(tmp)
            _, _, _, oracle, field, _ = self.fixture(directory, risk=7.)
            a, b = [.7, 1.2], [2.2, 1.2]
            expected = 1.5*7.
            for path in ([a, b], [b, a], [a, [1.2, 1.2], b]):
                values = objective_metrics(oracle, field, path, 4.)
                self.assertAlmostEqual(values["sem_exposure_risk_m"], expected)
                self.assertAlmostEqual(values["objective_value"], 1.5+4.*expected/40.)
            zero = objective_metrics(oracle, field, [a], 16.)
            self.assertEqual(zero["objective_value"], 0.)
            field.valid[11, 7] = False
            with self.assertRaisesRegex(ValueError, "missing_sem_support"):
                objective_metrics(oracle, field, [a], 0.)

    def test_boundary_uses_max_risk_and_isolated_corner_has_zero_measure(self):
        with self.workspace() as tmp:
            _, _, _, oracle, field, _ = self.fixture(Path(tmp))
            field.risk[11, :] = 3.
            field.risk[12, :] = 9.
            boundary = field.bounds[1]+12*field.resolution
            value = objective_metrics(oracle, field, [[.7, boundary], [2.2, boundary]], 1.)
            self.assertAlmostEqual(value["sem_exposure_risk_m"], 13.5)
            field.risk[:] = 0.
            field.risk[10, 9] = 30.
            diagonal = [[.95, .95], [1.05, 1.05]]
            self.assertGreater(field.score_path(diagonal)["pri"], 0.)
            self.assertAlmostEqual(objective_metrics(oracle, field, diagonal, 1.)["sem_exposure_risk_m"], 0.)

    def test_exact_grid_density_is_additive_inside_the_legacy_ULP_band(self):
        with self.workspace() as tmp:
            terrain = Path(tmp)/"synthetic_boundary_band.t25d"
            write_map(terrain, np.zeros((7, 7)).tolist(), spacing=(1., 1.), radius=.01, margin=0.)
            oracle = NativePaperOracle(terrain)
            field = synthetic_field(np.full((6, 6), 2.))
            field.resolution, field.bounds = 1., (0., 0., 6., 6.)
            field.risk[5, :] = 8.
            unit = math.ulp(5.)
            a, b = [1.25, 5.-8*unit], [1.75, 5.-32*unit]
            mid = [(a[k]+b[k])/2 for k in (0, 1)]
            expected = 2.*math.dist(a, b)
            for points in ([a, b], [b, a], [a, mid, b]):
                self.assertAlmostEqual(exact_sem_exposure(oracle, field, points), expected, places=13)
            # A double midpoint rounds onto the boundary here; its correct
            # interior is strictly below it for every positive parameter.
            a, b = [1.25, 5.], [1.75, math.nextafter(5., 0.)]
            self.assertEqual((a[1]+b[1])/2, 5.)
            for points in ([a, b], [b, a]):
                self.assertAlmostEqual(exact_sem_exposure(oracle, field, points), 2.*math.dist(a, b), places=13)
            self.assertAlmostEqual(exact_sem_exposure(oracle, field, [[1.25, 5.], [1.75, 5.]]), 4., places=13)

    def test_exposure_decrease_does_not_imply_PRI_decrease(self):
        with self.workspace() as tmp:
            _, _, _, oracle, field, _ = self.fixture(Path(tmp))
            field.risk[:] = 2.
            field.risk[15, 10] = 10.
            long_low = [[.7, 1.25], [2.2, 1.25]]
            short_high = [[1.01, 1.51], [1.02, 1.51]]
            self.assertLess(objective_metrics(oracle, field, short_high, 1.)["sem_exposure_risk_m"],
                            objective_metrics(oracle, field, long_low, 1.)["sem_exposure_risk_m"])
            self.assertGreater(field.score_path(short_high)["pri"], field.score_path(long_low)["pri"])

    def test_true_nodata_gap_and_near_diagonal_remain_distinct(self):
        with self.workspace() as tmp:
            directory = Path(tmp)
            _, _, _, oracle, field, _ = self.fixture(directory, risk=2.)
            edge = [[.7, .7], [2.2, math.nextafter(2.2, math.inf)]]
            value = objective_metrics(oracle, field, edge, 1.)
            self.assertAlmostEqual(value["sem_exposure_risk_m"], 2.*math.dist(*edge))
            heights = np.zeros((7, 7)).tolist()
            heights[3][3] = None
            hole = directory/"synthetic_hole.t25d"
            write_map(hole, heights, spacing=(.5, .5))
            with self.assertRaises(ValueError):
                objective_metrics(NativePaperOracle(hole), field, [[.7, 1.5], [2.2, 1.5]], 1.)

    def test_original_edge_exact_coverage_repairs_only_false_geometry_gap(self):
        with self.workspace() as tmp:
            directory = Path(tmp)
            terrain = directory/"synthetic_known_diagonal.t25d"
            write_map(terrain, np.zeros((101, 101)).tolist(), spacing=(.2, .2))
            oracle = NativePaperOracle(terrain)
            field = synthetic_field(np.full((200, 200), 2.))
            points = [[13.3, 5.300000000000001], [18., 10.]]
            expected = math.dist(*points)
            self.assertIn("NoData gap", oracle.validate_path(points, *points)["errors"])
            checked = validate_geometry(oracle, points, *points, expected)
            self.assertTrue(checked["valid"], checked)
            self.assertEqual(checked["surface_coverage_repair"], "exact_original_binary64_edge_intervals")
            values = objective_metrics(oracle, field, points, 4.)
            self.assertAlmostEqual(values["sem_exposure_risk_m"], 2.*expected, places=10)
            heights = np.zeros((101, 101)); heights[37, 77] = np.nan
            hole = directory/"synthetic_known_diagonal_hole.t25d"
            write_map(hole, heights.tolist(), spacing=(.2, .2))
            checked = validate_geometry(NativePaperOracle(hole), points, *points, expected)
            self.assertFalse(checked["valid"])
            self.assertIn("NoData gap", checked["errors"])


if __name__ == "__main__":
    unittest.main()
