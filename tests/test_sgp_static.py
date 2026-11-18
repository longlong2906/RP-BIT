"""Analytic fixtures for the static adapter; no experimental data fabricated."""
import importlib.util
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import sgp_static_adapter as sgp
from terrain_io import write_map
from terrain_compare_external_worker import atomic_json, validate_request

HAS_GP = all(importlib.util.find_spec(name) for name in ("torch", "gpytorch"))


class FeatureTests(unittest.TestCase):
    def test_pinned_upstream_blobs_unchanged(self):
        root = Path(__file__).resolve().parents[1]/"external"/"sgp"
        lock = json.loads((root/"SOURCE_LOCK.json").read_text(encoding="utf-8-sig"))
        self.assertEqual(lock["revision"], sgp.UPSTREAM_COMMIT)
        for source in lock["files"]:
            data = (root/lock["revision"]/source["path"]).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), source["sha256"])
            self.assertEqual(hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest(), source["upstream_git_blob"])

    def test_analytic_plane_and_offset_invariance(self):
        xx, yy = np.meshgrid(np.arange(9)*.2, np.arange(7)*.2)
        z = .1*xx + .2*yy + 7.
        grad = np.tile([.1, .2], (7, 9, 1))
        actual = sgp.feature_fields(z, grad, np.ones_like(z), [0, 0, 1.6, 1.2])
        alpha = math.atan(math.hypot(.1, .2))
        np.testing.assert_allclose(actual["slope_rad"], alpha, atol=1e-14)
        np.testing.assert_allclose(actual["plane_tilt_rad"], alpha, atol=2e-13)
        self.assertAlmostEqual(actual["local_relief_m"][3, 4], .12)
        expected = .7*alpha/.5236 + .3*.12/.30
        self.assertAlmostEqual(actual["traversability"][3, 4], expected)
        offset = sgp.feature_fields(z+100., grad, np.ones_like(z), [100, -80, 101.6, -78.8])
        np.testing.assert_allclose(actual["traversability"], offset["traversability"], atol=2e-12)

    def test_constant_field_has_zero_risk_and_gradient_not_elevation(self):
        z = np.full((7, 7), 100.)
        field = sgp.feature_fields(z, np.zeros((7, 7, 2)), np.ones_like(z), [0, 0, 1.2, 1.2])
        self.assertLess(np.max(field["traversability"]), 1e-12)
        gradient = np.zeros((7, 7, 2)); gradient[:, :, 1] = 1.
        steep = sgp.feature_fields(z, gradient, np.ones_like(z), [0, 0, 1.2, 1.2])
        np.testing.assert_allclose(steep["slope_rad"], math.pi/4)

    def test_uncertainty_has_no_exempt_center(self):
        z = np.zeros((7, 7)); variance = np.ones_like(z); variance[3, 3] = 100.
        field = sgp.feature_fields(z, np.zeros((7, 7, 2)), variance, [0, 0, 1.2, 1.2])
        self.assertTrue(field["uncertainty_mask"][3, 3])
        self.assertEqual(field["traversability"][3, 3], 1.)
        with self.assertRaises(sgp.NativeModelError):
            sgp.feature_fields(z, np.zeros((7, 7, 2)), variance*np.nan, [0, 0, 1.2, 1.2])

    def test_raw_nodes_nodata_and_global_grid_correspondence(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/"map.t25d"
            h = [[1., 2., 3.], [4., None, 6.], [7., 8., 9.]]
            write_map(p, h, origin=(101., -47.), spacing=(.2, .2))
            points, meta = sgp.raw_ground_cloud(p)
            self.assertEqual(len(points), 8)
            self.assertEqual(meta["omitted_nodata_nodes"], 1)
            np.testing.assert_array_equal(points[0], [101., -47., 1.])
            grid, shape = sgp.global_grid(meta["bounds"])
            self.assertEqual(shape, (3, 3))
            np.testing.assert_allclose(grid[3], [101., -46.8], atol=1e-5)

    def test_native_filter_checks_interior_and_global_offset(self):
        model = object.__new__(sgp.SGPStaticModel)
        model.bounds, model.resolution = (100., -50., 104., -46.), .2
        model.traversability = np.zeros((21, 21))
        a, b = (100.6, -48.), (103.4, -48.)
        self.assertTrue(model.valid_motion(a, b))
        model.traversability[10, 10] = 1.
        self.assertTrue(model.valid_state(a)); self.assertTrue(model.valid_state(b))
        self.assertFalse(model.valid_motion(a, b))
        self.assertFalse(model.valid_motion(b, a))
        self.assertFalse(model.valid_state((100., -50.)))


class AnalyticTerrain:
    bounds = (0., 0., 5., 3.)
    def __init__(self, block_goal_connector=False):
        self.cost_calls = 0
        self.valid_calls = 0
        self.block_goal_connector = block_goal_connector
    def height(self, point):
        return .1*point[0]
    def valid_motion(self, a, b):
        self.valid_calls += 1
        if not all(0. <= p[0] <= 5. and 0. <= p[1] <= 3. for p in (a, b)):
            return False
        # Deliberately asymmetric test API guard for terminal direction; the
        # adapter must test the full q->goal connector, not just goal state.
        return not (self.block_goal_connector and tuple(b) == (4., 1.) and tuple(a) != tuple(b))
    def edge_cost(self, a, b):
        self.cost_calls += 1
        length = math.sqrt((a[0]-b[0])**2*1.01 + (a[1]-b[1])**2)
        return (length, length*.2, length*.4, length*.3, length*1.3)


class NativeFree:
    def valid_state(self, point):
        return True
    def valid_motion(self, a, b):
        return True


class PlannerTests(unittest.TestCase):
    def test_atomic_worker_json_preserves_old_value_during_reader_sharing_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"phase.json"
            atomic_json(path, {"phase": "SETUP"})
            replace = os.replace
            attempts = []
            def sharing_conflict(source, destination):
                attempts.append(1)
                if len(attempts) <= 2:
                    self.assertEqual(json.loads(path.read_text())["phase"], "SETUP")
                    raise PermissionError("synthetic Windows reader sharing race")
                return replace(source, destination)
            with mock.patch("terrain_compare_external_worker.os.replace", side_effect=sharing_conflict):
                if os.name == "nt":
                    atomic_json(path, {"phase": "SOLVING"})
                    self.assertEqual(json.loads(path.read_text())["phase"], "SOLVING")
                    self.assertEqual(len(attempts), 3)
                else:
                    with self.assertRaises(PermissionError):
                        atomic_json(path, {"phase": "SOLVING"})
                    self.assertEqual(len(attempts), 1)
                    self.assertEqual(json.loads(path.read_text())["phase"], "SETUP")

    def collect(self, own=False, blocked=False, seed=11):
        terrain = AnalyticTerrain(blocked)
        planner = sgp.SGPStaticPlanner(terrain, NativeFree(), own)
        events = []
        status = planner.solve((1., 1.), (4., 1.), seed, 1., lambda *values: events.append(values), 300)
        return terrain, planner, events, status

    def test_fixedwork_replay_and_common_objective_hooks(self):
        terrain, planner, events, status = self.collect()
        self.assertEqual(status, "EXACT_SOLUTION")
        self.assertGreater(terrain.cost_calls, 0)
        self.assertEqual(planner.counters["iterations"], 300)
        self.assertGreater(planner.counters["goal_connector_checks"], 0)
        self.assertGreater(planner.counters["rewires"], 0)
        for _, path, raw, cost in events:
            self.assertEqual(path[0], [1., 1.]); self.assertEqual(path[-1], [4., 1.])
            self.assertEqual(len(path), len(raw))
            expected = sum(terrain.edge_cost(a, b)[4] for a, b in zip(path, path[1:]))
            self.assertAlmostEqual(cost, expected)
        _, other, replay, _ = self.collect()
        self.assertEqual([e[1:] for e in events], [e[1:] for e in replay])
        self.assertEqual(planner.counters, other.counters)
        for i in range(len(planner.nodes)):
            self.assertTrue(math.isfinite(planner.cost(i)))

    def test_own_cost_is_xy_and_not_common_cost(self):
        terrain, _, events, status = self.collect(own=True)
        self.assertEqual(status, "EXACT_SOLUTION")
        self.assertEqual(terrain.cost_calls, 0)
        for _, path, _, cost in events:
            self.assertAlmostEqual(cost, sum(math.dist(a, b) for a, b in zip(path, path[1:])))

    def test_terminal_edge_rejection_cannot_become_exact_solution(self):
        _, planner, events, status = self.collect(blocked=True)
        self.assertEqual(status, "TIMEOUT")
        self.assertFalse(events)
        self.assertGreater(planner.counters["goal_connector_checks"], 0)
        self.assertGreater(planner.counters["common_checker_rejections"], 0)

    def test_native_model_is_active_in_planning(self):
        class Blocked(NativeFree):
            def valid_motion(self, a, b):
                return False
        planner = sgp.SGPStaticPlanner(AnalyticTerrain(), Blocked())
        events = []
        status = planner.solve((1., 1.), (4., 1.), 11, 1., lambda *v: events.append(v), 50)
        self.assertEqual(status, "TIMEOUT")
        self.assertEqual(planner.counters["native_filter_rejections"], 50)
        self.assertFalse(events)

    def test_request_rejects_ambiguous_track_or_nonfinite(self):
        valid = dict(backend="sgp", track="common", map_path="a", start=[1., 1.], goal=[4., 1.],
                     planner_seed=11, planning_budget_s=2., bridge_library="b")
        validate_request(valid)
        with self.assertRaises(ValueError):
            validate_request(dict(valid, track="unknown"))
        with self.assertRaises(ValueError):
            validate_request(dict(valid, start=[math.nan, 1.]))


@unittest.skipUnless(HAS_GP, "BLOCKED_DEPENDENCY: real torch and gpytorch unavailable")
class RealGPTests(unittest.TestCase):
    def test_real_gp_predictions_gradients_and_block_invariance(self):
        xx, yy = np.meshgrid(np.arange(7)*.2, np.arange(7)*.2)
        points = np.column_stack((xx.ravel(), yy.ravel(), (.05*xx+.02*yy).ravel()))
        original = sgp.PREDICTION_BLOCK_SIZE
        try:
            sgp.PREDICTION_BLOCK_SIZE = 1024
            first = sgp.SGPStaticModel(points, (0., 0., 1.2, 1.2), 11)
            sgp.PREDICTION_BLOCK_SIZE = 13
            second = sgp.SGPStaticModel(points, (0., 0., 1.2, 1.2), 11)
        finally:
            sgp.PREDICTION_BLOCK_SIZE = original
        self.assertEqual(first.metadata["training_steps"], 1)
        self.assertEqual(first.metadata["training_observations"], 49)
        self.assertEqual(first.metadata["inducing_points"], 49)
        self.assertEqual(first.metadata["thread_count"], 1)
        self.assertEqual(first.metadata["interop_thread_count"], 1)
        self.assertGreater(np.max(np.abs(first.gradient)), 0.)
        for attr in ("mean", "variance", "gradient", "traversability"):
            np.testing.assert_allclose(getattr(first, attr), getattr(second, attr), rtol=3e-4, atol=3e-5)
        self.assertTrue(np.array_equal(first.traversability <= .3, second.traversability <= .3))


if __name__ == "__main__":
    unittest.main()
