"""Analytic and structural tests; all generated maps are synthetic fixtures."""
from pathlib import Path
import json
import math
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tools"))
from sem_native_planner import (SEMNativeModel, SEMNativePlanner, accumulate_native_path_cost,
                                _Node, _turn, WEIGHTS)

FIXTURES = ROOT/"tests/fixtures/native_paper"


def cloud(extent=6., gradient=(0., 0.)):
    x = (np.arange(round(extent/.1))+.5)*.1
    xx, yy = np.meshgrid(x, x)
    return np.column_stack((xx.ravel(), yy.ravel(), (gradient[0]*xx+gradient[1]*yy).ravel()))


class AnalyticTerrain:
    """Analytic fixture, not a substitute for production C++ hard geometry."""
    def __init__(self, extent=6., wall=None, delay=0.):
        self.bounds, self.wall, self.delay = (0., 0., extent, extent), wall, delay
        self.calls = 0
    def valid_motion(self, a, b):
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        if not all(0 <= p[0] <= self.bounds[2] and 0 <= p[1] <= self.bounds[3] for p in (a, b)):
            return False
        return self.wall is None or not min(a[0], b[0]) <= self.wall <= max(a[0], b[0])
    def height(self, point):
        return 0.


class RasterTests(unittest.TestCase):
    def test_top_filter_mean_population_variance(self):
        points = cloud(3.)
        points = np.vstack((points, points-[0, 0, .05], points-[0, 0, .1], points-[0, 0, .5]))
        model = SEMNativeModel(points, (0, 0, 3, 3))
        self.assertTrue(np.all(model.count == 2))  # Equality at max-res is excluded.
        np.testing.assert_allclose(model.mean, -.025, atol=1e-15)
        np.testing.assert_allclose(model.variance, .000625, atol=1e-15)
        np.testing.assert_allclose(model.risk, .125, atol=1e-13)

    def test_normalized_four_directions_are_exact_on_plane(self):
        for gradient in ((.1, 0), (.1, .2), (-.3, .15)):
            model = SEMNativeModel(cloud(6., gradient), (0, 0, 6, 6))
            expected = math.degrees(math.atan(math.hypot(*gradient)))
            np.testing.assert_allclose(model.slope_raw_deg[2:-2, 2:-2], expected, atol=2e-12)
            np.testing.assert_allclose(model.slope_risk_deg[13:-13, 13:-13], expected, atol=2e-12)
            np.testing.assert_allclose(model.risk[13:-13, 13:-13], .75*expected, atol=2e-12)

    def test_nodata_support_propagates_and_cannot_be_crossed(self):
        points = cloud()
        points = points[~((points[:, 0] > 2.9) & (points[:, 0] < 3.))]
        model = SEMNativeModel(points, (0, 0, 6, 6))
        self.assertFalse(model.valid_state((3.05, 3.05)))
        self.assertIsNone(model.edge((1.5, 3), (4.5, 3)))
        self.assertTrue(model.valid_state((.5, .5)))
        self.assertFalse(model.valid_state((-1e-5, .5)))

    def test_risk_cutoff_and_inclusive_bresenham(self):
        model = SEMNativeModel(cloud(3.), (0, 0, 3, 3))
        model.risk[:] = 2.
        a, b = (.25, .25), (1.25, .25)
        length, risk = model.edge(a, b)
        self.assertAlmostEqual(length, 10.)
        self.assertAlmostEqual(risk, 20.)
        model.risk[2, 7] = 40.000001
        self.assertIsNone(model.edge(a, b))
        self.assertIsNone(model.edge(b, a))
        model.risk[2, 7] = 40.
        self.assertIsNotNone(model.edge(a, b))
        self.assertEqual(model.edge(a, a), (0., 0.))

    def test_independent_accumulator_matches_all_octants_and_ties(self):
        model = SEMNativeModel(cloud(3.), (0, 0, 3, 3))
        model.risk[:] = np.random.default_rng(11).uniform(0, 30, model.risk.shape)
        a = (1.55, 1.55)
        for dx in range(-8, 9):
            for dy in range(-8, 9):
                b = (a[0]+dx*.1, a[1]+dy*.1)
                native = model.edge(a, b)
                expected = accumulate_native_path_cost([a, b], model)
                self.assertAlmostEqual(native[0], expected["length_grid"], places=11)
                self.assertAlmostEqual(native[1], expected["risk_integral_grid"], places=10)
                np.testing.assert_allclose(model.edge(a, b), model.edge(b, a), rtol=1e-13)

    def test_frozen_raster_load_and_tamper_rejection(self):
        FIXTURES.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="synthetic_sem_", dir=FIXTURES) as folder:
            original = SEMNativeModel(cloud(3.), (0, 0, 3, 3), folder)
            loaded = SEMNativeModel.from_artifacts(folder)
            np.testing.assert_array_equal(original.risk, loaded.risk)
            actual = accumulate_native_path_cost([(.2, .2), (2., 2.)], Path(folder)/"raster.npz")
            self.assertAlmostEqual(actual["total"], .48*math.hypot(18, 18))
            code = ("import sys;sys.path.insert(0,sys.argv[1]);"
                    "from sem_native_planner import SEMNativeModel,accumulate_native_path_cost;"
                    "m=SEMNativeModel.from_artifacts(sys.argv[2]);"
                    "assert accumulate_native_path_cost([[.2,.2],[2.,2.]],m)['total']>0;"
                    "assert not any(n=='scipy' or n.startswith('scipy.') for n in sys.modules)")
            checked = subprocess.run([sys.executable, "-c", code, str(ROOT/"tools"), folder],
                                     capture_output=True, text=True, timeout=30)
            self.assertEqual(checked.returncode, 0, checked.stdout+checked.stderr)
            with (Path(folder)/"raster.npz").open("ab") as stream:
                stream.write(b"synthetic_tampering")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                SEMNativeModel.from_artifacts(folder)


class PlannerTests(unittest.TestCase):
    def setUp(self):
        self.model = SEMNativeModel(cloud(), (0, 0, 6, 6))
        self.planner = SEMNativePlanner(AnalyticTerrain(), self.model)

    def assert_tree_costs(self, planner):
        for i, node in enumerate(planner.nodes):
            expected = accumulate_native_path_cost(planner._path(node.point, i), planner.model)
            self.assertAlmostEqual(node.cost, expected["total"], places=10)

    def test_turn_threshold_and_endpoint_conventions(self):
        self.assertEqual(_turn((0, 0), (1, 0), (2, 0)), 0.)
        self.assertAlmostEqual(_turn((0, 0), (1, 0), (1, 1)), math.pi/3)
        self.assertAlmostEqual(_turn((0, 0), (1, 0), (0, 0)), 5*math.pi/6)
        self.assertEqual(_turn((0, 0), (0, 0), (1, 0)), 0.)
        angle = math.pi/12
        self.assertEqual(_turn((0, 0), (1, 0), (1+math.cos(angle), math.sin(angle))), 0.)

    def test_parent_cost_uses_grandparent_and_full_start_connection(self):
        p = self.planner
        p.nodes = [_Node((3., 3.), None)]
        a = p._insert((2., 3.), 0, self.model.edge((2, 3), (3, 3)))
        b = p._insert((3., 2.), 0, self.model.edge((3, 2), (3, 3)))
        query = (1., 3.)
        ca = p._candidate(query, a, self.model.edge(query, p.nodes[a].point))
        cb = p._candidate(query, b, self.model.edge(query, p.nodes[b].point))
        self.assertEqual(ca[1][1], 0.)
        self.assertGreater(cb[1][1], 0.)
        for index, candidate in ((a, ca), (b, cb)):
            audit = accumulate_native_path_cost([query, p.nodes[index].point, (3, 3)], self.model)
            self.assertAlmostEqual(candidate[0], audit["total"])

    def test_rewire_recomputes_child_turn_and_descendant_totals(self):
        p = self.planner
        p.nodes = [_Node((5., 3.), None)]
        a = p._insert((3., 3.), 0, self.model.edge((3, 3), (5, 3)))
        b = p._insert((3., 4.), 0, self.model.edge((3, 4), (5, 3)))
        q = p._insert((2., 3.), a, self.model.edge((2, 3), (3, 3)))
        child = p._insert((1., 3.), q, self.model.edge((1, 3), (2, 3)))
        p._insert((.5, 3.), child, self.model.edge((.5, 3), (1, 3)))
        old = p.nodes[child].local[1]
        p._reparent(q, b, self.model.edge((2, 3), (3, 4)))
        self.assertGreater(p.nodes[child].local[1], old)
        self.assertEqual(p.counters["descendant_cost_updates"], 2)
        self.assert_tree_costs(p)
        with self.assertRaisesRegex(ValueError, "cycle"):
            p._reparent(b, child, (1., 0.))

    def test_incumbent_copied_before_tree_rewiring(self):
        p = self.planner
        p.nodes = [_Node((5., 3.), None)]
        a = p._insert((3., 3.), 0, self.model.edge((3, 3), (5, 3)))
        b = p._insert((3., 4.), 0, self.model.edge((3, 4), (5, 3)))
        q = p._insert((2., 3.), a, self.model.edge((2, 3), (3, 3)))
        events = []
        began = time.perf_counter()
        p._publish((1, 3), {q: self.model.edge((1, 3), (2, 3))}, began, began+10, lambda *e: events.append(e))
        saved, cost = p.best_path, p.best_cost
        p._reparent(q, b, self.model.edge((2, 3), (3, 4)))
        self.assertEqual(p.best_path, saved)
        self.assertEqual(p.best_cost, cost)
        self.assertNotEqual(p._path((1, 3), q), saved)
        self.assertAlmostEqual(accumulate_native_path_cost(saved, self.model)["total"], cost)

    def test_weighted_informed_bound_in_physical_units(self):
        p = self.planner
        start, goal = (1., 3.), (5., 3.)
        p.best_cost = .48*5/.1
        rng = np.random.default_rng(8)
        for _ in range(100):
            q = p._sample(rng, start, goal)
            self.assertLessEqual(math.dist(start, q)+math.dist(q, goal), 5+1e-12)

    def test_fixed_work_actual_search_exact_endpoint_cost_and_replay(self):
        runs = []
        for _ in range(2):
            p = SEMNativePlanner(AnalyticTerrain(), self.model)
            events = []
            status = p.solve((.5, .5), (5.5, 5.5), 73135, 10., lambda *e: events.append(e), work_iterations=150)
            self.assertEqual(status, "EXACT_SOLUTION")
            self.assertEqual(p.counters["iterations"], 150)
            self.assertTrue(p.counters["rewires"] > 0)
            self.assertTrue(events)
            costs = [event[3] for event in events]
            self.assertTrue(all(a > b for a, b in zip(costs, costs[1:])))
            for elapsed, path, raw, cost in events:
                self.assertLess(elapsed, 10.)
                self.assertEqual(path[0], [.5, .5]); self.assertEqual(path[-1], [5.5, 5.5])
                self.assertAlmostEqual(cost, accumulate_native_path_cost(path, self.model)["total"], places=9)
            self.assert_tree_costs(p)
            runs.append((p.best_path, p.best_cost, p.counters))
        self.assertEqual(runs[0], runs[1])

    def test_hard_and_native_filters_and_deadline(self):
        events = []
        p = SEMNativePlanner(AnalyticTerrain(wall=3.), self.model)
        self.assertEqual(p.solve((1, 3), (5, 3), 2, .05, lambda *e: events.append(e)), "TIMEOUT")
        self.assertEqual(events, [])
        self.assertGreater(p.counters["hard_rejections"], 0)
        self.model.risk[10, 10] = 41
        self.assertEqual(p.solve((1.05, 1.05), (5, 3), 2, .05, lambda *e: None), "INVALID_NATIVE_ENDPOINTS")
        self.model.risk[10, 10] = 0
        p = SEMNativePlanner(AnalyticTerrain(delay=.002), self.model)
        # Endpoint checks alone exceed this budget; no late initial incumbent.
        self.assertEqual(p.solve((1, 1), (2, 1), 2, .001, lambda *e: events.append(e)), "TIMEOUT")
        self.assertEqual(events, [])

    def test_identity_and_bad_requests(self):
        events = []
        self.assertEqual(self.planner.solve((1, 1), (1, 1), 1, 1., lambda *e: events.append(e)), "EXACT_SOLUTION")
        self.assertEqual(events[0][1], [[1., 1.]])
        self.assertEqual(events[0][3], 0.)
        for budget in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                self.planner.solve((1, 1), (2, 1), 1, budget, lambda *e: None)


if __name__ == "__main__":
    unittest.main()
