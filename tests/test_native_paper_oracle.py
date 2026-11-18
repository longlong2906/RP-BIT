"""Synthetic exact-memoization fixtures; no planner or benchmark execution."""
import copy
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tools"))
from independent_verifier import Oracle
from native_paper_oracle import NativePaperOracle
from terrain_io import write_map

CONFIG = dict(feature_version="cell_envelope_v1", **{"lambda":1.}, slope_weight=.5,
              clearance_weight=.5, clearance_comfort_m=.4)


class NativeOracleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="synthetic_native_oracle_", dir=ROOT/"tests/fixtures")
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def terrain(self, name="plane", **options):
        heights = options.pop("heights", [[.05*i+.025*j for i in range(9)] for j in range(9)])
        path = self.folder/(name+".t25d")
        write_map(path, heights, spacing=(.5,.5), radius=.15, margin=.05, slope=35.,
                  metadata=dict(synthetic_test_fixture=True), **options)
        return path

    def assert_same(self, a, b):
        if isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for key in a:
                if key != "validation_wall_s":
                    self.assert_same(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            self.assertEqual(type(a), type(b))
            self.assertEqual(len(a), len(b))
            for left, right in zip(a, b):
                self.assert_same(left, right)
        else:
            self.assertEqual(a, b)

    def test_safe_shared_snapshot_edges_preserve_all_geometric_metrics(self):
        path = self.terrain(rectangles=[(3.6,.6,3.65,.8)])
        fresh, cached = Oracle(path), NativePaperOracle(path)
        a,b,c,d,e = [0.8,.8],[1.4,.9],[2.1,1.8],[3.2,3.1],[2.2,2.2]
        snapshots = [[a,b,c,d], [a,b,c,e,d], [a,b,c,d], [d,c,b,a]]
        unique = set()
        for xy in snapshots:
            unique.update((tuple(u),tuple(v)) for u,v in zip(xy,xy[1:]))
            expected = fresh.validate_path(xy,xy[0],xy[-1],geometric_safety_metrics=True)
            actual = cached.validate_path(xy,xy[0],xy[-1],geometric_safety_metrics=True)
            self.assertTrue(actual["valid"], actual)
            self.assert_same(expected, actual)
        counts = cached.cache_info()["operations"]["swept_slope"]
        self.assertEqual(counts["misses"],len(unique))
        self.assertGreater(counts["hits"],0)
        # Whole-path claims are checked afresh; only edge computations are reused.
        bad = cached.validate_path(snapshots[0], a, d, claimed_cost=-100., geometric_safety_metrics=True)
        self.assertIn("cost_mismatch",bad["errors"])
        bad = cached.validate_path(snapshots[0], [0.9,.8], d, geometric_safety_metrics=True)
        self.assertIn("not_exact_start_goal",bad["errors"])

    def test_thin_obstacle_footprint_nodata_and_steep_cells_are_not_bypassed(self):
        missing = [[0.]*9 for _ in range(9)]; missing[4][4] = None
        steep = [[0. if i < 4 else 2. for i in range(9)] for _ in range(9)]
        cases = [("thin",dict(rectangles=[(1.99,.2,2.01,3.8)])),
                 ("footprint",dict(rectangles=[(1.99,2.05,2.01,2.1)])),
                 ("nodata",dict(heights=missing)), ("steep",dict(heights=steep))]
        xy = [[1.,2.],[3.,2.]]
        for name, options in cases:
            with self.subTest(case=name):
                path = self.terrain(name, **options)
                fresh, cached = Oracle(path), NativePaperOracle(path)
                for _ in range(2):
                    actual = cached.validate_path(xy,*xy,geometric_safety_metrics=True)
                    self.assertFalse(actual["valid"])
                    self.assert_same(fresh.validate_path(xy,*xy,geometric_safety_metrics=True),actual)
                self.assertGreater(cached.cache_info()["operations"]["valid_edge"]["hits"],0)

    def test_surface_export_and_traversability_checks_still_run(self):
        path = self.terrain(rectangles=[(3.5,.5,3.7,2.)])
        fresh, cached = Oracle(path), NativePaperOracle(path)
        xy = [[1.,1.],[1.8,1.4],[2.5,2.7]]
        expected = fresh.validate_path(xy,xy[0],xy[-1],traversability=CONFIG)
        for _ in range(2):
            actual = cached.validate_path(xy,xy[0],xy[-1],traversability=CONFIG)
            self.assert_same(expected,actual)
        wrong = cached.validate_path(xy,xy[0],xy[-1],surface=[[1.,1.,100.]],traversability=CONFIG)
        self.assertIn("off_surface_point",wrong["errors"])
        self.assertIn("surface_length_mismatch",wrong["errors"])
        for key in ("objective_value","risk_slope_integral_m","risk_clearance_integral_m","risk_integral_m"):
            self.assertEqual(actual[key],expected[key])

    def test_cached_mutable_outputs_cannot_poison_later_reads_or_map(self):
        path = self.terrain(); fresh, cached = Oracle(path), NativePaperOracle(path)
        a,b = [1.,1.],[2.5,2.2]
        value = cached.surface(a,b); value[1][0][2] = 999.
        self.assert_same(cached.surface(a,b),fresh.surface(a,b))
        intervals = cached.intervals(a,b); intervals[0][2][0] = 999.
        self.assert_same(cached.intervals(a,b),fresh.intervals(a,b))
        cost = cached.traversability_edge(a,b,CONFIG); cost["objective_value"] = -999.
        self.assert_same(cached.traversability_edge(a,b,CONFIG),fresh.traversability_edge(a,b,CONFIG))
        with self.assertRaises(ValueError):cached.heights[0,0] = 9.
        with self.assertRaises(AttributeError):cached.radius = 2.
        with self.assertRaises(AttributeError):cached.coefficients = np.zeros_like(cached.coefficients)

    def test_keys_keep_endpoint_order_bits_dtype_and_exact_config(self):
        cached = NativePaperOracle(self.terrain())
        a,b = [1.25,1.25],[2.25,1.25]
        cached.valid_edge(a,b); cached.valid_edge(a,b)
        cached.valid_edge(b,a)
        cached.valid_edge([math.nextafter(a[0],math.inf),a[1]],b)
        cached.valid_edge(np.asarray(a,dtype=np.float32),np.asarray(b,dtype=np.float32))
        counts = cached.cache_info()["operations"]["valid_edge"]
        self.assertEqual(counts,dict(misses=4,hits=1,bypasses=0))
        for x in (0.,-0.):cached.valid_edge([x,1.],b)
        self.assertEqual(cached.cache_info()["operations"]["valid_edge"]["misses"],6)
        config = copy.deepcopy(CONFIG)
        cached.traversability_edge(a,b,config)
        cached.traversability_edge(a,b,dict(reversed(list(config.items()))))
        config["lambda"] = math.nextafter(1.,math.inf)
        cached.traversability_edge(a,b,config)
        counts = cached.cache_info()["operations"]["traversability_edge"]
        self.assertEqual(counts,dict(misses=2,hits=1,bypasses=0))
        cached.intervals(a,b,use_index=False)
        self.assert_same(cached.intervals(a,b,use_index=False),Oracle(self.folder/"plane.t25d").intervals(a,b,use_index=False))

    def test_bounded_lru_zero_capacity_and_distinct_maps(self):
        path = self.terrain(); cached = NativePaperOracle(path,max_entries=2)
        for x in (1.,1.1,1.2,1.):cached.valid_edge([x,1.],[2.,2.])
        self.assertEqual(cached.cache_info()["entries"],2)
        self.assertEqual(cached.cache_info()["evictions"],2)
        tiny = NativePaperOracle(path,max_cache_bytes=1)
        tiny.surface([1.,1.],[2.,2.]); tiny.surface([1.,1.],[2.,2.])
        self.assertEqual(tiny.cache_info()["entries"],0)
        self.assertLessEqual(tiny.cache_info()["accounted_bytes"],1)
        zero = NativePaperOracle(path,max_entries=0)
        zero.valid_edge([1.,1.],[2.,2.]); zero.valid_edge([1.,1.],[2.,2.])
        self.assertEqual(zero.cache_info()["hits"],0)
        other = NativePaperOracle(self.terrain("blocked",rectangles=[(1.4,1.4,1.6,1.6)]))
        self.assertTrue(cached.valid_edge([1.,1.],[2.,2.]))
        self.assertFalse(other.valid_edge([1.,1.],[2.,2.]))
        self.assertNotEqual(cached.map_sha256,other.map_sha256)
        cached.clear_edge_cache()
        self.assertEqual(cached.cache_info()["accounted_bytes"],0)
        for bad in (-1,True,1.5):
            with self.assertRaises(ValueError):NativePaperOracle(path,max_entries=bad)

    def test_nonfinite_and_nodata_failures_match_fresh_oracle(self):
        heights = [[0.]*9 for _ in range(9)]; heights[4][4] = None
        path = self.terrain(heights=heights); fresh,cached = Oracle(path),NativePaperOracle(path)
        self.assertFalse(cached.valid_edge([math.nan,1.],[2.,2.]))
        for a,b in (([1.,2.],[3.,2.]),([-1.,1.],[1.,1.]),([math.nan,1.],[2.,2.])):
            failures=[]
            for oracle in (fresh,cached,cached):
                try:oracle.surface(a,b)
                except ValueError as error:failures.append((type(error),str(error)))
            self.assertEqual(len(failures),3)
            self.assertEqual(failures,[failures[0]]*3)


if __name__ == "__main__":unittest.main(verbosity=2)
