"""Analytic SEM and closed-grid traversal controls; no experimental results."""
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tools"))
from native_paper_v3_sem_evaluator import SharedSEMField, ensure_shared_field, file_hash


def lattice(extent=4., slope=(0., 0.)):
    axis = np.arange(.025, extent, .05)
    x, y = np.meshgrid(axis, axis)
    return np.column_stack((x.ravel(), y.ravel(), (slope[0]*x+slope[1]*y).ravel()))


class SharedSEMTests(unittest.TestCase):
    def setUp(self):
        self.fixtures = ROOT/"tests/fixtures/native_paper_v3"
        self.fixtures.mkdir(parents=True, exist_ok=True)

    def test_flat_and_population_variance(self):
        field = SharedSEMField(lattice(), (0., 0., 4., 4.))
        self.assertTrue(field.valid.all())
        np.testing.assert_array_equal(field.risk, 0.)
        self.assertTrue(field.score_path([[1., 1.], [3., 1.]])["valid"])
        points = lattice()
        doubled = np.concatenate((points, points+np.array([0., 0., .04])))
        field = SharedSEMField(doubled, (0., 0., 4., 4.))
        self.assertAlmostEqual(float(field.variance[20, 20]), .0004, places=14)
        self.assertAlmostEqual(float(field.risk[20, 20]), .08, places=12)

    def test_plane_exact_slope_and_known_cell_variance(self):
        ax, ay = .15, -.2
        field = SharedSEMField(lattice(6., (ax, ay)), (0., 0., 6., 6.))
        expected = math.degrees(math.atan(math.hypot(ax, ay)))
        self.assertAlmostEqual(float(field.slope_raw_deg[30, 30]), expected, places=11)
        self.assertAlmostEqual(float(field.slope_risk_deg[30, 30]), expected, places=11)
        # Four lattice samples per cell at (+/- .025, +/- .025).
        variance = (ax*ax+ay*ay)*.025**2
        self.assertAlmostEqual(float(field.variance[30, 30]), variance, places=13)
        self.assertAlmostEqual(float(field.risk[30, 30]), 200*variance+.75*expected, places=10)

    def test_missing_support_propagates_all_stencils(self):
        points = lattice(6.)
        keep = ~((points[:, 0] >= 2.9) & (points[:, 0] < 3.) & (points[:, 1] >= 2.9) & (points[:, 1] < 3.))
        field = SharedSEMField(points[keep], (0., 0., 6., 6.))
        self.assertFalse(field.valid[29, 29])
        self.assertFalse(field.valid[39, 29])
        self.assertTrue(field.valid[45, 45])
        scored = field.score_path([[2.95, 2.95], [3.05, 3.05]])
        self.assertFalse(scored["valid"])
        self.assertIsNone(scored["pri"])
        self.assertIn("missing_sem_support", scored["reasons"])

    def test_closed_boundary_supercover_reverse_split_and_distinct_cells(self):
        field = SharedSEMField(lattice(), (0., 0., 4., 4.))
        path = [[.15, .15], [3.85, 2.35]]
        mid = [(path[0][k]+path[1][k])/2 for k in (0, 1)]
        whole = field.path_cells(path)
        self.assertEqual(whole, field.path_cells(path[::-1]))
        self.assertEqual(whole, field.path_cells([path[0], mid, path[1]]))
        self.assertEqual(whole, field.path_cells([path[0], path[1], path[0]]))
        self.assertEqual(len(whole), len(set(whole)))
        on_boundary = set(field.path_cells([[1., 1.05], [1., 1.25]]))
        self.assertIn((9, 11), on_boundary)
        self.assertIn((10, 11), on_boundary)
        at_corner = field.path_cells([[1., 1.]])
        self.assertEqual(set(at_corner), {(9, 9), (9, 10), (10, 9), (10, 10)})
        self.assertFalse(field.score_path([[-.1, 1.], [1., 1.]])["valid"])

    def test_risk_40_inclusive_and_unique_cell_mean(self):
        field = SharedSEMField(lattice(), (0., 0., 4., 4.))
        # Explicit scorer fixture: independence of scoring from construction.
        risk = field.risk.copy()
        risk[9:11, 9:11] = 40.
        field.risk = risk
        result = field.score_path([[1., 1.]])
        self.assertTrue(result["valid"])
        self.assertEqual(result["pri"], 40.)
        self.assertEqual(result["cell_count"], 4)
        risk[9, 9] = math.nextafter(40., math.inf)
        self.assertFalse(field.score_path([[1., 1.]])["valid"])

    def test_supercover_matches_independent_rectangle_clipping(self):
        field = SharedSEMField(lattice(), (0., 0., 4., 4.))
        rng = np.random.default_rng(717)
        for _ in range(24):
            a, b = rng.integers(2, 78, size=(2, 2))*.05
            expected = set()
            # Brute-force closed rectangle clipping is deliberately different
            # from evaluator grid-crossing interval traversal.
            for j in range(40):
                for i in range(40):
                    low, high = 0., 1.
                    for axis, index in ((0, i), (1, j)):
                        delta = b[axis]-a[axis]
                        minimum, maximum = index*.1, (index+1)*.1
                        if abs(delta) < 1e-14:
                            if not minimum-1e-12 <= a[axis] <= maximum+1e-12:
                                low, high = 1., 0.
                                break
                        else:
                            x, y = sorted(((minimum-a[axis])/delta, (maximum-a[axis])/delta))
                            low, high = max(low, x), min(high, y)
                    if low <= high+1e-12:
                        expected.add((i, j))
            self.assertEqual(set(field.path_cells([a, b])), expected, (a, b))
            midpoint = (a+b)/2
            self.assertEqual(set(field.path_cells([a, midpoint, b])), expected, (a, b))

    def test_saved_field_hash_identity_and_readonly(self):
        with tempfile.TemporaryDirectory(prefix="analytic_shared_sem_", dir=self.fixtures) as directory:
            directory = Path(directory)
            cloud = directory/"fixture.cloud.npy"
            np.save(cloud, lattice(), allow_pickle=False)
            output = directory/"evaluator"
            original = ensure_shared_field(output, cloud, file_hash(cloud), (0., 0., 4., 4.), "fixture_map_hash")
            restored = ensure_shared_field(output, cloud, file_hash(cloud), (0., 0., 4., 4.), "fixture_map_hash")
            np.testing.assert_array_equal(original.risk, restored.risk)
            self.assertFalse(restored.risk.flags.writeable)
            with self.assertRaisesRegex(ValueError, "identity"):
                ensure_shared_field(output, cloud, file_hash(cloud), (0., 0., 4., 4.), "changed_map_hash")
            with (output/"field.npz").open("ab") as stream:
                stream.write(b"tampered_fixture")
            with self.assertRaisesRegex(ValueError, "hash"):
                ensure_shared_field(output, cloud, file_hash(cloud), (0., 0., 4., 4.), "fixture_map_hash")


if __name__ == "__main__":
    unittest.main()
