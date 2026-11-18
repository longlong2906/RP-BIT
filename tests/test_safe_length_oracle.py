"""Differential safety checks on labelled test fixtures, never benchmark data.

The planner field is checked against the independently grouped SEM evaluator;
continuous footprint/slope checks use the separate raw-map Python Oracle.
"""
import argparse
import csv
import json
import math
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
from safe_length_field import build_field
from terrain_io import write_map

CLI = None


def lattice(extent=6., slope=(0., 0.)):
    axis = np.arange(.025, extent, .05)
    x, y = np.meshgrid(axis, axis)
    return np.column_stack((x.ravel(), y.ravel(), (slope[0]*x+slope[1]*y).ravel()))


def write_exchange(path, support, risk, *, bounds=(0., 0., 6., 6.), resolution=.1):
    """Explicit test-only field writer, independent of production serialization."""
    ny, nx = support.shape
    lines = ["SAFE_SEM_FIELD_V1", " ".join(map(lambda x: format(x, ".17g"),
             (nx, ny, bounds[0], bounds[1], resolution, bounds[2], bounds[3])))]
    lines += [f"{int(v)} {format(float(r) if v else 0., '.17g')}"
              for v, r in zip(support.flat, risk.flat)]
    path.write_text("\n".join(lines)+"\n", encoding="ascii")


class SafeLengthOracleTests(unittest.TestCase):
    def test_planner_field_matches_independent_raw_cloud_evaluator(self):
        flat = lattice()
        plane = lattice(slope=(.13, -.17))
        doubled = np.concatenate((plane, plane+np.array([0., 0., .04])))
        missing = flat[~((flat[:, 0] >= 2.9) & (flat[:, 0] < 3.) &
                         (flat[:, 1] >= 2.9) & (flat[:, 1] < 3.))]
        rng = np.random.default_rng(713)
        noisy = plane.copy()
        noisy[:, 2] += rng.uniform(-.003, .003, len(noisy))
        for points in (flat, plane, doubled, missing, noisy):
            with self.subTest(points=len(points), height=float(points[0, 2])):
                actual = build_field(points, (0., 0., 6., 6.))
                oracle = SharedSEMField(points, (0., 0., 6., 6.))
                np.testing.assert_array_equal(actual.support, oracle.valid)
                np.testing.assert_allclose(actual.mean, oracle.mean, rtol=1e-13,
                                           atol=1e-13, equal_nan=True)
                np.testing.assert_allclose(actual.variance, oracle.variance, rtol=1e-11,
                                           atol=1e-13, equal_nan=True)
                np.testing.assert_allclose(actual.risk, oracle.risk, rtol=1e-11,
                                           atol=1e-10, equal_nan=True)

    def run_cli(self, folder, map_path, field_path, path, tau=40., alpha=35.):
        csv_path = folder/"edge.csv"
        with csv_path.open("w", newline="", encoding="ascii") as stream:
            writer = csv.writer(stream)
            writer.writerow(("x", "y"))
            writer.writerows(path)
        result = subprocess.run([CLI, "--map", str(map_path), "--field", str(field_path),
                                 "--risk-cap", repr(tau), "--slope-cap", repr(alpha),
                                 "--path", str(csv_path)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_closed_cell_risk_support_and_whole_edge_match_independent_evaluator(self):
        if not CLI:
            self.skipTest("requires the built safety_check executable")
        with tempfile.TemporaryDirectory(prefix="safe_length_oracle_") as tmp:
            folder = Path(tmp)
            map_path, field_path = folder/"map.t25d", folder/"field.txt"
            write_map(map_path, [[0.]*13 for _ in range(13)], spacing=(.5, .5),
                      radius=.05, margin=.01, slope=35.)
            oracle = Oracle(map_path)
            evaluator = SharedSEMField(lattice(), (0., 0., 6., 6.))
            risk, support = evaluator.risk.copy(), evaluator.valid.copy()
            support[29:32, 29:32] = False
            risk[15:18, 20:23] = 41.
            risk[40:42, 39:42] = 40.
            evaluator.risk, evaluator.valid = risk, support
            write_exchange(field_path, support, risk)
            rng = random.Random(508)
            paths = [[[rng.uniform(.2, 5.8), rng.uniform(.2, 5.8)] for _ in range(2)]
                     for _ in range(36)]
            paths += [[[1., 3.], [5., 3.]], [[2., 1.6], [2.3, 1.6]],
                      [[3., 3.]], [[4., 4.]], [[1., 1.], [5., 5.]],
                      [[1., 1.6], [4., 1.6]], [[.06, .06], [5.94, .06]]]
            for v in (3., math.nextafter(3., 0.), math.nextafter(3., math.inf)):
                paths.append([[v, 2.8], [v, 3.3]])
            for path in paths:
                expected = all(oracle.valid_edge(a, b) for a, b in
                               (list(zip(path, path[1:])) or [(path[0], path[0])]))
                expected &= evaluator.score_path(path)["valid"]
                result = self.run_cli(folder, map_path, field_path, path)
                self.assertEqual(result["valid"], expected, (path, result))
                reverse = self.run_cli(folder, map_path, field_path, path[::-1])
                self.assertEqual(reverse["valid"], expected, (path, reverse))
                if len(path) == 2:
                    a, b = path
                    mid = [(a[k]+b[k])/2 for k in (0, 1)]
                    split = self.run_cli(folder, map_path, field_path, [a, mid, b])
                    self.assertEqual(split["valid"], expected, (path, split))

    def test_swept_footprint_slope_matches_independent_triangle_oracle(self):
        if not CLI:
            self.skipTest("requires the built safety_check executable")
        with tempfile.TemporaryDirectory(prefix="safe_slope_oracle_") as tmp:
            folder = Path(tmp)
            map_path, field_path = folder/"map.t25d", folder/"field.txt"
            support, risk = np.ones((60, 60), dtype=bool), np.zeros((60, 60))
            write_exchange(field_path, support, risk)
            rng = random.Random(1948)
            for case in ("plane", "fold", "off_center"):
                heights = [[.09*i+.04*j for i in range(13)] for j in range(13)]
                if case == "fold":
                    heights = [[.15*abs(i-6) for i in range(13)] for _ in range(13)]
                if case == "off_center":
                    heights = [[.10*max(0, j-6) for i in range(13)] for j in range(13)]
                write_map(map_path, heights, spacing=(.5, .5), radius=.3, margin=.1, slope=35.)
                oracle = Oracle(map_path)
                edges = [[[rng.uniform(.5, 5.5), rng.uniform(.5, 5.5)] for _ in range(2)]
                         for _ in range(12)]
                edges += [[[1., 2.7], [5., 2.7]], [[1., 2.5], [5., 2.5]],
                          [[2., 2.], [2., 2.]], [[1., 1.], [5., 5.]]]
                for a, b in edges:
                    expected = oracle._swept_slope(np.asarray(a), np.asarray(b))
                    result = self.run_cli(folder, map_path, field_path, [a, b])
                    self.assertAlmostEqual(result["maximum_swept_slope_deg"], expected,
                                           delta=1e-9, msg=(case, a, b, result))
                    for alpha in (5., 15., 35.):
                        result = self.run_cli(folder, map_path, field_path, [a, b], alpha=alpha)
                        self.assertEqual(result["valid"], oracle.valid_edge(a, b) and expected <= alpha,
                                         (case, a, b, alpha, expected, result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli")
    args, remaining = parser.parse_known_args()
    CLI = args.cli
    unittest.main(argv=[sys.argv[0], *remaining])
