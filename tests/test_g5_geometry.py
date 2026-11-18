"""Synthetic correctness fixtures only; does not generate the held-out dataset."""
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
from g5_dataset import (FAMILIES, MAP_SEEDS, PARAMETER_RANGES, geometry_definition,
                        informed_geometry, oracle_heights, qualify_queries,
                        query_seed, raster_free_domain)
from independent_verifier import Oracle
from terrain_io import write_map


class G5GeometryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="g5_geometry_fixture_")
        self.path = Path(self.temporary.name)/"fixture.t25d"

    def tearDown(self):
        self.temporary.cleanup()

    def test_flat_ellipse_area_and_identical_informed_sets(self):
        write_map(self.path, [[2.]*11 for _ in range(11)], radius=0., margin=0.)
        result = informed_geometry(Oracle(self.path), [3., 5.], [7., 5.], 6., spacing=.05)
        exact_area = math.pi*3.*math.sqrt(5.)
        self.assertLess(abs(result["area_E"]-exact_area), .05)
        self.assertEqual(result["area_E"], result["area_I"])
        self.assertEqual(result["informed_area_ratio"], 1.)
        self.assertEqual(result["max_delta_f"], 0.)

    def test_raster_footprint_matches_continuous_oracle(self):
        heights = [[.1*x+.05*y for x in range(9)] for y in range(9)]
        heights[5][5] = None
        write_map(self.path, heights, radius=.2, margin=.1,
                  rectangles=[(3.9, 0., 4.1, 3.), (1., 6., 2., 7.)])
        oracle = Oracle(self.path)
        points, details = raster_free_domain(oracle, .25)
        actual = {tuple(point) for point in points}
        expected = {(x, y) for x in np.arange(.125, 8., .25)
                    for y in np.arange(.125, 8., .25) if oracle.valid_edge([x, y], [x, y])}
        self.assertEqual(actual, expected)
        self.assertEqual(details["free_midpoints"], len(expected))
        z = oracle_heights(oracle, points)
        self.assertTrue(np.allclose(z, .1*points[:, 0]+.05*points[:, 1], atol=1e-12))

    def test_non_affine_triangle_heights_and_informed_containment(self):
        heights = [[2.-.4*math.hypot(x-5., y-5.) for x in range(11)] for y in range(11)]
        write_map(self.path, heights, radius=.1, margin=0.)
        oracle = Oracle(self.path)
        points = np.array([[1.2, 1.3], [4.8, 5.1], [7.3, 2.7], [5., 5.], [3.5, 3.5]])
        for point, height in zip(points, oracle_heights(oracle, points)):
            self.assertAlmostEqual(height, oracle.surface(point, point)[1][0][2], places=12)
        result = informed_geometry(oracle, [1., 5.], [9., 5.], 9., spacing=.1)
        self.assertLess(result["area_I"], result["area_E"])
        self.assertGreater(result["median_delta_f"], 0.)
        self.assertLessEqual(result["median_delta_f"], result["p90_delta_f"])
        self.assertLessEqual(result["p90_delta_f"], result["max_delta_f"])

    def test_empty_sublevel_is_null_not_fabricated_ratio(self):
        write_map(self.path, [[0.]*11 for _ in range(11)], radius=0., margin=0.)
        result = informed_geometry(Oracle(self.path), [1., 5.], [9., 5.], 8., spacing=.25)
        self.assertEqual(result["area_E"], 0.)
        self.assertIsNone(result["informed_area_ratio"])
        self.assertIsNone(result["median_delta_f"])
        self.assertIsNotNone(result["unavailable_reason"])

    def test_development_geometry_witnesses_and_query_determinism(self):
        # 301 is a development fixture seed, never one of the held-out map seeds.
        model = {"fixture": "synthetic_not_experimental_results"}
        for family in FAMILIES:
            fn, rectangles, definitions, params = geometry_definition(family, 301)
            _, _, repeated, params_again = geometry_definition(family, 301)
            self.assertEqual(definitions, repeated)
            self.assertEqual(params, params_again)
            for key, bounds in PARAMETER_RANGES[family].items():
                self.assertLessEqual(bounds[0], params[key])
                self.assertLessEqual(params[key], bounds[1])
            # A coarser synthetic fixture exercises construction/qualification.
            meta = write_map(self.path, [[fn(x*.5, y*.5) for x in range(81)] for y in range(81)],
                             spacing=(.5, .5), rectangles=rectangles)
            queries = qualify_queries(Oracle(self.path), definitions, meta["data_sha256"], model)
            self.assertEqual(len(queries), 4)
            self.assertTrue(all(query["reference"]["reference_type"] == "WITNESS_UPPER_BOUND" for query in queries))
            self.assertTrue(all(query["reference"]["validation"]["valid"] for query in queries))
        self.assertFalse({201, 202} & set(MAP_SEEDS))
        self.assertEqual(len({query_seed(family, seed, index) for family in FAMILIES
                              for seed in MAP_SEEDS for index in range(4)}), 160)

    def test_qualification_rejects_invalid_witness_without_reselection(self):
        write_map(self.path, [[0.]*11 for _ in range(11)], rectangles=[(4., 0., 6., 10.)])
        definition = {"query_id": "q0", "query_seed": 1, "start": [1., 5.],
                      "goal": [9., 5.], "paths": [[[1., 5.], [9., 5.]]],
                      "geometry_label": "blocked_direct"}
        with self.assertRaisesRegex(ValueError, "witness criterion failed"):
            qualify_queries(Oracle(self.path), [definition], "fixture_hash", {})


if __name__ == "__main__":
    unittest.main()
