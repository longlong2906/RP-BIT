"""Schema-12 policy and full-footprint validation without clearance reporting."""
import math
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from independent_verifier import Oracle
from terrain_io import write_map
import terrain_compare_scheduler as scheduler


class V3RuntimeTests(unittest.TestCase):
    def test_registered_runtime_and_watchdogs(self):
        plan = dict(schema_version=12, protocol_id="a3t_sem_putn_kaggle_v3",
                    runtime_policy="native_paper_sessions_v3",
                    phase_timeouts=dict(setup_s=600., serialize_s=60., solve_grace_s=5.))
        old = dict(platform="linux", machine="x86_64", python="3.12.7", cpu_model="A")
        new = {**old, "cpu_model": "B"}
        self.assertEqual(scheduler.campaign_runtime_identity(plan, old),
                         scheduler.campaign_runtime_identity(plan, new))
        self.assertEqual(scheduler.phase_timeout_policy(plan)["setup_watchdog_s"], 600.)
        for changes in (dict(schema_version=11), dict(runtime_policy="native_paper_sessions_v2"),
                        dict(phase_timeouts=dict(setup_s=180., serialize_s=60., solve_grace_s=5.))):
            with self.assertRaises(scheduler.IntegrityError):
                scheduler.phase_timeout_policy({**plan, **changes})
        self.assertNotEqual(scheduler.campaign_runtime_identity({**plan, "schema_version": 11}, old),
                            scheduler.campaign_runtime_identity({**plan, "schema_version": 11}, new))

    def test_clearance_disabled_retains_capsule_collisions_and_slope(self):
        with tempfile.TemporaryDirectory(prefix="v3-oracle-fixture-", dir=ROOT / "tests/fixtures") as tmp:
            path = Path(tmp) / "plane.t25d"
            write_map(path, [[.1 * i * .2 for i in range(31)] for _ in range(31)],
                      spacing=(.2, .2), rectangles=[(2.99, 2., 3.01, 4.)],
                      metadata=dict(synthetic_test_fixture=True))
            oracle = Oracle(path)
            for points, expected in (([[1., 1.], [5., 1.]], True),
                                     ([[1., 3.], [5., 3.]], False),
                                     ([[1., .1], [5., .1]], False),
                                     ([[1., 1.6], [5., 1.6]], False)):
                with self.subTest(points=points):
                    old = oracle.validate_path(points, points[0], points[-1], geometric_safety_metrics=True)
                    new = oracle.validate_path(points, points[0], points[-1],
                                               geometric_safety_metrics=True, clearance_metrics=False)
                    self.assertEqual(new["valid"], expected, new)
                    self.assertEqual(new["valid"], old["valid"])
                    self.assertEqual(new["cost_terrain_m"], old["cost_terrain_m"])
                    self.assertFalse(any("clearance" in key for key in new), new)
                    self.assertAlmostEqual(new["maximum_swept_footprint_slope_deg"],
                                           math.degrees(math.atan(.1)), places=10)

    def test_a3_native_objective_preserved_without_clearance_metric(self):
        with tempfile.TemporaryDirectory(prefix="v3-objective-fixture-", dir=ROOT / "tests/fixtures") as tmp:
            path = Path(tmp) / "flat.t25d"
            write_map(path, [[0.] * 16 for _ in range(16)], spacing=(.2, .2),
                      rectangles=[(2., 1., 2.2, 2.)], metadata=dict(synthetic_test_fixture=True))
            oracle = Oracle(path)
            from a3_traversability_protocol import DEFAULT_TRAVERSABILITY
            points = [[.6, .6], [1.4, .6]]
            old = oracle.validate_path(points, *points, traversability=DEFAULT_TRAVERSABILITY)
            new = oracle.validate_path(points, *points, traversability=DEFAULT_TRAVERSABILITY,
                                       clearance_metrics=False)
            self.assertTrue(new["valid"], new)
            self.assertEqual(old["objective_value"], new["objective_value"])
            self.assertEqual(old["risk_clearance_integral_m"], new["risk_clearance_integral_m"])
            self.assertNotIn("minimum_footprint_clearance_m", new)
            self.assertNotIn("minimum_obstacle_footprint_clearance_m", new)


if __name__ == "__main__":
    unittest.main()
