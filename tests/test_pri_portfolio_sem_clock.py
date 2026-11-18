"""Synthetic clock regressions; these are not experimental observations."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))
from test_sem_native_planner import AnalyticTerrain, cloud
from sem_native_planner import SEMNativeModel, SEMNativePlanner


class AlgorithmClockTests(unittest.TestCase):
    def setUp(self):
        self.planner = SEMNativePlanner(AnalyticTerrain(), SEMNativeModel(cloud(), (0., 0., 6., 6.)))

    def test_budget_spent_before_solve_cannot_publish_direct_solution(self):
        events = []
        with patch("sem_native_planner.time.perf_counter", return_value=100.):
            status = self.planner.solve((1., 1.), (2., 1.), 11, 1., lambda *event: events.append(event),
                                        algorithm_started_at=98.)
        self.assertEqual(status, "TIMEOUT")
        self.assertEqual(events, [])
        self.assertEqual(self.planner.counters["iterations"], 0)

    def test_callback_clock_includes_prior_construction(self):
        events = []
        with patch("sem_native_planner.time.perf_counter", return_value=100.):
            status = self.planner.solve((1., 1.), (2., 1.), 11, 2., lambda *event: events.append(event),
                                        work_iterations=1, algorithm_started_at=99.)
        self.assertEqual(status, "EXACT_SOLUTION")
        self.assertTrue(events)
        self.assertTrue(all(event[0] == 1. for event in events))

    def test_legacy_origin_and_future_clock(self):
        events = []
        with patch("sem_native_planner.time.perf_counter", return_value=100.):
            self.planner.solve((1., 1.), (2., 1.), 11, 2., lambda *event: events.append(event), work_iterations=1)
            self.assertTrue(all(event[0] == 0. for event in events))
            for origin in (101., float("nan"), float("inf")):
                with self.assertRaises(ValueError):
                    self.planner.solve((1., 1.), (2., 1.), 11, 2., lambda *args: None,
                                       algorithm_started_at=origin)


if __name__ == "__main__":
    unittest.main()
