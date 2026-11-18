"""Synthetic protocol/selection fixtures; never benchmark observations."""
import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from run_safe_length import load_config, request_for, run, validate_trial, dump
from safe_length_field import file_hash
from terrain_io import write_map
from native_paper_v3_sem_evaluator import SharedSEMField
from independent_verifier import Oracle


class SafeLengthRunnerTests(unittest.TestCase):
    def test_ablation_switches_and_baseline_identity(self):
        config = load_config()
        info = dict(start=[1, 1], goal=[3, 3], map_path="synthetic.t25d", map_sha256="a"*64,
                    cloud_path="synthetic.npy", cloud_sha256="b"*64)
        meta = dict(field_sha256="c"*64, cloud_sha256="b"*64, builder_source_sha256="d"*64)
        requests = {method:request_for(method, info, config, "synthetic.txt", meta)
                    for method in config["methods"] if method != "sem_native"}
        self.assertFalse(requests["constrained_fixed200"]["safe_length"]["atlas_sampling"])
        self.assertTrue(requests["atlas_fixed200"]["safe_length"]["atlas_sampling"])
        adaptive = requests["atlas_adaptive100_200"]
        self.assertEqual((adaptive["samples_per_batch"], adaptive["batch_after_first_solution"]), (100, 200))
        common = requests["sem_common_safe_length"]
        self.assertEqual(common["planner_id"], "sem_rrtstar_common_v1")
        self.assertFalse(common["safe_length"]["atlas_sampling"])
        self.assertEqual(common["batch_after_first_solution"], 0)
        for request in requests.values():
            self.assertEqual(request["objective_type"], "surface_length")
            self.assertNotIn("traversability", request)
            self.assertEqual(request["safe_length"]["policy_id"], "a3t_safe_length_v1")
        native = request_for("sem_native", info, config, bridge="synthetic.dll")
        self.assertEqual(native["planner_id"], "sem_native_paper_v2")
        self.assertNotIn("safe_length", native)
        self.assertEqual(native["native_objective_id"], "sem_length_turn_risk_paper_v2")

    def test_pilot_full_never_execute_smoke_silently(self):
        for profile in ("pilot", "full"):
            for allow in (False, True):
                with self.assertRaises(ValueError):
                    run("does-not-exist", "does-not-exist", profile=profile,
                        allow_pilot=allow, allow_full=allow)

    def test_seed_and_cap_changes_rejected(self):
        with tempfile.TemporaryDirectory(prefix="safe_runner_synthetic_", dir=ROOT/"tests/fixtures") as tmp:
            path = Path(tmp)/"config.json"
            for key, value in (("planner_seeds", [22]), ("risk_cap", 16.), ("max_workers", 4)):
                config = copy.deepcopy(load_config())
                config[key] = value
                dump(path, config)
                with self.assertRaises(ValueError):
                    load_config(path)

    def _fixture(self, directory):
        mapfile = directory/"synthetic_selection.t25d"
        write_map(mapfile, np.zeros((11, 11)).tolist(), spacing=(1, 1))
        x, y = np.meshgrid((np.arange(100)+.5)*.1, (np.arange(100)+.5)*.1)
        points = np.column_stack((x.ravel(), y.ravel(), np.zeros(10000)))
        points = points[~((points[:, 0] > 5) & (points[:, 0] < 5.1) &
                         (points[:, 1] > 5) & (points[:, 1] < 5.1))]
        cp = directory/"synthetic_selection.cloud.npy"
        np.save(cp, points)
        info = dict(map_path=str(mapfile), map_sha256=file_hash(mapfile), cloud_path=str(cp),
                    cloud_sha256=file_hash(cp), bounds=[0, 0, 10, 10])
        request = dict(start=[1., 5.], goal=[9., 5.], planning_budget_s=2.)
        paths = ([[1., 5.], [1., 2.], [9., 2.], [9., 5.]], [[1., 5.], [9., 5.]])
        events = []
        for index, path in enumerate(paths):
            sid = "synthetic_"+str(index)
            target = directory/"snapshots"/sid/"path_xy.csv"
            target.parent.mkdir(parents=True)
            with target.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["x", "y"])
                writer.writerows(path)
            events.append(dict(snapshot_id=sid, elapsed_s=.1*(index+1), native_objective_value=14.-6*index,
                               path_xy=path, cost_terrain_m=14.-6*index))
        (directory/"path_xy.csv").write_bytes((directory/"snapshots/synthetic_1/path_xy.csv").read_bytes())
        dump(directory/"worker.json", dict(status="EXACT_SOLUTION", snapshots=events))
        return request, info, Oracle(mapfile), SharedSEMField(points, info["bounds"])

    def test_unsafe_final_not_replaced_with_safe_snapshot(self):
        with tempfile.TemporaryDirectory(prefix="safe_selection_synthetic_", dir=ROOT/"tests/fixtures") as tmp:
            directory = Path(tmp)
            request, info, oracle, field = self._fixture(directory)
            result = validate_trial(directory, request, info, dict(status="COMPLETED", field_build_s=0.),
                                    oracle=oracle, field=field)
            self.assertTrue(result["attained_safe"])
            self.assertFalse(result["success"])
            self.assertEqual(result["first_safe_s"], .1)
            self.assertEqual(result["selected_snapshot_id"], "synthetic_1")
            self.assertEqual(result["correctness_errors"], [])
            failed = validate_trial(directory, request, info, dict(status="CRASH", field_build_s=0.),
                                    oracle=oracle, field=field)
            self.assertFalse(failed["attained_safe"])
            self.assertIsNone(failed["first_safe_s"])
            self.assertTrue(failed["safe_snapshot_observed_in_failed_attempt"])
            request["safe_length"] = dict(risk_cap=40., slope_cap_deg=35.)
            constrained = validate_trial(directory, request, info, dict(status="COMPLETED", field_build_s=0.),
                                         oracle=oracle, field=field)
            self.assertIn("synthetic_1:hard_safety_contract_failed", constrained["correctness_errors"])

    def test_missing_worker_and_input_change_are_visible(self):
        with tempfile.TemporaryDirectory(prefix="safe_missing_synthetic_", dir=ROOT/"tests/fixtures") as tmp:
            directory = Path(tmp)
            request, info, _, _ = self._fixture(directory)
            (directory/"worker.json").unlink()
            result = validate_trial(directory, request, info, dict(status="CRASH"))
            self.assertFalse(result["success"])
            self.assertEqual(result["unavailable_reason"], "no_worker_output")
            Path(info["map_path"]).write_text("changed synthetic test fixture")
            with self.assertRaises(ValueError):
                validate_trial(directory, request, info, dict(status="CRASH"))

    def test_late_snapshot_retained_but_not_selected(self):
        with tempfile.TemporaryDirectory(prefix="safe_deadline_synthetic_", dir=ROOT/"tests/fixtures") as tmp:
            directory = Path(tmp)
            request, info, oracle, field = self._fixture(directory)
            worker = json.loads((directory/"worker.json").read_text())
            worker["snapshots"][-1]["elapsed_s"] = 2.1
            dump(directory/"worker.json", worker)
            result = validate_trial(directory, request, info, dict(status="COMPLETED", field_build_s=0.),
                                    oracle=oracle, field=field)
            self.assertTrue(result["success"])
            self.assertEqual(result["selected_snapshot_id"], "synthetic_0")
            self.assertEqual(result["late_snapshot_count"], 1)
            self.assertEqual(len(result["events"]), 2)
            self.assertEqual(result["correctness_errors"], [])


if __name__ == "__main__":
    unittest.main()
