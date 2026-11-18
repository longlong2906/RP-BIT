"""Synthetic-only tests for the locked A3/SEM common-objective analyzer."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from analyze_a3_sem_rrtstar_common import (
    BOOTSTRAP_HIERARCHY,
    CHECKPOINTS_S,
    EXPECTED_TRIALS,
    SECONDARY_EFFORT_METRICS,
    analyze,
    checkpoint_counter,
    classify_status,
    km_rmst,
    sample_count_checkpoint,
    time_to_first_record,
)
from experiment_common import ROOT, dump, file_hash, read_json


FIXTURE_PATH = ROOT / "tests/fixtures/a3_sem_common_analysis_cases.json"


def _write_snapshot(directory, snapshot_id, elapsed_s, cost_m, counters):
    snapshot = directory / "snapshots" / snapshot_id
    snapshot.mkdir(parents=True)
    xy = snapshot / "path_xy.csv"
    surface = snapshot / "path_surface.csv"
    xy.write_text("x_m,y_m\n0,0\n8,0\n", encoding="utf-8")
    surface.write_text("x_m,y_m,z_m\n0,0,0\n8,0,0\n", encoding="utf-8")
    dump(snapshot / "validation.json", {
        "valid": True,
        "cost_terrain_m": cost_m,
        "xy_sha256": file_hash(xy),
        "surface_sha256": file_hash(surface),
        "synthetic_test_fixture": True,
    })
    return {
        "snapshot_id": snapshot_id,
        "elapsed_s": elapsed_s,
        "best_cost_terrain_m": cost_m,
        "independent_valid": True,
        "counters": counters,
    }


def build_fixture(directory):
    """Expand the labelled fixture to the exact 300-row protocol shape."""
    specification = read_json(FIXTURE_PATH)
    assert specification["synthetic_test_fixture"] is True
    source = Path(directory) / "raw"
    source.mkdir()
    config = yaml.safe_load((ROOT / "configs/a3_vs_sem_rrtstar_common_v1.yaml").read_text(encoding="utf-8"))
    dump(source / "resolved_config.json", config)
    entries = []
    for family in specification["families"]:
        for map_seed in specification["map_seeds"]:
            map_id = f"{family}_{map_seed}"
            for seed_rank, planner_seed in enumerate(specification["planner_seeds"]):
                for planner in specification["planners"]:
                    behavior = specification["behavior"][planner]
                    trial_id = f"{map_id}_q0_{planner}_s{planner_seed}"
                    entry = {
                        "trial_id": trial_id,
                        "family": family,
                        "map_id": map_id,
                        "map_seed": map_seed,
                        "map_hash": "synthetic_" + map_id,
                        "query_id": "q0",
                        "planner_id": planner,
                        "planner_seed": planner_seed,
                    }
                    entries.append(entry)
                    trial_directory = source / trial_id
                    success = seed_rank in behavior["successful_seed_ranks"]
                    scale = behavior["effort_scale"]
                    progress = []
                    for checkpoint in CHECKPOINTS_S:
                        counters = {
                            name: scale * int(round(checkpoint * 100)) + index
                            for index, name in enumerate(SECONDARY_EFFORT_METRICS)
                        }
                        progress.append({"event_type": "poll", "elapsed_s": checkpoint,
                                         "feasible": success and checkpoint >= behavior["first_solution_s"],
                                         "best_cost_terrain_m": None, "counters": counters,
                                         "snapshot_id": None})
                    trial_directory.mkdir(parents=True)
                    (trial_directory / "progress.jsonl").write_text(
                        "".join(json.dumps(record, allow_nan=False) + "\n" for record in progress),
                        encoding="utf-8",
                    )
                    events = []
                    if success:
                        events.append(_write_snapshot(
                            trial_directory, "s0", behavior["first_solution_s"],
                            behavior["initial_cost_m"], progress[3]["counters"],
                        ))
                        events.append(_write_snapshot(
                            trial_directory, "s1", behavior["improved_time_s"],
                            behavior["improved_cost_m"], progress[-2]["counters"],
                        ))
                    dump(trial_directory / "validated_events.json", events)
                    dump(trial_directory / "validation.json", {
                        "valid": success,
                        "synthetic_test_fixture": True,
                    })
                    final_cost = behavior["improved_cost_m"] if success else None
                    final_counters = progress[-1]["counters"]
                    dump(trial_directory / "trial.json", {
                        **entry,
                        "execution_status": "COMPLETED" if success else "TIMEOUT",
                        "planner_status": "EXACT_SOLUTION" if success else "NO_SOLUTION",
                        "path_available": success,
                        "independent_valid": True if success else False,
                        "time_budget_s": 10.0,
                        "best_cost_terrain_m": final_cost,
                        "first_solution_time_s": behavior["first_solution_s"] if success else None,
                        "length_xy_m": 8.0 if success else None,
                        "independently_recomputed_cost_m": final_cost,
                        "actual_solve_wall_s": 10.0,
                        "worker_wall_s": 10.1,
                        "validation_wall_s": 0.01,
                        "peak_rss_mb": 42.0 + scale,
                        "termination_reason": "TIME_BUDGET",
                        "counters": final_counters,
                        "synthetic_test_fixture": True,
                    })
    assert len(entries) == EXPECTED_TRIALS
    dump(source / "manifest.json", {
        "schema_version": 3,
        "experiment_id": "synthetic_a3_vs_sem_common_fixture",
        "profile": "sem_comparison",
        "planned_trials": len(entries),
        "status": "SYNTHETIC_TEST_FIXTURE",
        "synthetic_test_fixture": True,
        "trials": entries,
    })
    return source


class A3SemCommonAnalysisTests(unittest.TestCase):
    def test_km_rmst_and_right_censoring_do_not_fabricate_timeout_events(self):
        result = km_rmst([
            {"time_s": 1.0, "event_observed": True},
            {"time_s": 3.0, "event_observed": False},
        ], 10.0)
        self.assertAlmostEqual(result["rmst_s"], 5.5)
        self.assertEqual((result["events"], result["right_censored"]), (1, 1))
        record = time_to_first_record({
            "executed": True, "success": False, "validation_pending": False,
            "status": "TIMEOUT", "time_budget_s": 10.0,
            "metrics": {"actual_solve_wall_s": 10.0, "worker_wall_s": 10.2},
        })
        self.assertFalse(record["event_observed"])
        self.assertEqual((record["status"], record["time_s"]), ("RIGHT_CENSORED", 10.0))

    def test_checkpoint_counter_never_backfills_a_future_poll(self):
        progress = [
            {"elapsed_s": 0.2, "counters": {"motion_validity_calls": 2}},
            {"elapsed_s": 0.8, "counters": {"motion_validity_calls": 8}},
        ]
        self.assertIsNone(checkpoint_counter(progress, "motion_validity_calls", 0.1)["value"])
        at_half = checkpoint_counter(progress, "motion_validity_calls", 0.5)
        self.assertEqual(at_half["value"], 2)
        self.assertAlmostEqual(at_half["poll_lag_s"], 0.3)

    def test_sample_checkpoint_never_uses_post_budget_poll(self):
        row = {
            "time_budget_s": 10.0,
            "progress": [
                {"elapsed_s": 9.9, "counters": {"sample_proposals": 4999}},
                {"elapsed_s": 10.01, "counters": {"sample_proposals": 5001}},
            ],
            "events": [],
            "success": False,
        }
        checkpoint = sample_count_checkpoint(row)
        self.assertFalse(checkpoint["available"])
        self.assertEqual(checkpoint["reason"], "sample_proposal_checkpoint_not_observed")

    def test_status_taxonomy_is_exclusive(self):
        complete = [{"executed": True, "validation_pending": False}]
        partial = [{"executed": False, "validation_pending": False}]
        self.assertEqual(classify_status([], [], complete), "COMPLETE")
        self.assertEqual(classify_status([], [], partial), "PARTIAL")
        self.assertEqual(classify_status(["protocol"], [], partial), "FAILED_CORRECTNESS")
        self.assertEqual(classify_status([], [{"trial_id": "x", "errors": ["bad"]}], complete),
                         "FAILED_CORRECTNESS")

    def test_complete_300_trial_fixture_primary_metrics_coverage_and_schema(self):
        with tempfile.TemporaryDirectory(prefix="a3_sem_analysis_", dir=ROOT / "tests/fixtures") as directory:
            source = build_fixture(directory)
            output = Path(directory) / "report"
            summary = analyze(source, output)
            self.assertEqual(summary["status"], "COMPLETE")
            self.assertEqual(summary["bootstrap_hierarchy"], list(BOOTSTRAP_HIERARCHY))
            self.assertEqual((summary["planned"], summary["executed"], summary["not_run"]), (300, 300, 0))
            self.assertEqual(summary["validated_success_at_10s"], 250)
            self.assertEqual(summary["timeout"], 50)
            paired = read_json(output / "paired_bootstrap.json")
            by_metric = {item["metric"]: item for item in paired}
            success = by_metric["validated_success_at_10s"]
            self.assertAlmostEqual(success["estimate"], 1 / 3)
            self.assertEqual(success["units"]["usable_paired_seed_slots"], 150)
            cost = by_metric["cost_terrain_m_at_10s_both_success"]
            self.assertEqual(cost["estimate"], 1.0)
            self.assertEqual(cost["units"]["usable_paired_seed_slots"], 100)
            xy_length = by_metric["length_xy_m_final_both_success"]
            self.assertEqual(xy_length["estimate"], 0.0)
            self.assertEqual(xy_length["units"]["usable_paired_seed_slots"], 100)
            rmst = by_metric["t_first_rmst_tau_10s"]
            self.assertAlmostEqual(rmst["estimate"], -11 / 3)
            self.assertEqual(rmst["units"]["usable_paired_seed_slots"], 150)
            effort = by_metric["effort_motion_validity_calls_at_10s"]
            self.assertEqual(effort["estimate"], 1000.0)
            descriptive = read_json(output / "descriptive_metrics.json")
            a3_descriptive = next(group for group in descriptive["groups"]
                                  if group["scope"] == "planner" and group["planner_id"] == "bitstar_a3")
            self.assertEqual(a3_descriptive["metrics"]["actual_solve_wall_s"]["available"], 150)
            self.assertEqual(a3_descriptive["metrics"]["peak_rss_mb"]["median"], 44.0)
            checkpoints = read_json(output / "checkpoints.json")
            a3 = next(group for group in checkpoints["groups"]
                      if group["scope"] == "planner" and group["planner_id"] == "bitstar_a3")
            self.assertEqual(a3["points"][0]["validated_success"], 0)
            self.assertEqual(a3["points"][3]["validated_success"], 150)
            report_text = (output / "REPORT.md").read_text(encoding="utf-8")
            self.assertIn("SEM-RRT*-adapted", report_text)
            self.assertNotIn("NaN", (output / "summary.json").read_text(encoding="utf-8"))
            self.assertNotIn("Infinity", (output / "trial_metrics.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
