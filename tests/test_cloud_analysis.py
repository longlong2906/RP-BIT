"""Synthetic-only v2 analysis regression; fixtures remain under tests/fixtures."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from analyze_cloud_campaign import (
    analyze, validate_design, time_record, km_rmst, paired_bootstrap, read_rows,
    group_counts, checkpoints, FAMILIES, PLANNERS, MAP_SEEDS, PLANNER_SEEDS,
)
from experiment_common import ROOT, digest, dump, file_hash, read_json, load_config


def fixture_spec():
    return read_json(ROOT / "tests/fixtures/cloud_analysis_cases.json")


def entries():
    return [{"trial_id": f"{family}_{map_seed}_q0_{planner}_s{seed}", "family": family,
             "map_id": f"{family}_{map_seed}", "map_seed": map_seed,
             "map_hash": "synthetic_map_" + f"{family}_{map_seed}", "query_id": "q0",
             "query_hash": "synthetic_query_" + f"{family}_{map_seed}", "planner_id": planner, "planner_seed": seed}
            for family in FAMILIES for map_seed in MAP_SEEDS for seed in PLANNER_SEEDS for planner in PLANNERS]


def inventory(source):
    records = [{"path": path.relative_to(source).as_posix(), "size_bytes": path.stat().st_size,
                "sha256": file_hash(path)} for path in sorted(source.rglob("*"))
               if path.is_file() and path.name != "raw_artifact_inventory.json"]
    dump(source / "raw_artifact_inventory.json", {"schema_version": 1, "entries": records, "inventory_sha256": digest(records)})


def write_trial(source, entry, *, pending=False, status="COMPLETED", invalid=False):
    directory = source / entry["trial_id"]
    directory.mkdir(parents=True)
    successful = status == "COMPLETED"
    events = []
    if successful and not pending:
        for event in fixture_spec()["events"]:
            snapshot = directory / "snapshots" / event["snapshot_id"]
            snapshot.mkdir(parents=True)
            cost = event["best_cost_terrain_m"]
            (snapshot / "path_xy.csv").write_text(f"x_m,y_m\n0,0\n{cost},0\n", encoding="utf-8")
            (snapshot / "path_surface.csv").write_text(f"x_m,y_m,z_m\n0,0,0\n{cost},0,0\n", encoding="utf-8")
            dump(snapshot / "validation.json", {"valid": not invalid, "cost_terrain_m": cost, "errors": ["forbidden_capsule"] if invalid else [],
                "xy_sha256": file_hash(snapshot / "path_xy.csv"), "surface_sha256": file_hash(snapshot / "path_surface.csv"),
                "synthetic_test_fixture": True})
            events.append(dict(event, independent_valid=not invalid, counters={}))
    trial = dict(entry, execution_status=status, planner_status="EXACT_SOLUTION" if successful else "NO_SOLUTION",
                 independent_valid=None if pending else (successful and not invalid), path_available=successful,
                 time_budget_s=30., best_cost_terrain_m=8. if successful else None,
                 first_solution_time_s=.5 if successful else None, length_xy_m=8. if successful else None,
                 independently_recomputed_cost_m=8. if successful else None,
                 actual_solve_wall_s=30.3 if successful else 8., worker_wall_s=30.5 if successful else 9.,
                 validation_wall_s=None if pending else .01, peak_rss_mb=20.,
                 termination_reason="TIME_BUDGET", counters={"motion_validity_calls": 10}, synthetic_test_fixture=True)
    if status == "CRASH":
        trial["actual_solve_wall_s"] = None
    dump(directory / "trial.json", trial)
    if not pending:
        dump(directory / "validated_events.json", events)
        dump(directory / "validation.json", {"valid": successful and not invalid, "cost_terrain_m": trial["best_cost_terrain_m"],
             "length_xy_m": trial["length_xy_m"], "synthetic_test_fixture": True})
    return directory


def raw_fixture(root):
    source = Path(root) / "raw"
    source.mkdir()
    config = load_config(ROOT / "configs/a3_vs_all_v2.yaml", "combined")
    planned = entries()
    dump(source / "resolved_config.json", config)
    dump(source / "manifest.json", {"schema_version": 3, "protocol_id": "a3_vs_all_v2", "profile": "combined",
         "experiment_id": "synthetic_cloud_analysis", "planned_trials": 1500, "trials": planned,
         "synthetic_test_fixture": True, "identity": {"synthetic_test_fixture": True}})
    # All five planners have a paired vector in each family; other slots are NOT_RUN.
    selected = [entry for entry in planned if entry["map_seed"] == MAP_SEEDS[0] and entry["planner_seed"] == PLANNER_SEEDS[0]]
    for entry in selected:
        write_trial(source, entry)
    inventory(source)
    return source, planned


def statistical_rows():
    rows = []
    for entry in entries():
        family_rank = FAMILIES.index(entry["family"])
        planner_rank = PLANNERS.index(entry["planner_id"])
        common = entry["map_seed"] / 100 + entry["planner_seed"] / 1000
        # Each family has a different contrast, constant across maps/seeds.
        cost = common + planner_rank * (1 + family_rank)
        first = .5 + planner_rank * .1
        rows.append(dict(entry, executed=True, success=True, outcome_known=True,
            validation_pending=False, cost_terrain_m_at_30s=cost, first_solution_time_s=first,
            t_first={"time_s": first, "event_observed": True, "reason": None}))
    return rows


class CloudAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="cloud_analysis_", dir=ROOT / "tests/fixtures")
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_design_rejects_budget_seed_hash_and_cardinality_drift(self):
        config = load_config(ROOT / "configs/a3_vs_all_v2.yaml", "combined")
        manifest = {"profile": "combined", "protocol_id": "a3_vs_all_v2", "planned_trials": 1500, "trials": entries()}
        validate_design(config, manifest)
        for key, value in (("planning_budget_s", 20.), ("planner_seeds", [94001]), ("maps_per_family", 10)):
            changed = copy.deepcopy(config)
            changed["profiles"]["combined"][key] = value
            with self.assertRaisesRegex(ValueError, "design mismatch"):
                validate_design(changed, manifest)
        changed = copy.deepcopy(manifest)
        changed["trials"].pop()
        with self.assertRaisesRegex(ValueError, "1500"):
            validate_design(config, changed)
        changed = copy.deepcopy(manifest)
        changed["trials"][1]["query_hash"] = "other"
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            validate_design(config, changed)

    def test_budget_cost_and_anytime_never_backfill_future_event(self):
        source, planned = raw_fixture(self.root)
        rows = read_rows(source, planned)
        row = rows[0]
        self.assertTrue(row["success"])
        self.assertEqual(row["cost_terrain_m_at_30s"], 10.)
        self.assertEqual(row["final_cost_terrain_m"], 8.)
        self.assertEqual(row["events_after_budget"], 1)
        points = [record for record in checkpoints(rows) if record["family"] == "ALL" and record["planner_id"] == "bitstar_a3"]
        at_early = next(record for record in points if record["elapsed_s"] == .25)
        self.assertEqual(at_early["validated_solution_count"], 0)
        self.assertIsNone(at_early["conditional_cost_terrain_m"]["median"])
        self.assertEqual(next(record for record in points if record["elapsed_s"] == 10.)["conditional_cost_terrain_m"]["median"], 12.)
        self.assertEqual(points[-1]["conditional_cost_terrain_m"]["median"], 10.)
        self.assertEqual(row["metrics"]["sample_proposals"], None)
        self.assertIn("sample_proposals", row["metric_unavailable_reasons"])

    def test_pending_unexecuted_timeout_and_crash_remain_distinct(self):
        planned = entries()[:5]
        source = self.root / "raw"
        source.mkdir()
        write_trial(source, planned[0], pending=True)
        write_trial(source, planned[1], status="TIMEOUT")
        write_trial(source, planned[2], status="CRASH")
        rows = read_rows(source, planned)
        counts = group_counts(rows)
        self.assertEqual((counts["planned"], counts["executed"], counts["validation_pending"], counts["unexecuted"]), (5, 3, 1, 2))
        self.assertEqual(counts["known_non_success"], 2)
        self.assertEqual(rows[1]["t_first"]["time_s"], 8.)
        self.assertFalse(rows[1]["t_first"]["event_observed"])
        self.assertIsNone(rows[2]["t_first"]["time_s"])
        self.assertEqual(rows[2]["t_first"]["status"], "TECHNICAL_MISSING")
        self.assertFalse(counts["success_rate_executed_is_final"])

    def test_rmst_known_value_and_unsupported_censor_tail(self):
        spec = fixture_spec()
        self.assertAlmostEqual(km_rmst(spec["weighted_km_example"])["rmst_s"], spec["weighted_km_expected_rmst_s"])
        early = km_rmst([{"time_s": 8., "event_observed": False}])
        self.assertIsNone(early["rmst_s"])
        self.assertEqual(early["reason"], "positive_survival_beyond_last_observed_endpoint")
        all_events = km_rmst([{"time_s": 1., "event_observed": True, "weight": 1.}, {"time_s": 3., "event_observed": True, "weight": 3.}])
        self.assertAlmostEqual(all_events["rmst_s"], 2.5)
        row = {"executed": True, "validation_pending": False, "success": False, "status": "TIMEOUT",
               "errors": [], "metrics": {"worker_wall_s": 35.}, "raw": {"planner_status": "NO_SOLUTION"}}
        self.assertIsNone(time_record(row)["time_s"])
        row["metrics"]["actual_solve_wall_s"] = 30.02
        row["raw"]["planner_status"] = "APPROXIMATE_SOLUTION"
        observed = time_record(row)
        self.assertEqual(observed["time_s"], 30.)
        self.assertFalse(observed["event_observed"])

    def test_bootstrap_fixed_strata_and_paired_vectors_are_reproducible(self):
        rows = statistical_rows()
        first = paired_bootstrap(rows, replicates=24)
        self.assertEqual(first, paired_bootstrap(rows, replicates=24))
        self.assertEqual(len(first), 6 * 4 * 4)
        overall = [record for record in first if record["family"] == "ALL" and record["metric"] == "cost_terrain_m_at_30s"]
        for rank, record in enumerate(overall, 1):
            self.assertAlmostEqual(record["estimate_a_minus_b"], -3. * rank)
            for limit in record["confidence_interval_95"]:
                self.assertAlmostEqual(limit, -3. * rank)
            self.assertEqual((record["eligible_pairs"], record["eligible_maps"], record["planned_pairs"]), (300, 100, 300))
        # All sampled first-solution pair differences must preserve planner pairing.
        for record in first:
            if record["metric"] in ("first_solution_time_s", "t_first_rmst_tau_30s"):
                expected = -.1 * PLANNERS.index(record["planner_b"])
                self.assertAlmostEqual(record["estimate_a_minus_b"], expected)

    def test_report_retains_all_planned_rows_and_immutable_raw(self):
        source, planned = raw_fixture(self.root)
        before = {path.relative_to(source).as_posix(): file_hash(path) for path in source.rglob("*") if path.is_file()}
        output = self.root / "report"
        summary = analyze(source, output, make_plots=False, bootstrap_replicates=8)
        self.assertEqual((summary["planned"], summary["executed"], summary["validated_success"], summary["unexecuted"]), (1500, 25, 25, 1475))
        self.assertEqual(summary["status"], "PARTIAL")
        records = read_json(output / "trial_metrics.json")
        self.assertEqual(len(records), 1500)
        self.assertFalse(summary["figures"])
        self.assertIn("SYNTHETIC TEST FIXTURE", (output / "REPORT.md").read_text(encoding="utf-8"))
        self.assertEqual(before, {path.relative_to(source).as_posix(): file_hash(path) for path in source.rglob("*") if path.is_file()})
        artifacts = read_json(output / "report_artifact_inventory.json")
        self.assertEqual(artifacts["inventory_sha256"], digest(artifacts["entries"]))

    def test_corrupt_snapshot_and_unfrozen_inventory_fail_closed(self):
        source, planned = raw_fixture(self.root)
        snapshot = source / planned[0]["trial_id"] / "snapshots/s0/path_xy.csv"
        snapshot.write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "size/hash mismatch"):
            analyze(source, self.root / "bad_report", make_plots=False, bootstrap_replicates=8)
        inventory(source)
        with self.assertRaisesRegex(ValueError, "corrupt validated event provenance"):
            read_rows(source, planned)

    def test_temporary_worker_evidence_is_inventory_protected(self):
        source, planned = raw_fixture(self.root)
        temporary = source / planned[0]["trial_id"] / "worker.partial.tmp"
        temporary.write_text("retained crash evidence", encoding="utf-8")
        inventory(source)
        analyze(source, self.root / "tmp_evidence_report", make_plots=False, bootstrap_replicates=4)
        temporary.write_text("tampered", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "size/hash mismatch"):
            analyze(source, self.root / "tampered_report", make_plots=False, bootstrap_replicates=4)

    def test_metadata_failure_does_not_claim_geometric_collision(self):
        source = self.root / "raw"
        source.mkdir()
        planned = entries()[:5]
        directory = write_trial(source, planned[0])
        (directory / "validated_events.json").unlink()
        rows = read_rows(source, planned)
        self.assertFalse(rows[0]["success"])
        self.assertFalse(rows[0]["geometric_invalid"])
        self.assertTrue(rows[0]["validation_failure_unclassified"])
        self.assertTrue(rows[0]["technical_failure"])

    def test_independently_invalid_path_is_not_success(self):
        source = self.root / "raw"
        source.mkdir()
        planned = entries()[:5]
        write_trial(source, planned[0], invalid=True)
        rows = read_rows(source, planned)
        self.assertFalse(rows[0]["success"])
        self.assertTrue(rows[0]["geometric_invalid"])
        self.assertIsNone(rows[0]["cost_terrain_m_at_30s"])
        self.assertEqual(group_counts(rows)["geometric_invalid"], 1)


if __name__ == "__main__":
    unittest.main()
