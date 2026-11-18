"""Labeled synthetic parser/statistics fixtures, confined to tests/fixtures."""
import copy
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tools"))
from a3_pilot_protocol import PROTOCOL_ID, PLANNERS, expected_config, plan_trials
from analyze_a3_pilot import (analyze, attainment, enrich_row, freeze_references,
                             paired_results, runtime_sessions, tradeoff)
from experiment_common import digest, dump, file_hash, read_json
from cloud_campaign import inventory


def seal(directory):
    records = inventory(directory, {"artifact_inventory.json"})
    dump(directory/"artifact_inventory.json", {"entries": records, "sha256": digest(records)})


def session_evidence(source, sid="session-a", cpu="Synthetic CPU A", system="Linux synthetic"):
    identity = {"compiler": "synthetic compiler", "worker_sha256": "f"*64,
                "python": "3.12.synthetic", "packages": {"synthetic": "fixture"}}
    dump(source/"runtime_identity.json", identity)
    environment = {"os": system, "cpu": cpu, "synthetic_test_fixture": True}
    session = {"session_id": sid, "stratum_id": digest({"platform": system, "cpu": cpu,
        "runtime_identity": identity}), "platform": system, "cpu": cpu,
        "environment_sha256": digest(environment), "runtime_identity_sha256": digest(identity)}
    dump(source/"sessions"/(sid+".json"), dict(session, environment=environment))
    return session, environment


def write_trial(source, entry, session, environment, events=None, status="COMPLETED", validation=True):
    events = [[.5, 11.], [1., 10.], [2.5, 9.]] if events is None else events
    directory = source/entry["trial_id"]
    directory.mkdir(parents=True, exist_ok=True)
    trial = dict(entry, execution_status=status, time_budget_s=2.,
        planner_status="EXACT_SOLUTION" if events else "TIMEOUT", path_available=bool(events),
        independent_valid=validation if events else None, best_cost_terrain_m=events[-1][1] if events else None,
        first_solution_time_s=.01,  # Deliberately wrong worker summary: snapshots are authoritative.
        validation_wall_s=.01, actual_solve_wall_s=2.05, load_s=1., preprocessing_s=2., setup_s=3.,
        collision_index_build_s=1.5, worker_wall_s=8.2, runtime_session_id=session["session_id"],
        runtime_stratum_id=session["stratum_id"], environment=environment, synthetic_test_fixture=True)
    dump(directory/"trial.json", trial)
    checked = []
    for index, (elapsed, cost) in enumerate(events):
        snapshot_id = f"s{index}"
        snapshot = directory/"snapshots"/snapshot_id
        snapshot.mkdir(parents=True, exist_ok=True)
        (snapshot/"path_xy.csv").write_text("x_m,y_m\n0,0\n8,0\n", encoding="utf-8")
        (snapshot/"path_surface.csv").write_text("x_m,y_m,z_m\n0,0,0\n8,0,0\n", encoding="utf-8")
        dump(snapshot/"validation.json", {"valid": validation, "cost_terrain_m": cost,
            "xy_sha256": file_hash(snapshot/"path_xy.csv"),
            "surface_sha256": file_hash(snapshot/"path_surface.csv"), "synthetic_test_fixture": True})
        checked.append({"snapshot_id": snapshot_id, "elapsed_s": elapsed, "best_cost_terrain_m": cost,
                        "independent_valid": validation})
    dump(directory/"validated_events.json", checked)
    dump(directory/"validation.json", {"valid": bool(events) and validation,
        "exact_goal_reached": bool(events) and validation, "synthetic_test_fixture": True})
    seal(directory)


def fixture(directory):
    source = directory/"raw"
    source.mkdir()
    config = expected_config()
    maps = {"profile": "smoke", "config_hash": digest(config), "protocol_id": PROTOCOL_ID,
            "seed_replacements": [], "seed_replacement_policy": "forbidden", "maps": []}
    for family in config["profiles"]["smoke"]["families"]:
        map_id = family+"_101"
        map_hash = digest({"synthetic_map": map_id})
        points = [[0., 0.], [8., 0.]]
        query = {"query_id": "q0", "start": points[0], "goal": points[1],
            "feasibility": "feasible_witness_validated", "reference": {
                "reference_type": "validated_witness", "cost_terrain_m": 8., "path_xy": points,
                "path_sha256": digest(points), "map_sha256": map_hash,
                "validation": {"valid": True, "cost_terrain_m": 8.},
                "provenance": "SYNTHETIC TEST FIXTURE ONLY"}}
        query["query_hash"] = digest({"start": points[0], "goal": points[1], "map_hash": map_hash, "model": config["model"]})
        maps["maps"].append({"family": family, "map_seed": 101, "map_id": map_id,
                             "path": map_id+".t25d", "sha256": map_hash, "queries": [query]})
    entries = plan_trials(config, "smoke", maps)
    session, environment = session_evidence(source)
    for entry in entries:
        write_trial(source, entry, session, environment)
    manifest = {"experiment_id": "synthetic_a3_analysis", "profile": "smoke", "planned_trials": 52,
        "protocol_id": PROTOCOL_ID, "identity": {"config_hash": digest(config),
        "plan_sha256": digest(entries), "map_manifest_sha256": digest(maps),
        "runtime_identity_sha256": digest(read_json(source/"runtime_identity.json"))},
        "trials": entries, "status": "COMPLETE_VALIDATED", "synthetic_test_fixture": True,
        "runtime_sessions": [session]}
    dump(source/"resolved_config.json", config)
    dump(source/"map_manifest.json", maps)
    dump(source/"manifest.json", manifest)
    return source, manifest, session, environment


class PilotAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="synthetic_a3_analysis_", dir=ROOT/"tests/fixtures")
        self.directory = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def report(self, source, name="report"):
        return analyze(source, self.directory/name, make_plots=False, _test_bootstrap_replicates=30)

    def test_two_clocks_deadline_explicit_pairs_and_all_artifacts(self):
        source, manifest, _, _ = fixture(self.directory)
        summary = self.report(source)
        self.assertEqual(summary["status"], "COMPLETE_VALIDATED")
        self.assertEqual((summary["executed"], summary["validated_success"]), (52, 52))
        self.assertEqual(summary["primary_metrics"], ["cost_terrain_m_at_budget", "first_solution_time_s"])
        self.assertEqual(len(summary["comparison_pairs"]), 19)
        self.assertFalse(any("abitstar_optimized" in pair for pair in summary["comparison_pairs"]))
        records = read_json(self.directory/"report/trial_metrics.json")
        self.assertTrue(all(row["cost_terrain_m_at_budget"] == 10. for row in records))
        self.assertTrue(all(row["first_solution_time_s"] == .5 for row in records))
        self.assertTrue(all(row["first_solution_including_preparation_s"] == 6.5 for row in records))
        self.assertTrue(all(row["events_after_budget"] == 1 for row in records))
        for path in ("summary.json", "paired_bootstrap.json", "paired_bootstrap.csv", "trials.csv", "REPORT.md",
                     "runtime_sessions.json", "anytime_curves.json", "reference_targets.json", "analysis_input_inventory.json"):
            self.assertTrue((self.directory/"report"/path).is_file())
        paired = read_json(self.directory/"report/paired_bootstrap.json")
        self.assertEqual({(row["candidate"], row["reference"]) for row in paired},
                         {tuple(pair) for pair in summary["comparison_pairs"]})
        self.assertTrue(any(row["family"] == "ALL" and row["ci"] is not None for row in paired))
        self.assertEqual({row["query_id"] for row in paired}, {"ALL", "q0"})
        self.assertTrue(all(row["ci"] is None for row in paired if row["family"] != "ALL"))

    def test_failure_denominator_missing_trials_and_censoring(self):
        source, manifest, session, env = fixture(self.directory)
        entries = manifest["trials"]
        write_trial(source, entries[0], session, env, [], "TIMEOUT")
        write_trial(source, entries[1], session, env, [[2.001, 8.]])
        write_trial(source, entries[2], session, env, [], "CRASH")
        crash_path = source/entries[2]["trial_id"]/"trial.json"
        trial = read_json(crash_path)
        trial["actual_solve_wall_s"] = None
        dump(crash_path, trial)
        seal(crash_path.parent)
        # A planned but unstarted trial has no record; old files here are ignored.
        (source/entries[3]["trial_id"]/"trial.json").unlink()
        (source/entries[3]["trial_id"]/"artifact_inventory.json").unlink()
        summary = self.report(source)
        self.assertEqual((summary["executed"], summary["validated_success"], summary["not_run"]), (51, 48, 1))
        self.assertEqual(summary["success_rate_executed"], 48/51)
        self.assertEqual(summary["status"], "STOPPED_FAILURE")
        records = {row["trial_id"]: row for row in read_json(self.directory/"report/trial_metrics.json")}
        for entry in entries[:3]:
            row = records[entry["trial_id"]]
            self.assertIsNone(row["first_solution_time_s"])
            self.assertIsNone(row["cost_terrain_m_at_budget"])
            self.assertFalse(row["success_attainment"]["event_observed"])
        self.assertEqual(records[entries[0]["trial_id"]]["success_attainment"]["observation_end_s"], 2.)
        self.assertIsNone(records[entries[2]["trial_id"]]["success_attainment"]["observation_end_s"])
        curve = next(row for row in read_json(self.directory/"report/anytime_curves.json")
                     if row["planner_id"] == entries[0]["planner_id"] and row["family"] == entries[0]["family"])
        self.assertEqual(curve["points"][-1]["success_attainment"], 0.)
        self.assertIsNone(curve["points"][-1]["median_cost_terrain_m"])

    def test_missing_setup_phase_stays_null(self):
        source, manifest, _, _ = fixture(self.directory)
        path = source/manifest["trials"][0]["trial_id"]/"trial.json"
        record = read_json(path)
        del record["setup_s"]
        dump(path, record)
        seal(path.parent)
        self.report(source)
        rows = read_json(self.directory/"report/trial_metrics.json")
        self.assertIsNone(rows[0]["first_solution_including_preparation_s"])
        self.assertEqual(rows[0]["first_solution_time_s"], .5)

    def test_distinct_cpu_and_os_never_pool_timing_or_pairs(self):
        source, manifest, session, env = fixture(self.directory)
        session_b, env_b = session_evidence(source, "session-b", "Synthetic CPU B")
        session_c, env_c = session_evidence(source, "session-c", "Synthetic CPU A", "Windows synthetic")
        for entry in manifest["trials"]:
            if entry["planner_id"] == "a3_cache":
                write_trial(source, entry, session_b, env_b)
            if entry["planner_id"] == "a3_scalar":
                write_trial(source, entry, session_c, env_c)
        manifest["runtime_sessions"] += [session_b, session_c]
        dump(source/"manifest.json", manifest)
        summary = self.report(source)
        self.assertEqual(len({group["runtime_stratum_id"] for group in summary["runtime_groups"]}), 3)
        pairs = read_json(self.directory/"report/paired_bootstrap.json")
        unpaired = [row for row in pairs if row["candidate"] == "a3_cache" and row["reference"] == "bitstar_a3"]
        self.assertTrue(unpaired)
        self.assertTrue(all(row["estimate"] is None for row in unpaired))
        self.assertTrue(all(row["units"]["paired_seed_slots"] == 0 for row in unpaired))
        audit = read_json(self.directory/"report/pair_coverage.json")
        record = next(row for row in audit if row["family"] == "ALL" and row["candidate"] == "a3_cache"
                      and row["reference"] == "bitstar_a3")
        self.assertEqual(record["cross_runtime"], 4)
        self.assertEqual(record["same_runtime"], 0)

    def test_missing_runtime_is_not_a_common_machine(self):
        source, manifest, _, _ = fixture(self.directory)
        path = source/manifest["trials"][0]["trial_id"]/"trial.json"
        trial = read_json(path)
        del trial["runtime_session_id"]
        dump(path, trial)
        seal(path.parent)
        summary = self.report(source)
        self.assertEqual(summary["status"], "FAILED_INTEGRITY")
        self.assertEqual(len(summary["runtime_errors"]), 1)

    def test_session_environment_hash_and_trial_binding(self):
        source, manifest, session, env = fixture(self.directory)
        path = source/"sessions"/(session["session_id"]+".json")
        record = read_json(path)
        record["environment"]["cpu"] = "changed CPU"
        dump(path, record)
        with self.assertRaisesRegex(ValueError, "evidence mismatch"):
            self.report(source)
        dump(path, dict(session, environment=env))
        trial_path = source/manifest["trials"][0]["trial_id"]/"trial.json"
        trial = read_json(trial_path)
        trial["environment"]["cpu"] = "wrong trial CPU"
        dump(trial_path, trial)
        seal(trial_path.parent)
        summary = self.report(source)
        self.assertEqual(summary["status"], "FAILED_INTEGRITY")

    def test_bad_snapshot_hash_never_becomes_valid_quality(self):
        source, manifest, _, _ = fixture(self.directory)
        path = source/manifest["trials"][0]["trial_id"]/"snapshots/s0/path_xy.csv"
        path.write_text("changed after validation", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "inventory size/hash mismatch"):
            self.report(source)
        # Even after a new artifact seal, stale snapshot-validation hashes fail.
        seal(source/manifest["trials"][0]["trial_id"])
        summary = self.report(source)
        self.assertEqual(summary["status"], "FAILED_CORRECTNESS")
        self.assertEqual(summary["validated_success"], 51)

    def test_frozen_inline_witness_cannot_follow_winner(self):
        source, manifest, _, _ = fixture(self.directory)
        self.report(source)
        for entry in manifest["trials"]:
            if entry["map_id"] == manifest["trials"][0]["map_id"]:
                entry["reference"]["cost_terrain_m"] = 7.
                entry["reference"]["validation"]["cost_terrain_m"] = 7.
        dump(source/"manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "frozen witness changed"):
            freeze_references(manifest, self.directory/"report", True)

    def test_unequal_failure_coverage_prevents_frontier_claim(self):
        rows = []
        for planner, successful, first, cost in (("A", [True, True], 2., 10.), ("B", [True, False], 1., 9.)):
            for index, success in enumerate(successful):
                rows.append({"planner_id": planner, "map_id": "m", "query_id": "q", "planner_seed": index,
                    "executed": True, "runtime_valid": True, "success": success,
                    "first_solution_time_s": first if success else None,
                    "cost_terrain_m": cost if success else None})
        result = tradeoff(rows, ["A", "B"])
        self.assertEqual(result["common_success_slots"], 1)
        self.assertEqual(result["frontier"], [])
        self.assertEqual(result["status"], "UNAVAILABLE_UNEQUAL_SUCCESS_OR_EXECUTION_COVERAGE")
        rows[-1].update(success=True, first_solution_time_s=1., cost_terrain_m=9.)
        result = tradeoff(rows, ["A", "B"])
        self.assertEqual(result["frontier"], ["B"])

    def test_hierarchy_is_paired_and_missing_success_never_zero(self):
        rows = []
        for map_index in range(2):
            for seed in range(3):
                for planner, cost in (("A", 10.), ("B", 12.)):
                    rows.append({"planner_id": planner, "map_id": str(map_index), "query_id": "q", "family": "f",
                        "planner_seed": seed, "runtime_stratum_id": "same", "runtime_valid": True,
                        "executed": True, "success": not (map_index == seed == 0 and planner == "A"),
                        "cost_terrain_m": cost, "first_solution_time_s": cost/10})
        pairs = paired_results(rows, [("A", "B")], 30, 73130, .95)
        cost = next(row for row in pairs if row["family"] == "f" and row["metric"] == "cost_terrain_m")
        self.assertEqual(cost["estimate"], -2.)
        self.assertEqual(cost["ci"], [-2., -2.])
        self.assertEqual(cost["units"]["paired_seed_slots"], 5)
        self.assertEqual(cost["units"]["unavailable_complete_pairs"], 1)

    def test_protocol_mutation_and_unsafe_paths_rejected(self):
        source, manifest, _, _ = fixture(self.directory)
        config_path = source/"resolved_config.json"
        config = read_json(config_path)
        config["a3_pilot_benchmark"]["comparison_pairs"].reverse()
        dump(config_path, config)
        with self.assertRaisesRegex(ValueError, "comparison pairs"):
            self.report(source)
        dump(config_path, expected_config())
        manifest["trials"][0]["trial_id"] = "../escape"
        dump(source/"manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "escapes"):
            self.report(source)

    def test_actual_campaign_state_and_missing_artifact_seal(self):
        source, manifest, _, _ = fixture(self.directory)
        del manifest["status"]
        dump(source/"manifest.json", manifest)
        dump(source/"campaign_state.json", {"protocol_id": PROTOCOL_ID, "status": "COMPLETE_VALIDATED"})
        summary = self.report(source)
        self.assertEqual(summary["status"], "COMPLETE_VALIDATED")
        (source/manifest["trials"][0]["trial_id"]/"artifact_inventory.json").unlink()
        summary = self.report(source)
        self.assertEqual(summary["status"], "PARTIAL")
        self.assertEqual(len(summary["artifact_inventories_missing"]), 1)

    def test_real_plotting_of_labeled_synthetic_fixture(self):
        source, _, _, _ = fixture(self.directory)
        # Fresh Python is essential: an earlier import would conceal the
        # notebook-backend failure observed in the isolated Kaggle controller.
        env = dict(os.environ, MPLBACKEND="synthetic_invalid_notebook_backend")
        code = ("import sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);"
                "from analyze_a3_pilot import analyze;"
                "analyze(Path(sys.argv[2]),Path(sys.argv[3]),make_plots=True,_test_bootstrap_replicates=10);"
                "import matplotlib;assert matplotlib.get_backend().lower()=='agg'")
        completed = subprocess.run([sys.executable, "-c", code, str(ROOT/"tools"),
                                    str(source), str(self.directory/"plots")],
                                   env=env, capture_output=True, text=True, timeout=120)
        self.assertEqual(completed.returncode, 0, completed.stdout+completed.stderr)
        plots = list((self.directory/"plots").glob("*.png"))
        self.assertEqual(len(plots), 3)
        for path in plots:
            self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertGreater(path.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
