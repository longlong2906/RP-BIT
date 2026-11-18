"""Direct production-reader regressions; fixtures are never performance data."""
import argparse
import copy
import csv
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
from analyze_results import COUNTERS
from experiment_common import ROOT, code_hash, digest, dump, file_hash, read_json
from g5_analysis import analyze, prepare_observations, read_g5_trial
from independent_verifier import Oracle
from terrain_io import write_map
from validate_results import validate_trial


FIXTURES = ROOT/"tests/fixtures/g5_taxonomy"
RUNTIME_ROOT = ROOT/"build-g5-taxonomy-runtime"
IDENTITY = {"code_sha256": "a"*64, "config_hash": "b"*64}
FIELDS = ("path_present", "geometry_valid", "exact_goal_reached",
          "exact_incumbent_present", "planner_status", "execution_status",
          "validated_success", "correctness_error")


def write_csv(path, headings, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headings)
        writer.writerows(rows)


def flat_path(directory, start, endpoint):
    """Analytic horizontal path and mesh breaks, without calling the oracle."""
    assert start[1] == endpoint[1] == 5.
    xy = [start] if start == endpoint else [start, endpoint]
    xs = [start[0], *[float(x) for x in range(math.floor(start[0])+1, math.ceil(endpoint[0]))], endpoint[0]]
    surface = [[x, 5., 0.] for x in dict.fromkeys(xs)]
    write_csv(directory/"path_xy.csv", ["x_m", "y_m"], xy)
    write_csv(directory/"path_surface.csv", ["x_m", "y_m", "z_m"], surface)
    return math.dist(start, endpoint)


def materialize(source, case_name, *, trial_id=None):
    registry = read_json(FIXTURES/"cases.json")
    assert registry["synthetic_test_fixture"] is True
    case = copy.deepcopy(registry["cases"][case_name])
    trial_id = trial_id or case_name
    directory = source/trial_id
    directory.mkdir(parents=True)
    map_path = directory/"map.t25d"
    write_map(map_path, [[0.]*11 for _ in range(11)], radius=.1, margin=0., slope=70.,
              rectangles=case.get("rectangles", []), metadata={"synthetic_test_fixture": True})
    entry = {"trial_id": trial_id, "family": "SYNTHETIC_CORRECTNESS_FIXTURE",
             "map_id": "synthetic_flat", "map_path": str(map_path), "map_hash": file_hash(map_path),
             "query_id": case_name, "planner_id": "rrtstar", "planner_seed": 771}
    request = {"map_path": str(map_path), "planner_id": "rrtstar", "planner_seed": 771,
               "start": case["start"], "goal": case["goal"], "planning_budget_s": registry["budget_s"],
               "synthetic_test_fixture": True}
    dump(directory/"request.json", request)
    has_path = "endpoint" in case
    cost = flat_path(directory, case["start"], case["endpoint"]) if has_path else None
    admin = case.get("administrative_interruption", False)
    origin = case.get("event_origin", "ompl_intermediate_solution_callback")
    trial = dict(entry, **IDENTITY, synthetic_test_fixture=True,
                 execution_status=case["execution_status"], planner_status=case["planner_status"],
                 path_available=has_path, independent_valid=None, success=False,
                 time_budget_s=registry["budget_s"], best_cost_terrain_m=cost,
                 actual_solve_wall_s=None if admin else 1.2 if case_name == "late_exact_incumbent" else 1.,
                 event_origin=origin, first_solution_time_s=case.get("event_time_s"),
                 termination_reason="INTERRUPTED_BY_CORRECTNESS_STOP_GATE" if admin else
                                    "ANALYTIC_IDENTITY_QUERY" if origin == "analytic_identity_query" else "TIME_BUDGET",
                 counters={} if admin else {"full_surface_cost_calls": 17, "motion_validity_calls": 19,
                                           "height_queries": 23, "triangle_visits": 29})
    if admin:
        trial.update(administrative_interruption=True, actual_solve_wall_s_unavailable_reason="administrative_stop_before_worker_serialization",
                     unavailable_counter_reason="administrative_stop_before_worker_serialization")
    else:
        dump(directory/"worker.json", trial)
    events = []
    if "event_time_s" in case:
        snapshot = directory/"snapshots/snapshot_0"
        snapshot.mkdir(parents=True)
        for name in ("path_xy.csv", "path_surface.csv"):
            shutil.copyfile(directory/name, snapshot/name)
        events.append({"event_type": "improvement", "elapsed_s": case["event_time_s"], "feasible": True,
                       "best_cost_terrain_m": cost, "snapshot_id": "snapshot_0", "event_origin": origin,
                       "counters": {"full_surface_cost_calls": 7, "motion_validity_calls": 9,
                                    "height_queries": 13, "triangle_visits": 15}})
    (directory/"progress.jsonl").write_text("".join(json.dumps(e)+"\n" for e in events), encoding="utf-8")
    dump(directory/"trial.json", trial)
    validate_trial(directory, Oracle(map_path))
    return entry


def retained_approximate(source, trial_id="retained_approximate"):
    provenance = read_json(FIXTURES/"retained_provenance.json")
    retained = FIXTURES/"retained_approximate"
    for name, expected in provenance["files"].items():
        assert file_hash(retained/name) == expected, name
    assert provenance["synthetic_test_fixture"] is False
    directory = source/trial_id
    shutil.copytree(retained, directory)
    request = read_json(directory/"request.json")
    request["map_path"] = str(directory/"map.t25d")
    dump(directory/"request.json", request)
    worker = read_json(directory/"worker.json")
    assert worker["planner_status"] == "APPROXIMATE_SOLUTION"
    assert worker["execution_status"] == "TIMEOUT"
    assert file_hash(directory/"worker.json") == "e50533a65199c8c1293e65c011b88a297768b55d5ccc185f624098faf336896d"
    entry = {"trial_id": trial_id, "family": "RETAINED_REAL_CTEST_FIXTURE",
             "map_id": "CTEST_FIXTURE_NOT_G5", "map_path": request["map_path"],
             "map_hash": file_hash(directory/"map.t25d"), "query_id": "blocked_goal_fixture",
             "planner_id": "rrtstar", "planner_seed": 771}
    trial = dict(worker)
    trial.update(entry)
    trial.update(IDENTITY)
    dump(directory/"trial.json", trial)
    checked = validate_trial(directory, Oracle(request["map_path"]))
    assert checked["independent_valid"] is True and checked["success"] is False
    return entry


def lock_artifacts(directory):
    names = ["request.json", "worker.json", "progress.jsonl", "path_xy.csv", "path_surface.csv"]
    paths = [directory/name for name in names if (directory/name).is_file()]
    paths += sorted((directory/"snapshots").glob("*/*.csv"))
    dump(directory/"artifact_hashes.json", {"schema_version": 1,
         "files": {p.relative_to(directory).as_posix(): file_hash(p) for p in paths}})


def read_fixture(source, entry, *, schema2=False):
    identity = dict(IDENTITY)
    if schema2:
        identity["analyzer_schema_version"] = 2
    budget = read_json(source/entry["trial_id"]/"request.json")["planning_budget_s"]
    return read_g5_trial(source, entry, budget, 1e-8, 1e-8, identity)


class G5TaxonomyTests(unittest.TestCase):
    def setUp(self):
        RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="run_", dir=RUNTIME_ROOT)
        self.source = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def case(self, name):
        entry = materialize(self.source, name)
        return entry, self.source/entry["trial_id"]

    def check(self, row, **expected):
        for field in FIELDS:
            self.assertIn(field, row)
        for key, value in expected.items():
            self.assertEqual(row[key], value, (key, row.get("errors")))
        self.assertEqual(row["success"], row["validated_success"])
        self.assertEqual(bool(row["errors"]), row["correctness_error"])

    def assert_correctness_failure(self, row):
        self.check(row, validated_success=False, correctness_error=True)
        self.assertIsNone(row["cost_terrain_m"])

    def test_A_exact_solution_valid_snapshot(self):
        entry, _ = self.case("exact")
        row = read_fixture(self.source, entry)
        self.check(row, path_present=True, geometry_valid=True, exact_goal_reached=True,
                   exact_incumbent_present=True, planner_status="EXACT_SOLUTION",
                   execution_status="COMPLETED", validated_success=True, correctness_error=False)
        self.assertEqual(row["cost_terrain_m"], 8.)

    def test_B_retained_real_ctest_approximate_hits_failed_v1_branch(self):
        entry = retained_approximate(self.source)
        directory = self.source/entry["trial_id"]
        before = {p.relative_to(directory).as_posix(): file_hash(p) for p in directory.rglob("*") if p.is_file()}
        row = read_fixture(self.source, entry)
        self.check(row, path_present=True, geometry_valid=True, exact_goal_reached=False,
                   exact_incumbent_present=False, planner_status="APPROXIMATE_SOLUTION",
                   execution_status="TIMEOUT", validated_success=False, correctness_error=False)
        self.assertEqual(row["events"], [])
        self.assertEqual(before, {p.relative_to(directory).as_posix(): file_hash(p) for p in directory.rglob("*") if p.is_file()})

    def test_B_synthetic_valid_approximate_is_normal_non_success(self):
        entry, _ = self.case("approximate")
        self.check(read_fixture(self.source, entry), path_present=True, geometry_valid=True,
                   exact_goal_reached=False, exact_incumbent_present=False,
                   validated_success=False, correctness_error=False)

    def test_C_approximate_needs_no_exact_snapshot_or_events_file(self):
        entry, directory = self.case("approximate")
        (directory/"validated_events.json").unlink()
        self.assertFalse((directory/"snapshots").exists())
        self.check(read_fixture(self.source, entry), exact_incumbent_present=False,
                   validated_success=False, correctness_error=False)

    def test_D_invalid_approximate_without_validity_claim(self):
        entry, directory = self.case("invalid_approximate")
        self.assertFalse(read_json(directory/"validation.json")["valid"])
        self.check(read_fixture(self.source, entry), path_present=True, geometry_valid=False,
                   validated_success=False, correctness_error=False)

    def test_D_invalid_approximate_claimed_valid_is_contradiction(self):
        entry, directory = self.case("invalid_approximate")
        trial = read_json(directory/"trial.json")
        trial["independent_valid"] = True
        dump(directory/"trial.json", trial)
        row = read_fixture(self.source, entry)
        self.assert_correctness_failure(row)
        self.assertIs(row["geometry_valid"], False)

    def test_E_timeout_without_path_is_not_correctness_error(self):
        entry, _ = self.case("timeout")
        row = read_fixture(self.source, entry)
        self.check(row, path_present=False, exact_goal_reached=False, exact_incumbent_present=False,
                   validated_success=False, correctness_error=False, execution_status="TIMEOUT")
        self.assertIsNone(row["cost_terrain_m"])

    def test_F_timeout_keeps_exact_event_without_backfill(self):
        entry, _ = self.case("timeout_exact_incumbent")
        row = read_fixture(self.source, entry)
        self.check(row, geometry_valid=True, exact_goal_reached=True, exact_incumbent_present=True,
                   validated_success=True, correctness_error=False, execution_status="TIMEOUT")
        original_events = copy.deepcopy(row["events"])
        prepare_observations(row, {"status": "FROZEN_VALIDATED_TARGET", "cost_terrain_m": 8.}, [.399, .4, 1., 2.])
        self.assertEqual([p["cost_terrain_m"] for p in row["checkpoint_observations"]], [None, 8., 8., 8.])
        self.assertEqual(row["frozen_target"]["event_time_s"], .4)
        self.assertEqual(row["frozen_target"]["counters_at_attainment"]["height_queries"], 13)
        self.assertEqual(row["events"], original_events)

    def test_F_event_after_budget_cannot_create_success(self):
        entry, _ = self.case("late_exact_incumbent")
        row = read_fixture(self.source, entry)
        self.check(row, validated_success=False, correctness_error=False)
        self.assertEqual(row["events_after_budget"], 1)
        prepare_observations(row, {}, [.5, 1., 2.])
        self.assertTrue(all(p["cost_terrain_m"] is None for p in row["checkpoint_observations"]))

    def test_F_timeout_snapshot_only_preserves_real_incumbent(self):
        entry, directory = self.case("timeout_exact_incumbent")
        trial = read_json(directory/"trial.json")
        trial.update(planner_status="UNKNOWN", path_available=False, success=False,
                     independent_valid=None, best_cost_terrain_m=None)
        dump(directory/"trial.json", trial)
        for name in ("path_xy.csv", "path_surface.csv"):
            (directory/name).unlink()
        row = read_fixture(self.source, entry)
        self.check(row, path_present=True, geometry_valid=True, exact_goal_reached=True,
                   exact_incumbent_present=True, validated_success=True, correctness_error=False)
        self.assertFalse(row["final_path_present"])
        self.assertEqual(row["first_solution_time_s"], .4)
        self.assertEqual(row["cost_terrain_m"], 8.)

    def test_raw_improvement_cannot_disappear_from_validation(self):
        entry, directory = self.case("approximate")
        event = {"event_type": "improvement", "elapsed_s": .4, "feasible": True,
                 "best_cost_terrain_m": 8., "snapshot_id": "undeclared_exact_snapshot"}
        (directory/"progress.jsonl").write_text(json.dumps(event)+"\n", encoding="utf-8")
        self.assert_correctness_failure(read_fixture(self.source, entry))

    def test_G_administrative_interruption_unknown_is_null_with_reason(self):
        entry, _ = self.case("administrative_interruption")
        row = read_fixture(self.source, entry)
        self.check(row, validated_success=False, correctness_error=False, execution_status="CRASH")
        self.assertEqual(row["status"], "ADMINISTRATIVE_INTERRUPTION")
        self.assertTrue(row["administrative_interruption"])
        self.assertFalse(row["native_ompl_crash"])
        for name in ("actual_solve_wall_s", *COUNTERS):
            self.assertIsNone(row["metrics"][name], name)
            self.assertTrue(row["metric_unavailable_reasons"][name], name)
        prepare_observations(row, {}, [0., 1.])
        observation = row["success_attainment"]
        self.assertIsNone(observation["observation_end_s"])
        self.assertTrue(observation["observation_end_unavailable_reason"])

    def test_H_identity_and_near_goal_preserve_actual_evidence(self):
        for name, expected_zero in (("identity", True), ("near_goal", False)):
            with self.subTest(name=name):
                entry, _ = self.case(name)
                row = read_fixture(self.source, entry)
                self.check(row, geometry_valid=True, exact_goal_reached=True, exact_incumbent_present=True,
                           validated_success=True, correctness_error=False)
                self.assertEqual(row["cost_terrain_m"] == 0., expected_zero)
                self.assertEqual(row["raw"]["event_origin"] == "analytic_identity_query", expected_zero)

    def test_exact_wrong_endpoint_is_correctness_failure(self):
        entry, _ = self.case("wrong_exact_endpoint")
        row = read_fixture(self.source, entry)
        self.assert_correctness_failure(row)
        self.assertTrue(row["geometry_valid"])
        self.assertFalse(row["exact_goal_reached"])

    def test_exact_missing_snapshot_is_correctness_failure(self):
        entry, directory = self.case("exact")
        (directory/"snapshots/snapshot_0/validation.json").unlink()
        self.assert_correctness_failure(read_fixture(self.source, entry))

    def test_I_missing_and_corrupt_snapshot_checksums_fail(self):
        for mode in ("missing", "corrupt"):
            with self.subTest(mode=mode):
                entry = materialize(self.source, "exact", trial_id=mode)
                path = self.source/mode/"snapshots/snapshot_0/validation.json"
                validation = read_json(path)
                if mode == "missing":
                    validation.pop("xy_sha256")
                else:
                    validation["xy_sha256"] = "0"*64
                dump(path, validation)
                self.assert_correctness_failure(read_fixture(self.source, entry))

    def test_I_final_path_checksum_claims_fail_when_partial_or_corrupt(self):
        for mode in ("partial", "corrupt"):
            with self.subTest(mode=mode):
                entry = materialize(self.source, "approximate", trial_id=mode)
                directory = self.source/mode
                validation = read_json(directory/"validation.json")
                validation["xy_sha256"] = file_hash(directory/"path_xy.csv")
                if mode == "corrupt":
                    validation["surface_sha256"] = "0"*64
                dump(directory/"validation.json", validation)
                self.assert_correctness_failure(read_fixture(self.source, entry))

    def test_I_provenance_mismatches_fail(self):
        for key in ("code_sha256", "config_hash", "map_hash", "planner_seed"):
            with self.subTest(key=key):
                entry = materialize(self.source, "exact", trial_id=key)
                path = self.source/key/"trial.json"
                trial = read_json(path)
                trial[key] = 999 if key == "planner_seed" else "0"*64
                dump(path, trial)
                self.assert_correctness_failure(read_fixture(self.source, entry))

    def test_I_schema2_requires_raw_checksums(self):
        for mode in ("valid", "missing_sidecar", "missing_entry", "corrupt_hash"):
            with self.subTest(mode=mode):
                entry = materialize(self.source, "exact", trial_id=mode)
                directory = self.source/mode
                if mode != "missing_sidecar":
                    lock_artifacts(directory)
                    path = directory/"artifact_hashes.json"
                    hashes = read_json(path)
                    if mode == "missing_entry":
                        hashes["files"].pop("request.json")
                    elif mode == "corrupt_hash":
                        hashes["files"]["request.json"] = "0"*64
                    dump(path, hashes)
                row = read_fixture(self.source, entry, schema2=True)
                if mode == "valid":
                    self.check(row, validated_success=True, correctness_error=False)
                else:
                    self.assert_correctness_failure(row)

    def test_real_v2_harness_binding_is_accepted_and_cannot_be_replaced(self):
        from g5_v2_run import bind_artifacts
        entry, directory = self.case("exact")
        bind_artifacts(directory)
        self.check(read_fixture(self.source, entry, schema2=True),
                   validated_success=True, correctness_error=False)
        before = file_hash(directory/"artifact_hashes.json")
        with self.assertRaisesRegex(ValueError, "never replace"):
            bind_artifacts(directory)
        self.assertEqual(file_hash(directory/"artifact_hashes.json"), before)
        self.assertEqual(file_hash(directory/"trial_before_validation.json"), file_hash(directory/"trial.json"))

    def test_v2_harness_detects_raw_trial_change_after_validation(self):
        from g5_v2_run import bind_artifacts
        entry, directory = self.case("exact")
        bind_artifacts(directory)
        identity = dict(IDENTITY, analyzer_schema_version=2, require_raw_trial_backup=True)
        self.assertFalse(read_g5_trial(self.source, entry, 1., 1e-8, 1e-8, identity)["correctness_error"])
        trial = read_json(directory/"trial.json")
        trial["planner_status"] = "APPROXIMATE_SOLUTION"
        dump(directory/"trial.json", trial)
        row = read_g5_trial(self.source, entry, 1., 1e-8, 1e-8, identity)
        self.assert_correctness_failure(row)
        self.assertIn("raw_trial_field_changed_after_validation:planner_status", row["errors"])

    def test_final_cost_or_surface_tampering_is_correctness_failure(self):
        for mode in ("cost", "surface", "malformed_path"):
            with self.subTest(mode=mode):
                entry = materialize(self.source, "approximate", trial_id=mode)
                directory = self.source/mode
                if mode == "cost":
                    trial = read_json(directory/"trial.json")
                    trial["best_cost_terrain_m"] = 100.
                    dump(directory/"trial.json", trial)
                elif mode == "surface":
                    (directory/"path_surface.csv").write_text("x_m,y_m,z_m\n1,5,0\n4,5,1\n", encoding="utf-8")
                else:
                    (directory/"path_xy.csv").write_text("x_m,y_m\nnot,a_number\n", encoding="utf-8")
                self.assert_correctness_failure(read_fixture(self.source, entry))

    def test_malformed_trial_and_events_are_explicit_correctness_failure(self):
        for name in ("trial.json", "validated_events.json"):
            with self.subTest(name=name):
                trial_id = name.replace(".", "_")
                entry = materialize(self.source, "exact", trial_id=trial_id)
                (self.source/trial_id/name).write_text("{broken JSON", encoding="utf-8")
                self.assert_correctness_failure(read_fixture(self.source, entry))

    def test_primary_analyze_refuses_historical_v1_before_output(self):
        dump(self.source/"manifest.json", {"experiment_id": "full_g5_v1", "synthetic_test_fixture": True})
        output = self.source/"forbidden_report"
        with self.assertRaisesRegex(ValueError, "immutable historical data"):
            analyze(self.source, output)
        self.assertFalse(output.exists())

    def test_primary_inference_refuses_partial_blocked_and_unvalidated_v2(self):
        from unittest.mock import patch
        source_hash = code_hash()
        for mode in ("partial", "blocked", "unvalidated"):
            with self.subTest(mode=mode):
                source = self.source/mode
                source.mkdir()
                config = {"synthetic_test_fixture": True}
                dump(source/"resolved_config.json", config)
                geometry_path = source/"terrain_informed_geometry.json"
                dump(geometry_path, {"synthetic_test_fixture": True,
                     "queries": [{"map_id": "synthetic", "query_id": "q"+str(i)} for i in range(160)]})
                identity = {"config_hash": digest(config), "code_sha256": source_hash,
                            "g5_frozen_files": {"synthetic/terrain_informed_geometry.json": file_hash(geometry_path)}}
                dump(source/"manifest.json", {"experiment_id": "SYNTHETIC_V2_INFERENCE_GUARD",
                     "identity": identity, "trials": [{"trial_id": "synthetic_0"}], "synthetic_test_fixture": True})
                dump(source/"manifest.lock.json", {"sha256": file_hash(source/"manifest.json")})
                dump(source/"execution_state.json", {"status": "COMPLETE_VALIDATED" if mode == "unvalidated" else mode.upper()})
                dump(source/"validation_summary.json", {"status": "NOT_RUN" if mode == "unvalidated" else "COMPLETE_VALIDATED",
                     "executed": 1, "not_run": 0, "correctness_errors": 0})
                output = self.source/(mode+"_report")
                with patch("g5_analysis.paired_bootstrap", side_effect=AssertionError("forbidden inference")) as bootstrap, \
                     patch("g5_analysis.geometry_association", side_effect=AssertionError("forbidden inference")) as correlation:
                    with self.assertRaisesRegex(ValueError, "full completion and independent validation required"):
                        analyze(source, output)
                    bootstrap.assert_not_called()
                    correlation.assert_not_called()
                self.assertFalse(output.exists())


def preflight(output):
    """Fresh fixture-only run after lock; no planner, primary result, or inference."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = output/"fixtures"
    entries = [materialize(source, "exact"), retained_approximate(source),
               materialize(source, "timeout"), materialize(source, "administrative_interruption")]
    rows = []
    for entry in entries:
        lock_artifacts(source/entry["trial_id"])
        row = read_fixture(source, entry, schema2=True)
        prepare_observations(row, {}, [0., row["time_budget_s"]])
        rows.append(row)
    expected = [True, False, False, False]
    checks = {"all_correctness_errors_absent": all(not r["correctness_error"] for r in rows),
              "success_taxonomy": [r["validated_success"] for r in rows] == expected,
              "approximate_geometry_valid": rows[1]["geometry_valid"] is True,
              "approximate_goal_not_reached": rows[1]["exact_goal_reached"] is False,
              "timeout_without_path": rows[2]["path_present"] is False,
              "administrative_is_not_native_crash": rows[3]["native_ompl_crash"] is False,
              "administrative_duration_unknown": rows[3]["success_attainment"]["observation_end_s"] is None}
    summary = {"passed": all(checks.values()), "checks": checks, "primary_dataset": False,
               "statistical_inference": "NOT_RUN_FIXTURE_PREFLIGHT", "planner_execution": False,
               "analyzer_sha256": file_hash(ROOT/"tools/g5_analysis.py"), "rows": rows}
    dump(output/"summary.json", summary)
    print(json.dumps({"passed": summary["passed"], "checks": checks, "output": str(output)}))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--preflight-output", type=Path)
    args, remaining = parser.parse_known_args()
    if args.runtime_root:
        RUNTIME_ROOT = args.runtime_root
    if args.preflight_output:
        raise SystemExit(preflight(args.preflight_output))
    unittest.main(argv=[sys.argv[0], *remaining], verbosity=2)
