"""Synthetic parser/statistics fixtures live only under tests/fixtures."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
from experiment_common import ROOT, digest, dump, file_hash, read_json
from analyze_results import (analyze, attainment_record, checked_events, freeze_references,
                             metric, paired_hierarchical_bootstrap, step_cost,
                             verify_raw_artifact_inventory)


def write_raw_inventory(source):
    excluded={"raw_artifact_inventory.json","campaign_state.json","analysis_attempt.json"}
    entries=[]
    for path in sorted((path for path in source.rglob("*") if path.is_file()),
                       key=lambda item:item.relative_to(source).as_posix()):
        relative=path.relative_to(source).as_posix()
        if relative in excluded or relative.endswith(".tmp"):
            continue
        entries.append({"path":relative,"size_bytes":path.stat().st_size,"sha256":file_hash(path)})
    inventory={"schema_version":1,"entries":entries,"inventory_sha256":digest(entries)}
    dump(source/"raw_artifact_inventory.json",inventory)
    return inventory["inventory_sha256"]


def fixture(directory):
    source = directory/"raw"
    source.mkdir()
    config = {"analysis": {"bootstrap_unit_order": ["map", "query", "seed"], "confidence_level": .95,
                           "target_relative_tolerances": [.10, .05, .02]},
              "model": {"numerical_abs_tolerance_m": 1e-8, "numerical_rel_tolerance": 1e-8},
              "profiles": {"pilot": {"planning_budget_s": 5.}}}
    dump(source/"resolved_config.json", config)
    entries = []
    cases = read_json(ROOT/"tests/fixtures/analysis_cases.json")
    assert cases["synthetic_test_fixture"] is True
    for case in cases["cases"]:
        planner, seed = case["planner"], case["seed"]
        entry = {"trial_id": f"fixture_{planner}_{seed}", "family": "fixture",
                 "map_id": "fixture_map", "map_hash": "synthetic", "query_id": "q0",
                 "planner_id": planner, "planner_seed": seed}
        entries.append(entry)
        if case["status"] == "NOT_RUN":
            continue
        trial = dict(entry, execution_status=case["status"], planner_status="EXACT_SOLUTION" if "costs" in case else "NO_SOLUTION",
                     path_available="costs" in case, independent_valid=case.get("valid"),
                     time_budget_s=5., best_cost_terrain_m=case.get("costs", [[None, None]])[-1][1],
                     first_solution_time_s=case.get("costs", [[None]])[0][0], length_xy_m=8.,
                     independently_recomputed_cost_m=case.get("costs", [[None, None]])[-1][1],
                     actual_solve_wall_s=1. if "costs" in case else 5.,
                     termination_reason="lower_bound" if "costs" in case else "deadline",
                     counters={"motion_validity_calls": 100, "sample_proposals": 20,
                               "samples_returned_before_validity": 10, "rejected_bounds": 5, "rejected_heuristic": 5})
        dump(source/entry["trial_id"]/"trial.json", trial)
        events = []
        for index, (elapsed, cost) in enumerate(case.get("costs", [])):
            sid = f"s{index}"
            snapshot = source/entry["trial_id"]/"snapshots"/sid
            snapshot.mkdir(parents=True)
            (snapshot/"path_xy.csv").write_text("x_m,y_m\n0,0\n8,0\n")
            (snapshot/"path_surface.csv").write_text("x_m,y_m,z_m\n0,0,0\n8,0,0\n")
            dump(snapshot/"validation.json", {"valid": case["valid"], "cost_terrain_m": cost,
                  "xy_sha256": file_hash(snapshot/"path_xy.csv"), "surface_sha256": file_hash(snapshot/"path_surface.csv"),
                  "synthetic_test_fixture": True})
            events.append({"snapshot_id": sid, "elapsed_s": elapsed, "best_cost_terrain_m": cost,
                           "independent_valid": case["valid"], "counters": {"motion_validity_calls": 10*(index+1)}})
        dump(source/entry["trial_id"]/"validated_events.json", events)
        dump(source/entry["trial_id"]/"validation.json", {"valid": case.get("valid", False), "synthetic_test_fixture": True})
    manifest = {"schema_version": 1, "experiment_id": "synthetic_analysis_fixture", "profile": "pilot",
                "planned_trials": len(entries), "status": "PARTIAL", "identity": {"synthetic": True}, "trials": entries}
    dump(source/"manifest.json", manifest)
    return source


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="analysis_runtime_", dir=ROOT/"tests/fixtures")
        self.directory = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_denominators_null_censoring_partial_and_invalid_snapshot(self):
        source = fixture(self.directory)
        output = self.directory/"report"
        summary = analyze(source, output, bootstrap_replicates=30, make_plots=False)
        self.assertEqual((summary["planned"], summary["executed"], summary["validated_success"], summary["not_run"]), (8, 6, 2, 2))
        self.assertEqual(summary["status"], "PARTIAL")
        self.assertEqual(summary["success_rate_executed"], 2/6)
        self.assertEqual((summary["timeout"], summary["crash"], summary["watchdog"], summary["invalid"], summary["blocked"]), (1, 1, 1, 1, 1))
        self.assertEqual(len(summary["missing_groups"]), 2)
        metrics = read_json(output/"trial_metrics.json")
        timed_out = next(r for r in metrics if r["status"] == "TIMEOUT")
        self.assertIsNone(timed_out["success_attainment"]["event_time_s"])
        self.assertEqual(timed_out["success_attainment"]["observation_end_s"], 5.)
        self.assertEqual(timed_out["quality_targets"], [])
        self.assertIsNone(timed_out["metrics"]["rewires"])
        self.assertIn("rewires", timed_out["metric_unavailable_reasons"])
        early = next(r for r in metrics if r["planner_id"] == "informed_rrtstar" and r["success"])
        self.assertEqual(early["cost_terrain_m"], 9.)
        self.assertEqual(early["final_cost_terrain_m"], 8.)
        self.assertEqual(early["events_after_budget"], 1)
        self.assertEqual(early["improvements_beyond_tolerance"], 1)
        self.assertEqual(early["custom_sampler_acceptance_all_phases"]["value"], .5)
        self.assertIsNone(early["custom_sampler_acceptance_all_phases"]["terrain_validity_acceptance_ratio"])
        self.assertIsNone(early["informed_acceptance"]["value"])
        curves = read_json(output/"anytime_curves.json")
        early_curve = next(c for c in curves if c["planner_id"] == "informed_rrtstar")
        self.assertIsNone(early_curve["points"][0]["median_cost_terrain_m"])
        self.assertEqual(early_curve["points"][-1]["time_s"], 5.)
        self.assertEqual(early_curve["points"][-1]["median_cost_terrain_m"], 9.)
        self.assertEqual(early_curve["points"][-1]["success_attainment"], .5)
        self.assertNotIn("NaN", (output/"summary.json").read_text())
        self.assertNotIn("Infinity", (output/"trial_metrics.json").read_text())

    def test_snapshot_hash_tampering_blocks_quality(self):
        source = fixture(self.directory)
        folder = source/"fixture_bitstar_a0_11"
        (folder/"snapshots/s0/path_xy.csv").write_text("changed after validation")
        events, errors = checked_events(folder, read_json(folder/"trial.json"), 1e-8, 1e-8)
        self.assertEqual(errors[0]["error"], "snapshot_hash_mismatch")
        self.assertEqual(len(events), 1)
        summary = analyze(source, self.directory/"report", bootstrap_replicates=10, make_plots=False)
        self.assertEqual(summary["validated_success"], 1)
        self.assertEqual(summary["invalid"], 2)

    def test_step_no_backfill_and_not_run_not_censored(self):
        events = [{"elapsed_s": 1., "best_cost_terrain_m": 12.}, {"elapsed_s": 2., "best_cost_terrain_m": 10.}]
        self.assertIsNone(step_cost(events, .99))
        self.assertEqual(step_cost(events, 1.), 12.)
        self.assertEqual(step_cost(events, 1.999), 12.)
        self.assertEqual(step_cost(events, 5.), 10.)
        record = attainment_record({"executed": False, "time_budget_s": 5.})
        self.assertEqual(record["status"], "NOT_RUN")
        self.assertIsNone(record["observation_end_s"])

    def test_crash_censoring_uses_measured_worker_fallback_or_null(self):
        base={"executed":True,"success":False,"status":"CRASH","events":[],
              "time_budget_s":5.,"metrics":{"actual_solve_wall_s":None,"worker_wall_s":.375}}
        record=attainment_record(base)
        self.assertEqual(record["observation_end_s"],.375)
        self.assertEqual(record["observation_end_measurement"],"worker_wall_s_fallback")
        self.assertEqual(record["observation_end_reason"],"actual_solve_wall_s_unavailable")
        unavailable=copy.deepcopy(base);unavailable["metrics"]["worker_wall_s"]=None
        record=attainment_record(unavailable)
        self.assertIsNone(record["observation_end_s"])
        self.assertEqual(record["observation_end_reason"],"crash_or_watchdog_duration_unavailable")

    def test_frozen_witness_targets_cannot_follow_pilot_winner(self):
        source = fixture(self.directory)
        manifest = read_json(source/"manifest.json")
        witness = self.directory/"witness.csv"
        witness.write_text("synthetic fixture witness")
        reference = {"reference_type": "validated_witness", "cost_terrain_m": 10., "path_xy": str(witness),
                     "path_sha256": file_hash(witness), "validation": {"valid": True, "cost_terrain_m": 10.},
                     "provenance": "SYNTHETIC TEST FIXTURE ONLY"}
        for entry in manifest["trials"]:
            entry["reference"] = reference
        output = self.directory/"reference_report"
        output.mkdir()
        refs, before_hash = freeze_references(manifest, output, 1e-8, 1e-8)
        self.assertEqual(refs["fixture_map/q0"]["cost_terrain_m"], 10.)
        _, after_hash = freeze_references(manifest, output, 1e-8, 1e-8)
        self.assertEqual(before_hash, after_hash)
        changed = copy.deepcopy(manifest)
        for entry in changed["trials"]:
            entry["reference"]["cost_terrain_m"] = 9.
            entry["reference"]["validation"]["cost_terrain_m"] = 9.
        with self.assertRaisesRegex(ValueError, "frozen references changed"):
            freeze_references(changed, output, 1e-8, 1e-8)

    def test_paired_hierarchy_and_seed_pairs(self):
        rows = []
        for map_id in ("m1", "m2"):
            for query_id in ("q1", "q2"):
                for seed in (11, 22):
                    for planner in ("a", "b"):
                        rows.append({"map_id": map_id, "query_id": query_id, "planner_seed": seed,
                                     "planner_id": planner, "executed": True, "success": True,
                                     "cost_terrain_m": (100 if map_id == "m1" else 200)+(3 if planner == "a" else 0)})
        result = paired_hierarchical_bootstrap(rows, "a", "b", "cost_terrain_m", 100, 887)
        self.assertEqual(result["estimate"], 3.)
        self.assertEqual(result["ci"], [3., 3.])
        self.assertEqual((result["units"]["maps"], result["units"]["map_query_units"], result["units"]["paired_seed_slots"]), (2, 4, 8))
        self.assertEqual(result, paired_hierarchical_bootstrap(rows, "a", "b", "cost_terrain_m", 100, 887))
        rows[-1]["executed"] = False
        partial = paired_hierarchical_bootstrap(rows, "a", "b", "success", 100)
        self.assertEqual(partial["units"]["unpaired_executed_slots"], 1)
        self.assertEqual(partial["units"]["paired_seed_slots"], 7)
        self.assertEqual(partial["units"]["planned_seed_slots"], 8)
        self.assertEqual((partial["units"]["both_planners_executed_slots"],
                          partial["units"]["one_planner_executed_slots"],
                          partial["units"]["neither_planner_executed_slots"]),(7,1,0))
        self.assertEqual(partial["estimate"], 0.)
        rows[-2]["executed"] = False
        neither = paired_hierarchical_bootstrap(rows, "a", "b", "success", 100)
        self.assertEqual((neither["units"]["both_planners_executed_slots"],
                          neither["units"]["one_planner_executed_slots"],
                          neither["units"]["neither_planner_executed_slots"]),(7,0,1))
        self.assertEqual(neither["units"]["planned_seed_slots"],8)
        one_map = paired_hierarchical_bootstrap(rows[:8], "a", "b", "cost_terrain_m", 100)
        self.assertEqual(one_map["status"], "INSUFFICIENT_INDEPENDENT_MAPS_FOR_CI")
        self.assertIsNone(one_map["ci"])

    def test_metric_aliases_preserve_zero_and_missing(self):
        self.assertEqual(metric({"counters": {"rejected_bounds": 0}}, "sample_rejected_bounds"), (0, None))
        self.assertEqual(metric({"load_s": .2}, "load_wall_s"), (.2, None))
        value, reason = metric({}, "rewires")
        self.assertIsNone(value)
        self.assertIsInstance(reason, str)

    def test_bootstrap_equal_map_weight_and_failure_conditioning(self):
        rows = []
        for map_id, query_count, difference in (("small", 1, 0), ("large", 3, 10)):
            for query in range(query_count):
                for seed in range(2):
                    for planner in ("a", "b"):
                        rows.append({"map_id": map_id, "query_id": f"q{query}", "planner_seed": seed,
                                     "planner_id": planner, "executed": True, "success": True,
                                     "cost_terrain_m": 100+(difference if planner == "a" else 0)})
        result = paired_hierarchical_bootstrap(rows, "a", "b", "cost_terrain_m", 100, 22)
        self.assertEqual(result["estimate"], 5.)  # A pooled trial average would be 7.5.
        self.assertEqual(result["ci"], [0., 10.])
        rows[0]["success"] = False
        success = paired_hierarchical_bootstrap(rows, "a", "b", "success", 100, 22)
        cost = paired_hierarchical_bootstrap(rows, "a", "b", "cost_terrain_m", 100, 22)
        self.assertEqual(success["units"]["paired_seed_slots"], 8)
        self.assertEqual(cost["units"]["paired_seed_slots"], 7)
        self.assertEqual(cost["units"]["unavailable_complete_pairs"], 1)

    def test_witness_target_is_not_optimality_gap_and_finite_sampler_denominator(self):
        source = fixture(self.directory)
        manifest = read_json(source/"manifest.json")
        witness = self.directory/"witness.csv"
        witness.write_text("synthetic fixture witness")
        for entry in manifest["trials"]:
            entry["reference"] = {"reference_type": "validated_witness", "cost_terrain_m": 10.,
                  "path_xy": str(witness), "path_sha256": file_hash(witness),
                  "validation": {"valid": True, "cost_terrain_m": 10.}, "provenance": "SYNTHETIC TEST FIXTURE ONLY"}
        dump(source/"manifest.json", manifest)
        path = source/"fixture_informed_rrtstar_22/trial.json"
        trial = read_json(path)
        trial["counters"].update(finite_bound_sample_proposals=12, finite_bound_samples_returned=3)
        dump(path, trial)
        analyze(source, self.directory/"report", bootstrap_replicates=10, make_plots=False)
        metrics = read_json(self.directory/"report/trial_metrics.json")
        row = next(r for r in metrics if r["trial_id"] == "fixture_informed_rrtstar_22")
        self.assertEqual(row["reference_relative_difference"], -.1)
        self.assertEqual(row["quality_targets"][1]["event_time_s"], .25)
        self.assertEqual(row["quality_targets"][1]["reference_type"], "validated_witness")
        self.assertEqual(row["informed_acceptance"]["value"], .25)
        self.assertEqual(row["informed_acceptance"]["denominator"], 12)
        self.assertEqual(row["custom_sampler_acceptance_all_phases"]["value"], .5)
        a0 = next(r for r in metrics if r["trial_id"] == "fixture_bitstar_a0_11")
        self.assertIsNone(a0["quality_targets"][1]["event_time_s"])
        self.assertFalse(a0["quality_targets"][1]["event_observed"])

    def test_embedded_generator_reference_schema(self):
        source = fixture(self.directory)
        manifest = read_json(source/"manifest.json")
        witness = [[0., 0.], [8., 0.]]
        reference = {"reference_type": "validated_witness", "cost_terrain_m": 8.,
                     "path_xy": witness, "path_sha256": digest(witness), "map_sha256": "synthetic",
                     "validation": {"valid": True, "cost_terrain_m": 8.}, "provenance": "SYNTHETIC TEST FIXTURE ONLY"}
        for entry in manifest["trials"]:
            entry["reference"] = reference
        output = self.directory/"references"
        output.mkdir()
        frozen, _ = freeze_references(manifest, output, 1e-8, 1e-8)
        self.assertEqual(frozen["fixture_map/q0"]["path_xy"], witness)
        reference["path_xy"][1][0] = 7.
        with self.assertRaisesRegex(ValueError, "reference path hash mismatch"):
            freeze_references(manifest, output, 1e-8, 1e-8)

    def test_advanced_analysis_is_config_driven_a3_vs_three_with_locked_bootstrap(self):
        source=self.directory/"advanced_raw";source.mkdir()
        planners=["bitstar_a3","abitstar","aitstar","lazyprmstar_hauser2015"]
        config={"analysis":{"bootstrap_unit_order":["map","query","seed"],"confidence_level":.95,
                            "target_relative_tolerances":[.10,.05,.02]},
                "model":{"numerical_abs_tolerance_m":1e-8,"numerical_rel_tolerance":1e-8},
                "profiles":{"advanced":{"planning_budget_s":5.,"planners":planners}},
                "advanced_benchmark":{"primary_planner_id":"bitstar_a3",
                    "comparator_planner_ids":planners[1:],"informed_planner_ids":planners[:3],
                    "bootstrap_replicates":2000,"bootstrap_seed":73132,
                    "report_metrics":["validated_success","first_solution_time_s","cost_terrain_m_at_budget",
                                      "length_xy_m_final","actual_solve_wall_s","coverage"]}}
        dump(source/"resolved_config.json",config)
        entries=[]
        for map_id in ("heldout_3001","heldout_3002"):
            for seed in (91001,91002):
                for planner in planners:
                    entries.append({"trial_id":f"{map_id}_{planner}_{seed}","family":"heldout_fixture",
                        "map_id":map_id,"map_hash":"synthetic","query_id":"q0","planner_id":planner,
                        "planner_seed":seed})
        dump(source/"manifest.json",{"schema_version":3,"experiment_id":"advanced_synthetic_fixture",
             "profile":"advanced","planned_trials":len(entries),"status":"PLANNED","trials":entries})
        inventory_hash=write_raw_inventory(source)
        output=self.directory/"advanced_report"
        summary=analyze(source,output,make_plots=False)
        self.assertEqual(summary["raw_artifact_inventory_sha256"],inventory_hash)
        self.assertEqual(verify_raw_artifact_inventory(source,required=True),inventory_hash)
        self.assertEqual(summary["planner_ids"],planners)
        self.assertEqual(summary["bootstrap_replicates"],2000)
        self.assertEqual(summary["bootstrap_seed"],73132)
        self.assertEqual({(item["planner_a"],item["planner_b"]) for item in summary["primary_comparisons"]},
                         {("bitstar_a3",planner) for planner in planners[1:]})
        bootstrap=read_json(output/"paired_bootstrap.json")
        self.assertEqual(len(bootstrap),9)
        self.assertEqual({(item["planner_a"],item["planner_b"]) for item in bootstrap},
                         {("bitstar_a3",planner) for planner in planners[1:]})
        self.assertEqual({item["seed"] for item in bootstrap},{73132})
        self.assertTrue(all("held_out_finite_budget_2_map_campaign" in item["interpretation"]
                            for item in bootstrap))
        self.assertEqual({item["raw_artifact_inventory_sha256"] for item in bootstrap},{inventory_hash})
        self.assertEqual(len(summary["metric_coverage"]),len(planners)*6)
        report=(output/"REPORT.md").read_text(encoding="utf-8")
        for planner in planners[1:]:self.assertIn("bitstar_a3 − "+planner,report)
        self.assertIn(inventory_hash,report)

    def test_raw_inventory_detects_post_inventory_mutation(self):
        source=fixture(self.directory)
        inventory_hash=write_raw_inventory(source)
        self.assertEqual(verify_raw_artifact_inventory(source),inventory_hash)
        path=source/"resolved_config.json"
        path.write_text(path.read_text(encoding="utf-8")+" ",encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"inventory (size|hash) mismatch"):
            analyze(source,self.directory/"mutated_report",bootstrap_replicates=10,make_plots=False)

    def test_report_failure_never_leaves_canonical_completion_summary(self):
        source=fixture(self.directory);output=self.directory/"failed_report";output.mkdir()
        dump(output/"summary.json",{"status":"COMPLETE","synthetic_stale_marker":True})
        with patch("analyze_results.write_report",side_effect=OSError("synthetic REPORT write failure")):
            with self.assertRaisesRegex(OSError,"synthetic REPORT write failure"):
                analyze(source,output,bootstrap_replicates=10,make_plots=False)
        self.assertFalse((output/"summary.json").exists())
        self.assertTrue((output/".summary.previous.json").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
