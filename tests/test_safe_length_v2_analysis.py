"""Synthetic parser/selection tests; never planner benchmark evidence."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tools"))
from analyze_safe_length_v2 import (BOOTSTRAP_REPLICATES, PROTOCOL_ID, _bootstrap, build_report,
    digest, held_out_gate, markdown, select_pilot, validate_registration)
from safe_length_v2_protocol import default_config, inventory

SPEC = json.loads((ROOT/"tests/fixtures/safe_length_v2_analysis_cases.json").read_text())


def fixture(profile="pilot"):
    entries, rows, maps = [], [], []
    for family in SPEC["families"]:
        for map_index in range(SPEC["maps_per_family"]):
            map_id = family + "_" + str(map_index)
            maps.append(dict(map_id=map_id, family=family))
            for seed in SPEC["planner_seeds"]:
                for method, (slack, length, pri) in SPEC[profile+"_methods"].items():
                    entry = dict(trial_id=f"{map_id}__s{seed}__{method}", method=method,
                        map_id=map_id, family=family, planner_seed=seed, risk_slack=slack, status="COMPLETED")
                    entries.append(entry)
                    rows.append(dict(**{key: entry[key] for key in entry if key != "status"},
                        synthetic_fixture=True, execution_launched=True,
                        execution_status="COMPLETED", planner_status="EXACT_SOLUTION",
                        success=True, attained_safe=True, first_safe_s=.1, first_safe_latency_s=1.1,
                        final_cost_terrain_m=length, final_pri=pri, final_swept_slope_deg=3.,
                        correctness_errors=[], field_build_s=.2, calibration_s=.3, solve_time_s=20.))
    manifest = dict(protocol_id=PROTOCOL_ID, profile=profile, status="COMPLETE", entries=entries,
                    maps=maps, planned_trials=len(entries))
    summary = dict(status="COMPLETE", profile=profile, planned=len(entries), attempted=len(rows),
                   success=len(rows), correctness_failures=0, rows=rows, synthetic_fixture=True)
    return summary, manifest, default_config()


def fail(summary, manifest, method, *, status="COMPLETED", error=None):
    row = next(row for row in summary["rows"] if row["method"] == method)
    entry = next(entry for entry in manifest["entries"] if entry["trial_id"] == row["trial_id"])
    row.update(execution_status=status, planner_status="NO_SOLUTION", success=False, attained_safe=False,
               first_safe_s=None, first_safe_latency_s=None, final_cost_terrain_m=None, final_pri=None,
               final_swept_slope_deg=None)
    entry["status"] = status
    summary["success"] -= 1
    if error:
        row["correctness_errors"] = [error]
        summary["correctness_failures"] += 1
    return row, entry


class AnalysisTests(unittest.TestCase):
    def test_pilot_all_three_gates_and_deterministic_tie(self):
        report = build_report(*fixture())
        self.assertEqual(report["status"], "DONE")
        selection = select_pilot(report)
        self.assertEqual((selection["status"], selection["selected_method"], selection["selected_risk_slack"]),
                         ("SELECTED", "a3t_v2_d0p5", .5))
        candidates = {row["method"]: row for row in selection["candidates"]}
        self.assertFalse(candidates["a3t_v2_d2"]["checks"]["pri"])
        self.assertFalse(candidates["a3t_v2_d4"]["checks"]["length"])
        self.assertTrue(candidates["a3t_v2_d1"]["passes"])
        self.assertEqual(report["bootstrap_replicates"], 0)
        self.assertEqual(report["held_out_gate"]["status"], "NOT_APPLICABLE")

    def test_no_candidate_when_psr_is_lower_even_if_quality_improves(self):
        summary, manifest, config = fixture()
        for method in SPEC["pilot_methods"]:
            if method.startswith("a3t_v2_d"):
                fail(summary, manifest, method)
        report = build_report(summary, manifest, config)
        self.assertEqual(report["pilot_selection"]["status"], "NO_CANDIDATE")
        self.assertIsNone(report["pilot_selection"]["selected_risk_slack"])
        self.assertTrue(all(not candidate["checks"]["psr"] for candidate in report["pilot_selection"]["candidates"]))
        pair = next(row for row in report["paired_comparisons"] if row["scope"] == "all" and row["method"] == "a3t_v2_d0p5")
        self.assertEqual((pair["attempted_pairs"], pair["safe_pairs"]), (8, 7))
        self.assertEqual(pair["metrics"]["psr"]["difference"]["mean"], -1/8)
        self.assertEqual(pair["metrics"]["final_pri"]["coverage"], 7/8)

    def test_incomplete_pilot_cannot_select_and_preserves_not_run(self):
        summary, manifest, config = fixture()
        missing = summary["rows"].pop()
        summary.update(status="INCOMPLETE", attempted=summary["attempted"]-1, success=summary["success"]-1)
        manifest["status"] = "INCOMPLETE"
        next(entry for entry in manifest["entries"] if entry["trial_id"] == missing["trial_id"])["status"] = "NOT_RUN"
        report = build_report(summary, manifest, config)
        self.assertEqual(report["status"], "INCOMPLETE")
        self.assertEqual(report["pilot_selection"]["status"], "BLOCKED")
        self.assertEqual(len(report["trials"]), summary["planned"])
        self.assertEqual(report["trials"][-1]["execution_status"], "NOT_RUN")

    def test_crash_is_psr_failure_not_deadline_solution(self):
        summary, manifest, config = fixture()
        fail(summary, manifest, "a3t_v2_d0p5", status="CRASH")
        report = build_report(summary, manifest, config)
        group = next(row for row in report["tables"] if row["scope"] == "all" and row["method"] == "a3t_v2_d0p5")
        self.assertEqual(group["psr"], 7/8)
        self.assertEqual(group["attempted"], 8)
        self.assertIsNone(group["restricted_first_safe"]["mean"])
        self.assertEqual(group["restricted_first_safe"]["unavailable"], 1)
        self.assertEqual(group["first_safe_s"]["n"], 7)
        self.assertEqual(report["status"], "DONE")

    def test_timeout_is_censored_not_a_first_safe_event(self):
        summary, manifest, config = fixture()
        fail(summary, manifest, "a3t_v2_d0p5")
        report = build_report(summary, manifest, config)
        group = next(row for row in report["tables"] if row["scope"] == "all" and row["method"] == "a3t_v2_d0p5")
        self.assertEqual(group["restricted_first_safe"]["censored_at_budget"], 1)
        self.assertAlmostEqual(group["restricted_first_safe"]["mean"], (7*.1+20)/8)
        self.assertEqual(group["first_safe_s"]["n"], 7)

    def test_blocked_before_launch_excluded_from_psr_but_blocks_pilot(self):
        summary, manifest, config = fixture()
        row, _ = fail(summary, manifest, "a3t_v2_d0p5", status="BLOCKED")
        row["execution_launched"] = False
        report = build_report(summary, manifest, config)
        group = next(item for item in report["tables"] if item["scope"] == "all" and item["method"] == "a3t_v2_d0p5")
        self.assertEqual((group["attempted"], group["executed"], group["failed_before_launch"], group["psr"]), (8, 7, 1, 1.))
        self.assertEqual(group["restricted_first_safe"]["n"], 7)
        self.assertEqual(group["restricted_first_safe"]["unavailable"], 0)
        pair = next(item for item in report["paired_comparisons"] if item["scope"] == "all" and item["method"] == "a3t_v2_d0p5")
        self.assertEqual((pair["attempted_pairs"], pair["executed_pairs"]), (8, 7))
        self.assertEqual(report["status"], "INCOMPLETE_EXECUTION")
        self.assertTrue(report["complete"])
        self.assertFalse(report["executed_complete"])
        self.assertEqual(report["pilot_selection"]["status"], "BLOCKED")

    def test_failed_and_interrupted_after_launch_stay_in_psr(self):
        for status in ("FAILED", "INTERRUPTED"):
            summary, manifest, config = fixture()
            row, _ = fail(summary, manifest, "a3t_v2_d0p5", status=status)
            row["execution_launched"] = True
            report = build_report(summary, manifest, config)
            group = next(item for item in report["tables"] if item["scope"] == "all" and item["method"] == "a3t_v2_d0p5")
            self.assertEqual((group["executed"], group["failed_before_launch"], group["psr"]), (8, 0, 7/8))
            self.assertTrue(report["executed_complete"])
            self.assertEqual(group["restricted_first_safe"]["unavailable"], 1)
            row["execution_launched"] = False
            report = build_report(summary, manifest, config)
            self.assertEqual(report["pilot_selection"]["status"], "BLOCKED")
            group = next(item for item in report["tables"] if item["scope"] == "all" and item["method"] == "a3t_v2_d0p5")
            self.assertEqual((group["executed"], group["psr"]), (7, 1.))

    def test_unlaunched_full_trial_cannot_pass_even_when_paired_ci_passes(self):
        summary, manifest, config = fixture("full")
        row, _ = fail(summary, manifest, "a3t_v2", status="BLOCKED")
        row["execution_launched"] = False
        report = build_report(summary, manifest, config, bootstrap_replicates=100)
        self.assertEqual(report["held_out_gate"]["status"], "BLOCKED")
        self.assertFalse(report["held_out_gate"]["passes"])

    def test_completed_outcome_cannot_claim_no_process(self):
        summary, manifest, config = fixture()
        summary["rows"][0]["execution_launched"] = False
        report = build_report(summary, manifest, config)
        self.assertEqual(report["status"], "FAILED_CORRECTNESS")
        self.assertTrue(any("contradicts missing launch" in error for error in report["errors"]))

    def test_correctness_failure_blocks_selection(self):
        summary, manifest, config = fixture()
        fail(summary, manifest, "a3t_v2_d4", error="synthetic_wrong_cap")
        report = build_report(summary, manifest, config)
        self.assertEqual(report["status"], "FAILED_CORRECTNESS")
        self.assertEqual(report["pilot_selection"]["status"], "BLOCKED")

    def test_exact_goal_finite_metric_and_deadline_contract(self):
        summary, manifest, config = fixture()
        summary["rows"][0]["planner_status"] = "APPROXIMATE_SOLUTION"
        report = build_report(summary, manifest, config)
        self.assertEqual(report["pilot_selection"]["status"], "BLOCKED")
        summary, manifest, config = fixture()
        summary["rows"][0]["final_pri"] = float("nan")
        with self.assertRaisesRegex(ValueError, "quality metric"):
            build_report(summary, manifest, config)
        summary, manifest, config = fixture()
        summary["rows"][0].update(first_safe_s=21., first_safe_latency_s=22.)
        self.assertEqual(build_report(summary, manifest, config)["pilot_selection"]["status"], "BLOCKED")

    def test_duplicate_identity_and_wrong_seed_rejected(self):
        summary, manifest, config = fixture()
        summary["rows"].append(copy.deepcopy(summary["rows"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            build_report(summary, manifest, config)
        summary, manifest, config = fixture()
        summary["rows"][0]["planner_seed"] = 999
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            build_report(summary, manifest, config)

    def test_smoke_never_selects_or_reports_superiority(self):
        report = build_report(*fixture("smoke"))
        self.assertEqual(report["pilot_selection"]["status"], "BLOCKED")
        self.assertEqual(report["held_out_gate"]["status"], "NOT_APPLICABLE")
        self.assertEqual(report["inference"], "smoke_correctness_only")
        self.assertEqual(report["bootstrap_replicates"], 0)
        self.assertIn("không chứng minh ưu thế", markdown(report))
        self.assertTrue(any(row["baseline"] == "sem_common_v2" for row in report["paired_comparisons"]))
        json.dumps(report, allow_nan=False)

    def test_full_joint_ci_gate_pass_and_strict_pri_failure(self):
        summary, manifest, config = fixture("full")
        report = build_report(summary, manifest, config)
        self.assertEqual(report["bootstrap_replicates"], BOOTSTRAP_REPLICATES)
        self.assertEqual(report["held_out_gate"]["status"], "PASS")
        for row in summary["rows"]:
            if row["method"] == "a3t_v2":
                row["final_pri"] = 5.
        report = build_report(summary, manifest, config, bootstrap_replicates=100)
        self.assertEqual(report["held_out_gate"]["status"], "NOT_MET")
        self.assertFalse(report["held_out_gate"]["checks"]["final_pri"])

    def test_full_psr_ci_rejects_even_one_failure(self):
        summary, manifest, config = fixture("full")
        fail(summary, manifest, "a3t_v2")
        report = build_report(summary, manifest, config, bootstrap_replicates=300)
        self.assertFalse(report["held_out_gate"]["checks"]["psr"])
        self.assertEqual(report["held_out_gate"]["status"], "NOT_MET")

    def test_bootstrap_keeps_family_fixed_and_resamples_maps(self):
        def pair(family, map_id, seed, difference):
            return dict(family=family, map_id=map_id, planner_seed=seed, psr=0.,
                        final_cost_terrain_m=difference, final_pri=difference,
                        final_swept_slope_deg=difference)
        fixed = [pair("a", "a1", seed, -2.) for seed in range(20)] + [pair("b", "b1", seed, -4.) for seed in range(20)]
        result = _bootstrap(fixed, 200, 3)["final_pri"]
        self.assertEqual((result["lower"], result["upper"]), (-3., -3.))
        clusters = [pair("a", "a1", seed, -1.) for seed in range(20)] + [pair("a", "a2", seed, 1.) for seed in range(20)]
        result = _bootstrap(clusters, 200, 3)["final_pri"]
        self.assertEqual((result["lower"], result["upper"]), (-1., 1.))
        self.assertEqual(result, _bootstrap(clusters, 200, 3)["final_pri"])

    def test_empty_quality_resamples_do_not_make_ci_from_selected_draws(self):
        pairs = [dict(family="a", map_id="m", planner_seed=1, psr=-1., final_cost_terrain_m=None,
                      final_pri=None, final_swept_slope_deg=None)]
        result = _bootstrap(pairs, 20, 3)
        self.assertIsNone(result["final_pri"]["upper"])
        self.assertEqual(result["final_pri"]["available_replicates"], 0)
        self.assertEqual(result["psr"]["lower"], -1.)

    def test_production_registration_rejects_truncated_schedule_and_unlocked_full(self):
        config = default_config()
        manifest = inventory(config, "pilot")
        validate_registration(manifest, config)
        manifest["entries"].pop()
        with self.assertRaisesRegex(ValueError, "complete registered profile"):
            validate_registration(manifest, config)
        manifest = inventory(config, "full", .5)
        with self.assertRaisesRegex(ValueError, "pilot selection"):
            validate_registration(manifest, config)
        manifest["selection_sha256"] = "synthetic_fixture_only"
        validate_registration(manifest, config)


if __name__ == "__main__":
    unittest.main()
