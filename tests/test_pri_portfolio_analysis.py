"""SYNTHETIC parser/statistics fixtures; never experimental observations."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tools"))

from analyze_pri_portfolio import (ALPHAS, FAMILIES, NATIVE, PROTOCOL_ID, aggregate, build_report,
                                  hierarchical_upper, method_for, paired, wilson, write_report)
from corridor_v4_execution import digest
from pri_portfolio_protocol import default_config


def synthetic_fixture(phase="pilot", selected=1.):
    config = default_config()
    entries, rows = [], []
    methods = [(method_for(a), a) for a in ALPHAS] if phase == "pilot" else [(method_for(selected), selected)]
    methods.append((NATIVE, None))
    for family in FAMILIES:
        for seed in range(3):
            for planner in range(2):
                for method, alpha in methods:
                    tid = f"synthetic_{family}_{seed}_{planner}_{method}"
                    entry = dict(trial_id=tid, method=method, family=family, map_id=f"{family}_{seed}",
                                 map_seed=seed, planner_seed=planner, query_id="q0", alpha=alpha, phase=phase, status="COMPLETED")
                    entries.append(entry)
                    rows.append(dict(entry, execution_status="COMPLETED", execution_launched=True, planner_status="EXACT_SOLUTION",
                                     success=True, attained_safe=True, correctness_errors=[], final_cost_terrain_m=12. if alpha is None else 10.,
                                     final_pri=8. if alpha is None else 6., final_swept_slope_deg=3., final_length_xy_m=9.,
                                     final_sharp_turn_count=2, solve_time_s=20., first_safe_s=.2, first_safe_latency_s=1.2,
                                     field_build_s=.3, preprocessing_s=.2, final_exposure=8., final_risk_sum=100., final_risk_count=20,
                                     synthetic_fixture=True))
    manifest = dict(protocol_id=PROTOCOL_ID, config_sha256=digest(config), phase=phase, entries=entries,
                    selected_alpha=selected if phase == "confirm" else None, planned_trials=len(entries), status="COMPLETE",
                    sources={"synthetic":"not_real_code"}, synthetic_fixture=True)
    summary = dict(protocol_id=PROTOCOL_ID, phase=phase, planned=len(entries), attempted=len(rows), validated=len(rows),
                   status="COMPLETE", success=len(rows), correctness_failures=0, rows=rows, synthetic_fixture=True)
    return summary, manifest, config


def mark_failure(summary, manifest, index, status="CRASH", launched=True):
    summary["rows"][index].update(success=False, attained_safe=False, first_safe_s=None, first_safe_latency_s=None,
                                   execution_status=status, execution_launched=launched, planner_status=None)
    manifest["entries"][index]["status"] = status
    summary["success"] -= 1


class PortfolioAnalysisTests(unittest.TestCase):
    def test_all_five_families_are_required_not_global_average(self):
        args = synthetic_fixture()
        for row in args[0]["rows"]:
            if row["method"] != NATIVE and row["family"] == "rolling_terrain":
                row["final_pri"] = 8.1
        report = build_report(*args)
        self.assertEqual(report["status"], "DONE")
        self.assertEqual(report["selection"]["status"], "NO_CANDIDATE")
        self.assertTrue(all(c["families"][2]["checks"]["pri"] is False for c in report["selection"]["candidates"]))

    def test_one_global_alpha_and_preregistered_tie_break(self):
        report = build_report(*synthetic_fixture())
        self.assertEqual(report["selection"]["status"], "CANDIDATE_LOCKED")
        self.assertEqual(report["selection"]["selected_alpha"], 1.)
        self.assertEqual(report["selection"]["common_cohort_count"], 30)
        self.assertEqual(set(report["selection"]["common_cohort_map_counts"].values()), {3})

    def test_shortest_not_ranked_before_worst_family_pri(self):
        args = synthetic_fixture()
        for row in args[0]["rows"]:
            if row["alpha"] == 1.025:
                row.update(final_pri=5., final_cost_terrain_m=11.9)
        self.assertEqual(build_report(*args)["selection"]["selected_alpha"], 1.025)

    def test_equality_length_allowed_but_equality_pri_fails(self):
        args = synthetic_fixture()
        for row in args[0]["rows"]:
            if row["method"] != NATIVE:
                row["final_cost_terrain_m"] = 12.
        self.assertEqual(build_report(*args)["selection"]["status"], "CANDIDATE_LOCKED")
        for row in args[0]["rows"]:
            if row["method"] != NATIVE:
                row["final_pri"] = 8.
        self.assertEqual(build_report(*args)["selection"]["status"], "NO_CANDIDATE")

    def test_missing_map_common_support_is_unrankable(self):
        args = synthetic_fixture()
        indexes = [i for i,r in enumerate(args[0]["rows"]) if r["method"] == method_for(1.1) and r["map_id"] == "ridge_pass_0"]
        for index in indexes:
            mark_failure(args[0], args[1], index)
        report = build_report(*args)
        self.assertEqual(report["selection"]["status"], "UNRANKABLE")
        self.assertEqual(report["selection"]["common_cohort_map_counts"]["ridge_pass"], 2)

    def test_prelaunch_failure_stays_in_psr_and_blocks_its_candidate(self):
        args = synthetic_fixture()
        index = next(i for i,r in enumerate(args[0]["rows"]) if r["method"] == method_for(1.))
        mark_failure(args[0], args[1], index, "FAILED", launched=False)
        report = build_report(*args)
        row = next(t for t in report["tables"] if t["scope"] == "all" and t["method"] == method_for(1.))
        self.assertEqual(row["attempted"], 30)
        self.assertEqual(row["executed"], 30)
        self.assertEqual(row["backend_launched"], 29)
        self.assertEqual(row["psr"], 29/30)
        self.assertEqual(report["selection"]["selected_alpha"], 1.025)

    def test_partial_and_correctness_error_never_lock_candidate(self):
        args = synthetic_fixture()
        args[0]["rows"].pop(); args[0]["success"] -= 1; args[0]["validated"] -= 1
        args[1]["entries"][-1]["status"] = "STARTED"
        self.assertEqual(build_report(*args)["selection"]["status"], "INCOMPLETE")
        args = synthetic_fixture()
        row = args[0]["rows"][0]
        row.update(success=False, attained_safe=False, correctness_errors=["synthetic invalid final"])
        args[0]["correctness_failures"] = 1; args[0]["success"] -= 1
        self.assertEqual(build_report(*args)["selection"]["status"], "FAILED_CORRECTNESS")

    def test_exactly_paired_quality_and_sample_sd(self):
        report = build_report(*synthetic_fixture())
        table = next(t for t in report["tables"] if t["scope"] == "all" and t["method"] == method_for(1.))
        self.assertEqual(table["final_cost_terrain_m"]["sd"], 0.)
        self.assertAlmostEqual(table["sti"], .2)
        comparison = next(c for c in report["paired_comparisons"] if c["scope"] == "all")
        self.assertEqual(comparison["safe_pairs"], 30)
        self.assertEqual(comparison["metrics"]["final_pri"]["difference"]["mean"], -2.)

    def test_psr_intervals_do_not_prove_zero_difference_at_all_successes(self):
        interval = wilson(100, 100)
        self.assertLess(interval["lower"], 1.)
        report = build_report(*synthetic_fixture())
        interval = report["paired_comparisons"][0]["psr_difference_interval"]
        self.assertLess(interval["lower"], 0.)
        self.assertGreater(interval["upper"], 0.)

    def test_bootstrap_is_paired_map_then_seed_and_keeps_missing_quality(self):
        pairs = [dict(map_id=str(m), final_cost_terrain_m=-2., final_pri=-1.) for m in range(3) for _ in range(2)]
        result = hierarchical_upper(pairs, replicates=2049)
        self.assertEqual(result["final_cost_terrain_m"]["upper"], -2.)
        self.assertEqual(result["final_pri"]["available_replicates"], 2049)
        for row in pairs:
            row["final_pri"] = None
        self.assertIsNone(hierarchical_upper(pairs, replicates=10)["final_pri"]["upper"])

    def test_confirmation_100000_replicates_and_no_population_psr_claim(self):
        selection = build_report(*synthetic_fixture())["selection"]
        report = build_report(*synthetic_fixture("confirm"), tuning_selection=selection)
        decision = report["confirmation"]
        self.assertEqual(decision["status"], "CONFIRMED")
        self.assertEqual(len(decision["families"]), 5)
        self.assertEqual(decision["families"][0]["upper_bounds"]["final_pri"]["replicates"], 100000)
        self.assertLess(decision["families"][0]["psr_difference_interval"]["lower"], 0.)
        args = synthetic_fixture("confirm", selected=1.025)
        with self.assertRaises(ValueError):
            build_report(*args, tuning_selection=selection)

    def test_tampered_identity_or_hash_fails(self):
        for key in ("method", "map_id", "planner_seed", "alpha"):
            args = synthetic_fixture()
            args[0]["rows"][0][key] = "wrong"
            with self.assertRaises(ValueError):
                build_report(*args)
        args = synthetic_fixture(); args[1]["config_sha256"] = "wrong"
        with self.assertRaises(ValueError):
            build_report(*args)

    def test_export_labels_synthetic_and_prevents_results_directory(self):
        args = synthetic_fixture()
        with tempfile.TemporaryDirectory(prefix="synthetic_pri_analysis_", dir=ROOT/"tests/fixtures") as tmp:
            output = Path(tmp)
            write_report(*args, output)
            self.assertIn("SYNTHETIC TEST FIXTURE", (output/"REPORT.md").read_text(encoding="utf-8"))
            self.assertTrue((output/"paired_trials.csv").is_file())
            with self.assertRaises(ValueError):
                write_report(*args, output/"results")


if __name__ == "__main__":
    unittest.main()
