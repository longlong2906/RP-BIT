"""SYNTHETIC development-analysis fixtures, never experimental observations."""
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tools"))
import pri_pathsearch_protocol as protocol
from analyze_pri_pathsearch import build_report, write_report
from corridor_v4_execution import digest


def synthetic_fixture(phase="pilot"):
    config, entries, rows = protocol.default_config(), [], []
    for family in protocol.FAMILIES:
        for seed in range(3):
            for planner in range(2):
                for backend in protocol.METHODS:
                    tid = f"synthetic_{family}_{seed}_{planner}_{backend}"
                    entry = dict(trial_id=tid, method=backend, family=family, map_id=f"{family}_{seed}",
                        map_seed=seed, planner_seed=planner, query_id="q0", phase=phase,
                        alpha=1.1 if backend == protocol.LEGACY else None, status="COMPLETED")
                    entries.append(entry)
                    metrics = dict(final_cost_terrain_m=12. if backend == protocol.NATIVE else 10.,
                        final_pri=8. if backend == protocol.NATIVE else 6., final_swept_slope_deg=3.,
                        final_length_xy_m=9., final_sharp_turn_count=2, solve_time_s=20., first_safe_s=.2,
                        first_safe_latency_s=1.2, field_build_s=.3, preprocessing_s=.2,
                        final_exposure=8., final_risk_sum=100., final_risk_count=20)
                    selectors = [dict(alpha=alpha, success=True, attained_safe=True,
                        selected_snapshot_id="synthetic_candidate_0", **metrics) for alpha in protocol.ALPHAS]
                    rows.append(dict(entry, execution_status="COMPLETED", execution_launched=True,
                        planner_status="EXACT_SOLUTION", success=True, attained_safe=True,
                        correctness_errors=[], selector_results=selectors if backend in protocol.SELECTOR_BACKENDS else [],
                        synthetic_fixture=True, **metrics))
    manifest = dict(protocol_id=protocol.PROTOCOL_ID, config_sha256=digest(config), phase=phase,
        entries=entries, selected_alpha=None, planned_trials=len(entries), status="COMPLETE",
        sources={"synthetic":"not_real_code"}, synthetic_fixture=True)
    summary = dict(protocol_id=protocol.PROTOCOL_ID, phase=phase, planned=len(entries), attempted=len(rows),
        validated=len(rows), status="COMPLETE", success=len(rows), correctness_failures=0,
        rows=rows, synthetic_fixture=True)
    return summary, manifest, config


def failure(args, index, status="CRASH"):
    summary, manifest, _ = args
    row = summary["rows"][index]
    row.update(success=False, attained_safe=False, execution_status=status, planner_status=None)
    for selector in row["selector_results"]:
        selector.update(success=False, attained_safe=False)
    manifest["entries"][index]["status"] = status
    summary["success"] -= 1


class AnalysisTests(unittest.TestCase):
    def test_physical_attempt_counts_are_not_inflated_by_correlated_selectors(self):
        report = build_report(*synthetic_fixture())
        self.assertEqual((report["planned"], report["attempted"], report["validated"]), (150,150,150))
        self.assertEqual(report["selector_projection_rows"], 330)
        self.assertEqual(len(report["physical_trials"]), 150)
        self.assertEqual(report["selection"]["common_cohort_count"], 30)
        for row in report["tables"]:
            if row["scope"] == "all":
                self.assertEqual(row["attempted"],30)
        self.assertTrue(report["stop_after_pilot"])
        self.assertFalse(report["held_out_evidence"])

    def test_both_length_and_pri_must_be_strictly_lower_in_every_family(self):
        for metric, value in (("final_cost_terrain_m",12.),("final_pri",8.)):
            args = synthetic_fixture()
            for row in args[0]["rows"]:
                if row["method"] == protocol.FULL and row["family"] == "rolling_terrain":
                    for selector in row["selector_results"]:
                        selector[metric] = value
            self.assertEqual(build_report(*args)["selection"]["status"],"NO_CANDIDATE")

    def test_only_full_can_win_shortcut_or_legacy_never_replace_it(self):
        args = synthetic_fixture()
        for row in args[0]["rows"]:
            if row["method"] == protocol.FULL:
                for selector in row["selector_results"]:
                    selector["final_pri"] = 9.
        self.assertEqual(build_report(*args)["selection"]["status"],"NO_CANDIDATE")
        good = build_report(*synthetic_fixture())["selection"]
        self.assertEqual(good["selected_method"],protocol.selector_method(protocol.FULL,1.))

    def test_one_alpha_selected_with_registered_worst_family_tie_break(self):
        args = synthetic_fixture()
        for row in args[0]["rows"]:
            if row["method"] == protocol.FULL:
                row["selector_results"][1].update(final_pri=5., final_cost_terrain_m=11.9)
        selected = build_report(*args)["selection"]
        self.assertEqual(selected["selected_alpha"],1.025)
        self.assertEqual(selected["status"],"CANDIDATE_LOCKED")
        self.assertTrue(selected["stop_after_pilot"])

    def test_shortcut_failure_does_not_remove_primary_full_pairs(self):
        args = synthetic_fixture()
        for index, row in enumerate(args[0]["rows"]):
            if row["method"] == protocol.SHORTCUT:
                failure(args,index)
        report = build_report(*args)
        self.assertEqual(report["selection"]["status"],"CANDIDATE_LOCKED")
        self.assertEqual(report["selection"]["common_cohort_count"],30)

    def test_missing_map_in_shared_full_cohort_is_unrankable(self):
        args = synthetic_fixture()
        for row in args[0]["rows"]:
            if row["method"] == protocol.FULL and row["map_id"] == "ridge_pass_0":
                row["selector_results"][3].update(success=False, attained_safe=False)
                row["success"] = False
                args[0]["success"] -= 1
        report = build_report(*args)
        self.assertEqual(report["selection"]["status"],"UNRANKABLE")
        self.assertEqual(report["selection"]["common_cohort_map_counts"]["ridge_pass"],2)

    def test_failed_attempt_stays_in_all_its_selector_psr_denominators(self):
        args = synthetic_fixture()
        index = next(i for i,r in enumerate(args[0]["rows"]) if r["method"] == protocol.FULL)
        failure(args,index,status="FAILED")
        args[0]["rows"][index]["execution_launched"] = False
        report = build_report(*args)
        for row in report["tables"]:
            if row["scope"] == "all" and row["physical_backend"] == protocol.FULL:
                self.assertEqual(row["attempted"],30)
                self.assertEqual(row["backend_launched"],29)
                self.assertEqual(row["psr"],29/30)
        self.assertEqual(report["selection"]["status"],"NO_CANDIDATE")

    def test_partial_or_correctness_failure_never_selects(self):
        args = synthetic_fixture()
        args[0]["rows"].pop(); args[0]["success"]-=1; args[0]["validated"]-=1
        args[1]["entries"][-1]["status"]="STARTED"
        self.assertEqual(build_report(*args)["selection"]["status"],"INCOMPLETE")
        args = synthetic_fixture()
        index = next(i for i,r in enumerate(args[0]["rows"]) if r["method"]==protocol.FULL)
        failure(args,index,status="FAILED_CORRECTNESS")
        args[0]["rows"][index]["correctness_errors"]=["synthetic invalid final"]
        args[0]["correctness_failures"]=1
        self.assertEqual(build_report(*args)["selection"]["status"],"FAILED_CORRECTNESS")

    def test_duplicate_or_missing_selector_verdict_rejected(self):
        for action in ("duplicate","missing"):
            args = synthetic_fixture()
            row = next(r for r in args[0]["rows"] if r["method"]==protocol.FULL)
            if action=="duplicate": row["selector_results"].append(dict(row["selector_results"][0]))
            else: row["selector_results"].pop()
            with self.assertRaises(ValueError):
                build_report(*args)

    def test_smoke_does_not_select_and_synthetic_export_is_labeled(self):
        report=build_report(*synthetic_fixture("smoke_serial"))
        self.assertNotIn("selection",report)
        with tempfile.TemporaryDirectory(prefix="__pathsearch_synthetic_",dir=ROOT/"tests/fixtures") as tmp:
            report=write_report(*synthetic_fixture(),Path(tmp)/"report")
            self.assertIn("SYNTHETIC",(Path(tmp)/"report/REPORT.md").read_text(encoding="utf-8"))
            self.assertTrue((Path(tmp)/"report/physical_trials.csv").is_file())
            with self.assertRaises(ValueError):
                write_report(*synthetic_fixture(),Path(tmp)/"results")


if __name__ == "__main__":
    unittest.main()
