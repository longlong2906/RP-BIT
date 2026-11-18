"""Labeled synthetic parser fixtures; not experimental results."""
import copy
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from corridor_v3_protocol import ALPHAS,PROTOCOL_ID,default_config,inventory
from analyze_corridor_v3 import build_report,select_pilot,validate_registration


def fixture(profile="pilot"):
    manifest=inventory(default_config(),profile,1.1 if profile=="full" else None)
    manifest.update(status="COMPLETE",maps=[],selection_sha256="synthetic-selected")
    rows=[]
    for entry in manifest["entries"]:
        entry["status"]="COMPLETED"
        candidate=entry["method"].startswith("a3t_corridor")
        rows.append(dict(**{key:entry[key] for key in entry if key!="status"},synthetic_fixture=True,
            execution_launched=True,execution_status="COMPLETED",planner_status="EXACT_SOLUTION",
            success=True,attained_safe=True,first_safe_s=.1,first_safe_latency_s=1.,
            final_cost_terrain_m=9. if candidate else 10.,final_pri=2. if candidate else 3.,
            final_swept_slope_deg=3.,correctness_errors=[],corridor_s=.3,graph_s=.1,
            guidance_s=.1,mask_s=.1,final_risk_sum=4.,final_risk_count=2,final_exposure=18.))
    summary=dict(status="COMPLETE",profile=profile,planned=len(rows),attempted=len(rows),
        success=len(rows),correctness_failures=0,rows=rows,synthetic_fixture=True)
    return summary,manifest,default_config()


class AnalysisTests(unittest.TestCase):
    def test_selects_three_gates_and_alpha_tie(self):
        report=build_report(*fixture())
        self.assertEqual(report["status"],"DONE")
        self.assertEqual(report["pilot_selection"]["selected_alpha"],1.05)
        self.assertEqual(len(report["pilot_selection"]["candidates"]),3)
        self.assertEqual(report["bootstrap_replicates"],0)
        self.assertTrue(any(row["baseline"].startswith("sem_common_corridor") for row in report["paired_comparisons"]))

    def test_worse_pri_rejects_all_candidates(self):
        summary,manifest,config=fixture()
        for row in summary["rows"]:
            if row["method"].startswith("a3t_corridor"):
                row["final_pri"]=4.
        report=build_report(summary,manifest,config)
        self.assertEqual(report["pilot_selection"]["status"],"NO_CANDIDATE")

    def test_failure_remains_in_psr_and_blocks_winner(self):
        summary,manifest,config=fixture()
        for alpha in ALPHAS:
            row=next(r for r in summary["rows"] if r["method"].startswith("a3t_corridor") and r["alpha"]==alpha)
            row.update(success=False,attained_safe=False,execution_status="CRASH",planner_status="NO_SOLUTION")
            next(e for e in manifest["entries"] if e["trial_id"]==row["trial_id"])["status"]="CRASH"
            summary["success"]-=1
        report=build_report(summary,manifest,config)
        self.assertEqual(report["pilot_selection"]["status"],"NO_CANDIDATE")
        candidate=next(r for r in report["tables"] if r["scope"]=="all" and r["method"]=="a3t_corridor_a1p05")
        self.assertEqual(candidate["psr"],29/30)
        self.assertIsNone(candidate["restricted_first_safe"]["mean"])

    def test_full_joint_ci_gates_and_auxiliary_metrics(self):
        report=build_report(*fixture("full"),bootstrap_replicates=25)
        self.assertTrue(report["held_out_gate"]["passes"])
        aggregate=next(r for r in report["tables"] if r["scope"]=="all" and r["method"]=="a3t_corridor_v3")
        self.assertEqual(aggregate["corridor_s"]["mean"],.3)
        self.assertEqual(aggregate["final_risk_sum"]["mean"],4.)

    def test_schedule_or_identity_changes_fail_closed(self):
        config=default_config()
        manifest=inventory(config,"pilot")
        validate_registration(manifest,config)
        manifest["entries"][0]["alpha"]=1.2
        with self.assertRaisesRegex(ValueError,"schedule"):
            validate_registration(manifest,config)
        summary,manifest,config=fixture()
        summary["rows"][0]["alpha"]=1.2
        with self.assertRaises(ValueError):
            build_report(summary,manifest,config)


if __name__=="__main__":
    unittest.main()
