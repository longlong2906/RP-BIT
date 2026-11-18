"""Synthetic parser cases are generated only under tests/fixtures when needed."""
import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import sem_exposure_protocol as protocol
from analyze_sem_exposure import build_report, markdown


def synthetic_case():
    config = protocol.default_config()
    manifest = protocol.inventory(config)
    manifest.update(status="COMPLETE", synthetic_fixture=True, maps=[])
    rows = []
    for entry in manifest["entries"]:
        entry.update(status="COMPLETED")
        rows.append(dict(**{key: entry[key] for key in ("trial_id", "method", "map_id", "family", "planner_seed", "alpha", "sem_lambda")},
            synthetic_fixture=True, execution_status="COMPLETED", execution_launched=True,
            success=True, attained_safe=True, planner_status="EXACT_SOLUTION", correctness_errors=[],
            first_safe_s=.1, first_safe_latency_s=.3, final_cost_terrain_m=10., final_pri=2.,
            final_swept_slope_deg=3., final_exposure=20., final_sem_exposure_normalized_m=.5,
            final_objective_value=10. + (entry["sem_lambda"] or 0)*.5))
    summary = dict(protocol_id=protocol.PROTOCOL_ID, profile="smoke", status="COMPLETE", synthetic_fixture=True,
        planned=42, attempted=42, validated=42, success=42, correctness_failures=0, rows=rows)
    return config, manifest, summary


class AnalysisTests(unittest.TestCase):
    def test_smoke_never_selects_weight_or_claims_superiority(self):
        config, manifest, summary = synthetic_case()
        report = build_report(summary, manifest, config)
        self.assertEqual(report["status"], "DONE")
        self.assertEqual(report["selection"], dict(status="NOT_RUN_SMOKE_ONLY", selected_lambda=None))
        self.assertTrue(report["no_superiority_claim"])
        self.assertIn("không chọn λ", markdown(report))
        self.assertEqual(len([r for r in report["paired_comparisons"] if r["scope"] == "all"]), 7)

    def test_failure_retained_and_paired_quality_excludes_it(self):
        config, manifest, summary = synthetic_case()
        index = next(i for i, row in enumerate(summary["rows"]) if row["method"] == protocol.NATIVE)
        summary["rows"][index].update(success=False, attained_safe=False, execution_status="CRASH", planner_status=None)
        manifest["entries"][index]["status"] = "CRASH"
        summary["success"] -= 1
        report = build_report(summary, manifest, config)
        row = next(row for row in report["paired_comparisons"] if row["scope"] == "all" and row["method"] == protocol.NOMINAL and row["baseline"] == protocol.NATIVE)
        self.assertEqual(row["safe_pairs"], 6)
        self.assertEqual(row["planned_pairs"], 7)
        native = next(row for row in report["tables"] if row["scope"] == "all" and row["method"] == protocol.NATIVE)
        self.assertEqual(native["executed"], 7)
        self.assertEqual(native["success"], 6)

    def test_raw_pending_is_incomplete_without_fabricated_verdict(self):
        config, manifest, summary = synthetic_case()
        pending = summary["rows"].pop()
        manifest["entries"][-1]["status"] = "RAW_READY"
        manifest["status"] = summary["status"] = "FAILED_INFRASTRUCTURE"
        summary.update(validated=41, success=41)
        report = build_report(summary, manifest, config)
        self.assertEqual(report["status"], "INCOMPLETE")
        self.assertEqual(report["attempted"], 42)
        self.assertEqual(report["trials"][-1]["execution_status"], "AWAITING_VERDICT")
        self.assertIsNone(report["trials"][-1]["final_cost_terrain_m"])

    def test_mutated_weight_and_duplicate_verdict_are_rejected(self):
        config, manifest, summary = synthetic_case()
        summary["rows"][0]["sem_lambda"] = 2.
        with self.assertRaisesRegex(ValueError, "weight"):
            build_report(summary, manifest, config)
        config, manifest, summary = synthetic_case()
        summary["rows"].append(copy.deepcopy(summary["rows"][0]))
        with self.assertRaises(ValueError):
            build_report(summary, manifest, config)


if __name__ == "__main__":
    unittest.main()
