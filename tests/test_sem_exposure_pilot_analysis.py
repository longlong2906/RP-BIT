"""Labeled synthetic pilot parser/selection fixtures; never campaign observations."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import sem_exposure_pilot_protocol as protocol
from analyze_sem_exposure_pilot import build_report, evaluate_confirmation, write_report
from corridor_v4_execution import digest


def fixture(phase="tune", selected_lambda=None, eligible=(1., 4.)):
    config = protocol.default_config()
    manifest = protocol.inventory(config, phase, selected_lambda)
    manifest.update(status="COMPLETE", sources={"fixture": "SYNTHETIC_NOT_EXPERIMENT"},
                    maps=[], synthetic_fixture=True)
    rows = []
    for entry in manifest["entries"]:
        entry["status"] = "COMPLETED"
        weight = entry["sem_lambda"]
        baseline = 30. + 2*protocol.FAMILIES.index(entry["family"]) + entry["map_seed"] % 10
        length = baseline if weight is None else baseline-2. if weight in eligible else baseline-1.
        pri = 5. if weight is None else 4. if weight in eligible else 6.
        rows.append(dict(**{key: entry[key] for key in ("trial_id", "method", "phase", "map_id", "map_seed",
            "query_id", "family", "planner_seed", "alpha", "sem_lambda")},
            synthetic_fixture=True, execution_status="COMPLETED", execution_launched=True,
            success=True, attained_safe=True, planner_status="EXACT_SOLUTION", correctness_errors=[],
            first_safe_s=.1, first_safe_latency_s=.2, final_cost_terrain_m=length, final_pri=pri,
            final_swept_slope_deg=10., final_exposure=40., final_sem_exposure_normalized_m=1.,
            final_objective_value=length+(weight or 0.), final_risk_sum=pri*100, final_risk_count=100))
    summary = dict(protocol_id=protocol.PROTOCOL_ID, phase=phase, status="COMPLETE", synthetic_fixture=True,
                   planned=len(rows), attempted=len(rows), validated=len(rows), rows=rows)
    recount(summary)
    return config, manifest, summary


def recount(summary):
    summary.update(success=sum(row["success"] for row in summary["rows"]),
        validated=len(summary["rows"]), correctness_failures=sum(bool(row["correctness_errors"]) for row in summary["rows"]))


def fail(manifest, row, *, launched=True, status="CRASH"):
    row.update(success=False, attained_safe=False, planner_status="NO_SOLUTION",
               execution_status=status, execution_launched=launched, first_safe_s=None, first_safe_latency_s=None)
    next(entry for entry in manifest["entries"] if entry["trial_id"] == row["trial_id"])["status"] = status


def locked_selection(weight=1.):
    return dict(protocol_id=protocol.PROTOCOL_ID, config_sha256=digest(protocol.default_config()),
        status="CANDIDATE_LOCKED", selected_lambda=weight, selected_method=protocol.method_for(weight))


class PilotProtocolTests(unittest.TestCase):
    def test_exact_counts_seeds_order_and_rejection(self):
        config = protocol.load_config()
        self.assertEqual(config, protocol.default_config())
        tune = protocol.inventory(config)
        confirm = protocol.inventory(config, "confirm", 16.)
        self.assertEqual((len(tune["entries"]), len(confirm["entries"])), (2500, 600))
        for plan, per_block in ((tune, 10), (confirm, 3)):
            self.assertEqual(len({entry["trial_id"] for entry in plan["entries"]}), plan["planned_trials"])
            for block in sorted({entry["block_index"] for entry in plan["entries"]}):
                self.assertEqual(sum(entry["block_index"] == block for entry in plan["entries"]), per_block)
        self.assertFalse({row["map_seed"] for row in tune["entries"]} & {row["map_seed"] for row in confirm["entries"]})
        self.assertFalse({row["planner_seed"] for row in tune["entries"]} & {row["planner_seed"] for row in confirm["entries"]})
        first_maps = list(dict.fromkeys(row["map_id"] for row in tune["entries"]))[:5]
        self.assertEqual(first_maps, [family+"_37001" for family in protocol.FAMILIES])
        for value in (None, 0., 3., True):
            with self.assertRaises(ValueError):
                protocol.inventory(config, "confirm", value)
        with self.assertRaises(ValueError):
            protocol.inventory(config, "full")
        edited = copy.deepcopy(config)
        edited["sem_exposure"]["lambdas"][2] = True
        with self.assertRaises(ValueError):
            protocol.validate_config(edited)

    def test_registered_requests_do_not_leak_smoke_seed_or_budget(self):
        config = protocol.default_config()
        info = dict(map_path=str(ROOT / "tests/fixtures/synthetic_map.t25d"),
            cloud_path=str(ROOT / "tests/fixtures/synthetic_cloud.npy"), cloud_sha256="cloud", map_sha256="map",
            start=[2., 20.], goal=[38., 20.], query_id="q0")
        metadata = dict(field_sha256="field", cloud_sha256="cloud", builder_source_sha256="builder")
        for phase, selected in (("tune", None), ("confirm", 16.)):
            plan = protocol.inventory(config, phase, selected)
            for method, _ in protocol.method_specs(phase, selected):
                entry = next(row for row in reversed(plan["entries"]) if row["method"] == method)
                request = protocol.request_for(entry, dict(info, map_id=entry["map_id"]), config,
                    ROOT / "tests/fixtures/synthetic_field.txt", metadata, ROOT / "tests/fixtures/synthetic_bridge.so")
                self.assertEqual(request["planner_seed"], entry["planner_seed"])
                self.assertEqual(request["planning_budget_s"], 20.)
                if method != protocol.NATIVE:
                    self.assertEqual(request["sem_exposure"], {"lambda": entry["sem_lambda"], "risk_scale": 40.})
                    self.assertEqual(request["safe_length"]["policy_id"], "a3t_safe_length_v1")
                    self.assertNotIn("corridor_policy", request["safe_length"])
                else:
                    self.assertNotIn("sem_exposure", request)


class PilotAnalysisTests(unittest.TestCase):
    def test_tie_break_uses_PRI_then_smaller_positive_lambda(self):
        config, manifest, summary = fixture()
        report = build_report(summary, manifest, config)
        self.assertEqual(report["selection"]["status"], "CANDIDATE_LOCKED")
        self.assertEqual(report["selection"]["selected_lambda"], 1.)
        self.assertNotIn(protocol.CONTROL, report["selection"]["eligible_methods"])
        for row in summary["rows"]:
            if row["sem_lambda"] == 4.:
                row["final_pri"] = 3.5
        report = build_report(summary, manifest, config)
        self.assertEqual(report["selection"]["selected_lambda"], 4.)
        self.assertEqual(report["selection"]["common_cohort_count"], 250)
        self.assertEqual(report["selection"]["common_cohort_sha256"], digest(report["selection"]["common_cohort"]))

    def test_common_cohort_prevents_ranking_different_valid_supports(self):
        config, manifest, summary = fixture()
        native = [row for row in summary["rows"] if row["method"] == protocol.NATIVE]
        easy, hard, absent = [(row["map_id"], row["planner_seed"]) for row in native[:3]]
        for row in summary["rows"]:
            slot = row["map_id"], row["planner_seed"]
            weight = row["sem_lambda"]
            if weight is None:
                row["final_cost_terrain_m"] = 10. if slot == easy else 1000. if slot == hard else 30.
                if slot == absent:
                    fail(manifest, row)
            elif weight in (1., 4.):
                row["final_cost_terrain_m"] = (28. if weight == 1. else 29.)
                if slot == easy:
                    row["final_cost_terrain_m"] = 8.
                if slot == hard:
                    row["final_cost_terrain_m"] = 900.
                if (weight == 1. and slot == easy) or (weight == 4. and slot == hard):
                    fail(manifest, row)
        recount(summary)
        report = build_report(summary, manifest, config)
        own_means = {row["method"]: row["metrics"]["final_cost_terrain_m"]["left_mean"]
            for row in report["paired_comparisons"] if row["scope"] == "all" and row["baseline"] == protocol.NATIVE}
        self.assertLess(own_means[protocol.method_for(4.)], own_means[protocol.method_for(1.)])
        selection = report["selection"]
        self.assertEqual(selection["selected_lambda"], 1.)
        self.assertEqual(selection["common_cohort_count"], 247)
        self.assertEqual([row["mean_surface_length_m"] for row in selection["ranking"]], [28., 29.])

    def test_unrepresented_family_and_no_candidate_are_distinct(self):
        config, manifest, summary = fixture(eligible=())
        self.assertEqual(build_report(summary, manifest, config)["selection"]["status"], "NO_CANDIDATE")
        config, manifest, summary = fixture()
        for row in summary["rows"]:
            if row["family"] == protocol.FAMILIES[0] and row["method"] == protocol.NATIVE:
                fail(manifest, row)
        recount(summary)
        report = build_report(summary, manifest, config)
        self.assertEqual(report["selection"]["status"], "UNRANKABLE_COMMON_SUPPORT")
        self.assertIsNone(report["selection"]["selected_lambda"])

    def test_prelaunch_interruption_counts_in_PSR_and_does_not_block_complete_selection(self):
        config, manifest, summary = fixture()
        for method in (protocol.NATIVE, protocol.method_for(1.)):
            row = next(row for row in summary["rows"] if row["method"] == method)
            fail(manifest, row, launched=False, status="INTERRUPTED")
        recount(summary)
        report = build_report(summary, manifest, config)
        self.assertTrue(report["executed_complete"])
        self.assertEqual(report["executed"], 2500)
        self.assertEqual(report["actual_planner_launched"], 2498)
        self.assertEqual(report["selection"]["status"], "CANDIDATE_LOCKED")
        native = next(row for row in report["tables"] if row["scope"] == "all" and row["method"] == protocol.NATIVE)
        self.assertEqual((native["success"], native["executed"], native["failed_before_launch"]), (249, 250, 1))

    def test_partial_or_correctness_failure_never_selects(self):
        config, manifest, summary = fixture()
        summary["rows"].pop()
        manifest["entries"][-1]["status"] = "RAW_READY"
        manifest["status"] = summary["status"] = "PARTIAL"
        recount(summary)
        report = build_report(summary, manifest, config)
        self.assertEqual(report["selection"]["status"], "INCOMPLETE")
        self.assertEqual(report["unresolved_started"], 1)
        config, manifest, summary = fixture()
        row = summary["rows"][0]
        row.update(success=False, attained_safe=False, correctness_errors=["synthetic_metric_mismatch"])
        recount(summary)
        report = build_report(summary, manifest, config)
        self.assertEqual(report["selection"]["status"], "FAILED_CORRECTNESS")

    def test_confirmation_is_reproducible_and_only_tests_locked_candidate(self):
        config, manifest, summary = fixture("confirm", 16., eligible=(16.,))
        selection = locked_selection(16.)
        first = build_report(summary, manifest, config, selection)
        second = build_report(summary, manifest, config, selection)
        self.assertEqual(first["confirmation"], second["confirmation"])
        self.assertEqual(first["confirmation"]["status"], "CONFIRMED")
        self.assertEqual(first["confirmation"]["recommended_lambda"], 16.)
        intervals = [row for row in first["paired_comparisons"] if row["metrics"]["psr"]["ci95"] is not None]
        self.assertEqual(len(intervals), 1)
        self.assertEqual(intervals[0]["metrics"]["psr"]["ci95"]["replicates"], 2000)
        with self.assertRaises(ValueError):
            build_report(summary, manifest, config, locked_selection(4.))

    def test_negative_means_with_uncertain_CI_are_not_confirmed(self):
        config, manifest, summary = fixture("confirm", 4., eligible=(4.,))
        for row in summary["rows"]:
            if row["sem_lambda"] == 4.:
                baseline = 30. + 2*protocol.FAMILIES.index(row["family"]) + row["map_seed"] % 10
                row["final_cost_terrain_m"] = baseline + (-1. if row["map_seed"] % 2 else .9)
        report = build_report(summary, manifest, config, locked_selection(4.))
        decision = report["confirmation"]
        self.assertTrue(all(decision["mean_checks"].values()))
        self.assertFalse(decision["ci_checks"]["final_cost_terrain_m"])
        self.assertEqual(decision["status"], "NOT_CONFIRMED")
        self.assertIsNone(decision["recommended_lambda"])
        self.assertEqual(decision["candidate_lambda"], 4.)

    def test_duplicate_identity_schedule_or_weight_drift_rejected(self):
        config, manifest, summary = fixture()
        summary["rows"].append(copy.deepcopy(summary["rows"][0]))
        with self.assertRaises(ValueError):
            build_report(summary, manifest, config)
        config, manifest, summary = fixture()
        summary["rows"][0]["sem_lambda"] = 3.
        with self.assertRaises(ValueError):
            build_report(summary, manifest, config)
        config, manifest, summary = fixture()
        manifest["entries"][0], manifest["entries"][1] = manifest["entries"][1], manifest["entries"][0]
        with self.assertRaises(ValueError):
            build_report(summary, manifest, config)

    def test_writer_exports_charts_and_blocks_fake_results_outside_fixtures(self):
        config, manifest, summary = fixture(eligible=())
        report = build_report(summary, manifest, config)
        with self.assertRaises(ValueError):
            write_report(report, ROOT / "reports/SHOULD_NEVER_EXIST_SYNTHETIC")
        with tempfile.TemporaryDirectory(prefix="synthetic_pilot_report_", dir=ROOT / "tests/fixtures") as temp:
            output = Path(temp) / "synthetic_output"
            published = write_report(report, output, provenance={"fixture_label": "SYNTHETIC_NOT_EXPERIMENT"})
            self.assertEqual(published["provenance"]["fixture_label"], "SYNTHETIC_NOT_EXPERIMENT")
            for name in ("REPORT.md", "summary.json", "selection.json", "aggregate.csv", "paired_comparisons.csv",
                         "trials.csv", "ranking.csv", "common_cohort.csv", "lambda_tradeoff.png", "lambda_tradeoff.svg"):
                self.assertTrue((output / name).is_file(), name)
            self.assertEqual((output / "lambda_tradeoff.png").read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(json.loads((output / "selection.json").read_text())["status"], "NO_CANDIDATE")
            with self.assertRaises(FileExistsError):
                write_report(report, output)


if __name__ == "__main__":
    unittest.main()
