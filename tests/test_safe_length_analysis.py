"""Synthetic analysis fixtures only; no benchmark data or fake experiment files."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tools"))
from analyze_safe_length import analyze, build_report, file_hash, main, read


def synthetic_fixture():
    entries, rows = [], []
    specifications = {
        "atlas_fixed200": ((.1, 10., 2., 3.), (None, None, None, None)),
        "sem_native": ((.5, 12., 1., 2.), (1., 14., 1., 3.)),
        "sem_common_safe_length": ((.2, 11., 1.5, 2.5), (None, None, None, None)),
    }
    for method, samples in specifications.items():
        for index, (first, length, risk, slope) in enumerate(samples):
            trial_id = "synthetic_"+method+"_"+str(index)
            entry = dict(trial_id=trial_id, method=method, map_id="synthetic_map_"+str(index),
                         planner_seed=11, status="COMPLETED")
            row = dict(trial_id=trial_id, method=method, map_id=entry["map_id"], family="synthetic",
                execution_status="COMPLETED", planner_status="EXACT_SOLUTION" if first is not None else "NO_SOLUTION",
                success=first is not None, success_geometry=first is not None, attained_safe=first is not None,
                first_safe_s=first, first_safe_latency_s=first+.6 if first is not None else None,
                final_cost_terrain_m=length, final_pri=risk, final_swept_slope_deg=slope,
                correctness_errors=[], field_build_s=.1, solve_time_s=2., events=[])
            entries.append(entry)
            rows.append(row)
    manifest = dict(status="COMPLETE", protocol_id="synthetic_analysis_fixture_only", profile="smoke", entries=entries)
    summary = dict(status="COMPLETE", planned=6, attempted=6, success=4, correctness_failures=0, rows=rows)
    return summary, manifest


class SafeLengthAnalysisTests(unittest.TestCase):
    def test_censoring_keeps_timeout_without_inventing_solution(self):
        summary, manifest = synthetic_fixture()
        report = build_report(summary, manifest)
        self.assertEqual(report["status"], "DONE")
        a3 = next(row for row in report["aggregates"] if row["method"] == "atlas_fixed200")
        self.assertEqual((a3["executed"], a3["safe_success"], a3["ever_safe"]), (2, 1, 1))
        self.assertEqual(a3["first_safe_s"]["n"], 1)
        self.assertEqual(a3["first_safe_s"]["median"], .1)
        self.assertAlmostEqual(a3["restricted_first_safe"]["mean_time_without_safe_s"], 1.05)
        self.assertEqual(a3["restricted_first_safe"]["censored_at_budget_count"], 1)
        censored = next(row for row in report["trials"] if row["method"] == "atlas_fixed200" and not row["attained"])
        self.assertIsNone(censored["first_safe_s"])
        self.assertFalse(censored["restricted_observation"]["event"])
        self.assertTrue(censored["restricted_observation"]["censored"])

    def test_paired_metrics_use_correct_conditional_samples(self):
        summary, manifest = synthetic_fixture()
        report = build_report(summary, manifest)
        pairs = {row["metric"]:row for row in report["paired_comparisons"] if row["baseline"] == "sem_native"}
        length = pairs["final_cost_terrain_m"]
        self.assertEqual((length["available_pairs"], length["planned_pairs"], length["coverage"]), (1, 2, .5))
        self.assertEqual(length["difference"]["mean"], -2.)
        self.assertAlmostEqual(pairs["first_safe_s"]["difference"]["mean"], -.4)
        restricted = pairs["restricted_time_without_safe_s"]
        self.assertEqual(restricted["available_pairs"], 2)
        self.assertAlmostEqual(restricted["difference"]["mean"], .3)

    def test_crash_does_not_become_budget_observation_or_vanish(self):
        summary, manifest = synthetic_fixture()
        summary["status"], manifest["status"] = "INCOMPLETE", "INCOMPLETE"
        summary["rows"][1]["execution_status"] = "CRASH"
        manifest["entries"][1]["status"] = "CRASH"
        report = build_report(summary, manifest)
        self.assertEqual(report["status"], "FAILED")
        self.assertTrue(report["errors"])
        a3 = next(row for row in report["aggregates"] if row["method"] == "atlas_fixed200")
        self.assertEqual(a3["executed"], 2)
        self.assertEqual(a3["restricted_first_safe"]["unavailable_observation_count"], 1)
        self.assertIsNone(a3["restricted_first_safe"]["mean_time_without_safe_s"])
        pair = next(row for row in report["paired_comparisons"] if row["baseline"] == "sem_native" and
                    row["metric"] == "restricted_time_without_safe_s")
        self.assertEqual((pair["eligible_pairs"], pair["available_pairs"], pair["unavailable_pairs"]), (2, 1, 1))
        self.assertIsNone(pair["difference"]["mean"])

    def test_correctness_failure_and_missing_rows_are_visible(self):
        summary, manifest = synthetic_fixture()
        summary["rows"][1]["correctness_errors"] = ["synthetic_bad_edge"]
        summary["correctness_failures"] = 1
        report = build_report(summary, manifest)
        self.assertEqual(report["status"], "FAILED")
        self.assertTrue(any("correctness" in error for error in report["errors"]))
        summary["rows"].pop()
        summary["attempted"] -= 1
        report = build_report(summary, manifest)
        self.assertEqual(len(report["trials"]), 6)
        self.assertEqual(report["trials"][-1]["execution_status"], "NOT_RUN")
        self.assertTrue(any("no observed row" in error for error in report["errors"]))

    def test_duplicate_or_invalid_quality_rejected(self):
        summary, manifest = synthetic_fixture()
        duplicate = copy.deepcopy(summary)
        duplicate["rows"].append(duplicate["rows"][0])
        with self.assertRaises(ValueError):
            build_report(duplicate, manifest)
        summary["rows"][0]["final_pri"] = float("nan")
        with self.assertRaises(ValueError):
            build_report(summary, manifest)

    def _write_fixture(self, source, summary, manifest):
        source.mkdir()
        config = dict(planning_budget_s=2., purpose="SYNTHETIC_TEST_FIXTURE_NOT_BENCHMARK")
        manifest["config_sha256"] = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        for name, value in (("summary.json", summary), ("manifest.json", manifest), ("config.json", config)):
            (source/name).write_text(json.dumps(value, allow_nan=False), encoding="utf-8")

    def test_artifacts_provenance_and_input_immutability(self):
        with tempfile.TemporaryDirectory(prefix="safe_analysis_synthetic_", dir=ROOT/"tests/fixtures") as temporary:
            base = Path(temporary)
            source, output = base/"synthetic_input", base/"synthetic_analysis"
            summary, manifest = synthetic_fixture()
            self._write_fixture(source, summary, manifest)
            before = {path.name:file_hash(path) for path in source.iterdir()}
            report = analyze(source, output)
            self.assertEqual(report["status"], "DONE")
            self.assertEqual(before, {path.name:file_hash(path) for path in source.iterdir()})
            self.assertEqual(read(output/"provenance.json")["input_sha256"], before)
            for name in ("methods.csv", "trials.csv", "paired_comparisons.csv", "REPORT.md", "analysis.json"):
                self.assertTrue((output/name).is_file())
            self.assertNotIn("NaN", (output/"analysis.json").read_text())
            with self.assertRaises(FileExistsError):
                analyze(source, output)
            with self.assertRaises(ValueError):
                analyze(source, source/"bad_output")

    def test_cli_preserves_failed_status_and_nonzero_exit(self):
        with tempfile.TemporaryDirectory(prefix="safe_analysis_failure_synthetic_", dir=ROOT/"tests/fixtures") as temporary:
            base = Path(temporary)
            source, output = base/"synthetic_input", base/"synthetic_analysis"
            summary, manifest = synthetic_fixture()
            summary["status"] = manifest["status"] = "FAILED_CORRECTNESS"
            self._write_fixture(source, summary, manifest)
            code = main(["--input", str(source), "--output", str(output)])
            self.assertEqual(code, 1)
            self.assertEqual(read(output/"analysis.json")["status"], "FAILED")


if __name__ == "__main__":
    unittest.main()
