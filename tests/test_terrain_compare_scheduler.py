"""Synthetic subprocess scheduler tests; no planner campaigns are executed."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import terrain_compare_scheduler as s


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="synthetic-scheduler-", dir=ROOT / "tests/fixtures")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.registry = {"fake": {"argv": [sys.executable, str(ROOT / "tests/fixtures/terrain_compare_fake_worker.py")]}}
        self.cpus = s.psutil.Process().cpu_affinity()[:4]
        self.runtime = {"platform": sys.platform, "eligible_cpus": self.cpus,
                        "cpu_capacity": len(self.cpus), "memory_limit_bytes": 16 * s.GIB,
                        "memory_sources": [], "test_fixture": True}
        self.plan = self.root / "plan.json"

    def manifest(self, blocks=4, arms=1, **request):
        entries = []
        for block in range(blocks):
            for arm in range(arms):
                entries.append({"trial_id": f"b{block}_a{arm}", "block_order": block,
                                "planned_order": arm, "backend": "fake", "execution_status": "PLANNED",
                                "track": "common", "map_id": "synthetic", "seed": 11,
                                "request": {"planning_budget_s": .1, "seed": 11, **request}})
        s.atomic_json(self.plan, {"entries": entries, "profile": "smoke", "synthetic_test_fixture": True})
        return entries

    def run_fixture(self, output="run", **options):
        with patch.object(s, "detect_runtime", return_value=self.runtime), \
             patch.object(s, "memory_pressure", return_value=.1):
            return s.run_campaign(self.plan, self.registry, self.root / output, force_local=True,
                                  reservations={"fake": 512 * s.MIB}, **options)

    def test_four_parallel_fresh_processes_affinity_and_whole_block_lane(self):
        if len(self.cpus) < 4:
            self.skipTest("requires four eligible CPUs")
        self.manifest(4, 2, work_s=.18)
        result = self.run_fixture()
        self.assertEqual(result["counts"], {"COMPLETED": 8},
                         [(entry["trial_id"],entry["execution_status"],entry.get("termination_reason"))
                          for entry in result["entries"]])
        self.assertEqual(max(event["concurrency"] for event in result["concurrency_timeline"]), 4)
        self.assertEqual(len({entry["pid"] for entry in result["entries"]}), 8)
        for block in range(4):
            arms = [e for e in result["entries"] if e["block_order"] == block]
            self.assertEqual(len({e["lane"] for e in arms}), 1)
            self.assertLessEqual(arms[0]["completed_utc"], arms[1]["started_utc"])
        for entry in result["entries"]:
            self.assertGreater(entry["peak_rss_bytes"], 0)
            self.assertTrue(entry["co_runners"])
            self.assertFalse(entry["success"])
            launch = s.read_json(self.root / "run" / entry["trial_id"] / "launch.json")
            self.assertEqual(launch["cpu_affinity"], [entry["core"]])
            self.assertTrue(all(value == "1" for value in launch["thread_environment"].values()))

    def test_resume_never_repeats_started_or_failed_trials(self):
        self.manifest(2, crash=True)
        first = self.run_fixture(max_workers=2)
        journal = (self.root / "run/start_journal.jsonl").read_bytes()
        again = self.run_fixture(max_workers=2, resume=True)
        self.assertEqual(first["counts"], {"CRASH": 2})
        self.assertEqual(again["counts"], first["counts"])
        self.assertEqual((self.root / "run/start_journal.jsonl").read_bytes(), journal)

    def test_interruption_retains_four_attempts_and_resume_only_unstarted(self):
        if len(self.cpus) < 4:
            self.skipTest("requires four eligible CPUs")
        self.manifest(6, work_s=.2)
        real_sleep = time.sleep
        with patch.object(s, "time") as clock:
            clock.perf_counter.side_effect = time.perf_counter
            clock.sleep.side_effect = KeyboardInterrupt
            result = self.run_fixture()
        self.assertEqual(result["counts"], {"INTERRUPTED": 4, "PLANNED": 2})
        result = self.run_fixture(resume=True)
        self.assertEqual(result["counts"], {"INTERRUPTED": 4, "COMPLETED": 2})
        self.assertEqual(len((self.root / "run/start_journal.jsonl").read_text().splitlines()), 6)
        self.assertFalse((self.root / "run/controller.lock").exists())

    def test_fixed_work_serial_parallel_preserve_signature(self):
        self.manifest(4)
        serial = self.run_fixture("serial", max_workers=1)
        parallel = self.run_fixture("parallel", max_workers=4)
        for entry in serial["entries"]:
            name = entry["trial_id"]
            left = s.read_json(self.root / "serial" / name / "worker.json")
            right = s.read_json(self.root / "parallel" / name / "worker.json")
            self.assertEqual(left["fixed_work_signature"], right["fixed_work_signature"])
        self.assertEqual(serial["sessions"][0]["timing_mode"], "serial_trials")
        self.assertEqual(parallel["sessions"][0]["timing_mode"], "concurrent_trials")

    def test_resume_hash_mismatch_before_launch(self):
        self.manifest(1)
        self.run_fixture(max_workers=1)
        with self.assertRaisesRegex(s.IntegrityError, "hash mismatch"):
            self.run_fixture(max_workers=2, resume=True)
        contents = s.read_json(self.plan)
        contents["entries"][0]["request"]["seed"] = 12
        s.atomic_json(self.plan, contents)
        with self.assertRaisesRegex(s.IntegrityError, "hash mismatch"):
            self.run_fixture(max_workers=1, resume=True)

    def test_ram_admission_lowers_concurrency(self):
        self.runtime["memory_limit_bytes"] = 1400 * s.MIB
        self.manifest(4, work_s=.15)
        result = self.run_fixture()
        self.assertEqual(result["counts"], {"COMPLETED": 4})
        self.assertEqual(max(e["concurrency"] for e in result["concurrency_timeline"]), 2)

    def test_insufficient_ram_does_not_start_or_label_algorithm_failure(self):
        self.runtime["memory_limit_bytes"] = 600 * s.MIB
        self.manifest(1)
        result = self.run_fixture()
        self.assertEqual(result["counts"], {"PLANNED": 1})
        self.assertEqual(result["sessions"][-1]["stop_reason"], "INSUFFICIENT_RAM_RESERVATION")
        self.assertFalse((self.root / "run/start_journal.jsonl").exists())

    def test_resource_pressure_stops_active_attempts(self):
        self.manifest(4, work_s=1)
        calls = [0]
        def pressure(_):
            calls[0] += 1
            return .1 if calls[0] == 1 else .95
        with patch.object(s, "detect_runtime", return_value=self.runtime), \
             patch.object(s, "memory_pressure", side_effect=pressure), \
             patch.object(s, "PRESSURE_S", .04):
            result = s.run_campaign(self.plan, self.registry, self.root / "run", force_local=True,
                                    reservations={"fake": 512 * s.MIB})
        self.assertEqual(set(result["counts"]), {"RESOURCE_PRESSURE"})
        self.assertTrue(all(not e["success"] for e in result["entries"]))

    def test_memory_watchdog_and_affinity_drift_are_not_success(self):
        for name, affinity, rss in (("rss", None, s.TRIAL_RSS_LIMIT + 1),
                                     ("affinity", [99999], 0)):
            self.manifest(1, work_s=.5)
            original = s._sample_process
            def sample(active):
                original(active)
                if (active["directory"] / "launch.json").exists() and affinity is not None:
                    active["affinities"] = {"synthetic-observation": affinity}
                return rss
            with patch.object(s, "_sample_process", side_effect=sample):
                result = self.run_fixture(name, max_workers=1)
            expected = "MEMORY_LIMIT" if name == "rss" else "AFFINITY_DRIFT"
            self.assertEqual(result["entries"][0]["termination_reason"], expected)
            self.assertFalse(result["entries"][0]["success"])

    def test_setup_solve_serialization_watchdogs_retain_trial(self):
        for phase, request, changes in (
            ("SETUP", {"setup_s": .8}, {"SETUP_LIMIT_S": .15}),
            ("SOLVING", {"work_s": .8}, {"SOLVE_GRACE_S": .05}),
            ("SERIALIZING", {"serialization_s": .8}, {"SERIALIZATION_LIMIT_S": .1}),
        ):
            self.manifest(1, **request)
            with patch.multiple(s, **changes):
                result = self.run_fixture(phase, max_workers=1)
            self.assertEqual(result["entries"][0]["execution_status"], "WATCHDOG")
            self.assertEqual(result["entries"][0]["termination_reason"], phase + "_WATCHDOG")

    def test_phase_regression_and_malformed_result_stop_campaign(self):
        for name, request in (("phase", {"regress_phase": True}),
                              ("json", {"malformed_result": True})):
            self.manifest(2, **request)
            result = self.run_fixture(name, max_workers=1)
            self.assertEqual(result["status"], "FAILED_CORRECTNESS")
            self.assertEqual(result["entries"][1]["execution_status"], "PLANNED")

    def test_soft_stop_does_not_open_blocks_and_resume_can_continue(self):
        self.manifest(2)
        result = self.run_fixture(soft_stop_s=0)
        self.assertEqual(result["counts"], {"PLANNED": 2})
        self.assertEqual(result["status"], "PARTIAL")

    def test_blocked_source_remains_without_registry_entry(self):
        entries = self.manifest(1)
        entries.append({"trial_id": "blocked", "block_order": 0, "planned_order": 1,
                        "backend": "absent", "execution_status": "BLOCKED_SOURCE"})
        s.atomic_json(self.plan, {"entries": entries, "profile": "smoke"})
        result = self.run_fixture(max_workers=1)
        self.assertEqual(result["counts"], {"COMPLETED": 1, "BLOCKED_SOURCE": 1})

    def test_duplicate_ids_rejected_and_single_writer_lock(self):
        entries = self.manifest(2)
        entries[1]["trial_id"] = entries[0]["trial_id"]
        s.atomic_json(self.plan, {"entries": entries})
        with self.assertRaisesRegex(s.IntegrityError, "duplicate"):
            self.run_fixture()
        directory = self.root / "locktest"
        directory.mkdir()
        lock = s._lock(directory)
        with self.assertRaisesRegex(s.IntegrityError, "active"):
            s._lock(directory)
        lock.unlink()

    def test_reservation_formula_and_collect(self):
        self.assertEqual(s.reservation_bytes(0), 512 * s.MIB)
        self.assertEqual(s.reservation_bytes(2 * s.GIB), 3 * s.GIB + 256 * s.MIB)
        self.assertEqual(s.reservation_bytes(8 * s.GIB), 8 * s.GIB)
        self.assertEqual(s.collect_reservations({"entries": [{"backend": "x", "peak_rss_bytes": 0}]}),
                         {"x": 512 * s.MIB})
        for bad in (-1, float("nan"), float("inf")):
            with self.assertRaises(s.IntegrityError):
                s.reservation_bytes(bad)

    def test_runtime_intersects_parent_cpuset_quota_and_memory(self):
        cgroups = self.root / "cgroups"
        child = cgroups / "nested/leaf"
        child.mkdir(parents=True)
        membership = self.root / "membership"
        membership.write_text("0::/nested/leaf\n")
        (cgroups / "cpu.max").write_text("400000 100000")
        (cgroups / "nested/cpu.max").write_text("250000 100000")
        (child / "cpu.max").write_text("max 100000")
        (cgroups / "nested/cpuset.cpus.effective").write_text("2-5")
        (cgroups / "nested/memory.max").write_text(str(4 * s.GIB))
        (child / "memory.max").write_text("max")
        with patch.object(s.sys, "platform", "linux"), \
             patch.object(s.psutil, "Process", return_value=SimpleNamespace(cpu_affinity=lambda: list(range(8)))), \
             patch.object(s.psutil, "virtual_memory", return_value=SimpleNamespace(total=16 * s.GIB)):
            result = s.detect_runtime(cgroups, membership)
        self.assertEqual(result["eligible_cpus"], [2, 3, 4, 5])
        self.assertEqual(result["cpu_capacity"], 2)
        self.assertEqual(result["memory_limit_bytes"], 4 * s.GIB)


if __name__ == "__main__":
    unittest.main()
