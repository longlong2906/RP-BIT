"""Synthetic launch-barrier regressions; never launches a planner.

The optional paths allow an external recovery patch to be tested against an
unchanged, restored repository on Linux as well as the development workspace.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch


arguments = argparse.ArgumentParser(add_help=False)
arguments.add_argument("--scheduler-path", type=Path)
arguments.add_argument("--root", type=Path)
options, remaining = arguments.parse_known_args()
ROOT = (options.root or Path(__file__).resolve().parents[1]).resolve()
SCHEDULER = (options.scheduler_path or ROOT / "tools/terrain_compare_scheduler.py").resolve()
spec = importlib.util.spec_from_file_location("affinity_scheduler_under_test", SCHEDULER)
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
s.LAUNCHER_PATH = ROOT / "tools/terrain_compare_launch.py"


class AffinityLaunchRaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="synthetic-affinity-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.core = s.psutil.Process().cpu_affinity()[0]
        self.registry = {"fixture": {"argv": [sys.executable,
            str(ROOT / "tests/fixtures/terrain_compare_fake_worker.py")]}}
        self.runtime = {"platform": sys.platform, "eligible_cpus": [self.core],
            "cpu_capacity": 1, "memory_limit_bytes": 16 * s.GIB,
            "memory_sources": [], "synthetic_test_fixture": True}
        self.plan = self.root / "plan.json"
        s.atomic_json(self.plan, {"profile": "smoke", "synthetic_test_fixture": True,
            "entries": [{"trial_id": "fixture", "block_order": 0, "planned_order": 0,
                "backend": "fixture", "execution_status": "PLANNED",
                "request": {"planning_budget_s": 1., "seed": 11, "work_s": .35}}]})

    def run_fixture(self, name="run", **kwargs):
        with patch.object(s, "detect_runtime", return_value=self.runtime), \
                patch.object(s, "memory_pressure", return_value=.1):
            result = s.run_campaign(self.plan, self.registry, self.root / name,
                max_workers=1, reservations={"fixture": 512 * s.MIB}, force_local=True, **kwargs)
        return result["entries"][0]

    def proof_active(self):
        active = {"directory": self.root, "proc": SimpleNamespace(pid=123, poll=lambda: None),
            "entry": {"pid": 123, "pid_created": 456., "core": self.core,
                      "command": ["launcher", "--", "worker"]},
            "began": time.perf_counter(), "cpu_by_pid": {}, "peak_rss": 0, "max_threads": 0,
            "affinities": {}}
        proof = {"pid": 123, "process_created": 456., "cpu_affinity": [self.core],
            "affinity_before_worker_import": True,
            "thread_environment": {key: "1" for key in s.THREAD_ENV}, "command": ["worker"]}
        s.atomic_json(self.root / "launch.json", proof)
        return active, proof

    def test_pre_pin_sample_then_launch_publication_does_not_fail_actual_loop(self):
        read_proof, sample = s._read_launch_proof, s._sample_process
        injected = {"read": False, "sample": False, "legacy_would_reject": False}

        def before(active):
            if not injected["read"]:
                injected["read"] = True
                return None  # Proof was absent at the beginning of this poll.
            return read_proof(active)

        def during(active):
            rss = sample(active)
            if not injected["sample"]:
                injected["sample"] = True
                # Model a sample taken before the launcher pins its CPU. The
                # launcher then publishes its actual proof before sample returns.
                active["affinities"] = {str(active["proc"].pid): [self.core, self.core + 999]}
                deadline = time.monotonic() + 5
                while not (active["directory"] / "launch.json").exists():
                    if time.monotonic() >= deadline:
                        self.fail("synthetic launcher did not publish proof")
                    time.sleep(.005)
                injected["legacy_would_reject"] = any(
                    cpus != [self.core] for cpus in active["affinities"].values())
            return rss

        with patch.object(s, "_read_launch_proof", side_effect=before), \
                patch.object(s, "_sample_process", side_effect=during):
            entry = self.run_fixture()
        self.assertTrue(injected["legacy_would_reject"])
        self.assertEqual(entry["execution_status"], "COMPLETED", entry)
        guard = entry["affinity_guard"]
        self.assertGreaterEqual(guard["prelaunch_samples"], 1)
        self.assertGreaterEqual(guard["postlaunch_samples"], 1)
        self.assertNotIn("failure_observation", guard)
        # Compare the deterministic fixture output, excluding clocks, PID and
        # resource observations; this is not a planner quality comparison.
        baseline = self.run_fixture("baseline")
        self.assertEqual(baseline["execution_status"], "COMPLETED")
        left = s.read_json(self.root / "run/fixture/worker.json")
        right = s.read_json(self.root / "baseline/fixture/worker.json")
        self.assertEqual(left["fixed_work_signature"], right["fixed_work_signature"])

    def test_real_postlaunch_bad_affinity_rejects_and_records_sample(self):
        sample = s._sample_process
        def wrong(active):
            rss = sample(active)
            if active.get("verified_launch_proof"):
                active["affinities"][str(active["proc"].pid)] = [self.core + 999]
            return rss
        with patch.object(s, "_sample_process", side_effect=wrong):
            entry = self.run_fixture()
        self.assertEqual(entry["termination_reason"], "AFFINITY_DRIFT")
        self.assertEqual(entry["execution_status"], "FAILED_CORRECTNESS")
        guard = entry["affinity_guard"]
        observation = guard["failure_observation"]
        self.assertTrue(observation["launch_ready_before_sample"])
        self.assertEqual(observation["observed_affinities"][str(entry["pid"])], [self.core + 999])
        self.assertEqual(observation["expected_process_created"], entry["pid_created"])
        self.assertEqual(guard["scheduler_sha256"], s.file_hash(SCHEDULER))
        self.assertEqual(guard["launcher_sha256"], s.file_hash(s.LAUNCHER_PATH))
        self.assertEqual(s.read_json(self.root / "run/fixture/execution.json")["affinity_guard"], guard)
        journal = (self.root / "run/start_journal.jsonl").read_bytes()
        again = self.run_fixture(resume=True)
        self.assertEqual(again["pid"], entry["pid"])
        self.assertEqual((self.root / "run/start_journal.jsonl").read_bytes(), journal)

    def test_malformed_launch_and_pid_mismatch_stop_actual_loop(self):
        read_proof = s._read_launch_proof
        for name in ("malformed", "pid"):
            changed = [False]
            def corrupt(active):
                path = active["directory"] / "launch.json"
                if path.exists() and not changed[0]:
                    changed[0] = True
                    if name == "malformed":
                        path.write_text("{malformed", encoding="utf-8")
                    else:
                        value = s.read_json(path)
                        value["pid"] += 999
                        s.atomic_json(path, value)
                return read_proof(active)
            with patch.object(s, "_read_launch_proof", side_effect=corrupt):
                entry = self.run_fixture(name)
            self.assertTrue(changed[0])
            self.assertEqual(entry["termination_reason"], "AFFINITY_PROTOCOL_ERROR")
            self.assertIn("failure_observation", entry["affinity_guard"])

    def test_physical_postlaunch_cpu_change_is_rejected(self):
        eligible = s.psutil.Process().cpu_affinity()
        if len(eligible) < 2:
            self.skipTest("physical affinity drift fixture requires two eligible CPUs")
        other = next(core for core in eligible if core != self.core)
        sample = s._sample_process
        changed = [False]
        def change_core(active):
            if active.get("verified_launch_proof") and not changed[0]:
                s.psutil.Process(active["proc"].pid).cpu_affinity([other])
                changed[0] = True
            return sample(active)
        with patch.object(s, "_sample_process", side_effect=change_core):
            entry = self.run_fixture()
        self.assertTrue(changed[0])
        self.assertEqual(entry["termination_reason"], "AFFINITY_DRIFT")
        observation = entry["affinity_guard"]["failure_observation"]
        self.assertEqual(observation["observed_affinities"][str(entry["pid"])], [other])

    def test_completed_worker_without_launch_proof_is_not_accepted(self):
        with patch.object(s, "_read_launch_proof", return_value=None):
            entry = self.run_fixture()
        self.assertEqual(entry["execution_status"], "FAILED_CORRECTNESS")
        self.assertEqual(entry["termination_reason"], "AFFINITY_LAUNCH_PROOF_MISSING")

    def test_launch_proof_identity_pinning_threads_and_command_fail_closed(self):
        changes = ({"pid": True}, {"pid": 124}, {"process_created": 457.},
            {"process_created": None}, {"process_created": True},
            {"cpu_affinity": [self.core + 999]}, {"cpu_affinity": []},
            {"affinity_before_worker_import": 1}, {"thread_environment": {}},
            {"thread_environment": {key: "2" for key in s.THREAD_ENV}}, {"command": ["other"]})
        for change in changes:
            with self.subTest(change=change):
                active, proof = self.proof_active()
                proof.update(change)
                s.atomic_json(self.root / "launch.json", proof)
                with self.assertRaises(s.AffinityProtocolError):
                    s._read_launch_proof(active)

    def test_duplicate_nonfinite_and_nonobject_proofs_reject(self):
        for value in ('{"pid":123,"pid":123}', '{"process_created":NaN}', '[]', 'null'):
            active, _ = self.proof_active()
            (self.root / "launch.json").write_text(value, encoding="utf-8")
            with self.assertRaises(s.AffinityProtocolError):
                s._read_launch_proof(active)

    def test_valid_proof_cannot_change_after_barrier(self):
        active, proof = self.proof_active()
        self.assertEqual(s._read_launch_proof(active), proof)
        proof["extra"] = "changed"
        s.atomic_json(self.root / "launch.json", proof)
        with self.assertRaisesRegex(s.AffinityProtocolError, "changed"):
            s._read_launch_proof(active)

    def test_affinity_api_missing_denied_and_unsupported_fail_closed(self):
        for failure in (AttributeError("cpu_affinity"), NotImplementedError("unsupported"),
                        s.psutil.AccessDenied(123)):
            active, _ = self.proof_active()
            process = SimpleNamespace(pid=123, create_time=lambda: 456., children=lambda recursive: [],
                cpu_times=lambda: SimpleNamespace(user=0., system=0.),
                memory_info=lambda: SimpleNamespace(rss=1), num_threads=lambda: 1)
            if not isinstance(failure, AttributeError):
                def unsupported():
                    raise failure
                process.cpu_affinity = unsupported
            with patch.object(s.psutil, "Process", return_value=process):
                with self.assertRaises(s.AffinityProtocolError):
                    s._sample_with_affinity_guard(active)
            receipt = active["entry"]["affinity_guard"]["failure_observation"]
            self.assertEqual(receipt["observation_errors"][0]["error"], type(failure).__name__)

    def test_no_live_observation_fails_but_exited_between_samples_is_not_drift(self):
        active, _ = self.proof_active()
        with patch.object(s.psutil, "Process", side_effect=s.psutil.NoSuchProcess(123)):
            with self.assertRaisesRegex(s.AffinityProtocolError, "no live"):
                s._sample_with_affinity_guard(active)
            active["proc"].poll = lambda: 0
            self.assertEqual(s._sample_with_affinity_guard(active), (0, None))

    def test_reused_pid_is_not_accepted(self):
        active, _ = self.proof_active()
        process = SimpleNamespace(pid=123, create_time=lambda: 457., children=lambda recursive: [])
        with patch.object(s.psutil, "Process", return_value=process):
            with self.assertRaises(s.AffinityProtocolError):
                s._sample_with_affinity_guard(active)
        receipt = active["entry"]["affinity_guard"]["failure_observation"]
        self.assertEqual(receipt["observed_process_created"], {"123": 457.})
        self.assertEqual(receipt["observation_errors"][0]["error"], "process_identity_mismatch")

    def test_external_launcher_path_and_actual_scheduler_are_hashed(self):
        _, _, artifacts = s._freeze_inputs(self.plan, self.registry)
        self.assertEqual(artifacts[str(SCHEDULER)], s.file_hash(SCHEDULER))
        self.assertEqual(artifacts[str(s.LAUNCHER_PATH.resolve())], s.file_hash(s.LAUNCHER_PATH))

    def test_external_module_uses_frozen_launcher_without_editing_it(self):
        external = self.root / "terrain_compare_scheduler_r2.py"
        external.write_bytes(SCHEDULER.read_bytes())
        imported = importlib.util.spec_from_file_location("external_scheduler_fixture", external)
        module = importlib.util.module_from_spec(imported)
        imported.loader.exec_module(module)
        module.LAUNCHER_PATH = s.LAUNCHER_PATH
        before = module.file_hash(module.LAUNCHER_PATH)
        with patch.object(module, "detect_runtime", return_value=self.runtime), \
                patch.object(module, "memory_pressure", return_value=.1):
            result = module.run_campaign(self.plan, self.registry, self.root / "external_run",
                max_workers=1, reservations={"fixture": 512 * module.MIB}, force_local=True)
        entry = result["entries"][0]
        self.assertEqual(entry["execution_status"], "COMPLETED", entry)
        self.assertEqual(entry["command"][1], str(s.LAUNCHER_PATH))
        self.assertEqual(result["identity"]["artifact_sha256"][str(external.resolve())], module.file_hash(external))
        self.assertEqual(entry["affinity_guard"]["scheduler_sha256"], module.file_hash(external))
        self.assertEqual(module.file_hash(module.LAUNCHER_PATH), before)

    def test_eight_fixed_work_trials_serial_and_four_lanes_preserve_results(self):
        eligible = s.psutil.Process().cpu_affinity()
        if len(eligible) < 4:
            if sys.platform.startswith("linux"):
                self.fail("Linux recovery gate requires four eligible CPUs")
            self.skipTest("four-lane synthetic fixture requires four eligible CPUs")
        runtime = dict(self.runtime, eligible_cpus=eligible[:4], cpu_capacity=4)
        entries = [{"trial_id": f"block{block}_arm{arm}", "block_order": block,
                    "planned_order": arm, "backend": "fixture", "execution_status": "PLANNED",
                    "request": {"planning_budget_s": 1., "seed": 11 + block, "work_s": .35}}
                   for block in range(4) for arm in range(2)]
        s.atomic_json(self.plan, {"profile": "smoke", "synthetic_test_fixture": True, "entries": entries})

        def collect(name, workers, resume=False):
            with patch.object(s, "detect_runtime", return_value=runtime), \
                    patch.object(s, "memory_pressure", return_value=.1):
                return s.run_campaign(self.plan, self.registry, self.root / name,
                    max_workers=workers, reservations={"fixture": 512 * s.MIB},
                    force_local=True, resume=resume)

        results = {"serial": collect("serial", 1), "parallel": collect("parallel", 4)}
        signatures = {}
        expected_ids = {entry["trial_id"] for entry in entries}
        for name, result in results.items():
            self.assertEqual(result["counts"], {"COMPLETED": 8}, result["entries"])
            self.assertEqual(max(event["concurrency"] for event in result["concurrency_timeline"]),
                             1 if name == "serial" else 4)
            identities = {(entry["pid"], entry["pid_created"]) for entry in result["entries"]}
            self.assertEqual(len(identities), 8)
            journal = [json.loads(line) for line in
                       (self.root / name / "start_journal.jsonl").read_text().splitlines()]
            self.assertEqual(len(journal), 8)
            self.assertEqual({row["trial_id"] for row in journal}, expected_ids)
            signatures[name] = {}
            for entry in result["entries"]:
                directory = self.root / name / entry["trial_id"]
                proof = s.read_json(directory / "launch.json")
                self.assertEqual(proof["cpu_affinity"], [entry["core"]])
                self.assertEqual(proof["pid"], entry["pid"])
                self.assertEqual(proof["process_created"], entry["pid_created"])
                self.assertIs(proof["affinity_before_worker_import"], True)
                self.assertEqual(proof["thread_environment"], {key: "1" for key in s.THREAD_ENV})
                self.assertGreater(entry["affinity_guard"]["postlaunch_samples"], 0)
                self.assertNotIn("failure_observation", entry["affinity_guard"])
                worker = s.read_json(directory / "worker.json")
                # Only fixture seed/path/cost are compared. Clock, PID and
                # resource fields are deliberately excluded from determinism.
                signatures[name][entry["trial_id"]] = worker["fixed_work_signature"]
        self.assertEqual(signatures["serial"], signatures["parallel"])
        parallel = results["parallel"]
        for block in range(4):
            arms = [entry for entry in parallel["entries"] if entry["block_order"] == block]
            self.assertEqual(len({entry["lane"] for entry in arms}), 1)
        journal_path = self.root / "parallel/start_journal.jsonl"
        journal_before = journal_path.read_bytes()
        resumed = collect("parallel", 4, resume=True)
        self.assertEqual(resumed["entries"], parallel["entries"])
        self.assertEqual(journal_path.read_bytes(), journal_before)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0], *remaining])
