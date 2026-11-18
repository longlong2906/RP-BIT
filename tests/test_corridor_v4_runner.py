"""Synthetic orchestration fixtures only; no scientific planner benchmark."""
from __future__ import annotations

import json
import os
from pathlib import Path
import random
import sys
import tempfile
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from corridor_v4_execution import (THREAD_ENV, atomic_json, read, resource_preflight,
    retry_file_access, run_spawn_lanes, append_journal, digest, file_hash)
from run_corridor_v4 import execution_gate, prepare, run, main, _reconcile, actual_planner_overlap


def synthetic_task(task):
    import psutil
    if task.get("fail"):
        raise ValueError("synthetic intentional failure")
    rng = random.Random(task["seed"])
    signature = [rng.randrange(10**9) for _ in range(20)]
    # Enough overlap to prove actual four-process execution, independent of
    # spawn/import speed. This fixture never creates an experimental result.
    time.sleep(task.get("delay", .25))
    return dict(synthetic_fixture=True, signature=signature, pid=os.getpid(),
                affinity=psutil.Process().cpu_affinity(),
                environment={key: os.environ[key] for key in THREAD_ENV})


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="corridor_v4_runner_synthetic_", dir=ROOT / "tests/fixtures")
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def cores(self):
        import psutil
        cores = psutil.Process().cpu_affinity()
        if len(cores) < 4:
            self.skipTest("four eligible CPUs required for four-spawn evidence")
        return cores[:4]

    def test_profile_gates_are_before_io(self):
        for profile in ("pilot", "full"):
            with self.assertRaisesRegex(ValueError, "requires"):
                run("missing", "missing", profile=profile)
        execution_gate("pilot", allow_pilot=True)
        execution_gate("full", allow_full=True)

    def test_inventory_is_registered_four_workers_and_never_launches(self):
        with mock.patch("subprocess.Popen", side_effect=AssertionError("unexpected planner")):
            result = prepare(self.root / "inventory", profile="smoke", dry_run=True)
        self.assertEqual(result["planned_trials"], 42)
        self.assertEqual(result["timing"]["max_workers"], 4)
        self.assertEqual(result["timing"]["validation_workers"], 4)
        self.assertEqual(result["maps"], [])
        self.assertFalse((self.root / "inventory/inputs").exists())
        with self.assertRaises(FileExistsError):
            prepare(self.root / "inventory", dry_run=True)
        with self.assertRaisesRegex(ValueError, "not prepared"):
            run(self.root / "inventory", "missing")

    def test_worker_override_is_frozen_and_invalid_counts_rejected(self):
        result = prepare(self.root / "serial", dry_run=True, workers=1, validation_workers=1)
        self.assertEqual(result["timing"]["max_workers"], 1)
        with self.assertRaisesRegex(ValueError, "worker count"):
            run(self.root / "serial", "missing", workers=4)
        with self.assertRaises(ValueError):
            prepare(self.root / "bad", dry_run=True, workers=2)
        self.assertFalse((self.root / "bad").exists())

    def test_worker_override_preserves_registered_beam_scheduler(self):
        from corridor_v4_protocol import default_config
        config = default_config(beam_scheduler="length_first_stable")
        path = self.root / "label_churn.json"
        atomic_json(path, config)
        prepared = prepare(self.root / "stable", path, dry_run=True, workers=1, validation_workers=1)
        actual = read(self.root / "stable/config.json")
        self.assertEqual(actual["corridor_policy"]["beam_scheduler"], "length_first_stable")
        self.assertEqual(actual["guidance"]["equal_score_policy"], "keep_first")
        self.assertEqual(prepared["timing"]["max_workers"], 1)

    def test_l0_reuse_is_smoke_preparation_only(self):
        for profile, dry_run in (("pilot", False), ("full", False), ("smoke", True)):
            with self.assertRaisesRegex(ValueError, "actual smoke preparation"):
                prepare(self.root / "forbidden", profile=profile, dry_run=dry_run, reuse_smoke_l0="missing")
        self.assertFalse((self.root / "forbidden").exists())
        with mock.patch("sys.stderr"):
            for command in ("run", "inventory"):
                with self.assertRaises(SystemExit) as error:
                    main([command, "--output", str(self.root / "missing"), "--reuse-smoke-l0", "missing"])
                self.assertEqual(error.exception.code, 2)

    def test_l0_cache_bytes_are_frozen_before_any_trial_launch(self):
        output = self.root / "cache_tamper"
        manifest = prepare(output, dry_run=True)
        bundle = dict(synthetic_fixture=True, reason="integrity_guard_only_never_seeded")
        atomic_json(output / "smoke_l0_cache.json", bundle)
        manifest.update(status="PREPARED", smoke_l0_cache=dict(file="smoke_l0_cache.json",
            file_sha256=file_hash(output / "smoke_l0_cache.json"), content_sha256=digest(bundle)))
        atomic_json(output / "manifest.json", manifest)
        atomic_json(output / "smoke_l0_cache.json", dict(bundle, altered=True))
        with mock.patch("subprocess.Popen", side_effect=AssertionError("unexpected planner")):
            with self.assertRaisesRegex(ValueError, "cache bytes changed"):
                run(output, "missing")

    def test_four_spawn_lanes_fixed_work_affinity_and_parent_callbacks(self):
        cores = self.cores()
        tasks = [dict(task_id="s" + str(seed), seed=seed, delay=.6) for seed in (11, 22, 33, 44)]
        callbacks = []
        parallel = run_spawn_lanes([[task] for task in tasks], synthetic_task, self.root / "parallel", cores=cores,
            on_start=lambda task, value: callbacks.append(("start", os.getpid(), task["task_id"])),
            on_complete=lambda task, value: callbacks.append(("complete", os.getpid(), task["task_id"])))
        serial = run_spawn_lanes([tasks], synthetic_task, self.root / "serial", cores=[cores[0]])
        self.assertEqual(parallel["status"], "COMPLETE")
        self.assertEqual(parallel["max_concurrency"], 4)
        self.assertLessEqual(max(item["concurrency"] for item in parallel["concurrency_timeline"]), 4)
        self.assertEqual(len({item["payload"]["pid"] for item in parallel["results"]}), 4)
        self.assertEqual({pid for _, pid, _ in callbacks}, {os.getpid()})
        left = {item["task_id"]: item["payload"]["signature"] for item in parallel["results"]}
        right = {item["task_id"]: item["payload"]["signature"] for item in serial["results"]}
        self.assertEqual(left, right)
        for index, task in enumerate(tasks):
            directory = self.root / "parallel" / task["task_id"]
            proof = read(directory / "child.json")
            self.assertEqual(proof["cpu_affinity"], [cores[index]])
            self.assertTrue(proof["affinity_before_backend_import"])
            self.assertEqual(set(proof["thread_environment"].values()), {"1"})
            self.assertEqual(read(directory / "exit.json")["exit_code"], 0)
            self.assertEqual(read(directory / "start.json")["task_id"], task["task_id"])

    def test_failure_stops_dispatch_and_preserves_exit_receipts(self):
        cores = self.cores()
        lanes = [[dict(task_id=f"a{i}", seed=i, fail=i == 0, delay=.6),
                  dict(task_id=f"b{i}", seed=i + 10)] for i in range(4)]
        result = run_spawn_lanes(lanes, synthetic_task, self.root / "failed", cores=cores)
        self.assertEqual(result["status"], "FAILED_INFRASTRUCTURE")
        self.assertEqual({item["task_id"] for item in result["results"]}, {f"a{i}" for i in range(4)})
        self.assertFalse(any((self.root / "failed" / f"b{i}").exists() for i in range(4)))
        for i in range(4):
            self.assertTrue((self.root / "failed" / f"a{i}" / "exit.json").is_file())

    def test_duplicate_tasks_and_existing_control_refuse_relaunch(self):
        cores = self.cores()
        task = dict(task_id="same", seed=11)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            run_spawn_lanes([[task, task]], synthetic_task, self.root / "duplicate", cores=[cores[0]])
        (self.root / "existing/same").mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            run_spawn_lanes([[task]], synthetic_task, self.root / "existing", cores=[cores[0]])

    def test_ram_preflight_and_windows_atomic_retry(self):
        from types import SimpleNamespace
        with mock.patch("psutil.virtual_memory", return_value=SimpleNamespace(available=100, total=1000)), \
             mock.patch("psutil.Process", return_value=SimpleNamespace(cpu_affinity=lambda: [0, 1, 2, 3])):
            admission = resource_preflight(4, 30)
        self.assertFalse(admission["admitted"])
        self.assertEqual(admission["reason"], "INSUFFICIENT_AVAILABLE_RAM")
        error = PermissionError("synthetic busy Windows reader")
        error.winerror = 32
        operation = mock.Mock(side_effect=[error, "written"])
        with mock.patch("corridor_v4_execution.time.sleep"):
            self.assertEqual(retry_file_access(operation), "written")
        self.assertEqual(operation.call_count, 2)
        atomic_json(self.root / "atomic.json", dict(synthetic_fixture=True))
        self.assertEqual(read(self.root / "atomic.json"), dict(synthetic_fixture=True))

    def test_interrupt_retains_four_started_children_without_retry(self):
        cores = self.cores()
        lanes = [[dict(task_id=f"active{i}", seed=i, delay=2.),
                  dict(task_id=f"pending{i}", seed=i + 10)] for i in range(4)]
        with mock.patch("corridor_v4_execution.time.sleep", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                run_spawn_lanes(lanes, synthetic_task, self.root / "interrupted", cores=cores)
        summary = read(self.root / "interrupted/summary.json")
        self.assertEqual(summary["status"], "INTERRUPTED")
        self.assertEqual(len(summary["results"]), 4)
        for i in range(4):
            self.assertTrue((self.root / "interrupted" / f"active{i}" / "exit.json").is_file())
            self.assertFalse((self.root / "interrupted" / f"pending{i}").exists())

    def test_resume_reconciles_started_without_inventing_exit_or_relaunching(self):
        output = self.root / "resume"
        manifest = dict(sources={"synthetic": "known"}, config_sha256="synthetic-config",
                        entries=[dict(trial_id="started", status="STARTED"),
                                 dict(trial_id="unstarted", status="NOT_RUN")])
        append_journal(output / "execution_control/start_journal.jsonl", dict(trial_id="started",
            started_utc="2026-11-11T00:00:00+00:00", config_sha256=manifest["config_sha256"],
            sources_sha256=digest(manifest["sources"])))
        with mock.patch("corridor_v4_validation.record_corridor_receipt", return_value=None), \
             mock.patch("subprocess.Popen", side_effect=AssertionError("unexpected retry")):
            _reconcile(output, manifest)
            _reconcile(output, manifest)
        self.assertEqual(manifest["entries"][0]["status"], "RAW_READY")
        self.assertEqual(manifest["entries"][1]["status"], "NOT_RUN")
        execution = read(output / "trials/started/execution_recovery.json")
        self.assertEqual(execution["status"], "INTERRUPTED")
        self.assertIsNone(execution["exit_code"])
        self.assertFalse((output / "trials/unstarted").exists())
        manifest["sources"] = {"synthetic": "modified"}
        with self.assertRaisesRegex(ValueError, "provenance"):
            _reconcile(output, manifest)

    def test_actual_planner_overlap_excludes_wrapper_only_counts(self):
        entries = [dict(status="COMPLETED", planner_interval=dict(
            launch_utc=f"2026-11-11T00:00:0{i}+00:00",
            process_finished_utc=f"2026-11-11T00:00:0{i+2}+00:00")) for i in range(4)]
        entries.append(dict(status="INTERRUPTED"))
        measured = actual_planner_overlap(entries)
        self.assertEqual(measured["max_concurrency"], 2)
        self.assertEqual(measured["recorded_intervals"], 4)
        self.assertEqual(measured["unavailable_intervals"], 1)


if __name__ == "__main__":
    unittest.main()
