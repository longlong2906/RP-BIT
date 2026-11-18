"""Synthetic runtime/deadline regression; no scientific planner campaign."""
from __future__ import annotations

import functools
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tools"))

from corridor_v4_execution import atomic_json, file_hash, read, run_spawn_lanes
from sem_exposure_runtime import executable_path, runtime_admission

GIB = 1024**3


def synthetic_work(task, *, marker="plain"):
    rng = random.Random(task["seed"])
    signature = [rng.randrange(10**9) for _ in range(50)]
    descendant = None
    if task.get("descendant_record"):
        import psutil
        descendant = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        atomic_json(task["descendant_record"], dict(pid=descendant.pid,
            created=psutil.Process(descendant.pid).create_time(), synthetic_fixture=True))
    try:
        time.sleep(task.get("delay", .1))
    finally:
        if descendant is not None and descendant.poll() is None:
            descendant.terminate()
            descendant.wait(timeout=5)
    return dict(signature=signature, marker=marker, synthetic_fixture=True)


def request_builder_fixture(*args):
    raise RuntimeError("SYNTHETIC_REQUEST_BUILDER_REACHED")


class SEMExposureRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="synthetic_sem_exposure_runtime_", dir=ROOT/"tests/fixtures")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cgroups = self.root/"cgroups"
        self.member = self.root/"membership"
        self.host = SimpleNamespace(total=16*GIB, available=12*GIB)

    def write(self, name, value):
        path = self.cgroups/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(value), encoding="ascii")

    def admission(self, workers=4, reservation=GIB, **options):
        with mock.patch("terrain_compare_scheduler.sys.platform", "linux"), \
             mock.patch("psutil.Process", return_value=SimpleNamespace(cpu_affinity=lambda: list(range(8)))), \
             mock.patch("psutil.virtual_memory", return_value=self.host):
            return runtime_admission(workers, reservation, cgroup_root=self.cgroups, membership=self.member, **options)

    def test_v2_ancestor_quota_and_cpuset_are_both_enforced(self):
        self.member.write_text("0::/parent/child\n")
        self.write("cpu.max", "800000 100000")
        self.write("parent/cpu.max", "250000 100000")
        self.write("parent/child/cpu.max", "max 100000")
        self.write("parent/cpuset.cpus.effective", "2-5")
        result = self.admission()
        self.assertFalse(result["admitted"])
        self.assertEqual(result["effective_cpu_capacity"], 2)
        self.assertEqual(result["eligible_cpus"], [2, 3, 4, 5])
        self.assertEqual(result["reason"], "INSUFFICIENT_CPU_QUOTA_OR_AFFINITY")
        self.write("parent/cpu.max", "400000 100000")
        result = self.admission()
        self.assertTrue(result["admitted"], result)
        self.assertEqual(result["cores"], [2, 3, 4, 5])

    def test_fractional_quota_never_rounds_up_to_one_worker(self):
        self.member.write_text("0::/\n")
        self.write("cpu.max", "50000 100000")
        result = self.admission(workers=1)
        self.assertFalse(result["admitted"])
        self.assertEqual(result["effective_cpu_capacity"], 0)

    def test_v1_cpu_parent_and_memory_headroom(self):
        self.member.write_text("2:cpu,cpuacct:/parent/child\n3:memory:/parent/child\n")
        self.write("cpu,cpuacct/parent/cpu.cfs_quota_us", "400000")
        self.write("cpu,cpuacct/parent/cpu.cfs_period_us", "100000")
        self.write("cpu,cpuacct/parent/child/cpu.cfs_quota_us", "-1")
        self.write("cpu,cpuacct/parent/child/cpu.cfs_period_us", "100000")
        self.write("memory/parent/memory.limit_in_bytes", 8*GIB)
        self.write("memory/parent/memory.usage_in_bytes", 6*GIB)
        self.write("memory/parent/memory.use_hierarchy", 1)
        result = self.admission()
        self.assertFalse(result["admitted"])
        self.assertEqual(result["reason"], "INSUFFICIENT_AVAILABLE_RAM")
        self.assertEqual(result["available_memory_bytes"], 2*GIB)
        self.assertEqual(result["memory_evidence"]["sources"][0]["credited_clean_inactive_bytes"], 0)

    def test_live_memory_ancestors_and_clean_cache_credit(self):
        self.member.write_text("0::/parent/child\n")
        self.write("cpu.max", "400000 100000")
        self.write("parent/memory.max", 8*GIB)
        self.write("parent/memory.current", 6*GIB)
        self.write("parent/memory.stat", f"inactive_file {4*GIB}\nfile {4*GIB}\nshmem 0\nfile_dirty 0\nfile_writeback 0\n")
        self.write("parent/child/memory.max", "max")
        result = self.admission()
        self.assertTrue(result["admitted"], result)
        self.assertEqual(result["available_memory_bytes"], 6*GIB)
        self.write("parent/memory.stat", f"inactive_file {4*GIB}\nfile {4*GIB}\nshmem 0\nfile_dirty {3*GIB}\nfile_writeback 0\n")
        result = self.admission()
        self.assertFalse(result["admitted"])
        self.assertEqual(result["available_memory_bytes"], 3*GIB)

    def test_unreadable_memory_and_invalid_quota_block(self):
        self.member.write_text("0::/\n")
        self.write("cpu.max", "400000 100000")
        self.write("memory.max", 8*GIB)
        result = self.admission()
        self.assertFalse(result["admitted"])
        self.assertEqual(result["reason"], "UNREADABLE_OR_INVALID_RUNTIME_LIMITS")
        self.write("cpu.max", "400000 0")
        self.assertEqual(self.admission()["reason"], "UNREADABLE_OR_INVALID_RUNTIME_LIMITS")

    def test_disk_preflight_and_parameter_validation(self):
        self.member.write_text("0::/\n")
        self.write("cpu.max", "max 100000")
        with mock.patch("sem_exposure_runtime.shutil.disk_usage", return_value=SimpleNamespace(total=100*GIB, free=2*GIB)):
            result = self.admission(required_disk_bytes=3*GIB, disk_path=self.root/"not_created/subdir")
        self.assertFalse(result["admitted"])
        self.assertEqual(result["reason"], "INSUFFICIENT_FREE_DISK")
        self.assertEqual(result["disk_probe_path"], str(self.root))
        for workers, reservation, options in ((2, GIB, {}), (4, 0, {}), (4, GIB, {"required_disk_bytes": 1}),
                                               (4, GIB, {"required_disk_bytes": -1})):
            with self.assertRaises(ValueError):
                runtime_admission(workers, reservation, **options)

    def test_interpreter_path_does_not_resolve_virtualenv_symlinks(self):
        path = self.root/"venv/bin/python"
        with mock.patch.object(Path, "resolve", side_effect=AssertionError("must not dereference executable symlink")):
            self.assertEqual(executable_path(path), Path(os.path.abspath(path)))
        if sys.platform.startswith("linux"):
            subprocess.run([sys.executable, "-m", "venv", "--without-pip", "--symlinks", str(self.root/"venv")], check=True)
            self.assertTrue(path.is_symlink())
            process = subprocess.run([str(executable_path(path)), "-c",
                "import json,sys; print(json.dumps(dict(prefix=sys.prefix,base=sys.base_prefix,executable=sys.executable)))"],
                check=True, capture_output=True, text=True)
            result = json.loads(process.stdout)
            self.assertEqual(result["prefix"], str(self.root/"venv"))
            self.assertNotEqual(result["prefix"], result["base"])
            self.assertEqual(result["executable"], str(path))

    def core(self):
        import psutil
        return psutil.Process().cpu_affinity()[0]

    def test_soft_deadline_drains_active_and_keeps_pending_unstarted(self):
        tasks = [dict(task_id=f"soft_{i}", seed=i, delay=1.) for i in range(3)]
        deadline = time.monotonic()+.5
        result = run_spawn_lanes([tasks], synthetic_work, self.root/"soft", cores=[self.core()],
                                 stop_dispatch_at_monotonic=deadline, terminate_at_monotonic=deadline+10.)
        self.assertEqual(result["status"], "DEADLINE")
        self.assertEqual(result["stop_reason"], "SOFT_SESSION_DEADLINE")
        self.assertEqual([item["task_id"] for item in result["results"]], ["soft_0"])
        self.assertEqual(result["results"][0]["status"], "COMPLETED")
        self.assertEqual(result["unstarted_task_ids"], ["soft_1", "soft_2"])
        self.assertFalse((self.root/"soft/soft_1").exists())

    def test_hard_deadline_interrupts_owned_descendants_and_keeps_receipts(self):
        import psutil
        evidence = self.root/"descendant.json"
        tasks = [dict(task_id="active", seed=1, delay=10., descendant_record=str(evidence)),
                 dict(task_id="pending", seed=2)]
        began = time.monotonic()
        result = run_spawn_lanes([tasks], synthetic_work, self.root/"hard", cores=[self.core()],
                                 terminate_at_monotonic=began+3.)
        self.assertEqual(result["status"], "DEADLINE")
        self.assertEqual(result["stop_reason"], "HARD_SESSION_DEADLINE")
        self.assertEqual(result["unstarted_task_ids"], ["pending"])
        self.assertEqual(result["results"][0]["status"], "INTERRUPTED")
        self.assertTrue(read(self.root/"hard/active/exit.json")["interrupted"])
        self.assertTrue(evidence.is_file(), "target did not reach the owned descendant fixture")
        child = read(evidence)
        try:
            process = psutil.Process(child["pid"])
            live = abs(process.create_time()-child["created"]) < .001 and process.is_running() and process.status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            live = False
        self.assertFalse(live)
        self.assertFalse((self.root/"hard/pending").exists())

    def test_deadline_before_dispatch_and_invalid_deadlines(self):
        tasks = [[dict(task_id="untouched", seed=1)]]
        result = run_spawn_lanes(tasks, synthetic_work, self.root/"expired", cores=[self.core()],
                                 stop_dispatch_at_monotonic=time.monotonic()-1.)
        self.assertEqual(result["results"], [])
        self.assertEqual(result["unstarted_task_ids"], ["untouched"])
        self.assertEqual(result["max_concurrency"], 0)
        for options in ({"stop_dispatch_at_monotonic": math_nan()},
                        {"stop_dispatch_at_monotonic": 2., "terminate_at_monotonic": 1.}):
            with self.assertRaises(ValueError):
                run_spawn_lanes(tasks, synthetic_work, self.root/"bad", cores=[self.core()], **options)

    def test_optional_deadlines_preserve_fixed_work_and_partial_target(self):
        tasks = [dict(task_id=f"fixed_{i}", seed=i) for i in range(3)]
        target = functools.partial(synthetic_work, marker="partial")
        ordinary = run_spawn_lanes([tasks], target, self.root/"plain", cores=[self.core()])
        bounded = run_spawn_lanes([tasks], target, self.root/"bounded", cores=[self.core()],
                                  stop_dispatch_at_monotonic=time.monotonic()+30.,
                                  terminate_at_monotonic=time.monotonic()+60.)
        self.assertEqual(ordinary["status"], "COMPLETE")
        self.assertEqual(bounded["status"], "COMPLETE")
        self.assertEqual([item["payload"] for item in ordinary["results"]],
                         [item["payload"] for item in bounded["results"]])

    def test_collector_request_builder_is_explicit_and_native_path_is_preserved(self):
        import run_sem_exposure as runner
        binary = self.root/"synthetic_binary"
        bridge = self.root/"synthetic_bridge"
        for path in (binary, bridge):
            path.write_text("synthetic fixture, never execute")
        task = dict(directory=str(self.root/"collector"), info={}, entry=dict(method="sem_native", trial_id="synthetic"),
                    worker=str(binary), binary_sha256=file_hash(binary), bridge=str(bridge),
                    bridge_sha256=file_hash(bridge), sem_python=str(self.root/"venv/bin/python"),
                    config={}, profile="tune")
        with mock.patch.object(runner, "_verify_inputs"), \
             mock.patch.object(runner, "_sample_rss", return_value=0), \
             mock.patch.object(runner.subprocess, "Popen", side_effect=AssertionError("no planner launch")):
            result = runner._collect_trial(task, request_builder=request_builder_fixture)
        self.assertEqual(result["execution"]["status"], "FAILED")
        self.assertEqual(result["execution"]["reason"], "SYNTHETIC_REQUEST_BUILDER_REACHED")
        self.assertFalse(result["execution"]["launched"])


def math_nan():
    return float("nan")


if __name__ == "__main__":
    unittest.main()
