"""G5 integrity regressions with synthetic fixtures outside experimental results.

No planner is launched and no synthetic observation enters results/ or reports/.
The tests exercise the real G5 orchestration with only environment gates and the
worker subprocess boundary replaced by controlled fixtures.
"""
import contextlib
import copy
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import g5_run as g5
from experiment_common import ROOT, digest, dump, file_hash, read_json


class G5RunnerTests(unittest.TestCase):
    def setUp(self):
        parent = ROOT / "build/g5_runner_regression"
        parent.mkdir(parents=True, exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix="synthetic_", dir=parent))
        (self.root / "FIXTURE.txt").write_text("SYNTHETIC TEST FIXTURE; no experimental observations.\n")
        self.out = self.root / "fixture_run"
        self.out.mkdir()
        self.log = self.root / "fixture_logs"
        self.log.mkdir()
        self.data = self.root / "fixture_data"
        self.data.mkdir()
        self.worker = self.root / "worker_fixture.exe"
        self.worker.write_bytes(b"SYNTHETIC NONEXECUTABLE WORKER IDENTITY")
        self.dll = self.root / ".deps/install/lib/libompl.dll"
        self.dll.parent.mkdir(parents=True)
        self.dll.write_bytes(b"SYNTHETIC NONEXECUTABLE DEPENDENCY IDENTITY")
        self.protocol = self.root / "protocol_fixture.md"
        self.protocol.write_text("SYNTHETIC frozen protocol\n")
        self.map_path = self.data / "map_fixture.t25d"
        self.map_path.write_bytes(b"SYNTHETIC NONPARSED MAP IDENTITY")
        self.config = {"profiles": {"full": {"planning_budget_s": 5.}}}
        self.maps = {"maps": [{"path": str(self.map_path), "sha256": file_hash(self.map_path)}]}
        dump(self.data / "manifest.json", self.maps)
        self.identity = {"code_sha256": "synthetic_source_identity", "worker_sha256": file_hash(self.worker),
                         "ompl_library_sha256": file_hash(self.dll), "config_hash": digest(self.config),
                         "map_manifest_sha256": digest(self.maps),
                         "g5_frozen_files": {self.protocol.name: file_hash(self.protocol)}}
        self.base_identity = {key: value for key, value in self.identity.items() if key != "g5_frozen_files"}
        self.entries = [{"trial_id": "fixture_" + str(i), "map_path": str(self.map_path),
                         "map_hash": file_hash(self.map_path), "shuffle_order": i,
                         "planner_id": "bitstar_a0", "planner_seed": i + 1,
                         "query_id": "synthetic_query", "map_id": "synthetic_map"} for i in range(4)]
        self.manifest = {"schema_version": 2, "experiment_id": self.out.name,
                         "profile": "full", "status": "LOCKED", "identity": copy.deepcopy(self.identity),
                         "planned_trials": len(self.entries), "trials": self.entries}
        dump(self.out / "manifest.json", self.manifest)
        dump(self.out / "manifest.lock.json", {"sha256": file_hash(self.out / "manifest.json"),
                                               "identity_sha256": digest(self.identity)})
        dump(self.out / "environment.json", {"code_sha256": self.identity["code_sha256"], "code_commit": None})
        archive = self.out / "source_at_run.zip"
        archive.write_bytes(b"SYNTHETIC HASH FIXTURE, NOT A SOURCE ARCHIVE")
        dump(self.out / "source_manifest.json", {"archive_sha256": file_hash(archive)})
        patches = {"ROOT": self.root, "OUT": self.out, "LOG": self.log, "DATA": self.data,
                   "CONFIG": self.root / "config_fixture.yaml"}
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        for name, value in patches.items():
            self.stack.enter_context(patch.object(g5, name, value))
        self.stack.enter_context(patch.object(g5, "default_worker", return_value=self.worker))
        self.stack.enter_context(patch.object(g5, "code_hash", return_value=self.identity["code_sha256"]))
        self.stack.enter_context(patch.object(g5, "full_config", return_value=self.config))
        self.stack.enter_context(patch.object(g5, "provenance_identity", return_value=self.base_identity))
        self.stack.enter_context(patch.object(g5, "gate_matches"))

    def invoke_run(self, worker):
        with patch.object(g5, "run_trial", side_effect=worker) as mocked, contextlib.redirect_stdout(io.StringIO()):
            g5.run()
        return mocked

    def fake_worker(self, entry, config, directory, worker, env, experiment_id, config_hash, **kwargs):
        self.assertEqual(kwargs, {"profile": "full"})
        directory.mkdir(exist_ok=False)
        record = dict(entry, execution_status="TIMEOUT", planner_status="NO_SOLUTION", success=False,
                      path_available=False, termination_reason="TIME_BUDGET", fixture=True,
                      attempt_history=[{"event": "SYNTHETIC_ATTEMPT"}])
        dump(directory / "trial.json", record)
        return record

    def test_identity_accepts_exact_frozen_bytes(self):
        g5.verify_identity(self.manifest)
        g5.verify_identity(self.manifest, all_inputs=False)

    def test_each_frozen_input_mismatch_blocks(self):
        paths = [(self.worker, "source/binary"), (self.dll, "linked dependency"),
                 (self.protocol, "frozen input"), (self.map_path, "map changed"),
                 (self.out / "source_at_run.zip", "source archive")]
        for path, message in paths:
            with self.subTest(path=path.name):
                original = path.read_bytes()
                path.write_bytes(original + b" MUTATED")
                try:
                    with self.assertRaisesRegex(RuntimeError, message):
                        g5.verify_identity(self.manifest)
                finally:
                    path.write_bytes(original)
        with patch.object(g5, "code_hash", return_value="changed_source"):
            with self.assertRaisesRegex(RuntimeError, "source/binary"):
                g5.verify_identity(self.manifest)

    def test_manifest_and_identity_mutations_block(self):
        original = (self.out / "manifest.json").read_bytes()
        changed = copy.deepcopy(self.manifest)
        changed["trials"][0]["planner_seed"] = 999
        dump(self.out / "manifest.json", changed)
        with self.assertRaisesRegex(RuntimeError, "manifest changed"):
            g5.verify_identity(changed)
        (self.out / "manifest.json").write_bytes(original)
        changed = copy.deepcopy(self.manifest)
        changed["identity"]["code_sha256"] = "different_in_memory_identity"
        with self.assertRaisesRegex(RuntimeError, "identity changed"):
            g5.verify_identity(changed)

    def test_existing_failures_retained_and_only_absent_trial_runs_once(self):
        preserved = {}
        for entry, status in zip(self.entries, ("CRASH", "TIMEOUT", "WATCHDOG")):
            directory = self.out / entry["trial_id"]
            directory.mkdir()
            record = dict(entry, execution_status=status, success=False, fixture=True,
                          path_available=status == "COMPLETED", independent_valid=False if status == "COMPLETED" else None)
            dump(directory / "trial.json", record)
            (directory / "stdout.log").write_text("SYNTHETIC retained failure log\n")
            preserved.update({path: path.read_bytes() for path in directory.iterdir()})
        manifest_before = (self.out / "manifest.json").read_bytes()
        first = self.invoke_run(self.fake_worker)
        self.assertEqual(first.call_count, 1)
        self.assertEqual(first.call_args.args[0]["trial_id"], self.entries[3]["trial_id"])
        second = self.invoke_run(self.fake_worker)
        self.assertEqual(second.call_count, 0)
        for path, content in preserved.items():
            self.assertEqual(path.read_bytes(), content)
        self.assertEqual((self.out / "manifest.json").read_bytes(), manifest_before)
        state = read_json(self.out / "execution_state.json")
        self.assertEqual(state["counts"]["executed"], 4)
        self.assertEqual(state["counts"]["not_run"], 0)
        self.assertEqual(state["counts"]["crash"], 1)
        self.assertEqual(state["counts"]["invalid"], 0)

    def test_known_invalid_stops_further_trials(self):
        directory = self.out / self.entries[0]["trial_id"]
        directory.mkdir()
        dump(directory / "trial.json", dict(self.entries[0], execution_status="COMPLETED", success=False,
             path_available=True, independent_valid=False))
        with patch.object(g5, "run_trial") as worker:
            with self.assertRaisesRegex(RuntimeError, "previously detected invalid"):
                g5.run()
            worker.assert_not_called()

    def test_concurrent_execution_is_rejected(self):
        with g5.exclusive_execution():
            with self.assertRaises((RuntimeError, BlockingIOError)):
                with g5.exclusive_execution():
                    self.fail("second lock unexpectedly acquired")

    def test_normal_interruption_resumes_only_not_run(self):
        def interrupted(entry, config, directory, *args, **kwargs):
            directory.mkdir(exist_ok=False)
            dump(directory / "attempt.json", {"event": "SYNTHETIC_PROCESS_STARTED", "fixture": True})
            (directory / "stderr.log").write_text("SYNTHETIC interruption evidence\n")
            raise KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            self.invoke_run(interrupted)
        state = read_json(self.out / "execution_state.json")
        self.assertEqual(state["status"], "PARTIAL")
        interrupted_dir = self.out / self.entries[0]["trial_id"]
        evidence = {p: p.read_bytes() for p in interrupted_dir.iterdir()}
        resumed = self.invoke_run(self.fake_worker)
        self.assertEqual(resumed.call_count, 3)
        record = read_json(interrupted_dir / "trial.json")
        self.assertEqual(record["execution_status"], "CRASH")
        self.assertEqual(record["termination_reason"], "INTERRUPTED_PREVIOUS_RUN")
        for path, content in evidence.items():
            self.assertEqual(path.read_bytes(), content)
        self.assertEqual(self.invoke_run(self.fake_worker).call_count, 0)

    def test_stop_gate_prevents_next_attempt_and_records_blocked(self):
        self.worker.write_bytes(b"MUTATED SYNTHETIC WORKER")
        with patch.object(g5, "run_trial") as worker:
            with self.assertRaisesRegex(RuntimeError, "source/binary"):
                g5.run()
        worker.assert_not_called()
        state = read_json(self.out / "execution_state.json")
        self.assertEqual(state["status"], "BLOCKED")
        self.assertEqual(state["counts"]["not_run"], 4)
        self.assertFalse(any((self.out / entry["trial_id"]).exists() for entry in self.entries))

    def test_full_actions_require_explicit_flag(self):
        for action in ("freeze", "run"):
            with self.subTest(action=action):
                with patch.object(sys, "argv", ["g5_run.py", action]), patch.object(g5, action) as operation:
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                        g5.main()
                    self.assertEqual(caught.exception.code, 2)
                    operation.assert_not_called()
                with patch.object(sys, "argv", ["g5_run.py", action, "--allow-full"]), patch.object(g5, action) as operation:
                    g5.main()
                    operation.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
