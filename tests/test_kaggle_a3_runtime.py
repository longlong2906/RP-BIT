"""No Kaggle job or planner: isolated orchestration/runtime safety regression tests."""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import kaggle_a3 as runtime
from kaggle_a3_bundle import archive, sha256, verify


class KaggleRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="synthetic-kaggle-runtime-", dir=ROOT / "tests/fixtures")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.output = self.root / "output"
        self.output.mkdir()
        self.started = time.monotonic()

    def file(self, name, data="synthetic test fixture"):
        path = self.source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data, encoding="utf-8")
        return path

    def input_archive(self):
        path = self.file("tools/test_source.py", "# synthetic source fixture\n")
        target = self.root / "input.zip"
        archive([("tools/test_source.py", path)], target, "A3_KAGGLE_SOURCE_INPUT",
                {"protocol_id": runtime.PROTOCOL_ID})
        return target

    def ready_runtime(self):
        self.file("build-kaggle/terrain_benchmark")
        self.file("build-kaggle/CMakeCache.txt")
        self.file("build-kaggle/object.o", "must not enter runtime archive")
        self.file(".venv-kaggle/bin/python", "must not enter runtime archive")
        self.file(".deps/install/lib/libompl.so.1.7.0")
        self.file(".deps/install/include/ompl-1.7/ompl/config.h")
        self.file(".deps/boost_1_82_0/boost/version.hpp")
        self.file(".deps/install/include/eigen3/Eigen/src/Core/util/Macros.h")
        runtime.write(self.output / "setup/build_gate.json", {"status": "PASS", "ctest_passed": True})
        runtime.write(self.output / "setup/runtime.json", {"runtime_identity_sha256": "synthetic-test-hash"})
        runtime.write(self.output / "pilot/trial_fixture/attempt.json", {"synthetic_test_fixture": True})

    def test_input_archive_bound_to_extracted_sources(self):
        target = self.input_archive()
        manifest = runtime.verify_source(target, sha256(target), self.source)
        self.assertEqual(manifest["kind"], "A3_KAGGLE_SOURCE_INPUT")
        self.file("tools/test_source.py", "modified")
        with self.assertRaisesRegex(ValueError, "differs"):
            runtime.verify_source(target, sha256(target), self.source)
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            runtime.verify_source(target, "0" * 64, self.source)

    def test_checkpoint_roundtrip_preserves_attempts_and_exact_runtime_without_cache(self):
        original = self.input_archive()
        self.ready_runtime()
        target = runtime.export_checkpoint(self.output, original, sha256(original), "test", root=self.source)
        manifest = verify(target)
        names = {row["path"] for row in manifest["files"]}
        self.assertTrue(manifest["metadata"]["runtime_ready"])
        self.assertIn("input/source.zip", names)
        self.assertIn("runtime/build-kaggle/terrain_benchmark", names)
        self.assertIn("artifacts/pilot/trial_fixture/attempt.json", names)
        self.assertFalse(any("object.o" in name or ".venv" in name or "exports/" in name for name in names))
        restored_root, restored_output = self.root / "restored", self.root / "resumed-output"
        runtime.restore_checkpoint(target, sha256(original), restored_output, root=restored_root)
        self.assertEqual(sha256(restored_root / "build-kaggle/terrain_benchmark"),
                         sha256(self.source / "build-kaggle/terrain_benchmark"))
        self.assertEqual(runtime.read(restored_output / "pilot/trial_fixture/attempt.json"),
                         {"synthetic_test_fixture": True})
        # An identical second restore is safe; a changed file is never overwritten.
        runtime.restore_checkpoint(target, sha256(original), restored_output, root=restored_root)
        conflict = restored_output / "pilot/trial_fixture/attempt.json"
        conflict.write_text("different")
        with self.assertRaisesRegex(ValueError, "conflicts"):
            runtime.restore_checkpoint(target, sha256(original), restored_output, root=restored_root)
        self.assertEqual(conflict.read_text(), "different")
        with self.assertRaisesRegex(ValueError, "different source"):
            runtime.restore_checkpoint(target, "f" * 64, self.root / "other", root=self.root / "other-root")

    def test_failed_bootstrap_checkpoint_is_explicitly_not_resumable(self):
        original = self.input_archive()
        target = runtime.export_checkpoint(self.output, original, sha256(original), "BLOCKED", root=self.source)
        self.assertFalse(verify(target)["metadata"]["runtime_ready"])
        with self.assertRaisesRegex(ValueError, "not resumable"):
            runtime.restore_checkpoint(target, sha256(original), self.root / "other", root=self.root / "other-root")

    def test_exports_are_immutable_and_latest_is_only_pointer(self):
        original = self.input_archive()
        self.ready_runtime()
        first = runtime.export_checkpoint(self.output, original, sha256(original), "before_pilot", root=self.source)
        first_hash = sha256(first)
        second = runtime.export_checkpoint(self.output, original, sha256(original), "PARTIAL", root=self.source)
        self.assertNotEqual(first, second)
        self.assertEqual(sha256(first), first_hash)
        self.assertEqual(runtime.read(self.output / "exports/latest.json")["path"], second.name)

    def test_export_respects_output_quota_even_when_disk_has_free_space(self):
        original = self.input_archive()
        self.ready_runtime()
        attempt = self.output / "pilot/trial_fixture/attempt.json"
        before = sha256(attempt)
        with patch.object(runtime, "output_size", return_value=runtime.MAX_OUTPUT_BYTES - 1), \
                patch.object(runtime.shutil, "disk_usage") as disk:
            disk.return_value.free = 100 * 1024**3
            with self.assertRaisesRegex(RuntimeError, "retained-output quota"):
                runtime.export_checkpoint(self.output, original, sha256(original), "PARTIAL", root=self.source)
        self.assertEqual(sha256(attempt), before)
        self.assertFalse((self.output / "exports").exists())

    def test_collection_reserves_next_full_export_and_old_exports_count_toward_quota(self):
        original = self.input_archive()
        self.ready_runtime()
        input_bytes = original.stat().st_size
        before = runtime.reserved_checkpoint_bytes(self.output, input_bytes=input_bytes, root=self.source)
        old_export = self.output / "exports/old.zip"
        old_export.parent.mkdir()
        old_export.write_bytes(b"old archive" * 100)
        self.assertEqual(runtime.reserved_checkpoint_bytes(self.output, input_bytes=input_bytes, root=self.source), before)
        self.assertGreater(runtime.output_size(self.output), old_export.stat().st_size)
        used = runtime.output_size(self.output)
        # The collector stops while there is still enough quota for the full
        # export, rather than waiting until raw data alone reaches the limit.
        quota = used + before + runtime.EXPORT_GROWTH_HEADROOM_BYTES // 2
        with patch.object(runtime, "MAX_OUTPUT_BYTES", quota), \
                patch.object(runtime.shutil, "disk_usage") as disk:
            disk.return_value.free = 100 * 1024**3
            with self.assertRaisesRegex(RuntimeError, "reserves space for a full checkpoint"):
                runtime.guard_disk(self.output, input_bytes=input_bytes, root=self.source)
            checkpoint = runtime.export_checkpoint(self.output, original, sha256(original), "PARTIAL", root=self.source)
        self.assertTrue(checkpoint.is_file())
        self.assertTrue(verify(checkpoint)["metadata"]["runtime_ready"])

    def test_collection_reserves_full_export_on_filesystem_as_well_as_output_quota(self):
        original = self.input_archive()
        self.ready_runtime()
        needed = runtime.reserved_checkpoint_bytes(self.output, input_bytes=original.stat().st_size, root=self.source)
        with patch.object(runtime.shutil, "disk_usage") as disk:
            disk.return_value.free = runtime.MIN_FREE_BYTES + needed
            with self.assertRaisesRegex(RuntimeError, "disk guard"):
                runtime.guard_disk(self.output, input_bytes=original.stat().st_size, root=self.source)
            disk.return_value.free += runtime.EXPORT_GROWTH_HEADROOM_BYTES + 1
            runtime.guard_disk(self.output, input_bytes=original.stat().st_size, root=self.source)

    def test_real_command_logs_success_failure_and_timeout_without_overwrite(self):
        console = io.StringIO()
        with contextlib.redirect_stdout(console):
            stdout = runtime.run_command("fixture", [sys.executable, "-c", "print('fixture-ok')"],
                                         self.output, self.started, cwd=self.root)
            self.assertEqual(stdout, "fixture-ok")
            with self.assertRaisesRegex(RuntimeError, "exited 7"):
                runtime.run_command("fixture", [sys.executable, "-c",
                    "import sys; print('diagnostic on stdout'); print('diagnostic on stderr',file=sys.stderr); raise SystemExit(7)"],
                                    self.output, self.started, cwd=self.root)
            began = time.monotonic()
            with self.assertRaisesRegex(RuntimeError, "deadline"):
                runtime.run_command("timeout", [sys.executable, "-c", "import time; time.sleep(30)"],
                                    self.output, self.started, cwd=self.root, max_seconds=.2)
            self.assertLess(time.monotonic() - began, 5)
        self.assertIn("diagnostic on stdout", console.getvalue())
        self.assertIn("diagnostic on stderr", console.getvalue())
        logs = [runtime.read(path) for path in (self.output / "setup/commands").glob("*.json")]
        self.assertEqual(len(logs), 3)
        self.assertEqual(sum(row["exit_code"] == 0 for row in logs), 1)
        self.assertEqual(sum(row["exit_code"] == 7 for row in logs), 1)
        self.assertTrue(any("deadline" in row.get("error", "") for row in logs))

    def test_expired_deadline_and_disk_guard_do_not_launch_commands(self):
        with patch.object(runtime.subprocess, "Popen") as spawn:
            with self.assertRaisesRegex(RuntimeError, "deadline"):
                runtime.run_command("late", ["must-not-run"], self.output,
                    time.monotonic() - runtime.HARD_SECONDS, cwd=self.root)
            with patch.object(runtime.shutil, "disk_usage") as usage:
                usage.return_value.free = 0
                with self.assertRaisesRegex(RuntimeError, "disk guard"):
                    runtime.run_command("no-disk", ["must-not-run"], self.output, self.started, cwd=self.root)
            spawn.assert_not_called()

    def test_bootstrap_rejects_windows_and_missing_compiler_before_installs(self):
        args = argparse.Namespace(input_archive="unused", input_sha256="unused", output=self.output)
        with patch.object(runtime.sys, "platform", "win32"), patch.object(runtime, "isolated_python") as install:
            with self.assertRaisesRegex(RuntimeError, "requires Linux"):
                runtime.bootstrap(args)
            install.assert_not_called()
        with patch.object(runtime.sys, "platform", "linux"), patch.object(runtime.shutil, "which", return_value=None), \
                patch.object(runtime, "isolated_python") as install:
            with self.assertRaisesRegex(RuntimeError, "g\\+\\+"):
                runtime.bootstrap(args)
            install.assert_not_called()

    def test_failure_finalizer_preserves_error_and_exports_evidence(self):
        target = self.input_archive()
        calls = ["--stage", "execute", "--input-archive", str(target), "--input-sha256", sha256(target),
                 "--output", str(self.output), "--notebook-started-monotonic", str(self.started)]
        with patch.object(runtime, "execute", side_effect=RuntimeError("BLOCKED: fixture dependency")), \
                patch.object(runtime, "export_checkpoint", return_value=Path("checkpoint.zip")) as export, \
                contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(runtime.main(calls), 1)
        state = runtime.read(self.output / "RUN_STATUS.json")
        self.assertEqual(state["status"], "BLOCKED")
        self.assertEqual(state["pilot_status"], "NOT_RUN")
        self.assertIn("fixture dependency", state["error"])
        export.assert_called_once()

    def test_handoff_keeps_explicit_authorization_and_original_session_clock(self):
        args = argparse.Namespace(input_archive=Path("source.zip"), input_sha256="f" * 64,
            output=self.output, notebook_started_monotonic=123., allow_a3_pilot=False,
            resume_archive=None)
        command = runtime.execution_command(args, "/isolated/bin/python")
        self.assertEqual(command[0], "/isolated/bin/python")
        self.assertEqual(command[command.index("--stage") + 1], "execute")
        self.assertEqual(command[command.index("--notebook-started-monotonic") + 1], "123.0")
        self.assertNotIn("--allow-a3-pilot", command)
        args.allow_a3_pilot, args.resume_archive = True, Path("previous.zip")
        command = runtime.execution_command(args, "/isolated/bin/python")
        self.assertIn("--allow-a3-pilot", command)
        self.assertEqual(command[command.index("--resume-archive") + 1], "previous.zip")

    def test_image_provenance_reads_only_allowlisted_identifiers(self):
        release = self.file("os-release", "NAME=FixtureLinux\n")
        with patch.dict(os.environ, {"KAGGLE_DOCKER_IMAGE": "fixture-image", "SECRET_TOKEN": "never-record"}, clear=True):
            observed = runtime.image_evidence(release)
        self.assertEqual(observed["os_release"], "NAME=FixtureLinux\n")
        self.assertEqual(observed["image_identifiers"], {"KAGGLE_DOCKER_IMAGE": "fixture-image"})
        self.assertNotIn("never-record", json.dumps(observed))
        with patch.dict(os.environ, {}, clear=True):
            missing = runtime.image_evidence(self.source / "missing")
        self.assertIsNone(missing["os_release"])
        self.assertEqual(missing["image_identifier_status"], "unavailable")

    def test_execute_orders_correctness_gates_and_requires_analyzer_acceptance(self):
        self.ready_runtime()
        library = self.source / ".deps/install/lib/libompl.so.1.7.0"
        # Test orchestration, not planner data: child commands are recorded
        # in memory and their minimal synthetic receipts stay under fixtures.
        for report_status, expected_status in (("COMPLETE_VALIDATED", "COMPLETE"),
                                               ("PARTIAL", "PARTIAL"),
                                               ("FAILED_INTEGRITY", "FAILED")):
            with self.subTest(report_status=report_status):
                output = self.root / ("sequence-" + report_status)
                output.mkdir()
                args = argparse.Namespace(output=output, notebook_started_monotonic=self.started,
                    input_archive=self.root / "source.zip", input_sha256="f" * 64,
                    resume_archive=None, allow_a3_pilot=True)
                calls = []

                def child(name, argv, *_args, **_kwargs):
                    calls.append(name)
                    command = list(map(str, argv))
                    if name == "ctest":
                        self.assertFalse((output / "setup/build_gate.json").exists())
                        self.assertNotIn("lock_780_trial_plan", calls)
                    elif name.startswith("generate_"):
                        profile = command[command.index("--profile") + 1]
                        runtime.write(output / "data" / profile / "manifest.json", {"synthetic_test_fixture": True})
                    elif name == "collect_smoke":
                        self.assertIn("ctest", calls)
                        gate = runtime.read(output / "setup/build_gate.json")
                        self.assertIs(gate["ctest_passed"], True)
                        runtime.write(output / "smoke/campaign_state.json", {
                            "status": "COMPLETE_VALIDATED", "counts": {"planned": 52, "offline_validated": 52}})
                        runtime.write(output / "smoke/smoke_gate.json", {
                            "status": "PASS", "planned_trials": 52, "validated_trials": 52})
                    elif name == "lock_780_trial_plan":
                        self.assertIn("--dry-run", command)
                        self.assertIn("collect_smoke", calls)
                        smoke = runtime.read(output / "smoke/smoke_gate.json")
                        self.assertEqual((smoke["planned_trials"], smoke["validated_trials"]), (52, 52))
                        runtime.write(output / "pilot/manifest.json", {"synthetic_test_fixture": True, "planned_trials": 780})
                    elif name == "collect_a3_pilot":
                        self.assertIn("--allow-a3-pilot", command)
                        self.assertIn("--resume", command)
                        self.assertIn("checkpoint_before_pilot", calls)
                        runtime.write(output / "pilot/campaign_state.json", {
                            "status": "COMPLETE_VALIDATED", "counts": {"planned": 780, "offline_validated": 780}})
                    elif name == "analyze_pilot":
                        runtime.write(output / "report/summary.json", {"status": report_status})
                    return ""

                def checkpoint(*_args, **_kwargs):
                    calls.append("checkpoint_before_pilot")
                    return Path("synthetic-checkpoint.zip")

                def source_hash():
                    self.assertIn("ctest", calls)
                    return "test-source-hash"

                with contextlib.ExitStack() as stack:
                    for target, value in (("kaggle_a3.ROOT", self.source),
                            ("setup_cloud_benchmark.require_prerequisites", lambda: None),
                            ("setup_cloud_benchmark.verify_pinned_archives", lambda: {}),
                            ("setup_cloud_benchmark.observed_dependency_versions", lambda: {}),
                            ("setup_cloud_benchmark.project_configure", lambda *_: ["cmake"]),
                            ("experiment_common.cloud_code_hash", source_hash),
                            ("experiment_common.cloud_environment", lambda *_: {"cpu": "test-cpu"}),
                            ("experiment_common.cloud_ompl_library", lambda *_: library),
                            ("a3_pilot_campaign.runtime_identity", lambda *_: {"fixture": True}),
                            ("kaggle_a3.verify_source", lambda *_: {}),
                            ("kaggle_a3.run_command", child),
                            ("kaggle_a3.export_checkpoint", checkpoint)):
                        stack.enter_context(patch(target, value))
                    result = runtime.execute(args)
                self.assertEqual(result["status"], expected_status)
                self.assertEqual(result["analysis_status"], report_status)
                self.assertLess(calls.index("ctest"), calls.index("lock_780_trial_plan"))
                self.assertLess(calls.index("collect_smoke"), calls.index("collect_a3_pilot"))
                self.assertLess(calls.index("collect_a3_pilot"), calls.index("analyze_pilot"))


if __name__ == "__main__":
    unittest.main()
