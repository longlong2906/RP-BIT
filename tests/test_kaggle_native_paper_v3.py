"""Schema-12 packaging contracts. Synthetic archives stay under tests/fixtures."""
from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import kaggle_native_paper_v3 as runtime
import kaggle_native_paper_v3_bundle as bundle


class KaggleNativeV3(unittest.TestCase):
    def setUp(self):
        # WSL's /mnt/d can be case-insensitive. Exact Linux-runtime archives
        # must be restored on a case-sensitive filesystem; keep the production
        # policy intact and allow an explicit temporary test-fixture location.
        fixtures = Path(os.environ.get("NATIVE_PAPER_V3_TEST_TMP", ROOT / "tests/fixtures"))
        fixtures.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="synthetic-native-v3-package-", dir=fixtures)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_notebook_defaults_schema_and_python(self):
        notebook = json.loads((ROOT / "notebooks/native_paper_kaggle_v3.ipynb").read_text(encoding="utf-8"))
        source = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
        compile(source, "native-v3-notebook", "exec")
        self.assertIn("STAGES = ['prepare', 'smoke']", source)
        self.assertIn("ALLOW_MAIN = False", source)
        self.assertIn(bundle.KIND, source)
        self.assertIn("kaggle_native_paper_v3_bundle", source)
        self.assertIn("tools/kaggle_native_paper_v3.py", source)
        self.assertIn("/kaggle/working/native_paper_v3", source)
        self.assertNotIn("SCHEMA11", source)

    def test_one_pinned_python_and_no_gp_requirement(self):
        self.assertEqual(runtime.PYTHONS, {"controller": "3.12.7"})
        lock = (ROOT / "requirements-native-paper-v3.txt").read_text()
        requirements = re.findall(r"^([A-Za-z0-9_-]+)==([^\s\\]+)", lock, re.M)
        selected = dict(requirements)
        self.assertEqual(selected["numpy"], "1.26.4")
        self.assertEqual(selected["scipy"], "1.11.4")
        self.assertEqual(len(selected), len(requirements))
        self.assertFalse({"torch", "gpytorch", "linear-operator", "scikit-learn"} & set(selected))
        for block in re.split(r"(?m)(?=^[A-Za-z0-9_-]+==)", lock)[1:]:
            self.assertIn("--hash=sha256:", block)

    def test_restored_package_set_must_match_lock_without_repair(self):
        lock = self.root / "requirements.txt"
        lock.write_text("numpy==1.26.4 \\\n    --hash=sha256:synthetic\nPyYAML==6.0.2\n")
        runtime.assert_locked_packages({"packages": [["numpy", "1.26.4"], ["pyyaml", "6.0.2"]]}, lock)
        for packages in ([["numpy", "1.26.4"]], [["numpy", "1.26.4"], ["PyYAML", "6.0.2"], ["torch", "2.0.1"]],
                         [["numpy", "2.0.0"], ["PyYAML", "6.0.2"]]):
            with self.assertRaisesRegex(ValueError, "package set"):
                runtime.assert_locked_packages({"packages": packages}, lock)

    def test_campaign_command_new_module_and_sem_interpreter(self):
        rt = {"controller": Path("/pinned/python")}
        prepare = runtime.campaign_command("prepare", self.root, rt)
        self.assertIn("native_paper_v3_campaign.py", str(prepare[1]))
        self.assertIn("--sem-python", prepare)
        self.assertNotIn("--sgp-python", prepare)
        self.assertIn(ROOT / "build-native-paper-v3", prepare)
        collect = runtime.campaign_command("collect", self.root, rt, mode="main", resume=True,
                                           allow_main=True, started=time.monotonic())
        self.assertIn("--allow-main", collect)
        self.assertIn("--resume", collect)
        self.assertNotIn("validate", collect)
        validation = runtime.campaign_command("validate", self.root, rt, mode="main")
        self.assertEqual(validation[-2:], ["--jobs", 4])
        self.assertNotIn("--allow-main", validation)

    def test_build_jobs_obey_cpu_quota_and_ram(self):
        self.assertEqual(runtime.choose_build_jobs({"cpu_capacity": 12, "memory_limit_bytes": 32 * 1024**3}), 4)
        self.assertEqual(runtime.choose_build_jobs({"cpu_capacity": 2, "memory_limit_bytes": 32 * 1024**3}), 2)
        self.assertEqual(runtime.choose_build_jobs({"cpu_capacity": 4, "memory_limit_bytes": 4 * 1024**3}), 1)

    def test_restore_missing_python_never_downloads(self):
        with patch.object(runtime, "ROOT", self.root), patch.object(runtime, "bootstrap_uv") as install:
            with self.assertRaisesRegex(RuntimeError, "exact Python runtime"):
                runtime.isolated_runtimes(self.root / "campaign", time.monotonic(), restore=True)
        install.assert_not_called()

    def test_main_requires_flag_before_source_or_dependency_work(self):
        args = SimpleNamespace(stage="collect", allow_main=False)
        with patch.object(runtime.sys, "platform", "linux"), patch.object(runtime.platform, "machine", return_value="x86_64"), \
             patch.object(runtime, "verify") as verify:
            with self.assertRaisesRegex(ValueError, "allow-main"):
                runtime.execute(args)
        verify.assert_not_called()

    def test_failure_receipt_keeps_real_exit_and_stderr(self):
        output = io.StringIO()
        with contextlib.redirect_stderr(output):
            with self.assertRaisesRegex(RuntimeError, "probe failed/partial"):
                runtime.command("probe", [sys.executable, "-c", "import sys;print('SYNTHETIC_DIAGNOSTIC',file=sys.stderr);sys.exit(7)"],
                                self.root, time.monotonic(), cwd=self.root)
        receipt = next((self.root / "logs/bootstrap").glob("probe_*.json"))
        self.assertEqual(json.loads(receipt.read_text())["exit_code"], 7)
        self.assertIn("SYNTHETIC_DIAGNOSTIC", output.getvalue())
        self.assertIn("SYNTHETIC_DIAGNOSTIC", receipt.with_suffix(".stderr.log").read_text())

    def test_checkpoint_round_trip_retains_failed_attempt(self):
        repo, workspace = self.root / "repo", self.root / "campaign"
        binary = repo / "build-native-paper-v3/terrain_benchmark"
        python = repo / ".venv-native-paper-v3/bin/python"
        for item in (binary, python):
            item.parent.mkdir(parents=True, exist_ok=True)
            item.write_bytes(b"SYNTHETIC_ARCHIVE_FIXTURE_not_executable")
        workspace.mkdir()
        (workspace / "start_journal.jsonl").write_text('{"trial_id":"fixture","status":"INTERRUPTED"}\n')
        source = self.root / "source.zip"
        source.write_bytes(b"synthetic source archive identity")
        archive = self.root / "checkpoint.zip"
        manifest = bundle.export_checkpoint(workspace, archive, root=repo, input_archive=source)
        self.assertEqual(manifest["kind"], bundle.CHECKPOINT_KIND)
        self.assertTrue(manifest["metadata"]["contains_exact_runtime"])
        paths = {entry["path"] for entry in manifest["files"]}
        self.assertIn("repo/.venv-native-paper-v3/bin/python", paths)
        self.assertIn("campaign/start_journal.jsonl", paths)
        bundle.restore_checkpoint(archive, self.root, bundle.sha256(archive), source_sha256=bundle.sha256(source), root=repo, workspace=workspace)
        self.assertIn("INTERRUPTED", (workspace / "start_journal.jsonl").read_text())
        with self.assertRaisesRegex(ValueError, "source identity"):
            bundle.restore_checkpoint(archive, self.root, bundle.sha256(archive), source_sha256="changed", root=repo, workspace=workspace)
        with self.assertRaisesRegex(ValueError, "absolute"):
            bundle.restore_checkpoint(archive, self.root, bundle.sha256(archive), source_sha256=bundle.sha256(source), root=repo / "other")

    if sys.platform.startswith("linux"):
        def test_linux_exact_runtime_symlink_roundtrip(self):
            """Real POSIX link export, recreation and idempotent second restore."""
            repo, workspace = self.root / "repo", self.root / "campaign"
            native = repo / ".managed-python/cpython/bin/python3.12"
            alias = repo / ".venv-native-paper-v3/bin/python"
            native.parent.mkdir(parents=True)
            alias.parent.mkdir(parents=True)
            workspace.mkdir()
            native.write_bytes(b"SYNTHETIC_INTERPRETER_FIXTURE_not_executable")
            alias.symlink_to("../../.managed-python/cpython/bin/python3.12")
            self.assertEqual(alias.resolve(), native.resolve())
            source = self.root / "source.zip"
            source.write_bytes(b"synthetic source archive identity")
            archive = self.root / "symlink-checkpoint.zip"
            manifest = bundle.export_checkpoint(workspace, archive, root=repo, input_archive=source)
            self.assertEqual(manifest["path_case_policy"], bundle.LINUX_CASE_POLICY)
            links = manifest["metadata"]["internal_runtime_symlinks"]
            self.assertEqual(len(links), 1)
            self.assertEqual(links[0]["path"], "repo/.venv-native-paper-v3/bin/python")
            self.assertEqual(links[0]["target"], "repo/.managed-python/cpython/bin/python3.12")
            alias.unlink()
            bundle.restore_checkpoint(archive, self.root, bundle.sha256(archive), source_sha256=bundle.sha256(source), root=repo, workspace=workspace)
            self.assertTrue(alias.is_symlink())
            self.assertEqual(alias.resolve(), native.resolve())
            bundle.restore_checkpoint(archive, self.root, bundle.sha256(archive), source_sha256=bundle.sha256(source), root=repo, workspace=workspace)
            self.assertTrue(alias.is_symlink())
            self.assertEqual(alias.read_bytes(), native.read_bytes())

    def test_export_cli_requires_source_archive(self):
        argv = ["bundle", "export", "--workspace", str(self.root), "--output", str(self.root / "output.zip")]
        with patch.object(sys, "argv", argv), patch.object(bundle, "export_checkpoint") as export, \
             contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as failure:
                bundle.main()
        self.assertEqual(failure.exception.code, 2)
        export.assert_not_called()

    def test_tampered_checkpoint_fails_before_restore(self):
        payload = self.root / "fixture.txt"
        payload.write_bytes(b"synthetic")
        archive = self.root / "checkpoint.zip"
        bundle.archive([("campaign/fixture.txt", payload)], archive, bundle.CHECKPOINT_KIND)
        checksum = bundle.sha256(archive)
        with archive.open("ab") as stream:
            stream.write(b"tamper")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            bundle.restore_checkpoint(archive, self.root, checksum, source_sha256="unused", root=self.root / "repo")

    def test_checkpoint_protocol_is_not_interchangeable(self):
        payload = self.root / "fixture.txt"
        payload.write_bytes(b"synthetic")
        archive = self.root / "wrong-protocol.zip"
        bundle.archive([("campaign/fixture.txt", payload)], archive, bundle.CHECKPOINT_KIND,
                       {"protocol_id": "different", "runtime_policy": "native_paper_sessions_v3"})
        with self.assertRaisesRegex(ValueError, "protocol/runtime"):
            bundle.restore_checkpoint(archive, self.root, bundle.sha256(archive), source_sha256="unused", root=self.root / "repo")

    def test_source_bundle_injects_hash_and_opaque_payload(self):
        root = self.root / "repo"
        (root / "notebooks").mkdir(parents=True)
        template = ROOT / "notebooks/native_paper_kaggle_v3.ipynb"
        (root / "notebooks" / template.name).write_bytes(template.read_bytes())
        fixture = root / "fixture.txt"
        fixture.write_bytes(b"SYNTHETIC_SOURCE_FIXTURE")
        target = self.root / "source.zip"
        with patch.object(bundle, "source_entries", return_value=[("fixture.txt", fixture)]):
            manifest = bundle.build_source(target, root=root)
        self.assertEqual(manifest["metadata"]["config_schema"], 12)
        notebook = target.with_suffix(".ipynb").read_text(encoding="utf-8")
        self.assertIn(bundle.sha256(target), notebook)
        self.assertNotIn("SOURCE_SHA256_PLACEHOLDER", notebook)
        import base64
        self.assertEqual(base64.b64decode(target.with_suffix(".b64").read_bytes()), target.read_bytes())

    def test_export_space_uses_streaming_reserve(self):
        # A large uncompressed input must not revive the old export-size guard.
        workspace = self.root / "campaign"
        workspace.mkdir()
        free = SimpleNamespace(free=1024**3)
        with patch.object(runtime.shutil, "disk_usage", return_value=free), \
             patch.object(runtime, "export_checkpoint", return_value={}) as export:
            result = runtime.retain_checkpoint(workspace, None)
        self.assertIsNotNone(result)
        export.assert_called_once()

    def test_export_failure_preserves_stage_exception(self):
        workspace = self.root / "campaign"
        workspace.mkdir()
        original = RuntimeError("SYNTHETIC_PRIMARY_FAILURE")
        with patch.object(runtime, "export_checkpoint", side_effect=ValueError("SYNTHETIC_EXPORT_FAILURE")), \
             contextlib.redirect_stderr(io.StringIO()):
            self.assertIsNone(runtime.retain_checkpoint(workspace, None, stage_error=original))
        receipt = next((workspace / "logs/bootstrap").glob("*_export_failure.json"))
        saved = json.loads(receipt.read_text())
        self.assertIn("SYNTHETIC_PRIMARY_FAILURE", saved["stage_error"])
        self.assertIn("SYNTHETIC_EXPORT_FAILURE", saved["export_error"])


if __name__ == "__main__":
    unittest.main()
