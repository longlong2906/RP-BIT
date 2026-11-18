"""Schema-10 Kaggle packaging/bootstrap tests. No Kaggle or planner run."""
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import kaggle_terrain_compare as k
import kaggle_terrain_compare_bundle as b


class KaggleTerrainTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="synthetic-kaggle-terrain-", dir=ROOT / "tests/fixtures")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_build_jobs_respect_four_cpu_quota_and_memory(self):
        self.assertEqual(k.choose_build_jobs({"cpu_capacity": 12, "memory_limit_bytes": 32 * 1024**3}), 4)
        self.assertEqual(k.choose_build_jobs({"cpu_capacity": 4, "memory_limit_bytes": 4 * 1024**3}), 1)
        self.assertEqual(k.choose_build_jobs({"cpu_capacity": 2, "memory_limit_bytes": 32 * 1024**3}), 2)

    def test_failed_command_exposes_actual_stderr_and_retains_full_logs(self):
        output = io.StringIO()
        with contextlib.redirect_stderr(output):
            with self.assertRaisesRegex(RuntimeError, "prepare failed/partial"):
                k.command("prepare", [sys.executable, "-c",
                          "import sys;print('synthetic stdout');print('PRIMARY_PREPARE_ERROR',file=sys.stderr);sys.exit(7)"],
                          self.root, time.monotonic(), cwd=self.root)
        self.assertIn("PRIMARY_PREPARE_ERROR", output.getvalue())
        self.assertIn("synthetic stdout", output.getvalue())
        receipt = next((self.root / "logs/bootstrap").glob("prepare_*.json"))
        self.assertEqual(json.loads(receipt.read_text())["exit_code"], 7)
        self.assertIn("PRIMARY_PREPARE_ERROR", receipt.with_suffix(".stderr.log").read_text())

    def test_failure_log_tail_is_bounded(self):
        stem = self.root / "failure"
        stem.with_suffix(".stderr.log").write_bytes(b"hidden-prefix" + b"x"*200 + b"visible-tail")
        stem.with_suffix(".stdout.log").write_bytes(b"")
        output = io.StringIO()
        with contextlib.redirect_stderr(output):
            k.show_command_failure(stem, {"exit_code": 1}, tail_bytes=32)
        self.assertIn("visible-tail", output.getvalue())
        self.assertNotIn("hidden-prefix", output.getvalue())

    def test_export_failure_preserves_original_stage_exception(self):
        args = SimpleNamespace(stage="prepare", allow_main=False, input_archive=self.root / "source.zip",
                               input_sha256="synthetic", workspace=self.root, session_started_monotonic=time.monotonic(),
                               restored_runtime=False)
        original = RuntimeError("PRIMARY_PREPARE_ERROR")
        with patch.object(k.sys, "platform", "linux"), patch.object(k.platform, "machine", return_value="x86_64"), \
             patch.object(k, "verify", return_value={"kind": k.KIND, "files": []}), \
             patch.object(k, "isolated_runtimes", return_value=({"controller": sys.executable, "sgp": sys.executable}, {})), \
             patch.object(k, "build"), patch.object(k, "command", side_effect=original), \
             patch.object(k, "export_checkpoint", side_effect=ValueError("SECONDARY_EXPORT_ERROR")), \
             contextlib.redirect_stderr(io.StringIO()) as output:
            with self.assertRaises(RuntimeError) as observed:
                k.execute(args)
        self.assertIs(observed.exception, original)
        self.assertIn("SECONDARY_EXPORT_ERROR", output.getvalue())
        log = next((self.root / "logs/bootstrap").glob("*_export_failure.json"))
        self.assertIn("PRIMARY_PREPARE_ERROR", json.loads(log.read_text())["stage_error"])

    def test_export_failure_without_prior_error_still_fails(self):
        with patch.object(k, "export_checkpoint", side_effect=ValueError("EXPORT_FAILED")), \
             contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "EXPORT_FAILED"):
                k.retain_checkpoint(self.root, self.root / "source.zip")

    def test_campaign_commands_keep_explicit_main_and_separate_phases(self):
        runtimes = {"controller": Path("/isolated/controller/python"), "sgp": Path("/isolated/sgp/python")}
        prepare = k.campaign_command("prepare", self.root, runtimes)
        self.assertIn("--sgp-python", prepare)
        self.assertNotIn("--allow-main", prepare)
        run = k.campaign_command("run", self.root, runtimes, mode="main", resume=True,
                                 allow_main=True, started=0)
        self.assertIn("--resume", run)
        self.assertIn("--allow-main", run)
        self.assertIn("--session-elapsed-s", run)
        validate = k.campaign_command("validate", self.root, runtimes, mode="main")
        self.assertEqual(validate[-2:], ["--jobs", 4])

    def test_checkpoint_round_trip_retains_attempts_runtime_and_source(self):
        repo = self.root / "repo"
        workspace = self.root / "campaign"
        (repo / "build-terrain-compare").mkdir(parents=True)
        binary = repo / "build-terrain-compare/terrain_benchmark"
        binary.write_bytes(b"SYNTHETIC-not-a-real-binary")
        workspace.mkdir()
        (workspace / "attempt.json").write_text('{"status":"INTERRUPTED"}')
        source = self.root / "source_input.zip"
        source.write_bytes(b"synthetic source identity")
        target = self.root / "checkpoint.zip"
        manifest = b.export_checkpoint(workspace, target, root=repo, input_archive=source)
        self.assertEqual(manifest["kind"], b.CHECKPOINT_KIND)
        self.assertIn("campaign/attempt.json", [r["path"] for r in manifest["files"]])
        restored = self.root
        b.restore_checkpoint(target, restored, b.sha256(target), source_sha256=b.sha256(source),
                             root=repo, workspace=workspace)
        self.assertEqual((restored / "repo/build-terrain-compare/terrain_benchmark").read_bytes(), binary.read_bytes())
        with self.assertRaisesRegex(ValueError, "source identity"):
            b.restore_checkpoint(target, restored, b.sha256(target), source_sha256="changed", root=repo)
        with self.assertRaisesRegex(ValueError, "absolute"):
            b.restore_checkpoint(target, restored, b.sha256(target), source_sha256=b.sha256(source), root=repo / "other")

    def test_internal_venv_symlinks_are_recorded_and_safely_reconstructed(self):
        repo, workspace = self.root / "repo", self.root / "campaign"
        native = repo / ".managed-python/python/bin/python3.10"
        alias = repo / ".venv-terrain-sgp/bin/python"
        native.parent.mkdir(parents=True)
        alias.parent.mkdir(parents=True)
        workspace.mkdir()
        native.write_bytes(b"synthetic interpreter")
        try:
            alias.symlink_to(native)
        except OSError:
            self.skipTest("host cannot create symlinks; Linux integration remains required")
        source = self.root / "source_input.zip"
        source.write_bytes(b"synthetic source")
        archive = self.root / "checkpoint.zip"
        manifest = b.export_checkpoint(workspace, archive, root=repo, input_archive=source)
        links = manifest["metadata"]["internal_runtime_symlinks"]
        self.assertEqual(len(links), 1)
        self.assertEqual(links[0]["target"], "repo/.managed-python/python/bin/python3.10")
        alias.unlink()
        b.restore_checkpoint(archive, self.root, b.sha256(archive), source_sha256=b.sha256(source), root=repo, workspace=workspace)
        self.assertTrue(alias.is_symlink())
        self.assertEqual(alias.resolve(), native.resolve())
        # Idempotent verified restore preserves the interpreter link.
        b.restore_checkpoint(archive, self.root, b.sha256(archive), source_sha256=b.sha256(source), root=repo, workspace=workspace)
        self.assertEqual(alias.resolve(), native.resolve())

    def test_restore_rejects_symlink_metadata_escape(self):
        repo, workspace = self.root / "repo", self.root / "campaign"
        repo.mkdir(); workspace.mkdir()
        item = self.root / "synthetic"
        item.write_bytes(b"fixture")
        archive = self.root / "malicious.zip"
        b.archive([("campaign/synthetic", item)], archive, b.CHECKPOINT_KIND,
                  {"root_path": str(repo), "workspace_path": str(workspace), "source_input_sha256": "source",
                   "internal_runtime_symlinks": [{"path": "repo/alias", "target": "repo/../../outside",
                                                   "target_is_directory": False, "target_sha256": "x"}]})
        with self.assertRaisesRegex(ValueError, "unsafe"):
            b.restore_checkpoint(archive, self.root, b.sha256(archive), source_sha256="source", root=repo, workspace=workspace)

    def test_verified_regular_alias_is_removed_before_recreating_internal_link(self):
        repo, workspace = self.root / "repo", self.root / "campaign"
        repo.mkdir(); workspace.mkdir()
        binary = self.root / "binary_fixture"
        binary.write_bytes(b"synthetic executable")
        checksum = b.sha256(binary)
        archive = self.root / "restore-links.zip"
        link = {"path": "repo/venv/bin/python", "target": "repo/native/python",
                "target_is_directory": False, "target_sha256": checksum}
        b.archive([(link["path"], binary), (link["target"], binary)], archive, b.CHECKPOINT_KIND,
                  {"root_path": str(repo), "workspace_path": str(workspace), "source_input_sha256": "source",
                   "internal_runtime_symlinks": [link]})
        calls = []
        def record_link(alias, target, target_is_directory=False):
            calls.append((alias, target, target_is_directory))
            self.assertFalse(alias.exists())
            self.assertEqual(b.sha256(alias.parent / target), checksum)
        with patch.object(Path, "symlink_to", record_link):
            b.restore_checkpoint(archive, self.root, b.sha256(archive), source_sha256="source", root=repo, workspace=workspace)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], self.root / link["path"])

    def test_runtime_restore_never_installs_missing_packages(self):
        with patch.object(k, "ROOT", self.root):
            with self.assertRaisesRegex(RuntimeError, "exact Python"):
                k.isolated_runtimes(self.root / "campaign", 0, restore=True)

    def test_runtime_commands_use_headless_backend_without_changing_notebook_environment(self):
        for role in k.PYTHONS:
            interpreter = self.root / (".venv-terrain-"+role) / "bin/python"
            interpreter.parent.mkdir(parents=True)
            interpreter.write_bytes(b"synthetic interpreter, never executed")
        observed = []
        def command(name, argv, workspace, started, **kwargs):
            observed.append(kwargs["env"].copy())
            role = name.removeprefix("python_probe_")
            return json.dumps({"version": k.PYTHONS[role], "packages": []})
        inherited = "module://matplotlib_inline.backend_inline"
        with patch.object(k, "ROOT", self.root), patch.object(k, "command", side_effect=command), \
             patch.dict(os.environ, {"MPLBACKEND": inherited}):
            _, env = k.isolated_runtimes(self.root / "campaign", 0, restore=True)
            self.assertEqual(env["MPLBACKEND"], "Agg")
            self.assertEqual(os.environ["MPLBACKEND"], inherited)
        self.assertEqual(len(observed), 2)
        self.assertTrue(all(env["MPLBACKEND"] == "Agg" for env in observed))

    def test_source_notebook_valid_python_and_safe_default(self):
        notebook = json.loads((ROOT / "notebooks/terrain_compare_kaggle.ipynb").read_text(encoding="utf-8"))
        code = []
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                source = "".join(cell["source"])
                ast.parse(source)
                code.append(source)
        text = "\n".join(code)
        self.assertIn("ALLOW_MAIN = False", text)
        self.assertIn("STAGES = ['prepare', 'smoke']", text)
        self.assertIn("SOURCE_SHA256_PLACEHOLDER", text)
        self.assertIn("checksum(temporary) != SOURCE_SHA256", text)
        self.assertIn("--session-started-monotonic", text)

    def test_hash_locks_fully_pin_and_no_cuda_packages(self):
        import re
        for role in ("controller", "sgp"):
            text = (ROOT / f"requirements-terrain-compare-{role}.txt").read_text()
            packages = [line for line in text.splitlines() if line and not line.startswith((" ", "#", "--"))]
            self.assertGreater(len(packages), 10)
            self.assertTrue(all(re.match(r"[a-zA-Z0-9_.-]+==[^ ]+", line) for line in packages))
            self.assertGreaterEqual(text.count("--hash=sha256:"), len(packages))
            self.assertNotIn("nvidia-", text)
        self.assertEqual(b.sha256(ROOT / ".deps/uv-0.9.7-linux.whl"), k.UV_WHEEL_SHA256)


if __name__ == "__main__":
    unittest.main()
