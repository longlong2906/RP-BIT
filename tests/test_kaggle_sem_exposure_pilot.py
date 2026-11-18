"""Synthetic packaging/runtime tests; no network or planner is launched."""
from __future__ import annotations

import ast
import base64
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tools"))
import kaggle_sem_exposure_pilot as bootstrap
import kaggle_sem_exposure_pilot_bundle as bundle
import kaggle_native_paper_v4_bundle as legacy
from kaggle_a3_bundle import sha256, verify


class PilotKaggleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="synthetic-sem-exposure-kaggle-")
        self.base = Path(self.temporary.name)
        self.repo, self.campaign = self.base/"repo", self.base/"campaign"
        self.repo.mkdir();self.campaign.mkdir()

    def tearDown(self):
        self.temporary.cleanup()

    def put(self, path, text="synthetic fixture; never executed"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def checkpoint_fixture(self):
        source = self.put(self.base/"source_input.zip")
        self.put(self.repo/bundle.BUILD_DIR/"terrain_benchmark")
        self.put(self.campaign/"pilot/tune/manifest.json", '{"status":"PAUSED_TIME"}')
        shard = self.put(self.campaign/"pilot/tune/shards/map1.zip")
        self.put(self.repo/bundle.VENV_DIR/"bin/python")
        return source, shard

    def test_source_payload_notebook_and_checksum(self):
        for directory in ("notebooks", "docs"):
            (self.repo/directory).mkdir()
        shutil.copy2(ROOT/"notebooks/sem_exposure_pilot_kaggle.ipynb", self.repo/"notebooks/sem_exposure_pilot_kaggle.ipynb")
        self.put(self.repo/"docs/KAGGLE_SEM_EXPOSURE_PILOT.md", "# Synthetic delivery fixture\n")
        entry = self.put(self.repo/"tools/new_pilot.py")
        output = self.base/"sem_exposure_pilot_fixture.zip"
        with patch.object(bundle, "source_entries", return_value=[("tools/new_pilot.py", entry)]):
            bundle.build_source(output, root=self.repo)
            with self.assertRaises(FileExistsError):
                bundle.build_source(output, root=self.repo)
        self.assertEqual(verify(output)["kind"], bundle.KIND)
        self.assertEqual(base64.b64decode(output.with_suffix(".b64").read_bytes()), output.read_bytes())
        notebook = json.loads(output.with_suffix(".ipynb").read_text(encoding="utf-8"))
        code = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
        self.assertNotIn("SOURCE_SHA256_PLACEHOLDER", code)
        self.assertIn(sha256(output), code)
        self.assertNotIn("ALLOW_MAIN", code)
        self.assertIn("ALLOW_PILOT = True", code)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))
        self.assertIn(sha256(output), output.with_suffix(".README.md").read_text(encoding="utf-8"))

    def test_nested_shards_external_and_incremental_restore(self):
        source, shard = self.checkpoint_fixture()
        first = self.campaign/"exports/base.payload"
        bundle.export_checkpoint(self.campaign, first, root=self.repo, input_archive=source)
        manifest = verify(first)
        name = "campaign/pilot/tune/shards/map1.zip"
        self.assertNotIn(name, {item["path"] for item in manifest["files"]})
        self.assertIn(name, {item["path"] for item in manifest["metadata"]["external_files"]})
        self.assertTrue(manifest["metadata"]["contains_exact_runtime"])
        trial = self.put(self.campaign/"pilot/tune/new_raw.txt", "new synthetic raw")
        second = self.campaign/"exports/delta.payload"
        bundle.export_checkpoint(self.campaign, second, root=self.repo, input_archive=source)
        second_manifest = verify(second)
        self.assertEqual(second_manifest["metadata"]["payload_type"], "delta")
        self.assertNotIn("repo/"+bundle.BUILD_DIR+"/terrain_benchmark", {x["path"] for x in second_manifest["files"]})
        index_path = self.campaign/"exports/checkpoint_index.json"
        inputs = self.base/"attached-input"
        external = inputs/name
        external.parent.mkdir(parents=True)
        shutil.copy2(shard, external)
        shard.unlink();trial.unlink()
        bundle.restore_checkpoint(index_path, self.base, sha256(index_path), source_sha256=sha256(source),
                                  root=self.repo, workspace=self.campaign, external_root=inputs)
        self.assertEqual(trial.read_text(), "new synthetic raw")
        self.assertEqual(sha256(shard), sha256(external))
        with self.assertRaises(ValueError):
            legacy.restore_checkpoint(index_path, self.base, sha256(index_path), source_sha256=sha256(source),
                                      root=self.repo, workspace=self.campaign, external_root=inputs)

    def test_checkpoint_rejects_changed_runtime_or_shard(self):
        source, shard = self.checkpoint_fixture()
        bundle.export_checkpoint(self.campaign, self.campaign/"exports/base.payload", root=self.repo, input_archive=source)
        original = shard.read_bytes();shard.write_bytes(b"changed fixture")
        with self.assertRaisesRegex(ValueError, "immutable checkpoint asset changed"):
            bundle.export_checkpoint(self.campaign, self.campaign/"exports/rejected.payload", root=self.repo, input_archive=source)
        shard.write_bytes(original)
        self.put(self.repo/bundle.BUILD_DIR/"terrain_benchmark", "changed runtime")
        with self.assertRaisesRegex(ValueError, "immutable checkpoint asset changed"):
            bundle.export_checkpoint(self.campaign, self.campaign/"exports/rejected.payload", root=self.repo, input_archive=source)

    def test_mutable_ctest_logs_are_not_immutable_runtime(self):
        source, _ = self.checkpoint_fixture()
        ctest_log = self.put(self.repo/bundle.BUILD_DIR/"Testing/Temporary/LastTest.log")
        bundle.export_checkpoint(self.campaign, self.campaign/"exports/base.payload", root=self.repo, input_archive=source)
        ctest_log.write_text("rerun synthetic test output", encoding="utf-8")
        bundle.export_checkpoint(self.campaign, self.campaign/"exports/delta.payload", root=self.repo, input_archive=source)
        index = json.loads((self.campaign/"exports/checkpoint_index.json").read_text())
        self.assertFalse(any("/Testing/" in name for name in index["inventory"]))

    def test_restore_rejects_newer_physical_start_evidence(self):
        source, _ = self.checkpoint_fixture()
        bundle.export_checkpoint(self.campaign, self.campaign/"exports/base.payload", root=self.repo, input_archive=source)
        self.put(self.campaign/"pilot/tune/active/map1/execution_control/start_journal.jsonl", '{"trial_id":"newer_synthetic_start"}\n')
        index = self.campaign/"exports/checkpoint_index.json"
        with self.assertRaisesRegex(ValueError, "physical attempt evidence is newer"):
            bundle.restore_checkpoint(index, self.base, sha256(index), source_sha256=sha256(source),
                root=self.repo, workspace=self.campaign)

    def test_old_v4_defaults_preserved(self):
        source = self.put(self.base/"source_input.zip")
        self.put(self.repo/"build-native-paper-v4/terrain_benchmark")
        self.put(self.campaign/"shards/old.zip")
        output = self.campaign/"exports/old.payload"
        legacy.export_checkpoint(self.campaign, output, root=self.repo, input_archive=source)
        value = verify(output)
        self.assertEqual(value["kind"], legacy.CHECKPOINT_KIND)
        self.assertEqual(value["metadata"]["protocol_id"], legacy.PROTOCOL_ID)
        self.assertEqual(value["metadata"]["runtime_policy"], "native_paper_sessions_v4")
        self.assertEqual(legacy._index(self.campaign/"exports/checkpoint_index.json")["schema_version"], 13)

    def test_resource_gate_observes_effective_quota(self):
        runtime = dict(cpu_capacity=3, eligible_cpus=[0, 1, 2, 3], cpu_quota=3.9, memory_limit_bytes=16*1024**3)
        with patch.object(bootstrap, "command", return_value=json.dumps(runtime)), \
             patch.object(bootstrap.shutil, "disk_usage", return_value=SimpleNamespace(free=20*1024**3)):
            with self.assertRaisesRegex(RuntimeError, "four effective CPUs"):
                bootstrap.resource_gate(self.campaign, 0., Path("python"), {}, "fixture")
        self.assertEqual(bootstrap.read(self.campaign/"setup/resource_fixture.json")["hardware"]["cpu_quota"], 3.9)

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux venv symlink semantics")
    def test_venv_execution_path_is_not_resolved_to_base(self):
        managed = self.put(self.repo/".managed-python/bin/python")
        python = self.repo/bundle.VENV_DIR/"bin/python"
        python.parent.mkdir(parents=True);python.symlink_to(managed)
        evidence = dict(version=bootstrap.PYTHON_VERSION, full="synthetic", packages=[])
        with patch.object(bootstrap, "ROOT", self.repo), patch.object(bootstrap, "command", return_value=json.dumps(evidence)), \
             patch.object(bootstrap.supervised, "assert_locked_packages"):
            actual, _ = bootstrap.isolated_runtime(self.campaign, 0., restored=True)
        self.assertEqual(actual, python.absolute())
        self.assertNotEqual(actual, python.resolve())

    def execute_mocks(self, tune_status="PAUSED_TIME", stage="all"):
        args = SimpleNamespace(stage=stage, allow_pilot=True, input_archive=self.base/"source.zip",
            input_sha256="a"*64, workspace=self.campaign, restored_runtime=False, session_started_monotonic=None)
        identity = dict(identity_sha256="b"*64)
        with patch.object(bootstrap.sys, "platform", "linux"), patch.object(bootstrap.platform, "machine", return_value="x86_64"), \
             patch.object(bootstrap, "verify_source"), patch.object(bootstrap, "isolated_runtime", return_value=(Path("python"), {})), \
             patch.object(bootstrap, "build") as build, patch.object(bootstrap, "runtime_identity", return_value=(Path("identity.json"), identity)), \
             patch.object(bootstrap, "correctness") as correctness, patch.object(bootstrap, "require_correctness"), \
             patch.object(bootstrap, "smoke_gate"), patch.object(bootstrap, "retain_checkpoint"), \
             patch.object(bootstrap, "pilot_phase", return_value=dict(status=tune_status)) as phase:
            value = bootstrap.execute(args)
        self.correctness_calls = correctness.call_count
        return value, phase.call_args_list, build.call_count

    def test_partial_tuning_never_dispatches_confirmation(self):
        value, calls, build_count = self.execute_mocks()
        self.assertEqual(value["status"], "PAUSED_TIME")
        self.assertEqual([call.args[-1] for call in calls], ["tune"])
        self.assertEqual(build_count, 1)

    def test_existing_identity_is_resumed_without_rebuild(self):
        self.put(self.campaign/"setup/runtime_identity.json", "{}")
        _, _, build_count = self.execute_mocks()
        self.assertEqual(build_count, 0)
        self.assertEqual(self.correctness_calls, 1)

    def test_interrupted_prepare_reruns_only_correctness_on_frozen_runtime(self):
        self.put(self.campaign/"setup/runtime_identity.json", "{}")
        value, calls, builds = self.execute_mocks(stage="prepare")
        self.assertEqual(value["status"], "PREPARED")
        self.assertEqual((builds, self.correctness_calls, len(calls)), (0, 1, 0))

    def test_existing_correctness_is_not_rerun(self):
        self.put(self.campaign/"setup/runtime_identity.json", "{}")
        self.put(self.campaign/"correctness.json", "{}")
        _, _, builds = self.execute_mocks()
        self.assertEqual((builds, self.correctness_calls), (0, 0))

    def test_missing_correctness_cannot_rerun_after_attempt_evidence(self):
        self.put(self.campaign/"setup/runtime_identity.json", "{}")
        self.put(self.campaign/"smoke/execution_control/start_journal.jsonl", "synthetic started")
        with self.assertRaisesRegex(ValueError, "after smoke/pilot evidence"):
            self.execute_mocks()

    def test_confirmation_requires_locked_candidate(self):
        self.put(self.campaign/"pilot/selection.json", '{"status":"NO_CANDIDATE"}')
        with patch.object(bootstrap, "command") as command:
            value = bootstrap.pilot_phase(self.campaign, 0., Path("python"), {}, Path("identity.json"), {}, "confirm")
        self.assertEqual(value["status"], "NO_CANDIDATE")
        command.assert_not_called()

    def test_exact_prior_scientific_gate_count(self):
        self.assertEqual(len(bootstrap.REQUIRED_CTESTS), 39)
        self.assertEqual(len(set(bootstrap.REQUIRED_CTESTS)), 39)

    def test_required_linux_test_skip_is_not_a_pass(self):
        with patch.object(bootstrap, "command", return_value="Ran 4 tests\nOK (skipped=1)\n"):
            with self.assertRaisesRegex(RuntimeError, "cannot accept skipped"):
                bootstrap.checked_test("fixture", [], self.campaign, 0., {})


if __name__ == "__main__":
    unittest.main()
