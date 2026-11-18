"""Standard-library-only schema-13 packaging tests; all data are synthetic."""
from __future__ import annotations

import base64
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tools"))
import kaggle_native_paper_v4 as runtime
import kaggle_native_paper_v4_bundle as bundle


class KaggleNativeV4(unittest.TestCase):
    def setUp(self):
        fixtures = Path(os.environ.get("NATIVE_PAPER_V4_TEST_TMP", ROOT/"tests/fixtures"))
        fixtures.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="synthetic-native-v4-package-", dir=fixtures)
        self.addCleanup(temporary.cleanup)
        self.temp = Path(temporary.name).resolve()
        self.base, self.inputs = self.temp/"retained", self.temp/"mounted_input"
        self.repo, self.workspace = self.base/"repo", self.base/"campaign"
        self.workspace.mkdir(parents=True)
        self.source = self.base/"source_input.zip"
        self.source.write_bytes(b"SYNTHETIC source archive identity")
        for relative in ("build-native-paper-v4/terrain_benchmark", ".venv-native-paper-v4/bin/python"):
            self.put(self.repo/relative, b"SYNTHETIC_NOT_EXECUTABLE")
        self.put(self.workspace/"data/main/map.json", b"SYNTHETIC_FROZEN_MAP")

    @staticmethod
    def put(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def export(self, name="base.payload", **kwargs):
        return bundle.export_checkpoint(self.workspace, self.workspace/"exports"/name,
            root=self.repo, input_archive=self.source, **kwargs)

    def mounted_index(self):
        # Simulate Kaggle's read-only prior Notebook Output, preserving layout.
        shutil.copytree(self.workspace/"exports", self.inputs/"campaign/exports")
        if (self.workspace/"shards").exists():
            shutil.copytree(self.workspace/"shards", self.inputs/"campaign/shards")
        return self.inputs/"campaign/exports/checkpoint_index.json"

    def restore(self, index):
        return bundle.restore_checkpoint(index, self.base, bundle.sha256(index),
            source_sha256=bundle.sha256(self.source), root=self.repo, workspace=self.workspace)

    def test_notebook_runs_complete_authorized_plan(self):
        value = json.loads((ROOT/"notebooks/native_paper_kaggle_v4.ipynb").read_text(encoding="utf-8"))
        code = "\n".join("".join(c["source"]) for c in value["cells"] if c["cell_type"] == "code")
        compile(code, "v4-notebook", "exec")
        for text in ("ALLOW_MAIN = True", "STAGE = 'all'", bundle.KIND,
                     "/kaggle/working/native_paper_v4", "checkpoint_index.json", "kaggle_native_paper_v4.py"):
            self.assertIn(text, code)
        self.assertNotIn("SCHEMA12", code)
        self.assertTrue(all(not cell.get("outputs") for cell in value["cells"]))

    def test_exact_v3_package_pins_in_new_runtime(self):
        self.assertEqual(runtime.PYTHONS, {"controller": "3.12.7"})
        for suffix in ("in", "txt"):
            self.assertEqual((ROOT/("requirements-native-paper-v3."+suffix)).read_bytes(),
                             (ROOT/("requirements-native-paper-v4."+suffix)).read_bytes())
        lock = ROOT/"requirements-native-paper-v4.txt"
        self.assertIn("scipy==1.11.4", lock.read_text())
        with self.assertRaisesRegex(ValueError, "package set"):
            runtime.assert_locked_packages({"packages": []}, lock)

    def test_main_flag_checked_before_any_source_or_install_work(self):
        with patch.object(runtime.sys, "platform", "linux"), patch.object(runtime.platform, "machine", return_value="x86_64"), patch.object(runtime, "verify") as verify:
            with self.assertRaisesRegex(ValueError, "allow-main"):
                runtime.execute(SimpleNamespace(stage="all", allow_main=False))
            verify.assert_not_called()

    def test_command_uses_v4_python_and_no_force_main(self):
        rt = {"controller": Path("/exact/python")}
        prepare = runtime.campaign_command("prepare", self.workspace, rt)
        self.assertIn("--python", prepare)
        self.assertIn(ROOT/"build-native-paper-v4", prepare)
        self.assertIn("native_paper_v4_campaign.py", str(prepare[1]))
        argv = runtime.campaign_command("run", self.workspace, rt, mode="main", resume=True,
                                        allow_main=True, started=time.monotonic())
        self.assertIn("--resume", argv)
        self.assertIn("--allow-main", argv)
        with self.assertRaisesRegex(ValueError, "forbidden"):
            runtime.campaign_command("run", self.workspace, rt, mode="main", force_local=True)

    def test_missing_restored_runtime_never_downloads(self):
        with patch.object(runtime, "ROOT", self.temp/"missing"), patch.object(runtime, "bootstrap_uv") as install:
            with self.assertRaisesRegex(RuntimeError, "exact Python runtime"):
                runtime.isolated_runtimes(self.workspace, time.monotonic(), restore=True)
        install.assert_not_called()

    def test_base_delta_without_duplicate_runtime_shared_or_shard(self):
        self.put(self.workspace/"state.json", b'{"status":"PARTIAL"}')
        first = self.export()
        self.assertEqual(first["metadata"]["payload_type"], "base")
        self.assertTrue(first["metadata"]["contains_exact_runtime"])
        self.put(self.workspace/"state.json", b'{"status":"COMPLETE"}')
        self.put(self.workspace/"shared/raster.npz", b"SYNTHETIC raster bytes")
        self.put(self.workspace/"shards/wave__map.zip", b"SYNTHETIC raw shard")
        self.put(self.workspace/"shards/index.json", b'{"generation":1}')
        second = self.export("delta1.payload")
        self.assertEqual({r["path"] for r in second["files"]}, {
            "campaign/state.json", "campaign/shared/raster.npz", "campaign/shards/index.json"})
        self.assertEqual(second["metadata"]["external_files"][0]["path"], "campaign/shards/wave__map.zip")
        self.put(self.workspace/"shards/index.json", b'{"generation":2}')
        third = self.export("delta2.payload")
        self.assertEqual({r["path"] for r in third["files"]}, {"campaign/shards/index.json"})
        self.assertEqual(len(json.loads((self.workspace/"exports/checkpoint_index.json").read_text())["payloads"]), 3)

    def test_chain_roundtrip_latest_metadata_deleted_active_and_raw_shards(self):
        self.put(self.workspace/"active/wave/map/partial.json", b"SYNTHETIC interrupted raw trial")
        self.put(self.workspace/"state.json", b"old")
        self.export()
        self.put(self.workspace/"state.json", b"new")
        (self.workspace/"active/wave/map/partial.json").unlink()
        self.put(self.workspace/"shards/wave__map.zip", b"SYNTHETIC packed trial bytes")
        self.put(self.workspace/"shared/raster.npz", b"SYNTHETIC raster")
        self.export("delta.payload")
        index = self.mounted_index()
        # Emulate a fresh session at the required exact same absolute root.
        shutil.rmtree(self.workspace)
        shutil.rmtree(self.repo)
        self.restore(index)
        self.assertEqual((self.workspace/"state.json").read_bytes(), b"new")
        self.assertFalse((self.workspace/"active/wave/map/partial.json").exists())
        self.assertEqual((self.workspace/"shards/wave__map.zip").read_bytes(), b"SYNTHETIC packed trial bytes")
        self.assertEqual((self.repo/".venv-native-paper-v4/bin/python").read_bytes(), b"SYNTHETIC_NOT_EXECUTABLE")
        self.restore(index)  # Idempotent restore keeps all identities.
        delta = self.export("resumed.payload")
        self.assertFalse(delta["files"])

    def test_runtime_and_frozen_inputs_are_immutable(self):
        self.export()
        (self.repo/".venv-native-paper-v4/bin/python").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "immutable"):
            self.export("rejected.payload")
        self.assertFalse((self.workspace/"exports/rejected.payload").exists())

    def test_tampered_payload_or_external_shard_rejected_before_restore(self):
        self.put(self.workspace/"shards/wave__map.zip", b"SYNTHETIC shard")
        self.export()
        index = self.mounted_index()
        shard = self.inputs/"campaign/shards/wave__map.zip"
        shard.write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "external raw shard"):
            self.restore(index)
        shard.write_bytes(b"SYNTHETIC shard")
        (index.parent/"base.payload").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "payload size"):
            self.restore(index)

    def test_refuses_unverified_destination_and_wrong_source_identity(self):
        self.put(self.workspace/"state.json", b"known")
        self.export()
        index = self.mounted_index()
        (self.workspace/"state.json").write_bytes(b"unrelated user data")
        with self.assertRaisesRegex(ValueError, "unverified"):
            self.restore(index)
        self.assertEqual((self.workspace/"state.json").read_bytes(), b"unrelated user data")
        with self.assertRaisesRegex(ValueError, "source identity"):
            bundle.restore_checkpoint(index, self.base, bundle.sha256(index), source_sha256="wrong", root=self.repo, workspace=self.workspace)

    def test_preflight_space_and_output_cap_precede_archive(self):
        with patch.object(bundle.shutil, "disk_usage", return_value=SimpleNamespace(free=1)):
            with self.assertRaisesRegex(RuntimeError, "reserve"):
                self.export()
        with self.assertRaisesRegex(RuntimeError, "16 GiB"):
            self.export(max_output_bytes=1)
        self.assertFalse((self.workspace/"exports/base.payload").exists())

    def test_restore_preflight_disk_and_shared_model_identity(self):
        self.put(self.workspace/"shared/raster.npz", b"SYNTHETIC frozen raster")
        self.export()
        index = self.mounted_index()
        with patch.object(bundle.shutil, "disk_usage", return_value=SimpleNamespace(free=1)):
            with self.assertRaisesRegex(RuntimeError, "restore reserve"):
                self.restore(index)
        (self.workspace/"shared/raster.npz").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "immutable"):
            self.export("changed.payload")

    def test_interrupted_restore_resumes_only_verified_prefix(self):
        self.put(self.workspace/"state.json", b"SYNTHETIC interrupted restore data")
        self.export()
        index = self.mounted_index()
        shutil.rmtree(self.workspace)
        def interrupted(source, destination, *args):
            destination.write(source.read(4))
            raise RuntimeError("SYNTHETIC extraction interruption")
        with patch.object(bundle.shutil,"copyfileobj",side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError,"extraction interruption"):
                self.restore(index)
        self.assertTrue(list(self.workspace.rglob("*.restore-partial")))
        self.restore(index)
        self.assertEqual((self.workspace/"state.json").read_bytes(),b"SYNTHETIC interrupted restore data")
        self.assertFalse(list(self.workspace.rglob("*.restore-partial")))
        (self.workspace/"state.json").unlink()
        unknown=self.workspace/"state.json.restore-partial";unknown.write_bytes(b"unknown prefix")
        with self.assertRaisesRegex(ValueError,"unverified interrupted restore prefix"):
            self.restore(index)
        self.assertEqual(unknown.read_bytes(),b"unknown prefix")

    def test_all_stage_runs_ordered_gates_and_stops_on_error(self):
        args = SimpleNamespace(stage="all", allow_main=True, input_archive=self.source,
            input_sha256=bundle.sha256(self.source), workspace=self.workspace,
            session_started_monotonic=time.monotonic(), restored_runtime=False, mode="main")
        calls=[]
        def record(name, *args, **kwargs):
            calls.append(name)
            if name == "run_smoke-parallel":
                raise RuntimeError("SYNTHETIC gate failure")
            return "{}"
        with patch.object(runtime.sys, "platform", "linux"), patch.object(runtime.platform, "machine", return_value="x86_64"), \
             patch.object(runtime, "verify", return_value={"kind":bundle.KIND,"files":[]}), \
             patch.object(runtime, "isolated_runtimes", return_value=({"controller":"python"},{})), \
             patch.object(runtime, "build"), patch.object(runtime, "retain_checkpoint") as retained, \
             patch.object(runtime, "command", side_effect=record):
            with self.assertRaisesRegex(RuntimeError, "gate failure"):
                runtime.execute(args)
        self.assertEqual(calls,["prepare","correctness","run_smoke-serial","report_smoke-serial","run_smoke-parallel"])
        self.assertEqual(retained.call_count,3)
        self.assertIsInstance(retained.call_args.kwargs["stage_error"], RuntimeError)

    def test_source_bundle_pins_notebook_and_base64(self):
        self.put(self.repo/"notebooks/native_paper_kaggle_v4.ipynb", (ROOT/"notebooks/native_paper_kaggle_v4.ipynb").read_bytes())
        fixture = self.repo/"synthetic.txt"
        fixture.write_text("SYNTHETIC source")
        output = self.temp/"native_paper_v4_source.zip"
        with patch.object(bundle, "source_entries", return_value=[("synthetic.txt", fixture)]):
            result = bundle.build_source(output, root=self.repo)
        self.assertEqual(result["kind"], bundle.KIND)
        self.assertEqual(base64.b64decode(output.with_suffix(".b64").read_bytes()), output.read_bytes())
        notebook = output.with_suffix(".ipynb").read_text(encoding="utf-8")
        self.assertIn(bundle.sha256(output), notebook)
        self.assertNotIn("SOURCE_SHA256_PLACEHOLDER", notebook)

    @unittest.skipUnless(os.name != "nt" and sys.platform.startswith("linux"), "exact Linux runtime symlinks require Linux")
    def test_linux_runtime_links_are_preserved(self):
        real = self.repo/".managed-python/bin/python3.12"
        self.put(real, b"SYNTHETIC managed interpreter")
        alias = self.repo/".venv-native-paper-v4/bin/python"
        alias.unlink()
        alias.symlink_to(os.path.relpath(real, alias.parent))
        self.export()
        index = self.mounted_index()
        shutil.rmtree(self.repo)
        self.restore(index)
        self.assertTrue(alias.is_symlink())
        self.assertEqual(alias.resolve(), real)

    def test_export_failure_preserves_original_stage_exception(self):
        original = RuntimeError("SYNTHETIC_PRIMARY_FAILURE")
        with patch.object(runtime, "export_checkpoint", side_effect=ValueError("SYNTHETIC_EXPORT_FAILURE")), contextlib.redirect_stderr(io.StringIO()):
            self.assertIsNone(runtime.retain_checkpoint(self.workspace, self.source, stage_error=original))
        receipt = next((self.workspace/"logs/bootstrap").glob("*_export_failure.json"))
        value = json.loads(receipt.read_text())
        self.assertIn("SYNTHETIC_PRIMARY_FAILURE", value["stage_error"])
        self.assertIn("SYNTHETIC_EXPORT_FAILURE", value["export_error"])

    def test_failed_process_cleanup_blocks_export_even_without_journal(self):
        with patch.object(runtime,"SUPERVISION_ERROR","SYNTHETIC live orphan"), \
             patch.object(runtime,"export_checkpoint") as export, contextlib.redirect_stderr(io.StringIO()):
            self.assertIsNone(runtime.retain_checkpoint(self.workspace,self.source,stage_error=RuntimeError("original")))
        export.assert_not_called()

    def test_main_preflight_captures_fresh_hardware_and_rejects_under_four_cpu(self):
        evidence = json.dumps(dict(cpu_capacity=2, memory_limit_bytes=8*1024**3))
        with patch.object(runtime, "command", return_value=evidence), patch.object(runtime.shutil, "disk_usage", return_value=SimpleNamespace(free=10*1024**3)):
            with self.assertRaisesRegex(RuntimeError, "four"):
                runtime.main_preflight(self.workspace, time.monotonic(), {"controller": "python"}, {})
        self.assertTrue((self.workspace/"setup/main_preflight.json").is_file())

    def test_orphan_tracking_uses_identity_and_new_subreaper_children(self):
        pid=os.getpid()
        snapshot={10:dict(parent=pid,start_ticks=100,state="S",created=1.),
                  20:dict(parent=10,start_ticks=200,state="S",created=2.),
                  30:dict(parent=pid,start_ticks=300,state="S",created=3.)}
        tracked={}
        self.assertEqual(runtime.track_descendants(snapshot,10,tracked,{(30,300)}),{10,20})
        # Leader exited; its worker started a separate session and reparented.
        snapshot.pop(10);snapshot[20]["parent"]=pid
        self.assertEqual(runtime.track_descendants(snapshot,10,tracked,{(30,300)}),{20})
        snapshot[20]=dict(parent=1,start_ticks=999,state="S",created=9.)
        self.assertEqual(runtime.track_descendants(snapshot,10,tracked,{(30,300)}),set())

    def test_export_refuses_live_launch_identity_and_accepts_reused_pid(self):
        self.put(self.workspace/"active/trial/launch.json",json.dumps(dict(pid=123,process_created=10.)).encode())
        with patch.object(runtime,"linux_host",return_value=True), \
             patch.object(runtime,"process_snapshot",return_value={123:dict(parent=1,start_ticks=100,state="S",created=10.)}):
            with self.assertRaisesRegex(RuntimeError,"live campaign"):
                runtime.assert_workspace_quiescent(self.workspace)
        with patch.object(runtime,"linux_host",return_value=True), \
             patch.object(runtime,"process_snapshot",return_value={123:dict(parent=1,start_ticks=200,state="S",created=20.)}):
            runtime.assert_workspace_quiescent(self.workspace)

    @unittest.skipUnless(os.name != "nt" and sys.platform.startswith("linux"),"Linux subreaper/process groups require Linux")
    def test_exited_leader_cannot_leave_independent_session_worker_alive(self):
        pidfile=self.temp/"orphan_pid.txt"
        child="import time;time.sleep(60)"
        parent=("import subprocess,sys;from pathlib import Path;"
                f"p=subprocess.Popen([sys.executable,'-c',{child!r}],start_new_session=True);"
                f"Path({str(pidfile)!r}).write_text(str(p.pid))")
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError,"failed/partial"):
                runtime.command("synthetic_orphan",[sys.executable,"-c",parent],self.workspace,time.monotonic(),cwd=self.temp)
        orphan=int(pidfile.read_text())
        self.assertNotIn(orphan,{pid for pid,value in runtime.process_snapshot().items() if value["state"] not in ("Z","X")})


if __name__ == "__main__":
    unittest.main()
