"""Preparation/guard tests with labeled synthetic temporary outputs; no solves."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from run_corridor_v3 import prepare,run,execution_gate
from corridor_v3_protocol import default_config,inventory,source_hashes
from run_safe_length import dump,read
from safe_length_field import file_hash


class RunnerTests(unittest.TestCase):
    def test_pilot_and_full_require_explicit_flags_before_io(self):
        for profile in ("pilot","full"):
            with self.assertRaisesRegex(ValueError,"requires"):
                run("does-not-exist","does-not-exist",profile=profile)
        execution_gate("pilot",allow_pilot=True)
        execution_gate("full",allow_full=True)

    def test_inventory_does_not_launch_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory(prefix="corridor_inventory_synthetic_",dir=ROOT/"tests/fixtures") as tmp:
            output=Path(tmp)/"inventory"
            with mock.patch("subprocess.Popen",side_effect=AssertionError("unexpected planner")):
                manifest=prepare(output,profile="pilot",dry_run=True)
            self.assertEqual(manifest["status"],"PREPARED_INVENTORY")
            self.assertEqual(manifest["planned_trials"],270)
            self.assertEqual(manifest["maps"],[])
            self.assertFalse((output/"inputs").exists())
            with self.assertRaises(FileExistsError):
                prepare(output,profile="pilot",dry_run=True)
            with self.assertRaisesRegex(ValueError,"unattempted"):
                run(output,"does-not-exist",profile="pilot",allow_pilot=True)

    def test_full_selection_required_except_inventory(self):
        with tempfile.TemporaryDirectory(prefix="corridor_full_synthetic_",dir=ROOT/"tests/fixtures") as tmp:
            output=Path(tmp)/"full"
            with self.assertRaisesRegex(ValueError,"selection"):
                prepare(output,profile="full")
            self.assertFalse(output.exists())
            manifest=prepare(output,profile="full",dry_run=True)
            self.assertEqual(manifest["planned_trials"],2000)
            self.assertTrue(manifest["requires_selected_pilot"])

    def test_interrupt_is_journaled_without_retry_or_fake_success(self):
        with tempfile.TemporaryDirectory(prefix="corridor_interrupt_synthetic_",dir=ROOT/"tests/fixtures") as tmp:
            output=Path(tmp)/"campaign"
            output.mkdir()
            worker=Path(tmp)/"synthetic_worker.exe"
            worker.write_bytes(b"synthetic mock dependency, never executed")
            rawmap=Path(tmp)/"synthetic.t25d"
            rawmap.write_bytes(b"synthetic raw input for journal test")
            cloud=Path(tmp)/"synthetic.npy"
            np.save(cloud,np.zeros((1,3)),allow_pickle=False)
            config=default_config()
            manifest=inventory(config)
            manifest.update(status="PREPARED",sources=source_hashes(),maps=[dict(
                map_id=entry["map_id"],family=entry["family"],map_path=str(rawmap),
                map_sha256=file_hash(rawmap),cloud_path=str(cloud),cloud_sha256=file_hash(cloud),
                bounds=[0.,0.,20.,20.],start=[1.,1.],goal=[5.,5.])
                for entry in manifest["entries"]])
            dump(output/"config.json",config)
            dump(output/"manifest.json",manifest)
            def build_field(cloud,map_path,bounds,path):
                Path(path).write_bytes(b"synthetic field not passed to a process")
                return dict(field_sha256=file_hash(path),cloud_sha256=file_hash(cloud),builder_source_sha256="synthetic")
            process=mock.Mock(pid=123,returncode=130)
            process.wait.side_effect=KeyboardInterrupt
            process.poll.return_value=130
            with mock.patch("run_corridor_v3.build_field_file",side_effect=build_field), \
                 mock.patch("run_corridor_v3.platform.platform",return_value="synthetic-test"), \
                 mock.patch("subprocess.Popen",return_value=process) as launch, \
                 mock.patch("native_paper_oracle.NativePaperOracle"), \
                 mock.patch("native_paper_v3_sem_evaluator.SharedSEMField"):
                result=run(output,worker)
            self.assertEqual(launch.call_count,1)
            self.assertEqual(result["status"],"INTERRUPTED")
            self.assertEqual(result["attempted"],1)
            self.assertEqual(result["success"],0)
            execution=read(output/"trials"/manifest["entries"][0]["trial_id"]/"execution.json")
            self.assertEqual(execution["exit_code"],130)
            self.assertTrue(execution["launched"])
            self.assertEqual(execution["status"],"INTERRUPTED")
            with self.assertRaisesRegex(ValueError,"unattempted"):
                run(output,worker)


if __name__=="__main__":
    unittest.main()
