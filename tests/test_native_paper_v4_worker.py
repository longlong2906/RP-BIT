"""Synthetic adapter and lossless-storage tests; no benchmark solver is run."""
import copy
import csv
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tools"))
import native_paper_v4_protocol as protocol
import native_paper_v4_worker as worker
from safe_length_field import build_field, build_field_file


class WorkerTests(unittest.TestCase):
    def setUp(self):
        fixtures = ROOT/"tests/fixtures/native_paper_v4"
        fixtures.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="synthetic_worker_", dir=fixtures)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.map = self.root/"synthetic.t25d"
        self.map.write_text("SYNTHETIC_PROVENANCE_ONLY_NOT_PLANNER_INPUT")
        self.cloud = self.map.with_suffix(".cloud.npy")
        x, y = np.meshgrid((np.arange(21)+.5)*.1, (np.arange(21)+.5)*.1)
        self.points = np.column_stack([x.ravel(),y.ravel(),(.1*x).ravel()])
        np.save(self.cloud,self.points,allow_pickle=False)
        self.field = self.map.with_suffix(".safe_sem.txt")
        build_field_file(self.cloud,self.map,[0,0,2.1,2.1],self.field)

    def request(self, backend="a3t_safe"):
        return protocol.request_for(backend,dict(start=[.5,.5],goal=[1.5,1.5]),self.map,self.cloud,
            worker.sha256(self.cloud),11,2.,protocol.default_config())

    def event_trial(self, name="trial", backend="a3t_old", xyz=False):
        directory = self.root/name
        snap = directory/"snapshots/incumbent_000000"
        snap.mkdir(parents=True)
        xy = [[.5,.5],[1.,1.25],[1.5,1.5]]
        raw = [[x,y,x/10] for x,y in xy]
        for filename, coords, header in (("path_xy.csv",xy,["x","y"]),("path_raw_xyz.csv",raw,["x","y","z"])):
            with (snap/filename).open("w",newline="",encoding="utf-8") as stream:
                writer = csv.writer(stream); writer.writerow(header); writer.writerows(coords)
        event = dict(snapshot_id="incumbent_000000",elapsed_s=.125,timestamp_s=.125,
            native_objective_value=4.2,path_xy=xy,counters=dict(samples=20))
        if xyz: event["raw_path_xyz"] = raw
        result = dict(status="EXACT_SOLUTION",snapshots=[event],backend=backend,
            solve_started_utc="2026-01-01T00:00:00+00:00",scalar_unmodified=1.2345678901234567)
        worker.dump(directory/"worker.json",result)
        events = [dict(snapshot_id=None,elapsed_s=0.,counter=0),event,dict(snapshot_id=None,elapsed_s=1.,counter=300)]
        (directory/"progress.jsonl").write_text("".join(json.dumps(e)+"\n" for e in events),encoding="utf-8")
        worker.dump(directory/"phase.json",dict(phase="FINISHED"))
        return directory, result, events

    def add_native(self,directory,result,array=None,zip_variant=False):
        folder = directory/"native_model"
        folder.mkdir()
        raster = folder/"raster.npz"
        values = np.arange(9,dtype=np.float64).reshape(3,3) if array is None else array
        np.savez_compressed(raster,risk=values,observed=np.ones(values.shape,dtype=np.uint8))
        if zip_variant:
            # Same arrays, different valid ZIP serialization: keep a separate byte object.
            with zipfile.ZipFile(raster,"a") as archive: archive.comment = b"synthetic_second_serialization"
        metadata = dict(version="SYNTHETIC_NATIVE_MODEL_ONLY",resolution=.1,
            construction_wall_s=.123,raster_sha256=worker.sha256(raster),raster_file="raster.npz",
            raster_artifacts={"raster.npz":worker.sha256(raster)})
        worker.dump(folder/"raster.json",metadata)
        result["native_model"] = dict(raster_artifacts={"raster.npz":worker.sha256(raster),
            "raster.json":worker.sha256(folder/"raster.json")})
        worker.dump(directory/"worker.json",result)
        return raster,metadata

    def test_per_trial_rebuild_is_byte_identical_and_immutable(self):
        req = self.request()
        before = self.field.read_bytes()
        field = build_field(self.points,[0,0,2.1,2.1])
        self.assertEqual(worker.field_bytes(field),before)
        with patch("safe_length_field.build_field",wraps=build_field) as build:
            first, second = worker.prepare_safe_field(req), worker.prepare_safe_field(req)
            self.assertEqual(build.call_count,2)
        self.assertEqual(first["field_sha256"],worker.sha256(self.field))
        self.assertGreaterEqual(second["field_build_s"],0.)
        self.assertTrue(first["rebuilt_for_this_trial"])
        self.assertEqual(self.field.read_bytes(),before)

    def test_rebuild_rejects_tampered_bytes_and_policy_drift(self):
        req = self.request()
        worker.validate_request(req)
        for key,value in (("risk_cap",41),("slope_cap_deg",36),("atlas_sampling",False)):
            altered = copy.deepcopy(req);altered["safe_length"][key] = value
            with self.assertRaises(ValueError): worker.validate_request(altered)
        altered = copy.deepcopy(req);altered["sampler_query_cache"] = False
        with self.assertRaises(ValueError): worker.validate_request(altered)
        self.field.write_bytes(self.field.read_bytes()+b"\n")
        with self.assertRaises(ValueError): worker.prepare_safe_field(req)

    def test_wrapper_prepares_then_launches_cpp_with_original_request(self):
        req = self.request()
        binary = self.root/"synthetic_binary_no_execution"
        binary.write_bytes(b"NOT_EXECUTABLE_SYNTHETIC")
        req["worker_binary"] = str(binary)
        reqfile = self.root/"request.json";worker.dump(reqfile,req)
        output = self.root/"adapter"
        # Actual Windows launch boundary is mocked; no planner/fake result runs.
        with patch.object(worker.sys,"platform","win32"), patch.object(worker.subprocess,"call",return_value=0) as launch:
            self.assertEqual(worker.run(reqfile,output),0)
        self.assertEqual(launch.call_args.args[0],[str(binary.resolve()),str(reqfile.resolve()),str(output.resolve())])
        self.assertEqual(worker.read(reqfile),req)
        prep = worker.read(output/"preparation.json")
        self.assertEqual(prep["status"],"PREPARED")
        self.assertTrue(prep["rebuilt_for_this_trial"])
        self.assertTrue(all(v=="1" for v in prep["thread_environment"].values()))
        with self.assertRaises(ValueError):worker.run(reqfile,output)

    def test_linux_launch_uses_exec_preserving_process_identity(self):
        class ExecObserved(BaseException): pass
        req=self.request("a3t_old")
        binary=self.root/"synthetic_binary_no_execution";binary.write_bytes(b"NOT_EXECUTABLE_SYNTHETIC")
        req["worker_binary"]=str(binary)
        reqfile=self.root/"linux.json";worker.dump(reqfile,req)
        output=self.root/"linux_adapter"
        with patch.object(worker.sys,"platform","linux"), patch.object(worker.os,"execvpe",side_effect=ExecObserved) as launch:
            with self.assertRaises(ExecObserved):worker.run(reqfile,output)
        self.assertEqual(launch.call_args.args[0],str(binary.resolve()))
        self.assertEqual(worker.read(output/"preparation.json")["status"],"PREPARED")

    def test_native_wrapper_preserves_frozen_request_and_translates_only_backend(self):
        req = self.request("sem_native")
        reqfile = self.root/"native.json";worker.dump(reqfile,req)
        output = self.root/"native_adapter"
        with patch("native_paper_worker.run",return_value=0) as run:
            self.assertEqual(worker.run(reqfile,output),0)
        translated = worker.read(run.call_args.args[0])
        expected = copy.deepcopy(req);expected["backend"]="sem"
        self.assertEqual(translated,expected)
        self.assertEqual(worker.read(reqfile),req)
        self.assertEqual(worker.read(output/"preparation.json")["field_build_s"],0.)

    def test_invalid_policy_is_recorded_as_failure(self):
        req = self.request();req["safe_length"]["risk_cap"] = 99
        reqfile = self.root/"invalid.json";worker.dump(reqfile,req)
        output = self.root/"invalid_adapter"
        self.assertEqual(worker.run(reqfile,output),1)
        self.assertEqual(worker.read(output/"worker.json")["execution_status"],"FAILED_CORRECTNESS")
        self.assertEqual(worker.read(output/"preparation.json")["status"],"FAILED")

    def test_compaction_preserves_scalars_order_and_canonical_csv(self):
        directory,result,events = self.event_trial(xyz=True)
        csvhash = worker.sha256(directory/"snapshots/incumbent_000000/path_xy.csv")
        receipt = worker.compact_trial(directory,self.request("a3t_old"),self.root/"shared")
        stripped = copy.deepcopy(result)
        for event in stripped["snapshots"]:
            del event["path_xy"];del event["raw_path_xyz"]
        self.assertEqual(worker.read(directory/"worker.json"),stripped)
        for event in events:
            event.pop("path_xy",None);event.pop("raw_path_xyz",None)
        self.assertEqual([json.loads(line) for line in (directory/"progress.jsonl").read_text().splitlines()],events)
        self.assertEqual(worker.sha256(directory/"snapshots/incumbent_000000/path_xy.csv"),csvhash)
        self.assertEqual(receipt["removed_coordinate_fields"],4)
        self.assertEqual(worker.compact_trial(directory,self.request("a3t_old"),self.root/"shared"),receipt)

    def test_coordinate_mismatch_and_active_trial_fail_without_mutation(self):
        directory,result,_ = self.event_trial()
        worker.dump(directory/"phase.json",dict(phase="SOLVING"))
        with self.assertRaises(ValueError):worker.compact_trial(directory,self.request("a3t_old"),self.root/"shared")
        worker.dump(directory/"phase.json",dict(phase="FINISHED"))
        result["snapshots"][0]["path_xy"][1][0] = 999.
        worker.dump(directory/"worker.json",result)
        before = (directory/"worker.json").read_bytes()
        with self.assertRaises(ValueError):worker.compact_trial(directory,self.request("a3t_old"),self.root/"shared")
        self.assertEqual((directory/"worker.json").read_bytes(),before)
        self.assertFalse((directory/"compaction.json").exists())

    def test_cpp_csv_meter_headers_are_explicitly_supported(self):
        directory,_,_=self.event_trial()
        path=directory/"snapshots/incumbent_000000/path_xy.csv"
        path.write_text(path.read_text().replace("x,y","x_m,y_m",1))
        self.assertEqual(worker.compact_trial(directory,self.request("a3t_old"),self.root/"shared")["status"],"COMPLETE")

    def test_native_sharing_requires_identical_bytes_and_keeps_metadata(self):
        req = self.request("sem_native")
        shared = self.root/"shared"
        references = []
        for index in range(3):
            directory,result,_ = self.event_trial("native_"+str(index),"sem",xyz=True)
            raster,metadata = self.add_native(directory,result,zip_variant=index==2)
            original = raster.read_bytes()
            receipt = worker.compact_trial(directory,req,shared)
            ref = receipt["native_raster_reference"]
            references.append(ref)
            self.assertEqual((directory/ref["native_raster_path"]).read_bytes(),original)
            self.assertFalse(raster.exists())
            self.assertEqual(worker.read(directory/"native_model/raster.json"),metadata)
            self.assertEqual(worker.read(directory/"worker.json")["native_model"],result["native_model"])
        self.assertEqual(references[0]["native_raster_sha256"],references[1]["native_raster_sha256"])
        self.assertNotEqual(references[1]["native_raster_sha256"],references[2]["native_raster_sha256"])
        self.assertEqual(len({r["native_raster_semantic_sha256"] for r in references}),1)
        self.assertEqual(len(list(shared.rglob("*.npz"))),2)

    def test_semantic_identity_binds_shape_dtype_and_config(self):
        directory,result,_=self.event_trial("semantic", "sem",xyz=True)
        raster,meta=self.add_native(directory,result)
        req=self.request("sem_native")
        original=worker.native_raster_identity(raster,meta,req)[0]
        altered=copy.deepcopy(req);altered["representation_config"]["sem_resolution_m"] = .2
        self.assertNotEqual(original,worker.native_raster_identity(raster,meta,altered)[0])
        np.savez_compressed(raster,risk=np.arange(9,dtype=np.float32).reshape(3,3),observed=np.ones((3,3),dtype=np.uint8))
        self.assertNotEqual(original,worker.native_raster_identity(raster,meta,req)[0])

    def test_partial_compaction_resumes_without_double_counting(self):
        directory,_,_=self.event_trial(xyz=True)
        req=self.request("a3t_old")
        original_atomic=worker.atomic_bytes
        failed=[False]
        def interrupt(path,data):
            if Path(path).name=="progress.jsonl" and not failed[0]:
                failed[0]=True
                raise OSError("synthetic interruption after worker rewrite")
            return original_atomic(path,data)
        with patch.object(worker,"atomic_bytes",side_effect=interrupt):
            with self.assertRaises(OSError):worker.compact_trial(directory,req,self.root/"shared")
        self.assertTrue((directory/"compaction.pending.json").exists())
        receipt=worker.compact_trial(directory,req,self.root/"shared")
        self.assertEqual(receipt["removed_coordinate_fields"],4)
        self.assertFalse((directory/"compaction.pending.json").exists())


if __name__ == "__main__":unittest.main()
