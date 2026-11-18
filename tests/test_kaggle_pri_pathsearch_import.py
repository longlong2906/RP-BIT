"""Synthetic transport faults; no planner or measured experiment is created."""
import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
sys.path.insert(0,str(ROOT/"kaggle_pri_pathsearch"))
import import_windows as importer
import pri_pathsearch_protocol as protocol
from corridor_v4_execution import atomic_json as dump, digest, file_hash, read
from run_corridor_v4 import raw_inventory


class ImportTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="__kaggle_import_synthetic_",dir=ROOT/"tests/fixtures")
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.phase=self.root/"checkpoint/pilot"
        self.out=self.root/"validation"
        config=protocol.default_config()
        self.manifest=protocol.inventory(config,"pilot")
        self.manifest.update(status="INTERRUPTED",synthetic_fixture=True,sources={"synthetic":"not_scientific_code"},
            maps=[],environment={"platform":"Windows synthetic fixture"})
        for entry in self.manifest["entries"]:
            if entry["map_id"] in {r["map_id"] for r in self.manifest["maps"]}:continue
            mid=entry["map_id"]
            folder=self.phase/"inputs";folder.mkdir(parents=True,exist_ok=True)
            for suffix in (".t25d",".cloud.npy",".json"):
                (folder/(mid+suffix)).write_text("SYNTHETIC TRANSPORT FIXTURE",encoding="utf-8")
            self.manifest["maps"].append(dict(map_id=mid,family=entry["family"],map_seed=entry["map_seed"],
                query_id="q0",start=[1.,1.],goal=[2.,2.],bounds=[0.,0.,40.,40.],
                map_path="D:\\original\\pilot\\inputs\\"+mid+".t25d",
                cloud_path="D:\\original\\pilot\\inputs\\"+mid+".cloud.npy",
                map_sha256=file_hash(folder/(mid+".t25d")),cloud_sha256=file_hash(folder/(mid+".cloud.npy")),
                metadata_sha256=file_hash(folder/(mid+".json"))))
        journal=[]
        for index,entry in enumerate(self.manifest["entries"][:3]):
            trial=self.phase/"trials"/entry["trial_id"]
            dump(trial/"execution.json",dict(status="COMPLETED",exit_code=0,launched=True))
            dump(trial/"process_exit.json",dict(status="COMPLETED",exit_code=0,process_wall_s=20.))
            inv=raw_inventory(trial);dump(trial/"raw_inventory.json",inv)
            entry.update(status="COMPLETED" if index==0 else "RAW_READY",raw_inventory_sha256=digest(inv))
            journal.append(dict(trial_id=entry["trial_id"],config_sha256=self.manifest["config_sha256"],
                sources_sha256=digest(self.manifest["sources"])))
            if index==0:
                row=self.row(entry)
                original=next(i for i in self.manifest["maps"] if i["map_id"]==entry["map_id"])
                proof=dict(identity=dict(trial_id=entry["trial_id"],raw_inventory_sha256=digest(inv),
                    sources_sha256=digest(self.manifest["sources"]),info_sha256=digest(original)),row=row)
                spool=self.phase/"execution_control/validation_spool"/(entry["trial_id"]+".json")
                dump(trial/"validation.json",row);dump(spool,proof)
                entry.update(validation_sha256=file_hash(trial/"validation.json"),validation_spool_sha256=file_hash(spool))
        dump(self.phase/"manifest.json",self.manifest);dump(self.phase/"config.json",config)
        (self.phase/"execution_control/start_journal.jsonl").write_text(
            "\n".join(__import__('json').dumps(r) for r in journal)+"\n",encoding="utf-8")
        self.dispatched=[]

    @staticmethod
    def row(entry):
        return dict(trial_id=entry["trial_id"],success=False,execution_status="CRASH",correctness_errors=[],
            synthetic_fixture=True,attained_safe=False,planner_status=None,selector_results=[])

    def spawn(self,lanes,target,control,**kwargs):
        self.assertIs(target,importer._validate_map)
        Path(control).mkdir(parents=True,exist_ok=True)
        for lane in lanes:
            for task in lane:
                for item in task["items"]:
                    entry=item["entry"];self.dispatched.append(entry["trial_id"])
                    value=dict(identity=item["identity"],row=self.row(entry),validation_platform="synthetic Linux",physical_planner_execution_count=0)
                    dump(Path(task["spool"])/(entry["trial_id"]+".json"),dict(value,spool_sha256=digest(value)))
        return {"status":"COMPLETE"}

    def execute(self,**kwargs):
        with patch.object(importer,"resource_preflight",return_value=dict(admitted=True,cores=[0,1,2,3])),\
             patch.object(importer,"run_spawn_lanes",side_effect=self.spawn):
            return importer.import_checkpoint(self.phase,self.out,**kwargs)

    def make_orphan(self,index=1):
        entry=self.manifest["entries"][index]
        original=next(i for i in self.manifest["maps"] if i["map_id"]==entry["map_id"])
        row=self.row(entry)
        row.update({k:entry[k] for k in ("method","map_id","planner_seed","family","alpha","sem_lambda","policy","map_seed","query_id","phase")})
        row.update(execution_status="COMPLETED",execution_launched=True,validation_s=.25,events=[])
        proof=dict(identity=dict(trial_id=entry["trial_id"],raw_inventory_sha256=entry["raw_inventory_sha256"],
            sources_sha256=digest(self.manifest["sources"]),info_sha256=digest(original)),row=row,
            created_utc="2026-11-13T00:00:00+00:00",validator_pid=123)
        path=self.phase/"execution_control/validation_spool"/(entry["trial_id"]+".json")
        dump(path,proof)
        return path

    def test_existing_verdict_unchanged_pending_only_and_resume_skips_completed(self):
        before={str(p.relative_to(self.phase)):file_hash(p) for p in self.phase.rglob('*') if p.is_file()}
        result=self.execute()
        self.assertEqual(result["status"],"COMPLETE")
        self.assertEqual(result["original_committed_verdict_count"],1)
        self.assertEqual(result["imported_pending_verdict_count"],2)
        self.assertEqual(result["pending_trial_ids"],[])
        self.assertEqual(len(result["unstarted_trial_ids"]),372)
        self.assertEqual(len(self.dispatched),2)
        self.assertEqual(result["physical_planner_execution_count"],0)
        self.dispatched.clear();self.execute(resume=True)
        self.assertEqual(self.dispatched,[])
        self.assertEqual(before,{str(p.relative_to(self.phase)):file_hash(p) for p in self.phase.rglob('*') if p.is_file()})

    def test_verified_original_orphan_spool_is_reused_without_revalidation(self):
        path=self.make_orphan();before=file_hash(path)
        result=self.execute()
        self.assertEqual(result["status"],"COMPLETE")
        self.assertEqual(result["original_committed_verdict_count"],1)
        self.assertEqual(result["recovered_original_spool_count"],1)
        self.assertEqual(result["imported_pending_verdict_count"],1)
        self.assertEqual(self.dispatched,[self.manifest["entries"][2]["trial_id"]])
        self.assertEqual(file_hash(path),before)
        receipt=next(r for r in result["original_verdict_receipts"] if r["kind"]=="uncommitted_original_spool")
        self.assertEqual(receipt["spool_sha256"],before)
        self.assertFalse((self.phase/"trials"/self.manifest["entries"][1]["trial_id"]/"validation.json").exists())
        self.dispatched.clear();self.execute(resume=True)
        self.assertEqual(self.dispatched,[])

    def test_invalid_original_orphan_spool_fails_integrity_never_revalidates(self):
        path=self.make_orphan();original=read(path)
        for section,key,value in (("identity","sources_sha256","wrong"),("identity","info_sha256","wrong"),
                                  ("row","method","sem_native"),("row","success",1),
                                  ("row","execution_status","CRASH")):
            altered=copy.deepcopy(original);altered[section][key]=value;dump(path,altered)
            with self.assertRaisesRegex(ValueError,"original orphan spool"):
                self.execute()
            self.assertEqual(self.dispatched,[])
            dump(path,original)

    def test_mutated_raw_is_rejected_before_dispatch(self):
        e=self.manifest["entries"][1]
        (self.phase/"trials"/e["trial_id"]/"execution.json").write_text("tampered",encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"raw trial changed"):
            self.execute()
        self.assertEqual(self.dispatched,[])

    def test_existing_verdict_and_spool_are_hash_bound(self):
        e=self.manifest["entries"][0]
        dump(self.phase/"trials"/e["trial_id"]/"validation.json",dict(trial_id=e["trial_id"],success=True))
        with self.assertRaisesRegex(ValueError,"verdict or spool changed"):
            self.execute()

    def test_resume_spool_tamper_and_duplicate_journal_fail(self):
        self.execute()
        path=next((self.out/"validation_spool").glob('*.json'))
        value=read(path);value["row"]["success"]=True;dump(path,value)
        with self.assertRaisesRegex(ValueError,"spool changed"):
            self.execute(resume=True)
        journal=self.phase/"execution_control/start_journal.jsonl"
        journal.write_text(journal.read_text()+journal.read_text().splitlines()[0]+"\n",encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"duplicate original"):
            importer.audit_checkpoint(self.phase)

    def test_deadline_keeps_pending_and_does_not_invent_failures(self):
        with patch.object(importer,"resource_preflight",return_value=dict(admitted=True,cores=[0,1,2,3])),\
             patch.object(importer,"run_spawn_lanes",return_value={"status":"DEADLINE"}) as spawn:
            result=importer.import_checkpoint(self.phase,self.out,deadline_seconds=120.)
        self.assertEqual(result["status"],"PARTIAL")
        self.assertEqual(len(result["rows"]),1)
        self.assertEqual(len(result["pending_trial_ids"]),2)
        self.assertEqual(result["correctness_failures"],0)
        self.assertLess(spawn.call_args.kwargs['stop_dispatch_at_monotonic'],spawn.call_args.kwargs['terminate_at_monotonic'])

    def test_missing_exit_receipt_is_interrupted_without_planner_retry(self):
        trial=self.phase/"trials"/self.manifest["entries"][1]["trial_id"]
        (trial/"process_exit.json").unlink()
        execution=importer._execution(trial)
        self.assertEqual(execution["status"],"INTERRUPTED")
        self.assertIsNone(execution["exit_code"])
        self.assertEqual(read(trial/"execution.json")["status"],"COMPLETED")

    def test_field_adapter_changes_only_file_read_and_restores_bindings(self):
        import pri_portfolio_validation as old
        import pri_pathsearch_validation as new
        trial=self.root/"read_adapter";trial.mkdir()
        field=trial/"planner_sem_field.txt";field.write_text("SYNTHETIC FIELD CONTENT",encoding="ascii")
        request=dict(safe_length=dict(field_path="D:\\original\\trial\\planner_sem_field.txt",
            field_sha256=file_hash(field),risk_cap=40.,slope_cap_deg=35.))
        frozen=copy.deepcopy(request)
        def read_boundary(raw,policy,numeric):
            self.assertEqual(Path(policy["field_path"]),field.resolve())
            self.assertEqual(policy["risk_cap"],40.)
            return Path(policy["field_path"]).read_text()
        with patch.object(old,"verify_mask_input",side_effect=read_boundary) as before_old:
            before_new=new.verify_mask_input
            with importer.field_read_adapter(trial,request) as receipt:
                self.assertEqual(old.verify_mask_input(None,request["safe_length"],None),"SYNTHETIC FIELD CONTENT")
                self.assertEqual(new.verify_mask_input(None,request["safe_length"],None),"SYNTHETIC FIELD CONTENT")
                self.assertFalse(receipt["raw_request_changed"])
                with self.assertRaisesRegex(ValueError,"different frozen policy"):
                    new.verify_mask_input(None,dict(request["safe_length"],risk_cap=41.),None)
            self.assertIs(old.verify_mask_input,before_old)
            self.assertIs(new.verify_mask_input,before_new)
        self.assertEqual(request,frozen)

    def test_scientific_catalog_unchanged_by_transport_modules(self):
        hashes=protocol.source_hashes()
        self.assertNotIn("kaggle_pri_pathsearch/import_windows.py",hashes)
        self.assertNotIn("tests/test_kaggle_pri_pathsearch_import.py",hashes)

    def numerical_artifact(self, folder=None):
        import numpy as np
        folder=folder or self.root/"numeric_reference";folder.mkdir(parents=True,exist_ok=True)
        risk=np.array([[1.,2.],[3.,np.nan]],dtype=np.float64)
        valid=np.array([[True,True],[True,False]],dtype=np.bool_)
        path=folder/"synthetic.npz"
        np.savez_compressed(path,risk=risk,valid=valid,bounds=np.array([0.,0.,.2,.2]),resolution=np.asarray(.1))
        return dict(path=str(path),file_sha256=file_hash(path),shape=[2,2],bounds=[0.,0.,.2,.2],resolution=.1,
            risk_bits_sha256=importer._array_digest(risk.view(np.uint64)),valid_bits_sha256=importer._array_digest(valid))

    def test_windows_array_transport_preserves_bits_and_detects_one_ulp_change(self):
        import numpy as np
        reference=self.numerical_artifact()
        value=importer.load_windows_reference(reference)
        self.assertFalse(value.risk.flags.writeable)
        self.assertTrue(np.isnan(value.risk[1,1]))
        changed=value.risk.copy();changed[0,0]=np.nextafter(changed[0,0],np.inf)
        np.savez_compressed(reference["path"],risk=changed,valid=value.valid,bounds=np.asarray(value.bounds),resolution=np.asarray(value.resolution))
        with self.assertRaisesRegex(ValueError,"reference changed"):
            importer.load_windows_reference(reference)
        reference["file_sha256"]=file_hash(reference["path"])
        with self.assertRaisesRegex(ValueError,"array bits"):
            importer.load_windows_reference(reference)

    def test_linux_pending_import_requires_explicit_windows_reference(self):
        with patch.object(importer.platform,"system",return_value="Linux"):
            with self.assertRaisesRegex(ValueError,"Windows independent numerical reference required"):
                self.execute()
        self.assertEqual(self.dispatched,[])

    def test_transported_reference_does_not_weaken_ulp_or_risk40_checks(self):
        import numpy as np
        from types import SimpleNamespace
        from pri_portfolio_validation import verify_mask_input
        path=self.root/"strict_field.txt"
        raw=SimpleNamespace(bounds=(0.,0.,.2,.2),resolution=.1,risk=np.ones((2,2)),
            valid=np.ones((2,2),dtype=bool),path_cells=lambda path:[])
        numeric=SimpleNamespace(bounds=raw.bounds,resolution=raw.resolution,risk=np.ones((2,2)),valid=raw.valid)
        def policy(value):
            path.write_text('SAFE_SEM_FIELD_V1\n2 2 0 0 0.1 0.2 0.2\n'+''.join('1 '+repr(value)+'\n' for _ in range(4)),encoding='ascii')
            return dict(field_path=str(path),field_sha256=file_hash(path))
        _,audit=verify_mask_input(raw,policy(1.),numeric)
        self.assertTrue(audit["exact_numerical_replay"])
        with self.assertRaisesRegex(ValueError,"exact_arithmetic_replay_mismatch"):
            verify_mask_input(raw,policy(float(np.nextafter(1.,2.))),numeric)
        raw.risk[:]=np.nextafter(40.,np.inf);numeric.risk[:]=40.
        with self.assertRaisesRegex(ValueError,"hard_risk_classification_mismatch"):
            verify_mask_input(raw,policy(40.),numeric)


if __name__=="__main__":
    unittest.main()
