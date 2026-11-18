"""Synthetic R2 preservation fixtures. No planner or geometry oracle is run.

The large fixture checks the real 893/607/53 lock cardinalities using tiny,
explicitly synthetic records. Smaller fixtures exercise interruptions/tamper.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import ModuleType,SimpleNamespace
import unittest
from unittest.mock import patch

parser=argparse.ArgumentParser(add_help=False)
parser.add_argument('--runner-path',type=Path);parser.add_argument('--root',type=Path)
parser.add_argument('--patch-dir',type=Path)
options,remaining=parser.parse_known_args()
ROOT=(options.root or Path(__file__).resolve().parents[1]).resolve()
RUNNER=(options.runner_path or ROOT/'tools/native_paper_v4_recovery_r2.py').resolve()
sys.path.insert(0,str(ROOT/'tools'))
spec=importlib.util.spec_from_file_location('synthetic_r2_under_test',RUNNER)
r2=importlib.util.module_from_spec(spec);sys.modules[spec.name]=r2;spec.loader.exec_module(r2)
from native_paper_v4_storage import ArtifactStore


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes((json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode())


def owned(path,base):
    path,base=Path(path).absolute(),Path(base).resolve()
    if not path.resolve().is_relative_to(base):raise ValueError('synthetic owned path escape')
    for item in (path,*path.parents):
        if item==base:break
        if item.is_symlink():raise ValueError('synthetic owned path link')
    return path


def copy(source,target,expected):
    assert r2.sha(source)==expected
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():assert r2.sha(target)==expected
    else:shutil.copyfile(source,target)


FIXED_SCHEDULER='''"""Synthetic scheduler: file operations only, never launches a process."""
import json
from pathlib import Path
def _put(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 path.write_bytes((json.dumps(value,sort_keys=True,indent=2)+'\\n').encode())
def _append_journal(path,event):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('ab') as output:output.write((json.dumps(event,sort_keys=True)+'\\n').encode())
def _run_locked(plan,registry,output,**kwargs):
 output=Path(output);existing=output/'manifest.json'
 if existing.exists():return json.loads(existing.read_bytes())
 entries=json.loads(Path(plan).read_bytes())['entries']
 for entry in entries:
  if entry['execution_status']!='PLANNED':continue
  _append_journal(output/'start_journal.jsonl',dict(trial_id=entry['trial_id'],synthetic_fixture=True))
  entry['execution_status']='COMPLETED';entry['directory']=str(output/entry['trial_id'])
  _put(Path(entry['directory'])/'request.json',entry['request'])
  _put(Path(entry['directory'])/'execution.json',dict(status='COMPLETED',synthetic_fixture=True))
 return dict(status='COLLECTED_VALIDATION_PENDING',entries=entries,synthetic_fixture=True,sessions=[{}])
'''


class FakeAnalysis:
    def __init__(self,fixture):self.fixture=fixture
    def validate_map(self,workspace,info,entries,jobs):
        self.fixture.validation_calls.extend(entry['trial_id'] for entry in entries)
        for entry in entries:self.fixture.ledger(entry,Path(entry['directory']))
        return [r2.read(Path(workspace)/'verdicts/main'/(entry['trial_id']+'.json'))['row'] for entry in entries]
    def analyze_campaign(self,*args,**kwargs):return dict(completion='SYNTHETIC_COMPLETE')


class Fixture:
    def __init__(self,base,started=4,pending=3,missing=3,failed=2):
        self.base=base;self.root=base/'repo';self.w=base/'campaign';self.patch_dir=base/'patch'
        self.root.joinpath('tools').mkdir(parents=True);self.w.mkdir();self.patch_dir.mkdir()
        self.identity='SYNTHETIC_R1_IDENTITY';self.validation_calls=[];self.new_compactions=[];self.run_calls=0
        self.fail_after_child=False;self.raise_before_collect=False
        self.old=[];self.future=[];self.missing_count=missing;self.failed=failed
        for i in range(started):
            tid=f'old{i:04d}';directory=self.w/'collection/main-w000'/tid
            entry=dict(trial_id=tid,map_id='oldmap',backend='synthetic',request=dict(synthetic_fixture=True,seed=i),
                execution_status='FAILED_CORRECTNESS' if i>=started-failed else 'COMPLETED')
            dump(directory/'request.json',entry['request']);dump(directory/'execution.json',dict(status=entry['execution_status'],synthetic_fixture=True))
            dump(directory/'worker.json',dict(synthetic_fixture=True,original_raw=True))
            self.old.append(entry)
            if i<started-missing:self.ledger(entry,directory)
        for i in range(pending):
            self.future.append(dict(trial_id=f'pending{i:04d}',map_id='newmap',backend='synthetic',
                request=dict(synthetic_fixture=True,seed=10000+i),execution_status='PLANNED'))
        main=dict(status='FAILED_CORRECTNESS',synthetic_fixture=True,entries=self.old+self.future,
            waves=[dict(wave_id='main-w000',map_ids=['oldmap'],status='COLLECTING'),
                   dict(wave_id='main-w001',map_ids=['newmap'],status='PLANNED')])
        dump(self.w/'manifests/main.json',main);dump(self.w/'shards/index.json',dict(schema_version=1,entries={}))
        dump(self.w/'maps.json',dict(main=[dict(map_id='oldmap'),dict(map_id='newmap')]))
        for mode in ('smoke-serial','smoke-parallel'):dump(self.w/'manifests'/(mode+'.json'),dict(synthetic_fixture=True))
        journal=self.w/'collection/main-w000/start_journal.jsonl';journal.parent.mkdir(parents=True,exist_ok=True)
        journal.write_bytes(b''.join((json.dumps(dict(trial_id=e['trial_id'],synthetic_fixture=True))+'\n').encode() for e in self.old))
        dump(self.w/'plans/pending.json',dict(entries=self.future,synthetic_fixture=True))
        self.base_module=ModuleType('terrain_compare_scheduler');self.worker=ModuleType('native_paper_v4_worker')
        original=self.root/'tools/terrain_compare_scheduler.py';original.write_text('SYNTHETIC_FROZEN_BASE_SCHEDULER')
        self.base_hash=r2.sha(original)
        self.base_module.__file__=str(original)
        self.base_module._freeze_inputs=lambda *a,**k:dict(scheduler_sha256=r2.sha(original),synthetic_fixture=True)
        self.base_module._run_locked=lambda *a,**k:(_ for _ in ()).throw(AssertionError('base loop must be overlaid'))
        self.base_module.run_campaign=lambda *a,**k:self.base_module._run_locked(*a,**k)
        self.base_module.atomic_json=dump
        def compact(directory,request,shared):
            self.new_compactions.append(Path(directory).name)
            dump(Path(directory)/'compaction.json',dict(synthetic_fixture=True));return dict(status='SYNTHETIC_COMPACTED')
        self.worker.compact_trial=compact
        self.campaign=SimpleNamespace(run=self.campaign_run,run_campaign=self.base_module.run_campaign,
            require_smoke=lambda *a,**k:dict(synthetic_fixture=True))
        self.r1=SimpleNamespace(campaign=self.campaign,validation_identity=self.identity,dump=dump,owned=owned,copy=copy,
            guard=self.guard,prepare=lambda:dict(status='PASS',synthetic_fixture=True),
            analysis_module=self.analysis_module,baseline_smokes=lambda:dict(synthetic={}),
            verify_smoke_gate=lambda *a,**k:None,_verify_files=lambda files:None)
        records={}
        for path in sorted(self.w.rglob('*')):
            if path.is_file():
                name='campaign/'+path.relative_to(self.w).as_posix()
                records[name]=dict(path=name,size_bytes=path.stat().st_size,sha256=r2.sha(path),mode=0o644)
        checkpoint=dict(inventory=records,inventory_sha256=r2.digest(records),payloads=[],synthetic_fixture=True)
        dump(self.w/'exports/checkpoint_index.json',checkpoint)
        (self.patch_dir/'native_paper_v4_recovery_r2.py').write_bytes(RUNNER.read_bytes())
        (self.patch_dir/'terrain_compare_scheduler_r2.py').write_text(FIXED_SCHEDULER)
        for name in ('native_paper_v4_recovery.py','native_paper_v4_analysis.py','recovery.json','test_affinity_launch_race.py','test_native_paper_v4_recovery_r2.py'):
            (self.patch_dir/name).write_text('SYNTHETIC_BOUND_UNUSED')
        manifest=dict(policy_id=r2.POLICY_ID,original_scheduler_sha256=self.base_hash,
            source_checkpoint_index_sha256=r2.sha(self.w/'exports/checkpoint_index.json'),
            source_checkpoint_inventory_sha256=checkpoint['inventory_sha256'],base_payloads=[])
        for key,name in [('runner','native_paper_v4_recovery_r2.py'),('scheduler','terrain_compare_scheduler_r2.py'),
                ('r1_runner','native_paper_v4_recovery.py'),('analyzer','native_paper_v4_analysis.py'),('r1_manifest','recovery.json'),
                ('test_scheduler','test_affinity_launch_race.py'),('test_runner','test_native_paper_v4_recovery_r2.py')]:
            manifest[key]=dict(path=name,sha256=r2.sha(self.patch_dir/name))
        for stem,ids in [('started',[e['trial_id'] for e in self.old]),('pending',[e['trial_id'] for e in self.future]),
                         ('missing',[e['trial_id'] for e in self.old[started-missing:]])]:
            manifest['initial_'+stem+'_count']=len(ids);manifest['initial_'+stem+'_ids_sha256']=r2.digest(sorted(ids))
        self.manifest=self.patch_dir/'recovery_r2.json';dump(self.manifest,manifest)

    def guard(self):
        if r2.sha(self.root/'tools/terrain_compare_scheduler.py')!=self.base_hash:raise ValueError('synthetic frozen source changed')

    @contextmanager
    def analysis_module(self,*args,**kwargs):yield FakeAnalysis(self)

    def ledger(self,entry,directory):
        status=entry['execution_status'];row=dict(trial_id=entry['trial_id'],backend='synthetic',synthetic_fixture=True,
            execution_status=status,validation_code_identity=self.identity,errors=[],correctness_errors=[],events=[],
            success=False,attained_safe=False,attained_geometry=False)
        verdict=(json.dumps(row,sort_keys=True,indent=2)+'\n').encode();expected=hashlib.sha256(verdict).hexdigest()
        receipt=dict(verdict_sha256=expected,validation_code_identity=self.identity,synthetic_fixture=True)
        dump(directory/'validation.json',row);dump(directory/'validation_receipt.json',receipt)
        dump(self.w/'verdicts/main'/(entry['trial_id']+'.json'),dict(row=row,receipt=receipt,verdict_sha256=expected))

    def runner(self):return r2.RecoveryV2(self.root,self.w,self.patch_dir,self.manifest,r2.sha(self.manifest),_r1=self.r1)

    def campaign_run(self,workspace,mode,**kwargs):
        self.run_calls+=1
        assert len(list((self.w/'verdicts/main').glob('*.json')))==len(self.old),'all original verdicts must precede any launch'
        main=r2.read(self.w/'manifests/main.json')
        if self.raise_before_collect:raise RuntimeError('SYNTHETIC_CONTROLLER_EXCEPTION')
        for smoke in ('smoke-serial','smoke-parallel'):self.campaign.require_smoke(workspace,smoke)
        if main['waves'][0]['status']!='PACKED':
            for entry in main['entries'][:len(self.old)]:
                old=self.w/'collection/main-w000'/entry['trial_id'];new=self.w/'active/main-w000/oldmap'/entry['trial_id']
                new.parent.mkdir(parents=True,exist_ok=True);old.rename(new)
                entry.update(directory=str(new),wave_id='main-w000')
                self.worker.compact_trial(new,entry['request'],self.w/'shared')
            dump(self.w/'manifests/main.json',main);ArtifactStore(self.w).pack_map('main-w000','oldmap')
            main['waves'][0]['status']='PACKED';dump(self.w/'manifests/main.json',main)
        output=self.w/'collection/main-w001'
        child=self.campaign.run_campaign(self.w/'plans/pending.json',{},output,resume=(output/'manifest.json').exists())
        if self.fail_after_child:
            self.fail_after_child=False;raise RuntimeError('SYNTHETIC_PARENT_BEHIND_CHILD')
        updated={entry['trial_id']:entry for entry in child['entries']}
        for entry in main['entries'][len(self.old):]:
            entry.update(updated[entry['trial_id']]);self.worker.compact_trial(Path(entry['directory']),entry['request'],self.w/'shared')
        main['status']='COMPLETE';dump(self.w/'manifests/main.json',main);return main


class RecoveryV2Tests(unittest.TestCase):
    def fixture(self,**kwargs):
        temporary=tempfile.TemporaryDirectory(prefix='synthetic-r2-');self.addCleanup(temporary.cleanup)
        fixture=Fixture(Path(temporary.name),**kwargs)
        modules=patch.dict(sys.modules,{'terrain_compare_scheduler':fixture.base_module,'native_paper_v4_worker':fixture.worker})
        modules.start();self.addCleanup(modules.stop);return fixture

    @contextmanager
    def linux_fork(self):
        with patch.object(r2.sys,'platform','linux'),patch.object(r2.multiprocessing,'get_context',return_value=SimpleNamespace(get_start_method=lambda:'fork')):yield

    def test_full_893_preserved_607_eligible_and_53_prevalidated_before_launch(self):
        fixture=self.fixture(started=893,pending=607,missing=53,failed=4);runner=fixture.runner()
        origin,unused,gate=runner.prepare()
        self.assertEqual(len(origin['initial_started_ids']),893);self.assertEqual(len(origin['initial_pending_ids']),607)
        self.assertEqual(gate['newly_validated'],53);self.assertEqual(gate['validation_failures_retained'],4)
        self.assertEqual(len(set(fixture.validation_calls)),53);self.assertEqual(fixture.run_calls,0)
        self.assertEqual(len(list((fixture.w/'verdicts/main').glob('*.json'))),893)
        runner.prepare();self.assertEqual(fixture.run_calls,0)

    def test_original_statuses_and_raw_survive_move_pack_and_only_pending_ids_launch(self):
        fixture=self.fixture();runner=fixture.runner();origin,gate,unused=runner.prepare()
        with self.linux_fork():result=runner.run(allow_main=True)
        self.assertEqual(result['preserved_attempts'],4);self.assertEqual(result['permitted_new_attempts'],3)
        runner.protect_origin(origin)
        self.assertEqual(runner.raw_inventory(runner.main()['entries'][:4]),origin['raw_inventory'])
        self.assertEqual(set(fixture.new_compactions),{entry['trial_id'] for entry in fixture.future})
        self.assertEqual(len(runner.journal_events()),7)
        for tid in [entry['trial_id'] for entry in fixture.old[-2:]]:self.assertEqual(runner.verify_ledger(tid)['execution_status'],'FAILED_CORRECTNESS')

    def test_overlay_preserves_frozen_input_identity_restores_functions_and_blocks_old_or_unknown(self):
        fixture=self.fixture();runner=fixture.runner();origin,gate,unused=runner.prepare()
        before=fixture.base_module._freeze_inputs();loop=fixture.base_module._run_locked
        collect=fixture.campaign.run_campaign;compact=fixture.worker.compact_trial;append=runner.fixed._append_journal
        with self.assertRaisesRegex(RuntimeError,'SYNTHETIC_EXCEPTION'):
            with runner.scheduler_overlay(origin,gate):
                self.assertIs(fixture.base_module._run_locked,runner.fixed._run_locked)
                self.assertEqual(fixture.base_module._freeze_inputs(),before)
                for tid in (fixture.old[0]['trial_id'],'unknown_slot'):
                    with self.assertRaisesRegex(ValueError,'preserved or unknown'):runner.fixed._append_journal(fixture.w/'ignored.jsonl',dict(trial_id=tid))
                raise RuntimeError('SYNTHETIC_EXCEPTION')
        self.assertIs(fixture.base_module._run_locked,loop);self.assertIs(fixture.campaign.run_campaign,collect)
        self.assertIs(fixture.worker.compact_trial,compact);self.assertIs(runner.fixed._append_journal,append)

    def test_child_ahead_of_parent_resume_never_relaunches_an_attempt(self):
        fixture=self.fixture();runner=fixture.runner();fixture.fail_after_child=True
        with self.linux_fork():
            with self.assertRaisesRegex(RuntimeError,'PARENT_BEHIND_CHILD'):runner.run(allow_main=True)
            self.assertEqual(len(runner.journal_events()),7)
            result=fixture.runner().run(allow_main=True)
        self.assertEqual(result['status'],'COMPLETE');self.assertEqual(len(runner.journal_events()),7)

    def test_journal_prefix_duplicate_unknown_and_missing_overlay_provenance_rejected(self):
        for kind in ('prefix','duplicate','unknown','provenance'):
            with self.subTest(kind=kind):
                fixture=self.fixture();runner=fixture.runner();origin,unused,gate=runner.prepare()
                journal=fixture.w/'collection/main-w000/start_journal.jsonl';data=journal.read_bytes()
                if kind=='prefix':journal.write_bytes(b' '+data[1:])
                else:
                    tid=fixture.old[0]['trial_id'] if kind=='duplicate' else 'UNKNOWN' if kind=='unknown' else fixture.future[0]['trial_id']
                    with journal.open('ab') as output:output.write((json.dumps(dict(trial_id=tid))+'\n').encode())
                with self.assertRaises(ValueError):runner.protect_origin(origin)

    def test_authenticated_origin_raw_entries_and_bound_files_reject_tamper(self):
        fixture=self.fixture();runner=fixture.runner();origin,unused,gate=runner.prepare()
        origin['raw_inventory'][fixture.old[0]['trial_id']]['request.json']='0'*64
        with self.assertRaisesRegex(ValueError,'raw seal'):runner.protect_origin(origin)
        origin=r2.read(runner.policy_dir/'origin.json');origin['original_entries'][fixture.old[0]['trial_id']]['execution_status']='PLANNED'
        with self.assertRaisesRegex(ValueError,'states differ'):runner.protect_origin(origin)
        (fixture.patch_dir/'terrain_compare_scheduler_r2.py').write_text('SYNTHETIC_TAMPER')
        with self.assertRaisesRegex(ValueError,'implementation changed'):runner.prepare()

    def test_controller_exception_restores_overlay_and_failed_attempts_are_not_relabelled(self):
        fixture=self.fixture();runner=fixture.runner();fixture.raise_before_collect=True
        loop=fixture.base_module._run_locked;compact=fixture.worker.compact_trial
        with self.linux_fork():
            with self.assertRaisesRegex(RuntimeError,'CONTROLLER_EXCEPTION'):runner.run(allow_main=True)
        self.assertIs(fixture.base_module._run_locked,loop);self.assertIs(fixture.worker.compact_trial,compact)
        self.assertEqual([entry['execution_status'] for entry in runner.main()['entries'][:4]],['COMPLETED','COMPLETED','FAILED_CORRECTNESS','FAILED_CORRECTNESS'])

    def test_origin_cannot_drop_original_journal_or_ledger_seal_coverage(self):
        fixture=self.fixture();runner=fixture.runner();origin,unused,gate=runner.prepare()
        origin['journal_prefixes'].clear()
        with self.assertRaisesRegex(ValueError,'coverage'):runner.protect_origin(origin)
        origin=r2.read(runner.policy_dir/'origin.json');origin['original_ledger_hashes'].clear()
        with self.assertRaisesRegex(ValueError,'coverage'):runner.protect_origin(origin)


if __name__=='__main__':unittest.main(argv=[sys.argv[0],*remaining])
