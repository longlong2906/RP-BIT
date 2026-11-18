"""Synthetic external-recovery fixtures; no planner or geometry oracle runs."""
from pathlib import Path
import hashlib
import importlib.util
import json
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"tools"))
import native_paper_v4_recovery as recovery
from native_paper_v4_storage import ArtifactStore

ANALYZER = '''from pathlib import Path
import hashlib,json
ROOT=Path('.')
def validation_code_identity(): return 'original'
def put(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 path.write_bytes((json.dumps(value,indent=2,sort_keys=True)+'\\n').encode())
def validate_map(workspace,info,entries,jobs=1):
 rows=[]
 for entry in entries:
  row=dict(trial_id=entry['trial_id'],backend=entry['backend'],errors=[],correctness_errors=[],success=True,
           planner_status='EXACT_SOLUTION',validation_code_identity=validation_code_identity())
  path=Path(entry['directory'])/'validation.json';put(path,row)
  h=hashlib.sha256(path.read_bytes()).hexdigest()
  receipt=dict(verdict_sha256=h,validation_code_identity=validation_code_identity())
  put(Path(workspace)/'verdicts'/info['mode']/(entry['trial_id']+'.json'),dict(row=row,receipt=receipt,verdict_sha256=h))
  rows.append(row)
 return rows
def analyze_campaign(*args,**kwargs): return dict(completion='COMPLETE')
'''


class FrozenCampaign:
    def __init__(self,root,workspace,source_files):
        self.root,self.workspace,self.sources=root,workspace,source_files
        self.calls=[]
    def check_frozen(self,workspace):
        for name,expected in self.sources.items():
            if recovery.sha(self.root/name)!=expected:raise ValueError('frozen source changed')
        return recovery.read(self.workspace/'prepared.json')
    def require_smoke(self,workspace,mode):
        if sys.modules['native_paper_v4_analysis'].validation_code_identity()!='original':
            raise ValueError('baseline must be verified by the original analyzer')
        proof=recovery.read(self.workspace/'readiness'/(mode+'.json'))
        for tid,expected in proof['ledgers'].items():
            if recovery.sha(self.workspace/'verdicts'/mode/(tid+'.json'))!=expected:raise ValueError('baseline ledger changed')
        for name,expected in proof['shards'].items():
            if recovery.sha(self.workspace/name)!=expected:raise ValueError('baseline shard changed')
        return recovery.read(self.workspace/'manifests'/(mode+'.json'))
    def run(self,workspace,mode,**kwargs):
        self.calls.append(kwargs)
        for smoke in recovery.SMOKES:self.require_smoke(workspace,smoke)
        main=recovery.read(self.workspace/'manifests/main.json')
        started=[e['trial_id'] for e in main['entries'] if e['execution_status']!='PLANNED']
        if sorted(started)!=['old0','old1']:raise AssertionError('unexpected repeated attempted slot')
        for entry in main['entries'][:2]:
            recovery.dump(Path(entry['directory'])/'validation.json',dict(corrected_derived=True))
        ArtifactStore(self.workspace).pack_map('main-w000','mainmap')
        main['waves'][0]['status']='PACKED'
        future=main['entries'][2]
        future['execution_status']='COMPLETED'
        journal=self.workspace/'collection/main-w001/start_journal.jsonl';journal.parent.mkdir(parents=True)
        journal.write_text(json.dumps(dict(trial_id=future['trial_id']))+'\n')
        main['status']='COMPLETE';recovery.dump(self.workspace/'manifests/main.json',main)
        return main


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='r1_synthetic_',dir=ROOT/'tests/fixtures')
        self.addCleanup(self.tmp.cleanup)
        self.base=Path(self.tmp.name);self.root=self.base/'repo';self.workspace=self.base/'campaign'
        (self.root/'tools').mkdir(parents=True);self.workspace.mkdir()
        (self.root/'tools/native_paper_v4_analysis.py').write_text(ANALYZER)
        for name in recovery.HELPERS:(self.root/'tools'/name).write_text('SYNTHETIC_FROZEN_HELPER')
        self.sources={p.relative_to(self.root).as_posix():recovery.sha(p) for p in (self.root/'tools').glob('*')}
        self.original_sources=dict(self.sources)
        recovery.dump(self.workspace/'prepared.json',dict(identity=dict(source_sha256='synthetic_source')))
        (self.base/'source_input.zip').write_bytes(b'SYNTHETIC_FROZEN_SOURCE_PACKAGE')
        store=ArtifactStore(self.workspace);maps=[]
        for mode in recovery.SMOKES:
            entries=[]
            for i in range(7):
                mid='fixture_'+str(i);wave=mode+'-w000'
                if mode==recovery.SMOKES[0]:maps.append(dict(map_id=mid))
                for backend in ('a3t_old','a3t_safe','sem_native'):
                    tid=mid+'__'+backend;directory=self.workspace/'active'/wave/mid/tid
                    recovery.dump(directory/'request.json',dict(backend=backend))
                    recovery.dump(directory/'worker.json',dict(synthetic=True))
                    entries.append(dict(trial_id=tid,backend=backend,map_id=mid,wave_id=wave,request=dict(backend=backend)))
                    recovery.dump(self.workspace/'verdicts'/mode/(tid+'.json'),dict(synthetic_original=True))
                store.pack_map(wave,mid)
            recovery.dump(self.workspace/'manifests'/(mode+'.json'),dict(status='COMPLETE',entries=entries))
            index=recovery.read(self.workspace/'shards/index.json')['entries']
            proof=dict(ledgers={e['trial_id']:recovery.sha(self.workspace/'verdicts'/mode/(e['trial_id']+'.json')) for e in entries},
                       shards={v['archive']:v['sha256'] for k,v in index.items() if k.startswith(mode+'-')})
            recovery.dump(self.workspace/'readiness'/(mode+'.json'),proof)
        recovery.dump(self.workspace/'maps.json',dict(smoke=maps))
        entries=[]
        for i in range(2):
            tid='old'+str(i);directory=self.workspace/'active/main-w000/mainmap'/tid
            recovery.dump(directory/'request.json',dict(planner_seed=11+i))
            recovery.dump(directory/'worker.json',dict(original_raw=True))
            recovery.dump(directory/'validation.json',dict(original_derived=True))
            recovery.dump(directory/'validation_receipt.json',dict(original_derived=True))
            recovery.dump(self.workspace/'verdicts/main'/(tid+'.json'),dict(original_derived=True))
            entries.append(dict(trial_id=tid,map_id='mainmap',wave_id='main-w000',directory=str(directory),execution_status='COMPLETED'))
        entries.append(dict(trial_id='future0',map_id='futuremap',execution_status='PLANNED'))
        recovery.dump(self.workspace/'manifests/main.json',dict(status='FAILED_CORRECTNESS',entries=entries,
            waves=[dict(wave_id='main-w000',map_ids=['mainmap'],status='DRAINED'),dict(wave_id='main-w001',map_ids=['futuremap'],status='PLANNED')]))
        journal=self.workspace/'collection/main-w000/start_journal.jsonl';journal.parent.mkdir(parents=True)
        journal.write_text(''.join(json.dumps(dict(trial_id=e['trial_id']))+'\n' for e in entries[:2]))
        self.campaign=FrozenCampaign(self.root,self.workspace,self.sources)
        self.patch=self.base/'corrected_analysis.py';self.patch.write_text(ANALYZER.replace("return 'original'","return 'corrected'"))
        self.manifest=self.base/'recovery.json'
        recovery.dump(self.manifest,dict(policy_id=recovery.POLICY_ID,
            analyzer=dict(path=self.patch.name,sha256=recovery.sha(self.patch)),
            runner=dict(path=Path(recovery.__file__).name,sha256=recovery.sha(recovery.__file__)),
            prepared_source_sha256='synthetic_source',source_input_sha256=recovery.sha(self.base/'source_input.zip'),
            initial_started_count=2,initial_started_ids_sha256=recovery.digest(['old0','old1'])))

    def runner(self):
        return recovery.RecoveryRunner(self.root,self.workspace,self.patch,self.manifest,recovery.sha(self.manifest),
                                       analyzer_sha256=recovery.sha(self.patch),_campaign=self.campaign)

    def test_prepare_is_idempotent_and_does_not_rewrite_original_smokes_or_raw(self):
        runner=self.runner();before=runner.main_raw_inventory()
        old_shards=recovery.read(self.workspace/'shards/index.json')
        first=runner.prepare();second=runner.prepare()
        self.assertEqual(first,second)
        self.assertEqual(before,runner.main_raw_inventory())
        self.assertEqual(old_shards,recovery.read(self.workspace/'shards/index.json'))
        self.assertEqual({p:recovery.sha(self.root/p) for p in self.sources},self.original_sources)
        self.assertEqual(self.campaign.calls,[])
        self.assertEqual(len(first['ledger_files']),42)
        self.assertTrue((runner.policy_dir/'original_derived/manifests/main.json').exists())

    def test_recovery_never_relaunches_original_attempts_and_keeps_bytes_after_pack(self):
        runner=self.runner();before=runner.main_raw_inventory()
        with patch.object(recovery.sys,'platform','linux'),patch('multiprocessing.get_context',return_value=SimpleNamespace(get_start_method=lambda:'fork')):
            result=runner.run(allow_main=True)
        self.assertEqual(result['status'],'COMPLETE')
        self.assertEqual(result['original_attempted_slots_retained'],2)
        self.assertEqual(result['report_status'],'COMPLETE')
        self.assertEqual(self.campaign.calls,[dict(allow_main=True,resume=True,session_elapsed_s=0.)])
        self.assertEqual(before,runner.main_raw_inventory({'old0','old1'}))
        self.assertEqual(len((self.workspace/'collection/main-w000/start_journal.jsonl').read_text().splitlines()),2)

    def test_manifest_patch_and_frozen_source_tamper_fail_closed(self):
        bad=self.manifest.read_bytes();self.manifest.write_bytes(bad+b' ')
        with self.assertRaisesRegex(ValueError,'manifest SHA'):
            recovery.RecoveryRunner(self.root,self.workspace,self.patch,self.manifest,hashlib.sha256(bad).hexdigest(),_campaign=self.campaign)
        self.manifest.write_bytes(bad)
        runner=self.runner();runner.prepare()
        self.patch.write_text('changed')
        with self.assertRaisesRegex(ValueError,'implementation changed'):runner.prepare()

    def test_missing_smoke_gate_coverage_is_not_accepted(self):
        runner=self.runner();runner.prepare()
        path=runner.policy_dir/'smoke_gate.json';gate=recovery.read(path);gate['ledger_files']={};recovery.dump(path,gate)
        with self.assertRaisesRegex(ValueError,'missing or extra'):runner.prepare()

    def test_spawn_context_refused_before_campaign_run(self):
        runner=self.runner()
        with patch.object(recovery.sys,'platform','linux'),patch('multiprocessing.get_context',return_value=SimpleNamespace(get_start_method=lambda:'spawn')):
            with self.assertRaisesRegex(ValueError,'fork context'):runner.run(allow_main=True)
        self.assertEqual(self.campaign.calls,[])

    def test_preexisting_raw_mutation_rejected_even_with_cached_smoke_gate(self):
        runner=self.runner();runner.prepare()
        target=self.workspace/'active/main-w000/mainmap/old0/worker.json';target.write_text('changed raw')
        with self.assertRaisesRegex(ValueError,'raw bytes changed'):runner.prepare()

    def test_backup_interruption_recovers_without_overwriting_unrelated_backup(self):
        runner=self.runner();original=recovery._copy;calls=0
        def interrupted(*args):
            nonlocal calls
            calls+=1
            if calls==2:raise RuntimeError('synthetic backup interruption')
            return original(*args)
        with patch.object(recovery,'_copy',side_effect=interrupted):
            with self.assertRaises(RuntimeError):runner.prepare()
        runner.prepare()
        backup=runner.policy_dir/'original_derived/manifests/main.json';backup.write_text('unrelated backup')
        with self.assertRaisesRegex(ValueError,'backup was changed'):runner.prepare()


class TemporaryWriteTests(unittest.TestCase):
    def test_pending_directory_and_link_are_rejected_before_write(self):
        with tempfile.TemporaryDirectory(prefix='recovery_temp_') as directory:
            root=Path(directory);source=root/'source';source.write_bytes(b'unchanged source')
            target=root/'target';pending=root/'target.pending';pending.mkdir()
            with self.assertRaisesRegex(ValueError,'not a regular'):recovery.dump(target,dict(test=True))
            copy_target=root/'copy';copy_pending=root/'copy.copy-pending';copy_pending.mkdir()
            with self.assertRaisesRegex(ValueError,'not a regular'):recovery._copy(source,copy_target,recovery.sha(source))
            original=Path.is_symlink
            def temporary_link(path):return path==root/'linked.pending' or original(path)
            with patch.object(Path,'is_symlink',temporary_link):
                with self.assertRaisesRegex(ValueError,'contains a link'):recovery.dump(root/'linked',dict(test=True))
            self.assertEqual(source.read_bytes(),b'unchanged source')
            self.assertFalse(target.exists());self.assertFalse(copy_target.exists())


if __name__=='__main__':unittest.main()
