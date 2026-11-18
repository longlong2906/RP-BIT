"""Labeled synthetic orchestration/transport fixtures; no planner trials run.

Fixtures exist only in temporary test directories, never in results/. They test
failure retention, clocks, frozen identity and unsafe archive rejection.
"""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch,Mock
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import cloud_campaign as c
from experiment_common import ROOT,digest,dump,file_hash,load_config


def fixture_maps(config):
    maps=[];s=config['profiles']['combined']
    for family in s['families']:
        for seed in s['map_seeds']:
            query={'query_id':'q0','query_hash':'fixture-query','start':[0.,0.],'goal':[1.,1.],
                   'reference':{'reference_type':'synthetic_test_fixture','cost_terrain_m':2.}}
            maps.append({'map_id':f'{family}_{seed}','family':family,'map_seed':seed,
                         'path':f'{family}_{seed}.t25d','sha256':'fixture-map','queries':[query]})
    return {'synthetic_test_fixture':True,'maps':maps}


class CampaignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=load_config(ROOT/'configs/a3_vs_all_v2.yaml','combined')
        cls.maps=fixture_maps(cls.config);cls.trials=c.plan(cls.config,cls.maps)
        cls.identity={'config_hash':digest(cls.config),'code_sha256':'synthetic-source','worker_sha256':'synthetic-worker','ompl_library_sha256':'synthetic-library'}
        cls.manifest={'experiment_id':'synthetic_test_fixture','identity':cls.identity,'trials':cls.trials}

    def test_full_schedule_is_paired_balanced_and_all_four_tasks_equal(self):
        audit=c.audit_plan(self.trials,self.config)
        self.assertEqual(audit['planned_trials'],1500)
        self.assertEqual(audit['task_trial_counts'],{str(i):375 for i in range(4)})
        self.assertTrue(all(v==[60]*5 for v in audit['position_counts'].values()))
        for shard in range(20):
            entries=[t for t in self.trials if t['shard_id']==shard]
            self.assertEqual(len(entries),75)
            self.assertEqual(len({e['map_id'] for e in entries}),5)
            self.assertEqual({e['planner_seed'] for e in entries},{94001,94002,94003})
        self.assertEqual(c.plan(self.config,self.maps),self.trials)

    def test_schedule_rejects_duplicate_and_wrong_shard(self):
        wrong=copy.deepcopy(self.trials);wrong[1]['trial_id']=wrong[0]['trial_id']
        with self.assertRaises(c.IntegrityError): c.audit_plan(wrong,self.config)
        wrong=copy.deepcopy(self.trials);wrong[0]['shard_id']=19
        with self.assertRaises(c.IntegrityError): c.audit_plan(wrong,self.config)

    def test_task_selection_covers_disjoint_shards(self):
        groups=[c.selected_shards(self.config,task_index=i) for i in range(4)]
        self.assertEqual(sorted(sum(groups,[])),list(range(20)))
        self.assertEqual(groups[0],[0,4,8,12,16])
        for args in ({},{'task_index':4},{'shard_id':20},{'task_index':0,'shard_id':0}):
            with self.assertRaises(c.IntegrityError): c.selected_shards(self.config,**args)

    def test_portable_paths_reject_escape_windows_drive_and_alias(self):
        for path in ('../escape','/absolute','C:/secret','a\\b','a//b','a/./b',''):
            with self.subTest(path=path),self.assertRaises(c.IntegrityError): c.relative(path)
        self.assertEqual(c.relative('map.t25d').as_posix(),'map.t25d')

    def test_duplicate_json_and_nonfinite_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix='synthetic-cloud-test-',dir=ROOT/'tests/fixtures') as temp:
            path=Path(temp)/'x.json'
            for content in ('{"a":1,"a":2}','{"a":NaN}'):
                path.write_text(content)
                with self.assertRaises(c.IntegrityError): c.read_json(path)

    def test_inventory_retains_temporary_raw_evidence_and_rejects_drift(self):
        with tempfile.TemporaryDirectory(prefix='synthetic-cloud-test-',dir=ROOT/'tests/fixtures') as temp:
            root=Path(temp);(root/'phase.json.tmp').write_text('partial raw worker evidence')
            raw=c.freeze_raw(root)
            self.assertEqual(raw['entries'][0]['path'],'phase.json.tmp')
            c.verify_raw(root);(root/'extra').write_text('unexpected')
            with self.assertRaises(c.IntegrityError): c.verify_raw(root)

    def test_cloud_gate_requires_real_tests_all_planners_and_library_binding(self):
        with tempfile.TemporaryDirectory(prefix='synthetic-cloud-test-',dir=ROOT/'tests/fixtures') as temp:
            worker=Path(temp)/'worker';worker.write_text('synthetic fixture')
            gate=dict(self.identity,status='PASS',ctest_passed=True,smoke_passed=True,smoke_planner_ids=list(c.PLANNERS))
            dump(worker.parent/'cloud_setup.json',gate);c.correctness_gate(worker,self.identity)
            for key,value in [('smoke_passed',False),('ctest_passed',False),('ompl_library_sha256','changed'),('smoke_planner_ids',['bitstar_a3'])]:
                bad=dict(gate,**{key:value});dump(worker.parent/'cloud_setup.json',bad)
                with self.assertRaises(c.IntegrityError): c.correctness_gate(worker,self.identity)

    def test_started_attempt_is_crash_retained_and_never_retried(self):
        with tempfile.TemporaryDirectory(prefix='synthetic-cloud-test-',dir=ROOT/'tests/fixtures') as temp:
            root=Path(temp);entry=self.trials[0];d=root/entry['trial_id'];d.mkdir();(d/'attempt.json').write_text('{}')
            self.assertTrue(c.retain_started(d,entry,self.config,self.manifest))
            self.assertEqual(c.read_json(d/'trial.json')['execution_status'],'CRASH')
            original=file_hash(d/'trial.json')
            self.assertFalse(c.retain_started(d,entry,self.config,self.manifest))
            self.assertEqual(file_hash(d/'trial.json'),original)

    def test_cross_root_resume_uses_portable_global_identity(self):
        with tempfile.TemporaryDirectory(prefix='synthetic-cloud-test-',dir=ROOT/'tests/fixtures') as temp:
            root=Path(temp);source=root/'global';source.mkdir();dump(source/'manifest.json',self.manifest)
            d=c.load_shard(root/'campaign',source,self.manifest,0,False,{'cpu':'host-one'})
            c.load_shard(root/'campaign',source,self.manifest,0,True,{'cpu':'host-two'})
            changed=copy.deepcopy(self.manifest);changed['identity']['worker_sha256']='wrong'
            with self.assertRaises(c.IntegrityError): c.load_shard(root/'campaign',source,changed,0,True,{})
            with self.assertRaises(c.IntegrityError): c.load_shard(root/'campaign',source,self.manifest,0,False,{})

    def launcher_watchdog(self,phase,ticks):
        with tempfile.TemporaryDirectory(prefix='synthetic-cloud-test-',dir=ROOT/'tests/fixtures') as temp:
            root=Path(temp);proc=Mock();proc.pid=123;proc.poll.return_value=None;proc.wait.return_value=-9
            with patch.object(c.subprocess,'Popen',return_value=proc),patch.object(c.psutil,'Process',side_effect=c.psutil.NoSuchProcess(123)),patch.object(c,'phase_read',return_value=phase),patch.object(c.time,'perf_counter',side_effect=ticks),patch.object(c.time,'sleep'):
                trial=c.launch_trial(self.trials[0],self.config,root/'trial',root/'worker',{},self.manifest,root)
            proc.kill.assert_called();return trial

    def test_setup_watchdog_is_distinct_from_thirty_second_solve(self):
        trial=self.launcher_watchdog(None,[0.,0.,61.,62.])
        self.assertEqual(trial['execution_status'],'WATCHDOG');self.assertEqual(trial['termination_reason'],'SETUP_WATCHDOG')
        self.assertEqual(trial['time_budget_s'],30.)

    def test_solve_watchdog_uses_solve_phase_plus_five_seconds(self):
        trial=self.launcher_watchdog({'phase':'SOLVING'},[0.,0.,36.,37.])
        self.assertEqual(trial['termination_reason'],'SOLVING_WATCHDOG')

    def test_collection_resume_does_not_launch_any_existing_attempt(self):
        with tempfile.TemporaryDirectory(prefix='synthetic-cloud-test-',dir=ROOT/'tests/fixtures') as temp:
            root=Path(temp);source=root/'global';source.mkdir();dump(source/'manifest.json',self.manifest)
            d=c.load_shard(root/'runs',source,self.manifest,0,False,{})
            entries=[t for t in self.trials if t['shard_id']==0]
            for entry in entries:
                trial=d/entry['trial_id'];trial.mkdir();dump(trial/'attempt.json',{'synthetic_test_fixture':True})
            with patch.object(c,'load_global',return_value=(self.config,self.maps,self.manifest)),patch.object(c.common,'cloud_environment',return_value={}),patch.object(c,'launch_trial') as launch:
                result=c.collect(source,root/'runs',root,root/'worker',shard_id=0,resume=True)
            launch.assert_not_called();self.assertEqual(result[0]['crash'],75)

    def test_chunk_transport_hash_order_and_existing_output_guard(self):
        with tempfile.TemporaryDirectory(prefix='synthetic-cloud-test-',dir=ROOT/'tests/fixtures') as temp:
            root=Path(temp);archive=root/'fixture.zip';archive.write_bytes(b'synthetic test transport'*300)
            c.build_chunks(archive,root/'parts');out=c.reassemble(root/'parts/parts.json',root/'reassembled/fixture.zip')
            self.assertEqual(file_hash(out),file_hash(archive));self.assertEqual(c.reassemble(root/'parts/parts.json',out),out)
            out.write_bytes(b'bad')
            with self.assertRaises(c.IntegrityError): c.reassemble(root/'parts/parts.json',out)
            part=root/'parts/part_0000.bin';part.write_bytes(b'changed')
            with self.assertRaises(c.IntegrityError): c.reassemble(root/'parts/parts.json',root/'new.zip')

    def test_zip_slip_and_symlink_rejected_before_extraction(self):
        with tempfile.TemporaryDirectory(prefix='synthetic-cloud-test-',dir=ROOT/'tests/fixtures') as temp:
            root=Path(temp)
            for index,name in enumerate(('../escape','C:/escape')):
                archive=root/f'bad{index}.zip'
                with zipfile.ZipFile(archive,'w') as z:z.writestr(name,b'bad')
                archive.with_name(archive.name+'.sha256').write_text(file_hash(archive)+'  '+archive.name+'\n')
                target=root/f'out{index}'
                with self.assertRaises(c.IntegrityError):c.extract_bundle(archive,target)
                self.assertFalse(target.exists())
            archive=root/'symlink.zip'
            with zipfile.ZipFile(archive,'w') as z:
                item=zipfile.ZipInfo('link');item.external_attr=(0o120777<<16);z.writestr(item,b'target')
            archive.with_name(archive.name+'.sha256').write_text(file_hash(archive)+'  '+archive.name+'\n')
            with self.assertRaises(c.IntegrityError): c.extract_bundle(archive,root/'symlink-out')

    def test_export_roundtrip_binding_and_missing_merge_shards(self):
        with tempfile.TemporaryDirectory(prefix='synthetic-cloud-test-',dir=ROOT/'tests/fixtures') as temp:
            root=Path(temp);source=root/'global';source.mkdir();dump(source/'manifest.json',self.manifest)
            d=c.load_shard(root/'runs',source,self.manifest,0,False,{})
            entries=[e for e in self.trials if e['shard_id']==0]
            for entry in entries:
                td=d/entry['trial_id'];td.mkdir()
                dump(td/'trial.json',dict(entry,config_hash=self.identity['config_hash'],code_sha256=self.identity['code_sha256'],execution_status='COMPLETED',planner_status='NO_SOLUTION',path_available=False,independent_valid=None,success=False,validation_wall_s=.01,synthetic_test_fixture=True))
                dump(td/'validation.json',{'valid':False,'errors':['no_path'],'synthetic_test_fixture':True})
            c.checkpoint(d,entries,'VALIDATED');c.freeze_raw(d)
            c.export_shard(root/'runs',0,root/'shard.zip')
            bundle=c.extract_bundle(root/'shard.zip',root/'extracted')
            self.assertEqual(bundle['trial_ids'],[e['trial_id'] for e in entries])
            with patch.object(c,'load_global',return_value=(self.config,self.maps,self.manifest)):
                with self.assertRaisesRegex(c.IntegrityError,'all20'):c.merge(source,[root/'shard.zip'],root/'merged')
                with self.assertRaisesRegex(c.IntegrityError,'duplicate'):c.merge(source,[root/'shard.zip',root/'shard.zip'],root/'merged')
            self.assertFalse((root/'merged').exists())

    def test_real_worker_five_planner_fixture_pipeline(self):
        # Analytical terrain is a correctness fixture, not experiment data.
        import os
        worker=os.environ.get('TERRAIN_CLOUD_WORKER')
        if not worker: self.skipTest('set TERRAIN_CLOUD_WORKER for real build integration')
        from terrain_io import write_map
        with tempfile.TemporaryDirectory(prefix='synthetic-cloud-test-',dir=ROOT/'tests/fixtures') as temp:
            root=Path(temp);path=root/'fixture.t25d'
            metadata=write_map(path,[[.02*i+.01*j for i in range(21)] for j in range(21)],spacing=(.5,.5),metadata={'synthetic_test_fixture':True})
            config=copy.deepcopy(self.config);config['profiles']['combined']['planning_budget_s']=1.
            for index,planner in enumerate(c.PLANNERS):
                entry=dict(self.trials[0],trial_id='fixture_'+planner,planner_id=planner,map_path=path.name,map_hash=metadata['data_sha256'],start=[1.,1.],goal=[9.,9.])
                td=root/f'trial_{index}'
                trial=c.launch_trial(entry,config,td,Path(worker),{},self.manifest,root)
                self.assertEqual(trial['execution_status'],'COMPLETED',(trial,(td/'stderr.log').read_text(errors='replace')))
                self.assertEqual(c.read_json(td/'phase.json')['phase'],'FINISHED')
                checked=c.subprocess.run([sys.executable,'-c',c.VALIDATION_CODE,str(ROOT/'tools'),str(td.resolve())],cwd=root,capture_output=True,text=True)
                self.assertEqual(checked.returncode,0,checked.stderr)
                self.assertTrue(c.read_json(td/'trial.json')['success'],planner)
    def test_serialization_watchdog_does_not_reduce_solve_budget(self):
        trial=self.launcher_watchdog({'phase':'SERIALIZING'},[0.,0.,61.,62.])
        self.assertEqual(trial['termination_reason'],'SERIALIZING_WATCHDOG')
        self.assertEqual(trial['time_budget_s'],30.)

    def test_all_twenty_bundles_merge_exactly_once_and_preserve_raw_trial_bytes(self):
        with tempfile.TemporaryDirectory(prefix='synthetic-cloud-test-',dir=ROOT/'tests/fixtures') as temp:
            root=Path(temp);source=root/'global';source.mkdir()
            dump(source/'manifest.json',self.manifest)
            for name in c.GLOBAL_FILES:
                if name!='manifest.json': (source/name).write_text('synthetic_test_fixture')
            archives=[];original_hashes={}
            for index in range(20):
                d=c.load_shard(root/'runs',source,self.manifest,index,False,{'synthetic_test_fixture':True})
                entries=[e for e in self.trials if e['shard_id']==index]
                for entry in entries:
                    td=d/entry['trial_id'];td.mkdir()
                    dump(td/'trial.json',dict(entry,config_hash=self.identity['config_hash'],code_sha256=self.identity['code_sha256'],execution_status='COMPLETED',planner_status='NO_SOLUTION',path_available=False,independent_valid=None,success=False,validation_wall_s=.01,synthetic_test_fixture=True))
                    dump(td/'validation.json',{'valid':False,'errors':['no_path'],'synthetic_test_fixture':True})
                    original_hashes[entry['trial_id']]=file_hash(td/'trial.json')
                c.checkpoint(d,entries,'VALIDATED');c.freeze_raw(d)
                archive=root/f'shard_{index}.zip';c.export_shard(root/'runs',index,archive);archives.append(archive)
            with patch.object(c,'load_global',return_value=(self.config,self.maps,self.manifest)):
                state=c.merge(source,archives,root/'merged')
            self.assertEqual(state['counts']['executed'],1500)
            self.assertEqual(state['counts']['validated_success'],0)
            self.assertEqual(state['counts']['validation_pending'],0)
            self.assertEqual(state['counts']['not_run'],0)
            c.verify_raw(root/'merged')
            for trial_id,original in original_hashes.items(): self.assertEqual(file_hash(root/'merged'/trial_id/'trial.json'),original)
            with patch.object(c,'load_global',return_value=(self.config,self.maps,self.manifest)):
                with self.assertRaises(c.IntegrityError): c.merge(source,archives,root/'merged')
            # A terminal BLOCKED attempt is not executed and cannot make a
            # campaign COMPLETE. Construct a separate labeled fixture campaign.
            import shutil
            original_shard=root/'runs/shard_00'
            blocked_shard=root/'blocked_runs/shard_00'
            shutil.copytree(original_shard,blocked_shard)
            (blocked_shard/'raw_artifact_inventory.json').unlink()
            blocked_entry=next(e for e in self.trials if e['shard_id']==0)
            blocked_path=blocked_shard/blocked_entry['trial_id']/'trial.json'
            blocked_trial=c.read_json(blocked_path)
            blocked_trial.update(execution_status='BLOCKED',planner_status='NOT_RUN',success=False)
            dump(blocked_path,blocked_trial)
            shard_state=c.checkpoint(blocked_shard,[e for e in self.trials if e['shard_id']==0],'VALIDATED')
            self.assertEqual(shard_state['status'],'PARTIAL')
            self.assertEqual(shard_state['counts']['not_run'],1)
            c.freeze_raw(blocked_shard)
            c.export_shard(root/'blocked_runs',0,root/'blocked_shard.zip')
            with patch.object(c,'load_global',return_value=(self.config,self.maps,self.manifest)):
                partial=c.merge(source,[root/'blocked_shard.zip',*archives[1:]],root/'blocked_merged')
            self.assertEqual(partial['status'],'PARTIAL')
            self.assertEqual(partial['counts']['executed'],1499)
            self.assertEqual(partial['counts']['not_run'],1)
            self.assertEqual(c.read_json(root/'blocked_merged/manifest.json')['status'],'PARTIAL')
            c.verify_raw(root/'blocked_merged')
    def test_partial_checkpoint_import_preserves_started_attempt_without_freezing_resume(self):
        with tempfile.TemporaryDirectory(prefix='synthetic-cloud-test-',dir=ROOT/'tests/fixtures') as temp:
            root=Path(temp);source=root/'global';source.mkdir();dump(source/'manifest.json',self.manifest)
            d=c.load_shard(root/'runs',source,self.manifest,0,False,{'synthetic_test_fixture':True})
            entry=next(e for e in self.trials if e['shard_id']==0)
            trial=d/entry['trial_id'];trial.mkdir();dump(trial/'attempt.json',{'synthetic_test_fixture':True})
            (trial/'phase.json.tmp').write_text('interrupted worker bytes')
            c.checkpoint(d,[e for e in self.trials if e['shard_id']==0],'INTERRUPTED')
            c.export_shard(root/'runs',0,root/'partial.zip',allow_partial=True)
            self.assertFalse((d/'raw_artifact_inventory.json').exists())
            with patch.object(c,'load_global',return_value=(self.config,self.maps,self.manifest)):
                imported=c.import_bundle(source,root/'partial.zip',root/'replacement')
                with self.assertRaises(c.IntegrityError): c.import_bundle(source,root/'partial.zip',root/'replacement')
                with self.assertRaisesRegex(c.IntegrityError,'completed shard'): c.merge(source,[root/'partial.zip'],root/'merged')
            self.assertTrue(imported['partial_checkpoint'])
            restored=root/'replacement/shard_00'
            self.assertFalse((restored/'raw_artifact_inventory.json').exists())
            self.assertEqual((restored/entry['trial_id']/'phase.json.tmp').read_text(),'interrupted worker bytes')
            self.assertTrue(c.retain_started(restored/entry['trial_id'],entry,self.config,self.manifest))
            self.assertEqual(c.read_json(restored/entry['trial_id']/'trial.json')['execution_status'],'CRASH')
            c.load_shard(root/'replacement',source,self.manifest,0,True,{'cpu':'replacement-host'})
    def test_zero_exit_with_worker_result_still_requires_finished_handshake(self):
        with tempfile.TemporaryDirectory(prefix='synthetic-cloud-test-',dir=ROOT/'tests/fixtures') as temp:
            root=Path(temp);proc=Mock();proc.pid=123;proc.poll.return_value=0;proc.wait.return_value=0
            def start(*args,**kwargs):
                dump(root/'trial/worker.json',{'execution_status':'COMPLETED','planner_status':'NO_SOLUTION','synthetic_test_fixture':True})
                dump(root/'trial/phase.json',{'phase':'SERIALIZING'})
                return proc
            with patch.object(c.subprocess,'Popen',side_effect=start),patch.object(c.psutil,'Process',side_effect=c.psutil.NoSuchProcess(123)):
                trial=c.launch_trial(self.trials[0],self.config,root/'trial',root/'worker',{},self.manifest,root)
            self.assertEqual(trial['execution_status'],'CRASH')
            self.assertEqual(trial['termination_reason'],'INCOMPLETE_PHASE_HANDSHAKE')
    def test_blocked_terminal_attempt_checkpoint_is_partial(self):
        with tempfile.TemporaryDirectory(prefix='synthetic-cloud-test-',dir=ROOT/'tests/fixtures') as temp:
            root=Path(temp);entry=self.trials[0];trial=root/entry['trial_id'];trial.mkdir()
            dump(trial/'trial.json',dict(entry,execution_status='BLOCKED',planner_status='NOT_RUN',path_available=False,independent_valid=None,success=False,validation_status='NO_REQUEST_INTERRUPTED_OR_BLOCKED',synthetic_test_fixture=True))
            state=c.checkpoint(root,[entry],'VALIDATED')
            self.assertEqual(state['status'],'PARTIAL')
            self.assertEqual(state['counts']['attempted'],1)
            self.assertEqual(state['counts']['executed'],0)
            self.assertEqual(state['counts']['not_run'],1)
            self.assertEqual(state['counts']['validation_pending'],0)
if __name__=='__main__':unittest.main()







