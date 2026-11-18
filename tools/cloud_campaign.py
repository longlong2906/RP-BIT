"""A3 v2: frozen Cloud worklists, sequential trials, verified shard transport.

No network actions or Cloud task dispatch happen in this module. Every started
planner attempt is retained; resume skips it even after process interruption.
"""
from __future__ import annotations
import argparse
from collections import Counter
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import zipfile
import psutil
import experiment_common as common
from experiment_common import ROOT, digest, dump, file_hash, load_config, worker_env
from independent_verifier import Oracle
from run_experiments import request_for, bitstar_log_counters

PROTOCOL_ID = 'a3_vs_all_v2'
PLANNERS = ('bitstar_a3','abitstar','aitstar','lazyprmstar_hauser2015','sem_rrtstar_common_v1')
CHUNK_BYTES = 32 * 1024 * 1024
MAX_BYTES = 32 * 1024 ** 3
MUTABLE = {'campaign_state.json','analysis_attempt.json','raw_artifact_inventory.json'}
GLOBAL_FILES = ('manifest.json','resolved_config.json','map_manifest.json','environment.json','source_at_run.zip','source_manifest.json')
class IntegrityError(ValueError): pass

def utc(): return dt.datetime.now(dt.timezone.utc).isoformat()
def read_json(path):
    def pairs(items):
        result = {}
        for k,v in items:
            if k in result: raise IntegrityError('duplicate JSON key: '+k)
            result[k]=v
        return result
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),object_pairs_hook=pairs,
        parse_constant=lambda x: (_ for _ in ()).throw(IntegrityError('nonfinite JSON: '+x)))
def relative(value):
    if not isinstance(value,str) or not value or any(c in value for c in '\\:\x00'):
        raise IntegrityError('unsafe relative path: '+repr(value))
    p=PurePosixPath(value)
    if p.is_absolute() or p.as_posix()!=value or '..' in p.parts or '.' in p.parts:
        raise IntegrityError('unsafe/noncanonical path: '+value)
    return p

def files(root):
    root=Path(root)
    if root.is_symlink() or not root.is_dir(): raise IntegrityError('nonregular artifact root')
    for p in sorted(root.rglob('*')):
        if p.is_symlink(): raise IntegrityError('symlink artifact: '+str(p))
        if p.is_file():
            if not stat.S_ISREG(p.stat().st_mode): raise IntegrityError('nonregular artifact')
            yield p
        elif not p.is_dir(): raise IntegrityError('nonregular artifact')
def inventory(root,excluded=()):
    return [{'path':p.relative_to(root).as_posix(),'size_bytes':p.stat().st_size,'sha256':file_hash(p)}
            for p in files(root) if p.relative_to(root).as_posix() not in excluded]
def check_inventory(root,records,excluded=()):
    if not isinstance(records,list) or records!=sorted(records,key=lambda r:r.get('path','')):
        raise IntegrityError('inventory not sorted')
    names=set()
    for r in records:
        if not isinstance(r,dict) or set(r)!={'path','size_bytes','sha256'}: raise IntegrityError('inventory schema')
        name=relative(r['path']).as_posix()
        if name in names: raise IntegrityError('duplicate inventory path')
        names.add(name);p=Path(root).joinpath(*PurePosixPath(name).parts)
        if p.is_symlink() or not p.is_file() or not isinstance(r['size_bytes'],int) or isinstance(r['size_bytes'],bool) or p.stat().st_size!=r['size_bytes'] or file_hash(p)!=r['sha256']:
            raise IntegrityError('inventory size/hash mismatch: '+name)
    if names!={r['path'] for r in inventory(root,excluded)}: raise IntegrityError('inventory allowlist mismatch')
def freeze_raw(root):
    records=inventory(root,MUTABLE);r={'schema_version':1,'entries':records,'inventory_sha256':digest(records)}
    p=Path(root)/'raw_artifact_inventory.json'
    if p.exists() and read_json(p)!=r: raise IntegrityError('immutable inventory differs')
    if not p.exists(): dump(p,r)
    return r
def verify_raw(root):
    r=read_json(Path(root)/'raw_artifact_inventory.json')
    if r.get('schema_version')!=1 or r.get('inventory_sha256')!=digest(r.get('entries')): raise IntegrityError('raw inventory digest')
    check_inventory(root,r['entries'],MUTABLE);return r

def dataset_file(root,name):
    parts=relative(name).parts;root=Path(root).resolve();p=root.joinpath(*parts)
    cursor=p
    while cursor!=root:
        if cursor.is_symlink(): raise IntegrityError('dataset symlink')
        cursor=cursor.parent
    if root not in p.resolve().parents or not p.is_file(): raise IntegrityError('missing dataset file: '+name)
    return p

def validate_dataset(config,maps,root,witnesses=True):
    s=config['profiles']['combined'];items=maps.get('maps',[])
    expected={(f,m) for f in s['families'] for m in s['map_seeds']}
    if maps.get('schema_version')!=5 or maps.get('generator_version')!='6.0.0' or maps.get('declared_map_seeds')!=s['map_seeds']: raise IntegrityError('dataset schema/generator/seed declaration')
    if maps.get('config_hash')!=digest(config) or maps.get('held_out') is not True or maps.get('seed_replacements')!=[] or maps.get('seed_replacement_policy')!='forbidden': raise IntegrityError('dataset config/heldout/seed identity')
    if len(items)!=len(expected) or {(m['family'],m['map_seed']) for m in items}!=expected or len({m['map_id'] for m in items})!=len(items): raise IntegrityError('dataset cardinality')
    if len({m['path'] for m in items})!=len(items): raise IntegrityError('duplicate map path')
    for m in items:
        p=dataset_file(root,m['path']);meta=read_json(p.with_suffix('.json'))
        if meta.get('generator_version')!='6.0.0' or meta.get('dataset_role')!='held_out_evaluation': raise IntegrityError('metadata generation role')
        if file_hash(p)!=m['sha256'] or meta.get('data_sha256')!=m['sha256'] or meta.get('queries')!=m['queries'] or meta.get('map_id')!=m['map_id'] or meta.get('generator_seed')!=m['map_seed']: raise IntegrityError('map/sidecar changed: '+m['map_id'])
        if [q['query_id'] for q in m['queries']]!=config['combined_benchmark']['required_query_ids']: raise IntegrityError('query set')
        oracle=Oracle(p) if witnesses else None
        for q in m['queries']:
            if q.get('query_hash')!=digest({'start':q['start'],'goal':q['goal'],'map_hash':m['sha256'],'model':config['model']}) or q.get('feasibility')!='feasible_witness_validated': raise IntegrityError('query qualification')
            ref=q.get('reference',{})
            if not ref.get('validation',{}).get('valid') or not ref.get('provenance') or ref.get('map_sha256')!=m['sha256'] or ref.get('path_sha256')!=digest(ref.get('path_xy')): raise IntegrityError('reference provenance')
            if witnesses:
                for xy,cost in [(ref['path_xy'],ref['cost_terrain_m'])]+[(w['path_xy'],w['validation']['cost_terrain_m']) for w in q.get('witnesses',[])]:
                    checked=oracle.validate_path(xy,q['start'],q['goal'])
                    tol=config['model']['numerical_abs_tolerance_m']+config['model']['numerical_rel_tolerance']*max(abs(cost),abs(checked.get('cost_terrain_m') or 0))
                    if not checked['valid'] or not math.isfinite(cost) or abs(cost-checked['cost_terrain_m'])>tol: raise IntegrityError('independent witness changed')

def plan(config,maps):
    s=config['profiles']['combined'];proto=config['combined_benchmark'];blocks=[]
    if proto['protocol_id']!=PROTOCOL_ID or tuple(s['planners'])!=PLANNERS: raise IntegrityError('unsupported protocol')
    for m in maps['maps']:
        f=s['families'].index(m['family']);rank=s['map_seeds'].index(m['map_seed'])
        for q in m['queries']:
            for seed_rank,seed in enumerate(s['planner_seeds']):
                row=(f+rank+seed_rank)%5;slot=f"{m['map_id']}_{q['query_id']}_s{seed}";group=[]
                for pos,planner in enumerate(PLANNERS[row:]+PLANNERS[:row]):
                    group.append({'trial_id':f"{m['map_id']}_{q['query_id']}_{planner}_s{seed}",'map_id':m['map_id'],'map_path':m['path'],'map_hash':m['sha256'],'family':m['family'],'map_seed':m['map_seed'],'query_id':q['query_id'],'query_hash':q['query_hash'],'start':q['start'],'goal':q['goal'],'reference':q['reference'],'geometry_label':q.get('geometry_label'),'planner_id':planner,'planner_seed':seed,'problem_slot_id':slot,'within_slot_order':pos,'latin_row':row,'shard_id':rank,'task_index':rank%proto['max_cloud_tasks'],'trial_order_policy':proto['order_policy']})
                blocks.append(group)
    random.Random(config['execution']['shuffle_trial_order_seed']).shuffle(blocks);trials=[]
    for i,group in enumerate(blocks):
        for t in group:
            t.update(problem_slot_order=i,shuffle_order=len(trials));trials.append(t)
    audit_plan(trials,config);return trials

def audit_plan(trials,config):
    s=config['profiles']['combined'];proto=config['combined_benchmark']
    if len(trials)!=proto['expected_trials'] or len({t['trial_id'] for t in trials})!=len(trials): raise IntegrityError('trial count/duplicate')
    groups={};shards={i:{'count':0,'maps':set(),'rows':Counter(),'families':set()} for i in range(proto['shard_count'])};positions={p:[0]*5 for p in PLANNERS}
    for t in trials:
        relative(t['trial_id']);groups.setdefault(t['problem_slot_id'],[]).append(t)
    for group in groups.values():
        t=group[0];f=s['families'].index(t['family']);m=s['map_seeds'].index(t['map_seed']);p=s['planner_seeds'].index(t['planner_seed']);row=(f+m+p)%5
        if len(group)!=5 or [g['planner_id'] for g in group]!=list(PLANNERS[row:]+PLANNERS[:row]) or len({(g['map_id'],g['map_hash'],g['query_hash'],g['planner_seed'],g['shard_id'],g['task_index']) for g in group})!=1 or t['shard_id']!=m or t['task_index']!=m%proto['max_cloud_tasks']: raise IntegrityError('slot Latin/shard identity')
        shard=shards[m];shard['count']+=5;shard['maps'].add(t['map_id']);shard['families'].add(t['family']);shard['rows'][row]+=1
        for pos,g in enumerate(group):
            if g['within_slot_order']!=pos: raise IntegrityError('within slot order')
            positions[g['planner_id']][pos]+=1
    if any(v!=60 for counts in positions.values() for v in counts): raise IntegrityError('global Latin imbalance')
    for shard in shards.values():
        if shard['count']!=75 or len(shard['maps'])!=5 or shard['families']!=set(s['families']) or sorted(shard['rows'].values())!=[3]*5: raise IntegrityError('shard balance/cardinality')
    return {'planned_trials':len(trials),'problem_slots':len(groups),'shard_count':len(shards),'trials_per_shard':75,'position_counts':positions,'task_trial_counts':{str(i):sum(t['task_index']==i for t in trials) for i in range(proto['max_cloud_tasks'])}}

def correctness_gate(worker,identity):
    p=Path(worker).resolve().parent/'cloud_setup.json'
    if not p.is_file(): raise IntegrityError('Cloud build/test/five planner smoke gate absent')
    gate=read_json(p)
    if (gate.get('worker_sha256')!=identity['worker_sha256'] or gate.get('code_sha256')!=identity['code_sha256']
        or gate.get('ompl_library_sha256')!=identity['ompl_library_sha256'] or gate.get('status')!='PASS'
        or gate.get('ctest_passed') is not True or gate.get('smoke_passed') is not True
        or set(gate.get('smoke_planner_ids',[]))!=set(PLANNERS)):
        raise IntegrityError('stale/failed Cloud correctness gate')

def prepare(config_path,map_manifest,dataset_root,output,worker):
    config=load_config(config_path,'combined');maps=read_json(map_manifest);validate_dataset(config,maps,dataset_root)
    trials=plan(config,maps);identity=common.cloud_identity(config,maps,Path(worker));correctness_gate(worker,identity)
    identity.update(plan_sha256=digest(trials),combined_protocol_sha256=digest(config['combined_benchmark']))
    output=Path(output)
    if output.exists() and any(output.iterdir()): raise IntegrityError('prepare output nonempty')
    output.mkdir(parents=True,exist_ok=True)
    manifest={'schema_version':3,'protocol_id':PROTOCOL_ID,'experiment_id':output.name,'profile':'combined','planned_trials':len(trials),'trials':trials,'identity':identity,'schedule':audit_plan(trials,config),'created_utc':utc(),'status':'FROZEN_PLANNED','attempt_policy':'one_primary_attempt_no_retry_no_seed_replacement','map_path_policy':'relative_to_explicit_dataset_root'}
    dump(output/'manifest.json',manifest);dump(output/'resolved_config.json',config);dump(output/'map_manifest.json',maps);dump(output/'environment.json',common.cloud_environment(Path(worker)))
    paths=list(dict.fromkeys(p.resolve() for p in list(common.source_files())+[Path(config_path),ROOT/'configs/dependencies.lock.json']))
    records={p.relative_to(ROOT).as_posix():file_hash(p) for p in paths}
    with zipfile.ZipFile(output/'source_at_run.zip','x',zipfile.ZIP_DEFLATED) as z:
        for p in paths: z.write(p,p.relative_to(ROOT).as_posix())
    dump(output/'source_manifest.json',{'code_sha256':identity['code_sha256'],'files':records,'archive_sha256':file_hash(output/'source_at_run.zip')})
    freeze_raw(output);return manifest

def load_global(source,dataset_root=None,worker=None):
    source=Path(source);verify_raw(source);config=read_json(source/'resolved_config.json');maps=read_json(source/'map_manifest.json');manifest=read_json(source/'manifest.json')
    if manifest.get('protocol_id')!=PROTOCOL_ID or manifest['identity']['config_hash']!=digest(config) or manifest['trials']!=plan(config,maps) or manifest['identity'].get('plan_sha256')!=digest(manifest['trials']): raise IntegrityError('global config/worklist mismatch')
    if dataset_root is not None: validate_dataset(config,maps,dataset_root,False)
    if worker is not None:
        current=common.cloud_identity(config,maps,Path(worker));expected=dict(manifest['identity']);expected.pop('plan_sha256');expected.pop('combined_protocol_sha256')
        if current!=expected: raise IntegrityError('source/binary/dependency/config/map drift')
        correctness_gate(worker,current)
    return config,maps,manifest

def selected_shards(config,task_index=None,shard_id=None):
    proto=config['combined_benchmark']
    if (task_index is None)==(shard_id is None): raise IntegrityError('select task-index XOR shard-id')
    if shard_id is not None:
        if not 0<=shard_id<proto['shard_count']: raise IntegrityError('shard-id range')
        return [shard_id]
    if not 0<=task_index<proto['max_cloud_tasks']: raise IntegrityError('task-index range')
    return [i for i in range(proto['shard_count']) if i%proto['max_cloud_tasks']==task_index]

def load_shard(output,source,manifest,index,resume,env):
    d=Path(output)/f'shard_{index:02d}';ids=[t['trial_id'] for t in manifest['trials'] if t['shard_id']==index]
    expected={'schema_version':1,'protocol_id':PROTOCOL_ID,'shard_id':index,'global_manifest_sha256':file_hash(Path(source)/'manifest.json'),'identity':manifest['identity'],'trial_ids':ids}
    if d.exists():
        if not resume or d.is_symlink(): raise IntegrityError('existing shard requires resume / regular directory')
        old=read_json(d/'shard_manifest.json')
        if any(old.get(k)!=v for k,v in expected.items()): raise IntegrityError('shard resume identity')
        if (d/'raw_artifact_inventory.json').exists(): verify_raw(d)
    else:
        d.mkdir(parents=True);dump(d/'shard_manifest.json',dict(expected,environment=env,created_utc=utc()))
    return d

def terminal_validation(d,t):
    wall=t.get('validation_wall_s')
    return t.get('validation_status')=='NO_REQUEST_INTERRUPTED_OR_BLOCKED' or ((Path(d)/'validation.json').is_file() and isinstance(wall,(int,float)) and math.isfinite(wall) and wall>=0)

def counts(directory,entries):
    r={k:0 for k in ('attempted','executed','not_run','offline_validated','validation_pending','validated_success','geometric_invalid','crash','watchdog','blocked','timeout')};r['planned']=len(entries)
    for e in entries:
        d=Path(directory)/e['trial_id']
        if not d.exists(): r['not_run']+=1;continue
        r['attempted']+=1
        if not (d/'trial.json').is_file(): r['validation_pending']+=1;continue
        t=read_json(d/'trial.json');executed=t.get('execution_status') not in ('BLOCKED','NOT_RUN','PLANNED');r['executed']+=executed;r['not_run']+=not executed
        for k in ('crash','watchdog','blocked','timeout'): r[k]+=t.get('execution_status')==k.upper()
        valid=terminal_validation(d,t);r['offline_validated']+=valid;r['validation_pending']+=not valid;r['validated_success']+=bool(t.get('success')) and valid
        r['geometric_invalid']+=valid and bool((t.get('path_available') and t.get('independent_valid') is False) or t.get('snapshots_valid') is False)
    return r

def checkpoint(d,entries,phase,**extra):
    r=counts(d,entries);status='COMPLETE' if r['executed']==r['planned'] and r['not_run']==0 and not r['validation_pending'] else 'PARTIAL'
    if r['geometric_invalid']: status='FAILED_CORRECTNESS'
    state={'schema_version':1,'protocol_id':PROTOCOL_ID,'status':status,'phase':phase,'utc':utc(),'counts':r,**extra};dump(Path(d)/'campaign_state.json',state);return state

def retain_started(d,e,config,manifest):
    if not d.exists() or (d/'trial.json').exists(): return False
    t=dict(e,schema_version=2,experiment_id=manifest['experiment_id'],config_hash=manifest['identity']['config_hash'],code_sha256=manifest['identity']['code_sha256'],time_budget_s=config['profiles']['combined']['planning_budget_s'],utc=utc(),execution_status='CRASH',planner_status='UNKNOWN',independent_valid=None,success=False,path_available=False,best_cost_terrain_m=None,first_solution_time_s=None,termination_reason='INTERRUPTED_PREVIOUS_RUN_RETAINED_NO_RETRY',administrative_interruption=True)
    if (d/'worker.json').is_file():
        try: t.update(read_json(d/'worker.json'));t.update(execution_status='CRASH',success=False,termination_reason='INTERRUPTED_PREVIOUS_RUN_RETAINED_NO_RETRY')
        except (OSError,ValueError) as exc: t['worker_read_error']=str(exc)
    dump(d/'trial.json',t);return True

def phase_read(path):
    if not path.exists(): return None
    try: value=read_json(path)
    except (OSError,ValueError): return None
    if value.get('phase') not in ('SETUP','SOLVING','SERIALIZING','FINISHED'): raise IntegrityError('invalid worker phase')
    return value

def launch_trial(e,config,d,worker,env,manifest,dataset_root):
    d=Path(d);d.mkdir(parents=True,exist_ok=False);request=request_for(e,config,'combined');request['cloud_phase_handshake']=True
    dump(d/'request.json',request);dump(d/'resolved_config.json',config)
    t=dict(e,schema_version=2,experiment_id=manifest['experiment_id'],config_hash=manifest['identity']['config_hash'],environment=env,code_sha256=manifest['identity']['code_sha256'],code_commit=env.get('code_commit'),utc=utc(),execution_status='BLOCKED',planner_status='NOT_RUN',independent_valid=None,success=False,path_available=False,time_budget_s=request['planning_budget_s'],best_cost_terrain_m=None,first_solution_time_s=None)
    cmd=[str(Path(worker).resolve()),str((d/'request.json').resolve()),str(d.resolve())];t['command']=cmd;dump(d/'attempt.json',{'utc':utc(),'status':'START_REQUESTED','entry':e,'command':cmd,'cwd_policy':'explicit_dataset_root'})
    began=time.perf_counter();phase='SETUP';phase_began=began;proc=None;failure=None;interrupted=False;peak=0.;exit_code=None;proto=config['combined_benchmark']
    with (d/'stdout.log').open('wb') as stdout,(d/'stderr.log').open('wb') as stderr:
        try:
            proc=subprocess.Popen(cmd,stdout=stdout,stderr=stderr,env=worker_env(),cwd=Path(dataset_root).resolve())
            try: monitor=psutil.Process(proc.pid)
            except psutil.NoSuchProcess: monitor=None
            while proc.poll() is None:
                now=time.perf_counter();observed=phase_read(d/'phase.json')
                if observed and observed['phase']!=phase:
                    nxt=observed['phase'];order={'SETUP':0,'SOLVING':1,'SERIALIZING':2,'FINISHED':3}
                    if order[nxt]<order[phase]: failure='PHASE_REGRESSION';break
                    phase=nxt;phase_began=now
                    timestamp=observed.get('solve_started_utc') if phase=='SOLVING' else observed.get('updated_utc')
                    if timestamp:
                        age=max(0.,(dt.datetime.now(dt.timezone.utc)-dt.datetime.fromisoformat(timestamp.replace('Z','+00:00'))).total_seconds());phase_began=now-age
                if monitor:
                    try: peak=max(peak,monitor.memory_info().rss/2**20)
                    except psutil.NoSuchProcess: pass
                limit=proto['setup_watchdog_s'] if phase=='SETUP' else request['planning_budget_s']+config['execution']['watchdog_grace_s'] if phase=='SOLVING' else proto['serialization_watchdog_s']
                if peak>config['execution']['memory_limit_mb']: failure='MEMORY_LIMIT'
                elif now-phase_began>limit: failure=phase+'_WATCHDOG'
                if failure: break
                time.sleep(.02)
            if failure and proc.poll() is None: proc.kill()
            exit_code=proc.wait()
        except OSError as exc: failure='BLOCKED';stderr.write(str(exc).encode())
        except KeyboardInterrupt: failure='INTERRUPTED';interrupted=True
        except (IntegrityError,ValueError) as exc: failure='PHASE_PROTOCOL_ERROR';t['phase_error']=str(exc)
        finally:
            if proc is not None and proc.poll() is None: proc.kill()
            if proc is not None: exit_code=proc.wait()
    if (d/'worker.json').is_file():
        try: t.update(read_json(d/'worker.json'))
        except (OSError,ValueError) as exc: failure='MALFORMED_WORKER_JSON';t['worker_read_error']=str(exc)
    if failure: t.update(execution_status='BLOCKED' if failure=='BLOCKED' else 'WATCHDOG' if failure.endswith('_WATCHDOG') or failure=='MEMORY_LIMIT' else 'CRASH',termination_reason=failure)
    elif exit_code!=0: t.update(execution_status='CRASH',termination_reason='NONZERO_WORKER_EXIT')
    elif not (d/'worker.json').exists(): t.update(execution_status='CRASH',termination_reason='MISSING_WORKER_RESULT')
    elif (phase_read(d/'phase.json') or {}).get('phase')!='FINISHED': t.update(execution_status='CRASH',termination_reason='INCOMPLETE_PHASE_HANDSHAKE')
    t.update(exit_code=exit_code,peak_rss_mb=peak if exit_code is not None else None,worker_wall_s=time.perf_counter()-began,watchdog_last_phase=phase,rss_measurement='sampled_peak_20ms_not_OS_exact_peak',success=False)
    if (d/'ompl.log').is_file() and e['planner_id'].startswith('bitstar'):
        text=(d/'ompl.log').read_text(errors='replace');matches=re.findall(r'Pruning disconnected (\d+) vertices.*?removed (\d+) samples',text)
        t.update(graph_prunings_log_observed=len(matches),vertices_disconnected_log_observed=sum(int(a) for a,b in matches),samples_pruned_log_observed=sum(int(b) for a,b in matches));t.update(bitstar_log_counters(text))
    dump(d/'trial.json',t)
    if interrupted: raise KeyboardInterrupt
    return t

def collect(source,output,dataset_root,worker,task_index=None,shard_id=None,resume=False):
    config,_,manifest=load_global(source,dataset_root,worker);env=common.cloud_environment(Path(worker));indices=selected_shards(config,task_index,shard_id);result={}
    for index in indices:
        d=load_shard(output,source,manifest,index,resume,env);entries=[t for t in manifest['trials'] if t['shard_id']==index]
        if not (d/'raw_artifact_inventory.json').exists():
            try:
                for e in entries:
                    td=d/e['trial_id'];retain_started(td,e,config,manifest)
                    if (td/'trial.json').exists(): continue
                    checkpoint(d,entries,'ACTIVE_TRIAL',active_trial_id=e['trial_id'])
                    t=launch_trial(e,config,td,worker,env,manifest,dataset_root);checkpoint(d,entries,'COLLECTION')
                    print(f"shard {index:02d} {e['trial_id']}: {t['execution_status']}/{t['planner_status']}",flush=True)
                checkpoint(d,entries,'EXECUTED_PENDING_VALIDATION')
            except KeyboardInterrupt: checkpoint(d,entries,'INTERRUPTED');raise
        result[index]=counts(d,entries)
    return result

VALIDATION_CODE="""import sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from validate_results import validate_trial
from independent_verifier import Oracle
from experiment_common import read_json
oracle=None
for d in sys.argv[2:]:
    r=read_json(Path(d)/'request.json') if (Path(d)/'request.json').exists() else None
    if r is not None and oracle is None: oracle=Oracle(r['map_path'])
    validate_trial(d,oracle)
    print('validated '+d,flush=True)
"""

def validate(source,output,dataset_root,task_index=None,shard_id=None,resume=False):
    config,_,manifest=load_global(source,dataset_root);result={}
    for index in selected_shards(config,task_index,shard_id):
        expected=Path(output)/f'shard_{index:02d}'
        if not expected.is_dir(): raise IntegrityError('collect shard before validation')
        d=load_shard(output,source,manifest,index,True,{});entries=[t for t in manifest['trials'] if t['shard_id']==index]
        if (d/'raw_artifact_inventory.json').exists(): result[index]=counts(d,entries);continue
        groups={};already=0
        for e in entries:
            td=d/e['trial_id']
            if not (td/'trial.json').is_file(): continue
            t=read_json(td/'trial.json')
            if terminal_validation(td,t): already+=1;continue
            groups.setdefault(t['map_path'],[]).append(td)
        if already and not resume: raise IntegrityError('partial validation requires resume')
        try:
            for pending in groups.values():
                cmd=[sys.executable,'-c',VALIDATION_CODE,str(ROOT/'tools'),*(str(p.resolve()) for p in pending)];checkpoint(d,entries,'OFFLINE_VALIDATION',active_validation_trial_ids=[p.name for p in pending])
                name='validation_command_'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');began=time.perf_counter();interrupted=False
                with (d/(name+'.stdout.log')).open('wb') as stdout,(d/(name+'.stderr.log')).open('wb') as stderr:
                    proc=subprocess.Popen(cmd,stdout=stdout,stderr=stderr,cwd=Path(dataset_root).resolve())
                    try: code=proc.wait()
                    except KeyboardInterrupt: interrupted=True;proc.kill();code=proc.wait()
                dump(d/(name+'.json'),{'command':cmd,'utc':utc(),'exit_code':code,'wall_s':time.perf_counter()-began,'administrative_interruption':interrupted})
                if interrupted: raise KeyboardInterrupt
                if code: checkpoint(d,entries,'VALIDATION_FAILED',validation_exit_code=code);raise IntegrityError('validator failed; retain logs and resume pending work')
                checkpoint(d,entries,'OFFLINE_VALIDATION')
            state=checkpoint(d,entries,'VALIDATED')
            if state['counts']['attempted']==len(entries) and not state['counts']['validation_pending']: freeze_raw(d)
        except KeyboardInterrupt: checkpoint(d,entries,'INTERRUPTED');raise
        result[index]=counts(d,entries)
    return result

def write_zip(source,output,metadata_name,metadata,records):
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists() or output.with_name(output.name+'.tmp').exists(): raise IntegrityError('transport output exists; do not overwrite')
    temp=output.with_name(output.name+'.tmp')
    with zipfile.ZipFile(temp,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for r in records: z.write(Path(source).joinpath(*relative(r['path']).parts),r['path'])
        z.writestr(metadata_name,json.dumps(metadata,sort_keys=True,ensure_ascii=False,allow_nan=False))
    temp.replace(output);output.with_name(output.name+'.sha256').write_text(file_hash(output)+'  '+output.name+'\n',encoding='ascii');return output

def export_shard(source,index,output,chunks_dir=None,allow_partial=False):
    if not isinstance(index,int) or not 0<=index<20: raise IntegrityError('shard-id range')
    d=Path(source)/f'shard_{index:02d}'
    if allow_partial and not (d/'raw_artifact_inventory.json').exists():
        return export_checkpoint(d,index,output,chunks_dir)
    raw=verify_raw(d);shard=read_json(d/'shard_manifest.json');state=read_json(d/'campaign_state.json')
    if shard.get('shard_id')!=index or state.get('counts',{}).get('attempted')!=75 or state.get('counts',{}).get('validation_pending'): raise IntegrityError('export requires 75 terminal and validated attempts')
    records=inventory(d);metadata={'schema_version':1,'artifact_type':'A3_V2_CLOUD_SHARD','protocol_id':PROTOCOL_ID,'shard_id':index,'global_manifest_sha256':shard['global_manifest_sha256'],'identity':shard['identity'],'trial_ids':shard['trial_ids'],'raw_inventory_sha256':raw['inventory_sha256'],'entries':records}
    result=write_zip(d,output,'bundle_manifest.json',metadata,records)
    return build_chunks(result,chunks_dir) if chunks_dir else {'archive':str(result),'sha256':file_hash(result)}

def build_chunks(archive,output):
    archive=Path(archive);output=Path(output)
    if output.exists() and any(output.iterdir()): raise IntegrityError('chunk directory nonempty')
    output.mkdir(parents=True,exist_ok=True);records=[]
    with archive.open('rb') as stream:
        while data:=stream.read(CHUNK_BYTES):
            name=f'part_{len(records):04d}.bin';(output/name).write_bytes(data);records.append({'path':name,'size_bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
    r={'schema_version':1,'artifact_type':'A3_V2_CLOUD_CHUNKS','archive_name':archive.name,'archive_size_bytes':archive.stat().st_size,'archive_sha256':file_hash(archive),'chunk_bytes':CHUNK_BYTES,'chunks':records};dump(output/'parts.json',r);return r

def reassemble(parts_path,output):
    parts_path=Path(parts_path);m=read_json(parts_path);records=m.get('chunks');name=m.get('archive_name')
    if m.get('schema_version')!=1 or m.get('artifact_type')!='A3_V2_CLOUD_CHUNKS' or m.get('chunk_bytes')!=CHUNK_BYTES or not isinstance(records,list) or not records or len(relative(name).parts)!=1: raise IntegrityError('chunk manifest schema')
    total=0
    for index,r in enumerate(records):
        if set(r)!={'path','size_bytes','sha256'} or r['path']!=f'part_{index:04d}.bin': raise IntegrityError('chunk name/order')
        size=r['size_bytes']
        if not isinstance(size,int) or isinstance(size,bool) or not 0<size<=CHUNK_BYTES or (index<len(records)-1 and size!=CHUNK_BYTES): raise IntegrityError('chunk size')
        p=dataset_file(parts_path.parent,r['path'])
        if p.stat().st_size!=size or file_hash(p)!=r['sha256']: raise IntegrityError('chunk hash/size')
        total+=size
    if total!=m.get('archive_size_bytes') or total>MAX_BYTES: raise IntegrityError('total transport size')
    output=Path(output)
    if output.exists():
        if output.is_symlink() or not output.is_file() or output.stat().st_size!=total or file_hash(output)!=m['archive_sha256']: raise IntegrityError('refuse mismatched existing archive')
        return output
    output.parent.mkdir(parents=True,exist_ok=True);temp=output.with_name(output.name+'.tmp')
    with temp.open('xb') as target:
        for r in records:
            with (parts_path.parent/r['path']).open('rb') as source: shutil.copyfileobj(source,target)
    if temp.stat().st_size!=total or file_hash(temp)!=m['archive_sha256']: raise IntegrityError('reassembled archive hash')
    temp.replace(output);output.with_name(output.name+'.sha256').write_text(m['archive_sha256']+'  '+output.name+'\n',encoding='ascii');return output

def extract_bundle(archive,output):
    archive=Path(archive);sidecar=archive.with_name(archive.name+'.sha256');output=Path(output)
    if archive.is_symlink() or sidecar.is_symlink() or not sidecar.is_file(): raise IntegrityError('regular archive/SHA256 sidecar required')
    side=sidecar.read_text(encoding='ascii').strip().split()
    if len(side)!=2 or side[1]!=archive.name or side[0]!=file_hash(archive): raise IntegrityError('archive hash mismatch')
    if output.exists(): raise IntegrityError('extraction output exists')
    with zipfile.ZipFile(archive) as z:
        entries=z.infolist();names=set()
        if len(entries)>500000 or sum(e.file_size for e in entries)>MAX_BYTES: raise IntegrityError('archive extraction limits')
        for e in entries:
            name=relative(e.filename).as_posix();mode=e.external_attr>>16
            if e.is_dir() or stat.S_IFMT(mode) not in (0,stat.S_IFREG) or name in names: raise IntegrityError('duplicate/nonregular ZIP member')
            names.add(name)
        if 'bundle_manifest.json' not in names: raise IntegrityError('missing bundle metadata')
        with tempfile.TemporaryDirectory() as scratch:
            p=Path(scratch)/'metadata.json';p.write_bytes(z.read('bundle_manifest.json'));bundle=read_json(p)
        if bundle.get('schema_version')!=1 or bundle.get('artifact_type') not in ('A3_V2_CLOUD_SHARD','A3_V2_CLOUD_CHECKPOINT') or bundle.get('protocol_id')!=PROTOCOL_ID: raise IntegrityError('bundle protocol/schema')
        records=bundle.get('entries',[]);listed=[relative(r['path']).as_posix() for r in records]
        if len(listed)!=len(set(listed)) or set(listed)|{'bundle_manifest.json'}!=names: raise IntegrityError('bundle allowlist')
        output.mkdir(parents=True)
        for r in records:
            entry=z.getinfo(r['path'])
            if entry.file_size!=r['size_bytes']: raise IntegrityError('bundle member size')
            target=output.joinpath(*relative(r['path']).parts);target.parent.mkdir(parents=True,exist_ok=True)
            with z.open(entry) as source,target.open('xb') as dest: shutil.copyfileobj(source,dest)
            if file_hash(target)!=r['sha256']: raise IntegrityError('bundle member hash')
    check_inventory(output,records);raw=verify_raw(output)
    if raw['inventory_sha256']!=bundle.get('raw_inventory_sha256'): raise IntegrityError('bundle raw inventory')
    return bundle

def merge(source,bundles,output):
    config,_,manifest=load_global(source);output=Path(output)
    if output.exists(): raise IntegrityError('frozen merge output exists; never overwrite')
    global_hash=file_hash(Path(source)/'manifest.json');seen=set();prepared=[]
    with tempfile.TemporaryDirectory(prefix='a3-v2-merge-') as scratch:
        scratch=Path(scratch)
        for i,value in enumerate(bundles):
            archive=Path(value)
            if archive.name=='parts.json': archive=reassemble(archive,scratch/f'archive_{i}'/read_json(archive)['archive_name'])
            extracted=scratch/f'bundle_{i}';b=extract_bundle(archive,extracted);index=b.get('shard_id')
            if b.get('artifact_type')!='A3_V2_CLOUD_SHARD': raise IntegrityError('merge requires completed shard, not partial checkpoint')
            if not isinstance(index,int) or isinstance(index,bool) or index in seen or not 0<=index<20: raise IntegrityError('duplicate/out-of-range shard')
            seen.add(index);expected=[t['trial_id'] for t in manifest['trials'] if t['shard_id']==index]
            if b.get('identity')!=manifest['identity'] or b.get('global_manifest_sha256')!=global_hash or b.get('trial_ids')!=expected: raise IntegrityError('bundle global identity/worklist')
            internal=read_json(extracted/'shard_manifest.json')
            if internal.get('identity')!=manifest['identity'] or internal.get('trial_ids')!=expected or internal.get('shard_id')!=index or internal.get('global_manifest_sha256')!=global_hash: raise IntegrityError('internal shard identity')
            for e in [t for t in manifest['trials'] if t['shard_id']==index]:
                td=extracted/e['trial_id'];t=read_json(td/'trial.json')
                for k in ('trial_id','map_id','map_hash','query_id','query_hash','planner_id','planner_seed','shard_id'):
                    if t.get(k)!=e.get(k): raise IntegrityError('transported trial identity: '+k)
                if t.get('config_hash')!=manifest['identity']['config_hash'] or t.get('code_sha256')!=manifest['identity']['code_sha256'] or not terminal_validation(td,t): raise IntegrityError('transported trial config/source/validation')
            prepared.append((index,archive,extracted,b))
        if seen!=set(range(20)): raise IntegrityError('merge requires all20 disjoint shards')
        output.parent.mkdir(parents=True,exist_ok=True);staging=Path(tempfile.mkdtemp(prefix=output.name+'.staging-',dir=output.parent))
        for name in GLOBAL_FILES: shutil.copyfile(Path(source)/name,staging/name)
        imports=[]
        for index,archive,extracted,b in prepared:
            meta=staging/'shards'/f'shard_{index:02d}'
            for p in files(extracted):
                rel=p.relative_to(extracted);target=(staging/rel) if rel.parts[0] in b['trial_ids'] else (meta/rel);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
            imports.append({'shard_id':index,'archive_sha256':file_hash(archive),'raw_inventory_sha256':b['raw_inventory_sha256']})
        r=counts(staging,manifest['trials']);status='FAILED_CORRECTNESS' if r['geometric_invalid'] else 'PARTIAL' if r['not_run'] or r['validation_pending'] else 'COMPLETE_VALIDATED'
        dump(staging/'manifest.json',dict(manifest,status=status,counts=r,global_manifest_sha256=global_hash,shard_imports=sorted(imports,key=lambda x:x['shard_id']),merged_utc=utc()))
        state={'schema_version':1,'protocol_id':PROTOCOL_ID,'status':'FAILED_CORRECTNESS' if r['geometric_invalid'] else 'PARTIAL' if r['not_run'] or r['validation_pending'] else 'COMPLETE','phase':'MERGED_VALIDATED','counts':r,'utc':utc(),'analysis_status':'NOT_RUN'};dump(staging/'campaign_state.json',state);freeze_raw(staging);staging.rename(output)
    return state


def export_checkpoint(directory,index,output,chunks_dir=None):
    # Freeze a copy, never the resumable live shard. A concurrent writer causes
    # the complete before/after inventory check to reject this snapshot.
    before=inventory(directory)
    shard=read_json(Path(directory)/'shard_manifest.json')
    if shard.get('shard_id')!=index: raise IntegrityError('checkpoint shard identity')
    with tempfile.TemporaryDirectory(prefix='a3-v2-checkpoint-') as temp:
        snapshot=Path(temp)/'snapshot';shutil.copytree(directory,snapshot)
        check_inventory(directory,before)
        raw=freeze_raw(snapshot);records=inventory(snapshot)
        metadata={'schema_version':1,'artifact_type':'A3_V2_CLOUD_CHECKPOINT','protocol_id':PROTOCOL_ID,'shard_id':index,'global_manifest_sha256':shard['global_manifest_sha256'],'identity':shard['identity'],'trial_ids':shard['trial_ids'],'raw_inventory_sha256':raw['inventory_sha256'],'entries':records,'resume_policy':'retain_started_attempts_without_retry'}
        archive=write_zip(snapshot,output,'bundle_manifest.json',metadata,records)
    return build_chunks(archive,chunks_dir) if chunks_dir else {'archive':str(archive),'sha256':file_hash(archive),'partial_checkpoint':True}

def import_bundle(source,bundle_path,output):
    config,_,manifest=load_global(source)
    with tempfile.TemporaryDirectory(prefix='a3-v2-import-') as temp:
        temp=Path(temp);archive=Path(bundle_path)
        if archive.name=='parts.json': archive=reassemble(archive,temp/read_json(archive)['archive_name'])
        extracted=temp/'extracted';bundle=extract_bundle(archive,extracted);index=bundle.get('shard_id')
        if not isinstance(index,int) or isinstance(index,bool) or not 0<=index<20: raise IntegrityError('import shard-id range')
        expected=[t['trial_id'] for t in manifest['trials'] if t['shard_id']==index]
        if bundle.get('identity')!=manifest['identity'] or bundle.get('global_manifest_sha256')!=file_hash(Path(source)/'manifest.json') or bundle.get('trial_ids')!=expected: raise IntegrityError('import global identity/worklist')
        internal=read_json(extracted/'shard_manifest.json')
        if internal.get('identity')!=manifest['identity'] or internal.get('trial_ids')!=expected or internal.get('shard_id')!=index: raise IntegrityError('import internal shard identity')
        destination=Path(output)/f'shard_{index:02d}'
        if destination.exists(): raise IntegrityError('import refuses overwrite of existing shard')
        for e in [t for t in manifest['trials'] if t['shard_id']==index]:
            trial_path=extracted/e['trial_id']/'trial.json'
            if trial_path.exists():
                trial=read_json(trial_path)
                if any(trial.get(k)!=e.get(k) for k in ('trial_id','map_id','map_hash','query_id','query_hash','planner_id','planner_seed','shard_id')): raise IntegrityError('import trial identity')
        destination.parent.mkdir(parents=True,exist_ok=True)
        staging=Path(tempfile.mkdtemp(prefix=destination.name+'.import-',dir=destination.parent))
        shutil.copytree(extracted,staging,dirs_exist_ok=True)
        if bundle['artifact_type']=='A3_V2_CLOUD_CHECKPOINT':
            (staging/'raw_artifact_inventory.json').unlink()
            dump(staging/'import_provenance.json',{'schema_version':1,'archive_sha256':file_hash(archive),'snapshot_inventory_sha256':bundle['raw_inventory_sha256'],'imported_utc':utc(),'resume_policy':'retain_started_attempts_without_retry'})
        staging.rename(destination)
    return {'shard_id':index,'imported_directory':str(destination),'partial_checkpoint':bundle['artifact_type']=='A3_V2_CLOUD_CHECKPOINT'}
def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    for name in ('prepare','dry-run'):
        p=sub.add_parser(name);p.add_argument('--config',type=Path,required=True);p.add_argument('--map-manifest',type=Path,required=True);p.add_argument('--dataset-root',type=Path)
        if name=='prepare': p.add_argument('--output',type=Path,required=True);p.add_argument('--worker',type=Path,required=True)
        p.add_argument('--allow-combined-benchmark',action='store_true')
    for name in ('collect','validate'):
        p=sub.add_parser(name);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--dataset-root',type=Path,required=True)
        if name=='collect': p.add_argument('--worker',type=Path,required=True)
        select=p.add_mutually_exclusive_group(required=True);select.add_argument('--task-index',type=int);select.add_argument('--shard-id',type=int)
        p.add_argument('--resume',action='store_true');p.add_argument('--allow-combined-benchmark',action='store_true')
    p=sub.add_parser('export');p.add_argument('--input',type=Path,required=True);p.add_argument('--shard-id',type=int,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--chunks-dir',type=Path);p.add_argument('--allow-partial',action='store_true');p.add_argument('--allow-combined-benchmark',action='store_true')
    p=sub.add_parser('reassemble');p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--allow-combined-benchmark',action='store_true')
    p=sub.add_parser('import');p.add_argument('--input',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--allow-combined-benchmark',action='store_true')
    p=sub.add_parser('merge');p.add_argument('--input',type=Path,required=True);p.add_argument('--bundles',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--allow-combined-benchmark',action='store_true')
    args=parser.parse_args(argv)
    if not args.allow_combined_benchmark: parser.error('explicit --allow-combined-benchmark required')
    try:
        if args.command in ('prepare','dry-run'):
            root=args.dataset_root or args.map_manifest.parent
            if args.command=='dry-run':
                config=load_config(args.config,'combined');maps=read_json(args.map_manifest);validate_dataset(config,maps,root);result=audit_plan(plan(config,maps),config)
            else: result=prepare(args.config,args.map_manifest,root,args.output,args.worker)['schedule']
        elif args.command in ('collect','validate'):
            options={'task_index':args.task_index,'shard_id':args.shard_id,'resume':args.resume}
            result=collect(args.input,args.output,args.dataset_root,args.worker,**options) if args.command=='collect' else validate(args.input,args.output,args.dataset_root,**options)
        elif args.command=='export': result=export_shard(args.input,args.shard_id,args.output,args.chunks_dir,args.allow_partial)
        elif args.command=='import': result=import_bundle(args.input,args.bundle,args.output)
        elif args.command=='reassemble': result={'archive':str(reassemble(args.input,args.output))}
        else:
            result=merge(args.input,args.bundles,args.output);print(json.dumps(result,allow_nan=False));return 1 if result['counts']['geometric_invalid'] else 2 if result['counts']['not_run'] or result['counts']['validation_pending'] else 0
        print(json.dumps(result,allow_nan=False));return 0
    except KeyboardInterrupt: print('PARTIAL: interrupted; retain artifacts and resume same frozen identity',file=sys.stderr);return 130
    except (OSError,ValueError,KeyError,zipfile.BadZipFile) as exc: print('FAILED: '+str(exc),file=sys.stderr);return 2

if __name__=='__main__': raise SystemExit(main())






