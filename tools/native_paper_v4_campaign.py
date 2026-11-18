"""Schema-13 wave controller. Only the explicit Linux main gate permits 1500 trials."""
from __future__ import annotations
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
import datetime as dt
import json
import os
from pathlib import Path
import random
import sys
import time

from experiment_common import dump, digest
from native_paper_v4_protocol import (ROOT, PROTOCOL_ID, SCHEMA_VERSION, BACKENDS,
    default_config, load_config, generate_dataset, plan_trials, source_identity, sha256)
from terrain_compare_campaign import binary_paths, environment_identity
from terrain_compare_runtime import capture_native_dependencies
from terrain_compare_scheduler import (run_campaign, detect_runtime, read_json,
    collect_reservations, GIB)

MODES = ('smoke-serial','smoke-parallel','main')

def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def owned_path(workspace,path,base):
    w=Path(workspace).resolve(); path=Path(path).absolute(); base=Path(base).absolute()
    if not path.is_relative_to(base) or not base.is_relative_to(w): raise ValueError('active path outside owned workspace')
    for candidate in (path,*path.parents):
        if candidate==w: break
        if candidate.is_symlink() or getattr(candidate,'is_junction',lambda:False)():
            raise ValueError('links are forbidden in active collection paths')
    resolved=path.resolve()
    if not resolved.is_relative_to(base.resolve()) or not resolved.is_relative_to(w):
        raise ValueError('active path resolves outside the owned target')
    return path

def normalize_map(m, directory, mode=None):
    p=Path(directory); q=m['queries'][0]
    return dict(map_id=m['map_id'], family=m['family'], map_path=str(p/m['path']),
        map_sha256=m['sha256'], cloud_path=str(p/m['cloud_path']), cloud_sha256=m['cloud_sha256'],
        bounds=m.get('bounds',[0.,0.,40.,40.]), start=q['start'], goal=q['goal'], mode=mode)

def prepare(workspace, build, python, profile=None):
    w=Path(workspace).resolve(); b=Path(build).resolve(); python=Path(os.path.abspath(python))
    if (w/'prepared.json').exists(): raise ValueError('already prepared; resume the frozen workspace')
    binary,library=binary_paths(b)
    for p in (binary,library,python):
        if not p.is_file(): raise ValueError('BLOCKED missing built dependency: '+str(p))
    w.mkdir(parents=True,exist_ok=True)
    identity=source_identity()
    roots=list(dict.fromkeys(map(str,(binary,library,python))))
    native=capture_native_dependencies(roots)
    identity.update(binary_hashes={**native['binary_hashes'],**{p:sha256(p) for p in roots}},
        native_roots=roots,native_linkage=native['linkage'],environment=environment_identity(python))
    registry={k:dict(argv=[str(python),str(ROOT/'tools/native_paper_v4_worker.py')],cwd=str(ROOT)) for k in BACKENDS}
    dump(w/'registry.json',registry)
    config=load_config(ROOT/'configs/native_paper_kaggle_v4.yaml')
    profiles=[profile] if profile else ['smoke','main']; maps={}
    for name in profiles:
        folder=w/'data'/name
        if (folder/'manifest.json').exists():
            ds=read_json(folder/'manifest.json')
            if ds.get('config_hash')!=digest(config) or ds.get('profile')!=name:
                raise ValueError('pre-generated dataset does not match the frozen configuration')
        else: ds=generate_dataset(config,name,folder)
        plan=plan_trials(config,name,ds,folder,identity)
        for e in plan['entries']:
            e['request'].update(worker_binary=str(binary),bridge_library=str(library))
            if os.name=='nt': e['request']['library_search_paths']=[str(b),'C:/mingw64/bin']
        dump(w/'plans'/(name+'.json'),plan)
        maps[name]=[normalize_map(m,folder) for m in ds['maps']]
    dump(w/'maps.json',maps)
    frozen={p.relative_to(w).as_posix():sha256(p) for folder in ('plans','data')
            for p in sorted((w/folder).rglob('*')) if p.is_file()}
    frozen.update({p:sha256(w/p) for p in ('registry.json','maps.json')})
    dump(w/'prepared.json',dict(protocol_id=PROTOCOL_ID,schema_version=SCHEMA_VERSION,
        build_dir=str(b),controller_python=str(python),source_root=str(ROOT),identity=identity,
        frozen_inputs=frozen,profiles=profiles,created_utc=utc()))
    return dict(status='PREPARED',profiles=profiles)

def check_frozen(workspace):
    w=Path(workspace); p=read_json(w/'prepared.json')
    if (p.get('protocol_id'),p.get('schema_version'))!=(PROTOCOL_ID,SCHEMA_VERSION):
        raise ValueError('not a schema-13 workspace')
    if source_identity()['source_sha256']!=p['identity']['source_sha256']:
        raise ValueError('source changed after preparation')
    for name,expected in p['identity']['binary_hashes'].items():
        if sha256(name)!=expected: raise ValueError('runtime bytes changed: '+name)
    n=capture_native_dependencies(p['identity']['native_roots'])
    if n['linkage']!=p['identity']['native_linkage']: raise ValueError('native linkage drift')
    if environment_identity(p['controller_python'])!=p['identity']['environment']:
        raise ValueError('Python package/runtime drift')
    for name,expected in p['frozen_inputs'].items():
        if sha256(w/name)!=expected: raise ValueError('frozen input changed: '+name)
    return p

def correctness(workspace):
    from native_paper_v3_campaign import REQUIRED_CTESTS
    from terrain_compare_receipt import run as recorded
    w=Path(workspace); p=check_frozen(w)
    names=tuple(dict.fromkeys((*REQUIRED_CTESTS,'safe_length_unit','safe_length_field',
        'safe_length_runner','safe_length_oracle','safe_length_ompl')))
    regex='^('+'|'.join(names)+')$'
    import subprocess
    listing=subprocess.run(['ctest','--test-dir',p['build_dir'],'--show-only=json-v1','-R',regex],
        check=True,capture_output=True,text=True)
    if {t['name'] for t in json.loads(listing.stdout)['tests']}!=set(names):
        raise ValueError('missing required correctness tests')
    commands=[(['ctest','--test-dir',p['build_dir'],'--output-on-failure','-R',regex],'ctest')]
    for name in ('native_paper_v4_protocol','native_paper_v4_worker','native_paper_v4_storage',
                 'native_paper_v4_analysis','native_paper_v4_campaign','native_paper_v4_resume','kaggle_native_paper_v4','safe_length_analysis'):
        commands.append(([p['controller_python'],str(ROOT/'tests'/('test_'+name+'.py'))],name))
    binary,library=binary_paths(Path(p['build_dir']))
    commands.append(([p['controller_python'],str(ROOT/'tests/test_native_paper_v4_fixed_work.py'),
        '--worker',str(binary),'--library',str(library)],'fixed_work'))
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    if os.name=='nt': env['PATH']='C:/mingw64/bin;'+str(ROOT/'.deps/install/lib')+';'+env.get('PATH','')
    records=[]; passed=True; platform_gaps=[]
    for command,name in commands:
        r=recorded(command,w/'logs/correctness',name,cwd=ROOT,env=env); records.append(r)
        err=(w/'logs/correctness'/(name+'.stderr.log')).read_text(errors='replace')
        if 'skipped=' in err and os.name=='nt' and name in ('native_paper_v4_storage','kaggle_native_paper_v4'):
            platform_gaps.append('Windows cannot exercise Linux symlink/subreaper controls in '+name+'; Linux gate remains mandatory')
        elif 'skipped=' in err: passed=False
        if r['exit_code']: passed=False
        if not passed: break
    status='FAILED' if not passed else 'PASS_LOCAL_WITH_PLATFORM_GAPS' if platform_gaps else 'PASS'
    result=dict(status=status,identity_sha256=digest(p['identity']),platform_gaps=platform_gaps,
                commands=records,required_test_count=len(names),created_utc=utc())
    dump(w/'correctness.json',result)
    if not passed: raise RuntimeError('correctness gate failed; raw logs retained')
    return result

def wave_groups(plan):
    ids=sorted({e['map_id'] for e in plan['entries']})
    random.Random(73139).shuffle(ids)
    return [ids[k:k+4] for k in range(0,len(ids),4)]

def _validate_map_task(args):
    from native_paper_v4_analysis import validate_map
    return validate_map(*args)

def smoke_evidence(workspace,mode):
    from native_paper_v4_analysis import validation_code_identity
    w=Path(workspace); m=read_json(w/'manifests'/(mode+'.json'))
    plan=read_json(w/'plans/smoke.json')
    expected={e['trial_id']:e for e in plan['entries']}
    if len(expected)!=21 or len(m['entries'])!=21 or {e['trial_id'] for e in m['entries']}!=set(expected):
        raise ValueError('smoke must contain exactly the 21 frozen trials')
    if m['status']!='COMPLETE' or any(e['execution_status']!='COMPLETED' for e in m['entries']):
        raise ValueError('required smoke is incomplete')
    ledgers={}; shards={}; index=read_json(w/'shards/index.json')['entries']
    for e in m['entries']:
        if digest(e['request'])!=digest(expected[e['trial_id']]['request']): raise ValueError('smoke request drift')
        ledgerpath=w/'verdicts'/mode/(e['trial_id']+'.json'); ledger=read_json(ledgerpath)
        v=ledger['row']
        if v.get('errors') or v.get('correctness_errors'): raise ValueError('smoke correctness failed')
        serialized=(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n').encode('utf-8')
        import hashlib
        if hashlib.sha256(serialized).hexdigest()!=ledger['verdict_sha256'] or ledger['receipt']['verdict_sha256']!=ledger['verdict_sha256']:
            raise ValueError('smoke verdict hash mismatch')
        if v.get('trial_id')!=e['trial_id'] or v.get('backend')!=e['backend']:
            raise ValueError('smoke verdict identity mismatch')
        if v.get('planner_status',v.get('native_status')) in ('ERROR','MODEL_ERROR','BLOCKED_DEPENDENCY'):
            raise ValueError('smoke model/dependency failed')
        if v.get('validation_code_identity')!=validation_code_identity(): raise ValueError('stale smoke verifier')
        key=e['wave_id']+'/'+e['map_id']; receipt=index[key]
        if receipt['inventory'][e['trial_id']+'/validation.json']['sha256']!=ledger['verdict_sha256']:
            raise ValueError('shard verdict and ledger differ')
        shards[receipt['archive']]=receipt['sha256']; ledgers[e['trial_id']]=sha256(ledgerpath)
    for name,expected_hash in shards.items():
        if sha256(w/name)!=expected_hash: raise ValueError('smoke shard changed')
    if mode=='smoke-parallel' and m.get('max_concurrency',0)!=4:
        raise ValueError('parallel smoke did not demonstrate four workers')
    return dict(prepared_identity_sha256=digest(read_json(w/'prepared.json')['identity']),
        plan_sha256=sha256(w/'plans/smoke.json'),manifest_sha256=sha256(w/'manifests'/(mode+'.json')),
        validation_code_identity=validation_code_identity(),ledgers=ledgers,shards=shards)

def require_smoke(workspace,mode):
    w=Path(workspace); evidence=smoke_evidence(w,mode)
    if read_json(w/'readiness'/(mode+'.json'))!=evidence: raise ValueError('stale or missing smoke readiness receipt')
    return read_json(w/'manifests'/(mode+'.json'))

def run(workspace,mode,*,allow_main=False,force_local=False,resume=False,session_elapsed_s=0.):
    from native_paper_v4_storage import ArtifactStore,preflight
    from native_paper_v4_worker import compact_trial
    w=Path(workspace).resolve(); p=check_frozen(w)
    if mode not in MODES: raise ValueError('unknown mode')
    gate=read_json(w/'correctness.json')
    allowed={'PASS'}
    if force_local and mode!='main' and os.name=='nt': allowed.add('PASS_LOCAL_WITH_PLATFORM_GAPS')
    if gate.get('status') not in allowed or gate.get('identity_sha256')!=digest(p['identity']):
        raise ValueError('current correctness gate required')
    if mode=='main' and (not allow_main or force_local or not sys.platform.startswith('linux')):
        raise ValueError('main requires --allow-main on Linux; local runs are smoke only')
    runtime=detect_runtime(); workers=1 if mode=='smoke-serial' else 4
    if workers==4 and runtime['cpu_capacity']<4: raise RuntimeError('BLOCKED_RESOURCE: four CPU capacity required')
    reservations={b:8*GIB for b in BACKENDS}
    if mode!='smoke-serial': reservations=collect_reservations(require_smoke(w,'smoke-serial'))
    if set(reservations)!=set(BACKENDS): raise RuntimeError('BLOCKED_RESOURCE: missing measured backend RSS')
    if mode=='main': require_smoke(w,'smoke-parallel')
    if workers==4 and 4*max(reservations.values())>.8*runtime['memory_limit_bytes']:
        raise RuntimeError('BLOCKED_RESOURCE: measured reservations cannot admit four planner lanes')
    profile='main' if mode=='main' else 'smoke'
    plan=read_json(w/'plans'/(profile+'.json')); registry=read_json(w/'registry.json')
    maps={m['map_id']:dict(m,mode=mode) for m in read_json(w/'maps.json')[profile]}
    target=w/'manifests'/(mode+'.json')
    if target.exists():
        if not resume: raise ValueError('resume required for existing mode')
        manifest=read_json(target)
        if manifest['plan_sha256']!=digest(plan): raise ValueError('resume plan drift')
    else:
        if resume: raise ValueError('no existing campaign to resume')
        manifest=dict(protocol_id=PROTOCOL_ID,schema_version=13,mode=mode,profile=profile,
            plan_sha256=digest(plan),entries=copy.deepcopy(plan['entries']),status='RUNNING',
            wave_order_seed=73139,
            waves=[dict(wave_id=mode+'-w%03d'%i,map_ids=group,status='PLANNED')
                   for i,group in enumerate(wave_groups(plan))],max_concurrency=0,created_utc=utc())
        dump(target,manifest)
    entries={e['trial_id']:e for e in manifest['entries']}; store=ArtifactStore(w)
    started=time.monotonic(); raw_budget=1.5*GIB; pack_budget=1.875*GIB
    # A resumed session must retain already observed high-water marks.
    shard_index=read_json(w/'shards/index.json')['entries'] if (w/'shards/index.json').exists() else {}
    for previous in manifest['waves']:
        observed=previous.get('disk_observed',{})
        measured=observed.get('uncompressed_bytes',sum(r['uncompressed_bytes'] for k,r in shard_index.items()
            if k.startswith(previous['wave_id']+'/')))
        raw_budget=max(raw_budget,1.25*measured); pack_budget=max(pack_budget,1.25*measured)
    for wi,wave in enumerate(manifest['waves']):
        if wave['status']=='PACKED': continue
        elapsed=session_elapsed_s+time.monotonic()-started
        if elapsed>=28800:
            manifest['status']='PARTIAL_SESSION'; break
        wave_id=wave['wave_id']; output=w/'collection'/wave_id
        if wave['status'] in ('PLANNED','COLLECTING'):
            # Kaggle saves all of /kaggle/working, including this repo/runtime,
            # not only the campaign directory. The scope is measured, never cleaned.
            scope=Path('/kaggle/working') if sys.platform.startswith('linux') and w.is_relative_to('/kaggle/working') else w.parent if w.name=='campaign' else w
            admission=preflight(scope,output_path=scope,first_wave=wi==0,
                raw_budget_bytes=int(raw_budget),pack_budget_bytes=int(pack_budget))
            wave['disk_admission']=admission; dump(target,manifest)
            if not admission['allowed']:
                manifest['status']='PARTIAL_DISK'; break
            sub=copy.deepcopy(plan); sub['entries']=[copy.deepcopy(e) for e in plan['entries'] if e['map_id'] in wave['map_ids']]
            sub['planned_slots']=sub['planned_executable']=len(sub['entries'])
            subpath=w/'waves'/(wave_id+'.json')
            if subpath.exists():
                if read_json(subpath)!=sub: raise ValueError('wave plan drift')
            else: dump(subpath,sub)
            wave['status']='COLLECTING'; dump(target,manifest)
            collected=run_campaign(subpath,registry,output,max_workers=workers,reservations=reservations,
                resume=(output/'manifest.json').exists(),force_local=force_local,
                session_elapsed_s=elapsed,soft_stop_s=28800)
            for e in collected['entries']: entries[e['trial_id']].update(e)
            observed=max((c['concurrency'] for c in collected.get('concurrency_timeline',[])),default=0)
            manifest['max_concurrency']=max(manifest['max_concurrency'],observed)
            if collected['status']!='COLLECTED_VALIDATION_PENDING':
                manifest['status']=collected['status']; dump(target,manifest); return manifest
            wave['status']='DRAINED'; dump(target,manifest)
        # No measured process is alive beyond this barrier. Moves are entirely
        # inside the new workspace and idempotent after an interrupted controller.
        current=[]
        for mid in wave['map_ids']:
            destination=owned_path(w,w/'active'/wave_id/mid,w/'active'); destination.mkdir(parents=True,exist_ok=True)
            for e in manifest['entries']:
                if e['map_id']!=mid: continue
                old=owned_path(w,output/e['trial_id'],w/'collection')
                new=owned_path(w,destination/e['trial_id'],w/'active')
                if old.exists() and not new.exists(): old.rename(new)
                if not new.is_dir():
                    if wave['status']=='VALIDATED': continue
                    raise ValueError('missing raw trial after barrier: '+e['trial_id'])
                e['directory']=str(new); e['wave_id']=wave_id
                if not (new/'compaction.json').exists(): compact_trial(new,e['request'],w/'shared')
                current.append(e)
        dump(target,manifest)
        if wave['status']!='VALIDATED':
            # Admission is based on observed serial worker memory, not core count.
            import psutil
            from native_paper_v3_memory import admission_snapshot
            def host_memory():
                vm=psutil.virtual_memory();return dict(total=int(vm.total),available=int(vm.available))
            admission=admission_snapshot(runtime,host_memory)
            # Independent verification has a different memory profile from
            # planning. Preserve the proven 2 GiB reservation per verifier.
            need=2*GIB; available=.8*admission['absolute_headroom_bytes']
            jobs=min(workers,len(wave['map_ids']),int(available//need))
            wave['validation_ram_admission']=admission
            if jobs<1:
                manifest['status']='PARTIAL_RAM';dump(target,manifest);return manifest
            tasks=[(w,maps[mid],[e for e in current if e['map_id']==mid],1) for mid in wave['map_ids']]
            if jobs==1: groups=list(map(_validate_map_task,tasks))
            else:
                with ProcessPoolExecutor(max_workers=jobs) as pool: groups=list(pool.map(_validate_map_task,tasks))
            rows=[r for group in groups for r in group]
            if any(r.get('errors') or r.get('correctness_errors') for r in rows):
                manifest['status']='FAILED_CORRECTNESS'; dump(target,manifest); return manifest
            wave.update(status='VALIDATED',validation_workers=jobs,validated_trials=len(rows)); dump(target,manifest)
        receipts=[]
        for mid in wave['map_ids']: receipts.append(store.pack_map(wave_id,mid,cleanup=True))
        measured=sum(r['uncompressed_bytes'] for r in receipts)
        wave.update(status='PACKED',shards=[r['archive'] for r in receipts],packed_utc=utc(),
            disk_observed=dict(uncompressed_bytes=measured,archive_bytes=sum(r['archive_bytes'] for r in receipts)))
        raw_budget=max(raw_budget,1.25*measured); pack_budget=max(pack_budget,1.25*measured)
        dump(target,manifest)
    else: manifest['status']='COMPLETE'
    manifest['updated_utc']=utc(); dump(target,manifest)
    if manifest['status']=='COMPLETE' and mode!='main':
        dump(w/'readiness'/(mode+'.json'),smoke_evidence(w,mode))
    return manifest

def plan_summary(workspace=None,profile='main'):
    if workspace:
        w=Path(workspace); check_frozen(w); plan=read_json(w/'plans'/(profile+'.json'))
        entries=plan['entries']; maps=len({e['map_id'] for e in entries}); blocks=len({(e['map_id'],e['planner_seed']) for e in entries})
    else:
        c=default_config()['profiles'][profile]; maps=len(c['families'])*len(c['map_seeds']); blocks=maps*len(c['planner_seeds']); entries=range(blocks*3)
    return dict(protocol_id=PROTOCOL_ID,schema_version=13,dry_run=True,dataset_verified=bool(workspace),
        maps=maps,paired_blocks=blocks,slots=len(entries),planning_budget_s=20. if profile=='main' else 2.)

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('command',choices=('prepare','correctness','run','report','plan'))
    p.add_argument('--workspace',type=Path); p.add_argument('--build-dir',type=Path); p.add_argument('--python',type=Path,default=Path(sys.executable))
    p.add_argument('--profile',choices=('smoke','main')); p.add_argument('--mode',choices=MODES)
    p.add_argument('--allow-main',action='store_true'); p.add_argument('--force-local',action='store_true'); p.add_argument('--resume',action='store_true')
    p.add_argument('--session-elapsed-s',type=float,default=0.)
    p.add_argument('--dry-run',action='store_true',help='plan is always read-only')
    a=p.parse_args()
    if a.command=='plan': result=plan_summary(a.workspace,a.profile or 'main')
    elif a.command=='prepare': result=prepare(a.workspace,a.build_dir,a.python,a.profile)
    elif a.command=='correctness': result=correctness(a.workspace)
    elif a.command=='report':
        from native_paper_v4_analysis import analyze_campaign
        result=analyze_campaign(a.workspace,a.workspace/'reports'/a.mode,mode=a.mode)
    else: result=run(a.workspace,a.mode,allow_main=a.allow_main,force_local=a.force_local,resume=a.resume,session_elapsed_s=a.session_elapsed_s)
    print(json.dumps({k:result[k] for k in ('status','maps','paired_blocks','slots','protocol_id') if k in result}))
    return 0 if result.get('status','PASS') in ('PASS','PASS_LOCAL_WITH_PLATFORM_GAPS','PREPARED','COMPLETE','DONE') else 2

if __name__=='__main__': raise SystemExit(main())
