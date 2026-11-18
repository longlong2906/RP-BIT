"""Real solver reproducibility control, never part of measured main results."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from experiment_common import dump
from native_paper_v4_protocol import default_config,generate_dataset,plan_trials,BACKENDS
from terrain_compare_scheduler import run_campaign,detect_runtime,GIB

def trace(folder,entry):
    worker=json.loads((folder/entry['trial_id']/'worker.json').read_text())
    values=worker.get('snapshots')
    if values is None:
        values=[json.loads(line) for line in (folder/entry['trial_id']/'progress.jsonl').read_text().splitlines()
                if json.loads(line).get('event_type')=='improvement']
    paths=[]
    for event in values:
        paths.append(dict(path=(folder/entry['trial_id']/'snapshots'/event['snapshot_id']/'path_xy.csv').read_text(),
            cost=event.get('native_objective_value',event.get('best_cost_terrain_m')),
            components=event.get('native_components')))
    return dict(status=worker.get('planner_status',worker.get('status')),paths=paths,
        counters=worker.get('counters'),iterations=worker.get('iterations'),batches=worker.get('batches'),
        batch_transition_count=worker.get('batch_transition_count'),
        field_hash=worker.get('requested_safe_length',{}).get('field_sha256'))

def run(worker,library,output=None):
    runtime=detect_runtime()
    if runtime['cpu_capacity']<4: raise RuntimeError('BLOCKED_RESOURCE: fixed-work requires four CPUs')
    temp=None
    if output is None:
        base=ROOT/'tests/fixtures';base.mkdir(exist_ok=True)
        temp=tempfile.TemporaryDirectory(prefix='synthetic_v4_fixed_work_',dir=base); output=Path(temp.name)
    else: output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    try:
        config=default_config(); dataset=generate_dataset(config,'smoke',output/'data')
        plan=plan_trials(config,'smoke',dataset,output/'data')
        families={'unsafe_short_safe_detour','nodata_thin_obstacle'}
        plan['entries']=[e for e in plan['entries'] if e['family'] in families]
        original=copy.deepcopy(plan['entries'])
        for e in original:
            e['trial_id']=e['trial_id'].replace('__s11__','__s22__')
            e['planner_seed']=22;e['request']['planner_seed']=22
            plan['entries'].append(e)
        blocks={}
        for e in plan['entries']:
            key=(e['map_id'],e['planner_seed'])
            blocks.setdefault(key,len(blocks))
            e['block_order']=blocks[key];e['planned_order']=plan['entries'].index(e)
        for e in plan['entries']:
            r=e['request'];r.update(worker_binary=str(Path(worker).resolve()),bridge_library=str(Path(library).resolve()),
                work_iterations=2000,planning_budget_s=60.)
            if os.name=='nt':r['library_search_paths']=[str(Path(library).resolve().parent),'C:/mingw64/bin']
        plan['planned_slots']=plan['planned_executable']=len(plan['entries'])
        dump(output/'plan.json',plan)
        registry={b:dict(argv=[sys.executable,str(ROOT/'tools/native_paper_v4_worker.py')],cwd=str(ROOT)) for b in BACKENDS}
        serial=run_campaign(output/'plan.json',registry,output/'serial',max_workers=1,
            reservations={b:512*1024**2 for b in BACKENDS},force_local=not sys.platform.startswith('linux'))
        parallel=run_campaign(output/'plan.json',registry,output/'parallel',max_workers=4,
            reservations={b:512*1024**2 for b in BACKENDS},force_local=not sys.platform.startswith('linux'))
        for m in (serial,parallel):
            if any(e['execution_status']!='COMPLETED' for e in m['entries']):
                raise AssertionError('fixed-work process failed: '+str(m['counts']))
        if max(c['concurrency'] for c in parallel['concurrency_timeline'])!=4:
            raise AssertionError('fixed-work parallel test did not reach four lanes')
        for entry in plan['entries']:
            if trace(output/'serial',entry)!=trace(output/'parallel',entry):
                raise AssertionError('fixed-work traces differ: '+entry['trial_id'])
        result=dict(status='PASS',real_trials=2*len(plan['entries']),paired_traces=len(plan['entries']),
            snapshot_timestamps_compared=False,main_trials=0)
        dump(output/'fixed_work_result.json',result)
        return result
    finally:
        if temp is not None: temp.cleanup()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',type=Path,required=True);p.add_argument('--library',type=Path,required=True)
    p.add_argument('--output',type=Path);a=p.parse_args()
    print(json.dumps(run(a.worker,a.library,a.output)))
