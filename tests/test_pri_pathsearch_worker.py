"""Real persistent BIT* replay; fixtures are explicitly synthetic, not benchmark data."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'tools'), str(ROOT/'tests')]
from corridor_v4_execution import THREAD_ENV
for key in THREAD_ENV:
    os.environ[key] = '1'
import numpy as np
from native_point_cloud import generate_cloud
from safe_length_field import build_field_file, file_hash
from terrain_io import write_map
from pri_pathsearch_validation import validate_trial
from native_paper_oracle import NativePaperOracle
from native_paper_v3_sem_evaluator import SharedSEMField


def run(worker, directory):
    terrain = directory/'synthetic_pathsearch.t25d'
    height = lambda x,y: .45*np.exp(-((x-3.)**2+(y-3.)**2)/1.8)
    write_map(terrain, [[float(height(i*.2,j*.2)) for i in range(31)] for j in range(31)],
              spacing=(.2,.2), rectangles=[], metadata=dict(test_fixture=True,purpose='synthetic_pathsearch_replay'))
    cloud_path=directory/'synthetic_cloud.npy'
    np.save(cloud_path,generate_cloud(terrain),allow_pickle=False)
    field_path=directory/'synthetic_field.txt'
    metadata=build_field_file(cloud_path,terrain,[0.,0.,6.,6.],field_path)
    info=dict(map_path=str(terrain),map_sha256=file_hash(terrain),cloud_path=str(cloud_path),
              cloud_sha256=file_hash(cloud_path),bounds=[0.,0.,6.,6.])
    base=dict(map_path=str(terrain),planner_id='bitstar_a3',variant_id='synthetic_pathsearch',
        planner_seed=771,start=[.8,3.],goal=[5.2,3.],planning_budget_s=120.,
        test_only_fixed_work=True,pathsearch_test_work=dict(anchor_iterations=1200,iterations_per_slice=300,
            optimizer_proposals_per_slice=96,cycles=3),
        samples_per_batch=100,batch_after_first_solution=200,rewire_factor=1.1,sample_attempts=100,
        state_cache=True,scalar_cost=True,sampler_query_cache=True,collision_index=True,exact_flat_certificate=False,
        objective_type='surface_length',pri_pathsearch={'mode':'full'},
        safe_length=dict(policy_id='a3t_safe_length_v1',field_path=str(field_path),field_sha256=metadata['field_sha256'],
                         risk_cap=40.,slope_cap_deg=35.,atlas_sampling=True))
    env=os.environ.copy()
    env['PATH']=os.pathsep.join((str(ROOT/'.deps/install/lib'),'C:/mingw64/bin',env.get('PATH','')))

    def execute(name,request,expected=0):
        out=directory/name;out.mkdir()
        (out/'request.json').write_text(json.dumps(request),encoding='utf-8')
        command=[str(worker),str(out/'request.json'),str(out)]
        process=subprocess.run(command,env=env,capture_output=True,text=True,timeout=150)
        (out/'stdout.log').write_text(process.stdout,encoding='utf-8')
        (out/'stderr.log').write_text(process.stderr,encoding='utf-8')
        (out/'command.json').write_text(json.dumps(dict(argv=command,exit_code=process.returncode)),encoding='utf-8')
        assert process.returncode==expected,(name,process.returncode,process.stderr)
        if expected:return None
        return out,json.loads((out/'worker.json').read_text())

    requests=[]
    for mode in ('length','shortcut','full'):
        value=copy.deepcopy(base);value['pri_pathsearch']['mode']=mode;requests.append(value)
    serial=[execute('serial_'+str(i),value) for i,value in enumerate(requests)]
    with ThreadPoolExecutor(max_workers=3) as pool:
        parallel=list(pool.map(lambda pair:execute('parallel_'+str(pair[0]),pair[1]),enumerate(requests)))
    oracle=NativePaperOracle(terrain)
    field=SharedSEMField(np.load(cloud_path,allow_pickle=False),info['bounds'])
    snapshot_count=0
    for request,(left,a),(right,b) in zip(requests,serial,parallel):
        assert a['planner_status']==b['planner_status']=='EXACT_SOLUTION'
        assert a['planner_instances']==b['planner_instances']==1
        assert a['selected_snapshot_id']==b['selected_snapshot_id']
        assert a['selectors']==b['selectors'],'fixed-work changed selectors/metrics'
        assert (left/'path_xy.csv').read_bytes()==(right/'path_xy.csv').read_bytes()
        assert a['counters']==b['counters'],'fixed-work changed BIT* counters'
        assert a['search_counters']==b['search_counters'],'fixed-work changed optimizer counters'
        for key in ('motion_cost_calls','motion_heuristic_calls','cost_to_go_calls','sampler_allocations'):
            assert a['counters'][key]>0,key
        assert a['effective_policy']['motion_objective']=='surface_length'
        assert a['effective_policy']['external_length_cap_in_sampler'] is False
        aa=[json.loads(line) for line in (left/'pathsearch_candidates.jsonl').read_text().splitlines()]
        bb=[json.loads(line) for line in (right/'pathsearch_candidates.jsonl').read_text().splitlines()]
        assert len(aa)==len(bb)
        for x,y in zip(aa,bb):
            for key in ('snapshot_id','candidate_id','parent_ids','proposal_id','move_type','anchor_phase','pool_admitted',
                        'published_alphas','reason','feasible','cost_terrain_m','pri','risk_sum','risk_count','path_xy'):
                assert x[key]==y[key],(request['pri_pathsearch'],x['snapshot_id'],key)
        mode=request['pri_pathsearch']['mode'];counts=a['search_counters']
        if mode=='length':assert counts['proposals']==0
        elif mode=='shortcut':
            assert counts['shortcut_proposals']>0 and counts['deform_proposals']==counts['splice_proposals']==0
        else:
            assert all(counts[key]>0 for key in ('shortcut_proposals','deform_proposals','splice_proposals'))
        verdict=validate_trial(left,request,info,dict(status='COMPLETED',field_build_s=0.),oracle=oracle,field=field)
        assert verdict['success'] and not verdict['correctness_errors'],(mode,verdict['correctness_errors'])
        (left/'independent_full_verdict.json').write_text(json.dumps(verdict),encoding='utf-8')
        snapshot_count+=len(aa)

    invalid=[]
    for key,value in (('unknown',1),('mode','fake')):
        req=copy.deepcopy(base);req['pri_pathsearch'][key]=value;invalid.append(req)
    req=copy.deepcopy(base);req['safe_length']['risk_cap']=39.;invalid.append(req)
    req=copy.deepcopy(base);req['sem_exposure']={'lambda':1.};invalid.append(req)
    req=copy.deepcopy(base);req.pop('test_only_fixed_work');invalid.append(req)
    req=copy.deepcopy(base);req['pathsearch_test_work']['cycles']=0;invalid.append(req)
    req=copy.deepcopy(base);req['planning_budget_s']=0.;invalid.append(req)
    req=copy.deepcopy(base);req['pri_portfolio']={'length_cap_ratio':1.};invalid.append(req)
    for i,req in enumerate(invalid):execute('rejected_'+str(i),req,expected=2)
    expired=copy.deepcopy(base);expired.pop('pathsearch_test_work');expired['planning_budget_s']=1e-12
    _,record=execute('expired_setup',expired)
    assert not record['path_available'] and all(s['selected_snapshot_id'] is None for s in record['selectors'])
    identity=copy.deepcopy(base);identity['goal']=identity['start']
    out,record=execute('identity_query',identity)
    assert record['path_available'] and record['cost_terrain_m']==0.
    verdict=validate_trial(out,identity,info,dict(status='COMPLETED',field_build_s=0.),oracle=oracle,field=field)
    assert verdict['success'] and not verdict['correctness_errors'],verdict['correctness_errors']
    print(json.dumps(dict(status='PASS',test_fixture=True,real_fixed_work_trials=7,
        serial_parallel_paths_counters_identical=True,independently_validated_candidate_paths=snapshot_count,
        full_validator_trials=4,rejected_requests=len(invalid))))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',required=True,type=Path);args=parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='synthetic_pri_pathsearch_worker_',dir=ROOT/'tests/fixtures') as folder:
        run(args.worker.resolve(),Path(folder))
