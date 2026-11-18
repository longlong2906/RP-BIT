"""Small correctness trials, kept outside experimental results."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from experiment_common import ROOT,code_hash,dump,file_hash,load_config,worker_env
from terrain_io import write_map
from independent_verifier import Oracle
from validate_results import classify_success,read_path
from generate_scenarios import generate
from run_experiments import plan


def run(worker,directory,case,planner,*,budget=3.,work=2000,start=(1.,5.),goal=(9.,5.),rectangles=(),height=None,cloud_handshake=False):
    directory.mkdir();map_path=directory/"map.t25d"
    fn=height or (lambda x,y:0.)
    heights=[[fn(i,j) for i in range(11)] for j in range(11)]
    write_map(map_path,heights,radius=.1,margin=0,slope=70,rectangles=rectangles,metadata={"fixture":True,"case":case})
    request={"map_path":str(map_path),"planner_id":planner,"planner_seed":771,"start":start,"goal":goal,
        "planning_budget_s":budget,"samples_per_batch":40,"rewire_factor":1.1,"sample_attempts":100,"progress_interval_s":.05}
    if planner=="sem_rrtstar_common_v1":
        request.update(sem_range_m=1.0,sem_neighbor_radius_m=3.0)
    if work:request["work_iterations"]=work
    if cloud_handshake:request["cloud_phase_handshake"]=True
    dump(directory/"request.json",request)
    result=subprocess.run([str(worker),str(directory/"request.json"),str(directory)],capture_output=True,text=True,env=worker_env(),timeout=15)
    (directory/"stdout.log").write_text(result.stdout);(directory/"stderr.log").write_text(result.stderr)
    assert result.returncode==0,(case,result.stderr)
    record=json.loads((directory/"worker.json").read_text())
    assert "NaN" not in (directory/"worker.json").read_text() and "Infinity" not in (directory/"worker.json").read_text()
    # The requested description must match ordinary RRT*'s uniform sampler.
    # Also cover invalid-query early returns, which do not reach planner setup.
    if planner=="rrtstar":
        assert record["requested_policy"]=="d_phi/D",(case,record)
        effective=record["effective_policy"]
        assert effective["sample_domain"]=="D",(case,effective)
        assert effective["informed_sampling"] is False,(case,effective)
        assert effective["pruning"] is False,(case,effective)
    if planner in ("abitstar","aitstar"):
        assert record["requested_policy"]=="d_xy/E_xy",(case,record)
        effective=record["effective_policy"]
        assert effective["edge_lower_bound"]=="d_xy" and effective["sample_domain"]=="E_xy",(case,effective)
        if "sampler_class" in effective:
            assert effective["sampler_class"]=="ompl::base::PathLengthDirectInfSampler",(case,effective)
    if planner=="lazyprmstar_hauser2015":
        assert record["requested_policy"]=="d_xy/D",(case,record)
        effective=record["effective_policy"]
        assert effective["edge_lower_bound"]=="d_xy" and effective["sample_domain"]=="D",(case,effective)
        assert effective["informed_sampling"] is False and effective["pruning"] is False,(case,effective)
        if "sample_attempts" in effective:
            assert effective["sample_attempts"] is None,(case,effective)
            assert effective["sample_attempts_status"]=="NOT_APPLICABLE_ONE_UNIFORM_PROPOSAL_PER_ITERATION",(case,effective)
        if "planner_class" in effective:
            assert effective["planner_class"]=="terrain25d::HauserLazyPRMstar",(case,effective)
            assert effective["algorithm_profile"]=="LazyPRMstar_Hauser_2015_Algorithms_1_2_5_core",(case,effective)
            assert effective["bidirectional_path_precheck"] is False,(case,effective)
            assert effective["section_III_E_roadmap_pruning"] is False,(case,effective)
    if planner=="sem_rrtstar_common_v1":
        assert record["requested_policy"]=="d_xy/D_then_E_xy",(case,record)
        effective=record["effective_policy"]
        assert effective["sample_domain"]=="D_then_E_xy",(case,effective)
        if "planner_class" in effective:  # Invalid-query records stop before planner setup.
            assert effective["planner_class"]=="terrain25d::SemRRTstarCommon",(case,effective)
            assert effective["official_sem_rrtstar_implementation"] is False,(case,effective)
            assert effective["tree_direction"]=="goal_to_start",(case,effective)
            assert effective["solution_output_order"]=="start_to_goal",(case,effective)
            assert effective["sample_domain_before_incumbent"]=="uniform_D",(case,effective)
            assert effective["sample_domain_after_incumbent"]=="d_xy_E_xy",(case,effective)
            assert effective["steer_range_m"]==1.0 and effective["near_radius_m"]==3.0,(case,effective)
            assert effective["risk_assessment_enabled"] is False,(case,effective)
            assert effective["turn_constraint_enabled"] is False,(case,effective)
            assert effective["rolling_constraint_enabled"] is False,(case,effective)
    oracle=Oracle(map_path)
    if record["path_available"]:
        xy=read_path(directory/"path_xy.csv")
        endpoint=goal if record["planner_status"]=="EXACT_SOLUTION" else xy[-1]
        check=oracle.validate_path(xy,start,endpoint,record["best_cost_terrain_m"],read_path(directory/"path_surface.csv"))
        assert check["valid"],(case,check)
        events=[json.loads(line) for line in (directory/"progress.jsonl").read_text().splitlines()]
        events=[e for e in events if e["event_type"]=="improvement"]
        if record["planner_status"]=="EXACT_SOLUTION":
            assert events,"callback not invoked"
            assert events[0]["elapsed_s"]==record["first_solution_time_s"]
        else:assert not events,"approximate path emitted exact incumbent event"
        for event in events:
            snapshot=directory/"snapshots"/event["snapshot_id"]
            valid=oracle.validate_path(read_path(snapshot/"path_xy.csv"),start,goal,event["best_cost_terrain_m"],read_path(snapshot/"path_surface.csv"))
            assert valid["valid"],(case,valid)
    return record

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--worker",type=Path,required=True);args=parser.parse_args()
    fixture_root=args.worker.resolve().parent/"correctness_trials"
    fixture_root.mkdir(parents=True,exist_ok=True)
    # Each invocation gets a unique directory, preserving earlier failures and raw logs.
    directory=Path(tempfile.mkdtemp(prefix="run_",dir=fixture_root))
    planners=("bitstar_a0","bitstar_a3","rrtstar","informed_rrtstar",
              "abitstar","aitstar","lazyprmstar_hauser2015","sem_rrtstar_common_v1")
    for planner in planners:
        base_work=500 if planner in ("abitstar","aitstar","lazyprmstar_hauser2015") else 2000
        for case,fn in [("flat",lambda x,y:3.),("tilted",lambda x,y:.3*x+.2*y)]:
            result=run(args.worker,directory/f"{planner}_{case}",case,planner,height=fn,work=base_work)
            assert result["planner_status"]=="EXACT_SOLUTION",result
            expected=math.hypot(8.,fn(9,5)-fn(1,5))
            assert expected-1e-8<=result["best_cost_terrain_m"]<=expected*1.05,result
        # SEM's all-neighbor surface costs grow with the tree. A smaller fixed
        # work count exercises rewiring while leaving ample wall-time headroom.
        determinism_work=500 if planner=="sem_rrtstar_common_v1" else base_work
        first=run(args.worker,directory/f"{planner}_det1","determinism",planner,
                  height=lambda x,y:.5*(4-abs(x-5)),work=determinism_work,budget=10.)
        second=run(args.worker,directory/f"{planner}_det2","determinism",planner,
                   height=lambda x,y:.5*(4-abs(x-5)),work=determinism_work,budget=10.)
        # Wall-clock truncation is not a deterministic work-budget observation.
        assert first["termination_reason"]==second["termination_reason"]=="WORK_BUDGET", (planner,first,second)
        assert first["iterations"]==second["iterations"]==determinism_work, (planner,first,second)
        assert first["counters"]==second["counters"],"work budget counters differ"
        assert first["best_cost_terrain_m"]==second["best_cost_terrain_m"],"work budget cost differs"
        assert (directory/f"{planner}_det1/path_xy.csv").read_bytes()==(directory/f"{planner}_det2/path_xy.csv").read_bytes()
        for case,kwargs in [("timeout",{"budget":.02,"work":0,"rectangles":[(4.999,0,5.001,10)]}),
            ("empty_free",{"rectangles":[(0,0,10,10)]}),("invalid_goal",{"goal":(-1.,5.)}),
            ("start_goal",{"start":(5.,5.),"goal":(5.,5.),"work":base_work}),
            ("near_goal",{"start":(5.,5.),"goal":(5.+1e-10,5.),"work":base_work})]:
            record=run(args.worker,directory/f"{planner}_{case}",case,planner,**kwargs)
            if case=="timeout":assert record["execution_status"]=="TIMEOUT" and record["planner_status"]!="EXACT_SOLUTION"
            if case=="empty_free":assert record["planner_status"]=="INVALID_START"
            if case=="invalid_goal":assert record["planner_status"]=="INVALID_GOAL"
            if case in ("start_goal","near_goal"):assert record["planner_status"]=="EXACT_SOLUTION"
        detour=run(args.worker,directory/f"{planner}_blocked_direct","blocked_direct",planner,
            rectangles=[(4.,3.,6.,7.)],work=1500 if planner in ("abitstar","aitstar","lazyprmstar_hauser2015") else 4000)
        assert detour["planner_status"]=="EXACT_SOLUTION",detour
        xy=read_path(directory/f"{planner}_blocked_direct/path_xy.csv")
        assert len(xy)>2,"blocked edge incorrectly accepted as two-point solution"
    # Deliberately fabricated parser cases stay in tests/fixtures, never results.
    cases=json.loads((ROOT/"tests/fixtures/status_cases.json").read_text())
    for case in cases["cases"]:assert classify_success(case["trial"],case["validation"],case["events"])==case["expected"]
    config=load_config(ROOT/"configs/experiments.yaml")
    first=generate(config,output=directory/"maps1");second=generate(config,output=directory/"maps2")
    assert [m["sha256"] for m in first["maps"]]==[m["sha256"] for m in second["maps"]]
    assert len(plan(config,"smoke",first))==24
    # Unknown fields must fail, rather than silently producing a mislabeled variant.
    import yaml
    bad=config.copy();bad["extra_unknown_option"]=True
    bad_path=directory/"invalid.yaml";bad_path.write_text(yaml.safe_dump(bad))
    try:load_config(bad_path)
    except ValueError:pass
    else:raise AssertionError("unknown field accepted")
    # The gate covers the runner protocol and independent verifier index, too.
    subprocess.run([sys.executable,str(ROOT/"tests/test_runner.py")],check=True,env=worker_env())
    dump(args.worker.resolve().parent/"correctness_gate.json",{"worker_sha256":file_hash(args.worker),"code_sha256":code_hash(),"pipeline_passed":True,"evidence_directory":str(directory)})
    print(f"T14-T17 PASS: deterministic inputs/work budget; timeout/invalid/exact singleton; strict JSON and approximate-result rejection. Logs: {directory}")

if __name__=="__main__":main()
