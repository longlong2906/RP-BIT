"""Real backends under the actual serial/four-lane scheduler, never fake workers.

The small analytic map and retained artifacts are correctness fixtures. They are
not scientific benchmark results and do not rank latency or solution quality.
"""
import argparse
import copy
import csv
import json
import math
import os
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from experiment_common import dump,worker_env
from independent_verifier import Oracle
from terrain_io import write_map
from terrain_compare_protocol import request_for,sha256,DEFAULT_TRAVERSABILITY
from terrain_compare_scheduler import run_campaign,collect_reservations,read_json,detect_runtime,MIB


def coordinates(path):
    with Path(path).open(newline="",encoding="utf-8") as f:
        return [[float(v) for v in row] for row in list(csv.reader(f))[1:]]


def trace(directory,oracle,request):
    worker=read_json(directory/"worker.json")
    assert worker.get("planner_status",worker.get("status"))=="EXACT_SOLUTION",worker
    events=worker.get("snapshots")
    if events is None:
        events=[json.loads(line) for line in (directory/"progress.jsonl").read_text().splitlines()
                if json.loads(line).get("event_type")=="improvement"]
    assert events,"fixed-work correctness fixture must exercise actual incumbent paths"
    output=[]
    fields=("cost_terrain_m","length_xy_m","risk_slope_integral_m","risk_clearance_integral_m","risk_integral_m")
    for event in events:
        folder=directory/"snapshots"/event["snapshot_id"]
        xy=event.get("path_xy")
        if xy is None:xy=coordinates(folder/"path_xy.csv")
        raw=event.get("raw_path_xyz")
        if raw is None:raw=coordinates(folder/"path_surface.csv")
        checked=oracle.validate_path(xy,request["start"],request["goal"],traversability=DEFAULT_TRAVERSABILITY)
        assert checked["valid"],checked
        for key in fields:
            assert math.isclose(event[key],checked[key],rel_tol=1e-8,abs_tol=1e-8),(key,event,checked)
        common=event.get("evaluation_objective_value",event.get("objective_value"))
        assert math.isclose(common,checked["objective_value"],rel_tol=1e-8,abs_tol=1e-8)
        native=event.get("planner_objective_value",event.get("objective_value"))
        output.append(dict(path_xy=xy,raw_xyz=raw,planner_objective=native,
                           metrics={key:event[key] for key in fields},evaluation_objective=common))
    if request["backend"]=="sgp":
        model=worker["native_model"]
        assert model["training_steps"]==1 and model["training_observations"]==441
        assert model["thread_count"]==1 and model["interop_thread_count"]==1
        assert worker["counters"]["iterations"]==request["work_iterations"]
        model_signature={key:value for key,value in model.items() if key!="conversion"}
    elif request["backend"]=="putn":
        stats=worker["counters"]["putn_statistics"]
        assert stats[0]==request["work_iterations"] and stats[1]>0 and stats[4]>0 and stats[10]==1
        model_signature={"work_statistics":stats[:8],"tree_consistent":stats[10]}
    else:
        assert worker["termination_reason"]=="WORK_BUDGET"
        assert worker["iterations"]==request["work_iterations"]
        assert worker["counters"]["traversability_motion_cost_calls"]>0
        model_signature=worker["effective_policy"]
    return dict(events=output,model=model_signature,final_xy=coordinates(directory/"path_xy.csv"))


def backend_registry(worker, sgp_python, controller_python=None):
    """Keep venv aliases for execution; resolving them loses site-packages."""
    registry={name:{"argv":[str(worker)],"cwd":str(ROOT)} for name in ("a3","sem")}
    controller_python = sys.executable if controller_python is None else controller_python
    for backend, interpreter in (("putn", controller_python), ("sgp", sgp_python)):
        registry[backend]={"argv":[os.path.abspath(interpreter),
            str(ROOT/"tools/terrain_compare_external_worker.py")],"cwd":str(ROOT)}
    return registry


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--worker",type=Path,required=True)
    parser.add_argument("--library",type=Path,required=True)
    parser.add_argument("--sgp-python",type=Path,required=True)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    worker,library=(p.resolve() for p in (args.worker,args.library))
    for p in (worker,library,args.sgp_python):assert p.is_file(),p
    if args.output:
        root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    else:
        parent=worker.parent/"terrain_compare_fixed_work";parent.mkdir(exist_ok=True)
        root=Path(tempfile.mkdtemp(prefix="run_",dir=parent))
    print("Fixed-work correctness artifacts:", root, flush=True)
    runtime=detect_runtime()
    if runtime["cpu_capacity"]<2:
        dump(root/"result.json",{"status":"BLOCKED","reason":"requires at least two actual eligible CPUs","runtime":runtime})
        raise RuntimeError("fixed-work parallel test requires at least two eligible CPUs")
    map_path=root/"analytic_flat.t25d"
    write_map(map_path,[[0.]*21 for _ in range(21)],spacing=(.2,.2),metadata={"test_fixture":True,"purpose":"real_fixed_work_scheduler"})
    entries=[]
    query=dict(start=[1.2,2.],goal=[2.8,2.])
    blocks=[(i,seed,"common",["a3","sem","putn","sgp"]) for i,seed in enumerate((11,22,33,44))]
    blocks.append((4,11,"own",["putn","sgp"]))
    for block,seed,track,backends in blocks:
        # Same complete block stays on one lane; rotation avoids assigning a
        # particular backend permanently to a particular processor.
        n=block%len(backends);backends=backends[n:]+backends[:n]
        for backend in backends:
            request=request_for(backend,track,query,map_path,seed,10.)
            request.update(work_iterations=100,exact_flat_certificate=False)
            if backend in ("putn","sgp"):
                request["bridge_library"]=str(library)
                if os.name=="nt":request["library_search_paths"]=[str(library.parent),"C:/mingw64/bin"]
            entries.append(dict(trial_id=f"{track}_s{seed}_{backend}",block_order=block,planned_order=len(entries),
                backend=backend,track=track,family="analytic_flat_fixture",map_id="analytic_flat",
                map_sha256=sha256(map_path),planner_seed=seed,execution_status="PLANNED",request=request))
    plan=root/"plan.json";dump(plan,dict(profile="correctness",entries=entries,test_fixture=True))
    registry=backend_registry(worker,args.sgp_python)
    old_env=os.environ.copy()
    try:
        # Process-local loader paths only. Scheduler controls numerical threads
        # and affinity before either C++ or Python backend starts.
        os.environ.update(worker_env())
        serial=run_campaign(plan,registry,root/"serial",max_workers=1,
                            reservations={b:512*MIB for b in registry},force_local=True)
        assert serial["counts"]=={"COMPLETED":len(entries)},("serial",serial["counts"],"artifacts",str(root))
        parallel=run_campaign(plan,registry,root/"parallel",max_workers=4,
                              reservations=collect_reservations(serial),force_local=True)
        assert parallel["counts"]=={"COMPLETED":len(entries)},("parallel",parallel["counts"],"artifacts",str(root))
    finally:
        os.environ.clear();os.environ.update(old_env)
    peak=max(x["concurrency"] for x in parallel["concurrency_timeline"])
    assert peak>=2,("parallel regime never overlapped actual workers",peak)
    oracle=Oracle(map_path);comparisons=[]
    for entry in entries:
        name=entry["trial_id"]
        left=trace(root/"serial"/name,oracle,entry["request"])
        right=trace(root/"parallel"/name,oracle,entry["request"])
        assert left==right,("real fixed-work trace changed under concurrency",name,left,right)
        launch=read_json(root/"parallel"/name/"launch.json")
        assert len(launch["cpu_affinity"])==1
        assert all(v=="1" for v in launch["thread_environment"].values())
        comparisons.append(dict(trial_id=name,snapshots=len(left["events"]),full_trace_equal=True,independent_valid=True))
    result=dict(status="PASS",test_fixture=True,real_backend_trials=2*len(entries),
                pairs=len(comparisons),actual_parallel_peak=peak,maximum_configured_workers=4,
                compared="all incumbent XY/raw paths and L/Rs/Rc/R/J/native tuple, final path, deterministic model/work metadata; clocks excluded",
                comparisons=comparisons)
    dump(root/"result.json",result)
    print(json.dumps({"status":"PASS","pairs":len(comparisons),"actual_parallel_peak":peak,"artifacts":str(root)}))


if __name__=="__main__":main()
