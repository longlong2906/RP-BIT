"""Actual SEM common-cost hooks and independent tuple validation, not post-hoc cost only."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from experiment_common import dump,worker_env
from independent_verifier import Oracle
from terrain_io import write_map
from terrain_compare_protocol import request_for


def main():
    p=argparse.ArgumentParser();p.add_argument("--worker",type=Path,required=True);a=p.parse_args()
    worker=a.worker.resolve();parent=worker.parent/"sem_traversability_tests";parent.mkdir(exist_ok=True)
    root=Path(tempfile.mkdtemp(prefix="run_",dir=parent));map_path=root/"tilted_obstacle.t25d"
    write_map(map_path,[[.12*i+.08*j for i in range(11)] for j in range(11)],
              rectangles=[(4.2,3.,5.8,7.)],metadata={"test_fixture":True})
    oracle=Oracle(map_path)
    for lam in (0.,1.):
        output=root/f"lambda_{lam:g}";output.mkdir()
        request=request_for("sem","common",dict(start=[1.,5.],goal=[9.,5.]),map_path,771,15.)
        request.update(cloud_phase_handshake=False,work_iterations=500)
        request["traversability"]["lambda"]=lam
        dump(output/"request.json",request)
        command=[str(worker),str(output/"request.json"),str(output)]
        result=subprocess.run(command,capture_output=True,text=True,env=worker_env(),timeout=35)
        (output/"stdout.log").write_text(result.stdout);(output/"stderr.log").write_text(result.stderr)
        dump(output/"command.json",dict(argv=command,exit_code=result.returncode))
        assert result.returncode==0,(result.returncode,result.stderr)
        record=json.loads((output/"worker.json").read_text())
        assert record["planner_status"]=="EXACT_SOLUTION",record
        assert record["termination_reason"]=="WORK_BUDGET",record
        assert record["effective_policy"]["informed_cost_bound"]=="objective_value_J"
        assert record["effective_policy"]["motion_objective"]=="surface_length_traversability"
        assert record["effective_policy"]["nearest_neighbor_metric"]=="euclidean_xy"
        assert record["counters"]["traversability_motion_cost_calls"]>0
        assert record["planner_specific_counters"]["rewires"]>0,record["planner_specific_counters"]
        events=[json.loads(x) for x in (output/"progress.jsonl").read_text().splitlines()]
        improvements=[e for e in events if e["event_type"]=="improvement"]
        assert improvements
        previous=math.inf
        for event in improvements:
            import csv
            with (output/"snapshots"/event["snapshot_id"]/"path_xy.csv").open() as f:
                rows=list(csv.reader(f));xy=[[float(v) for v in row] for row in rows[1:]]
            checked=oracle.validate_path(xy,request["start"],request["goal"],traversability=request["traversability"])
            assert checked["valid"],checked
            for key in ("objective_value","cost_terrain_m","risk_slope_integral_m","risk_clearance_integral_m"):
                assert math.isclose(event[key],checked[key],abs_tol=1e-8,rel_tol=1e-8),(key,event,checked)
            assert event["objective_value"]<=previous+1e-8
            previous=event["objective_value"]
        if lam==0:assert math.isclose(record["objective_value"],record["cost_terrain_m"],abs_tol=1e-8)
        else:assert record["objective_value"]>record["cost_terrain_m"]
    print("PASS SEM real fixed-work cost/rewire hooks and independent L/R/J snapshots")


if __name__=="__main__":main()
