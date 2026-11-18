import argparse
import csv
from pathlib import Path
import time
import math

from experiment_common import dump,file_hash,read_json
from independent_verifier import Oracle
from run_experiments import refresh_manifest,summarize


def read_path(path):
    with Path(path).open(newline="",encoding="utf-8") as handle:
        return [[float(value) for value in row] for row in list(csv.reader(handle))[1:]]

def classify_success(trial,validation,events):
    return bool(trial.get("execution_status")=="COMPLETED" and trial.get("planner_status")=="EXACT_SOLUTION" and
        validation.get("valid") and events and events[0]["elapsed_s"]<=trial["time_budget_s"] and
        all(event.get("independent_valid") for event in events))

def validate_trial(directory,oracle=None):
    directory=Path(directory);trial=read_json(directory/"trial.json");began=time.perf_counter()
    if not (directory/"request.json").exists():
        if trial.get("path_available"):raise ValueError("claimed path without request")
        trial.update(success=False,independent_valid=None,validation_status="NO_REQUEST_INTERRUPTED_OR_BLOCKED")
        dump(directory/"trial.json",trial);return trial
    request=read_json(directory/"request.json")
    if file_hash(request["map_path"])!=trial["map_hash"]:raise ValueError("map hash changed")
    oracle=oracle or Oracle(request["map_path"])
    traversability=request.get("traversability")
    if traversability is not None:
        import json
        progress_file=directory/"progress.jsonl"
        has_snapshots=progress_file.exists() and any(json.loads(line).get("event_type")=="improvement"
            for line in progress_file.read_text().splitlines() if line.strip())
        observed_policy=(trial.get("path_available") or trial.get("effective_traversability") is not None or
                         trial.get("objective_value") is not None or has_snapshots)
        if observed_policy:
            if trial.get("objective_type")!="surface_length_traversability":raise ValueError("traversability objective type mismatch")
            for key in ("requested_traversability","effective_traversability"):
                if trial.get(key)!=traversability:raise ValueError("traversability policy mismatch: "+key)
        else:
            trial["effective_traversability"]=None
            trial["effective_traversability_unavailable_reason"]="worker_policy_not_observed_before_failure_or_no_output"
    validation={"valid":False,"errors":["no_path"],"cost_terrain_m":None}
    if trial.get("path_available"):
        try:
            xy=read_path(directory/"path_xy.csv")
            endpoint=request["goal"] if trial.get("planner_status")=="EXACT_SOLUTION" else xy[-1]
            validation=oracle.validate_path(xy,request["start"],endpoint,trial.get("best_cost_terrain_m"),read_path(directory/"path_surface.csv"),
                traversability=traversability,claimed_metrics=trial if traversability is not None else None)
            validation["exact_goal_reached"]=bool(xy and math.dist(xy[-1],request["goal"])<=1e-9)
        except (OSError,ValueError,IndexError) as error:validation["errors"]=["missing_or_malformed_path: "+str(error)]
    progress_path=directory/"progress.jsonl"
    records=[];improvements=[];snapshots=[];certificate_snapshots={}
    if progress_path.exists():
        import json
        records=[json.loads(line,parse_constant=lambda value:(_ for _ in ()).throw(ValueError("nonfinite progress: "+value))) for line in progress_path.read_text().splitlines() if line.strip()]
        previous=float("inf");previous_time=-math.inf;seen=set()
        for event in records:
            if event["event_type"]!="improvement":continue
            snapshot=directory/"snapshots"/event["snapshot_id"]
            snapshot_xy=[]
            try:
                snapshot_xy=read_path(snapshot/"path_xy.csv")
                checked=oracle.validate_path(snapshot_xy,request["start"],request["goal"],
                    event["best_cost_terrain_m"],read_path(snapshot/"path_surface.csv"),
                    traversability=traversability,claimed_metrics=event if traversability is not None else None)
            except (OSError,ValueError,IndexError) as error:checked={"valid":False,"errors":["missing_or_malformed_snapshot: "+str(error)]}
            value=event.get("objective_value") if traversability is not None else event["best_cost_terrain_m"]
            if not isinstance(value,(int,float)) or not math.isfinite(value) or value>previous+1e-8:
                checked["valid"]=False;checked["errors"].append("increasing_or_invalid_objective_incumbent" if traversability is not None else "increasing_incumbent")
            if event["elapsed_s"]<previous_time or event["elapsed_s"]<0:checked["valid"]=False;checked["errors"].append("nonmonotone_event_time")
            if event["snapshot_id"] in seen:checked["valid"]=False;checked["errors"].append("reused_snapshot_id")
            seen.add(event["snapshot_id"]);previous_time=event["elapsed_s"]
            previous=value if isinstance(value,(int,float)) else previous
            checked["snapshot_id"]=event["snapshot_id"]
            checked["xy_sha256"]=file_hash(snapshot/"path_xy.csv") if (snapshot/"path_xy.csv").exists() else None
            checked["surface_sha256"]=file_hash(snapshot/"path_surface.csv") if (snapshot/"path_surface.csv").exists() else None
            dump(snapshot/"validation.json",checked);snapshots.append(checked)
            if "exact_flat_certificate" in request or "exact_certificate" in trial:
                certificate_snapshots[event["snapshot_id"]] = {
                    "valid": checked["valid"], "path_xy": snapshot_xy}
            event["independent_valid"]=checked["valid"]
            if traversability is not None:
                for key in ("minimum_obstacle_footprint_clearance_m","obstacle_clearance_status","maximum_swept_footprint_slope_deg"):
                    event[key]=checked.get(key)
            improvements.append(event)
    validation["snapshots"]=snapshots
    if "exact_flat_certificate" in request or "exact_certificate" in trial:
        from flat_direct_certificate import verify_claim
        certificate_check=verify_claim(oracle,request,trial,certificate_snapshots)
        validation["exact_certificate_validation"]=certificate_check
        trial["exact_certificate_independent_valid"]=certificate_check["valid"]
        if not certificate_check["valid"]:
            validation["valid"]=False
            validation["errors"].extend(certificate_check["errors"])
    dump(directory/"validation.json",validation)
    dump(directory/"validated_events.json",improvements)
    trial["independent_valid"]=validation["valid"] if trial.get("path_available") else None
    trial["independently_recomputed_cost_m"]=validation.get("cost_terrain_m")
    trial["length_xy_m"]=validation.get("length_xy_m")
    trial["cost_abs_error_m"]=validation.get("cost_abs_error_m")
    trial["minimum_footprint_clearance_m"]=validation.get("minimum_footprint_clearance_m")
    trial["maximum_slope_deg"]=validation.get("maximum_slope_deg")
    if traversability is not None:
        trial["independently_recomputed_objective_value"]=validation.get("objective_value")
        for key in ("minimum_obstacle_footprint_clearance_m","obstacle_clearance_status","maximum_swept_footprint_slope_deg"):
            trial[key]=validation.get(key)
        trial["first_solution_time_semantics"]="native_callback_timestamp_of_snapshot_validated_offline; excludes offline_validation_latency"
    trial["success"]=classify_success(trial,validation,improvements)
    trial["validation_wall_s"]=time.perf_counter()-began
    trial["snapshots_valid"]=all(s["valid"] for s in snapshots) if snapshots else (False if trial.get("planner_status")=="EXACT_SOLUTION" and trial.get("path_available") else None)
    dump(directory/"trial.json",trial)
    return trial

def validate_experiment(directory):
    directory=Path(directory);manifest=read_json(directory/"manifest.json")
    if directory.resolve()==(Path(__file__).resolve().parents[1]/"results/smoke").resolve():
        raise ValueError("completed smoke is immutable; validate a copy in a new audit directory")
    groups={}
    for entry in manifest["trials"]:groups.setdefault(entry["map_path"],[]).append(entry)
    for map_path,entries in groups.items():
        oracle=None
        for entry in entries:
            path=directory/entry["trial_id"]
            if not (path/"trial.json").exists():continue
            if (path/"request.json").exists() and oracle is None:oracle=Oracle(map_path)
            trial=validate_trial(path,oracle)
            print(f'validated {entry["trial_id"]}: success={trial["success"]}',flush=True)
            if trial.get("snapshots_valid") is False:
                refresh_manifest(directory,manifest,validated=True)
                dump(directory/"validation_summary.json",summarize(directory,manifest))
                raise ValueError("invalid incumbent snapshot: "+entry["trial_id"])
    refresh_manifest(directory,manifest,validated=True)
    summary=summarize(directory,manifest)
    dump(directory/"validation_summary.json",summary);print(summary)
    return summary

def main():
    p=argparse.ArgumentParser();p.add_argument("--input",required=True);args=p.parse_args()
    summary=validate_experiment(args.input)
    return 1 if summary["invalid"] else 0

if __name__=="__main__":raise SystemExit(main())
