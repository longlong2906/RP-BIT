"""Independent raw-map/cloud validation for calibrated A3-T policies.

The graph's bottleneck is a discrete calibration result, never a certificate of
continuous optimality or low PRI. No planner field or edge routine is an oracle.
"""
from __future__ import annotations

from datetime import datetime
import math
from pathlib import Path

import numpy as np

from run_safe_length import _csv_path, _events, read, dump
from safe_length_field import file_hash

ROOT = Path(__file__).resolve().parents[1]
RISK_ROUNDOFF_ULPS = 64


def finite(value):
    return type(value) in (int,float) and math.isfinite(value)


def risk_cap_comparison(risk_max, cap, *, roundoff_ulps=RISK_ROUNDOFF_ULPS):
    """Declared cross-implementation allowance; raw metrics remain untouched.

The independent common SEM <=40 gate is applied separately and remains strict.
This is not a tuning parameter, a risk margin, or a claim covering every input.
"""
    result=dict(raw_risk_max=risk_max,risk_cap=cap,roundoff_ulps=roundoff_ulps,
                tolerance=None,numeric_excess=None,signed_difference=None,
                within_roundoff=False,passed=False)
    if not finite(risk_max) or not finite(cap):
        return result
    tolerance=roundoff_ulps*math.ulp(max(1.,abs(cap),abs(risk_max)))
    difference=risk_max-cap
    excess=max(0.,difference)
    result.update(tolerance=tolerance,numeric_excess=excess,signed_difference=difference,
                  within_roundoff=0.<difference<=tolerance,passed=difference<=tolerance)
    return result


def _graph_endpoint_matches(actual, requested):
    """Graph identity only: allow the same fixed 16-ULP coordinate ties.

    Boost.JSON 1.82 decimal parsing can differ from Python by one ULP. This
    check does not change the requested endpoint used by geometry validation,
    the endpoint-to-grid radius, CSV/inline equality, or any path risk value.
    """
    return (len(actual)==len(requested)==2 and all(
        finite(a) and finite(b) and
        abs(a-b)<=16*math.ulp(max(1.,abs(a),abs(b)))
        for a,b in zip(actual,requested)))


def witness_on_registered_graph(path, start, goal, bounds):
    """Check the fixed lattice independently of C++ graph construction.

The real endpoints may be off-grid. Intermediate vertices must be grid nodes;
only adjacent nodes and the explicitly registered endpoint edges are allowed.
"""
    if not path or any(len(point)!=2 or not all(finite(v) for v in point) for point in path):
        return False
    if len(path)==1:
        return start==goal and _graph_endpoint_matches(path[0],start)
    if not _graph_endpoint_matches(path[0],start) or not _graph_endpoint_matches(path[-1],goal):
        return False
    if len(path)==2:
        return True  # Explicit direct start-goal edge, separately safety checked.
    indices=[]
    for point in path[1:-1]:
        pair=[]
        for axis in (0,1):
            coordinate=point[axis]
            index=round((coordinate-bounds[axis])/.1)
            if not 0<=index<=math.floor((bounds[axis+2]-bounds[axis])/.1):
                return False
            expected=bounds[axis]+index*.1
            if abs(coordinate-expected)>16*math.ulp(max(1.,abs(coordinate),abs(expected))):
                return False
            pair.append(index)
        indices.append(pair)
    for left,right in zip(indices,indices[1:]):
        if max(abs(a-b) for a,b in zip(left,right))!=1:
            return False
    for endpoint,node in ((start,path[1]),(goal,path[-2])):
        magnitude=max(1.,*(abs(v) for v in endpoint),*(abs(v) for v in node),.2)
        if math.dist(endpoint,node)>.2+16*math.ulp(magnitude):
            return False
    return True


def record_calibration_receipt(directory):
    directory = Path(directory)
    worker_file = directory/"worker.json"
    if not worker_file.is_file():
        return None
    worker = read(worker_file)
    if "calibration" not in worker:
        return None
    receipt = dict(worker_sha256=file_hash(worker_file),
                   calibration_status=worker["calibration"].get("status"))
    path = directory/"calibration_witness_xy.csv"
    receipt["witness_sha256"] = file_hash(path) if path.is_file() else None
    target = directory/"calibration_receipt.json"
    if target.exists():
        if read(target) != receipt:
            raise ValueError("immutable calibration receipt mismatch")
    else:
        dump(target, receipt)
    return receipt


def validate_calibration(directory, worker, request, oracle, field):
    errors = []
    calibration = worker.get("calibration", {})
    effective = worker.get("effective_safe_length", {})
    policy = request["safe_length"]
    result = dict(status=calibration.get("status"), independent_valid=False,
                  effective_risk_cap=None, errors=errors)
    for key,value in (("risk_slack",policy["risk_slack"]),("grid_spacing_m",.1),
                      ("endpoint_radius_m",.2),("risk_ceiling",40.),("slope_cap_deg",35.),
                      ("endpoint_radius_tie_ulps",16),
                      ("endpoint_radius_tie_scale","max(1,abs(endpoint_xy),abs(grid_node_xy),radius)"),
                      ("witness_used_as_incumbent",False)):
        if calibration.get(key) != value:
            errors.append("calibration_metadata_mismatch:"+key)
    if not finite(calibration.get("preprocessing_s")) or calibration["preprocessing_s"] < 0:
        errors.append("invalid_calibration_time")
    receipt_file = Path(directory)/"calibration_receipt.json"
    if not receipt_file.is_file():
        errors.append("missing_calibration_receipt")
        receipt = {}
    else:
        receipt = read(receipt_file)
        if receipt.get("worker_sha256") != file_hash(Path(directory)/"worker.json"):
            errors.append("calibration_worker_hash_mismatch")
        if receipt.get("calibration_status") != calibration.get("status"):
            errors.append("calibration_receipt_status_mismatch")
    status = calibration.get("status")
    if status != "CALIBRATED":
        if status not in ("CALIBRATION_NO_CONNECTION","INVALID_START","INVALID_GOAL"):
            errors.append("unknown_calibration_status")
        if (calibration.get("b_graph") is not None or calibration.get("effective_risk_cap") is not None or
                effective.get("risk_cap") is not None or calibration.get("witness_xy") or
                _events(Path(directory),worker) or worker.get("planner_status") == "EXACT_SOLUTION"):
            errors.append("calibration_failure_published_solution_or_bound")
        if worker.get("planner_solve_called") is not False:
            errors.append("planner_called_after_calibration_failure")
        if calibration.get("witness_file") is not None or receipt.get("witness_sha256") is not None:
            errors.append("calibration_failure_has_witness_file")
        if status in ("INVALID_START","INVALID_GOAL"):
            point = request["start" if status == "INVALID_START" else "goal"]
            check = oracle.validate_path([point],point,point,geometric_safety_metrics=True,clearance_metrics=False)
            sem = field.score_path([point])
            if check["valid"] and sem["valid"] and check["maximum_swept_footprint_slope_deg"] <= 35.:
                errors.append("calibration_claimed_invalid_endpoint_is_valid")
        result.update(independent_valid=not errors, no_continuous_infeasibility_claim=True)
        return result
    b,cap,slack = calibration.get("b_graph"),calibration.get("effective_risk_cap"),policy["risk_slack"]
    if (not finite(b) or not 0 <= b <= 40. or not finite(cap) or cap != min(40.,b+slack) or
            effective.get("risk_cap") != cap):
        errors.append("calibration_cap_formula_mismatch")
    witness_file = Path(directory)/"calibration_witness_xy.csv"
    if calibration.get("witness_file") != witness_file.name or not witness_file.is_file():
        errors.append("missing_or_unregistered_calibration_witness")
    else:
        if receipt.get("witness_sha256") != file_hash(witness_file):
            errors.append("calibration_witness_hash_mismatch")
        witness = _csv_path(witness_file)
        if witness != calibration.get("witness_xy"):
            errors.append("calibration_witness_csv_mismatch")
        if not witness_on_registered_graph(witness,request["start"],request["goal"],field.bounds):
            errors.append("calibration_witness_not_on_registered_graph")
        check = oracle.validate_path(witness,request["start"],request["goal"],
            geometric_safety_metrics=True,clearance_metrics=False)
        sem = field.score_path(witness)
        if not check["valid"] or not sem["valid"] or check["maximum_swept_footprint_slope_deg"] > 35.:
            errors.append("independent_calibration_witness_invalid")
        comparison=risk_cap_comparison(sem.get("risk_max"),b)
        if (not finite(comparison["signed_difference"]) or
                abs(comparison["signed_difference"])>comparison["tolerance"]):
            errors.append("independent_calibration_bottleneck_mismatch")
        result.update(witness_geometry=check,witness_sem=sem,witness_risk_comparison=comparison,
                      witness_sha256=receipt.get("witness_sha256"))
    result.update(independent_valid=not errors,effective_risk_cap=cap if finite(cap) else None,
                  bottleneck_semantics="minimum_on_registered_graph; not_continuous_optimum; not_PRI_minimum")
    return result


def validate_trial(directory, request, info, execution, *, oracle=None, field=None):
    from native_paper_oracle import NativePaperOracle
    from native_paper_v3_sem_evaluator import SharedSEMField
    directory = Path(directory)
    for key,hashkey in (("map_path","map_sha256"),("cloud_path","cloud_sha256")):
        if file_hash(info[key]) != info[hashkey]:
            raise ValueError("raw input hash mismatch during independent validation")
    row = dict(success=False,success_geometry=False,attained_safe=False,first_safe_s=None,
        first_safe_latency_s=None,final_cost_terrain_m=None,final_pri=None,final_swept_slope_deg=None,
        events=[],correctness_errors=[],execution_status=execution["status"],planner_status=None,
        field_build_s=execution.get("field_build_s"),solve_time_s=None,
        selection="last_published_incumbent_in_budget_before_validation")
    if not (directory/"worker.json").is_file():
        row["unavailable_reason"] = "no_worker_output"
        if execution["status"] == "COMPLETED":
            row["correctness_errors"].append("completed_process_missing_worker_output")
        return row
    worker = read(directory/"worker.json")
    oracle = oracle or NativePaperOracle(info["map_path"])
    field = field or SharedSEMField(np.load(info["cloud_path"],allow_pickle=False),info["bounds"])
    errors = row["correctness_errors"]
    row.update(planner_status=worker.get("planner_status",worker.get("status")),
        worker_execution_status=worker.get("execution_status"),termination_reason=worker.get("termination_reason"),
        ompl_version=worker.get("ompl_version"),compiler=worker.get("compiler"),
        preprocessing_s=worker.get("preprocessing_s",worker.get("phase_timings",{}).get("preprocessing_s")),
        solve_time_s=worker.get("actual_solve_wall_s",worker.get("phase_timings",{}).get("solve_s")))
    policy = request.get("safe_length")
    constrained = policy is not None
    cap,slope = 40.,35.
    if constrained:
        effective = worker.get("effective_safe_length",{})
        if worker.get("requested_safe_length") != policy:
            errors.append("requested_safety_policy_mismatch")
        for key in ("policy_id","slope_cap_deg","atlas_sampling"):
            if effective.get(key) != policy.get(key):
                errors.append("effective_safety_policy_mismatch:"+key)
        if not effective.get("thresholds_immutable_during_solve"):
            errors.append("effective_safety_contract_missing")
        if policy["policy_id"] == "a3t_safe_length_v2":
            checked = validate_calibration(directory,worker,request,oracle,field)
            row.update(calibration=checked,calibration_s=worker.get("calibration",{}).get("preprocessing_s"),
                calibration_status=checked["status"],b_graph=worker.get("calibration",{}).get("b_graph"))
            errors.extend(checked["errors"])
            cap = checked["effective_risk_cap"]
            if effective.get("witness_used_as_incumbent") is not False:
                errors.append("calibration_witness_used_as_incumbent")
        elif policy["policy_id"] == "a3t_safe_length_v1":
            cap = policy["risk_cap"]
            if effective.get("risk_cap") != cap:
                errors.append("effective_safety_policy_mismatch:risk_cap")
        else:
            errors.append("unknown_safety_policy")
        effective_policy = worker.get("effective_policy",{})
        # Early calibration failures have no planner setup. Once configured,
        # record the actual atlas/batch hooks rather than merely the request.
        if "safety_atlas_sampling" in effective_policy and effective_policy["safety_atlas_sampling"] != policy["atlas_sampling"]:
            errors.append("effective_atlas_policy_mismatch")
        if "batch_policy" in effective_policy:
            for key in ("batch_after_first_solution","samples_per_batch"):
                if key == "samples_per_batch" and request["planner_id"] != "bitstar_a3":
                    continue
                if effective_policy.get(key) != request[key]:
                    errors.append("effective_batch_policy_mismatch:"+key)
    row["effective_risk_cap"] = cap
    events,seen,previous_time,previous_cost = _events(directory,worker),set(),-1.,math.inf
    selected,before_solve,total_before_solve = None,None,None
    if worker.get("solve_started_utc") and execution.get("launch_utc"):
        before_solve = (datetime.fromisoformat(worker["solve_started_utc"].replace("Z","+00:00"))-
                        datetime.fromisoformat(execution["launch_utc"])).total_seconds()
        if before_solve < 0:
            raise ValueError("negative startup time")
        if execution.get("started_utc"):
            total_before_solve = (datetime.fromisoformat(worker["solve_started_utc"].replace("Z","+00:00"))-
                                 datetime.fromisoformat(execution["started_utc"])).total_seconds()
            if total_before_solve < before_solve:
                raise ValueError("attempt started after launch")
    row["launch_to_solve_s"] = before_solve
    row["attempt_to_solve_s"] = total_before_solve
    row["latency_origin"] = "attempt_start_before_input_hash_field_build_and_process_launch"
    for event in events:
        sid = event["snapshot_id"]
        if not isinstance(sid,str) or Path(sid).name != sid or sid in (".","..") or sid in seen:
            raise ValueError("unsafe or repeated snapshot id")
        seen.add(sid)
        elapsed = event.get("elapsed_s",event.get("timestamp_s"))
        cost = event.get("native_objective_value",event.get("objective_value",event.get("best_cost_terrain_m")))
        if not finite(elapsed) or elapsed < max(0.,previous_time) or not finite(cost) or cost > previous_cost+1e-8:
            raise ValueError("invalid incumbent time or cost order")
        path = _csv_path(directory/"snapshots"/sid/"path_xy.csv")
        if event.get("path_xy") is not None and event["path_xy"] != path:
            raise ValueError("snapshot CSV differs from callback coordinates")
        check = oracle.validate_path(path,request["start"],request["goal"],
            claimed_cost=event.get("cost_terrain_m",event.get("best_cost_terrain_m")),
            geometric_safety_metrics=True,clearance_metrics=False)
        sem = field.score_path(path)
        common_safe = bool(check["valid"] and sem["valid"] and check["maximum_swept_footprint_slope_deg"] <= slope)
        comparison=risk_cap_comparison(sem.get("risk_max"),cap,
            roundoff_ulps=RISK_ROUNDOFF_ULPS if constrained and policy["policy_id"]=="a3t_safe_length_v2" else 0)
        safe = bool(common_safe and comparison["passed"])
        if not check["valid"]:
            errors.append(sid+":independent_geometry_invalid")
        if constrained and not safe:
            errors.append(sid+":hard_safety_contract_failed")
        if constrained and abs(check["cost_terrain_m"]-cost) > 1e-8+1e-8*max(abs(cost),abs(check["cost_terrain_m"])):
            errors.append(sid+":surface_length_objective_mismatch")
        measured = dict(snapshot_id=sid,elapsed_s=elapsed,in_budget=elapsed<=request["planning_budget_s"],
                        safe=safe,common_safe=common_safe,geometry_valid=check["valid"],geometry=check,sem=sem,
                        risk_comparison=comparison)
        row["events"].append(measured)
        if measured["in_budget"]:
            selected = measured
            if safe and row["first_safe_s"] is None:
                row["first_safe_s"] = elapsed
                if total_before_solve is not None:
                    row["first_safe_latency_s"] = total_before_solve+elapsed
                if before_solve is not None:
                    row["first_safe_launch_latency_s"] = before_solve+elapsed
        previous_time,previous_cost = elapsed,cost
    row["attained_safe"] = row["first_safe_s"] is not None and execution["status"] == "COMPLETED" and not errors
    if execution["status"] != "COMPLETED" or errors:
        row["safe_snapshot_observed_in_failed_attempt"] = row["first_safe_s"] is not None
        row["first_safe_s"],row["first_safe_latency_s"] = None,None
    row["late_snapshot_count"] = sum(not e["in_budget"] for e in row["events"])
    if constrained and events and request.get("batch_after_first_solution") and worker.get("batch_transition_count") != 1:
        errors.append("adaptive_batch_transition_not_observed")
    if selected is not None:
        final_path = directory/"path_xy.csv"
        if not final_path.is_file() or _csv_path(final_path) != _csv_path(directory/"snapshots"/events[-1]["snapshot_id"]/"path_xy.csv"):
            errors.append("final_path_not_last_published_incumbent")
        row.update(selected_snapshot_id=selected["snapshot_id"],success_geometry=selected["geometry_valid"],
            final_cost_terrain_m=selected["geometry"]["cost_terrain_m"],final_pri=selected["sem"]["pri"],
            final_swept_slope_deg=selected["geometry"]["maximum_swept_footprint_slope_deg"],
            success=bool(execution["status"] == "COMPLETED" and row["planner_status"] == "EXACT_SOLUTION" and
                         selected["safe"] and not errors))
    elif row["planner_status"] == "EXACT_SOLUTION":
        errors.append("exact_solution_without_in_budget_snapshot")
    if errors:
        row["attained_safe"] = False
        row["first_safe_s"],row["first_safe_latency_s"] = None,None
    row["oracle_sources"] = {name:file_hash(ROOT/"tools"/name) for name in
        ("independent_verifier.py","native_paper_oracle.py","native_paper_v3_sem_evaluator.py","safe_length_v2_validation.py")}
    return row
