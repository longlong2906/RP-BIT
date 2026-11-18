"""Schema-13 independent snapshot validation and paired benchmark reports.

Native objectives are audited separately. Final selection precedes validation;
an unsafe last incumbent is never replaced with an earlier safe snapshot.
"""
from __future__ import annotations

import argparse
import collections
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time

import numpy as np

from a3_traversability_protocol import DEFAULT_TRAVERSABILITY
from native_paper_analysis import sharp_turn_count
from native_paper_oracle import NativePaperOracle
from native_paper_v3_analysis import audit_sem_native_cost
from native_paper_v3_sem_evaluator import SharedSEMField
from safe_length_field import _atomic_write, file_hash

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "a3t_old_safe_sem_kaggle_v4"
BACKENDS = ("a3t_old", "a3t_safe", "sem_native")
LABELS = {"a3t_old": "A3-T v1", "a3t_safe": "A3-T safe-length", "sem_native": "SEM-RRT* native reconstruction"}
UNSTARTED = {"NOT_RUN", "BLOCKED", "BLOCKED_SOURCE", "PLANNED", "PREPARED"}
QUALITY = ("cost_terrain_m", "length_xy_m", "pri", "maximum_swept_footprint_slope_deg")
TIMING = ("solve_time_s", "first_geometry_s", "first_safe_s", "first_geometry_latency_s", "first_safe_latency_s")
DERIVED = {"validation.json", "validation_receipt.json"}
# configs/experiments.yaml and the frozen schema-13 model specify this numeric
# roundoff tolerance for a singleton goal; it is not a physical goal region.
GOAL_ROUNDOFF_TOLERANCE_M = 1e-9


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump(path, value):
    _atomic_write(path, (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n").encode())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validation_code_identity():
    names = ("native_paper_v4_analysis.py", "independent_verifier.py", "native_paper_oracle.py",
             "native_paper_v3_sem_evaluator.py", "native_paper_v3_analysis.py", "native_paper_analysis.py",
             "a3_traversability_protocol.py")
    return digest({name: file_hash(ROOT/"tools"/name) for name in names})


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def close(a, b):
    return finite(a) and finite(b) and abs(a-b) <= 1e-8+1e-8*max(abs(a), abs(b))


def singleton_endpoint_errors(path, start, goal):
    distances = (math.dist(path[0], start), math.dist(path[-1], goal))
    return distances, all(math.isfinite(value) and value <= GOAL_ROUNDOFF_TOLERANCE_M for value in distances)


def path_csv(path, columns=2):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        records = list(csv.reader(stream))
    expected = ["x", "y"] if columns == 2 else ["x", "y", "z"]
    if not records or records[0] not in (expected, [name+"_m" for name in expected]):
        raise ValueError("invalid path CSV header: "+str(path))
    values = [[float(x) for x in row] for row in records[1:]]
    if not values or any(len(row) != columns or not all(map(math.isfinite, row)) for row in values):
        raise ValueError("empty/nonfinite/malformed path CSV")
    return values


def events_from(directory, worker):
    if worker.get("snapshots") is not None and all(isinstance(item,dict) for item in worker["snapshots"]):
        return worker["snapshots"]
    if worker.get("snapshot_ids") is not None and not (directory/"progress.jsonl").exists():
        raise ValueError("compact worker missing progress.jsonl")
    path = directory/"progress.jsonl"
    if not path.exists():
        return []
    return [value for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
            for value in [json.loads(line)] if value.get("event_type") == "improvement" or
            (value.get("event_type") is None and "snapshot_id" in value)]


def raw_inventory(directory):
    return {p.relative_to(directory).as_posix(): file_hash(p) for p in sorted(directory.rglob("*"))
            if p.is_file() and p.name not in DERIVED and not p.name.endswith(".tmp")}


def _backend(request):
    for key in ("backend_id", "v4_backend", "method", "backend"):
        if request.get(key) in BACKENDS:
            return request[key]
    if "safe_length" in request:
        return "a3t_safe"
    if request.get("backend") == "sem" or request.get("planner_id") in ("sem_native", "sem_native_paper_v2"):
        return "sem_native"
    if "traversability" in request:
        return "a3t_old"
    raise ValueError("unregistered schema-13 backend")


def _safe_policy_errors(worker, request):
    requested = request["safe_length"]
    effective = worker.get("effective_safe_length", {})
    errors = []
    for key in ("risk_cap", "slope_cap_deg", "atlas_sampling"):
        if effective.get(key) != requested.get(key):
            errors.append("effective_safety_policy_mismatch:"+key)
    if effective.get("policy_id") != "a3t_safe_length_v1" or not effective.get("thresholds_immutable_during_solve"):
        errors.append("effective_safety_contract_missing")
    if worker.get("requested_safe_length") != requested:
        errors.append("requested_safety_policy_mismatch")
    policy = worker.get("effective_policy", {})
    if policy.get("safety_atlas_sampling") != requested.get("atlas_sampling"):
        errors.append("effective_atlas_policy_mismatch")
    if "batch_policy" in policy:
        for key in ("batch_after_first_solution", "samples_per_batch"):
            if policy.get(key) != request.get(key):
                errors.append("effective_batch_policy_mismatch:"+key)
    return errors


def _native_raster(directory, worker):
    sidecar = Path(directory)/"artifact_refs.json"
    references = read(sidecar) if sidecar.exists() else worker.get("artifact_refs", {})
    if sidecar.exists() and worker.get("artifact_refs") is not None and references != worker["artifact_refs"]:
        raise ValueError("native raster sidecar disagrees with worker reference")
    value = references.get("native_raster_path")
    if value is not None:
        path = Path(value)
        if not path.is_absolute():
            path = (directory/path).resolve()
        expected = references.get("native_raster_sha256", references.get("native_raster_hash"))
        if not expected or file_hash(path) != expected:
            raise ValueError("native raster reference hash mismatch")
        return path
    path = directory/"native_model/raster.npz"
    expected = worker.get("native_model", {}).get("raster_artifacts", {}).get("raster.npz")
    if not expected or file_hash(path) != expected:
        raise ValueError("native raster inventory/hash mismatch")
    return path


def _setup_latency(directory, worker, execution):
    launch = execution.get("launch_utc")
    if launch is None and (directory/"launch.json").exists():
        metadata = read(directory/"launch.json")
        launch = metadata.get("launch_utc", metadata.get("process_created"))
    def epoch(value):
        return float(value) if finite(value) else datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    if launch is not None and worker.get("solve_started_utc"):
        elapsed = epoch(worker["solve_started_utc"])-epoch(launch)
        if elapsed < 0:
            raise ValueError("negative launch-to-solve latency")
        # Wrapper field construction occurs after launch_utc by default. Explicit
        # legacy callers can mark field construction outside that launch clock.
        return elapsed+(execution.get("field_build_s", 0.) if execution.get("field_build_outside_launch", False) else 0.), "wrapper_launch_to_snapshot"
    timings = worker.get("phase_timings", worker)
    fields = [timings.get(key) for key in ("load_s", "conversion_s", "preprocessing_s", "setup_s")]
    if not any(finite(value) for value in fields):
        return None, "UNAVAILABLE"
    return math.fsum(value for value in fields if finite(value))+float(execution.get("field_build_s") or 0.), "reported_setup_plus_field_to_snapshot"


def validate_trial(directory, request, info, execution, oracle=None, field=None):
    """Read raw inputs and every snapshot; return a complete auditable verdict."""
    directory = Path(directory)
    began = time.perf_counter()
    backend = _backend(request)
    for key, hashkey in (("map_path", "map_sha256"), ("cloud_path", "cloud_sha256")):
        if file_hash(info[key]) != info[hashkey]:
            raise ValueError("raw input hash mismatch: "+key)
    if (directory/"request.json").exists() and digest(read(directory/"request.json")) != digest(request):
        raise ValueError("raw request differs from frozen plan")
    status = execution.get("status", execution.get("execution_status", "NOT_RUN"))
    row = dict(backend=backend, method=backend, execution_status=status, budget_s=request["planning_budget_s"],
        success=False, success_geometry=False, success_paper=False, attained_safe=False, attained_geometry=False,
        first_safe_s=None, first_geometry_s=None, first_safe_latency_s=None, first_geometry_latency_s=None,
        selected_snapshot_id=None, events=[], correctness_errors=[], errors=[], planning_failures=[],
        planner_status=None, native_status=None, solve_time_s=None, field_build_s=execution.get("field_build_s", 0.),
        endpoint_policy="singleton_roundoff_euclidean", goal_roundoff_tolerance_m=GOAL_ROUNDOFF_TOLERANCE_M,
        selection="last_published_incumbent_within_budget_before_validation", validation_cache_hit=False)
    row.update({key: info.get(key) for key in ("map_id", "family")})
    row.update(planner_seed=request.get("planner_seed"), query_id=request.get("query_id", "q0"),
               timing_stratum=execution.get("timing_stratum", request.get("timing_stratum")))
    if not (directory/"worker.json").exists():
        row["unavailable_reason"] = "no_worker_output"
        row["validation_wall_s"] = time.perf_counter()-began
        return row
    worker = read(directory/"worker.json")
    oracle = oracle or NativePaperOracle(info["map_path"])
    field = field or SharedSEMField(np.load(info["cloud_path"], allow_pickle=False), info["bounds"])
    row["planner_status"] = worker.get("planner_status", worker.get("status"))
    row["native_status"] = worker.get("native_status", worker.get("status", row["planner_status"]))
    row["worker_execution_status"] = worker.get("execution_status")
    for failure in sorted({worker.get(key) for key in ("planner_status", "status", "native_status")}
                          & {"ERROR", "MODEL_ERROR", "BLOCKED_DEPENDENCY"}):
        row["correctness_errors"].append("worker_status_failure:"+failure)
    if row["worker_execution_status"] in ("FAILED_CORRECTNESS", "PROVENANCE_MISMATCH"):
        row["correctness_errors"].append("worker_execution_failure:"+row["worker_execution_status"])
    row["termination_reason"] = worker.get("termination_reason")
    row["solve_time_s"] = worker.get("actual_solve_wall_s", worker.get("phase_timings", {}).get("solve_s"))
    row["timings"] = worker.get("phase_timings", {key:worker.get(key) for key in
        ("load_s", "preprocessing_s", "safety_preprocessing_s", "setup_s", "actual_solve_wall_s")})
    row["effective_policy"] = worker.get("effective_policy", {})
    if backend == "a3t_safe":
        row["correctness_errors"] += _safe_policy_errors(worker, request)
    elif backend == "a3t_old":
        if request.get("traversability") != DEFAULT_TRAVERSABILITY:
            row["correctness_errors"].append("old_a3_traversability_not_frozen")
        if worker.get("effective_traversability") != request.get("traversability"):
            row["correctness_errors"].append("old_a3_effective_traversability_mismatch")
    setup, row["latency_scope"] = _setup_latency(directory, worker, execution)
    source_events = events_from(directory, worker)
    seen, previous_time, previous_cost = set(), -1., math.inf
    last_path = None
    raster = _native_raster(directory, worker) if backend == "sem_native" and source_events else None
    for source in source_events:
        sid = source["snapshot_id"]
        if not isinstance(sid, str) or Path(sid).name != sid or sid in (".", "..") or sid in seen or "\\" in sid:
            raise ValueError("unsafe/repeated snapshot id")
        seen.add(sid)
        elapsed = source.get("elapsed_s", source.get("timestamp_s"))
        native = source.get("native_objective_value", source.get("planner_objective_value",
                 source.get("objective_value", source.get("best_cost_terrain_m"))))
        if not finite(elapsed) or elapsed < 0 or elapsed < previous_time or not finite(native) or native < 0 or native > previous_cost+1e-8:
            raise ValueError("invalid incumbent timestamp/native cost order")
        snapshot = directory/"snapshots"/sid
        path = path_csv(snapshot/"path_xy.csv")
        if source.get("path_xy") is not None and source["path_xy"] != path:
            raise ValueError("snapshot CSV/native event mismatch")
        endpoint_errors, exact = singleton_endpoint_errors(path, request["start"], request["goal"])
        surface = path_csv(snapshot/"path_surface.csv", 3) if backend != "sem_native" else None
        checked = oracle.validate_path(path, request["start"], request["goal"], surface=surface,
            traversability=request["traversability"] if backend == "a3t_old" else None,
            geometric_safety_metrics=True, clearance_metrics=False)
        geometry_errors = list(checked.get("errors", []))
        if not exact:
            geometry_errors.append("not_exact_start_goal")
        audit_errors = []
        if backend == "sem_native":
            raw = path_csv(snapshot/"path_raw_xyz.csv", 3)
            if source.get("raw_path_xyz") is not None and source["raw_path_xyz"] != raw:
                audit_errors.append("raw_xyz_event_csv_mismatch")
            if len(raw) != len(path) or any(p[:2] != q for p, q in zip(raw, path)):
                audit_errors.append("raw_xyz_vertex_xy_mismatch")
            else:
                for p, q in zip(raw, path):
                    try:
                        _, lifted = oracle.surface(q, q)
                        if not close(p[2], lifted[0][2]):
                            audit_errors.append("raw_xyz_off_heightfield")
                            break
                    except ValueError:
                        audit_errors.append("raw_xyz_missing_support")
                        break
        for key in ("cost_terrain_m", "length_xy_m"):
            if source.get(key) is not None and not close(source[key], checked.get(key)):
                audit_errors.append(key+"_mismatch")
        try:
            recomputed = audit_sem_native_cost(path, raster) if backend == "sem_native" else checked.get(
                "objective_value" if backend == "a3t_old" else "cost_terrain_m")
            if not close(native, recomputed):
                audit_errors.append("native_objective_mismatch")
        except (ValueError, OSError, KeyError, IndexError, TypeError) as error:
            recomputed = None
            audit_errors.append("native_cost_audit:"+str(error))
        if backend == "a3t_old":
            for key in ("risk_slope_integral_m", "risk_clearance_integral_m", "risk_integral_m"):
                if key not in source or not close(source[key], checked.get(key)):
                    audit_errors.append(key+"_mismatch")
        geometry_valid = bool(checked["valid"] and exact)
        sem = field.score_path(path)
        caps = request.get("safe_length", {"risk_cap":40., "slope_cap_deg":35.})
        slope = checked.get("maximum_swept_footprint_slope_deg")
        safe = bool(geometry_valid and sem["valid"] and finite(sem.get("risk_max")) and
                    sem["risk_max"] <= caps["risk_cap"] and finite(slope) and slope <= caps["slope_cap_deg"])
        if backend == "a3t_safe" and not safe:
            audit_errors.append("hard_safety_contract_failed")
        if geometry_errors:
            if backend == "a3t_safe":
                audit_errors += geometry_errors
            else:
                row["planning_failures"] += [sid+":"+error for error in geometry_errors]
        if not sem["valid"]:
            row["planning_failures"] += [sid+":"+reason for reason in sem["reasons"]]
        row["correctness_errors"] += [sid+":"+error for error in audit_errors]
        metrics = {key:checked.get(key) for key in ("cost_terrain_m", "length_xy_m", "maximum_swept_footprint_slope_deg")}
        metrics.update(pri=sem.get("pri"), sharp_turn_count=sharp_turn_count(path))
        row["events"].append(dict(snapshot_id=sid, elapsed_s=elapsed, in_budget=elapsed<=row["budget_s"],
            native_objective_value=native, native_recomputed=recomputed, geometry_valid=geometry_valid,
            independent_valid=geometry_valid, safe=safe, paper_valid=safe, metrics=metrics,
            geometry_errors=geometry_errors, sem_score=sem, audit_errors=audit_errors))
        row["events"][-1]["endpoint_errors_m"] = dict(start=endpoint_errors[0], goal=endpoint_errors[1])
        last_path = path
        previous_time, previous_cost = elapsed, native
    if source_events:
        if not (directory/"path_xy.csv").exists() or path_csv(directory/"path_xy.csv") != last_path:
            row["correctness_errors"].append("final_path_not_last_published_incumbent")
        if backend == "a3t_safe" and request.get("batch_after_first_solution") and worker.get("batch_transition_count") != 1:
            row["correctness_errors"].append("adaptive_batch_transition_not_observed")
    elif row["planner_status"] == "EXACT_SOLUTION":
        row["correctness_errors"].append("exact_status_without_snapshot")
    in_budget = [event for event in row["events"] if event["in_budget"]]
    selected = in_budget[-1] if in_budget else None
    row["late_snapshot_count"] = len(row["events"])-len(in_budget)
    row["correctness_errors"] = sorted(set(row["correctness_errors"]))
    row["errors"] = row["correctness_errors"]
    completed = status == "COMPLETED" and row["planner_status"] == "EXACT_SOLUTION" and not row["correctness_errors"]
    for kind, key in (("geometry", "geometry_valid"), ("safe", "safe")):
        first = next((event for event in in_budget if event[key]), None)
        row["attained_"+kind] = bool(completed and first)
        row["first_"+kind+"_s"] = first["elapsed_s"] if completed and first else None
        row["first_"+kind+"_latency_s"] = setup+first["elapsed_s"] if completed and first and setup is not None else None
        row["first_"+kind+"_snapshot_id"] = first["snapshot_id"] if completed and first else None
        row[kind+"_snapshot_observed_in_failed_attempt"] = bool(first and not completed)
    if selected:
        row.update(selected_snapshot_id=selected["snapshot_id"], success_geometry=bool(completed and selected["geometry_valid"]),
                   success=bool(completed and selected["safe"]), success_paper=bool(completed and selected["safe"]))
        row.update(final_cost_terrain_m=selected["metrics"]["cost_terrain_m"], final_pri=selected["metrics"]["pri"],
                   final_swept_slope_deg=selected["metrics"]["maximum_swept_footprint_slope_deg"])
    row["validation_wall_s"] = time.perf_counter()-began
    row["validation_code_identity"] = validation_code_identity()
    return row


def validate_map(workspace, map_info, trial_entries, jobs=1):
    """Validate a quiescent map wave before packing; parallelism is across maps."""
    workspace = Path(workspace)
    if jobs < 1:
        raise ValueError("jobs must be positive")
    # Raw data and external raster references are verified even on cache hits.
    # A hash-bearing cached verdict must not hide later artifact corruption.
    for key, hashkey in (("map_path", "map_sha256"), ("cloud_path", "cloud_sha256")):
        if file_hash(map_info[key]) != map_info[hashkey]:
            raise ValueError("raw map/cloud changed before validation wave")
    oracle, field, rows = None, None, []
    code = validation_code_identity()
    for entry in trial_entries:
        raw_directory = entry.get("directory", str(Path("trials")/entry["trial_id"]))
        directory = Path(raw_directory)
        if not directory.is_absolute():
            directory = workspace/directory
        request = read(directory/"request.json") if (directory/"request.json").exists() else entry["request"]
        if entry.get("request") is not None and digest(entry["request"]) != digest(request):
            raise ValueError("trial request changed from frozen entry")
        execution = read(directory/"execution.json") if (directory/"execution.json").exists() else {
            "status":entry.get("execution_status", entry.get("status", "NOT_RUN"))}
        if execution.get("status", execution.get("execution_status")) in ("RUNNING", "STARTED", "START_REQUESTED"):
            raise ValueError("validation cannot overlap an active trial")
        execution.setdefault("timing_stratum", entry.get("timing_stratum"))
        if (directory/"preparation.json").exists():
            preparation = read(directory/"preparation.json")
            execution.setdefault("field_build_s", preparation.get("field_build_s", preparation.get("planner_field_build_s", 0.)))
        inventory = raw_inventory(directory)
        external = {}
        if (directory/"worker.json").exists() and _backend(request) == "sem_native":
            worker = read(directory/"worker.json")
            if events_from(directory, worker):
                raster = _native_raster(directory, worker)
                external["native_raster_sha256"] = file_hash(raster)
        key = digest(dict(inventory=inventory, request=request, map_sha256=map_info["map_sha256"],
                          cloud_sha256=map_info["cloud_sha256"], external_artifacts=external, code=code))
        verdict, receipt = directory/"validation.json", directory/"validation_receipt.json"
        if verdict.exists() and receipt.exists() and read(receipt).get("cache_key") == key and read(receipt).get("verdict_sha256") == file_hash(verdict):
            row = read(verdict)
        else:
            if (directory/"worker.json").exists() and oracle is None:
                oracle = NativePaperOracle(map_info["map_path"])
                field = SharedSEMField(np.load(map_info["cloud_path"], allow_pickle=False), map_info["bounds"])
            row = validate_trial(directory, request, map_info, execution, oracle, field)
            row.update({name:entry.get(name, row.get(name)) for name in
                        ("trial_id", "backend", "family", "map_id", "query_id", "planner_seed", "timing_stratum")})
            row["backend"] = row.get("backend") or _backend(request)
            row["method"] = row["backend"]
            row["validation_code_identity"] = code
            dump(verdict, row)
            if raw_inventory(directory) != inventory:
                raise ValueError("raw trial changed during validation")
            dump(receipt, dict(cache_key=key, verdict_sha256=file_hash(verdict), raw_inventory_sha256=digest(inventory),
                               validation_code_identity=code))
        # The ledger is a hash-bound copy, not a second independent verdict.
        ledger = dict(row=row, receipt=read(receipt), verdict_sha256=file_hash(verdict))
        dump(workspace/"verdicts"/map_info.get("mode", "main")/(entry["trial_id"]+".json"), ledger)
        rows.append(row)
    return rows


def final_event(row):
    available = [event for event in row.get("events", []) if event["elapsed_s"] <= row["budget_s"]]
    return available[-1] if available else None


def selected_metrics(row, selection="final_safe"):
    if selection == "first_safe":
        event = next((event for event in row.get("events", []) if event["elapsed_s"] <= row["budget_s"] and event["safe"]), None)
    else:
        event = final_event(row)
    return event["metrics"] if event else {}


def stats(values):
    values = [value for value in values if value is not None]
    if any(not finite(value) for value in values):
        raise ValueError("nonfinite metric")
    return dict(n=len(values), mean=statistics.mean(values) if values else None,
                sd=statistics.stdev(values) if len(values)>1 else None, median=statistics.median(values) if values else None)


def restricted_observation(row):
    if row["execution_status"] in UNSTARTED:
        return None
    if row["execution_status"] != "COMPLETED" or row.get("correctness_errors"):
        return None
    return row["first_safe_s"] if row.get("attained_safe") else row["budget_s"]


def group_summary(rows, selection="final_safe"):
    attempted = [row for row in rows if row["execution_status"] not in UNSTARTED]
    key = {"final_safe":"success", "final_geometry":"success_geometry", "first_safe":"attained_safe"}[selection]
    valid = [row for row in attempted if row.get(key, False)]
    measured = [selected_metrics(row, selection) for row in valid]
    total_length = math.fsum(item["cost_terrain_m"] for item in measured)
    observations = [restricted_observation(row) for row in attempted]
    result = dict(selection=selection, planned=len(rows), attempted=len(attempted), success=len(valid),
        psr=len(valid)/len(attempted) if attempted else None,
        psr_geometry=sum(row.get("success_geometry", False) for row in attempted)/len(attempted) if attempted else None,
        psr_paper=sum(row.get("success", False) for row in attempted)/len(attempted) if attempted else None,
        attained_safe=sum(row.get("attained_safe", False) for row in attempted),
        status_counts=dict(collections.Counter(row["execution_status"] for row in rows)),
        sti=math.fsum(item["sharp_turn_count"] for item in measured)/total_length if total_length>0 else None,
        restricted_first_safe_mean_s=statistics.mean(observations) if observations and all(value is not None for value in observations) else None,
        restricted_unavailable=sum(value is None for value in observations),
        censored_at_budget=sum(not row.get("attained_safe", False) and restricted_observation(row) is not None for row in attempted))
    for metric in QUALITY:
        result[metric] = stats(item.get(metric) for item in measured)
    for metric in TIMING:
        result[metric] = stats(row.get(metric) for row in valid)
    result["first_safe_all_attained"] = stats(row.get("first_safe_s") for row in attempted if row.get("attained_safe"))
    return result


def paired_bootstrap(rows, competitor, metric, replicates=2000, selection="final_safe"):
    if competitor not in ("a3t_old", "sem_native") or not isinstance(replicates, int) or replicates<1:
        raise ValueError("invalid contrast/bootstrap configuration")
    keyed = {}
    for row in rows:
        key = (row["family"], row["map_id"], row.get("query_id", "q0"), row["planner_seed"])
        methods = keyed.setdefault(key, {})
        if row["backend"] in methods:
            raise ValueError("duplicate paired trial")
        methods[row["backend"]] = row
    pairs, planned, attempted, mismatched, unavailable = [], 0, 0, 0, 0
    success_key = {"final_safe":"success", "final_geometry":"success_geometry", "first_safe":"attained_safe"}[selection]
    for (family, map_id, query, seed), methods in keyed.items():
        if "a3t_safe" not in methods or competitor not in methods:
            continue
        planned += 1
        a, b = methods["a3t_safe"], methods[competitor]
        if any(row["execution_status"] in UNSTARTED for row in (a,b)):
            continue
        attempted += 1
        if metric in TIMING+ ("restricted_first_safe_s",) and (not a.get("timing_stratum") or a.get("timing_stratum") != b.get("timing_stratum")):
            mismatched += 1
            continue
        if metric in ("psr_paper", "psr_geometry"):
            key = "success" if metric == "psr_paper" else "success_geometry"
            value = float(a[key]), float(b[key])
        elif metric == "restricted_first_safe_s":
            value = restricted_observation(a), restricted_observation(b)
            if None in value:
                unavailable += 1
                continue
        else:
            # First-solution comparisons condition on attaining that criterion,
            # not on the unrelated validity of a later final incumbent.
            required = "attained_geometry" if metric.startswith("first_geometry") else "attained_safe" if metric.startswith("first_safe") else success_key
            if not a.get(required) or not b.get(required):
                continue
            aa, bb = selected_metrics(a, selection), selected_metrics(b, selection)
            if metric == "sti":
                if aa.get("cost_terrain_m",0)<=0 or bb.get("cost_terrain_m",0)<=0:
                    continue
                value = (aa["sharp_turn_count"],aa["cost_terrain_m"]),(bb["sharp_turn_count"],bb["cost_terrain_m"])
            else:
                value = (a.get(metric),b.get(metric)) if metric in TIMING else (aa.get(metric),bb.get(metric))
                if any(v is None for v in value):
                    continue
        pairs.append((family,map_id,value))
    result = dict(competitor=competitor, metric=metric, selection=selection, planned_pairs=planned,
        attempted_pairs=attempted, usable_pairs=len(pairs), excluded_stratum_mismatch=mismatched,
        unavailable_pairs=unavailable, coverage=len(pairs)/planned if planned else None,
        maps=len({(family,map_id) for family,map_id,_ in pairs}), mean_difference=None, ci95=None)
    if not pairs or (metric == "restricted_first_safe_s" and unavailable):
        return result
    groups = {}
    for family,map_id,value in pairs:
        groups.setdefault(family,{}).setdefault(map_id,[]).append(value)
    def estimate(values):
        if metric == "sti":
            return math.fsum(a[0] for a,b in values)/math.fsum(a[1] for a,b in values)-math.fsum(b[0] for a,b in values)/math.fsum(b[1] for a,b in values)
        return statistics.mean(a-b for a,b in values)
    rng,draws = random.Random(83138),[]
    for _ in range(replicates):
        values=[]
        for family in sorted(groups):
            maps=groups[family];ids=sorted(maps)
            for _ in ids:
                block=maps[rng.choice(ids)]
                values.extend(rng.choice(block) for _ in block)
        draws.append(estimate(values))
    draws.sort()
    result.update(mean_difference=estimate([value for _,_,value in pairs]),
                  ci95=[draws[int(.025*(replicates-1))],draws[int(.975*(replicates-1))]])
    return result


def build_report(rows, planned_entries, budget, replicates=2000):
    if not finite(budget) or budget<=0:
        raise ValueError("invalid report budget")
    observed={}
    for row in rows:
        if row["trial_id"] in observed:
            raise ValueError("duplicate verdict")
        observed[row["trial_id"]]=row
    planned_ids=[entry["trial_id"] for entry in planned_entries]
    if len(set(planned_ids))!=len(planned_ids) or set(observed)-set(planned_ids):
        raise ValueError("duplicate/unplanned trial identity")
    merged=[]
    for entry in planned_entries:
        row=dict(observed.get(entry["trial_id"], {}))
        for key in ("trial_id","backend","family","map_id","query_id","planner_seed","timing_stratum"):
            if key in entry:
                if key in row and row[key] is not None and row[key]!=entry[key] and key!="timing_stratum":
                    raise ValueError("verdict/plan identity mismatch: "+key)
                row.setdefault(key,entry[key])
        if row.get("backend") not in BACKENDS:
            raise ValueError("invalid backend in planned trial")
        row.setdefault("execution_status",entry.get("execution_status",entry.get("status","NOT_RUN")))
        row.setdefault("budget_s",budget)
        if row["budget_s"]!=budget:
            raise ValueError("mixed planning budgets")
        for key in ("success","success_geometry","attained_safe","attained_geometry"):
            row.setdefault(key,False)
        row.setdefault("events",[]);row.setdefault("correctness_errors",[])
        if entry["trial_id"] not in observed and row["execution_status"] not in UNSTARTED:
            row["correctness_errors"].append("executed_trial_missing_verdict")
        merged.append(row)
    tables=[]
    for scope,key in (("all",None),("family","family"),("map","map_id"),("timing_stratum","timing_stratum")):
        groups=["all"] if key is None else sorted({str(row.get(key)) for row in merged})
        for value in groups:
            for backend in BACKENDS:
                subset=[row for row in merged if row["backend"]==backend and (key is None or str(row.get(key))==value)]
                if subset:
                    for selection in ("final_safe","final_geometry","first_safe"):
                        tables.append(dict(scope=scope,group=value,backend=backend,**group_summary(subset,selection)))
    contrasts=[paired_bootstrap(merged,competitor,metric,replicates,selection)
        for selection in ("final_safe","first_safe") for competitor in ("a3t_old","sem_native")
        for metric in ("psr_paper","psr_geometry",*QUALITY,"sti",*TIMING,"restricted_first_safe_s")]
    checkpoints=sorted({0.,budget,*[min(budget,t) for t in (.1,.25,.5,1.,2.,5.,10.,15.)],
        *[row["first_"+kind+"_s"] for row in merged for kind in ("safe","geometry") if row.get("attained_"+kind)]})
    attainment=[]
    for backend in BACKENDS:
        subset=[row for row in merged if row["backend"]==backend and row["execution_status"] not in UNSTARTED]
        for kind in ("geometry","safe"):
            for checkpoint in checkpoints:
                count=sum(row.get("attained_"+kind,False) and row["first_"+kind+"_s"]<=checkpoint for row in subset)
                attainment.append(dict(backend=backend,criterion=kind,elapsed_s=checkpoint,attained=count,
                    attempted=len(subset),fraction=count/len(subset) if subset else None))
    errors=sum(bool(row["correctness_errors"]) for row in merged)
    completion="FAILED_CORRECTNESS" if errors else "PARTIAL" if any(row["execution_status"] in UNSTARTED for row in merged) else "COMPLETE"
    return dict(protocol_id=PROTOCOL_ID,schema_version=13,completion=completion,budget_s=budget,
        planned=len(merged),attempted=sum(row["execution_status"] not in UNSTARTED for row in merged),
        correctness_failures=errors,rows=merged,tables=tables,paired_contrasts=contrasts,attainment=attainment,
        bootstrap_replicates=replicates,bootstrap_seed=83138,confidence=.95,
        bootstrap_unit="fixed_family_then_map_then_paired_seed",plots=False)


def _write_csv(path,records):
    with Path(path).open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=sorted({key for row in records for key in row}))
        writer.writeheader()
        for row in records:
            writer.writerow({key:json.dumps(value,ensure_ascii=False) if isinstance(value,(dict,list)) else value for key,value in row.items()})


def write_report(report, output, plots=True):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    _write_csv(output/"aggregate.csv",report["tables"])
    _write_csv(output/"paired_comparisons.csv",report["paired_contrasts"])
    _write_csv(output/"attainment.csv",report["attainment"])
    trials=[]
    for row in report["rows"]:
        record={key:row.get(key) for key in ("trial_id","backend","family","map_id","planner_seed","query_id",
            "execution_status","planner_status","timing_stratum","success","success_geometry","attained_safe",
            "correctness_errors","planning_failures","selected_snapshot_id",*TIMING)}
        record.update({"final_"+key:value for key,value in selected_metrics(row).items()})
        trials.append(record)
    _write_csv(output/"trials.csv",trials)
    if plots and report["attempted"]:
        if any(row.get("synthetic_fixture") for row in report["rows"]):
            raise ValueError("synthetic parser fixtures cannot generate benchmark plots")
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        figure,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
        for axis,kind in zip(axes,("geometry","safe")):
            for backend in BACKENDS:
                points=[p for p in report["attainment"] if p["backend"]==backend and p["criterion"]==kind and p["fraction"] is not None]
                if points:
                    axis.step([p["elapsed_s"] for p in points],[p["fraction"] for p in points],where="post",label=LABELS[backend])
            axis.set(xlabel="Solve time (s)",title=kind,xlim=(0,report["budget_s"]),ylim=(0,1));axis.grid(alpha=.25)
        axes[0].set_ylabel("Independent validated attainment / attempts")
        axes[-1].legend(fontsize=7);figure.tight_layout()
        figure.savefig(output/"attainment.png",dpi=160);figure.savefig(output/"attainment.pdf");plt.close(figure)
        report["plots"]=True
    def fmt(value): return "—" if value is None else f"{value:.6g}"
    def summary(value): return fmt(value["mean"])+(" ± "+fmt(value["sd"]) if value["sd"] is not None else "")+f" (n={value['n']})"
    lines=["# Benchmark A3-T cũ, A3-T safe-length và SEM-RRT* v4","",
        f"Trạng thái: {report['completion']}; {report['attempted']}/{report['planned']} attempt; correctness failures: {report['correctness_failures']}.","",
        "Bảng chính chọn incumbent cuối trong ngân sách trước kiểm chứng; không cứu bằng nghiệm safe cũ. PSR yêu cầu exact goal, kiểm toàn footprint và SEM support/risk≤40. PRI là chỉ số raster, không phải xác suất tai nạn.",
        "APL là chiều dài mặt địa hình; STI=tổng turn XY>30°/tổng L_H. Không so native objective giữa thuật toán. Time là solve thực; ưu tiên thời gian tới nghiệm an toàn đầu và báo latency gồm setup riêng.",
        "Restricted mean là diện tích chưa đạt nghiệm safe trên [0,budget]; timeout không bị gán thành nghiệm ở deadline. Crash/interrupted làm restricted observation unavailable và vẫn nằm mẫu số PSR.","",
        "| Chọn đường | Phạm vi | Nhóm | Thuật toán | Success/attempt/planned | PSR safe | APL (m) | PRI | STI (1/m) | Max swept slope (°) | Solve (s) | First safe (s) | Latency safe (s) | Restricted (s) |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for t in report["tables"]:
        lines.append("| "+" | ".join([t["selection"],t["scope"],t["group"],LABELS[t["backend"]],f"{t['success']}/{t['attempted']}/{t['planned']}",
            fmt(t["psr_paper"]),summary(t["cost_terrain_m"]),summary(t["pri"]),fmt(t["sti"]),summary(t["maximum_swept_footprint_slope_deg"]),
            summary(t["solve_time_s"]),summary(t["first_safe_s"]),summary(t["first_safe_latency_s"]),fmt(t["restricted_first_safe_mean_s"])])+" |")
    lines += ["","Hiệu A3-T safe-length − đối chứng; CI95 bootstrap family cố định, map rồi paired seed. Quality chỉ dùng cặp cùng success; timing chỉ ghép cùng timing_stratum. Coverage và mọi failure giữ trong CSV.","",
              "| Chọn đường | Đối chứng | Metric | Cặp dùng/attempt/planned | Hiệu | CI95 |","|---|---|---|---:|---:|---|"]
    for c in report["paired_contrasts"]:
        lines.append(f"| {c['selection']} | {c['competitor']} | {c['metric']} | {c['usable_pairs']}/{c['attempted_pairs']}/{c['planned_pairs']} | {fmt(c['mean_difference'])} | {c['ci95']} |")
    (output/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    dump(output/"summary.json",report)
    return report


def analyze_campaign(workspace, output, mode="main", replicates=2000, plots=True):
    workspace,output=Path(workspace),Path(output)
    candidates=[workspace/"manifests"/(mode+".json"),workspace/"plans"/(mode+".json"),workspace/"manifest.json"]
    manifest_path=next((path for path in candidates if path.exists()),None)
    if manifest_path is None:
        raise FileNotFoundError("missing schema-13 plan/manifest")
    manifest=read(manifest_path);entries=manifest["entries"]
    rows=[];ledger_hashes={};code=validation_code_identity()
    for entry in entries:
        path=workspace/"verdicts"/mode/(entry["trial_id"]+".json")
        if not path.exists():
            continue
        ledger=read(path);row=ledger["row"];receipt=ledger["receipt"]
        serialized=(json.dumps(row,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
        if hashlib.sha256(serialized).hexdigest()!=ledger["verdict_sha256"] or receipt["verdict_sha256"]!=ledger["verdict_sha256"]:
            raise ValueError("verdict ledger hash mismatch")
        if row.get("validation_code_identity")!=code or receipt.get("validation_code_identity")!=code:
            raise ValueError("validation code identity changed")
        rows.append(row);ledger_hashes[entry["trial_id"]]=file_hash(path)
    budget=manifest.get("planning_budget_s",manifest.get("budget_s",manifest.get("budget",20. if mode=="main" else 2.)))
    report=build_report(rows,entries,budget,replicates)
    report["provenance"]=dict(manifest_sha256=file_hash(manifest_path),ledger_hashes=ledger_hashes,validation_code_identity=code)
    return write_report(report,output,plots=plots and not manifest.get("synthetic_fixture",False))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--mode",default="main");parser.add_argument("--no-plots",action="store_true")
    args=parser.parse_args()
    result=analyze_campaign(args.workspace,args.output,args.mode,plots=not args.no_plots)
    print(json.dumps({key:result[key] for key in ("completion","planned","attempted","correctness_failures")}))
    raise SystemExit(0 if result["completion"]=="COMPLETE" else 2)
