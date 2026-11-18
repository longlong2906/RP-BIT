"""Independent raw-map/cloud validation for additive SEM exposure objectives.

The planner's field and C++ edge integration never serve as an oracle here.
PRI remains the mean over unique closed cells; exposure uses surface measure.
"""
from __future__ import annotations

from datetime import datetime
from bisect import bisect_left, bisect_right
from fractions import Fraction
import json
import math
from pathlib import Path

import numpy as np

from corridor_v3_validation import risk_metrics
from corridor_v4_validation import _candidate_surface_length, _exact_original_intervals
from run_safe_length import _csv_path, _events, read
from safe_length_field import file_hash
from safe_length_v2_validation import finite

ROOT = Path(__file__).resolve().parents[1]
VERSION = "independent_sem_exposure_v1"
OBJECTIVE = "surface_length_sem_exposure"
METRICS = ("objective_value", "cost_terrain_m", "best_cost_terrain_m", "length_xy_m",
           "sem_exposure_risk_m", "sem_exposure_normalized_m")


def _close(actual, expected):
    return finite(actual) and finite(expected) and abs(actual-expected) <= 1e-8+1e-8*max(abs(actual), abs(expected))


def exact_sem_exposure(oracle, field, path):
    """Integrate canonical raster density without expanding grid boundaries.

    Fractions retain original binary64 endpoints and canonical raster lines.
    This differs intentionally from the legacy 16-ULP midpoint-band proxy;
    the unchanged closed-cell scorer still supplies hard support and PRI.
    """
    touched = field.path_cells(path)
    if not touched or any(not field.valid[j, i] or not finite(float(field.risk[j, i])) for i, j in touched):
        raise ValueError("missing_sem_support")
    ny, nx = field.risk.shape
    # Field geometry is immutable during a trial. Cache only its exact grid,
    # never risk values or path results on the independently reconstructed field.
    geometry_key = (tuple(field.bounds), field.resolution, nx, ny)
    grid_cache = getattr(field, "_sem_exposure_canonical_grid", None)
    if grid_cache is None or grid_cache[0] != geometry_key:
        axes = tuple(tuple(Fraction(float(field.bounds[axis]+k*field.resolution)) for k in range(size))+
                     (Fraction(float(field.bounds[axis+2])),) for axis, size in ((0, nx), (1, ny)))
        grid_cache = geometry_key, axes
        field._sem_exposure_canonical_grid = grid_cache
    axes = grid_cache[1]

    def incident(value, boundaries):
        k = bisect_left(boundaries, value)
        if k < len(boundaries) and boundaries[k] == value:
            return tuple(index for index in (k-1, k) if 0 <= index < len(boundaries)-1)
        return (k-1,) if 0 < k < len(boundaries) else ()

    def integrate(a, b, raster, intervals):
        segments = sorted(((Fraction(low), Fraction(high), coefficient) for low, high, coefficient in intervals),
                          key=lambda item: item[:2])
        cuts = sorted({Fraction(0), Fraction(1), *raster,
                       *(value for low, high, _ in segments for value in (low, high))})
        delta = [b[k]-a[k] for k in (0, 1)]
        dx, dy = float(delta[0]), float(delta[1])
        active, cursor, parts = [], 0, []
        for low, high in zip(cuts, cuts[1:]):
            if high <= low:
                continue
            midpoint = (low+high)/2
            while cursor < len(segments) and segments[cursor][0] <= midpoint:
                active.append(segments[cursor]); cursor += 1
            active = [item for item in active if item[1] >= midpoint]
            if not active:
                raise ValueError("NoData gap")
            coefficient = active[-1][2]
            xs, ys = (incident(a[k]+midpoint*delta[k], axes[k]) for k in (0, 1))
            if not xs or not ys:
                raise ValueError("path outside shared evaluator")
            cells = [(i, j) for i in xs for j in ys]
            if any(not field.valid[j, i] for i, j in cells):
                raise ValueError("missing_sem_support")
            risk = max(float(field.risk[j, i]) for i, j in cells)
            speed = math.hypot(dx, dy, float(coefficient[0])*dx+float(coefficient[1])*dy)
            parts.append(float(high-low)*risk*speed)
        return math.fsum(parts)

    parts = []
    for aa, bb in (list(zip(path, path[1:])) or [(path[0], path[0])]):
        a, b = [Fraction(float(value)) for value in aa], [Fraction(float(value)) for value in bb]
        if a == b:
            oracle.surface(aa, bb)
            parts.append(0.)
            continue
        raster = set()
        for axis in (0, 1):
            if a[axis] == b[axis]:
                continue
            low, high = sorted((a[axis], b[axis]))
            for boundary in axes[axis][bisect_right(axes[axis], low):bisect_left(axes[axis], high)]:
                raster.add((boundary-a[axis])/(b[axis]-a[axis]))
        try:
            value = integrate(a, b, raster, oracle.intervals(aa, bb))
        except ValueError as error:
            if str(error) != "NoData gap":
                raise
            value = integrate(a, b, raster, _exact_original_intervals(oracle, aa, bb))
        parts.append(value)
    return math.fsum(parts)


def objective_metrics(oracle, field, path, weight):
    """Independent complete-path decomposition; no waypoint risk approximation."""
    if not finite(weight) or weight < 0:
        raise ValueError("invalid SEM exposure weight")
    # Check support even for a singleton or a zero-length edge. A zero measure
    # must not turn undefined terrain/risk into a valid zero-cost path.
    cells = field.path_cells(path)
    if not cells or any(not field.valid[j, i] or not finite(float(field.risk[j, i])) for i, j in cells):
        raise ValueError("missing_sem_support")
    length = math.fsum(_candidate_surface_length(oracle, a, b) for a, b in zip(path, path[1:]))
    if len(path) == 1:
        oracle.surface(path[0], path[0])
    exposure = exact_sem_exposure(oracle, field, path)
    normalized = exposure/40.
    return dict(objective_value=length+weight*normalized, cost_terrain_m=length,
                best_cost_terrain_m=length,
                length_xy_m=math.fsum(math.dist(a, b) for a, b in zip(path, path[1:])),
                sem_exposure_risk_m=exposure, sem_exposure_normalized_m=normalized)


def record_receipt(directory):
    """Only the unchanged corridor baseline requires a guide/mask receipt."""
    from corridor_v4_validation import record_corridor_receipt
    directory = Path(directory)
    if (directory/"worker.json").is_file() and read(directory/"worker.json").get("corridor"):
        return record_corridor_receipt(directory)
    return None


def validate_geometry(oracle, points, start, goal, claimed_cost):
    """Keep full capsule/slope checks; repair only a proven float clipping gap."""
    result = oracle.validate_path(points, start, goal, claimed_cost=claimed_cost,
                                  geometric_safety_metrics=True, clearance_metrics=False)
    if "NoData gap" not in result["errors"]:
        return result
    pairs = list(zip(points, points[1:])) or [(points[0], points[0])]
    try:
        lengths = [_candidate_surface_length(oracle, a, b) for a, b in pairs]
    except ValueError:
        return result  # Real absent intervals remain forbidden.
    errors = [error for error in result["errors"] if error not in ("NoData gap", "cost_mismatch")]
    length = math.fsum(lengths)
    if claimed_cost is not None and not _close(claimed_cost, length):
        errors.append("cost_mismatch")
    max_slope = result["maximum_slope_deg"]
    for a, b in pairs:
        for low, high, coefficient in _exact_original_intervals(oracle, a, b):
            if high > low:
                max_slope = max(max_slope, math.degrees(math.atan(math.hypot(*coefficient[:2]))))
    result.update(valid=not errors, errors=sorted(set(errors)), cost_terrain_m=length,
                  cost_abs_error_m=None if claimed_cost is None else abs(length-claimed_cost),
                  maximum_slope_deg=max_slope,
                  surface_coverage_repair="exact_original_binary64_edge_intervals")
    return result


def _check_tuple(actual, expected, prefix, errors):
    for key in METRICS:
        if not _close(actual.get(key), expected[key]):
            errors.append(prefix+":metric_mismatch:"+key)


def validate_poll_tuples(progress, events, metrics, errors):
    """A poll may refer only to the latest published incumbent at its time."""
    previous = -1.
    for index, poll in enumerate(progress):
        if poll.get("event_type") != "poll":
            continue
        elapsed = poll.get("elapsed_s")
        if not finite(elapsed) or elapsed < max(0., previous):
            errors.append("poll_invalid_time_order")
            continue
        previous = elapsed
        eligible = [event for event in events if event["elapsed_s"] <= elapsed]
        prefix = "poll_"+str(index)
        if eligible:
            sid = eligible[-1]["snapshot_id"]
            if poll.get("snapshot_id") != sid:
                errors.append(prefix+":incumbent_snapshot_mismatch")
            if sid in metrics:
                _check_tuple(poll, metrics[sid], prefix, errors)
            if poll.get("feasible") is not True:
                errors.append(prefix+":incumbent_feasibility_mismatch")
        elif poll.get("snapshot_id") is not None or poll.get("feasible") is not False or any(poll.get(key) is not None for key in METRICS):
            errors.append(prefix+":future_incumbent_backfill")


def _annotate_baseline(row, directory, oracle, field):
    """Report the same independent diagnostic components for every comparator."""
    row["exposure_definition"] = "canonical_raster_integral_risk_ds_surface_v1"
    for event in row.get("events", []):
        if event.get("common_safe", event.get("safe")):
            path = _csv_path(directory/"snapshots"/event["snapshot_id"]/"path_xy.csv")
            event["risk_metrics"] = risk_metrics(field, path)
            event["exposure"] = exact_sem_exposure(oracle, field, path)
            event["exposure_definition"] = "canonical_raster_integral_risk_ds_surface_v1"
    selected = next((event for event in row.get("events", []) if event["snapshot_id"] == row.get("selected_snapshot_id")), None)
    if selected and "risk_metrics" in selected:
        values = selected["risk_metrics"]
        row.update(final_risk_sum=values["risk_sum"], final_risk_count=values["risk_count"],
                   final_risk_quantiles=values["risk_quantiles"], final_exposure=selected["exposure"],
                   final_sem_exposure_normalized_m=selected["exposure"]/40.)
    return row


def validate_trial(directory, request, info, execution, *, oracle=None, field=None):
    from native_paper_oracle import NativePaperOracle
    from native_paper_v3_sem_evaluator import SharedSEMField

    directory = Path(directory)
    for key, hashkey in (("map_path", "map_sha256"), ("cloud_path", "cloud_sha256")):
        if file_hash(info[key]) != info[hashkey]:
            raise ValueError("raw input hash mismatch during SEM exposure validation")
    oracle = oracle or NativePaperOracle(info["map_path"])
    field = field or SharedSEMField(np.load(info["cloud_path"], allow_pickle=False), info["bounds"])
    if request.get("objective_type") != OBJECTIVE:
        from corridor_v4_validation import validate_trial as baseline_validate
        row = baseline_validate(directory, request, info, execution, oracle=oracle, field=field)
        return _annotate_baseline(row, directory, oracle, field)

    parameters = request.get("sem_exposure", {})
    weight = parameters.get("lambda", 1.)
    if not finite(weight) or weight < 0 or parameters.get("risk_scale", 40.) != 40.:
        raise ValueError("invalid SEM exposure request")
    policy = request.get("safe_length", {})
    # A4 refine-attribution: SEM-RRT* incumbents replayed with rpRefine, under the same safety domain.
    replay = "rp_refine_replay" in request
    if (request.get("planner_id") != ("rp_refine_replay" if replay else "bitstar_a3") or policy.get("policy_id") != "a3t_safe_length_v1"
            or policy.get("risk_cap") != 40. or policy.get("slope_cap_deg") != 35.
            or policy.get("atlas_sampling") is not True or "corridor_policy" in policy):
        raise ValueError("unregistered SEM exposure safety domain")
    row = dict(success=False, success_geometry=False, attained_safe=False, first_safe_s=None,
               first_safe_latency_s=None, final_cost_terrain_m=None, final_pri=None,
               final_swept_slope_deg=None, final_exposure=None, final_objective_value=None,
               final_sem_exposure_normalized_m=None, events=[], correctness_errors=[],
               execution_status=execution["status"], planner_status=None,
               field_build_s=execution.get("field_build_s"), solve_time_s=None,
               selection="last_published_incumbent_in_budget_before_validation", oracle_version=VERSION)
    row["exposure_definition"] = "canonical_raster_integral_risk_ds_surface_v1"
    errors = row["correctness_errors"]
    if not (directory/"worker.json").is_file():
        row["unavailable_reason"] = "no_worker_output"
        if execution["status"] == "COMPLETED":
            errors.append("completed_process_missing_worker_output")
        return row
    worker = read(directory/"worker.json")
    row.update(planner_status=worker.get("planner_status"), worker_execution_status=worker.get("execution_status"),
               termination_reason=worker.get("termination_reason"), preprocessing_s=worker.get("preprocessing_s"),
               solve_time_s=worker.get("actual_solve_wall_s"), ompl_version=worker.get("ompl_version"),
               compiler=worker.get("compiler"), effective_risk_cap=40.)
    if worker.get("objective_type") != OBJECTIVE or worker.get("requested_sem_exposure") != parameters:
        errors.append("requested_objective_mismatch")
    for key, expected in (("lambda", weight), ("risk_scale", 40.),
                          ("density_version", "exact_canonical_raster_v1")):
        if worker.get("effective_sem_exposure", {}).get(key) != expected:
            errors.append("effective_objective_mismatch:"+key)
    if worker.get("requested_safe_length") != policy:
        errors.append("requested_safety_policy_mismatch")
    for key, expected in (("policy_id", "a3t_safe_length_v1"), ("risk_cap", 40.),
                          ("slope_cap_deg", 35.), ("atlas_sampling", True),
                          ("thresholds_immutable_during_solve", True)):
        if worker.get("effective_safe_length", {}).get(key) != expected:
            errors.append("effective_safety_policy_mismatch:"+key)
    if worker.get("corridor") or worker.get("exact_certificate", {}).get("enabled"):
        errors.append("unexpected_corridor_or_length_certificate")
    effective = worker.get("effective_policy", {})
    if replay:
        for key, expected in (("motion_objective", OBJECTIVE), ("planner_class", "rp_refine_replay_of_sem_native")):
            if effective.get(key) != expected:
                errors.append("effective_planner_policy_mismatch:"+key)
    elif worker.get("solve_started_utc") or worker.get("planner_status") == "EXACT_SOLUTION":
        for key in ("samples_per_batch", "batch_after_first_solution"):
            if effective.get(key) != request.get(key):
                errors.append("effective_batch_policy_mismatch:"+key)
        if effective.get("safety_atlas_sampling") is not True:
            errors.append("effective_atlas_policy_mismatch")
        if effective.get("informed_cost_bound") != "objective_value_J":
            errors.append("informed_bound_is_not_J")
        for key, expected in (("motion_objective", OBJECTIVE), ("k_nearest", True), ("jit", False),
                              ("planner_class", "ompl::geometric::BITstar")):
            if effective.get(key) != expected:
                errors.append("effective_planner_policy_mismatch:"+key)

    before, total_before = None, None
    if worker.get("solve_started_utc") and execution.get("launch_utc"):
        parse = lambda value: datetime.fromisoformat(value.replace("Z", "+00:00"))
        before = (parse(worker["solve_started_utc"])-parse(execution["launch_utc"])).total_seconds()
        if before < 0:
            raise ValueError("negative startup time")
        if execution.get("started_utc"):
            total_before = (parse(worker["solve_started_utc"])-parse(execution["started_utc"])).total_seconds()
            if total_before < before:
                raise ValueError("attempt started after launch")
    row.update(launch_to_solve_s=before, attempt_to_solve_s=total_before,
               latency_origin="attempt_start_before_input_hash_field_build_and_process_launch")
    events = _events(directory, worker)
    seen, previous_time, previous_cost, selected = set(), -1., math.inf, None
    decompositions = {}
    for event in events:
        sid, elapsed = event["snapshot_id"], event.get("elapsed_s")
        if not isinstance(sid, str) or Path(sid).name != sid or sid in (".", "..") or sid in seen:
            raise ValueError("unsafe or repeated snapshot id")
        seen.add(sid)
        cost = event.get("objective_value")
        if not finite(elapsed) or elapsed < max(0., previous_time) or not finite(cost) or cost > previous_cost+1e-8:
            raise ValueError("invalid incumbent time or objective order")
        points = _csv_path(directory/"snapshots"/sid/"path_xy.csv")
        if event.get("path_xy") is not None and event["path_xy"] != points:
            raise ValueError("snapshot CSV differs from callback coordinates")
        geometry = validate_geometry(oracle, points, request["start"], request["goal"], event.get("cost_terrain_m"))
        sem = field.score_path(points)
        safe = bool(geometry["valid"] and sem["valid"] and sem["risk_max"] <= 40.
                    and geometry["maximum_swept_footprint_slope_deg"] <= 35.)
        if not geometry["valid"]:
            errors.append(sid+":independent_geometry_invalid")
        if not safe:
            errors.append(sid+":hard_safety_contract_failed")
        measured = dict(snapshot_id=sid, elapsed_s=elapsed, in_budget=elapsed <= request["planning_budget_s"],
                        safe=safe, common_safe=safe, geometry_valid=geometry["valid"], geometry=geometry, sem=sem)
        try:
            values = objective_metrics(oracle, field, points, weight)
            decompositions[sid] = values
            _check_tuple(event, values, sid, errors)
            measured.update(objective_metrics=values, exposure=values["sem_exposure_risk_m"],
                            risk_metrics=risk_metrics(field, points))
        except ValueError as error:
            errors.append(sid+":undefined_objective:"+str(error))
        row["events"].append(measured)
        if measured["in_budget"]:
            selected = measured
            if safe and row["first_safe_s"] is None:
                row["first_safe_s"] = elapsed
                if total_before is not None:
                    row["first_safe_latency_s"] = total_before+elapsed
                if before is not None:
                    row["first_safe_launch_latency_s"] = before+elapsed
        previous_time, previous_cost = elapsed, cost

    progress_file = directory/"progress.jsonl"
    if progress_file.is_file():
        progress = [json.loads(line) for line in progress_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        validate_poll_tuples(progress, events, decompositions, errors)
    elif events:
        errors.append("missing_progress_log")
    if events:
        last_sid = events[-1]["snapshot_id"]
        final = directory/"path_xy.csv"
        if not final.is_file() or _csv_path(final) != _csv_path(directory/"snapshots"/last_sid/"path_xy.csv"):
            errors.append("final_path_not_last_published_incumbent")
        if last_sid in decompositions:
            _check_tuple(worker, decompositions[last_sid], "worker_final", errors)
        if request.get("batch_after_first_solution") and worker.get("batch_transition_count") != 1:
            errors.append("adaptive_batch_transition_not_observed")
    elif row["planner_status"] in ("NO_SOLUTION", "INVALID_START", "INVALID_GOAL"):
        if any(worker.get(key) is not None for key in METRICS) or worker.get("path_available"):
            errors.append("no_solution_has_fabricated_final_metrics")
    if selected is not None:
        values, metrics = selected.get("objective_metrics", {}), selected.get("risk_metrics", {})
        row.update(selected_snapshot_id=selected["snapshot_id"], success_geometry=selected["geometry_valid"],
                   final_cost_terrain_m=selected["geometry"]["cost_terrain_m"], final_pri=selected["sem"]["pri"],
                   final_swept_slope_deg=selected["geometry"]["maximum_swept_footprint_slope_deg"],
                   final_exposure=values.get("sem_exposure_risk_m"),
                   final_sem_exposure_normalized_m=values.get("sem_exposure_normalized_m"),
                   final_objective_value=values.get("objective_value"), final_risk_sum=metrics.get("risk_sum"),
                   final_risk_count=metrics.get("risk_count"), final_risk_quantiles=metrics.get("risk_quantiles"))
    elif row["planner_status"] == "EXACT_SOLUTION":
        errors.append("exact_solution_without_in_budget_snapshot")
    row["success"] = bool(selected and selected["safe"] and execution["status"] == "COMPLETED"
                          and row["planner_status"] == "EXACT_SOLUTION" and not errors)
    row["attained_safe"] = bool(row["first_safe_s"] is not None and execution["status"] == "COMPLETED" and not errors)
    row["late_snapshot_count"] = sum(not event["in_budget"] for event in row["events"])
    if execution["status"] != "COMPLETED" or errors:
        row["safe_snapshot_observed_in_failed_attempt"] = row["first_safe_s"] is not None
        row["first_safe_s"], row["first_safe_latency_s"] = None, None
    row["oracle_sources"] = {name: file_hash(ROOT/"tools"/name) for name in
                             ("independent_verifier.py", "native_paper_oracle.py", "native_paper_v3_sem_evaluator.py",
                              "corridor_v3_validation.py", "corridor_v4_validation.py", "sem_exposure_validation.py")}
    return row
