"""Independent raw-map/cloud validation of the length-capped PRI portfolio.

Physical safety and path metrics use only independently reconstructed raw
data. Mask replay separately uses the exact serialized numerical input, after
independent full-field verification: strict binary64 comparisons can change
components on an otherwise constant plane after one-bit rounding changes.
The planner component traversal and surface-cost routine are never oracles.
In particular, an outer PRI improvement may increase length;
only the frozen cap, whole-path selection and each inner planner objective
are audited. An invalid final cannot be rescued using another snapshot.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
from scipy.ndimage import label

from corridor_v3_validation import risk_metrics
from corridor_v4_validation import _candidate_surface_length
from run_safe_length import _csv_path, _events, read
from safe_length_field import file_hash
from sem_exposure_validation import _close, exact_sem_exposure, validate_geometry

ROOT = Path(__file__).resolve().parents[1]
VERSION = "independent_pri_portfolio_v3_exact_raw_arithmetic_replay"
BACKEND = "a3t_pri_portfolio_v1"
FIELD_ROUNDING_UNITS = 128


class VerifiedMaskInput:
    """Numerical algorithm input only; deliberately has no safety/score API."""
    def __init__(self, raw_field, risk, valid):
        self.risk, self.valid = risk, valid
        self.path_cells = raw_field.path_cells


def reconstruct_numerical_input(points, bounds):
    """Independently replay the documented binary64 field arithmetic.

    Stable sorting groups observations, then scalar left folds retain their
    original order within each cell. This does not use the production indexed
    maxima/bincount or its correlation routine. Python's sum is intentionally
    avoided because recent Python versions may compensate floating sums.
    Filters use the same pinned SciPy primitives as the physical SEM oracle.
    No planner-produced arrays or cached planner metrics are consumed.
    """
    from native_paper_v3_sem_evaluator import _correlate3
    from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter
    cloud = np.asarray(points, dtype=np.float64)
    bounds = tuple(map(float, bounds))
    if len(bounds) != 4 or not all(map(math.isfinite, bounds)) or bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        raise ValueError("numerical_replay_invalid_bounds")
    if cloud.ndim != 2 or cloud.shape[1] != 3 or not len(cloud) or not np.isfinite(cloud).all():
        raise ValueError("numerical_replay_invalid_cloud")
    if np.any(cloud[:, :2] < bounds[:2]) or np.any(cloud[:, :2] > bounds[2:]):
        raise ValueError("numerical_replay_cloud_outside_bounds")
    resolution = .1
    nx, ny = (math.ceil((bounds[i+2]-bounds[i])/resolution) for i in (0, 1))
    ix = np.minimum(nx-1, np.floor((cloud[:, 0]-bounds[0])/resolution).astype(np.int64))
    iy = np.minimum(ny-1, np.floor((cloud[:, 1]-bounds[1])/resolution).astype(np.int64))
    cell_ids = iy*nx+ix
    order = np.argsort(cell_ids, kind="stable")
    ordered_ids, z = cell_ids[order], cloud[order, 2]
    starts = np.r_[0, np.flatnonzero(np.diff(ordered_ids))+1]
    ends = np.r_[starts[1:], len(z)]
    maxima = np.maximum.reduceat(z, starts)
    mean, variance = np.zeros(nx*ny), np.zeros(nx*ny)
    observed = np.zeros(nx*ny, dtype=bool)
    for begin, end, upper in zip(starts, ends, maxima):
        values = z[begin:end]
        kept = values[values > upper-resolution]
        total = 0.
        for value in kept:
            total = total+float(value)
        center = total/len(kept)
        squared = 0.
        for value in kept:
            difference = float(value)-center
            squared = squared+difference*difference
        cell = int(ordered_ids[begin])
        mean[cell], variance[cell], observed[cell] = center, squared/len(kept), True
    mean, variance, observed = (a.reshape(ny, nx) for a in (mean, variance, observed))
    kernels = (
        np.array([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])/4.,
        np.array([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])/4.,
        np.array([[-2., -1., 0.], [-1., 0., 1.], [0., 1., 2.]])/3.,
        np.array([[0., 1., 2.], [-1., 0., 1.], [-2., -1., 0.]])/3.,
    )
    derivatives = [_correlate3(mean, kernel) for kernel in kernels]
    slope = np.degrees(.5*np.arctan(np.hypot(derivatives[0], derivatives[1])/(2*resolution))
                       +.5*np.arctan(np.hypot(derivatives[2], derivatives[3])/(2*math.sqrt(2)*resolution)))
    valid = minimum_filter(observed, size=3, mode="nearest")
    valid = minimum_filter(valid, size=11, mode="nearest")
    valid = minimum_filter(valid, size=11, mode="nearest") & observed
    filtered = maximum_filter(gaussian_filter(slope, sigma=5/3, radius=5, mode="nearest"), size=11, mode="nearest")
    risk = np.where(valid, 200*variance+.75*filtered, np.nan)
    for array in (risk, valid):
        array.setflags(write=False)
    return SimpleNamespace(bounds=bounds, resolution=resolution, risk=risk, valid=valid)


def verify_mask_input(raw_field, policy, numerical_reference):
    """Check every serialized cell against the independent raw-cloud oracle.

    Serialized numbers must equal the independently reconstructed numerical
    reference exactly; even a one-bit change fails. The old 128*epsilon
    envelope remains unchanged as a diagnostic of the different accumulation
    schedules, not an acceptance rule. Support and risk<=40 classification
    must also agree exactly with the physical oracle. Published safety, PRI
    and exposure still use raw_field, never this numerical input reference.
    """
    path = Path(policy["field_path"])
    fingerprint = file_hash(path)
    if fingerprint != policy.get("field_sha256"):
        raise ValueError("mask_input_field_hash_mismatch")
    with path.open("r", encoding="ascii") as stream:
        if stream.readline().strip() != "SAFE_SEM_FIELD_V1":
            raise ValueError("mask_input_field_format_mismatch")
        header = stream.readline().split()
        if len(header) != 7:
            raise ValueError("mask_input_field_header_mismatch")
        nx, ny = int(header[0]), int(header[1])
        grid = tuple(map(float, header[2:]))
        expected_grid = (raw_field.bounds[0], raw_field.bounds[1], raw_field.resolution,
                         raw_field.bounds[2], raw_field.bounds[3])
        if (ny, nx) != raw_field.risk.shape or grid != expected_grid:
            raise ValueError("mask_input_field_grid_mismatch")
        cells = np.loadtxt(stream, dtype=np.float64, ndmin=2)
    if cells.shape != (nx*ny, 2) or not np.isfinite(cells).all():
        raise ValueError("mask_input_field_cells_malformed")
    if not np.isin(cells[:, 0], (0., 1.)).all():
        raise ValueError("mask_input_field_support_malformed")
    valid = cells[:, 0].reshape(ny, nx).astype(bool)
    risk = cells[:, 1].reshape(ny, nx)
    if not np.array_equal(valid, raw_field.valid):
        raise ValueError("mask_input_field_support_mismatch")
    if (numerical_reference.bounds != raw_field.bounds or numerical_reference.resolution != raw_field.resolution
            or numerical_reference.risk.shape != raw_field.risk.shape or not np.array_equal(valid, numerical_reference.valid)):
        raise ValueError("mask_input_numerical_reference_grid_or_support_mismatch")
    if np.any(risk[~valid] != 0.) or np.any(risk[valid] < 0.):
        raise ValueError("mask_input_field_risk_encoding_mismatch")
    reference = raw_field.risk[valid]
    if not np.isfinite(reference).all():
        raise ValueError("mask_input_raw_risk_nonfinite")
    actual = risk[valid]
    if not np.array_equal(actual <= 40., reference <= 40.):
        raise ValueError("mask_input_field_hard_risk_classification_mismatch")
    allowance = FIELD_ROUNDING_UNITS*np.finfo(np.float64).eps*np.maximum(1., np.abs(reference))
    delta = np.abs(actual-reference)
    if not np.array_equal(actual.view(np.uint64), numerical_reference.risk[valid].view(np.uint64)):
        raise ValueError("mask_input_field_exact_arithmetic_replay_mismatch")
    if file_hash(path) != fingerprint:
        raise ValueError("mask_input_field_changed_during_validation")
    risk.setflags(write=False)
    valid.setflags(write=False)
    audit = dict(definition="exact_raw_arithmetic_replay_for_mask_input_v1",
                 field_sha256=fingerprint, supported_cells=int(valid.sum()),
                 different_binary64_risk_cells=int(np.count_nonzero(delta)),
                 maximum_absolute_risk_difference=float(delta.max(initial=0.)),
                 maximum_risk_rounding_allowance=float(allowance.max(initial=0.)),
                 old_rounding_envelope_violation_count=int(np.count_nonzero(delta > allowance)),
                 maximum_old_rounding_envelope_ratio=float(np.divide(delta, allowance).max(initial=0.)),
                 exact_numerical_replay=True, numerical_replay_source="raw_cloud_stable_group_scalar_binary64_left_folds",
                 rounding_units=FIELD_ROUNDING_UNITS, support_exact=True,
                 hard_risk_classification_exact=True, physical_metrics_source="independent_raw_cloud")
    return VerifiedMaskInput(raw_field, risk, valid), audit


def finite(value):
    return type(value) in (float, int) and math.isfinite(value)


def sharp_turns(path):
    points = [p for i, p in enumerate(path) if i == 0 or p != path[i-1]]
    count = 0
    for a, b, c in zip(points, points[1:], points[2:]):
        u, v = np.asarray(b)-a, np.asarray(c)-b
        angle = math.acos(float(np.clip(float(u@v)/(math.hypot(*u)*math.hypot(*v)), -1., 1.)))
        count += angle > math.pi/6
    return count


def path_metrics(oracle, field, path, *, exposure=False):
    risk = risk_metrics(field, path)
    length = math.fsum(_candidate_surface_length(oracle, a, b) for a, b in zip(path, path[1:]))
    if len(path) == 1:
        oracle.surface(path[0], path[0])
    result = dict(**risk, cost_terrain_m=length, best_cost_terrain_m=length, objective_value=length,
                  length_xy_m=math.fsum(math.dist(a, b) for a, b in zip(path, path[1:])),
                  sharp_turn_count=sharp_turns(path))
    if exposure:
        result["sem_exposure_risk_m"] = exact_sem_exposure(oracle, field, path)
        result["sem_exposure_normalized_m"] = result["sem_exposure_risk_m"]/40.
    return result


def reconstruct_masks(field, anchor, start, goal):
    """Independent SciPy component construction on verified numerical input."""
    cells = field.path_cells(anchor)
    ny, nx = field.risk.shape
    anchor_ids = {j*nx+i for i, j in cells}
    endpoint_ids = {j*nx+i for p in (start, goal) for i, j in field.path_cells([p])}
    risks = np.array([field.risk[j, i] for i, j in cells], dtype=float)
    # The registered linear interpolation is left + fraction*(right-left),
    # not NumPy's equivalent alternate lerp branch with different last-bit
    # rounding. Exact rational averaging avoids round(sum)/N double rounding.
    pri = float(sum((Fraction(float(v)) for v in risks), Fraction(0))/len(risks))
    ordered = sorted(map(float, risks))
    def quantile(p):
        position = p*(len(ordered)-1)
        low = int(position)
        return ordered[low]+(position-low)*(ordered[min(low+1, len(ordered)-1)]-ordered[low])
    thresholds = [max(pri, quantile(p)) for p in (.9, .75)]
    rankings = []
    for threshold in thresholds:
        components, _ = label(field.valid & np.isfinite(field.risk) & (field.risk <= 40.) & (field.risk > threshold),
                              structure=np.ones((3, 3), dtype=np.int8))
        flat = components.ravel()
        endpoint_labels = {int(flat[i]) for i in endpoint_ids}
        crossing_labels = {int(flat[i]) for i in anchor_ids}-{0}-endpoint_labels
        ranked = []
        for component in crossing_labels:
            all_ids = np.flatnonzero(flat == component).tolist()
            crossing = sorted(anchor_ids & set(all_ids))
            surplus = math.fsum(float(field.risk.ravel()[i])-pri for i in crossing)
            ranked.append((surplus, all_ids[0], all_ids))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        rankings.append(ranked)
    output, seen = [], set()
    for episode, (threshold, ranking, count) in enumerate(((thresholds[0], rankings[0], 1), (thresholds[1], rankings[1], 1),
                                                          (thresholds[1], rankings[1], 2)), 1):
        excluded = sorted({i for component in ranking[:count] for i in component[2]})
        mask = np.ones(nx*ny, dtype=np.uint8)
        mask[excluded] = 0
        fingerprint = hashlib.sha256(mask.tobytes()).hexdigest()
        reason = "EMPTY" if not excluded else "DUPLICATE" if fingerprint in seen else "READY"
        if reason == "READY":
            seen.add(fingerprint)
        output.append(dict(episode_id=episode, threshold=threshold, excluded_cells=excluded,
                           mask_sha256=fingerprint, oracle_status=reason, nx=nx, ny=ny))
    return output


def _safe_id(value):
    return isinstance(value, str) and Path(value).name == value and value not in (".", "..")


def _load_records(path):
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _check_metrics(actual, expected, prefix, errors, *, required=True):
    keys = ("cost_terrain_m", "best_cost_terrain_m", "length_xy_m", "objective_value", "pri", "risk_sum", "risk_count")
    for key in keys:
        if key not in actual and not required:
            continue
        good = actual.get(key) == expected[key] if key == "risk_count" else _close(actual.get(key), expected[key])
        if not good:
            errors.append(prefix+":metric_mismatch:"+key)
    for key in ("sem_exposure_risk_m", "sem_exposure_normalized_m"):
        if key in actual and actual[key] is not None and key in expected and not _close(actual[key], expected[key]):
            errors.append(prefix+":metric_mismatch:"+key)


def _annotate_native(row, directory, oracle, field):
    """Use the same metrics for the native comparator without imposing A3 caps."""
    cache = {}
    for event in row.get("events", []):
        if event.get("safe", event.get("common_safe")):
            sid = event["snapshot_id"]
            path = _csv_path(directory/"snapshots"/sid/"path_xy.csv")
            # baseline_validate already used the independent canonical oracle;
            # do not repeat its expensive Fraction integration here.
            canonical = event.get("exposure_definition") == "canonical_raster_integral_risk_ds_surface_v1"
            exposure = event.get("exposure") if canonical and finite(event.get("exposure")) else exact_sem_exposure(oracle, field, path)
            risk = event.get("risk_metrics") or risk_metrics(field, path)
            values = dict(**risk, length_xy_m=math.fsum(math.dist(a,b) for a,b in zip(path,path[1:])),
                          sharp_turn_count=sharp_turns(path), sem_exposure_risk_m=exposure, sem_exposure_normalized_m=exposure/40.)
            cache[sid] = values
            event.update(risk_metrics={k:values[k] for k in ("pri", "risk_sum", "risk_count", "risk_quantiles")},
                         exposure=values["sem_exposure_risk_m"])
    selected = row.get("selected_snapshot_id")
    if selected in cache:
        values = cache[selected]
        row.update(final_length_xy_m=values["length_xy_m"], final_sharp_turn_count=values["sharp_turn_count"],
                   final_exposure=values["sem_exposure_risk_m"], final_sem_exposure_normalized_m=values["sem_exposure_normalized_m"],
                   final_risk_sum=values["risk_sum"], final_risk_count=values["risk_count"])
    row["oracle_version"] = VERSION
    return row


def audit_selection(records, worker, request, metrics, errors):
    """Audit the preregistered choice using independently recomputed path stats."""
    budget = request["planning_budget_s"]
    fixed_work = bool(request.get("work_iterations"))
    anchor_id = worker.get("anchor_snapshot_id")
    candidates = {r["snapshot_id"]:r for r in records}
    # Deterministic tie order is replayed on submitted binary64 stats only
    # after those values have independently agreed with the raw-data oracle.
    anchor_phase = [r for r in records if r.get("anchor_phase") and r.get("snapshot_id") in metrics
                    and r.get("feasible") and finite(r.get("score_completed_s")) and r["score_completed_s"] <= (budget if fixed_work else budget/4.)]
    expected_anchor = min(anchor_phase, key=lambda r:(r["cost_terrain_m"], r["episode_id"], r["callback_id"])) if anchor_phase else None
    if (expected_anchor["snapshot_id"] if expected_anchor else None) != anchor_id:
        errors.append("anchor_not_shortest_first_phase_in_budget")
    if anchor_id:
        if anchor_id not in metrics:
            errors.append("missing_anchor_snapshot")
            return None
        expected_cap = request["pri_portfolio"]["length_cap_ratio"]*metrics[anchor_id]["cost_terrain_m"]
        cap = request["pri_portfolio"]["length_cap_ratio"]*candidates[anchor_id]["cost_terrain_m"]
        if not _close(worker.get("length_cap_m"), expected_cap) or worker.get("length_cap_m") != cap:
            errors.append("length_cap_mismatch")
        eligible = [candidates[anchor_id]] if anchor_id in candidates else []
        for r in records:
            sid = r["snapshot_id"]
            if r.get("anchor_phase") or sid not in metrics:
                continue
            in_budget = finite(r.get("score_completed_s")) and 0 <= r["score_completed_s"] <= budget
            inside = r["cost_terrain_m"] <= cap
            path = r.get("path_xy", [])
            exact = bool(path and math.dist(path[0], request["start"]) <= 1e-9 and math.dist(path[-1], request["goal"]) <= 1e-9)
            if bool(r.get("eligible")) != bool(in_budget and inside and r.get("feasible") and exact):
                errors.append(sid+":candidate_admission_mismatch")
            if r.get("eligible") and not (in_budget and inside):
                errors.append(sid+":eligible_outside_cap_or_deadline")
            if r.get("eligible") and in_budget and inside:
                eligible.append(r)
        best = min(eligible, key=lambda r:(r["pri"], r["cost_terrain_m"],
                                           r["episode_id"], r["callback_id"])) if eligible else None
    else:
        if worker.get("length_cap_m") is not None:
            errors.append("cap_without_anchor")
        eligible = []
        for record in records:
            path = record.get("path_xy", [])
            exact = bool(path and math.dist(path[0], request["start"]) <= 1e-9 and math.dist(path[-1], request["goal"]) <= 1e-9)
            qualifies = bool(record["snapshot_id"] in metrics and record.get("feasible") and exact
                             and finite(record.get("score_completed_s")) and 0 <= record["score_completed_s"] <= budget)
            if bool(record.get("eligible")) != qualifies:
                errors.append(record["snapshot_id"]+":fallback_candidate_admission_mismatch")
            if qualifies:
                eligible.append(record)
        best = min(eligible, key=lambda r:(r["cost_terrain_m"], r["episode_id"], r["callback_id"])) if eligible else None
    expected = best["snapshot_id"] if best else None
    if worker.get("selected_snapshot_id") != expected:
        errors.append("final_not_registered_prevalidation_selection")
    return expected


def validate_trial(directory, request, info, execution, *, oracle=None, field=None, numerical_reference=None):
    from native_paper_oracle import NativePaperOracle
    from native_paper_v3_sem_evaluator import SharedSEMField
    directory = Path(directory)
    for pathkey, hashkey in (("map_path", "map_sha256"), ("cloud_path", "cloud_sha256")):
        if file_hash(info[pathkey]) != info[hashkey]:
            raise ValueError("raw input hash changed during portfolio validation")
    oracle = oracle or NativePaperOracle(info["map_path"])
    field = field or SharedSEMField(np.load(info["cloud_path"], allow_pickle=False), info["bounds"])
    if "pri_portfolio" not in request:
        from sem_exposure_validation import validate_trial as baseline_validate
        row = _annotate_native(baseline_validate(directory, request, info, execution, oracle=oracle, field=field), directory, oracle, field)
        if request.get("algorithm_budget_policy") == "pri_portfolio_v1" and (directory/"worker.json").is_file():
            native_worker = read(directory/"worker.json")
            if native_worker.get("solve_started_utc") and (native_worker.get("algorithm_budget_policy") != "pri_portfolio_v1"
                    or native_worker.get("planner_construction_in_algorithm_budget") is not True):
                row["correctness_errors"].append("native_algorithm_budget_policy_mismatch")
                row.update(success=False, attained_safe=False, first_safe_s=None, first_safe_latency_s=None)
        return row
    policy, parameters = request.get("safe_length", {}), request["pri_portfolio"]
    alpha = parameters.get("length_cap_ratio")
    if (not finite(alpha) or alpha < 1. or request.get("planner_id") != "bitstar_a3" or request.get("objective_type") != "surface_length"
            or policy.get("policy_id") != "a3t_safe_length_v1" or policy.get("risk_cap") != 40.
            or policy.get("slope_cap_deg") != 35. or policy.get("atlas_sampling") is not True):
        raise ValueError("unregistered portfolio objective or safety request")
    row = dict(success=False, success_geometry=False, attained_safe=False, first_safe_s=None, first_safe_latency_s=None,
               final_cost_terrain_m=None, final_pri=None, final_swept_slope_deg=None, final_exposure=None,
               events=[], correctness_errors=[], execution_status=execution["status"], planner_status=None,
               field_build_s=execution.get("field_build_s"), solve_time_s=None, oracle_version=VERSION,
               selection="registered_min_PRI_under_frozen_length_cap_before_independent_validation")
    errors = row["correctness_errors"]
    if not (directory/"worker.json").is_file():
        row["unavailable_reason"] = "no_worker_output"
        if execution["status"] == "COMPLETED":
            errors.append("completed_process_missing_worker_output")
        return row
    mask_input = None
    try:
        if numerical_reference is None:
            cached = getattr(field, "_pri_numerical_reference", None)
            if cached is None or cached[0] != info["cloud_sha256"]:
                numerical_reference = reconstruct_numerical_input(np.load(info["cloud_path"], allow_pickle=False), info["bounds"])
                field._pri_numerical_reference = (info["cloud_sha256"], numerical_reference)
            else:
                numerical_reference = cached[1]
        mask_input, row["mask_input_audit"] = verify_mask_input(field, policy, numerical_reference)
        row["mask_input_audit"].update(raw_cloud_sha256=info["cloud_sha256"], raw_map_sha256=info["map_sha256"])
    except (ValueError, OSError, KeyError) as error:
        errors.append("mask_input_verification_failed:"+str(error))
    worker = read(directory/"worker.json")
    row.update(planner_status=worker.get("planner_status"), worker_execution_status=worker.get("execution_status"),
               solve_time_s=worker.get("actual_solve_wall_s"), preprocessing_s=worker.get("preprocessing_s"),
               termination_reason=worker.get("termination_reason"), ompl_version=worker.get("ompl_version"),
               compiler=worker.get("compiler"), length_cap_m=worker.get("length_cap_m"), portfolio_stage=worker.get("portfolio_stage"))
    if worker.get("requested_pri_portfolio") != parameters or worker.get("objective_type") != "surface_length":
        errors.append("requested_portfolio_objective_mismatch")
    if worker.get("requested_safe_length") != policy:
        errors.append("requested_safety_mismatch")
    effective = worker.get("effective_pri_portfolio", {})
    if effective.get("length_cap_ratio") != alpha:
        errors.append("effective_length_cap_ratio_mismatch")
    for key, value in (("risk_cap", 40.), ("slope_cap_deg", 35.), ("atlas_sampling", True)):
        if worker.get("effective_safe_length", {}).get(key) != value:
            errors.append("effective_safety_mismatch:"+key)
    ep = worker.get("effective_policy", {})
    for key, value in (("planner_class", "ompl::geometric::BITstar"), ("motion_objective", "surface_length"),
                       ("k_nearest", True), ("jit", False), ("safety_atlas_sampling", True),
                       ("edge_lower_bound", "d_phi"), ("cost_to_go_lower_bound", "d_phi"), ("informed_sampling", "I_phi"),
                       ("external_length_cap_in_sampler", False), ("anchor_injected_as_incumbent", False)):
        if ep.get(key) != value:
            errors.append("effective_policy_mismatch:"+key)
    if worker.get("exact_certificate", {}).get("enabled"):
        errors.append("unexpected_certificate")
    instances = set()
    for episode in worker.get("episodes", []):
        if episode.get("anchor_injected") is not False:
            errors.append("episode_external_incumbent_injected")
        if episode.get("fresh_planner"):
            if episode.get("planner_instance_id") in instances:
                errors.append("fresh_episode_reuses_planner_instance")
            instances.add(episode.get("planner_instance_id"))
        elif episode.get("episode_id") != 4 or episode.get("planner_instance_id") != 1:
            errors.append("changed_mask_reuses_planner_instance")
    before = total_before = None
    if worker.get("solve_started_utc") and execution.get("launch_utc"):
        parse = lambda value: datetime.fromisoformat(value.replace("Z", "+00:00"))
        before = (parse(worker["solve_started_utc"])-parse(execution["launch_utc"])).total_seconds()
        total_before = (parse(worker["solve_started_utc"])-parse(execution.get("started_utc", execution["launch_utc"]))).total_seconds()
        if before < 0 or total_before < before:
            raise ValueError("invalid timing origin")
    row.update(launch_to_solve_s=before, attempt_to_solve_s=total_before)
    records = _load_records(directory/"portfolio_candidates.jsonl")
    if not (directory/"portfolio_candidates.jsonl").is_file():
        errors.append("missing_candidate_audit")
    measured, paths, seen, previous = {}, {}, set(), -1.
    for record in records:
        sid = record.get("snapshot_id")
        if not _safe_id(sid) or sid in seen:
            raise ValueError("unsafe or repeated candidate id")
        seen.add(sid)
        elapsed = record.get("score_completed_s")
        if not finite(elapsed) or elapsed < max(0., previous) or record.get("elapsed_s") != elapsed:
            errors.append(sid+":candidate_time_order")
        previous = elapsed if finite(elapsed) else previous
        if finite(record.get("callback_s")) and record["callback_s"] > elapsed:
            errors.append(sid+":score_before_callback")
        path_file = directory/"snapshots"/sid/"path_xy.csv"
        if not path_file.is_file():
            errors.append(sid+":missing_candidate_path")
            continue
        path = _csv_path(path_file)
        paths[sid] = path
        if record.get("path_xy") is not None and record["path_xy"] != path:
            errors.append(sid+":inline_path_mismatch")
        try:
            values = path_metrics(oracle, field, path)
            measured[sid] = values
            if record.get("feasible"):
                _check_metrics(record, values, sid, errors)
        except ValueError as error:
            if record.get("eligible") or record.get("published"):
                errors.append(sid+":undefined_path_metric:"+str(error))
    audit_selection(records, worker, request, measured, errors)
    anchor_id = worker.get("anchor_snapshot_id")
    if anchor_id in measured and mask_input is not None:
        row["anchor_length_m"] = measured[anchor_id]["cost_terrain_m"]
        mask_file = directory/"portfolio_masks.json"
        if not mask_file.is_file():
            errors.append("missing_mask_audit")
        else:
            data = read(mask_file)
            expected = reconstruct_masks(mask_input, paths[anchor_id], request["start"], request["goal"])
            actual = data.get("masks", [])
            row["mask_audit"] = []
            if data.get("nx") != field.risk.shape[1] or data.get("ny") != field.risk.shape[0]:
                errors.append("mask_dimensions_mismatch")
            for mask in actual:
                target = next((m for m in expected if m["episode_id"] == mask.get("episode_id")), None)
                if target is None:
                    errors.append("unknown_mask_episode")
                    continue
                prefix = "mask_"+str(mask["episode_id"])
                if not _close(mask.get("threshold"), target["threshold"]):
                    errors.append(prefix+":threshold_mismatch")
                cancelled = mask.get("status") == "SKIPPED_DEADLINE"
                empty = target["oracle_status"] == "EMPTY"
                if cancelled:
                    if any(e.get("episode_id") == mask["episode_id"] and e.get("solve_called") for e in worker.get("episodes", [])) or worker.get("actual_solve_wall_s", 0.) < request["planning_budget_s"]:
                        errors.append(prefix+":invalid_deadline_skip")
                    continue
                expected_status = {"EMPTY":"SKIPPED_NO_COMPONENT", "DUPLICATE":"SKIPPED_DUPLICATE", "READY":"READY"}[target["oracle_status"]]
                if mask.get("status") != expected_status:
                    errors.append(prefix+":mask_status_mismatch")
                if mask.get("excluded_cells") != target["excluded_cells"]:
                    errors.append(prefix+":independent_components_mismatch")
                if (empty and mask.get("mask_sha256") is not None) or (not empty and mask.get("mask_sha256") != target["mask_sha256"]):
                    errors.append(prefix+":independent_mask_hash_mismatch")
                if mask.get("mask_file"):
                    submitted = (directory/mask["mask_file"]).resolve()
                    if directory.resolve() not in submitted.parents or not submitted.is_file() or file_hash(submitted) != target["mask_sha256"]:
                        errors.append(prefix+":mask_file_mismatch")
                row["mask_audit"].append(dict(episode_id=target["episode_id"], expected_status=target["oracle_status"], excluded_count=len(target["excluded_cells"])))
                excluded = set(target["excluded_cells"])
                for candidate in records:
                    sid = candidate["snapshot_id"]
                    if candidate.get("episode_id") == target["episode_id"] and candidate.get("feasible") and sid in paths:
                        touched = {j*data["nx"]+i for i,j in field.path_cells(paths[sid])}
                        if touched & excluded:
                            errors.append(sid+":candidate_crosses_episode_exclusion")
            if len(actual) != 3:
                errors.append("mask_recipe_count_mismatch")
    events = _events(directory, worker)
    progress = _load_records(directory/"progress.jsonl")
    if not (directory/"progress.jsonl").is_file() or [p for p in progress if p.get("event_type") == "improvement"] != events:
        errors.append("published_progress_differs_from_worker_snapshots")
    selected = None
    seen_events, previous_time = set(), -1.
    for event in events:
        sid, elapsed = event.get("snapshot_id"), event.get("elapsed_s")
        if not _safe_id(sid) or sid in seen_events:
            raise ValueError("unsafe or repeated published id")
        seen_events.add(sid)
        if not finite(elapsed) or elapsed < max(0., previous_time):
            errors.append(sid+":published_time_order")
        previous_time = elapsed if finite(elapsed) else previous_time
        if sid not in measured:
            errors.append(sid+":published_without_valid_candidate_stats")
            continue
        record = next((r for r in records if r["snapshot_id"] == sid), None)
        if not record or not record.get("published") or not record.get("feasible") or elapsed != record["score_completed_s"]:
            errors.append(sid+":publication_audit_mismatch")
        path, values = paths[sid], measured[sid]
        _check_metrics(event, values, sid+":published", errors)
        geometry = validate_geometry(oracle, path, request["start"], request["goal"], values["cost_terrain_m"])
        sem = field.score_path(path)
        safe = bool(geometry["valid"] and sem["valid"] and geometry["maximum_swept_footprint_slope_deg"] <= 35.)
        if not safe:
            errors.append(sid+":hard_safety_contract_failed")
        try:
            values["sem_exposure_risk_m"] = exact_sem_exposure(oracle, field, path)
            values["sem_exposure_normalized_m"] = values["sem_exposure_risk_m"]/40.
            _check_metrics(event, values, sid+":published", errors, required=False)
        except ValueError as error:
            errors.append(sid+":undefined_exposure:"+str(error))
        entry = dict(snapshot_id=sid, elapsed_s=elapsed, in_budget=finite(elapsed) and elapsed <= request["planning_budget_s"],
                      safe=safe, common_safe=safe, geometry_valid=geometry["valid"], geometry=geometry, sem=sem,
                      risk_metrics={k:values[k] for k in ("pri", "risk_sum", "risk_count", "risk_quantiles")}, exposure=values.get("sem_exposure_risk_m"))
        row["events"].append(entry)
        if not entry["in_budget"]:
            errors.append(sid+":published_after_deadline")
        if safe and entry["in_budget"] and row["first_safe_s"] is None:
            row["first_safe_s"] = elapsed
            row["first_safe_latency_s"] = total_before+elapsed if total_before is not None else None
        if sid == worker.get("selected_snapshot_id"):
            selected = entry
    if {r["snapshot_id"] for r in records if r.get("published")} != seen_events:
        errors.append("publication_inventory_mismatch")
    if selected:
        sid = selected["snapshot_id"]
        if not events or events[-1]["snapshot_id"] != sid:
            errors.append("final_not_last_published_outer_incumbent")
        final = directory/"path_xy.csv"
        if not final.is_file() or _csv_path(final) != paths[sid]:
            errors.append("final_path_not_selected_snapshot")
        values = measured[sid]
        _check_metrics(worker, values, "worker_final", errors)
        row.update(selected_snapshot_id=sid, success_geometry=selected["geometry_valid"], final_cost_terrain_m=values["cost_terrain_m"],
                   final_pri=values["pri"], final_length_xy_m=values["length_xy_m"], final_sharp_turn_count=values["sharp_turn_count"],
                   final_swept_slope_deg=selected["geometry"]["maximum_swept_footprint_slope_deg"],
                   final_exposure=values.get("sem_exposure_risk_m"), final_sem_exposure_normalized_m=values.get("sem_exposure_normalized_m"),
                   final_risk_sum=values["risk_sum"], final_risk_count=values["risk_count"], final_risk_quantiles=values["risk_quantiles"])
    elif row["planner_status"] == "EXACT_SOLUTION":
        errors.append("exact_solution_without_selected_in_budget_snapshot")
    elif worker.get("path_available") or any(worker.get(k) is not None for k in ("cost_terrain_m", "pri", "length_xy_m")):
        errors.append("no_solution_has_fabricated_metrics")
    row["success"] = bool(selected and selected["safe"] and selected["in_budget"] and execution["status"] == "COMPLETED"
                          and row["planner_status"] == "EXACT_SOLUTION" and not errors)
    row["attained_safe"] = bool(row["first_safe_s"] is not None and execution["status"] == "COMPLETED" and not errors)
    row["late_snapshot_count"] = sum(not r["in_budget"] for r in row["events"])
    row["candidate_count"] = len(records)
    if execution["status"] != "COMPLETED" or errors:
        row["safe_snapshot_observed_in_failed_attempt"] = row["first_safe_s"] is not None
        row["first_safe_s"] = row["first_safe_latency_s"] = None
    row["oracle_sources"] = {name:file_hash(ROOT/"tools"/name) for name in
                             ("pri_portfolio_validation.py", "independent_verifier.py", "native_paper_oracle.py",
                              "native_paper_v3_sem_evaluator.py", "sem_exposure_validation.py", "corridor_v4_validation.py")}
    return row
