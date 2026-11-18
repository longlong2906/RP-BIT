"""Independent raw-map/cloud checks for immutable corridor-v3 trials.

No C++ graph, safety traversal, mask, costs or exposure are used as an oracle.
The shared Python evaluator enumerates closed raster cells; continuous geometry
comes from the independent triangle/capsule oracle. Graph optimality concerns
the registered finite graph only, never the continuous planning problem.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import heapq
import json
import math
from pathlib import Path
import time

import numpy as np

from run_safe_length import _csv_path, _events, dump, read
from safe_length_field import file_hash
from safe_length_v2_validation import finite, witness_on_registered_graph

ROOT = Path(__file__).resolve().parents[1]
VERSION = "independent_corridor_v3_v1"
READY = "GUIDANCE_READY"
FAILURES = {"GUIDANCE_NO_CONNECTION", "CORRIDOR_NO_CONNECTION", "INVALID_START",
            "INVALID_GOAL", "GUIDANCE_RESOURCE_LIMIT"}


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def record_corridor_receipt(directory):
    """Bind raw output to the exact inline guide and row-major mask, once."""
    directory = Path(directory)
    source = directory / "worker.json"
    if not source.is_file():
        return None
    corridor = read(source).get("corridor")
    if corridor is None:
        return None
    guide_file = directory/"corridor_guide_xy.csv"
    receipt = dict(worker_sha256=file_hash(source), status=corridor.get("status"),
                   mask_sha256=canonical_sha256(corridor.get("mask", [])),
                   guide_sha256=file_hash(guide_file) if guide_file.is_file() else None,
                   guide_inline_sha256=canonical_sha256(corridor.get("guide_xy", [])),
                   version=VERSION)
    target = directory / "corridor_receipt.json"
    if target.exists():
        if read(target) != receipt:
            raise ValueError("immutable corridor receipt mismatch")
    else:
        dump(target, receipt)
    return receipt


def risk_metrics(field, path):
    cells = field.path_cells(path)
    values = [float(field.risk[j, i]) for i, j in cells]
    if not cells or any(not field.valid[j, i] or not math.isfinite(value)
                        for (i, j), value in zip(cells, values)):
        raise ValueError("missing_sem_support")
    total = math.fsum(values)
    return dict(pri=total / len(values), risk_sum=total, risk_count=len(values),
                risk_quantiles=np.quantile(values, [0., .25, .5, .75, 1.]).tolist())


def reconstruct_mask(field, guide, dilation_cells=1):
    if type(dilation_cells) is not int or dilation_cells < 0:
        raise ValueError("invalid corridor dilation")
    ny, nx = field.risk.shape
    mask = np.zeros((ny, nx), dtype=np.int8)
    for i, j in field.path_cells(guide):
        mask[max(0, j-dilation_cells):min(ny, j+dilation_cells+1),
             max(0, i-dilation_cells):min(nx, i+dilation_cells+1)] = 1
    return mask.ravel().tolist()


def path_in_mask(field, path, mask):
    ny, nx = field.risk.shape
    if (not isinstance(mask, list) or len(mask) != nx*ny or
            any(type(value) is not int or value not in (0, 1) for value in mask)):
        return False
    try:
        cells = field.path_cells(path)
    except (ValueError, TypeError):
        return False
    return bool(cells) and all(mask[j*nx+i] for i, j in cells)


def risk_exposure(oracle, field, path):
    """Integrate risk * surface length by raster cuts and oracle mesh cuts.

    Risk on a positive-length raster boundary is the maximum of all incident
    closed cells. Cells touched at isolated points have zero exposure measure.
    This additive proxy is deliberately separate from unique-cell PRI.
    """
    parts = []
    for aa, bb in zip(path, path[1:]):
        a, b = np.asarray(aa, dtype=float), np.asarray(bb, dtype=float)
        delta = b-a
        cuts = {0., 1.}
        for axis in (0, 1):
            if delta[axis] == 0:
                continue
            origin = field.bounds[axis]
            lo, hi = sorted(((a[axis]-origin)/field.resolution,
                             (b[axis]-origin)/field.resolution))
            for k in range(math.ceil(lo), math.floor(hi)+1):
                t = (origin+k*field.resolution-a[axis])/delta[axis]
                if 0. < t < 1.:
                    cuts.add(float(t))
        ordered = sorted(cuts)
        for lo, hi in zip(ordered, ordered[1:]):
            if hi <= lo:
                continue
            mid = (a+((lo+hi)/2)*delta).tolist()
            cells = field.path_cells([mid])
            if not cells or any(not field.valid[j, i] for i, j in cells):
                raise ValueError("missing_sem_support")
            risk = max(float(field.risk[j, i]) for i, j in cells)
            length, _ = oracle.surface((a+lo*delta).tolist(), (a+hi*delta).tolist())
            parts.append(risk*length)
    return math.fsum(parts)


def _metric_close(actual, expected):
    return finite(actual) and finite(expected) and abs(actual-expected) <= 1e-8+1e-8*max(abs(actual), abs(expected))


def _risk_close(actual, expected):
    return finite(actual) and finite(expected) and abs(actual-expected) <= 64*math.ulp(max(1., abs(actual), abs(expected)))


def _hard_edge(oracle, field, a, b):
    if not oracle.valid_edge(a, b):
        return None
    score = field.score_path([a, b])
    if not score["valid"] or oracle._swept_slope(a, b) > 35.:
        return None
    try:
        return oracle.surface(a, b)[0]
    except ValueError:
        return None


def shortest_graph_length(oracle, field, start, goal, claimed_upper):
    """Independent A* with XY lower bounds on the registered finite graph.

    A claimed upper bound narrows work but never serves as an edge cost. The
    valid guide/shortest claim is separately checked. Reuse only per immutable
    oracle, exact query and raw field object. A direct chord equality supplies
    an independent global lower-bound certificate without graph enumeration.
    """
    cache = getattr(oracle, "_corridor_shortest_cache", {})
    key = (id(field), tuple(start), tuple(goal), float(claimed_upper))
    if key in cache:
        return dict(cache[key], cached=True)
    began = time.perf_counter()
    direct = _hard_edge(oracle, field, start, goal)
    try:
        z0 = oracle.surface(start, start)[1][0][2]
        z1 = oracle.surface(goal, goal)[1][0][2]
        chord = math.sqrt(math.dist(start, goal)**2+(z1-z0)**2)
    except ValueError:
        chord = math.inf
    if direct is not None and _metric_close(direct, chord):
        result = dict(length=direct, method="direct_surface_equals_3d_chord_lower_bound", settled_nodes=0)
    else:
        bounds = field.bounds
        nx = math.floor((bounds[2]-bounds[0])/.1)+1
        ny = math.floor((bounds[3]-bounds[1])/.1)+1
        count = nx*ny
        if count > 2000000:
            raise ValueError("registered independent graph exceeds 2000000 nodes")
        sid, gid = count, count+1
        def point(node):
            if node == sid:
                return start
            if node == gid:
                return goal
            return [bounds[0]+(node % nx)*.1, bounds[1]+(node // nx)*.1]
        def endpoint_links(endpoint):
            result = []
            for j in range(max(0, math.floor((endpoint[1]-bounds[1]-.2)/.1)),
                           min(ny-1, math.ceil((endpoint[1]-bounds[1]+.2)/.1))+1):
                for i in range(max(0, math.floor((endpoint[0]-bounds[0]-.2)/.1)),
                               min(nx-1, math.ceil((endpoint[0]-bounds[0]+.2)/.1))+1):
                    node = j*nx+i
                    candidate = point(node)
                    scale = max(1., *(abs(v) for v in endpoint), *(abs(v) for v in candidate), .2)
                    if math.dist(endpoint, candidate) <= .2+16*math.ulp(scale):
                        result.append(node)
            return result
        slinks, glinks = endpoint_links(start), endpoint_links(goal)
        sset, gset = set(slinks), set(glinks)
        def neighbours(node):
            if node == sid:
                return [*slinks, gid]
            if node == gid:
                return [*glinks, sid]
            i, j = node % nx, node // nx
            return [(j+dj)*nx+i+di for dj in (-1, 0, 1) for di in (-1, 0, 1)
                    if (di or dj) and 0 <= i+di < nx and 0 <= j+dj < ny]+(
                        [sid] if node in sset else [])+([gid] if node in gset else [])
        distance, settled = {sid: 0.}, set()
        queue = [(math.dist(start, goal), 0., sid)]
        tolerance = 1e-8+1e-8*abs(claimed_upper)
        answer = None
        while queue:
            lower, value, node = heapq.heappop(queue)
            if node in settled or value != distance[node]:
                continue
            if lower > claimed_upper+tolerance:
                break
            settled.add(node)
            if node == gid:
                answer = value
                break
            a = point(node)
            for other in neighbours(node):
                if other in settled:
                    continue
                b = point(other)
                cheap = value+math.dist(a, b)
                if cheap+math.dist(b, goal) > claimed_upper+tolerance:
                    continue
                cost = _hard_edge(oracle, field, a, b)
                if cost is None:
                    continue
                candidate = value+cost
                if candidate < distance.get(other, math.inf):
                    distance[other] = candidate
                    heapq.heappush(queue, (candidate+math.dist(b, goal), candidate, other))
        result = dict(length=answer, method="independent_registered_graph_Astar_XY_lower_bound",
                      settled_nodes=len(settled))
    result.update(validation_wall_s=time.perf_counter()-began, cached=False)
    cache[key] = result
    oracle._corridor_shortest_cache = cache
    return result


def validate_corridor(directory, worker, request, oracle, field):
    directory = Path(directory)
    corridor = worker.get("corridor", {})
    errors = []
    result = dict(status=corridor.get("status"), independent_valid=False, errors=errors,
                  no_continuous_infeasibility_claim=True)
    policy = request.get("corridor_policy", request["safe_length"].get("corridor_policy", {}))
    receipt_path = directory/"corridor_receipt.json"
    if not receipt_path.is_file():
        errors.append("missing_corridor_receipt")
    else:
        receipt = read(receipt_path)
        for key, actual in (("worker_sha256", file_hash(directory/"worker.json")),
                            ("status", corridor.get("status")),
                            ("mask_sha256", canonical_sha256(corridor.get("mask", []))),
                            ("guide_inline_sha256", canonical_sha256(corridor.get("guide_xy", [])))):
            if receipt.get(key) != actual:
                errors.append("corridor_receipt_mismatch:"+key)
        result.update(mask_sha256=receipt.get("mask_sha256"), guide_sha256=receipt.get("guide_sha256"))
    for key in ("graph_s", "guidance_s", "mask_s", "total_s"):
        if not finite(corridor.get(key)) or corridor[key] < 0:
            errors.append("invalid_corridor_time:"+key)
    status = corridor.get("status")
    if status != READY:
        if status not in FAILURES:
            errors.append("unknown_corridor_status")
        if (corridor.get("guide_xy") or corridor.get("mask") or _events(directory, worker) or
                worker.get("planner_solve_called") is not False or worker.get("planner_status") == "EXACT_SOLUTION"):
            errors.append("corridor_failure_published_guide_mask_or_solution")
        if status in ("INVALID_START", "INVALID_GOAL"):
            point = request["start" if status == "INVALID_START" else "goal"]
            endpoint = oracle.validate_path([point], point, point,
                                            geometric_safety_metrics=True, clearance_metrics=False)
            if (endpoint["valid"] and field.score_path([point])["valid"] and
                    endpoint["maximum_swept_footprint_slope_deg"] <= 35.):
                errors.append("corridor_claimed_invalid_endpoint_is_valid")
        result["independent_valid"] = not errors
        return result
    for key, expected in (("alpha", policy.get("alpha")), ("label_bins", policy.get("label_bins", 16)),
                          ("dilation_cells", policy.get("dilation_cells", 1)),
                          ("grid_spacing_m", .1), ("endpoint_radius_m", .2),
                          ("max_labels", policy.get("max_labels", 2000000)),
                          ("max_memory_bytes", policy.get("max_memory_bytes", 536870912)),
                          ("guide_used_as_incumbent", False)):
        if corridor.get(key) != expected:
            errors.append("corridor_metadata_mismatch:"+key)
    guide = corridor.get("guide_xy", [])
    guide_file = directory/"corridor_guide_xy.csv"
    if corridor.get("guide_file") != guide_file.name or not guide_file.is_file():
        errors.append("missing_or_unregistered_corridor_guide_csv")
    else:
        if _csv_path(guide_file) != guide:
            errors.append("corridor_guide_csv_inline_mismatch")
        if receipt_path.is_file() and read(receipt_path).get("guide_sha256") != file_hash(guide_file):
            errors.append("corridor_guide_csv_hash_mismatch")
    if not witness_on_registered_graph(guide, request["start"], request["goal"], field.bounds):
        errors.append("guide_not_on_registered_graph")
    # Endpoint-grid zero edges are allowed by the graph. Other repeated
    # coordinates are cycles, even when their risk contribution is zero.
    compressed = [point for k, point in enumerate(guide) if k == 0 or point != guide[k-1]]
    if len({tuple(point) for point in compressed}) != len(compressed):
        errors.append("guide_has_cycle")
    if not guide:
        errors.append("missing_corridor_guide")
        return result
    check = oracle.validate_path(guide, request["start"], request["goal"],
                                 geometric_safety_metrics=True, clearance_metrics=False)
    sem = field.score_path(guide)
    if not check["valid"] or not sem["valid"] or check["maximum_swept_footprint_slope_deg"] > 35.:
        errors.append("independent_guide_hard_safety_failed")
    length = check["cost_terrain_m"]
    L0, alpha = corridor.get("L0"), policy.get("alpha")
    if not finite(L0) or L0 < 0 or not finite(alpha) or alpha < 1:
        errors.append("invalid_guide_length_budget")
    elif length > alpha*L0+1e-8+1e-8*abs(alpha*L0):
        errors.append("guide_exceeds_length_budget")
    if not _metric_close(corridor.get("guide_length"), length):
        errors.append("guide_length_mismatch")
    if finite(L0) and L0 >= 0:
        certificate = shortest_graph_length(oracle, field, request["start"], request["goal"], L0)
        result["L0_certificate"] = certificate
        if not _metric_close(certificate["length"], L0):
            errors.append("L0_not_independent_graph_shortest_length")
    try:
        metrics = risk_metrics(field, guide)
        exposure = risk_exposure(oracle, field, guide)
        for key, measured in (("guide_pri", metrics["pri"]), ("risk_sum", metrics["risk_sum"])):
            if not _risk_close(corridor.get(key), measured):
                errors.append("guide_metric_mismatch:"+key)
        if corridor.get("risk_count") != metrics["risk_count"]:
            errors.append("guide_metric_mismatch:risk_count")
        quantiles = corridor.get("risk_quantiles")
        if (not isinstance(quantiles, list) or len(quantiles) != 5 or
                not all(_risk_close(a, b) for a, b in zip(quantiles, metrics["risk_quantiles"]))):
            errors.append("guide_metric_mismatch:risk_quantiles")
        if not _metric_close(corridor.get("guide_exposure"), exposure):
            errors.append("guide_metric_mismatch:guide_exposure")
        rebuilt = reconstruct_mask(field, guide, policy.get("dilation_cells", 1))
        if corridor.get("mask") != rebuilt:
            errors.append("corridor_mask_not_exact_guide_dilation")
        if not path_in_mask(field, guide, corridor.get("mask")):
            errors.append("guide_outside_corridor_mask")
        result.update(guide_metrics=metrics, guide_exposure=exposure,
                      reconstructed_mask_sha256=canonical_sha256(rebuilt))
    except (ValueError, TypeError) as error:
        errors.append("guide_metrics_unavailable:"+str(error))
    result.update(independent_valid=not errors, geometry=check, sem=sem)
    return result


def validate_trial(directory, request, info, execution, *, oracle=None, field=None):
    if request.get("safe_length", {}).get("policy_id") != "a3t_corridor_v3":
        from safe_length_v2_validation import validate_trial as previous
        return previous(directory, request, info, execution, oracle=oracle, field=field)
    from native_paper_oracle import NativePaperOracle
    from native_paper_v3_sem_evaluator import SharedSEMField
    directory = Path(directory)
    for key, hashkey in (("map_path", "map_sha256"), ("cloud_path", "cloud_sha256")):
        if file_hash(info[key]) != info[hashkey]:
            raise ValueError("raw input hash mismatch during corridor validation")
    row = dict(success=False, success_geometry=False, attained_safe=False, first_safe_s=None,
               first_safe_latency_s=None, final_cost_terrain_m=None, final_pri=None,
               final_swept_slope_deg=None, events=[], correctness_errors=[],
               execution_status=execution["status"], planner_status=None,
               field_build_s=execution.get("field_build_s"), solve_time_s=None,
               selection="last_published_incumbent_in_budget_before_validation")
    if not (directory/"worker.json").is_file():
        row["unavailable_reason"] = "no_worker_output"
        if execution["status"] == "COMPLETED":
            row["correctness_errors"].append("completed_process_missing_worker_output")
        return row
    worker = read(directory/"worker.json")
    oracle = oracle or NativePaperOracle(info["map_path"])
    field = field or SharedSEMField(np.load(info["cloud_path"], allow_pickle=False), info["bounds"])
    errors = row["correctness_errors"]
    policy = request["safe_length"]
    effective = worker.get("effective_safe_length", {})
    row.update(planner_status=worker.get("planner_status", worker.get("status")),
               worker_execution_status=worker.get("execution_status"), termination_reason=worker.get("termination_reason"),
               preprocessing_s=worker.get("preprocessing_s"),
               solve_time_s=worker.get("actual_solve_wall_s"),
               ompl_version=worker.get("ompl_version"), compiler=worker.get("compiler"),
               effective_risk_cap=40., corridor_s=worker.get("corridor", {}).get("total_s"),
               corridor_status=worker.get("corridor", {}).get("status"))
    row.update({key: worker.get("corridor", {}).get(key) for key in
                ("graph_s", "guidance_s", "mask_s", "guide_length", "guide_pri", "guide_exposure", "L0")})
    if worker.get("requested_safe_length") != policy:
        errors.append("requested_safety_policy_mismatch")
    for key, expected in (("policy_id", "a3t_corridor_v3"), ("slope_cap_deg", 35.),
                          ("atlas_sampling", policy.get("atlas_sampling")),
                          ("thresholds_immutable_during_solve", True),
                          ("domain", "immutable_centerline_corridor")):
        if effective.get(key) != expected:
            errors.append("effective_safety_policy_mismatch:"+key)
    checked = validate_corridor(directory, worker, request, oracle, field)
    row["corridor"] = checked
    errors.extend(checked["errors"])
    if checked["status"] == READY:
        if effective.get("risk_cap") != 40.:
            errors.append("effective_safety_policy_mismatch:risk_cap")
        if worker.get("planner_solve_called") is not True:
            errors.append("planner_not_called_after_guidance_ready")
    if effective.get("guide_used_as_incumbent", worker.get("corridor", {}).get("guide_used_as_incumbent")) is not False:
        errors.append("guide_incumbent_contract_missing")
    effective_policy = worker.get("effective_policy", {})
    if ("safety_atlas_sampling" in effective_policy and
            effective_policy["safety_atlas_sampling"] != policy.get("atlas_sampling")):
        errors.append("effective_atlas_policy_mismatch")
    if "batch_policy" in effective_policy:
        for key in ("batch_after_first_solution", "samples_per_batch"):
            if key == "samples_per_batch" and request["planner_id"] != "bitstar_a3":
                continue
            if effective_policy.get(key) != request[key]:
                errors.append("effective_batch_policy_mismatch:"+key)
    before, total_before = None, None
    if worker.get("solve_started_utc") and execution.get("launch_utc"):
        parse = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00"))
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
    mask = worker.get("corridor", {}).get("mask", [])
    for event in events:
        sid = event["snapshot_id"]
        if not isinstance(sid, str) or Path(sid).name != sid or sid in (".", "..") or sid in seen:
            raise ValueError("unsafe or repeated snapshot id")
        seen.add(sid)
        elapsed = event.get("elapsed_s", event.get("timestamp_s"))
        cost = event.get("native_objective_value", event.get("objective_value", event.get("best_cost_terrain_m")))
        if not finite(elapsed) or elapsed < max(0., previous_time) or not finite(cost) or cost > previous_cost+1e-8:
            raise ValueError("invalid incumbent time or cost order")
        path = _csv_path(directory/"snapshots"/sid/"path_xy.csv")
        if event.get("path_xy") is not None and event["path_xy"] != path:
            raise ValueError("snapshot CSV differs from callback coordinates")
        geometry = oracle.validate_path(path, request["start"], request["goal"],
                                        claimed_cost=event.get("cost_terrain_m", event.get("best_cost_terrain_m")),
                                        geometric_safety_metrics=True, clearance_metrics=False)
        sem = field.score_path(path)
        common = bool(geometry["valid"] and sem["valid"] and geometry["maximum_swept_footprint_slope_deg"] <= 35.)
        inside = path_in_mask(field, path, mask)
        safe = common and inside
        if not geometry["valid"]:
            errors.append(sid+":independent_geometry_invalid")
        if not safe:
            errors.append(sid+":corridor_hard_safety_contract_failed")
        if not _metric_close(geometry["cost_terrain_m"], cost):
            errors.append(sid+":surface_length_objective_mismatch")
        measured = dict(snapshot_id=sid, elapsed_s=elapsed, in_budget=elapsed <= request["planning_budget_s"],
                        safe=safe, common_safe=common, geometry_valid=geometry["valid"],
                        geometry=geometry, sem=sem, inside_corridor=inside)
        if common:
            measured.update(risk_metrics=risk_metrics(field, path), exposure=risk_exposure(oracle, field, path))
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
    if selected is not None:
        final = directory/"path_xy.csv"
        if not final.is_file() or _csv_path(final) != _csv_path(directory/"snapshots"/events[-1]["snapshot_id"]/"path_xy.csv"):
            errors.append("final_path_not_last_published_incumbent")
        metrics = selected.get("risk_metrics", {})
        row.update(selected_snapshot_id=selected["snapshot_id"], success_geometry=selected["geometry_valid"],
                   final_cost_terrain_m=selected["geometry"]["cost_terrain_m"], final_pri=selected["sem"]["pri"],
                   final_swept_slope_deg=selected["geometry"]["maximum_swept_footprint_slope_deg"],
                   final_risk_sum=metrics.get("risk_sum"), final_risk_count=metrics.get("risk_count"),
                   final_risk_quantiles=metrics.get("risk_quantiles"), final_exposure=selected.get("exposure"))
    elif row["planner_status"] == "EXACT_SOLUTION":
        errors.append("exact_solution_without_in_budget_snapshot")
    if events and request.get("batch_after_first_solution") and worker.get("batch_transition_count") != 1:
        errors.append("adaptive_batch_transition_not_observed")
    row["success"] = bool(selected is not None and selected["safe"] and execution["status"] == "COMPLETED" and
                          row["planner_status"] == "EXACT_SOLUTION" and not errors)
    row["attained_safe"] = bool(row["first_safe_s"] is not None and execution["status"] == "COMPLETED" and not errors)
    row["late_snapshot_count"] = sum(not e["in_budget"] for e in row["events"])
    if execution["status"] != "COMPLETED" or errors:
        row["safe_snapshot_observed_in_failed_attempt"] = row["first_safe_s"] is not None
        row["first_safe_s"], row["first_safe_latency_s"] = None, None
    row["oracle_sources"] = {name: file_hash(ROOT/"tools"/name) for name in
                             ("independent_verifier.py", "native_paper_oracle.py", "native_paper_v3_sem_evaluator.py",
                              "safe_length_v2_validation.py", "corridor_v3_validation.py")}
    return row
