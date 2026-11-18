"""Independent validation of whole-path PRI search, never a planner oracle.

Geometry and raster risk are reconstructed from raw terrain/cloud inputs.  All
archived complete paths are checked, including candidates no longer in the
working pool.  Output selectors are replayed before their independent verdict;
an invalid selected final is never replaced by a different snapshot.
"""
from __future__ import annotations

from datetime import datetime
from fractions import Fraction
import math
from pathlib import Path
import re
import struct

import numpy as np

from pri_portfolio_validation import (
    _check_metrics, _load_records, finite, path_metrics,
    reconstruct_numerical_input, verify_mask_input, sharp_turns,
)
from corridor_v3_validation import risk_metrics
from corridor_v4_validation import _candidate_surface_length
from run_safe_length import _csv_path, read
from safe_length_field import file_hash
from sem_exposure_validation import _close, exact_sem_exposure, validate_geometry

ROOT = Path(__file__).resolve().parents[1]
VERSION = "independent_pri_pathsearch_v1"
BACKEND = "a3t_pri_pathsearch_v1"
ALPHAS = (1., 1.025, 1.05, 1.1)
MODES = ("length", "shortcut", "full", "segment")
POLICY = dict(anchor_fraction=.25, bitstar_slice_fraction=.005,
              optimizer_slice_fraction=.015, internal_length_cap_ratio=1.1,
              output_cap_ratios=list(ALPHAS), length_bins=16, paths_per_bin=4,
              pool_capacity=64, single_persistent_planner=True,
              sem_reference_used=False, deadline_after_scoring=True)


def _edge_key(a, b):
    return struct.pack(">dddd", *a, *b)


class _MemoField:
    """Bounded reuse of the independent raster traversal, with exact endpoints."""
    def __init__(self, field):
        self.source, self.cells, self.hits, self.misses = field, {}, 0, 0

    def __getattr__(self, name):
        return getattr(self.source, name)

    def path_cells(self, path):
        if not path:
            return self.source.path_cells(path)
        unique = set()
        for a, b in (list(zip(path, path[1:])) or [(path[0], path[0])]):
            key = _edge_key(a, b)
            if key not in self.cells:
                self.misses += 1
                if len(self.cells) >= 8192:
                    self.cells.clear()
                self.cells[key] = tuple(self.source.path_cells([a, b]))
            else:
                self.hits += 1
            unique.update(self.cells[key])
        return sorted(unique)

    def score_path(self, path):
        from native_paper_v3_sem_evaluator import SharedSEMField
        return SharedSEMField.score_path(self, path)


class _EdgeGeometry:
    """Every cache miss runs the independent full continuous edge verifier."""
    def __init__(self, oracle):
        self.oracle, self.edges, self.hits, self.misses = oracle, {}, 0, 0

    def path(self, path, start, goal):
        if not path:
            raise ValueError("missing_path")
        pieces = []
        for a, b in (list(zip(path, path[1:])) or [(path[0], path[0])]):
            key = _edge_key(a, b)
            if key not in self.edges:
                self.misses += 1
                if len(self.edges) >= 8192:
                    self.edges.clear()
                length = _candidate_surface_length(self.oracle, a, b)
                self.edges[key] = validate_geometry(self.oracle, [a, b], a, b, length)
            else:
                self.hits += 1
            pieces.append(self.edges[key])
        errors = {error for item in pieces for error in item["errors"]}
        if math.dist(path[0], start) > 1e-9 or math.dist(path[-1], goal) > 1e-9:
            errors.add("not_exact_start_goal")
        return dict(valid=not errors, errors=sorted(errors),
                    cost_terrain_m=math.fsum(p["cost_terrain_m"] for p in pieces),
                    length_xy_m=math.fsum(p["length_xy_m"] for p in pieces),
                    maximum_slope_deg=max(p["maximum_slope_deg"] for p in pieces),
                    maximum_swept_footprint_slope_deg=max(p["maximum_swept_footprint_slope_deg"] for p in pieces),
                    method="independent_exact_endpoint_edge_memo_full_triangle_and_capsule_checks")


def _candidate_id(value):
    match = re.fullmatch(r"candidate_(0|[1-9][0-9]*)", value or "") if isinstance(value, str) else None
    return int(match.group(1)) if match else None


def simple_path(points):
    """Exact orientation checks on original binary64 coordinates.

    Consecutive duplicate endpoints have no geometric extent.  Nonadjacent
    contacts and crossings, and adjacent backtracking overlap, are forbidden.
    Bounding boxes eliminate most pairs before rational orientation work.
    """
    if not points or any(len(p) != 2 or not all(finite(v) for v in p) for p in points):
        return False
    clean = [tuple(p) for i, p in enumerate(points) if not i or p != points[i-1]]
    fractions = [tuple(Fraction(v) for v in point) for point in clean]
    def cross(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    edges = list(zip(clean, clean[1:]))
    for i, (a, b) in enumerate(edges):
        aa, bb = fractions[i:i+2]
        for j in range(i+1, len(edges)):
            c, d = edges[j]
            if (max(a[0], b[0]) < min(c[0], d[0]) or max(c[0], d[0]) < min(a[0], b[0])
                    or max(a[1], b[1]) < min(c[1], d[1]) or max(c[1], d[1]) < min(a[1], b[1])):
                continue
            cc, dd = fractions[j:j+2]
            if j == i+1:
                if cross(aa, bb, dd) == 0 and sum((aa[k]-bb[k])*(dd[k]-bb[k]) for k in (0, 1)) > 0:
                    return False
                continue
            if cross(aa, bb, cc)*cross(aa, bb, dd) <= 0 and cross(cc, dd, aa)*cross(cc, dd, bb) <= 0:
                return False
    return True


def expected_selection(records, worker, request, measured, errors):
    """Replay strict in-budget selectors after independently checking metrics."""
    budget = request["planning_budget_s"]
    mode = request["pri_pathsearch"]["mode"]
    fixed = request.get("test_only_fixed_work") is True
    valid = [r for r in records if r["snapshot_id"] in measured and r.get("feasible") is True
             and finite(r.get("score_completed_s")) and 0 <= r["score_completed_s"] <= budget]
    anchor_rows = [r for r in valid if r.get("move_type") == "bitstar" and r.get("anchor_phase") is True
                   and (fixed or r["score_completed_s"] <= budget*.25)]
    anchor = min(anchor_rows, key=lambda r:(r["cost_terrain_m"], r["pri"], _candidate_id(r["snapshot_id"]))) if anchor_rows else None
    if mode == "length":
        anchor = None
    elif worker.get("anchor_snapshot_id") is None and anchor is not None:
        stopped_before_lock = (worker.get("pathsearch_stage") == "NO_ANCHOR_LOCK_BEFORE_DEADLINE"
                               and finite(worker.get("actual_solve_wall_s")) and worker["actual_solve_wall_s"] >= budget)
        stopped_before_lock = stopped_before_lock or worker.get("optimizer_resource_limit") is True
        if stopped_before_lock:
            anchor = None
    if worker.get("anchor_snapshot_id") != (anchor["snapshot_id"] if anchor else None):
        errors.append("anchor_not_shortest_scored_raw_first_phase")
    output = []
    for alpha in ALPHAS:
        if mode == "length" or anchor is None:
            eligible = [r for r in valid if r.get("move_type") == "bitstar"]
            key = lambda r:(r["cost_terrain_m"], r["pri"], _candidate_id(r["snapshot_id"]))
            cap = None
        else:
            cap = alpha*anchor["cost_terrain_m"]
            eligible = [r for r in valid if r["cost_terrain_m"] <= cap]
            key = lambda r:(r["pri"], r["cost_terrain_m"], _candidate_id(r["snapshot_id"]))
        best = min(eligible, key=key) if eligible else None
        output.append(dict(alpha=alpha, selected_snapshot_id=best["snapshot_id"] if best else None,
                           length_cap_m=cap))
    return output


def _base_row(execution):
    return dict(success=False, success_geometry=False, attained_safe=False, first_safe_s=None,
                first_safe_latency_s=None, final_cost_terrain_m=None, final_pri=None,
                final_swept_slope_deg=None, final_exposure=None, events=[], selector_results=[],
                correctness_errors=[], execution_status=execution["status"], planner_status=None,
                field_build_s=execution.get("field_build_s"), solve_time_s=None, oracle_version=VERSION,
                selection="whole_path_min_PRI_under_frozen_length_cap_before_independent_validation")


def _check_request(request):
    mode = request.get("pri_pathsearch", {}).get("mode")
    policy = request.get("safe_length", {})
    if (mode not in MODES or set(request["pri_pathsearch"]) != {"mode"}
            or request.get("planner_id") != "bitstar_a3" or request.get("objective_type") != "surface_length"
            or policy.get("policy_id") != "a3t_safe_length_v1" or policy.get("risk_cap") != 40.
            or policy.get("slope_cap_deg") != 35. or policy.get("atlas_sampling") is not True
            or not finite(request.get("planning_budget_s")) or request["planning_budget_s"] <= 0):
        raise ValueError("unregistered_pathsearch_request")
    if mode == "segment" and request.get("variant_id") != "a3t_pri_segment_v2":
        raise ValueError("unregistered_segment_backend_label")
    return mode, policy


def _check_policy(worker, request, mode, policy, errors):
    backend = "a3t_pri_segment_v2" if mode == "segment" else BACKEND
    if worker.get("backend_id") != backend or worker.get("objective_type") != "surface_length":
        errors.append("backend_or_objective_mismatch")
    if worker.get("requested_pri_pathsearch") != request["pri_pathsearch"]:
        errors.append("requested_pathsearch_mismatch")
    if mode == "segment" and worker.get("planner_id") != request.get("variant_id"):
        errors.append("segment_planner_label_mismatch")
    if worker.get("state_dimension") != 2 or worker.get("ompl_version") != "1.7.0":
        errors.append("pinned_state_or_ompl_version_mismatch")
    if worker.get("pathsearch_test_work") != request.get("pathsearch_test_work"):
        errors.append("test_work_policy_mismatch")
    effective = worker.get("effective_pri_pathsearch", {})
    for key, value in dict(POLICY, mode=mode).items():
        if effective.get(key) != value:
            errors.append("effective_pathsearch_mismatch:"+key)
    if mode == "segment":
        from pri_segment_protocol import SEGMENT_POLICY
        if effective.get("segment_search") != SEGMENT_POLICY:
            errors.append("effective_segment_search_mismatch")
    if worker.get("requested_safe_length") != policy:
        errors.append("requested_safety_mismatch")
    for key, value in (("policy_id", "a3t_safe_length_v1"), ("risk_cap", 40.),
                       ("slope_cap_deg", 35.), ("atlas_sampling", True)):
        if worker.get("effective_safe_length", {}).get(key) != value:
            errors.append("effective_safety_mismatch:"+key)
    expected = dict(planner_class="ompl::geometric::BITstar", motion_objective="surface_length",
                    k_nearest=True, jit=False, safety_atlas_sampling=True, edge_lower_bound="d_phi",
                    cost_to_go_lower_bound="d_phi", informed_sampling="I_phi",
                    external_length_cap_in_sampler=False, anchor_injected_as_incumbent=False,
                    nearest_neighbor_metric="euclidean_xy", pruning=True)
    for key, value in expected.items():
        if worker.get("effective_policy", {}).get(key) != value:
            errors.append("effective_policy_mismatch:"+key)
    if worker.get("timing_origin") != "before_initial_policy_and_planner_creation":
        errors.append("algorithm_timing_origin_mismatch")
    if worker.get("primary_output_alpha") != 1.1:
        errors.append("primary_output_alpha_mismatch")
    if worker.get("corridor") or worker.get("exact_certificate", {}).get("enabled"):
        errors.append("unexpected_corridor_or_certificate")
    if worker.get("candidate_archive") != "pathsearch_candidates.jsonl":
        errors.append("candidate_archive_name_mismatch")
    for episode in worker.get("episodes", []):
        if episode.get("planner_instance_id") != 1:
            errors.append("nonpersistent_planner_instance")
        if episode.get("anchor_injected") is not False:
            errors.append("external_incumbent_injected")
    if worker.get("planner_instances") not in (0, 1) or (worker.get("planner_status") == "EXACT_SOLUTION" and worker.get("planner_instances") != 1):
        errors.append("planner_instance_count_mismatch")
    counters = worker.get("search_counters", {})
    for key in ("proposals", "shortcut_proposals", "deform_proposals", "splice_proposals", "accepted",
                "pool_evictions", "edge_cache_hits", "edge_evaluations", "incidence_adds", "incidence_removes",
                "full_recomputations", "current_pool_size", "peak_pool_size", "peak_edge_cache_bytes", "archive_bytes"):
        value = counters.get(key)
        if type(value) is not int or value < 0:
            errors.append("invalid_search_counter:"+key)
    operator_counts = ("shortcut_proposals", "deform_proposals", "splice_proposals")
    if mode == "segment":
        operator_counts += ("segment_proposals",)
        if type(counters.get("segment_proposals")) is not int or counters["segment_proposals"] < 0:
            errors.append("invalid_search_counter:segment_proposals")
        if counters.get("splice_proposals") != 0:
            errors.append("segment_mode_legacy_splice_active")
    if all(type(counters.get(k)) is int for k in ("proposals",)+operator_counts):
        if counters["proposals"] != sum(counters[k] for k in operator_counts):
            errors.append("operator_proposal_counter_mismatch")
    if counters.get("peak_pool_size", 0) > 64 or counters.get("current_pool_size", 0) > counters.get("peak_pool_size", 0):
        errors.append("working_pool_bound_exceeded")
    if mode == "length" and counters.get("proposals") != 0:
        errors.append("length_ablation_optimizer_active")
    if mode == "shortcut" and (counters.get("deform_proposals") != 0 or counters.get("splice_proposals") != 0):
        errors.append("shortcut_ablation_other_operator_active")


def _check_segment_lineage(records, worker, measured, paths, request, errors):
    """Replay segment targets and unchanged context independently of the search.

    Strict inequalities on the serialized production values are checked first;
    independently reconstructed values use the established metric agreement
    tolerance only for floating-point differences between the two evaluators.
    """
    attempts = worker.get("segment_attempts")
    if not isinstance(attempts, list) or len(attempts) != worker.get("search_counters", {}).get("segment_proposals"):
        errors.append("segment_attempt_inventory_mismatch")
        return
    by_id = {r["snapshot_id"]:r for r in records}
    anchor = by_id.get(worker.get("anchor_snapshot_id"))
    accepted = {r["proposal_id"]:r for r in records if r.get("move_type") in ("segment_reduce","segment_contract")}
    if len(accepted) != sum(r.get("move_type") in ("segment_reduce","segment_contract") for r in records):
        errors.append("segment_candidate_proposal_duplicate")
    previous_id, previous_finish, seen = 0, 0., set()
    counter_keys = ("nodes_proposed","nodes_retained","edge_evaluations","edge_cache_hits","expansions",
        "labels_created","beam_discarded","distinct_sets_same_node","prefix_above_pri_kept","length_rejected",
        "invalid_edges","non_simple_rejected","completed_paths","pri_rejected","full_recomputations","cancel_checks","soft_stops")
    for index, attempt in enumerate(attempts):
        pid = attempt.get("proposal_id")
        label = "segment_attempt:"+str(pid)
        if type(pid) is not int or pid <= previous_id or pid > worker.get("search_counters", {}).get("proposals",0):
            errors.append(label+":proposal_identity")
        previous_id = pid if type(pid) is int else previous_id
        move = "segment_reduce" if index%2 == 0 else "segment_contract"
        if attempt.get("move_type") != move or (type(pid) is int and pid%4 != (3 if move=="segment_reduce" else 0)):
            errors.append(label+":operator_schedule")
        began, finished = attempt.get("began_s"), attempt.get("finished_s")
        if (not finite(began) or not finite(finished) or began < previous_finish or finished < began
                or began > request["planning_budget_s"] or finished > worker.get("actual_solve_wall_s",0)):
            errors.append(label+":timing_order")
        if finite(finished):
            previous_finish = finished
        counters = attempt.get("counters", {})
        if any(type(counters.get(k)) is not int or counters[k] < 0 for k in counter_keys):
            errors.append(label+":counter_schema")
        if counters.get("nodes_retained",0)>24 or counters.get("expansions",0)>128:
            errors.append(label+":search_bound_exceeded")
        status = attempt.get("status")
        if status not in ("NO_PARENT","IDENTITY","CANCELLED","NO_CANDIDATE","ACCEPTED",
                           "ADMISSION_REJECTED","INVALID_PARENT","IDENTITY_PATH","LENGTH_BOUND"):
            errors.append(label+":unknown_status")
        termination = attempt.get("termination")
        if termination not in ("","WORK_COMPLETED","SLICE_STOP","HARD_CANCEL","EXPANSION_LIMIT"):
            errors.append(label+":unknown_termination")
        if termination=="HARD_CANCEL" and status!="CANCELLED":
            errors.append(label+":hard_cancel_returned_candidate")
        if (status == "ACCEPTED") != (pid in accepted):
            errors.append(label+":accepted_candidate_receipt_mismatch")
        if pid in accepted:
            seen.add(pid)
        if status == "NO_PARENT":
            if anchor is not None:
                errors.append(label+":unexpected_missing_parent")
            continue
        parent = by_id.get(attempt.get("parent_id"))
        if parent is None or anchor is None or not finite(began):
            errors.append(label+":parent_or_anchor_missing")
            continue
        cap_index = (0,1,2,2,3)[(index//2)%5]
        selected_cap = ALPHAS[cap_index]*anchor["cost_terrain_m"]
        eligible = [r for r in records if r.get("score_completed_s",float("inf")) <= began
                    and r.get("cost_terrain_m",float("inf")) <= selected_cap]
        selected = min(eligible,key=lambda r:(r["pri"],r["cost_terrain_m"],_candidate_id(r["snapshot_id"]))) if eligible else None
        if selected is None or selected["snapshot_id"] != parent["snapshot_id"]:
            errors.append(label+":parent_not_current_target_selector")
        rho = parent["pri"]*(.99 if move=="segment_reduce" else 1.)
        cap = selected_cap if move=="segment_reduce" else min(parent["cost_terrain_m"],selected_cap)
        if attempt.get("pri_cap") != rho or attempt.get("length_cap") != cap:
            errors.append(label+":target_definition_mismatch")
        record = accepted.get(pid)
        if record is None:
            continue
        sid = record["snapshot_id"]
        first, last = attempt.get("first"), attempt.get("last")
        parent_path, path = paths.get(parent["snapshot_id"]), paths.get(sid)
        if (record.get("parent_ids") != [parent["snapshot_id"]] or record.get("move_type") != move
                or record.get("segment_pri_cap") != rho or record.get("segment_length_cap") != cap
                or record.get("segment_first") != first or record.get("segment_last") != last):
            errors.append(sid+":segment_lineage_mismatch")
        if (not parent_path or not path or type(first) is not int or type(last) is not int
                or not 0 <= first < last < len(parent_path)):
            errors.append(sid+":segment_window_invalid")
        elif path[:first+1] != parent_path[:first+1] or path[-(len(parent_path)-last):] != parent_path[last:]:
            errors.append(sid+":segment_changed_outside_context")
        if not (began <= record["callback_s"] <= record["score_completed_s"] <= finished):
            errors.append(sid+":segment_candidate_outside_attempt_time")
        values = measured.get(sid)
        if record["pri"]>rho or record["cost_terrain_m"]>cap:
            errors.append(sid+":segment_raw_target_violation")
        if values is not None and any(values[k]>limit and not _close(values[k],limit)
                for k,limit in (("pri",rho),("cost_terrain_m",cap))):
            errors.append(sid+":segment_independent_target_violation")
        if move=="segment_contract" and not record["cost_terrain_m"]<parent["cost_terrain_m"]:
            errors.append(sid+":segment_contract_not_shorter")
    if set(accepted) != seen:
        errors.append("segment_candidate_missing_attempt_receipt")


def validate_trial(directory, request, info, execution, *, oracle=None, field=None, numerical_reference=None):
    """Return one physical row; four selectors share a run, never four trials."""
    if "pri_pathsearch" not in request:
        from pri_portfolio_validation import validate_trial as legacy_validate
        return legacy_validate(directory, request, info, execution, oracle=oracle, field=field,
                               numerical_reference=numerical_reference)
    from native_paper_oracle import NativePaperOracle
    from native_paper_v3_sem_evaluator import SharedSEMField
    directory = Path(directory)
    mode, policy = _check_request(request)
    for pathkey, hashkey in (("map_path", "map_sha256"), ("cloud_path", "cloud_sha256")):
        if file_hash(info[pathkey]) != info[hashkey]:
            raise ValueError("raw input hash changed during pathsearch validation")
    oracle = oracle or NativePaperOracle(info["map_path"])
    field = field or SharedSEMField(np.load(info["cloud_path"], allow_pickle=False), info["bounds"])
    row = _base_row(execution)
    row["mode"] = mode
    if mode == "segment":
        row["oracle_version"] = "independent_pri_segment_v2"
    errors = row["correctness_errors"]
    if not (directory/"worker.json").is_file():
        row["unavailable_reason"] = "no_worker_output"
        if execution["status"] == "COMPLETED":
            errors.append("completed_process_missing_worker_output")
        return row
    worker = read(directory/"worker.json")
    row.update(planner_status=worker.get("planner_status"), worker_execution_status=worker.get("execution_status"),
               solve_time_s=worker.get("actual_solve_wall_s"), preprocessing_s=worker.get("preprocessing_s"),
               termination_reason=worker.get("termination_reason"), ompl_version=worker.get("ompl_version"),
               compiler=worker.get("compiler"), pathsearch_stage=worker.get("pathsearch_stage"),
               search_counters=worker.get("search_counters", {}))
    _check_policy(worker, request, mode, policy, errors)
    try:
        if numerical_reference is None:
            cached = getattr(field, "_pri_numerical_reference", None)
            if cached is None or cached[0] != info["cloud_sha256"]:
                numerical_reference = reconstruct_numerical_input(np.load(info["cloud_path"], allow_pickle=False), info["bounds"])
                field._pri_numerical_reference = (info["cloud_sha256"], numerical_reference)
            else:
                numerical_reference = cached[1]
        _, row["field_input_audit"] = verify_mask_input(field, policy, numerical_reference)
        row["field_input_audit"].update(raw_cloud_sha256=info["cloud_sha256"], raw_map_sha256=info["map_sha256"])
    except (ValueError, OSError, KeyError) as error:
        errors.append("field_input_verification_failed:"+str(error))
    before = total_before = None
    if worker.get("solve_started_utc") and execution.get("launch_utc"):
        parse = lambda value: datetime.fromisoformat(value.replace("Z", "+00:00"))
        before = (parse(worker["solve_started_utc"])-parse(execution["launch_utc"])).total_seconds()
        total_before = (parse(worker["solve_started_utc"])-parse(execution.get("started_utc", execution["launch_utc"]))).total_seconds()
        if before < 0 or total_before < before:
            raise ValueError("invalid timing origin")
    row.update(launch_to_solve_s=before, attempt_to_solve_s=total_before)
    archive = directory/"pathsearch_candidates.jsonl"
    if not archive.is_file():
        errors.append("missing_candidate_archive")
    records = _load_records(archive)
    measured, paths, candidates, checked = {}, {}, {}, {}
    previous_time = -1.
    allowed = {"bitstar"} if mode == "length" else {"bitstar", "shortcut"} if mode == "shortcut" else {"bitstar", "shortcut", "deform", "splice"}
    if mode == "segment":
        allowed = {"bitstar", "shortcut", "deform", "segment_reduce", "segment_contract"}
    exposure_cache = {}
    memo_field, geometry_cache = _MemoField(field), _EdgeGeometry(oracle)
    output_ids = {s.get("selected_snapshot_id") for s in worker.get("selectors", [])}
    for record in records:
        sid = record.get("snapshot_id")
        cid = _candidate_id(sid)
        if cid is None or sid in candidates:
            raise ValueError("unsafe_or_repeated_candidate_id")
        if cid != len(candidates):
            errors.append(sid+":candidate_id_order")
        parents = record.get("parent_ids")
        move = record.get("move_type")
        if move not in allowed:
            errors.append(sid+":operator_not_allowed_in_mode")
        if not isinstance(parents, list) or any(parent not in candidates for parent in parents):
            errors.append(sid+":invalid_parent_provenance")
        elif len(parents) != (0 if move == "bitstar" else 2 if move == "splice" else 1):
            errors.append(sid+":invalid_operator_parent_count")
        candidates[sid] = record
        elapsed = record.get("score_completed_s")
        if not finite(elapsed) or elapsed < max(0., previous_time) or record.get("elapsed_s") != elapsed:
            errors.append(sid+":candidate_time_order")
        if finite(elapsed) and elapsed > request["planning_budget_s"]:
            errors.append(sid+":archived_after_deadline")
        previous_time = elapsed if finite(elapsed) else previous_time
        if not finite(record.get("callback_s")) or (finite(elapsed) and not 0 <= record["callback_s"] <= elapsed):
            errors.append(sid+":invalid_proposal_time")
        if record.get("accepted") is not True or record.get("feasible") is not True:
            errors.append(sid+":archive_contains_unaccepted_candidate")
        if type(record.get("proposal_id")) is not int or record["proposal_id"] < 0:
            errors.append(sid+":invalid_proposal_id")
        if record.get("anchor_phase") and move != "bitstar":
            errors.append(sid+":optimizer_active_during_anchor_phase")
        path_file = directory/"snapshots"/sid/"path_xy.csv"
        if not path_file.is_file():
            errors.append(sid+":missing_candidate_path")
            continue
        path = _csv_path(path_file)
        paths[sid] = path
        if record.get("path_xy") != path:
            errors.append(sid+":inline_path_mismatch")
        if not simple_path(path):
            errors.append(sid+":path_not_simple")
        try:
            geometry = geometry_cache.path(path, request["start"], request["goal"])
            values = dict(**risk_metrics(memo_field, path),
                          cost_terrain_m=geometry["cost_terrain_m"], best_cost_terrain_m=geometry["cost_terrain_m"],
                          objective_value=geometry["cost_terrain_m"], length_xy_m=geometry["length_xy_m"],
                          sharp_turn_count=sharp_turns(path))
            measured[sid] = values
            _check_metrics(record, values, sid, errors)
            sem = memo_field.score_path(path)
            safe = bool(geometry["valid"] and sem["valid"] and sem["risk_max"] <= 40.
                        and geometry["maximum_swept_footprint_slope_deg"] <= 35.)
            if not safe:
                errors.append(sid+":hard_safety_contract_failed")
            if record.get("published_alphas") or sid in output_ids:
                parts = []
                for a, b in (list(zip(path, path[1:])) or [(path[0], path[0])]):
                    key = _edge_key(a, b)
                    if key not in exposure_cache:
                        if len(exposure_cache) >= 8192:
                            exposure_cache.clear()
                        exposure_cache[key] = exact_sem_exposure(oracle, memo_field, [a, b])
                    parts.append(exposure_cache[key])
                values["sem_exposure_risk_m"] = math.fsum(parts)
                values["sem_exposure_normalized_m"] = values["sem_exposure_risk_m"]/40.
                if not finite(record.get("sem_exposure_risk_m")) or not finite(record.get("sem_exposure_normalized_m")):
                    errors.append(sid+":published_or_selected_exposure_unavailable")
            _check_metrics(record, values, sid, errors, required=False)
            checked[sid] = dict(snapshot_id=sid, elapsed_s=elapsed,
                in_budget=finite(elapsed) and 0 <= elapsed <= request["planning_budget_s"],
                safe=safe, common_safe=safe, geometry_valid=geometry["valid"], geometry=geometry, sem=sem,
                risk_metrics={k:values[k] for k in ("pri", "risk_sum", "risk_count", "risk_quantiles")},
                exposure=values.get("sem_exposure_risk_m"), objective_metrics=values,
                move_type=move, parent_ids=parents)
        except (ValueError, IndexError) as error:
            errors.append(sid+":undefined_candidate_metric:"+str(error))
    if mode == "segment":
        _check_segment_lineage(records, worker, measured, paths, request, errors)
        row["segment_attempt_count"] = len(worker.get("segment_attempts", []))
    expected = expected_selection(records, worker, request, measured, errors)
    if worker.get("candidate_count") != len(records) or worker.get("search_counters", {}).get("accepted") != len(records):
        errors.append("accepted_candidate_inventory_mismatch")
    actual = worker.get("selectors", [])
    if not isinstance(actual, list) or len(actual) != len(ALPHAS) or [s.get("alpha") for s in actual] != list(ALPHAS):
        errors.append("selector_inventory_mismatch")
        actual = []
    anchor = worker.get("anchor_snapshot_id")
    if anchor in measured:
        row["anchor_length_m"] = measured[anchor]["cost_terrain_m"]
    if anchor is None and any(r.get("move_type") != "bitstar" for r in records):
        errors.append("optimizer_active_without_anchor")
    locked = worker.get("anchor_locked_s")
    if anchor is not None and mode != "length":
        if not finite(locked) or not 0 <= locked <= request["planning_budget_s"] or locked < candidates.get(anchor, {}).get("score_completed_s", math.inf):
            errors.append("invalid_anchor_lock_time")
    elif locked is not None:
        errors.append("anchor_lock_without_anchor_or_in_length_mode")
    published = worker.get("snapshots", [])
    publication_rows = []
    seen_publications, previous_publication = set(), -1.
    for event in published if isinstance(published, list) else []:
        sid, event_id, alpha, elapsed = event.get("snapshot_id"), event.get("event_id"), event.get("alpha"), event.get("elapsed_s")
        if not isinstance(event_id, str) or not event_id or event_id in seen_publications:
            errors.append("invalid_or_duplicate_publication_id")
        seen_publications.add(event_id)
        if alpha not in ALPHAS:
            errors.append(str(event_id)+":unregistered_publication_alpha")
        if not finite(elapsed) or elapsed < max(0., previous_publication):
            errors.append(str(event_id)+":publication_time_order")
        previous_publication = elapsed if finite(elapsed) else previous_publication
        if sid in candidates and (not finite(elapsed) or elapsed < candidates[sid]["score_completed_s"]):
            errors.append(str(event_id)+":publication_before_scoring")
        if sid in measured:
            _check_metrics(event, measured[sid], str(event_id)+":published", errors)
            for metric in ("sem_exposure_risk_m", "sem_exposure_normalized_m"):
                if not finite(event.get(metric)):
                    errors.append(str(event_id)+":published_metric_unavailable:"+metric)
        reason = event.get("publication_reason")
        if reason not in ("LENGTH_FALLBACK", "ANCHOR_LOCK", "PRI_IMPROVEMENT"):
            errors.append(str(event_id)+":unknown_publication_reason")
        if reason == "ANCHOR_LOCK" and (mode == "length" or not finite(locked) or elapsed != locked):
            errors.append(str(event_id)+":anchor_publication_time_mismatch")
        if reason == "PRI_IMPROVEMENT" and (mode == "length" or not finite(locked) or not finite(elapsed) or elapsed < locked):
            errors.append(str(event_id)+":pri_publication_without_locked_anchor")
        if finite(elapsed) and alpha in ALPHAS:
            available = [r for r in records if r["snapshot_id"] in measured and r["score_completed_s"] <= elapsed]
            if mode == "length" or not finite(locked) or elapsed < locked or (elapsed == locked and reason == "LENGTH_FALLBACK"):
                available = [r for r in available if r["move_type"] == "bitstar"]
                key = lambda r:(r["cost_terrain_m"], r["pri"], _candidate_id(r["snapshot_id"]))
            else:
                cap = alpha*candidates[anchor]["cost_terrain_m"] if anchor in candidates else -math.inf
                available = [r for r in available if r["cost_terrain_m"] <= cap]
                key = lambda r:(r["pri"], r["cost_terrain_m"], _candidate_id(r["snapshot_id"]))
            best = min(available, key=key) if available else None
            if sid != (best["snapshot_id"] if best else None):
                errors.append(str(event_id)+":publication_not_registered_best_available")
        if sid not in checked or not finite(elapsed) or elapsed > request["planning_budget_s"]:
            errors.append(str(event_id)+":published_invalid_or_late")
            continue
        publication_rows.append(dict(checked[sid], event_id=event_id, alpha=alpha, elapsed_s=elapsed,
                                     score_completed_s=candidates[sid]["score_completed_s"],
                                     in_budget=0 <= elapsed <= request["planning_budget_s"]))
    first = min((e["elapsed_s"] for e in publication_rows if e["safe"] and e["in_budget"]), default=None)
    selector_results = []
    for registered in expected:
        alpha, sid = registered["alpha"], registered["selected_snapshot_id"]
        selected = next((s for s in actual if s.get("alpha") == alpha), {})
        if selected.get("selected_snapshot_id") != sid:
            errors.append(f"alpha_{alpha}:final_not_registered_prevalidation_selection")
        if selected.get("length_cap_m") != registered["length_cap_m"]:
            errors.append(f"alpha_{alpha}:length_cap_mismatch")
        final_sid = selected.get("selected_snapshot_id")
        alpha_events = [event for event in publication_rows if event["alpha"] == alpha]
        first_alpha = min((e["elapsed_s"] for e in alpha_events if e["safe"] and e["in_budget"]), default=None)
        if (alpha_events[-1]["snapshot_id"] if alpha_events else None) != final_sid:
            errors.append(f"alpha_{alpha}:final_not_last_publication")
        result = dict(alpha=alpha, selected_snapshot_id=final_sid, length_cap_m=selected.get("length_cap_m"),
                      success=False, attained_safe=False, success_geometry=False,
                      first_safe_s=first_alpha, first_safe_latency_s=total_before+first_alpha if total_before is not None and first_alpha is not None else None,
                      events=alpha_events,
                      final_cost_terrain_m=None, final_pri=None, final_exposure=None, final_swept_slope_deg=None)
        if final_sid in measured and final_sid in checked:
            values, event = measured[final_sid], checked[final_sid]
            _check_metrics(selected, values, f"alpha_{alpha}", errors)
            for metric in ("sem_exposure_risk_m", "sem_exposure_normalized_m"):
                if not finite(selected.get(metric)):
                    errors.append(f"alpha_{alpha}:selected_metric_unavailable:"+metric)
            if selected.get("path_available") is not True:
                errors.append(f"alpha_{alpha}:selected_path_unavailable")
            if not event["in_budget"]:
                errors.append(f"alpha_{alpha}:selected_after_deadline")
            result.update(success_geometry=event["geometry_valid"], final_cost_terrain_m=values["cost_terrain_m"],
                final_pri=values["pri"], final_length_xy_m=values["length_xy_m"], final_sharp_turn_count=values["sharp_turn_count"],
                final_risk_sum=values["risk_sum"], final_risk_count=values["risk_count"], final_risk_quantiles=values["risk_quantiles"],
                final_exposure=values["sem_exposure_risk_m"], final_sem_exposure_normalized_m=values["sem_exposure_normalized_m"],
                final_swept_slope_deg=event["geometry"]["maximum_swept_footprint_slope_deg"],
                success=event["safe"] and event["in_budget"])
        elif final_sid is not None or selected.get("path_available"):
            errors.append(f"alpha_{alpha}:selected_snapshot_unavailable")
        elif any(selected.get(k) is not None for k in ("cost_terrain_m", "pri", "length_xy_m")):
            errors.append(f"alpha_{alpha}:no_solution_fabricated_metrics")
        selector_results.append(result)
    primary = selector_results[-1]
    if worker.get("selected_snapshot_id") != primary["selected_snapshot_id"]:
        errors.append("primary_output_selection_mismatch")
    primary_sid = primary["selected_snapshot_id"]
    if primary_sid in paths:
        if not (directory/"path_xy.csv").is_file() or _csv_path(directory/"path_xy.csv") != paths[primary_sid]:
            errors.append("primary_path_not_selected_snapshot")
        _check_metrics(worker, measured[primary_sid], "worker_final", errors)
    elif worker.get("path_available") or any(worker.get(k) is not None for k in ("cost_terrain_m", "pri", "length_xy_m")):
        errors.append("no_solution_fabricated_primary_metrics")
    if worker.get("planner_status") == "EXACT_SOLUTION" and any(s["selected_snapshot_id"] is None for s in selector_results):
        errors.append("exact_solution_without_all_selected_snapshots")
    published_ids = {r["snapshot_id"] for r in records if r.get("published_alphas")}
    if not isinstance(published, list) or {e.get("snapshot_id") for e in published} != published_ids:
        errors.append("published_snapshot_inventory_mismatch")
    progress = _load_records(directory/"progress.jsonl")
    if not (directory/"progress.jsonl").is_file() or [p for p in progress if p.get("event_type") == "improvement"] != published:
        errors.append("published_progress_differs_from_worker_snapshots")
    for candidate in records:
        alphas = candidate.get("published_alphas")
        if not isinstance(alphas, list) or any(a not in ALPHAS for a in alphas) or len(set(alphas)) != len(alphas):
            errors.append(candidate["snapshot_id"]+":invalid_published_alpha_list")
        elif set(alphas) != {event["alpha"] for event in publication_rows if event["snapshot_id"] == candidate["snapshot_id"]}:
            errors.append(candidate["snapshot_id"]+":published_alpha_inventory_mismatch")
    completed = execution["status"] == "COMPLETED" and worker.get("execution_status") == "COMPLETED"
    for selected in selector_results:
        selected["success"] = bool(selected["success"] and completed and worker.get("planner_status") == "EXACT_SOLUTION" and not errors)
        selected["attained_safe"] = bool(selected["first_safe_s"] is not None and completed and not errors)
        if not completed or errors:
            selected["safe_snapshot_observed_in_failed_attempt"] = first is not None
            selected["first_safe_s"] = selected["first_safe_latency_s"] = None
    row.update({key:value for key,value in selector_results[-1].items() if key != "alpha"})
    row["success"] = all(s["success"] for s in selector_results)
    row["selector_results"] = [] if mode == "length" else selector_results
    row["events"] = [event for event in publication_rows if event["alpha"] == 1.1] if mode == "length" else publication_rows
    row["candidate_count"] = len(records)
    row["independently_checked_candidate_count"] = len(checked)
    row["independent_edge_cache"] = dict(geometry_hits=geometry_cache.hits, geometry_misses=geometry_cache.misses,
                                         raster_hits=memo_field.hits, raster_misses=memo_field.misses,
                                         max_entries_per_cache=8192)
    row["late_snapshot_count"] = sum(not e["in_budget"] for e in checked.values())
    row["oracle_sources"] = {name:file_hash(ROOT/"tools"/name) for name in
        ("pri_pathsearch_validation.py", "pri_portfolio_validation.py", "independent_verifier.py",
         "native_paper_oracle.py", "native_paper_v3_sem_evaluator.py", "sem_exposure_validation.py", "corridor_v4_validation.py")}
    return row
