"""Reconstruct descriptive pilot results from retained, independently checked paths.

The manifest defines the denominator. No failed or missing trial is imputed, and
the only quality events admitted have a matching snapshot validation and hashes.
Witness targets are frozen separately before any comparison is computed.
"""
import argparse
from collections import Counter, defaultdict
import csv
import itertools
import math
import os
from pathlib import Path
from pathlib import PurePosixPath
import random
import statistics
import tempfile

from experiment_common import digest, dump, file_hash, read_json


COUNTERS = (
    "iterations", "batches", "rewires", "num_prune_calls", "states_removed",
    "branches_disconnected", "graph_prunings_log_observed", "vertices_disconnected_log_observed",
    "samples_pruned_log_observed", "sample_proposals", "samples_returned_before_validity",
    "samples_accepted", "sample_rejected_bounds", "sample_rejected_nodata",
    "sample_rejected_heuristic", "sample_failures", "goal_biased_proposals",
    "finite_bound_sample_proposals", "finite_bound_samples_returned", "finite_bound_rejected_bounds",
    "finite_bound_rejected_nodata", "finite_bound_rejected_heuristic",
    "state_validity_calls", "motion_validity_calls", "full_surface_cost_calls",
    "height_queries", "triangle_visits", "height_cache_hits", "height_cache_misses",
    "fixed_heuristic_cache_hits", "fixed_heuristic_cache_misses", "bvh_node_visits",
    "collision_candidates", "collision_narrow_checks", "batch_transition_count",
    "motion_cost_calls", "traversability_motion_cost_calls",
    "sampler_interval_calls", "sampler_one_bound_calls", "sampler_exhausted_calls",
    "sampler_query_cache_hits", "sampler_query_cache_rebuilds",
    "sampler_bound_cache_hits", "sampler_bound_cache_rebuilds",
    "sampler_invalid_interval_calls", "sampler_lower_bound_rejected_calls",
    "sampler_proposal_limit_calls",
)
TIMINGS = (
    "load_wall_s", "preprocessing_wall_s", "setup_wall_s", "actual_solve_wall_s",
    "serialize_wall_s", "validation_wall_s", "worker_wall_s", "callback_overhead_s",
    "peak_rss_mb", "overshoot_s", "collision_index_build_s",
)
ALIASES = {"graph_prunings_log_observed": ("graph_prunings",),
           "samples_pruned_log_observed": ("samples_pruned",),
           "vertices_disconnected_log_observed": ("vertices_disconnected",),
           "sample_rejected_bounds": ("rejected_bounds",),
           "sample_rejected_nodata": ("rejected_nodata",),
           "sample_rejected_heuristic": ("rejected_heuristic",),
           "sample_failures": ("sampler_exhausted_calls",),
           "load_wall_s": ("load_s",), "preprocessing_wall_s": ("preprocessing_s",),
           "setup_wall_s": ("setup_s",), "serialize_wall_s": ("serialize_s",)}

RAW_INVENTORY_NAME = "raw_artifact_inventory.json"
RAW_INVENTORY_MUTABLE_FILES = {RAW_INVENTORY_NAME, "campaign_state.json", "analysis_attempt.json"}


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _raw_inventory_excluded(relative_path):
    relative_path = PurePosixPath(relative_path).as_posix()
    return relative_path in RAW_INVENTORY_MUTABLE_FILES or relative_path.endswith(".tmp")


def verify_raw_artifact_inventory(source, required=False):
    """Verify the runner's immutable raw-input inventory before analysis.

    The inventory hashes its sorted ``entries`` array rather than hashing its own
    JSON file. Mutable campaign bookkeeping and atomic-write temporary files are
    deliberately outside this immutable analysis input boundary.
    """
    source = Path(source)
    inventory_path = source/RAW_INVENTORY_NAME
    if not inventory_path.exists():
        if required:
            raise ValueError("locked comparison analysis requires raw_artifact_inventory.json")
        return None
    inventory = read_json(inventory_path)
    if inventory.get("schema_version") != 1 or not isinstance(inventory.get("entries"), list):
        raise ValueError("unsupported raw artifact inventory schema")
    entries = inventory["entries"]
    if entries != sorted(entries, key=lambda entry: entry.get("path", "")):
        raise ValueError("raw artifact inventory entries are not sorted by path")
    if digest(entries) != inventory.get("inventory_sha256"):
        raise ValueError("raw artifact inventory digest mismatch")

    listed = set()
    source_root = source.resolve()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"path", "size_bytes", "sha256"}:
            raise ValueError("malformed raw artifact inventory entry")
        relative = entry["path"]
        if not isinstance(relative, str) or not relative:
            raise ValueError("invalid raw artifact inventory path")
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != relative:
            raise ValueError("unsafe/noncanonical raw artifact inventory path: "+relative)
        if _raw_inventory_excluded(relative):
            raise ValueError("mutable/temporary file listed in raw artifact inventory: "+relative)
        if relative in listed:
            raise ValueError("duplicate raw artifact inventory path: "+relative)
        listed.add(relative)
        candidate = source.joinpath(*pure.parts)
        resolved = candidate.resolve()
        if source_root not in resolved.parents or not candidate.is_file():
            raise ValueError("raw artifact inventory file missing/outside source: "+relative)
        size = entry["size_bytes"]
        if not isinstance(size, int) or isinstance(size, bool) or size < 0 or candidate.stat().st_size != size:
            raise ValueError("raw artifact inventory size mismatch: "+relative)
        if file_hash(candidate) != entry["sha256"]:
            raise ValueError("raw artifact inventory hash mismatch: "+relative)

    actual = {
        path.relative_to(source).as_posix()
        for path in source.rglob("*")
        if path.is_file() and not _raw_inventory_excluded(path.relative_to(source).as_posix())
    }
    if actual != listed:
        missing = sorted(actual-listed)
        nonexistent = sorted(listed-actual)
        raise ValueError(f"raw artifact inventory coverage mismatch: unlisted={missing}, missing={nonexistent}")
    return inventory["inventory_sha256"]


def close(a, b, absolute=1e-8, relative=1e-8):
    return finite(a) and finite(b) and abs(a-b) <= absolute+relative*max(abs(a), abs(b))


def metric(trial, name):
    """Unavailable instrumentation never becomes a measured zero."""
    for key in (name, *ALIASES.get(name, ())):
        for obj in (trial, trial.get("counters", {})):
            value = obj.get(key)
            if finite(value):
                return value, None
    reasons = trial.get("unavailable_metrics", {})
    if not isinstance(reasons, dict):
        reasons = {}
    return None, reasons.get(name, trial.get(name+"_unavailable_reason",
                       trial.get("unavailable_counter_reason", "not_measured_or_not_applicable_for_this_planner")))


def distribution(values, total=None):
    values = sorted(v for v in values if finite(v))
    total = len(values) if total is None else total
    def quantile(p):
        x = (len(values)-1)*p
        lo, hi = math.floor(x), math.ceil(x)
        return values[lo]+(values[hi]-values[lo])*(x-lo)
    return {"n": len(values), "unavailable": total-len(values),
            "median": statistics.median(values) if values else None,
            "q25": quantile(.25) if values else None,
            "q75": quantile(.75) if values else None,
            "min": values[0] if values else None,
            "max": values[-1] if values else None}


def checked_events(directory, trial, absolute, relative):
    """Validate the provenance of offline verification, not geometry a second time."""
    directory = Path(directory)
    path = directory/"validated_events.json"
    events, errors = [], []
    if not path.exists():
        if trial.get("path_available"):
            errors.append("missing_validated_events")
        return events, errors
    previous_time, previous_cost = -1., float("inf")
    for event in read_json(path):
        sid = event.get("snapshot_id")
        error = None
        if not isinstance(sid, str) or Path(sid).name != sid:
            error = "invalid_snapshot_id"
        elif not finite(event.get("elapsed_s")) or event["elapsed_s"] < previous_time or event["elapsed_s"] < 0:
            error = "invalid_event_time"
        elif not finite(event.get("best_cost_terrain_m")) or event["best_cost_terrain_m"] < 0:
            error = "invalid_event_cost"
        elif not event.get("independent_valid"):
            error = "invalid_incumbent_snapshot"
        else:
            snapshot = directory/"snapshots"/sid
            try:
                validation = read_json(snapshot/"validation.json")
                if not validation.get("valid"):
                    error = "invalid_snapshot_validation"
                elif file_hash(snapshot/"path_xy.csv") != validation.get("xy_sha256") or file_hash(snapshot/"path_surface.csv") != validation.get("surface_sha256"):
                    error = "snapshot_hash_mismatch"
                elif not close(event["best_cost_terrain_m"], validation.get("cost_terrain_m"), absolute, relative):
                    error = "snapshot_cost_mismatch"
                elif event["best_cost_terrain_m"] > previous_cost and not close(event["best_cost_terrain_m"], previous_cost, absolute, relative):
                    error = "increasing_incumbent"
            except (OSError, ValueError, KeyError) as exc:
                error = "missing_or_invalid_snapshot_validation:"+type(exc).__name__
        if error:
            errors.append({"snapshot_id": sid, "error": error})
            continue
        previous_time, previous_cost = event["elapsed_s"], event["best_cost_terrain_m"]
        events.append(event)
    return events, errors


def read_trial(source, entry, budget, absolute, relative, informed_planners=None, allow_pending_validation=False):
    directory = source/entry["trial_id"]
    path = directory/"trial.json"
    row = dict(entry, executed=False, success=False, status="NOT_RUN", events=[],
               errors=[], time_budget_s=budget, first_solution_time_s=None,
               cost_terrain_m=None, length_xy_m=None, independently_recomputed_cost_m=None,
               improvements_beyond_tolerance=0, metrics={}, metric_unavailable_reasons={})
    if not path.exists():
        return row
    trial = read_json(path)
    for key in ("trial_id", "map_id", "map_hash", "query_id", "planner_id", "planner_seed"):
        if key in entry and trial.get(key) != entry[key]:
            raise ValueError(f"trial identity mismatch: {entry['trial_id']} / {key}")
    row["raw"] = trial
    row["status"] = trial["execution_status"]
    row["executed"] = row["status"] not in ("BLOCKED", "NOT_RUN", "PLANNED")
    row["time_budget_s"] = trial.get("time_budget_s", budget)
    row["termination_reason"] = trial.get("termination_reason", "unknown")
    row["pruning_counter_provenance"] = trial.get("num_prune_calls_provenance")
    validation_path = directory/"validation.json"
    no_request_terminal=trial.get("validation_status")=="NO_REQUEST_INTERRUPTED_OR_BLOCKED"
    wall=trial.get("validation_wall_s")
    committed_validation=(validation_path.exists() and finite(wall) and wall>=0)
    validation_pending = allow_pending_validation and not committed_validation and not no_request_terminal
    events, errors = ([], []) if validation_pending else checked_events(directory, trial, absolute, relative)
    row["events"], row["errors"] = events, errors
    validation = read_json(validation_path) if validation_path.exists() and not validation_pending else {}
    row["offline_validation_available"] = ((committed_validation if allow_pending_validation else validation_path.exists()) or
                                             no_request_terminal)
    row["validation_pending"] = validation_pending
    if trial.get("path_available") and not validation.get("valid") and not validation_pending:
        row["errors"].append("final_path_not_validated")
    in_budget = [e for e in events if e["elapsed_s"] <= row["time_budget_s"]]
    row["success"] = bool(row["executed"] and row["status"] == "COMPLETED" and
                          trial.get("planner_status") == "EXACT_SOLUTION" and
                          trial.get("independent_valid") is True and validation.get("valid") and
                          in_budget and not row["errors"])
    if row["success"]:
        row["first_solution_time_s"] = in_budget[0]["elapsed_s"]
        # Quality at the common budget never uses a callback after the deadline.
        row["cost_terrain_m"] = min(e["best_cost_terrain_m"] for e in in_budget)
    row["final_cost_terrain_m"] = trial.get("best_cost_terrain_m")
    row["length_xy_m"] = trial.get("length_xy_m")
    row["independently_recomputed_cost_m"] = trial.get("independently_recomputed_cost_m")
    row["improvements_beyond_tolerance"] = sum(
        previous["best_cost_terrain_m"]-current["best_cost_terrain_m"] >
        absolute+relative*max(abs(previous["best_cost_terrain_m"]), abs(current["best_cost_terrain_m"]))
        for previous, current in zip(in_budget, in_budget[1:]))
    row["snapshot_events"] = len(events)
    row["events_after_budget"] = len(events)-len(in_budget)
    row["snapshot_quality_usable"] = not row["errors"]
    for name in (*COUNTERS, *TIMINGS):
        value, reason = metric(trial, name)
        row["metrics"][name] = value
        if reason:
            row["metric_unavailable_reasons"][name] = reason
    proposals = row["metrics"]["sample_proposals"]
    returned = row["metrics"]["samples_returned_before_validity"]
    informed = (entry.get("planner_id") != "rrtstar" if informed_planners is None else
                entry.get("planner_id") in informed_planners)
    not_informed_reason=("not_applicable_uniform_rrtstar" if informed_planners is None else
                         "not_applicable_non_informed_planner")
    row["custom_sampler_acceptance_all_phases"] = {
        "value": returned/proposals if informed and finite(proposals) and proposals > 0 and finite(returned) else None,
        "numerator": returned if informed else None, "denominator": proposals if informed else None,
        "definition": "lifetime samples_returned_before_validity / sample_proposals; includes preincumbent uniform-D proposals; excludes goal bias; before footprint/state validity",
        "reason": None if informed and finite(proposals) and proposals > 0 and finite(returned) else
                  not_informed_reason if not informed else "no_measured_positive_proposal_denominator",
        "goal_biased_proposals": row["metrics"]["goal_biased_proposals"],
        "terrain_valid_samples": row["metrics"]["samples_accepted"],
        "terrain_validity_acceptance_ratio": None,
        "terrain_validity_ratio_reason": "state_validity_calls_also_include_nonsampling_queries; no_commensurate_denominator",
    }
    finite_proposals = row["metrics"]["finite_bound_sample_proposals"]
    finite_returns = row["metrics"]["finite_bound_samples_returned"]
    measured_finite = informed and finite(finite_proposals) and finite_proposals > 0 and finite(finite_returns)
    row["informed_acceptance"] = {
        "value": finite_returns/finite_proposals if measured_finite else None,
        "numerator": finite_returns if informed else None, "denominator": finite_proposals if informed else None,
        "definition": "finite_bound_samples_returned / finite_bound_sample_proposals; finite maxCost calls only; excludes goal bias and post-steer/state validity",
        "reason": None if measured_finite else "no_positive_measured_finite_bound_proposal_denominator" if informed else not_informed_reason}
    return row


def coverage(rows):
    executed = [r for r in rows if r["executed"]]
    return {"planned": len(rows), "executed": len(executed),
            "validated_success": sum(r["success"] for r in executed),
            "timeout": sum(r["status"] == "TIMEOUT" for r in executed),
            "invalid": sum(bool(r["errors"]) for r in executed),
            "crash": sum(r["status"] == "CRASH" for r in executed),
            "watchdog": sum(r["status"] == "WATCHDOG" for r in executed),
            "blocked": sum(r["status"] == "BLOCKED" for r in rows),
            "not_run": len(rows)-len(executed),
            "offline_validated": sum(bool(r.get("offline_validation_available")) for r in executed),
            "validation_pending": sum(bool(r.get("validation_pending")) for r in executed),
            "completion_coverage": len(executed)/len(rows) if rows else None,
            "offline_validation_coverage": (sum(bool(r.get("offline_validation_available")) for r in executed)/len(executed)
                                              if executed else None),
            "success_rate_executed": sum(r["success"] for r in executed)/len(executed) if executed else None}


def summarize_group(rows):
    result = coverage(rows)
    executed = [r for r in rows if r["executed"]]
    successes = [r for r in executed if r["success"]]
    result["first_solution_s"] = distribution([r["first_solution_time_s"] for r in successes], len(executed))
    result["cost_terrain_m_at_budget"] = distribution([r["cost_terrain_m"] for r in successes], len(executed))
    result["length_xy_m_final"] = distribution([r["length_xy_m"] for r in successes], len(executed))
    result["independently_recomputed_cost_m_final"] = distribution([r["independently_recomputed_cost_m"] for r in successes], len(executed))
    result["improvements_beyond_tolerance"] = sum(r["improvements_beyond_tolerance"] for r in executed)
    result["trials_with_improvements"] = sum(r["improvements_beyond_tolerance"] > 0 for r in executed)
    result["metrics"] = {name: distribution([r["metrics"].get(name) for r in executed], len(executed)) for name in (*COUNTERS, *TIMINGS)}
    result["termination_reasons"] = dict(Counter(r.get("termination_reason", "unknown") for r in executed))
    result["missing_trial_ids"] = [r["trial_id"] for r in rows if not r["executed"]]
    return result


def group_summaries(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k) for k in keys)].append(row)
    return [dict(zip(keys, key), **summarize_group(group)) for key, group in sorted(groups.items())]


def step_cost(events, time_s):
    past = [e["best_cost_terrain_m"] for e in events if e["elapsed_s"] <= time_s]
    return min(past) if past else None


def attainment_record(row, threshold=None):
    """Censoring is not a fabricated success at the budget endpoint."""
    budget = row["time_budget_s"]
    if not row["executed"]:
        return {"event_time_s": None, "event_observed": False, "observation_end_s": None,
                "observation_end_measurement": None, "observation_end_reason": "trial_not_executed",
                "status": "NOT_RUN"}
    events = row["events"] if row["success"] else []
    eligible = [e for e in events if e["elapsed_s"] <= budget and
                (threshold is None or e["best_cost_terrain_m"] <= threshold)]
    event_time = eligible[0]["elapsed_s"] if eligible else None
    actual = row["metrics"].get("actual_solve_wall_s")
    worker = row["metrics"].get("worker_wall_s")
    if row["status"] in ("CRASH", "WATCHDOG"):
        if finite(actual):
            observation = min(budget, actual)
            observation_measurement = "actual_solve_wall_s"
            observation_reason = None
        elif finite(worker):
            observation = min(budget, worker)
            observation_measurement = "worker_wall_s_fallback"
            observation_reason = "actual_solve_wall_s_unavailable"
        else:
            observation = None
            observation_measurement = None
            observation_reason = "crash_or_watchdog_duration_unavailable"
    else:
        observation = budget
        observation_measurement = "planning_budget_s"
        observation_reason = None
    return {"event_time_s": event_time, "event_observed": event_time is not None,
            "observation_end_s": observation,
            "observation_end_measurement": observation_measurement,
            "observation_end_reason": observation_reason,
            "status": "ATTAINED" if event_time is not None else "RIGHT_CENSORED_NO_EVENT",
            "failure_type": row["status"] if not row["success"] else None}


def freeze_references(manifest, output, absolute, relative, raw_artifact_inventory_sha256=None):
    references = {}
    hash_cache = {}
    def cached_hash(path):
        if path not in hash_cache:
            hash_cache[path] = file_hash(path)
        return hash_cache[path]
    for entry in manifest["trials"]:
        key = entry["map_id"]+"/"+entry["query_id"]
        ref = entry.get("reference")
        if not ref:
            candidate = {"map_id": entry["map_id"], "query_id": entry["query_id"],
                         "map_hash": entry.get("map_hash"), "reference_type": None,
                         "status": "UNAVAILABLE", "reason": "no_reference_in_manifest"}
        else:
            ref = dict(ref)
            validation = ref.get("validation", {})
            if not validation.get("valid") or not finite(ref.get("cost_terrain_m")) or ref["cost_terrain_m"] < 0:
                raise ValueError("reference lacks valid independent verification: "+key)
            if not close(ref["cost_terrain_m"], validation.get("cost_terrain_m"), absolute, relative):
                raise ValueError("reference cost mismatch: "+key)
            if not ref.get("provenance") or not ref.get("reference_type"):
                raise ValueError("reference provenance/type absent: "+key)
            path_hash = digest(ref["path_xy"]) if isinstance(ref["path_xy"], list) else cached_hash(ref["path_xy"])
            if path_hash != ref["path_sha256"]:
                raise ValueError("reference path hash mismatch: "+key)
            if ref.get("map_sha256", entry.get("map_hash")) != entry.get("map_hash"):
                raise ValueError("reference map identity mismatch: "+key)
            if entry.get("map_path") and cached_hash(entry["map_path"]) != entry["map_hash"]:
                raise ValueError("reference map hash mismatch: "+key)
            candidate = dict(ref, map_id=entry["map_id"], query_id=entry["query_id"],
                             map_hash=entry.get("map_hash"), status="FROZEN_VALIDATED_TARGET",
                             interpretation="feasible_target_not_certified_optimum")
        if key in references and references[key] != candidate:
            raise ValueError("reference differs between paired planners: "+key)
        references[key] = candidate
    frozen = {"schema_version": 1, "experiment_id": manifest["experiment_id"],
              "experiment_identity": manifest.get("identity"),
              "policy": "manifest geometric witnesses only; frozen before final comparisons; never selected from pilot winners",
              "references": references}
    if raw_artifact_inventory_sha256 is not None:
        frozen["raw_artifact_inventory_sha256"] = raw_artifact_inventory_sha256
    path = output/"reference_targets.json"
    if path.exists():
        if read_json(path) != frozen:
            raise ValueError("frozen references changed; use a new report directory")
    else:
        dump(path, frozen)
    return references, file_hash(path)


def paired_hierarchical_bootstrap(rows, planner_a, planner_b, metric_name, replicates=2000,
                                  seed=73130, confidence=.95, interpretation=None,
                                  raw_artifact_inventory_sha256=None):
    """Resample maps, then their queries, then paired seed slots within queries.

    Numeric seed labels pair repetitions, not random streams across algorithms.
    Cost/time comparisons explicitly condition on both methods succeeding.
    Missing/unexecuted pairs never become observed failures. Planned slots remain
    in the denominator audit even when neither planner was executed.
    """
    slots = defaultdict(dict)
    for row in rows:
        if row["planner_id"] not in (planner_a, planner_b):
            continue
        key = (row["map_id"], row["query_id"], row["planner_seed"])
        if row["planner_id"] in slots[key]:
            raise ValueError("duplicate planner/seed in paired bootstrap")
        slots[key][row["planner_id"]] = row
    hierarchy = defaultdict(lambda: defaultdict(list))
    both_executed = one_executed = neither_executed = incomplete_planned = unavailable_pair = 0
    for (map_id, query_id, _), pair in slots.items():
        if set(pair) != {planner_a, planner_b}:
            incomplete_planned += 1
            continue
        executed = [bool(pair[planner]["executed"]) for planner in (planner_a, planner_b)]
        if all(executed):
            both_executed += 1
        elif any(executed):
            one_executed += 1
            continue
        else:
            neither_executed += 1
            continue
        a, b = pair[planner_a], pair[planner_b]
        if metric_name == "success":
            value_a, value_b = int(a["success"]), int(b["success"])
        elif a["success"] and b["success"]:
            value_a, value_b = a.get(metric_name), b.get(metric_name)
        else:
            value_a = value_b = None
        if not finite(value_a) or not finite(value_b):
            unavailable_pair += 1
            continue
        hierarchy[map_id][query_id].append(value_a-value_b)
    maps = sorted(hierarchy)
    planned_maps = {map_id for map_id, _, _ in slots}
    planned_queries = {(map_id, query_id) for map_id, query_id, _ in slots}
    units = {"maps": len(maps), "map_query_units": sum(len(v) for v in hierarchy.values()),
             "paired_seed_slots": sum(len(v) for qs in hierarchy.values() for v in qs.values()),
             "distinct_seed_labels": len({r["planner_seed"] for r in rows}),
             "planned_maps": len(planned_maps), "planned_map_query_units": len(planned_queries),
             "planned_seed_slots": len(slots),
             "both_planners_executed_slots": both_executed,
             "one_planner_executed_slots": one_executed,
             "neither_planner_executed_slots": neither_executed,
             "incomplete_planned_slots": incomplete_planned,
             # Backward-compatible name: an unpaired executed slot has exactly
             # one observed planner; slots with neither are now explicit too.
             "unpaired_executed_slots": one_executed,
             "unavailable_complete_pairs": unavailable_pair}
    result = {"planner_a": planner_a, "planner_b": planner_b, "metric": metric_name,
              "difference": "a_minus_b", "units": units, "replicates": replicates,
              "seed": seed, "confidence_level": confidence, "estimate": None, "ci": None,
              "estimand": "equal_map_weight_then_equal_query_weight_mean_paired_seed_difference",
              "conditioning": "all_complete_executed_pairs" if metric_name == "success" else "both_planners_validated_success",
              "interpretation": interpretation or
                  "finite_dataset_paired_bootstrap; paired_seed_labels_do_not_mean_identical_sample_streams"}
    if raw_artifact_inventory_sha256 is not None:
        result["raw_artifact_inventory_sha256"] = raw_artifact_inventory_sha256
    if not maps:
        result["status"] = "UNAVAILABLE_NO_PAIRED_OBSERVATIONS"
        return result
    result["estimate"] = statistics.mean(statistics.mean(statistics.mean(v) for v in hierarchy[m].values()) for m in maps)
    if len(maps) < 2:
        result["status"] = "INSUFFICIENT_INDEPENDENT_MAPS_FOR_CI"
        return result
    rng = random.Random(seed)
    samples = []
    for _ in range(replicates):
        map_means = []
        for m in rng.choices(maps, k=len(maps)):
            queries = sorted(hierarchy[m])
            query_means = []
            for q in rng.choices(queries, k=len(queries)):
                values = hierarchy[m][q]
                query_means.append(statistics.mean(rng.choices(values, k=len(values))))
            map_means.append(statistics.mean(query_means))
        samples.append(statistics.mean(map_means))
    samples.sort()
    alpha = (1-confidence)/2
    result["ci"] = [samples[math.floor(alpha*(replicates-1))], samples[math.ceil((1-alpha)*(replicates-1))]]
    result["status"] = "EXPLORATORY"
    return result


def curves(rows, references, tolerances, raw_artifact_inventory_sha256=None):
    groups = defaultdict(list)
    for row in rows:
        if row["executed"]:
            groups[(row["family"], row["planner_id"])].append(row)
    result = []
    for (family, planner), group in sorted(groups.items()):
        budget = max(r["time_budget_s"] for r in group)
        times = sorted({0., budget, *(e["elapsed_s"] for r in group for e in r["events"] if e["elapsed_s"] <= budget)})
        points = []
        for time_s in times:
            costs, success, targets = [], 0, {str(eps): 0 for eps in tolerances}
            target_denominators = {str(eps): 0 for eps in tolerances}
            for row in group:
                cost = step_cost(row["events"], min(time_s, row["time_budget_s"])) if row["success"] else None
                if cost is not None:
                    costs.append(cost)
                    success += 1
                ref = references[row["map_id"]+"/"+row["query_id"]]
                if ref["status"] == "FROZEN_VALIDATED_TARGET":
                    for eps in tolerances:
                        target_denominators[str(eps)] += 1
                        targets[str(eps)] += int(cost is not None and cost <= (1+eps)*ref["cost_terrain_m"])
            points.append({"time_s": time_s, "median_cost_terrain_m": statistics.median(costs) if costs else None,
                           "cost_status": "AVAILABLE" if costs else "NO_VALIDATED_INCUMBENT",
                           "incumbent_coverage": len(costs)/len(group), "success_attainment": success/len(group),
                           "quality_target_attainment": {eps: targets[eps]/denom if denom else None for eps, denom in target_denominators.items()},
                           "quality_target_denominators": target_denominators})
        record = {"family": family, "planner_id": planner, "executed": len(group), "points": points}
        if raw_artifact_inventory_sha256 is not None:
            record["raw_artifact_inventory_sha256"] = raw_artifact_inventory_sha256
        result.append(record)
    return result


def plot_results(output, rows, curve_data, families, planners):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    colors = {planner: f"C{i}" for i, planner in enumerate(planners)}
    if not any(r["executed"] for r in rows):
        return []
    names = []
    fig, axes = plt.subplots(len(families), 3, figsize=(15, 3.0*len(families)), squeeze=False, layout="constrained")
    for index, family in enumerate(families):
        for data in curve_data:
            if data["family"] != family:
                continue
            x = [p["time_s"] for p in data["points"]]
            for col, key in enumerate(("success_attainment", "median_cost_terrain_m", "incumbent_coverage")):
                y = [p[key] if p[key] is not None else np.nan for p in data["points"]]
                axes[index, col].step(x, y, where="post", color=colors.get(data["planner_id"]), label=f'{data["planner_id"]} n={data["executed"]}')
                axes[index, col].set(xlabel="Solve time (s)", title=family,
                                     ylabel=("Validated success / executed", "Median surface cost (m), available only", "Incumbent coverage")[col])
                if col != 1:
                    axes[index, col].set_ylim(0, 1.02)
        for ax in axes[index]:
            if ax.lines:
                ax.legend(fontsize=7)
    name = "anytime_and_attainment_by_family.png"
    fig.savefig(output/name, dpi=140); plt.close(fig); names.append(name)
    fig, axes = plt.subplots(len(families), 1, figsize=(9, 3*len(families)), squeeze=False, layout="constrained")
    for index, family in enumerate(families):
        ax = axes[index, 0]
        used = set()
        for row in rows:
            if row["family"] != family or not row["success"]:
                continue
            events = [e for e in row["events"] if e["elapsed_s"] <= row["time_budget_s"] and finite(e.get("counters", {}).get("motion_validity_calls"))]
            if not events:
                continue
            planner = row["planner_id"]
            ax.step([e["counters"]["motion_validity_calls"] for e in events], [e["best_cost_terrain_m"] for e in events],
                    where="post", marker=".", alpha=.4, color=colors.get(planner), label=planner if planner not in used else None)
            used.add(planner)
        ax.set(xlabel="Continuous motion validity calls", ylabel="Surface cost (m)", title=family+"; actual verified events")
        if ax.lines:
            ax.legend(fontsize=7)
    name = "quality_by_edge_checks.png"
    fig.savefig(output/name, dpi=140); plt.close(fig); names.append(name)
    fig, axes = plt.subplots(len(families), 1, figsize=(9, 3*len(families)), squeeze=False, layout="constrained")
    for index, family in enumerate(families):
        ax = axes[index, 0]
        for data in curve_data:
            if data["family"] != family:
                continue
            target = "0.05" if "0.05" in data["points"][0]["quality_target_attainment"] else next(iter(data["points"][0]["quality_target_attainment"]), None)
            if target is None:
                continue
            ax.step([p["time_s"] for p in data["points"]],
                    [p["quality_target_attainment"][target] if p["quality_target_attainment"][target] is not None else np.nan for p in data["points"]],
                    where="post", label=data["planner_id"], color=colors.get(data["planner_id"]))
        ax.set(xlabel="Solve time (s)", ylabel="Fraction reaching 1.05 x witness target", title=family+"; no optimality claim", ylim=(0, 1.02))
        if ax.lines:
            ax.legend(fontsize=7)
    name = "witness_target_attainment.png"
    fig.savefig(output/name, dpi=140); plt.close(fig); names.append(name)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
    for col, name in enumerate(("motion_validity_calls", "height_queries", "actual_solve_wall_s")):
        for pi, planner in enumerate(planners):
            medians = []
            for family in families:
                values = [r["metrics"].get(name) for r in rows if r["family"] == family and r["planner_id"] == planner and r["executed"]]
                value = distribution(values)["median"]
                medians.append(value if value is not None else np.nan)
            axes[col].plot(range(len(families)), medians, marker="o", label=planner, color=colors[planner])
        axes[col].set_xticks(range(len(families)), families, rotation=40, ha="right")
        axes[col].set(ylabel="Median "+name, title="All measured executed trials")
        axes[col].legend(fontsize=7)
    name = "effort_by_family.png"
    fig.savefig(output/name, dpi=140); plt.close(fig); names.append(name)
    return names


def _analyze_into(source, output, bootstrap_replicates=None, bootstrap_seed=None, make_plots=True,
                  raw_artifact_inventory_sha256=None, previous_output=None):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = read_json(source/"manifest.json")
    entries = manifest["trials"]
    if len({e["trial_id"] for e in entries}) != len(entries):
        raise ValueError("duplicate trial id in manifest")
    if manifest.get("planned_trials", len(entries)) != len(entries):
        raise ValueError("planned trial count disagrees with manifest")
    config = read_json(source/"resolved_config.json")
    settings = config["analysis"]
    profile_settings=config["profiles"][manifest["profile"]]
    planners=list(profile_settings.get("planners") or dict.fromkeys(entry["planner_id"] for entry in entries))
    if set(planners)!={entry["planner_id"] for entry in entries}:
        raise ValueError("configured planner set disagrees with manifest")
    advanced=config.get("advanced_benchmark")
    combined=config.get("combined_benchmark")
    if advanced and combined:
        raise ValueError("config cannot declare both advanced_benchmark and combined_benchmark")
    benchmark=combined or advanced
    combined_protocol=bool(combined and combined.get("protocol_id")=="a3_vs_all_v1")
    if benchmark:
        primary=benchmark["primary_planner_id"]
        comparators=list(benchmark["comparator_planner_ids"])
        comparisons=[(primary,planner) for planner in comparators]
        informed_planners=set(benchmark["informed_planner_ids"])
        if primary not in planners or set(comparators)!=(set(planners)-{primary}):
            raise ValueError("benchmark primary/comparator planner set disagrees with profile")
        if combined_protocol:
            expected_comparators=["abitstar","aitstar","lazyprmstar_hauser2015","sem_rrtstar_common_v1"]
            expected_informed={"bitstar_a3","abitstar","aitstar","sem_rrtstar_common_v1"}
            if (primary!="bitstar_a3" or comparators!=expected_comparators or
                    informed_planners!=expected_informed or profile_settings.get("planning_budget_s")!=20.0 or
                    combined.get("sem_comparator_label")!="SEM-RRT*-adapted common-objective" or
                    combined.get("official_sem_rrtstar_implementation") is not False or
                    combined.get("terrain_risk_cost") is not False or
                    combined.get("turning_angle_cost") is not False or
                    combined.get("rolling_optimization") is not False):
                raise ValueError("combined benchmark planner/SEM scope differs from locked protocol")
        if bootstrap_replicates is None:bootstrap_replicates=benchmark["bootstrap_replicates"]
        if bootstrap_seed is None:bootstrap_seed=benchmark["bootstrap_seed"]
    else:
        primary="bitstar_a0" if "bitstar_a0" in planners and "bitstar_a3" in planners else planners[0]
        comparators=["bitstar_a3"] if primary=="bitstar_a0" and "bitstar_a3" in planners else planners[1:]
        comparisons=list(itertools.combinations(sorted(planners),2))
        informed_planners=None
        if config.get("planner_variants"):
            informed_planners={label for label in planners
                if config["planner_variants"].get(label,{}).get("planner_id",label)
                not in {"rrtstar","lazyprmstar_hauser2015"}}
        if bootstrap_replicates is None:bootstrap_replicates=2000
        if bootstrap_seed is None:bootstrap_seed=73130
    if settings["bootstrap_unit_order"] != ["map", "query", "seed"]:
        raise ValueError("unsupported bootstrap hierarchy")
    confidence = settings["confidence_level"]
    if not 0 < confidence < 1 or bootstrap_replicates < 2:
        raise ValueError("invalid bootstrap confidence/replicates")
    tolerances = settings["target_relative_tolerances"]
    absolute = config["model"]["numerical_abs_tolerance_m"]
    relative = config["model"]["numerical_rel_tolerance"]
    references, reference_hash = freeze_references(
        manifest, output, absolute, relative, raw_artifact_inventory_sha256)
    if previous_output is not None and (Path(previous_output)/"reference_targets.json").exists():
        previous_frozen = read_json(Path(previous_output)/"reference_targets.json")
        current_frozen = read_json(output/"reference_targets.json")
        # A resumed partial campaign legitimately obtains a new raw inventory as
        # more predeclared trials finish. The witness/reference content must stay
        # frozen; only its binding to the new immutable raw snapshot may change.
        previous_frozen.pop("raw_artifact_inventory_sha256", None)
        current_frozen.pop("raw_artifact_inventory_sha256", None)
        if previous_frozen != current_frozen:
            raise ValueError("frozen references changed; use a new report directory")
    budget = profile_settings["planning_budget_s"]
    rows = [read_trial(source, entry, budget, absolute, relative, informed_planners,
                       allow_pending_validation=bool(benchmark)) for entry in entries]
    summary = coverage(rows)
    summary.update(schema_version=2, experiment_id=manifest["experiment_id"], profile=manifest["profile"],
                   manifest_status=manifest.get("status"), raw_manifest_sha256=file_hash(source/"manifest.json"),
                   raw_artifact_inventory_sha256=raw_artifact_inventory_sha256,
                   reference_targets_sha256=reference_hash,
                   status="PARTIAL" if summary["not_run"] or summary["validation_pending"] else "FAILED_CORRECTNESS" if summary["invalid"] else "COMPLETE",
                   count_semantics="blocked is included in not_run; invalid may overlap execution statuses; success denominator includes all executed failures",
                    interpretation=("held-out finite-budget comparison; exact goal reached plus independent validation is success; finite-budget best known is not a certified global optimum"
                                    if benchmark else "descriptive exploratory pilot; finite-budget exact solution is not a global optimum; few maps cannot establish general superiority"),
                   memory_semantics="peak_rss_mb is a sampled process RSS peak, not an exact OS peak or hard quota",
                   log_format="custom harness JSON/JSONL and raw OMPL output; not Planner Arena format",
                   query_groups=group_summaries(rows, ("family", "map_id", "query_id", "planner_id")),
                   family_groups=group_summaries(rows, ("family", "planner_id")),
                   planner_groups=group_summaries(rows, ("planner_id",)),
                   correctness_errors=[{"trial_id": r["trial_id"], "errors": r["errors"]} for r in rows if r["errors"]])
    summary["planner_ids"]=planners
    summary["primary_planner_id"]=primary
    summary["comparator_planner_ids"]=comparators
    summary["protocol_id"]=benchmark.get("protocol_id") if benchmark else None
    summary["primary_comparisons"]=[{"planner_a":a,"planner_b":b,"difference":"a_minus_b"} for a,b in comparisons
                                    if a==primary and b in comparators]
    summary["bootstrap_replicates"]=bootstrap_replicates
    summary["bootstrap_seed"]=bootstrap_seed
    summary["independent_units"] = {"families": len({e["family"] for e in entries}),
                                    "maps_planned": len({e["map_id"] for e in entries}),
                                    "map_query_units_planned": len({(e["map_id"], e["query_id"]) for e in entries}),
                                    "distinct_seed_labels": len({e["planner_seed"] for e in entries})}
    if combined_protocol:
        summary["combined_design"]={
            "planned_campaign_trials":250,
            "planned_maps":50,
            "planning_budget_s":20.0,
            "planner_seed_labels_per_map":1,
            "stochastic_replication_limitation":"one planner seed per map; uncertainty is across paired maps, not repeated RNG realizations within a map",
        }
        summary["sem_comparator_scope"]={
            "label":combined["sem_comparator_label"],
            "planner_id":"sem_rrtstar_common_v1",
            "official_sem_rrtstar_implementation":False,
            "terrain_risk_cost":False,
            "turning_angle_cost":False,
            "rolling_optimization":False,
            "paper_faithful":False,
        }
    summary["missing_groups"] = [dict((key, group[key]) for key in ("family", "map_id", "query_id", "planner_id", "planned", "executed", "not_run")) for group in summary["query_groups"] if group["not_run"]]
    bootstrap = []
    families = list(dict.fromkeys(e["family"] for e in entries))
    bootstrap_interpretation = (
        f"held_out_finite_budget_{len({e['map_id'] for e in entries})}_map_campaign; "
        "paired_seed_labels_do_not_mean_identical_sample_streams"
        if benchmark else
        "exploratory_pilot_finite_dataset; paired_seed_labels_do_not_mean_identical_sample_streams")
    for family in families:
        group = [r for r in rows if r["family"] == family]
        for a, b in comparisons:
            for measure in ("success", "cost_terrain_m", "first_solution_time_s"):
                bootstrap.append(dict(family=family, **paired_hierarchical_bootstrap(
                    group, a, b, measure, bootstrap_replicates, bootstrap_seed, confidence,
                    bootstrap_interpretation, raw_artifact_inventory_sha256)))
    if combined_protocol:
        overall_interpretation=(
            "a3_vs_all_v1_overall_equal_map_paired_bootstrap_across_50_predeclared_maps; "
            "one_planner_seed_per_map_does_not_measure_within_map_rng_variability")
        for a,b in comparisons:
            for measure in ("success","cost_terrain_m","first_solution_time_s"):
                bootstrap.append(dict(family="ALL", **paired_hierarchical_bootstrap(
                    rows,a,b,measure,bootstrap_replicates,bootstrap_seed,confidence,
                    overall_interpretation,raw_artifact_inventory_sha256)))
    dump(output/"paired_bootstrap.json", bootstrap)
    curve_data = curves(rows, references, tolerances, raw_artifact_inventory_sha256)
    dump(output/"anytime_curves.json", curve_data)
    records = []
    for row in rows:
        record = {k: v for k, v in row.items() if k not in ("raw", "events", "reference")}
        record["raw_artifact_inventory_sha256"] = raw_artifact_inventory_sha256
        ref = references[row["map_id"]+"/"+row["query_id"]]
        record["success_attainment"] = attainment_record(row)
        record["quality_targets"] = []
        if ref["status"] == "FROZEN_VALIDATED_TARGET":
            for eps in tolerances:
                threshold = (1+eps)*ref["cost_terrain_m"]
                record["quality_targets"].append(dict(relative_tolerance=eps, cost_threshold_m=threshold,
                                                       reference_type=ref["reference_type"], **attainment_record(row, threshold)))
            record["reference_relative_difference"] = ((row["cost_terrain_m"]-ref["cost_terrain_m"])/ref["cost_terrain_m"]
                                                        if row["success"] and ref["cost_terrain_m"] > 0 else None)
            record["reference_difference_reason"] = "zero_reference_cost_ratio_undefined" if ref["cost_terrain_m"] == 0 else None
        else:
            record["reference_relative_difference"] = None
            record["reference_difference_reason"] = "missing_reference_no_gap_or_quality_target"
        records.append(record)
    dump(output/"trial_metrics.json", records)
    fields = ("trial_id", "family", "map_id", "query_id", "planner_id", "planner_seed", "status", "executed", "success",
              "raw_artifact_inventory_sha256",
              "first_solution_time_s", "cost_terrain_m", "final_cost_terrain_m", "length_xy_m", "independently_recomputed_cost_m",
              "improvements_beyond_tolerance", "termination_reason", *COUNTERS, *TIMINGS)
    with (output/"trials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in rows:
            flat = dict(row, **row["metrics"])
            writer.writerow({k: flat.get(k) for k in fields})
    metric_names=("validated_success","first_solution_time_s","cost_terrain_m_at_budget","length_xy_m_final",
                  "actual_solve_wall_s","worker_wall_s","validation_wall_s","peak_rss_mb",
                  "motion_validity_calls","state_validity_calls","coverage")
    requested_metrics=benchmark.get("report_metrics",metric_names) if benchmark else metric_names
    coverage_records=[]
    for planner in planners:
        planned_rows=[row for row in rows if row["planner_id"]==planner]
        executed_rows=[row for row in planned_rows if row["executed"]]
        for name in requested_metrics:
            if name=="coverage":
                available=len(executed_rows);denominator=len(planned_rows);conditioning="all_planned_trials"
            elif name=="validated_success":
                available=sum(bool(row.get("offline_validation_available")) for row in executed_rows)
                denominator=len(executed_rows);conditioning="all_executed_trials"
            elif name=="cost_terrain_m_at_budget":
                available=sum(finite(row.get("cost_terrain_m")) for row in executed_rows)
                denominator=len(executed_rows);conditioning="reported_over_all_executed; value_available_only_after_validated_success"
            elif name=="length_xy_m_final":
                available=sum(finite(row.get("length_xy_m")) for row in executed_rows)
                denominator=len(executed_rows);conditioning="reported_over_all_executed; value_available_only_for_validated_final_paths"
            elif name=="first_solution_time_s":
                available=sum(finite(row.get("first_solution_time_s")) for row in executed_rows)
                denominator=len(executed_rows);conditioning="reported_over_all_executed; value_available_only_after_validated_success"
            else:
                available=sum(finite(row["metrics"].get(name)) for row in executed_rows)
                denominator=len(executed_rows);conditioning="all_executed_trials"
            coverage_records.append({"planner_id":planner,"metric":name,"available":available,
                "denominator":denominator,"coverage":available/denominator if denominator else None,
                "conditioning":conditioning})
    summary["metric_coverage"]=coverage_records
    summary["figures"] = plot_results(output, rows, curve_data, families, planners) if make_plots else []
    if combined_protocol:
        summary["figure_status"]=("CREATED" if make_plots else
            "NOT_CREATED_BY_CAMPAIGN_NO_PLOTS_DUE_TO_PILLOW_MATPLOTLIB_DLL_RUNTIME; JSON_ANYTIME_AND_METRIC_ARTIFACTS_RETAINED")
    write_report(output, source, summary, bootstrap)
    # summary.json is the completion marker. It is intentionally emitted only
    # after every required report artifact, especially REPORT.md, succeeded.
    dump(output/"summary.json", summary)
    return summary


def _publish_staged_report(staging, output, previous_summary):
    staging, output = Path(staging), Path(output)
    required = {"summary.json", "REPORT.md", "reference_targets.json", "paired_bootstrap.json",
                "anytime_curves.json", "trial_metrics.json", "trials.csv"}
    staged_files = [path for path in staging.rglob("*") if path.is_file()]
    relative_files = {path.relative_to(staging).as_posix() for path in staged_files}
    if not required <= relative_files:
        raise RuntimeError("staged report lacks required artifacts: "+str(sorted(required-relative_files)))
    summary = staging/"summary.json"
    for path in sorted((path for path in staged_files if path != summary),
                       key=lambda item: item.relative_to(staging).as_posix()):
        destination = output/path.relative_to(staging)
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(path, destination)
    # The completion marker is committed last. A failed publication therefore
    # cannot leave a newly generated canonical summary claiming completion.
    os.replace(summary, output/"summary.json")
    if previous_summary.exists():
        try:
            previous_summary.unlink()
        except OSError:
            # The canonical report is already complete; a recoverable hidden
            # backup is harmless and must not turn successful publication into
            # a reported failure.
            pass


def analyze(source, output, bootstrap_replicates=None, bootstrap_seed=None, make_plots=True):
    """Build a report in a staging directory and publish summary.json last."""
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError("reports must be separate from immutable raw results")
    manifest = read_json(source/"manifest.json")
    config = read_json(source/"resolved_config.json")
    if config.get("schema_version")==9:
        from analyze_a3_traversability_perf import analyze as analyze_performance
        return analyze_performance(source,output,make_plots)
    if config.get("schema_version")==8:
        from analyze_a3_traversability import analyze as analyze_traversability
        return analyze_traversability(source,output,make_plots)
    locked_comparison=(bool(config.get("advanced_benchmark")) or
                       bool(config.get("combined_benchmark")) or
                       manifest.get("profile")=="advanced")
    raw_inventory_hash = verify_raw_artifact_inventory(source, required=locked_comparison)

    output.mkdir(parents=True, exist_ok=True)
    previous_summary = output/".summary.previous.json"
    canonical_summary = output/"summary.json"
    if canonical_summary.exists():
        os.replace(canonical_summary, previous_summary)
    try:
        with tempfile.TemporaryDirectory(prefix=output.name+".staging.", dir=output.parent) as directory:
            staging = Path(directory)
            summary = _analyze_into(source, staging, bootstrap_replicates, bootstrap_seed, make_plots,
                                    raw_inventory_hash, output)
            _publish_staged_report(staging, output, previous_summary)
            print({k: summary[k] for k in ("status", "planned", "executed", "validated_success",
                                           "timeout", "invalid", "crash", "watchdog", "blocked", "not_run")})
            return summary
    except BaseException:
        # Keep the previous summary recoverable, but never at the canonical path:
        # callers must not confuse an old completion marker with this failed run.
        raise


def write_report(output, source, summary, bootstrap):
    def fmt(value):
        return "N/A" if value is None else f"{value:.6g}"
    combined_protocol=summary.get("protocol_id")=="a3_vs_all_v1"
    lines = ["# Báo cáo "+summary["experiment_id"], "",
             f"Trạng thái **{summary['status']}**. Đã thực thi **{summary['executed']}/{summary['planned']}** trial "
             f"(coverage {fmt(summary['completion_coverage'])}); success qua verifier **{summary['validated_success']}/{summary['executed']}** "
             f"({fmt(summary['success_rate_executed'])}). Timeout {summary['timeout']}; invalid {summary['invalid']}; "
             f"crash {summary['crash']}; watchdog {summary['watchdog']}; blocked {summary['blocked']}; chưa chạy {summary['not_run']}; "
             f"validation pending {summary['validation_pending']}.", "",
             "Blocked nằm trong chưa chạy. Invalid có thể đồng thời thuộc một execution status. Trial chưa chạy không là thất bại đã quan sát; mọi trial đã chạy đều nằm trong mẫu số success.", "",
             "Kết quả theo family/query nằm trong `summary.json`; CSV giữ mọi trial theo manifest, kể cả chưa chạy. Các thống kê cost/T_first có điều kiện theo success và ghi số quan sát thiếu riêng."]
    if combined_protocol:
        lines += ["",
            "Protocol khóa `a3_vs_all_v1` có 250 trial = 50 map × 5 planner, một query `q0` và đúng một planner seed 93001 trên mỗi map; mỗi trial có budget solve 20 s. Một seed/map không đo được biến thiên RNG lặp lại trong cùng map, nên bootstrap ghép cặp chỉ mô tả biến thiên qua các map đã khóa.", "",
            "Success chính chỉ là nghiệm tới singleton goal với `EXACT_SOLUTION`, xuất hiện trong budget và vượt kiểm chứng đường đi độc lập. `EXACT_SOLUTION` không phải chứng minh tối ưu toàn cục; timeout không phải chứng minh vô nghiệm.", "",
            "`cost_terrain_m` là chiều dài đường bám bề mặt và là objective chung. `length_xy_m` là chiều dài chiếu bằng phụ trợ, được lưu và báo riêng, không thay thế terrain cost.", "",
            "Comparator `sem_rrtstar_common_v1` phải được gọi **SEM-RRT*-adapted common-objective**: `official_sem_rrtstar_implementation=false`, `terrain_risk_cost=false`, `turning_angle_cost=false`, `rolling_optimization=false`. Đây không phải implementation SEM-RRT* chính thức và không paper-faithful.", "",
            "Campaign này chạy analyzer với `--no-plots` vì Pillow/Matplotlib không nạp được DLL trong runtime hiện tại. Không tạo PNG; `anytime_curves.json`, `trial_metrics.json`, bảng coverage và paired bootstrap vẫn là artifact phân tích đầy đủ."]
    lines += ["",
              "| Family | Planner | Success / executed / planned | T_first median (s) | Cost tại budget median (m) | Improvements > tolerance |",
              "|---|---|---:|---:|---:|---:|"]
    for g in summary["family_groups"]:
        lines.append(f"| {g['family']} | {g['planner_id']} | {g['validated_success']}/{g['executed']}/{g['planned']} | "
                     f"{fmt(g['first_solution_s']['median'])} | {fmt(g['cost_terrain_m_at_budget']['median'])} | {g['improvements_beyond_tolerance']} |")
    lines += ["", "T_first và anytime chỉ dùng snapshot có verification/checksum khớp. Cost trước nghiệm đầu là null, không bằng 0; đường đồ thị là hàm bậc thang từ event thật. Incumbent cuối được giữ tới budget khi planner dừng sớm, không tạo event/công việc mới. Callback sau deadline không đóng góp quality tại budget. Snapshot lỗi làm trial mất success và được báo correctness error.", "",
              "Reference trong `reference_targets.json` được khóa từ witness hình học đã kiểm chứng trong manifest trước khi tạo so sánh; không lấy phương pháp thắng làm reference. Target là (1+epsilon) nhân witness cost; không phải chứng nhận epsilon-optimal. Chênh lệch tương đối có thể âm. Thiếu reference hoặc reference cost bằng 0 không tạo gap giả.", "",
              "`trial_metrics.json` giữ event time riêng với observation endpoint: timeout không được gán thời gian thành công bằng budget. Với crash/watchdog, endpoint dùng actual solve wall time; nếu thiếu thì dùng worker wall time có gắn nhãn fallback, và giữ null nếu cả hai đều thiếu. Crash/watchdog nằm trong mẫu số, kể cả có snapshot trước lỗi. Đây là empirical attainment, không giả định censoring độc lập hay ước lượng Kaplan–Meier.", "",
              "Acceptance toàn vòng đời custom sampler = samples_returned_before_validity / sample_proposals, gồm cả proposal uniform-D trước incumbent; không gồm goal bias và chưa kiểm footprint/state validity. Không gọi tỷ lệ này là acceptance riêng của miền informed hữu hạn. Planner không dùng informed sampling và counter chưa đo dùng null + lý do. Counter verifier không cộng vào planner. RSS là sampled peak. Các thời gian load/preprocess/setup/solve/serialize/validation/callback được báo riêng khi đo được.", "",
              "Bootstrap trong `paired_bootstrap.json` resample map → query → paired seed slot, giữ cặp phương pháp ở cùng bài toán. Cost/T_first chỉ xét cặp cùng success; success xét mọi cặp đã chạy. Seed số học giống nhau không bảo đảm chuỗi mẫu giống nhau. CI là bất định thực nghiệm hữu hạn trên bộ dữ liệu đã khóa, không là chứng nhận ưu thế tổng quát. Có dưới hai map thì không xuất CI.", "",
              "| Family | Paired difference | Metric | Observed maps / queries / usable pairs | Planned slots: both / one / none | Estimate | Exploratory CI |",
              "|---|---|---|---:|---:|---:|---|"]
    report_pairs={(item["planner_a"],item["planner_b"]) for item in summary["primary_comparisons"]}
    for b in bootstrap:
        if (b["planner_a"], b["planner_b"]) not in report_pairs:
            continue
        units = b["units"]
        interval = "N/A" if b["ci"] is None else "["+", ".join(fmt(x) for x in b["ci"])+"]"
        lines.append(f"| {b['family']} | {b['planner_a']} − {b['planner_b']} | {b['metric']} | "
                     f"{units['maps']}/{units['map_query_units']}/{units['paired_seed_slots']} | "
                     f"{units['both_planners_executed_slots']}/{units['one_planner_executed_slots']}/"
                     f"{units['neither_planner_executed_slots']} | {fmt(b['estimate'])} | {interval} |")
    lines += ["", "Coverage từng metric (`available/denominator`); giá trị cost và T_first chỉ khả dụng sau validated success:", "",
              "| Planner | Metric | Coverage | Conditioning |", "|---|---|---:|---|"]
    for item in summary["metric_coverage"]:
        lines.append(f"| {item['planner_id']} | {item['metric']} | {item['available']}/{item['denominator']} ({fmt(item['coverage'])}) | {item['conditioning']} |")
    if summary["missing_groups"]:
        lines += ["", "Nhóm còn thiếu (family / map / query / planner: executed/planned):"]
        lines += [f"- {g['family']} / {g['map_id']} / {g['query_id']} / {g['planner_id']}: {g['executed']}/{g['planned']}" for g in summary["missing_groups"]]
    caveat=("Không gọi nghiệm EXACT_SOLUTION là tối ưu toàn cục; không gọi timeout là vô nghiệm. "
            "Đây là so sánh finite-budget trên seeds held-out đã khóa; mọi thiếu hụt coverage vẫn được giữ trong báo cáo."
            if summary["profile"]=="advanced" or combined_protocol else
            "Không gọi nghiệm EXACT_SOLUTION là tối ưu toàn cục; không gọi timeout là vô nghiệm. G5/full chưa chạy. Cấu hình full chỉ nên được khóa sau khi xem coverage, failure, overhead và độ biến động ở các family; giữ map seeds full tách pilot, không dùng full để tune.")
    figure_note=("Không tạo PNG do campaign dùng `--no-plots` trước lỗi DLL Pillow/Matplotlib; dữ liệu JSON anytime/metrics vẫn được giữ."
                 if combined_protocol and not summary["figures"] else
                 "Hình: "+", ".join("`"+p+"`" for p in summary["figures"])+".")
    lines += ["", caveat, "",
              "Raw input: `"+str(source.resolve())+"`. Log thuộc harness riêng, không phải native Planner Arena. "+figure_note]
    if summary.get("raw_artifact_inventory_sha256"):
        lines += ["", "Raw artifact inventory SHA-256: `"+summary["raw_artifact_inventory_sha256"]+"`."]
    (output/"REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-replicates", type=int)
    parser.add_argument("--bootstrap-seed", type=int)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    summary = analyze(args.input, args.output, args.bootstrap_replicates, args.bootstrap_seed, not args.no_plots)
    return 1 if summary["invalid"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
