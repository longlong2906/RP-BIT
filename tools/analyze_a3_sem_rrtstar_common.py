"""Analyze the locked common-objective A3 versus SEM-RRT*-adapted campaign.

This analyzer is deliberately separate from the generic pilot analyzer.  It
implements the preregistered 10 s estimands, paired family/map/seed bootstrap,
right-censored first-solution RMST, and checkpoint effort accounting for
``a3_vs_sem_rrtstar_common_v1``.  It never treats a timeout as a solution event
and never uses unverified progress costs as path-quality observations.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
import os
from pathlib import Path
import random
import statistics
import tempfile

from analyze_results import (
    finite,
    read_trial,
    step_cost,
    verify_raw_artifact_inventory,
)
from experiment_common import dump, file_hash, read_json


PROTOCOL_ID = "a3_vs_sem_rrtstar_common_v1"
PROFILE = "sem_comparison"
PRIMARY_PLANNER = "bitstar_a3"
COMPARATOR_PLANNER = "sem_rrtstar_common_v1"
PLANNERS = (PRIMARY_PLANNER, COMPARATOR_PLANNER)
EXPECTED_TRIALS = 300
EXPECTED_FAMILIES = (
    "traversable_hill",
    "ridge_pass",
    "rolling_terrain",
    "narrow_passage",
    "mixed_obstacles",
)
EXPECTED_MAP_SEEDS = tuple(range(4001, 4011))
EXPECTED_PLANNER_SEEDS = (92001, 92002, 92003)
EXPECTED_QUERY_IDS = ("q0",)
TAU_S = 10.0
CHECKPOINTS_S = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0)
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 73135
BOOTSTRAP_HIERARCHY = ("family", "map", "seed")
CONFIDENCE_LEVEL = 0.95
PAPER_SAMPLE_CHECKPOINT = 5000
SECONDARY_EFFORT_METRICS = (
    "motion_validity_calls",
    "state_validity_calls",
    "full_surface_cost_calls",
    "triangle_visits",
    "height_queries",
    "sample_proposals",
)
REQUIRED_REPORT_FILES = {
    "summary.json",
    "REPORT.md",
    "trial_metrics.json",
    "trials.csv",
    "checkpoints.json",
    "metric_coverage.json",
    "descriptive_metrics.json",
    "paired_bootstrap.json",
}
DESCRIPTIVE_TIMING_METRICS = (
    "actual_solve_wall_s",
    "worker_wall_s",
    "validation_wall_s",
    "peak_rss_mb",
)


def _quantile(values: list[float], probability: float) -> float | None:
    values = sorted(value for value in values if finite(value))
    if not values:
        return None
    position = (len(values) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def distribution(values, denominator=None):
    values = sorted(value for value in values if finite(value))
    denominator = len(values) if denominator is None else denominator
    return {
        "available": len(values),
        "denominator": denominator,
        "coverage": len(values) / denominator if denominator else None,
        "median": statistics.median(values) if values else None,
        "q25": _quantile(values, 0.25),
        "q75": _quantile(values, 0.75),
        "min": values[0] if values else None,
        "max": values[-1] if values else None,
    }


def validate_locked_protocol(config, manifest):
    """Return protocol violations without turning missing trials into failures."""
    errors = []
    sem = config.get("sem_comparison", {})
    profile = config.get("profiles", {}).get(PROFILE, {})
    entries = manifest.get("trials", [])
    expected_pairs = {
        "protocol_id": PROTOCOL_ID,
        "planner_ids": list(PLANNERS),
        "primary_planner_id": PRIMARY_PLANNER,
        "comparator_planner_ids": [COMPARATOR_PLANNER],
        "expected_trials": EXPECTED_TRIALS,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    if config.get("schema_version") != 3:
        errors.append("resolved_config.schema_version_must_be_3")
    if manifest.get("schema_version") != 3:
        errors.append("experiment_manifest.schema_version_must_be_3")
    if manifest.get("profile") != PROFILE:
        errors.append("manifest.profile_must_be_sem_comparison")
    for key, expected in expected_pairs.items():
        if sem.get(key) != expected:
            errors.append(f"sem_comparison.{key}_differs_from_locked_protocol")
    if config.get("analysis", {}).get("bootstrap_unit_order") != list(BOOTSTRAP_HIERARCHY):
        errors.append("analysis.bootstrap_unit_order_must_be_family_map_seed")
    if profile.get("planning_budget_s") != TAU_S:
        errors.append("planning_budget_s_must_be_10")
    if profile.get("families") != list(EXPECTED_FAMILIES):
        errors.append("family_list_differs_from_locked_protocol")
    if profile.get("map_seeds") != list(EXPECTED_MAP_SEEDS):
        errors.append("map_seed_list_differs_from_locked_protocol")
    if profile.get("planner_seeds") != list(EXPECTED_PLANNER_SEEDS):
        errors.append("planner_seed_list_differs_from_locked_protocol")
    if profile.get("planners") != list(PLANNERS):
        errors.append("planner_list_differs_from_locked_protocol")
    if manifest.get("planned_trials") != EXPECTED_TRIALS or len(entries) != EXPECTED_TRIALS:
        errors.append("manifest_must_contain_exactly_300_planned_trials")

    trial_ids = [entry.get("trial_id") for entry in entries]
    if len(set(trial_ids)) != len(trial_ids) or any(not isinstance(value, str) or not value for value in trial_ids):
        errors.append("manifest_trial_ids_must_be_unique_nonempty_strings")

    slots = defaultdict(list)
    map_families = {}
    maps_by_family = defaultdict(set)
    map_seeds_by_family = defaultdict(set)
    required = ("trial_id", "family", "map_id", "map_seed", "query_id", "planner_id", "planner_seed")
    for entry in entries:
        missing = [key for key in required if key not in entry]
        if missing:
            errors.append(f"manifest_entry_missing_identity_fields:{entry.get('trial_id')}:{','.join(missing)}")
            continue
        family, map_id = entry["family"], entry["map_id"]
        if map_id in map_families and map_families[map_id] != family:
            errors.append(f"map_id_in_multiple_families:{map_id}")
        map_families[map_id] = family
        maps_by_family[family].add(map_id)
        map_seeds_by_family[family].add(entry["map_seed"])
        slots[(family, map_id, entry["query_id"], entry["planner_seed"])].append(entry["planner_id"])

    if set(maps_by_family) != set(EXPECTED_FAMILIES):
        errors.append("manifest_family_set_differs_from_locked_protocol")
    for family in EXPECTED_FAMILIES:
        if len(maps_by_family[family]) != 10:
            errors.append(f"family_must_have_10_maps:{family}")
        if map_seeds_by_family[family] != set(EXPECTED_MAP_SEEDS):
            errors.append(f"family_map_seeds_differ_from_locked_protocol:{family}")
    if {entry.get("query_id") for entry in entries} != set(EXPECTED_QUERY_IDS):
        errors.append("manifest_query_ids_differ_from_locked_protocol")
    if {entry.get("planner_seed") for entry in entries} != set(EXPECTED_PLANNER_SEEDS):
        errors.append("manifest_planner_seeds_differ_from_locked_protocol")
    if len(slots) != EXPECTED_TRIALS // len(PLANNERS):
        errors.append("manifest_must_have_150_paired_problem_slots")
    for key, planners in slots.items():
        if Counter(planners) != Counter(PLANNERS):
            errors.append("paired_slot_does_not_contain_each_planner_once:" + "/".join(map(str, key)))
    return sorted(set(errors))


def read_progress(directory):
    """Read timestamped counters only; progress costs are intentionally ignored."""
    path = Path(directory) / "progress.jsonl"
    if not path.exists():
        return [], [], "missing_progress_jsonl"
    records, errors = [], []
    previous_time = -1.0
    previous_counters = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(
                    line,
                    parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON: " + value)),
                )
            except (ValueError, json.JSONDecodeError) as exc:
                errors.append(f"progress_json_invalid_line_{line_number}:{type(exc).__name__}")
                continue
            elapsed = record.get("elapsed_s")
            counters = record.get("counters")
            if not finite(elapsed) or elapsed < 0 or elapsed < previous_time:
                errors.append(f"progress_time_invalid_line_{line_number}")
                continue
            if not isinstance(counters, dict):
                errors.append(f"progress_counters_missing_line_{line_number}")
                continue
            normalized = {}
            invalid = False
            for name, value in counters.items():
                if not finite(value) or value < 0:
                    errors.append(f"progress_counter_invalid_line_{line_number}:{name}")
                    invalid = True
                    continue
                if name in previous_counters and value < previous_counters[name]:
                    errors.append(f"progress_counter_decreased_line_{line_number}:{name}")
                    invalid = True
                normalized[name] = value
            if invalid:
                continue
            previous_time = elapsed
            previous_counters.update(normalized)
            records.append({"elapsed_s": elapsed, "counters": normalized})
    return records, errors, None if records else "progress_contains_no_usable_counter_records"


def checkpoint_counter(progress, metric_name, checkpoint_s):
    eligible = [
        record for record in progress
        if record["elapsed_s"] <= checkpoint_s and finite(record["counters"].get(metric_name))
    ]
    if not eligible:
        return {"value": None, "sample_elapsed_s": None, "poll_lag_s": None,
                "reason": "no_timestamped_counter_at_or_before_checkpoint"}
    record = eligible[-1]
    return {
        "value": record["counters"][metric_name],
        "sample_elapsed_s": record["elapsed_s"],
        "poll_lag_s": checkpoint_s - record["elapsed_s"],
        "reason": None,
    }


def sample_count_checkpoint(row, target=PAPER_SAMPLE_CHECKPOINT):
    """Locate the first poll at/after the paper's 5,000-sample diagnostic.

    Polling generally crosses the target between observations, so the observed
    proposal count and overshoot are retained instead of pretending the record
    was taken at exactly 5,000 proposals.  Quality still comes exclusively from
    independently validated snapshots no later than that poll.
    """
    trial_budget = row.get("time_budget_s")
    cutoff = min(TAU_S, trial_budget) if finite(trial_budget) and trial_budget >= 0 else TAU_S
    eligible = [
        record for record in row.get("progress", [])
        if record["elapsed_s"] <= cutoff
        and finite(record["counters"].get("sample_proposals"))
        and record["counters"]["sample_proposals"] >= target
    ]
    if not eligible:
        return {"available": False, "target_sample_proposals": target,
                "observed_sample_proposals": None, "proposal_overshoot": None,
                "elapsed_s": None, "validated_cost_terrain_m": None,
                "reason": "sample_proposal_checkpoint_not_observed"}
    record = eligible[0]
    cost = step_cost(row.get("events", []), record["elapsed_s"]) if row.get("success") else None
    return {"available": True, "target_sample_proposals": target,
            "observed_sample_proposals": record["counters"]["sample_proposals"],
            "proposal_overshoot": record["counters"]["sample_proposals"] - target,
            "elapsed_s": record["elapsed_s"], "validated_cost_terrain_m": cost,
            "reason": None}


def time_to_first_record(row, tau_s=TAU_S):
    """Return an event/censor record without fabricating an event at tau."""
    if not row.get("executed"):
        return {"time_s": None, "event_observed": False, "status": "NOT_RUN",
                "endpoint_source": None, "reason": "trial_not_executed"}
    if row.get("validation_pending"):
        return {"time_s": None, "event_observed": False, "status": "UNAVAILABLE",
                "endpoint_source": None, "reason": "offline_validation_pending"}
    first = row.get("first_solution_time_s")
    if row.get("success") and finite(first) and first <= tau_s:
        return {"time_s": first, "event_observed": True, "status": "EVENT",
                "endpoint_source": "first_independently_validated_snapshot", "reason": None}

    actual = row.get("metrics", {}).get("actual_solve_wall_s")
    worker = row.get("metrics", {}).get("worker_wall_s")
    if finite(actual) and actual >= 0:
        endpoint, source, reason = min(tau_s, actual), "actual_solve_wall_s", None
    elif row.get("status") == "TIMEOUT" and row.get("time_budget_s") == tau_s:
        endpoint, source, reason = tau_s, "timeout_budget_contract", "actual_solve_wall_s_unavailable"
    elif finite(worker) and worker >= 0:
        endpoint, source, reason = min(tau_s, worker), "worker_wall_s_fallback", "actual_solve_wall_s_unavailable"
    else:
        return {"time_s": None, "event_observed": False, "status": "UNAVAILABLE",
                "endpoint_source": None, "reason": "censor_endpoint_unavailable"}
    return {"time_s": endpoint, "event_observed": False, "status": "RIGHT_CENSORED",
            "endpoint_source": source, "reason": reason}


def km_rmst(records, tau_s=TAU_S):
    """Kaplan-Meier restricted mean with optional non-negative case weights."""
    usable = []
    for record in records:
        time_s = record.get("time_s")
        weight = record.get("weight", 1.0)
        if finite(time_s) and 0 <= time_s <= tau_s and finite(weight) and weight > 0:
            usable.append((time_s, bool(record.get("event_observed")), weight))
    if not usable:
        return {"rmst_s": None, "events": 0, "right_censored": 0, "n": 0,
                "total_weight": 0.0, "tau_s": tau_s}
    event_times = sorted({time_s for time_s, observed, _ in usable if observed and time_s <= tau_s})
    survival = 1.0
    previous = 0.0
    area = 0.0
    for event_time in event_times:
        area += survival * (event_time - previous)
        at_risk = sum(weight for time_s, _, weight in usable if time_s >= event_time)
        events = sum(weight for time_s, observed, weight in usable if observed and time_s == event_time)
        if at_risk > 0:
            survival *= max(0.0, 1.0 - events / at_risk)
        previous = event_time
    area += survival * (tau_s - previous)
    return {
        "rmst_s": area,
        "events": sum(observed for _, observed, _ in usable),
        "right_censored": sum(not observed for _, observed, _ in usable),
        "n": len(usable),
        "total_weight": sum(weight for _, _, weight in usable),
        "tau_s": tau_s,
    }


def _paired_slots(rows):
    slots = defaultdict(dict)
    duplicate_errors = []
    for row in rows:
        if row.get("planner_id") not in PLANNERS:
            continue
        key = (row.get("family"), row.get("map_id"), row.get("planner_seed"))
        if row["planner_id"] in slots[key]:
            duplicate_errors.append("duplicate_planner_in_pair:" + "/".join(map(str, key)))
        slots[key][row["planner_id"]] = row
    return slots, duplicate_errors


def _pair_units(slots, usable_keys):
    both_executed = one_executed = neither_executed = incomplete = 0
    both_success = 0
    for pair in slots.values():
        if set(pair) != set(PLANNERS):
            incomplete += 1
            continue
        executed = [bool(pair[planner].get("executed")) for planner in PLANNERS]
        if all(executed):
            both_executed += 1
        elif any(executed):
            one_executed += 1
        else:
            neither_executed += 1
        if all(pair[planner].get("success_at_10s") for planner in PLANNERS):
            both_success += 1
    planned_families = {key[0] for key in slots}
    planned_maps = {(key[0], key[1]) for key in slots}
    usable_families = {key[0] for key in usable_keys}
    usable_maps = {(key[0], key[1]) for key in usable_keys}
    return {
        "planned_families": len(planned_families),
        "planned_maps": len(planned_maps),
        "planned_paired_seed_slots": len(slots),
        "both_planners_executed_slots": both_executed,
        "one_planner_executed_slots": one_executed,
        "neither_planner_executed_slots": neither_executed,
        "incomplete_planned_slots": incomplete,
        "both_planners_validated_success_slots": both_success,
        "usable_families": len(usable_families),
        "usable_maps": len(usable_maps),
        "usable_paired_seed_slots": len(usable_keys),
        "usable_pair_coverage": len(usable_keys) / len(slots) if slots else None,
    }


def _hierarchy(values):
    hierarchy = defaultdict(lambda: defaultdict(list))
    for (family, map_id, _seed), value in values.items():
        hierarchy[family][map_id].append(value)
    return hierarchy


def _hierarchical_scalar_estimate(hierarchy):
    if not hierarchy:
        return None
    return statistics.mean(
        statistics.mean(statistics.mean(seed_values) for seed_values in maps.values())
        for maps in hierarchy.values()
    )


def _bootstrap_scalar(hierarchy, rng):
    families = sorted(hierarchy)
    family_means = []
    for family in rng.choices(families, k=len(families)):
        maps = sorted(hierarchy[family])
        map_means = []
        for map_id in rng.choices(maps, k=len(maps)):
            seed_values = hierarchy[family][map_id]
            map_means.append(statistics.mean(rng.choices(seed_values, k=len(seed_values))))
        family_means.append(statistics.mean(map_means))
    return statistics.mean(family_means)


def _weighted_rmst_for_pairs(pair_values, planner):
    hierarchy = _hierarchy(pair_values)
    records = []
    family_count = len(hierarchy)
    for maps in hierarchy.values():
        map_count = len(maps)
        for pairs in maps.values():
            pair_count = len(pairs)
            weight = 1.0 / (family_count * map_count * pair_count)
            for pair in pairs:
                records.append(dict(pair[planner], weight=weight))
    return km_rmst(records, TAU_S)


def _bootstrap_rmst(pair_values, rng):
    hierarchy = _hierarchy(pair_values)
    families = sorted(hierarchy)
    records = {planner: [] for planner in PLANNERS}
    for family in rng.choices(families, k=len(families)):
        maps = sorted(hierarchy[family])
        for map_id in rng.choices(maps, k=len(maps)):
            pairs = hierarchy[family][map_id]
            for pair in rng.choices(pairs, k=len(pairs)):
                for planner in PLANNERS:
                    records[planner].append(pair[planner])
    a = km_rmst(records[PRIMARY_PLANNER], TAU_S)["rmst_s"]
    b = km_rmst(records[COMPARATOR_PLANNER], TAU_S)["rmst_s"]
    return a - b


def _interval(samples, confidence=CONFIDENCE_LEVEL):
    samples = sorted(value for value in samples if finite(value))
    if not samples:
        return None
    alpha = (1.0 - confidence) / 2.0
    return [
        samples[math.floor(alpha * (len(samples) - 1))],
        samples[math.ceil((1.0 - alpha) * (len(samples) - 1))],
    ]


def paired_bootstrap(rows, metric_name, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED):
    """Locked family -> map -> paired-seed bootstrap for one contrast."""
    slots, errors = _paired_slots(rows)
    values = {}
    is_rmst = metric_name == "t_first_rmst_tau_10s"
    for key, pair in slots.items():
        if set(pair) != set(PLANNERS):
            continue
        a, b = pair[PRIMARY_PLANNER], pair[COMPARATOR_PLANNER]
        if metric_name == "validated_success_at_10s":
            eligible = all(
                row.get("executed") and row.get("offline_validation_available")
                and not row.get("validation_pending") and not row.get("errors")
                for row in (a, b)
            )
            value = int(a.get("success_at_10s", False)) - int(b.get("success_at_10s", False)) if eligible else None
        elif metric_name == "cost_terrain_m_at_10s_both_success":
            eligible = all(row.get("success_at_10s") and finite(row.get("cost_at_10s")) for row in (a, b))
            value = a.get("cost_at_10s") - b.get("cost_at_10s") if eligible else None
        elif metric_name == "length_xy_m_final_both_success":
            eligible = all(row.get("success_at_10s") and finite(row.get("length_xy_m")) for row in (a, b))
            value = a.get("length_xy_m") - b.get("length_xy_m") if eligible else None
        elif is_rmst:
            eligible = all(finite(row.get("t_first", {}).get("time_s")) for row in (a, b))
            value = {planner: pair[planner]["t_first"] for planner in PLANNERS} if eligible else None
        elif metric_name.startswith("effort_") and metric_name.endswith("_at_10s"):
            counter = metric_name[len("effort_"):-len("_at_10s")]
            av = a.get("effort_at_10s", {}).get(counter, {}).get("value")
            bv = b.get("effort_at_10s", {}).get(counter, {}).get("value")
            eligible = a.get("executed") and b.get("executed") and finite(av) and finite(bv)
            value = av - bv if eligible else None
        else:
            raise ValueError("unsupported paired metric: " + metric_name)
        if eligible and value is not None:
            values[key] = value

    units = _pair_units(slots, set(values))
    result = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "planner_a": PRIMARY_PLANNER,
        "planner_b": COMPARATOR_PLANNER,
        "difference": "planner_a_minus_planner_b",
        "metric": metric_name,
        "hierarchy": list(BOOTSTRAP_HIERARCHY),
        "replicates": replicates,
        "seed": seed,
        "confidence_level": CONFIDENCE_LEVEL,
        "units": units,
        "estimate": None,
        "ci": None,
        "status": "UNAVAILABLE_NO_USABLE_PAIRS",
        "errors": errors,
    }
    if not values:
        return result
    hierarchy = _hierarchy(values)
    if is_rmst:
        a_rmst = _weighted_rmst_for_pairs(values, PRIMARY_PLANNER)
        b_rmst = _weighted_rmst_for_pairs(values, COMPARATOR_PLANNER)
        result["planner_rmst"] = {PRIMARY_PLANNER: a_rmst, COMPARATOR_PLANNER: b_rmst}
        result["estimate"] = a_rmst["rmst_s"] - b_rmst["rmst_s"]
        result["conditioning"] = "complete paired trials with known event-or-censor endpoints"
        draw = _bootstrap_rmst
    else:
        result["estimate"] = _hierarchical_scalar_estimate(hierarchy)
        result["conditioning"] = (
            "both planners independently validated success by 10 s"
            if metric_name in ("cost_terrain_m_at_10s_both_success", "length_xy_m_final_both_success")
            else "complete paired executed trials with terminal offline-validation status"
            if metric_name == "validated_success_at_10s"
            else "both planners executed with timestamped counter at or before 10 s"
        )
        draw = _bootstrap_scalar
    if len(hierarchy) < 2:
        result["status"] = "INSUFFICIENT_INDEPENDENT_FAMILIES_FOR_CI"
        return result
    rng = random.Random(seed)
    samples = [draw(values if is_rmst else hierarchy, rng) for _ in range(replicates)]
    result["ci"] = _interval(samples)
    result["status"] = "ESTIMATED"
    return result


def _checkpoint_group(rows, checkpoint_s):
    executed = [row for row in rows if row.get("executed")]
    costs = []
    successes = 0
    effort = {name: [] for name in SECONDARY_EFFORT_METRICS}
    lags = {name: [] for name in SECONDARY_EFFORT_METRICS}
    for row in executed:
        cost = step_cost(row.get("events", []), checkpoint_s) if row.get("success") else None
        if finite(cost):
            successes += 1
            costs.append(cost)
        for name in SECONDARY_EFFORT_METRICS:
            item = row.get("effort_checkpoints", {}).get(str(checkpoint_s), {}).get(name, {})
            if finite(item.get("value")):
                effort[name].append(item["value"])
                lags[name].append(item["poll_lag_s"])
    return {
        "checkpoint_s": checkpoint_s,
        "planned": len(rows),
        "executed": len(executed),
        "validated_success": successes,
        "validated_success_rate_executed": successes / len(executed) if executed else None,
        "cost_terrain_m": distribution(costs, len(executed)),
        "secondary_effort": {
            name: dict(distribution(effort[name], len(executed)), poll_lag_s=distribution(lags[name], len(executed)))
            for name in SECONDARY_EFFORT_METRICS
        },
    }


def checkpoint_summaries(rows):
    groups = []
    for scope, keys in (("planner", ("planner_id",)), ("family_planner", ("family", "planner_id"))):
        grouped = defaultdict(list)
        for row in rows:
            grouped[tuple(row.get(key) for key in keys)].append(row)
        for values, group in sorted(grouped.items()):
            identity = dict(zip(keys, values))
            groups.append({
                "scope": scope,
                **identity,
                "points": [_checkpoint_group(group, checkpoint) for checkpoint in CHECKPOINTS_S],
            })
    sample_groups = []
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row.get("planner_id"),)].append(row)
        grouped[(row.get("family"), row.get("planner_id"))].append(row)
    for identity, group in sorted(grouped.items(), key=lambda item: (len(item[0]), item[0])):
        available = [row["paper_sample_checkpoint"] for row in group
                     if row.get("executed") and row["paper_sample_checkpoint"]["available"]]
        record = {
            "scope": "planner" if len(identity) == 1 else "family_planner",
            "planner_id": identity[-1],
            "planned": len(group),
            "executed": sum(row.get("executed", False) for row in group),
            "available": len(available),
            "target_sample_proposals": PAPER_SAMPLE_CHECKPOINT,
            "elapsed_s": distribution([item["elapsed_s"] for item in available],
                                      sum(row.get("executed", False) for row in group)),
            "proposal_overshoot": distribution([item["proposal_overshoot"] for item in available],
                                                sum(row.get("executed", False) for row in group)),
            "validated_cost_terrain_m": distribution(
                [item["validated_cost_terrain_m"] for item in available],
                sum(row.get("executed", False) for row in group)),
        }
        if len(identity) == 2:
            record["family"] = identity[0]
        sample_groups.append(record)
    return {"schema_version": 1, "protocol_id": PROTOCOL_ID,
            "checkpoints_s": list(CHECKPOINTS_S), "groups": groups,
            "paper_sample_checkpoint": {"target_sample_proposals": PAPER_SAMPLE_CHECKPOINT,
                                          "groups": sample_groups}}


def metric_coverage(rows):
    records = []
    for planner in PLANNERS:
        planned = [row for row in rows if row.get("planner_id") == planner]
        executed = [row for row in planned if row.get("executed")]
        terminal = [row for row in executed if row.get("offline_validation_available") and not row.get("validation_pending")]
        definitions = [
            ("validated_success_at_10s", len(terminal), len(executed), "all executed trials with terminal offline validation"),
            ("cost_terrain_m_at_10s", sum(finite(row.get("cost_at_10s")) for row in executed), len(executed),
             "all executed trials; value exists only after validated success by 10 s"),
            ("length_xy_m_final", sum(row.get("success_at_10s") and finite(row.get("length_xy_m")) for row in executed),
             len(executed), "all executed trials; final XY length exists only for independently validated success"),
            ("t_first_event_time_s", sum(row.get("t_first", {}).get("event_observed", False) for row in executed), len(executed),
             "event times only; right-censored observations are intentionally not filled with 10 s"),
            ("t_first_event_or_censor_endpoint", sum(finite(row.get("t_first", {}).get("time_s")) for row in executed), len(executed),
             "all executed trials with a measured event or censor endpoint"),
        ]
        for name in SECONDARY_EFFORT_METRICS:
            available = sum(finite(row.get("effort_at_10s", {}).get(name, {}).get("value")) for row in executed)
            definitions.append(("effort_" + name + "_at_10s", available, len(executed),
                                "latest timestamped progress counter at or before 10 s; no extrapolation"))
        for name in DESCRIPTIVE_TIMING_METRICS:
            available = sum(finite(row.get("metrics", {}).get(name)) for row in executed)
            definitions.append((name, available, len(executed),
                                "all executed trials; unavailable measurements remain null"))
        for metric_name, available, denominator, conditioning in definitions:
            records.append({
                "planner_id": planner,
                "metric": metric_name,
                "available": available,
                "denominator": denominator,
                "coverage": available / denominator if denominator else None,
                "conditioning": conditioning,
            })
    return {"schema_version": 1, "protocol_id": PROTOCOL_ID, "records": records}


def descriptive_metrics(rows):
    """Distributions for every non-primary metric locked by report_metrics."""
    groups = []
    metric_names = (
        "first_solution_time_s",
        "cost_terrain_m_at_10s",
        "length_xy_m_final",
        *DESCRIPTIVE_TIMING_METRICS,
        *(name + "_final" for name in SECONDARY_EFFORT_METRICS),
    )

    def value(row, name):
        if name == "first_solution_time_s":
            return row.get("first_solution_time_s") if row.get("success_at_10s") else None
        if name == "cost_terrain_m_at_10s":
            return row.get("cost_at_10s")
        if name == "length_xy_m_final":
            return row.get("length_xy_m") if row.get("success_at_10s") else None
        if name.endswith("_final"):
            return row.get("metrics", {}).get(name[:-len("_final")])
        return row.get("metrics", {}).get(name)

    for scope, keys in (("planner", ("planner_id",)), ("family_planner", ("family", "planner_id"))):
        grouped = defaultdict(list)
        for row in rows:
            grouped[tuple(row.get(key) for key in keys)].append(row)
        for identity, group in sorted(grouped.items()):
            executed = [row for row in group if row.get("executed")]
            groups.append({
                "scope": scope,
                **dict(zip(keys, identity)),
                "planned": len(group),
                "executed": len(executed),
                "validated_success_at_10s": sum(row.get("success_at_10s", False) for row in executed),
                "metrics": {
                    name: distribution([value(row, name) for row in executed], len(executed))
                    for name in metric_names
                },
            })
    return {"schema_version": 1, "protocol_id": PROTOCOL_ID,
            "conditioning": {
                "first_solution_time_s": "validated success by 10 s",
                "cost_terrain_m_at_10s": "validated success by 10 s",
                "length_xy_m_final": "final independently validated path after success by 10 s",
                "timing_rss_and_final_effort": "all executed trials with the measurement available",
            },
            "groups": groups}


def classify_status(protocol_errors, correctness_errors, rows):
    if protocol_errors or correctness_errors:
        return "FAILED_CORRECTNESS"
    if any(not row.get("executed") or row.get("validation_pending") for row in rows):
        return "PARTIAL"
    return "COMPLETE"


def _safe_read_rows(source, entries, budget, absolute, relative):
    rows, errors = [], []
    for entry in entries:
        try:
            row = read_trial(
                source,
                entry,
                budget,
                absolute,
                relative,
                informed_planners=set(PLANNERS),
                allow_pending_validation=True,
            )
        except (KeyError, OSError, TypeError, ValueError) as exc:
            errors.append({"trial_id": entry.get("trial_id"), "errors": ["trial_parse_error:" + type(exc).__name__]})
            row = dict(entry, executed=False, success=False, status="NOT_RUN", events=[], errors=[],
                       validation_pending=False, offline_validation_available=False, metrics={},
                       time_budget_s=budget, first_solution_time_s=None, cost_terrain_m=None)
        progress, progress_errors, progress_reason = read_progress(source / entry["trial_id"])
        row["progress"] = progress
        row["progress_errors"] = progress_errors
        row["progress_unavailable_reason"] = progress_reason
        row["success_at_10s"] = bool(row.get("success") and finite(row.get("first_solution_time_s"))
                                      and row["first_solution_time_s"] <= TAU_S)
        row["cost_at_10s"] = step_cost(row.get("events", []), TAU_S) if row["success_at_10s"] else None
        row["t_first"] = time_to_first_record(row, TAU_S)
        row["effort_checkpoints"] = {}
        for checkpoint in CHECKPOINTS_S:
            row["effort_checkpoints"][str(checkpoint)] = {
                name: checkpoint_counter(progress, name, checkpoint)
                for name in SECONDARY_EFFORT_METRICS
            }
        row["effort_at_10s"] = row["effort_checkpoints"][str(TAU_S)]
        row["paper_sample_checkpoint"] = sample_count_checkpoint(row)
        rows.append(row)
        if row.get("errors"):
            errors.append({"trial_id": entry.get("trial_id"), "errors": row["errors"]})
        if progress_errors:
            errors.append({"trial_id": entry.get("trial_id"), "errors": progress_errors})
    return rows, errors


def _summary_counts(rows):
    executed = [row for row in rows if row.get("executed")]
    return {
        "planned": len(rows),
        "executed": len(executed),
        "not_run": len(rows) - len(executed),
        "validated_success_at_10s": sum(row.get("success_at_10s", False) for row in executed),
        "success_rate_executed_at_10s": (
            sum(row.get("success_at_10s", False) for row in executed) / len(executed) if executed else None
        ),
        "timeout": sum(row.get("status") == "TIMEOUT" for row in executed),
        "crash": sum(row.get("status") == "CRASH" for row in executed),
        "watchdog": sum(row.get("status") == "WATCHDOG" for row in executed),
        "blocked": sum(row.get("status") == "BLOCKED" for row in rows),
        "validation_pending": sum(row.get("validation_pending", False) for row in executed),
        "offline_validation_available": sum(row.get("offline_validation_available", False) for row in executed),
        "completion_coverage": len(executed) / len(rows) if rows else None,
    }


def _trial_records(rows, raw_inventory_hash):
    records = []
    for row in rows:
        records.append({
            key: row.get(key)
            for key in ("trial_id", "family", "map_id", "map_seed", "query_id", "planner_id", "planner_seed",
                        "status", "executed", "validation_pending", "offline_validation_available",
                        "success_at_10s", "cost_at_10s", "length_xy_m", "termination_reason",
                        "progress_unavailable_reason")
        } | {
            "t_first": row.get("t_first"),
            "effort_at_10s": row.get("effort_at_10s"),
            "paper_sample_checkpoint": row.get("paper_sample_checkpoint"),
            "reported_metrics": {
                name: row.get("metrics", {}).get(name)
                for name in (*DESCRIPTIVE_TIMING_METRICS, *SECONDARY_EFFORT_METRICS)
            },
            "metric_unavailable_reasons": row.get("metric_unavailable_reasons", {}),
            "snapshot_errors": row.get("errors", []),
            "progress_errors": row.get("progress_errors", []),
            "raw_artifact_inventory_sha256": raw_inventory_hash,
        })
    return records


def _write_csv(path, rows):
    fields = (
        "trial_id", "family", "map_id", "map_seed", "query_id", "planner_id", "planner_seed",
        "status", "executed", "validation_pending", "success_at_10s", "cost_at_10s",
        "t_first_time_s", "t_first_event_observed", "t_first_status", "length_xy_m",
        *DESCRIPTIVE_TIMING_METRICS,
        *(name + "_at_10s" for name in SECONDARY_EFFORT_METRICS),
        *(name + "_final" for name in SECONDARY_EFFORT_METRICS),
    )
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            record = {key: row.get(key) for key in fields}
            record.update(
                t_first_time_s=row.get("t_first", {}).get("time_s"),
                t_first_event_observed=row.get("t_first", {}).get("event_observed"),
                t_first_status=row.get("t_first", {}).get("status"),
            )
            for name in SECONDARY_EFFORT_METRICS:
                record[name + "_at_10s"] = row.get("effort_at_10s", {}).get(name, {}).get("value")
                record[name + "_final"] = row.get("metrics", {}).get(name)
            for name in DESCRIPTIVE_TIMING_METRICS:
                record[name] = row.get("metrics", {}).get(name)
            writer.writerow(record)


def write_report(output, summary, paired, coverage, descriptive):
    def fmt(value):
        return "N/A" if value is None else f"{value:.6g}"

    lines = [
        "# Báo cáo A3 so với SEM-RRT*-adapted common-objective",
        "",
        f"Trạng thái **{summary['status']}**. Đã thực thi **{summary['executed']}/{summary['planned']}** trial; "
        f"validated success trước hoặc tại 10 s: **{summary['validated_success_at_10s']}/{summary['executed']}**. "
        f"Timeout {summary['timeout']}; crash {summary['crash']}; watchdog {summary['watchdog']}; "
        f"chưa chạy {summary['not_run']}; validation pending {summary['validation_pending']}.",
        "",
        "Đây là comparator common-objective: cả hai planner tối thiểu hóa cùng `c_H` và dùng cùng validity contract. "
        "Nó không phải SEM-RRT* đa mục tiêu nguyên bản của bài báo: terrain-risk cost, turning-angle cost và rolling optimization đều tắt.",
        "",
        "## Primary paired estimands",
        "",
        "Chênh lệch luôn là `bitstar_a3 - sem_rrtstar_common_v1`. `c_H@10` và `length_xy_m_final` chỉ dùng cặp cả hai validated success; XY length là số đo phụ, không thay objective. "
        "T_first dùng Kaplan-Meier RMST với tau=10 s; timeout là right-censored, không phải event tại 10 s.",
        "",
        "| Metric | Usable pairs / planned | Estimate | 95% hierarchical bootstrap CI |",
        "|---|---:|---:|---|",
    ]
    for item in paired:
        if item["metric"].startswith("effort_"):
            continue
        interval = "N/A" if item["ci"] is None else "[" + ", ".join(fmt(value) for value in item["ci"]) + "]"
        units = item["units"]
        lines.append(
            f"| {item['metric']} | {units['usable_paired_seed_slots']}/{units['planned_paired_seed_slots']} | "
            f"{fmt(item['estimate'])} | {interval} |"
        )
    lines += [
        "",
        "Bootstrap khóa 2.000 replicate, seed 73135, resample `family → map → paired seed`; nhãn seed ghép cặp không hàm ý hai planner dùng cùng random stream.",
        "",
        "## Secondary effort tại 10 s",
        "",
        "Effort lấy bản ghi progress cuối cùng không muộn hơn checkpoint; không ngoại suy tới 10 s và không dùng counter cuối sau overshoot.",
        "",
        "| Metric | Usable pairs / planned | A3 − SEM estimate | 95% CI |",
        "|---|---:|---:|---|",
    ]
    for item in paired:
        if not item["metric"].startswith("effort_"):
            continue
        interval = "N/A" if item["ci"] is None else "[" + ", ".join(fmt(value) for value in item["ci"]) + "]"
        units = item["units"]
        lines.append(
            f"| {item['metric']} | {units['usable_paired_seed_slots']}/{units['planned_paired_seed_slots']} | "
            f"{fmt(item['estimate'])} | {interval} |"
        )
    lines += [
        "",
        "## Coverage",
        "",
        "| Planner | Metric | Available / denominator | Coverage |",
        "|---|---|---:|---:|",
    ]
    for item in coverage["records"]:
        lines.append(
            f"| {item['planner_id']} | {item['metric']} | {item['available']}/{item['denominator']} | {fmt(item['coverage'])} |"
        )
    lines += [
        "",
        "## Descriptive timing, RSS và final effort",
        "",
        "Các phân phối này là descriptive; timing/RSS/final counters không thay primary estimands và giá trị thiếu không được điền 0.",
        "",
        "| Planner | Metric | Available / executed | Median [Q25, Q75] |",
        "|---|---|---:|---|",
    ]
    for group in descriptive["groups"]:
        if group["scope"] != "planner":
            continue
        for metric_name, stats in group["metrics"].items():
            lines.append(
                f"| {group['planner_id']} | {metric_name} | {stats['available']}/{stats['denominator']} | "
                f"{fmt(stats['median'])} [{fmt(stats['q25'])}, {fmt(stats['q75'])}] |"
            )
    lines += [
        "",
        "Checkpoint chi tiết 0,1/0,25/0,5/1/2/5/10 s và diagnostic khi progress lần đầu đạt ít nhất 5.000 sample proposals nằm trong `checkpoints.json`. Mốc sample giữ cả overshoot do polling, không giả là đo đúng tại proposal thứ 5.000; `trial_metrics.json` giữ censor endpoint, snapshot errors và lý do thiếu counter theo trial.",
        "",
        "Chỉ exact singleton-goal path qua independent validation mới là success. Không gọi timeout là vô nghiệm, không gọi incumbent tốt nhất trong 10 s là tối ưu toàn cục, và không diễn giải conditional cost như hiệu quả vô điều kiện.",
    ]
    if summary["protocol_errors"] or summary["correctness_errors"]:
        lines += ["", "## Correctness errors", ""]
        lines += ["- " + error for error in summary["protocol_errors"]]
        lines += [f"- {item['trial_id']}: {', '.join(map(str, item['errors']))}" for item in summary["correctness_errors"]]
    (Path(output) / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _analyze_into(source, output):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = read_json(source / "resolved_config.json")
    manifest = read_json(source / "manifest.json")
    synthetic = manifest.get("synthetic_test_fixture") is True
    protocol_errors = validate_locked_protocol(config, manifest)
    raw_inventory_hash = None
    try:
        raw_inventory_hash = verify_raw_artifact_inventory(source, required=not synthetic)
    except ValueError as exc:
        protocol_errors.append("raw_artifact_inventory:" + str(exc))

    profile = config.get("profiles", {}).get(PROFILE, {})
    budget = profile.get("planning_budget_s", TAU_S)
    absolute = config.get("model", {}).get("numerical_abs_tolerance_m", 1e-8)
    relative = config.get("model", {}).get("numerical_rel_tolerance", 1e-8)
    entries = manifest.get("trials", [])
    rows, correctness_errors = _safe_read_rows(source, entries, budget, absolute, relative)

    paired_metrics = (
        "validated_success_at_10s",
        "cost_terrain_m_at_10s_both_success",
        "length_xy_m_final_both_success",
        "t_first_rmst_tau_10s",
        *("effort_" + name + "_at_10s" for name in SECONDARY_EFFORT_METRICS),
    )
    paired = [paired_bootstrap(rows, metric) for metric in paired_metrics]
    for item in paired:
        if item.get("errors"):
            correctness_errors.append({"trial_id": "PAIRED_SCHEMA", "errors": item["errors"]})

    checkpoints = checkpoint_summaries(rows)
    coverage = metric_coverage(rows)
    descriptive = descriptive_metrics(rows)
    status = classify_status(protocol_errors, correctness_errors, rows)
    counts = _summary_counts(rows)
    summary = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "experiment_id": manifest.get("experiment_id"),
        "profile": manifest.get("profile"),
        "status": status,
        "status_semantics": {
            "COMPLETE": "all 300 planned trials executed and validation terminal, with no protocol or correctness error",
            "PARTIAL": "protocol-correct retained subset with at least one not-run or validation-pending trial",
            "FAILED_CORRECTNESS": "protocol, identity, snapshot-validation, progress-integrity, or raw-inventory error",
        },
        **counts,
        "tau_s": TAU_S,
        "checkpoints_s": list(CHECKPOINTS_S),
        "paper_sample_checkpoint": PAPER_SAMPLE_CHECKPOINT,
        "bootstrap_hierarchy": list(BOOTSTRAP_HIERARCHY),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "primary_planner_id": PRIMARY_PLANNER,
        "comparator_planner_id": COMPARATOR_PLANNER,
        "contrast": "bitstar_a3_minus_sem_rrtstar_common_v1",
        "cost_conditioning": "both planners independently validated success by 10 s",
        "length_xy_conditioning": "final independently validated paths; both planners validated success by 10 s",
        "t_first_estimand": "Kaplan-Meier restricted mean time to first independently validated exact solution, tau=10 s",
        "secondary_effort_metrics": list(SECONDARY_EFFORT_METRICS),
        "descriptive_timing_metrics": list(DESCRIPTIVE_TIMING_METRICS),
        "protocol_errors": sorted(set(protocol_errors)),
        "correctness_errors": correctness_errors,
        "raw_manifest_sha256": file_hash(source / "manifest.json"),
        "raw_artifact_inventory_sha256": raw_inventory_hash,
        "interpretation": "held-out finite-budget common-objective comparison; SEM comparator is adapted, not paper-faithful native multi-cost SEM-RRT*",
    }

    records = _trial_records(rows, raw_inventory_hash)
    dump(output / "trial_metrics.json", records)
    _write_csv(output / "trials.csv", rows)
    dump(output / "checkpoints.json", checkpoints)
    dump(output / "metric_coverage.json", coverage)
    dump(output / "descriptive_metrics.json", descriptive)
    dump(output / "paired_bootstrap.json", paired)
    write_report(output, summary, paired, coverage, descriptive)
    dump(output / "summary.json", summary)  # Completion marker, always published last.
    return summary


def _publish(staging, output):
    staging, output = Path(staging), Path(output)
    relative = {path.relative_to(staging).as_posix() for path in staging.rglob("*") if path.is_file()}
    if not REQUIRED_REPORT_FILES <= relative:
        raise RuntimeError("staged report missing required files: " + str(sorted(REQUIRED_REPORT_FILES - relative)))
    output.mkdir(parents=True, exist_ok=True)
    previous = output / ".summary.previous.json"
    canonical = output / "summary.json"
    if canonical.exists():
        os.replace(canonical, previous)
    summary = staging / "summary.json"
    for path in sorted((path for path in staging.rglob("*") if path.is_file() and path != summary),
                       key=lambda value: value.relative_to(staging).as_posix()):
        destination = output / path.relative_to(staging)
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(path, destination)
    os.replace(summary, canonical)
    if previous.exists():
        previous.unlink()


def analyze(source, output):
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError("report output must be separate from immutable raw results")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=output.name + ".staging.", dir=output.parent) as directory:
        summary = _analyze_into(source, Path(directory))
        _publish(Path(directory), output)
    print({key: summary[key] for key in (
        "status", "planned", "executed", "validated_success_at_10s", "timeout", "crash",
        "watchdog", "not_run", "validation_pending",
    )})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Immutable raw campaign directory")
    parser.add_argument("--output", required=True, help="Separate report directory")
    args = parser.parse_args()
    analyze(args.input, args.output)


if __name__ == "__main__":
    main()
