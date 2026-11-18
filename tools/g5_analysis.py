"""Frozen G5 budget audit and family-stratified held-out analysis.

No planner execution, target selection, retries, or mutation of raw artifacts.
"""
import argparse
from collections import defaultdict
import csv
import datetime as dt
import json
import math
from pathlib import Path
import random
import statistics

from analyze_results import (COUNTERS, TIMINGS, close, coverage, finite,
                             freeze_references, group_summaries, read_trial,
                             step_cost)
from experiment_common import ROOT, dump, file_hash, read_json


WORK_METRICS = ("full_surface_cost_calls", "motion_validity_calls", "height_queries", "triangle_visits")
GEOMETRY_METRICS = ("informed_area_ratio", "mean_delta_f", "median_delta_f", "p90_delta_f", "max_delta_f")


def quantile(values, p):
    values = sorted(v for v in values if finite(v))
    if not values:
        return None
    index = (len(values)-1)*p
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo]+(values[hi]-values[lo])*(index-lo)


def tail_record(events, budget=5.):
    before, after = step_cost(events, .8*budget), step_cost(events, budget)
    previous, strict = None, False
    for event in sorted(events, key=lambda e: e["elapsed_s"]):
        cost = event["best_cost_terrain_m"]
        if event["elapsed_s"] > budget:
            break
        if .8*budget < event["elapsed_s"] and previous is not None:
            strict |= previous-cost > max(1e-8, 1e-8*abs(previous))
        previous = cost if previous is None else min(previous, cost)
    available = finite(before) and finite(after) and before > 0
    return {"cost_at_80pct_m": before, "cost_at_budget_m": after,
            "relative_tail_gain": (before-after)/before if available else None,
            "gain_unavailable_reason": None if available else "no_positive_validated_incumbent_at_80pct",
            "strict_late_improvement": bool(strict)}


def tail_groups(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    result = []
    for key, group in sorted(groups.items()):
        gains = [r["relative_tail_gain"] for r in group if finite(r["relative_tail_gain"])]
        q25, q75 = quantile(gains, .25), quantile(gains, .75)
        result.append(dict(zip(keys, key), trials=len(group), gain_available=len(gains),
                           gain_null=len(group)-len(gains), median=quantile(gains, .5),
                           q25=q25, q75=q75, iqr=q75-q25 if gains else None,
                           p90=quantile(gains, .9),
                           strict_late_improvement_trials=sum(r["strict_late_improvement"] for r in group),
                           strict_late_improvement_fraction=sum(r["strict_late_improvement"] for r in group)/len(group)))
    return result


def choose_budget(groups):
    return 5. if groups and all(g["gain_null"] == 0 and g["median"] is not None and
                g["median"] <= .005 and g["p90"] <= .02 and
                g["strict_late_improvement_fraction"] <= .25 for g in groups) else 10.


def pilot_tail(output, source=ROOT/"results/pilot_g3g4_v1", report=ROOT/"reports/pilot_g3g4_v1"):
    source, report, output = Path(source), Path(report), Path(output)
    if output.exists():
        raise ValueError("budget audit is immutable; output already exists")
    rule = ROOT/"docs/G5_BUDGET_RULE.md"
    rule_hash = file_hash(rule)
    manifest = read_json(source/"manifest.json")
    curves = read_json(report/"anytime_curves.json")
    rows, trial_sources = [], []
    for entry in manifest["trials"]:
        row = read_trial(source, entry, 5., 1e-8, 1e-8)
        if not row["executed"] or row["errors"]:
            raise ValueError("pilot audit requires executed, independently checked trials: "+entry["trial_id"])
        rows.append(dict((k, entry[k]) for k in ("trial_id", "family", "map_id", "query_id", "planner_id", "planner_seed")) |
                    tail_record(row["events"]))
        trial_sources.append({"trial_id": entry["trial_id"],
                              "validated_events_sha256": file_hash(source/entry["trial_id"]/"validated_events.json")})
    groups = tail_groups(rows, ("family", "planner_id"))
    checks = []
    for curve in curves:
        group = [r for r in rows if (r["family"], r["planner_id"]) == (curve["family"], curve["planner_id"])]
        for time_s, field in ((4., "cost_at_80pct_m"), (5., "cost_at_budget_m")):
            point = [p for p in curve["points"] if p["time_s"] <= time_s][-1]
            expected = quantile([r[field] for r in group], .5)
            observed = point["median_cost_terrain_m"]
            match = expected == observed or close(expected, observed)
            checks.append({"family": curve["family"], "planner_id": curve["planner_id"], "time_s": time_s,
                           "median_from_trial_events": expected, "median_from_pilot_anytime_curves": observed, "match": match})
            if not match:
                raise ValueError("pilot aggregate curve disagrees with retained trial events")
    audit = {"schema_version": 1, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
             "rule_path": str(rule.relative_to(ROOT)), "rule_sha256": rule_hash,
             "pilot_budget_s": 5., "tail_start_s": 4., "chosen_full_budget_s": choose_budget(groups),
             "source_anytime_curves_sha256": file_hash(report/"anytime_curves.json"),
             "source_manifest_sha256": file_hash(source/"manifest.json"),
             "source_note": "Pilot anytime_curves.json contains aggregate curves only; per-trial gain distributions are reconstructed from independently validated snapshot events and cross-checked at 4 s and 5 s against every aggregate curve.",
             "group_rule": "5s iff EVERY family/planner median<=0.005,p90<=0.02,strict_late_fraction<=0.25,no_null; else10s",
             "family_planner": groups,
             "family_planner_query_id": tail_groups(rows, ("family", "planner_id", "query_id")),
             "map_query_planner": tail_groups(rows, ("family", "map_id", "query_id", "planner_id")),
             "aggregate_curve_checks": checks, "trials": rows, "trial_sources": trial_sources}
    dump(output, audit)
    print({"trial_count": len(rows), "group_count": len(groups), "chosen_full_budget_s": audit["chosen_full_budget_s"],
           "groups_with_late_improvement_fraction_gt_025": sum(g["strict_late_improvement_fraction"] > .25 for g in groups),
           "max_late_fraction": max(g["strict_late_improvement_fraction"] for g in groups)})
    return audit


def checkpoints(budget):
    return sorted({0., budget, *(budget*p for p in (.01, .02, .05, .10, .20, .50)),
                   *(s for s in (.1, .25, .5, 1., 2., 5.) if s <= budget)})


def observation(row, threshold=None):
    """Events, observation duration and retained incumbent are distinct fields."""
    budget = row["time_budget_s"]
    actual = row["metrics"].get("actual_solve_wall_s")
    # A deadline is a planned upper bound, not an observed duration. In
    # particular, an administrative interruption has no implicit 10 s exposure.
    end = min(budget, actual) if finite(actual) else None
    eligible = [e for e in row["events"] if row["success"] and e["elapsed_s"] <= budget and
                (threshold is None or e["best_cost_terrain_m"] <= threshold)]
    event = eligible[0] if eligible else None
    counters = {k: event.get("counters", {}).get(k) if event else None for k in WORK_METRICS}
    return {"event_time_s": event["elapsed_s"] if event else None,
            "event_observed": event is not None,
            "observation_end_s": end if row["executed"] else None,
            "observation_end_unavailable_reason": None if row["executed"] and end is not None else
                "not_executed" if not row["executed"] else "solve_duration_not_recorded",
            "termination_time_s": actual,
            "termination_reason": row.get("termination_reason"),
            "status": "NOT_RUN" if not row["executed"] else "ATTAINED" if event else "NO_EVENT_OBSERVED",
            "counters_at_attainment": counters,
            "counter_unavailable_reasons": {k: "no_attainment_event" if event is None else "counter_not_exposed_at_event"
                                            for k, v in counters.items() if v is None}}


def prepare_observations(row, reference, times):
    row["success_attainment"] = observation(row)
    usable_ref = reference.get("status") == "FROZEN_VALIDATED_TARGET"
    threshold = reference.get("cost_terrain_m") if usable_ref else None
    row["frozen_target"] = dict(observation(row, threshold),
                                reference_type="WITNESS_UPPER_BOUND" if usable_ref else None,
                                cost_threshold_m=threshold,
                                reason=None if usable_ref else "no_frozen_reference")
    if not usable_ref:
        row["frozen_target"].update(event_time_s=None, event_observed=False,
                                    counters_at_attainment={k: None for k in WORK_METRICS})
    row["secondary_witness_targets"] = [{"relative_tolerance": epsilon,
        "reference_type": "WITNESS_UPPER_BOUND", "interpretation": "witness_based_target_attainment_not_epsilon_optimality",
        "cost_threshold_m": (1+epsilon)*threshold if usable_ref else None,
        "observation": observation(row, (1+epsilon)*threshold) if usable_ref else None,
        "unavailable_reason": None if usable_ref else "no_frozen_reference"} for epsilon in (.02, .05, .10)]
    row["analysis_values"] = {"success": float(row["success"]),
                               "cost_terrain_m": row["cost_terrain_m"],
                               "first_solution_time_s": row["first_solution_time_s"],
                               "witness_attainment": float(row["frozen_target"]["event_observed"]) if usable_ref else None,
                               "time_to_witness_s": row["frozen_target"]["event_time_s"]}
    row["analysis_values"].update({"total_"+k: row["metrics"].get(k) for k in WORK_METRICS})
    row["analysis_values"].update({"at_witness_"+k: row["frozen_target"]["counters_at_attainment"][k] for k in WORK_METRICS})
    row["checkpoint_observations"] = []
    actual = row["metrics"].get("actual_solve_wall_s")
    for time_s in times:
        cost = step_cost(row["events"], min(time_s, row["time_budget_s"])) if row["success"] else None
        target = bool(usable_ref and cost is not None and cost <= threshold)
        point = {"time_s": time_s, "cost_terrain_m": cost, "valid_attainment": cost is not None,
                 "witness_target_attainment": target if usable_ref else None,
                 "planner_already_terminated": bool(finite(actual) and actual <= time_s),
                 "retained_incumbent_after_termination": bool(cost is not None and finite(actual) and actual < time_s)}
        row["checkpoint_observations"].append(point)
        suffix = format(time_s, ".10g")+"s"
        row["analysis_values"]["attainment_at_"+suffix] = float(cost is not None)
        row["analysis_values"]["cost_at_"+suffix] = cost
        row["analysis_values"]["witness_attainment_at_"+suffix] = float(target) if usable_ref else None


def paired_values(rows, measure, direction="A0_MINUS_A3"):
    slots = defaultdict(dict)
    for row in rows:
        if row["planner_id"] not in ("bitstar_a0", "bitstar_a3") or not row["executed"]:
            continue
        key = (row["map_id"], row["query_id"], row["planner_seed"])
        if row["planner_id"] in slots[key]:
            raise ValueError("duplicate paired slot")
        slots[key][row["planner_id"]] = row
    tree = defaultdict(lambda: defaultdict(list))
    unpaired = unavailable = 0
    for (map_id, query_id, _), pair in slots.items():
        if set(pair) != {"bitstar_a0", "bitstar_a3"}:
            unpaired += 1
            continue
        a, b = pair["bitstar_a0"]["analysis_values"].get(measure), pair["bitstar_a3"]["analysis_values"].get(measure)
        if not finite(a) or not finite(b):
            unavailable += 1
            continue
        tree[map_id][query_id].append((a-b) if direction == "A0_MINUS_A3" else (b-a))
    units = {"maps": len(tree), "map_query_units": sum(len(q) for q in tree.values()),
             "paired_observations": sum(len(v) for q in tree.values() for v in q.values()),
             "unpaired_executed_slots": unpaired, "unavailable_complete_pairs": unavailable}
    return tree, units


def hierarchy_draws(tree, replicates, seed, geometry=None):
    """Vectorized independent map -> query -> paired-seed bootstrap draws.

    Family is a fixed stratum. Every map and query gets equal weight even with
    unequal numbers of complete paired observations. Geometry adds x/y moments.
    """
    import numpy as np
    maps = sorted(tree)
    rng = np.random.default_rng(seed)
    selected_maps = rng.integers(len(maps), size=(replicates, len(maps)))
    dimensions = 5 if geometry is not None else 1
    draws = np.zeros((replicates, len(maps), dimensions))
    for index, map_id in enumerate(maps):
        locations = np.where(selected_maps == index)
        occurrences = len(locations[0])
        queries = sorted(tree[map_id])
        selected_queries = rng.integers(len(queries), size=(occurrences, len(queries)))
        values = np.zeros((occurrences, len(queries), dimensions))
        for qi, query_id in enumerate(queries):
            selected = np.where(selected_queries == qi)
            samples = np.asarray(tree[map_id][query_id])
            indices = rng.integers(len(samples), size=(len(selected[0]), len(samples)))
            means = samples[indices].mean(axis=1)
            if geometry is None:
                values[selected[0], selected[1], 0] = means
            else:
                x = geometry[(map_id, query_id)]
                values[selected[0], selected[1]] = np.column_stack((np.full(len(means), x), means,
                    np.full(len(means), x*x), means*means, x*means))
        draws[locations] = values.mean(axis=1)
    return draws.mean(axis=1)


def paired_bootstrap(rows, measure, replicates=2000, seed=73130):
    tree, units = paired_values(rows, measure)
    result = {"metric": measure, "difference": "A0_MINUS_A3", "units": units,
              "replicates": replicates, "seed": seed, "confidence_level": .95,
              "estimate": None, "ci": None,
              "estimand": "equal_map_weight_then_equal_query_weight_mean_paired_seed_difference",
              "hierarchy": ["family_fixed_stratum", "map", "query", "paired_seed"],
              "conditioning": "complete_executed_pairs_with_both_metric_values; cost requires validated incumbent; time/work at target requires both attaining frozen target"}
    if not tree:
        result["status"] = "UNAVAILABLE_NO_PAIRED_OBSERVATIONS"
        return result
    result["estimate"] = statistics.mean(statistics.mean(statistics.mean(v) for v in qs.values()) for qs in tree.values())
    if len(tree) < 2:
        result["status"] = "INSUFFICIENT_INDEPENDENT_MAPS_FOR_CI"
        return result
    samples = hierarchy_draws(tree, replicates, seed)[:, 0].tolist()
    result.update(status="AVAILABLE_POINTWISE_CI", ci=[quantile(samples, .025), quantile(samples, .975)])
    return result


def geometry_association(rows, geometry, measure, covariate, replicates=2000, seed=73130):
    tree, _ = paired_values(rows, measure, "A3_MINUS_A0")
    xs, clean, query_effects = {}, defaultdict(dict), []
    for map_id, queries in tree.items():
        for query_id, values in queries.items():
            x = geometry.get((map_id, query_id), {}).get(covariate)
            if finite(x):
                xs[(map_id, query_id)] = x
                clean[map_id][query_id] = values
                query_effects.append({"map_id": map_id, "query_id": query_id, covariate: x,
                                      "mean_effect_A3_minus_A0": statistics.mean(values), "paired_seed_count": len(values)})
    result = {"metric": measure, "covariate": covariate, "effect_direction": "A3_MINUS_A0",
              "maps": len(clean), "queries": len(query_effects),
              "paired_observations": sum(len(v) for q in clean.values() for v in q.values()),
              "pearson_r": None, "pearson_r_ci": None, "slope": None, "slope_ci": None,
              "query_effects": query_effects, "replicates": replicates, "seed": seed,
              "interpretation": "within_family_equal_map_and_query_weight_association; map_query_paired_seed_bootstrap; descriptive_not_causal"}
    if not clean:
        result["status"] = "UNAVAILABLE_NO_PAIRED_GEOMETRY"
        return result
    moments = []
    for map_id, queries in clean.items():
        qm = []
        for query_id, values in queries.items():
            x, y = xs[(map_id, query_id)], statistics.mean(values)
            qm.append((x, y, x*x, y*y, x*y))
        moments.append([statistics.mean(v[i] for v in qm) for i in range(5)])
    point = [statistics.mean(v[i] for v in moments) for i in range(5)]
    def relation(m):
        vx, vy, cov = m[2]-m[0]*m[0], m[3]-m[1]*m[1], m[4]-m[0]*m[1]
        slope = cov/vx if vx > 1e-16 else None
        corr = max(-1., min(1., cov/math.sqrt(vx*vy))) if vx > 1e-16 and vy > 1e-16 else None
        return slope, corr
    result["slope"], result["pearson_r"] = relation(point)
    if len(clean) < 2:
        result["status"] = "INSUFFICIENT_INDEPENDENT_MAPS_FOR_CI"
        return result
    samples = [relation(m) for m in hierarchy_draws(clean, replicates, seed, xs)]
    slopes, corrs = [s[0] for s in samples], [s[1] for s in samples]
    result.update(status="AVAILABLE_DESCRIPTIVE_ASSOCIATION", slope_ci=[quantile(slopes, .025), quantile(slopes, .975)],
                  pearson_r_ci=[quantile(corrs, .025), quantile(corrs, .975)],
                  undefined_bootstrap_slopes=sum(v is None for v in slopes),
                  undefined_bootstrap_correlations=sum(v is None for v in corrs))
    return result


def _path_points(path, dimensions):
    """Read exported bytes strictly; malformed paths are not geometry failures."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header not in (["x_m", "y_m", "z_m"][:dimensions], ["x", "y", "z"][:dimensions]):
            raise ValueError("invalid_path_header")
        points = [[float(value) for value in record] for record in reader]
    if not points or any(len(point) != dimensions or not all(finite(v) for v in point) for point in points):
        raise ValueError("empty_or_malformed_path")
    return points


def _artifact_binding(directory, raw, identity):
    """Bind raw exports, without changing the worker or verifier schema.

    The v2 harness writes this sidecar before offline validation. Legacy v1
    artifacts have no sidecar and retain their snapshot checksum contract.
    """
    path = directory/"artifact_hashes.json"
    required = identity.get("analyzer_schema_version") == 2
    if not path.exists():
        return ["missing_artifact_checksums"] if required else []
    document = read_json(path)
    files = document.get("files")
    if document.get("schema_version") != 1 or not isinstance(files, dict):
        return ["malformed_artifact_checksums"]
    errors = []
    for name, checksum in files.items():
        candidate = directory/name
        if (not isinstance(name, str) or not candidate.resolve().is_relative_to(directory.resolve()) or
                not isinstance(checksum, str) or len(checksum) != 64):
            errors.append("malformed_artifact_checksum:"+str(name))
        elif not candidate.is_file() or file_hash(candidate) != checksum:
            errors.append("artifact_checksum_mismatch:"+name)
    # Administrative interruption may precede worker serialization. Every raw
    # file that actually exists still needs a checksum; never invent a worker.
    administrative = bool(raw.get("administrative_interruption"))
    published = raw.get("execution_status") in ("COMPLETED", "TIMEOUT") and not administrative
    mandatory = {"request.json", "worker.json", "progress.jsonl"} if published else set()
    mandatory.update(name for name in ("request.json", "worker.json", "progress.jsonl", "path_xy.csv", "path_surface.csv")
                     if (directory/name).exists())
    if raw.get("path_available"):
        mandatory.update(("path_xy.csv", "path_surface.csv"))
    mandatory.update(str(p.relative_to(directory)).replace("\\", "/")
                     for p in (directory/"snapshots").glob("*/path_*.csv"))
    normalized = {name.replace("\\", "/") for name in files}
    errors.extend("missing_artifact_checksum:"+name for name in sorted(mandatory-normalized))
    backup_path = directory/"trial_before_validation.json"
    if identity.get("require_raw_trial_backup") and "trial_before_validation.json" not in normalized:
        errors.append("missing_raw_trial_backup_checksum")
    if backup_path.exists():
        original = read_json(backup_path)
        immutable = ("experiment_id", "trial_id", "map_id", "map_hash", "query_id", "planner_id",
                     "planner_seed", "query_hash", "execution_status", "planner_status", "path_available",
                     "best_cost_terrain_m", "time_budget_s", "termination_reason", "code_sha256",
                     "config_hash", "source_sha256", "binary_sha256", "dependency_sha256", "config_sha256")
        errors.extend("raw_trial_field_changed_after_validation:"+key for key in immutable
                      if key in original and original.get(key) != raw.get(key))
    return errors


def _check_event_origins(directory, events):
    """Validated snapshots must refer to real retained improvement events."""
    path = directory/"progress.jsonl"
    if not path.exists():
        return ["missing_progress_for_incumbent_events"] if events else []
    def invalid_constant(value):
        raise ValueError("nonfinite_progress:"+value)
    records = [json.loads(line, parse_constant=invalid_constant)
               for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    improvements = [r for r in records if r.get("event_type") == "improvement"]
    if len(improvements) != len(events):
        return ["validated_event_count_mismatch"]
    errors = []
    for original, checked in zip(improvements, events):
        # independent_valid is added by the offline verifier, not the worker.
        if original != {k: v for k, v in checked.items() if k != "independent_valid"}:
            errors.append("validated_event_origin_mismatch:"+str(checked.get("snapshot_id")))
    return errors


def read_g5_trial(source, entry, budget, absolute, relative, identity):
    """Classify execution, geometry and exact attainment independently.

    Approximate output is a candidate path, not an exact incumbent claim. An
    invalid candidate only becomes a correctness error when evidence contradicts
    a claim or is malformed/unbound. No event or duration is fabricated here.
    The legacy G3/G4 reader and independent geometry implementation stay intact.
    """
    source = Path(source)
    directory = source/entry["trial_id"]
    taxonomy = dict(path_present=False, final_path_present=False, geometry_valid=None,
                    exact_goal_reached=False, exact_incumbent_present=False,
                    planner_status="NOT_RUN", execution_status="NOT_RUN",
                    validated_success=False, correctness_error=False,
                    administrative_interruption=False, native_ompl_crash=None)
    try:
        row = read_trial(source, entry, budget, absolute, relative)
        row.update(taxonomy)
        if "raw" not in row:
            return row
        raw = row["raw"]
        row.update(planner_status=raw.get("planner_status", "UNKNOWN"),
                   execution_status=raw["execution_status"],
                   administrative_interruption=bool(raw.get("administrative_interruption")))
        if row["administrative_interruption"]:
            row.update(status="ADMINISTRATIVE_INTERRUPTION", native_ompl_crash=False)
            for name in (*COUNTERS, *TIMINGS):
                if row["metrics"].get(name) is None:
                    row["metric_unavailable_reasons"][name] = raw.get(name+"_unavailable_reason",
                        "not_recorded_before_administrative_interruption")
        errors = row["errors"]
        for key in ("code_sha256", "config_hash"):
            if key in identity and raw.get(key) != identity[key]:
                errors.append("trial_provenance_mismatch:"+key)
        for key in ("query_hash", "source_sha256", "binary_sha256", "dependency_sha256", "config_sha256"):
            if key in entry and raw.get(key) != entry[key]:
                errors.append("trial_provenance_mismatch:"+key)
        errors.extend(_artifact_binding(directory, raw, identity))
        exact_claim = raw.get("planner_status") == "EXACT_SOLUTION"
        path_claim = raw.get("path_available") is True
        if not isinstance(raw.get("path_available", False), bool):
            errors.append("malformed_path_available")
        # The legacy reader assumes every path is an exact solution. Remove only
        # those two generic errors; the specific evidence checks below replace
        # them, while snapshot checksum and all other errors remain untouched.
        errors[:] = [e for e in errors if e not in ("missing_validated_events", "final_path_not_validated")]
        request_path = directory/"request.json"
        request = read_json(request_path) if request_path.exists() else {}
        if request:
            for key in ("planner_id", "planner_seed", "start", "goal"):
                if key in entry and request.get(key) != entry[key]:
                    errors.append("request_provenance_mismatch:"+key)
            if not close(request.get("planning_budget_s"), budget, absolute, relative):
                errors.append("request_budget_mismatch")
            if "map_path" in entry and Path(request.get("map_path", "")).resolve() != Path(entry["map_path"]).resolve():
                errors.append("request_provenance_mismatch:map_path")
        if not close(row["time_budget_s"], budget, absolute, relative):
            errors.append("trial_budget_mismatch")
        if exact_claim and not path_claim:
            errors.append("exact_solution_without_final_path_claim")
        errors.extend(_check_event_origins(directory, row["events"]))
        if row["events"]:
            for event in row["events"]:
                points = _path_points(directory/"snapshots"/event["snapshot_id"]/"path_xy.csv", 2)
                if (not request or math.dist(points[0], request["start"]) > 1e-9 or
                        math.dist(points[-1], request["goal"]) > 1e-9):
                    errors.append("snapshot_not_exact_start_goal:"+event["snapshot_id"])
        row["exact_incumbent_present"] = bool(row["events"])
        row["final_path_present"] = (directory/"path_xy.csv").exists()
        row["path_present"] = row["final_path_present"] or row["exact_incumbent_present"]
        if row["final_path_present"] != path_claim:
            errors.append("final_path_presence_claim_mismatch")
        if path_claim:
            xy = _path_points(directory/"path_xy.csv", 2)
            surface = _path_points(directory/"path_surface.csv", 3)
            validation = read_json(directory/"validation.json")
            if not isinstance(validation.get("valid"), bool) or not request:
                raise ValueError("missing_final_validation_or_request")
            row["geometry_errors"] = validation.get("errors", [])
            # Oracle.validate_path also checks its requested endpoint and the
            # claimed cost. Neither changes whether the exported surface path
            # itself is geometrically traversable. Keep these claims separate.
            nongeometry_claim_errors = {"not_exact_start_goal", "cost_mismatch"}
            row["geometry_valid"] = bool(validation["valid"] or
                (row["geometry_errors"] and set(row["geometry_errors"]) <= nongeometry_claim_errors))
            row["exact_goal_reached"] = math.dist(xy[-1], request["goal"]) <= 1e-9
            row["final_path_sha256"] = {name: file_hash(directory/name) for name in ("path_xy.csv", "path_surface.csv")}
            if "exact_goal_reached" in validation and validation["exact_goal_reached"] != row["exact_goal_reached"]:
                errors.append("final_goal_validation_mismatch")
            if exact_claim and (not row["exact_goal_reached"] or validation.get("exact_goal_reached") is not True):
                errors.append("final_goal_not_independently_verified")
            # Legacy final exports do not carry checksums in validation.json.
            # If either optional checksum is present, both must be correct.
            if "xy_sha256" in validation or "surface_sha256" in validation:
                for key, name in (("xy_sha256", "path_xy.csv"), ("surface_sha256", "path_surface.csv")):
                    if validation.get(key) != row["final_path_sha256"][name]:
                        errors.append("final_path_hash_mismatch:"+name)
            if exact_claim or row["exact_incumbent_present"]:
                if not row["events"]:
                    errors.append("missing_exact_incumbent_snapshot")
                else:
                    snapshot = directory/"snapshots"/row["events"][-1]["snapshot_id"]
                    for name in ("path_xy.csv", "path_surface.csv"):
                        if row["final_path_sha256"][name] != file_hash(snapshot/name):
                            errors.append("final_path_snapshot_hash_mismatch:"+name)
            if not (exact_claim or row["exact_incumbent_present"]) or not validation["valid"] or not row["exact_goal_reached"]:
                # Approximate paths have no exact snapshot by design. Recheck
                # their geometry read-only with the unchanged independent Oracle
                # so legacy verification is still bound to the actual bytes.
                from independent_verifier import Oracle
                map_path = Path(request["map_path"])
                if file_hash(map_path) != entry.get("map_hash"):
                    errors.append("approximate_map_hash_mismatch")
                oracle = Oracle(map_path)
                checked = oracle.validate_path(xy, request["start"], xy[-1], None, surface)
                row["geometry_valid"] = checked["valid"]
                row["geometry_errors"] = checked["errors"]
                row["final_verification_method"] = "read_only_independent_oracle_for_nonincumbent_path"
                # The legacy verifier includes endpoint agreement in `valid`.
                # Separate it here: a wrong-goal exact claim may have perfectly
                # valid geometry while remaining a correctness contradiction.
                expected_validation_valid = checked["valid"] and (row["exact_goal_reached"] if exact_claim else True)
                if expected_validation_valid != validation["valid"]:
                    errors.append("final_geometry_validation_mismatch")
                if finite(checked.get("cost_terrain_m")) and not close(
                        checked["cost_terrain_m"], raw.get("best_cost_terrain_m"), absolute, relative):
                    errors.append("final_oracle_cost_claim_mismatch")
                if any(e in checked["errors"] for e in ("off_surface_point", "surface_length_mismatch",
                        "missing_triangle_breakpoint", "missing_or_nonfinite_surface", "exported_nodata")):
                    errors.append("final_surface_evidence_mismatch")
            if finite(validation.get("cost_terrain_m")) and not close(
                    raw.get("best_cost_terrain_m"), validation["cost_terrain_m"], absolute, relative):
                errors.append("final_validated_cost_mismatch")
            if not row["geometry_valid"] and (exact_claim or raw.get("independent_valid") is True or raw.get("success") is True):
                errors.append("final_geometry_contradicts_validity_claim")
        elif row["exact_incumbent_present"]:
            # An interruption/timeout can retain a fully verified incumbent even
            # if final serialization was never published. Only its actual event
            # time contributes to quality; there is no backfilled final event.
            row.update(geometry_valid=True, exact_goal_reached=True,
                       final_verification_method="retained_exact_incumbent_snapshot")
        if raw.get("planner_status") == "APPROXIMATE_SOLUTION" and row["exact_incumbent_present"]:
            errors.append("approximate_status_contradicts_exact_incumbent")
        if raw.get("success") is True and not (row["geometry_valid"] and row["exact_goal_reached"] and row["exact_incumbent_present"]):
            errors.append("success_claim_without_exact_evidence")
        in_budget = [e for e in row["events"] if e["elapsed_s"] <= budget]
        if raw.get("success") is True and not in_budget:
            errors.append("success_claim_without_in_budget_incumbent")
        row["correctness_error"] = bool(errors)
        row["validated_success"] = bool(row["executed"] and row["geometry_valid"] and row["exact_goal_reached"] and
                                         in_budget and not row["correctness_error"])
        row["success"] = row["validated_success"]
        row["cost_terrain_m"] = min(e["best_cost_terrain_m"] for e in in_budget) if row["success"] else None
        row["first_solution_time_s"] = in_budget[0]["elapsed_s"] if row["success"] else None
        row["snapshot_quality_usable"] = not row["correctness_error"]
        return row
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError) as exc:
        # A malformed artifact is a structured correctness failure, not a
        # timeout and not an unhandled exception that loses the trial identity.
        if "row" not in locals():
            row = dict(entry, executed=(directory/"trial.json").exists(), status="MALFORMED_ARTIFACT",
                       events=[], errors=[], time_budget_s=budget, metrics={}, metric_unavailable_reasons={},
                       improvements_beyond_tolerance=0, length_xy_m=None, independently_recomputed_cost_m=None)
            row.update(taxonomy)
        row["errors"].append("malformed_or_missing_artifact:"+type(exc).__name__+":"+str(exc))
        row.update(correctness_error=True, validated_success=False, success=False,
                   cost_terrain_m=None, first_solution_time_s=None, snapshot_quality_usable=False)
        return row


def checkpoint_curves(rows, times):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["family"], row["planner_id"])].append(row)
    result = []
    for (family, planner), group in sorted(groups.items()):
        executed = [r for r in group if r["executed"]]
        points = []
        for index, time_s in enumerate(times):
            observed = [r["checkpoint_observations"][index] for r in executed]
            costs = [p["cost_terrain_m"] for p in observed if p["cost_terrain_m"] is not None]
            refs = [p for p in observed if p["witness_target_attainment"] is not None]
            points.append({"time_s": time_s, "executed_denominator": len(executed),
                           "valid_solution_count": len(costs),
                           "valid_solution_attainment": len(costs)/len(executed) if executed else None,
                           "median_cost_terrain_m": quantile(costs, .5),
                           "q25_cost_terrain_m": quantile(costs, .25), "q75_cost_terrain_m": quantile(costs, .75),
                           "witness_target_denominator": len(refs),
                           "witness_target_attainment": sum(p["witness_target_attainment"] for p in refs)/len(refs) if refs else None,
                           "terminated_trials": sum(p["planner_already_terminated"] for p in observed)})
        result.append({"family": family, "planner_id": planner, "planned": len(group), "executed": len(executed), "points": points})
    return result


def write_full_report(output, summary, bootstrap, associations):
    def fmt(value):
        return "null" if value is None else f"{value:.6g}"
    def line(item):
        units = item["units"]
        interval = "null" if item["ci"] is None else "["+", ".join(fmt(x) for x in item["ci"])+"]"
        return f"| {item['family']} | {item['metric']} | {fmt(item['estimate'])} | {interval} | {units['maps']} / {units['map_query_units']} / {units['paired_observations']} |"
    def table(metrics):
        return ["", "| Family | Metric (A0 − A3) | Paired estimate | 95% CI | Maps / queries / paired observations |",
                "|---|---|---:|---|---:|", *[line(b) for b in bootstrap if b["metric"] in metrics]]
    lines = ["# G5 held-out evaluation", "",
             f"Trạng thái **{summary['status']}**; planned/executed/validated success/not_run = "
             f"**{summary['planned']}/{summary['executed']}/{summary['validated_success']}/{summary['not_run']}**. "
             f"Correctness errors: {summary['invalid']}; crash: {summary['crash']}; watchdog: {summary['watchdog']}; timeout: {summary['timeout']}.", "",
             "Bốn planner dùng cùng query, map, footprint, singleton goal, terrain objective và ngân sách. "
             "Không hậu xử lý hoặc đưa witness vào planner. Witness là **WITNESS_UPPER_BOUND**, không phải optimum. "
             "Primary target là cost ≤ witness cost đã khóa trước trial.", "",
             "Đơn vị độc lập là map. Family là tầng cố định; bootstrap map → query → paired seed, "
             "2.000 resample seed 73130, trọng số đều cho map rồi query. CI 95% là pointwise, "
             "không điều chỉnh đa so sánh. Seed slot ghép cặp không bảo đảm cùng chuỗi mẫu. "
             "CI chứa 0 không chứng minh equivalence. Không có xếp hạng gộp các family.", "",
             "| Family | Planner | Valid / executed / planned | Median c_H tại budget (m) | Median first solution (s) |",
             "|---|---|---:|---:|---:|"]
    for g in summary["family_groups"]:
        lines.append(f"| {g['family']} | {g['planner_id']} | {g['validated_success']}/{g['executed']}/{g['planned']} | "
                     f"{fmt(g['cost_terrain_m_at_budget']['median'])} | {fmt(g['first_solution_s']['median'])} |")
    lines += ["", "## RQ1. A3 có giữ correctness/success so với A0 không?", "",
              "Success yêu cầu exact goal qua oracle độc lập và snapshot hợp lệ trong budget; mọi failure đã thực thi nằm trong mẫu số. "
              "Đọc paired attainment theo thời gian và success tại budget dưới đây. CI [0,0] khi mọi quan sát đều success "
              "chỉ mô tả mẫu đã thấy; không bảo đảm success quần thể hoặc chứng minh non-inferiority."]
    lines += table({"success"})
    lines += ["", "## RQ2. A3 có giảm computational work ở cùng mức quality không?", "",
              "So sánh bốn counter tại event hợp lệ đầu tiên đạt cùng frozen witness target. "
              "Time/counter tại target là null nếu không đạt; các paired work estimate có điều kiện cả hai đạt target. "
              "Bảng witness_attainment báo riêng coverage nên không diễn giải việc thiếu trial khó thành giảm work. "
              "Counter cuối solve là mô tả budget chung, không tự chứng minh cùng quality."]
    lines += table({"witness_attainment", *("at_witness_"+k for k in WORK_METRICS)})
    lines += ["", "## RQ3. A3 có cải thiện cost-vs-time hoặc time-to-target không?", "",
              "Cost tại checkpoint dùng step function từ event thật; trước first solution là null. "
              "Không backfill; nếu dừng sớm, lưu thời gian/lý do dừng và giữ incumbent, không tạo event hoặc công việc mới. "
              "Cost so sánh có điều kiện cả hai có incumbent; time-to-target có điều kiện cả hai đạt target. "
              "anytime_curves.json giữ từng event, checkpoint và attainment; paired_bootstrap.json giữ CI mọi checkpoint."]
    lines += table({"cost_terrain_m", "time_to_witness_s"})
    lines += ["", "## RQ4. Lợi ích/chi phí A3 thay đổi thế nào theo informed_area_ratio và delta_f?", "",
              "Geometry được đóng băng trước planner result analysis. Phân tích effect A3−A0 theo từng family; "
              "Pearson r và slope trên trung bình seed của query, trọng số đều map/query, CI bootstrap map/query/paired seed. "
              "Đây là association mô tả, không phải quan hệ nhân quả. Không dùng geometry hoặc thắng/thua để loại query.", "",
              "| Family | Effect metric (A3 − A0) | Geometry | Pearson r | 95% CI | Maps / queries |",
              "|---|---|---|---:|---|---:|"]
    for item in associations:
        ci = item["pearson_r_ci"]
        interval = "null" if ci is None else "["+", ".join(fmt(v) for v in ci)+"]"
        lines.append(f"| {item['family']} | {item['metric']} | {item['covariate']} | {fmt(item['pearson_r'])} | {interval} | {item['maps']} / {item['queries']} |")
    lines += ["", "## RQ5. Có family nào A3 bất lợi ổn định không?", "",
              "Chỉ đánh dấu bằng chứng bất lợi cho một metric/family khi CI A0−A3 nằm hoàn toàn dưới 0 "
              "đối với cost/time/work (hoặc hoàn toàn trên 0 đối với attainment). Đây là CI pointwise; "
              "không suy ra bất lợi ổn định trên mọi metric, query hay family chưa quan sát."]
    adverse = []
    for b in bootstrap:
        if b["metric"] not in {"success", "cost_terrain_m", "time_to_witness_s", *("at_witness_"+k for k in WORK_METRICS)} or not b["ci"]:
            continue
        bad = b["ci"][0] > 0 if b["metric"] == "success" else b["ci"][1] < 0
        if bad:
            adverse.append(b)
    lines += ["", *([f"- {b['family']}: {b['metric']}, CI {b['ci']}." for b in adverse] or
                     ["Chưa có metric/family primary nào thỏa tiêu chí CI bất lợi nói trên; điều này không chứng minh equivalence."])]
    lines += ["", "## Giới hạn và provenance", "",
              "Kết quả cost có điều kiện success; time/work tại target có điều kiện attainment. "
              "Null đi kèm lý do trong trial_metrics.json; RSS là sampled peak. "
              "Full terrain edge evaluations dùng full_surface_cost_calls, không dùng PathGeometric::length(). "
              "Timeout không có nghĩa vô nghiệm; EXACT_SOLUTION không có nghĩa tối ưu toàn cục.", "",
              f"Manifest SHA-256: `{summary['raw_manifest_sha256']}`. Geometry SHA-256: `{summary['geometry_sha256']}`. "
              "Source/binary/dependency/config identity nằm trong summary.json và immutable manifest. "
              "Nếu trạng thái PARTIAL/BLOCKED, các bảng chỉ mô tả số trial thực có, không thay cho full completion."]
    (output/"REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def analyze(source, output):
    source, output = Path(source), Path(output)
    manifest = read_json(source/"manifest.json")
    if manifest.get("experiment_id") == "full_g5_v1":
        raise ValueError("G5-v1 is immutable historical data; use the separate read-only forensic harness")
    if output.resolve() == source.resolve() or source.resolve() in output.resolve().parents:
        raise ValueError("analysis output must be outside the raw artifact directory")
    lock = read_json(source/"manifest.lock.json")
    if file_hash(source/"manifest.json") != lock["sha256"]:
        raise ValueError("STOP_GATE manifest hash mismatch")
    config = read_json(source/"resolved_config.json")
    identity = manifest["identity"]
    from experiment_common import digest, code_hash
    if digest(config) != identity["config_hash"] or code_hash() != identity["code_sha256"]:
        raise ValueError("STOP_GATE config/source hash mismatch")
    geometry_path = source/"terrain_informed_geometry.json"
    candidates = [value for name, value in identity["g5_frozen_files"].items()
                  if Path(name).name == "terrain_informed_geometry.json"]
    if len(candidates) != 1:
        raise ValueError("STOP_GATE missing or ambiguous frozen geometry identity")
    expected_geometry = candidates[0]
    if file_hash(geometry_path) != expected_geometry:
        raise ValueError("STOP_GATE frozen geometry hash mismatch")
    geometry_document = read_json(geometry_path)
    geometry = {(g["map_id"], g["query_id"]): g for g in geometry_document["queries"]}
    if len(geometry) != 160:
        raise ValueError("G5 geometry must contain 160 unique map/query units")
    # Never call bootstrap or correlation for a partial/stopped/unvalidated
    # dataset, even if the subset that happened to finish has valid geometry.
    state = read_json(source/"execution_state.json")
    validation_summary = read_json(source/"validation_summary.json")
    expected_count = len(manifest["trials"])
    if (state.get("status") != "COMPLETE_VALIDATED" or
            validation_summary.get("status") != "COMPLETE_VALIDATED" or
            validation_summary.get("executed") != expected_count or
            validation_summary.get("not_run") != 0 or
            validation_summary.get("correctness_errors", validation_summary.get("invalid", 0)) not in (0, [])):
        raise ValueError("STOP_GATE full completion and independent validation required before inference")
    output.mkdir(parents=True, exist_ok=True)
    absolute, relative = config["model"]["numerical_abs_tolerance_m"], config["model"]["numerical_rel_tolerance"]
    budget = config["profiles"]["full"]["planning_budget_s"]
    times = checkpoints(budget)
    references, reference_hash = freeze_references(manifest, output, absolute, relative)
    rows = []
    for entry in manifest["trials"]:
        g = geometry[(entry["map_id"], entry["query_id"])]
        if g["query_hash"] != entry["query_hash"] or g["map_sha256"] != entry["map_hash"]:
            raise ValueError("STOP_GATE geometry query identity mismatch")
        row = read_g5_trial(source, entry, budget, absolute, relative, identity)
        ref = references[entry["map_id"]+"/"+entry["query_id"]]
        prepare_observations(row, ref, times)
        row["geometry"] = {k: g[k] for k in GEOMETRY_METRICS}
        rows.append(row)
    summary = coverage(rows)
    summary.update(status="BLOCKED" if state["status"] == "BLOCKED" or summary["invalid"] else "PARTIAL" if summary["not_run"] else "COMPLETE_VALIDATED",
                   identity=identity, budget_s=budget, raw_manifest_sha256=file_hash(source/"manifest.json"),
                   geometry_sha256=file_hash(geometry_path), reference_sha256=reference_hash,
                   family_groups=group_summaries(rows, ("family", "planner_id")),
                   query_groups=group_summaries(rows, ("family", "map_id", "query_id", "planner_id")),
                   correctness_errors=[{"trial_id": r["trial_id"], "errors": r["errors"]} for r in rows if r["errors"]])
    summary.update(administrative_interrupted_attempts=sum(r["administrative_interruption"] for r in rows),
                   geometric_invalid_outputs=sum(r["geometry_valid"] is False for r in rows),
                   approximate_non_success=sum(r["planner_status"] == "APPROXIMATE_SOLUTION" and not r["validated_success"] for r in rows),
                   full_statistical_analysis="NOT_RUN" if summary["status"] != "COMPLETE_VALIDATED" else "AUTHORIZED_COMPLETE_VALIDATED")
    if summary["status"] != "COMPLETE_VALIDATED":
        dump(output/"summary.json", summary)
        raise ValueError("STOP_GATE every primary trial must pass analyzer correctness before inference")
    bootstrap, associations = [], []
    measures = list(rows[0]["analysis_values"]) if rows else []
    for family in config["profiles"]["full"]["families"]:
        selected = [r for r in rows if r["family"] == family]
        for measure in measures:
            bootstrap.append(dict(paired_bootstrap(selected, measure), family=family))
        for measure in ("cost_terrain_m", "witness_attainment", "time_to_witness_s",
                        *("at_witness_"+k for k in WORK_METRICS), *(m for m in measures if m.startswith("cost_at_"))):
            for covariate in GEOMETRY_METRICS:
                associations.append(dict(geometry_association(selected, geometry, measure, covariate), family=family))
    fields = ["trial_id", "family", "map_id", "query_id", "planner_id", "planner_seed", "status", "executed", "success",
              "path_present", "geometry_valid", "exact_goal_reached", "exact_incumbent_present", "planner_status",
              "execution_status", "validated_success", "correctness_error", "administrative_interruption",
              "cost_terrain_m", "first_solution_time_s", "termination_reason", *COUNTERS, *TIMINGS,
              "witness_target_attained", "time_to_witness_s", *GEOMETRY_METRICS]
    with (output/"trials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = {k: row.get(k) for k in fields}
            flat.update(row["metrics"])
            flat.update(row["geometry"])
            flat.update(witness_target_attained=row["frozen_target"]["event_observed"], time_to_witness_s=row["frozen_target"]["event_time_s"])
            writer.writerow(flat)
    dump(output/"summary.json", summary)
    dump(output/"paired_bootstrap.json", {"comparisons": bootstrap, "geometry_associations": associations,
         "hierarchy": ["family_fixed_stratum", "map", "query", "paired_seed"], "confidence_level": .95,
         "replicates": 2000, "seed": 73130, "multiplicity": "pointwise_CI_no_familywise_adjustment"})
    dump(output/"terrain_informed_geometry.json", geometry_document)
    dump(output/"anytime_curves.json", {"checkpoints_s": times, "family_curves": checkpoint_curves(rows, times),
         "trials": [{"trial_id": r["trial_id"], "events": r["events"], "checkpoints": r["checkpoint_observations"],
                     "termination": r["success_attainment"], "target": r["frozen_target"]} for r in rows],
         "interpretation": "right-continuous step function of verified events; null before first solution; retained incumbent is not continued planner work"})
    dump(output/"trial_metrics.json", [{k: v for k, v in r.items() if k != "raw"} for r in rows])
    write_full_report(output, summary, bootstrap, associations)
    print({k: summary[k] for k in ("status", "planned", "executed", "validated_success", "not_run", "invalid")})
    if summary["invalid"]:
        raise ValueError("STOP_GATE correctness errors in G5 analysis")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    tail = sub.add_parser("pilot-tail")
    tail.add_argument("--output", default="logs/g5/pilot_tail_audit.json")
    full = sub.add_parser("analyze")
    full.add_argument("--input", required=True)
    full.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "pilot-tail":
        pilot_tail(args.output)
    elif args.command == "analyze":
        analyze(args.input, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
