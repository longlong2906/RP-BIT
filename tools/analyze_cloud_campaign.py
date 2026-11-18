"""Analyze independently validated raw Cloud trials for a3_vs_all_v2.

Immutable manifests provide denominators; hash-bound snapshots provide quality.
Families are fixed strata and bootstrap draws retain all five planner vectors.
This module never starts a planner or changes raw experiment artifacts.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import math
import os
from pathlib import Path
import random
import statistics
import tempfile
from analyze_results import COUNTERS, TIMINGS, distribution, finite, read_trial, step_cost
from experiment_common import digest, dump, file_hash
from cloud_campaign import verify_raw, read_json

PROTOCOL_ID = "a3_vs_all_v2"
PLANNERS = ("bitstar_a3", "abitstar", "aitstar", "lazyprmstar_hauser2015", "sem_rrtstar_common_v1")
PRIMARY = PLANNERS[0]
LABELS = dict(zip(PLANNERS, ("A3", "ABIT*", "AIT*", "LazyPRM* Hauser", "SEM-RRT*-adapted common-objective")))
FAMILIES = ("traversable_hill", "ridge_pass", "rolling_terrain", "narrow_passage", "mixed_obstacles")
MAP_SEEDS = tuple(range(6001, 6021))
PLANNER_SEEDS = (94001, 94002, 94003)
BUDGET_S = 30.0
EXPECTED_TRIALS = 1500
CHECKPOINTS_S = (.1, .25, .5, 1., 2., 5., 10., 20., 30.)
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 73139
PAIRED_METRICS = ("validated_success", "cost_terrain_m_at_30s", "first_solution_time_s", "t_first_rmst_tau_30s")
TECHNICAL_STATUSES = {"CRASH", "WATCHDOG", "INTERRUPTED", "WORKER_FAILURE", "FAILED"}


def validate_design(config, manifest):
    profile = config.get("profiles", {}).get("combined", {})
    protocol = config.get("combined_benchmark", {})
    checks = {
        "config schema": config.get("schema_version") == 5,
        "protocol": protocol.get("protocol_id") == PROTOCOL_ID,
        "profile": manifest.get("profile") == "combined",
        "manifest protocol": manifest.get("protocol_id") == PROTOCOL_ID,
        "bootstrap hierarchy": config.get("analysis", {}).get("bootstrap_unit_order") == ["family", "map", "seed"],
        "confidence level": config.get("analysis", {}).get("confidence_level") == .95,
        "budget": profile.get("planning_budget_s") == BUDGET_S,
        "families": profile.get("families") == list(FAMILIES),
        "map seeds": profile.get("map_seeds") == list(MAP_SEEDS),
        "maps per family": profile.get("maps_per_family") == 20,
        "query count": profile.get("queries_per_map") == 1,
        "planner seeds": profile.get("planner_seeds") == list(PLANNER_SEEDS),
        "planners": profile.get("planners") == list(PLANNERS),
        "protocol planners": protocol.get("planner_ids") == list(PLANNERS),
        "primary": protocol.get("primary_planner_id") == PRIMARY,
        "comparators": protocol.get("comparator_planner_ids") == list(PLANNERS[1:]),
        "expected trials": protocol.get("expected_trials") == EXPECTED_TRIALS,
        "manifest cardinality": manifest.get("planned_trials") == EXPECTED_TRIALS,
        "bootstrap replicates": protocol.get("bootstrap_replicates") == BOOTSTRAP_REPLICATES,
        "bootstrap seed": protocol.get("bootstrap_seed") == BOOTSTRAP_SEED,
        "SEM scope": protocol.get("official_sem_rrtstar_implementation") is False,
        "SEM label": protocol.get("sem_comparator_label") == LABELS[PLANNERS[-1]],
        "SEM risk": protocol.get("terrain_risk_cost") is False,
        "SEM turn": protocol.get("turning_angle_cost") is False,
        "SEM rolling": protocol.get("rolling_optimization") is False,
    }
    failed = [key for key, valid in checks.items() if not valid]
    if failed:
        raise ValueError("v2 design mismatch: " + ", ".join(failed))
    entries = manifest.get("trials", [])
    if len(entries) != EXPECTED_TRIALS:
        raise ValueError("v2 manifest requires exactly 1500 planned entries")
    ids, slots, maps = set(), defaultdict(list), defaultdict(dict)
    for entry in entries:
        required = ("trial_id", "family", "map_id", "map_hash", "map_seed", "query_id", "query_hash", "planner_id", "planner_seed")
        if any(key not in entry for key in required):
            raise ValueError("planned entry lacks trial identity")
        trial_id = entry["trial_id"]
        if not isinstance(trial_id, str) or not trial_id or Path(trial_id).name != trial_id or "\\" in trial_id or trial_id in (".", "..") or trial_id in ids:
            raise ValueError("duplicate or unsafe trial_id")
        ids.add(trial_id)
        family, map_id = entry["family"], entry["map_id"]
        if family not in FAMILIES or entry["map_seed"] not in MAP_SEEDS or entry["planner_seed"] not in PLANNER_SEEDS or entry["query_id"] != "q0":
            raise ValueError("unregistered family/map/query/planner seed")
        identity = (entry["map_seed"], entry["map_hash"], entry.get("query_hash"))
        if map_id in maps[family] and maps[family][map_id] != identity:
            raise ValueError("paired map/query hash mismatch")
        maps[family][map_id] = identity
        slots[(family, map_id, entry["query_id"], entry["planner_seed"])].append(entry["planner_id"])
    if len({map_id for group in maps.values() for map_id in group}) != 100:
        raise ValueError("v2 requires exactly 100 distinct map IDs")
    for family in FAMILIES:
        if len(maps[family]) != 20 or {item[0] for item in maps[family].values()} != set(MAP_SEEDS):
            raise ValueError("v2 requires every map seed in each family")
    if len(slots) != 300 or any(Counter(planners) != Counter(PLANNERS) for planners in slots.values()):
        raise ValueError("v2 requires 300 complete five-planner seed slots")


def time_record(row):
    missing = {"time_s": None, "event_observed": False, "reason": None, "endpoint_source": None, "status": "TECHNICAL_MISSING"}
    if not row["executed"]:
        return dict(missing, reason="trial_not_executed", status="NOT_RUN")
    if row["validation_pending"]:
        return dict(missing, reason="offline_validation_pending")
    if row["success"]:
        return {"time_s": row["first_solution_time_s"], "event_observed": True, "reason": None,
                "endpoint_source": "first_hash_bound_independently_validated_snapshot", "status": "EVENT"}
    raw = row.get("raw", {})
    if row["status"] in TECHNICAL_STATUSES or row["errors"]:
        return dict(missing, reason="technical_failure_or_invalid_output")
    if raw.get("planner_status") in ("INVALID_START", "INVALID_GOAL"):
        return dict(missing, reason="invalid_problem")
    elapsed = row["metrics"].get("actual_solve_wall_s")
    if not finite(elapsed) or elapsed < 0:
        return dict(missing, reason="measured_solve_endpoint_unavailable")
    return {"time_s": min(BUDGET_S, elapsed), "event_observed": False, "reason": None,
            "endpoint_source": "actual_solve_wall_s", "status": "RIGHT_CENSORED"}


def km_rmst(records, tau_s=BUDGET_S):
    """Weighted Kaplan-Meier RMST without extending unsupported survival."""
    usable = [(record["time_s"], record["event_observed"], record.get("weight", 1.))
              for record in records if finite(record.get("time_s")) and 0 <= record["time_s"] <= tau_s
              and finite(record.get("weight", 1.)) and record.get("weight", 1.) > 0]
    result = {"rmst_s": None, "n": len(usable), "events": sum(bool(item[1]) for item in usable),
              "right_censored": sum(not item[1] for item in usable), "tau_s": tau_s, "reason": None}
    if not usable:
        return dict(result, reason="no_event_or_measured_censor_records")
    groups = defaultdict(lambda: [0., 0.])
    for elapsed, observed, weight in usable:
        groups[elapsed][0] += weight if observed else 0.
        groups[elapsed][1] += weight
    at_risk = sum(item[2] for item in usable)
    survival, previous, area = 1., 0., 0.
    for elapsed, (events, removals) in sorted(groups.items()):
        area += survival * (elapsed - previous)
        if events and at_risk > 0:
            survival *= max(0., 1. - events / at_risk)
        at_risk -= removals
        previous = elapsed
    if previous < tau_s and survival > 1e-12:
        return dict(result, reason="positive_survival_beyond_last_observed_endpoint")
    result["rmst_s"] = area + survival * (tau_s - previous)
    return result


def read_rows(source, entries):
    rows = []
    for entry in entries:
        row = read_trial(source, entry, BUDGET_S, 1e-8, 1e-8,
                         informed_planners=set(PLANNERS) - {"lazyprmstar_hauser2015"}, allow_pending_validation=True)
        row.setdefault("validation_pending", False)
        raw = row.get("raw", {})
        if raw and raw.get("time_budget_s") != BUDGET_S:
            raise ValueError("trial solve budget differs from 30 s: " + row["trial_id"])
        if raw and entry.get("query_hash") is not None and raw.get("query_hash") != entry["query_hash"]:
            raise ValueError("trial query identity mismatch: " + row["trial_id"])
        if raw and not raw.get("path_available") and row["status"] != "COMPLETED":
            row["validation_pending"] = False
        corrupt = ("snapshot_hash_mismatch", "snapshot_cost_mismatch", "invalid_snapshot_id",
                   "invalid_event_time", "invalid_event_cost", "increasing_incumbent")
        if any(isinstance(error, dict) and error.get("error") in corrupt for error in row["errors"]):
            raise ValueError("corrupt validated event provenance: " + row["trial_id"])
        row["outcome_known"] = row["executed"] and not row["validation_pending"]
        directory = Path(source) / entry["trial_id"]
        validation_errors = []
        for validation_path in [directory / "validation.json", *sorted((directory / "snapshots").glob("*/validation.json"))]:
            if validation_path.is_file():
                validation_errors.extend(read_json(validation_path).get("errors", []))
        geometry_errors = {"not_exact_start_goal", "forbidden_capsule", "NoData gap", "NoData point",
                           "cost_mismatch", "missing_or_nonfinite_surface", "missing_triangle_breakpoint",
                           "surface_length_mismatch", "off_surface_point", "exported_nodata"}
        row["validation_errors"] = validation_errors
        row["independent_validation_failed"] = bool(row["outcome_known"] and raw.get("path_available") and
                                                      (row["errors"] or raw.get("independent_valid") is False))
        row["geometric_invalid"] = bool(row["independent_validation_failed"] and geometry_errors.intersection(validation_errors))
        row["validation_failure_unclassified"] = row["independent_validation_failed"] and not row["geometric_invalid"]
        row["technical_failure"] = row["status"] in TECHNICAL_STATUSES or row["validation_failure_unclassified"]
        row["t_first"] = time_record(row)
        row["cost_terrain_m_at_30s"] = row["cost_terrain_m"]
        row["length_xy_m_final"] = row["length_xy_m"] if row["success"] else None
        rows.append(row)
    return rows


def group_counts(rows):
    executed = [row for row in rows if row["executed"]]
    known = [row for row in executed if row["outcome_known"]]
    success = sum(row["success"] for row in executed)
    return {"planned": len(rows), "executed": len(executed), "outcome_known": len(known),
            "validated_success": success, "known_non_success": len(known) - success,
            "validation_pending": len(executed) - len(known), "unexecuted": len(rows) - len(executed),
            "geometric_invalid": sum(row["geometric_invalid"] for row in rows),
            "technical_failures": sum(row["technical_failure"] for row in rows),
            "validation_failures_unclassified": sum(row["validation_failure_unclassified"] for row in rows),
            "execution_status_counts": dict(Counter(row["status"] for row in rows)),
            "completion_coverage": len(executed) / len(rows) if rows else None,
            "success_rate_executed": success / len(executed) if executed else None,
            "success_rate_known_outcomes": success / len(known) if known else None,
            "success_rate_executed_is_final": len(executed) == len(known)}


def descriptive(rows):
    groups = []
    for scope in ("ALL", *FAMILIES):
        for planner in PLANNERS:
            group = [row for row in rows if row["planner_id"] == planner and (scope == "ALL" or row["family"] == scope)]
            executed = [row for row in group if row["executed"]]
            metrics = {name: distribution([row.get(name) for row in executed], len(executed))
                       for name in ("cost_terrain_m_at_30s", "first_solution_time_s", "length_xy_m_final")}
            metrics.update({name: distribution([row["metrics"].get(name) for row in executed], len(executed))
                            for name in (*COUNTERS, *TIMINGS)})
            groups.append({"family": scope, "planner_id": planner, **group_counts(group), "metrics": metrics,
                           "first_solution_rmst": km_rmst([row["t_first"] for row in executed]),
                           "t_first_missing_reasons": dict(Counter(row["t_first"]["reason"] for row in executed if row["t_first"]["reason"]))})
    return groups


def checkpoints(rows):
    records = []
    for scope in ("ALL", *FAMILIES):
        for planner in PLANNERS:
            group = [row for row in rows if row["planner_id"] == planner and (scope == "ALL" or row["family"] == scope)]
            executed = [row for row in group if row["executed"]]
            for checkpoint in CHECKPOINTS_S:
                available = [cost for row in executed if row["success"] for cost in [step_cost(row["events"], checkpoint)] if finite(cost)]
                records.append({"family": scope, "planner_id": planner, "elapsed_s": checkpoint,
                                "planned": len(group), "executed": len(executed), "validated_solution_count": len(available),
                                "solution_coverage_executed": len(available) / len(executed) if executed else None,
                                "validation_pending": sum(row["validation_pending"] for row in executed),
                                "conditional_cost_terrain_m": distribution(available, len(executed))})
    return records


def _hierarchy(rows, scope):
    slots = defaultdict(dict)
    for row in rows:
        if scope == "ALL" or row["family"] == scope:
            slots[(row["family"], row["map_id"], row["query_id"], row["planner_seed"])][row["planner_id"]] = row
    hierarchy = defaultdict(lambda: defaultdict(list))
    for (family, map_id, query, seed), slot in sorted(slots.items()):
        if set(slot) != set(PLANNERS):
            raise ValueError("incomplete planned five-planner slot")
        vector = {}
        for comparator in PLANNERS[1:]:
            a, b = slot[PRIMARY], slot[comparator]
            pair = {"validated_success": float(a["success"]) - float(b["success"])
                    if a["outcome_known"] and b["outcome_known"] else None}
            for name in ("cost_terrain_m_at_30s", "first_solution_time_s"):
                pair[name] = a[name] - b[name] if a["success"] and b["success"] else None
            pair["t_first_rmst_tau_30s"] = (a["t_first"], b["t_first"]) if finite(a["t_first"]["time_s"]) and finite(b["t_first"]["time_s"]) else None
            vector[comparator] = pair
        hierarchy[family][map_id].append(vector)
    return {family: list(maps.values()) for family, maps in sorted(hierarchy.items())}


def _estimate(hierarchy, comparator, name):
    family_values, records_a, records_b = [], [], []
    units = {"planned_pairs": sum(len(seeds) for maps in hierarchy.values() for seeds in maps),
             "eligible_pairs": 0, "eligible_maps": 0, "families": len(hierarchy)}
    unsupported_stratum = False
    for family, maps in hierarchy.items():
        eligible_maps = [[vector[comparator][name] for vector in seeds if vector[comparator][name] is not None] for seeds in maps]
        eligible_maps = [values for values in eligible_maps if values]
        units["eligible_maps"] += len(eligible_maps)
        units["eligible_pairs"] += sum(len(values) for values in eligible_maps)
        if not eligible_maps:
            unsupported_stratum = True
            continue
        if name == "t_first_rmst_tau_30s":
            for values in eligible_maps:
                weight = 1. / len(hierarchy) / len(eligible_maps) / len(values)
                for first_a, first_b in values:
                    records_a.append(dict(first_a, weight=weight))
                    records_b.append(dict(first_b, weight=weight))
        else:
            family_values.append(statistics.mean(statistics.mean(values) for values in eligible_maps))
    if unsupported_stratum:
        return None, units, "no_eligible_pair_in_fixed_family_stratum"
    if name == "t_first_rmst_tau_30s":
        a, b = km_rmst(records_a), km_rmst(records_b)
        if a["rmst_s"] is None or b["rmst_s"] is None:
            return None, units, a["reason"] or b["reason"]
        return a["rmst_s"] - b["rmst_s"], units, None
    return statistics.mean(family_values), units, None


def _interval(values):
    if not values:
        return None
    values = sorted(values)
    def percentile(probability):
        position = (len(values) - 1) * probability
        lower, upper = math.floor(position), math.ceil(position)
        return values[lower] + (values[upper] - values[lower]) * (position - lower)
    return [percentile(.025), percentile(.975)]


def paired_bootstrap(rows, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED):
    """One draw preserves all five planner/metric vectors in each paired seed."""
    if not isinstance(replicates, int) or isinstance(replicates, bool) or replicates < 1:
        raise ValueError("positive integer bootstrap replicate count required")
    results = []
    rng = random.Random(seed)
    for scope in ("ALL", *FAMILIES):
        hierarchy = _hierarchy(rows, scope)
        points = {(comparator, name): _estimate(hierarchy, comparator, name)
                  for comparator in PLANNERS[1:] for name in PAIRED_METRICS}
        samples = {key: [] for key in points}
        eligible_keys = [key for key, point in points.items() if point[0] is not None]
        if eligible_keys:
            for _ in range(replicates):
                draw = {}
                for family, maps in hierarchy.items():
                    draw[family] = [[rng.choice(selected) for _ in range(len(selected))]
                                    for selected in (rng.choice(maps) for _ in range(len(maps)))]
                for comparator, name in eligible_keys:
                    estimate, _, _ = _estimate(draw, comparator, name)
                    if finite(estimate):
                        samples[(comparator, name)].append(estimate)
        for (comparator, name), (estimate, units, reason) in points.items():
            values = samples[(comparator, name)]
            ci = _interval(values) if len(values) >= .9 * replicates else None
            results.append({"family": scope, "planner_a": PRIMARY, "planner_b": comparator,
                            "metric": name, "estimate_a_minus_b": estimate, "confidence_interval_95": ci,
                            **units, "bootstrap_replicates_requested": replicates,
                            "bootstrap_replicates_usable": len(values), "bootstrap_seed": seed,
                            "reason": reason or (None if ci else "insufficient_usable_bootstrap_replicates"),
                            "weighting": "fixed_family_equal_weight_then_equal_map_then_equal_eligible_paired_seed",
                            "bootstrap_hierarchy": "fixed_family_strata -> map -> whole_five_planner_paired_seed_vector",
                            "conditioning": "both_validated_success" if name in ("cost_terrain_m_at_30s", "first_solution_time_s") else
                            "both_known_executed_outcomes" if name == "validated_success" else
                            "both_known_event_or_measured_censor_endpoint"})
    return results


def _trial_record(row):
    names = ("trial_id", "family", "map_id", "map_seed", "map_hash", "query_id", "query_hash",
             "planner_id", "planner_seed", "status", "executed", "success", "outcome_known",
             "validation_pending", "geometric_invalid", "technical_failure", "independent_validation_failed",
             "validation_failure_unclassified", "validation_errors", "termination_reason",
             "first_solution_time_s", "cost_terrain_m_at_30s", "length_xy_m_final", "final_cost_terrain_m",
             "events_after_budget", "metrics", "metric_unavailable_reasons", "errors", "t_first")
    return {name: row.get(name) for name in names}


def _plots(output, rows, checkpoint_rows, paired):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = []
    overall = [record for record in checkpoint_rows if record["family"] == "ALL"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for planner in PLANNERS:
        series = [record for record in overall if record["planner_id"] == planner]
        x = [record["elapsed_s"] for record in series]
        y = [record["conditional_cost_terrain_m"]["median"] for record in series]
        axes[0].step(x, [value if value is not None else math.nan for value in y], where="post", label=LABELS[planner])
        axes[1].step(x, [record["solution_coverage_executed"] if record["solution_coverage_executed"] is not None else math.nan
                        for record in series], where="post", label=LABELS[planner])
    axes[0].set_ylabel("Median terrain cost c_H (m), conditional on valid solution")
    axes[1].set_ylabel("Validated solution / executed trials")
    axes[1].set_xlabel("Elapsed solve time (s)")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xscale("log")
    axes[0].legend(fontsize=8)
    fig.suptitle("100 maps x 3 paired seeds; observations at or before each checkpoint")
    fig.tight_layout()
    fig.savefig(output / "anytime_cost_and_coverage.png", dpi=160)
    plt.close(fig)
    figures.append("anytime_cost_and_coverage.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for index, planner in enumerate(PLANNERS):
        costs = [row["cost_terrain_m_at_30s"] for row in rows if row["planner_id"] == planner and row["success"]]
        times = [row["first_solution_time_s"] for row in rows if row["planner_id"] == planner and row["success"]]
        for ax, values in zip(axes, (costs, times)):
            if values:
                ax.boxplot([values], positions=[index + 1], widths=.6)
    for ax in axes:
        ax.set_xticks(range(1, 6), [LABELS[planner] for planner in PLANNERS], rotation=25, ha="right", fontsize=8)
    axes[0].set_ylabel("Terrain cost c_H at 30 s (m)")
    axes[1].set_ylabel("First independently validated solution time (s)")
    fig.suptitle("Conditional on validated success; eligible counts in descriptive_metrics.json")
    fig.tight_layout()
    fig.savefig(output / "final_quality_and_first_solution.png", dpi=160)
    plt.close(fig)
    figures.append("final_quality_and_first_solution.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric_name, units in zip(axes, ("cost_terrain_m_at_30s", "t_first_rmst_tau_30s"), ("m", "s")):
        selected = [record for record in paired if record["family"] == "ALL" and record["metric"] == metric_name]
        for index, record in enumerate(selected):
            estimate, interval = record["estimate_a_minus_b"], record["confidence_interval_95"]
            if finite(estimate):
                ax.plot(estimate, index, "o")
                if interval:
                    ax.hlines(index, *interval)
        ax.axvline(0, color="gray", linewidth=1)
        ax.set_yticks(range(4), [LABELS[planner] for planner in PLANNERS[1:]], fontsize=8)
        ax.set_xlabel(f"A3 - comparator ({units}); 95% stratified paired bootstrap CI")
        ax.set_title("Terrain cost @30 s" if units == "m" else "First solution RMST, tau=30 s")
    fig.tight_layout()
    fig.savefig(output / "paired_contrasts.png", dpi=160)
    plt.close(fig)
    figures.append("paired_contrasts.png")
    return figures


def _write_report(output, summary, groups, paired):
    def fmt(value):
        return "unavailable" if value is None else f"{value:.6g}"
    lines = ["# A3 so với bốn planner — Codex Cloud, 30 s/trial", "",
             f"Trạng thái: **{summary['status']}**. Thiết kế khóa: 5 planner × 100 map × 3 seed = 1500 trial; một query `q0`/map.", "",
             "Objective `cost_terrain_m` là chiều dài đường bám bề mặt; `length_xy_m` là chiều dài chiếu bằng phụ trợ.",
             "Thành công cần singleton goal chính xác và independent validation. `EXACT_SOLUTION` không chứng minh tối ưu toàn cục; timeout không chứng minh vô nghiệm.",
             "Comparator SEM: **SEM-RRT*-adapted common-objective**, `official_sem_rrtstar_implementation=false`; risk/turn/rolling đều tắt. Kết quả không áp dụng cho SEM-RRT* chính thức.", "",
             "| Planner | Planned / executed / validated success | Pending / unexecuted | Cost @30 s median [Q25,Q75] m | First solution median [Q25,Q75] s |",
             "|---|---|---|---|---|"]
    for group in groups:
        if group["family"] != "ALL":
            continue
        cost, first = group["metrics"]["cost_terrain_m_at_30s"], group["metrics"]["first_solution_time_s"]
        lines.append(f"| {LABELS[group['planner_id']]} | {group['planned']} / {group['executed']} / {group['validated_success']} | "
                     f"{group['validation_pending']} / {group['unexecuted']} | {fmt(cost['median'])} [{fmt(cost['q25'])}, {fmt(cost['q75'])}] (n={cost['n']}) | "
                     f"{fmt(first['median'])} [{fmt(first['q25'])}, {fmt(first['q75'])}] (n={first['n']}) |")
    lines += ["", "| Planner | Solve median s | Worker median s | Validation median s | Peak RSS median MB |",
              "|---|---|---|---|---|"]
    for group in groups:
        if group["family"] == "ALL":
            timing = group["metrics"]
            lines.append(f"| {LABELS[group['planner_id']]} | {fmt(timing['actual_solve_wall_s']['median'])} | "
                         f"{fmt(timing['worker_wall_s']['median'])} | {fmt(timing['validation_wall_s']['median'])} | "
                         f"{fmt(timing['peak_rss_mb']['median'])} |")
    lines += ["", "Phase timing/counter có median/IQR và coverage trong `descriptive_metrics.json`, theo overall và từng family. Build/source identity và environment từng shard được giữ trong raw inventory; các paired planner trong một map/query/seed block phải chạy trên cùng VM.", ""]
    lines += ["", "Ghép cặp A3 − comparator cố định năm family, resample map rồi cả vector năm planner cùng paired seed: 2000 replicate, seed 73139, CI 95%. Cost và first-solution có điều kiện cả hai thành công, kèm số cặp đủ điều kiện. RMST tau=30 s dùng censor endpoint từ solve wall đo thật; crash/watchdog/validation pending/đồng hồ thiếu là technical missing, không thay bằng 30 s. Không ngoại suy RMST khi support chưa đến tau và survival còn dương.", "",
              "| Comparator | Metric | Eligible pairs / planned | A3 − comparator | 95% CI |", "|---|---|---|---|---|"]
    for record in paired:
        if record["family"] != "ALL":
            continue
        interval = record["confidence_interval_95"]
        ci = f"[{fmt(interval[0])}, {fmt(interval[1])}]" if interval else "unavailable"
        lines.append(f"| {LABELS[record['planner_b']]} | {record['metric']} | {record['eligible_pairs']} / {record['planned_pairs']} | {fmt(record['estimate_a_minus_b'])} | {ci} |")
    lines += ["", "Anytime là step function của event kiểm chứng xảy ra ≤ checkpoint; không nội suy hoặc dùng nghiệm tương lai. Cost chart có coverage kèm theo. Mọi failure, pending và chưa chạy được giữ trong `trial_metrics.json`/`trials.csv`. Success rate trên executed là cận dưới tạm thời khi còn pending; partial không phải benchmark hoàn chỉnh.",
              "Solve budget tách setup/preprocessing/serialization/offline validation. RSS/counter/phase timing không đo được giữ null và lý do; counter khác nghĩa không phải đơn vị công việc hoán đổi trực tiếp.", "",
              f"Raw inventory SHA-256: `{summary['raw_artifact_inventory_sha256']}`.",
              f"Manifest SHA-256: `{summary['manifest_sha256']}`. Config SHA-256: `{summary['resolved_config_sha256']}`.",
              "Bootstrap là so sánh thăm dò; CI chưa điều chỉnh multiplicity của nhiều comparator/metric/family. Không khẳng định ưu thế tổng quát hoặc optimum đã chứng minh.", ""]
    if summary["synthetic_test_fixture"]:
        lines += ["SYNTHETIC TEST FIXTURE — không phải dữ liệu hoặc báo cáo thực nghiệm.", ""]
    for figure in summary["figures"]:
        lines += [f"![{figure}]({figure})", ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def analyze(source, output, *, make_plots=True, bootstrap_replicates=BOOTSTRAP_REPLICATES):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("report and raw trees must be disjoint")
    manifest = read_json(source / "manifest.json")
    config = read_json(source / "resolved_config.json")
    validate_design(config, manifest)
    fixture = manifest.get("synthetic_test_fixture") is True
    if (not make_plots or bootstrap_replicates != BOOTSTRAP_REPLICATES) and not fixture:
        raise ValueError("production v2 analysis requires real figures and locked 2000 replicates")
    inventory_hash = verify_raw(source)["inventory_sha256"]
    rows = read_rows(source, manifest["trials"])
    counts = group_counts(rows)
    status = "COMPLETE_VALIDATED" if counts["executed"] == EXPECTED_TRIALS and not counts["validation_pending"] else "PARTIAL"
    if counts["geometric_invalid"] or counts["validation_failures_unclassified"]:
        status = "CORRECTNESS_FAILURE"
    groups, checkpoint_rows = descriptive(rows), checkpoints(rows)
    paired = paired_bootstrap(rows, replicates=bootstrap_replicates)
    summary = {"schema_version": 1, "protocol_id": PROTOCOL_ID, "experiment_id": manifest["experiment_id"],
               "status": status, **counts, "planner_ids": list(PLANNERS), "map_count": 100,
               "planner_seed_labels_per_map": 3, "planning_budget_s": BUDGET_S,
               "checkpoints_s": list(CHECKPOINTS_S), "bootstrap_replicates": bootstrap_replicates,
               "bootstrap_seed": BOOTSTRAP_SEED, "confidence_level": .95,
               "bootstrap_hierarchy": "fixed_family_strata -> map -> whole_five_planner_paired_seed_vector",
               "raw_artifact_inventory_sha256": inventory_hash, "manifest_sha256": file_hash(source / "manifest.json"),
               "resolved_config_sha256": file_hash(source / "resolved_config.json"),
               "identity": manifest.get("identity", {}), "figures": [], "synthetic_test_fixture": fixture,
               "warnings": ["exploratory_CI_without_multiplicity_adjustment"]}
    if output.exists() and any(output.iterdir()):
        raise ValueError("report output is nonempty; preserve it and choose a new output")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".cloud-analysis-" + output.name + "-", dir=output.parent))
    records = [_trial_record(row) for row in rows]
    dump(stage / "trial_metrics.json", records)
    dump(stage / "descriptive_metrics.json", groups)
    dump(stage / "checkpoints.json", checkpoint_rows)
    dump(stage / "paired_bootstrap.json", paired)
    with (stage / "trials.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["trial_id", "family", "map_id", "query_id", "planner_id", "planner_seed", "status", "executed",
                  "success", "validation_pending", "cost_terrain_m_at_30s", "first_solution_time_s", "length_xy_m_final", *COUNTERS, *TIMINGS]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({name: record["metrics"].get(name) if name in (*COUNTERS, *TIMINGS) else record.get(name) for name in fields})
    try:
        summary["figures"] = _plots(stage, rows, checkpoint_rows, paired) if make_plots else []
    except Exception as error:
        dump(stage / "summary.json", dict(summary, status="BLOCKED_PLOTTING", plot_error=f"{type(error).__name__}: {error}"))
        raise RuntimeError(f"matplotlib figure generation failed; metrics retained in {stage}") from error
    dump(stage / "summary.json", summary)
    _write_report(stage, summary, groups, paired)
    artifacts = [{"path": path.name, "size_bytes": path.stat().st_size, "sha256": file_hash(path)} for path in sorted(stage.iterdir()) if path.is_file()]
    dump(stage / "report_artifact_inventory.json", {"schema_version": 1, "entries": artifacts, "inventory_sha256": digest(artifacts)})
    if output.exists():
        output.rmdir()
    os.replace(stage, output)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="immutable merged raw campaign root")
    parser.add_argument("--output", type=Path, required=True, help="new report directory")
    args = parser.parse_args()
    summary = analyze(args.input, args.output)
    print({key: summary[key] for key in ("status", "planned", "executed", "validated_success", "validation_pending", "unexecuted")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

