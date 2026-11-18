"""Analyze registered A3-T corridor v3 trials without running a planner.

Smoke is descriptive. Pilot selection and held-out gates are deliberately
separate: a pilot winner is not evidence of held-out superiority. All started
failures remain in PSR; quality uses paired independently safe final paths.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

PROTOCOL_ID = "a3t_corridor_v3"
BOOTSTRAP_SEED = 83158
BOOTSTRAP_REPLICATES = 2000
ALPHAS = (1.05,1.10,1.20)
QUALITY = ("final_cost_terrain_m", "final_pri", "final_swept_slope_deg")
TERMINAL = {"COMPLETED", "CRASH", "WATCHDOG", "TIMEOUT", "BLOCKED", "FAILED",
            "INTERRUPTED", "FAILED_CORRECTNESS"}


def read(path):
    def reject(value):
        raise ValueError("nonstandard JSON number: " + value)
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def stats(values):
    values = [float(value) for value in values if finite(value)]
    return dict(n=len(values), mean=statistics.fmean(values) if values else None,
                median=statistics.median(values) if values else None,
                minimum=min(values) if values else None, maximum=max(values) if values else None)


def _rows(summary, manifest):
    entries, observed = manifest["entries"], summary["rows"]
    errors = []
    ids = [entry["trial_id"] for entry in entries]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate planned trial identity")
    lookup = {row["trial_id"]: row for row in observed}
    if len(lookup) != len(observed) or set(lookup) - set(ids):
        raise ValueError("duplicate or unplanned observed trial")
    if summary.get("planned") != len(entries) or summary.get("attempted") != len(observed):
        errors.append("summary cardinality differs from manifest/observations")
    if manifest.get("planned_trials", len(entries)) != len(entries):
        errors.append("manifest planned count differs from entries")
    families = {item["map_id"]: item["family"] for item in manifest.get("maps", [])}
    rows, slots = [], set()
    for entry in entries:
        identity = (entry["method"], entry["map_id"], entry["planner_seed"])
        if identity in slots:
            raise ValueError("duplicate method/map/seed slot")
        slots.add(identity)
        family = families.get(entry["map_id"], entry.get("family"))
        source = lookup.get(entry["trial_id"])
        base = dict(trial_id=entry["trial_id"], method=entry["method"], map_id=entry["map_id"],
                    planner_seed=entry["planner_seed"], family=family,
                    alpha=entry.get("alpha"))
        if source is None:
            rows.append(dict(**base, attempted=False, executed=False, execution_launched=False,
                             completed=False, safe=False, attained=False,
                             execution_status="NOT_RUN", correctness_errors=[],
                             **{key: None for key in QUALITY}, first_safe_s=None,
                             first_safe_latency_s=None))
            continue
        for key in ("method", "map_id", "planner_seed"):
            if source.get(key) != entry[key]:
                raise ValueError("trial identity mismatch: " + entry["trial_id"] + ":" + key)
        if family is None or source.get("family") != family:
            raise ValueError("unknown or inconsistent map family")
        if source.get("alpha") != entry.get("alpha"):
            raise ValueError("trial risk alpha differs from registration")
        status = source.get("execution_status")
        if status not in TERMINAL:
            errors.append(entry["trial_id"] + ": observation has no terminal execution status")
        if entry.get("status") != status:
            errors.append(entry["trial_id"] + ": manifest execution status mismatch")
        # A controller can attempt a trial but fail before creating its child.
        # Such a dependency/launch failure is not an executed planner trial.
        launched = source.get("execution_launched")
        inferred_execution = status in {"COMPLETED", "CRASH", "WATCHDOG", "TIMEOUT"}
        if launched is not None and type(launched) is not bool:
            raise ValueError("execution_launched must be a boolean")
        executed = inferred_execution if launched is None else launched
        if launched is False and inferred_execution:
            errors.append(entry["trial_id"] + ": process outcome contradicts missing launch")
        trial_errors = source.get("correctness_errors", [])
        if not isinstance(trial_errors, list):
            raise ValueError("correctness_errors must be a list")
        if trial_errors:
            errors.append(entry["trial_id"] + ": correctness errors present")
        completed = status == "COMPLETED"
        good = completed and executed and not trial_errors
        safe = bool(source.get("success") and good and source.get("planner_status") == "EXACT_SOLUTION")
        if source.get("success") and not safe:
            errors.append(entry["trial_id"] + ": inconsistent exact-safe success")
        attained = bool(source.get("attained_safe") and good)
        if source.get("attained_safe") and not attained:
            errors.append(entry["trial_id"] + ": inconsistent safe attainment")
        if safe and not attained:
            errors.append(entry["trial_id"] + ": final safe without observed safe attainment")
        if safe and any(not finite(source.get(key)) or source[key] < 0 for key in QUALITY):
            raise ValueError("safe success has invalid quality metric")
        first = source.get("first_safe_s") if attained else None
        latency = source.get("first_safe_latency_s") if attained else None
        if attained and (not finite(first) or first < 0):
            raise ValueError("attained trial has invalid first-safe time")
        if latency is not None and (not finite(latency) or latency < first):
            raise ValueError("invalid first-safe latency")
        rows.append(dict(**base, attempted=True, executed=executed, execution_launched=executed,
            completed=completed, safe=safe, attained=attained,
            execution_status=status, correctness_errors=trial_errors, first_safe_s=first,
            first_safe_latency_s=latency, planner_status=source.get("planner_status"),
            **{key: source.get(key) if safe else None for key in QUALITY},
            **{key: source.get(key) for key in ("field_build_s", "calibration_s", "preprocessing_s",
                "solve_time_s", "calibration_status", "effective_risk_cap", "b_graph",
                "corridor_s", "corridor_status", "graph_s", "guidance_s", "mask_s",
                "final_risk_sum", "final_risk_count", "final_risk_quantiles", "final_exposure",
                "guide_length", "guide_pri", "guide_exposure", "L0")}))
    if summary.get("success") != sum(row["safe"] for row in rows):
        errors.append("summary safe-success count mismatch")
    if "executed" in summary and summary["executed"] != sum(row["executed"] for row in rows):
        errors.append("summary executed count mismatch")
    if summary.get("correctness_failures") != sum(bool(row.get("correctness_errors")) for row in observed):
        errors.append("summary correctness count mismatch")
    methods = {row["method"] for row in rows}
    expected_slots = None
    for method in sorted(methods):
        group_slots = {(row["map_id"], row["planner_seed"]) for row in rows if row["method"] == method}
        if expected_slots is not None and group_slots != expected_slots:
            errors.append("method schedules are not fully paired")
        expected_slots = group_slots
    if "sem_native" not in methods:
        errors.append("SEM native comparator is missing")
    complete = (len(observed) == len(entries) and all(row["execution_status"] in TERMINAL for row in rows)
                and manifest.get("status") == "COMPLETE" and summary.get("status") == "COMPLETE")
    return rows, errors, complete


def aggregate(rows, budget):
    attempted = [row for row in rows if row["attempted"]]
    executed = [row for row in rows if row["executed"]]
    successful = [row for row in executed if row["safe"]]
    restricted = []
    for row in executed:
        if not row["completed"] or row["correctness_errors"]:
            restricted.append(None)
        else:
            restricted.append(row["first_safe_s"] if row["attained"] else budget)
    unavailable = sum(value is None for value in restricted)
    return dict(planned=len(rows), attempted=len(attempted), not_run=len(rows)-len(attempted),
        executed=len(executed), failed_before_launch=len(attempted)-len(executed),
        completed=sum(row["completed"] for row in executed), success=len(successful),
        psr=len(successful)/len(executed) if executed else None,
        completion_coverage=len(attempted)/len(rows) if rows else None,
        execution_coverage=len(executed)/len(rows) if rows else None,
        execution_status_counts={value: sum(row["execution_status"] == value for row in rows)
                                 for value in sorted({row["execution_status"] for row in rows})},
        **{key: stats(row[key] for row in successful) for key in QUALITY},
        **{key: stats(row.get(key) for row in attempted) for key in
           ("first_safe_s", "first_safe_latency_s", "field_build_s", "calibration_s", "preprocessing_s", "solve_time_s", "corridor_s", "graph_s",
            "guidance_s", "mask_s", "final_risk_sum", "final_risk_count", "final_exposure",
            "guide_length", "guide_pri", "guide_exposure", "L0")},
        restricted_first_safe=dict(horizon_s=budget, n=len(executed),
            censored_at_budget=sum(row["completed"] and not row["attained"] and not row["correctness_errors"] for row in executed),
            unavailable=unavailable, mean=statistics.fmean(restricted) if restricted and not unavailable else None,
            semantics="integral_of_nonattainment; deadline_censoring_is_not_a_solution_event"))


def _quantile(values, probability):
    ordered = sorted(values)
    position = (len(ordered)-1)*probability
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high]-ordered[low])*(position-low)


def _bootstrap(pairs, replicates, seed):
    """Resample maps within each fixed family, then paired seeds within map.

    Failed trials remain sampled for PSR. Quality estimates condition on both
    safe paths, including after resampling; an empty replicate is unavailable.
    """
    groups = {}
    for pair in pairs:
        groups.setdefault(pair["family"], {}).setdefault(pair["map_id"], []).append(pair)
    randomizer = random.Random(seed)
    draws = {key: [] for key in ("psr", *QUALITY)}
    for _ in range(replicates):
        sample = []
        for family in sorted(groups):
            maps = sorted(groups[family])
            for _ in maps:
                candidates = groups[family][randomizer.choice(maps)]
                sample.extend(randomizer.choice(candidates) for _ in candidates)
        for metric in draws:
            values = [item[metric] for item in sample if item[metric] is not None]
            draws[metric].append(statistics.fmean(values) if values else None)
    result = {}
    for metric, values in draws.items():
        available = [value for value in values if value is not None]
        complete = bool(available) and len(available) == replicates
        result[metric] = dict(lower=_quantile(available, .025) if complete else None,
            upper=_quantile(available, .975) if complete else None,
            replicates=replicates, available_replicates=len(available), confidence=.95,
            unavailable_reason=None if complete else "empty_paired_quality_in_resample")
    return result


def paired(rows, method, *, replicates=0, seed=BOOTSTRAP_SEED, baseline="sem_native"):
    left = {(row["map_id"], row["planner_seed"]): row for row in rows if row["method"] == method}
    right = {(row["map_id"], row["planner_seed"]): row for row in rows if row["method"] == baseline}
    slots = sorted(set(left) & set(right))
    pairs = []
    for slot in slots:
        a, b = left[slot], right[slot]
        if not (a["executed"] and b["executed"]):
            continue
        safe = a["safe"] and b["safe"]
        pairs.append(dict(family=a["family"], map_id=a["map_id"], planner_seed=a["planner_seed"],
            psr=float(a["safe"])-float(b["safe"]),
            **{key: a[key]-b[key] if safe else None for key in QUALITY}))
    paired_safe = [(left[slot], right[slot]) for slot in slots if left[slot]["safe"] and right[slot]["safe"]]
    metrics = {}
    confidence = _bootstrap(pairs, replicates, seed) if replicates and pairs else {}
    for key in ("psr", *QUALITY):
        values = [item[key] for item in pairs if item[key] is not None]
        result = dict(difference=stats(values), coverage=len(values)/len(slots) if slots else None,
                      maps=len({item["map_id"] for item in pairs if item[key] is not None}),
                      ci95=confidence.get(key))
        if key in QUALITY:
            result.update(left_mean=statistics.fmean(a[key] for a, _ in paired_safe) if paired_safe else None,
                          right_mean=statistics.fmean(b[key] for _, b in paired_safe) if paired_safe else None)
        metrics[key] = result
    return dict(method=method, baseline=baseline, sign="method_minus_baseline", planned_pairs=len(slots),
                attempted_pairs=sum(left[slot]["attempted"] and right[slot]["attempted"] for slot in slots),
                executed_pairs=len(pairs), safe_pairs=len(paired_safe), metrics=metrics)


def select_pilot(report):
    result = dict(status="BLOCKED", selected_method=None, selected_alpha=None, candidates=[],
                  rule="PSR>=SEM and paired mean delta L_H<0 and delta PRI<0; min paired candidate L_H, PRI, alpha")
    if report["profile"] != "pilot":
        result["reason"] = "selection_requires_pilot"
        return result
    if not report["complete"] or not report["executed_complete"] or report["errors"]:
        result["reason"] = "incomplete_execution_or_invalid_pilot"
        return result
    eligible = []
    methods = sorted({row["method"] for row in report["trials"] if row["method"].startswith("a3t_corridor_a")})
    alphas_seen = []
    for method in methods:
        alphas = {row["alpha"] for row in report["trials"] if row["method"] == method}
        if len(alphas) != 1 or next(iter(alphas)) not in ALPHAS:
            raise ValueError("pilot method has inconsistent/unregistered risk alpha")
        alpha = next(iter(alphas))
        alphas_seen.append(alpha)
        comparison = next(item for item in report["paired_comparisons"]
                          if item["method"] == method and item["scope"] == "all" and item["baseline"] == "sem_native")
        metrics = comparison["metrics"]
        checks = dict(psr=finite(metrics["psr"]["difference"]["mean"]) and metrics["psr"]["difference"]["mean"] >= 0,
                      length=finite(metrics[QUALITY[0]]["difference"]["mean"]) and metrics[QUALITY[0]]["difference"]["mean"] < 0,
                      pri=finite(metrics[QUALITY[1]]["difference"]["mean"]) and metrics[QUALITY[1]]["difference"]["mean"] < 0)
        candidate = dict(method=method, alpha=alpha, passes=all(checks.values()), checks=checks,
                         paired_length_mean=metrics[QUALITY[0]]["left_mean"],
                         paired_pri_mean=metrics[QUALITY[1]]["left_mean"],
                         safe_pairs=comparison["safe_pairs"], planned_pairs=comparison["planned_pairs"])
        result["candidates"].append(candidate)
        if candidate["passes"]:
            eligible.append(candidate)
    if sorted(alphas_seen) != list(ALPHAS):
        result["reason"] = "pilot_does_not_include_exactly_all_three_alphas"
        return result
    if not eligible:
        result.update(status="NO_CANDIDATE", reason="no_candidate_passes_all_registered_gates")
        return result
    winner = min(eligible, key=lambda row: (row["paired_length_mean"], row["paired_pri_mean"], row["alpha"]))
    result.update(status="SELECTED", selected_method=winner["method"], selected_alpha=winner["alpha"],
                  reason="pilot_selection_only_not_held_out_superiority")
    return result


def held_out_gate(report):
    result = dict(status="NOT_APPLICABLE", passes=False, checks={})
    if report["profile"] != "full":
        return result
    if not report["complete"] or not report["executed_complete"] or report["errors"]:
        result.update(status="BLOCKED", reason="incomplete_execution_or_invalid_held_out")
        return result
    comparison = next((item for item in report["paired_comparisons"]
        if item["method"] == "a3t_corridor_v3" and item["scope"] == "all" and item["baseline"] == "sem_native"), None)
    if comparison is None:
        result.update(status="BLOCKED", reason="missing_registered_primary_contrast")
        return result
    for key in ("psr", QUALITY[0], QUALITY[1]):
        ci = comparison["metrics"][key]["ci95"]
        endpoint = ci.get("lower" if key == "psr" else "upper") if ci else None
        result["checks"][key] = finite(endpoint) and (endpoint >= 0 if key == "psr" else endpoint < 0)
    result.update(status="PASS" if all(result["checks"].values()) else "NOT_MET",
                  passes=all(result["checks"].values()),
                  reason="joint_registered_gates; quality_is_conditional_on_paired_safe_finals")
    return result


def validate_registration(manifest, config):
    from corridor_v3_protocol import default_config, inventory
    if digest(config) != digest(default_config()) or manifest.get("config_sha256") != digest(config):
        raise ValueError("unregistered config or config provenance mismatch")
    alpha = None
    if manifest["profile"] == "full":
        alphas = {entry.get("alpha") for entry in manifest["entries"] if entry["method"] == "a3t_corridor_v3"}
        if len(alphas) != 1 or next(iter(alphas)) not in ALPHAS:
            raise ValueError("full requires one locked pilot alpha")
        alpha = next(iter(alphas))
        if not manifest.get("selection_sha256"):
            raise ValueError("full manifest is not bound to pilot selection")
    planned = inventory(config, manifest["profile"], alpha)["entries"]
    keys = ("trial_id", "method", "map_id", "family", "map_seed", "planner_seed", "alpha", "risk_slack")
    actual = [{key: entry.get(key) for key in keys} for entry in manifest["entries"]]
    expected = [{key: entry.get(key) for key in keys} for entry in planned]
    if actual != expected:
        raise ValueError("manifest schedule differs from the complete registered profile")


def build_report(summary, manifest, config, *, bootstrap_replicates=None):
    if manifest.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("not an A3-T corridor v3 campaign")
    profile = manifest.get("profile")
    if profile not in ("smoke", "pilot", "full") or summary.get("profile", profile) != profile:
        raise ValueError("unknown or inconsistent profile")
    if not summary.get("synthetic_fixture"):
        validate_registration(manifest, config)
    budget = config["profiles"][profile]["planning_budget_s"]
    if not finite(budget) or budget <= 0:
        raise ValueError("invalid planning budget")
    rows, errors, complete = _rows(summary, manifest)
    for row in rows:
        if row["attained"] and row["first_safe_s"] > budget:
            errors.append(row["trial_id"] + ": safe attainment after deadline")
    replicates = BOOTSTRAP_REPLICATES if profile == "full" else 0
    if bootstrap_replicates is not None:
        # Reduced resampling is only a test API, never a production CLI switch.
        if not summary.get("synthetic_fixture"):
            raise ValueError("bootstrap override is reserved for labeled test fixtures")
        if type(bootstrap_replicates) is not int or bootstrap_replicates <= 0:
            raise ValueError("invalid test replicate count")
        replicates = bootstrap_replicates if profile == "full" else 0
    tables, contrasts = [], []
    methods = sorted({row["method"] for row in rows})
    for scope in ("all", "family", "map"):
        groups = ["all"] if scope == "all" else sorted({row["family" if scope == "family" else "map_id"] for row in rows})
        for group in groups:
            subset = rows if scope == "all" else [row for row in rows if row["family" if scope == "family" else "map_id"] == group]
            for method in methods:
                method_rows = [row for row in subset if row["method"] == method]
                if method_rows:
                    tables.append(dict(scope=scope, group=group, method=method, **aggregate(method_rows, budget)))
            if scope != "map":
                for method in methods:
                    if method != "sem_native":
                        contrasts.append(dict(scope=scope, group=group, **paired(subset, method,
                            replicates=replicates if method in ("a3t_corridor_v3", "a3t_cap40") else 0)))
                for candidate in methods:
                    if candidate.startswith("a3t_corridor"):
                        control=candidate.replace("a3t_corridor","sem_common_corridor",1)
                        if control in methods:
                            contrasts.append(dict(scope=scope,group=group,
                                **paired(subset,candidate,baseline=control)))
    executed_complete = bool(rows) and all(row["executed"] for row in rows)
    report = dict(schema_version=3, protocol_id=PROTOCOL_ID, profile=profile,
        status="FAILED_CORRECTNESS" if errors else "INCOMPLETE" if not complete else
               "DONE" if executed_complete else "INCOMPLETE_EXECUTION",
        complete=complete, executed_complete=executed_complete, errors=errors, budget_s=budget, planned=len(rows),
        attempted=sum(row["attempted"] for row in rows), executed=sum(row["executed"] for row in rows),
        success=sum(row["safe"] for row in rows),
        tables=tables, paired_comparisons=contrasts, trials=rows,
        bootstrap_replicates=replicates, bootstrap_seed=BOOTSTRAP_SEED, confidence=.95,
        bootstrap_unit="fixed_family_then_map_then_paired_seed",
        quality_selection="last_published_in_budget_native_incumbent_before_validation; paired_both_safe",
        failure_policy="retain_all_executed_failures_in_PSR; report_before_launch_failures_and_unattempted_slots_separately",
        inference="smoke_correctness_only" if profile == "smoke" else "pilot_development_only" if profile == "pilot" else "registered_held_out_gate")
    report["pilot_selection"] = select_pilot(report)
    report["held_out_gate"] = held_out_gate(report)
    return report


def _write_csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True, allow_nan=False)
                             if isinstance(value, (dict, list)) else value for key, value in row.items()})


def markdown(report):
    def fmt(value):
        return "—" if value is None else format(value, ".6g")
    lines = ["# A3-T corridor v3", "", f"Profile: {report['profile']}; trạng thái: **{report['status']}**.",
        f"Đã thử khởi chạy {report['attempted']}/{report['planned']} lượt, thực sự chạy {report['executed']}; {report['success']} đường cuối exact và hợp lệ độc lập.", "",
        "| Nhóm | Phương pháp | Success/executed/attempt/planned | PSR | L_H mean (m) | PRI mean | Corridor mean (s) | First-safe mean (s) | Latency mean (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["tables"]:
        if row["scope"] == "map":
            continue
        lines.append(f"| {row['group']} | {row['method']} | {row['success']}/{row['executed']}/{row['attempted']}/{row['planned']} | " +
                     " | ".join(fmt(row[key]["mean"] if key != "psr" else row[key]) for key in
                                ("psr", QUALITY[0], QUALITY[1], "corridor_s", "first_safe_s", "first_safe_latency_s")) + " |")
    lines += ["", "| Nhóm | So sánh | Cặp safe/planned | ΔL_H mean | ΔPRI mean | ΔPSR |",
              "|---|---|---:|---:|---:|---:|"]
    for row in report["paired_comparisons"]:
        lines.append(f"| {row['group']} | {row['method']} − {row['baseline']} | {row['safe_pairs']}/{row['planned_pairs']} | " +
                     " | ".join(fmt(row["metrics"][key]["difference"]["mean"]) for key in (QUALITY[0], QUALITY[1], "psr")) + " |")
    if report["profile"] == "smoke":
        lines += ["", "Smoke chỉ kiểm pipeline/correctness; không chọn α và không chứng minh ưu thế trước SEM. Pilot/full NOT_RUN."]
    elif report["profile"] == "pilot":
        selection = report["pilot_selection"]
        lines += ["", f"Lựa chọn pilot: **{selection['status']}**; α={fmt(selection['selected_alpha'])}.",
                  "Không ứng viên đạt thì dừng trước full; không thay gate/seed. Lựa chọn pilot chưa phải kết quả held-out."]
    else:
        lines += ["", f"Gate held-out: **{report['held_out_gate']['status']}**.",
                  "CI95% dùng 2.000 bootstrap family cố định → map → paired seed; upper ΔL_H và ΔPRI <0, lower ΔPSR ≥0."]
        for row in report["paired_comparisons"]:
            if row["scope"] == "all" and row["method"] == "a3t_corridor_v3":
                for key in (QUALITY[0], QUALITY[1], "psr"):
                    ci = row["metrics"][key]["ci95"] or {}
                    lines.append(f"- {key}: CI95% [{fmt(ci.get('lower'))}; {fmt(ci.get('upper'))}].")
    lines += ["", "PRI là trung bình risk trên tập ô đóng duy nhất đường tâm chạm, không phải xác suất tai nạn. "
              "Giảm exposure của guide không bảo đảm giảm PRI của incumbent cuối. Chiều dài là L_H do oracle tính lại; quality chỉ trên các cặp cùng final-safe.",
              "Mọi thất bại đã thực thi giữ trong mẫu số PSR; lỗi trước khi tạo process và lượt chưa thử được báo riêng. "
              "Pilot/full bị chặn nếu chưa thực thi đủ schedule, dù collection đã kết thúc. Timeout không phải chứng minh vô nghiệm. "
              "Restricted time giữ censoring; crash/interruption làm quan sát thời gian unavailable.",
              "Preprocessing/corridor được báo riêng; latency gồm khởi tạo và solve, không gồm oracle ngoại tuyến."]
    if report["errors"]:
        lines += ["", "Lỗi:", ""] + ["- " + value for value in report["errors"]]
    return "\n".join(lines) + "\n"


def analyze(input_directory, output_directory):
    source, output = Path(input_directory).resolve(), Path(output_directory).resolve()
    if output == source or source in output.parents:
        raise ValueError("analysis output must be separate from immutable campaign")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("refusing to overwrite analysis artifacts")
    manifest, summary, config = (read(source/name) for name in ("manifest.json", "summary.json", "config.json"))
    if summary.get("synthetic_fixture") or any(row.get("synthetic_fixture") for row in summary.get("rows", [])):
        raise ValueError("synthetic parser fixtures cannot be analyzed as benchmark artifacts")
    if manifest.get("config_sha256") != digest(config):
        raise ValueError("config provenance mismatch")
    # Validate the registered experiment, not just its self-reported count.
    from corridor_v3_protocol import load_config
    load_config(source/"config.json")
    if manifest.get("profile") == "full":
        selection_path = source/"pilot_selection.json"
        if not selection_path.is_file() or file_hash(selection_path) != manifest.get("selection_sha256"):
            raise ValueError("full pilot selection artifact is missing or changed")
        selection = read(selection_path)
        expected = dict(status="SELECTED", protocol_id=PROTOCOL_ID,
            selected_alpha=manifest.get("selected_alpha"),
            config_sha256=manifest["config_sha256"],
            sources_sha256=digest(manifest.get("sources", {})), binary_sha256=manifest.get("binary_sha256"))
        if any(selection.get(key) != value for key, value in expected.items()):
            raise ValueError("full pilot selection differs from locked experiment identity")
    identities = {entry["trial_id"]: entry for entry in manifest["entries"]}
    for row in summary["rows"]:
        entry = identities.get(row["trial_id"])
        if entry is None:
            raise ValueError("unplanned trial in summary")
        directory = source/"trials"/row["trial_id"]
        validation = directory/"validation.json"
        if file_hash(validation) != entry.get("validation_sha256") or digest(read(validation)) != digest(row):
            raise ValueError("independent verdict provenance mismatch: " + row["trial_id"])
        inventory = read(directory/"raw_inventory.json")
        for relative, expected in inventory.items():
            path = (directory/relative).resolve()
            if directory.resolve() not in path.parents or file_hash(path) != expected:
                raise ValueError("raw artifact hash mismatch or unsafe path: " + row["trial_id"])
    report = build_report(summary, manifest, config)
    hashes = {name: file_hash(source/name) for name in ("manifest.json", "summary.json", "config.json")}
    provenance = dict(created_utc=datetime.now(timezone.utc).isoformat(), input_directory=str(source),
                      input_sha256=hashes, analyzer_sha256=file_hash(__file__),
                      sources_sha256=digest(manifest.get("sources", {})), binary_sha256=manifest.get("binary_sha256"))
    report["provenance"] = provenance
    output.mkdir(parents=True, exist_ok=True)
    for name, value in (("summary.json", report), ("provenance.json", provenance)):
        (output/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    _write_csv(output/"aggregate.csv", report["tables"])
    _write_csv(output/"paired_comparisons.csv", report["paired_comparisons"])
    _write_csv(output/"trials.csv", report["trials"])
    (output/"REPORT.md").write_text(markdown(report), encoding="utf-8")
    if report["profile"] == "pilot":
        selection = dict(report["pilot_selection"], protocol_id=PROTOCOL_ID, schema_version=3,
            config_sha256=manifest["config_sha256"], sources_sha256=provenance["sources_sha256"],
            binary_sha256=provenance["binary_sha256"], analyzer_sha256=provenance["analyzer_sha256"],
            pilot_manifest_path=str(source/"manifest.json"), pilot_manifest_sha256=hashes["manifest.json"],
            pilot_summary_path=str(source/"summary.json"), pilot_summary_sha256=hashes["summary.json"])
        (output/"selection.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = analyze(args.input, args.output)
    print(json.dumps({key: report[key] for key in ("status", "profile", "planned", "attempted", "executed", "success")}, allow_nan=False))
    return 0 if report["status"] == "DONE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
