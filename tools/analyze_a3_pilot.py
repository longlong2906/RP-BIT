"""Predeclared, descriptive A3 pilot analysis; no synthetic observations.

Only independently validated in-budget snapshots contribute quality or first
solution time. Runtime strata are never pooled for either co-primary metric.
Legacy campaign analysis stays in analyze_results.py unchanged.
"""
import argparse
from collections import defaultdict
import copy
import csv
import math
import os
from pathlib import Path
import statistics

from analyze_results import (COUNTERS, TIMINGS, close, coverage, distribution,
    finite, paired_hierarchical_bootstrap, read_trial, step_cost,
    verify_raw_artifact_inventory)
from experiment_common import digest, dump, file_hash, read_json
from a3_pilot_protocol import PROTOCOL_ID, PLANNERS, comparison_pairs, plan_trials, validate_config
from cloud_campaign import check_inventory


EXPLORATORY = ("exploratory_pilot; two_maps_per_family; paired_seed_labels_do_not_"
               "imply_identical_random_streams; no_global_optimality_claim")
CO_PRIMARY = ("cost_terrain_m_at_budget", "first_solution_time_s")


def safe_child(root, relative):
    root = Path(root).resolve()
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("noncanonical artifact path")
    path = root / relative
    if path.is_absolute() and Path(relative).is_absolute():
        raise ValueError("artifact path must be relative")
    if root not in path.resolve().parents:
        raise ValueError("artifact path escapes results directory")
    return path


def freeze_references(manifest, output, required):
    """Use inline witnesses bound to map hashes, without original absolute paths."""
    references = {}
    for entry in manifest["trials"]:
        key = entry["map_id"] + "/" + entry["query_id"]
        ref = entry.get("reference")
        if not ref:
            if required:
                raise ValueError("pilot query missing frozen witness: " + key)
            candidate = {"status": "UNAVAILABLE", "cost_terrain_m": None}
        else:
            ref = copy.deepcopy(ref)
            validation = ref.get("validation", {})
            if (not isinstance(ref.get("path_xy"), list) or
                    digest(ref["path_xy"]) != ref.get("path_sha256") or
                    not validation.get("valid") or
                    not finite(ref.get("cost_terrain_m")) or ref["cost_terrain_m"] < 0 or
                    not close(ref["cost_terrain_m"], validation.get("cost_terrain_m")) or
                    ref.get("map_sha256") != entry.get("map_hash") or
                    not ref.get("provenance") or ref.get("reference_type") != "validated_witness"):
                raise ValueError("invalid witness provenance: " + key)
            candidate = dict(ref, status="FROZEN_VALIDATED_TARGET",
                map_hash=entry["map_hash"], interpretation="feasible_target_not_certified_optimum")
        if key in references and candidate != references[key]:
            raise ValueError("witness differs between paired trials: " + key)
        references[key] = candidate
    frozen = {"protocol_id": PROTOCOL_ID, "experiment_id": manifest["experiment_id"],
              "references": references, "policy": "predeclared_manifest_witnesses_only"}
    destination = Path(output) / "reference_targets.json"
    if destination.exists() and read_json(destination) != frozen:
        raise ValueError("frozen witness changed; use a new report directory")
    dump(destination, frozen)
    return references


def runtime_sessions(source, manifest):
    """Load session provenance. Exact identity binding is audited per trial."""
    result = {}
    declared = manifest.get("runtime_sessions", [])
    identity = read_json(Path(source)/"runtime_identity.json") if declared else None
    for session in manifest.get("runtime_sessions", []):
        sid = session.get("session_id")
        if not isinstance(sid, str) or not sid or sid in result or Path(sid).name != sid:
            raise ValueError("invalid/duplicate runtime session")
        if not session.get("stratum_id") or not session.get("platform") or not session.get("cpu"):
            raise ValueError("runtime session lacks platform/CPU/stratum")
        path = safe_child(source, "sessions/" + sid + ".json")
        if not path.is_file():
            raise ValueError("runtime session evidence missing: " + sid)
        evidence = read_json(path)
        environment = evidence.get("environment", {})
        environment_hash = digest(environment)
        identity_hash = digest(identity)
        stratum = digest({"platform": environment.get("os"), "cpu": environment.get("cpu"),
                          "runtime_identity": identity})
        expected = {"session_id": sid, "stratum_id": stratum,
                    "platform": environment.get("os"), "cpu": environment.get("cpu"),
                    "environment_sha256": environment_hash, "runtime_identity_sha256": identity_hash}
        if any(session.get(key) != value or evidence.get(key) != value for key, value in expected.items()):
            raise ValueError("runtime session/identity evidence mismatch: " + sid)
        result[sid] = dict(session, evidence=evidence, evidence_file_sha256=file_hash(path))
    return result


def enrich_row(row, sessions):
    row["cost_terrain_m_at_budget"] = row["cost_terrain_m"]
    row["runtime_errors"] = []
    row["runtime_valid"] = False
    row["runtime_session_id"] = row.get("raw", {}).get("runtime_session_id")
    row["runtime_stratum_id"] = row.get("raw", {}).get("runtime_stratum_id")
    if row["executed"]:
        session = sessions.get(row["runtime_session_id"])
        if session is None or session["stratum_id"] != row["runtime_stratum_id"]:
            row["runtime_errors"].append("missing_or_mismatched_runtime_session")
        else:
            interrupted_without_record = (row["status"] == "INTERRUPTED" and "environment" not in row["raw"])
            if not interrupted_without_record and digest(row["raw"].get("environment", {})) != session["environment_sha256"]:
                row["runtime_errors"].append("trial_environment_differs_from_session")
            else:
                row["runtime_valid"] = True
                row["runtime_platform"] = session["platform"]
                row["runtime_cpu"] = session["cpu"]
    timings = row["metrics"]
    components = [timings.get(name) for name in
                  ("load_wall_s", "preprocessing_wall_s", "setup_wall_s")]
    row["first_solution_including_preparation_s"] = (
        sum(components) + row["first_solution_time_s"]
        if row["success"] and all(finite(value) and value >= 0 for value in components)
        else None)
    row["first_solution_including_preparation_unavailable_reason"] = (
        None if row["first_solution_including_preparation_s"] is not None else
        "no_validated_in_budget_solution" if not row["success"] else
        "one_or_more_preparation_phases_not_measured")
    # BVH construction is already a component of preprocessing, never added again.
    row["preparation_time_formula"] = "load_s + preprocessing_s + setup_s"
    return row


def attainment(row, threshold=None):
    events = row["events"] if row["success"] else []
    times = [event["elapsed_s"] for event in events
             if event["elapsed_s"] <= row["time_budget_s"] and
             (threshold is None or event["best_cost_terrain_m"] <= threshold)]
    event_time = min(times) if times else None
    if not row["executed"]:
        end, reason = None, "trial_not_executed"
    elif event_time is not None:
        end, reason = event_time, "event_observed"
    elif row["status"] in ("CRASH", "WATCHDOG", "INTERRUPTED", "MEMORY_LIMIT"):
        actual = row["metrics"].get("actual_solve_wall_s")
        end = min(row["time_budget_s"], actual) if finite(actual) and actual >= 0 else None
        reason = "measured_solve_end" if end is not None else "solve_duration_unavailable"
    else:
        end, reason = row["time_budget_s"], "planning_budget_no_validated_event"
    return {"event_observed": event_time is not None, "event_time_s": event_time,
            "observation_end_s": end, "observation_end_reason": reason,
            "status": "NOT_RUN" if not row["executed"] else
                      "ATTAINED" if event_time is not None else "RIGHT_CENSORED_NO_EVENT",
            "failure_type": row["status"] if row["executed"] and not row["success"] else None}


def summarize(rows):
    result = coverage(rows)
    executed = [row for row in rows if row["executed"]]
    # Missing runtime evidence is never an implicit common machine.
    usable = [row for row in executed if row["runtime_valid"]]
    successes = [row for row in usable if row["success"]]
    result["runtime_unavailable"] = len(executed) - len(usable)
    result["first_solution_s"] = distribution([r["first_solution_time_s"] for r in successes], len(executed))
    result["cost_terrain_m_at_budget"] = distribution([r["cost_terrain_m"] for r in successes], len(executed))
    result["first_solution_including_preparation_s"] = distribution(
        [r["first_solution_including_preparation_s"] for r in successes], len(executed))
    result["metrics"] = {name: distribution([r["metrics"].get(name) for r in usable], len(executed))
                         for name in (*COUNTERS, *TIMINGS)}
    return result


def grouped(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        if row["runtime_valid"]:
            groups[tuple(row.get(key) for key in keys)].append(row)
    return [dict(zip(keys, key), **summarize(group)) for key, group in sorted(groups.items())]


def curves(rows, references, tolerances):
    groups = defaultdict(list)
    for row in rows:
        if row["executed"] and row["runtime_valid"]:
            groups[(row["runtime_stratum_id"], row["family"], row["query_id"], row["planner_id"])].append(row)
    output = []
    for (stratum, family, query, planner), group in sorted(groups.items()):
        times = sorted({0., *(row["time_budget_s"] for row in group),
            *(e["elapsed_s"] for row in group if row["success"] for e in row["events"]
              if e["elapsed_s"] <= row["time_budget_s"])})
        points = []
        for elapsed in times:
            costs, reached = [], 0
            targets, denominators = defaultdict(int), defaultdict(int)
            for row in group:
                cost = step_cost(row["events"], min(elapsed, row["time_budget_s"])) if row["success"] else None
                if cost is not None:
                    costs.append(cost)
                    reached += 1
                ref = references[row["map_id"] + "/" + row["query_id"]]
                if ref["status"] == "FROZEN_VALIDATED_TARGET":
                    for tolerance in tolerances:
                        key = str(tolerance)
                        denominators[key] += 1
                        targets[key] += int(cost is not None and cost <= (1+tolerance)*ref["cost_terrain_m"])
            points.append({"time_s": elapsed, "success_attainment": reached/len(group),
                "successes": reached, "executed_denominator": len(group),
                "median_cost_terrain_m": statistics.median(costs) if costs else None,
                "cost_observations": len(costs), "cost_conditioning": "validated_incumbents_available_at_t",
                "quality_target_attainment": {key: targets[key]/n for key, n in denominators.items()},
                "quality_target_denominators": dict(denominators)})
        output.append({"runtime_stratum_id": stratum, "family": family, "query_id": query,
                       "planner_id": planner, "executed": len(group), "points": points})
    return output


def slot(row):
    return row["map_id"], row["query_id"], row["planner_seed"]


def tradeoff(rows, planners):
    """Plot common successful slots; allow frontier only with equal coverage.

    A planner that fails hard slots cannot become a frontier winner merely by
    reporting medians on its easier successful subset.
    """
    by_planner = {planner: {slot(row): row for row in rows if row["planner_id"] == planner
                           and row["executed"] and row["runtime_valid"]} for planner in planners}
    executed_sets = [set(by_planner[planner]) for planner in planners]
    success_sets = [{key for key, row in by_planner[planner].items() if row["success"]} for planner in planners]
    common_executed = set.intersection(*executed_sets) if executed_sets else set()
    common_success = set.intersection(*success_sets) if success_sets else set()
    equal_coverage = bool(common_executed) and all(items == executed_sets[0] for items in executed_sets)
    equal_success = bool(common_success) and all(items == success_sets[0] for items in success_sets)
    points = []
    for planner in planners:
        matched = [by_planner[planner][key] for key in sorted(common_success)]
        points.append({"planner_id": planner, "executed": len(by_planner[planner]),
            "validated_success": sum(row["success"] for row in by_planner[planner].values()),
            "common_success_slots": len(matched),
            "cost_median_m": distribution([row["cost_terrain_m"] for row in matched])["median"],
            "first_solution_median_s": distribution([row["first_solution_time_s"] for row in matched])["median"]})
    eligible = equal_coverage and equal_success
    frontier = []
    if eligible:
        for point in points:
            x, y = point["first_solution_median_s"], point["cost_median_m"]
            dominated = any(other["first_solution_median_s"] <= x and other["cost_median_m"] <= y and
                (other["first_solution_median_s"] < x or other["cost_median_m"] < y) for other in points)
            if not dominated:
                frontier.append(point["planner_id"])
    return {"status": "DESCRIPTIVE_EQUAL_COVERAGE" if eligible else
            "UNAVAILABLE_UNEQUAL_SUCCESS_OR_EXECUTION_COVERAGE" if common_success else "UNAVAILABLE_NO_COMMON_SUCCESS",
            "common_executed_slots": len(common_executed), "common_success_slots": len(common_success),
            "conditioning": "same_seed_slots_successful_for_every_configured_planner",
            "frontier_condition": "all_planners_have_identical_executed_and_success_slot_sets",
            "interpretation": "descriptive_median_tradeoff_not_statistical_dominance_or_certified_optimum",
            "points": points, "frontier": frontier}


def paired_results(rows, comparisons, replicates, seed, confidence):
    results = []
    strata = sorted({row["runtime_stratum_id"] for row in rows if row["runtime_valid"]})
    families = sorted({row["family"] for row in rows})
    for stratum in strata:
        same_runtime = [row for row in rows if row["runtime_valid"] and row["runtime_stratum_id"] == stratum]
        for family in ["ALL", *families]:
            group = same_runtime if family == "ALL" else [row for row in same_runtime if row["family"] == family]
            if not group:
                continue
            for query in ["ALL", *sorted({row["query_id"] for row in group})]:
                query_group = group if query == "ALL" else [row for row in group if row["query_id"] == query]
                for candidate, reference in comparisons:
                    for metric in ("success", "cost_terrain_m", "first_solution_time_s"):
                        record = paired_hierarchical_bootstrap(query_group, candidate, reference, metric,
                            replicates, seed, confidence, EXPLORATORY)
                        record.update(runtime_stratum_id=stratum, family=family, query_id=query,
                            candidate=candidate, reference=reference,
                            favorable_direction="positive" if metric == "success" else "negative")
                        results.append(record)
    return results


def pair_coverage(rows, comparisons):
    """Audit losses before stratification, including cross-runtime seed pairs."""
    output = []
    families = sorted({row["family"] for row in rows})
    for family in ["ALL", *families]:
        subset = rows if family == "ALL" else [row for row in rows if row["family"] == family]
        for candidate, reference in comparisons:
            slots = defaultdict(dict)
            for row in subset:
                if row["planner_id"] in (candidate, reference):
                    slots[slot(row)][row["planner_id"]] = row
            record = dict(family=family, candidate=candidate, reference=reference,
                planned_slots=len(slots), incomplete_planned_slots=0, both_executed=0,
                one_executed=0, neither_executed=0, runtime_missing=0, cross_runtime=0,
                same_runtime=0, both_success_same_runtime=0, one_or_both_failure_same_runtime=0)
            for pair in slots.values():
                if set(pair) != {candidate, reference}:
                    record["incomplete_planned_slots"] += 1
                    continue
                a, b = pair[candidate], pair[reference]
                if not (a["executed"] and b["executed"]):
                    record["one_executed" if a["executed"] or b["executed"] else "neither_executed"] += 1
                    continue
                record["both_executed"] += 1
                if not (a["runtime_valid"] and b["runtime_valid"]):
                    record["runtime_missing"] += 1
                elif a["runtime_stratum_id"] != b["runtime_stratum_id"]:
                    record["cross_runtime"] += 1
                else:
                    record["same_runtime"] += 1
                    record["both_success_same_runtime" if a["success"] and b["success"] else
                           "one_or_both_failure_same_runtime"] += 1
            output.append(record)
    return output


def _csv(path, records, fields):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def plot_results(output, curve_data, tradeoffs, planners):
    # This batch reporter writes files. Notebook backends inherited by a worker
    # may be unavailable in its venv and can fail during matplotlib's import.
    os.environ["MPLBACKEND"] = "Agg"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {planner: plt.get_cmap("tab20")(index) for index, planner in enumerate(planners)}
    files = []
    strata = sorted({record["runtime_stratum_id"] for record in curve_data})
    for stratum_index, stratum in enumerate(strata):
        subset = [record for record in curve_data if record["runtime_stratum_id"] == stratum]
        panels = sorted({(record["family"], record["query_id"]) for record in subset})
        if not panels:
            continue
        for metric, ylabel, label in (("success_attainment", "Fraction with first valid solution", "first_solution_attainment"),
                                      ("median_cost_terrain_m", "Median surface cost (m), incumbents only", "anytime_cost")):
            fig, axes = plt.subplots(math.ceil(len(panels)/2), 2, figsize=(15, 3.3*math.ceil(len(panels)/2)),
                                      squeeze=False, layout="constrained")
            for ax, (family, query) in zip(axes.flat, panels):
                for record in subset:
                    if (record["family"], record["query_id"]) == (family, query):
                        ax.step([point["time_s"] for point in record["points"]],
                                [point[metric] for point in record["points"]], where="post",
                                color=colors[record["planner_id"]], label=record["planner_id"])
                ax.set(title=family+" / "+query, xlabel="Solve time (s)", ylabel=ylabel)
                if metric == "success_attainment":
                    ax.set_ylim(0, 1.02)
            for ax in list(axes.flat)[len(panels):]:
                ax.set_visible(False)
            axes.flat[0].legend(fontsize=6)
            fig.suptitle("Exploratory pilot; runtime stratum " + stratum[:12])
            filename = f"{label}_runtime_{stratum_index:02d}.png"
            fig.savefig(output/filename, dpi=150)
            plt.close(fig)
            files.append(filename)
        selected = [record for record in tradeoffs if record["runtime_stratum_id"] == stratum]
        fig, axes = plt.subplots(math.ceil(len(selected)/2), 2, figsize=(15, 4*math.ceil(len(selected)/2)),
                                 squeeze=False, layout="constrained")
        for ax, record in zip(axes.flat, selected):
            for point in record["points"]:
                if point["cost_median_m"] is not None:
                    ax.scatter(point["first_solution_median_s"], point["cost_median_m"],
                               color=colors[point["planner_id"]], label=point["planner_id"])
            ax.set(title=f"{record['family']}: {record['common_success_slots']} shared successful slots",
                   xlabel="Median first valid solution, solve time (s)", ylabel="Median surface cost at budget (m)")
            if not record["common_success_slots"]:
                ax.text(.5, .5, "No common successful slots", ha="center", transform=ax.transAxes)
        for ax in list(axes.flat)[len(selected):]:
            ax.set_visible(False)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        if handles:
            axes.flat[0].legend(fontsize=6)
        fig.suptitle("Common-slot tradeoff, not a weighted score; runtime " + stratum[:12])
        filename = f"cost_time_tradeoff_runtime_{stratum_index:02d}.png"
        fig.savefig(output/filename, dpi=150)
        plt.close(fig)
        files.append(filename)
    return files


def write_report(output, summary, plot_files):
    def fmt(value):
        return "N/A" if value is None else f"{value:.6g}"
    lines = ["# Pilot A3: cost bám bề mặt và thời gian có nghiệm đầu", "",
        f"Trạng thái: **{summary['status']}**; profile `{summary['profile']}`; protocol `{PROTOCOL_ID}`.", "",
        f"Đã chạy {summary['executed']}/{summary['planned']}; success đã kiểm chứng {summary['validated_success']}; "
        f"chưa chạy {summary['not_run']}; validation còn thiếu {summary['validation_pending']}.", "",
        "Hai tiêu chí đồng chính là cost tại ngân sách solve và thời gian có nghiệm hợp lệ đầu từ `solve()`. "
        "Không gộp thành điểm trọng số. Success cần exact goal và kiểm chứng độc lập; nghiệm sau deadline không được dùng.", "",
        ("Đây là smoke kiểm tra pipeline, không dùng để xếp hạng khoa học. " if summary["profile"] == "smoke" else
         "Đây là pilot thăm dò, không phải held-out. Chỉ có hai map mỗi family; ") +
        "CI bootstrap 95% không chứng minh ưu thế tổng quát. "
        "Các bảng đo bản triển khai cải tiến; không quy toàn bộ chênh lệch cho riêng thuật toán.", "",
        "## Kết quả theo runtime và family", "",
        "Mọi trung vị chất lượng/thời gian dưới đây chỉ dùng success. Mẫu số success gồm tất cả lượt đã thực thi, "
        "kể cả timeout, crash và watchdog. Không trộn CPU/runtime khác nhau.", "",
        "| Runtime | Family | Planner | Success / executed | Cost median (m) | Tfirst median (s) | Tfirst IQR (s) |",
        "|---|---|---|---:|---:|---:|---:|"]
    for group in summary["family_groups"]:
        first = group["first_solution_s"]
        lines.append(f"| {group['runtime_stratum_id'][:12]} | {group['family']} | {group['planner_id']} | "
            f"{group['validated_success']}/{group['executed']} | {fmt(group['cost_terrain_m_at_budget']['median'])} | "
            f"{fmt(first['median'])} | {fmt(first['q25'])}–{fmt(first['q75'])} |")
    lines += ["", "## Ghép cặp và đánh đổi", "",
        "`paired_bootstrap.json` và `.csv` chứa đúng 19 cặp khóa trước, theo runtime/family/query và tổng trong cùng runtime. "
        "Chênh lệch candidate − reference: âm thuận lợi cho cost và Tfirst, dương thuận lợi cho success. "
        "Cost/Tfirst chỉ xét cặp cùng thành công; số cặp không đủ dữ liệu được lưu riêng. "
        f"Bootstrap map → query → seed, {summary['bootstrap_replicates']} lần, seed 73130; thiếu hai map độc lập thì không dựng CI. "
        "`pair_coverage.json` phân biệt cặp chưa chạy, khác runtime, thiếu provenance và cặp có failure.", "",
        "Biểu đồ cost–time dùng đúng các slot thành công chung của cả 13 cấu hình. Frontier chỉ công bố khi "
        "mọi cấu hình có cùng tập slot đã chạy và cùng tập slot thành công; khi coverage khác nhau không ép chọn bản thắng. "
        "Frontier trung vị là mô tả, không phải kiểm định ưu thế thống kê.", "",
        "## Đồng hồ, censoring và bằng chứng", "",
        "Tfirst có chuẩn bị = load + preprocessing + setup + Tfirst từ solve. BVH nằm trong preprocessing và không cộng lần hai. "
        "Pha chưa đo giữ null; validation không tính vào solve. RSS là đỉnh được lấy mẫu, không phải đỉnh chính xác của hệ điều hành.", "",
        "Timeout không có event vẫn là chưa có nghiệm, không được gán Tfirst = deadline. Đường cong đạt nghiệm dùng số lượt đã "
        "thực thi làm mẫu số; không phải Kaplan–Meier. Crash/watchdog thiếu thời lượng solve có observation_end = null.", "",
        "Witness trong `reference_targets.json` đã khóa từ manifest trước phân tích, không lấy từ planner thắng và không phải optimum. "
        "Bảng target là khả năng đạt (1 + epsilon) × witness cost.", "",
        "CSV/JSON: `trials.csv`, `trial_metrics.json`, `summary.json`, `paired_bootstrap.*`, `anytime_curves.json`, "
        "`tradeoffs.json`, `pair_coverage.json`, `runtime_sessions.json`, `analysis_input_inventory.json`. Không thay đổi dữ liệu thô."]
    if not plot_files:
        lines += ["", "Biểu đồ: NOT_RUN (`--no-plots` hoặc chưa có dữ liệu runtime hợp lệ)."]
    for filename in plot_files:
        lines += ["", f"![{filename}]({filename})"]
    (output/"REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def analyze(source, output, make_plots=True, *, _test_bootstrap_replicates=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents:
        raise ValueError("report must be outside immutable raw results")
    manifest = read_json(source/"manifest.json")
    config = read_json(source/"resolved_config.json")
    protocol = config.get("a3_pilot_benchmark", {})
    if config.get("schema_version") != 7 or protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("analysis requires schema 7 A3 pilot protocol")
    profile = manifest["profile"]
    if profile not in ("smoke", "a3_pilot"):
        raise ValueError("unsupported A3 pilot profile")
    planners = list(config["profiles"][profile]["planners"])
    if planners != list(PLANNERS):
        raise ValueError("planner order/set differs from the 13 locked configurations")
    comparisons = comparison_pairs()
    configured_pairs = protocol.get("comparison_pairs")
    if configured_pairs is not None and configured_pairs != [list(pair) for pair in comparisons]:
        raise ValueError("comparison pairs differ from the predeclared 19 pairs")
    if (protocol.get("bootstrap_replicates") != 2000 or protocol.get("bootstrap_seed") != 73130 or
            config["analysis"].get("bootstrap_unit_order") != ["map", "query", "seed"] or
            config["analysis"].get("confidence_level") != .95):
        raise ValueError("bootstrap protocol differs from the locked pilot")
    validate_config(config, profile)
    expected = 52 if profile == "smoke" else 780
    entries = manifest["trials"]
    if (manifest.get("planned_trials") != expected or len(entries) != expected or
            len({entry["trial_id"] for entry in entries}) != expected or
            {entry["planner_id"] for entry in entries} != set(planners)):
        raise ValueError("trial identities/cardinality disagree with locked profile")
    for entry in entries:
        safe_child(source, entry["trial_id"])
    maps = read_json(source/"map_manifest.json")
    identity = manifest.get("identity", {})
    if (manifest.get("protocol_id") != PROTOCOL_ID or identity.get("config_hash") != digest(config) or
            identity.get("map_manifest_sha256") != digest(maps) or identity.get("plan_sha256") != digest(entries) or
            entries != plan_trials(config, profile, maps)):
        raise ValueError("immutable config/maps/plan identity mismatch")
    budget = config["profiles"][profile]["planning_budget_s"]
    if budget != (2. if profile == "smoke" else 10.):
        raise ValueError("planning budget differs from the locked pilot")
    raw_hash = verify_raw_artifact_inventory(source, required=False)
    output.mkdir(parents=True, exist_ok=True)
    references = freeze_references(manifest, output, required=profile == "a3_pilot")
    sessions = runtime_sessions(source, manifest)
    if identity.get("runtime_identity_sha256") != digest(read_json(source/"runtime_identity.json")):
        raise ValueError("runtime identity differs from frozen manifest")
    informed = set(planners) - {"lazyprmstar_hauser2015"}
    rows = []
    for entry in entries:
        inventory_path = source / entry["trial_id"] / "artifact_inventory.json"
        if inventory_path.exists():
            frozen = read_json(inventory_path)
            if frozen.get("sha256") != digest(frozen.get("entries")):
                raise ValueError("trial artifact inventory digest mismatch")
            check_inventory(inventory_path.parent, frozen["entries"], {"artifact_inventory.json"})
        events_path = source / entry["trial_id"] / "validated_events.json"
        if events_path.exists():
            for event in read_json(events_path):
                safe_child(events_path.parent / "snapshots", event.get("snapshot_id"))
        row = read_trial(source, entry, budget, config["model"]["numerical_abs_tolerance_m"],
            config["model"]["numerical_rel_tolerance"], informed, allow_pending_validation=True)
        if row["time_budget_s"] != budget:
            raise ValueError("trial budget differs from manifest: " + entry["trial_id"])
        row["artifact_inventory_available"] = inventory_path.is_file()
        rows.append(enrich_row(row, sessions))
    summary = coverage(rows)
    unsealed = [row["trial_id"] for row in rows if row["executed"] and not row["artifact_inventory_available"]]
    state_path = source/"campaign_state.json"
    state = read_json(state_path) if state_path.exists() else {}
    if state and state.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("campaign state protocol differs from manifest")
    campaign_status = state.get("status", manifest.get("status"))
    runtime_errors = [{"trial_id": row["trial_id"], "errors": row["runtime_errors"]}
                      for row in rows if row["runtime_errors"]]
    failure = summary["crash"] or summary["watchdog"] or any(
        row["status"] in ("MEMORY_LIMIT", "INTERRUPTED", "FAILED") for row in rows)
    status = ("FAILED_INTEGRITY" if runtime_errors else "FAILED_CORRECTNESS" if summary["invalid"] else
        "STOPPED_FAILURE" if failure else "PARTIAL" if summary["not_run"] or summary["validation_pending"] or unsealed or
        campaign_status != "COMPLETE_VALIDATED" else "COMPLETE_VALIDATED")
    if campaign_status in ("FAILED_CORRECTNESS", "FAILED_INTEGRITY", "STOPPED_FAILURE"):
        status = campaign_status
    summary.update(protocol_id=PROTOCOL_ID, schema_version=1, profile=profile,
        experiment_id=manifest["experiment_id"], status=status, campaign_status=campaign_status,
        raw_manifest_sha256=file_hash(source/"manifest.json"), raw_artifact_inventory_sha256=raw_hash,
        primary_metrics=list(CO_PRIMARY), planner_ids=planners,
        comparison_pairs=[list(pair) for pair in comparisons],
        interpretation=EXPLORATORY, bootstrap_replicates=2000, bootstrap_seed=73130,
        runtime_errors=runtime_errors, runtime_sessions=len(sessions),
        artifact_inventories_missing=unsealed,
        family_groups=grouped(rows, ("runtime_stratum_id", "family", "planner_id")),
        query_groups=grouped(rows, ("runtime_stratum_id", "family", "map_id", "query_id", "planner_id")),
        runtime_groups=grouped(rows, ("runtime_stratum_id", "planner_id")),
        planner_coverage=[dict(planner_id=planner, **coverage([r for r in rows if r["planner_id"] == planner])) for planner in planners],
        missing_trial_ids=[row["trial_id"] for row in rows if not row["executed"]])
    replicates = _test_bootstrap_replicates or 2000
    if _test_bootstrap_replicates is not None:
        if not manifest.get("synthetic_test_fixture") or "tests/fixtures" not in source.as_posix():
            raise ValueError("bootstrap override is restricted to labeled parser fixtures under tests/fixtures")
        summary["synthetic_test_fixture"] = True
        summary["bootstrap_replicates"] = replicates
    paired = paired_results(rows, comparisons, replicates, 73130, .95)
    paired_coverage = pair_coverage(rows, comparisons)
    curve_data = curves(rows, references, config["analysis"]["target_relative_tolerances"])
    tradeoffs = []
    for stratum in sorted({row["runtime_stratum_id"] for row in rows if row["runtime_valid"]}):
        subset = [row for row in rows if row["runtime_valid"] and row["runtime_stratum_id"] == stratum]
        for family in ["ALL", *sorted({row["family"] for row in subset})]:
            group = subset if family == "ALL" else [row for row in subset if row["family"] == family]
            tradeoffs.append(dict(runtime_stratum_id=stratum, family=family, **tradeoff(group, planners)))
    records = []
    for row in rows:
        record = {key: value for key, value in row.items() if key not in ("raw", "events", "reference")}
        record["success_attainment"] = attainment(row)
        ref = references[row["map_id"] + "/" + row["query_id"]]
        record["quality_targets"] = [dict(relative_tolerance=tolerance,
            cost_threshold_m=(1+tolerance)*ref["cost_terrain_m"], reference_type=ref["reference_type"],
            **attainment(row, (1+tolerance)*ref["cost_terrain_m"]))
            for tolerance in config["analysis"]["target_relative_tolerances"]
            if ref["status"] == "FROZEN_VALIDATED_TARGET"]
        records.append(record)
    for name, value in (("summary.json", summary), ("trial_metrics.json", records),
                        ("paired_bootstrap.json", paired), ("anytime_curves.json", curve_data),
                        ("tradeoffs.json", tradeoffs), ("pair_coverage.json", paired_coverage),
                        ("runtime_sessions.json", sessions)):
        dump(output/name, value)
    _csv(output/"trials.csv", [dict(row, **row["metrics"]) for row in records],
        ("trial_id", "family", "map_id", "query_id", "planner_id", "planner_seed", "status", "executed", "success",
         "runtime_session_id", "runtime_stratum_id", "runtime_valid", "cost_terrain_m_at_budget",
         "first_solution_time_s", "first_solution_including_preparation_s", "events_after_budget", *COUNTERS, *TIMINGS))
    _csv(output/"paired_bootstrap.csv", [dict(row, ci_low=row["ci"][0] if row["ci"] else None,
        ci_high=row["ci"][1] if row["ci"] else None, **row["units"]) for row in paired],
        ("runtime_stratum_id", "family", "query_id", "candidate", "reference", "metric", "estimate", "ci_low", "ci_high", "status",
         "conditioning", "maps", "map_query_units", "paired_seed_slots", "both_planners_executed_slots",
         "one_planner_executed_slots", "incomplete_planned_slots", "unavailable_complete_pairs"))
    plots = plot_results(output, curve_data, tradeoffs, planners) if make_plots else []
    write_report(output, summary, plots)
    inventory = [{"path": path.relative_to(source).as_posix(), "sha256": file_hash(path), "size_bytes": path.stat().st_size}
                 for path in sorted(source.rglob("*")) if path.is_file() and not path.name.endswith(".tmp")]
    dump(output/"analysis_input_inventory.json", {"entries": inventory, "inventory_sha256": digest(inventory)})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    result = analyze(args.input, args.output, make_plots=not args.no_plots)
    print({key: result[key] for key in ("status", "planned", "executed", "validated_success")})
    return 0 if result["status"] in ("COMPLETE_VALIDATED", "PARTIAL") else 1


if __name__ == "__main__":
    raise SystemExit(main())
