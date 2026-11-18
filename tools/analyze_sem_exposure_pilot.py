"""Pure analysis of audited pilot rows; collection/shard trust belongs to the runner.

Tuning screens all positive weights and ranks them on a single common support.
Confirmation tests exactly the frozen candidate. It cannot select a runner-up.
"""
from __future__ import annotations

import json
from pathlib import Path
import statistics

from analyze_corridor_v3 import QUALITY, TERMINAL, _rows, _write_csv, aggregate, finite, paired, stats
from corridor_v4_execution import atomic_json as dump, digest, file_hash, utc_now
from sem_exposure_pilot_protocol import (CONFIG, CONTROL, FAMILIES, LAMBDAS, NATIVE,
    POSITIVE_LAMBDAS, PROTOCOL_ID, ROOT, inventory, method_for, method_specs, validate_config)

EXTRA = ("final_exposure", "final_sem_exposure_normalized_m", "final_objective_value",
         "final_risk_sum", "final_risk_count", "queue_wait_s", "validation_s", "peak_rss_bytes")


def _slot(row):
    return row["map_id"], row.get("query_id", "q0"), row["planner_seed"]


def _mean_gate(comparison):
    metrics = comparison["metrics"]
    values = {key: metrics[key]["difference"]["mean"] for key in ("psr", *QUALITY[:2])}
    return dict(psr=finite(values["psr"]) and values["psr"] >= 0.,
                length=finite(values[QUALITY[0]]) and values[QUALITY[0]] < 0.,
                pri=finite(values[QUALITY[1]]) and values[QUALITY[1]] < 0.)


def select_tuning(report):
    if report["phase"] != "tune":
        raise ValueError("weight selection is restricted to tuning")
    result = dict(protocol_id=PROTOCOL_ID, schema_version=1, config_sha256=report["config_sha256"],
        sources_sha256=report.get("sources_sha256"), status="INCOMPLETE", selected_lambda=None,
        selected_method=None, recommended_lambda=None, candidates=[], eligible_methods=[],
        common_cohort=[], common_cohort_sha256=None, ranking=[],
        ranking_rule="same_common_cohort_mean_L_H_then_mean_PRI_then_smaller_lambda",
        scope="pilot_candidate_only_not_global_optimum_or_confirmed_superiority")
    if report["errors"]:
        result.update(status="FAILED_CORRECTNESS", reason="audited_rows_have_errors")
        return result
    if not report["complete"] or not report["executed_complete"]:
        result["reason"] = "all_registered_tuning_starts_need_terminal_verdicts"
        return result
    for weight in POSITIVE_LAMBDAS:
        method = method_for(weight)
        comparison = next(row for row in report["paired_comparisons"]
            if row["scope"] == "all" and row["method"] == method and row["baseline"] == NATIVE)
        checks = _mean_gate(comparison)
        candidate = dict(method=method, sem_lambda=weight, checks=checks, passes=all(checks.values()),
            safe_pairs=comparison["safe_pairs"], planned_pairs=comparison["planned_pairs"],
            paired_mean_delta_length=comparison["metrics"][QUALITY[0]]["difference"]["mean"],
            paired_mean_delta_pri=comparison["metrics"][QUALITY[1]]["difference"]["mean"],
            delta_psr=comparison["metrics"]["psr"]["difference"]["mean"])
        result["candidates"].append(candidate)
        if candidate["passes"]:
            result["eligible_methods"].append(method)
    if not result["eligible_methods"]:
        result.update(status="NO_CANDIDATE", reason="no_positive_lambda_passes_all_registered_mean_gates")
        return result
    methods = [NATIVE, *result["eligible_methods"]]
    lookup = {method: {_slot(row): row for row in report["trials"] if row["method"] == method and row["safe"]}
              for method in methods}
    common = set.intersection(*(set(lookup[method]) for method in methods))
    cohort = [dict(map_id=slot[0], query_id=slot[1], planner_seed=slot[2],
                   family=lookup[NATIVE][slot]["family"]) for slot in sorted(common)]
    counts = {family: sum(row["family"] == family for row in cohort) for family in FAMILIES}
    result.update(common_cohort=cohort, common_cohort_sha256=digest(cohort), common_cohort_count=len(cohort),
                  common_cohort_family_counts=counts, eligible_methods_sha256=digest(result["eligible_methods"]))
    if not common or any(count == 0 for count in counts.values()):
        result.update(status="UNRANKABLE_COMMON_SUPPORT", reason="common_safe_cohort_must_represent_every_registered_family")
        return result
    for candidate in result["candidates"]:
        if not candidate["passes"]:
            continue
        rows = [lookup[candidate["method"]][slot] for slot in sorted(common)]
        result["ranking"].append(dict(method=candidate["method"], sem_lambda=candidate["sem_lambda"],
            common_cohort_n=len(rows), mean_surface_length_m=statistics.fmean(row[QUALITY[0]] for row in rows),
            mean_pri=statistics.fmean(row[QUALITY[1]] for row in rows)))
    result["ranking"].sort(key=lambda row: (row["mean_surface_length_m"], row["mean_pri"], row["sem_lambda"]))
    winner = result["ranking"][0]
    result.update(status="CANDIDATE_LOCKED", selected_lambda=winner["sem_lambda"], selected_method=winner["method"],
        reason="shortest_eligible_on_identical_common_safe_support; confirmation_not_yet_claimed")
    return result


def _validate_tuning_selection(selection, config_sha256):
    if (not isinstance(selection, dict) or selection.get("status") != "CANDIDATE_LOCKED" or
            selection.get("protocol_id") != PROTOCOL_ID or selection.get("config_sha256") != config_sha256 or
            type(selection.get("selected_lambda")) not in (int, float) or
            selection["selected_lambda"] not in POSITIVE_LAMBDAS or
            selection.get("selected_method") != method_for(selection["selected_lambda"])):
        raise ValueError("confirmation requires this protocol's hash-bound locked tuning candidate")
    return selection


def evaluate_confirmation(report, tuning_selection):
    if report["phase"] != "confirm":
        raise ValueError("confirmation gate only applies to confirm phase")
    selection = _validate_tuning_selection(tuning_selection, report["config_sha256"])
    if selection["selected_lambda"] != report["selected_lambda"]:
        raise ValueError("confirmation candidate differs from locked tuning lambda")
    result = dict(protocol_id=PROTOCOL_ID, schema_version=1, config_sha256=report["config_sha256"],
        status="INCOMPLETE", candidate_lambda=selection["selected_lambda"],
        candidate_method=selection["selected_method"], recommended_lambda=None, passes=False,
        tuning_selection_sha256=digest(selection), primary_baseline=NATIVE, mean_checks={}, ci_checks={},
        bootstrap=dict(replicates=2000, seed=83158, confidence=.95, order=["family_fixed", "map", "paired_seed"]),
        no_runner_up=True, scope="independent_pilot_confirmation_conditional_on_paired_safe_paths; full_NOT_RUN")
    if report["errors"]:
        result.update(status="FAILED_CORRECTNESS", reason="audited_rows_have_errors")
        return result
    if not report["complete"] or not report["executed_complete"]:
        result["reason"] = "all_600_confirmation_starts_need_terminal_verdicts"
        return result
    comparison = next(row for row in report["paired_comparisons"] if row["scope"] == "all" and
                      row["method"] == selection["selected_method"] and row["baseline"] == NATIVE)
    result.update(mean_checks=_mean_gate(comparison), safe_pairs=comparison["safe_pairs"],
                  planned_pairs=comparison["planned_pairs"], metrics=comparison["metrics"])
    for key in ("psr", *QUALITY[:2]):
        interval = comparison["metrics"][key].get("ci95")
        endpoint = interval.get("lower" if key == "psr" else "upper") if interval else None
        result["ci_checks"][key] = finite(endpoint) and (endpoint >= 0 if key == "psr" else endpoint < 0)
    passed = all(result["mean_checks"].values()) and all(result["ci_checks"].values())
    result.update(status="CONFIRMED" if passed else "NOT_CONFIRMED", passes=passed,
        recommended_lambda=selection["selected_lambda"] if passed else None,
        reason="all_frozen_point_and_CI_gates_passed" if passed else "point_or_CI_gate_not_met; no_runner_up_or_retuning")
    return result


def build_report(summary, manifest, config, tuning_selection=None):
    validate_config(config)
    phase = manifest.get("phase", manifest.get("profile"))
    if phase not in ("tune", "confirm") or any(value.get("protocol_id") != PROTOCOL_ID for value in (manifest, summary)):
        raise ValueError("not a registered SEM-exposure pilot")
    if summary.get("phase", summary.get("profile", phase)) != phase:
        raise ValueError("summary phase mismatch")
    config_sha = digest(config)
    if manifest.get("config_sha256") != config_sha:
        raise ValueError("pilot config hash mismatch")
    selected = None
    if phase == "confirm":
        selection = _validate_tuning_selection(tuning_selection, config_sha)
        selected = selection["selected_lambda"]
        if manifest.get("selected_lambda") != selected:
            raise ValueError("manifest confirmation lambda differs from locked selection")
    elif tuning_selection is not None:
        raise ValueError("tuning must not receive a selection from confirmation")
    planned = inventory(config, phase, selected)["entries"]
    for actual, expected in zip(manifest["entries"], planned, strict=True):
        if any(actual.get(key) != value for key, value in expected.items() if key != "status"):
            raise ValueError("pilot schedule identity/order differs from registration")
    observed = {row["trial_id"]: row for row in summary["rows"]}
    if len(observed) != len(summary["rows"]) or summary.get("validated") != len(observed):
        raise ValueError("duplicate verdict or validation coverage mismatch")
    normalized_summary = dict(summary, attempted=len(observed),
        success=summary.get("success", sum(bool(row.get("success")) for row in observed.values())),
        correctness_failures=summary.get("correctness_failures", sum(bool(row.get("correctness_errors")) for row in observed.values())))
    rows, errors, complete = _rows(normalized_summary, manifest)
    attempted = sum(entry["status"] != "NOT_RUN" for entry in manifest["entries"])
    if summary.get("attempted") != attempted:
        errors.append("attempt count differs from durable registered starts")
    for row, entry in zip(rows, manifest["entries"], strict=True):
        source = observed.get(row["trial_id"], {})
        for key in ("sem_lambda", "phase", "map_seed", "query_id"):
            if source and key in source and source[key] != entry[key]:
                raise ValueError("audited row identity differs: " + key)
        row.update(**{key: entry[key] for key in ("phase", "map_seed", "query_id", "sem_lambda")},
                   **{key: source.get(key) for key in EXTRA})
        # Durable starts consume their registered slots even if backend launch
        # fails. Keep actual launch evidence separate from benchmark attempts.
        row["backend_launched"] = row["execution_launched"]
        row["executed"] = bool(source and entry["status"] != "NOT_RUN" and row["execution_status"] in TERMINAL)
        if not source and entry["status"] != "NOT_RUN":
            row.update(attempted=True, execution_status="AWAITING_VERDICT", raw_status=entry["status"])
    executed_complete = bool(complete and all(row["executed"] for row in rows))
    methods = [method for method, _ in method_specs(phase, selected)]
    tables, comparisons = [], []
    scopes = [("all", "all", rows)] + [("family", family, [row for row in rows if row["family"] == family]) for family in FAMILIES]
    for scope, group, subset in scopes:
        for method in methods:
            values = [row for row in subset if row["method"] == method]
            table = dict(scope=scope, group=group, method=method, sem_lambda=values[0]["sem_lambda"],
                         **aggregate(values, 20.))
            table.update(actual_planner_launched=sum(row["backend_launched"] for row in values),
                failed_before_launch=sum(row["executed"] and not row["backend_launched"] for row in values),
                unresolved_started=sum(row["attempted"] and not row["executed"] for row in values),
                psr_denominator="durably_started_attempts_with_terminal_verdict; pending_is_unresolved")
            table.update(**{key: stats(row.get(key) for row in values if row["safe"]) for key in EXTRA})
            tables.append(table)
        for method in (item for item in methods if item != NATIVE):
            is_primary = phase == "confirm" and scope == "all" and method == method_for(selected)
            replicates = config["analysis"]["bootstrap_replicates"] if is_primary and executed_complete and not errors else 0
            comparisons.append(dict(scope=scope, group=group, primary=is_primary,
                **paired(subset, method, baseline=NATIVE, replicates=replicates,
                         seed=config["analysis"]["bootstrap_seed"])))
        for method in (item for item in methods if item not in (NATIVE, CONTROL)):
            comparisons.append(dict(scope=scope, group=group, primary=False,
                                    **paired(subset, method, baseline=CONTROL)))
    report = dict(protocol_id=PROTOCOL_ID, schema_version=1, phase=phase, profile=phase,
        config_sha256=config_sha, sources_sha256=digest(manifest.get("sources", {})),
        selected_lambda=selected, status="FAILED_CORRECTNESS" if errors else "DONE" if executed_complete else "INCOMPLETE",
        errors=errors, complete=complete, executed_complete=executed_complete,
        planned=len(rows), attempted=attempted, validated=len(observed),
        executed=sum(row["executed"] for row in rows), success=sum(row["safe"] for row in rows),
        actual_planner_launched=sum(row["backend_launched"] for row in rows),
        unresolved_started=sum(row["attempted"] and not row["executed"] for row in rows),
        tables=tables, paired_comparisons=comparisons, trials=rows,
        no_global_lambda_optimality_claim=True, full_status="NOT_RUN",
        campaign_timing={key: summary.get(key) for key in ("collection_s", "validation_s", "session_wall_s")})
    if manifest.get("synthetic_fixture") or summary.get("synthetic_fixture") or any(row.get("synthetic_fixture") for row in summary["rows"]):
        report["synthetic_fixture"] = True
    if phase == "tune":
        report["selection"] = select_tuning(report)
    else:
        report["confirmation"] = evaluate_confirmation(report, tuning_selection)
    return report


def markdown(report):
    def fmt(value):
        return "—" if value is None else f"{value:.6g}"
    phase = report["phase"]
    decision = report["selection"] if phase == "tune" else report["confirmation"]
    lines = ["# Pilot A3-T SEM exposure — " + ("chọn trọng số" if phase == "tune" else "xác nhận độc lập"), "",
        f"Dữ liệu: **{report['status']}**. Quyết định: **{decision['status']}**.",
        f"{report['attempted']}/{report['planned']} lượt đã bắt đầu; {report['validated']} verdict; "
        f"{report['success']}/{report['executed']} nghiệm cuối exact hợp lệ.", "",
        "J=L_H+λE/40; E là tích phân risk theo chiều dài bề mặt. PRI giữ nguyên trung bình risk của các ô đóng "
        "duy nhất đường tâm chạm. Không so scalar J giữa các λ hoặc với SEM.", "",
        "| Phương pháp | Success/attempt có verdict | L_H mean (m) | PRI mean | E mean | ΔL_H ghép cặp | ΔPRI ghép cặp |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for row in report["tables"]:
        if row["scope"] != "all":
            continue
        contrast = next((item for item in report["paired_comparisons"] if item["scope"] == "all" and
                         item["method"] == row["method"] and item["baseline"] == NATIVE), None)
        deltas = [contrast["metrics"][key]["difference"]["mean"] if contrast else None for key in QUALITY[:2]]
        values = [row[key]["mean"] for key in (*QUALITY[:2], "final_exposure")] + deltas
        lines.append(f"| {row['method']} | {row['success']}/{row['executed']} | " + " | ".join(map(fmt, values)) + " |")
    if phase == "tune":
        lines += ["", "λ0 là control, không được chọn. Gate xét mọi λ dương: PSR≥SEM, meanΔL_H<0 và meanΔPRI<0. "
            "Chỉ xếp hạng trên cùng tập trial mà SEM và mọi λ đủ điều kiện đều thành công; tập này phải có đủ năm family."]
        if decision["ranking"]:
            lines += ["", "| Hạng | λ | Cặp chung | L_H mean trên tập chung | PRI mean trên tập chung |",
                      "|---:|---:|---:|---:|---:|"]
            for index, row in enumerate(decision["ranking"], 1):
                lines.append(f"| {index} | {fmt(row['sem_lambda'])} | {row['common_cohort_n']} | "
                             f"{fmt(row['mean_surface_length_m'])} | {fmt(row['mean_pri'])} |")
        lines += ["", "Candidate đã khóa: " + fmt(decision["selected_lambda"]) + ". Đây là lựa chọn trên tuning, chưa xác nhận ưu thế."]
    else:
        lines += ["", f"Candidate λ={fmt(decision['candidate_lambda'])}; λ được khuyến nghị sau gate: "
            f"{fmt(decision['recommended_lambda'])}.", "",
            "Gate xác nhận giữ dấu của ba mean và yêu cầu CI95: upperΔL_H<0, upperΔPRI<0, lowerΔPSR≥0. "
            "Bootstrap 2.000 lần, seed83158, family cố định → map → paired seed. Chỉ candidate-vs-SEM là đối chiếu chính."]
        if decision.get("metrics"):
            lines += ["", "| Chỉ số A3-T − SEM | Mean | CI95 |", "|---|---:|---:|"]
            for key in (QUALITY[0], QUALITY[1], "psr"):
                metric = decision["metrics"][key]
                ci = metric.get("ci95") or {}
                lines.append(f"| {key} | {fmt(metric['difference']['mean'])} | [{fmt(ci.get('lower'))}, {fmt(ci.get('upper'))}] |")
        lines += ["", "Không dùng confirmation để chọn runner-up hoặc tinh chỉnh λ."]
    lines += ["", "Quality chỉ tính trên cặp có hai đường cuối hợp lệ; PSR giữ mọi lần bắt đầu đã có verdict, "
        "kể cả interruption hoặc failure trước khi backend khởi chạy. Lượt chờ verdict được báo riêng và chặn lựa chọn. "
        "Coverage, từng family, exposure, số ô, tổng risk và thời gian được giữ trong CSV. "
        "SEM là bản tái dựng paper đã đăng ký. Kết quả không chứng minh λ tối ưu toàn cục; full vẫn NOT_RUN."]
    if report["errors"]:
        lines += ["", *["- " + error for error in report["errors"]]]
    return "\n".join(lines) + "\n"


def _plot(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    comparisons = [row for row in report["paired_comparisons"] if row["scope"] == "all" and row["baseline"] == NATIVE]
    weights = {row["method"]: row["sem_lambda"] for row in report["tables"] if row["scope"] == "all"}
    comparisons.sort(key=lambda row: weights[row["method"]])
    figure, axes = plt.subplots(1, 3, figsize=(12., 3.8), constrained_layout=True)
    for axis, metric, label in zip(axes, (*QUALITY[:2], "psr"), ("Mean paired ΔL_H (m)", "Mean paired ΔPRI", "ΔPSR")):
        for row in comparisons:
            value = row["metrics"][metric]["difference"]["mean"]
            if not finite(value):
                continue
            weight = weights[row["method"]]
            axis.scatter([weight], [value], color="#165a85", s=36)
            interval = row["metrics"][metric].get("ci95")
            if interval and finite(interval.get("lower")) and finite(interval.get("upper")):
                axis.vlines(weight, interval["lower"], interval["upper"], color="#165a85", linewidth=1.5)
        axis.axhline(0., color="#777777", linestyle="--", linewidth=.8)
        axis.set_xscale("symlog", linthresh=.5)
        axis.set_xlabel("λ (0 = length control)")
        axis.set_ylabel(label)
        axis.grid(alpha=.2)
    figure.suptitle(f"SEM exposure pilot: {report['phase']} · {report['validated']}/{report['planned']} validated trials")
    figure.savefig(output / "lambda_tradeoff.png", dpi=180)
    figure.savefig(output / "lambda_tradeoff.svg")
    plt.close(figure)


def write_report(report, output, provenance=None):
    """Publish to a fresh directory after the controller has audited all inputs."""
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("refusing to overwrite pilot analysis")
    fixture_root = (ROOT / "tests/fixtures").resolve()
    if report.get("synthetic_fixture") and output != fixture_root and fixture_root not in output.parents:
        raise ValueError("synthetic parser reports/charts must stay under tests/fixtures")
    output.mkdir(parents=True, exist_ok=True)
    published = dict(report)
    published["provenance"] = dict(provenance or {}, analyzer_sha256=file_hash(__file__), created_utc=utc_now())
    dump(output / "summary.json", published)
    dump(output / "provenance.json", published["provenance"])
    decision_key = "selection" if report["phase"] == "tune" else "confirmation"
    dump(output / (decision_key + ".json"), report[decision_key])
    for name, key in (("aggregate.csv", "tables"), ("paired_comparisons.csv", "paired_comparisons"), ("trials.csv", "trials")):
        _write_csv(output / name, report[key])
    if decision_key == "selection":
        _write_csv(output / "ranking.csv", report["selection"]["ranking"])
        _write_csv(output / "common_cohort.csv", report["selection"]["common_cohort"])
    (output / "REPORT.md").write_text(markdown(report), encoding="utf-8")
    _plot(report, output)
    return published
