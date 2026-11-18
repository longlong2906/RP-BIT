"""Registered PRI portfolio analysis; never runs or selects a planner after validation.

Quality is conditional on paired final-safe paths. Every durable attempt stays
in the PSR denominator, including failures before backend launch. Synthetic
parser fixtures are explicitly marked and may not be exported to results/.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import NormalDist, fmean, median, stdev

import numpy as np

from corridor_v4_execution import atomic_json as dump, digest, file_hash

PROTOCOL_ID = "a3t_pri_portfolio_v1"
FAMILIES = ("traversable_hill", "ridge_pass", "rolling_terrain", "narrow_passage", "mixed_obstacles")
ALPHAS = (1., 1.025, 1.05, 1.1)
NATIVE = "sem_native"
QUALITY = ("final_cost_terrain_m", "final_pri")
TERMINAL = {"COMPLETED", "CRASH", "WATCHDOG", "TIMEOUT", "BLOCKED", "FAILED", "INTERRUPTED", "FAILED_CORRECTNESS"}
EXTRA = ("final_swept_slope_deg", "final_length_xy_m", "final_exposure", "final_risk_sum", "final_risk_count",
         "final_sharp_turn_count", "first_safe_s", "first_safe_latency_s", "preprocessing_s", "field_build_s",
         "solve_time_s", "validation_s", "queue_wait_s", "peak_rss_bytes", "anchor_length_m", "length_cap_m")


def method_for(alpha):
    if alpha not in ALPHAS:
        raise ValueError("unregistered length cap ratio")
    return "a3t_pri_a" + str(round(alpha*1000))


def finite(value):
    return type(value) in (float, int) and math.isfinite(value)


def stats(values):
    values = [float(v) for v in values if finite(v)]
    return dict(n=len(values), mean=fmean(values) if values else None,
                sd=stdev(values) if len(values) > 1 else None,
                median=median(values) if values else None,
                minimum=min(values) if values else None, maximum=max(values) if values else None)


def wilson(success, attempted, confidence=.975):
    """Finite-sample, non-degenerate descriptive binomial interval.

97.5% marginal intervals yield a conservative 95% difference by a union
bound. These intervals do not prove population PSR non-inferiority, nor do
they account for generalization from the sampled map distribution.
"""
    if attempted <= 0:
        return dict(lower=None, upper=None, confidence=confidence)
    z = NormalDist().inv_cdf((1+confidence)/2)
    p, n = success/attempted, attempted
    center = (p+z*z/(2*n))/(1+z*z/n)
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return dict(lower=max(0., center-half), upper=min(1., center+half), confidence=confidence,
                method="Wilson_marginal_descriptive_binomial_not_map_population_guarantee")


def _slot(row):
    return row["map_id"], row.get("query_id", "q0"), row["planner_seed"]


def _normalize(summary, manifest):
    entries, observations = manifest["entries"], summary["rows"]
    ids = [e["trial_id"] for e in entries]
    by_id = {r["trial_id"]: r for r in observations}
    if len(set(ids)) != len(ids) or len(by_id) != len(observations) or set(by_id)-set(ids):
        raise ValueError("duplicate or unregistered trial identity")
    errors, rows = [], []
    if manifest.get("planned_trials", len(ids)) != len(ids) or summary.get("planned", len(ids)) != len(ids):
        errors.append("planned_count_mismatch")
    for entry in entries:
        source = by_id.get(entry["trial_id"])
        started = bool(source or entry.get("status", "NOT_RUN") != "NOT_RUN")
        row = {k: entry.get(k) for k in ("trial_id", "method", "map_id", "family", "map_seed", "planner_seed", "alpha", "phase")}
        row.update(query_id=entry.get("query_id", "q0"), attempted=started, executed=False,
                   backend_launched=False, safe=False, attained=False, completed=False,
                   execution_status="AWAITING_VERDICT" if started else "NOT_RUN", correctness_errors=[],
                   **{k: None for k in (*QUALITY, *EXTRA)})
        if source:
            for key in ("method", "map_id", "family", "planner_seed", "alpha"):
                if source.get(key) != entry.get(key):
                    raise ValueError("audited row identity mismatch: "+key)
            status = source.get("execution_status")
            trial_errors = source.get("correctness_errors", [])
            if not isinstance(trial_errors, list):
                raise ValueError("invalid correctness_errors")
            if status not in TERMINAL:
                errors.append(entry["trial_id"]+":nonterminal_verdict")
            if entry.get("status", status) != status:
                errors.append(entry["trial_id"]+":manifest_status_mismatch")
            if trial_errors:
                errors.append(entry["trial_id"]+":correctness_errors")
            good = status == "COMPLETED" and not trial_errors
            safe = bool(good and source.get("success") and source.get("planner_status") == "EXACT_SOLUTION")
            if bool(source.get("success")) != safe:
                errors.append(entry["trial_id"]+":inconsistent_success")
            if safe and (not source.get("attained_safe") or any(not finite(source.get(k)) or source[k] < 0 for k in QUALITY)):
                raise ValueError("safe trial lacks safe attainment or finite quality")
            row.update(executed=status in TERMINAL, backend_launched=bool(source.get("execution_launched", status in {"COMPLETED", "CRASH", "TIMEOUT", "WATCHDOG"})),
                       completed=status == "COMPLETED", safe=safe, attained=bool(good and source.get("attained_safe")),
                       execution_status=status, planner_status=source.get("planner_status"), correctness_errors=trial_errors,
                       **{k: source.get(k) for k in EXTRA},
                       **{k: source.get(k) if safe else None for k in QUALITY})
            if not row["attained"]:
                row["first_safe_s"] = row["first_safe_latency_s"] = None
        rows.append(row)
    attempted = sum(r["attempted"] for r in rows)
    if summary.get("attempted", attempted) != attempted:
        errors.append("durable_attempt_count_mismatch")
    if summary.get("validated", len(by_id)) != len(by_id):
        errors.append("validation_count_mismatch")
    if summary.get("success", sum(r["safe"] for r in rows)) != sum(r["safe"] for r in rows):
        errors.append("success_count_mismatch")
    if summary.get("correctness_failures", sum(bool(r.get("correctness_errors")) for r in observations)) != sum(bool(r.get("correctness_errors")) for r in observations):
        errors.append("correctness_count_mismatch")
    slots = {}
    for row in rows:
        key = (row["method"], *_slot(row))
        if key in slots:
            raise ValueError("duplicate method/map/query/seed")
        slots[key] = row
    methods = sorted({r["method"] for r in rows})
    registered = [{_slot(r) for r in rows if r["method"] == m} for m in methods]
    if not methods or NATIVE not in methods or any(s != registered[0] for s in registered):
        errors.append("unpaired_or_missing_native_schedule")
    return rows, errors, len(by_id) == len(entries) and all(r["executed"] for r in rows)


def aggregate(rows):
    attempted = [r for r in rows if r["attempted"]]
    safe = [r for r in rows if r["safe"]]
    length = sum(r[QUALITY[0]] for r in safe)
    turns = [r.get("final_sharp_turn_count") for r in safe]
    return dict(planned=len(rows), attempted=len(attempted), executed=sum(r["executed"] for r in rows),
                pending=sum(r["attempted"] and not r["executed"] for r in rows), not_run=len(rows)-len(attempted),
                success=len(safe), psr=len(safe)/len(attempted) if attempted else None,
                psr_interval=wilson(len(safe), len(attempted)),
                psr_denominator="all_durable_attempts_including_prelaunch_failure; pending_explicit",
                backend_launched=sum(r["backend_launched"] for r in rows),
                failure_counts={s: sum(r["execution_status"] == s for r in rows) for s in sorted({r["execution_status"] for r in rows})},
                sti=sum(turns)/length if length > 0 and all(finite(t) for t in turns) else 0. if safe and length == 0 and all(t == 0 for t in turns) else None,
                **{k: stats(r[k] for r in safe) for k in (*QUALITY, "final_swept_slope_deg", "final_length_xy_m", "final_exposure", "final_risk_sum", "final_risk_count", "solve_time_s")},
                **{k: stats(r[k] for r in attempted) for k in ("first_safe_s", "first_safe_latency_s", "preprocessing_s", "field_build_s", "validation_s", "peak_rss_bytes")})


def paired(rows, method, *, cohort=None):
    left = {_slot(r): r for r in rows if r["method"] == method}
    right = {_slot(r): r for r in rows if r["method"] == NATIVE}
    slots = sorted(set(left) & set(right))
    good = [s for s in slots if left[s]["safe"] and right[s]["safe"] and (cohort is None or s in cohort)]
    a, b = aggregate(list(left.values())), aggregate(list(right.values()))
    ia, ib = a["psr_interval"], b["psr_interval"]
    interval = dict(lower=ia["lower"]-ib["upper"], upper=ia["upper"]-ib["lower"], confidence=.95,
                    method="conservative_difference_of_97.5pct_Wilson_marginals; descriptive_only") if a["attempted"] and b["attempted"] else None
    return dict(method=method, baseline=NATIVE, sign="A3_minus_SEM", planned_pairs=len(slots), safe_pairs=len(good),
                safe_maps=len({s[0] for s in good}), paired_coverage=len(good)/len(slots) if slots else None,
                psr_a3=a["psr"], psr_sem=b["psr"], delta_psr=a["psr"]-b["psr"] if a["psr"] is not None and b["psr"] is not None else None,
                psr_difference_interval=interval,
                metrics={k: dict(difference=stats(left[s][k]-right[s][k] for s in good),
                                  left=stats(left[s][k] for s in good), right=stats(right[s][k] for s in good)) for k in QUALITY},
                pairs=[dict(map_id=s[0], query_id=s[1], planner_seed=s[2], family=left[s]["family"],
                            **{k:left[s][k]-right[s][k] if s in good else None for k in QUALITY}) for s in slots])


def hierarchical_upper(pairs, *, replicates=100000, seed=83178):
    """Map bootstrap followed by paired-seed bootstrap; vectorized bounded batches."""
    grouped = {}
    for row in pairs:
        grouped.setdefault(row["map_id"], []).append(row)
    if not grouped:
        return {k: dict(upper=None, available_replicates=0, replicates=replicates) for k in QUALITY}
    maps = sorted(grouped)
    counts = np.array([len(grouped[m]) for m in maps], dtype=np.int64)
    width = int(counts.max())
    values = np.full((len(maps), width, len(QUALITY)), np.nan)
    for i, name in enumerate(maps):
        for j, row in enumerate(grouped[name]):
            values[i, j] = [row.get(k) if finite(row.get(k)) else np.nan for k in QUALITY]
    random = np.random.default_rng(seed)
    chunks = []
    for start in range(0, replicates, 2048):
        size = min(2048, replicates-start)
        picked = random.integers(0, len(maps), (size, len(maps)))
        sizes = counts[picked]
        seeds = (random.random((size, len(maps), width))*sizes[..., None]).astype(np.int64)
        sampled = values[picked[..., None], seeds]
        present = (np.arange(width)[None, None, :] < sizes[..., None])[..., None] & np.isfinite(sampled)
        numerator = np.where(present, sampled, 0.).sum(axis=(1, 2))
        denominator = present.sum(axis=(1, 2))
        means = np.full_like(numerator, np.nan)
        np.divide(numerator, denominator, out=means, where=denominator > 0)
        chunks.append(means)
    draws = np.concatenate(chunks)
    result = {}
    for index, key in enumerate(QUALITY):
        available = draws[:, index][np.isfinite(draws[:, index])]
        result[key] = dict(upper=float(np.quantile(available, .995)) if len(available) == replicates else None,
                           available_replicates=len(available), replicates=replicates, seed=seed,
                           one_sided_confidence=.995, familywise_alpha=.05, registered_quality_conditions=10,
                           order=["map", "paired_seed"], method="percentile_hierarchical_bootstrap_Bonferroni")
    return result


def select_pilot(report):
    result = dict(protocol_id=PROTOCOL_ID, schema_version=1, config_sha256=report["config_sha256"],
                  sources_sha256=report["sources_sha256"], status="INCOMPLETE", selected_alpha=None, selected_method=None,
                  candidates=[], ranking=[], common_cohort=[], common_cohort_sha256=None,
                  ranking_rule="minimum_worst_family_delta_PRI_then_mean_family_delta_L_then_alpha")
    if report["errors"]:
        result.update(status="FAILED_CORRECTNESS", reason="correctness_errors")
        return result
    if not report["complete"]:
        return result
    methods = [NATIVE, *(method_for(a) for a in ALPHAS)]
    lookup = {m: {_slot(r):r for r in report["trials"] if r["method"] == m and r["safe"]} for m in methods}
    common = set.intersection(*(set(lookup[m]) for m in methods))
    cohort = [dict(map_id=s[0], query_id=s[1], planner_seed=s[2], family=lookup[NATIVE][s]["family"]) for s in sorted(common)]
    result.update(common_cohort=cohort, common_cohort_sha256=digest(cohort), common_cohort_count=len(cohort))
    expected = {f: {r["map_id"] for r in report["trials"] if r["family"] == f} for f in FAMILIES}
    covered = {f: {r["map_id"] for r in cohort if r["family"] == f} for f in FAMILIES}
    result["common_cohort_map_counts"] = {f:len(covered[f]) for f in FAMILIES}
    if any(not expected[f] or covered[f] != expected[f] for f in FAMILIES):
        result.update(status="UNRANKABLE", reason="common_final_safe_cohort_must_cover_every_pilot_map_in_all_five_families")
        return result
    for alpha in ALPHAS:
        comparisons = []
        for family in FAMILIES:
            comparison = paired([r for r in report["trials"] if r["family"] == family], method_for(alpha), cohort=common)
            dl = comparison["metrics"][QUALITY[0]]["difference"]["mean"]
            dr = comparison["metrics"][QUALITY[1]]["difference"]["mean"]
            checks = dict(pri=finite(dr) and dr < 0., length=finite(dl) and dl <= 0., psr=finite(comparison["delta_psr"]) and comparison["delta_psr"] >= 0.)
            comparisons.append(dict(family=family, mean_delta_length=dl, mean_delta_pri=dr, delta_psr=comparison["delta_psr"], checks=checks,
                                    safe_pairs=comparison["safe_pairs"], safe_maps=comparison["safe_maps"]))
        candidate = dict(alpha=alpha, method=method_for(alpha), families=comparisons,
                         passes=all(all(c["checks"].values()) for c in comparisons))
        result["candidates"].append(candidate)
        if candidate["passes"]:
            result["ranking"].append(dict(alpha=alpha, method=method_for(alpha), worst_family_delta_pri=max(c["mean_delta_pri"] for c in comparisons),
                                           mean_family_delta_length=fmean(c["mean_delta_length"] for c in comparisons)))
    result["ranking"].sort(key=lambda r:(r["worst_family_delta_pri"], r["mean_family_delta_length"], r["alpha"]))
    if result["ranking"]:
        best = result["ranking"][0]
        result.update(status="CANDIDATE_LOCKED", selected_alpha=best["alpha"], selected_method=best["method"], reason="all_five_family_observed_gates_pass; independent_confirmation_required")
    else:
        result.update(status="NO_CANDIDATE", reason="no_alpha_passes_every_family_gate")
    return result


def evaluate_confirmation(report, selection):
    if not isinstance(selection, dict) or selection.get("status") != "CANDIDATE_LOCKED" or selection.get("config_sha256") != report["config_sha256"]:
        raise ValueError("confirmation requires the hash-bound pilot selection")
    alpha = selection.get("selected_alpha")
    if alpha not in ALPHAS or selection.get("selected_method") != method_for(alpha) or report.get("selected_alpha") != alpha:
        raise ValueError("confirmation candidate differs from locked selection")
    result = dict(protocol_id=PROTOCOL_ID, status="INCOMPLETE", selected_alpha=alpha, selected_method=method_for(alpha),
                  selection_sha256=digest(selection), families=[], passes=False, no_runner_up=True,
                  psr_gate="observed_PSRA3>=PSRSEM_each_family; not_population_noninferiority_proof")
    if report["errors"]:
        result["status"] = "FAILED_CORRECTNESS"
        return result
    if not report["complete"]:
        return result
    for family in FAMILIES:
        comparison = next(c for c in report["paired_comparisons"] if c["scope"] == "family" and c["group"] == family and c["method"] == method_for(alpha))
        bounds = hierarchical_upper(comparison["pairs"])
        dl, dr = (comparison["metrics"][k]["difference"]["mean"] for k in QUALITY)
        u_l, u_r = (bounds[k]["upper"] for k in QUALITY)
        expected_maps = {r["map_id"] for r in report["trials"] if r["family"] == family}
        checks = dict(pri_mean=finite(dr) and dr < 0., length_mean=finite(dl) and dl <= 0.,
                      psr_observed=finite(comparison["delta_psr"]) and comparison["delta_psr"] >= 0.,
                      pri_upper=finite(u_r) and u_r < 0., length_upper=finite(u_l) and u_l <= 0.,
                      map_coverage=comparison["safe_maps"] == len(expected_maps) and bool(expected_maps))
        result["families"].append(dict(family=family, checks=checks, passes=all(checks.values()),
                                       mean_delta_length=dl, mean_delta_pri=dr, delta_psr=comparison["delta_psr"],
                                       upper_bounds=bounds, psr_difference_interval=comparison["psr_difference_interval"], safe_pairs=comparison["safe_pairs"]))
    result["passes"] = all(r["passes"] for r in result["families"])
    result["status"] = "CONFIRMED" if result["passes"] else "NOT_CONFIRMED"
    return result


def build_report(summary, manifest, config, tuning_selection=None):
    if any(v.get("protocol_id") != PROTOCOL_ID for v in (summary, manifest, config)):
        raise ValueError("unregistered portfolio protocol")
    phase = manifest.get("phase", manifest.get("profile"))
    if phase not in ("pilot", "confirm", "smoke_serial", "smoke_parallel", "smoke"):
        raise ValueError("unknown portfolio phase")
    if summary.get("phase", summary.get("profile", phase)) != phase:
        raise ValueError("summary phase mismatch")
    config_hash = digest(config)
    if manifest.get("config_sha256") != config_hash:
        raise ValueError("configuration hash mismatch")
    synthetic = bool(manifest.get("synthetic_fixture") or summary.get("synthetic_fixture"))
    if not synthetic:
        import pri_portfolio_protocol as protocol
        protocol.validate_config(config)
        registered = protocol.inventory(config, phase, manifest.get("selected_alpha"))
        if len(registered["entries"]) != len(manifest["entries"]):
            raise ValueError("registered inventory cardinality mismatch")
        for expected, actual in zip(registered["entries"], manifest["entries"], strict=True):
            if any(actual.get(k) != v for k,v in expected.items() if k != "status"):
                raise ValueError("registered inventory differs")
    rows, errors, complete = _normalize(summary, manifest)
    methods = list(dict.fromkeys(r["method"] for r in rows))
    tables, comparisons = [], []
    families = list(dict.fromkeys(r["family"] for r in rows))
    scopes = [("all", "all", rows)] + [("family", f, [r for r in rows if r["family"] == f]) for f in families]
    scopes += [("map", m, [r for r in rows if r["map_id"] == m]) for m in dict.fromkeys(r["map_id"] for r in rows)]
    for scope, group, subset in scopes:
        for method in methods:
            values = [r for r in subset if r["method"] == method]
            if values:
                tables.append(dict(scope=scope, group=group, method=method, alpha=values[0]["alpha"], **aggregate(values)))
        for method in methods:
            if method != NATIVE:
                comparisons.append(dict(scope=scope, group=group, **paired(subset, method)))
    report = dict(protocol_id=PROTOCOL_ID, schema_version=1, phase=phase, profile=phase,
                  config_sha256=config_hash, sources_sha256=digest(manifest.get("sources", {})),
                  selected_alpha=manifest.get("selected_alpha"), status="FAILED_CORRECTNESS" if errors else "DONE" if complete else "INCOMPLETE",
                  complete=complete, executed_complete=complete, errors=errors, planned=len(rows),
                  attempted=sum(r["attempted"] for r in rows), validated=len(summary["rows"]),
                  executed=sum(r["executed"] for r in rows), success=sum(r["safe"] for r in rows),
                  tables=tables, paired_comparisons=comparisons, trials=rows, synthetic_fixture=synthetic,
                  runtime_comparison="same_algorithm_budget; not_original_paper_runtime_replication",
                  sem_implementation="static_paper_reconstruction; official_implementation=false")
    if phase == "pilot":
        report["selection"] = select_pilot(report)
    elif phase == "confirm":
        report["confirmation"] = evaluate_confirmation(report, tuning_selection)
    return report


def markdown(report):
    fmt = lambda v: "—" if v is None else f"{v:.6g}"
    def spread(metric):
        return fmt(metric["mean"])+" ± "+fmt(metric["sd"])
    decision = report.get("selection", report.get("confirmation", {}))
    lines = ["# A3-T PRI portfolio — báo cáo thực nghiệm", "",
             f"Pha **{report['phase']}**; trạng thái **{report['status']}**; quyết định **{decision.get('status', 'SMOKE_DESCRIPTIVE')}**.",
             f"Đã bắt đầu {report['attempted']}/{report['planned']} lượt; {report['validated']} verdict; {report['success']} nghiệm cuối exact hợp lệ.", "",
             "SEM-RRT* là bản tái dựng tĩnh, không phải mã tác giả. Bản đồ tổng hợp và ngân sách khác paper; không so trực tiếp runtime với Bảng II.",
             "APL=L_H được oracle tính lại; PRI là trung bình risk trên tập ô đóng duy nhất đường tâm chạm. PRI không phải xác suất tai nạn.",
             "PSR giữ mọi attempt, gồm lỗi trước launch. Chất lượng mean ± SD chỉ trên final-safe; SD không phải CI. STI=tổng góc XY>30°/tổng L_H.", "",
             "| Nhóm map | Thuật toán | Success/attempt | PSR | APL (m) | PRI | STI (1/m) | Time solve TB (ms) |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in report["tables"]:
        if row["scope"] == "family":
            lines.append(f"| {row['group']} | {row['method']} | {row['success']}/{row['attempted']} | {fmt(100*row['psr'] if row['psr'] is not None else None)}% | {spread(row[QUALITY[0]])} | {spread(row[QUALITY[1]])} | {fmt(row['sti'])} | {fmt(1000*row['solve_time_s']['mean'] if row['solve_time_s']['mean'] is not None else None)} |")
    lines += ["", "| Nhóm map | A3-T − SEM | Cặp final-safe/dự kiến | ΔL_H TB (m) | ΔPRI TB | ΔPSR quan sát |",
              "|---|---|---:|---:|---:|---:|"]
    for row in report["paired_comparisons"]:
        if row["scope"] == "family":
            lines.append(f"| {row['group']} | {row['method']} | {row['safe_pairs']}/{row['planned_pairs']} | {fmt(row['metrics'][QUALITY[0]]['difference']['mean'])} | {fmt(row['metrics'][QUALITY[1]]['difference']['mean'])} | {fmt(row['delta_psr'])} |")
    if report["phase"] == "pilot":
        lines += ["", "Gate pilot áp riêng cho cả năm nhóm trên giao final-safe của SEM và cả bốn α, đủ mọi map pilot. Một α chung được chọn theo worst-family ΔPRI, mean-family ΔL rồi α nhỏ.",
                  f"α được khóa: {fmt(decision.get('selected_alpha'))}. Không có ứng viên đạt thì dừng; không chạy xác nhận để chọn lại."]
    elif report["phase"] == "confirm":
        lines += ["", "Xác nhận dùng 100.000 bootstrap map→paired seed, seed83178. Cận trên một phía99,5% cho10 điều kiện chất lượng; PSR chỉ xét không giảm quan sát.",
                  "Khoảng PSR mô tả dùng Wilson không suy biến; 100/100 ở hai bên không chứng minh population PSR bằng nhau. Không đổi α hoặc chọn runner-up sau xác nhận.", "",
                  "| Nhóm | Upper ΔL_H | Upper ΔPRI | Đạt tất cả gate |", "|---|---:|---:|---|"]
        for row in decision.get("families", []):
            lines.append(f"| {row['family']} | {fmt(row['upper_bounds'][QUALITY[0]]['upper'])} | {fmt(row['upper_bounds'][QUALITY[1]]['upper'])} | {row['passes']} |")
    else:
        lines += ["", "Smoke chỉ kiểm pipeline; không chọn α hoặc kết luận thắng/thua tổng quát từ smoke."]
    lines += ["", "Chi tiết theo từng map/trial, failure, paired coverage, first-safe, preprocessing, exposure, tổng risk và số ô nằm trong CSV/JSON kèm theo."]
    if report["synthetic_fixture"]:
        lines.insert(2, "**SYNTHETIC TEST FIXTURE — KHÔNG PHẢI KẾT QUẢ THỰC NGHIỆM.**")
    if report["errors"]:
        lines += ["", "Lỗi correctness: "+json.dumps(report["errors"], ensure_ascii=False)]
    return "\n".join(lines)+"\n"


def _write_csv(path, rows):
    flat = []
    for row in rows:
        item = {}
        for key, value in row.items():
            if key == "pairs":
                continue
            if isinstance(value, dict) and "mean" in value:
                item.update({key+"_"+k:v for k,v in value.items()})
            else:
                item[key] = json.dumps(value, sort_keys=True, allow_nan=False) if isinstance(value, (dict, list)) else value
        flat.append(item)
    fields = list(dict.fromkeys(k for row in flat for k in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(flat)


def write_report(summary, manifest, config, output, tuning_selection=None):
    report = build_report(summary, manifest, config, tuning_selection)
    output = Path(output).resolve()
    if report["synthetic_fixture"] and "results" in {p.lower() for p in output.parts}:
        raise ValueError("synthetic fixtures must not enter experimental results")
    output.mkdir(parents=True, exist_ok=True)
    dump(output/"report.json", report)
    (output/"REPORT.md").write_text(markdown(report), encoding="utf-8")
    for name, key in (("aggregate.csv", "tables"), ("trials.csv", "trials"), ("paired_comparisons.csv", "paired_comparisons")):
        _write_csv(output/name, report[key])
    pair_rows = [dict(method=c["method"], baseline=NATIVE, **r) for c in report["paired_comparisons"] if c["scope"] == "all" for r in c["pairs"]]
    _write_csv(output/"paired_trials.csv", pair_rows)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--selection", type=Path)
    args = parser.parse_args(argv)
    # The production CLI reconstructs verdicts from the runner's hash-bound
    # raw inventory/spool audit; a cached summary is never trusted as input.
    from run_pri_portfolio import analyze
    directory = args.input.resolve()
    if args.selection is not None and args.selection.resolve() != directory.parent/"selection.json":
        raise ValueError("confirmation selection must be the campaign's audited receipt")
    result = analyze(directory.parent, directory.name, report_output=args.output)
    print(json.dumps({k:result[k] for k in ("status", "planned", "attempted", "success")}))
    return 0 if result["status"] == "DONE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
