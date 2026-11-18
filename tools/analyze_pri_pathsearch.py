"""Analyze physical attempts and their correlated PRI selector outputs.

Pilot selection is a development decision only. No held-out/full workflow is
implemented or authorized by this analyzer. Synthetic fixtures stay in tests.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean

from corridor_v4_execution import atomic_json as dump, digest
from analyze_pri_portfolio import aggregate, paired, finite, _write_csv, QUALITY, EXTRA, TERMINAL
from pri_pathsearch_protocol import (ALPHAS, FAMILIES, NATIVE, LENGTH, LEGACY, SHORTCUT, FULL,
    METHODS, SELECTOR_BACKENDS, PHASES, PROTOCOL_ID, selector_method)


def _slot(row):
    return row["map_id"], row.get("query_id", "q0"), row["planner_seed"]


def _normalize(summary, manifest):
    entries, observations = manifest["entries"], summary["rows"]
    ids = [e["trial_id"] for e in entries]
    by_id = {r["trial_id"]: r for r in observations}
    if len(ids) != len(set(ids)) or len(by_id) != len(observations) or set(by_id)-set(ids):
        raise ValueError("duplicate or unregistered physical trial identity")
    if manifest.get("planned_trials", len(ids)) != len(ids) or summary.get("planned", len(ids)) != len(ids):
        raise ValueError("physical attempt count mismatch")
    physical, projected, errors = [], [], []
    for entry in entries:
        source = by_id.get(entry["trial_id"])
        started = bool(source or entry.get("status", "NOT_RUN") != "NOT_RUN")
        backend = entry["method"]
        if backend not in METHODS:
            raise ValueError("unregistered physical backend")
        row = {k: entry.get(k) for k in ("trial_id", "method", "map_id", "family", "map_seed", "planner_seed", "alpha", "phase")}
        row.update(query_id=entry.get("query_id", "q0"), physical_trial_id=entry["trial_id"],
            physical_backend=backend, attempted=started, executed=False, backend_launched=False,
            safe=False, attained=False, completed=False, correctness_errors=[],
            execution_status="AWAITING_VERDICT" if started else "NOT_RUN",
            **{k: None for k in (*QUALITY, *EXTRA)})
        selectors = []
        if source:
            for key in ("method", "map_id", "family", "planner_seed", "alpha", "phase"):
                if source.get(key) != entry.get(key):
                    raise ValueError("physical verdict identity mismatch: " + key)
            status = source.get("execution_status")
            faults = source.get("correctness_errors", [])
            if not isinstance(faults, list):
                raise ValueError("invalid correctness errors")
            if status not in TERMINAL:
                errors.append(entry["trial_id"]+":nonterminal_verdict")
            if entry.get("status", status) != status:
                errors.append(entry["trial_id"]+":manifest_status_mismatch")
            if faults:
                errors.append(entry["trial_id"]+":correctness_errors")
            good = status == "COMPLETED" and not faults
            safe = bool(good and source.get("success") and source.get("planner_status") == "EXACT_SOLUTION")
            if bool(source.get("success")) != safe:
                errors.append(entry["trial_id"]+":inconsistent_physical_success")
            row.update(executed=status in TERMINAL, completed=status == "COMPLETED", safe=safe,
                attained=bool(good and source.get("attained_safe")),
                backend_launched=bool(source.get("execution_launched", False)),
                execution_status=status, planner_status=source.get("planner_status"), correctness_errors=faults,
                **{k: source.get(k) for k in EXTRA},
                **{k: source.get(k) if safe else None for k in QUALITY})
            selectors = source.get("selector_results", [])
            if not isinstance(selectors, list):
                raise ValueError("selector_results must be an array")
            if backend in SELECTOR_BACKENDS:
                alphas = [r.get("alpha") for r in selectors]
                if len(alphas) != len(set(alphas)) or any(type(a) not in (float, int) or a not in ALPHAS for a in alphas):
                    raise ValueError("duplicate or unknown correlated selector")
                if good and set(alphas) != set(ALPHAS):
                    raise ValueError("completed backend lacks four registered selector verdicts")
                if safe != bool(good and len(selectors) == 4 and all(r.get("success") for r in selectors)):
                    errors.append(entry["trial_id"]+":selector_physical_success_disagrees")
            elif selectors:
                raise ValueError("scalar backend must not create independent selector trials")
        physical.append(dict(row))
        outputs = ALPHAS if backend in SELECTOR_BACKENDS else (None,)
        for alpha in outputs:
            view = dict(row)
            if alpha is not None:
                selector = next((s for s in selectors if s["alpha"] == alpha), {})
                success = bool(source and source.get("execution_status") == "COMPLETED"
                               and not source.get("correctness_errors") and selector.get("success"))
                view.update(method=selector_method(backend, alpha), alpha=alpha,
                    trial_id=entry["trial_id"]+"__a"+str(round(alpha*1000)),
                    selector_alpha=alpha, correlated_selectors=True, safe=success,
                    attained=bool(success and selector.get("attained_safe")),
                    selected_snapshot_id=selector.get("selected_snapshot_id"),
                    **{k: selector.get(k, view.get(k)) if success else None for k in QUALITY},
                    **{k: selector.get(k, view.get(k)) for k in EXTRA})
            else:
                view.update(selector_alpha=None, correlated_selectors=False)
            if view["safe"] and (not view["attained"] or any(not finite(view[k]) or view[k] < 0 for k in QUALITY)):
                raise ValueError("final-safe result lacks finite metrics or safe attainment")
            if not view["attained"]:
                view["first_safe_s"] = view["first_safe_latency_s"] = None
            projected.append(view)
    for name, actual in (("attempted", sum(r["attempted"] for r in physical)),
                         ("validated", len(by_id)), ("success", sum(r["safe"] for r in physical)),
                         ("correctness_failures", sum(bool(r.get("correctness_errors")) for r in observations))):
        if summary.get(name, actual) != actual:
            errors.append("physical_"+name+"_count_mismatch")
    registered_slots = [{_slot(r) for r in physical if r["method"] == m} for m in METHODS]
    if not all(registered_slots) or any(s != registered_slots[0] for s in registered_slots):
        errors.append("unpaired_or_missing_physical_backend_schedule")
    if len({(r["method"], *_slot(r)) for r in physical}) != len(physical):
        raise ValueError("duplicate physical backend/map/query/seed")
    complete = len(by_id) == len(entries) and all(r["executed"] for r in physical)
    return physical, projected, errors, complete


def select_pilot(report):
    result = dict(protocol_id=PROTOCOL_ID, schema_version=1, config_sha256=report["config_sha256"],
        sources_sha256=report["sources_sha256"], status="INCOMPLETE", selected_alpha=None, selected_method=None,
        candidates=[], ranking=[], common_cohort=[], common_cohort_sha256=None, stop_after_pilot=True,
        primary_backend=FULL, selectors_are_correlated=True, held_out_evidence=False,
        ranking_rule="minimum_worst_family_delta_PRI_then_mean_family_delta_L_then_alpha")
    if report["errors"]:
        result.update(status="FAILED_CORRECTNESS", reason="correctness_or_integrity_errors")
        return result
    if not report["complete"]:
        return result
    methods = [NATIVE, *(selector_method(FULL, a) for a in ALPHAS)]
    lookup = {m: {_slot(r): r for r in report["trials"] if r["method"] == m and r["safe"]} for m in methods}
    common = set.intersection(*(set(lookup[m]) for m in methods))
    cohort = [dict(map_id=s[0], query_id=s[1], planner_seed=s[2], family=lookup[NATIVE][s]["family"]) for s in sorted(common)]
    result.update(common_cohort=cohort, common_cohort_sha256=digest(cohort), common_cohort_count=len(cohort))
    expected = {f: {r["map_id"] for r in report["physical_trials"] if r["family"] == f} for f in FAMILIES}
    covered = {f: {r["map_id"] for r in cohort if r["family"] == f} for f in FAMILIES}
    result["common_cohort_map_counts"] = {f: len(covered[f]) for f in FAMILIES}
    if any(not expected[f] or covered[f] != expected[f] for f in FAMILIES):
        result.update(status="UNRANKABLE", reason="common_full_selector_cohort_must_cover_every_development_map")
        return result
    for alpha in ALPHAS:
        method = selector_method(FULL, alpha)
        comparisons = []
        for family in FAMILIES:
            comparison = paired([r for r in report["trials"] if r["family"] == family], method, cohort=common)
            dl, dr = (comparison["metrics"][k]["difference"]["mean"] for k in QUALITY)
            checks = dict(pri=finite(dr) and dr < 0., length=finite(dl) and dl < 0.,
                psr=finite(comparison["delta_psr"]) and comparison["delta_psr"] >= 0.)
            comparisons.append(dict(family=family, mean_delta_length=dl, mean_delta_pri=dr,
                delta_psr=comparison["delta_psr"], checks=checks,
                safe_pairs=comparison["safe_pairs"], safe_maps=comparison["safe_maps"]))
        candidate = dict(alpha=alpha, method=method, families=comparisons,
            passes=all(all(c["checks"].values()) for c in comparisons))
        result["candidates"].append(candidate)
        if candidate["passes"]:
            result["ranking"].append(dict(alpha=alpha, method=method,
                worst_family_delta_pri=max(c["mean_delta_pri"] for c in comparisons),
                mean_family_delta_length=fmean(c["mean_delta_length"] for c in comparisons)))
    result["ranking"].sort(key=lambda r: (r["worst_family_delta_pri"], r["mean_family_delta_length"], r["alpha"]))
    if result["ranking"]:
        best = result["ranking"][0]
        result.update(status="CANDIDATE_LOCKED", selected_alpha=best["alpha"], selected_method=best["method"],
            reason="development_gates_pass; stop_after_pilot; no_held_out_claim")
    else:
        result.update(status="NO_CANDIDATE", reason="no_full_selector_passes_strict_length_and_PRI_gates_in_every_family")
    return result


def build_report(summary, manifest, config):
    import pri_pathsearch_protocol as protocol
    if any(v.get("protocol_id") != PROTOCOL_ID for v in (summary, manifest, config)):
        raise ValueError("unregistered pathsearch protocol")
    phase = manifest.get("phase")
    if phase not in PHASES or summary.get("phase") != phase:
        raise ValueError("unregistered pathsearch phase")
    protocol.validate_config(config)
    config_hash = digest(config)
    if manifest.get("config_sha256") != config_hash:
        raise ValueError("configuration hash mismatch")
    synthetic = bool(manifest.get("synthetic_fixture") or summary.get("synthetic_fixture") or
                     any(r.get("synthetic_fixture") for r in summary["rows"]))
    if not synthetic:
        registered = protocol.inventory(config, phase)
        if len(registered["entries"]) != len(manifest["entries"]) or any(
            not protocol.entry_matches(actual, expected)
            for actual, expected in zip(manifest["entries"], registered["entries"], strict=True)):
            raise ValueError("registered physical inventory differs")
    physical, rows, errors, complete = _normalize(summary, manifest)
    methods = list(dict.fromkeys(r["method"] for r in rows))
    tables, comparisons = [], []
    scopes = [("all", "all", rows)]
    scopes += [("family", f, [r for r in rows if r["family"] == f]) for f in dict.fromkeys(r["family"] for r in rows)]
    scopes += [("map", m, [r for r in rows if r["map_id"] == m]) for m in dict.fromkeys(r["map_id"] for r in rows)]
    for scope, group, subset in scopes:
        for method in methods:
            values = [r for r in subset if r["method"] == method]
            if values:
                tables.append(dict(scope=scope, group=group, method=method, alpha=values[0]["alpha"],
                    physical_backend=values[0]["physical_backend"], correlated_selectors=values[0]["correlated_selectors"],
                    **aggregate(values)))
                if method != NATIVE:
                    comparisons.append(dict(scope=scope, group=group, **paired(subset, method)))
    report = dict(protocol_id=PROTOCOL_ID, schema_version=1, phase=phase, profile=phase,
        config_sha256=config_hash, sources_sha256=digest(manifest.get("sources", {})),
        status="FAILED_CORRECTNESS" if errors else "DONE" if complete else "INCOMPLETE",
        complete=complete, errors=errors, planned=len(physical), attempted=sum(r["attempted"] for r in physical),
        validated=len(summary["rows"]), executed=sum(r["executed"] for r in physical),
        success=sum(r["safe"] for r in physical), physical_trials=physical, trials=rows,
        selector_projection_rows=len(rows), count_unit="physical_backend_attempt",
        tables=tables, paired_comparisons=comparisons, synthetic_fixture=synthetic,
        selector_interpretation="four_correlated_outputs_of_one_run; never_four_independent_trials",
        held_out_evidence=False, stop_after_pilot=True,
        sem_implementation="static_paper_reconstruction; official_implementation=false")
    if phase == "pilot":
        report["selection"] = select_pilot(report)
    return report


def markdown(report):
    def fmt(v):
        return "—" if v is None else f"{v:.6g}"
    decision = report.get("selection", {})
    lines = ["# A3-T PRI path search — báo cáo", "",
        f"Pha **{report['phase']}**, trạng thái **{report['status']}**, quyết định **{decision.get('status', 'SMOKE_DESCRIPTIVE')}**.",
        f"Đã bắt đầu {report['attempted']}/{report['planned']} lượt backend thực; {report['validated']} verdict; {report['success']} backend có đầu ra cuối exact hợp lệ.",
        "Bốn α của full hoặc shortcut là bốn đầu ra tương quan từ cùng một lượt, không phải bốn trial độc lập. "
        "Success backend đòi hỏi mọi đầu ra đăng ký hợp lệ; PSR và quality theo từng α được báo riêng bên dưới.", "",
        "SEM là bản tái dựng tĩnh, không phải mã tác giả. APL là chiều dài bề mặt độc lập; PRI là mean risk trên hợp ô đóng duy nhất. "
        "PSR giữ mọi attempt; mean ± SD chỉ trên final-safe; SD không phải CI.",
        "Pilot dùng map phát triển đã xem. Dừng sau pilot dù có ứng viên; chưa có bằng chứng held-out.", "",
        "| Family | Backend/selector | Success/attempt | APL (m), mean ± SD | PRI, mean ± SD |",
        "|---|---|---:|---:|---:|"]
    for row in report["tables"]:
        if row["scope"] == "family":
            length, risk = (row[k] for k in QUALITY)
            lines.append(f"| {row['group']} | {row['method']} | {row['success']}/{row['attempted']} | {fmt(length['mean'])} ± {fmt(length['sd'])} | {fmt(risk['mean'])} ± {fmt(risk['sd'])} |")
    lines += ["", "| Family | Hiệu so với SEM | Cặp final-safe/dự kiến | Mean ΔL_H (m) | Mean ΔPRI | ΔPSR |",
        "|---|---|---:|---:|---:|---:|"]
    for row in report["paired_comparisons"]:
        if row["scope"] == "family":
            dl, dr = (row["metrics"][k]["difference"]["mean"] for k in QUALITY)
            lines.append(f"| {row['group']} | {row['method']} | {row['safe_pairs']}/{row['planned_pairs']} | {fmt(dl)} | {fmt(dr)} | {fmt(row['delta_psr'])} |")
    if report["phase"] == "pilot":
        lines += ["", "Chỉ chọn full trên giao final-safe SEM và cả bốn selector full, phủ mọi map. "
            "Từng family phải có ΔL_H < 0, ΔPRI < 0 và PSR quan sát không giảm. "
            "Xếp worst-family ΔPRI, rồi mean-family ΔL_H, rồi α nhỏ nhất.",
            "α được khóa: "+fmt(decision.get("selected_alpha"))+". Việc khóa không mở thêm pha thực nghiệm."]
    else:
        lines += ["", "Smoke kiểm pipeline và hiệu lực ablation; không chọn α hoặc kết luận ưu thế."]
    if report["synthetic_fixture"]:
        lines.insert(2, "**SYNTHETIC TEST FIXTURE — KHÔNG PHẢI KẾT QUẢ THỰC NGHIỆM.**")
    if report["errors"]:
        lines += ["", "Lỗi: "+json.dumps(report["errors"], ensure_ascii=False)]
    return "\n".join(lines)+"\n"


def write_report(summary, manifest, config, output):
    report = build_report(summary, manifest, config)
    output = Path(output).resolve()
    if report["synthetic_fixture"] and "results" in {p.lower() for p in output.parts}:
        raise ValueError("synthetic fixtures must not enter experimental results")
    output.mkdir(parents=True, exist_ok=True)
    dump(output/"report.json", report)
    (output/"REPORT.md").write_text(markdown(report), encoding="utf-8")
    for name, key in (("aggregate.csv", "tables"), ("physical_trials.csv", "physical_trials"),
                      ("selector_trials.csv", "trials"), ("paired_comparisons.csv", "paired_comparisons")):
        _write_csv(output/name, report[key])
    _write_csv(output/"paired_trials.csv", [dict(method=c["method"], baseline=NATIVE, **r)
        for c in report["paired_comparisons"] if c["scope"] == "all" for r in c["pairs"]])
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    from run_pri_pathsearch import analyze
    directory = args.input.resolve()
    result = analyze(directory.parent, directory.name, report_output=args.output)
    print(json.dumps({k: result[k] for k in ("status", "planned", "attempted", "success")}))
    return 0 if result["status"] == "DONE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
