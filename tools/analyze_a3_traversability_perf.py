"""Performance report with kernel work separated from certified termination."""
from collections import defaultdict
from pathlib import Path

from a3_traversability_perf_protocol import PROTOCOL_ID, validate_config
from analyze_a3_traversability import analyze as analyze_objective
from experiment_common import dump, read_json


def analyze(source, output, make_plots=True):
    source, output = Path(source), Path(output)
    validate_config(read_json(source/"resolved_config.json"))
    summary = analyze_objective(source, output, make_plots)
    rows = read_json(output/"trial_metrics.json")
    summary["protocol_id"] = PROTOCOL_ID
    summary["certificate_claimed"] = sum(bool((r.get("exact_certificate") or {}).get("certified")) for r in rows)
    summary["certificate_verified"] = sum(bool((r.get("exact_certificate") or {}).get("certified") and
                                                  r.get("exact_certificate_independent_valid") is True) for r in rows)
    summary["performance_scope"] = "application_cold_process; OS filesystem page cache uncontrolled; one seed descriptive only"
    groups = defaultdict(dict)
    for row in rows:
        groups[(row["family"], row["query_id"], row["planner_seed"])][row["planner_id"]] = row
    comparisons = []
    # Kernel arms exclude certification. Separate certificate contrasts expose work saved.
    pairs = (("a3_perf_baseline", "a3_perf_sampler", "sampler_kernel"),
             ("a3_perf_baseline", "a3_perf_preprocessing", "preprocessing_kernel"),
             ("a3_perf_baseline", "a3_perf_both_kernels", "both_kernels"),
             ("a3_perf_baseline", "a3_perf_certificate", "certified_termination"),
             ("a3_perf_both_kernels", "a3_perf_all", "certified_termination_after_kernels"))
    def difference(first, second):
        return second-first if first is not None and second is not None else None
    for (family, query, seed), group in groups.items():
        for control, treatment, purpose in pairs:
            if control not in group or treatment not in group:
                continue
            a, b = group[control], group[treatment]
            metrics = ("first_solution_time_s", "first_solution_end_to_end_s", "actual_solve_wall_s",
                       "preprocessing_s", "traversability_preprocessing_s", "objective_value",
                       "cost_terrain_m", "risk_slope_integral_m", "risk_clearance_integral_m")
            comparisons.append(dict(family=family,query_id=query,seed=seed,control=control,
                treatment=treatment,purpose=purpose,control_success=a["success"],treatment_success=b["success"],
                delta_treatment_minus_control={k:difference(a.get(k),b.get(k)) for k in metrics},
                sampler_interval_calls_difference=difference(a["metrics"].get("sampler_interval_calls"),
                                                            b["metrics"].get("sampler_interval_calls")),
                certificate_verified=bool((b.get("exact_certificate") or {}).get("certified") and
                                          b.get("exact_certificate_independent_valid") is True)))
    dump(output/"performance_comparisons.json", comparisons)
    instrumentation = [{"trial_id":r["trial_id"],"planner_id":r["planner_id"],"family":r["family"],
        "status":r["status"],"metrics":r["metrics"],"unavailable":r["metric_unavailable_reasons"],
        "traversability_build_stats":r.get("traversability_build_stats"),
        "exact_certificate":r.get("exact_certificate"),
        "exact_certificate_independent_valid":r.get("exact_certificate_independent_valid")} for r in rows]
    dump(output/"instrumentation_details.json", instrumentation)
    def fmt(value): return "N/A" if value is None else f"{value:.6g}"
    lines = ["# A3-T — sampler và cold start", "",
        f"{summary['executed']}/{summary['planned']} trial đã chạy; {summary['validated_success']} nghiệm thành công được kiểm chứng.",
        f"Chứng nhận flat_direct_v1: {summary['certificate_verified']}/{summary['certificate_claimed']} claim đã kiểm chứng độc lập.", "",
        "Các nhánh cache/builder giữ certificate tắt để đo tối ưu triển khai. Nhánh certificate đo công việc tìm kiếm được tránh nhờ chứng minh hẹp. Giảm thời gian solve ở nhánh này không được diễn giải là sampler xử lý mỗi lời gọi nhanh hơn.", "",
        "| Fixture | Nhánh | Success | Preprocess (ms) | T_first solve (ms) | T_first gồm setup (ms) | Solve (ms) | Sampler calls | Certificate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        ms=lambda key: None if row.get(key) is None else row[key]*1000
        lines.append(f"| {row['family']} | {row['planner_id']} | {int(row['success'])} | {fmt(ms('preprocessing_s'))} | {fmt(ms('first_solution_time_s'))} | {fmt(ms('first_solution_end_to_end_s'))} | {fmt(ms('actual_solve_wall_s'))} | {fmt(row['metrics'].get('sampler_interval_calls'))} | {bool((row.get('exact_certificate') or {}).get('certified'))} |")
    lines += ["", "L/R/J trong trial_metrics.json và anytime_curves.json luôn thuộc cùng incumbent. Thời điểm nghiệm đầu là callback của snapshot được xác minh offline; không cộng thời gian xác minh vào T_first.",
        "", "Cold start ở đây là tiến trình ứng dụng mới; không xóa hoặc kiểm soát OS page cache. Một seed và ba lần lặp microbenchmark không chứng minh ưu thế thống kê. Timeout, crash, invalid, missing và metric chưa đo được giữ nguyên.",
        "", "Microbenchmark field và replay sampler được báo riêng trong artifact của các harness; không trộn thời gian chúng vào 36 planner trial. Miền informed nhỏ nhưng chưa có chứng nhận vẫn có thể có tỷ lệ chấp nhận thấp."]
    (output/"OBSERVATIONS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    dump(output/"summary.json", summary)
    return summary
