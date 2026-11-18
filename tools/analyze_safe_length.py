"""Descriptive analysis of immutable safe-length smoke results.

No planner, oracle or bootstrap is run. Quality comparisons use paired safe
successes; first-safe times use paired attainment. Restricted times retain
completed attempts with no safe event as right-censored at the budget, never
as a solution event. Incomplete observations make the restricted mean null.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
BASELINES = ("sem_native", "sem_common_safe_length")
QUALITY = ("final_cost_terrain_m", "final_pri", "final_swept_slope_deg")
STARTED_STATUSES = {"COMPLETED", "CRASH", "WATCHDOG"}


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    def reject(value):
        raise ValueError("nonstandard JSON number: " + value)
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def stats(values):
    values = [float(value) for value in values if finite(value)]
    return dict(n=len(values), mean=statistics.fmean(values) if values else None,
                median=statistics.median(values) if values else None,
                minimum=min(values) if values else None, maximum=max(values) if values else None)


def _normalized_rows(summary, manifest):
    errors = []
    entries, source_rows = manifest.get("entries", []), summary.get("rows", [])
    if not isinstance(entries, list) or not isinstance(source_rows, list):
        raise ValueError("manifest entries and summary rows must be lists")
    identities = [entry["trial_id"] for entry in entries]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate planned trial identity")
    lookup = {}
    for row in source_rows:
        if row["trial_id"] in lookup:
            raise ValueError("duplicate observed trial identity")
        lookup[row["trial_id"]] = row
    unknown = set(lookup)-set(identities)
    if unknown:
        raise ValueError("unplanned observed trials: " + ",".join(sorted(unknown)))
    if summary.get("planned") != len(entries) or summary.get("attempted") != len(source_rows):
        errors.append("summary cardinality differs from raw rows/manifest")
    if manifest.get("status") != "COMPLETE" or summary.get("status") != "COMPLETE":
        errors.append("campaign incomplete or failed; no successful analysis status permitted")
    rows = []
    for entry in entries:
        source = lookup.get(entry["trial_id"])
        if source is None:
            errors.append(entry["trial_id"]+": planned slot has no observed row")
            rows.append(dict(trial_id=entry["trial_id"], method=entry["method"], map_id=entry["map_id"],
                             planner_seed=entry["planner_seed"], execution_status="NOT_RUN", attempted=False,
                             executed=False, completed=False, geometry=False, safe=False, attained=False,
                             first_safe_s=None, first_safe_latency_s=None, correctness_errors=[],
                             final_cost_terrain_m=None, final_pri=None, final_swept_slope_deg=None))
            continue
        for key in ("method", "map_id"):
            if source.get(key) != entry[key]:
                raise ValueError("trial identity mismatch: " + entry["trial_id"]+":"+key)
        status = source.get("execution_status")
        if entry.get("status") != status:
            errors.append(entry["trial_id"]+": manifest execution status mismatch")
        completed = status == "COMPLETED"
        trial_errors = list(source.get("correctness_errors", []))
        if trial_errors:
            errors.append(entry["trial_id"]+": correctness errors present")
        good = completed and not trial_errors
        attained = bool(source.get("attained_safe") and good)
        safe = bool(source.get("success") and good and source.get("planner_status") == "EXACT_SOLUTION")
        geometry = bool(source.get("success_geometry") and good and source.get("planner_status") == "EXACT_SOLUTION")
        if source.get("success") and not safe:
            errors.append(entry["trial_id"]+": inconsistent reported safe success")
        if source.get("attained_safe") and not attained:
            errors.append(entry["trial_id"]+": inconsistent reported safe attainment")
        first = source.get("first_safe_s") if attained else None
        latency = source.get("first_safe_latency_s") if attained else None
        if attained and (not finite(first) or first < 0):
            raise ValueError("invalid attained first-safe time")
        if latency is not None and (not finite(latency) or latency < first):
            raise ValueError("invalid first-safe latency")
        if safe and not all(finite(source.get(key)) for key in QUALITY):
            raise ValueError("safe success has unavailable quality metric")
        if not completed and status != "BLOCKED":
            errors.append(entry["trial_id"]+": unsuccessful process status "+str(status))
        if status == "BLOCKED":
            errors.append(entry["trial_id"]+": blocked dependency/attempt")
        # If launch information exists, it resolves FAILED-before-launch versus
        # FAILED-after-launch. COMPLETED/CRASH/WATCHDOG always imply execution.
        launched = source.get("_launched")
        executed = status in STARTED_STATUSES or launched is True
        rows.append(dict(trial_id=entry["trial_id"], method=entry["method"], map_id=entry["map_id"],
            planner_seed=entry["planner_seed"], family=source.get("family"), execution_status=status,
            attempted=True, executed=executed, completed=completed, geometry=geometry, safe=safe,
            attained=attained, first_safe_s=first, first_safe_latency_s=latency,
            correctness_errors=trial_errors, solve_time_s=source.get("solve_time_s"),
            field_build_s=source.get("field_build_s"),
            **{key:source.get(key) if safe else None for key in QUALITY}))
    if summary.get("correctness_failures") != sum(bool(row.get("correctness_errors")) for row in source_rows):
        errors.append("summary correctness count mismatch")
    if summary.get("success") != sum(row["safe"] for row in rows):
        errors.append("summary safe-success count mismatch")
    return rows, errors


def restricted_observation(row, horizon):
    """Time contribution plus event/censoring flag; failures stay unavailable."""
    if not row["executed"]:
        return dict(time_s=None, event=False, censored=False, reason="not_executed")
    if not row["completed"] or row["correctness_errors"]:
        return dict(time_s=None, event=False, censored=False, reason="incomplete_or_invalid_observation")
    if row["attained"] and row["first_safe_s"] <= horizon:
        return dict(time_s=row["first_safe_s"], event=True, censored=False, reason=None)
    return dict(time_s=horizon, event=False, censored=True, reason="no_validated_safe_event_by_horizon")


def restricted_summary(rows, horizon):
    observations = [restricted_observation(row, horizon) for row in rows if row["executed"]]
    missing = sum(value["time_s"] is None for value in observations)
    return dict(horizon_s=horizon, n=len(observations),
        event_count=sum(value["event"] for value in observations),
        censored_at_budget_count=sum(value["censored"] for value in observations),
        unavailable_observation_count=missing,
        mean_time_without_safe_s=(statistics.fmean(value["time_s"] for value in observations)
                                  if observations and not missing else None),
        unavailable_reason="incomplete_or_invalid_observation" if missing else "no_executed_trials" if not observations else None,
        semantics="integral_of_empirical_nonattainment_0_to_budget; censored_time_is_not_a_solution_event")


def build_report(summary, manifest, *, horizon_s=2.):
    if not finite(horizon_s) or horizon_s <= 0:
        raise ValueError("positive finite horizon required")
    rows, errors = _normalized_rows(summary, manifest)
    for row in rows:
        if row["attained"] and row["first_safe_s"] > horizon_s:
            errors.append(row["trial_id"]+": first safe exceeds registered budget")
        row["restricted_observation"] = restricted_observation(row, horizon_s)
    methods = sorted(set(row["method"] for row in rows))
    aggregates = []
    for method in methods:
        group = [row for row in rows if row["method"] == method]
        aggregates.append(dict(method=method, planned=len(group), attempted=sum(row["attempted"] for row in group),
            executed=sum(row["executed"] for row in group), completed=sum(row["completed"] for row in group),
            geometry_success=sum(row["geometry"] for row in group), safe_success=sum(row["safe"] for row in group),
            ever_safe=sum(row["attained"] for row in group),
            execution_status_counts={status:sum(row["execution_status"] == status for row in group)
                                     for status in sorted(set(row["execution_status"] for row in group))},
            first_safe_s=stats(row["first_safe_s"] for row in group if row["attained"]),
            first_safe_latency_s=stats(row["first_safe_latency_s"] for row in group if row["attained"]),
            field_build_s=stats(row.get("field_build_s") for row in group if row["completed"]),
            solve_time_s=stats(row.get("solve_time_s") for row in group if row["completed"]),
            restricted_first_safe=restricted_summary(group, horizon_s),
            **{key:stats(row[key] for row in group if row["safe"]) for key in QUALITY}))
    paired = []
    for baseline in BASELINES:
        right = {(row["map_id"], row["planner_seed"]):row for row in rows if row["method"] == baseline}
        if len(right) != sum(row["method"] == baseline for row in rows):
            raise ValueError("duplicate baseline map/seed slot")
        for method in methods:
            if method in BASELINES:
                continue
            left = {(row["map_id"], row["planner_seed"]):row for row in rows if row["method"] == method}
            if len(left) != sum(row["method"] == method for row in rows):
                raise ValueError("duplicate method map/seed slot")
            slots = sorted(set(left) & set(right))
            for metric in (*QUALITY, "first_safe_s", "first_safe_latency_s", "restricted_time_without_safe_s"):
                pairs, values, unavailable = [], [], 0
                for slot in slots:
                    a, b = left[slot], right[slot]
                    if metric in QUALITY:
                        eligible = a["safe"] and b["safe"]
                    elif metric.startswith("first_safe"):
                        eligible = a["attained"] and b["attained"]
                    else:
                        eligible = a["executed"] and b["executed"]
                    if not eligible:
                        continue
                    pairs.append(slot)
                    av, bv = ((a["restricted_observation"]["time_s"], b["restricted_observation"]["time_s"])
                              if metric == "restricted_time_without_safe_s" else (a[metric], b[metric]))
                    if finite(av) and finite(bv):
                        values.append(float(av)-float(bv))
                    else:
                        unavailable += 1
                difference = stats(values)
                if metric == "restricted_time_without_safe_s" and unavailable:
                    # Do not discard failed observations to manufacture a
                    # favorable complete-case restricted-time comparison.
                    difference = dict(n=len(values), mean=None, median=None, minimum=None, maximum=None)
                paired.append(dict(method=method, baseline=baseline, metric=metric,
                    sign="method_minus_baseline", planned_pairs=len(slots), eligible_pairs=len(pairs),
                    available_pairs=len(values), unavailable_pairs=unavailable,
                    coverage=len(values)/len(slots) if slots else None,
                    maps=len({slot[0] for slot in pairs}), difference=difference,
                    unavailable_reason="incomplete_paired_observation" if unavailable else "no_eligible_pairs" if not pairs else None))
    return dict(schema_version=1, protocol_id=manifest.get("protocol_id"), profile="smoke",
        status="FAILED" if errors else "DONE", errors=errors, horizon_s=horizon_s,
        aggregates=aggregates, paired_comparisons=paired, trials=rows,
        inference="descriptive_smoke_only_no_bootstrap_no_ranking_no_general_superiority_claim",
        timing="serial_one_thread; first_safe_latency includes planner field build, startup/setup and solve callback time; excludes offline validation",
        quality_selection="conditional_on_final_independently_safe_exact_success; paired_quality_uses_both_safe",
        failure_policy="all_planned_slots_retained; no_seed_replacement; no_final_path_reselection",
        pilot_status="NOT_RUN", full_status="NOT_RUN")


def _csv(path, rows):
    keys = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key:json.dumps(value, sort_keys=True, allow_nan=False)
                             if isinstance(value, (dict, list)) else value for key, value in row.items()})


def _format(value, digits=6):
    return "—" if value is None else format(value, "."+str(digits)+"g")


def markdown(report):
    lines = ["# Phân tích smoke A3-T safe-length", "", "Trạng thái: **"+report["status"]+"**.", "",
        "Đây là kiểm thử tích hợp trên bảy fixture, một seed; không có bootstrap, xếp hạng hoặc kết luận ưu thế tổng quát.", "",
        "| Phương pháp | Executed/planned | Hình học | Safe cuối | Từng safe | T_first safe median (s), n | Latency median (s), n | Restricted mean (s), n/censored |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["aggregates"]:
        first, latency, restricted = row["first_safe_s"], row["first_safe_latency_s"], row["restricted_first_safe"]
        lines.append(f"| {row['method']} | {row['executed']}/{row['planned']} | {row['geometry_success']} | {row['safe_success']} | "
            f"{row['ever_safe']} | {_format(first['median'])}, {first['n']} | {_format(latency['median'])}, {latency['n']} | "
            f"{_format(restricted['mean_time_without_safe_s'])}, {restricted['n']}/{restricted['censored_at_budget_count']} |")
    lines += ["", "T_first chỉ tính lượt đã đạt. Restricted mean tích phân tỷ lệ chưa có nghiệm safe từ 0 tới ngân sách; "
        "lượt hoàn tất mà chưa đạt được giữ dưới dạng kiểm duyệt ở ngân sách, không coi ngân sách là thời điểm tìm thấy nghiệm. "
        "Nếu quan sát bị ngắt/lỗi, restricted mean là null và giữ số lượt thiếu quan sát. Latency gồm dựng field, startup/setup và callback solve, không gồm oracle offline.", "",
        "| Phương pháp | N safe cuối | L_H mean/median (m) | PRI mean/median | Swept slope mean/median (°) |",
        "|---|---:|---:|---:|---:|"]
    for row in report["aggregates"]:
        metrics = [row[key] for key in QUALITY]
        lines.append(f"| {row['method']} | {row['safe_success']} | "+" | ".join(
            _format(metric["mean"])+" / "+_format(metric["median"]) for metric in metrics)+" |")
    lines += ["", "Các metric chất lượng cùng lấy tập final-safe của từng phương pháp; so sánh dùng các cặp cùng map/seed bên dưới. "
        "SEM native giữ objective gốc; SEM common dùng cùng chiều dài và ràng buộc. PRI không phải xác suất tai nạn.", "",
        "| Phương pháp − baseline | Metric | Mean Δ | Median Δ | N/planned | Coverage |",
        "|---|---|---:|---:|---:|---:|"]
    for row in report["paired_comparisons"]:
        lines.append(f"| {row['method']} − {row['baseline']} | {row['metric']} | {_format(row['difference']['mean'])} | "
            f"{_format(row['difference']['median'])} | {row['available_pairs']}/{row['planned_pairs']} | {_format(row['coverage'])} |")
    lines += ["", "Chất lượng ghép cặp yêu cầu cả hai safe cuối; T_first yêu cầu cả hai từng đạt; restricted time giữ mọi cặp đã thực thi. "
        "Số cặp/coverage phải được đọc cùng hiệu số. Không diễn giải solve gần ngân sách như tốc độ hội tụ.", "",
        "Pilot/full: **NOT_RUN**. Input và nguồn analyzer được băm trong provenance.json; dữ liệu đầu vào không được sửa."]
    if report["errors"]:
        lines += ["", "Lỗi giữ nguyên:", ""]+["- "+error for error in report["errors"]]
    return "\n".join(lines)+"\n"


def analyze(input_directory, output_directory):
    source, output = Path(input_directory).resolve(), Path(output_directory).resolve()
    if output == source or source in output.parents:
        raise ValueError("analysis output must be separate from immutable input campaign")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("refusing to overwrite analysis artifacts")
    input_hashes = {name:file_hash(source/name) for name in ("manifest.json", "summary.json", "config.json")}
    manifest, summary, config = (read(source/name) for name in ("manifest.json", "summary.json", "config.json"))
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"),
                                           allow_nan=False).encode()).hexdigest()
    if manifest.get("config_sha256") != config_hash or manifest.get("profile") != "smoke":
        raise ValueError("config provenance mismatch or non-smoke input")
    entries = {entry["trial_id"]:entry for entry in manifest["entries"]}
    for row in summary["rows"]:
        entry = entries[row["trial_id"]]
        directory = source/"trials"/row["trial_id"]
        if Path(row["trial_id"]).name != row["trial_id"] or row["trial_id"] in (".", ".."):
            raise ValueError("unsafe trial id")
        validation = directory/"validation.json"
        if entry.get("validation_sha256"):
            if file_hash(validation) != entry["validation_sha256"] or read(validation) != row:
                raise ValueError("summary/validation provenance mismatch: "+row["trial_id"])
            input_hashes[validation.relative_to(source).as_posix()] = file_hash(validation)
        execution = directory/"execution.json"
        if execution.exists():
            record = read(execution)
            row["_launched"] = record.get("launch_utc") is not None
            input_hashes[execution.relative_to(source).as_posix()] = file_hash(execution)
    report = build_report(summary, manifest, horizon_s=config["planning_budget_s"])
    # No reader may silently accept a concurrent input mutation.
    for name, expected in input_hashes.items():
        if file_hash(source/name) != expected:
            raise ValueError("input changed while analyzing: "+name)
    output.mkdir(parents=True, exist_ok=True)
    provenance = dict(input_directory=str(source), input_sha256=input_hashes,
        analyzer_source_sha256=file_hash(__file__), created_utc=datetime.now(timezone.utc).isoformat(),
        input_mutations=0, planner_executions=0, oracle_executions=0)
    for name, value in (("analysis.json", report), ("provenance.json", provenance)):
        (output/name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")
    _csv(output/"methods.csv", report["aggregates"])
    _csv(output/"paired_comparisons.csv", report["paired_comparisons"])
    _csv(output/"trials.csv", report["trials"])
    (output/"REPORT.md").write_text(markdown(report), encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT/"results/a3t_safe_length_smoke_20261108")
    parser.add_argument("--output", type=Path, default=ROOT/"reports/a3t_safe_length_smoke_20261108")
    args = parser.parse_args(argv)
    report = analyze(args.input, args.output)
    print(json.dumps(dict(status=report["status"], errors=report["errors"], methods=len(report["aggregates"]),
                          paired_contrasts=len(report["paired_comparisons"])), allow_nan=False))
    return 0 if report["status"] == "DONE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
