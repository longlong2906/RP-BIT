"""Descriptive, hash-audited SEM-exposure smoke analysis; never select a weight."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_corridor_v3 import QUALITY, _rows, _write_csv, aggregate, paired, stats
from corridor_v4_execution import atomic_json as dump, digest, file_hash, read, utc_now
from run_corridor_v4 import _verify_inputs, actual_planner_overlap, raw_inventory
from sem_exposure_protocol import (CORRIDOR, METHODS, NATIVE, NOMINAL, PROTOCOL_ID,
                                    inventory, load_config, validate_config)

EXTRA = ("final_exposure", "final_sem_exposure_normalized_m", "final_objective_value",
         "final_risk_sum", "final_risk_count", "queue_wait_s", "validation_s", "peak_rss_bytes")


def build_report(summary, manifest, config):
    validate_config(config)
    if any(value.get("protocol_id") != PROTOCOL_ID or value.get("profile") != "smoke"
           for value in (summary, manifest)):
        raise ValueError("not a registered SEM-exposure smoke")
    if manifest.get("config_sha256") != digest(config):
        raise ValueError("frozen configuration differs")
    expected = inventory(config)["entries"]
    for actual, registered in zip(manifest["entries"], expected, strict=True):
        if any(actual.get(key) != value for key, value in registered.items() if key != "status"):
            raise ValueError("frozen schedule differs")
    observed = {row["trial_id"]: row for row in summary["rows"]}
    if summary.get("validated") != len(observed):
        raise ValueError("validated count mismatch")
    # The shared normalizer's attempted count means observed verdicts. This
    # runner separately retains all starts, including raw awaiting validation.
    rows, errors, complete = _rows(dict(summary, attempted=len(summary["rows"])), manifest)
    started = sum(entry["status"] != "NOT_RUN" for entry in manifest["entries"])
    if summary.get("attempted") != started:
        errors.append("started-attempt count differs from manifest")
    for row, entry in zip(rows, manifest["entries"], strict=True):
        source = observed.get(row["trial_id"], {})
        if source and source.get("sem_lambda") != entry["sem_lambda"]:
            raise ValueError("verdict weight differs from frozen registration")
        row.update(sem_lambda=entry["sem_lambda"], nominal=row["method"] == NOMINAL,
                   **{key: source.get(key) for key in EXTRA})
        if not source and entry["status"] != "NOT_RUN":
            row.update(attempted=True, execution_status="AWAITING_VERDICT", raw_status=entry["status"])
    tables, contrasts = [], []
    for scope, group, subset in [("all", "all", rows)] + [
            ("family", family, [row for row in rows if row["family"] == family])
            for family in config["profiles"]["smoke"]["families"]]:
        for method in METHODS:
            selected = [row for row in subset if row["method"] == method]
            table = dict(scope=scope, group=group, method=method, **aggregate(selected, 2.))
            table.update(**{key: stats(row.get(key) for row in selected if row["safe"]) for key in EXTRA})
            tables.append(table)
        for method in METHODS[:-1]:
            contrasts.append(dict(scope=scope, group=group, **paired(subset, method, baseline=NATIVE)))
        for baseline in (CORRIDOR, METHODS[0]):
            contrasts.append(dict(scope=scope, group=group, **paired(subset, NOMINAL, baseline=baseline)))
    return dict(protocol_id=PROTOCOL_ID, schema_version=1, profile="smoke",
        status="FAILED_CORRECTNESS" if errors else "DONE" if complete else "INCOMPLETE",
        errors=errors, planned=len(rows), attempted=started, validated=len(observed),
        executed=sum(row["executed"] for row in rows), success=sum(row["safe"] for row in rows),
        tables=tables, paired_comparisons=contrasts, trials=rows, nominal_method=NOMINAL,
        selection=dict(status="NOT_RUN_SMOKE_ONLY", selected_lambda=None),
        pilot_status="NOT_RUN", full_status="NOT_RUN", no_superiority_claim=True,
        campaign_timing={key: summary.get(key) for key in ("collection_s", "validation_s", "session_wall_s")},
        planner_overlap=actual_planner_overlap(manifest["entries"]), sessions=manifest.get("sessions", []))


def markdown(report):
    def fmt(value):
        return "—" if value is None else f"{value:.6g}"
    lines = ["# A3-T — chiều dài và SEM exposure", "",
        f"Smoke **{report['status']}**: {report['success']}/{report['executed']} nghiệm exact hợp lệ; "
        f"{report['attempted']}/{report['planned']} lượt bắt đầu, {report['validated']} verdict.", "",
        "Mục tiêu mới J=L_H+λE/40, với E=∫F_R ds_H. E là exposure cộng được; PRI vẫn là "
        "trung bình risk trên tập ô đóng duy nhất đường tâm chạm. λ=1 được đăng ký trước là cấu hình danh nghĩa.", "",
        "| Phương pháp | Success/executed/planned | L_H (m) | PRI | E (risk·m) | E/40 (m) | Preprocess (s) | Solve (s) | Latency first-safe (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["tables"]:
        if row["scope"] == "all":
            keys = ("final_cost_terrain_m", "final_pri", "final_exposure", "final_sem_exposure_normalized_m",
                    "preprocessing_s", "solve_time_s", "first_safe_latency_s")
            lines.append(f"| {row['method']} | {row['success']}/{row['executed']}/{row['planned']} | " +
                         " | ".join(fmt(row[key]["mean"]) for key in keys) + " |")
    lines += ["", "| So sánh | Cặp cùng hợp lệ/dự kiến | ΔL_H (m) | ΔPRI | ΔPSR |",
              "|---|---:|---:|---:|---:|"]
    for row in report["paired_comparisons"]:
        if row["scope"] == "all":
            lines.append(f"| {row['method']} − {row['baseline']} | {row['safe_pairs']}/{row['planned_pairs']} | " +
                " | ".join(fmt(row["metrics"][key]["difference"]["mean"]) for key in (*QUALITY[:2], "psr")) + " |")
    lines += ["", "Quality ghép cặp chỉ dùng hai đường cuối hợp lệ. Các failure đã thực thi nằm trong PSR; "
        "lượt đã bắt đầu nhưng chưa có verdict được ghi riêng. Không cứu đường cuối bằng snapshot cũ.", "",
        "SEM native là bản tái dựng paper đã đăng ký, giữ objective length/turn/risk. "
        "Không so tổng objective khác nghĩa giữa các phương pháp.", "",
        "Smoke chỉ kiểm correctness và mô tả trade-off, không chọn λ hoặc chứng minh vượt SEM ở cả PRI và chiều dài. "
        "Pilot/full: **NOT_RUN**.", "",
        f"Planner chồng thời gian tối đa theo receipt: {report['planner_overlap']['max_concurrency']}. "
        "Toàn bộ collection kết thúc trước independent validation."]
    if report["errors"]:
        lines += ["", *["- " + error for error in report["errors"]]]
    return "\n".join(lines) + "\n"


def analyze(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or (output.exists() and any(output.iterdir())):
        raise ValueError("analysis requires a separate unused directory")
    config = load_config(source / "config.json")
    manifest, summary = read(source / "manifest.json"), read(source / "summary.json")
    for info in manifest["maps"]:
        _verify_inputs(info)
    entries = {entry["trial_id"]: entry for entry in manifest["entries"]}
    for row in summary["rows"]:
        trial_id = row["trial_id"]
        if trial_id not in entries or Path(trial_id).name != trial_id:
            raise ValueError("unregistered trial identity")
        directory = source / "trials" / trial_id
        if (file_hash(directory / "validation.json") != entries[trial_id]["validation_sha256"] or
                read(directory / "validation.json") != row):
            raise ValueError("independent verdict changed: " + trial_id)
        raw = read(directory / "raw_inventory.json")
        if raw_inventory(directory) != raw or digest(raw) != entries[trial_id]["raw_inventory_sha256"]:
            raise ValueError("raw artifacts changed: " + trial_id)
        for name in raw:
            if directory not in (directory / name).resolve().parents:
                raise ValueError("unsafe raw artifact path")
    report = build_report(summary, manifest, config)
    report["provenance"] = dict(created_utc=utc_now(), input_directory=str(source),
        input_sha256={name: file_hash(source / name) for name in ("manifest.json", "summary.json", "config.json")},
        analyzer_sha256=file_hash(__file__), sources_sha256=digest(manifest["sources"]),
        binary_sha256=manifest.get("binary_sha256"))
    output.mkdir(parents=True, exist_ok=True)
    dump(output / "summary.json", report)
    dump(output / "provenance.json", report["provenance"])
    for name, key in (("aggregate.csv", "tables"), ("paired_comparisons.csv", "paired_comparisons"), ("trials.csv", "trials")):
        _write_csv(output / name, report[key])
    (output / "REPORT.md").write_text(markdown(report), encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = analyze(args.input, args.output)
    print(json.dumps({key: report[key] for key in ("status", "planned", "attempted", "success")}))
    return 0 if report["status"] == "DONE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
