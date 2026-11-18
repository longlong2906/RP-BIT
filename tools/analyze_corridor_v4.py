"""Analyze the registered v4 campaign and bind any selection to raw evidence."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from analyze_corridor_v3 import QUALITY, _rows, _write_csv, aggregate, finite, paired, read, stats
from corridor_v4_protocol import (ALPHAS, COMMON, NATIVE, PRIMARY, PROTOCOL_ID,
                                  inventory, load_config, validate_config)
from run_safe_length import digest
from safe_length_field import file_hash

TIMING = ("queue_wait_s", "validation_s", "peak_rss_bytes")
DIAGNOSTICS = ("cell_delta_diagnostics", "preview_s", "final_risk_quantiles")
SLOT_KEYS = ("trial_id", "map_id", "query_id", "family", "map_seed", "planner_seed",
             "method", "alpha", "policy", "block_index")


def _selection(report):
    result = dict(status="NOT_APPLICABLE", selected_alpha=None, selected_method=None, candidates=[])
    if report["profile"] != "pilot":
        return result
    if report["status"] != "DONE":
        return dict(result, status="BLOCKED", reason="incomplete_or_invalid_campaign")
    for row in report["paired_comparisons"]:
        if (row["scope"] != "all" or row["baseline"] != NATIVE
                or not row["method"].startswith(PRIMARY + "_a")):
            continue
        metrics = row["metrics"]
        psr, length, pri = (metrics[key]["difference"]["mean"] for key in ("psr", *QUALITY[:2]))
        checks = dict(psr=finite(psr) and psr >= 0, length=finite(length) and length < 0,
                      pri=finite(pri) and pri < 0)
        result["candidates"].append(dict(method=row["method"], alpha=row["alpha"],
            checks=checks, passes=all(checks.values()), safe_pairs=row["safe_pairs"],
            planned_pairs=row["planned_pairs"], paired_length_mean=metrics[QUALITY[0]]["left_mean"],
            paired_pri_mean=metrics[QUALITY[1]]["left_mean"]))
    if sorted(row["alpha"] for row in result["candidates"]) != list(ALPHAS):
        raise ValueError("pilot contrasts must cover all registered primary alphas")
    eligible = [row for row in result["candidates"] if row["passes"]]
    if not eligible:
        return dict(result, status="NO_CANDIDATE")
    winner = min(eligible, key=lambda row: (row["paired_length_mean"], row["paired_pri_mean"], row["alpha"]))
    return dict(result, status="SELECTED", selected_alpha=winner["alpha"], selected_method=winner["method"])


def _held_out(report):
    if report["profile"] != "full":
        return dict(status="NOT_APPLICABLE", checks={})
    if report["status"] != "DONE":
        return dict(status="BLOCKED", checks={})
    contrast = next(row for row in report["paired_comparisons"]
                    if row["scope"] == "all" and row["method"] == PRIMARY and row["baseline"] == NATIVE)
    checks = {}
    for key in ("psr", *QUALITY[:2]):
        ci = contrast["metrics"][key]["ci95"] or {}
        endpoint = ci.get("lower" if key == "psr" else "upper")
        checks[key] = finite(endpoint) and (endpoint >= 0 if key == "psr" else endpoint < 0)
    return dict(status="PASS" if all(checks.values()) else "NOT_MET", checks=checks)


def build_report(summary, manifest, config):
    validate_config(config)
    profile = manifest["profile"]
    if (manifest.get("protocol_id") != PROTOCOL_ID or summary.get("protocol_id") != PROTOCOL_ID
            or summary.get("profile") != profile or digest(config) != manifest.get("config_sha256")):
        raise ValueError("campaign protocol/profile/config mismatch")
    if summary.get("synthetic_fixture") or manifest.get("synthetic_fixture"):
        raise ValueError("fixture records are not campaign evidence")
    alpha = manifest.get("selected_alpha") if profile == "full" else None
    if profile == "full" and (alpha not in ALPHAS or not manifest.get("selection_sha256")):
        raise ValueError("full manifest lacks a selected pilot")
    expected = inventory(config, profile, alpha)["entries"]
    identity = lambda entries: [tuple(row.get(key) for key in SLOT_KEYS) for row in entries]
    if identity(manifest["entries"]) != identity(expected):
        raise ValueError("campaign schedule differs from registration")

    # A started slot may still await an independent verdict after an interruption.
    # Keep it distinct from both an observed outcome and an unstarted slot.
    observed_count = len(summary["rows"])
    if summary.get("validated", observed_count) != observed_count:
        raise ValueError("validation count differs from recorded verdicts")
    rows, errors, complete = _rows(dict(summary, attempted=observed_count), manifest)
    observed = {row["trial_id"]: row for row in summary["rows"]}
    budget = config["profiles"][profile]["planning_budget_s"]
    for row, entry in zip(rows, manifest["entries"], strict=True):
        raw = observed.get(row["trial_id"], {})
        row.update(query_id=entry["query_id"], policy=entry["policy"], block_index=entry["block_index"])
        row.update({key: raw.get(key) for key in TIMING + DIAGNOSTICS})
        if not raw and entry["status"] != "NOT_RUN":
            row.update(attempted=True, execution_status="PENDING_VALIDATION")
        corridor = raw.get("corridor") or {}
        row.update(raw_guide_stats=corridor.get("raw_guide_stats"),
                   selected_guide_stats=corridor.get("guide_stats"),
                   beam_scheduler=corridor.get("beam_scheduler"),
                   **corridor.get("beam_counters", {}))
        if row["attained"] and row["first_safe_s"] > budget:
            errors.append(row["trial_id"] + ": attainment after the solve budget")
    tables, contrasts = [], []
    methods = sorted({row["method"] for row in rows})
    for scope in ("all", "family", "map"):
        for group in (["all"] if scope == "all" else sorted({row["map_id" if scope == "map" else scope] for row in rows})):
            key = "map_id" if scope == "map" else scope
            subset = [row for row in rows if scope == "all" or row[key] == group]
            for method in methods:
                selected = [row for row in subset if row["method"] == method]
                tables.append(dict(scope=scope, group=group, method=method,
                    **aggregate(selected, budget), **{k: stats(row[k] for row in selected) for k in TIMING}))
                if scope != "map" and method != NATIVE:
                    repeats = config["analysis"]["bootstrap_replicates"] if profile == "full" and method == PRIMARY else 0
                    contrasts.append(dict(scope=scope, group=group, alpha=selected[0]["alpha"],
                        **paired(subset, method, replicates=repeats, seed=config["analysis"]["bootstrap_seed"])))
                if scope != "map" and (method == PRIMARY or method.startswith(PRIMARY + "_a")):
                    control = COMMON + method[len(PRIMARY):]
                    contrasts.append(dict(scope=scope, group=group, alpha=selected[0]["alpha"],
                        **paired(subset, method, baseline=control)))
    executed = sum(row["executed"] for row in rows)
    report = dict(protocol_id=PROTOCOL_ID, schema_version=4, profile=profile,
        status="FAILED_CORRECTNESS" if errors else "DONE" if complete and executed == len(rows) else "INCOMPLETE",
        complete=complete, errors=errors, budget_s=budget, planned=len(rows),
        attempted=sum(row["attempted"] for row in rows), executed=executed,
        success=sum(row["safe"] for row in rows), tables=tables, paired_comparisons=contrasts,
        trials=rows, execution=config["execution"],
        sessions=manifest.get("sessions", []), waves=manifest.get("waves", []),
        campaign_timing={key: summary.get(key) for key in ("collection_s", "validation_s",
            "session_wall_s", "max_concurrency")})
    report["pilot_selection"] = _selection(report)
    report["held_out_gate"] = _held_out(report)
    return report


def markdown(report):
    def fmt(value):
        return "—" if value is None else f"{value:.6g}"

    lines = ["# A3-T corridor v4", "",
        f"Profile {report['profile']} · {report['status']} · {report['success']}/{report['executed']} nghiệm exact hợp lệ",
        f"Lượt có verdict {report['attempted']}/{report['planned']}.",
        f"Cấu hình {report['execution']['max_workers']} worker trial và "
        f"{report['execution']['validation_workers']} worker kiểm chứng.",
        "", "| Nhóm | Phương pháp | Success/executed/planned | PSR | L_H (m) | PRI | Preprocess (s) | Solve (s) | Latency (s) | Queue (s) | Oracle (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["tables"]:
        if row["scope"] != "map":
            values = [row["psr"]] + [row[key]["mean"] for key in (*QUALITY[:2], "preprocessing_s",
                "solve_time_s", "first_safe_latency_s", "queue_wait_s", "validation_s")]
            lines.append(f"| {row['group']} | {row['method']} | {row['success']}/{row['executed']}/{row['planned']} | "
                         + " | ".join(map(fmt, values)) + " |")
    lines += ["", "| Nhóm | So sánh | Cặp hợp lệ / dự kiến | ΔL_H (m) | ΔPRI | ΔPSR |",
              "|---|---|---:|---:|---:|---:|"]
    for row in report["paired_comparisons"]:
        values = [row["metrics"][key]["difference"]["mean"] for key in (*QUALITY[:2], "psr")]
        lines.append(f"| {row['group']} | {row['method']} − {row['baseline']} | "
                     f"{row['safe_pairs']}/{row['planned_pairs']} | " + " | ".join(map(fmt, values)) + " |")
    if report["profile"] == "smoke":
        lines += ["", "Smoke dùng để kiểm chứng pipeline. Pilot và full ở trạng thái NOT_RUN."]
    elif report["profile"] == "pilot":
        lines += ["", f"Kết quả chọn pilot **{report['pilot_selection']['status']}**."]
    else:
        lines += ["", f"Gate held-out **{report['held_out_gate']['status']}**."]
    lines.extend(["", "PRI và L_H của phép so sánh ghép cặp chỉ dùng các cặp có hai nghiệm cuối hợp lệ.",
                  "PSR giữ các lượt planner thất bại đã thực thi. Báo cáo dùng thời gian đo dưới tải của campaign."])
    if report["errors"]:
        lines.extend(["", *["- " + error for error in report["errors"]]])
    timing = report["campaign_timing"]
    lines.extend(["", f"Tổng thời gian session {fmt(timing['session_wall_s'])} s. "
        f"Thu raw {fmt(timing['collection_s'])} s, kiểm chứng {fmt(timing['validation_s'])} s."])
    if "planner_overlap" in report:
        lines.append(f"Số tiến trình planner chồng thời gian cao nhất theo receipt: "
                     f"{report['planner_overlap']['peak']}.")
    return "\n".join(lines) + "\n"


def analyze(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or (output.exists() and any(output.iterdir())):
        raise ValueError("analysis requires a separate unused directory")
    manifest, summary = (read(source / name) for name in ("manifest.json", "summary.json"))
    config = load_config(source / "config.json")
    if manifest["profile"] == "full":
        selection = source / "pilot_selection.json"
        if file_hash(selection) != manifest["selection_sha256"]:
            raise ValueError("full selection artifact hash mismatch")
    entries = {entry["trial_id"]: entry for entry in manifest["entries"]}
    intervals = []
    for row in summary["rows"]:
        trial_id = row["trial_id"]
        if trial_id not in entries or Path(trial_id).name != trial_id:
            raise ValueError("unplanned or invalid trial identity")
        directory = source / "trials" / trial_id
        validation = directory / "validation.json"
        if file_hash(validation) != entries[trial_id]["validation_sha256"] or read(validation) != row:
            raise ValueError(f"independent verdict changed for {trial_id}")
        raw = read(directory / "raw_inventory.json")
        if set(raw) != {path.relative_to(directory).as_posix() for path in directory.rglob("*")
                       if path.is_file() and path.name not in ("validation.json", "raw_inventory.json")}:
            raise ValueError(f"raw inventory membership differs for {trial_id}")
        for name, expected in raw.items():
            path = (directory / name).resolve()
            if directory not in path.parents or file_hash(path) != expected:
                raise ValueError(f"raw artifact hash mismatch for {trial_id}/{name}")
        if "execution.json" in raw and "process_exit.json" in raw:
            execution = read(directory / "execution.json")
            finished = read(directory / "process_exit.json")
            if execution.get("launched") is True and finished.get("pid") == execution.get("pid"):
                begin = datetime.fromisoformat(execution["launch_utc"])
                end = datetime.fromisoformat(finished["finished_utc"])
                if end < begin:
                    raise ValueError(f"process exit precedes launch for {trial_id}")
                if end > begin:
                    intervals.extend([(begin, 1), (end, -1)])
    report = build_report(summary, manifest, config)
    peak = active = 0
    for _, change in sorted(intervals):
        active += change
        peak = max(peak, active)
    report["planner_overlap"] = dict(peak=peak, recorded_intervals=len(intervals)//2,
                                     origin="subprocess launch and OS exit receipts")
    files = ("manifest.json", "summary.json", "config.json")
    hashes = {name: file_hash(source / name) for name in files}
    provenance = dict(created_utc=datetime.now(timezone.utc).isoformat(), input_directory=str(source),
        input_sha256=hashes, analyzer_sha256=file_hash(__file__),
        sources_sha256=digest(manifest["sources"]), binary_sha256=manifest["binary_sha256"])
    report["provenance"] = provenance
    output.mkdir(parents=True, exist_ok=True)

    def write_json(name, value):
        (output / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                                   encoding="utf-8")

    write_json("summary.json", report)
    write_json("provenance.json", provenance)
    for name, key in (("aggregate.csv", "tables"), ("paired_comparisons.csv", "paired_comparisons"),
                      ("trials.csv", "trials")):
        _write_csv(output / name, report[key])
    (output / "REPORT.md").write_text(markdown(report), encoding="utf-8")
    if report["profile"] == "pilot":
        write_json("selection.json", dict(report["pilot_selection"], protocol_id=PROTOCOL_ID,
            config_sha256=digest(config), sources_sha256=provenance["sources_sha256"],
            binary_sha256=provenance["binary_sha256"], analyzer_sha256=provenance["analyzer_sha256"],
            pilot_manifest_path=str(source / "manifest.json"), pilot_manifest_sha256=hashes["manifest.json"],
            pilot_summary_path=str(source / "summary.json"), pilot_summary_sha256=hashes["summary.json"]))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = analyze(args.input, args.output)
    print(json.dumps({key: report[key] for key in ("status", "profile", "planned", "attempted", "executed", "success")}))
    return 0 if report["status"] == "DONE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
