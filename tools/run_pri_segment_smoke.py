"""Collect only the registered segment-v2 versus full-v1 fixture smoke.

The established execution engine is loaded in a private module namespace with
an explicit protocol/collector binding. Its original module globals and CLI
remain unchanged. Spawn workers use importable, top-level functions. There is
no pilot or full command, no alpha selection, and no retry of a started slot.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import multiprocessing
from pathlib import Path

from corridor_v4_execution import atomic_json as dump, digest, file_hash, read, utc_now
import pri_segment_protocol as protocol
import run_pri_pathsearch as established
from run_sem_exposure import _collect_trial as collect_registered_trial


def _collect_trial(task):
    established._verify_map(task["info"], task["entry"]["phase"])
    return collect_registered_trial(task, request_builder=protocol.request_for)


def _isolated_engine():
    spec = importlib.util.spec_from_file_location("_pri_segment_smoke_engine", established.__file__)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    engine.protocol = protocol
    engine._collect_trial = _collect_trial
    # Keep the spawn target importable by its original stable module name.
    # This verifier receives the real segment request and checks every alpha.
    engine._validate_map = established._validate_map
    return engine


_engine = _isolated_engine()
_engine_audit = _engine._audit


def _phase(phase):
    if phase not in protocol.PHASES:
        raise ValueError("only segment smoke_serial and smoke_parallel are registered")
    return phase


def _audit(directory, require_complete=False):
    manifest = read(Path(directory) / "manifest.json")
    _phase(manifest.get("phase"))
    if manifest.get("protocol_id") != protocol.PROTOCOL_ID:
        raise ValueError("existing output belongs to another protocol; trials cannot be relabelled")
    if manifest.get("sources") != protocol.source_hashes():
        raise ValueError("segment scientific sources changed after preparation")
    expected = protocol.inventory(protocol.load_config(Path(directory) / "config.json"), manifest["phase"])
    if len(manifest.get("entries", [])) != len(expected["entries"]):
        raise ValueError("segment physical inventory count changed")
    return _engine_audit(directory, require_complete=require_complete)


_engine._audit = _audit


def prepare(output, config_path=protocol.CONFIG, phase="smoke_serial", *, dry_run=False):
    _phase(phase)
    workers = 1 if phase == "smoke_serial" else 4
    manifest = _engine.prepare(output, config_path, phase, dry_run=dry_run,
                               workers=workers, validation_workers=workers)
    manifest.pop("native_baseline_objective", None)
    manifest.update(comparator=protocol.BASELINE, evidence_scope="correctness_and_fixture_smoke_only",
        objective="surface_length_with_bounded_whole_path_PRI_selectors", stop_after_smoke=True)
    dump(Path(output) / phase / "manifest.json", manifest)
    return manifest


def run(output, phase, worker, bridge, sem_python=None, *, resume=False):
    _phase(phase)
    # Gate identity before entering the established engine, which owns durable
    # starts, process exit proof, hash checks, resource admission and spools.
    manifest = read(Path(output) / phase / "manifest.json")
    if manifest.get("protocol_id") != protocol.PROTOCOL_ID:
        raise ValueError("refusing to reuse another campaign's physical trials")
    if manifest.get("workers") != (1 if phase == "smoke_serial" else 4):
        raise ValueError("segment serial/parallel lane schedule changed")
    return _engine.run(output, phase, worker, bridge, sem_python, resume=resume)


METRICS = ("final_cost_terrain_m", "final_pri", "final_risk_sum", "final_risk_count", "final_exposure")


def build_report(summary, manifest, config):
    protocol.validate_config(config)
    phase = _phase(manifest["phase"])
    registered = protocol.inventory(config, phase)
    entries = registered["entries"]
    actual_entries = manifest.get("entries", [])
    if (manifest.get("protocol_id") != protocol.PROTOCOL_ID or len(actual_entries) != len(entries)
            or any(not protocol.entry_matches(a,b) for a,b in zip(actual_entries,entries,strict=True))):
        raise ValueError("smoke report physical schedule differs")
    rows = summary.get("rows", [])
    by_id = {row["trial_id"]:row for row in rows}
    actual_by_id = {entry["trial_id"]:entry for entry in actual_entries}
    expected_ids = {e["trial_id"] for e in entries}
    if len(by_id) != len(rows) or set(by_id)-expected_ids:
        raise ValueError("smoke verdict duplicates or unknown physical IDs")
    selectors, physical = [], []
    for entry in entries:
        row = by_id.get(entry["trial_id"])
        physical.append(dict(trial_id=entry["trial_id"], method=entry["method"], family=entry["family"],
            map_id=entry["map_id"], planner_seed=entry["planner_seed"],
            execution_status=row.get("execution_status") if row else actual_by_id[entry["trial_id"]]["status"],
            success=bool(row and row.get("success")), validated=row is not None,
            correctness_errors=row.get("correctness_errors", []) if row else [],
            search_counters=row.get("search_counters", {}) if row else {}))
        outputs = row.get("selector_results", []) if row else []
        if outputs and (len(outputs) != 4 or [s.get("alpha") for s in outputs] != list(protocol.ALPHAS)):
            raise ValueError("physical run has a changed four-selector inventory")
        if row and row.get("success") and (len(outputs)!=4 or not all(s.get("success") for s in outputs)):
            raise ValueError("physical success lacks four successful independent selectors")
        for alpha in protocol.ALPHAS:
            selected = next((s for s in outputs if s["alpha"] == alpha), None)
            success = bool(row and not row.get("correctness_errors") and selected and selected.get("success"))
            if success and any(type(selected.get(k)) not in (int,float) or not math.isfinite(selected[k])
                               or selected[k] < 0 for k in METRICS):
                raise ValueError("successful selector lacks independent risk/length diagnostics")
            selectors.append(dict(physical_trial_id=entry["trial_id"], method=entry["method"],
                family=entry["family"], map_id=entry["map_id"], planner_seed=entry["planner_seed"],
                alpha=alpha, success=success, selected_snapshot_id=selected.get("selected_snapshot_id") if selected else None,
                **{k:selected.get(k) if success else None for k in METRICS}))
    indexed = {(s["map_id"],s["planner_seed"],s["alpha"],s["method"]):s for s in selectors}
    comparisons = []
    for selected in selectors:
        if selected["method"] != protocol.SEGMENT:
            continue
        base = indexed[(selected["map_id"],selected["planner_seed"],selected["alpha"],protocol.BASELINE)]
        safe = selected["success"] and base["success"]
        comparisons.append(dict(map_id=selected["map_id"], family=selected["family"],
            planner_seed=selected["planner_seed"], alpha=selected["alpha"], paired_final_safe=safe,
            segment_success=selected["success"], baseline_success=base["success"],
            **{"delta_"+k:selected[k]-base[k] if safe else None for k in METRICS}))
    complete = len(rows)==len(entries) and manifest.get("status")=="COMPLETE"
    errors = sum(bool(r.get("correctness_errors")) for r in rows)
    return dict(protocol_id=protocol.PROTOCOL_ID, schema_version=1, phase=phase,
        status="FAILED_CORRECTNESS" if errors else "DONE" if complete else "INCOMPLETE",
        complete=complete, planned=len(entries), attempted=summary["attempted"], validated=len(rows),
        success=sum(r["success"] for r in physical), correctness_failures=errors,
        physical_trials=physical, selector_trials=selectors, paired_comparisons=comparisons,
        actual_planner_overlap=summary.get("actual_planner_overlap"),
        config_sha256=digest(config), sources_sha256=digest(manifest["sources"]),
        runtime_sha256=manifest.get("runtime_sha256"), selected_alpha=None, alpha_selection=False,
        selector_unit="four_correlated_outputs_per_physical_run", held_out_evidence=False,
        synthetic_fixture=bool(manifest.get("synthetic_fixture") or any(r.get("synthetic_fixture") for r in rows)),
        smoke_cannot_establish_superiority=True, stop_after_smoke=True,
        created_utc=utc_now())


def analyze(output, phase):
    directory = Path(output) / _phase(phase)
    manifest, summary, config = _audit(directory)
    dump(directory / "summary.json", summary)
    report = build_report(summary, manifest, config)
    report["manifest_sha256"] = file_hash(directory / "manifest.json")
    report["report_sha256"] = digest(report)
    destination = directory / "reports"
    destination.mkdir(exist_ok=True)
    dump(destination / "report.json", report)
    lines = ["# Smoke segment-v2 so với full-v1", "",
        f"{report['status']}: {report['validated']}/{report['planned']} verdict, {report['success']} backend thành công, "
        f"{report['correctness_failures']} lỗi correctness.", "",
        "Bốn alpha là đầu ra tương quan của cùng một lượt. Đây chỉ là kiểm tra correctness/fixture; "
        "không chọn alpha, không kết luận ưu thế, không mở pilot.", "",
        "| Fixture | Backend | Alpha | Thành công | L (m) | PRI | Risk sum | Số ô duy nhất | Exposure |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|"]
    fmt = lambda v: "—" if v is None else f"{v:.8g}"
    for row in report["selector_trials"]:
        lines.append("| "+" | ".join([row["family"],row["method"],str(row["alpha"]),str(row["success"])]
            +[fmt(row[k]) for k in METRICS])+" |")
    (destination / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inventory","prepare","run","smoke","analyze"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", default=protocol.CONFIG, type=Path)
    parser.add_argument("--phase", choices=protocol.PHASES, default="smoke_serial")
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--bridge", type=Path)
    parser.add_argument("--sem-python", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    phases = protocol.PHASES if args.command == "smoke" else (args.phase,)
    execute = args.command in ("run","smoke") and not args.dry_run
    if execute and (not args.worker or not args.bridge):
        parser.error("actual smoke requires --worker and --bridge")
    result = None
    for phase in phases:
        path = args.output / phase / "manifest.json"
        if args.command == "analyze":
            result = analyze(args.output,phase)
            continue
        if args.command in ("inventory","prepare") or args.dry_run or not path.exists():
            result = prepare(args.output,args.config,phase,dry_run=args.dry_run or args.command=="inventory")
        if execute:
            if args.resume and read(path).get("status") == "COMPLETE":
                _,result,_ = _audit(args.output/phase,require_complete=True)
            else:
                result = run(args.output,phase,args.worker,args.bridge,args.sem_python,resume=args.resume)
            if result["status"] != "COMPLETE":
                break
            result = analyze(args.output,phase)
    print(json.dumps({k:result[k] for k in ("status","planned_trials","planned","attempted","validated",
        "success","correctness_failures","selected_alpha") if k in result},ensure_ascii=False))
    return 0 if result["status"] in ("PREPARED","PREPARED_INVENTORY","DONE","COMPLETE") else 1


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
