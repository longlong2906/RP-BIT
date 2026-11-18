"""G5 held-out protocol: immutable plan, one attempt, strict provenance gates."""
import argparse
import contextlib
import datetime as dt
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from experiment_common import ROOT, code_hash, digest, dump, environment, file_hash, load_config, read_json, worker_env
from run_experiments import archive_source, plan, provenance_identity, request_for, run_trial, summarize

OUT = ROOT / "results/full_g5_v1"
LOG = ROOT / "logs/g5"
CONFIG = ROOT / "configs/g5_full_v1.yaml"
DATA = ROOT / "data/full_g5_v1"
BUILD = ROOT / "build-g5"
FAMILIES = ["traversable_hill", "ridge_pass", "rolling_terrain", "narrow_passage", "mixed_obstacles"]
PLANNERS = ["bitstar_a0", "bitstar_a3", "informed_rrtstar", "rrtstar"]


def default_worker():
    return BUILD / ("terrain_benchmark.exe" if os.name == "nt" else "terrain_benchmark")


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def prepare_protocol():
    import yaml
    audit = read_json(LOG / "pilot_tail_audit.json")
    if CONFIG.exists():
        raise ValueError("G5 config already prepared")
    config = load_config(ROOT / "configs/experiments.yaml", "pilot")
    config["profiles"]["full"].update(maps_per_family=8, map_seeds=list(range(1001, 1009)),
        planner_seeds=list(range(1, 11)), planners=PLANNERS, planning_budget_s=audit["chosen_full_budget_s"])
    CONFIG.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    protocol = ROOT / "docs/G5_PROTOCOL.md"
    text = protocol.read_text(encoding="utf-8")
    marker = "**Budget cuối và số liệu audit phải được điền trước generate/freeze.**"
    if marker not in text:
        raise ValueError("protocol budget marker missing")
    lines = [f"**Budget full đã chọn: {audit['chosen_full_budget_s']:g} giây/trial.** "
             "Audit 400 trial, 20 nhóm family/planner: không thiếu gain; 16/20 nhóm có late-improvement fraction >0,25. "
             "Median và p90 gain đều nhỏ hơn ngưỡng; điều kiện cải thiện muộn kích hoạt mức 10 s. "
             "Quy tắc áp dụng đúng như đăng ký, không đổi ngưỡng sau audit. Tổng upper solve budget 64.000 s ≈17,78 h; "
             "không phải dự báo tổng wall time.", "",
             "| Family | Planner | Median tail gain | IQR | p90 | Strict late fraction |",
             "|---|---|---:|---:|---:|---:|"]
    for group in audit["family_planner"]:
        lines.append(f"| {group['family']} | {group['planner_id']} | {group['median']:.8g} | {group['iqr']:.8g} | {group['p90']:.8g} | {group['strict_late_improvement_fraction']:.2f} |")
    lines += ["", "Gain là tỷ lệ, không phải phần trăm. Chi tiết từng query và map/query/planner cùng q25/q75 nằm trong "
              "[pilot_tail_audit.json](../logs/g5/pilot_tail_audit.json). Đối chiếu toàn bộ 40 mốc median4s/5s của anytime aggregate đạt."]
    protocol.write_text(text.replace(marker, "\n".join(lines)), encoding="utf-8")
    print("prepared G5 budget", audit["chosen_full_budget_s"], "s", flush=True)


def command_index():
    records = []
    for path in sorted(LOG.glob("*.json")):
        if path.name == "command_index.json":
            continue
        record = read_json(path)
        if isinstance(record, dict) and "command" in record and "exit_code" in record:
            records.append(dict(record, record_path=str(path.relative_to(ROOT))))
    dump(LOG / "command_index.json", {"updated_utc": utc(), "commands": records})


def recorded(name, command):
    LOG.mkdir(parents=True, exist_ok=True)
    stem = LOG / name
    if stem.with_suffix(".json").exists():
        raise ValueError("command record already exists: " + name)
    started = time.perf_counter()
    record = {"command": list(map(str, command)), "cwd": str(ROOT), "utc": utc()}
    with stem.with_suffix(".stdout.log").open("wb") as out, stem.with_suffix(".stderr.log").open("wb") as err:
        try:
            result = subprocess.run(command, cwd=ROOT, env=worker_env(), stdout=out, stderr=err)
            record["exit_code"] = result.returncode
        except OSError as exc:
            err.write(str(exc).encode())
            record["exit_code"] = 127
    record["wall_s"] = time.perf_counter() - started
    dump(stem.with_suffix(".json"), record)
    command_index()
    print(name, "exit", record["exit_code"], flush=True)
    if record["exit_code"]:
        raise RuntimeError("STOP_GATE command failed: " + name)
    return record


def preservation(check=False):
    path = LOG / "preservation_before.json"
    if check:
        before = read_json(path)
        changed = [p for p, h in before["files"].items() if not (ROOT / p).exists() or file_hash(ROOT / p) != h]
        result = {"checked": len(before["files"]), "changed": changed, "utc": utc(), "valid": not changed}
        dump(LOG / "preservation_after.json", result)
        if changed:
            raise RuntimeError("STOP_GATE old artifacts changed")
        return result
    if path.exists():
        raise ValueError("preservation snapshot already exists")
    folders = ["results/smoke", "results/diagnostic_g3g4_v1", "results/pilot_g3g4_v1", "reports/smoke", "reports/diagnostic_g3g4_v1", "reports/pilot_g3g4_v1", "data/smoke", "data/pilot", "data/diagnostic_g3g4_v1"]
    paths = [p for folder in folders for p in (ROOT / folder).rglob("*") if p.is_file()]
    paths += [ROOT / "configs/experiments.yaml", ROOT / "configs/dependencies.lock.json"]
    dump(path, {"utc": utc(), "files": {str(p.relative_to(ROOT)): file_hash(p) for p in paths}})
    print("preserved hashes", len(paths), flush=True)


def full_config():
    config = load_config(CONFIG, "pilot")
    profile = config["profiles"]["full"]
    expected = {"families": FAMILIES, "maps_per_family": 8, "queries_per_map": 4,
                "map_seeds": list(range(1001, 1009)), "planner_seeds": list(range(1, 11)),
                "planners": PLANNERS, "map_extent_m": [40., 40.], "grid_spacing_m": .1,
                "purpose": "held_out_evaluation"}
    if any(profile[k] != v for k, v in expected.items()):
        raise ValueError("G5 config differs from registered design")
    audit = read_json(LOG / "pilot_tail_audit.json")
    if profile["planning_budget_s"] != audit["chosen_full_budget_s"]:
        raise ValueError("budget differs from pilot rule")
    return config


def correctness_gate():
    # No full run is permitted by the old pipeline gate alone.
    suffix = "" if not (LOG / "configure_release_g5.json").exists() else "_" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S")
    configure = recorded("configure_release_g5" + suffix, ["cmake", "-S", ".", "-B", str(BUILD), "-G", "Ninja",
        "-DCMAKE_CXX_COMPILER=C:/mingw64/bin/g++.exe", "-DCMAKE_BUILD_TYPE=Release",
        "-DPython3_EXECUTABLE="+sys.executable,
        "-DCMAKE_PREFIX_PATH="+str(ROOT / ".deps/install"), "-DBOOST_ROOT="+str(ROOT / ".deps/boost_1_82_0"),
        "-DBOOST_LIBRARYDIR="+str(ROOT / ".deps/boost_1_82_0/stage/lib"), "-DBoost_NO_SYSTEM_PATHS=ON",
        "-DBoost_NO_BOOST_CMAKE=ON", "-DBoost_USE_STATIC_LIBS=ON"])
    build = recorded("build_release_g5" + suffix, ["cmake", "--build", str(BUILD), "--parallel", "2"])
    test = recorded("ctest_release_g5" + suffix, ["ctest", "--test-dir", str(BUILD), "--output-on-failure"])
    dump(LOG / "correctness_gate.json", {"status": "PASS", "utc": utc(), "code_sha256": code_hash(),
         "worker_sha256": file_hash(default_worker()), "configure": configure, "build": build, "ctest": test})


def gate_matches():
    gate = read_json(LOG / "correctness_gate.json")
    if gate["status"] != "PASS" or gate["code_sha256"] != code_hash() or gate["worker_sha256"] != file_hash(default_worker()):
        raise RuntimeError("STOP_GATE correctness gate missing/stale")
    audit = read_json(LOG / "dependency_audit.json")
    if audit.get("status") != "PASS" or audit.get("code_sha256") != code_hash():
        raise RuntimeError("STOP_GATE dependency patch/link audit not PASS")
    for name, expected in audit["hashes"].items():
        if file_hash(ROOT/name) != expected:
            raise RuntimeError("STOP_GATE audited dependency/binary changed: " + name)
    integration = read_json(LOG / "integration_check.json")
    if integration.get("status") != "PASS" or integration.get("code_sha256") != code_hash() or integration.get("worker_sha256") != file_hash(default_worker()):
        raise RuntimeError("STOP_GATE integration check missing/stale")


def dependency_audit():
    suffix = "" if not (LOG / "dependency_and_baseline_check.json").exists() else "_" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S")
    directory = BUILD / ("dependency_audit" + suffix)
    recorded("dependency_and_baseline_check" + suffix, [sys.executable, "diagnostics/audit_g5_dependency.py", "--output", str(directory), "--worker", str(default_worker()), "--build-dir", str(BUILD)])
    audit = read_json(directory / "audit.json")
    audit["status"] = "PASS" if audit["valid"] else "FAIL"
    dump(LOG / "dependency_audit.json", audit)
    dump(LOG / "integration_check.json", {"status": audit["status"], "code_sha256": audit["code_sha256"],
         "worker_sha256": file_hash(default_worker()), "primary_dataset": False, "checks": audit["integration_checks"],
         "audit_sha256": file_hash(directory / "audit.json")})


def extra_hashes():
    return {str(p.relative_to(ROOT)): file_hash(p) for p in [CONFIG, ROOT / "docs/G5_PROTOCOL.md", ROOT / "docs/G5_BUDGET_RULE.md", DATA / "terrain_informed_geometry.json", LOG / "pilot_tail_audit.json", LOG / "correctness_gate.json", LOG / "dependency_audit.json", LOG / "integration_check.json"]}


def freeze():
    gate_matches()
    config = full_config()
    maps = read_json(DATA / "manifest.json")
    if maps["config_hash"] != digest(config):
        raise ValueError("dataset config mismatch")
    for item in maps["maps"]:
        if file_hash(item["path"]) != item["sha256"]:
            raise ValueError("map hash mismatch")
        for query in item["queries"]:
            if query["query_hash"] != digest({"start": query["start"], "goal": query["goal"], "map_hash": item["sha256"], "model": config["model"]}):
                raise ValueError("query hash does not bind model/endpoints/map")
            ref = query["reference"]
            if query["feasibility"] != "feasible_witness_validated" or ref["reference_type"] != "WITNESS_UPPER_BOUND" or not ref["validation"]["valid"]:
                raise ValueError("query lacks independent witness")
    trials = plan(config, "full", maps)
    if len(trials) != 6400:
        raise ValueError("G5 must have exactly 6400 planned trials")
    if OUT.exists() and any(OUT.iterdir()):
        raise ValueError("full output is not empty; never replace manifest")
    OUT.mkdir(parents=True, exist_ok=True)
    identity = provenance_identity(config, maps, default_worker())
    identity["g5_frozen_files"] = extra_hashes()
    identity["generator_version"] = maps["generator_version"]
    query_seeds = {(m["map_id"], q["query_id"]): q["query_seed"] for m in maps["maps"] for q in m["queries"]}
    for entry in trials:
        entry["query_seed"] = query_seeds[(entry["map_id"], entry["query_id"])]
        entry["source_sha256"] = identity["code_sha256"]
        entry["binary_sha256"] = identity["worker_sha256"]
        entry["dependency_sha256"] = digest({k: v for k, v in identity.items() if "dependency" in k or "ompl" in k or "patch" in k})
        entry["config_sha256"] = identity["config_hash"]
    archive_source(OUT, identity)
    shutil.copyfile(CONFIG, OUT / CONFIG.name)
    shutil.copyfile(ROOT / "docs/G5_PROTOCOL.md", OUT / "G5_PROTOCOL.md")
    shutil.copyfile(DATA / "terrain_informed_geometry.json", OUT / "terrain_informed_geometry.json")
    dump(OUT / "environment.json", environment())
    dump(OUT / "resolved_config.json", config)
    dump(OUT / "map_manifest.json", maps)
    manifest = {"schema_version": 2, "experiment_id": OUT.name, "profile": "full", "status": "LOCKED",
                "created_utc": utc(), "identity": identity, "planned_trials": 6400, "trials": trials,
                "resume_policy": "NOT_RUN only, never rerun an existing attempt; immutable manifest; state stored separately",
                "planner_jobs": 1, "postprocessing": False}
    dump(OUT / "manifest.json", manifest)
    dump(OUT / "manifest.lock.json", {"sha256": file_hash(OUT / "manifest.json"), "locked_utc": utc(), "identity_sha256": digest(identity)})
    update_state(manifest, "NOT_RUN")
    print("LOCKED 6400 trials", file_hash(OUT / "manifest.json"), flush=True)


def update_state(manifest, status, reason=None):
    counts = summarize(OUT, manifest)
    dump(OUT / "execution_state.json", {"status": status, "reason": reason, "utc": utc(), "counts": counts})
    return counts


def verify_identity(manifest, *, all_inputs=True):
    lock = read_json(OUT / "manifest.lock.json")
    if file_hash(OUT / "manifest.json") != lock["sha256"]:
        raise RuntimeError("STOP_GATE manifest changed")
    if digest(manifest["identity"]) != lock["identity_sha256"]:
        raise RuntimeError("STOP_GATE identity changed")
    identity = manifest["identity"]
    if identity["worker_sha256"] != file_hash(default_worker()) or identity["code_sha256"] != code_hash():
        raise RuntimeError("STOP_GATE source/binary changed")
    if file_hash(ROOT / ".deps/install/lib/libompl.dll") != identity["ompl_library_sha256"]:
        raise RuntimeError("STOP_GATE linked dependency changed")
    for name, sha in identity["g5_frozen_files"].items():
        if file_hash(ROOT / name) != sha:
            raise RuntimeError("STOP_GATE frozen input changed: " + name)
    if all_inputs:
        config = full_config()
        maps = read_json(DATA / "manifest.json")
        now = provenance_identity(config, maps, default_worker())
        if any(identity[k] != v for k, v in now.items()):
            raise RuntimeError("STOP_GATE provenance mismatch")
        for item in maps["maps"]:
            if file_hash(item["path"]) != item["sha256"]:
                raise RuntimeError("STOP_GATE map changed")
        source = read_json(OUT / "source_manifest.json")
        if file_hash(OUT / "source_at_run.zip") != source["archive_sha256"]:
            raise RuntimeError("STOP_GATE source archive changed")


@contextlib.contextmanager
def exclusive_execution():
    """OS lock is released after a process crash; stale file never implies a retry."""
    with (OUT / "execution.lock").open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("G5 runner/verifier already active; no concurrent attempt allowed") from exc
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def run():
    with exclusive_execution():
        return run_locked()


def run_locked():
    manifest = read_json(OUT / "manifest.json")
    try:
        verify_identity(manifest)
        gate_matches()
        config = full_config()
        env = read_json(OUT / "environment.json")
        if (OUT / "execution_state.json").exists() and read_json(OUT / "execution_state.json")["status"] == "BLOCKED":
            raise RuntimeError("STOP_GATE previous BLOCKED state requires investigation outside primary run")
        if summarize(OUT, manifest)["invalid"]:
            raise RuntimeError("STOP_GATE previously detected invalid output")
        for entry in manifest["trials"]:
            directory = OUT / entry["trial_id"]
            if (directory / "trial.json").exists():
                continue
            if directory.exists():
                dump(directory / "trial.json", dict(entry, execution_status="CRASH", planner_status="UNKNOWN", success=False,
                     path_available=False, independent_valid=None, time_budget_s=config["profiles"]["full"]["planning_budget_s"],
                     termination_reason="INTERRUPTED_PREVIOUS_RUN", experiment_id=OUT.name,
                     code_sha256=manifest["identity"]["code_sha256"], config_hash=manifest["identity"]["config_hash"],
                     attempt_history=[read_json(directory / "attempt.json") if (directory / "attempt.json").exists() else None,
                                      {"event": "RETAINED_NO_RETRY", "utc": utc()}]))
                continue
            verify_identity(manifest, all_inputs=False)
            if file_hash(entry["map_path"]) != entry["map_hash"]:
                raise RuntimeError("STOP_GATE trial map changed")
            result = run_trial(entry, config, directory, default_worker(), env, OUT.name, digest(config), profile="full")
            if result.get("execution_status") == "BLOCKED":
                raise RuntimeError("STOP_GATE worker cannot start")
            update_state(manifest, "RUNNING")
            print(f'{entry["shuffle_order"]+1}/6400 {entry["trial_id"]}: {result["execution_status"]}', flush=True)
        update_state(manifest, "EXECUTED_PENDING_VALIDATION")
    except KeyboardInterrupt:
        update_state(manifest, "PARTIAL", "normal interruption; resume only NOT_RUN")
        raise
    except Exception as exc:
        update_state(manifest, "BLOCKED", str(exc))
        raise


def validate():
    with exclusive_execution():
        return validate_locked()


def validate_locked():
    from concurrent.futures import ProcessPoolExecutor, as_completed
    manifest = read_json(OUT / "manifest.json")
    try:
        verify_identity(manifest)
        groups = {}
        for entry in manifest["trials"]:
            groups.setdefault(entry["map_path"], []).append(entry)
        with ProcessPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(validate_group, str(OUT), map_path, entries) for map_path, entries in groups.items()]
            for future in as_completed(futures):
                try:
                    print("validated group", future.result(), flush=True)
                    update_state(manifest, "VALIDATING")
                except BaseException:
                    for pending in futures:
                        pending.cancel()
                    raise
        summary = summarize(OUT, manifest)
        summary["status"] = "COMPLETE_VALIDATED" if not summary["not_run"] else "PARTIAL"
        dump(OUT / "validation_summary.json", summary)
        update_state(manifest, summary["status"])
    except Exception as exc:
        summary = summarize(OUT, manifest)
        summary.update(status="BLOCKED", reason=str(exc))
        dump(OUT / "validation_summary.json", summary)
        update_state(manifest, "BLOCKED", str(exc))
        raise


def validate_group(output, map_path, entries):
    """Offline map group; geometry oracle is unchanged and no planner is launched."""
    from independent_verifier import Oracle
    from validate_results import validate_trial
    from analyze_results import checked_events
    oracle = None
    checked = retained = 0
    for entry in entries:
        directory = Path(output) / entry["trial_id"]
        if not (directory / "trial.json").exists():
            continue
        existing = read_json(directory / "trial.json")
        if existing.get("success") and existing.get("snapshots_valid"):
            events, errors = checked_events(directory, existing, 1e-8, 1e-8)
            last = directory / "snapshots" / events[-1]["snapshot_id"] if events else None
            final_matches = last and all(file_hash(directory/name) == file_hash(last/name) for name in ("path_xy.csv", "path_surface.csv"))
            if events and not errors and final_matches:
                retained += 1
                continue
        if oracle is None:
            oracle = Oracle(map_path)
        result = validate_trial(directory, oracle)
        checked += 1
        if result.get("independent_valid") is False or result.get("snapshots_valid") is False:
            raise RuntimeError("STOP_GATE verifier mismatch: " + entry["trial_id"])
        print("validated", entry["trial_id"], flush=True)
    return {"map_path": map_path, "checked": checked, "retained": retained}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["preserve", "check-preservation", "prepare", "gate", "dependency-audit", "freeze", "run", "validate", "index"])
    parser.add_argument("--allow-full", action="store_true")
    parser.add_argument("--resume", action="store_true", help="existing attempts are always preserved")
    args = parser.parse_args()
    if args.action in ("freeze", "run") and not args.allow_full:
        parser.error("explicit --allow-full required")
    actions = {"preserve": preservation, "check-preservation": lambda: preservation(True), "prepare": prepare_protocol, "gate": correctness_gate,
               "dependency-audit": dependency_audit, "freeze": freeze, "run": run, "validate": validate, "index": command_index}
    actions[args.action]()


if __name__ == "__main__":
    main()
