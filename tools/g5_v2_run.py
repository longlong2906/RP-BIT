"""G5-v2 harness: new identities, immutable v1, unchanged worker and geometry.

Every public action records its exact command, output and return code. Primary
execution is sequential; offline validation separates fixed 160-attempt blocks.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from experiment_common import (ROOT, code_hash, digest, dump, environment,
                               file_hash, load_config, read_json, worker_env)
from run_experiments import archive_source, plan, provenance_identity, run_trial

OUT = ROOT / "results/full_g5_v2"
REPORT = ROOT / "reports/full_g5_v2"
LOG = ROOT / "logs/g5_v2"
DATA = ROOT / "data/full_g5_v2"
CONFIG = ROOT / "configs/g5_full_v2.yaml"
PROTOCOL = ROOT / "docs/G5_V2_PROTOCOL.md"
BUILD = ROOT / "build-g5-v2"
WORKER = BUILD / "retained-worker/terrain_benchmark.exe"
OLD_WORKER = ROOT / "build-g5/terrain_benchmark.exe"
V1 = ROOT / "results/full_g5_v1"
BLOCK_SIZE = 160
FAMILIES = ["traversable_hill", "ridge_pass", "rolling_terrain", "narrow_passage", "mixed_obstacles"]
PLANNERS = ["bitstar_a0", "bitstar_a3", "informed_rrtstar", "rrtstar"]


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def command_index():
    records = []
    for path in sorted(LOG.glob("command_*.json")):
        if path.name == "command_index.json":
            continue
        record = read_json(path)
        if "command" in record:
            records.append(dict(record, record_path=path.relative_to(ROOT).as_posix()))
    dump(LOG / "command_index.json", {"updated_utc": utc(), "commands": records})


def recorded(name, command):
    LOG.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    stem = LOG / ("command_" + name + "_" + stamp)
    record = {"command": list(map(str, command)), "cwd": str(ROOT), "started_utc": utc(),
              "status": "RUNNING", "exit_code": None,
              "stdout": stem.with_suffix(".stdout.log").relative_to(ROOT).as_posix(),
              "stderr": stem.with_suffix(".stderr.log").relative_to(ROOT).as_posix()}
    dump(stem.with_suffix(".json"), record)
    command_index()
    began = time.perf_counter()
    with stem.with_suffix(".stdout.log").open("xb") as stdout, stem.with_suffix(".stderr.log").open("xb") as stderr:
        try:
            process = subprocess.Popen(command, cwd=ROOT, env=worker_env(), stdout=stdout, stderr=stderr)
            record["pid"] = process.pid
            dump(stem.with_suffix(".json"), record)
            result = process.wait()
        except OSError as exc:
            stderr.write(str(exc).encode())
            result = 127
        except KeyboardInterrupt:
            record.update(status="INTERRUPTED", reason="operator_interruption")
            raise
        finally:
            record.update(finished_utc=utc(), wall_s=time.perf_counter()-began)
    record.update(exit_code=result, status="PASS" if result == 0 else "FAIL")
    dump(stem.with_suffix(".json"), record)
    command_index()
    print(name, "exit", result, "record", stem.with_suffix(".json"), flush=True)
    if result:
        raise RuntimeError("STOP_GATE command failed: " + name)
    return record


def full_config():
    config = load_config(CONFIG, "pilot")
    old = load_config(ROOT / "configs/g5_full_v1.yaml", "pilot")
    registered = read_json(LOG / "seed_history_adjudication.json")["registered_map_seeds"]
    expected = json.loads(json.dumps(old))
    expected["profiles"]["full"]["map_seeds"] = registered
    if config != expected:
        raise ValueError("STOP_GATE only held-out map seeds may differ from v1 config")
    p = config["profiles"]["full"]
    if (p["families"] != FAMILIES or p["planners"] != PLANNERS or len(registered) != 8
            or p["planning_budget_s"] != 10. or p["planner_seeds"] != list(range(1, 11))
            or p["queries_per_map"] != 4):
        raise ValueError("STOP_GATE unexpected design")
    return config


def preservation_check():
    baseline = read_json(LOG / "preservation_baseline.json")
    source_baseline = json.loads((LOG / "initial_source_inventory.json").read_text(encoding="utf-8-sig"))
    # All old data/reports/logs and every pre-existing document except STATUS
    # remain immutable. Only the analyzer and its CTest registration may change.
    allowed = {"tools/g5_analysis.py", "cmake/OmplTargets.cmake", "docs/STATUS.md"}
    expected = dict(baseline["files"])
    expected.update(source_baseline["files"])
    changed = [name for name, sha in expected.items() if name not in allowed and
               (not (ROOT / name).is_file() or file_hash(ROOT / name) != sha)]
    result = {"status": "PASS" if not changed else "FAIL", "utc": utc(),
              "checked": len(expected)-len(allowed), "changed": changed,
              "allowed_current_source_edits": sorted(allowed),
              "baseline_sha256": file_hash(LOG / "preservation_baseline.json"),
              "v1_manifest_sha256": file_hash(V1 / "manifest.json"),
              "v1_source_sha256": read_json(V1 / "manifest.json")["identity"]["code_sha256"]}
    dump(LOG / ("preservation_" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%f") + ".json"), result)
    dump(LOG / "preservation_latest.json", result)
    print({k: result[k] for k in ("status", "checked", "changed")}, flush=True)
    if changed:
        raise RuntimeError("STOP_GATE historical or out-of-scope file modified")
    return result


def correctness_gate():
    if (OUT / "manifest.json").exists():
        raise ValueError("cannot rebuild/retest frozen experiment source")
    recorded("configure_release", ["cmake", "-S", ".", "-B", str(BUILD), "-G", "Ninja",
        "-DCMAKE_CXX_COMPILER=C:/mingw64/bin/g++.exe", "-DCMAKE_BUILD_TYPE=Release",
        "-DPython3_EXECUTABLE=" + sys.executable,
        "-DCMAKE_PREFIX_PATH=" + str(ROOT / ".deps/install"),
        "-DBOOST_ROOT=" + str(ROOT / ".deps/boost_1_82_0"),
        "-DBOOST_LIBRARYDIR=" + str(ROOT / ".deps/boost_1_82_0/stage/lib"),
        "-DBoost_NO_SYSTEM_PATHS=ON", "-DBoost_NO_BOOST_CMAKE=ON", "-DBoost_USE_STATIC_LIBS=ON"])
    recorded("build_release", ["cmake", "--build", str(BUILD), "--parallel", "2"])
    recorded("ctest_release", ["ctest", "--test-dir", str(BUILD), "--output-on-failure"])
    expected = read_json(V1 / "manifest.json")["identity"]["worker_sha256"]
    if file_hash(OLD_WORKER) != expected:
        raise RuntimeError("historical worker bytes changed")
    WORKER.parent.mkdir(parents=True, exist_ok=True)
    if WORKER.exists():
        if file_hash(WORKER) != expected:
            raise RuntimeError("retained worker copy hash mismatch")
    else:
        shutil.copyfile(OLD_WORKER, WORKER)
    recorded("retained_worker_pipeline", [sys.executable, "tests/test_pipeline.py", "--worker", str(WORKER)])
    gate = read_json(WORKER.parent / "correctness_gate.json")
    if gate.get("worker_sha256") != expected or gate.get("code_sha256") != code_hash() or not gate.get("pipeline_passed"):
        raise RuntimeError("retained worker pipeline gate mismatch")
    dump(LOG / "correctness_gate.json", {"status": "PASS", "utc": utc(), "code_sha256": code_hash(),
         "worker_sha256": expected, "worker_identical_to_v1": True,
         "release_rebuild_sha256": file_hash(BUILD / "terrain_benchmark.exe"),
         "analyzer_sha256": file_hash(ROOT / "tools/g5_analysis.py"),
         "pipeline_gate_sha256": file_hash(WORKER.parent / "correctness_gate.json")})


def dependency_audit():
    target = BUILD / ("dependency_audit_" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%f"))
    recorded("dependency_integration", [sys.executable, "diagnostics/audit_g5_dependency.py",
             "--output", str(target), "--worker", str(WORKER), "--build-dir", str(BUILD)])
    audit = read_json(target / "audit.json")
    if not audit["valid"]:
        raise RuntimeError("dependency audit failed")
    dump(LOG / "dependency_audit.json", dict(audit, status="PASS", evidence=str(target)))


def gate_matches():
    for name in ("correctness_gate.json", "dependency_audit.json"):
        gate = read_json(LOG / name)
        if gate.get("status") != "PASS" or gate.get("code_sha256") != code_hash():
            raise RuntimeError("STOP_GATE missing or stale " + name)
    gate = read_json(LOG / "correctness_gate.json")
    if gate["worker_sha256"] != file_hash(WORKER):
        raise RuntimeError("STOP_GATE worker gate mismatch")
    for name, sha in read_json(LOG / "dependency_audit.json")["hashes"].items():
        if file_hash(ROOT / name) != sha:
            raise RuntimeError("STOP_GATE dependency bytes changed: " + name)
    forensic = read_json(ROOT / "reports/full_g5_v1_forensic_analyzer_v2/summary.json")
    if forensic.get("status") != "PASS" or forensic.get("analyzer_sha256") != file_hash(ROOT / "tools/g5_analysis.py"):
        raise RuntimeError("STOP_GATE forensic check missing or stale")
    if read_json(LOG / "preservation_latest.json").get("status") != "PASS":
        raise RuntimeError("STOP_GATE preservation failed")
    seed = read_json(LOG / "seed_history_adjudication.json")
    if seed.get("status") != "PASS":
        raise RuntimeError("STOP_GATE seed provenance missing")


def generate():
    gate_matches()
    import g5_dataset
    config = full_config()
    # Seed selection is harness configuration; all generator functions and
    # distributions are reused byte-for-byte, and both hashes are locked.
    g5_dataset.MAP_SEEDS = tuple(config["profiles"]["full"]["map_seeds"])
    g5_dataset.generate(config, DATA, PROTOCOL)


def frozen_files():
    paths = [CONFIG, PROTOCOL, ROOT / "docs/G5_BUDGET_RULE.md", DATA / "manifest.json",
             DATA / "terrain_informed_geometry.json", LOG / "seed_history_audit.json",
             LOG / "seed_history_adjudication.json",
             LOG / "correctness_gate.json", LOG / "dependency_audit.json",
             ROOT / "reports/full_g5_v1_forensic_analyzer_v2/summary.json"]
    return {p.relative_to(ROOT).as_posix(): file_hash(p) for p in paths}


def freeze():
    gate_matches()
    preservation_check()
    config = full_config()
    maps = read_json(DATA / "manifest.json")
    if maps["config_hash"] != digest(config) or maps["protocol_sha256"] != file_hash(PROTOCOL):
        raise ValueError("dataset config/protocol mismatch")
    for item in maps["maps"]:
        if file_hash(item["path"]) != item["sha256"]:
            raise ValueError("map hash mismatch")
        for query in item["queries"]:
            if query["query_hash"] != digest({"start": query["start"], "goal": query["goal"], "map_hash": item["sha256"], "model": config["model"]}):
                raise ValueError("query hash mismatch")
            ref = query["reference"]
            if query["feasibility"] != "feasible_witness_validated" or ref["reference_type"] != "WITNESS_UPPER_BOUND" or not ref["validation"]["valid"]:
                raise ValueError("query lacks independent witness qualification")
    entries = plan(config, "full", maps)
    if len(entries) != 6400:
        raise ValueError("G5-v2 requires exactly 6400 primary trials")
    if OUT.exists() and any(OUT.iterdir()):
        raise ValueError("never overwrite/relock a nonempty experiment directory")
    OUT.mkdir(parents=True, exist_ok=True)
    identity = provenance_identity(config, maps, WORKER)
    identity.update(analyzer_schema_version=2, analyzer_sha256=file_hash(ROOT / "tools/g5_analysis.py"),
        require_raw_trial_backup=True,
        worker_identical_to_v1=identity["worker_sha256"] == read_json(V1 / "manifest.json")["identity"]["worker_sha256"],
        g5_frozen_files=frozen_files(), generator_version=maps["generator_version"],
        generator_sha256=file_hash(ROOT / "tools/g5_dataset.py"),
        harness_sha256=file_hash(__file__), geometry_file_key=DATA.relative_to(ROOT).as_posix()+"/terrain_informed_geometry.json")
    query_seeds = {(m["map_id"], q["query_id"]): q["query_seed"] for m in maps["maps"] for q in m["queries"]}
    for entry in entries:
        entry.update(query_seed=query_seeds[(entry["map_id"], entry["query_id"])],
            source_sha256=identity["code_sha256"], analyzer_sha256=identity["analyzer_sha256"],
            binary_sha256=identity["worker_sha256"], config_sha256=identity["config_hash"],
            dependency_sha256=digest({k: v for k, v in identity.items() if "dependency" in k or "ompl" in k or "patch" in k}))
    archive_source(OUT, identity)
    identity["source_bundle_sha256"] = file_hash(OUT / "source_at_run.zip")
    identity["source_manifest_sha256"] = file_hash(OUT / "source_manifest.json")
    for path in (CONFIG, PROTOCOL):
        shutil.copyfile(path, OUT / path.name)
    shutil.copyfile(DATA / "terrain_informed_geometry.json", OUT / "terrain_informed_geometry.json")
    dump(OUT / "environment.json", environment())
    dump(OUT / "resolved_config.json", config)
    dump(OUT / "map_manifest.json", maps)
    manifest = {"schema_version": 3, "experiment_id": OUT.name, "profile": "full", "status": "LOCKED",
        "created_utc": utc(), "identity": identity, "planned_trials": len(entries), "trials": entries,
        "planner_jobs": 1, "validation_block_size": BLOCK_SIZE, "validation_jobs": 4,
        "attempt_policy": "one primary attempt; no retry or replacement; correctness stop prohibits resume",
        "postprocessing": False}
    dump(OUT / "manifest.json", manifest)
    dump(OUT / "manifest.lock.json", {"sha256": file_hash(OUT / "manifest.json"), "identity_sha256": digest(identity), "locked_utc": utc()})
    update_state("NOT_RUN", manifest)
    print("LOCKED", len(entries), file_hash(OUT / "manifest.json"), flush=True)


def verify_identity(manifest, all_inputs=True):
    identity = manifest["identity"]
    lock = read_json(OUT / "manifest.lock.json")
    if file_hash(OUT / "manifest.json") != lock["sha256"] or digest(identity) != lock["identity_sha256"]:
        raise RuntimeError("STOP_GATE manifest/identity changed")
    if code_hash() != identity["code_sha256"] or file_hash(WORKER) != identity["worker_sha256"]:
        raise RuntimeError("STOP_GATE source/worker changed")
    if (file_hash(OUT / "source_at_run.zip") != identity["source_bundle_sha256"] or
            file_hash(OUT / "source_manifest.json") != identity["source_manifest_sha256"]):
        raise RuntimeError("STOP_GATE full source bundle/source manifest changed")
    if file_hash(ROOT / ".deps/install/lib/libompl.dll") != identity["ompl_library_sha256"]:
        raise RuntimeError("STOP_GATE OMPL DLL changed")
    for name, sha in identity["g5_frozen_files"].items():
        if file_hash(ROOT / name) != sha:
            raise RuntimeError("STOP_GATE frozen input changed: " + name)
    if all_inputs:
        now = provenance_identity(full_config(), read_json(DATA / "manifest.json"), WORKER)
        if any(identity[k] != value for k, value in now.items()):
            raise RuntimeError("STOP_GATE provenance mismatch")
        for entry in read_json(DATA / "manifest.json")["maps"]:
            if file_hash(entry["path"]) != entry["sha256"]:
                raise RuntimeError("STOP_GATE map changed")
        if file_hash(OUT / "source_at_run.zip") != read_json(OUT / "source_manifest.json")["archive_sha256"]:
            raise RuntimeError("STOP_GATE source archive changed")


def preflight():
    manifest = read_json(OUT / "manifest.json")
    verify_identity(manifest)
    if any((OUT / e["trial_id"]).exists() for e in manifest["trials"]):
        raise ValueError("preflight must precede every primary attempt")
    target = BUILD / ("preflight_" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%f"))
    try:
        command = recorded("preflight_reader", [sys.executable, "tests/test_g5_taxonomy.py", "--preflight-output", str(target)])
    except BaseException as exc:
        update_state("BLOCKED", manifest, "PREFLIGHT_READER_FAILURE: " + str(exc))
        raise
    dump(LOG / "preflight.json", {"status": "PASS", "utc": utc(), "code_sha256": code_hash(),
         "manifest_sha256": file_hash(OUT / "manifest.json"), "primary_dataset": False,
         "evidence_directory": str(target), "command": command})


@contextlib.contextmanager
def exclusive_execution():
    with (OUT / "execution.lock").open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def bind_artifacts(directory):
    target = directory / "artifact_hashes.json"
    if target.exists():
        raise ValueError("raw artifact bindings already exist; never replace")
    shutil.copyfile(directory / "trial.json", directory / "trial_before_validation.json")
    files = {p.relative_to(directory).as_posix(): file_hash(p) for p in sorted(directory.rglob("*"))
             if p.is_file() and p.name not in ("trial.json", "validation.json", "validated_events.json")}
    dump(target, {"schema_version": 1, "utc": utc(), "files": files})


def validate_group(entries, identity):
    from independent_verifier import Oracle
    from validate_results import validate_trial
    from g5_analysis import read_g5_trial
    oracle = None
    for entry in entries:
        directory = OUT / entry["trial_id"]
        trial = read_json(directory / "trial.json")
        if trial.get("path_available") or (directory / "progress.jsonl").exists():
            if oracle is None:
                oracle = Oracle(entry["map_path"])
        validate_trial(directory, oracle)
        row = read_g5_trial(OUT, entry, 10., 1e-8, 1e-8, identity)
        semantic = {k: v for k, v in row.items() if k != "raw"}
        semantic["analyzer_sha256"] = identity["analyzer_sha256"]
        dump(directory / "analysis_validation.json", semantic)
        # Bind the independent verifier output too, without changing its code.
        validation_files = [p for p in directory.rglob("validation.json")]
        if (directory / "validated_events.json").exists():
            validation_files.append(directory / "validated_events.json")
        validation_files.append(directory / "analysis_validation.json")
        dump(directory / "validation_hashes.json", {"files": {
            p.relative_to(directory).as_posix(): file_hash(p) for p in validation_files}})
        if row["correctness_error"]:
            raise RuntimeError("STOP_GATE analyzer/verifier mismatch: " + entry["trial_id"] + " " + str(row["errors"]))
    return len(entries)


def validate_block(entries, manifest):
    groups = {}
    for entry in entries:
        groups.setdefault(entry["map_path"], []).append(entry)
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(validate_group, group, manifest["identity"]) for group in groups.values()]
        for future in as_completed(futures):
            try:
                print("independently validated", future.result(), "attempts", flush=True)
            except BaseException:
                for pending in futures:
                    pending.cancel()
                raise


def counts(manifest):
    result = dict(planned=len(manifest["trials"]), executed=0, validated_success=0,
                  timeout=0, crash=0, watchdog=0, administrative_interrupted_attempts=0,
                  correctness_errors=0, geometric_invalid_outputs=0,
                  final_paths_available=0, final_paths_validated=0, snapshots_available=0, snapshots_validated=0,
                  validation_pending=0)
    for entry in manifest["trials"]:
        directory = OUT / entry["trial_id"]
        if not (directory / "trial.json").exists():
            continue
        trial = read_json(directory / "trial.json")
        result["executed"] += 1
        admin = trial.get("administrative_interruption") or trial.get("termination_reason", "").startswith("INTERRUPTED")
        result["administrative_interrupted_attempts"] += bool(admin)
        for key in ("timeout", "crash", "watchdog"):
            result[key] += trial.get("execution_status") == key.upper() and not (key == "crash" and admin)
        result["final_paths_available"] += bool(trial.get("path_available"))
        if not (directory / "analysis_validation.json").exists():
            result["validation_pending"] += 1
            continue
        row = read_json(directory / "analysis_validation.json")
        result["validated_success"] += row["validated_success"]
        result["correctness_errors"] += row["correctness_error"]
        result["geometric_invalid_outputs"] += row["path_present"] and row["geometry_valid"] is False
        result["final_paths_validated"] += bool(row["path_present"] and row["geometry_valid"])
        events = read_json(directory / "validated_events.json") if (directory / "validated_events.json").exists() else []
        result["snapshots_available"] += len(events)
        result["snapshots_validated"] += sum(e.get("independent_valid") is True for e in events)
    result["not_run"] = result["planned"]-result["executed"]
    result["invalid"] = result["correctness_errors"]
    return result


def update_state(status, manifest, reason=None):
    summary = counts(manifest)
    dump(OUT / "execution_state.json", {"status": status, "utc": utc(), "reason": reason, "counts": summary})
    return summary


def final_validation(manifest):
    from g5_analysis import read_g5_trial
    verify_identity(manifest)
    for entry in manifest["trials"]:
        directory = OUT / entry["trial_id"]
        if not (directory / "analysis_validation.json").exists():
            raise RuntimeError("STOP_GATE trial lacks validation: " + entry["trial_id"])
        for name, sha in read_json(directory / "validation_hashes.json")["files"].items():
            if file_hash(directory / name) != sha:
                raise RuntimeError("STOP_GATE validation checksum mismatch")
        row = read_g5_trial(OUT, entry, 10., 1e-8, 1e-8, manifest["identity"])
        if row["correctness_error"]:
            raise RuntimeError("STOP_GATE final reader failure: " + entry["trial_id"])
    summary = counts(manifest)
    if summary["not_run"] or summary["validation_pending"] or summary["correctness_errors"]:
        raise RuntimeError("STOP_GATE full completion/validation missing")
    summary.update(status="COMPLETE_VALIDATED", verification_status="PASS", utc=utc(),
        code_sha256=code_hash(), analyzer_sha256=manifest["identity"]["analyzer_sha256"],
        manifest_sha256=file_hash(OUT / "manifest.json"))
    dump(OUT / "validation_summary.json", summary)
    update_state("COMPLETE_VALIDATED", manifest)
    return summary


def run():
    with exclusive_execution():
        manifest = read_json(OUT / "manifest.json")
        try:
            verify_identity(manifest)
            gate_matches()
            state = read_json(OUT / "execution_state.json")
            if state["status"] == "BLOCKED":
                raise RuntimeError("STOP_GATE blocked manifest cannot resume")
            pre = read_json(LOG / "preflight.json")
            if pre.get("status") != "PASS" or pre["code_sha256"] != code_hash() or pre["manifest_sha256"] != file_hash(OUT / "manifest.json"):
                raise RuntimeError("STOP_GATE preflight missing or stale")
            config = full_config()
            env = read_json(OUT / "environment.json")
            pending = []
            for entry in manifest["trials"]:
                directory = OUT / entry["trial_id"]
                if directory.exists():
                    if not (directory / "trial.json").exists():
                        # An existing attempt is retained, never restarted.
                        dump(directory / "trial.json", dict(entry, experiment_id=OUT.name,
                            execution_status="CRASH", planner_status="UNKNOWN", success=False,
                            path_available=False, independent_valid=None, time_budget_s=10.,
                            code_sha256=manifest["identity"]["code_sha256"], config_hash=manifest["identity"]["config_hash"],
                            administrative_interruption=True, actual_solve_wall_s=None,
                            termination_reason="INTERRUPTED_PREVIOUS_RUN_RETAINED_NO_RETRY"))
                    if not (directory / "artifact_hashes.json").exists():
                        bind_artifacts(directory)
                    if not (directory / "analysis_validation.json").exists():
                        pending.append(entry)
                else:
                    verify_identity(manifest, all_inputs=False)
                    if file_hash(entry["map_path"]) != entry["map_hash"]:
                        raise RuntimeError("STOP_GATE trial map changed")
                    result = run_trial(entry, config, directory, WORKER, env, OUT.name, digest(config), profile="full")
                    bind_artifacts(directory)
                    if result["execution_status"] == "BLOCKED":
                        raise RuntimeError("STOP_GATE worker launch blocked")
                    pending.append(entry)
                    print(f'{entry["shuffle_order"]+1}/6400 {entry["trial_id"]}: {result["execution_status"]}', flush=True)
                if len(pending) >= BLOCK_SIZE:
                    update_state("RUNNING", manifest, "OFFLINE_VALIDATION_NO_PLANNER_ACTIVE")
                    validate_block(pending, manifest)
                    pending = []
                # Raw attempt markers are written per trial. Recompute the
                # expensive full-manifest status every ten attempts; a crash
                # is reconstructed from those markers without retrying them.
                if (entry["shuffle_order"] + 1) % 10 == 0:
                    update_state("RUNNING", manifest)
            if pending:
                validate_block(pending, manifest)
            final_validation(manifest)
        except BaseException as exc:
            summary = update_state("BLOCKED", manifest, str(exc))
            dump(OUT / "validation_summary.json", dict(summary, status="BLOCKED", reason=str(exc), utc=utc()))
            raise


def analyze():
    from g5_analysis import analyze as analyze_results
    manifest = read_json(OUT / "manifest.json")
    verify_identity(manifest)
    if read_json(OUT / "validation_summary.json").get("status") != "COMPLETE_VALIDATED":
        raise RuntimeError("STOP_GATE inference requires complete validation")
    return analyze_results(OUT, REPORT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["gate", "dependency-audit", "preserve-check", "forensic",
        "generate", "freeze", "preflight", "run", "validate", "analyze", "index"])
    parser.add_argument("--allow-full", action="store_true")
    parser.add_argument("--internal", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.action in ("generate", "freeze", "run") and not args.allow_full:
        parser.error("explicit --allow-full required")
    if not args.internal and args.action != "index":
        recorded(args.action, [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--internal"])
        return
    if args.action == "forensic":
        recorded("forensic_readonly", [sys.executable, "tools/g5_v2_forensic.py", "--output",
                 str(ROOT / "reports/full_g5_v1_forensic_analyzer_v2")])
        return
    actions = {"gate": correctness_gate, "dependency-audit": dependency_audit,
        "preserve-check": preservation_check, "generate": generate, "freeze": freeze,
        "preflight": preflight, "run": run, "validate": lambda: final_validation(read_json(OUT / "manifest.json")),
        "analyze": analyze, "index": command_index}
    actions[args.action]()


if __name__ == "__main__":
    main()
