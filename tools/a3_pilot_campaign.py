"""Sequential, checkpointed A3 pilot collection; no upload or remote dispatch.

The default is the 52-trial smoke. The 780-trial pilot requires both its
explicit profile and permission flag, a Linux worker, CTest evidence and the
validated smoke gate from this exact source/config/runtime.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import contextlib
import datetime as dt
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

import psutil

import experiment_common as common
from experiment_common import ROOT, digest, dump, file_hash, load_config, worker_env
from cloud_campaign import IntegrityError, dataset_file, inventory, check_inventory, read_json, relative
from independent_verifier import Oracle
from run_experiments import request_for, verify_variant_result, bitstar_log_counters

PROTOCOL_ID = "a3_improvements_pilot_v1"
IDENTITY_GATE_KEYS = ("code_sha256", "worker_sha256", "ompl_library_sha256")
FATAL_EXECUTION = {"CRASH", "WATCHDOG", "BLOCKED", "POLICY_ERROR"}


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def runtime_identity(environment):
    """CPU is a timing stratum, not a reason to silently change runtime code."""
    keys = ("os", "python", "packages", "compiler", "cmake", "build_flags",
            "ompl_version", "boost_version", "eigen_version", "worker_sha256",
            "ompl_library_sha256")
    result = {key: environment.get(key) for key in keys}
    result["dependency_headers"] = {name: {key: row.get(key) for key in ("version", "sha256")}
        for name, row in environment.get("dependency_version_evidence", {}).items()}
    # ldd also binds the C/C++ runtime used by the measured worker. Absolute
    # mount paths are not identity; the bytes and SONAMEs are.
    linked = environment.get("version_probe_evidence", {}).get("linked_libraries")
    if linked and linked.get("exit_code") == 0:
        libraries = {}
        for line in linked.get("stdout", "").splitlines():
            words = line.split()
            candidate = words[2] if len(words) > 2 and words[1] == "=>" else words[0] if words else ""
            if candidate.startswith("/") and Path(candidate).is_file():
                libraries[Path(candidate).name] = file_hash(candidate)
        result["linked_libraries"] = libraries
    return result


def check_build_gate(path, identity):
    if path is None or not Path(path).is_file():
        raise IntegrityError("build/CTest gate absent")
    gate = read_json(path)
    if gate.get("status") != "PASS" or gate.get("ctest_passed") is not True:
        raise IntegrityError("build/CTest gate failed")
    if any(gate.get(key) != identity.get(key) for key in IDENTITY_GATE_KEYS):
        raise IntegrityError("build/CTest gate is stale")
    return gate


def check_smoke_gate(path, identity, runtime):
    if path is None or not Path(path).is_file():
        raise IntegrityError("52-trial independently validated smoke gate absent")
    gate = read_json(path)
    expected = {key: identity[key] for key in (*IDENTITY_GATE_KEYS, "config_hash")}
    if (gate.get("protocol_id") != PROTOCOL_ID or gate.get("status") != "PASS"
            or gate.get("planned_trials") != 52 or gate.get("validated_trials") != 52
            or gate.get("identity") != expected or gate.get("runtime_identity_sha256") != digest(runtime)):
        raise IntegrityError("52-trial smoke gate failed or differs from pilot runtime/config")
    # A copied gate alone is insufficient: it must still resolve to the raw
    # smoke campaign beside it, including verified immutable trial inventories.
    smoke = Path(path).parent
    manifest = read_json(smoke / "manifest.json")
    if file_hash(smoke / "manifest.json") != gate.get("manifest_sha256"):
        raise IntegrityError("smoke manifest changed after gate")
    from a3_pilot_protocol import plan_trials
    config, maps = read_json(smoke / "resolved_config.json"), read_json(smoke / "map_manifest.json")
    if (manifest.get("profile") != "smoke" or manifest.get("protocol_id") != PROTOCOL_ID
            or digest(config) != identity["config_hash"] or manifest["trials"] != plan_trials(config, "smoke", maps)
            or digest(read_json(smoke / "runtime_identity.json")) != digest(runtime)):
        raise IntegrityError("smoke config/worklist/runtime changed after gate")
    if any(not (smoke / entry["trial_id"] / "artifact_inventory.json").is_file() for entry in manifest["trials"]):
        raise IntegrityError("smoke trial inventory missing")
    verify_completed(smoke, manifest["trials"])
    state = summarize(smoke, manifest["trials"])
    if state["executed"] != 52 or state["offline_validated"] != 52 or state["fatal"] or state["invalid"]:
        raise IntegrityError("smoke evidence does not satisfy gate")
    return gate


def validate_dataset(config, profile, maps, dataset_root, *, witnesses=True):
    settings = config["profiles"][profile]
    expected = {(family, seed) for family in settings["families"] for seed in settings["map_seeds"]}
    entries = maps.get("maps", [])
    if (len(entries) != len(expected) or {(m["family"], m["map_seed"]) for m in entries} != expected
            or len({m["map_id"] for m in entries}) != len(entries)):
        raise IntegrityError("map cardinality/family/seed differs from frozen protocol")
    if maps.get("config_hash") != digest(config) or maps.get("profile") != profile:
        raise IntegrityError("map manifest config/profile mismatch")
    if maps.get("seed_replacements", []) != []:
        raise IntegrityError("map seed replacement is forbidden")
    expected_queries = [f"q{i}" for i in range(settings["queries_per_map"])]
    for m in entries:
        path = dataset_file(dataset_root, m["path"])
        sidecar = read_json(path.with_suffix(".json"))
        if (file_hash(path) != m["sha256"] or sidecar.get("data_sha256") != m["sha256"]
                or sidecar.get("queries") != m["queries"] or sidecar.get("map_id") != m["map_id"]
                or sidecar.get("generator_seed") != m["map_seed"]):
            raise IntegrityError("map or metadata hash/identity changed: " + m["map_id"])
        if [q["query_id"] for q in m["queries"]] != expected_queries:
            raise IntegrityError("query cardinality/order mismatch")
        oracle = Oracle(path) if witnesses else None
        for q in m["queries"]:
            expected_hash = digest({"start": q["start"], "goal": q["goal"], "map_hash": m["sha256"], "model": config["model"]})
            ref = q.get("reference", {})
            if (q.get("query_hash") != expected_hash or q.get("feasibility") != "feasible_witness_validated"
                    or not ref.get("validation", {}).get("valid") or not ref.get("provenance")
                    or ref.get("map_sha256") != m["sha256"] or ref.get("path_sha256") != digest(ref.get("path_xy"))):
                raise IntegrityError("query/witness provenance mismatch")
            if witnesses:
                candidates = ([(ref["path_xy"], ref["cost_terrain_m"])] +
                    [(w["path_xy"], w["validation"]["cost_terrain_m"]) for w in q.get("witnesses", [])])
                for xy, cost in candidates:
                    verified = oracle.validate_path(xy, q["start"], q["goal"])
                    tol = config["model"]["numerical_abs_tolerance_m"] + config["model"]["numerical_rel_tolerance"] * max(abs(cost), abs(verified.get("cost_terrain_m") or 0))
                    if not verified["valid"] or not math.isfinite(cost) or abs(cost-verified["cost_terrain_m"]) > tol:
                        raise IntegrityError("independent witness verification failed")


def verify_policy(entry, config, record):
    verify_variant_result(entry, config, record)
    if record.get("planner_id") != entry["planner_id"]:
        raise IntegrityError("worker planner label differs from locked entry")
    if record.get("termination_reason") == "INVALID_QUERY":
        raise IntegrityError("qualified query rejected by worker")
    if entry["planner_id"] not in config["planner_variants"]:
        effective = record.get("effective_policy", {})
        if record.get("backend_planner_id", entry["planner_id"]) != entry["planner_id"]:
            raise IntegrityError("baseline backend changed")
        if any(effective.get(key) is not False for key in ("state_cache", "scalar_cost", "collision_index")):
            raise IntegrityError("baseline optimization unexpectedly enabled")
        if effective.get("batch_after_first_solution") != 0 or effective.get("batch_policy") != "fixed":
            raise IntegrityError("baseline batch policy changed")
        if entry["planner_id"] != "lazyprmstar_hauser2015" and effective.get("samples_per_batch") != 200:
            raise IntegrityError("baseline batch is not 200")


def phase_read(path):
    if not Path(path).exists():
        return None
    try:
        value = read_json(path)
    except (FileNotFoundError, PermissionError):
        # Windows replaces its diagnostic phase file via remove+rename. A
        # reader can hit that brief gap/sharing lock; retain the previous phase
        # and its monotonic watchdog deadline until the next 20 ms poll.
        return None
    if value.get("phase") not in ("SETUP", "SOLVING", "SERIALIZING", "FINISHED"):
        raise IntegrityError("invalid worker phase")
    return value


def phase_limit(phase, budget_s, config):
    proto = config["a3_pilot_benchmark"]
    if phase == "SETUP":
        return proto["setup_watchdog_s"]
    if phase == "SOLVING":
        return budget_s + config["execution"]["watchdog_grace_s"]
    return proto["serialization_watchdog_s"]


def launch_trial(entry, config, directory, worker, environment, manifest, dataset_root, session):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    request = request_for(entry, config, manifest["profile"])
    request["cloud_phase_handshake"] = True
    dump(directory / "request.json", request)
    dump(directory / "resolved_config.json", config)
    trial = dict(entry, schema_version=2, experiment_id=manifest["experiment_id"],
        config_hash=manifest["identity"]["config_hash"], code_sha256=manifest["identity"]["code_sha256"],
        environment=environment, runtime_session_id=session["session_id"], runtime_stratum_id=session["stratum_id"],
        utc=utc(), execution_status="BLOCKED", planner_status="NOT_RUN", independent_valid=None,
        success=False, path_available=False, time_budget_s=request["planning_budget_s"],
        best_cost_terrain_m=None, first_solution_time_s=None)
    command = [str(Path(worker).resolve()), str((directory / "request.json").resolve()), str(directory.resolve())]
    trial["command"] = command
    dump(directory / "attempt.json", {"utc": utc(), "status": "START_REQUESTED", "entry": entry,
        "command": command, "cwd_policy": "explicit_dataset_root", "runtime_session_id": session["session_id"]})
    began = phase_began = time.perf_counter()
    phase = "SETUP"
    proc = None
    failure = None
    interrupted = False
    peak = 0.0
    exit_code = None
    with (directory / "stdout.log").open("wb") as stdout, (directory / "stderr.log").open("wb") as stderr:
        try:
            proc = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=worker_env(), cwd=Path(dataset_root).resolve())
            try:
                monitor = psutil.Process(proc.pid)
            except psutil.NoSuchProcess:
                monitor = None
            while proc.poll() is None:
                now = time.perf_counter()
                observed = phase_read(directory / "phase.json")
                if observed and observed["phase"] != phase:
                    nxt = observed["phase"]
                    order = {"SETUP": 0, "SOLVING": 1, "SERIALIZING": 2, "FINISHED": 3}
                    if order[nxt] < order[phase]:
                        failure = "PHASE_REGRESSION"
                        break
                    phase, phase_began = nxt, now
                    timestamp = observed.get("solve_started_utc") if phase == "SOLVING" else observed.get("updated_utc")
                    if timestamp:
                        age = max(0.0, (dt.datetime.now(dt.timezone.utc)-dt.datetime.fromisoformat(timestamp.replace("Z", "+00:00"))).total_seconds())
                        phase_began = now-age
                if monitor:
                    with contextlib.suppress(psutil.NoSuchProcess):
                        peak = max(peak, monitor.memory_info().rss / 2**20)
                if peak > config["execution"]["memory_limit_mb"]:
                    failure = "MEMORY_LIMIT"
                elif now-phase_began > phase_limit(phase, request["planning_budget_s"], config):
                    failure = phase+"_WATCHDOG"
                if failure:
                    break
                time.sleep(0.02)
            if failure and proc.poll() is None:
                proc.kill()
            exit_code = proc.wait()
        except OSError as error:
            failure = "BLOCKED"
            trial["launch_error"] = str(error)
            stderr.write(str(error).encode())
        except KeyboardInterrupt:
            failure, interrupted = "INTERRUPTED", True
        except (ValueError, IntegrityError) as error:
            failure = "PHASE_PROTOCOL_ERROR"
            trial["phase_error"] = str(error)
        finally:
            if proc is not None:
                if proc.poll() is None:
                    proc.kill()
                exit_code = proc.wait()
    if (directory / "worker.json").is_file():
        try:
            record = read_json(directory / "worker.json")
            # Worker output must not overwrite planner/block/provenance metadata.
            trial.update(record)
            verify_policy(entry, config, record)
            trial.update(entry)
            trial.update(environment=environment, config_hash=manifest["identity"]["config_hash"],
                code_sha256=manifest["identity"]["code_sha256"], experiment_id=manifest["experiment_id"],
                time_budget_s=request["planning_budget_s"])
            trial["policy_verified"] = True
        except (OSError, ValueError) as error:
            failure = "POLICY_OR_WORKER_ERROR"
            trial["policy_error"] = str(error)
    if failure:
        status = "BLOCKED" if failure == "BLOCKED" else "WATCHDOG" if failure.endswith("_WATCHDOG") or failure == "MEMORY_LIMIT" else "POLICY_ERROR" if failure == "POLICY_OR_WORKER_ERROR" else "CRASH"
        trial.update(execution_status=status, termination_reason=failure)
    elif exit_code != 0:
        trial.update(execution_status="CRASH", termination_reason="NONZERO_WORKER_EXIT")
    elif not (directory / "worker.json").is_file():
        trial.update(execution_status="CRASH", termination_reason="MISSING_WORKER_RESULT")
    elif (phase_read(directory / "phase.json") or {}).get("phase") != "FINISHED":
        trial.update(execution_status="CRASH", termination_reason="INCOMPLETE_PHASE_HANDSHAKE")
    trial.update(exit_code=exit_code, peak_rss_mb=peak if exit_code is not None else None,
        worker_wall_s=time.perf_counter()-began, watchdog_last_phase=phase,
        rss_measurement="sampled_peak_20ms_not_OS_exact_peak", success=False,
        runtime_session_id=session["session_id"], runtime_stratum_id=session["stratum_id"])
    if (directory / "ompl.log").is_file():
        trial.update(bitstar_log_counters((directory / "ompl.log").read_text(errors="replace")))
    dump(directory / "trial.json", trial)
    if interrupted:
        raise KeyboardInterrupt
    return trial


def retain_interrupted(directory, entry, manifest, config=None):
    directory = Path(directory)
    if not directory.exists() or (directory / "trial.json").is_file():
        return False
    trial = dict(entry, schema_version=2, experiment_id=manifest["experiment_id"],
        config_hash=manifest["identity"]["config_hash"], code_sha256=manifest["identity"]["code_sha256"],
        time_budget_s=manifest["planning_budget_s"], utc=utc(), execution_status="CRASH", planner_status="UNKNOWN",
        independent_valid=None, success=False, path_available=False, best_cost_terrain_m=None,
        first_solution_time_s=None, administrative_interruption=True,
        termination_reason="INTERRUPTED_PREVIOUS_RUN_RETAINED_NO_RETRY")
    if (directory / "attempt.json").is_file():
        trial["runtime_session_id"] = read_json(directory / "attempt.json").get("runtime_session_id")
    # If serialization finished before the parent was stopped, retain that
    # final path for independent checking too. The attempt remains a crash;
    # recovering metadata never turns an interrupted attempt into success.
    if (directory / "worker.json").is_file():
        try:
            record = read_json(directory / "worker.json")
            if config is not None:
                verify_policy(entry, config, record)
            trial.update(record)
            trial.update(entry)
            trial.update(execution_status="CRASH", success=False, administrative_interruption=True,
                termination_reason="INTERRUPTED_PREVIOUS_RUN_RETAINED_NO_RETRY")
        except (OSError, ValueError) as error:
            trial["worker_read_error"] = str(error)
    session_id = trial.get("runtime_session_id")
    if session_id and (directory.parent / "sessions" / (session_id+".json")).is_file():
        session = read_json(directory.parent / "sessions" / (session_id+".json"))
        trial["runtime_stratum_id"] = session["stratum_id"]
        trial["environment"] = session["environment"]
    dump(directory / "trial.json", trial)
    return True


def terminal_validation(directory, trial):
    if trial.get("validation_status") == "NO_REQUEST_INTERRUPTED_OR_BLOCKED":
        return True
    return (Path(directory) / "validation.json").is_file() and isinstance(trial.get("validation_wall_s"), (int, float))


def summarize(output, entries):
    counts = dict(planned=len(entries), attempted=0, executed=0, not_run=0, offline_validated=0,
                  validation_pending=0, validated_success=0, invalid=0, fatal=0, timeout=0)
    for entry in entries:
        directory = Path(output) / entry["trial_id"]
        if not directory.exists():
            counts["not_run"] += 1
            continue
        counts["attempted"] += 1
        if not (directory / "trial.json").is_file():
            counts["validation_pending"] += 1
            continue
        trial = read_json(directory / "trial.json")
        executed = trial.get("execution_status") not in {"BLOCKED", "NOT_RUN", "PLANNED"}
        counts["executed"] += executed
        counts["not_run"] += not executed
        validated = terminal_validation(directory, trial)
        counts["offline_validated"] += validated
        counts["validation_pending"] += not validated
        counts["validated_success"] += bool(trial.get("success")) and validated
        counts["invalid"] += bool(validated and ((trial.get("path_available") and trial.get("independent_valid") is False) or trial.get("snapshots_valid") is False))
        counts["fatal"] += trial.get("execution_status") in FATAL_EXECUTION
        counts["timeout"] += trial.get("execution_status") == "TIMEOUT"
    return counts


def checkpoint(output, manifest, phase, **details):
    counts = summarize(output, manifest["trials"])
    complete = counts["executed"] == counts["planned"] and counts["offline_validated"] == counts["planned"]
    status = "FAILED_CORRECTNESS" if counts["invalid"] else "STOPPED_FAILURE" if counts["fatal"] else "COMPLETE_VALIDATED" if complete else "PARTIAL"
    state = dict(schema_version=1, protocol_id=PROTOCOL_ID, status=status, phase=phase,
                 utc=utc(), counts=counts, **details)
    dump(Path(output) / "campaign_state.json", state)
    return state


def freeze_trial(directory):
    directory = Path(directory)
    records = inventory(directory, {"artifact_inventory.json"})
    dump(directory / "artifact_inventory.json", {"entries": records, "sha256": digest(records)})


def verify_completed(output, entries):
    for entry in entries:
        directory = Path(output) / entry["trial_id"]
        path = directory / "artifact_inventory.json"
        if path.is_file():
            frozen = read_json(path)
            if frozen.get("sha256") != digest(frozen.get("entries")):
                raise IntegrityError("trial artifact inventory checksum changed")
            check_inventory(directory, frozen["entries"], {"artifact_inventory.json"})
        elif (directory / "trial.json").is_file() and terminal_validation(directory, read_json(directory / "trial.json")):
            # A crash between validator completion and inventory commit leaves
            # a valid checkpoint; independently revalidate before sealing it.
            continue


VALIDATION_CODE = """import sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from validate_results import validate_trial
from independent_verifier import Oracle
from experiment_common import read_json
oracle=None
for name in sys.argv[2:]:
    directory=Path(name)
    request=read_json(directory/'request.json') if (directory/'request.json').exists() else None
    if request is not None and oracle is None: oracle=Oracle(request['map_path'])
    validate_trial(directory,oracle)
    print('validated '+name,flush=True)
"""


def validate_block(output, entries, dataset_root):
    output = Path(output)
    pending = [output / entry["trial_id"] for entry in entries
               if (output / entry["trial_id"] / "trial.json").is_file()
               and not (output / entry["trial_id"] / "artifact_inventory.json").is_file()]
    if not pending:
        return
    command = [sys.executable, "-c", VALIDATION_CODE, str(ROOT / "tools"), *(str(p.resolve()) for p in pending)]
    name = "validation_"+dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    began = time.perf_counter()
    with (output / (name+".stdout.log")).open("wb") as stdout, (output / (name+".stderr.log")).open("wb") as stderr:
        proc = subprocess.Popen(command, stdout=stdout, stderr=stderr, cwd=Path(dataset_root).resolve())
        try:
            exit_code = proc.wait()
        except KeyboardInterrupt:
            proc.kill()
            exit_code = proc.wait()
            dump(output / (name+".json"), dict(command=command, utc=utc(), exit_code=exit_code, administrative_interruption=True))
            raise
    dump(output / (name+".json"), dict(command=command, utc=utc(), exit_code=exit_code, elapsed_s=time.perf_counter()-began))
    if exit_code:
        raise IntegrityError("independent validator failed; raw paths and validation logs retained")
    for directory in pending:
        freeze_trial(directory)
        trial = read_json(directory / "trial.json")
        if (trial.get("path_available") and not trial.get("independent_valid")) or trial.get("snapshots_valid") is False:
            raise IntegrityError("independent path/snapshot validation failed")


def disk_ready(output, min_free_bytes, max_output_bytes):
    output = Path(output)
    used = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
    return shutil.disk_usage(output).free >= min_free_bytes and used < max_output_bytes


def verify_live_code(worker, identity):
    if (common.cloud_code_hash() != identity["code_sha256"] or file_hash(worker) != identity["worker_sha256"]
            or file_hash(common.cloud_ompl_library(worker)) != identity["ompl_library_sha256"]):
        raise IntegrityError("source/worker/library changed during campaign")


def run_campaign(config_path, map_manifest, dataset_root, output, worker, *, profile="smoke",
        allow_a3_pilot=False, dry_run=False, resume=False, build_gate=None, smoke_gate=None,
        session_elapsed_s=0.0, soft_stop_s=28800.0, min_free_bytes=2*1024**3,
        max_output_bytes=18*1024**3):
    from a3_pilot_protocol import plan_trials
    if profile not in {"smoke", "a3_pilot"}:
        raise IntegrityError("only smoke/a3_pilot supported; full prohibited")
    if profile == "a3_pilot" and not dry_run and not allow_a3_pilot:
        raise IntegrityError("pilot requires explicit --allow-a3-pilot")
    if profile == "a3_pilot" and not dry_run and sys.platform != "linux":
        raise IntegrityError("pilot collection is Linux/Kaggle only; Windows permits smoke and dry-run")
    if not math.isfinite(session_elapsed_s) or session_elapsed_s < 0 or not 0 < soft_stop_s <= 28800:
        raise IntegrityError("invalid session/soft deadline")
    began = time.monotonic()
    output, dataset_root, worker = Path(output).resolve(), Path(dataset_root).resolve(), Path(worker).resolve()
    config = load_config(config_path, profile)
    maps = read_json(map_manifest)
    validate_dataset(config, profile, maps, dataset_root, witnesses=not resume)
    entries = plan_trials(config, profile, maps)
    identity = common.cloud_identity(config, maps, worker)
    identity["plan_sha256"] = digest(entries)
    environment = common.cloud_environment(worker)
    # platform.processor() can be the generic architecture on Linux. Use the
    # observed CPU model for timing strata even when that generic value exists.
    cpuinfo = Path("/proc/cpuinfo")
    if sys.platform == "linux" and cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("model name"):
                environment["cpu"] = line.split(":", 1)[1].strip()
                break
    runtime = runtime_identity(environment)
    identity["runtime_identity_sha256"] = digest(runtime)
    if not dry_run:
        check_build_gate(build_gate, identity)
        if profile == "a3_pilot":
            check_smoke_gate(smoke_gate, identity, runtime)
    expected = dict(schema_version=1, protocol_id=PROTOCOL_ID, profile=profile,
        experiment_id=output.name, planned_trials=len(entries), planning_budget_s=config["profiles"][profile]["planning_budget_s"],
        identity=identity, trials=entries, attempt_policy="one_primary_attempt_no_retry_no_seed_replacement",
        map_path_policy="relative_to_explicit_dataset_root")
    if output.exists() and any(output.iterdir()):
        if not resume:
            raise IntegrityError("nonempty output requires --resume")
        manifest = read_json(output / "manifest.json")
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise IntegrityError("resume source/config/runtime/maps/worklist identity mismatch")
        if read_json(output / "resolved_config.json") != config or read_json(output / "map_manifest.json") != maps:
            raise IntegrityError("frozen config/map manifest altered")
        verify_completed(output, entries)
    else:
        output.mkdir(parents=True, exist_ok=True)
        manifest = dict(expected, created_utc=utc(), runtime_sessions=[])
        dump(output / "manifest.json", manifest)
        dump(output / "resolved_config.json", config)
        dump(output / "map_manifest.json", maps)
        dump(output / "environment.json", environment)
        dump(output / "runtime_identity.json", runtime)
    if dry_run:
        return checkpoint(output, manifest, "PLANNED_DRY_RUN")
    session_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S")+"_"+uuid.uuid4().hex[:8]
    stratum = digest({"platform": environment["os"], "cpu": environment["cpu"], "runtime_identity": runtime})
    session = dict(session_id=session_id, stratum_id=stratum, platform=environment["os"], cpu=environment["cpu"],
        environment_sha256=digest(environment), runtime_identity_sha256=digest(runtime), created_utc=utc(),
        elapsed_at_campaign_start_s=session_elapsed_s)
    dump(output / "sessions" / (session_id+".json"), dict(session, environment=environment))
    manifest["runtime_sessions"].append(session)
    dump(output / "manifest.json", manifest)
    groups = OrderedDict()
    for entry in entries:
        relative(entry["trial_id"])
        groups.setdefault(entry["block_order"], []).append(entry)
    try:
        for block, group in groups.items():
            for entry in group:
                retain_interrupted(output / entry["trial_id"], entry, manifest, config)
            # Finish validation of an interrupted block before opening another.
            validate_block(output, group, dataset_root)
            if all((output / e["trial_id"] / "trial.json").is_file() for e in group):
                continue
            if session_elapsed_s+time.monotonic()-began >= soft_stop_s:
                return checkpoint(output, manifest, "SOFT_DEADLINE", active_block=block)
            if not disk_ready(output, min_free_bytes, max_output_bytes):
                return checkpoint(output, manifest, "DISK_GUARD", active_block=block)
            verify_live_code(worker, identity)
            for entry in group:
                directory = output / entry["trial_id"]
                if (directory / "trial.json").is_file():
                    continue
                if not disk_ready(output, min_free_bytes, max_output_bytes):
                    validate_block(output, group, dataset_root)
                    return checkpoint(output, manifest, "DISK_GUARD", active_block=block)
                if file_hash(dataset_file(dataset_root, entry["map_path"])) != entry["map_hash"]:
                    raise IntegrityError("map changed immediately before worker launch")
                checkpoint(output, manifest, "ACTIVE_TRIAL", active_trial_id=entry["trial_id"])
                trial = launch_trial(entry, config, directory, worker, environment, manifest, dataset_root, session)
                checkpoint(output, manifest, "COLLECTION", active_block=block)
                print(f'{entry["shuffle_order"]+1}/{len(entries)} {entry["trial_id"]}: {trial["execution_status"]}/{trial["planner_status"]}', flush=True)
                if trial["execution_status"] in FATAL_EXECUTION:
                    validate_block(output, group, dataset_root)
                    return checkpoint(output, manifest, "HALTED_WORKER_FAILURE", active_trial_id=entry["trial_id"])
            validate_block(output, group, dataset_root)
            state = checkpoint(output, manifest, "BLOCK_VALIDATED", active_block=block)
            if state["counts"]["invalid"]:
                return state
        state = checkpoint(output, manifest, "VALIDATED")
        if profile == "smoke" and state["status"] == "COMPLETE_VALIDATED":
            dump(output / "smoke_gate.json", dict(schema_version=1, protocol_id=PROTOCOL_ID, status="PASS",
                planned_trials=52, validated_trials=52,
                identity={key: identity[key] for key in (*IDENTITY_GATE_KEYS, "config_hash")},
                runtime_identity_sha256=digest(runtime), manifest_sha256=file_hash(output / "manifest.json"), utc=utc()))
        return state
    except KeyboardInterrupt:
        checkpoint(output, manifest, "INTERRUPTED")
        raise
    except (OSError, ValueError) as error:
        state = checkpoint(output, manifest, "INTEGRITY_OR_VALIDATION_FAILURE", error=str(error))
        if not state["counts"]["invalid"]:
            state["status"] = "FAILED_INTEGRITY"
            dump(output / "campaign_state.json", state)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/a3_improvements_pilot_v1.yaml")
    parser.add_argument("--map-manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--profile", choices=("smoke", "a3_pilot"), default="smoke")
    parser.add_argument("--allow-a3-pilot", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--build-gate", type=Path)
    parser.add_argument("--smoke-gate", type=Path)
    parser.add_argument("--session-elapsed-s", type=float, default=0.0)
    parser.add_argument("--soft-stop-s", type=float, default=28800.0)
    parser.add_argument("--min-free-bytes", type=int, default=2*1024**3)
    parser.add_argument("--max-output-bytes", type=int, default=18*1024**3)
    args = vars(parser.parse_args(argv))
    args["config_path"] = args.pop("config")
    try:
        result = run_campaign(**args)
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(result, flush=True)
    return 0 if result["status"] in {"COMPLETE_VALIDATED", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
