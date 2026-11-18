"""Registered corridor-v4 campaigns with four spawn lanes and offline validation.

Every method in a map/query/seed block uses the same lane. Waves contain at
most four blocks; no independent verifier runs while a measured trial is alive.
The default is smoke. Pilot/full remain explicit opt-ins and full needs selection.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import multiprocessing
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

from corridor_v4_execution import (THREAD_ENV, append_journal, atomic_json as dump, digest,
    file_hash, process_alive, read, resource_preflight, run_spawn_lanes, set_thread_limits,
    terminate_tree, utc_now)

# Spawn imports this module before entering the task. Numerical packages are
# imported lazily below, only after these process-local environment limits.
set_thread_limits()
ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/a3t_corridor_v4.json"


def execution_gate(profile, *, allow_pilot=False, allow_full=False):
    if profile not in ("smoke", "pilot", "full"):
        raise ValueError("unknown profile")
    if profile == "pilot" and not allow_pilot:
        raise ValueError("pilot requires --allow-pilot; NOT_RUN")
    if profile == "full" and not allow_full:
        raise ValueError("full requires --allow-full and selected pilot; NOT_RUN")


def prepare(output, config_path=CONFIG, *, profile="smoke", selection_path=None,
            dry_run=False, workers=None, validation_workers=None, reuse_smoke_l0=None):
    import corridor_v4_protocol as protocol
    if reuse_smoke_l0 is not None and (profile != "smoke" or dry_run):
        raise ValueError("--reuse-smoke-l0 only applies to actual smoke preparation")
    config, output = protocol.load_config(config_path), Path(output).resolve()
    if workers is not None:
        config["execution"]["max_workers"] = workers
    if validation_workers is not None:
        config["execution"]["validation_workers"] = validation_workers
    protocol.validate_config(config)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("refusing to overwrite an existing campaign")
    selection = protocol.check_selection(selection_path, config) if selection_path else None
    if profile == "full" and not dry_run and selection is None:
        raise ValueError("full preparation requires a successful pilot selection artifact")
    l0_bundle = None
    if reuse_smoke_l0 is not None:
        from corridor_v4_l0_cache import prepare_smoke_l0_cache
        l0_bundle = prepare_smoke_l0_cache(Path(reuse_smoke_l0).resolve())
    selected_alpha = selection["selected_alpha"] if selection else None
    manifest = protocol.inventory(config, profile, selected_alpha)
    output.mkdir(parents=True, exist_ok=True)
    dump(output / "config.json", config)
    manifest.update(created_utc=utc_now(), sources=protocol.source_hashes(), maps=[],
        objective="surface_length_subject_to_immutable_single_corridor",
        native_baseline_objective="sem_length_turn_risk_paper_v2",
        timing=dict(max_workers=config["execution"]["max_workers"],
            validation_workers=config["execution"]["validation_workers"], threads_per_trial=1,
            start_method="spawn", phase_order=["collection", "independent_validation"],
            latency_origin="attempt_start_before_input_hash_field_build_and_process_launch"),
        pilot_status="NOT_RUN", full_status="NOT_RUN", sessions=[], waves=[])
    if l0_bundle is not None:
        cache_path = output / "smoke_l0_cache.json"
        dump(cache_path, l0_bundle)
        manifest["smoke_l0_cache"] = dict(file="smoke_l0_cache.json", file_sha256=file_hash(cache_path),
            content_sha256=digest(l0_bundle), source_directory=str(Path(reuse_smoke_l0).resolve()),
            scope="independent_finite_graph_L0_certificate_only; every_path_verdict_is_fresh",
            field_fingerprint_origin="rederived_during_cache_preparation_not_archived_at_original_solve")
    if selection:
        dump(output / "pilot_selection.json", selection)
        manifest.update(selection_sha256=file_hash(output / "pilot_selection.json"), selected_alpha=selected_alpha)
    if dry_run:
        dump(output / "manifest.json", manifest)
        return manifest
    import numpy as np
    from a3_traversability_protocol import fixture_geometry
    from g5_dataset import geometry_definition, qualify_queries
    from independent_verifier import Oracle
    from native_point_cloud import generate_cloud
    from terrain_compare_protocol import MODEL
    from terrain_io import write_map
    settings = config["profiles"][profile]
    try:
        for family in settings["families"]:
            for seed in settings["map_seeds"]:
                extent = 40. if family == "flat_40m" else settings["extent_m"]
                definitions = None
                if profile == "smoke":
                    if family == "flat_40m":
                        height, rectangles, start, goal, label = (lambda x, y: 0.), [], [2., 20.], [38., 20.], "flat_full_scale_control"
                    else:
                        height, rectangles, start, goal, _, label = fixture_geometry(family)
                    params = dict(fixed_fixture=True, geometry_label=label)
                else:
                    height, rectangles, definitions, params = geometry_definition(family, seed)
                    definitions = definitions[:1]
                    start, goal = definitions[0]["start"], definitions[0]["goal"]
                spacing, map_id = settings["spacing_m"], f"{family}_{seed}"
                n = round(extent / spacing) + 1
                path = output / "inputs" / (map_id + ".t25d")
                metadata = write_map(path, [[height(i*spacing, j*spacing) for i in range(n)] for j in range(n)],
                    spacing=(spacing, spacing), rectangles=rectangles,
                    metadata=dict(map_id=map_id, family=family, generator_seed=seed,
                        generator_version="a3t_corridor_v4", parameters=params,
                        purpose="synthetic_geometry_for_real_planner; not_measured_sensor_data"))
                if definitions is not None:
                    metadata["queries"] = qualify_queries(Oracle(path), definitions, file_hash(path), MODEL, clearance_metrics=False)
                    dump(path.with_suffix(".json"), metadata)
                cloud = path.with_suffix(".cloud.npy")
                np.save(cloud, generate_cloud(path), allow_pickle=False)
                manifest["maps"].append(dict(map_id=map_id, family=family, map_seed=seed, map_path=str(path),
                    map_sha256=file_hash(path), metadata_sha256=file_hash(path.with_suffix(".json")),
                    cloud_path=str(cloud), cloud_sha256=file_hash(cloud), bounds=[0., 0., extent, extent],
                    start=list(start), goal=list(goal), query_id="q0"))
        manifest["status"] = "PREPARED"
    except Exception as error:
        manifest["status"] = "PREPARATION_FAILED"
        dump(output / "qualification_failure.json", dict(error=str(error), traceback=traceback.format_exc(),
             seed_replacements=[], planner_execution_count=0))
        dump(output / "manifest.json", manifest)
        raise
    dump(output / "manifest.json", manifest)
    return manifest


def raw_inventory(directory):
    return {path.relative_to(directory).as_posix(): file_hash(path)
            for path in sorted(directory.rglob("*")) if path.is_file() and
            path.name not in ("validation.json", "raw_inventory.json")}


def _verify_inputs(info):
    for key in ("map", "cloud"):
        if file_hash(info[key + "_path"]) != info[key + "_sha256"]:
            raise ValueError("frozen input changed: " + key)
    if info.get("metadata_sha256") and file_hash(Path(info["map_path"]).with_suffix(".json")) != info["metadata_sha256"]:
        raise ValueError("frozen map metadata changed")


def _sample_rss():
    import psutil
    process = psutil.Process()
    rss = 0
    for item in [process, *process.children(recursive=True)]:
        try:
            rss += item.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    return rss


def _collect_trial(task):
    """One fresh spawn process; own field build and one actual planner process."""
    import psutil
    import corridor_v4_protocol as protocol
    from safe_length_field import build_field_file
    from corridor_v4_validation import record_corridor_receipt
    directory, info, entry = Path(task["directory"]), task["info"], task["entry"]
    directory.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter()
    execution = dict(status="STARTED", field_build_s=0., command=None, exit_code=None,
        started_utc=utc_now(), launched=False, collection_pid=os.getpid(),
        collection_process_created=psutil.Process().create_time(),
        cpu_affinity=psutil.Process().cpu_affinity(),
        thread_environment={key: os.environ[key] for key in THREAD_ENV},
        queue_wait_s=task.get("queue_wait_s"), peak_process_tree_rss_bytes=_sample_rss())
    dump(directory / "execution.json", execution)
    process = None
    try:
        _verify_inputs(info)
        if file_hash(task["worker"]) != task["binary_sha256"]:
            raise ValueError("frozen worker changed")
        native = entry["method"] == "sem_native"
        metadata = None
        field_path = directory / "planner_sem_field.txt"
        if native:
            if not task.get("bridge") or file_hash(task["bridge"]) != task["bridge_sha256"]:
                raise ValueError("native bridge missing or changed")
        else:
            started = time.perf_counter()
            metadata = build_field_file(info["cloud_path"], info["map_path"], info["bounds"], field_path)
            execution["field_build_s"] = time.perf_counter() - started
            if file_hash(field_path) != metadata["field_sha256"]:
                raise ValueError("field hash mismatch before launch")
        request = protocol.request_for(entry, info, task["config"], task["profile"], field_path,
                                       metadata, task.get("bridge"))
        dump(directory / "request.json", request)
        execution["request_sha256"] = file_hash(directory / "request.json")
        command = ([task["sem_python"], str(ROOT / "tools/native_paper_worker.py")]
                   if native else [task["worker"]]) + [str(directory / "request.json"), str(directory)]
        environment = os.environ.copy()
        if os.name == "nt":
            environment["PATH"] = os.pathsep.join((str(Path(task["worker"]).parent), str(ROOT / ".deps/install/lib"),
                "C:/mingw64/bin", str(Path(task["sem_python"]).parent / "Library/bin"), environment.get("PATH", "")))
        execution.update(command=command, launch_utc=utc_now(), status="RUNNING")
        dump(directory / "execution.json", execution)
        launched = time.perf_counter()
        with (directory / "stdout.log").open("wb") as stdout, (directory / "stderr.log").open("wb") as stderr:
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=environment, cwd=ROOT)
            try:
                process_created = psutil.Process(process.pid).create_time()
            except psutil.NoSuchProcess:
                process_created = None
            execution.update(launched=True, pid=process.pid, process_created=process_created)
            dump(directory / "execution.json", execution)
            while process.poll() is None:
                execution["peak_process_tree_rss_bytes"] = max(execution["peak_process_tree_rss_bytes"], _sample_rss())
                if time.perf_counter() - launched > task["config"]["execution"]["process_watchdog_s"]:
                    execution.update(status="WATCHDOG", reason="registered process watchdog exceeded")
                    terminate_tree(process.pid)
                    break
                time.sleep(.02)
            code = process.wait()
            execution.update(exit_code=code, process_wall_s=time.perf_counter() - launched,
                             process_finished_utc=utc_now())
            if execution["status"] != "WATCHDOG":
                execution["status"] = "COMPLETED" if code == 0 else "CRASH"
            # Durable actual subprocess exit proof before receipt construction,
            # inventories, result IPC or any independent verification.
            dump(directory / "process_exit.json", dict(pid=process.pid, exit_code=code,
                 process_created=execution["process_created"], status=execution["status"],
                 process_wall_s=execution["process_wall_s"], finished_utc=execution["process_finished_utc"]))
            dump(directory / "execution.json", execution)
        if (directory / "worker.json").is_file() and read(directory / "worker.json").get("status") == "BLOCKED_DEPENDENCY":
            execution.update(status="BLOCKED", reason=read(directory / "worker.json").get("error"))
        if not native and file_hash(field_path) != metadata["field_sha256"]:
            raise ValueError("field changed during trial")
        if file_hash(directory / "request.json") != execution["request_sha256"]:
            raise ValueError("request changed during trial")
        record_corridor_receipt(directory)
    except Exception as error:
        execution.update(status="FAILED", reason=str(error), traceback=traceback.format_exc())
    finally:
        if process is not None and process.poll() is None:
            terminate_tree(process.pid)
            execution.update(exit_code=process.wait())
    execution.update(finished_utc=utc_now(), attempt_wall_before_validation_s=time.perf_counter() - began,
                     resource_measurement="sampled_20ms_collection_process_tree; short_lived_processes_may_be_missed")
    dump(directory / "execution.json", execution)
    inventory = raw_inventory(directory)
    dump(directory / "raw_inventory.json", inventory)
    return dict(trial_id=entry["trial_id"], execution_sha256=file_hash(directory / "execution.json"),
                raw_inventory_sha256=digest(inventory), execution=execution)


def _execution(directory):
    recovery = directory / "execution_recovery.json"
    return read(recovery if recovery.is_file() else directory / "execution.json")


def _validate_map(task):
    import numpy as np
    from corridor_v4_validation import validate_trial
    from native_paper_oracle import NativePaperOracle
    from native_paper_v3_sem_evaluator import SharedSEMField
    info = task["info"]
    _verify_inputs(info)
    oracle = NativePaperOracle(info["map_path"])
    field = SharedSEMField(np.load(info["cloud_path"], allow_pickle=False), info["bounds"])
    l0_preload = None
    if task.get("smoke_l0_cache") is not None:
        from corridor_v4_l0_cache import seed_map_l0_cache, validate_smoke_l0_cache
        validate_smoke_l0_cache(task["smoke_l0_cache"], task["smoke_l0_cache_content_sha256"])
        l0_preload = seed_map_l0_cache(task["smoke_l0_cache"], info, oracle, field, profile=task["profile"])
    completed = []
    began = time.perf_counter()
    for item in task["items"]:
        directory = Path(item["directory"])
        inventory = read(directory / "raw_inventory.json")
        if digest(inventory) != item["raw_inventory_sha256"] or raw_inventory(directory) != inventory:
            raise ValueError("raw inventory changed before validation: " + item["entry"]["trial_id"])
        spool = Path(task["spool"]) / (item["entry"]["trial_id"] + ".json")
        identity = dict(trial_id=item["entry"]["trial_id"], raw_inventory_sha256=digest(inventory),
                        sources_sha256=task["sources_sha256"], info_sha256=digest(info))
        if spool.exists():
            if read(spool).get("identity") != identity:
                raise ValueError("validation spool identity mismatch")
            completed.append(item["entry"]["trial_id"])
            continue
        started = time.perf_counter()
        execution = _execution(directory)
        if (directory / "request.json").is_file():
            row = validate_trial(directory, read(directory / "request.json"), info, execution,
                                 oracle=oracle, field=field)
        else:
            row = dict(success=False, attained_safe=False, execution_status=execution["status"],
                       correctness_errors=[], events=[], unavailable_reason="no_request_before_launch")
        entry = item["entry"]
        row.update(trial_id=entry["trial_id"], method=entry["method"], map_id=entry["map_id"],
                   planner_seed=entry["planner_seed"], family=entry["family"], alpha=entry["alpha"],
                   policy=entry["policy"], execution_launched=execution.get("launched", False),
                   queue_wait_s=entry.get("queue_wait_s"),
                   peak_rss_bytes=execution.get("peak_process_tree_rss_bytes"),
                   validation_s=time.perf_counter() - started)
        if l0_preload is not None:
            certificate = row.get("corridor", {}).get("L0_certificate", {})
            row["smoke_l0_cache_preload"] = l0_preload
            row["smoke_l0_cache_reused"] = bool(certificate.get("smoke_reuse_binding_sha256"))
        dump(spool, dict(identity=identity, row=row, validator_pid=os.getpid(), created_utc=utc_now()))
        completed.append(entry["trial_id"])
    return dict(map_id=info["map_id"], completed_trial_ids=completed,
                validation_s=time.perf_counter() - began, process_tree_rss_bytes_at_finish=_sample_rss(),
                smoke_l0_cache_preload=l0_preload)


def _acquire_lock(output):
    import psutil
    path = output / "controller.lock"
    if path.exists():
        prior = read(path)
        if process_alive(prior.get("pid"), prior.get("process_created")):
            raise ValueError("another campaign controller is still alive")
        append_journal(output / "execution_control" / "stale_locks.jsonl", prior)
        path.unlink()
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        import json
        json.dump(dict(pid=os.getpid(), process_created=psutil.Process().create_time(), started_utc=utc_now()), stream)
        stream.flush()
        os.fsync(stream.fileno())
    return path


def _reconcile(output, manifest):
    """A started identity is never made eligible for dispatch again."""
    from corridor_v4_validation import record_corridor_receipt
    for proof in (output / "execution_control").glob("**/child.json"):
        value = read(proof)
        if process_alive(value.get("pid"), value.get("process_created")):
            raise ValueError("previous spawned child still alive; refuse duplicate controller")
    journal = output / "execution_control" / "start_journal.jsonl"
    starts = {}
    if journal.is_file():
        import json
        for line in journal.read_text(encoding="utf-8").splitlines():
            value = json.loads(line)
            if value["trial_id"] in starts:
                raise ValueError("duplicate trial start journal")
            if (value.get("config_sha256") != manifest["config_sha256"] or
                    value.get("sources_sha256") != digest(manifest["sources"])):
                raise ValueError("start journal provenance changed")
            starts[value["trial_id"]] = value
    known = {entry["trial_id"] for entry in manifest["entries"]}
    if set(starts) - known:
        raise ValueError("unknown trial in start journal")
    for entry in manifest["entries"]:
        directory = output / "trials" / entry["trial_id"]
        if entry.get("validation_sha256"):
            if (file_hash(directory / "validation.json") != entry["validation_sha256"] or
                    raw_inventory(directory) != read(directory / "raw_inventory.json")):
                raise ValueError("committed trial changed")
            continue
        if entry["trial_id"] not in starts:
            if entry["status"] != "NOT_RUN" or directory.exists():
                raise ValueError("trial state exists without durable start journal")
            continue
        directory.mkdir(parents=True, exist_ok=True)
        if (directory / "execution.json").is_file():
            execution = read(directory / "execution.json")
        else:
            execution = dict(status="INTERRUPTED", exit_code=None, launched=False,
                             started_utc=starts[entry["trial_id"]]["started_utc"])
        if process_alive(execution.get("pid"), execution.get("process_created")):
            raise ValueError("previous actual planner still alive")
        if (directory / "process_exit.json").is_file():
            receipt = read(directory / "process_exit.json")
            if execution.get("exit_code") not in (None, receipt["exit_code"]):
                raise ValueError("subprocess exit proof disagrees with execution")
            execution.update(exit_code=receipt["exit_code"], process_wall_s=receipt["process_wall_s"])
            if execution.get("status") in ("STARTED", "RUNNING"):
                execution["status"] = receipt["status"]
        if execution.get("status") in ("STARTED", "RUNNING") or not (directory / "execution.json").is_file():
            execution.update(status="INTERRUPTED", reason="controller_interrupted; no retry of started attempt")
        # Preserve the original record; reconciliation has its own visible receipt.
        if not (directory / "execution.json").is_file() or execution != read(directory / "execution.json"):
            dump(directory / "execution_recovery.json", execution)
        record_corridor_receipt(directory)
        inventory = raw_inventory(directory)
        if (directory / "raw_inventory.json").is_file() and inventory != read(directory / "raw_inventory.json"):
            raise ValueError("collected raw trial changed before resume")
        dump(directory / "raw_inventory.json", inventory)
        entry.update(status="RAW_READY", execution_status=execution["status"], raw_inventory_sha256=digest(inventory))


def _commit_spool(output, entry, manifest, spool_dir):
    path = spool_dir / (entry["trial_id"] + ".json")
    if not path.is_file() or entry.get("validation_sha256"):
        return
    value = read(path)
    directory = output / "trials" / entry["trial_id"]
    expected = dict(trial_id=entry["trial_id"], raw_inventory_sha256=entry["raw_inventory_sha256"],
                    sources_sha256=digest(manifest["sources"]),
                    info_sha256=digest(next(info for info in manifest["maps"] if info["map_id"] == entry["map_id"])))
    if value.get("identity") != expected:
        raise ValueError("spool differs from frozen trial identity")
    if raw_inventory(directory) != read(directory / "raw_inventory.json"):
        raise ValueError("raw trial changed during independent validation")
    row = value["row"]
    if row.get("trial_id") != entry["trial_id"]:
        raise ValueError("spool row trial identity mismatch")
    corridor_receipt = directory / "corridor_receipt.json"
    if corridor_receipt.is_file() and row.get("corridor_status") == "GUIDANCE_READY":
        receipt = read(corridor_receipt)
        key = digest([entry["map_id"], entry["query_id"], entry["alpha"], entry["policy"]])
        identity = {name: receipt.get(name) for name in ("mask_sha256", "guide_sha256", "raw_guide_sha256", "preview_sha256")}
        matched = manifest.setdefault("matched_corridors", {})
        if key in matched and matched[key] != identity:
            row["correctness_errors"].append("matched_corridor_or_guide_mismatch")
            row["success"] = False
            row["attained_safe"] = False
        else:
            matched[key] = identity
    dump(directory / "validation.json", row)
    entry.update(status=row["execution_status"], success=row["success"],
                 validation_sha256=file_hash(directory / "validation.json"), validation_spool_sha256=file_hash(path))


def actual_planner_overlap(entries):
    """Peak of recorded subprocess intervals; collection wrappers are excluded."""
    events, measured, unavailable = [], 0, 0
    for entry in entries:
        interval = entry.get("planner_interval", {})
        if not interval.get("launch_utc") or not interval.get("process_finished_utc"):
            if entry["status"] != "NOT_RUN":
                unavailable += 1
            continue
        start = datetime.fromisoformat(interval["launch_utc"].replace("Z", "+00:00"))
        finish = datetime.fromisoformat(interval["process_finished_utc"].replace("Z", "+00:00"))
        if finish < start:
            raise ValueError("negative recorded planner interval")
        events.extend(((start, 1), (finish, -1)))
        measured += 1
    running, peak = 0, 0
    for _, delta in sorted(events):
        running += delta
        peak = max(peak, running)
    return dict(max_concurrency=peak, recorded_intervals=measured, unavailable_intervals=unavailable,
                measurement="launch_before_Popen_to_observed_process_exit; includes_subprocess_startup_and_polling")


def run(output, worker, bridge=None, sem_python=None, *, profile="smoke", allow_pilot=False,
        allow_full=False, workers=None, validation_workers=None, resume=False):
    execution_gate(profile, allow_pilot=allow_pilot, allow_full=allow_full)
    import corridor_v4_protocol as protocol
    output, worker = Path(output).resolve(), Path(worker).resolve()
    config, manifest = protocol.load_config(output / "config.json"), read(output / "manifest.json")
    execution_config = config["execution"]
    for value, key in ((workers, "max_workers"), (validation_workers, "validation_workers")):
        if value is not None and value != execution_config[key]:
            raise ValueError("worker count differs from frozen prepared configuration")
    workers, validation_workers = execution_config["max_workers"], execution_config["validation_workers"]
    if manifest["profile"] != profile or manifest["status"] == "PREPARED_INVENTORY":
        raise ValueError("campaign is not prepared for this execution profile")
    if not resume and manifest["status"] != "PREPARED":
        raise ValueError("campaign was already attempted; --resume required")
    if manifest.get("status") == "FAILED_CORRECTNESS":
        raise ValueError("correctness stop requires a new protocol/campaign")
    if manifest["config_sha256"] != digest(config) or manifest["sources"] != protocol.source_hashes():
        raise ValueError("frozen source/config changed after preparation")
    l0_bundle = None
    if manifest.get("smoke_l0_cache") is not None:
        if profile != "smoke":
            raise ValueError("archived L0 certificate reuse is restricted to smoke")
        from corridor_v4_l0_cache import validate_smoke_l0_cache
        cache = manifest["smoke_l0_cache"]
        if cache.get("file") != "smoke_l0_cache.json":
            raise ValueError("unexpected smoke L0 cache path")
        cache_path = output / cache["file"]
        if file_hash(cache_path) != cache["file_sha256"]:
            raise ValueError("frozen smoke L0 cache bytes changed")
        l0_bundle = read(cache_path)
        validate_smoke_l0_cache(l0_bundle, cache["content_sha256"])
    expected = protocol.inventory(config, profile, manifest.get("selected_alpha"))["entries"]
    for left, right in zip(manifest["entries"], expected, strict=True):
        if {key: left[key] for key in right if key != "status"} != {key: value for key, value in right.items() if key != "status"}:
            raise ValueError("frozen schedule changed")
    bridge = Path(bridge).resolve() if bridge else None
    sem_python = Path(sem_python or sys.executable).resolve()
    if not worker.is_file() or not sem_python.is_file() or bridge is None or not bridge.is_file():
        raise FileNotFoundError("BLOCKED: worker, SEM interpreter and bridge dependencies are required")
    identity = dict(binary_sha256=file_hash(worker), bridge_sha256=file_hash(bridge), sem_python=str(sem_python))
    if resume and any(manifest.get(key) != value for key, value in identity.items()):
        raise ValueError("resume binary/interpreter identity mismatch")
    if profile == "full":
        selected = output / "pilot_selection.json"
        if not selected.is_file() or file_hash(selected) != manifest["selection_sha256"]:
            raise ValueError("full selection artifact changed")
        protocol.check_selection(selected, config, binary_sha256=identity["binary_sha256"])
    lock = _acquire_lock(output)
    began = time.perf_counter()
    session = None
    try:
        if resume:
            _reconcile(output, manifest)
        manifest.update(identity, status="RUNNING", worker_path=str(worker), worker_sha256=identity["binary_sha256"],
                        environment=dict(python=sys.version, platform=platform.platform(),
                            thread_environment={key: os.environ[key] for key in THREAD_ENV}))
        session_index = len(manifest.setdefault("sessions", []))
        session = dict(session_index=session_index, started_utc=utc_now(), workers=workers,
                       validation_workers=validation_workers, timing_mode="concurrent_trials" if workers == 4 else "serial_trials",
                       collection_s=0., validation_s=0., max_concurrency=0)
        manifest["sessions"].append(session)
        dump(output / "manifest.json", manifest)
        maps = {info["map_id"]: info for info in manifest["maps"]}
        blocks = defaultdict(list)
        for entry in manifest["entries"]:
            blocks[entry["block_index"]].append(entry)
        ordered = sorted(blocks)
        for offset in range(0, len(ordered), 4):
            wave_id = f"wave_{offset // 4:04d}"
            wave_entries = [entry for block in ordered[offset:offset + 4] for entry in blocks[block]]
            if all(entry.get("validation_sha256") for entry in wave_entries):
                continue
            wave = dict(wave_id=wave_id, session_index=session_index, block_indices=ordered[offset:offset+4],
                        phase="COLLECTION", started_utc=utc_now())
            manifest.setdefault("waves", []).append(wave)
            control = output / "execution_control" / f"session_{session_index:04d}" / wave_id
            lanes = [[] for _ in range(workers)]
            for index, block in enumerate(ordered[offset:offset + 4]):
                for entry in blocks[block]:
                    if entry["status"] != "NOT_RUN":
                        continue
                    lanes[index % workers].append(dict(task_id=entry["trial_id"], entry=dict(entry),
                        directory=str(output / "trials" / entry["trial_id"]), info=maps[entry["map_id"]],
                        config=config, profile=profile, worker=str(worker), bridge=str(bridge),
                        sem_python=str(sem_python), binary_sha256=identity["binary_sha256"], bridge_sha256=identity["bridge_sha256"]))
            entry_by_id = {entry["trial_id"]: entry for entry in wave_entries}

            def on_start(task, receipt):
                entry = entry_by_id[task["task_id"]]
                if entry["status"] != "NOT_RUN":
                    raise ValueError("attempt already started")
                append_journal(output / "execution_control" / "start_journal.jsonl",
                    dict(trial_id=entry["trial_id"], started_utc=receipt["started_utc"],
                         config_sha256=manifest["config_sha256"], sources_sha256=digest(manifest["sources"]),
                         session_index=session_index, wave_id=wave_id, lane=receipt["lane"], core=receipt["core"]))
                entry.update(status="STARTED", lane=receipt["lane"], core=receipt["core"],
                             queue_wait_s=receipt["queue_wait_s"], started_utc=receipt["started_utc"])
                dump(output / "manifest.json", manifest)

            def on_complete(task, value):
                entry = entry_by_id[task["task_id"]]
                if value["status"] == "COMPLETED":
                    payload = value["payload"]
                    measured = payload["execution"]
                    entry.update(status="RAW_READY", execution_status=payload["execution"]["status"],
                        collection_sha256=digest(payload), raw_inventory_sha256=payload["raw_inventory_sha256"],
                        planner_interval={key: measured.get(key) for key in
                            ("pid", "process_created", "launch_utc", "process_finished_utc", "exit_code")})
                dump(output / "manifest.json", manifest)

            if any(lanes):
                admission = resource_preflight(workers, execution_config["reserve_bytes_per_collection_worker"])
                wave["collection_admission"] = admission
                if not admission["admitted"]:
                    manifest.update(status="BLOCKED_RESOURCES", blocked_reason=admission["reason"])
                    break
                collected = run_spawn_lanes(lanes, _collect_trial, control / "collection", cores=admission["cores"],
                                             on_start=on_start, on_complete=on_complete)
            else:
                collected = dict(status="COMPLETE", wall_s=0., max_concurrency=0,
                                 skipped="all_wave_attempts_already_collected", concurrency_timeline=[])
                dump(control / "collection" / "summary.json", collected)
                wave["collection_admission"] = dict(admitted=True, skipped="validation_only_resume")
            wave.update(collection_s=collected["wall_s"], max_concurrency=collected["max_concurrency"],
                        collection_summary_sha256=file_hash(control / "collection" / "summary.json"))
            session["collection_s"] += collected["wall_s"]
            session["max_concurrency"] = max(session["max_concurrency"], collected["max_concurrency"])
            if collected["status"] != "COMPLETE":
                manifest["status"] = collected["status"]
                break
            wave.update(phase="INDEPENDENT_VALIDATION", collection_finished_utc=utc_now())
            dump(output / "manifest.json", manifest)
            validation_admission = resource_preflight(validation_workers, execution_config["reserve_bytes_per_validation_worker"])
            wave["validation_admission"] = validation_admission
            if not validation_admission["admitted"]:
                manifest.update(status="BLOCKED_RESOURCES", blocked_reason=validation_admission["reason"])
                break
            spool = output / "execution_control" / "validation_spool"
            for entry in wave_entries:
                _commit_spool(output, entry, manifest, spool)
            grouped = defaultdict(list)
            for entry in wave_entries:
                if not entry.get("validation_sha256") and entry["status"] == "RAW_READY":
                    grouped[entry["map_id"]].append(dict(entry=dict(entry), directory=str(output / "trials" / entry["trial_id"]),
                                                        raw_inventory_sha256=entry["raw_inventory_sha256"]))
            validation_lanes = [[] for _ in range(validation_workers)]
            for index, (map_id, items) in enumerate(grouped.items()):
                validation_lanes[index % validation_workers].append(dict(task_id=map_id, info=maps[map_id], items=items,
                    sources_sha256=digest(manifest["sources"]), spool=str(spool), profile=profile,
                    smoke_l0_cache=l0_bundle,
                    smoke_l0_cache_content_sha256=manifest.get("smoke_l0_cache", {}).get("content_sha256")))

            def commit_validation(task, value):
                for entry in wave_entries:
                    _commit_spool(output, entry, manifest, spool)
                dump(output / "manifest.json", manifest)

            verified = run_spawn_lanes(validation_lanes, _validate_map, control / "validation",
                cores=validation_admission["cores"], on_complete=commit_validation)
            wave.update(validation_s=verified["wall_s"], validation_finished_utc=utc_now(), phase="VALIDATED",
                        validation_summary_sha256=file_hash(control / "validation" / "summary.json"))
            session["validation_s"] += verified["wall_s"]
            if verified["status"] != "COMPLETE":
                manifest["status"] = verified["status"]
                break
            if any(read(output / "trials" / entry["trial_id"] / "validation.json")["correctness_errors"]
                   for entry in wave_entries if entry.get("validation_sha256")):
                manifest["status"] = "FAILED_CORRECTNESS"
                break
            dump(output / "manifest.json", manifest)
        else:
            manifest["status"] = "COMPLETE" if all(entry.get("validation_sha256") for entry in manifest["entries"]) else "INCOMPLETE"
    except KeyboardInterrupt:
        manifest["status"] = "INTERRUPTED"
    except Exception as error:
        manifest.update(status="FAILED_INFRASTRUCTURE", error=str(error), traceback=traceback.format_exc())
    finally:
        if session is not None:
            session.update(finished_utc=utc_now(), session_wall_s=time.perf_counter() - began)
        manifest["updated_utc"] = utc_now()
        dump(output / "manifest.json", manifest)
        lock.unlink(missing_ok=True)
    rows = [read(output / "trials" / entry["trial_id"] / "validation.json")
            for entry in manifest["entries"] if entry.get("validation_sha256")]
    attempted = sum(entry["status"] != "NOT_RUN" for entry in manifest["entries"])
    summary = dict(protocol_id=protocol.PROTOCOL_ID, schema_version=4, profile=profile, status=manifest["status"],
        planned=len(manifest["entries"]), attempted=attempted, validated=len(rows),
        success=sum(row["success"] for row in rows), correctness_failures=sum(bool(row["correctness_errors"]) for row in rows),
        rows=rows, requested_workers=workers, validation_workers=validation_workers,
        max_concurrency=max((session["max_concurrency"] for session in manifest["sessions"]), default=0),
        collection_s=sum(session["collection_s"] for session in manifest["sessions"]),
        validation_s=sum(session["validation_s"] for session in manifest["sessions"]),
        session_wall_s=sum(session["session_wall_s"] for session in manifest["sessions"]),
        pilot_status="COMPLETE" if profile == "pilot" and manifest["status"] == "COMPLETE" else "NOT_RUN",
        full_status="COMPLETE" if profile == "full" and manifest["status"] == "COMPLETE" else "NOT_RUN",
        inference="smoke_only_no_performance_claim" if profile == "smoke" else "requires_registered_analysis_gates")
    summary["actual_planner_overlap"] = actual_planner_overlap(manifest["entries"])
    dump(output / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "smoke", "inventory"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--profile", choices=("smoke", "pilot", "full"), default="smoke")
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--bridge", type=Path)
    parser.add_argument("--sem-python", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--reuse-smoke-l0", type=Path,
                        help="reuse verified independent L0 certificates from a completed smoke; never path verdicts")
    parser.add_argument("--workers", type=int, choices=(1, 4))
    parser.add_argument("--validation-workers", type=int, choices=(1, 4))
    parser.add_argument("--allow-pilot", action="store_true")
    parser.add_argument("--allow-full", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "smoke" and args.profile != "smoke":
        parser.error("smoke command only accepts the smoke profile")
    if args.resume and args.command != "run":
        parser.error("--resume only applies to run")
    if args.reuse_smoke_l0 is not None and (args.command not in ("prepare", "smoke") or
                                           args.profile != "smoke" or args.dry_run):
        parser.error("--reuse-smoke-l0 is fixed at actual smoke prepare/smoke, never at run/resume")
    if args.command in ("run", "smoke") and not args.dry_run and not args.worker:
        parser.error("--worker required for execution")
    if args.command in ("prepare", "smoke", "inventory") or args.dry_run:
        result = prepare(args.output, args.config, profile=args.profile, selection_path=args.selection,
            dry_run=args.dry_run or args.command == "inventory", workers=args.workers, validation_workers=args.validation_workers,
            reuse_smoke_l0=args.reuse_smoke_l0)
    if args.command in ("run", "smoke") and not args.dry_run:
        result = run(args.output, args.worker, args.bridge, args.sem_python, profile=args.profile,
            allow_pilot=args.allow_pilot, allow_full=args.allow_full, workers=args.workers,
            validation_workers=args.validation_workers, resume=args.resume)
    import json
    print(json.dumps({key: result[key] for key in ("status", "planned_trials", "planned", "attempted", "success",
                                                "correctness_failures", "max_concurrency") if key in result}))
    return 0 if result["status"] in ("PREPARED", "PREPARED_INVENTORY", "COMPLETE") else 1


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
