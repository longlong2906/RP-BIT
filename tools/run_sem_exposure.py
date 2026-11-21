"""Collect the frozen SEM-exposure smoke, then independently validate all raw trials.

No pilot/full runner is provided. Started identities are never retried; the
existing v4 spawn/journal primitives provide durable exits and private lanes.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import multiprocessing
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

from corridor_v4_execution import (THREAD_ENV, append_journal, atomic_json as dump,
    digest, file_hash, read, resource_preflight, run_spawn_lanes, set_thread_limits,
    terminate_tree, utc_now)
from run_corridor_v4 import (_acquire_lock, _commit_spool, _execution, _reconcile,
    _sample_rss, _verify_inputs, actual_planner_overlap, raw_inventory)
from sem_exposure_runtime import executable_path

set_thread_limits()
ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/a3t_sem_exposure_v1.json"


def _collect_trial(task, *, request_builder=None, is_native=None, field_exchange=None):
    """One fresh spawn process; own field build and one actual planner process.

    `is_native(entry)` selects the trials run by the native worker (default: the method `sem_native`).
    `field_exchange(entry)` may return another (file name, builder) for that trial's planner SEM field."""
    import psutil
    import sem_exposure_protocol as protocol
    from safe_length_field import build_field_file
    from sem_exposure_validation import record_receipt
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
        native = is_native(entry) if is_native else entry["method"] == "sem_native"
        metadata = None
        name, build = (field_exchange(entry) if field_exchange else None) or ("planner_sem_field.txt", build_field_file)
        field_path = directory / name
        if native:
            if not task.get("bridge") or file_hash(task["bridge"]) != task["bridge_sha256"]:
                raise ValueError("native bridge missing or changed")
        else:
            # Loading the library is process start-up, not field construction: it happens before the clock.
            from scipy import ndimage  # noqa: F401
            started = time.perf_counter()
            metadata = build(info["cloud_path"], info["map_path"], info["bounds"], field_path)
            execution["field_build_s"] = time.perf_counter() - started
            if file_hash(field_path) != metadata["field_sha256"]:
                raise ValueError("field hash mismatch before launch")
        builder = request_builder or protocol.request_for
        request = builder(entry, info, task["config"], task["profile"], field_path,
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
        record_receipt(directory)
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


def prepare(output, config_path=CONFIG, *, dry_run=False, workers=None, validation_workers=None):
    import sem_exposure_protocol as protocol
    config, output = protocol.load_config(config_path), Path(output).resolve()
    for value, key in ((workers, "max_workers"), (validation_workers, "validation_workers")):
        if value is not None:
            config["execution"][key] = value
    protocol.validate_config(config)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("refusing to overwrite an existing campaign")
    manifest = protocol.inventory(config)
    output.mkdir(parents=True, exist_ok=True)
    dump(output / "config.json", config)
    manifest.update(created_utc=utc_now(), sources=protocol.source_hashes(), maps=[], sessions=[],
        objective="surface_length_plus_normalized_SEM_exposure_not_PRI", nominal_method=protocol.NOMINAL,
        pilot_status="NOT_RUN", full_status="NOT_RUN", weight_selection="NOT_RUN_SMOKE_ONLY",
        timing=dict(threads_per_trial=1, start_method="spawn",
            phase_order=["collection", "independent_validation"],
            latency_origin="attempt_start_before_input_hash_field_build_and_process_launch"))
    if dry_run:
        dump(output / "manifest.json", manifest)
        return manifest
    import numpy as np
    from a3_traversability_protocol import fixture_geometry
    from native_point_cloud import generate_cloud
    from terrain_io import write_map
    settings = config["profiles"]["smoke"]
    try:
        for family in settings["families"]:
            extent = 40. if family == "flat_40m" else settings["extent_m"]
            if family == "flat_40m":
                height, rectangles, start, goal, label = (lambda x, y: 0.), [], [2., 20.], [38., 20.], "flat_full_scale_control"
            else:
                height, rectangles, start, goal, _, label = fixture_geometry(family)
            spacing, map_id = settings["spacing_m"], family + "_101"
            n = round(extent / spacing) + 1
            path = output / "inputs" / (map_id + ".t25d")
            write_map(path, [[height(i*spacing, j*spacing) for i in range(n)] for j in range(n)],
                spacing=(spacing, spacing), rectangles=rectangles,
                metadata=dict(map_id=map_id, family=family, generator_seed=101,
                    generator_version=protocol.PROTOCOL_ID, geometry_label=label,
                    purpose="fixed_synthetic_geometry_for_real_planner_smoke; not_measured_sensor_data"))
            cloud = path.with_suffix(".cloud.npy")
            np.save(cloud, generate_cloud(path), allow_pickle=False)
            manifest["maps"].append(dict(map_id=map_id, family=family, map_seed=101, map_path=str(path),
                map_sha256=file_hash(path), metadata_sha256=file_hash(path.with_suffix(".json")),
                cloud_path=str(cloud), cloud_sha256=file_hash(cloud), bounds=[0., 0., extent, extent],
                start=list(start), goal=list(goal), query_id="q0"))
        manifest["status"] = "PREPARED"
    except Exception as error:
        manifest.update(status="PREPARATION_FAILED", error=str(error))
        dump(output / "qualification_failure.json", dict(error=str(error), traceback=traceback.format_exc(),
            seed_replacements=[], planner_execution_count=0))
        dump(output / "manifest.json", manifest)
        raise
    dump(output / "manifest.json", manifest)
    return manifest


def _validate_map(task):
    import numpy as np
    from sem_exposure_validation import validate_trial
    from native_paper_oracle import NativePaperOracle
    from native_paper_v3_sem_evaluator import SharedSEMField
    info = task["info"]
    _verify_inputs(info)
    oracle = NativePaperOracle(info["map_path"])
    field = SharedSEMField(np.load(info["cloud_path"], allow_pickle=False), info["bounds"])
    completed = []
    for item in task["items"]:
        directory, entry = Path(item["directory"]), item["entry"]
        inventory = read(directory / "raw_inventory.json")
        if digest(inventory) != item["raw_inventory_sha256"] or raw_inventory(directory) != inventory:
            raise ValueError("raw inventory changed before validation: " + entry["trial_id"])
        spool = Path(task["spool"]) / (entry["trial_id"] + ".json")
        identity = dict(trial_id=entry["trial_id"], raw_inventory_sha256=digest(inventory),
            sources_sha256=task["sources_sha256"], info_sha256=digest(info))
        if spool.exists():
            if read(spool).get("identity") != identity:
                raise ValueError("validation spool identity mismatch")
            completed.append(entry["trial_id"])
            continue
        began, execution = time.perf_counter(), _execution(directory)
        if (directory / "request.json").is_file():
            row = validate_trial(directory, read(directory / "request.json"), info, execution,
                                 oracle=oracle, field=field)
        else:
            row = dict(success=False, attained_safe=False, execution_status=execution["status"],
                correctness_errors=[], events=[], unavailable_reason="no_request_before_launch")
        row.update(**{key: entry[key] for key in ("trial_id", "method", "map_id", "planner_seed",
            "family", "alpha", "sem_lambda", "policy")}, execution_launched=execution.get("launched", False),
            queue_wait_s=entry.get("queue_wait_s"), peak_rss_bytes=execution.get("peak_process_tree_rss_bytes"),
            validation_s=time.perf_counter()-began)
        row.update({key: entry[key] for key in ("map_seed", "query_id", "phase") if key in entry})
        dump(spool, dict(identity=identity, row=row, validator_pid=os.getpid(), created_utc=utc_now()))
        completed.append(entry["trial_id"])
    return dict(map_id=info["map_id"], completed_trial_ids=completed)


def run(output, worker, bridge, sem_python=None, *, resume=False, workers=None, validation_workers=None):
    import sem_exposure_protocol as protocol
    output, worker, bridge = Path(output).resolve(), Path(worker).resolve(), Path(bridge).resolve()
    config, manifest = protocol.load_config(output / "config.json"), read(output / "manifest.json")
    settings = config["execution"]
    for value, key in ((workers, "max_workers"), (validation_workers, "validation_workers")):
        if value is not None and value != settings[key]:
            raise ValueError("worker count differs from frozen prepared configuration")
    workers, validation_workers = settings["max_workers"], settings["validation_workers"]
    if manifest["profile"] != "smoke" or manifest["status"] == "PREPARED_INVENTORY":
        raise ValueError("campaign is not prepared for actual smoke execution")
    if not resume and manifest["status"] != "PREPARED":
        raise ValueError("campaign already attempted; --resume required")
    if manifest["status"] == "FAILED_CORRECTNESS":
        raise ValueError("correctness stop requires a new campaign")
    if manifest["config_sha256"] != digest(config) or manifest["sources"] != protocol.source_hashes():
        raise ValueError("frozen source/config changed after preparation")
    expected = protocol.inventory(config)["entries"]
    for left, right in zip(manifest["entries"], expected, strict=True):
        if any(left.get(key) != value for key, value in right.items() if key != "status"):
            raise ValueError("frozen schedule changed")
    sem_python = executable_path(sem_python or sys.executable)
    if not all(path.is_file() for path in (worker, bridge, sem_python)):
        raise FileNotFoundError("BLOCKED: worker, bridge and SEM interpreter are required")
    identity = dict(binary_sha256=file_hash(worker), bridge_sha256=file_hash(bridge),
                    sem_python=str(sem_python), sem_python_sha256=file_hash(sem_python))
    if resume and any(manifest.get(key) != value for key, value in identity.items()):
        raise ValueError("resume binary/interpreter identity mismatch")
    lock = _acquire_lock(output)
    began, session = time.perf_counter(), None
    try:
        if resume:
            _reconcile(output, manifest)
        manifest.update(identity, status="RUNNING", worker_path=str(worker),
            environment=dict(python=sys.version, platform=platform.platform(),
                             thread_environment={key: os.environ[key] for key in THREAD_ENV}))
        index = len(manifest["sessions"])
        session = dict(session_index=index, started_utc=utc_now(), collection_s=0., validation_s=0., max_concurrency=0,
                       phase="COLLECTION", workers=workers, validation_workers=validation_workers)
        manifest["sessions"].append(session)
        control = output / "execution_control" / f"session_{index:04d}"
        maps = {info["map_id"]: info for info in manifest["maps"]}
        entries = {entry["trial_id"]: entry for entry in manifest["entries"]}
        lanes = [[] for _ in range(workers)]
        for entry in manifest["entries"]:
            if entry["status"] == "NOT_RUN":
                lanes[entry["block_index"] % workers].append(dict(task_id=entry["trial_id"], entry=dict(entry),
                    directory=str(output / "trials" / entry["trial_id"]), info=maps[entry["map_id"]],
                    config=config, profile="smoke", worker=str(worker), bridge=str(bridge), sem_python=str(sem_python),
                    binary_sha256=identity["binary_sha256"], bridge_sha256=identity["bridge_sha256"]))

        def on_start(task, receipt):
            entry = entries[task["task_id"]]
            if entry["status"] != "NOT_RUN":
                raise ValueError("attempt already started")
            append_journal(output / "execution_control" / "start_journal.jsonl", dict(trial_id=entry["trial_id"],
                started_utc=receipt["started_utc"], config_sha256=manifest["config_sha256"],
                sources_sha256=digest(manifest["sources"]), session_index=index, lane=receipt["lane"], core=receipt["core"]))
            entry.update(status="STARTED", lane=receipt["lane"], core=receipt["core"],
                         queue_wait_s=receipt["queue_wait_s"], started_utc=receipt["started_utc"])
            dump(output / "manifest.json", manifest)

        def on_complete(task, value):
            if value["status"] == "COMPLETED":
                payload = value["payload"]
                measured = payload["execution"]
                entries[task["task_id"]].update(status="RAW_READY", execution_status=measured["status"],
                    collection_sha256=digest(payload), raw_inventory_sha256=payload["raw_inventory_sha256"],
                    planner_interval={key: measured.get(key) for key in
                        ("pid", "process_created", "launch_utc", "process_finished_utc", "exit_code")})
            dump(output / "manifest.json", manifest)

        dump(output / "manifest.json", manifest)
        admitted = resource_preflight(workers, settings["reserve_bytes_per_collection_worker"]) if any(lanes) else {"admitted": True}
        session["collection_admission"] = admitted
        if not admitted["admitted"]:
            manifest.update(status="BLOCKED_RESOURCES", blocked_reason=admitted["reason"])
        else:
            collected = run_spawn_lanes(lanes, _collect_trial, control / "collection", cores=admitted["cores"],
                on_start=on_start, on_complete=on_complete) if any(lanes) else dict(status="COMPLETE", wall_s=0., max_concurrency=0)
            session.update(collection_s=collected["wall_s"], max_concurrency=collected["max_concurrency"],
                           collection_finished_utc=utc_now())
            if collected["status"] != "COMPLETE":
                manifest["status"] = collected["status"]
            else:
                # The complete collection joins before any oracle process can start.
                session.update(phase="INDEPENDENT_VALIDATION", validation_started_utc=utc_now())
                dump(output / "manifest.json", manifest)
                admitted = resource_preflight(validation_workers, settings["reserve_bytes_per_validation_worker"])
                session["validation_admission"] = admitted
                if not admitted["admitted"]:
                    manifest.update(status="BLOCKED_RESOURCES", blocked_reason=admitted["reason"])
                else:
                    spool = output / "execution_control" / "validation_spool"
                    grouped = defaultdict(list)
                    for entry in manifest["entries"]:
                        _commit_spool(output, entry, manifest, spool)
                        if not entry.get("validation_sha256") and entry["status"] == "RAW_READY":
                            grouped[entry["map_id"]].append(dict(entry=dict(entry),
                                directory=str(output / "trials" / entry["trial_id"]),
                                raw_inventory_sha256=entry["raw_inventory_sha256"]))
                    validation_lanes = [[] for _ in range(validation_workers)]
                    for lane, (map_id, items) in enumerate(grouped.items()):
                        validation_lanes[lane % validation_workers].append(dict(task_id=map_id, info=maps[map_id], items=items,
                            sources_sha256=digest(manifest["sources"]), spool=str(spool)))

                    def commit_validation(task, value):
                        for entry in manifest["entries"]:
                            _commit_spool(output, entry, manifest, spool)
                        dump(output / "manifest.json", manifest)

                    verified = run_spawn_lanes(validation_lanes, _validate_map, control / "validation",
                        cores=admitted["cores"], on_complete=commit_validation)
                    session.update(validation_s=verified["wall_s"], validation_finished_utc=utc_now(), phase="VALIDATED")
                    manifest["status"] = ("COMPLETE" if all(entry.get("validation_sha256") for entry in manifest["entries"])
                                          else "INCOMPLETE") if verified["status"] == "COMPLETE" else verified["status"]
                    if any(read(output / "trials" / entry["trial_id"] / "validation.json")["correctness_errors"]
                           for entry in manifest["entries"] if entry.get("validation_sha256")):
                        manifest["status"] = "FAILED_CORRECTNESS"
    except KeyboardInterrupt:
        manifest["status"] = "INTERRUPTED"
    except Exception as error:
        manifest.update(status="FAILED_INFRASTRUCTURE", error=str(error), traceback=traceback.format_exc())
    finally:
        if session is not None:
            session.update(finished_utc=utc_now(), session_wall_s=time.perf_counter()-began)
        manifest["updated_utc"] = utc_now()
        dump(output / "manifest.json", manifest)
        lock.unlink(missing_ok=True)
    rows = [read(output / "trials" / entry["trial_id"] / "validation.json")
            for entry in manifest["entries"] if entry.get("validation_sha256")]
    summary = dict(protocol_id=protocol.PROTOCOL_ID, schema_version=1, profile="smoke", status=manifest["status"],
        planned=len(manifest["entries"]), attempted=sum(entry["status"] != "NOT_RUN" for entry in manifest["entries"]),
        validated=len(rows), success=sum(row["success"] for row in rows),
        correctness_failures=sum(bool(row["correctness_errors"]) for row in rows), rows=rows,
        requested_workers=workers, validation_workers=validation_workers,
        max_concurrency=max((s["max_concurrency"] for s in manifest["sessions"]), default=0),
        collection_s=sum(s["collection_s"] for s in manifest["sessions"]),
        validation_s=sum(s["validation_s"] for s in manifest["sessions"]),
        session_wall_s=sum(s["session_wall_s"] for s in manifest["sessions"]),
        actual_planner_overlap=actual_planner_overlap(manifest["entries"]),
        pilot_status="NOT_RUN", full_status="NOT_RUN", weight_selection="NOT_RUN_SMOKE_ONLY")
    dump(output / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "smoke", "inventory"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--profile", choices=("smoke",), default="smoke")
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--bridge", type=Path)
    parser.add_argument("--sem-python", type=Path)
    parser.add_argument("--workers", type=int, choices=(1, 4))
    parser.add_argument("--validation-workers", type=int, choices=(1, 4))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.resume and args.command != "run":
        parser.error("--resume only applies to run")
    if args.command in ("run", "smoke") and not args.dry_run and (not args.worker or not args.bridge):
        parser.error("actual execution requires --worker and --bridge")
    if args.command in ("prepare", "smoke", "inventory") or args.dry_run:
        result = prepare(args.output, args.config, dry_run=args.dry_run or args.command == "inventory",
                         workers=args.workers, validation_workers=args.validation_workers)
    if args.command in ("run", "smoke") and not args.dry_run:
        result = run(args.output, args.worker, args.bridge, args.sem_python, resume=args.resume,
                     workers=args.workers, validation_workers=args.validation_workers)
    import json
    print(json.dumps({key: result[key] for key in ("status", "planned_trials", "planned", "attempted", "success",
                      "correctness_failures", "max_concurrency") if key in result}))
    return 0 if result["status"] in ("PREPARED", "PREPARED_INVENTORY", "COMPLETE") else 1


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())

