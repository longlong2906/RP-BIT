"""Registered A3-T PRI path search collection. Stops after development pilot.

One durable attempt per physical backend; four selector outputs are correlated.
No confirm/full command exists. Prior protocols and experiment data are untouched.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
import multiprocessing
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback
from corridor_v4_execution import (THREAD_ENV, append_journal, atomic_json as dump,
    digest, file_hash, read, resource_preflight, run_spawn_lanes, set_thread_limits, utc_now)
set_thread_limits()
from run_corridor_v4 import (_acquire_lock, _commit_spool, _execution, _reconcile,
    _verify_inputs, actual_planner_overlap, raw_inventory)
from run_sem_exposure import _collect_trial as collect_registered_trial
from sem_exposure_runtime import executable_path
import pri_pathsearch_protocol as protocol
ROOT = Path(__file__).resolve().parents[1]

class _CommittedCorrectnessError(ValueError):
    pass

def _runtime_identity(worker, bridge, python):
    from terrain_compare_campaign import environment_identity
    from terrain_compare_runtime import capture_native_dependencies
    return dict(native=capture_native_dependencies([worker, bridge, python]),
                python_environment=environment_identity(python))


def _collect_trial(task):
    _verify_map(task["info"], task["entry"]["phase"])
    return collect_registered_trial(task, request_builder=protocol.request_for)


def _verify_map(info, phase):
    _verify_inputs(info)
    metadata = read(Path(info["map_path"]).with_suffix(".json"))
    for key, source in (("map_id", "map_id"), ("family", "family"), ("map_seed", "generator_seed")):
        if info[key] != metadata[source]:
            raise ValueError("map metadata identity differs: " + key)
    if info.get("query_id") != "q0":
        raise ValueError("unregistered query id")
    if not phase.startswith("smoke"):
        query, = metadata["queries"]
        if info["bounds"] != [0., 0., 40., 40.] or any(
                info[key] != query[key] for key in ("query_id", "query_hash", "start", "goal")):
            raise ValueError("frozen map/query differs from qualified metadata")
    else:
        from a3_traversability_protocol import fixture_geometry
        if info["family"] == "flat_40m":
            start, goal, extent = [2., 20.], [38., 20.], 40.
        else:
            _, _, start, goal, _, _ = fixture_geometry(info["family"])
            extent = 20.
        if info["start"] != list(start) or info["goal"] != list(goal) or info["bounds"] != [0., 0., extent, extent]:
            raise ValueError("smoke query differs from registered fixture")


def _prepare_map(directory, entry, config):
    import numpy as np
    from native_point_cloud import generate_cloud
    from terrain_io import write_map
    family, seed, phase = entry["family"], entry["map_seed"], entry["phase"]
    settings = config["profiles"][phase]
    if phase.startswith("smoke"):
        from a3_traversability_protocol import fixture_geometry
        extent = 40. if family == "flat_40m" else 20.
        if family == "flat_40m":
            height, rectangles, start, goal = (lambda x, y: 0.), [], [2., 20.], [38., 20.]
            label = "flat_full_scale_control"
        else:
            height, rectangles, start, goal, _, label = fixture_geometry(family)
        parameters = dict(geometry_label=label)
        definition = None
    else:
        from g5_dataset import geometry_definition
        height, rectangles, definitions, parameters = geometry_definition(family, seed)
        definition = definitions[0]
        if definition["query_id"] != "q0":
            raise ValueError("generator did not provide the registered q0")
        extent, start, goal = 40., definition["start"], definition["goal"]
    spacing = settings["spacing_m"]
    n = round(extent / spacing) + 1
    path = directory / "inputs" / (entry["map_id"] + ".t25d")
    metadata = write_map(path, [[height(i*spacing, j*spacing) for i in range(n)] for j in range(n)],
        spacing=(spacing, spacing), rectangles=rectangles,
        metadata=dict(map_id=entry["map_id"], family=family, generator_seed=seed,
            generator_version=protocol.PROTOCOL_ID, parameters=parameters,
            purpose="registered_synthetic_terrain_for_real_planning_not_measured_sensor_data"))
    if definition is not None:
        from g5_dataset import qualify_queries
        from independent_verifier import Oracle
        from terrain_compare_protocol import MODEL
        metadata["queries"] = qualify_queries(Oracle(path), [definition], file_hash(path), MODEL, clearance_metrics=False)
        dump(path.with_suffix(".json"), metadata)
    cloud = path.with_suffix(".cloud.npy")
    np.save(cloud, generate_cloud(path), allow_pickle=False)
    return dict(map_id=entry["map_id"], family=family, map_seed=seed, map_path=str(path),
        map_sha256=file_hash(path), metadata_sha256=file_hash(path.with_suffix(".json")),
        cloud_path=str(cloud), cloud_sha256=file_hash(cloud), bounds=[0., 0., extent, extent],
        start=list(start), goal=list(goal), query_id="q0",
        query_hash=metadata.get("queries", [{}])[0].get("query_hash"))


def _validate_map(task):
    import numpy as np
    from pri_pathsearch_validation import validate_trial
    from native_paper_oracle import NativePaperOracle
    from native_paper_v3_sem_evaluator import SharedSEMField
    info = task["info"]
    _verify_map(info, task["items"][0]["entry"]["phase"])
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
        row.update({key: entry[key] for key in ("trial_id", "method", "map_id", "planner_seed",
            "family", "alpha", "sem_lambda", "policy", "map_seed", "query_id", "phase")})
        row.update(execution_launched=execution.get("launched", False),
            queue_wait_s=entry.get("queue_wait_s"), peak_rss_bytes=execution.get("peak_process_tree_rss_bytes"),
            validation_s=time.perf_counter()-began)
        dump(spool, dict(identity=identity, row=row, validator_pid=os.getpid(), created_utc=utc_now()))
        completed.append(entry["trial_id"])
    return dict(map_id=info["map_id"], completed_trial_ids=completed)


def _audit(directory, require_complete=False):
    """Read only: reconstruct summary from hash-bound verdicts, not cached CSV."""
    directory = Path(directory)
    manifest = read(directory / "manifest.json")
    grandfathered = {}
    config = protocol.load_config(directory / "config.json")
    expected = protocol.inventory(config, manifest["phase"], manifest["selected_alpha"])
    if manifest["config_sha256"] != expected["config_sha256"]:
        raise ValueError("configuration identity drift")
    for actual, registered in zip(manifest["entries"], expected["entries"], strict=True):
        if not protocol.entry_matches(actual, registered):
            raise ValueError("registered trial schedule changed")
    maps = {info["map_id"]: info for info in manifest["maps"]}
    if len(maps) != len(manifest["maps"]):
        raise ValueError("duplicate registered map metadata")
    for info in maps.values():
        _verify_map(info, manifest["phase"])
    journal = directory / "execution_control" / "start_journal.jsonl"
    starts = [] if not journal.is_file() else [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    ids = [item["trial_id"] for item in starts]
    if len(set(ids)) != len(ids) or set(ids) != {entry["trial_id"] for entry in manifest["entries"] if entry["status"] != "NOT_RUN"}:
        raise ValueError("attempt journal disagrees with registered started identities")
    if any(item["config_sha256"] != manifest["config_sha256"] or
           item["sources_sha256"] != grandfathered.get(item["trial_id"], digest(manifest["sources"])) for item in starts):
        raise ValueError("attempt journal provenance changed")
    rows = []
    for entry in manifest["entries"]:
        if not entry.get("validation_sha256"):
            continue
        trial = directory / "trials" / entry["trial_id"]
        if file_hash(trial / "validation.json") != entry["validation_sha256"]:
            raise ValueError("validation verdict changed")
        inventory = read(trial / "raw_inventory.json")
        if digest(inventory) != entry["raw_inventory_sha256"] or raw_inventory(trial) != inventory:
            raise ValueError("raw trial changed")
        row = read(trial / "validation.json")
        if row.get("trial_id") != entry["trial_id"]:
            raise ValueError("validation identity mismatch")
        spool = directory / "execution_control" / "validation_spool" / (entry["trial_id"] + ".json")
        if file_hash(spool) != entry["validation_spool_sha256"]:
            raise ValueError("validation spool changed")
        proof = read(spool)
        binding = dict(trial_id=entry["trial_id"], raw_inventory_sha256=entry["raw_inventory_sha256"],
            sources_sha256=digest(manifest["sources"]), info_sha256=digest(maps[entry["map_id"]]))
        if proof["identity"] != binding or proof["row"] != row:
            raise ValueError("validation spool no longer binds the raw/map/query/source identity")
        rows.append(row)
    if require_complete and (manifest["status"] != "COMPLETE" or len(rows) != len(expected["entries"])):
        raise ValueError("complete independently validated campaign required")
    summary = dict(protocol_id=protocol.PROTOCOL_ID, schema_version=1, phase=manifest["phase"],
        profile=manifest["phase"], status=manifest["status"], planned=len(manifest["entries"]),
        attempted=sum(entry["status"] != "NOT_RUN" for entry in manifest["entries"]),
        validated=len(rows), success=sum(row["success"] for row in rows),
        correctness_failures=sum(bool(row["correctness_errors"]) for row in rows), rows=rows,
        max_concurrency=max((s.get("max_concurrency", 0) for s in manifest["sessions"]), default=0),
        collection_s=sum(s.get("collection_s", 0.) for s in manifest["sessions"]),
        validation_s=sum(s.get("validation_s", 0.) for s in manifest["sessions"]),
        actual_planner_overlap=actual_planner_overlap(manifest["entries"]))
    return manifest, summary, config


def _smoke_gate(output, binary_sha256=None, bridge_sha256=None, runtime_sha256=None):
    from analyze_pri_pathsearch import build_report
    bindings = {}
    for phase in ("smoke_serial", "smoke_parallel"):
        manifest, summary, config = _audit(Path(output) / phase, require_complete=True)
        if manifest.get("synthetic_fixture") or any(row.get("synthetic_fixture") for row in summary["rows"]):
            raise ValueError("synthetic fixtures cannot authorize experimental phases")
        if manifest["sources"] != protocol.source_hashes() or summary["correctness_failures"]:
            raise ValueError("smoke scientific sources changed or correctness failed")
        report = build_report(summary, manifest, config)
        if report["errors"] or not report["complete"]:
            raise ValueError("smoke analysis did not pass its complete-data checks")
        if binary_sha256 is not None and manifest.get("binary_sha256") != binary_sha256:
            raise ValueError("pilot binary differs from tested smoke binary")
        if bridge_sha256 is not None and manifest.get("bridge_sha256") != bridge_sha256:
            raise ValueError("pilot bridge differs from tested smoke bridge")
        if runtime_sha256 is not None and manifest.get("runtime_sha256") != runtime_sha256:
            raise ValueError("pilot native/Python runtime differs from tested smoke runtime")
        if any(row.get("execution_status") != "COMPLETED" for row in summary["rows"]):
            raise ValueError("smoke has crashed/interrupted/blocked attempts")
        for method in protocol.METHODS:
            if not any(row["method"] == method and row["success"] for row in summary["rows"]):
                raise ValueError("smoke did not establish a final-safe path for " + method)
        reserve = protocol.default_config()["execution"]["reserve_bytes_per_collection_worker"]
        if any(row.get("peak_rss_bytes", 0) > reserve for row in summary["rows"]):
            raise ValueError("observed smoke process tree exceeded the registered memory reservation")
        bindings[phase] = file_hash(Path(output) / phase / "manifest.json")
    return bindings


def prepare(output, config_path=protocol.CONFIG, phase="smoke_serial", *, dry_run=False,
            workers=None, validation_workers=None):
    output = Path(output).resolve()
    config = protocol.load_config(config_path)
    selected = None
    manifest = protocol.inventory(config, phase, selected)
    directory = output / phase
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("refusing to overwrite an existing phase")
    defaults = config["execution"]
    expected_workers = 1 if phase == "smoke_serial" else 4
    workers = expected_workers if workers is None else workers
    validation_workers = defaults["validation_workers"] if validation_workers is None else validation_workers
    if workers not in (1, 4) or validation_workers not in (1, 4):
        raise ValueError("registered worker counts are 1 or 4")
    if phase.startswith("smoke") and workers != expected_workers:
        raise ValueError("smoke serial/parallel worker schedule is fixed")
    directory.mkdir(parents=True, exist_ok=True)
    dump(directory / "config.json", config)
    manifest.update(created_utc=utc_now(), sources=protocol.source_hashes(), maps=[], sessions=[],
        workers=workers, validation_workers=validation_workers,
        objective="BITstar_surface_length_with_bounded_whole_path_PRI_search",
        native_baseline_objective="SEM_paper_static_reconstruction_length_turn_risk",
        timing=dict(origin="before_planner_creation_after_immutable_field_preprocessing",
                    counts_planner_creation_pathsearch_scoring_and_callbacks=True, threads_per_trial=1))
    dump(directory / "manifest.json", manifest)
    if dry_run:
        return manifest
    try:
        seen = set()
        for entry in manifest["entries"]:
            if entry["map_id"] in seen:
                continue
            if shutil.disk_usage(directory).free < defaults["minimum_free_disk_bytes"]:
                raise OSError("BLOCKED_DISK: insufficient registered free disk reserve")
            info = _prepare_map(directory, entry, config)
            manifest["maps"].append(info)
            seen.add(entry["map_id"])
            dump(directory / "manifest.json", manifest)
        manifest["status"] = "PREPARED"
    except Exception as error:
        manifest.update(status="PREPARATION_FAILED", error=str(error), traceback=traceback.format_exc())
        dump(directory / "manifest.json", manifest)
        raise
    dump(directory / "manifest.json", manifest)
    return manifest


def run(output, phase, worker, bridge, sem_python=None, *, resume=False,
        allow_pilot=False, workers=None, validation_workers=None):
    output = Path(output).resolve()
    directory, worker, bridge = output / phase, Path(worker).resolve(), Path(bridge).resolve()
    config = protocol.load_config(directory / "config.json")
    manifest = read(directory / "manifest.json")
    registered = protocol.inventory(config, phase, manifest["selected_alpha"])
    if manifest["phase"] != phase or manifest["config_sha256"] != registered["config_sha256"]:
        raise ValueError("phase/configuration identity mismatch")
    for actual, expected in zip(manifest["entries"], registered["entries"], strict=True):
        if not protocol.entry_matches(actual, expected):
            raise ValueError("frozen schedule changed")
    if manifest["sources"] != protocol.source_hashes():
        raise ValueError("scientific sources changed after phase preparation")
    if manifest["status"] in ("PREPARED_INVENTORY", "PREPARATION_FAILED", "FAILED_CORRECTNESS"):
        raise ValueError("phase is not eligible for execution")
    if not resume and manifest["status"] != "PREPARED":
        raise ValueError("existing attempt history requires --resume")
    for value, key in ((workers, "workers"), (validation_workers, "validation_workers")):
        if value is not None and value != manifest[key]:
            raise ValueError("worker count differs from frozen phase")
    workers, validation_workers = manifest["workers"], manifest["validation_workers"]
    sem_python = executable_path(sem_python or sys.executable)
    if not all(path.is_file() for path in (worker, bridge, sem_python)):
        raise FileNotFoundError("BLOCKED: worker, bridge and interpreter are required")
    identity = dict(binary_sha256=file_hash(worker), bridge_sha256=file_hash(bridge),
        sem_python=str(sem_python), sem_python_sha256=file_hash(sem_python))
    runtime = _runtime_identity(worker, bridge, sem_python)
    identity["runtime_sha256"] = digest(runtime)
    if resume and any(manifest.get(key) != value for key, value in identity.items()):
        raise ValueError("resume binary/interpreter identity mismatch")
    if phase == "pilot":
        if not allow_pilot:
            raise ValueError("pilot requires --allow-pilot")
        manifest["smoke_gate"] = _smoke_gate(output, identity["binary_sha256"], identity["bridge_sha256"], identity["runtime_sha256"])
    if phase == "smoke_parallel":
        serial_manifest, serial_summary, _ = _audit(output / "smoke_serial", require_complete=True)
        if serial_manifest["sources"] != protocol.source_hashes() or serial_summary["correctness_failures"]:
            raise ValueError("parallel smoke requires current correctness-passing serial smoke")
        if any(serial_manifest.get(key) != identity[key] for key in ("binary_sha256", "bridge_sha256", "runtime_sha256")):
            raise ValueError("parallel binary differs from serial smoke")
        reserve = config["execution"]["reserve_bytes_per_collection_worker"]
        if any(row.get("peak_rss_bytes", 0) > reserve for row in serial_summary["rows"]):
            raise ValueError("BLOCKED_RESOURCES: serial peak exceeds the registered per-worker reservation")
    lock = _acquire_lock(directory)
    session, began = None, time.perf_counter()
    try:
        if resume:
            _reconcile(directory, manifest)
        if any(
                read(directory / "trials" / entry["trial_id"] / "validation.json")["correctness_errors"]
                for entry in manifest["entries"] if entry.get("validation_sha256")):
            raise _CommittedCorrectnessError("a committed correctness failure blocks further dispatch")
        manifest.update(identity, status="RUNNING", worker_path=str(worker),
            runtime=runtime,
            environment=dict(python=sys.version, platform=platform.platform(),
                thread_environment={key: os.environ[key] for key in THREAD_ENV}))
        index = len(manifest["sessions"])
        session = dict(session_index=index, started_utc=utc_now(), collection_s=0., validation_s=0.,
                       max_concurrency=0, workers=workers, validation_workers=validation_workers)
        manifest["sessions"].append(session)
        control = directory / "execution_control" / f"session_{index:04d}"
        maps = {info["map_id"]: info for info in manifest["maps"]}
        entries = {entry["trial_id"]: entry for entry in manifest["entries"]}
        ordered_maps = list(dict.fromkeys(entry["map_id"] for entry in manifest["entries"]))
        settings = config["execution"]
        spool = directory / "execution_control" / "validation_spool"

        def on_start(task, receipt):
            entry = entries[task["task_id"]]
            if entry["status"] != "NOT_RUN":
                raise ValueError("started identity cannot be retried")
            append_journal(directory / "execution_control" / "start_journal.jsonl",
                dict(trial_id=entry["trial_id"], started_utc=receipt["started_utc"],
                    config_sha256=manifest["config_sha256"], sources_sha256=digest(manifest["sources"]),
                    session_index=index, lane=receipt["lane"], core=receipt["core"]))
            entry.update(status="STARTED", lane=receipt["lane"], core=receipt["core"],
                         queue_wait_s=receipt["queue_wait_s"], started_utc=receipt["started_utc"])
            dump(directory / "manifest.json", manifest)

        def on_complete(task, value):
            if value["status"] == "COMPLETED":
                payload, entry = value["payload"], entries[task["task_id"]]
                execution = payload["execution"]
                entry.update(status="RAW_READY", execution_status=execution["status"],
                    collection_sha256=digest(payload), raw_inventory_sha256=payload["raw_inventory_sha256"],
                    planner_interval={key: execution.get(key) for key in
                        ("pid", "process_created", "launch_utc", "process_finished_utc", "exit_code")})
            dump(directory / "manifest.json", manifest)

        def commit_validation(task, value):
            for entry in manifest["entries"]:
                _commit_spool(directory, entry, manifest, spool)
            dump(directory / "manifest.json", manifest)

        dump(directory / "manifest.json", manifest)
        width = settings["wave_map_count"]
        for wave_index, offset in enumerate(range(0, len(ordered_maps), width)):
            wave_maps = set(ordered_maps[offset:offset+width])
            wave_entries = [entry for entry in manifest["entries"] if entry["map_id"] in wave_maps]
            if all(entry.get("validation_sha256") for entry in wave_entries):
                continue
            if shutil.disk_usage(directory).free < settings["minimum_free_disk_bytes"]:
                manifest.update(status="BLOCKED_DISK", blocked_reason="registered free disk reserve")
                break
            lanes = [[] for _ in range(workers)]
            for entry in wave_entries:
                if entry["status"] == "NOT_RUN":
                    lanes[entry["block_index"] % workers].append(dict(task_id=entry["trial_id"],
                        entry=dict(entry), directory=str(directory/"trials"/entry["trial_id"]),
                        info=maps[entry["map_id"]], config=config, profile=phase, worker=str(worker),
                        bridge=str(bridge), sem_python=str(sem_python),
                        binary_sha256=identity["binary_sha256"], bridge_sha256=identity["bridge_sha256"]))
            wave_control = control / f"wave_{wave_index:03d}"
            if any(lanes):
                admission = resource_preflight(workers, settings["reserve_bytes_per_collection_worker"])
                session["collection_admission"] = admission
                if not admission["admitted"]:
                    manifest.update(status="BLOCKED_RESOURCES", blocked_reason=admission["reason"])
                    break
                result = run_spawn_lanes(lanes, _collect_trial, wave_control/"collection", cores=admission["cores"],
                                         on_start=on_start, on_complete=on_complete)
                session["collection_s"] += result["wall_s"]
                session["max_concurrency"] = max(session["max_concurrency"], result["max_concurrency"])
                if result["status"] != "COMPLETE":
                    manifest["status"] = result["status"]
                    break
            grouped = defaultdict(list)
            for entry in wave_entries:
                _commit_spool(directory, entry, manifest, spool)
                if entry["status"] == "RAW_READY" and not entry.get("validation_sha256"):
                    grouped[entry["map_id"]].append(dict(entry=dict(entry),
                        directory=str(directory/"trials"/entry["trial_id"]),
                        raw_inventory_sha256=entry["raw_inventory_sha256"]))
            if grouped:
                admission = resource_preflight(validation_workers, settings["reserve_bytes_per_validation_worker"])
                session["validation_admission"] = admission
                if not admission["admitted"]:
                    manifest.update(status="BLOCKED_RESOURCES", blocked_reason=admission["reason"])
                    break
                lanes = [[] for _ in range(validation_workers)]
                for lane, (map_id, items) in enumerate(grouped.items()):
                    lanes[lane % validation_workers].append(dict(task_id=map_id, info=maps[map_id], items=items,
                        sources_sha256=digest(manifest["sources"]), spool=str(spool)))
                result = run_spawn_lanes(lanes, _validate_map, wave_control/"validation", cores=admission["cores"],
                                         on_complete=commit_validation)
                session["validation_s"] += result["wall_s"]
                if result["status"] != "COMPLETE":
                    manifest["status"] = result["status"]
                    break
            if any(read(directory/"trials"/entry["trial_id"]/"validation.json")["correctness_errors"]
                   for entry in wave_entries if entry.get("validation_sha256")):
                manifest["status"] = "FAILED_CORRECTNESS"
                break
        if manifest["status"] == "RUNNING":
            manifest["status"] = "COMPLETE" if all(entry.get("validation_sha256") for entry in manifest["entries"]) else "INCOMPLETE"
    except _CommittedCorrectnessError as error:
        manifest.update(status="FAILED_CORRECTNESS", error=str(error))
    except KeyboardInterrupt:
        manifest["status"] = "INTERRUPTED"
    except Exception as error:
        manifest.update(status="FAILED_INFRASTRUCTURE", error=str(error), traceback=traceback.format_exc())
    finally:
        if session is not None:
            session.update(finished_utc=utc_now(), session_wall_s=time.perf_counter()-began)
        manifest["updated_utc"] = utc_now()
        dump(directory / "manifest.json", manifest)
        lock.unlink(missing_ok=True)
    _, summary, _ = _audit(directory)
    dump(directory / "summary.json", summary)
    return summary


def analyze(output, phase, report_output=None):
    from analyze_pri_pathsearch import write_report
    if phase not in protocol.PHASES:
        raise ValueError("only registered smoke/pilot analysis is available")
    output = Path(output).resolve()
    directory = output / phase
    manifest, summary, config = _audit(directory)
    dump(directory / "summary.json", summary)
    report = write_report(summary, manifest, config, report_output or directory / "reports")
    if phase == "pilot":
        body = dict(protocol_id=protocol.PROTOCOL_ID, selection=report["selection"],
            pilot_manifest_sha256=file_hash(directory / "manifest.json"),
            binary_sha256=manifest.get("binary_sha256"), bridge_sha256=manifest.get("bridge_sha256"),
            runtime_sha256=manifest.get("runtime_sha256"), sources_sha256=digest(manifest["sources"]),
            config_sha256=manifest["config_sha256"], stop_after_pilot=True)
        dump(output / "selection.json", dict(body, receipt_sha256=digest(body)))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inventory", "prepare", "run", "smoke", "pilot", "analyze"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=protocol.CONFIG)
    parser.add_argument("--phase", choices=protocol.PHASES)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--bridge", type=Path)
    parser.add_argument("--sem-python", type=Path)
    parser.add_argument("--workers", type=int, choices=(1, 4))
    parser.add_argument("--validation-workers", type=int, choices=(1, 4))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-pilot", action="store_true")
    args = parser.parse_args(argv)
    phases = ("smoke_serial", "smoke_parallel") if args.command == "smoke" else (
        "pilot" if args.command == "pilot" else args.phase or "smoke_serial",)
    actual = args.command in ("run", "smoke", "pilot") and not args.dry_run
    if actual and (not args.worker or not args.bridge):
        parser.error("actual execution requires --worker and --bridge")
    if actual and "pilot" in phases and not args.allow_pilot:
        parser.error("pilot requires --allow-pilot")
    result = None
    for phase in phases:
        if args.command == "analyze":
            result = analyze(args.output, phase)
            continue
        path = args.output / phase / "manifest.json"
        if args.command in ("inventory", "prepare") or args.dry_run or not path.exists():
            result = prepare(args.output, args.config, phase,
                dry_run=args.dry_run or args.command == "inventory", workers=args.workers,
                validation_workers=args.validation_workers)
        if actual:
            if args.resume and read(path).get("status") == "COMPLETE":
                _, result, _ = _audit(args.output/phase, require_complete=True)
            else:
                result = run(args.output, phase, args.worker, args.bridge, args.sem_python,
                    resume=args.resume, allow_pilot=args.allow_pilot,
                    workers=args.workers, validation_workers=args.validation_workers)
            if result["status"] != "COMPLETE":
                break
            analyze(args.output, phase)
    print(json.dumps({key: result[key] for key in ("status", "planned_trials", "planned", "attempted",
        "success", "correctness_failures", "max_concurrency", "selection") if key in result}, ensure_ascii=False))
    return 0 if result.get("status") in ("PREPARED", "PREPARED_INVENTORY", "COMPLETE", "DONE") else 1


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
