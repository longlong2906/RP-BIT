"""Run the registered SEM exposure pilot in bounded, resumable map waves."""
from __future__ import annotations

import argparse
from collections import defaultdict
from functools import partial
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback
from zipfile import ZipFile

from corridor_v4_execution import (append_journal, atomic_json as dump, digest, file_hash,
    process_alive, read, run_spawn_lanes, set_thread_limits, utc_now)
from native_paper_v4_storage import ArtifactStore, preflight
from run_corridor_v4 import (_acquire_lock, _commit_spool, _reconcile, _verify_inputs,
    actual_planner_overlap, raw_inventory)
from run_sem_exposure import _collect_trial, _validate_map
import sem_exposure_pilot_protocol as protocol
from sem_exposure_runtime import executable_path, runtime_admission

set_thread_limits()
ROOT = Path(__file__).resolve().parents[1]
SOFT_SESSION_S = 8*3600
# Leave five minutes for controller reconciliation before the supervisor
# starts the final fifteen-minute export window.
WORK_DEADLINE_S = 9*3600-20*60
PHASES = ("tune", "confirm")
RECOVERABLE = {"PAUSED_TIME", "BLOCKED_RESOURCES", "BLOCKED_DISK", "INTERRUPTED"}


def _groups(entries):
    grouped = defaultdict(list)
    for entry in entries:
        grouped[entry["map_id"]].append(entry)
    return grouped


def _waves(entries):
    maps = list(_groups(entries))
    return [(f"wave_{i//4:03d}", maps[i:i+4]) for i in range(0, len(maps), 4)]


def _selection(output, config):
    from analyze_sem_exposure_pilot import build_report
    receipt = read(output/"selection.json")
    selection = {
        key: value for key, value in receipt.items()
        if key not in ("receipt_sha256", "tune_manifest_sha256")}
    binding = dict(selection=selection, tune_manifest_sha256=receipt["tune_manifest_sha256"])
    if (selection["status"] != "CANDIDATE_LOCKED" or receipt["receipt_sha256"] != digest(binding)
            or binding["tune_manifest_sha256"] != file_hash(output/"tune"/"manifest.json")):
        raise ValueError("confirmation requires the locked tuning selection and unchanged evidence")
    manifest, summary = _audited_summary(output/"tune", config)
    if build_report(summary, manifest, config)["selection"] != selection:
        raise ValueError("selection differs from the audited tuning result")
    return selection


def prepare(output, config_path=protocol.CONFIG, phase="tune"):
    if phase not in PHASES:
        raise ValueError("unknown pilot phase")
    output = Path(output).resolve()
    config = protocol.load_config(config_path)
    directory = output/phase
    directory.mkdir(parents=True, exist_ok=True)
    lock = _acquire_lock(directory)
    try:
        if (directory/"manifest.json").exists():
            raise FileExistsError("phase already prepared")
        selection = _selection(output, config) if phase == "confirm" else None
        weight = selection["selected_lambda"] if selection else None
        plan = protocol.inventory(config, phase, weight)
        manifest = dict(plan, phase=phase, selected_lambda=weight, status="PREPARED",
            sources=protocol.source_hashes(), maps=[], shards={}, sessions=[], created_utc=utc_now(),
            pilot_status="NOT_RUN", full_status="NOT_RUN")
        if selection:
            manifest["selection_sha256"] = file_hash(output/"selection.json")
        dump(directory/"config.json", config)
        dump(directory/"manifest.json", manifest)
        return manifest
    finally:
        lock.unlink(missing_ok=True)


def _check_manifest(directory, config, manifest):
    registered = protocol.inventory(config, manifest["phase"], manifest["selected_lambda"])
    if (manifest["config_sha256"] != registered["config_sha256"]
            or manifest["sources"] != protocol.source_hashes()
            or manifest["protocol_id"] != protocol.PROTOCOL_ID
            or manifest["planned_trials"] != len(registered["entries"])):
        raise ValueError("frozen phase configuration, sources or inventory changed")
    for actual, expected in zip(manifest["entries"], registered["entries"], strict=True):
        if any(actual.get(key) != value for key, value in expected.items() if key != "status"):
            raise ValueError("frozen trial schedule changed")
    if directory.name != manifest["phase"]:
        raise ValueError("phase location differs from manifest")


def _runtime_identity(path, worker, bridge, python):
    from terrain_compare_campaign import environment_identity
    from terrain_compare_runtime import capture_native_dependencies
    value = read(path)
    body = {key: entry for key, entry in value.items() if key != "identity_sha256"}
    if (executable_path(sys.executable) != python
            or value.get("schema_version") != 1 or value.get("protocol_id") != protocol.PROTOCOL_ID
            or value.get("identity_sha256") != digest(body)
            or value.get("worker_path") != str(worker) or value.get("bridge_path") != str(bridge)
            or value.get("controller_python") != str(python)
            or value.get("density_version") != "exact_canonical_raster_v1"
            or value.get("requirements_sha256") != file_hash(ROOT/"requirements-native-paper-v4.txt")):
        raise ValueError("runtime receipt differs from the requested pilot runtime")
    roots = value["native_roots"]
    if not {str(worker), str(bridge), str(python)} <= set(roots):
        raise ValueError("runtime receipt omits required native roots")
    current = capture_native_dependencies(roots)
    if (current["binary_hashes"] != value["binary_hashes"]
            or current["linkage"] != value["native_linkage"]
            or environment_identity(python) != value["environment"]):
        raise ValueError("runtime linkage, binary or Python package identity drift")
    return value


def _prepare_map(directory, entry):
    import numpy as np
    from g5_dataset import geometry_definition, qualify_queries
    from independent_verifier import Oracle
    from native_point_cloud import generate_cloud
    from terrain_compare_protocol import MODEL
    from terrain_io import write_map

    family, seed = entry["family"], entry["map_seed"]
    height, rectangles, definitions, parameters = geometry_definition(family, seed)
    # The experiment freezes q0 from the existing seeded map generator.
    definition = definitions[0]
    if definition["query_id"] != "q0":
        raise ValueError("unexpected generated query")
    path = directory/"inputs"/(entry["map_id"]+".t25d")
    metadata = write_map(path, [[height(i*.2, j*.2) for i in range(201)] for j in range(201)],
        spacing=(.2, .2), rectangles=rectangles, metadata=dict(map_id=entry["map_id"],
        family=family, generator_seed=seed, generator_version=protocol.PROTOCOL_ID, parameters=parameters))
    map_hash = file_hash(path)
    metadata["queries"] = qualify_queries(Oracle(path), [definition], map_hash, MODEL, clearance_metrics=False)
    dump(path.with_suffix(".json"), metadata)
    cloud = path.with_suffix(".cloud.npy")
    np.save(cloud, generate_cloud(path), allow_pickle=False)
    return dict(map_id=entry["map_id"], family=family, map_seed=seed, map_path=str(path),
        map_sha256=map_hash, metadata_sha256=file_hash(path.with_suffix(".json")),
        cloud_path=str(cloud), cloud_sha256=file_hash(cloud), bounds=[0., 0., 40., 40.],
        start=definition["start"], goal=definition["goal"], query_id="q0",
        query_hash=metadata["queries"][0]["query_hash"])


def _request(entry, info, config, phase, field_path, field_metadata, bridge):
    return protocol.request_for(entry, info, config, field_path, field_metadata, bridge)


def _check_query(info, metadata):
    query, = metadata["queries"]
    if (metadata["map_id"] != info["map_id"] or metadata["generator_seed"] != info["map_seed"]
            or metadata["family"] != info["family"] or info["bounds"] != [0., 0., 40., 40.]
            or any(query[key] != info[key] for key in ("query_id", "query_hash", "start", "goal"))):
        raise ValueError("frozen map/query metadata differs from trial input")


def _verify_map(info):
    _verify_inputs(info)
    _check_query(info, read(Path(info["map_path"]).with_suffix(".json")))


def _assert_quiescent(directory):
    for path in (directory/"execution_control").glob("**/child.json"):
        receipt = read(path)
        if process_alive(receipt.get("pid"), receipt.get("process_created")):
            raise ValueError("prior campaign worker is still alive")


def _journal(lines, entries, manifest):
    seen = set()
    expected = {entry["trial_id"] for entry in entries if entry["status"] != "NOT_RUN"}
    sources_sha256 = digest(manifest["sources"])
    for line in lines.splitlines():
        receipt = json.loads(line)
        trial_id = receipt["trial_id"]
        if (trial_id in seen or trial_id not in expected
                or receipt["config_sha256"] != manifest["config_sha256"]
                or receipt["sources_sha256"] != sources_sha256):
            raise ValueError("trial journal differs from registered attempt history")
        seen.add(trial_id)
    if seen != expected:
        raise ValueError("trial state lacks durable start proof")


def _checked_row(row, entry, manifest):
    if any(row.get(key) != entry[key] for key in ("trial_id", "method", "map_id", "family",
                                                "planner_seed", "sem_lambda", "success", "execution_status")):
        raise ValueError("independent verdict identity mismatch")
    # Snapshot verdicts stay in the immutable archive. The analyzer uses only
    # trial-level metrics and need not retain thousands of geometry trees.
    compact = {key: value for key, value in row.items() if key != "events"}
    return dict(compact, map_seed=entry["map_seed"], query_id=entry["query_id"], phase=manifest["phase"])


def _read_sealed(store, wave, mid, entries, manifest, info):
    indexed = store._read_index()["entries"][wave+"/"+mid]
    directory, archive, _ = store._paths(wave, mid)
    frozen = manifest["shards"][mid]
    if (indexed["archive"] != archive.relative_to(store.root).as_posix()
            or frozen["sha256"] != indexed["sha256"] or file_hash(archive) != frozen["sha256"]):
        raise ValueError("sealed map archive identity changed")
    with ZipFile(archive) as packed:
        # pack_map already verified every member, CRC and mode against this
        # immutable archive hash. Bind the index to that same embedded inventory.
        packed_inventory = json.loads(packed.read("__map_shard__.json"))
        inventory = packed_inventory["inventory"]
        if inventory != indexed["inventory"] or digest(inventory) != frozen["inventory_sha256"]:
            raise ValueError("sealed map inventory changed")
        for path_key, hash_key in (("map_path", "map_sha256"), ("cloud_path", "cloud_sha256"),
                                   ("map_path", "metadata_sha256")):
            path = Path(info[path_key])
            if hash_key == "metadata_sha256":
                path = path.with_suffix(".json")
            if inventory[path.relative_to(directory).as_posix()]["sha256"] != info[hash_key]:
                raise ValueError("sealed input differs from registered hash")
        metadata_path = Path(info["map_path"]).with_suffix(".json").relative_to(directory).as_posix()
        _check_query(info, json.loads(packed.read(metadata_path)))
        _journal(packed.read("execution_control/start_journal.jsonl").decode(), entries, manifest)
        rows = []
        raw_by_trial = defaultdict(dict)
        for name, record in inventory.items():
            parts = name.split("/", 2)
            if len(parts) == 3 and parts[0] == "trials" and parts[2] not in ("raw_inventory.json", "validation.json"):
                raw_by_trial[parts[1]][parts[2]] = record["sha256"]
        for entry in entries:
            tid = entry["trial_id"]
            prefix = "trials/"+tid+"/"
            raw = json.loads(packed.read(prefix+"raw_inventory.json"))
            if raw != raw_by_trial[tid] or digest(raw) != entry["raw_inventory_sha256"]:
                raise ValueError("sealed trial raw inventory changed")
            name = prefix+"validation.json"
            if inventory[name]["sha256"] != entry["validation_sha256"]:
                raise ValueError("sealed trial verdict hash changed")
            rows.append(_checked_row(json.loads(packed.read(name)), entry, manifest))
    return rows


def _audited_summary(directory, config):
    manifest = read(directory/"manifest.json")
    _check_manifest(directory, config, manifest)
    store = ArtifactStore(directory)
    by_map = _groups(manifest["entries"])
    infos = {info["map_id"]: info for info in manifest["maps"]}
    rows = []
    for wave, maps in _waves(manifest["entries"]):
        for mid in maps:
            entries = by_map[mid]
            if mid in manifest["shards"]:
                rows.extend(_read_sealed(store, wave, mid, entries, manifest, infos[mid]))
                continue
            active = store._paths(wave, mid)[0]
            if mid in infos:
                _verify_map(infos[mid])
            journal = active/"execution_control"/"start_journal.jsonl"
            _journal(journal.read_text(encoding="utf-8") if journal.is_file() else "", entries, manifest)
            for entry in entries:
                if not entry.get("validation_sha256"):
                    continue
                trial = active/"trials"/entry["trial_id"]
                raw = read(trial/"raw_inventory.json")
                if (raw != raw_inventory(trial) or digest(raw) != entry["raw_inventory_sha256"]
                        or file_hash(trial/"validation.json") != entry["validation_sha256"]):
                    raise ValueError("active trial raw artifacts or verdict changed")
                rows.append(_checked_row(read(trial/"validation.json"), entry, manifest))
    attempted = sum(entry["status"] != "NOT_RUN" for entry in manifest["entries"])
    timing = {key: sum(session.get(key, 0.) for session in manifest["sessions"])
              for key in ("collection_s", "validation_s", "session_wall_s")}
    summary = dict(protocol_id=protocol.PROTOCOL_ID, schema_version=1, phase=manifest["phase"],
        status=manifest["status"], planned=len(manifest["entries"]), attempted=attempted,
        validated=len(rows), success=sum(row["success"] for row in rows), rows=rows,
        planner_overlap=actual_planner_overlap(manifest["entries"]), **timing)
    return manifest, summary


def analyze(output, phase):
    from analyze_sem_exposure_pilot import build_report, write_report
    output = Path(output).resolve()
    directory = output/phase
    config = protocol.load_config(directory/"config.json")
    manifest, summary = _audited_summary(directory, config)
    selection = _selection(output, config) if phase == "confirm" else None
    report = build_report(summary, manifest, config, tuning_selection=selection)
    manifest_hash = file_hash(directory/"manifest.json")
    target = directory/"reports"/manifest_hash
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".report-", dir=target.parent) as stage:
            write_report(report, stage, provenance=dict(manifest_sha256=manifest_hash,
                sources_sha256=digest(manifest["sources"]),
                runtime_identity_sha256=manifest.get("runtime_identity_sha256")))
            os.replace(stage, target)
    dump(directory/"latest_report.json", dict(path=target.relative_to(directory).as_posix(),
                                               manifest_sha256=manifest_hash, status=report["status"]))
    if phase == "tune":
        result = report["selection"]
        binding = dict(selection=result, tune_manifest_sha256=manifest_hash)
        receipt = dict(result, tune_manifest_sha256=manifest_hash, receipt_sha256=digest(binding))
        path = output/"selection.json"
        if path.is_file() and read(path)["status"] == "CANDIDATE_LOCKED" and read(path) != receipt:
            raise ValueError("refusing to replace an immutable tuning selection")
        dump(path, receipt)
    return dict(report, campaign_status=manifest["status"])


def _recover_map(folder, entries, manifest):
    from run_corridor_v4 import _execution
    _reconcile(folder, dict(manifest, entries=entries))
    journal = folder/"execution_control"/"start_journal.jsonl"
    if not journal.is_file():
        return
    receipts = {value["trial_id"]: value for value in
                (json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines())}
    for entry in entries:
        if entry["trial_id"] in receipts:
            receipt = receipts[entry["trial_id"]]
            entry.update({key: receipt[key] for key in ("started_utc", "lane", "core", "queue_wait_s")})
            execution = _execution(folder/"trials"/entry["trial_id"])
            entry["planner_interval"] = {key: execution.get(key) for key in
                ("pid", "process_created", "launch_utc", "process_finished_utc", "exit_code")}


def _run_wave(directory, store, manifest, config, wave, maps, grouped, paths, runtime,
              session, admission, stop, hard):
    active = {mid: store._paths(wave, mid)[0] for mid in maps if mid not in manifest["shards"]}
    if not active:
        return "COMPLETE"
    infos = {info["map_id"]: info for info in manifest["maps"]}
    for mid, folder in active.items():
        if time.monotonic() >= stop:
            return "PAUSED_TIME"
        if mid not in infos:
            infos[mid] = _prepare_map(folder, grouped[mid][0])
            manifest["maps"].append(infos[mid])
            dump(directory/"manifest.json", manifest)
        # A crash after shard commit but before updating the phase manifest
        # leaves a verified index entry. Adopt it before touching active data.
        if wave+"/"+mid in store._read_index()["entries"]:
            if not all(entry.get("validation_sha256") for entry in grouped[mid]):
                raise ValueError("sealed map is missing committed verdicts")
            _seal(store, manifest, wave, mid)
            dump(directory/"manifest.json", manifest)
            continue
        _verify_map(infos[mid])
        _recover_map(folder, grouped[mid], manifest)
    active = {mid: folder for mid, folder in active.items() if mid not in manifest["shards"]}
    entries = {entry["trial_id"]: entry for mid in active for entry in grouped[mid]}
    worker, bridge, python = paths
    lanes = [[] for _ in range(4)]
    for entry in entries.values():
        if entry["status"] != "NOT_RUN":
            continue
        folder = active[entry["map_id"]]
        lanes[entry["block_index"] % 4].append(dict(task_id=entry["trial_id"], entry=dict(entry),
            directory=str(folder/"trials"/entry["trial_id"]), info=infos[entry["map_id"]], config=config,
            profile=manifest["phase"], worker=str(worker), bridge=str(bridge), sem_python=str(python),
            binary_sha256=runtime["binary_hashes"][str(worker)], bridge_sha256=runtime["binary_hashes"][str(bridge)]))
    control = directory/"execution_control"/f"session_{session['session_index']:04d}"/wave
    sources_sha256 = digest(manifest["sources"])

    def start(task, receipt):
        entry = entries[task["task_id"]]
        if entry["status"] != "NOT_RUN":
            raise ValueError("attempt already started")
        identity = {key: receipt[key] for key in ("started_utc", "lane", "core", "queue_wait_s")}
        append_journal(active[entry["map_id"]]/"execution_control"/"start_journal.jsonl",
            dict(identity, trial_id=entry["trial_id"], config_sha256=manifest["config_sha256"],
                 sources_sha256=sources_sha256, session_index=session["session_index"]))
        entry.update(status="STARTED", **identity)

    def complete(task, value):
        if value["status"] == "COMPLETED":
            payload = value["payload"]
            measured = payload["execution"]
            entries[task["task_id"]].update(status="RAW_READY", execution_status=measured["status"],
                raw_inventory_sha256=payload["raw_inventory_sha256"],
                planner_interval={key: measured.get(key) for key in
                                  ("pid", "process_created", "launch_utc", "process_finished_utc", "exit_code")})

    deadline = dict(stop_dispatch_at_monotonic=stop, terminate_at_monotonic=hard)
    if any(lanes):
        result = run_spawn_lanes(lanes, partial(_collect_trial, request_builder=_request),
            control/"collection", cores=admission["cores"], on_start=start, on_complete=complete, **deadline)
        session["collection_s"] += result["wall_s"]
        session["max_concurrency"] = max(session["max_concurrency"], result["max_concurrency"])
    _assert_quiescent(directory)
    dump(directory/"manifest.json", manifest)
    for mid, folder in active.items():
        if any(entry["status"] == "STARTED" for entry in grouped[mid]):
            _recover_map(folder, grouped[mid], manifest)
    dump(directory/"manifest.json", manifest)
    if time.monotonic() >= stop:
        return "PAUSED_TIME"

    admission = runtime_admission(4, config["execution"]["reserve_bytes_per_validation_worker"])
    session["validation_admission"] = admission
    if not admission["admitted"]:
        return "BLOCKED_RESOURCES"
    validation = [[] for _ in range(4)]
    for lane, (mid, folder) in enumerate(active.items()):
        spool = folder/"validation_spool"
        items = []
        for entry in grouped[mid]:
            _commit_spool(folder, entry, manifest, spool)
            if entry["status"] == "RAW_READY" and not entry.get("validation_sha256"):
                items.append(dict(entry=dict(entry), directory=str(folder/"trials"/entry["trial_id"]),
                                  raw_inventory_sha256=entry["raw_inventory_sha256"]))
        if items:
            validation[lane].append(dict(task_id=mid, info=infos[mid], items=items,
                                        sources_sha256=sources_sha256, spool=str(spool)))

    def commit(task, value):
        mid = task["task_id"]
        folder = active[mid]
        for entry in grouped[mid]:
            _commit_spool(folder, entry, manifest, folder/"validation_spool")
        dump(directory/"manifest.json", manifest)

    if any(validation):
        result = run_spawn_lanes(validation, _validate_map, control/"validation",
            cores=admission["cores"], on_complete=commit, **deadline)
        session["validation_s"] += result["wall_s"]
    for mid, folder in active.items():
        commit(dict(task_id=mid), None)
        if any(read(folder/"trials"/entry["trial_id"]/"validation.json")["correctness_errors"]
               for entry in grouped[mid] if entry.get("validation_sha256")):
            return "FAILED_CORRECTNESS"
    for mid in active:
        if time.monotonic() >= hard:
            return "PAUSED_TIME"
        if all(entry.get("validation_sha256") for entry in grouped[mid]):
            _seal(store, manifest, wave, mid)
            dump(directory/"manifest.json", manifest)
    if all(mid in manifest["shards"] for mid in maps):
        return "COMPLETE"
    return "PAUSED_TIME" if time.monotonic() >= stop else "FAILED_INFRASTRUCTURE"


def _seal(store, manifest, wave, mid):
    receipt = store.pack_map(wave, mid, cleanup=True)
    manifest["shards"][mid] = dict(archive=receipt["archive"], sha256=receipt["sha256"],
        archive_bytes=receipt["archive_bytes"], uncompressed_bytes=receipt["uncompressed_bytes"],
        inventory_sha256=digest(receipt["inventory"]))


def run(output, phase, worker, bridge, sem_python, runtime_identity, *, allow_pilot=False,
        resume=False, session_elapsed_s=0.):
    if not allow_pilot:
        raise ValueError("pilot execution requires --allow-pilot")
    if phase not in PHASES or not math.isfinite(session_elapsed_s) or session_elapsed_s < 0.:
        raise ValueError("invalid session elapsed time")
    began = time.monotonic()
    stop, hard = began+SOFT_SESSION_S-session_elapsed_s, began+WORK_DEADLINE_S-session_elapsed_s
    output = Path(output).resolve()
    directory = output/phase
    lock = _acquire_lock(directory)
    try:
        config = protocol.load_config(directory/"config.json")
        manifest = read(directory/"manifest.json")
        _check_manifest(directory, config, manifest)
        if manifest["status"] == "FAILED_CORRECTNESS":
            raise ValueError("correctness failure must be resolved in a separately registered campaign")
        if manifest["status"] != "PREPARED" and not resume:
            raise ValueError("existing phase requires --resume")
        if phase == "confirm":
            selected = _selection(output, config)
            if (manifest["selected_lambda"] != selected["selected_lambda"]
                    or manifest["selection_sha256"] != file_hash(output/"selection.json")):
                raise ValueError("confirmation selection changed")
        paths = Path(worker).resolve(), Path(bridge).resolve(), Path(executable_path(sem_python))
        runtime = _runtime_identity(runtime_identity, *paths)
        if manifest.get("runtime_identity_sha256", runtime["identity_sha256"]) != runtime["identity_sha256"]:
            raise ValueError("resume runtime identity differs")
        if phase == "confirm" and read(output/"tune"/"manifest.json")["runtime_identity_sha256"] != runtime["identity_sha256"]:
            raise ValueError("confirmation runtime differs from tuning")
        if manifest["status"] == "COMPLETE":
            return analyze(output, phase)
        _assert_quiescent(directory)
        admission = runtime_admission(4, config["execution"]["reserve_bytes_per_collection_worker"])
        session = dict(session_index=len(manifest["sessions"]), started_utc=utc_now(), admission=admission,
                       collection_s=0., validation_s=0., max_concurrency=0)
        manifest.update(runtime_identity_sha256=runtime["identity_sha256"], status="RUNNING", pilot_status="RUNNING")
        manifest["sessions"].append(session)
        dump(directory/"manifest.json", manifest)
        store = ArtifactStore(directory)
        try:
            if not admission["admitted"]:
                manifest["status"] = "BLOCKED_RESOURCES"
            else:
                grouped = _groups(manifest["entries"])
                for wave, maps in _waves(manifest["entries"]):
                    if all(mid in manifest["shards"] for mid in maps):
                        continue
                    if time.monotonic() >= stop:
                        manifest["status"] = "PAUSED_TIME"
                        break
                    current = runtime_admission(4, config["execution"]["reserve_bytes_per_collection_worker"])
                    append_journal(directory/"execution_control"/"cpu_ram_admission.jsonl", dict(current, wave=wave))
                    if not current["admitted"] or current["cores"] != admission["cores"]:
                        manifest["status"] = "BLOCKED_RESOURCES"
                        break
                    observed = max((s["uncompressed_bytes"] for s in manifest["shards"].values()), default=0)
                    raw = max(1536*1024**2, 5*observed)
                    disk = preflight(output.parent, first_wave=not manifest["shards"],
                                     raw_budget_bytes=raw, pack_budget_bytes=max(1920*1024**2, raw),
                                     output_path=output.parent)
                    append_journal(directory/"execution_control"/"disk_admission.jsonl", dict(disk, wave=wave))
                    if not disk["allowed"]:
                        manifest["status"] = "BLOCKED_DISK"
                        break
                    status = _run_wave(directory, store, manifest, config, wave, maps, grouped, paths,
                                       runtime, session, current, stop, hard)
                    if status != "COMPLETE":
                        manifest["status"] = status
                        break
                else:
                    manifest["status"] = "COMPLETE"
        except KeyboardInterrupt:
            manifest["status"] = "INTERRUPTED"
        except Exception:
            manifest["status"] = "FAILED_INFRASTRUCTURE"
            session["error"] = traceback.format_exc()
        finally:
            session.update(finished_utc=utc_now(), session_wall_s=time.monotonic()-began)
            manifest["pilot_status"] = manifest["status"]
            dump(directory/"manifest.json", manifest)
        return analyze(output, phase)
    finally:
        lock.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inventory", "prepare", "run", "analyze"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=protocol.CONFIG)
    parser.add_argument("--phase", choices=PHASES, default="tune")
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--bridge", type=Path)
    parser.add_argument("--sem-python", type=Path)
    parser.add_argument("--runtime-identity", type=Path)
    parser.add_argument("--allow-pilot", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--session-elapsed-s", type=float, default=0.)
    args = parser.parse_args(argv)
    if args.command == "inventory":
        config = protocol.load_config(args.config)
        tuning = protocol.inventory(config, "tune")
        confirmation = dict(config["profiles"]["confirm"], method_roles=["locked_candidate", protocol.CONTROL,
                            protocol.NATIVE], selection_required=True)
        value = dict(status="PREPARED_INVENTORY", tune=tuning, confirm=confirmation,
                     maximum_pilot_trials=tuning["planned_trials"]+confirmation["planned_trials"],
                     planner_execution_count=0)
        dump(args.output/"inventory.json", value)
    elif args.command == "prepare":
        value = prepare(args.output, args.config, args.phase)
    elif args.command == "analyze":
        value = analyze(args.output, args.phase)
    else:
        if not args.allow_pilot or not all((args.worker, args.bridge, args.sem_python, args.runtime_identity)):
            parser.error("run requires --allow-pilot, --worker, --bridge, --sem-python and --runtime-identity")
        value = run(args.output, args.phase, args.worker, args.bridge, args.sem_python, args.runtime_identity,
                    allow_pilot=args.allow_pilot, resume=args.resume, session_elapsed_s=args.session_elapsed_s)
    status = value.get("campaign_status", value["status"])
    print(json.dumps(dict(status=status, phase=args.phase, selection=value.get("selection"),
                          confirmation=value.get("confirmation"))))
    return 0 if status in {"PREPARED_INVENTORY", "PREPARED", "COMPLETE", *RECOVERABLE} else 1


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
