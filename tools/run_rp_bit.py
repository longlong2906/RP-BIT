"""Local A4 / RP-BIT* campaign runner: collection on fixed cores, then independent validation.

One fresh process per trial on one pinned core; started trials are never
retried and existing trial directories are never overwritten. The held-out
phase requires --allow-heldout. Dev maps are the already-seen pilot inputs.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import platform
import sys
import time

from corridor_v4_execution import (atomic_json as dump, digest, file_hash, read, run_spawn_lanes,
                                   set_thread_limits, utc_now)

set_thread_limits()
import rp_bit_protocol as protocol

ROOT = Path(__file__).resolve().parents[1]


def phase_config(config, phase):
    """Only what this phase depends on, so later phases can be added to the file."""
    settings = config["phases"][phase]
    return dict(protocol_id=config["protocol_id"], safety=config["safety"], execution=config["execution"],
                analysis=config["analysis"], phases={phase: settings},
                methods={name: config["methods"][name] for name in settings["methods"]})


def _collect(task):
    from run_sem_exposure import _collect_trial
    return _collect_trial(task, request_builder=protocol.request_for)


def _validate_map(task):
    import numpy as np
    from native_paper_oracle import NativePaperOracle
    from native_paper_v3_sem_evaluator import SharedSEMField
    from rp_bit_validation import validate_trial
    info = task["info"]
    oracle = NativePaperOracle(info["map_path"])
    field = SharedSEMField(np.load(info["cloud_path"], allow_pickle=False), info["bounds"])
    done = []
    for item in task["items"]:
        began = time.perf_counter()
        row = validate_trial(item["directory"], item["entry"], info, task["config"],
                             oracle=oracle, field=field, full=task["full"])
        row["validation_s"] = time.perf_counter() - began
        dump(item["verdict"], row)
        done.append(item["entry"]["trial_id"])
    return dict(map_id=info["map_id"], validated=done)


def _maps(config, phase, directory):
    wanted = [(family, seed, f"{family}_{seed}") for family, seed in protocol.maps_for(config, phase)]
    kind = config["phases"][phase]["maps"]
    if kind == "dev":
        known = {item["map_id"]: item for item in read(protocol.DEV_MANIFEST)["maps"]}
        return {name: known[name] for _, _, name in wanted}
    from run_pri_pathsearch import _prepare_map
    profile = "smoke" if kind == "smoke" else "generated"
    settings = dict(profiles={profile: dict(spacing_m=.2)})
    return {name: _prepare_map(directory, dict(map_id=name, family=family, map_seed=seed, phase=profile), settings)
            for family, seed, name in wanted}


def _lanes(tasks, cores, key):
    lanes = [[] for _ in cores]
    for task in tasks:
        lanes[key(task) % len(cores)].append(task)
    return lanes


def _memory_preflight(config):
    import psutil
    execution = config["execution"]
    need = len(execution["cores"]) * execution["reserve_bytes_per_worker"]
    available = psutil.virtual_memory().available
    if need > .8 * available:
        raise SystemExit(f"BLOCKED_MEMORY: need {need} bytes for {len(execution['cores'])} lanes, {available} available")


def collect(directory, config, phase, worker, bridge, stop_dispatch_epoch=None):
    manifest = read(directory / "manifest.json")
    maps, trials = manifest["maps"], directory / "trials"
    pending = [entry for entry in manifest["entries"] if not (trials / entry["trial_id"]).exists()]
    if not pending:
        return None
    _memory_preflight(config)
    common = dict(worker=str(worker), binary_sha256=file_hash(worker), bridge=str(bridge),
                  bridge_sha256=file_hash(bridge), sem_python=sys.executable, config=config, profile=phase)
    tasks = [dict(common, task_id=entry["trial_id"], directory=str(trials / entry["trial_id"]),
                  info=maps[entry["map_id"]], entry=entry) for entry in pending]
    cores = config["execution"]["cores"]
    control = directory / "execution_control" / ("collection_" + time.strftime("%Y%m%dT%H%M%S"))
    # Optional soft deadline (epoch seconds): stop dispatching, let the active trials finish.
    stop = None if stop_dispatch_epoch is None else time.monotonic() + (stop_dispatch_epoch - time.time())
    return run_spawn_lanes(_lanes(tasks, cores, lambda task: task["entry"]["block_index"]), _collect, control, cores=cores,
                           stop_dispatch_at_monotonic=stop)


def validate(directory, config, full, methods=None):
    """Write missing verdicts (optionally only for some methods) and the summary of all existing ones."""
    manifest = read(directory / "manifest.json")
    verdicts = directory / ("validation_full" if full else "validation")
    grouped = defaultdict(list)
    for entry in manifest["entries"]:
        trial, verdict = directory / "trials" / entry["trial_id"], verdicts / (entry["trial_id"] + ".json")
        if methods and entry["method"] not in methods:
            continue
        if (trial / "raw_inventory.json").is_file() and not verdict.exists():
            grouped[entry["map_id"]].append(dict(entry=entry, directory=str(trial), verdict=str(verdict)))
    if grouped:
        _memory_preflight(config)
        tasks = [dict(task_id="validate_" + name, index=index, info=manifest["maps"][name], items=items,
                      config=config, full=full) for index, (name, items) in enumerate(sorted(grouped.items()))]
        cores = config["execution"]["cores"]
        control = directory / "execution_control" / (("validation_full_" if full else "validation_") + time.strftime("%Y%m%dT%H%M%S"))
        outcome = run_spawn_lanes(_lanes(tasks, cores, lambda task: task["index"]), _validate_map, control, cores=cores)
        if outcome["status"] != "COMPLETE":
            raise SystemExit("validation infrastructure failure: see " + str(control))
    rows = [read(verdicts / (entry["trial_id"] + ".json")) for entry in manifest["entries"]
            if (verdicts / (entry["trial_id"] + ".json")).is_file()]
    summary = dict(protocol_id=protocol.PROTOCOL_ID, phase=manifest["phase"], config_sha256=manifest["config_sha256"],
                   full_validation=full, planned=len(manifest["entries"]), validated=len(rows),
                   success=sum(row["success"] for row in rows),
                   correctness_failures=sum(bool(row["correctness_errors"]) for row in rows), rows=rows)
    dump(directory / ("summary_full.json" if full else "summary.json"), summary)
    return summary


def run(output, phase, worker, bridge, *, config_path=protocol.CONFIG, resume=False, allow_heldout=False, full=False,
        stop_dispatch_epoch=None):
    config = protocol.load_config(config_path)
    if phase not in config["phases"]:
        raise SystemExit("unregistered phase: " + phase)
    if phase.startswith("heldout") and not allow_heldout:
        raise SystemExit("held-out requires --allow-heldout after explicit user confirmation")
    worker, bridge = Path(worker).resolve(), Path(bridge).resolve()
    if not worker.is_file() or not bridge.is_file():
        raise SystemExit("BLOCKED: worker and bridge binaries are required")
    directory = Path(output).resolve() / phase
    frozen, entries = phase_config(config, phase), protocol.inventory(config, phase)
    if (directory / "manifest.json").is_file():
        manifest = read(directory / "manifest.json")
        if not resume:
            raise SystemExit("phase already exists; use --resume to dispatch only unstarted trials")
        if manifest["config_sha256"] != digest(frozen) or manifest["entries"] != entries:
            raise SystemExit("frozen phase configuration or schedule changed")
        if manifest["binary_sha256"] != file_hash(worker) or manifest["bridge_sha256"] != file_hash(bridge):
            raise SystemExit("resume requires the same worker and bridge binaries")
    else:
        if directory.exists() and any(directory.iterdir()):
            raise SystemExit("refusing to overwrite a non-empty phase directory")
        directory.mkdir(parents=True, exist_ok=True)
        manifest = dict(protocol_id=protocol.PROTOCOL_ID, phase=phase, config=frozen, config_sha256=digest(frozen),
                        entries=entries, planned_trials=len(entries), maps=_maps(config, phase, directory),
                        binary_sha256=file_hash(worker), bridge_sha256=file_hash(bridge),
                        python=sys.version, platform=platform.platform(), created_utc=utc_now(), sessions=[],
                        dev_maps_are_held_out=False if config["phases"][phase]["maps"] == "dev" else None)
        dump(directory / "manifest.json", manifest)
    session = dict(started_utc=utc_now(), argv=sys.argv[1:])
    began = time.perf_counter()
    outcome = collect(directory, frozen, phase, worker, bridge, stop_dispatch_epoch)
    session.update(collection_status=outcome["status"] if outcome else "NOTHING_PENDING",
                   collection_s=time.perf_counter() - began)
    if outcome and outcome["status"] == "DEADLINE":
        # Unstarted trials stay pending for --resume; validation waits for the complete collection.
        session.update(unstarted=len(outcome.get("unstarted_task_ids", [])), finished_utc=utc_now())
        manifest["sessions"].append(session)
        dump(directory / "manifest.json", manifest)
        print(f"{phase}: collection=DEADLINE unstarted={session['unstarted']}")
        return 3
    began = time.perf_counter()
    summary = validate(directory, frozen, full)
    session.update(validation_s=time.perf_counter() - began, validated=summary["validated"],
                   success=summary["success"], correctness_failures=summary["correctness_failures"],
                   finished_utc=utc_now())
    manifest["sessions"].append(session)
    dump(directory / "manifest.json", manifest)
    print(f"{phase}: planned={summary['planned']} validated={summary['validated']} success={summary['success']} "
          f"correctness_failures={summary['correctness_failures']} collection={session['collection_status']}")
    return 0 if outcome is None or outcome["status"] == "COMPLETE" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    go = sub.add_parser("run")
    go.add_argument("--phase", required=True)
    go.add_argument("--output", required=True)
    go.add_argument("--worker", required=True)
    go.add_argument("--bridge", required=True)
    go.add_argument("--config", default=str(protocol.CONFIG))
    go.add_argument("--resume", action="store_true")
    go.add_argument("--allow-heldout", action="store_true")
    go.add_argument("--full-validation", action="store_true")
    go.add_argument("--stop-dispatch-epoch", type=float,
                    help="epoch seconds after which no new trial starts; exit 3 leaves the rest for --resume")
    check = sub.add_parser("validate")
    check.add_argument("--phase", required=True)
    check.add_argument("--output", required=True)
    check.add_argument("--full-validation", action="store_true")
    check.add_argument("--methods", nargs="+", help="validate only these methods (e.g. strict validation of A4 trials)")
    args = parser.parse_args()
    if args.command == "run":
        return run(args.output, args.phase, args.worker, args.bridge, config_path=args.config, resume=args.resume,
                   allow_heldout=args.allow_heldout, full=args.full_validation,
                   stop_dispatch_epoch=args.stop_dispatch_epoch)
    directory = Path(args.output).resolve() / args.phase
    summary = validate(directory, read(directory / "manifest.json")["config"], args.full_validation, args.methods)
    print(f"{args.phase}: validated={summary['validated']} success={summary['success']} "
          f"correctness_failures={summary['correctness_failures']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
