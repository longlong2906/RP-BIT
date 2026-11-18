"""Prepare or run registered A3-T corridor v3 campaigns; smoke is the default.

Each started slot has one immutable attempt. Pilot/full require explicit flags;
full additionally requires a complete, successful, hash-bound pilot selection.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import numpy as np

from run_safe_length import dump, read, digest, utc_now, THREAD_VARS
from safe_length_field import build_field_file, file_hash, _atomic_write
from corridor_v3_protocol import (ROOT, CONFIG, PROTOCOL_ID, load_config, source_hashes,
                                    inventory, method_specs, request_for, check_selection)


def prepare(output, config_path=CONFIG, *, profile="smoke", selection_path=None, dry_run=False):
    config,output = load_config(config_path),Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("refusing to overwrite an existing campaign")
    selection = check_selection(selection_path,config) if selection_path else None
    if profile == "full" and not dry_run and selection is None:
        raise ValueError("full preparation requires a successful pilot selection artifact")
    selected_alpha = selection["selected_alpha"] if selection else None
    manifest = inventory(config,profile,selected_alpha)
    output.mkdir(parents=True,exist_ok=True)
    dump(output/"config.json",config)
    manifest.update(created_utc=utc_now(),sources=source_hashes(),maps=[],
                    objective="surface_length_subject_to_immutable_single_corridor",
                    native_baseline_objective="sem_length_turn_risk_paper_v2",
                    timing=dict(max_workers=1,threads_per_trial=1,
                        latency_origin="attempt_start_before_input_hash_field_build_and_process_launch"),
                    pilot_status="NOT_RUN",full_status="NOT_RUN")
    if selection:
        dump(output/"pilot_selection.json",selection)
        manifest.update(selection_sha256=file_hash(output/"pilot_selection.json"),selected_alpha=selected_alpha)
    if dry_run:
        dump(output/"manifest.json",manifest)
        return manifest
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
                if profile == "smoke":
                    if family == "flat_40m":
                        height,rectangles,start,goal,label = (lambda x,y:0.),[],[2.,20.],[38.,20.],"flat_full_scale_control"
                    else:
                        height,rectangles,start,goal,_,label = fixture_geometry(family)
                    definitions,params = None,dict(fixed_fixture=True,geometry_label=label)
                else:
                    height,rectangles,definitions,params = geometry_definition(family,seed)
                    definitions = definitions[:1]
                    start,goal = definitions[0]["start"],definitions[0]["goal"]
                spacing = settings["spacing_m"]
                n = round(extent/spacing)+1
                map_id = f"{family}_{seed}"
                path = output/"inputs"/(map_id+".t25d")
                metadata = write_map(path,[[height(i*spacing,j*spacing) for i in range(n)] for j in range(n)],
                    spacing=(spacing,spacing),rectangles=rectangles,
                    metadata=dict(map_id=map_id,family=family,generator_seed=seed,
                        generator_version="a3t_corridor_v3",parameters=params,
                        purpose="synthetic_geometry_for_real_planner; not_measured_sensor_data"))
                if definitions is not None:
                    metadata["queries"] = qualify_queries(Oracle(path),definitions,file_hash(path),MODEL,clearance_metrics=False)
                    dump(path.with_suffix(".json"),metadata)
                cloud = path.with_suffix(".cloud.npy")
                np.save(cloud,generate_cloud(path),allow_pickle=False)
                manifest["maps"].append(dict(map_id=map_id,family=family,map_seed=seed,map_path=str(path),
                    map_sha256=file_hash(path),metadata_sha256=file_hash(path.with_suffix(".json")),
                    cloud_path=str(cloud),cloud_sha256=file_hash(cloud),bounds=[0.,0.,extent,extent],
                    start=list(start),goal=list(goal),query_id="q0"))
        manifest["status"] = "PREPARED"
    except Exception as error:
        manifest["status"] = "PREPARATION_FAILED"
        dump(output/"qualification_failure.json",dict(error=str(error),traceback=traceback.format_exc(),
                                                      seed_replacements=[],planner_execution_count=0))
        dump(output/"manifest.json",manifest)
        raise
    dump(output/"manifest.json",manifest)
    return manifest


def execution_gate(profile, *, allow_pilot=False, allow_full=False):
    if profile not in ("smoke","pilot","full"):
        raise ValueError("unknown profile")
    if profile == "pilot" and not allow_pilot:
        raise ValueError("pilot requires --allow-pilot; NOT_RUN")
    if profile == "full" and not allow_full:
        raise ValueError("full requires --allow-full and selected pilot; NOT_RUN")


def run(output, worker, bridge=None, sem_python=None, *, profile="smoke", allow_pilot=False, allow_full=False):
    execution_gate(profile,allow_pilot=allow_pilot,allow_full=allow_full)
    from corridor_v3_validation import validate_trial, record_corridor_receipt
    from native_paper_oracle import NativePaperOracle
    from native_paper_v3_sem_evaluator import SharedSEMField
    import scipy
    output,worker = Path(output).resolve(),Path(worker).resolve()
    bridge = Path(bridge).resolve() if bridge else None
    config,manifest = load_config(output/"config.json"),read(output/"manifest.json")
    if manifest.get("status") != "PREPARED" or manifest.get("profile") != profile:
        raise ValueError("campaign is not an unattempted prepared campaign for this profile")
    if manifest["config_sha256"] != digest(config) or manifest["sources"] != source_hashes():
        raise ValueError("frozen source/config changed after preparation")
    expected = inventory(config,profile,manifest.get("selected_alpha"))["entries"]
    if digest(manifest["entries"]) != digest(expected):
        raise ValueError("prepared schedule differs from the registered complete schedule")
    if not worker.is_file():
        manifest.update(status="BLOCKED",blocked_reason="missing terrain_benchmark dependency")
        dump(output/"manifest.json",manifest)
        raise FileNotFoundError("BLOCKED: terrain_benchmark unavailable")
    if profile == "full":
        selected = output/"pilot_selection.json"
        if not selected.is_file() or file_hash(selected) != manifest.get("selection_sha256"):
            raise ValueError("full pilot selection missing or changed")
        check_selection(selected,config,binary_sha256=file_hash(worker))
    environment = dict(os.environ,**{key:"1" for key in THREAD_VARS})
    if os.name == "nt":
        environment["PATH"] = os.pathsep.join((str(worker.parent),str(ROOT/".deps/install/lib"),
            "C:/mingw64/bin",str(Path(sys.executable).parent/"Library/bin"),environment.get("PATH","")))
    manifest.update(status="RUNNING",started_utc=utc_now(),worker_path=str(worker),
        binary_sha256=file_hash(worker),worker_sha256=file_hash(worker),
        bridge_sha256=file_hash(bridge) if bridge and bridge.is_file() else None,
        environment=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),
                         thread_environment={key:environment[key] for key in THREAD_VARS}))
    dump(output/"manifest.json",manifest)
    maps = {info["map_id"]:info for info in manifest["maps"]}
    rows,oracle,field,cached_map_id = [],None,None,None
    matched_corridors = {}
    for entry in manifest["entries"]:
        info = maps[entry["map_id"]]
        directory = output/"trials"/entry["trial_id"]
        directory.mkdir(parents=True,exist_ok=False)
        attempt_started=time.perf_counter()
        execution = dict(status="NOT_RUN",field_build_s=0.,command=None,exit_code=None,started_utc=utc_now(),launched=False)
        request = None
        interrupted = False
        # Journal the slot before any field or subprocess work. Interrupted
        # campaigns cannot accidentally retry a started attempt.
        entry["status"] = "STARTED"
        dump(output/"manifest.json",manifest)
        try:
            for key,hashkey in (("map_path","map_sha256"),("cloud_path","cloud_sha256")):
                if file_hash(info[key]) != info[hashkey]:
                    raise ValueError("frozen raw input changed before trial")
            if file_hash(worker) != manifest["binary_sha256"]:
                raise ValueError("worker changed during campaign")
            native = entry["method"] == "sem_native"
            if native and (bridge is None or not bridge.is_file()):
                execution.update(status="BLOCKED",reason="native SEM requires --bridge libterrain_compare_c")
            elif native and sem_python is not None and not Path(sem_python).is_file():
                execution.update(status="BLOCKED",reason="native SEM Python interpreter unavailable")
            else:
                metadata = None
                field_path = directory/"planner_sem_field.txt"
                if not native:
                    began = time.perf_counter()
                    metadata = build_field_file(info["cloud_path"],info["map_path"],info["bounds"],field_path)
                    execution["field_build_s"] = time.perf_counter()-began
                    if file_hash(field_path) != metadata["field_sha256"]:
                        raise ValueError("planner field changed before launch")
                elif file_hash(bridge) != manifest["bridge_sha256"]:
                    raise ValueError("native bridge changed during campaign")
                request = request_for(entry,info,config,profile,field_path,metadata,bridge)
                dump(directory/"request.json",request)
                execution["request_sha256"] = file_hash(directory/"request.json")
                command = ([str(sem_python or sys.executable),str(ROOT/"tools/native_paper_worker.py")]
                           if native else [str(worker)])+[str(directory/"request.json"),str(directory)]
                execution.update(command=command,launch_utc=utc_now(),status="RUNNING")
                dump(directory/"execution.json",execution)
                began = time.perf_counter()
                with (directory/"stdout.log").open("wb") as stdout,(directory/"stderr.log").open("wb") as stderr:
                    process = subprocess.Popen(command,stdout=stdout,stderr=stderr,env=environment,cwd=ROOT)
                    try:
                        execution.update(launched=True,pid=process.pid)
                        dump(directory/"execution.json",execution)
                        returncode = process.wait(timeout=config["execution"]["process_watchdog_s"])
                    finally:
                        if process.poll() is None:
                            process.kill()
                            process.wait()
                        execution.update(exit_code=process.returncode,process_wall_s=time.perf_counter()-began)
                execution.update(status="COMPLETED" if returncode == 0 else "CRASH")
                if (directory/"worker.json").is_file() and read(directory/"worker.json").get("status") == "BLOCKED_DEPENDENCY":
                    execution.update(status="BLOCKED",reason=read(directory/"worker.json").get("error"))
                if not native and file_hash(field_path) != metadata["field_sha256"]:
                    raise ValueError("planner field changed during trial")
                if file_hash(directory/"request.json") != execution["request_sha256"]:
                    raise ValueError("request changed during trial")
        except subprocess.TimeoutExpired:
            execution.update(status="WATCHDOG",reason="process exceeded registered watchdog")
        except KeyboardInterrupt:
            interrupted = True
            execution.update(status="INTERRUPTED",reason="user interrupted collection; started slot is never retried")
        except Exception as error:
            execution.update(status="FAILED",reason=str(error),traceback=traceback.format_exc())
        execution["finished_utc"] = utc_now()
        execution["attempt_wall_before_validation_s"] = time.perf_counter()-attempt_started
        try:
            receipt = record_corridor_receipt(directory)
            if receipt:
                execution["corridor_mask_sha256"] = receipt["mask_sha256"]
            if request is None:
                row = dict(success=False,attained_safe=False,execution_status=execution["status"],
                           correctness_errors=[],events=[],unavailable_reason=execution.get("reason"))
            else:
                if cached_map_id != info["map_id"]:
                    oracle = NativePaperOracle(info["map_path"])
                    field = SharedSEMField(np.load(info["cloud_path"],allow_pickle=False),info["bounds"])
                    cached_map_id = info["map_id"]
                row = validate_trial(directory,request,info,execution,oracle=oracle,field=field)
                if receipt:
                    key=(entry["map_id"],entry.get("alpha"))
                    identity=(receipt["mask_sha256"],receipt["guide_sha256"])
                    if key in matched_corridors and matched_corridors[key]!=identity:
                        row["correctness_errors"].append("matched_corridor_or_guide_mismatch")
                        row["success"]=False
                    matched_corridors[key]=identity
        except KeyboardInterrupt:
            interrupted = True
            execution.update(process_status=execution["status"],status="INTERRUPTED",
                reason="user interrupted independent validation; raw process receipt retained")
            row = dict(success=False,attained_safe=False,execution_status="INTERRUPTED",events=[],
                correctness_errors=[],unavailable_reason="independent_validation_interrupted")
        except Exception as error:
            row = dict(success=False,attained_safe=False,execution_status=execution["status"],events=[],
                correctness_errors=["validation_exception:"+str(error)],traceback=traceback.format_exc())
        dump(directory/"execution.json",execution)
        row.update(trial_id=entry["trial_id"],method=entry["method"],map_id=entry["map_id"],family=info["family"],
                   planner_seed=entry["planner_seed"],risk_slack=entry["risk_slack"],alpha=entry.get("alpha"),
                   execution_launched=execution.get("launched",False))
        dump(directory/"validation.json",row)
        dump(directory/"raw_inventory.json",{path.relative_to(directory).as_posix():file_hash(path)
            for path in sorted(directory.rglob("*")) if path.is_file() and path.name not in ("validation.json","raw_inventory.json")})
        rows.append(row)
        entry.update(status=execution["status"],success=row["success"],validation_sha256=file_hash(directory/"validation.json"))
        dump(output/"manifest.json",manifest)
        print(__import__("json").dumps(dict(trial_id=entry["trial_id"],status=execution["status"],success=row["success"],
                                           correctness_errors=len(row["correctness_errors"]))),flush=True)
        if interrupted:
            manifest["status"] = "INTERRUPTED"
            break
        if row["correctness_errors"]:
            manifest["status"] = "FAILED_CORRECTNESS"
            break
    if manifest["status"] == "RUNNING":
        manifest["status"] = "COMPLETE" if len(rows)==len(manifest["entries"]) else "INCOMPLETE"
    manifest["finished_utc"] = utc_now()
    summary = dict(protocol_id=PROTOCOL_ID,schema_version=3,profile=profile,status=manifest["status"],
        planned=len(manifest["entries"]),attempted=len(rows),success=sum(row["success"] for row in rows),
        correctness_failures=sum(bool(row["correctness_errors"]) for row in rows),rows=rows,
        pilot_status="COMPLETE" if profile=="pilot" and manifest["status"]=="COMPLETE" else "NOT_RUN",
        full_status="COMPLETE" if profile=="full" and manifest["status"]=="COMPLETE" else "NOT_RUN",
        inference="smoke_only_no_performance_claim" if profile=="smoke" else "requires_registered_analysis_gates")
    dump(output/"manifest.json",manifest)
    dump(output/"summary.json",summary)
    lines = ["# A3-T corridor v3 campaign", "", f"Profile: {profile}; collection: {manifest['status']}.","",
             "| Method | Attempts | Independently safe final |", "|---|---:|---:|"]
    for method,_ in method_specs(profile,manifest.get("selected_alpha")):
        subset = [row for row in rows if row["method"]==method]
        lines.append(f"| {method} | {len(subset)} | {sum(row['success'] for row in subset)} |")
    lines += ["", "Every snapshot and guidance corridor is checked against independent raw-map/cloud oracles.",
        "Guides create one immutable centerline mask; only real BIT* output is scored. Exposure is not PRI.",
        "All failed attempts are retained. Smoke is correctness evidence only; no superiority claim.",""]
    _atomic_write(output/"REPORT.md","\n".join(lines).encode("utf-8"))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("prepare","run","smoke","inventory"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--config",type=Path,default=CONFIG)
    parser.add_argument("--profile",choices=("smoke","pilot","full"),default="smoke")
    parser.add_argument("--worker",type=Path)
    parser.add_argument("--bridge",type=Path)
    parser.add_argument("--sem-python",type=Path)
    parser.add_argument("--selection",type=Path)
    parser.add_argument("--allow-pilot",action="store_true")
    parser.add_argument("--allow-full",action="store_true")
    parser.add_argument("--dry-run",action="store_true")
    args = parser.parse_args(argv)
    if args.command=="smoke" and args.profile!="smoke":
        parser.error("smoke command only accepts smoke profile")
    if args.command in ("run","smoke") and not args.dry_run and not args.worker:
        parser.error("--worker required for execution")
    if args.command in ("prepare","smoke","inventory") or args.dry_run:
        result = prepare(args.output,args.config,profile=args.profile,selection_path=args.selection,
                         dry_run=args.dry_run or args.command=="inventory")
    if args.command in ("run","smoke") and not args.dry_run:
        result = run(args.output,args.worker,args.bridge,args.sem_python,profile=args.profile,
                     allow_pilot=args.allow_pilot,allow_full=args.allow_full)
    print(__import__("json").dumps({key:result[key] for key in
        ("status","planned_trials","planned","attempted","success","correctness_failures") if key in result}))
    return 0 if result["status"] in ("PREPARED","PREPARED_INVENTORY","COMPLETE") else 1


if __name__ == "__main__":
    raise SystemExit(main())
