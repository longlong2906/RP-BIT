"""Reproducible smoke campaign for constrained A3 surface-length planning.

Only smoke execution is implemented/authorized by this protocol. The separate
20-policy tuning inventory is preparation only, never a claim of pilot results.
Old native-paper inputs, source bundles and reports are not modified.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import numpy as np

from safe_length_field import build_field_file, file_hash, _atomic_write

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/a3t_safe_length_smoke.json"
METHODS = ("constrained_fixed200", "atlas_fixed200", "atlas_adaptive100_200",
           "sem_common_safe_length", "sem_native")
FAMILIES = ("flat", "tilted", "unsafe_short_safe_detour", "two_corridors_slope",
            "two_corridors_clearance", "nodata_thin_obstacle", "flat_40m")
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")


def dump(path, value):
    _atomic_write(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def load_config(path=CONFIG):
    value = read(path)
    if value.get("protocol_id") != "a3t_safe_length_smoke_v1" or value.get("schema_version") != 1:
        raise ValueError("unknown safe-length protocol")
    if value.get("profile") != "smoke":
        raise ValueError("pilot/full execution requires a separately registered protocol; NOT_RUN")
    if value.get("families") != list(FAMILIES) or value.get("methods") != list(METHODS):
        raise ValueError("smoke requires every declared fixture and method")
    if value.get("planner_seeds") != [11] or value.get("map_seed") != 101:
        raise ValueError("smoke seed replacement is forbidden")
    if value.get("planning_budget_s") != 2. or value.get("spacing_m") != .2:
        raise ValueError("smoke uses fixed 2 s budget and 0.2 m terrain grid")
    if value.get("risk_cap") != 40. or value.get("slope_cap_deg") != 35.:
        raise ValueError("smoke safety policy is fixed at risk 40 / slope 35 degrees")
    if value.get("threads_per_trial") != 1 or value.get("max_workers") != 1:
        raise ValueError("smoke timing uses one serial worker and one thread")
    if value.get("seed_replacements") != []:
        raise ValueError("seed replacements are forbidden")
    if value.get("tuning_prepare_only") != dict(risk_caps=[2., 4., 8., 16., 40.],
            slope_caps_deg=[5., 10., 20., 35.], execution_status="NOT_RUN",
            requires_new_registered_pilot_protocol=True):
        raise ValueError("unregistered tuning inventory")
    watchdog = value.get("process_watchdog_s")
    if type(watchdog) not in (int, float) or not math.isfinite(watchdog) or watchdog < 10:
        raise ValueError("invalid process watchdog")
    return value


def source_hashes():
    # Bind the Python entrypoints and the scientific code they invoke. The
    # launched binary/DLL is independently hashed at collection time.
    names = ("run_safe_length.py", "safe_length_field.py", "terrain_io.py", "native_point_cloud.py",
             "a3_traversability_protocol.py", "independent_verifier.py", "native_paper_oracle.py",
             "native_paper_v3_sem_evaluator.py", "native_paper_worker.py", "sem_native_planner.py",
             "terrain_compare_bridge.py", "native_paper_v3_protocol.py", "native_paper_protocol.py")
    result = {"tools/"+name: file_hash(ROOT/"tools"/name) for name in names}
    for folder in ("src", "include", "apps", "cmake"):
        for path in sorted((ROOT/folder).rglob("*")):
            if path.is_file():
                result[path.relative_to(ROOT).as_posix()] = file_hash(path)
    result["CMakeLists.txt"] = file_hash(ROOT/"CMakeLists.txt")
    return result


def prepare(output, config_path=CONFIG):
    """Generate fixed geometric inputs and a frozen trial schedule, no solves."""
    from a3_traversability_protocol import fixture_geometry
    from native_point_cloud import generate_cloud
    from terrain_io import write_map

    config, output = load_config(config_path), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("refusing to overwrite an existing campaign")
    output.mkdir(parents=True, exist_ok=True)
    dump(output/"config.json", config)
    maps = []
    for family in config["families"]:
        extent = 40. if family == "flat_40m" else 20.
        if family == "flat_40m":
            height, rectangles, start, goal = (lambda x, y: 0.), [], [2., 20.], [38., 20.]
            label = "flat_full_scale_control"
        else:
            height, rectangles, start, goal, _, label = fixture_geometry(family)
        spacing = config["spacing_m"]
        n = round(extent/spacing)+1
        map_id = family + "_101"
        path = output/"inputs"/(map_id+".t25d")
        metadata = write_map(path, [[height(i*spacing, j*spacing) for i in range(n)] for j in range(n)],
            spacing=(spacing, spacing), rectangles=rectangles,
            metadata=dict(map_id=map_id, family=family, generator_seed=101,
                          generator_version="safe_length_smoke_v1", geometry_label=label,
                          purpose="fixed_synthetic_geometry_for_real_planner_smoke; not_measured_sensor_data"))
        cloud = path.with_suffix(".cloud.npy")
        np.save(cloud, generate_cloud(path), allow_pickle=False)
        maps.append(dict(map_id=map_id, family=family, map_path=str(path), map_sha256=file_hash(path),
                         metadata_sha256=file_hash(path.with_suffix(".json")), cloud_path=str(cloud),
                         cloud_sha256=file_hash(cloud), bounds=[0., 0., extent, extent], start=start, goal=goal))
    entries = []
    for mi, info in enumerate(maps):
        # Rotate order across fixtures while preserving exactly one attempt.
        order = METHODS[mi % len(METHODS):] + METHODS[:mi % len(METHODS)]
        for method in order:
            entries.append(dict(trial_id=info["map_id"]+"__s11__"+method, method=method,
                                map_id=info["map_id"], planner_seed=11, status="NOT_RUN"))
    manifest = dict(protocol_id=config["protocol_id"], schema_version=1, profile="smoke", status="PREPARED",
                    created_utc=utc_now(), config_sha256=digest(config), sources=source_hashes(),
                    maps=maps, entries=entries, planned_trials=len(entries),
                    objective="surface_length_subject_to_fixed_sem_and_swept_slope_caps",
                    native_baseline_objective="sem_length_turn_risk_paper_v2",
                    seed_replacements=[], timing=dict(max_workers=1, threads_per_trial=1),
                    pilot_status="NOT_RUN", full_status="NOT_RUN")
    dump(output/"manifest.json", manifest)
    dump(output/"tuning_plan.json", dict(status="PREPARED_ONLY_NOT_RUN", policies=[
        dict(risk_cap=tau, slope_cap_deg=alpha) for tau in config["tuning_prepare_only"]["risk_caps"]
        for alpha in config["tuning_prepare_only"]["slope_caps_deg"]],
        note="No pilot/main map seeds, trial budget or selection rule registered; execution disabled."))
    return manifest


def request_for(method, info, config, field_path=None, field_metadata=None, bridge=None):
    if method not in METHODS:
        raise ValueError("unknown method")
    if method == "sem_native":
        from native_paper_v3_protocol import request_for as native_request, default_config
        request = native_request("sem", dict(start=info["start"], goal=info["goal"]),
            info["map_path"], info["cloud_path"], info["cloud_sha256"], 11,
            config["planning_budget_s"], default_config())
        request.update(map_sha256=info["map_sha256"], bridge_library=str(Path(bridge).resolve()) if bridge else None,
                       library_search_paths=[str(Path(bridge).resolve().parent), "C:/mingw64/bin"] if bridge else [],
                       variant_id="sem_native_unchanged_in_safe_length_smoke")
        return request
    common = method == "sem_common_safe_length"
    adaptive = method == "atlas_adaptive100_200"
    atlas = method in ("atlas_fixed200", "atlas_adaptive100_200")
    request = dict(planner_id="sem_rrtstar_common_v1" if common else "bitstar_a3", variant_id=method,
        map_path=info["map_path"], planner_seed=11, start=info["start"], goal=info["goal"],
        planning_budget_s=config["planning_budget_s"], samples_per_batch=100 if adaptive else 200,
        batch_after_first_solution=200 if adaptive else 0, rewire_factor=1.1, sample_attempts=100,
        progress_interval_s=.05, collision_index=True, state_cache=True, scalar_cost=True,
        sampler_query_cache=not common, exact_flat_certificate=False, objective_type="surface_length",
        cloud_phase_handshake=True, safe_length=dict(field_path=str(Path(field_path).resolve()),
            risk_cap=config["risk_cap"], slope_cap_deg=config["slope_cap_deg"], atlas_sampling=atlas,
            policy_id="a3t_safe_length_v1", **{k:field_metadata[k] for k in
                ("field_sha256", "cloud_sha256", "builder_source_sha256")}))
    if common:
        request.update(sem_range_m=1., sem_neighbor_radius_m=3.)
    return request


def _csv_path(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    return [[float(v) for v in row] for row in rows[1:]]


def _events(directory, worker):
    if "snapshots" in worker:
        return worker["snapshots"]
    path = directory/"progress.jsonl"
    return [v for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
            for v in [json.loads(line)] if v.get("event_type") == "improvement"] if path.exists() else []


def validate_trial(directory, request, info, execution, *, oracle=None, field=None):
    """Recompute every published snapshot from raw inputs with separate oracles."""
    from native_paper_oracle import NativePaperOracle
    from native_paper_v3_sem_evaluator import SharedSEMField

    directory = Path(directory)
    for key, hashkey in (("map_path", "map_sha256"), ("cloud_path", "cloud_sha256")):
        if file_hash(info[key]) != info[hashkey]:
            raise ValueError("raw input hash mismatch during validation")
    worker_path = directory/"worker.json"
    row = dict(success=False, success_geometry=False, attained_safe=False, first_safe_s=None,
               first_safe_latency_s=None, final_cost_terrain_m=None, final_pri=None,
               final_swept_slope_deg=None, events=[], correctness_errors=[],
               execution_status=execution["status"], planner_status=None,
               field_build_s=execution.get("field_build_s"), solve_time_s=None,
               selection="last_published_incumbent_within_budget_before_validation")
    if not worker_path.exists():
        row["unavailable_reason"] = "no_worker_output"
        return row
    worker = read(worker_path)
    oracle = oracle or NativePaperOracle(info["map_path"])
    field = field or SharedSEMField(np.load(info["cloud_path"], allow_pickle=False), info["bounds"])
    row["planner_status"] = worker.get("planner_status", worker.get("status"))
    row["worker_execution_status"] = worker.get("execution_status")
    row["termination_reason"] = worker.get("termination_reason")
    row["ompl_version"] = worker.get("ompl_version")
    row["compiler"] = worker.get("compiler")
    row["solve_time_s"] = worker.get("actual_solve_wall_s", worker.get("phase_timings", {}).get("solve_s"))
    constrained = "safe_length" in request
    caps = request.get("safe_length", dict(risk_cap=40., slope_cap_deg=35.))
    if constrained:
        effective = worker.get("effective_safe_length", {})
        for key in ("risk_cap", "slope_cap_deg", "atlas_sampling"):
            if effective.get(key) != caps.get(key):
                row["correctness_errors"].append("effective_safety_policy_mismatch:"+key)
        if effective.get("policy_id") != "a3t_safe_length_v1" or not effective.get("thresholds_immutable_during_solve"):
            row["correctness_errors"].append("effective_safety_contract_missing")
        if worker.get("requested_safe_length") != caps:
            row["correctness_errors"].append("requested_safety_policy_mismatch")
        policy = worker.get("effective_policy", {})
        if policy.get("safety_atlas_sampling") != caps.get("atlas_sampling"):
            row["correctness_errors"].append("effective_atlas_policy_mismatch")
        # Invalid endpoints return before planner setup, so batch keys can be
        # unavailable. A setup planner must report the actual requested policy.
        if "batch_policy" in policy:
            for key in ("batch_after_first_solution", "samples_per_batch"):
                if key == "samples_per_batch" and request["planner_id"] != "bitstar_a3":
                    continue
                if policy.get(key) != request[key]:
                    row["correctness_errors"].append("effective_batch_policy_mismatch:"+key)
    events, previous_time, previous_cost, seen = _events(directory, worker), -1., math.inf, set()
    selected = None
    before_solve = None
    if worker.get("solve_started_utc") and execution.get("launch_utc"):
        before_solve = (datetime.fromisoformat(worker["solve_started_utc"].replace("Z", "+00:00"))-
                        datetime.fromisoformat(execution["launch_utc"])).total_seconds()
        if before_solve < 0:
            raise ValueError("negative startup time")
    for event in events:
        sid = event["snapshot_id"]
        if not isinstance(sid, str) or Path(sid).name != sid or sid in (".", "..") or sid in seen:
            raise ValueError("unsafe/repeated snapshot id")
        seen.add(sid)
        elapsed = event.get("elapsed_s", event.get("timestamp_s"))
        native_cost = event.get("native_objective_value", event.get("objective_value", event.get("best_cost_terrain_m")))
        if (type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < previous_time or elapsed < 0 or
                type(native_cost) not in (int, float) or not math.isfinite(native_cost) or native_cost > previous_cost+1e-8):
            raise ValueError("invalid incumbent time/cost order")
        path = _csv_path(directory/"snapshots"/sid/"path_xy.csv")
        if event.get("path_xy") is not None and event["path_xy"] != path:
            raise ValueError("snapshot CSV and native event differ")
        check = oracle.validate_path(path, request["start"], request["goal"],
            claimed_cost=event.get("cost_terrain_m", event.get("best_cost_terrain_m")),
            geometric_safety_metrics=True, clearance_metrics=False)
        sem = field.score_path(path)
        safe = bool(check["valid"] and sem["valid"] and sem["risk_max"] <= caps["risk_cap"] and
                    check["maximum_swept_footprint_slope_deg"] <= caps["slope_cap_deg"])
        if constrained and (abs(check["cost_terrain_m"]-native_cost) > 1e-8+1e-8*max(abs(native_cost), abs(check["cost_terrain_m"]))):
            row["correctness_errors"].append(sid+":surface_length_objective_mismatch")
        if not check["valid"]:
            row["correctness_errors"].append(sid+":independent_geometry_invalid")
        if constrained and not safe:
            row["correctness_errors"].append(sid+":hard_safety_contract_failed")
        in_budget = elapsed <= request["planning_budget_s"]
        measured = dict(snapshot_id=sid, elapsed_s=elapsed, in_budget=in_budget, safe=safe,
                        geometry_valid=check["valid"], geometry=check, sem=sem)
        row["events"].append(measured)
        if in_budget:
            selected = measured
            if safe and row["first_safe_s"] is None:
                row["first_safe_s"] = elapsed
                if before_solve is not None:
                    row["first_safe_latency_s"] = execution.get("field_build_s", 0.) + before_solve + elapsed
        previous_time, previous_cost = elapsed, native_cost
    row["attained_safe"] = row["first_safe_s"] is not None and execution["status"] == "COMPLETED"
    if execution["status"] != "COMPLETED":
        row["safe_snapshot_observed_in_failed_attempt"] = row["first_safe_s"] is not None
        row["first_safe_s"], row["first_safe_latency_s"] = None, None
    row["late_snapshot_count"] = sum(not event["in_budget"] for event in row["events"])
    if constrained and events and request.get("batch_after_first_solution"):
        if worker.get("batch_transition_count") != 1:
            row["correctness_errors"].append("adaptive_batch_transition_not_observed")
    if selected is not None:
        final_path = directory/"path_xy.csv"
        if not final_path.exists() or _csv_path(final_path) != _csv_path(directory/"snapshots"/events[-1]["snapshot_id"]/"path_xy.csv"):
            row["correctness_errors"].append("final_path_not_last_published_incumbent")
        row.update(selected_snapshot_id=selected["snapshot_id"],
            success_geometry=selected["geometry_valid"], final_cost_terrain_m=selected["geometry"]["cost_terrain_m"],
            final_pri=selected["sem"]["pri"], final_swept_slope_deg=selected["geometry"]["maximum_swept_footprint_slope_deg"],
            success=bool(execution["status"] == "COMPLETED" and row["planner_status"] == "EXACT_SOLUTION" and
                         selected["safe"] and not row["correctness_errors"]))
    elif row["planner_status"] == "EXACT_SOLUTION":
        row["correctness_errors"].append("exact_solution_without_in_budget_snapshot")
    row["oracle_sources"] = {name:file_hash(ROOT/"tools"/name) for name in
                            ("independent_verifier.py", "native_paper_oracle.py", "native_paper_v3_sem_evaluator.py")}
    return row


def run(output, worker, bridge=None, sem_python=None, *, profile="smoke", allow_pilot=False, allow_full=False):
    # Explicit flags never silently map pilot/full to this smoke design.
    if profile != "smoke":
        required = "--allow-pilot" if profile == "pilot" else "--allow-full"
        if not (allow_pilot if profile == "pilot" else allow_full):
            raise ValueError(profile + " requires explicit " + required + "; NOT_RUN")
        raise ValueError(profile + " is preparation-only: register a new protocol before execution; NOT_RUN")
    from native_paper_oracle import NativePaperOracle
    from native_paper_v3_sem_evaluator import SharedSEMField
    import scipy

    output, worker = Path(output).resolve(), Path(worker).resolve()
    config, manifest = load_config(output/"config.json"), read(output/"manifest.json")
    if manifest["status"] != "PREPARED":
        raise ValueError("campaign already attempted; no automatic retry or overwrite")
    if manifest["config_sha256"] != digest(config) or manifest["sources"] != source_hashes():
        raise ValueError("frozen config/source changed after preparation")
    if not worker.is_file():
        manifest.update(status="BLOCKED", blocked_reason="missing terrain_benchmark dependency")
        dump(output/"manifest.json", manifest)
        raise FileNotFoundError("BLOCKED: terrain_benchmark is unavailable")
    environment = dict(os.environ, **{key:"1" for key in THREAD_VARS})
    if os.name == "nt":
        environment["PATH"] = os.pathsep.join((str(worker.parent), str(ROOT/".deps/install/lib"),
            "C:/mingw64/bin", str(Path(sys.executable).parent/"Library/bin"), environment.get("PATH", "")))
    manifest.update(status="RUNNING", started_utc=utc_now(), worker_path=str(worker), worker_sha256=file_hash(worker),
        bridge_sha256=file_hash(bridge) if bridge and Path(bridge).is_file() else None,
        environment=dict(python=sys.version, numpy=np.__version__, platform=platform.platform(),
                         scipy=scipy.__version__,
                         thread_environment={key:environment[key] for key in THREAD_VARS}))
    dump(output/"manifest.json", manifest)
    maps = {m["map_id"]:m for m in manifest["maps"]}
    rows, oracle_cache, field_cache = [], {}, {}
    for entry in manifest["entries"]:
        info = maps[entry["map_id"]]
        directory = output/"trials"/entry["trial_id"]
        directory.mkdir(parents=True, exist_ok=False)
        execution = dict(status="NOT_RUN", field_build_s=0., command=None, exit_code=None, started_utc=utc_now())
        request = None
        try:
            for key, hashkey in (("map_path", "map_sha256"), ("cloud_path", "cloud_sha256")):
                if file_hash(info[key]) != info[hashkey]:
                    raise ValueError("frozen input changed before trial")
            if file_hash(worker) != manifest["worker_sha256"]:
                raise ValueError("worker changed during campaign")
            native = entry["method"] == "sem_native"
            if native and (bridge is None or not Path(bridge).is_file()):
                execution.update(status="BLOCKED", reason="native SEM requires --bridge libterrain_compare_c")
            elif native and sem_python is not None and not Path(sem_python).is_file():
                execution.update(status="BLOCKED", reason="native SEM Python interpreter unavailable")
            else:
                metadata = None
                field_path = directory/"planner_sem_field.txt"
                if not native:
                    began = time.perf_counter()
                    metadata = build_field_file(info["cloud_path"], info["map_path"], info["bounds"], field_path)
                    execution["field_build_s"] = time.perf_counter()-began
                    if file_hash(field_path) != metadata["field_sha256"]:
                        raise ValueError("planner field changed before launch")
                elif file_hash(bridge) != manifest["bridge_sha256"]:
                    raise ValueError("bridge changed during campaign")
                request = request_for(entry["method"], info, config, field_path, metadata, bridge)
                dump(directory/"request.json", request)
                command = ([str(sem_python or sys.executable), str(ROOT/"tools/native_paper_worker.py")]
                           if native else [str(worker)]) + [str(directory/"request.json"), str(directory)]
                execution.update(command=command, launch_utc=utc_now(), status="RUNNING")
                dump(directory/"execution.json", execution)
                began = time.perf_counter()
                with (directory/"stdout.log").open("wb") as stdout, (directory/"stderr.log").open("wb") as stderr:
                    result = subprocess.run(command, stdout=stdout, stderr=stderr, env=environment,
                                            cwd=ROOT, timeout=config["process_watchdog_s"], check=False)
                execution.update(exit_code=result.returncode, process_wall_s=time.perf_counter()-began,
                                 status="COMPLETED" if result.returncode == 0 else "CRASH")
                if (directory/"worker.json").exists() and read(directory/"worker.json").get("status") == "BLOCKED_DEPENDENCY":
                    execution.update(status="BLOCKED", reason=read(directory/"worker.json").get("error"))
                if not native and file_hash(field_path) != metadata["field_sha256"]:
                    raise ValueError("planner field changed during trial")
        except subprocess.TimeoutExpired:
            execution.update(status="WATCHDOG", reason="process exceeded declared watchdog")
        except Exception as error:
            execution.update(status="FAILED", reason=str(error), traceback=traceback.format_exc())
        execution["finished_utc"] = utc_now()
        dump(directory/"execution.json", execution)
        try:
            if request is None:
                row = dict(success=False, attained_safe=False, execution_status=execution["status"],
                           correctness_errors=[], events=[], unavailable_reason=execution.get("reason"))
            else:
                if info["map_id"] not in oracle_cache:
                    oracle_cache[info["map_id"]] = NativePaperOracle(info["map_path"])
                    field_cache[info["map_id"]] = SharedSEMField(np.load(info["cloud_path"], allow_pickle=False), info["bounds"])
                row = validate_trial(directory, request, info, execution, oracle=oracle_cache[info["map_id"]],
                                     field=field_cache[info["map_id"]])
        except Exception as error:
            row = dict(success=False, attained_safe=False, execution_status=execution["status"], events=[],
                       correctness_errors=["validation_exception:"+str(error)], traceback=traceback.format_exc())
        row.update(trial_id=entry["trial_id"], method=entry["method"], map_id=entry["map_id"], family=info["family"])
        dump(directory/"validation.json", row)
        dump(directory/"raw_inventory.json", {p.relative_to(directory).as_posix():file_hash(p)
            for p in sorted(directory.rglob("*")) if p.is_file() and p.name not in ("validation.json", "raw_inventory.json")})
        rows.append(row)
        entry.update(status=execution["status"], success=row["success"], validation_sha256=file_hash(directory/"validation.json"))
        dump(output/"manifest.json", manifest)
        print(json.dumps(dict(trial_id=entry["trial_id"], status=execution["status"], success=row["success"],
                              correctness_errors=len(row["correctness_errors"]))), flush=True)
        if row["correctness_errors"]:
            manifest["status"] = "FAILED_CORRECTNESS"
            break
    if manifest["status"] == "RUNNING":
        manifest["status"] = "COMPLETE" if all(e["status"] == "COMPLETED" for e in manifest["entries"]) else "INCOMPLETE"
    manifest["finished_utc"] = utc_now()
    summary = dict(status=manifest["status"], planned=len(manifest["entries"]), attempted=len(rows),
        success=sum(r["success"] for r in rows), correctness_failures=sum(bool(r["correctness_errors"]) for r in rows),
        rows=rows, pilot_status="NOT_RUN", full_status="NOT_RUN", inference="smoke_only_no_performance_claim")
    dump(output/"summary.json", summary)
    dump(output/"manifest.json", manifest)
    lines = ["# Safe-length smoke", "", "Smoke tích hợp trên dữ liệu hình học tổng hợp cố định; không dùng để tuyên bố ưu thế hiệu năng.",
        "", "| Method | Completed / planned | Independently safe final | Ever safe |", "|---|---:|---:|---:|"]
    for method in METHODS:
        subset = [r for r in rows if r["method"] == method]
        lines.append(f"| {method} | {sum(r['execution_status']=='COMPLETED' for r in subset)} / {len(manifest['maps'])} | "
                     f"{sum(r['success'] for r in subset)} | {sum(r.get('attained_safe',False) for r in subset)} |")
    lines += ["", "SEM native giữ objective gốc; SEM common là adaptation dùng cùng chiều dài và ràng buộc.",
              "Mỗi nghiệm/snapshot được kiểm lại từ raw terrain và cloud; không lấy planner field làm oracle.",
              "Latency nghiệm an toàn đầu gồm build field, process startup/setup và thời gian callback; không gồm oracle offline.",
              "Pilot/full NOT_RUN. Mọi timeout/crash/BLOCKED được giữ, không thay seed hoặc chọn lại đường cuối.", ""]
    _atomic_write(output/"REPORT.md", "\n".join(lines).encode("utf-8"))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "smoke"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--bridge", type=Path)
    parser.add_argument("--sem-python", type=Path)
    parser.add_argument("--profile", choices=("smoke", "pilot", "full"), default="smoke")
    parser.add_argument("--allow-pilot", action="store_true")
    parser.add_argument("--allow-full", action="store_true")
    args = parser.parse_args(argv)
    if args.profile != "smoke":
        parser.error("pilot/full are NOT_RUN; only prepare-only tuning inventory is supported by this protocol")
    if args.command in ("run", "smoke") and not args.worker:
        parser.error("--worker is required for run/smoke")
    if args.command in ("prepare", "smoke"):
        result = prepare(args.output, args.config)
    if args.command in ("run", "smoke"):
        result = run(args.output, args.worker, args.bridge, args.sem_python, profile=args.profile,
                     allow_pilot=args.allow_pilot, allow_full=args.allow_full)
    print(json.dumps({k:result[k] for k in ("status", "planned_trials", "planned", "attempted", "success", "correctness_failures") if k in result}))
    return 0 if result["status"] in ("PREPARED", "COMPLETE") else 1


if __name__ == "__main__":
    raise SystemExit(main())
