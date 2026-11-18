"""One-process-per-trial PUTN/SGP worker: REQUEST.json OUTPUT_DIRECTORY.

The worker's completed exact solution is only a candidate success. The campaign
must independently validate it before counting success.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import traceback

THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")
for _variable in THREAD_VARIABLES:
    os.environ[_variable] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    deadline = time.perf_counter() + .25
    while True:
        try:
            os.replace(temporary, path)
            break
        except PermissionError:
            # Windows readers can briefly hold the old file without delete
            # sharing. Never unlink it: the controller must see atomic states.
            if os.name != "nt" or time.perf_counter() >= deadline:
                raise
            time.sleep(.001)


def phase(output, name):
    value = {"phase": name, "updated_utc": datetime.now(timezone.utc).isoformat(),
             "monotonic_s": time.monotonic()}
    if name == "SOLVING":
        value["solve_started_utc"] = value["updated_utc"]
    atomic_json(output / "phase.json", value)
    return value


def source_hash():
    digest = hashlib.sha256()
    for name in ("terrain_compare_external_worker.py", "terrain_compare_bridge.py", "sgp_static_adapter.py"):
        digest.update(name.encode()); digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def validate_request(request):
    required = {"backend", "track", "map_path", "start", "goal", "planner_seed",
                "planning_budget_s", "bridge_library"}
    if missing := required - request.keys():
        raise ValueError("missing request fields: " + ",".join(sorted(missing)))
    if request["backend"] not in ("sgp", "putn") or request["track"] not in ("common", "own"):
        raise ValueError("unsupported external backend/track")
    for key in ("start", "goal"):
        if len(request[key]) != 2 or not all(type(x) in (int, float) and math.isfinite(x) for x in request[key]):
            raise ValueError("invalid " + key)
    if type(request["planner_seed"]) != int or not 0 <= request["planner_seed"] < 2**64:
        raise ValueError("invalid planner_seed")
    if not math.isfinite(request["planning_budget_s"]) or request["planning_budget_s"] <= 0:
        raise ValueError("invalid planning_budget_s")
    if type(request.get("work_iterations", 0)) != int or request.get("work_iterations", 0) < 0:
        raise ValueError("invalid work_iterations")


def write_path(directory, path, raw):
    directory.mkdir(parents=True, exist_ok=True)
    for filename, header, rows in (("path_xy.csv", ["x", "y"], path),
                                   ("path_raw_xyz.csv", ["x", "y", "z"], raw)):
        with (directory / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)


def run(request_file, output):
    began = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "worker.json").exists() or (output / "progress.jsonl").exists():
        raise ValueError("trial output already started; worker refuses overwrite")
    phase(output, "SETUP")
    request = json.loads(Path(request_file).read_text(encoding="utf-8-sig"))
    result = {"schema_version": 1, "backend": request.get("backend"),
              "track": request.get("track"), "status": "ERROR", "exact_goal": False,
              "execution_status": "COMPLETED", "snapshots": [], "counters": {},
              "phase_timings": {}, "native_model": {}, "independent_validation": "NOT_RUN",
              "provenance": {"python": platform.python_version(), "platform": platform.platform(),
                             "source_sha256": source_hash(), "thread_environment":
                             {v: os.environ[v] for v in THREAD_VARIABLES}}}
    terrain = native = None
    exit_code = 0
    try:
        validate_request(request)
        from terrain_compare_bridge import TerrainBridge, PutnBridge
        terrain = TerrainBridge(request["bridge_library"], request["map_path"], request.get("library_search_paths", []))
        timing = terrain.timings
        result["phase_timings"]["load_s"] = timing["load_s"]
        result["phase_timings"]["conversion_s"] = 0.
        result["phase_timings"]["preprocessing_s"] = timing["risk_preprocessing_s"] + timing["collision_index_s"]
        result["provenance"]["bridge_sha256"] = hashlib.sha256(Path(request["bridge_library"]).read_bytes()).hexdigest()
        model_begin = time.perf_counter()
        if request["backend"] == "sgp":
            from sgp_static_adapter import SGPStaticModel, SGPStaticPlanner, raw_ground_cloud, UPSTREAM_COMMIT
            points, conversion = raw_ground_cloud(request["map_path"])
            result["phase_timings"]["conversion_s"] = time.perf_counter()-model_begin
            model_begin = time.perf_counter()
            model = SGPStaticModel(points, terrain.bounds, request["planner_seed"])
            native = SGPStaticPlanner(terrain, model, request["track"] == "own")
            result["native_model"] = dict(model.metadata, conversion=conversion)
            result["provenance"]["upstream_commit"] = UPSTREAM_COMMIT
            result["phase_timings"]["preprocessing_s"] += time.perf_counter()-model_begin
        else:
            native = PutnBridge(terrain)
            native_timing = native.statistics
            result["phase_timings"]["conversion_s"] = native_timing[11]
            result["phase_timings"]["preprocessing_s"] += native_timing[8]
            result["native_model"] = {"version": "putn_pf_rrtstar_static_v1", "voxel_resolution_m": .1,
                "ground_spacing_m": .2, "plane_fit_radius_m": .5, "com_height_m": .7,
                "step_size_m": .4, "neighbor_radius_m": 1., "goal_threshold_m": 1., "goal_bias": .15,
                "weights": {"total": 1., "flatness": 1000., "slope": .1, "sparsity": 1.},
                "vacancy_ratio_min": .02, "vacancy_ratio_max": .05, "conv_threshold": .1152,
                "projection": "exact_H_at_support_xy", "nearest_metric": "euclidean_xy",
                "native_cost": "COM_distance_reciprocal_traversability",
                "body_samples": 21, "footprint_grid": [5, 5]}
            result["provenance"]["upstream_commit"] = "a12c900d0965a7a7ea3d4bb7408f8ccf3afdafad"
        total_setup = time.perf_counter()-began
        result["phase_timings"]["setup_s"] = max(0., total_setup - sum(result["phase_timings"].values()))
        result["phase_timings"]["total_before_solve_s"] = total_setup
        phase(output, "SOLVING")
        result["solve_started_utc"] = datetime.now(timezone.utc).isoformat()
        solve_begin = time.perf_counter()
        with (output / "progress.jsonl").open("w", encoding="utf-8") as events:
            def incumbent(elapsed, path, raw, planner_cost):
                if tuple(path[0]) != tuple(request["start"]) or tuple(path[-1]) != tuple(request["goal"]):
                    raise ValueError("native incumbent does not reach exact requested endpoints")
                if len(path) != len(raw) or not all(math.isfinite(v) for p in raw for v in p):
                    raise ValueError("malformed native trajectory")
                metrics = terrain.evaluate_path(path)
                index = len(result["snapshots"])
                snapshot_id = "incumbent_%06d" % index
                snapshot = {"snapshot_id": snapshot_id, "elapsed_s": elapsed,
                    "timestamp_s": elapsed, "path_xy": path, "raw_path_xyz": raw,
                    "planner_objective_value": planner_cost,
                    "native_geometric_length_m": sum(math.dist(a, b) for a, b in zip(raw, raw[1:])), **metrics}
                if not math.isfinite(planner_cost) or planner_cost < 0:
                    raise ValueError("invalid native incumbent cost")
                if result["snapshots"] and not planner_cost < result["snapshots"][-1]["planner_objective_value"]:
                    raise ValueError("incumbent native cost did not improve")
                write_path(output / "snapshots" / snapshot_id, path, raw)
                events.write(json.dumps(snapshot, allow_nan=False) + "\n")
                events.flush()
                result["snapshots"].append(snapshot)
            if request["backend"] == "sgp":
                status = native.solve(request["start"], request["goal"], request["planner_seed"],
                    request["planning_budget_s"], incumbent, request.get("work_iterations", 0))
                result["counters"] = native.counters
            else:
                rc = native.solve(request["start"], request["goal"], request["planner_seed"],
                    request["planning_budget_s"], request["track"] == "own",
                    request.get("work_iterations", 0), incumbent)
                status = {1: "EXACT_SOLUTION", 0: "TIMEOUT", -2: "INVALID_NATIVE_ENDPOINTS"}.get(rc, "ERROR")
                result["counters"] = {"putn_statistics": native.statistics}
        result["phase_timings"]["solve_s"] = time.perf_counter()-solve_begin
        result["status"] = status
        result["exact_goal"] = bool(result["snapshots"]) and status == "EXACT_SOLUTION"
        if status == "EXACT_SOLUTION" and not result["snapshots"]:
            raise ValueError("native success without a full incumbent")
        if result["snapshots"]:
            result["final_snapshot_id"] = result["snapshots"][-1]["snapshot_id"]
            result["native_geometric_length_m"] = result["snapshots"][-1]["native_geometric_length_m"]
            result["time_to_first_solution_s"] = result["snapshots"][0]["elapsed_s"]
            result["latency_to_first_solution_s"] = total_setup + result["time_to_first_solution_s"]
            write_path(output, result["snapshots"][-1]["path_xy"], result["snapshots"][-1]["raw_path_xyz"])
    except ModuleNotFoundError as exc:
        result.update(status="BLOCKED_DEPENDENCY", error=str(exc), exact_goal=False)
        exit_code = 3
    except Exception as exc:
        result.update(status="MODEL_ERROR" if type(exc).__name__ == "NativeModelError" else "ERROR",
                      error=str(exc), traceback=traceback.format_exc(), exact_goal=False)
        exit_code = 1
    finally:
        if native is not None and hasattr(native, "close"):
            native.close()
        if terrain is not None:
            terrain.close()
    phase(output, "SERIALIZING")
    serialization = time.perf_counter()
    result["planner_status"] = result["status"]
    result["phase_timings"]["serialization_s"] = 0.
    atomic_json(output / "worker.json", result)
    result["phase_timings"]["serialization_s"] = time.perf_counter()-serialization
    result["worker_wall_s"] = time.perf_counter()-began
    atomic_json(output / "worker.json", result)
    phase(output, "FINISHED")
    return exit_code


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: terrain_compare_external_worker.py REQUEST.json OUTPUT_DIRECTORY")
    raise SystemExit(run(sys.argv[1], sys.argv[2]))
