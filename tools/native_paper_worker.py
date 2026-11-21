"""Schema-11 native SEM/PUTN/SGP trial worker, with no shared SEM scorer.

Planner callbacks copy immutable candidates only. Geometry measurements, native
feature export and disk serialization occur after solve, as for the A3 worker.
Every candidate still requires the campaign's independent geometry oracle.
"""
from __future__ import annotations

import copy
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

from terrain_compare_external_worker import atomic_json, phase, write_path, THREAD_VARIABLES

OBJECTIVES = {"sem": "sem_length_turn_risk_paper_v2",
              "putn": "putn_com_traversability_paper_v2", "sgp": "sgp_xy_paper_v2"}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_request(request):
    required = {"backend", "track", "planner_id", "native_objective_id", "map_path",
                "cloud_path", "cloud_sha256", "start", "goal", "planner_seed",
                "planning_budget_s", "bridge_library", "representation_config"}
    if required - request.keys():
        raise ValueError("missing native request fields: " + ",".join(sorted(required-request.keys())))
    backend = request["backend"]
    if (backend not in OBJECTIVES or request["track"] != "native" or
            request["native_objective_id"] != OBJECTIVES[backend] or
            request["planner_id"] != backend+"_native_paper_v2"):
        raise ValueError("unregistered native planner/objective/track")
    budget_policy = request.get("algorithm_budget_policy")
    if budget_policy is not None and (budget_policy != "pri_portfolio_v1" or backend != "sem"):
        raise ValueError("unregistered native algorithm budget policy")
    for name in ("start", "goal"):
        if (len(request[name]) != 2 or
                not all(type(x) in (int, float) and math.isfinite(x) for x in request[name])):
            raise ValueError("invalid " + name)
    budget = request["planning_budget_s"]
    if type(budget) not in (int, float) or not math.isfinite(budget) or budget <= 0:
        raise ValueError("positive finite planning budget required")
    if type(request["planner_seed"]) != int or not 0 <= request["planner_seed"] < 2**64:
        raise ValueError("invalid planner seed")
    work = request.get("work_iterations", 0)
    if type(work) != int or work < 0:
        raise ValueError("invalid fixed work")
    # Optional search engine of the same SEM reconstruction: "cpp" is the C++ port (sem_rrtstar_c).
    engine = request.get("sem_engine", "python")
    if engine not in ("python", "cpp") or (engine == "cpp" and (backend != "sem" or not request.get("sem_library"))):
        raise ValueError("unregistered SEM search engine")
    if engine == "cpp" and request.get("sem_library_sha256") not in (None, file_hash(request["sem_library"])):
        raise ValueError("C++ SEM library hash mismatch")
    # Optional convergence stop of the C++ SEM search (A4 rule per iteration).
    if "sem_convergence_stop" in request:
        from sem_native_cpp import check_convergence_stop
        if engine != "cpp" or request["sem_convergence_stop"] is None:
            raise ValueError("unregistered SEM convergence stop")
        check_convergence_stop(request["sem_convergence_stop"])
    # Optional PUTN settings of the A4 comparison: the slope term of tau and the same convergence stop.
    if "putn_tau_slope_cap_deg" in request:
        cap = request["putn_tau_slope_cap_deg"]
        if backend != "putn" or type(cap) is not float or not math.isfinite(cap) or not 0 < cap <= 90:
            raise ValueError("unregistered PUTN tau slope cap")
    if "putn_convergence_stop" in request:
        from sem_native_cpp import check_convergence_stop
        if backend != "putn" or request["putn_convergence_stop"] is None:
            raise ValueError("unregistered PUTN convergence stop")
        try:
            check_convergence_stop(request["putn_convergence_stop"])
        except ValueError:
            raise ValueError("unregistered PUTN convergence stop") from None
    if file_hash(request["cloud_path"]) != request["cloud_sha256"]:
        raise ValueError("frozen point-cloud hash mismatch")
    if request.get("map_sha256") and file_hash(request["map_path"]) != request["map_sha256"]:
        raise ValueError("frozen map hash mismatch")


def _preload():
    """Loading the libraries and identifying the host are process start-up, not preprocessing:
    both happen before the clock starts."""
    platform.platform()  # cached by the platform module; the first call takes about 70 ms on Windows
    try:
        import numpy  # noqa: F401
        import scipy.ndimage  # noqa: F401  (SEMNativeModel imports it on first use)
        import sem_native_cpp  # noqa: F401
        import sem_native_planner  # noqa: F401
        import terrain_compare_bridge  # noqa: F401
    except ModuleNotFoundError:
        pass  # reported as BLOCKED_DEPENDENCY by run(), where the module is first needed


def run(request_file, output):
    _preload()
    began = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/"worker.json").exists() or (output/"progress.jsonl").exists():
        raise ValueError("native attempt already exists; refusing overwrite")
    phase(output, "SETUP")
    request = json.loads(Path(request_file).read_text(encoding="utf-8-sig"))
    result = dict(schema_version=11, backend=request.get("backend"), track="native",
                  native_objective_id=request.get("native_objective_id"),
                  status="ERROR", exact_goal=False, execution_status="COMPLETED",
                  snapshots=[], counters={}, phase_timings={}, native_model={},
                  independent_validation="NOT_RUN", provenance={
                      "python": platform.python_version(), "platform": platform.platform(),
                      "thread_environment": {v: os.environ[v] for v in THREAD_VARIABLES},
                      "worker_sha256": file_hash(__file__)},
                  callback_policy="immutable_copy_only; metrics_and_serialization_after_solve")
    terrain = native = model = None
    code = 0
    try:
        validate_request(request)
        import numpy as np
        from terrain_compare_bridge import TerrainBridge, PutnBridge
        terrain = TerrainBridge(request["bridge_library"], request["map_path"],
                                request.get("library_search_paths", []), risk_enabled=False)
        timings = result["phase_timings"]
        bt = terrain.timings
        timings.update(load_s=bt["load_s"], conversion_s=0.,
                       preprocessing_s=bt["risk_preprocessing_s"]+bt["collision_index_s"])
        conversion_start = time.perf_counter()
        points = np.load(request["cloud_path"], allow_pickle=False)
        if (points.ndim != 2 or points.shape[1] != 3 or not len(points) or
                not np.isfinite(points).all()):
            raise ValueError("invalid or empty master point cloud")
        conversion = {"master_cloud_sha256": request["cloud_sha256"],
                      "master_cloud_points": int(len(points)),
                      "representation_config": request["representation_config"]}
        if request["backend"] == "sgp":
            from native_point_cloud import voxel_centroids
            points = voxel_centroids(points, .2)
            conversion.update(training_cloud_points=int(len(points)), voxel_resolution_m=.2,
                              reduction="deterministic_3d_voxel_centroids")
        timings["conversion_s"] = time.perf_counter()-conversion_start
        model_begin = time.perf_counter()
        if request["backend"] == "sgp":
            from sgp_paper_adapter import SGPPaperModel
            from sgp_static_adapter import SGPStaticPlanner
            model = SGPPaperModel(points, terrain.bounds, request["planner_seed"])
            native = SGPStaticPlanner(terrain, model, own_cost=True)
            result["native_model"] = copy.deepcopy(model.metadata)
        elif request["backend"] == "sem":
            from sem_native_planner import SEMNativeModel, SEMNativePlanner
            model = SEMNativeModel(points, terrain.bounds)
            make_sem = lambda: SEMNativePlanner(terrain, model)
            if request.get("sem_engine", "python") == "cpp":
                # Same raster; the C++ copy and its terrain load are preprocessing, before the solve clock.
                from sem_native_cpp import SEMNativeCppEngine, SEMNativePlannerCpp
                engine = SEMNativeCppEngine(request["sem_library"], request["map_path"], model,
                                            request.get("library_search_paths", []))
                make_sem = lambda: SEMNativePlannerCpp(engine, request.get("sem_convergence_stop"))
                result["provenance"].update(sem_engine="cpp", sem_engine_version=engine.version,
                                            sem_library_sha256=file_hash(request["sem_library"]))
            # This opt-in comparison charges planner construction to the same
            # algorithm budget as the portfolio's fresh BIT* instances.
            if request.get("algorithm_budget_policy") != "pri_portfolio_v1":
                native = make_sem()
            result["native_model"] = copy.deepcopy(model.metadata)
        else:
            cap = request.get("putn_tau_slope_cap_deg")
            native = PutnBridge(terrain, paper=True, points=points, tau_slope_cap_deg=cap)
            result["native_model"] = {
                "version": "putn_pf_rrtstar_paper_v2", "voxel_resolution_m": .1,
                "plane_fit_radius_m": .5, "com_height_m": .7, "step_size_m": .4,
                "neighbor_radius_m": 1., "goal_threshold_m": 1., "goal_bias": .15,
                "native_cost": "COM_distance_times_1_plus_0.1_reciprocal_tau_minus_two",
                "tau_ge_one": "forbidden", "flatness_denominator": "N",
                "projection": "exact_H_support_xy", "nearest_metric": "euclidean_xy",
                "informed_proposal": "uniform_xy_filtered_by_com_distance_lower_bound",
                "upstream_commit": "a12c900d0965a7a7ea3d4bb7408f8ccf3afdafad"}
            if cap is not None:
                # Declared A4 adaptation: the slope term alone reaches tau = 1 at `cap` (source: 10 degrees).
                result["native_model"].update(version="putn_pf_rrtstar_paper_v2_slope_cap",
                                              tau_slope_cap_deg=cap, tau_slope_weight_per_deg=1./cap)
        timings["preprocessing_s"] += time.perf_counter()-model_begin
        result["native_model"]["conversion"] = conversion
        result["native_model"]["common_a3_risk_field_enabled"] = False
        result["native_model"]["exported_xyz_semantics"] = (
            "fitted_center_of_mass_vertices" if request["backend"] == "putn"
            else "support_vertices_lifted_by_canonical_H")
        result["native_model"]["native_geometric_length_semantics"] = (
            "sum_of_exported_vertex_chords; not_APL_or_cross_backend_score")
        result["provenance"].update(bridge_sha256=file_hash(request["bridge_library"]),
                                    map_sha256=file_hash(request["map_path"]),
                                    cloud_sha256=request["cloud_sha256"])
        total_setup = time.perf_counter()-began
        timings["setup_s"] = max(0., total_setup-sum(timings.values()))
        timings["total_before_solve_s"] = total_setup
        phase(output, "SOLVING")
        result["solve_started_utc"] = datetime.now(timezone.utc).isoformat()
        solve_begin = time.perf_counter()
        portfolio_clock = request.get("algorithm_budget_policy") == "pri_portfolio_v1"
        if portfolio_clock:
            native = make_sem()
            result["algorithm_budget_policy"] = "pri_portfolio_v1"
            result["planner_construction_in_algorithm_budget"] = True
        late_candidates = 0

        def incumbent(elapsed, path, raw, native_cost):
            nonlocal late_candidates
            if (not path or tuple(path[0]) != tuple(request["start"]) or
                    tuple(path[-1]) != tuple(request["goal"])):
                raise ValueError("native candidate has nonexact endpoints")
            if (len(path) != len(raw) or not math.isfinite(elapsed) or elapsed < 0 or
                    not math.isfinite(native_cost) or native_cost < 0 or
                    not all(math.isfinite(float(x)) for p in [*path, *raw] for x in p)):
                raise ValueError("invalid native candidate")
            # Legacy PUTN/SGP can finish one iteration across the deadline. Such
            # candidates are retained as a count, never published as incumbents.
            if elapsed > request["planning_budget_s"]:
                late_candidates += 1
                return
            previous = result["snapshots"][-1] if result["snapshots"] else None
            if previous and (native_cost >= previous["native_objective_value"] or elapsed < previous["elapsed_s"]):
                raise ValueError("native incumbent cost/time order violated")
            snapshot = dict(snapshot_id="incumbent_%06d" % len(result["snapshots"]),
                            elapsed_s=float(elapsed), timestamp_s=float(elapsed),
                            path_xy=[list(p) for p in path], raw_path_xyz=[list(p) for p in raw],
                            planner_objective_value=float(native_cost), native_objective_value=float(native_cost))
            if request["backend"] == "sem":
                snapshot["native_components"] = copy.deepcopy(native.last_components)
            if portfolio_clock:
                # Eligibility is measured after validation/copy, not backdated
                # to a planner callback that finished beyond the common budget.
                completed_elapsed = time.perf_counter()-solve_begin
                if completed_elapsed > request["planning_budget_s"]:
                    late_candidates += 1
                    return
                snapshot.update(elapsed_s=completed_elapsed, timestamp_s=completed_elapsed)
            result["snapshots"].append(snapshot)

        args = (request["start"], request["goal"], request["planner_seed"], request["planning_budget_s"])
        if request["backend"] == "putn":
            if "putn_convergence_stop" in request:
                native.set_convergence_stop(request["putn_convergence_stop"])
            rc = native.solve(*args, True, request.get("work_iterations", 0), incumbent)
            status = {1: "EXACT_SOLUTION", 0: "TIMEOUT", -2: "INVALID_NATIVE_ENDPOINTS"}.get(rc, "ERROR")
            statistics = native.statistics
            result["counters"] = {"putn_statistics": statistics, "iterations": int(statistics[0])}
        else:
            if portfolio_clock:
                status = native.solve(*args, incumbent, request.get("work_iterations", 0),
                                      algorithm_started_at=solve_begin)
            else:
                status = native.solve(*args, incumbent, request.get("work_iterations", 0))
            result["counters"] = copy.deepcopy(native.counters)
        timings["solve_s"] = time.perf_counter()-solve_begin
        outcome = dict(sem="sem_convergence", putn="putn_convergence").get(request["backend"])
        if outcome and request.get(outcome + "_stop"):
            work = request.get("work_iterations", 0)
            result[outcome] = dict(native.convergence, rule=copy.deepcopy(request[outcome + "_stop"]))
            result["termination_reason"] = (
                "CONVERGED_NO_IMPROVEMENT" if native.convergence["converged"] else
                "WORK_BUDGET" if work and result["counters"]["iterations"] >= work else
                "TIME_BUDGET" if status in ("EXACT_SOLUTION", "TIMEOUT") else status)
        result["native_reported_status"] = status
        result["counters"]["candidates_after_deadline"] = late_candidates
        if status == "EXACT_SOLUTION" and not result["snapshots"] and late_candidates:
            status = "TIMEOUT"
        result["status"] = status
        result["exact_goal"] = status == "EXACT_SOLUTION" and bool(result["snapshots"])
        if status == "EXACT_SOLUTION" and not result["snapshots"]:
            raise ValueError("native exact status without candidate")
        phase(output, "SERIALIZING")
        serialization = time.perf_counter()
        model_dir = output/"native_model"
        if request["backend"] == "sem":
            model.save(model_dir)
        elif request["backend"] == "sgp":
            model_dir.mkdir(parents=True, exist_ok=True)
            # Persist the actual GP fields used by planning, not a second evaluator.
            arrays = {key: value for key, value in model.fields.items()
                      if isinstance(value, np.ndarray)}
            np.savez_compressed(model_dir/"gp_fields.npz", mean=model.mean,
                                variance=model.variance, gradient=model.gradient,
                                bounds=np.asarray(model.bounds), **arrays)
            atomic_json(model_dir/"gp_metadata.json", result["native_model"])
        if model_dir.exists():
            result["native_model"]["artifact_directory"] = "native_model"
            result["native_model"]["raster_artifacts"] = {
                p.relative_to(model_dir).as_posix(): file_hash(p)
                for p in sorted(model_dir.rglob("*")) if p.is_file()}
        with (output/"progress.jsonl").open("w", encoding="utf-8") as progress:
            for snapshot in result["snapshots"]:
                xy, raw = snapshot["path_xy"], snapshot["raw_path_xyz"]
                measured = terrain.evaluate_path(xy)
                # These are measurements, never a surrogate for native objective.
                snapshot.update(cost_terrain_m=measured["cost_terrain_m"],
                                length_xy_m=measured["length_xy_m"],
                                native_geometric_length_m=sum(math.dist(a,b) for a,b in zip(raw,raw[1:])))
                if request["backend"] == "putn":
                    snapshot["putn_features"] = native.path_features(xy)
                write_path(output/"snapshots"/snapshot["snapshot_id"], xy, raw)
                progress.write(json.dumps(snapshot, allow_nan=False)+"\n")
        if result["snapshots"]:
            last = result["snapshots"][-1]
            result.update(final_snapshot_id=last["snapshot_id"],
                          time_to_first_solution_s=result["snapshots"][0]["elapsed_s"],
                          native_geometric_length_m=last["native_geometric_length_m"])
            write_path(output, last["path_xy"], last["raw_path_xyz"])
        timings["serialization_s"] = time.perf_counter()-serialization
    except ModuleNotFoundError as exc:
        result.update(status="BLOCKED_DEPENDENCY", error=str(exc), exact_goal=False)
        code = 3
    except Exception as exc:
        result.update(status="MODEL_ERROR" if type(exc).__name__ == "NativeModelError" else "ERROR",
                      error=str(exc), traceback=traceback.format_exc(), exact_goal=False)
        code = 1
    finally:
        if native is not None and hasattr(native, "close"):
            native.close()
        if terrain is not None:
            terrain.close()
    phase(output, "SERIALIZING")
    result["planner_status"] = result["status"]
    result["worker_wall_s"] = time.perf_counter()-began
    atomic_json(output/"worker.json", result)
    phase(output, "FINISHED")
    return code


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: native_paper_worker.py REQUEST.json OUTPUT_DIRECTORY")
    raise SystemExit(run(sys.argv[1], sys.argv[2]))
