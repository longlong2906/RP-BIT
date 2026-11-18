"""Actual schema-11 backends under serial and four-lane fixed-work execution.

The generated 4 m analytic map is a correctness fixture, not a benchmark result.
Every incumbent is independently validated after all measured workers exit.
Timestamps and file hashes containing timestamps are deliberately not compared.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from experiment_common import dump, worker_env
from native_point_cloud import generate_cloud
from native_paper_protocol import (BACKENDS, OBJECTIVES, PROTOCOL_ID, SCHEMA_VERSION,
                                   default_config, request_for, sha256)
from native_paper_analysis import _validate_group, validation_code_identity
from terrain_compare_scheduler import (MIB, collect_reservations, detect_runtime,
                                       read_json, run_campaign)
from terrain_io import write_map


def coordinates(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return [[float(v) for v in row] for row in list(csv.reader(stream))[1:]]


def array_signatures(directory):
    """Compare actual numeric model arrays, independently of ZIP header clocks."""
    result = {}
    for path in sorted((directory / "native_model").glob("*.npz")):
        with np.load(path, allow_pickle=False) as arrays:
            result[path.name] = {
                key: dict(shape=list(arrays[key].shape), dtype=arrays[key].dtype.str,
                          sha256=hashlib.sha256(arrays[key].tobytes()).hexdigest())
                for key in sorted(arrays.files)}
    return result


def trace(directory, request):
    worker = read_json(directory / "worker.json")
    backend = request["backend"]
    assert worker.get("planner_status", worker.get("status")) == "EXACT_SOLUTION", worker
    source = worker.get("snapshots")
    if source is None:
        source = [json.loads(line) for line in (directory / "progress.jsonl").read_text().splitlines()
                  if line.strip() and json.loads(line).get("event_type") == "improvement"]
    assert source, (backend, "fixture must exercise real incumbent paths")
    events = []
    for event in source:
        folder = directory / "snapshots" / event["snapshot_id"]
        xy = event.get("path_xy")
        if xy is None:
            xy = coordinates(folder / "path_xy.csv")
        raw = event.get("raw_path_xyz")
        if raw is None:
            raw = coordinates(folder / "path_surface.csv")
        native = event.get("native_objective_value",
                           event.get("planner_objective_value", event.get("objective_value")))
        assert xy[0] == request["start"] and xy[-1] == request["goal"]
        assert len(xy) >= 2 and len(raw) >= 2
        # Retain exact parsed doubles: no rounding can conceal nondeterminism.
        events.append(dict(path_xy=xy, raw_xyz=raw, native_objective=native,
                           cost_terrain_m=event["cost_terrain_m"], length_xy_m=event["length_xy_m"],
                           putn_features=event.get("putn_features"),
                           native_components=event.get("native_components")))
    counters = worker["counters"]
    if backend == "a3":
        assert worker["termination_reason"] == "WORK_BUDGET", worker
        assert worker["iterations"] == request["work_iterations"]
        assert counters["traversability_motion_cost_calls"] > 0
        assert request["planner_id"] == "bitstar_a3" and request["exact_flat_certificate"] is False
        signature = dict(policy=worker["effective_policy"], iterations=worker["iterations"])
    else:
        assert worker["native_objective_id"] == OBJECTIVES[backend]
        assert worker["native_model"]["common_a3_risk_field_enabled"] is False
        assert counters["candidates_after_deadline"] == 0
        if backend == "putn":
            stats = counters["putn_statistics"]
            assert stats[0] == request["work_iterations"] and stats[1] > 0 and stats[4] > 0
            assert stats[7] > 2 and stats[10] == 1
            # The excluded positions are preprocessing, solve and conversion clocks.
            signature = dict(work_statistics=stats[:8], tree_consistent=stats[10])
        else:
            assert counters["iterations"] == request["work_iterations"]
            assert counters["edge_evaluations"] > 0
            signature = dict(counters=counters, arrays=array_signatures(directory))
            assert signature["arrays"], (backend, "real model fields were not exported")
            if backend == "sgp":
                model = worker["native_model"]
                assert model["kernel"] == "scaled_RBF_SE" and model["training_steps"] == 1
                assert model["training_observations"] == 400 and model["inducing_points"] == 400
                assert model["thread_count"] == 1 and model["interop_thread_count"] == 1
                signature["training"] = {key: model[key] for key in (
                    "kernel", "training_observations", "inducing_points", "training_loss",
                    "initial_hyperparameters", "trained_hyperparameters", "reduced_cloud_sha256")}
            else:
                assert counters["tree_insertions"] > 0
    final = coordinates(directory / "path_xy.csv")
    assert final == events[-1]["path_xy"]
    return dict(events=events, model=signature, final_xy=final)


def backend_registry(worker, sgp_python):
    # Keep the venv interpreter alias: resolving a symlink can lose site-packages.
    registry = {"a3": dict(argv=[str(worker)], cwd=str(ROOT))}
    for backend in ("sem", "putn", "sgp"):
        interpreter = sgp_python if backend in ("sem", "sgp") else sys.executable
        registry[backend] = dict(argv=[os.path.abspath(interpreter),
                                       str(ROOT / "tools/native_paper_worker.py")], cwd=str(ROOT))
    return registry


def exercise(root, worker, library, sgp_python):
    runtime = detect_runtime()
    if runtime["cpu_capacity"] < 4:
        dump(root / "result.json", dict(status="BLOCKED", test_fixture=True,
             reason="requires four actual eligible CPUs for the four-lane gate", runtime=runtime))
        raise RuntimeError("four-lane fixed-work gate requires four eligible CPUs")
    map_path = root / "analytic_flat.t25d"
    write_map(map_path, [[0.] * 21 for _ in range(21)], spacing=(.2, .2),
              metadata=dict(test_fixture=True, purpose="native_paper_fixed_work_correctness"))
    cloud_path = root / "analytic_flat.cloud.npy"
    np.save(cloud_path, generate_cloud(map_path), allow_pickle=False)
    assert np.load(cloud_path, allow_pickle=False).shape == (6400, 3)
    query, config = dict(start=[1.2, 2.], goal=[2.8, 2.]), default_config()
    entries = []
    for block, seed in enumerate((11, 22, 33, 44)):
        # A complete paired block stays on one CPU; rotate its backend order.
        for backend in BACKENDS[block:] + BACKENDS[:block]:
            request = request_for(backend, query, map_path, cloud_path, sha256(cloud_path), seed, 15., config)
            request.update(work_iterations=100, exact_flat_certificate=False)
            if backend != "a3":
                request["bridge_library"] = str(library)
                if os.name == "nt":
                    request["library_search_paths"] = [str(library.parent), "C:/mingw64/bin"]
            entries.append(dict(trial_id=f"native_s{seed}_{backend}", block_order=block,
                planned_order=len(entries), backend=backend, track="native", family="analytic_flat_fixture",
                map_id="analytic_flat", map_sha256=sha256(map_path), query_id="q0", planner_seed=seed,
                execution_status="PLANNED", request=request))
    plan = root / "plan.json"
    dump(plan, dict(schema_version=SCHEMA_VERSION, protocol_id=PROTOCOL_ID,
                    runtime_policy="native_paper_sessions_v2",
                    phase_timeouts=dict(setup_s=600.,serialize_s=60.,solve_grace_s=5.),
                    profile="correctness", test_fixture=True, entries=entries))
    registry = backend_registry(worker, sgp_python)
    old_environment = os.environ.copy()
    try:
        os.environ.update(worker_env())
        serial = run_campaign(plan, registry, root / "serial", max_workers=1,
                              reservations={backend: 512 * MIB for backend in BACKENDS}, force_local=True)
        assert serial["counts"] == {"COMPLETED": len(entries)}, ("serial", serial["counts"], str(root))
        parallel = run_campaign(plan, registry, root / "parallel", max_workers=4,
                                reservations=collect_reservations(serial), force_local=True)
        assert parallel["counts"] == {"COMPLETED": len(entries)}, ("parallel", parallel["counts"], str(root))
    finally:
        os.environ.clear()
        os.environ.update(old_environment)
    peak = max(event["concurrency"] for event in parallel["concurrency_timeline"])
    assert peak == 4, ("four actual workers did not overlap", peak)
    # This is the production offline validator: independent capsule/triangle
    # geometry for every snapshot, plus the native objective's own audit only.
    # In particular no shared SEM risk threshold or success filter is applied.
    for mode, manifest in (("serial", serial), ("parallel", parallel)):
        rows = _validate_group((str(root / mode), manifest["entries"], validation_code_identity()))
        assert len(rows) == len(entries)
        for row in rows:
            assert row["success"] and not row["errors"], (mode, row)
            assert row["events"] and all(event["independent_valid"] for event in row["events"]), (mode, row)
        dump(root / mode / "fixed_work_validation.json", dict(test_fixture=True, rows=rows))
    comparisons = []
    for entry in entries:
        name = entry["trial_id"]
        left = trace(root / "serial" / name, entry["request"])
        right = trace(root / "parallel" / name, entry["request"])
        assert left == right, ("fixed-work trace changed under concurrency", name, left, right)
        launch = read_json(root / "parallel" / name / "launch.json")
        assert len(launch["cpu_affinity"]) == 1
        assert all(value == "1" for value in launch["thread_environment"].values())
        comparisons.append(dict(trial_id=name, backend=entry["backend"], snapshots=len(left["events"]),
                                exact_goal=True, full_trace_equal=True, independently_valid=True))
    result = dict(status="PASS", test_fixture=True, purpose="correctness_not_latency_or_quality_ranking",
                  protocol_id=PROTOCOL_ID, real_backend_trials=2 * len(entries), pairs=len(comparisons),
                  actual_parallel_peak=peak, maximum_configured_workers=4, work_iterations=100,
                  compared="all incumbent XY/raw paths, native objectives/features, final paths, deterministic model/work signatures; clocks excluded",
                  comparisons=comparisons)
    dump(root / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--sgp-python", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    worker, library = args.worker.resolve(), args.library.resolve()
    for path in (worker, library, args.sgp_python):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output:
        root = args.output.resolve()
        root.mkdir(parents=True, exist_ok=False)
    else:
        # Inherit the campaign's TEMP workspace, outside frozen source trees.
        root = Path(tempfile.mkdtemp(prefix="native_paper_fixed_work_")).resolve()
    print("Native fixed-work correctness artifacts:", root, flush=True)
    try:
        result = exercise(root, worker, library, args.sgp_python)
    except BaseException as error:
        if not (root / "result.json").exists():
            dump(root / "result.json", dict(status="FAILED", test_fixture=True,
                 error=repr(error), traceback=traceback.format_exc()))
        raise
    print(json.dumps(dict(status=result["status"], pairs=result["pairs"],
                          actual_parallel_peak=result["actual_parallel_peak"], artifacts=str(root))))


if __name__ == "__main__":
    main()
