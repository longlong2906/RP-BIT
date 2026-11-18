"""Actual BIT* fixed-work parity in one/four spawn lanes, synthetic inputs only.

No pilot dataset, campaign seed, selection or scientific result is produced.
Numerical libraries are imported only by the parent fixture builder or after
the spawn entry point has applied affinity and single-thread environment.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tools"))
sys.path.insert(0, str(ROOT/"tests"))

from corridor_v4_execution import THREAD_ENV, atomic_json, file_hash, read, run_spawn_lanes
from sem_exposure_runtime import runtime_admission

WEIGHTS = (0., .5, 16., 64.)
WORK_ITERATIONS = 4000


def collect_fixed_work(task):
    """Spawn target; affinity/thread limits are already active at entry."""
    import psutil
    directory = Path(task["directory"])
    directory.mkdir()
    atomic_json(directory/"request.json", task["request"])
    env = os.environ.copy()
    if os.name == "nt":
        env["PATH"] = os.pathsep.join((str(Path(task["worker"]).parent), str(ROOT/".deps/install/lib"),
                                       "C:/mingw64/bin", env.get("PATH", "")))
    command = [task["worker"], str(directory/"request.json"), str(directory)]
    if task.get("barrier_dir"):
        # Synchronize launches only, outside the planner's solve/work budget.
        # This makes actual four-process overlap observable even on fast hosts.
        barrier = Path(task["barrier_dir"])
        atomic_json(barrier/(task["task_id"]+".json"), dict(ready=True))
        limit = time.monotonic()+30.
        while len(list(barrier.glob("weight_*.json"))) != 4:
            if time.monotonic() >= limit:
                raise AssertionError("four-lane launch barrier timed out")
            time.sleep(.01)
    began = time.monotonic_ns()
    with (directory/"stdout.log").open("wb") as stdout, (directory/"stderr.log").open("wb") as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env)
        actual_affinity = psutil.Process(process.pid).cpu_affinity()
        try:
            exit_code = process.wait(timeout=180)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    ended = time.monotonic_ns()
    atomic_json(directory/"command.json", dict(argv=command, exit_code=exit_code,
        synthetic_fixture=True, planner_pid=process.pid, cpu_affinity=actual_affinity))
    if exit_code:
        raise AssertionError((directory/"stderr.log").read_text(encoding="utf-8", errors="replace"))
    worker = read(directory/"worker.json")
    if worker["planner_status"] != "EXACT_SOLUTION":
        raise AssertionError("synthetic fixed-work fixture did not reach the exact goal")
    if worker["iterations"] != WORK_ITERATIONS or worker["termination_reason"] != "WORK_BUDGET":
        raise AssertionError("test ended by time/native stop rather than the registered work budget")
    if (worker["effective_policy"]["planner_class"] != "ompl::geometric::BITstar" or
            worker["effective_sem_exposure"]["density_version"] != "exact_canonical_raster_v1"):
        raise AssertionError("actual BIT* canonical objective contract changed")
    metrics = ("objective_value", "cost_terrain_m", "best_cost_terrain_m", "length_xy_m",
               "sem_exposure_risk_m", "sem_exposure_normalized_m")
    events = [json.loads(line) for line in (directory/"progress.jsonl").read_text().splitlines()]
    improvements = [event for event in events if event["event_type"] == "improvement"]
    if not improvements:
        raise AssertionError("fixed-work fixture requires an actual incumbent callback")
    signature = dict(final_metrics={key: worker[key] for key in metrics},
        final_path_sha256=file_hash(directory/"path_xy.csv"),
        improvements=[dict(metrics={key: event[key] for key in metrics},
            path_sha256=file_hash(directory/"snapshots"/event["snapshot_id"]/"path_xy.csv"))
            for event in improvements],
        counters=worker["counters"], setup_policy_counters=worker["setup_policy_counters"],
        iterations=worker["iterations"], batches=worker["batches"],
        batch_transition_count=worker["batch_transition_count"],
        effective_policy=worker["effective_policy"])
    return dict(signature=signature, planner_pid=process.pid, began_ns=began, ended_ns=ended,
                cpu_affinity=actual_affinity, thread_environment={name: os.environ[name] for name in THREAD_ENV},
                sem_lambda=task["request"]["sem_exposure"]["lambda"], synthetic_fixture=True)


def build_fixture(directory):
    import numpy as np
    from native_paper_v3_sem_evaluator import SharedSEMField
    from terrain_io import write_map
    from test_safe_length_oracle import write_exchange

    terrain = directory/"synthetic_plane_obstacle.t25d"
    write_map(terrain, [[.01*(i*.2)+.005*(j*.2) for i in range(31)] for j in range(31)],
              spacing=(.2, .2), rectangles=[(2.5, 2., 3.5, 4.)],
              metadata=dict(test_fixture=True, purpose="synthetic_true_BITstar_serial_parallel_fixed_work"))
    x, y = np.meshgrid(np.arange(.025, 6., .05), np.arange(.025, 6., .05))
    cloud = directory/"synthetic_cloud.npy"
    np.save(cloud, np.column_stack((x.ravel(), y.ravel(), (.01*x+.005*y).ravel())), allow_pickle=False)
    field = SharedSEMField(np.load(cloud, allow_pickle=False), (0., 0., 6., 6.))
    exchange = directory/"synthetic_field.txt"
    write_exchange(exchange, field.valid, field.risk)
    info = dict(map_path=str(terrain), map_sha256=file_hash(terrain), cloud_path=str(cloud),
                cloud_sha256=file_hash(cloud), bounds=[0., 0., 6., 6.])
    request = dict(map_path=str(terrain), planner_id="bitstar_a3", variant_id="synthetic_pilot_fixed_work",
        planner_seed=771, start=[.8, 3.], goal=[5.2, 3.], planning_budget_s=120., work_iterations=WORK_ITERATIONS,
        samples_per_batch=100, batch_after_first_solution=200, rewire_factor=1.1,
        sample_attempts=100, progress_interval_s=.01, state_cache=True, scalar_cost=True,
        collision_index=True, sampler_query_cache=True, exact_flat_certificate=False, audit_tree=True,
        objective_type="surface_length_sem_exposure", sem_exposure={"lambda": 1., "risk_scale": 40.},
        safe_length=dict(policy_id="a3t_safe_length_v1", field_path=str(exchange),
            field_sha256=file_hash(exchange), cloud_sha256=file_hash(cloud),
            builder_source_sha256="synthetic_independent_test_field", risk_cap=40., slope_cap_deg=35., atlas_sampling=True))
    return info, field, request


def run(worker, directory):
    admitted = runtime_admission(4, 512*1024**2)
    atomic_json(directory/"runtime_admission.json", admitted)
    if not admitted["admitted"]:
        raise RuntimeError("BLOCKED_RESOURCE: "+admitted["reason"])
    info, field, template = build_fixture(directory)
    outcomes, all_tasks = {}, []
    for workers in (1, 4):
        lanes = [[] for _ in range(workers)]
        for index, weight in enumerate(WEIGHTS):
            request = copy.deepcopy(template)
            request.update(planner_seed=771+index, variant_id=f"synthetic_parity_l{weight:g}")
            request["sem_exposure"]["lambda"] = weight
            task = dict(task_id=f"weight_{index}", request=request, worker=str(worker),
                        directory=str(directory/f"jobs{workers}_weight{index}"))
            if workers == 4:
                task["barrier_dir"] = str(directory/"parallel_launch_barrier")
            lanes[index % workers].append(task)
            all_tasks.append(task)
        now = time.monotonic()
        result = run_spawn_lanes(lanes, collect_fixed_work, directory/f"jobs{workers}_control",
            cores=admitted["cores"][:workers], stop_dispatch_at_monotonic=now+300., terminate_at_monotonic=now+360.)
        if result["status"] != "COMPLETE" or len(result["results"]) != len(WEIGHTS):
            errors = [{key: item.get(key) for key in ("task_id", "status", "error", "traceback")}
                      for item in result["results"] if item["status"] != "COMPLETED"]
            raise AssertionError(dict(status=result["status"], errors=errors))
        if result["max_concurrency"] != workers:
            raise AssertionError("actual spawned lane concurrency mismatch")
        outcomes[workers] = {item["task_id"]: item["payload"] for item in result["results"]}
        for index in range(4):
            proof = read(directory/f"jobs{workers}_control"/f"weight_{index}"/"child.json")
            if not proof["affinity_before_backend_import"] or set(proof["thread_environment"].values()) != {"1"}:
                raise AssertionError("child backend import affinity/thread proof is missing")
    for task_id, serial in outcomes[1].items():
        parallel = outcomes[4][task_id]
        if serial["signature"] != parallel["signature"]:
            raise AssertionError("fixed-work path/counters differ for "+task_id)
        if len(parallel["cpu_affinity"]) != 1 or set(parallel["thread_environment"].values()) != {"1"}:
            raise AssertionError("actual planner process did not inherit one CPU/single-thread environment")
    parallel = list(outcomes[4].values())
    if len({item["planner_pid"] for item in parallel}) != 4 or len({tuple(item["cpu_affinity"]) for item in parallel}) != 4:
        raise AssertionError("four actual planner processes were not pinned independently")
    if max(item["began_ns"] for item in parallel) >= min(item["ended_ns"] for item in parallel):
        raise AssertionError("four actual planner process intervals did not overlap")

    # Verification follows all collection; no oracle competes with test planners.
    from native_paper_oracle import NativePaperOracle
    from sem_exposure_validation import validate_trial
    oracle = NativePaperOracle(info["map_path"])
    snapshots = 0
    for task in all_tasks:
        out = Path(task["directory"])
        checked = validate_trial(out, task["request"], info, dict(status="COMPLETED", field_build_s=0.),
                                 oracle=oracle, field=field)
        atomic_json(out/"validation.json", checked)
        if not checked["success"] or checked["correctness_errors"]:
            raise AssertionError((task["directory"], checked["correctness_errors"]))
        snapshots += len(checked["events"])
    summary = dict(status="PASS", synthetic_fixture=True, real_bitstar_trials=8, paired_traces=4,
                   sem_lambdas=list(WEIGHTS), work_iterations=WORK_ITERATIONS,
                   bitwise_paths_and_deterministic_counters=True, overlapping_actual_planners=4,
                   independently_validated_snapshots=snapshots, scientific_pilot_trials=0)
    atomic_json(directory/"summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="synthetic_sem_pilot_fixed_work_", dir=ROOT/"tests/fixtures") as temporary:
        run(args.worker.resolve(), Path(temporary))
