"""Real BIT* fixed-work runs in one and four Windows-compatible spawn lanes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from corridor_v4_execution import THREAD_ENV, resource_preflight, run_spawn_lanes


def solve(task):
    directory = Path(task["output"])
    directory.mkdir()
    request = directory / "request.json"
    request.write_text(json.dumps(task["request"]), encoding="utf-8")
    env = os.environ.copy()
    env["PATH"] = os.pathsep.join((str(ROOT / ".deps/install/lib"), "C:/mingw64/bin", env["PATH"]))
    with (directory / "stdout.log").open("wb") as out, (directory / "stderr.log").open("wb") as err:
        began = time.monotonic_ns()
        process = subprocess.Popen([task["worker"], str(request), str(directory)],
                                   stdout=out, stderr=err, env=env)
        try:
            code = process.wait(timeout=180)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
        ended = time.monotonic_ns()
    if code != 0:
        raise RuntimeError((directory / "stderr.log").read_text(encoding="utf-8", errors="replace"))
    worker = json.loads((directory / "worker.json").read_text())
    improvements = []
    for line in (directory / "progress.jsonl").read_text().splitlines():
        event = json.loads(line)
        if event.get("event_type") == "improvement":
            path = directory / "snapshots" / event["snapshot_id"] / "path_xy.csv"
            improvements.append([event["best_cost_terrain_m"], path.read_text()])
    if not improvements or worker["planner_status"] != "EXACT_SOLUTION":
        raise AssertionError("fixed-work fixture must exercise a genuine incumbent")
    if worker["effective_policy"]["motion_objective"] != "surface_length":
        raise AssertionError("unexpected objective")
    corridor = worker["corridor"]
    if corridor["beam_scheduler"] != "length_first_stable":
        raise AssertionError("optimized beam scheduler is not effective")
    return dict(pid=process.pid, began=began, ended=ended, seed=task["request"]["planner_seed"],
                threads={key: os.environ[key] for key in THREAD_ENV},
                signature=dict(improvements=improvements, counters=worker["counters"],
                    guide=corridor["guide_xy"], mask=corridor["mask"]))


def run(worker, base):
    from corridor_v4_protocol import PRIMARY, default_config, inventory, request_for
    from safe_length_field import file_hash
    from terrain_io import write_map

    resource = resource_preflight(4, 512*1024**2)
    if not resource["admitted"]:
        raise RuntimeError("BLOCKED_RESOURCE: " + resource["reason"])
    config = default_config(beam_scheduler="length_first_stable")
    slot = next(item for item in inventory(config)["entries"] if item["method"] == PRIMARY)
    map_path = base / "flat.t25d"
    write_map(map_path, [[0.] * 21 for _ in range(21)], spacing=(.2, .2),
              metadata=dict(test_fixture=True, purpose="fixed_work_corridor_v4"))
    field = base / "field.txt"
    # SEM field format is the same as the v3 fixture. Constant supported risk
    # leaves all variation attributable to the per-trial RNG and scheduling.
    field.write_text("SAFE_SEM_FIELD_V1\n20 20\n0 0 0.2 4 4\n"
                     + "1 1\n" * 400, encoding="utf-8")
    metadata = dict(field_sha256=file_hash(field), cloud_sha256="fixture", builder_source_sha256="fixture")
    query = dict(map_path=str(map_path), start=[.8, 1.9], goal=[3.2, 2.1])
    outputs = {}
    for jobs in (1, 4):
        lanes = [[] for _ in range(jobs)]
        for i in range(4):
            trial = f"fixture_{i}"
            request = request_for(dict(slot, planner_seed=17+i), query, config, "smoke", field, metadata)
            request.update(work_iterations=3500, planning_budget_s=90.)
            lanes[i % jobs].append(dict(task_id=trial, request=request, worker=str(worker),
                                        output=str(base / f"jobs{jobs}_{trial}")))
        result = run_spawn_lanes(lanes, solve, base / f"jobs{jobs}", cores=resource["cores"][:jobs])
        if result["status"] != "COMPLETE" or len(result["results"]) != 4:
            raise AssertionError(result["results"])
        if result["max_concurrency"] != jobs:
            raise AssertionError(f"expected concurrency {jobs}")
        outputs[jobs] = {item["task_id"]: item["payload"] for item in result["results"]}
    children = list(outputs[4].values())
    if len({item["pid"] for item in children}) != 4 or len({item["seed"] for item in children}) != 4:
        raise AssertionError("planner subprocesses and seeds must be independent")
    if not all(set(item["threads"].values()) == {"1"} for item in children):
        raise AssertionError("thread limits did not reach the worker")
    # The worker intervals begin immediately before launching their own actual
    # planner executable and end after it has exited.
    if max(item["began"] for item in children) >= min(item["ended"] for item in children):
        raise AssertionError("four actual planner subprocess intervals did not overlap")
    for trial, result in outputs[1].items():
        if result["signature"] != outputs[4][trial]["signature"]:
            raise AssertionError(f"fixed-work trace differs for {trial}")
    print(json.dumps(dict(status="PASS", paired_traces=4, real_trials=8, overlapping_planners=4)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True, type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="synthetic_corridor_v4_fixed_work_", dir=ROOT / "tests/fixtures") as folder:
        run(args.worker.resolve(), Path(folder))
