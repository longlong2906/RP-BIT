"""Real fixed-work A3 variants and independent path checks, outside results/."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from experiment_common import dump, worker_env
from independent_verifier import Oracle
from terrain_io import write_map
from validate_results import read_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--policy-test", type=Path)
    args = parser.parse_args()
    parent = args.worker.resolve().parent / "a3_worker_regression"
    parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="run_", dir=parent))
    if args.policy_test:
        traces, height_queries = [], []
        for mode in (0, 1, 2):
            command = [str(args.policy_test.resolve()), "--fixed-work", str(mode)]
            process = subprocess.run(command, capture_output=True, text=True, env=worker_env(), timeout=30)
            (root / f"policy_{mode}.stdout.log").write_text(process.stdout, encoding="utf-8")
            (root / f"policy_{mode}.stderr.log").write_text(process.stderr, encoding="utf-8")
            dump(root / f"policy_{mode}.json", {"argv": command, "exit_code": process.returncode})
            assert process.returncode == 0, (mode, process.stderr)
            traces.append(next(line for line in process.stdout.splitlines() if line.startswith("TRACE ")))
            height_queries.append(int(next(line for line in process.stdout.splitlines()
                                           if line.startswith("HEIGHT_QUERIES ")).split()[1]))
        assert traces[0] == traces[1] == traces[2], "native policy fixed-work traces differ"
        assert height_queries[1] < height_queries[0] and height_queries[1] == height_queries[2]
        dump(root / "fixed_work_equivalence.json", {"passed": True, "height_queries": height_queries,
             "modes": ["original", "cache", "cache_scalar"], "trace_equal": True})
        print("Fixed-work native traces identical; actual height queries:", height_queries, flush=True)
    map_path = root / "folded.t25d"
    heights = [[0.4 * (5 - abs(i - 5)) + .03 * j for i in range(11)] for j in range(11)]
    write_map(map_path, heights, radius=.1, margin=.02, slope=70,
              rectangles=[(4.2, 3.0, 5.8, 7.0)], metadata={"fixture": True})
    oracle = Oracle(map_path)

    def run(name, backend="bitstar_a3", expected_exit=0, label=True, **options):
        output = root / name
        output.mkdir()
        request = {"map_path": str(map_path), "planner_id": backend, "planner_seed": 771,
                   "start": [1., 5.], "goal": [9., 5.], "planning_budget_s": 15.,
                   "work_iterations": 2000, "samples_per_batch": 50, "rewire_factor": 1.1,
                   "sample_attempts": 100, "progress_interval_s": .05}
        if label and options:
            request["variant_id"] = name
        request.update(options)
        dump(output / "request.json", request)
        command = [str(args.worker.resolve()), str(output / "request.json"), str(output)]
        process = subprocess.run(command, capture_output=True, text=True, env=worker_env(), timeout=30)
        (output / "stdout.log").write_text(process.stdout, encoding="utf-8")
        (output / "stderr.log").write_text(process.stderr, encoding="utf-8")
        dump(output / "command.json", {"argv": command, "exit_code": process.returncode})
        assert process.returncode == expected_exit, (name, process.stderr)
        if expected_exit:
            assert not (output / "worker.json").exists()
            return None
        record = json.loads((output / "worker.json").read_text())
        assert record["planner_id"] == request.get("variant_id", backend), record
        assert record["backend_planner_id"] == backend, record
        assert record["planner_status"] == "EXACT_SOLUTION", record
        assert record["termination_reason"] == "WORK_BUDGET", record
        assert record["iterations"] == request["work_iterations"], record
        effective = record["effective_policy"]
        for flag in ("state_cache", "scalar_cost", "collision_index"):
            assert effective[flag] == options.get(flag, False), (name, flag, effective)
        assert record["state_dimension"] == 2 and effective["nearest_neighbor_metric"] == "euclidean_xy"
        checked = oracle.validate_path(read_path(output / "path_xy.csv"), request["start"], request["goal"],
                                       record["best_cost_terrain_m"], read_path(output / "path_surface.csv"))
        assert checked["valid"], (name, checked)
        events = [json.loads(line) for line in (output / "progress.jsonl").read_text().splitlines()]
        improvements = [event for event in events if event["event_type"] == "improvement"]
        assert improvements
        snapshots = []
        for event in improvements:
            directory = output / "snapshots" / event["snapshot_id"]
            check = oracle.validate_path(read_path(directory / "path_xy.csv"), request["start"], request["goal"],
                                         event["best_cost_terrain_m"], read_path(directory / "path_surface.csv"))
            assert check["valid"], (name, check)
            snapshots.append((event["best_cost_terrain_m"], (directory / "path_xy.csv").read_bytes()))
        record["test_snapshots"] = snapshots
        record["test_final_xy"] = (output / "path_xy.csv").read_bytes()
        return record

    original = run("original")
    variants = [("cache", {"state_cache": True}), ("scalar", {"scalar_cost": True}),
                ("bvh", {"collision_index": True}),
                ("optimized", {"state_cache": True, "scalar_cost": True, "collision_index": True})]
    for name, flags in variants:
        record = run(name, **flags)
        assert record["test_snapshots"] == original["test_snapshots"], (name, "incumbent sequence changed")
        assert record["test_final_xy"] == original["test_final_xy"], (name, "final path changed")
        assert record["best_cost_terrain_m"] == original["best_cost_terrain_m"], name
        if flags.get("state_cache"):
            assert record["counters"]["height_cache_hits"] + record["counters"]["fixed_heuristic_cache_hits"] > 0
            assert record["counters"]["height_queries"] < original["counters"]["height_queries"]
        if flags.get("collision_index"):
            assert record["counters"]["bvh_node_visits"] > 0
            assert record["collision_index_build_s"] >= 0
        assert record["batch_transition_count"] == 0
    for size in (50, 100, 200, 400):
        fixed = run("fixed_" + str(size), samples_per_batch=size)
        assert fixed["effective_policy"]["samples_per_batch"] == size
        assert fixed["effective_policy"]["samples_per_batch_final"] == size
    scheduled = run("two_phase", batch_after_first_solution=200)
    assert scheduled["batch_transition_count"] == 1
    assert len(scheduled["batch_transitions"]) == 1
    assert scheduled["effective_policy"]["samples_per_batch"] == 50
    assert scheduled["effective_policy"]["samples_per_batch_final"] == 200
    assert scheduled["batches"] > scheduled["batch_transitions"][0]["after_batch"], "no later batch exercised"
    # Shared geometry is available to comparison planners, with native policies retained.
    for planner in ("abitstar", "aitstar", "lazyprmstar_hauser2015"):
        run(planner + "_shared", backend=planner, scalar_cost=True, collision_index=True,
            work_iterations=500)
    run("unlabelled_cache", expected_exit=2, label=False, state_cache=True)
    run("wrong_backend_schedule", backend="abitstar", expected_exit=2, batch_after_first_solution=200)
    run("noop_schedule", expected_exit=2, batch_after_first_solution=50)
    print("PASS: fixed-work variants preserve exact incumbent paths; batch policies, shared geometry, independent oracle. " + str(root))


if __name__ == "__main__":
    main()
