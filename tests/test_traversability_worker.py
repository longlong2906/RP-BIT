"""Real fixed-work BIT* requests, retained command logs and independent costs.

Artifacts stay under the selected build directory, outside experiment results.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tools"))
from a3_traversability_protocol import DEFAULT_TRAVERSABILITY, OBJECTIVE_TYPE
from experiment_common import dump, file_hash, worker_env
from independent_verifier import Oracle
from terrain_io import write_map
from validate_results import read_path, validate_trial


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path, required=True)
    args = parser.parse_args()
    worker = args.worker.resolve()
    parent = worker.parent/"traversability_worker_regression"
    parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="run_", dir=parent))
    map_path = root/"tilted_obstacle.t25d"
    write_map(map_path, [[.12*i+.08*j for i in range(11)] for j in range(11)],
              radius=.3, margin=.1, slope=35., rectangles=[(4.2, 3., 5.8, 7.)],
              metadata={"test_fixture": True, "purpose": "fixed_work_worker_regression"})
    oracle = Oracle(map_path)
    evidence = []

    def run(name, *, risk=None, backend="bitstar_a3", expected_exit=0,
            expected_status="EXACT_SOLUTION", **options):
        output = root/name
        output.mkdir()
        request = {"map_path": str(map_path), "planner_id": backend, "planner_seed": 771,
                   "start": [1., 5.], "goal": [9., 5.], "planning_budget_s": 15.,
                   "work_iterations": 500, "samples_per_batch": 50, "rewire_factor": 1.1,
                   "sample_attempts": 100, "progress_interval_s": .05}
        if risk is not None:
            request.update(variant_id=name, objective_type=OBJECTIVE_TYPE,
                           traversability=copy.deepcopy(risk))
        request.update(options)
        dump(output/"request.json", request)
        command = [str(worker), str(output/"request.json"), str(output)]
        process = subprocess.run(command, capture_output=True, text=True,
                                 env=worker_env(), timeout=35)
        (output/"stdout.log").write_text(process.stdout, encoding="utf-8")
        (output/"stderr.log").write_text(process.stderr, encoding="utf-8")
        dump(output/"command.json", {"argv": command, "exit_code": process.returncode})
        assert process.returncode == expected_exit, (name, process.returncode, process.stderr)
        evidence.append({"case": name, "exit_code": process.returncode})
        if expected_exit:
            assert not (output/"worker.json").exists(), name
            return None
        record = json.loads((output/"worker.json").read_text())
        assert record["planner_status"] == expected_status, (name, record)
        # Worker output is provisional; independent verification owns success.
        assert record["success"] is False and record["independent_valid"] is None, name
        if risk is None:
            assert "objective_type" not in record and "objective_value" not in record, name
        else:
            assert record["objective_type"] == OBJECTIVE_TYPE, name
            assert record["requested_traversability"] == risk, name
            assert record["effective_traversability"] == risk, name
        if expected_status != "EXACT_SOLUTION":
            assert record["path_available"] is False and record["first_solution_time_s"] is None, name
            assert record["objective_value"] is None and record["risk_integral_m"] is None, name
            assert not (output/"path_xy.csv").exists(), name
            assert (output/"progress.jsonl").read_text() == "", name
            dump(output/"trial.json", dict(record, map_hash=file_hash(map_path)))
            wrapped = validate_trial(output, oracle)
            assert wrapped["success"] is False and wrapped["independent_valid"] is None, name
            return record
        assert record["termination_reason"] == "WORK_BUDGET", (name, record["termination_reason"])
        assert record["iterations"] == request["work_iterations"], (name, record["iterations"])
        assert record["state_dimension"] == 2, name
        assert record["effective_policy"]["nearest_neighbor_metric"] == "euclidean_xy", name
        for flag in ("state_cache", "scalar_cost", "collision_index"):
            assert record["effective_policy"][flag] == options.get(flag, False), (name, flag)

        def validate(directory, metrics):
            xy, surface = read_path(directory/"path_xy.csv"), read_path(directory/"path_surface.csv")
            kwargs = {} if risk is None else {"traversability": risk, "claimed_metrics": metrics}
            checked = oracle.validate_path(xy, request["start"], request["goal"],
                metrics["best_cost_terrain_m"], surface, **kwargs)
            dump(directory/"independent_test_validation.json", checked)
            assert checked["valid"], (name, checked)
            if risk is not None:
                assert close(metrics["objective_value"], metrics["cost_terrain_m"]+risk["lambda"]*metrics["risk_integral_m"]), name
                assert close(metrics["risk_integral_m"], risk["slope_weight"]*metrics["risk_slope_integral_m"]+
                             risk["clearance_weight"]*metrics["risk_clearance_integral_m"]), name
                assert metrics["objective_value"]+1e-8 >= metrics["cost_terrain_m"], name
                assert all(0. <= metrics[key] <= metrics["cost_terrain_m"]+1e-8 for key in
                           ("risk_integral_m", "risk_slope_integral_m", "risk_clearance_integral_m")), name
            return xy

        final_xy = validate(output, record)
        events = [json.loads(line) for line in (output/"progress.jsonl").read_text().splitlines()]
        improvements = [e for e in events if e["event_type"] == "improvement"]
        assert improvements, name
        snapshots = []
        previous = math.inf
        for event in improvements:
            xy = validate(output/"snapshots"/event["snapshot_id"], event)
            value = event["objective_value"] if risk is not None else event["best_cost_terrain_m"]
            assert value <= previous+1e-8, (name, "objective increased")
            assert event["elapsed_s"] <= request["planning_budget_s"], name
            previous = value
            snapshots.append((value, xy))
        if risk is not None:
            assert record["counters"]["traversability_motion_cost_calls"] > 0, name
            assert record["traversability_preprocessing_s"] >= 0., name
        # Exercise the exact runner-facing validation path, including objective
        # policy dictionaries and snapshot metadata, rather than only the Oracle.
        dump(output/"trial.json", dict(record, map_hash=file_hash(map_path)))
        wrapped = validate_trial(output, oracle)
        assert wrapped["success"] and wrapped["independent_valid"] and wrapped["snapshots_valid"], (name, wrapped)
        if risk is not None:
            assert close(wrapped["independently_recomputed_objective_value"], record["objective_value"]), name
        return {"record": record, "snapshots": snapshots, "final_xy": final_xy}

    def assert_equivalent(a, b, label):
        assert len(a["snapshots"]) == len(b["snapshots"]), (label, "snapshot count")
        for left, right in zip(a["snapshots"], b["snapshots"]):
            assert close(left[0], right[0]), (label, left[0], right[0])
            assert len(left[1]) == len(right[1]), (label, "path vertex count")
            assert all(math.dist(x, y) <= 1e-8 for x, y in zip(left[1], right[1])), (label, "path changed")
        assert len(a["final_xy"]) == len(b["final_xy"]), (label, "final vertex count")
        assert all(math.dist(x, y) <= 1e-8 for x, y in zip(a["final_xy"], b["final_xy"])), label

    legacy = run("legacy")
    zero = dict(DEFAULT_TRAVERSABILITY, **{"lambda": 0.})
    lambda0 = run("lambda0", risk=zero)
    assert_equivalent(legacy, lambda0, "lambda0 retains surface-only seeded path sequence")
    risk = run("risk", risk=DEFAULT_TRAVERSABILITY)
    optimized = run("risk_optimized", risk=DEFAULT_TRAVERSABILITY,
                    state_cache=True, scalar_cost=True, collision_index=True)
    assert_equivalent(risk, optimized, "optimized retains weighted seeded path sequence")
    run("a0_risk", risk=DEFAULT_TRAVERSABILITY, backend="bitstar_a0")
    run("invalid_start", risk=DEFAULT_TRAVERSABILITY, start=[5., 5.], expected_status="INVALID_START")
    run("negative_lambda", risk=dict(DEFAULT_TRAVERSABILITY, **{"lambda": -1.}), expected_exit=2)
    run("unknown_feature_field", risk=dict(DEFAULT_TRAVERSABILITY, roughness_weight=.2), expected_exit=2)
    run("unsupported_backend", risk=DEFAULT_TRAVERSABILITY, backend="rrtstar", expected_exit=2)
    dump(root/"summary.json", {"passed": True, "test_fixture": True, "cases": evidence,
                              "work_iterations": 500, "scientific_ranking": False})
    print("PASS: traversability worker J/L/R, independent snapshots, lambda0 and optimized fixed-work equivalence. " + str(root))


if __name__ == "__main__":
    main()
