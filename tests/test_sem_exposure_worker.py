"""Real OMPL worker regression on explicitly labeled synthetic test inputs."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tools"))
sys.path.insert(0, str(ROOT/"tests"))

from corridor_v4_execution import THREAD_ENV
from native_paper_oracle import NativePaperOracle
from native_paper_v3_sem_evaluator import SharedSEMField
from run_safe_length import dump
from safe_length_field import file_hash
from sem_exposure_validation import METRICS, OBJECTIVE, validate_trial
from terrain_io import write_map
from test_safe_length_oracle import write_exchange


def run(worker, directory):
    terrain = directory/"synthetic_plane_obstacle.t25d"
    write_map(terrain, [[.1*(i*.2)+.05*(j*.2) for i in range(31)] for j in range(31)],
              spacing=(.2, .2), rectangles=[(2.5, 2., 3.5, 4.)],
              metadata=dict(test_fixture=True, purpose="synthetic_sem_exposure_real_worker_regression"))
    x, y = np.meshgrid(np.arange(.025, 6., .05), np.arange(.025, 6., .05))
    cloud = directory/"synthetic_cloud.npy"
    np.save(cloud, np.column_stack((x.ravel(), y.ravel(), (.1*x+.05*y).ravel())), allow_pickle=False)
    field = SharedSEMField(np.load(cloud, allow_pickle=False), (0., 0., 6., 6.))
    exchange = directory/"synthetic_field.txt"
    write_exchange(exchange, field.valid, field.risk)
    info = dict(map_path=str(terrain), map_sha256=file_hash(terrain), cloud_path=str(cloud),
                cloud_sha256=file_hash(cloud), bounds=[0., 0., 6., 6.])
    base = dict(map_path=str(terrain), planner_id="bitstar_a3", variant_id="synthetic_sem_exposure",
                planner_seed=771, start=[.8, 3.], goal=[5.2, 3.], planning_budget_s=30., work_iterations=800,
                samples_per_batch=100, batch_after_first_solution=200, rewire_factor=1.1,
                sample_attempts=100, progress_interval_s=.005, state_cache=True, scalar_cost=True,
                collision_index=True, sampler_query_cache=True, exact_flat_certificate=False, audit_tree=True,
                objective_type=OBJECTIVE, sem_exposure={"lambda": 1., "risk_scale": 40.},
                safe_length=dict(policy_id="a3t_safe_length_v1", field_path=str(exchange),
                    field_sha256=file_hash(exchange), cloud_sha256=file_hash(cloud),
                    builder_source_sha256="labeled_synthetic_exchange_from_independent_test_field",
                    risk_cap=40., slope_cap_deg=35., atlas_sampling=True))
    env = os.environ.copy()
    env.update({name: "1" for name in THREAD_ENV})
    env["PATH"] = os.pathsep.join((str(ROOT/".deps/install/lib"), "C:/mingw64/bin", env["PATH"]))
    cases = []

    def execute(name, request, expected=0):
        out = directory/name
        out.mkdir()
        dump(out/"request.json", request)
        command = [str(worker), str(out/"request.json"), str(out)]
        completed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=50)
        (out/"stdout.log").write_text(completed.stdout, encoding="utf-8")
        (out/"stderr.log").write_text(completed.stderr, encoding="utf-8")
        dump(out/"command.json", dict(argv=command, exit_code=completed.returncode))
        assert completed.returncode == expected, (name, completed.returncode, completed.stderr)
        cases.append(dict(case=name, exit_code=completed.returncode))
        if expected:
            return None
        record = json.loads((out/"worker.json").read_text())
        progress = [json.loads(line) for line in (out/"progress.jsonl").read_text().splitlines()]
        events = [event for event in progress if event["event_type"] == "improvement"]
        assert record["planner_status"] == "EXACT_SOLUTION" and events, (name, record)
        assert record["iterations"] == 800 and record["termination_reason"] == "WORK_BUDGET", (name, record)
        assert record["tree_checkpoint"]["passed"], record["tree_checkpoint"]
        trace = [(event["best_cost_terrain_m"],
                  (out/"snapshots"/event["snapshot_id"]/"path_xy.csv").read_bytes()) for event in events]
        if request["objective_type"] == OBJECTIVE:
            checked = validate_trial(out, request, info, dict(status="COMPLETED", field_build_s=0.),
                                     oracle=NativePaperOracle(terrain), field=field)
            dump(out/"validation.json", checked)
            assert checked["success"] and not checked["correctness_errors"], (name, checked["correctness_errors"])
            assert record["effective_sem_exposure"]["lambda"] == request["sem_exposure"].get("lambda", 1.)
            assert record["effective_sem_exposure"]["risk_scale"] == 40.
            assert record["effective_policy"]["informed_cost_bound"] == "objective_value_J"
            assert record["effective_policy"]["planner_class"] == "ompl::geometric::BITstar"
            for key in ("motion_cost_calls", "motion_heuristic_calls", "from_start_heuristic_calls", "to_goal_heuristic_calls",
                        "cost_to_go_calls", "sampler_allocations", "sampler_finite_bound_calls"):
                assert record["counters"].get(key, 0)+record.get("setup_policy_counters", {}).get(key, 0) > 0, (name, key)
            for event in events:
                assert all(math.isfinite(event[key]) for key in METRICS), event
            costs = [event["objective_value"] for event in events]
            assert all(b <= a+1e-8 for a, b in zip(costs, costs[1:])), costs
        return record, trace, (out/"path_xy.csv").read_bytes()

    old_request = copy.deepcopy(base)
    old_request["objective_type"] = "surface_length"
    del old_request["sem_exposure"]
    old = execute("length_control", old_request)
    zero_request = copy.deepcopy(base)
    zero_request["sem_exposure"]["lambda"] = 0.
    zero = execute("lambda_zero", zero_request)
    assert zero[1:] == old[1:], "lambda0 changed fixed-work path/improvement trace"
    for key in ("motion_cost_calls", "motion_heuristic_calls", "sample_proposals", "samples_returned_before_validity"):
        assert zero[0]["counters"][key] == old[0]["counters"][key], key
    weighted = copy.deepcopy(base)
    weighted["sem_exposure"]["lambda"] = 4.
    result = execute("lambda_four", weighted)
    assert result[0]["objective_value"] > result[0]["cost_terrain_m"]
    assert result[0]["sem_exposure_risk_m"] > 0.
    defaults = copy.deepcopy(base)
    defaults["sem_exposure"] = {}
    execute("default_parameters", defaults)

    invalid = []
    for value in (-1., float("nan"), float("inf")):
        request = copy.deepcopy(base); request["sem_exposure"]["lambda"] = value
        invalid.append(("weight_"+str(value), request))
    request = copy.deepcopy(base); del request["safe_length"]; invalid.append(("missing_safety", request))
    request = copy.deepcopy(base); request["safe_length"].pop("field_path"); invalid.append(("missing_field", request))
    request = copy.deepcopy(base); request["sem_exposure"]["unknown"] = 1; invalid.append(("unknown_parameter", request))
    request = copy.deepcopy(base); request["sem_exposure"]["risk_scale"] = 20.; invalid.append(("wrong_scale", request))
    request = copy.deepcopy(base); request["objective_type"] = "surface_length"; invalid.append(("objective_mismatch", request))
    request = copy.deepcopy(base); del request["sem_exposure"]; invalid.append(("missing_objective_config", request))
    request = copy.deepcopy(base); request["exact_flat_certificate"] = True; invalid.append(("length_certificate", request))
    request = copy.deepcopy(base); request["safe_length"]["policy_id"] = "a3t_corridor_v4"; invalid.append(("corridor", request))
    request = copy.deepcopy(base); request["traversability"] = {}; invalid.append(("traversability", request))
    request = copy.deepcopy(base); request["planner_id"] = "bitstar_a0"; invalid.append(("wrong_backend", request))
    request = copy.deepcopy(base); request["safe_length"]["atlas_sampling"] = False; invalid.append(("no_atlas", request))
    for name, request in invalid:
        # Python's serializer intentionally emits non-standard NaN/Infinity only
        # for negative parser fixtures. Those requests must be rejected by C++.
        if any(isinstance(value, float) and not math.isfinite(value) for value in request.get("sem_exposure", {}).values()):
            out = directory/name; out.mkdir()
            (out/"request.json").write_text(json.dumps(request), encoding="utf-8")
            command = [str(worker), str(out/"request.json"), str(out)]
            completed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=10)
            (out/"stdout.log").write_text(completed.stdout, encoding="utf-8")
            (out/"stderr.log").write_text(completed.stderr, encoding="utf-8")
            dump(out/"command.json", dict(argv=command, exit_code=completed.returncode))
            assert completed.returncode == 2, (name, completed.stderr)
            cases.append(dict(case=name, exit_code=completed.returncode))
        else:
            execute(name, request, expected=2)
    dump(directory/"summary.json", dict(status="PASS", test_fixture=True, cases=cases,
         fixed_work_iterations=800, lambda_zero_bitwise_paths=True, independent_all_snapshot_validation=True))
    print(json.dumps(dict(status="PASS", cases=len(cases), real_fixed_work_trials=4,
                         lambda_zero_bitwise_paths=True, independent_all_snapshot_validation=True)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True, type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="synthetic_sem_exposure_worker_", dir=ROOT/"tests/fixtures") as folder:
        run(args.worker.resolve(), Path(folder))
