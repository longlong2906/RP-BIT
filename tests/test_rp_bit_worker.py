"""Real RP-BIT* worker regression on explicitly labeled synthetic test inputs."""
import argparse
import copy
import json
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
from pri_portfolio_validation import sharp_turns
from rp_bit_validation import validate_trial as score_trial
from run_safe_length import _csv_path, dump
from safe_length_field import file_hash
from sem_exposure_validation import OBJECTIVE, validate_trial
from terrain_io import write_map
from test_safe_length_oracle import write_exchange

WORK = 2500


def run(worker, directory):
    terrain = directory/"synthetic_plane_obstacle.t25d"
    write_map(terrain, [[.1*(i*.2)+.05*(j*.2) for i in range(31)] for j in range(31)],
              spacing=(.2, .2), rectangles=[(2.5, 2., 3.5, 4.)],
              metadata=dict(test_fixture=True, purpose="synthetic_rp_bit_real_worker_regression"))
    x, y = np.meshgrid(np.arange(.025, 6., .05), np.arange(.025, 6., .05))
    cloud = directory/"synthetic_cloud.npy"
    np.save(cloud, np.column_stack((x.ravel(), y.ravel(), (.1*x+.05*y).ravel())), allow_pickle=False)
    field = SharedSEMField(np.load(cloud, allow_pickle=False), (0., 0., 6., 6.))
    oracle = NativePaperOracle(terrain)
    exchange = directory/"synthetic_field.txt"
    write_exchange(exchange, field.valid, field.risk)
    info = dict(map_path=str(terrain), map_sha256=file_hash(terrain), cloud_path=str(cloud),
                cloud_sha256=file_hash(cloud), bounds=[0., 0., 6., 6.])
    base = dict(map_path=str(terrain), planner_id="bitstar_a3", variant_id="synthetic_rp_bit",
                planner_seed=771, start=[.8, 3.], goal=[5.2, 3.], planning_budget_s=60.,
                samples_per_batch=100, batch_after_first_solution=200, rewire_factor=1.1,
                sample_attempts=100, state_cache=True, scalar_cost=True, collision_index=True,
                sampler_query_cache=True, exact_flat_certificate=False, test_only_fixed_work=True,
                objective_type=OBJECTIVE, sem_exposure={"lambda": 30., "risk_scale": 40.},
                rp_bit=dict(refine=True, stop_batches=0, test_max_iterations=WORK),
                safe_length=dict(policy_id="a3t_safe_length_v1", field_path=str(exchange),
                    field_sha256=file_hash(exchange), cloud_sha256=file_hash(cloud),
                    builder_source_sha256="labeled_synthetic_exchange_from_independent_test_field",
                    risk_cap=40., slope_cap_deg=35., atlas_sampling=True))
    config = dict(methods=dict(synthetic_rp_bit=dict(kind="rp_bit")),
                  analysis=dict(sem_equivalent_lambda=30., checkpoints_s=[.5, 60.]))
    entry = dict(trial_id="synthetic", method="synthetic_rp_bit")
    env = os.environ.copy()
    env.update({name: "1" for name in THREAD_ENV})
    env["PATH"] = os.pathsep.join((str(ROOT/".deps/install/lib"), "C:/mingw64/bin", env["PATH"]))
    cases = []

    def execute(name, request, expected=0):
        out = directory/name
        out.mkdir()
        dump(out/"request.json", request)
        command = [str(worker), str(out/"request.json"), str(out)]
        completed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=150)
        (out/"stdout.log").write_text(completed.stdout, encoding="utf-8")
        (out/"stderr.log").write_text(completed.stderr, encoding="utf-8")
        dump(out/"command.json", dict(argv=command, exit_code=completed.returncode))
        assert completed.returncode == expected, (name, completed.returncode, completed.stderr)
        cases.append(dict(case=name, exit_code=completed.returncode))
        if expected:
            assert not (out/"worker.json").exists(), name
            return None
        record = json.loads((out/"worker.json").read_text())
        events = [json.loads(line) for line in (out/"progress.jsonl").read_text().splitlines()]
        assert record["planner_status"] == "EXACT_SOLUTION" and events, (name, record)
        # The project's strict independent validator accepts every published snapshot.
        checked = validate_trial(out, request, info, dict(status="COMPLETED", field_build_s=0.), oracle=oracle, field=field)
        dump(out/"validation.json", checked)
        assert checked["success"] and not checked["correctness_errors"], (name, checked["correctness_errors"])
        # The A4 scorer agrees and reads the same final path.
        dump(out/"execution.json", dict(status="COMPLETED", launched=True, field_build_s=0.))
        scored = score_trial(out, entry, info, config, oracle=oracle, field=field, full=True)
        assert scored["success"] and not scored["correctness_errors"], (name, scored["correctness_errors"])
        assert abs(scored["final_j_sem"]-record["objective_value"]) <= 1e-8*record["objective_value"], name
        assert scored["final_sharp_turn_count"] == record["final_sharp_turn_count"] == sharp_turns(_csv_path(out/"path_xy.csv")), name
        assert scored["checkpoints"]["60"]["snapshot_id"] == events[-1]["snapshot_id"], name
        assert record["effective_policy"]["planner_class"] == "ompl::geometric::BITstar"
        assert record["state_dimension"] == 2 and record["batch_transition_count"] == 1
        assert record["requested_rp_bit"] == request["rp_bit"] and record["effective_rp_bit"]["sem_reference_used"] is False
        counters = record["rp_counters"]
        assert counters["heuristic_violations"] == 0 and counters["fast_integral_calls"] > 0, (name, counters)
        assert record["fast_integral_max_rel_error"] <= 1e-8, name
        assert 0. < record["lower_bound_J"] <= record["objective_value"]*(1+1e-9), name
        costs = [event["objective_value"] for event in events]
        assert all(b <= a+1e-9 for a, b in zip(costs, costs[1:])), costs
        trace = [(out/"snapshots"/event["snapshot_id"]/"path_xy.csv").read_bytes() for event in events]
        return record, events, trace

    record, events, trace = execute("refine_on", base)
    assert record["iterations"] == WORK and record["termination_reason"] == "WORK_BUDGET", record["termination_reason"]
    assert all(event["event_origin"] == "rp_refinement" for event in events)
    assert record["rp_counters"]["potential_evaluations"] > 0 and record["rp_counters"]["sampler_cell_draws"] > 0
    assert record["rp_counters"]["refine_edge_evaluations"] > 0
    again = execute("refine_on_repeat", base)
    assert again[2] == trace and again[0]["iterations"] == WORK, "fixed-work RP-BIT* run is not reproducible"

    raw_request = copy.deepcopy(base); raw_request["rp_bit"]["refine"] = False
    raw = execute("refine_off", raw_request)
    assert all(event["event_origin"] == "ompl_intermediate_solution_callback" for event in raw[1])
    assert raw[0]["rp_counters"]["refine_edge_evaluations"] == 0
    cap = record["effective_rp_bit"]["refinement"]["round_cost_ratio"]
    assert record["objective_value"] <= raw[0]["objective_value"]*(1+cap)+1e-9, "refinement exceeded its declared cost cap"

    plain_request = copy.deepcopy(base); plain_request["rp_bit"]["potential"] = False
    plain = execute("potential_off", plain_request)
    assert plain[0]["rp_counters"]["potential_evaluations"] == 0 and plain[0]["rp_counters"]["sampler_cell_draws"] == 0
    assert plain[0]["effective_policy"]["cost_to_go_lower_bound"] == "d_phi"

    stop_request = copy.deepcopy(base)
    del stop_request["test_only_fixed_work"]
    stop_request["rp_bit"] = dict(refine=True, stop_batches=3, stop_relative_eps=1e-3)
    stopped = execute("convergence_stop", stop_request)
    assert stopped[0]["termination_reason"] == "CONVERGED_NO_IMPROVEMENT", stopped[0]["termination_reason"]
    assert stopped[0]["actual_solve_wall_s"] < 30. and stopped[0]["overshoot_s"] == 0.

    invalid = []
    request = copy.deepcopy(base); del request["rp_bit"]["refine"]; invalid.append(("missing_refine", request))
    request = copy.deepcopy(base); request["rp_bit"]["unknown"] = 1; invalid.append(("unknown_parameter", request))
    request = copy.deepcopy(base); del request["test_only_fixed_work"]; invalid.append(("unlabeled_work_limit", request))
    request = copy.deepcopy(base); request["work_iterations"] = 800; invalid.append(("legacy_work_limit", request))
    request = copy.deepcopy(base); request["objective_type"] = "surface_length"; invalid.append(("objective_mismatch", request))
    request = copy.deepcopy(base); request["variant_id"] = "bitstar_a3"; invalid.append(("no_variant", request))
    request = copy.deepcopy(base); request["samples_per_batch"] = 50; invalid.append(("batch_drift", request))
    request = copy.deepcopy(base); request["sem_exposure"]["risk_scale"] = 20.; invalid.append(("wrong_scale", request))
    request = copy.deepcopy(base); request["safe_length"]["risk_cap"] = 60.; invalid.append(("wrong_cap", request))
    request = copy.deepcopy(base); request["safe_length"]["corridor_policy"] = {}; invalid.append(("corridor", request))
    request = copy.deepcopy(base); del request["safe_length"]; invalid.append(("missing_safety", request))
    request = copy.deepcopy(base); request["pri_pathsearch"] = {}; invalid.append(("mixed_variant", request))
    for name, request in invalid:
        execute(name, request, expected=2)
    dump(directory/"summary.json", dict(status="PASS", test_fixture=True, cases=cases, fixed_work_iterations=WORK))
    print(json.dumps(dict(status="PASS", cases=len(cases), real_trials=5, independent_all_snapshot_validation=True,
                          refined_J=record["objective_value"], raw_J=raw[0]["objective_value"],
                          potential_off_J=plain[0]["objective_value"], lower_bound_J=record["lower_bound_J"],
                          refined_sharp_turns=record["final_sharp_turn_count"], raw_sharp_turns=raw[0]["final_sharp_turn_count"],
                          convergence_stop_s=stopped[0]["actual_solve_wall_s"])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True, type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="synthetic_rp_bit_worker_", dir=ROOT/"tests/fixtures") as folder:
        run(args.worker.resolve(), Path(folder))
