"""Real A5 worker regression on explicitly labeled synthetic test inputs."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tools"))
sys.path.insert(0, str(ROOT/"tests"))

import rp_bit_a5_field
from corridor_v4_execution import THREAD_ENV
from native_paper_oracle import NativePaperOracle
from native_paper_v3_sem_evaluator import SharedSEMField
from rp_bit_validation import a5_next_batch, validate_trial as score_trial
from run_safe_length import dump
from safe_length_field import file_hash
from sem_exposure_validation import OBJECTIVE, validate_trial
from terrain_io import write_map
from test_safe_length_oracle import write_exchange

WORK = 2500
OFF = dict(refine=True, window_samples=0, batch_min=200, batch_growth=False, fast_validity=False, lazy_refine=False, seed=False)
FULL = dict(refine=True, window_samples=4000, batch_min=500, batch_growth=True, fast_validity=True, lazy_refine=True, seed=True)


def run(worker, directory):
    terrain = directory/"synthetic_plane_obstacle.t25d"
    write_map(terrain, [[.1*(i*.2)+.05*(j*.2) for i in range(31)] for j in range(31)],
              spacing=(.2, .2), rectangles=[(2.5, 2., 3.5, 4.)],
              metadata=dict(test_fixture=True, purpose="synthetic_rp_bit_a5_real_worker_regression"))
    x, y = np.meshgrid(np.arange(.025, 6., .05), np.arange(.025, 6., .05))
    cloud = directory/"synthetic_cloud.npy"
    np.save(cloud, np.column_stack((x.ravel(), y.ravel(), (.1*x+.05*y).ravel())), allow_pickle=False)
    field = SharedSEMField(np.load(cloud, allow_pickle=False), (0., 0., 6., 6.))
    oracle = NativePaperOracle(terrain)
    exchange = directory/"synthetic_field.txt"
    write_exchange(exchange, field.valid, field.risk)
    info = dict(map_path=str(terrain), map_sha256=file_hash(terrain), cloud_path=str(cloud),
                cloud_sha256=file_hash(cloud), bounds=[0., 0., 6., 6.])
    # Across the obstacle (which the SEM field does not know) and along its side (where the lattice seed is safe).
    across, beside = dict(start=[.8, 3.], goal=[5.2, 3.]), dict(start=[.8, .8], goal=[5.2, 1.2])

    def request_for(options, query=across, fixed=True):
        request = dict(map_path=str(terrain), planner_id="bitstar_a3", variant_id="synthetic_rp_bit_a5", planner_seed=771,
                       planning_budget_s=60., samples_per_batch=100, batch_after_first_solution=options["batch_min"],
                       rewire_factor=1.1, sample_attempts=100, state_cache=True, scalar_cost=True, collision_index=True,
                       sampler_query_cache=True, exact_flat_certificate=False, objective_type=OBJECTIVE,
                       sem_exposure={"lambda": 30., "risk_scale": 40.}, rp_bit_a5=dict(options), **query,
                       safe_length=dict(policy_id="a3t_safe_length_v1", field_path=str(exchange),
                           field_sha256=file_hash(exchange), cloud_sha256=file_hash(cloud),
                           builder_source_sha256="labeled_synthetic_exchange_from_independent_test_field",
                           risk_cap=40., slope_cap_deg=35., atlas_sampling=True))
        if fixed:
            request["test_only_fixed_work"] = True
            request["rp_bit_a5"]["test_max_iterations"] = WORK
        return request

    config = dict(methods=dict(synthetic_rp_bit_a5=dict(kind="rp_bit_a5"), synthetic_rp_bit=dict(kind="rp_bit")),
                  analysis=dict(sem_equivalent_lambda=30., checkpoints_s=[.5, 60.]))
    env = os.environ.copy()
    env.update({name: "1" for name in THREAD_ENV})
    env["PATH"] = os.pathsep.join((str(ROOT/".deps/install/lib"), "C:/mingw64/bin", env["PATH"]))
    cases = []

    def score(out, request, method="synthetic_rp_bit_a5"):
        return score_trial(out, dict(trial_id="synthetic", method=method), info, config, oracle=oracle, field=field, full=True)

    def execute(name, request, expected=0, method="synthetic_rp_bit_a5"):
        out = directory/name
        out.mkdir()
        dump(out/"request.json", request)
        command = [str(worker), str(out/"request.json"), str(out)]
        completed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=200)
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
        # The project's strict independent validator accepts every published snapshot, unchanged.
        checked = validate_trial(out, request, info, dict(status="COMPLETED", field_build_s=0.), oracle=oracle, field=field)
        dump(out/"validation.json", checked)
        assert checked["success"] and not checked["correctness_errors"], (name, checked["correctness_errors"])
        dump(out/"execution.json", dict(status="COMPLETED", launched=True, field_build_s=0.))
        scored = score(out, request, method)
        assert scored["success"] and not scored["correctness_errors"], (name, scored["correctness_errors"])
        assert abs(scored["final_j_sem"]-record["objective_value"]) <= 1e-8*record["objective_value"], name
        assert scored["events_j_sem"] and len(scored["events_j_sem"]) == len(events), name
        assert record["effective_policy"]["planner_class"] == "ompl::geometric::BITstar"
        assert record["state_dimension"] == 2 and record["batch_transition_count"] == 1, name
        assert record["rp_counters"]["heuristic_violations"] == 0 and record["fast_integral_max_rel_error"] <= 1e-8, name
        costs = [event["objective_value"] for event in events]
        assert all(b <= a+1e-9 for a, b in zip(costs, costs[1:])), costs
        trace = [(out/"snapshots"/event["snapshot_id"]/"path_xy.csv").read_bytes() for event in events]
        return record, events, trace, out

    # 1. With every switch off, the A5 worker publishes exactly what the A4 worker publishes.
    off = execute("switches_off", request_for(OFF))
    a4 = copy.deepcopy(request_for(OFF))
    del a4["rp_bit_a5"]
    a4.update(variant_id="synthetic_rp_bit", rp_bit=dict(refine=True, stop_batches=0, test_max_iterations=WORK))
    reference = execute("a4_reference", a4, method="synthetic_rp_bit")
    assert off[0]["backend_id"] == "a5_rpbitstar_v1" and reference[0]["backend_id"] == "a4_rpbitstar_v1"
    assert off[2] == reference[2] and off[0]["iterations"] == reference[0]["iterations"] == WORK, "A5 with every switch off differs from A4"
    assert off[0]["requested_rp_bit_a5"] == request_for(OFF)["rp_bit_a5"] and off[0]["seed_status"] == "disabled"
    assert all(value == 0 for value in off[0]["a5_counters"].values()), off[0]["a5_counters"]

    # 2. Every switch on: reproducible at fixed work, in the path bytes and in the search itself.
    fixed = dict(FULL, window_samples=0)
    full = execute("switches_on", request_for(fixed, beside))
    again = execute("switches_on_repeat", request_for(fixed, beside))
    assert full[2] == again[2] and full[0]["iterations"] == again[0]["iterations"], "fixed-work A5 run is not reproducible"
    assert [b["iterations"] for b in full[0]["batch_log"]] == [b["iterations"] for b in again[0]["batch_log"]]
    assert full[0]["termination_reason"] == "WORK_BUDGET" and full[0]["seed_status"] == "published", full[0]["seed_status"]
    assert full[1][0]["event_origin"] == "a5_lattice_seed_refined" and full[1][0]["batch"] == 0 and full[1][0]["bitstar_cost"] is None
    assert all(event["event_origin"] != "a5_lattice_seed_refined" for event in full[1][1:])
    counters = full[0]["a5_counters"]
    assert counters["slope_certified"] > 0 and counters["seed_expansions"] > 0, counters
    # Here the best path is the straight edge BIT* tries first, so the published bound never binds before BIT*'s own.
    assert counters["bounded_samples"] == 0 and full[0]["effective_policy"]["informed_sampling"].endswith("capped_by_published_J")
    # The same field in the binary container: the worker reads the same cells, so the run is the same run.
    container = directory/rp_bit_a5_field.FILE_NAME
    container.write_bytes(rp_bit_a5_field.encode((0., 0., 6., 6.), .1, field.valid, field.risk))

    def in_container(request, path=container):
        request = copy.deepcopy(request)
        request["safe_length"].update(field_path=str(path), field_sha256=file_hash(path))
        return request
    packed = execute("switches_on_binary_field", in_container(request_for(fixed, beside)))
    assert packed[2] == full[2] and packed[0]["iterations"] == full[0]["iterations"], "the binary field container changed the run"
    assert [b["iterations"] for b in packed[0]["batch_log"]] == [b["iterations"] for b in full[0]["batch_log"]]
    # Around the obstacle (no seed possible): lazy refinement, the window stop and a refined final path.
    around = execute("lazy_around_obstacle", request_for(dict(FULL, window_samples=1000, seed=False), across, fixed=False))
    assert around[0]["termination_reason"] == "CONVERGED_NO_IMPROVEMENT" and around[0]["a5_counters"]["bounded_samples"] == 0
    assert around[1][-1]["event_origin"] == "rp_refinement", around[1][-1]["event_origin"]

    # 3. The same switches with the sample window: the run stops by itself on the registered rule.
    stopped = execute("window_stop", request_for(dict(FULL, window_samples=1000), beside, fixed=False))
    log = stopped[0]["batch_log"]
    assert stopped[0]["termination_reason"] == "CONVERGED_NO_IMPROVEMENT" and stopped[0]["overshoot_s"] == 0., stopped[0]["termination_reason"]
    assert log[-1]["stale_samples"] >= 1000 > log[-2]["stale_samples"], log[-2:]
    assert stopped[0]["effective_rp_bit_a5"]["stop_relative_eps"] == 1e-3 and stopped[0]["actual_solve_wall_s"] < 30.

    # 4. Each switch changes what the worker does.
    grown = execute("growth_on", request_for(dict(OFF, window_samples=4000, batch_min=500, batch_growth=True), fixed=False))
    flat = execute("growth_off", request_for(dict(OFF, window_samples=4000, batch_min=500), fixed=False))
    sizes = lambda record: [b["random_samples"] for b in record["batch_log"] if b["published_J"] is not None]
    assert set(sizes(flat[0])) == {500} and {1000, 2000} <= set(sizes(grown[0])), (sizes(flat[0]), sizes(grown[0]))
    assert grown[0]["batches"] < flat[0]["batches"], "growing batches did not reduce the number of batches"
    exact = execute("shortcut_off", request_for(dict(fixed, fast_validity=False), beside))
    assert exact[2] == full[2] and exact[0]["iterations"] == full[0]["iterations"], "the slope shortcut changed the run"
    assert exact[0]["a5_counters"]["slope_certified"] == 0 == exact[0]["a5_counters"]["slope_exact"]
    unseeded = execute("seed_off", request_for(dict(fixed, seed=False), beside))
    assert unseeded[0]["seed_status"] == "disabled" and unseeded[1][0]["event_origin"] == "rp_refinement"
    assert unseeded[0]["a5_counters"]["seed_expansions"] == 0 == unseeded[0]["a5_counters"]["bounded_samples"]
    assert unseeded[0]["effective_policy"]["informed_sampling"].endswith("exact_rejection")
    # Lazy refinement off: every batch boundary with a new incumbent refines, so at least as many refinements.
    lazy = execute("lazy_on", request_for(dict(OFF, window_samples=4000, batch_min=500, batch_growth=True, lazy_refine=True), fixed=False))
    assert grown[0]["a5_counters"]["refine_deferred"] == 0 < lazy[0]["a5_counters"]["refine_deferred"], lazy[0]["a5_counters"]
    assert lazy[0]["refinement_count"] < grown[0]["refinement_count"], (lazy[0]["refinement_count"], grown[0]["refinement_count"])
    assert lazy[0]["objective_value"] <= grown[0]["objective_value"]*1.001, "lazy refinement lost quality on the fixture"
    # The SEM field does not know the obstacle, so the lattice path crosses it: no seed, A4-style start.
    blocked = execute("seed_unsafe", request_for(fixed, across))
    assert blocked[0]["seed_status"] == "seed_unsafe" and blocked[1][0]["event_origin"] == "rp_refinement", blocked[0]["seed_status"]

    # 5. The scorer catches a worker record that contradicts its own log or the registered options.
    def tampered(name, source, change, expected):
        out = directory/name
        shutil.copytree(source[3], out)
        record = json.loads((out/"worker.json").read_text())
        events = [json.loads(line) for line in (out/"progress.jsonl").read_text().splitlines()]
        change(record, events)
        dump(out/"worker.json", record)
        (out/"progress.jsonl").write_text("".join(json.dumps(event)+"\n" for event in events), encoding="utf-8")
        errors = score(out, json.loads((out/"request.json").read_text()))["correctness_errors"]
        assert expected in errors, (name, errors)
        cases.append(dict(case=name, detected=expected))

    def shorter(record, events): record["batch_log"][-1]["stale_samples"] -= 1
    def earlier(record, events): record["batch_log"][-2]["stale_samples"] = 1000
    def resized(record, events): record["batch_log"][-1]["random_samples"] += 1
    def relabelled(record, events): record["termination_reason"] = "TIME_BUDGET"
    def unseeded_claim(record, events): record["seed_status"] = "disabled"
    def seed_label(record, events): events[0]["event_origin"] = "a5_lattice_seed_refined"
    def other_options(record, events): record["effective_rp_bit_a5"]["lazy_refine"] = False
    def other_request(record, events): record["requested_rp_bit_a5"]["window_samples"] = 4000
    def violation(record, events): record["rp_counters"]["heuristic_violations"] = 1
    for name, source, change, expected in (
            ("stale_count", stopped, shorter, "a5_convergence_inconsistent"),
            ("window_reached_earlier", stopped, earlier, "a5_convergence_inconsistent"),
            ("off_schedule_batch", stopped, resized, "a5_convergence_inconsistent"),
            ("stop_label", stopped, relabelled, "a5_convergence_inconsistent"),
            ("seed_status", stopped, unseeded_claim, "a5_seed_event_inconsistent"),
            ("seed_event_without_seed", around, seed_label, "a5_seed_event_inconsistent"),
            ("effective_options", stopped, other_options, "effective_rp_bit_a5_mismatch"),
            ("requested_options", stopped, other_request, "requested_rp_bit_a5_mismatch"),
            ("bound_violation", stopped, violation, "heuristic_bound_violation_on_incumbent")):
        tampered("tampered_"+name, source, change, expected)
    # The schedule restated in the scorer is the worker's.
    options = stopped[0]["requested_rp_bit_a5"]
    for before, after in zip(log, log[1:]):
        if before["published_J"] is not None:
            assert after["random_samples"] == a5_next_batch(before["stale_samples"], before["random_samples"], options)

    # 6. Requests outside the registered contract are rejected before any planning.
    base, invalid = request_for(FULL, beside), []
    for key in FULL:
        request = copy.deepcopy(base); del request["rp_bit_a5"][key]; invalid.append(("missing_"+key, request))
    request = copy.deepcopy(base); request["rp_bit_a5"]["unknown"] = 1; invalid.append(("unknown_parameter", request))
    request = copy.deepcopy(base); del request["test_only_fixed_work"]; invalid.append(("unlabeled_work_limit", request))
    request = copy.deepcopy(base); request["rp_bit"] = dict(refine=True); invalid.append(("mixed_with_a4", request))
    request = copy.deepcopy(base); request["batch_after_first_solution"] = 200; invalid.append(("batch_drift", request))
    request = copy.deepcopy(base); request["samples_per_batch"] = 50; invalid.append(("first_batch_drift", request))
    request = copy.deepcopy(base); request["rp_bit_a5"]["batch_min"] = 0; request["batch_after_first_solution"] = 0
    invalid.append(("empty_batch", request))
    request = copy.deepcopy(base); request["rp_bit_a5"]["refine"] = False; invalid.append(("seed_without_refine", request))
    request = copy.deepcopy(base); request["rp_bit_a5"].update(refine=False, seed=False); invalid.append(("lazy_without_refine", request))
    request = copy.deepcopy(base); request["objective_type"] = "surface_length"; invalid.append(("objective_mismatch", request))
    request = copy.deepcopy(base); request["variant_id"] = "bitstar_a3"; invalid.append(("no_variant", request))
    request = copy.deepcopy(base); request["sem_exposure"]["risk_scale"] = 20.; invalid.append(("wrong_scale", request))
    request = copy.deepcopy(base); request["safe_length"]["risk_cap"] = 60.; invalid.append(("wrong_cap", request))
    request = copy.deepcopy(base); del request["safe_length"]; invalid.append(("missing_safety", request))
    truncated = directory/"truncated_field.bin"
    truncated.write_bytes(container.read_bytes()[:-1])
    invalid.append(("truncated_binary_field", in_container(base, truncated)))
    for name, request in invalid:
        execute("invalid_"+name, request, expected=2)
    dump(directory/"summary.json", dict(status="PASS", test_fixture=True, cases=cases, fixed_work_iterations=WORK))
    print(json.dumps(dict(status="PASS", cases=len(cases), real_trials=sum(case.get("exit_code") == 0 for case in cases),
                          independent_all_snapshot_validation=True,
                          a4_J=reference[0]["objective_value"], full_J=stopped[0]["objective_value"],
                          full_stop_s=stopped[0]["actual_solve_wall_s"], full_batches=stopped[0]["batches"],
                          growth_batches=[flat[0]["batches"], grown[0]["batches"]],
                          slope_certified=counters["slope_certified"], slope_exact=counters["slope_exact"])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True, type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="synthetic_rp_bit_a5_worker_", dir=ROOT/"tests/fixtures") as folder:
        run(args.worker.resolve(), Path(folder))
