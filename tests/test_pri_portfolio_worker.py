"""Real BIT* portfolio replay and independent geometry/risk checks on test fixtures."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "tests")]
from corridor_v4_execution import THREAD_ENV
from native_paper_oracle import NativePaperOracle
from native_paper_v3_sem_evaluator import SharedSEMField
from sem_exposure_validation import validate_geometry
from pri_portfolio_validation import validate_trial
from safe_length_field import file_hash, build_field_file
from native_point_cloud import generate_cloud
from terrain_io import write_map
from test_safe_length_oracle import write_exchange


def run(worker, directory):
    terrain = directory / "synthetic_portfolio.t25d"
    height = lambda x, y: .45 * np.exp(-((x - 3.)**2 + (y - 3.)**2) / 1.8)
    write_map(terrain, [[float(height(i * .2, j * .2)) for i in range(31)] for j in range(31)],
              spacing=(.2, .2), rectangles=[],
              metadata=dict(test_fixture=True, purpose="synthetic_pri_portfolio_real_BITstar_replay"))
    x, y = np.meshgrid(np.arange(.025, 6., .05), np.arange(.025, 6., .05))
    cloud = np.column_stack((x.ravel(), y.ravel(), height(x, y).ravel()))
    cloud_file = directory / "synthetic_cloud.npy"
    np.save(cloud_file, cloud, allow_pickle=False)
    info = dict(map_path=str(terrain), map_sha256=file_hash(terrain), cloud_path=str(cloud_file),
                cloud_sha256=file_hash(cloud_file), bounds=[0.,0.,6.,6.])
    field = SharedSEMField(cloud, (0., 0., 6., 6.))
    exchange = directory / "synthetic_field.txt"
    build_field_file(cloud_file, terrain, (0., 0., 6., 6.), exchange)
    base = dict(map_path=str(terrain), planner_id="bitstar_a3", variant_id="synthetic_pri_portfolio",
        planner_seed=771, start=[.8, 3.], goal=[5.2, 3.], planning_budget_s=120., work_iterations=1200,
        test_only_fixed_work=True, samples_per_batch=100, batch_after_first_solution=200,
        rewire_factor=1.1, sample_attempts=100, progress_interval_s=.005, state_cache=True,
        scalar_cost=True, collision_index=True, sampler_query_cache=True, exact_flat_certificate=False,
        objective_type="surface_length", pri_portfolio=dict(length_cap_ratio=1.),
        safe_length=dict(policy_id="a3t_safe_length_v1", field_path=str(exchange), field_sha256=file_hash(exchange), risk_cap=40.,
                         slope_cap_deg=35., atlas_sampling=True))
    env = os.environ.copy()
    env.update({name: "1" for name in THREAD_ENV})
    env["PATH"] = os.pathsep.join((str(ROOT / ".deps/install/lib"), "C:/mingw64/bin", env["PATH"]))

    def execute(name, request, expected=0):
        out = directory / name
        out.mkdir()
        (out / "request.json").write_text(json.dumps(request), encoding="utf-8")
        command = [str(worker), str(out / "request.json"), str(out)]
        completed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=150)
        (out / "stdout.log").write_text(completed.stdout, encoding="utf-8")
        (out / "stderr.log").write_text(completed.stderr, encoding="utf-8")
        (out / "command.json").write_text(json.dumps(dict(argv=command, exit_code=completed.returncode)), encoding="utf-8")
        assert completed.returncode == expected, (name, completed.returncode, completed.stderr)
        if expected:
            return None
        record = json.loads((out / "worker.json").read_text())
        return out, record

    requests = []
    for alpha in (1., 1.025, 1.05, 1.1):
        request = copy.deepcopy(base)
        request["pri_portfolio"]["length_cap_ratio"] = alpha
        requests.append(request)
    serial = [execute(f"serial_{i}", request) for i, request in enumerate(requests)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        parallel = list(pool.map(lambda item: execute(f"parallel_{item[0]}", item[1]), enumerate(requests)))
    validated = 0
    oracle = NativePaperOracle(terrain)
    for index, ((left, record), (right, replay)) in enumerate(zip(serial, parallel)):
        assert record["planner_status"] == replay["planner_status"] == "EXACT_SOLUTION"
        assert record["anchor_snapshot_id"] == replay["anchor_snapshot_id"]
        assert record["selected_snapshot_id"] == replay["selected_snapshot_id"]
        assert (left / "path_xy.csv").read_bytes() == (right / "path_xy.csv").read_bytes(), "schedule changed fixed-work final path"
        assert record["counters"] == replay["counters"], "schedule changed fixed-work counters"
        for key in ("motion_cost_calls", "motion_heuristic_calls", "cost_to_go_calls", "sampler_allocations"):
            assert record["counters"][key] > 0, key
        assert record["effective_policy"]["motion_objective"] == "surface_length"
        assert not record["effective_policy"]["external_length_cap_in_sampler"]
        candidates = [json.loads(line) for line in (left / "portfolio_candidates.jsonl").read_text().splitlines()]
        repeats = [json.loads(line) for line in (right / "portfolio_candidates.jsonl").read_text().splitlines()]
        assert len(candidates) == len(repeats)
        for event, repeat in zip(candidates, repeats):
            for key in ("snapshot_id", "episode_id", "callback_id", "anchor_phase", "eligible", "published", "reason", "path_xy", "pri", "risk_sum", "risk_count", "cost_terrain_m"):
                assert event[key] == repeat[key], (index, event["snapshot_id"], key)
            if event["eligible"]:
                assert event["cost_terrain_m"] <= record["length_cap_m"]
                assert event["score_completed_s"] <= base["planning_budget_s"]
            geometry = validate_geometry(oracle, event["path_xy"], base["start"], base["goal"], event["cost_terrain_m"])
            sem = field.score_path(event["path_xy"])
            assert geometry["valid"] and sem["valid"] and sem["risk_max"] <= 40., (event["snapshot_id"], geometry, sem)
            cells = field.path_cells(event["path_xy"])
            pri = math.fsum(float(field.risk[j, i]) for i, j in cells) / len(cells)
            assert math.isclose(pri, event["pri"], abs_tol=1e-10, rel_tol=1e-10)
            validated += 1
        masks = json.loads((left / "portfolio_masks.json").read_text())
        for mask in masks["masks"]:
            if mask["mask_file"]:
                data = (left / mask["mask_file"]).read_bytes()
                assert len(data) == masks["nx"] * masks["ny"]
                assert hashlib.sha256(data).hexdigest() == mask["mask_sha256"]
                assert [i for i, value in enumerate(data) if value == 0] == mask["excluded_cells"]
        fresh = [episode for episode in record["episodes"] if episode["fresh_planner"]]
        assert len({episode["planner_instance_id"] for episode in fresh}) == len(fresh)
        assert all(not episode["anchor_injected"] for episode in fresh)
        verdict = validate_trial(left, requests[index], info, dict(status="COMPLETED", field_build_s=0.), oracle=oracle, field=field)
        assert verdict["success"] and not verdict["correctness_errors"], verdict["correctness_errors"]
        (left / "independent_full_verdict.json").write_text(json.dumps(verdict), encoding="utf-8")

    # Real worker + separately reconstructed raw-cloud oracle on numerically
    # constant fields. The planner field is built by the production builder,
    # so this also exercises quantile/component boundary parity.
    for name, extent, height_function in (("flat",6.,lambda x,y:0.), ("tilted",6.,lambda x,y:.12*x+.04*y),
                                          ("tilted20m",20.,lambda x,y:.24*x+.10*y+2.)):
        size=round(extent/.2)+1
        terrain_path = directory / ("synthetic_"+name+".t25d")
        write_map(terrain_path, [[height_function(i*.2,j*.2) for i in range(size)] for j in range(size)],
                  spacing=(.2,.2), metadata=dict(test_fixture=True, purpose="synthetic_pri_raw_field_parity"))
        raw_cloud = directory / ("synthetic_"+name+"_cloud.npy")
        np.save(raw_cloud, generate_cloud(terrain_path), allow_pickle=False)
        exchange_path = directory / ("synthetic_"+name+"_field.txt")
        metadata = build_field_file(raw_cloud, terrain_path, [0.,0.,extent,extent], exchange_path)
        request = copy.deepcopy(base)
        request.update(map_path=str(terrain_path), planning_budget_s=10., work_iterations=300)
        if extent==20.:
            request.update(start=[2.,10.],goal=[18.,10.])
        request["safe_length"].update(field_path=str(exchange_path), field_sha256=metadata["field_sha256"])
        out, _ = execute("raw_field_"+name, request)
        raw_info = dict(map_path=str(terrain_path), map_sha256=file_hash(terrain_path), cloud_path=str(raw_cloud),
                        cloud_sha256=file_hash(raw_cloud), bounds=[0.,0.,extent,extent])
        verdict = validate_trial(out, request, raw_info, dict(status="COMPLETED", field_build_s=0.))
        assert verdict["success"] and not verdict["correctness_errors"], (name,verdict["correctness_errors"])
        (out / "independent_full_verdict.json").write_text(json.dumps(verdict), encoding="utf-8")

    invalid = []
    for alpha in (.99, 1.11):
        value = copy.deepcopy(base); value["pri_portfolio"]["length_cap_ratio"] = alpha
        invalid.append(value)
    value = copy.deepcopy(base); value["pri_portfolio"]["unknown"] = 1; invalid.append(value)
    value = copy.deepcopy(base); value["safe_length"]["risk_cap"] = 39.; invalid.append(value)
    value = copy.deepcopy(base); value["sem_exposure"] = {"lambda": 1.}; invalid.append(value)
    value = copy.deepcopy(base); value.pop("test_only_fixed_work"); invalid.append(value)
    for i, request in enumerate(invalid):
        execute(f"rejected_{i}", request, expected=2)
    expired = copy.deepcopy(base); expired.pop("work_iterations"); expired["planning_budget_s"] = 1e-12
    _, record = execute("expired_initial_setup", expired)
    assert not record["path_available"] and record["selected_snapshot_id"] is None
    print(json.dumps(dict(status="PASS", real_fixed_work_trials=11, independently_validated_snapshots=validated,
                          serial_parallel_paths_and_counters_identical=True, rejected_requests=len(invalid),
                          full_validator_serial_trials=4, raw_field_flat_tilted_parity_trials=3,
                          test_fixture=True)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True, type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="synthetic_pri_portfolio_worker_", dir=ROOT / "tests/fixtures") as folder:
        run(args.worker.resolve(), Path(folder))
