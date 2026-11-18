"""SEM + rpRefine replay worker on explicitly labeled synthetic test inputs.

A real fixed-work RP-BIT* run with refinement off stands in for a SEM-RRT* source: its raw
incumbents are written in the SEM worker layout and replayed. The fixture is the one of
test_rp_bit_worker.py (copied, that file is locked).
"""
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

from corridor_v4_execution import THREAD_ENV
from native_paper_oracle import NativePaperOracle
from native_paper_v3_sem_evaluator import SharedSEMField
import rp_bit_protocol as protocol
from rp_bit_validation import validate_trial as score_trial
from run_safe_length import _csv_path, dump
from safe_length_field import file_hash
from sem_exposure_validation import OBJECTIVE, validate_trial
from terrain_io import write_map
from test_safe_length_oracle import write_exchange

WORK = 2500
SEED = 771


def run(worker, directory):
    terrain = directory/"synthetic_plane_obstacle.t25d"
    write_map(terrain, [[.1*(i*.2)+.05*(j*.2) for i in range(31)] for j in range(31)],
              spacing=(.2, .2), rectangles=[(2.5, 2., 3.5, 4.)],
              metadata=dict(test_fixture=True, purpose="synthetic_rp_refine_replay_regression"))
    x, y = np.meshgrid(np.arange(.025, 6., .05), np.arange(.025, 6., .05))
    cloud = directory/"synthetic_cloud.npy"
    np.save(cloud, np.column_stack((x.ravel(), y.ravel(), (.1*x+.05*y).ravel())), allow_pickle=False)
    field = SharedSEMField(np.load(cloud, allow_pickle=False), (0., 0., 6., 6.))
    oracle = NativePaperOracle(terrain)
    exchange = directory/"synthetic_field.txt"
    write_exchange(exchange, field.valid, field.risk)
    info = dict(map_path=str(terrain), map_sha256=file_hash(terrain), cloud_path=str(cloud),
                cloud_sha256=file_hash(cloud), bounds=[0., 0., 6., 6.])
    safety = dict(policy_id="a3t_safe_length_v1", field_path=str(exchange), field_sha256=file_hash(exchange),
                  cloud_sha256=file_hash(cloud), builder_source_sha256="labeled_synthetic_exchange_from_independent_test_field",
                  risk_cap=40., slope_cap_deg=35., atlas_sampling=True)
    a4 = dict(map_path=str(terrain), planner_id="bitstar_a3", variant_id="synthetic_rp_bit",
              planner_seed=SEED, start=[.8, 3.], goal=[5.2, 3.], planning_budget_s=60.,
              samples_per_batch=100, batch_after_first_solution=200, rewire_factor=1.1,
              sample_attempts=100, state_cache=True, scalar_cost=True, collision_index=True,
              sampler_query_cache=True, exact_flat_certificate=False, test_only_fixed_work=True,
              objective_type=OBJECTIVE, sem_exposure={"lambda": 30., "risk_scale": 40.},
              rp_bit=dict(refine=True, stop_batches=0, test_max_iterations=WORK), safe_length=safety)
    env = os.environ.copy()
    env.update({name: "1" for name in THREAD_ENV})
    env["PATH"] = os.pathsep.join((str(ROOT/".deps/install/lib"), "C:/mingw64/bin", env["PATH"]))
    cases = []

    def launch(out, request, expected=0):
        out.mkdir()
        dump(out/"request.json", request)
        command = [str(worker), str(out/"request.json"), str(out)]
        completed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=150)
        (out/"stderr.log").write_text(completed.stderr, encoding="utf-8")
        assert completed.returncode == expected, (out.name, completed.returncode, completed.stderr)
        cases.append(dict(case=out.name, exit_code=completed.returncode))
        if expected:
            assert not (out/"worker.json").exists(), out.name
            return None
        return json.loads((out/"worker.json").read_text())

    def events(out):
        return [json.loads(line) for line in (out/"progress.jsonl").read_text().splitlines() if line.strip()]

    # Two real A4 runs on the same seed and work: raw incumbents (refine off) and refined ones.
    request = copy.deepcopy(a4); request["rp_bit"]["refine"] = False
    raw_record = launch(directory/"a4_refine_off", request)
    raw_events = events(directory/"a4_refine_off")
    reference_record = launch(directory/"a4_refine_on", a4)
    reference = events(directory/"a4_refine_on")[-1]
    assert reference["event_origin"] == "rp_refinement"
    raw = [dict(snapshot_id=e["snapshot_id"], elapsed_s=e["elapsed_s"], bitstar_cost=e["bitstar_cost"],
                path_xy=_csv_path(directory/"a4_refine_off"/"snapshots"/e["snapshot_id"]/"path_xy.csv")) for e in raw_events]
    assert len(raw) >= 3, len(raw)

    def source(slot, incumbents, solve_s, status="COMPLETED"):
        """A finished trial in the SEM worker layout (the fields the replay and its checks read)."""
        out = directory/f"{slot}__s{SEED}__sem_native"
        out.mkdir()
        dump(out/"execution.json", dict(status=status))
        dump(out/"worker.json", dict(test_fixture=True, planner_status="EXACT_SOLUTION",
            termination_reason="CONVERGED_NO_IMPROVEMENT", phase_timings=dict(solve_s=solve_s, total_before_solve_s=.25),
            snapshots=[dict(snapshot_id=f"incumbent_{i:06d}", elapsed_s=t, path_xy=p) for i, (t, p) in enumerate(incumbents)]))
        return out

    def replay(slot, refine_at, budget=60., expected=0, mutate=None, suffix=""):
        method = {"kind": "sem_refine", "source": "sem_native", "lambda": 30., "refine_at_s": refine_at}
        spec = protocol.replay_spec(directory/f"{slot}__s{SEED}__sem_native", method)
        request = dict(planner_id="rp_refine_replay", variant_id="sem_refine", planner_seed=SEED, objective_type=OBJECTIVE,
                       sem_exposure={"lambda": 30., "risk_scale": 40.}, map_path=str(terrain), start=[.8, 3.], goal=[5.2, 3.],
                       planning_budget_s=budget, collision_index=True, cloud_phase_handshake=True,
                       safe_length=copy.deepcopy(safety), rp_refine_replay=spec)
        if mutate:
            mutate(request)
        out = directory/f"{slot}__s{SEED}__sem_refine{suffix}"
        record = launch(out, request, expected)
        if expected:
            return None
        config = dict(methods=dict(sem_refine=method), analysis=dict(sem_equivalent_lambda=30., checkpoints_s=[.5, 60.]))
        execution = dict(status="COMPLETED", launched=True, field_build_s=0.)
        dump(out/"execution.json", execution)
        strict = validate_trial(out, request, info, execution, oracle=oracle, field=field)
        entry = dict(trial_id=out.name, method="sem_refine", map_id=slot, planner_seed=SEED)
        scored = score_trial(out, entry, info, config, oracle=oracle, field=field, full=True)
        assert not strict["correctness_errors"] and not scored["correctness_errors"], (slot, strict["correctness_errors"],
                                                                                        scored["correctness_errors"])
        return record, events(out), scored, out

    # 1. The replay refines with A4's own rpRefine: the A4 final path, refined from one raw incumbent,
    #    comes out byte-identical when the replay refines that same incumbent.
    match = [i for i, item in enumerate(raw)
             if abs(item["bitstar_cost"]-reference["bitstar_cost"]) <= 1e-12*max(1., reference["bitstar_cost"])]
    assert len(match) == 1, match
    k = match[0]
    assert raw[k]["elapsed_s"] > 0 and (k+1 == len(raw) or raw[k+1]["elapsed_s"] > raw[k]["elapsed_s"])
    source("identity", [(item["elapsed_s"], item["path_xy"]) for item in raw], raw_record["actual_solve_wall_s"])
    record, published, scored, out = replay("identity", [raw[k]["elapsed_s"]])
    assert scored["success"], scored
    refined = [e for e in published if e["event_origin"] == "rp_refinement" and e["source_snapshot_id"] == f"incumbent_{k:06d}"]
    assert len(refined) == 1, published
    assert ((out/"snapshots"/refined[0]["snapshot_id"]/"path_xy.csv").read_bytes()
            == (directory/"a4_refine_on"/"path_xy.csv").read_bytes()), "replay refinement differs from A4's"

    # 2. Schedule and timing: standing incumbent, once per incumbent, raw first on ties, stamps on SEM's clock.
    picked = [raw[0], raw[len(raw)//2], raw[-1]]
    source("schedule", list(zip((.1, .5, 3.), (item["path_xy"] for item in picked))), 4.)
    record, published, scored, out = replay("schedule", [.5, 1., 2., 5.])
    assert scored["success"], scored
    steps = record["replay_steps"]
    assert [(s["kind"], s["at_s"], s["source_snapshot_id"], s["skipped"]) for s in steps] == [
        ("raw", .1, "incumbent_000000", None), ("raw", .5, "incumbent_000001", None),
        ("refine", .5, "incumbent_000001", None), ("refine", 1., "incumbent_000001", "already_refined"),
        ("refine", 2., "incumbent_000001", "already_refined"), ("raw", 3., "incumbent_000002", None),
        ("refine", 4., "incumbent_000002", None)], steps
    offset = record["setup_s"]
    for step in steps:
        offset += step["duration_s"]
        if step["stamp_s"] is not None:
            assert abs(step["stamp_s"]-(step["at_s"]+offset)) <= 1e-9, step
    assert abs(record["actual_solve_wall_s"]-(4.+offset)) <= 1e-9 and record["refinement_count"] == 2
    assert all(b["elapsed_s"] >= a["elapsed_s"] and b["objective_value"] <= a["objective_value"]+1e-9
               for a, b in zip(published, published[1:]))
    assert published[-1]["event_origin"] == "rp_refinement" and scored["replay"]["refinements"] == 2
    assert scored["events_j_sem"] and record["preprocessing_s"] >= .25
    trace = [(out/"snapshots"/e["snapshot_id"]/"path_xy.csv").read_bytes() for e in published]
    overhead = record["replay_overhead_s"]

    # 2b. The independent checks catch a changed source, schedule, timing, registration or raw path.
    registered = dict(methods=dict(sem_refine={"kind": "sem_refine", "source": "sem_native", "lambda": 30.,
                                               "refine_at_s": [.5, 1., 2., 5.]}),
                      analysis=dict(sem_equivalent_lambda=30., checkpoints_s=[.5, 60.]))

    def tampered(name, edit):
        copy_dir = directory/f"schedule__s{SEED}__sem_refine_{name}"
        shutil.copytree(out, copy_dir)
        edit(copy_dir)
        entry = dict(trial_id=copy_dir.name, method="sem_refine", map_id="schedule", planner_seed=SEED)
        return score_trial(copy_dir, entry, info, registered, oracle=oracle, field=field)["correctness_errors"]

    def rewrite(path, change):
        value = json.loads(path.read_text(encoding="utf-8"))
        change(value)
        dump(path, value)

    raw_snapshot = next(e["snapshot_id"] for e in published if e["event_origin"] == "sem_incumbent")

    def nudge(d):
        csv = d/"snapshots"/raw_snapshot/"path_xy.csv"
        points = _csv_path(csv)
        points[1][0] += 1e-6
        csv.write_text("x_m,y_m\n"+"".join(f"{x!r},{y!r}\n" for x, y in points), encoding="utf-8")

    checks = dict(
        timing=(lambda d: rewrite(d/"worker.json", lambda w: w["replay_steps"][0].update(duration_s=.5)), "replay_timing_mismatch"),
        schedule=(lambda d: rewrite(d/"worker.json", lambda w: w["replay_steps"][3].update(skipped=None)), "replay_schedule_mismatch"),
        registration=(lambda d: rewrite(d/"request.json", lambda r: r["rp_refine_replay"].update(refine_at_s=[.5, 1.])),
                      "replay_spec_not_registered"),
        raw_path=(nudge, "replay_raw_path_differs_from_source"))
    for name, (edit, expected) in checks.items():
        assert expected in tampered(name, edit), name
    original = (directory/f"schedule__s{SEED}__sem_native"/"worker.json").read_bytes()
    (directory/f"schedule__s{SEED}__sem_native"/"worker.json").write_bytes(original+b" ")
    assert "replay_source_missing_or_changed" in tampered("source", lambda d: None)
    (directory/f"schedule__s{SEED}__sem_native"/"worker.json").write_bytes(original)

    # 3. Deterministic: a second replay of the same source publishes the same paths.
    source("repeat", list(zip((.1, .5, 3.), (item["path_xy"] for item in picked))), 4.)
    again = replay("repeat", [.5, 1., 2., 5.])
    assert [(again[3]/"snapshots"/e["snapshot_id"]/"path_xy.csv").read_bytes() for e in again[1]] == trace

    # 4. Budget: a candidate stamped after the budget is never published.
    source("budget", list(zip((.05, .5), (item["path_xy"] for item in picked[:2]))), 1.)
    record, published, scored, out = replay("budget", [.5], budget=.4)
    assert published and all(e["elapsed_s"] <= .4 for e in published), published
    assert not any(s["published"] for s in record["replay_steps"] if s["at_s"] >= .5)
    assert scored["replay"]["late"] >= 2, scored["replay"]

    # 5. An unsafe SEM incumbent (straight through the obstacle) is neither published nor refined.
    source("unsafe", [(.1, [[.8, 3.], [5.2, 3.]])], 2.)
    record, published, scored, out = replay("unsafe", [.5, 1.])
    assert record["planner_status"] == "NO_SOLUTION" and not published and record["raw_rejected_count"] == 1
    assert [s["skipped"] for s in record["replay_steps"] if s["kind"] == "refine"] == ["unsafe_source"]*3
    assert not scored["success"] and not scored["correctness_errors"]

    # 6. A source that did not complete cannot be replayed.
    source("crashed", [(.1, picked[0]["path_xy"])], 1., status="CRASH")
    try:
        replay("crashed", [.5])
    except ValueError as error:
        assert "did not complete" in str(error)
    else:
        raise AssertionError("an incomplete source was replayed")

    # 7. Invalid requests exit 2 without output.
    source("invalid", [(item["elapsed_s"], item["path_xy"]) for item in raw[:3]], raw[2]["elapsed_s"]+1.)
    edits = dict(
        legacy_key=lambda r: r.update(samples_per_batch=100),
        rp_bit_mixed=lambda r: r.update(rp_bit=dict(refine=True)),
        unknown_replay_field=lambda r: r["rp_refine_replay"].update(extra=1),
        time_order=lambda r: r["rp_refine_replay"]["incumbents"].reverse(),
        endpoint=lambda r: r["rp_refine_replay"]["incumbents"][0]["path_bits"][0].__setitem__(0, protocol.coordinate_bits(.7)),
        bits=lambda r: r["rp_refine_replay"]["incumbents"][0]["path_bits"][1].__setitem__(0, "zz"*8),
        refine_order=lambda r: r["rp_refine_replay"].update(refine_at_s=[1., .5]),
        wrong_cap=lambda r: r["safe_length"].update(risk_cap=60.),
        wrong_scale=lambda r: r["sem_exposure"].update(risk_scale=20.))
    for name, edit in edits.items():
        replay("invalid", [.5], expected=2, mutate=edit, suffix="_"+name)
    dump(directory/"summary.json", dict(status="PASS", test_fixture=True, cases=cases))
    print(json.dumps(dict(status="PASS", cases=len(cases), identity_with_a4_refinement=True,
                          a4_refined_J=reference_record["objective_value"], raw_J=raw_record["objective_value"],
                          schedule_overhead_s=overhead)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True, type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="synthetic_rp_refine_replay_", dir=ROOT/"tests/fixtures") as folder:
        run(args.worker.resolve(), Path(folder))
