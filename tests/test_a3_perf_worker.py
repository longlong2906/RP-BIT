"""Actual BIT* fixed-work equivalence and certificate/PTC integration."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from a3_traversability_protocol import DEFAULT_TRAVERSABILITY, OBJECTIVE_TYPE
from experiment_common import dump, file_hash, worker_env
from independent_verifier import Oracle
from terrain_io import write_map
from validate_results import validate_trial


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path, required=True)
    args = parser.parse_args()
    worker = args.worker.resolve()
    parent = worker.parent / "a3_perf_worker_regression"
    parent.mkdir(exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="run_", dir=parent))
    tilted, flat = root / "tilted.t25d", root / "flat.t25d"
    write_map(tilted, [[.12*i+.08*j for i in range(11)] for j in range(11)],
              radius=.3, margin=.1, slope=35., rectangles=[(4.2, 3., 5.8, 7.)],
              metadata={"test_fixture": True})
    write_map(flat, [[0.]*11 for _ in range(11)], radius=.3, margin=.1, slope=35.,
              metadata={"test_fixture": True})
    metrics = ("objective_value", "cost_terrain_m", "length_xy_m", "risk_slope_integral_m",
               "risk_clearance_integral_m", "risk_integral_m")
    evidence = []

    def run(name, map_path=tilted, lam=1., expected_exit=0, **options):
        out = root / name
        out.mkdir()
        request = dict(map_path=str(map_path), planner_id="bitstar_a3", variant_id=name,
                       planner_seed=771, start=[1., 5.], goal=[9., 5.], planning_budget_s=15.,
                       work_iterations=300, samples_per_batch=50, rewire_factor=1.1,
                       sample_attempts=100, progress_interval_s=.05, state_cache=True,
                       scalar_cost=True, collision_index=True, objective_type=OBJECTIVE_TYPE,
                       traversability=dict(DEFAULT_TRAVERSABILITY, **{"lambda": lam}),
                       sampler_query_cache=False, traversability_build_policy="legacy",
                       exact_flat_certificate=False)
        request.update(options)
        dump(out / "request.json", request)
        command = [str(worker), str(out / "request.json"), str(out)]
        process = subprocess.run(command, capture_output=True, text=True, env=worker_env(), timeout=35)
        (out / "stdout.log").write_text(process.stdout, encoding="utf-8")
        (out / "stderr.log").write_text(process.stderr, encoding="utf-8")
        dump(out / "command.json", dict(argv=command, exit_code=process.returncode))
        assert process.returncode == expected_exit, (name, process.stderr)
        evidence.append(dict(case=name, exit_code=process.returncode))
        if expected_exit:
            return None
        record = json.loads((out / "worker.json").read_text())
        assert record["planner_status"] == "EXACT_SOLUTION", record
        for key in ("sampler_query_cache", "traversability_build_policy", "exact_flat_certificate"):
            assert record["effective_policy"][key] == request[key], (name, key)
        dump(out / "trial.json", dict(record, map_hash=file_hash(map_path)))
        checked = validate_trial(out, Oracle(map_path))
        assert checked["success"] and checked["independent_valid"] and checked["snapshots_valid"], (name, checked)
        events = [json.loads(line) for line in (out / "progress.jsonl").read_text().splitlines()]
        improved = [e for e in events if e["event_type"] == "improvement"]
        trace = [(tuple(e[k] for k in metrics),
                  (out / "snapshots" / e["snapshot_id"] / "path_xy.csv").read_bytes()) for e in improved]
        return record, trace, (out / "path_xy.csv").read_bytes()

    baseline = run("baseline")
    for name, options in (("sampler", dict(sampler_query_cache=True)),
                          ("preprocessing", dict(traversability_build_policy="skip_dominated")),
                          ("both", dict(sampler_query_cache=True, traversability_build_policy="skip_dominated"))):
        result = run(name, **options)
        assert result[1:] == baseline[1:], (name, "fixed-work incumbent trace changed")
        assert tuple(result[0][k] for k in metrics) == tuple(baseline[0][k] for k in metrics), name
        assert result[0]["iterations"] == baseline[0]["iterations"] == 300
        for key in ("sample_proposals", "samples_returned_before_validity", "rejected_bounds", "rejected_nodata",
                    "rejected_heuristic", "motion_cost_calls", "motion_heuristic_calls"):
            assert result[0]["counters"][key] == baseline[0]["counters"][key], (name, key)
        assert result[0]["termination_reason"] == "WORK_BUDGET", name
    negative = run("nonflat_certificate", exact_flat_certificate=True)
    assert not negative[0]["exact_certificate"]["certified"]
    assert negative[1:] == baseline[1:]
    for lam in (0., 1., 4.):
        result = run(f"flat_certificate_{lam}", flat, lam, exact_flat_certificate=True)
        record = result[0]
        assert record["termination_reason"] == "EXACT_FLAT_DIRECT_CERTIFICATE", record
        certificate = record["exact_certificate"]
        assert certificate["certified"] and certificate["triggered_termination"]
        assert certificate["snapshot_id"] == "snapshot_0" and certificate["checks"] == 1
        assert len(result[1]) == 1 and record["counters"]["sampler_interval_calls"] == 0, record
        assert record["iterations"] < 300
    run("bad_build_policy", expected_exit=2, traversability_build_policy="square_max")
    dump(root / "summary.json", dict(passed=True, test_fixture=True, cases=evidence,
                                     work_iterations=300, bitwise_incumbent_trace=True))
    print("PASS A3-T perf worker fixed-work trace, certificate PTC and independent verification:", root)


if __name__ == "__main__":
    main()
