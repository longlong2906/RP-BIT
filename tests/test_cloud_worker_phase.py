"""Real worker smoke for the optional Cloud phase protocol, outside results."""
import argparse
import datetime as dt
import json
from pathlib import Path
import tempfile
from test_pipeline import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path, required=True)
    args = parser.parse_args()
    fixture_root = args.worker.resolve().parent / "cloud_phase_smoke"
    fixture_root.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix="run_", dir=fixture_root))
    planners = ["bitstar_a3", "abitstar", "aitstar", "lazyprmstar_hauser2015", "sem_rrtstar_common_v1"]
    for planner in planners:
        output = folder / planner
        record = run(args.worker, output, "cloud_phase_smoke", planner,
                     work=250, budget=5., cloud_handshake=True)
        assert record["planner_status"] == "EXACT_SOLUTION", record
        assert record["termination_reason"] == "WORK_BUDGET", record
        phase = json.loads((output / "phase.json").read_text(encoding="utf-8"))
        assert phase["phase"] == "FINISHED", phase
        dt.datetime.fromisoformat(phase["updated_utc"].replace("Z", "+00:00"))
        assert not (output / "phase.json.tmp").exists()
    output = folder / "invalid_goal"
    run(args.worker, output, "cloud_invalid_goal", "bitstar_a3", goal=(-1., 5.), cloud_handshake=True)
    assert json.loads((output / "phase.json").read_text())["phase"] == "FINISHED"
    print("PASS: five real planners and invalid-query phase completion; paths independently validated")


if __name__ == "__main__": main()
