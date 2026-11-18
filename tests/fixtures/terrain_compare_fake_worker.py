"""SYNTHETIC scheduler fixture. This is not a planner or experiment result."""
import datetime
import json
import os
from pathlib import Path
import sys
import time

request = json.loads(Path(sys.argv[1]).read_text())
output = Path(sys.argv[2])


def write(name, value):
    path = output / name
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value))
    # Match the existing C++ phase writer's Windows replacement fallback;
    # a reader may briefly hold a handle without delete-sharing.
    for attempt in range(10):
        try:
            if os.name == "nt" and path.exists():
                path.unlink()
            temp.replace(path)
            break
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(.01 * (attempt + 1))


def phase(name):
    value = {"phase": name, "updated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    if name == "SOLVING":
        value["solve_started_utc"] = value["updated_utc"]
    write("phase.json", value)


assert request["cloud_phase_handshake"]
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    assert os.environ[key] == "1"
phase("SETUP")
time.sleep(request.get("setup_s", .01))
phase("SOLVING")
if request.get("regress_phase"):
    time.sleep(.06)
    phase("SETUP")
time.sleep(request.get("work_s", .12))
if request.get("crash"):
    sys.exit(9)
phase("SERIALIZING")
time.sleep(request.get("serialization_s", .01))
if request.get("malformed_result"):
    (output / "worker.json").write_text("{malformed")
else:
    write("worker.json", {"synthetic_test_fixture": True,
                          "execution_status": "COMPLETED",
                          "fixed_work_signature": {"seed": request.get("seed", 11),
                                                   "path": [[1, 1], [3, 1]], "cost": 2},
                          "pid": os.getpid(), "success": False})
phase("FINISHED")
