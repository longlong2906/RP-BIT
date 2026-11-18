"""Execute a real command, retaining exact argv, output, exit code and elapsed time."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import time


def run(command, log_dir, name, *, cwd=None, env=None, timeout=None):
    directory = Path(log_dir); directory.mkdir(parents=True, exist_ok=True)
    receipt = directory/(name+".json")
    if receipt.exists(): raise ValueError("receipt exists; use a new attempt name")
    began = time.perf_counter()
    record = {"argv": [str(x) for x in command], "cwd": str(Path(cwd or '.').resolve()),
              "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "exit_code": None}
    try:
        with (directory/(name+".stdout.log")).open("w", encoding="utf-8") as out, (directory/(name+".stderr.log")).open("w", encoding="utf-8") as err:
            result = subprocess.run(record["argv"], cwd=cwd, env=env, stdout=out, stderr=err, timeout=timeout)
            record["exit_code"] = result.returncode
    except BaseException as exc:
        record["exception"] = repr(exc)
        raise
    finally:
        record["elapsed_s"] = time.perf_counter()-began
        receipt.write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
    return record


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("--log-dir",default="logs/terrain_compare_20261023")
    p.add_argument("--name",required=True);p.add_argument("--timeout",type=float)
    p.add_argument("command",nargs=argparse.REMAINDER);a=p.parse_args()
    cmd=a.command[1:] if a.command[:1]==["--"] else a.command
    r=run(cmd,a.log_dir,a.name,timeout=a.timeout)
    print(json.dumps(r));raise SystemExit(r["exit_code"])
