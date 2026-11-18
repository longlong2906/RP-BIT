"""Run an argv without a shell and preserve command, exit status and both streams."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required")
    prefix = args.log.resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = {suffix: prefix.with_name(prefix.name + suffix) for suffix in (".json", ".stdout.log", ".stderr.log")}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("evidence prefix already exists: " + str(prefix))
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    if os.name == "nt":
        root = Path(__file__).resolve().parents[1]
        environment["PATH"] = os.pathsep.join((str(root / ".deps/install/lib"), "C:/mingw64/bin", environment.get("PATH", "")))
    receipt = dict(argv=command, cwd=str(Path.cwd()), started_utc=datetime.now(timezone.utc).isoformat(),
                   stdout=str(paths[".stdout.log"]), stderr=str(paths[".stderr.log"]), exit_code=None)
    paths[".json"].write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    began = time.perf_counter()
    with paths[".stdout.log"].open("wb") as stdout, paths[".stderr.log"].open("wb") as stderr:
        try:
            receipt["exit_code"] = subprocess.run(command, env=environment, stdout=stdout, stderr=stderr, check=False).returncode
        except OSError as error:
            receipt.update(exit_code=127, launch_error=str(error))
    receipt.update(finished_utc=datetime.now(timezone.utc).isoformat(), wall_s=time.perf_counter()-began)
    paths[".json"].write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt))
    for name in (".stdout.log", ".stderr.log"):
        lines = paths[name].read_text(encoding="utf-8", errors="replace").splitlines()
        if lines:
            print("\n".join(lines[-20:]))
    return receipt["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
