"""Run a command without a shell; retain its argv, exit code and complete output."""
import argparse
import datetime as dt
import json
import pathlib
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--log-dir", default="logs/build")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    directory = pathlib.Path(args.log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stem = directory / args.name
    if stem.with_suffix(".json").exists():
        parser.error("log name already exists; choose a new name to preserve evidence")
    started = time.perf_counter()
    utc = dt.datetime.now(dt.timezone.utc).isoformat()
    try:
        run = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
        code, stdout, stderr = run.returncode, run.stdout, run.stderr
    except OSError as error:
        code, stdout, stderr = 127, "", str(error)
    stem.with_suffix(".stdout.log").write_text(stdout, encoding="utf-8")
    stem.with_suffix(".stderr.log").write_text(stderr, encoding="utf-8")
    stem.with_suffix(".json").write_text(json.dumps({
        "command": command, "cwd": str(pathlib.Path.cwd()), "utc": utc,
        "exit_code": code, "wall_s": time.perf_counter() - started,
    }, indent=2), encoding="utf-8")
    print(stdout, end="")
    print(stderr, end="", file=sys.stderr)
    print(f"exit_code={code}; logs={stem}")
    return code


if __name__ == "__main__":
    sys.exit(main())
