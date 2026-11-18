"""Kaggle Linux build, smoke, Windows offline verification, and pilot continuation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import kaggle_native_paper_v4 as supervision
import kaggle_sem_exposure_pilot as linux_setup
import pri_pathsearch_protocol as protocol
from corridor_v4_execution import atomic_json, digest, file_hash, read, utc_now

TESTS = ["pri_pathsearch_" + component for component in
         ("core", "worker", "protocol", "runner", "validation", "analysis")]


def run_logged(name, argv, output, deadline, env, *, accepted=(0,)):
    logs = output / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stem = logs / f"{name}_{time.time_ns()}"
    receipt = dict(argv=list(map(str, argv)), started_utc=utc_now(), exit_code=None)
    print(name, "started", flush=True)
    process, cleanup_error = None, None
    supervision.enable_subreaper()
    baseline = {(pid, value["start_ticks"]) for pid, value in supervision.process_snapshot().items()
                if value["parent"] == os.getpid()}
    tracked = {}
    try:
        if time.monotonic() >= deadline:
            receipt["termination"] = "SESSION_DEADLINE_BEFORE_LAUNCH"
            raise RuntimeError("session deadline before " + name)
        with stem.with_suffix(".stdout.log").open("wb") as stdout, stem.with_suffix(".stderr.log").open("wb") as stderr:
            process = subprocess.Popen(receipt["argv"], cwd=ROOT, env=env, stdout=stdout,
                                       stderr=stderr, start_new_session=True)
            while process.poll() is None:
                supervision.track_descendants(supervision.process_snapshot(), process.pid, tracked, baseline)
                if time.monotonic() >= deadline:
                    receipt["termination"] = "SESSION_DEADLINE"
                    break
                time.sleep(.1)
    finally:
        if process is not None:
            try:
                if process.poll() is None:
                    receipt.setdefault("termination", "INTERRUPTED")
                    process.send_signal(signal.SIGINT)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                receipt["exit_code"] = process.returncode
                # Descendants can create their own process groups and outlive
                # the controller. Always drain them, even after an exit 0/2.
                grace = time.monotonic() + 2.
                while True:
                    snapshot = supervision.process_snapshot()
                    live = supervision.track_descendants(snapshot, process.pid, tracked, baseline)
                    if not live or time.monotonic() >= grace:
                        break
                    time.sleep(.05)
                trackers = set()
                for pid in live:
                    if snapshot[pid]["start_ticks"] != tracked.get(pid):
                        continue
                    try:
                        command = Path(f"/proc/{pid}/cmdline").read_bytes().rstrip(b"\x00").split(b"\x00")
                    except OSError:
                        continue
                    if (len(command) >= 3 and command[0].rsplit(b"/", 1)[-1].startswith(b"python")
                            and command[-2] == b"-c" and re.fullmatch(
                                rb"from multiprocessing\.resource_tracker import main;main\([0-9]+\)", command[-1])):
                        trackers.add(pid)
                terminated = supervision.quiesce_descendants(process.pid, tracked, baseline)
                receipt["descendants_quiescent"] = True
                if terminated:
                    receipt["terminated_descendant_pids"] = terminated
                    if set(terminated) <= trackers:
                        receipt["verified_resource_tracker_cleanup"] = terminated
                    else:
                        receipt.setdefault("termination", "ORPHANED_DESCENDANTS")
            except BaseException as error:
                cleanup_error = error
                receipt["quiescence_error"] = repr(error)
                supervision.SUPERVISION_ERROR = repr(error)
                # The notebook must refuse to archive mutable campaign state.
                atomic_json(output / "export_blocked.json", dict(command=name,
                    reason="descendant_quiescence_unproven", error=repr(error), utc=utc_now()))
        receipt["finished_utc"] = utc_now()
        atomic_json(stem.with_suffix(".json"), receipt)
    if cleanup_error is not None:
        raise RuntimeError("BLOCKED_EXPORT: subprocess descendants may still be alive") from cleanup_error
    result = stem.with_suffix(".stdout.log").read_text(encoding="utf-8", errors="replace")
    if receipt.get("termination") or receipt["exit_code"] not in accepted:
        diagnostic = stem.with_suffix(".stderr.log").read_text(encoding="utf-8", errors="replace")
        print(result[-5000:], diagnostic[-3000:], file=sys.stderr)
        raise RuntimeError(f"{name} failed, see {stem.name}.json and logs")
    print(name, "completed", flush=True)
    return result


def execute(args):
    if sys.platform != "linux" or platform.machine() != "x86_64":
        raise RuntimeError("Kaggle bootstrap requires x86_64 Linux")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = args.session_started_monotonic
    deadline = started + args.session_seconds - 180.
    if shutil.disk_usage(output).free < 6 * 1024**3:
        raise RuntimeError("BLOCKED_DISK: six GiB free required before build")
    migration = read(args.checkpoint / "migration.json")
    checkpoint_manifest = args.checkpoint / "pilot/manifest.json"
    if file_hash(checkpoint_manifest) != migration["pilot_manifest_sha256"]:
        raise ValueError("transported pilot manifest differs from receipt")
    if read(checkpoint_manifest)["sources"] != protocol.source_hashes():
        raise ValueError("frozen scientific sources differ")
    if time.monotonic() >= deadline:
        raise RuntimeError("session time exhausted before build")

    # Use the repository's already exercised pinned Linux dependency recipe.
    supervision.HARD_SESSION_S = args.session_seconds
    supervision.RESERVE_EXPORT_S = 180.
    ctest = output / "ctest_receipt.json"
    python, env = linux_setup.isolated_runtime(output, started, restored=ctest.exists())
    worker = ROOT / linux_setup.BUILD_DIR / "terrain_benchmark"
    bridge = ROOT / linux_setup.BUILD_DIR / "libterrain_compare_c.so"
    if not ctest.exists():
        linux_setup.build(output, started, python, env)
    runtime = linux_setup.resource_gate(output, started, python, env, "kaggle_pri_pathsearch")
    identity = dict(binary_sha256=file_hash(worker), bridge_sha256=file_hash(bridge),
                    sources_sha256=digest(protocol.source_hashes()))
    if not ctest.exists():
        regex = "^(" + "|".join(TESTS) + ")$"
        common = ["ctest", "--test-dir", worker.parent, "-R", regex]
        listing = json.loads(run_logged("ctest_inventory", [*common, "--show-only=json-v1"],
                                       output, deadline, env))
        if {item["name"] for item in listing["tests"]} != set(TESTS):
            raise ValueError("missing PRI pathsearch CTest suite")
        result = run_logged("ctest_correctness", [*common, "--verbose", "--no-tests=error"],
                            output, deadline, env)
        if re.search(r"OK\s*\(skipped=|\*\*\*Skipped|\*\*\*Not Run", result):
            raise ValueError("a required correctness test was skipped")
    elif any(read(ctest).get(key) != value for key, value in identity.items()):
        raise ValueError("restored build does not match CTest receipt")

    # Retain exact binaries, interpreter and shared libraries for a new session.
    runtime_archive = output / "linux_runtime.tar"
    runtime_receipt = output / "linux_runtime_sha256.json"
    if not runtime_archive.exists():
        temporary = output / "linux_runtime.partial"
        run_logged("retain_linux_runtime", ["tar", "-cf", temporary, "-C", ROOT,
                   linux_setup.VENV_DIR, ".managed-python", ".deps/install/lib",
                   worker.relative_to(ROOT), bridge.relative_to(ROOT)], output, deadline, env)
        temporary.replace(runtime_archive)
    if not runtime_receipt.exists():
        atomic_json(runtime_receipt, dict(sha256=file_hash(runtime_archive), **identity))
    if not ctest.exists():
        atomic_json(ctest, dict(exit_code=0, tests=TESTS, **identity))

    smoke = output / "linux_smoke"
    smoke_complete = output / "linux_smoke_done.json"
    if not smoke_complete.exists():
        runner = ROOT / "tools/run_pri_pathsearch.py"
        for phase in ("smoke_serial", "smoke_parallel"):
            phase_manifest = smoke / phase / "manifest.json"
            prior = read(phase_manifest) if phase_manifest.is_file() else {}
            command = [python, "-B", runner, "run", "--phase", phase, "--output", smoke,
                       "--worker", worker, "--bridge", bridge, "--sem-python", python]
            if prior and prior.get("status") != "PREPARED":
                command.append("--resume")
            run_logged("linux_" + phase, command, output, deadline, env)
        atomic_json(smoke_complete, dict(identity, status="DONE", utc=utc_now()))

    transport = ROOT / "kaggle_pri_pathsearch"
    imported = output / "windows_verification"
    report = imported / "report.json"
    if not report.exists() or read(report).get("status") != "COMPLETE":
        remaining = deadline - time.monotonic() - 60.
        if remaining <= 0:
            return "PAUSED_TIME"
        command = [python, "-B", transport / "import_windows.py", "--checkpoint", args.checkpoint,
                   "--output", imported, "--workers", "4", "--deadline-seconds", remaining,
                   "--windows-reference", ROOT / "windows_reference"]
        if imported.exists():
            command.append("--resume")
        run_logged("verify_windows_raw", command, output, deadline, env, accepted=(0, 2))
        if not report.is_file() or read(report).get("status") != "COMPLETE":
            return "PAUSED_WINDOWS_VERIFICATION"
    remaining = deadline - time.monotonic() - 60.
    if remaining <= 120.:
        return "PAUSED_TIME"
    continuation = output / "linux_continuation"
    command = [python, "-B", transport / "run.py", "run", "--checkpoint", args.checkpoint,
               "--output", continuation, "--worker", worker, "--bridge", bridge,
               "--sem-python", python, "--smoke-output", smoke, "--ctest-receipt", ctest,
               "--import-report", report, "--workers", "4", "--validation-workers", "4",
               "--deadline-seconds", remaining, "--drain-seconds", min(900., remaining / 4.),
               "--allow-pilot"]
    if (continuation / "manifest.json").is_file():
        command.append("--resume")
    summary = run_logged("continue_pilot", command, output, deadline, env)
    print(summary.strip(), flush=True)
    state = json.loads(summary.strip().splitlines()[-1])["status"]
    if state == "COMPLETE":
        run_logged("pilot_report", [python, "-B", transport / "report.py", "--checkpoint", args.checkpoint,
                   "--windows-report", report, "--continuation", continuation,
                   "--output", output / "mixed_platform_report"], output, deadline, env)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--session-started-monotonic", type=float, default=time.monotonic())
    parser.add_argument("--session-seconds", type=float, default=9 * 3600.)
    args = parser.parse_args()
    status = args.output / "bootstrap_status.json"
    try:
        state = execute(args)
        atomic_json(status, dict(status=state, updated_utc=utc_now(), pilot_only=True))
        print(state, flush=True)
    except BaseException as error:
        atomic_json(status, dict(status="FAILED_OR_BLOCKED", error=repr(error), updated_utc=utc_now()))
        raise


if __name__ == "__main__":
    main()
