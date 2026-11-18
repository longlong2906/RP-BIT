"""Small, spawn-only execution primitives for corridor v4.

The parent is the only campaign-state writer. Each child has one private task
directory and records its own result before it exits. No numerical library is
imported by this module; thread limits and affinity precede backend imports.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import signal
import tempfile
import time
import traceback

THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
              "OMP_THREAD_LIMIT")


def set_thread_limits():
    for name in THREAD_ENV:
        os.environ[name] = "1"
    os.environ.update(OMP_DYNAMIC="FALSE", MKL_DYNAMIC="FALSE")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def retry_file_access(operation, *args):
    deadline, delay = time.monotonic() + 5., .01
    while True:
        try:
            return operation(*args)
        except OSError as error:
            remaining = deadline - time.monotonic()
            if getattr(error, "winerror", None) not in (5, 32, 33) or remaining <= 0:
                raise
            time.sleep(min(delay, remaining))
            delay = min(.2, delay * 2)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        retry_file_access(os.replace, temporary, path)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()
    return hashlib.sha256(content).hexdigest()


def append_journal(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def process_alive(pid, created):
    if not pid or created is None:
        return False
    import psutil
    try:
        process = psutil.Process(pid)
        return process.is_running() and abs(process.create_time() - created) < .001
    except psutil.Error:
        return False


def terminate_tree(pid):
    """Stop only a process tree owned by the current controller."""
    import psutil
    try:
        root = psutil.Process(pid)
        children = root.children(recursive=True)
    except psutil.Error:
        return
    # Stop the launcher first, so it cannot create another descendant while
    # the known children are being stopped. All PIDs come from this live tree.
    for process in [root, *reversed(children)]:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs([root, *children], timeout=5)


def resource_preflight(workers, reservation_bytes):
    import psutil
    if type(workers) is not int or workers not in (1, 4):
        raise ValueError("workers must be 1 or 4")
    affinity = sorted(psutil.Process().cpu_affinity())
    vm = psutil.virtual_memory()
    required = workers * reservation_bytes
    result = dict(requested_workers=workers, eligible_cpus=affinity,
                  available_memory_bytes=int(vm.available), total_memory_bytes=int(vm.total),
                  reserved_bytes_per_worker=reservation_bytes, required_memory_bytes=required,
                  observed_utc=utc_now())
    result["admitted"] = len(affinity) >= workers and required <= .8 * vm.available
    result["reason"] = (None if result["admitted"] else
                        "INSUFFICIENT_ELIGIBLE_CPUS" if len(affinity) < workers else "INSUFFICIENT_AVAILABLE_RAM")
    result["cores"] = affinity[:workers] if result["admitted"] else []
    return result


def _child_entry(target, task, control, core):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    set_thread_limits()
    import psutil
    from terrain_compare_launch import pin_cpu
    control = Path(control)
    began = time.perf_counter()
    result = None
    try:
        actual = pin_cpu(core)
        atomic_json(control / "child.json", dict(pid=os.getpid(), process_created=psutil.Process().create_time(),
                    cpu_affinity=actual, thread_environment={key: os.environ[key] for key in THREAD_ENV},
                    affinity_before_backend_import=True, started_utc=utc_now(), task_id=task["task_id"]))
        result = dict(status="COMPLETED", payload=target(task))
    except BaseException as error:
        result = dict(status="FAILED_INFRASTRUCTURE", error=str(error), traceback=traceback.format_exc())
    result.update(task_id=task["task_id"], child_pid=os.getpid(), finished_utc=utc_now(),
                  child_wall_s=time.perf_counter() - began)
    atomic_json(control / "result.json", result)


def run_spawn_lanes(lanes, target, control_dir, *, cores, on_start=None, on_complete=None,
                    stop_dispatch_at_monotonic=None, terminate_at_monotonic=None):
    """Run FIFO queues on fixed cores, at most one fresh child per lane.

    Tasks require a unique, safe ``task_id``. Callbacks run only in the parent.
    The target is a top-level function, so Windows spawn never shares mutable
    planner state or RNG. A failed child stops further dispatch, drains active
    children, and leaves unstarted tasks untouched.

    Optional session deadlines use time.monotonic() absolute seconds. The soft
    deadline stops dispatch and drains active work. The hard deadline initiates
    termination of owned process trees and records their exits as INTERRUPTED;
    cleanup and durable receipt writes may finish shortly after that deadline.
    Omitting both arguments preserves the historical smoke scheduling policy.
    """
    if len(lanes) != len(cores) or len(lanes) not in (1, 4):
        raise ValueError("lane/core count must be exactly one or four")
    for deadline in (stop_dispatch_at_monotonic, terminate_at_monotonic):
        if deadline is not None and (type(deadline) not in (int, float) or not math.isfinite(deadline)):
            raise ValueError("session deadlines must be finite monotonic timestamps")
    if (stop_dispatch_at_monotonic is not None and terminate_at_monotonic is not None
            and stop_dispatch_at_monotonic > terminate_at_monotonic):
        raise ValueError("soft deadline must not exceed hard deadline")
    tasks = [task for lane in lanes for task in lane]
    ids = [task.get("task_id") for task in tasks]
    if (len(set(ids)) != len(ids) or any(not isinstance(x, str) or Path(x).name != x or
                                        x in (".", "..") for x in ids)):
        raise ValueError("duplicate or unsafe task identity")
    set_thread_limits()
    control_dir = Path(control_dir)
    control_dir.mkdir(parents=True, exist_ok=True)
    context = multiprocessing.get_context("spawn")
    queues = [list(lane) for lane in lanes]
    active = [None] * len(lanes)
    result = dict(status="COMPLETE", start_method="spawn", requested_workers=len(lanes),
                  results=[], concurrency_timeline=[], started_utc=utc_now())
    began = time.perf_counter()
    stop_dispatch = False
    deadline_enabled = stop_dispatch_at_monotonic is not None or terminate_at_monotonic is not None
    if deadline_enabled:
        result.update(stop_dispatch_at_monotonic=stop_dispatch_at_monotonic,
                      terminate_at_monotonic=terminate_at_monotonic, stop_reason=None)

    def deadline_reached():
        nonlocal stop_dispatch
        now = time.monotonic()
        hard = terminate_at_monotonic is not None and now >= terminate_at_monotonic
        soft = stop_dispatch_at_monotonic is not None and now >= stop_dispatch_at_monotonic
        if hard or soft:
            stop_dispatch = True
            if any(queues) or hard and any(active):
                if result["status"] == "COMPLETE":
                    result["status"] = "DEADLINE"
                result["stop_reason"] = "HARD_SESSION_DEADLINE" if hard else "SOFT_SESSION_DEADLINE"
        return hard

    def timeline():
        running = [item for item in active if item]
        result["concurrency_timeline"].append(dict(elapsed_s=time.perf_counter() - began,
             utc=utc_now(), concurrency=len(running), task_ids=[x["task"]["task_id"] for x in running],
             pids=[x["process"].pid for x in running]))
        atomic_json(control_dir / "progress.json", {k: v for k, v in result.items() if k != "results"})

    def complete(index, interrupted=False):
        item = active[index]
        process, task, directory = item["process"], item["task"], item["directory"]
        process.join()
        exit_record = dict(task_id=task["task_id"], lane=index, pid=process.pid,
                           exit_code=process.exitcode, finished_utc=utc_now(), interrupted=interrupted)
        # Capture OS exit before any result parsing or parent callback.
        atomic_json(directory / "exit.json", exit_record)
        if (directory / "result.json").is_file():
            value = read(directory / "result.json")
            if value.get("task_id") != task["task_id"]:
                raise ValueError("child result task identity mismatch")
        else:
            value = dict(task_id=task["task_id"], status="INTERRUPTED" if interrupted else "FAILED_INFRASTRUCTURE",
                         error="child_result_unavailable")
        value["process_exit"] = exit_record
        if process.exitcode != 0 or interrupted:
            value["status"] = "INTERRUPTED" if interrupted else "FAILED_INFRASTRUCTURE"
        result["results"].append(value)
        active[index] = None
        if on_complete:
            on_complete(task, value)
        timeline()
        return value

    try:
        while any(queues) or any(active):
            for index, item in enumerate(active):
                if item and not item["process"].is_alive():
                    value = complete(index)
                    if value["status"] != "COMPLETED":
                        stop_dispatch = True
                        result["status"] = "FAILED_INFRASTRUCTURE"
            if deadline_enabled and deadline_reached():
                for item in active:
                    if item and item["process"].pid:
                        terminate_tree(item["process"].pid)
                for index in range(len(active)):
                    if active[index]:
                        complete(index, interrupted=True)
                break
            if stop_dispatch and not any(active):
                break
            for index, queue in enumerate(queues):
                if deadline_enabled:
                    deadline_reached()
                if stop_dispatch or active[index] is not None or not queue:
                    continue
                task = dict(queue.pop(0))
                task["queue_wait_s"] = time.perf_counter() - began
                directory = control_dir / task["task_id"]
                directory.mkdir(exist_ok=False)
                started = dict(task_id=task["task_id"], lane=index, core=cores[index], started_utc=utc_now(),
                               task_sha256=digest(task), queue_wait_s=task["queue_wait_s"])
                atomic_json(directory / "start.json", started)
                if on_start:
                    on_start(task, started)
                process = context.Process(target=_child_entry, args=(target, task, directory, cores[index]))
                process.start()
                active[index] = dict(task=task, process=process, directory=directory)
                timeline()
            time.sleep(.02)
    except BaseException:
        result["status"] = "INTERRUPTED"
        for item in active:
            if item and item["process"].pid:
                terminate_tree(item["process"].pid)
        for index in range(len(active)):
            if active[index]:
                complete(index, interrupted=True)
        result.update(finished_utc=utc_now(), wall_s=time.perf_counter() - began)
        atomic_json(control_dir / "summary.json", result)
        raise
    result.update(finished_utc=utc_now(), wall_s=time.perf_counter() - began,
                  max_concurrency=max((x["concurrency"] for x in result["concurrency_timeline"]), default=0))
    if deadline_enabled:
        result["unstarted_task_ids"] = [task["task_id"] for queue in queues for task in queue]
    atomic_json(control_dir / "summary.json", result)
    return result
