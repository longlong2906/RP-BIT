"""Four-lane, single-writer scheduler for the static terrain comparison.

This module collects raw attempts only. Independent validation runs in a separate
phase with no measured workers alive. A completed process is not a success claim.
"""
from __future__ import annotations

import argparse
import collections
import copy
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import signal
import shutil
import subprocess
import sys
import time

import psutil

GIB = 1024 ** 3
MIB = 1024 ** 2
POLL_S = .02
PRESSURE_S = 5.0
SETUP_LIMIT_S = 180.0
SERIALIZATION_LIMIT_S = 60.0
SOLVE_GRACE_S = 5.0
TRIAL_RSS_LIMIT = 8 * GIB
PHASES = {"SETUP": 0, "SOLVING": 1, "SERIALIZING": 2, "FINISHED": 3}
PROTOCOL = "terrain_compare_four_lane_v1"
LAUNCHER_PATH = Path(__file__).with_name("terrain_compare_launch.py")
AFFINITY_GUARD_PROTOCOL = "launch_proof_before_sample_v2"
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
              "OMP_THREAD_LIMIT")


class IntegrityError(ValueError):
    pass


class AffinityProtocolError(IntegrityError):
    pass


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise IntegrityError("duplicate JSON key: " + key)
            result[key] = value
        return result
    def invalid(value):
        raise IntegrityError("nonfinite JSON: " + value)
    return json.loads(Path(path).read_text(encoding="utf-8-sig"),
                      object_pairs_hook=pairs, parse_constant=invalid)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def parse_cpuset(text):
    cpus = set()
    for part in text.strip().split(","):
        if not part:
            continue
        values = part.split("-")
        first, last = int(values[0]), int(values[-1])
        if first < 0 or last < first or len(values) > 2:
            raise IntegrityError("invalid cpuset")
        cpus.update(range(first, last + 1))
    return sorted(cpus)


def _text(path):
    try:
        return Path(path).read_text().strip()
    except (OSError, UnicodeError):
        return None


def _cgroup_dirs(root=Path("/sys/fs/cgroup"), membership=Path("/proc/self/cgroup")):
    """Include ancestors: a child can have max while its parent is constrained."""
    root = Path(root)
    paths = {root}
    for line in (_text(membership) or "").splitlines():
        pieces = line.split(":", 2)
        if len(pieces) != 3:
            continue
        _, controllers, member = pieces
        if ".." in Path(member).parts:
            continue
        bases = [root] if not controllers else [root / name for name in controllers.split(",")]
        if controllers:
            bases.append(root / controllers)
        for base in bases:
            cursor = base / member.lstrip("/")
            while cursor == base or base in cursor.parents:
                paths.add(cursor)
                if cursor == base:
                    break
                cursor = cursor.parent
    return sorted(paths)


def detect_runtime(cgroup_root=Path("/sys/fs/cgroup"), membership=Path("/proc/self/cgroup")):
    affinity = sorted(psutil.Process().cpu_affinity())
    vm = psutil.virtual_memory()
    quota = None
    memory_limits = [int(vm.total)]
    memory_sources = []
    constraints = []
    if sys.platform.startswith("linux"):
        for directory in _cgroup_dirs(cgroup_root, membership):
            cpus = _text(directory / "cpuset.cpus.effective") or _text(directory / "cpuset.cpus")
            if cpus:
                affinity = sorted(set(affinity).intersection(parse_cpuset(cpus)))
                constraints.append({"path": str(directory), "cpuset": cpus})
            cpu_max = _text(directory / "cpu.max")
            if cpu_max:
                amount, period = cpu_max.split()
                if amount != "max":
                    value = int(amount) / int(period)
                    quota = value if quota is None else min(value, quota)
            else:
                amount, period = _text(directory / "cpu.cfs_quota_us"), _text(directory / "cpu.cfs_period_us")
                if amount and period and int(amount) > 0:
                    value = int(amount) / int(period)
                    quota = value if quota is None else min(value, quota)
            for limit_name, current_name in (("memory.max", "memory.current"),
                                              ("memory.limit_in_bytes", "memory.usage_in_bytes")):
                limit = _text(directory / limit_name)
                if limit and limit != "max" and 0 < int(limit) <= vm.total:
                    memory_limits.append(int(limit))
                    memory_sources.append({"limit_bytes": int(limit),
                                           "current_file": str(directory / current_name)})
    if not affinity:
        raise IntegrityError("no eligible CPU after cpuset intersection")
    capacity = len(affinity)
    if quota is not None:
        if quota <= 0:
            raise IntegrityError("nonpositive CPU quota")
        capacity = min(capacity, max(1, math.floor(quota)))
    return {"platform": sys.platform, "os_release": platform.release(),
            "machine": platform.machine(), "host": platform.node(),
            "cpu_model": next((line.split(":", 1)[1].strip() for line in (_text("/proc/cpuinfo") or "").splitlines()
                               if line.startswith("model name") and ":" in line), platform.processor()),
            "python": platform.python_version(), "eligible_cpus": affinity,
            "cpu_quota": quota, "cpu_capacity": capacity,
            "physical_memory_bytes": vm.total, "memory_limit_bytes": min(memory_limits),
            "memory_sources": memory_sources, "cpuset_constraints": constraints}


def runtime_signature(runtime):
    """Host names/cgroup namespace paths change between equivalent sessions."""
    return {key: value for key, value in runtime.items()
            if key not in ("host", "memory_sources", "cpuset_constraints")}


def campaign_runtime_identity(plan, runtime):
    """Registered native-paper policies admit a different Kaggle CPU session.

    Source, dependency, binary, plan and scheduling-policy identities stay locked.
    The actual hardware is recorded independently in every session and attempt.
    Legacy campaigns retain the strict whole-runtime identity.
    """
    if (plan.get("protocol_id") == "a3t_native_paper_kaggle_v2" and
            plan.get("runtime_policy") == "native_paper_sessions_v2"):
        return {"policy": "native_paper_sessions_v2",
                **{k: runtime[k] for k in ("platform", "machine", "python")}}
    if (plan.get("protocol_id") == "a3t_sem_putn_kaggle_v3" and
            plan.get("schema_version") == 12 and
            plan.get("runtime_policy") == "native_paper_sessions_v3"):
        return {"policy": "native_paper_sessions_v3",
                **{k: runtime[k] for k in ("platform", "machine", "python")}}
    if (plan.get("protocol_id") == "a3t_old_safe_sem_kaggle_v4" and
            plan.get("schema_version") == 13 and plan.get("runtime_policy") == "native_paper_sessions_v4"):
        return {"policy": "native_paper_sessions_v4",
                **{k: runtime[k] for k in ("platform", "machine", "python")}}
    return runtime_signature(runtime)


def phase_timeout_policy(plan):
    """Only the registered native paper protocol admits a longer setup guard.

    These are resource watchdogs, never planning budgets. Legacy protocols keep
    their original limits and cannot obtain the native policy through a request.
    """
    native_policies={"a3t_native_paper_kaggle_v2":(11,"native_paper_sessions_v2"),
                     "a3t_sem_putn_kaggle_v3":(12,"native_paper_sessions_v3"),
                     "a3t_old_safe_sem_kaggle_v4":(13,"native_paper_sessions_v4")}
    if plan.get("protocol_id") in native_policies:
        schema,policy=native_policies[plan["protocol_id"]]
        if plan.get("schema_version") != schema or plan.get("runtime_policy") != policy:
            raise IntegrityError("unregistered native phase timeout policy")
        expected={"setup_s":600.,"serialize_s":60.,"solve_grace_s":5.}
        value=plan.get("phase_timeouts")
        if (not isinstance(value,dict) or set(value)!=set(expected) or
                any(isinstance(value[k],bool) or not isinstance(value[k],(int,float)) or
                    not math.isfinite(value[k]) or value[k]!=v for k,v in expected.items())):
            raise IntegrityError("native phase_timeouts differ from registered 600/60/5 policy")
        return {"setup_watchdog_s":value["setup_s"],"serialization_watchdog_s":value["serialize_s"],
                "solve_grace_s":value["solve_grace_s"]}
    return {"setup_watchdog_s":SETUP_LIMIT_S,"serialization_watchdog_s":SERIALIZATION_LIMIT_S,
            "solve_grace_s":SOLVE_GRACE_S}


def memory_pressure(runtime):
    vm = psutil.virtual_memory()
    ratios = [(vm.total - vm.available) / vm.total]
    for source in runtime["memory_sources"]:
        current = _text(source["current_file"])
        if current is not None:
            ratios.append(int(current) / source["limit_bytes"])
    return max(ratios)


def reservation_bytes(peak_rss_bytes):
    peak = float(peak_rss_bytes)
    if not math.isfinite(peak) or peak < 0:
        raise IntegrityError("invalid peak RSS")
    return min(TRIAL_RSS_LIMIT, max(512 * MIB, math.ceil(1.5 * peak + 256 * MIB)))


def collect_reservations(manifest):
    if isinstance(manifest, (str, Path)):
        manifest = read_json(manifest)
    peaks = {}
    for entry in manifest["entries"]:
        if entry.get("peak_rss_bytes") is not None:
            backend = entry["backend"]
            peaks[backend] = max(peaks.get(backend, 0), entry["peak_rss_bytes"])
    return {backend: reservation_bytes(peak) for backend, peak in peaks.items()}


def _safe_id(value):
    if not isinstance(value, str) or not value or value in (".", "..") or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_." for c in value):
        raise IntegrityError("unsafe trial_id")
    return value


def _freeze_inputs(manifest_path, registry):
    manifest_path = Path(manifest_path).resolve()
    plan = read_json(manifest_path)
    entries = plan.get("entries", plan.get("trials"))
    if not isinstance(entries, list) or not entries:
        raise IntegrityError("manifest must have nonempty entries")
    plan["entries"] = entries
    seen = set()
    for entry in entries:
        trial_id = _safe_id(entry["trial_id"])
        if trial_id in seen:
            raise IntegrityError("duplicate trial_id")
        seen.add(trial_id)
        if entry.get("execution_status", "PLANNED") not in ("PLANNED", "BLOCKED_SOURCE"):
            raise IntegrityError("input manifest must be an immutable plan")
        entry.setdefault("execution_status", "PLANNED")
        for field in ("block_order", "planned_order"):
            if not isinstance(entry.get(field), int) or isinstance(entry[field], bool):
                raise IntegrityError("missing integer " + field)
        if entry["execution_status"] == "BLOCKED_SOURCE":
            continue
        if entry["backend"] not in registry:
            raise IntegrityError("backend missing from registry: " + entry["backend"])
        request = entry.get("request")
        if isinstance(request, str):
            path = Path(request)
            request = read_json(path if path.is_absolute() else manifest_path.parent / path)
        if not isinstance(request, dict):
            raise IntegrityError("request must be an object or JSON path")
        budget = request.get("planning_budget_s")
        if isinstance(budget, bool) or not isinstance(budget, (int, float)) or not math.isfinite(budget) or budget <= 0:
            raise IntegrityError("positive planning_budget_s required")
        entry["request"] = request
    ordering = [(e["block_order"], e["planned_order"]) for e in entries]
    if len(ordering) != len(set(ordering)):
        raise IntegrityError("duplicate block/planned order")
    registry = copy.deepcopy(registry)
    artifacts = {}
    for backend, spec in registry.items():
        argv = spec.get("argv")
        if not isinstance(argv, list) or not argv or not all(isinstance(v, str) for v in argv):
            raise IntegrityError("registry argv must be a nonempty string list")
        for value in argv:
            candidate = Path(value)
            if candidate.is_file():
                artifacts[str(candidate.resolve())] = file_hash(candidate)
        if not Path(argv[0]).is_absolute() or not Path(argv[0]).is_file():
            raise IntegrityError("registry executable must be an existing absolute path")
    for module in (Path(__file__), Path(LAUNCHER_PATH)):
        artifacts[str(module.resolve())] = file_hash(module)
    return plan, registry, artifacts


def _process_alive(pid, created):
    try:
        process = psutil.Process(pid)
        return process.is_running() and process.create_time() == created
    except psutil.Error:
        return False


def _lock(output):
    lock = output / "controller.lock"
    if lock.exists():
        previous = read_json(lock)
        if _process_alive(previous["pid"], previous["created"]):
            raise IntegrityError("another controller is active")
        lock.unlink()
    with lock.open("x", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "created": psutil.Process().create_time()}, stream)
    return lock


def _append_journal(path, value):
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _read_phase(path):
    if not path.exists():
        return None
    try:
        value = read_json(path)
    except (FileNotFoundError, PermissionError):
        # Existing C++ Windows writer can unlink between exists() and open().
        # Missing observations never reset the watchdog clock.
        return None
    if value.get("phase") not in PHASES:
        raise IntegrityError("unknown worker phase")
    return value


def _sample_process(active):
    rss, affinities, threads = 0, {}, 0
    identities, errors = {}, []
    try:
        root = psutil.Process(active["proc"].pid)
        processes = [root] + root.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        processes = []
    except psutil.Error as exc:
        processes = []
        errors.append({"pid": active["proc"].pid, "error": type(exc).__name__})
    for process in processes:
        try:
            key = (process.pid, process.create_time())
            identities[str(process.pid)] = key[1]
            if process.pid == active["proc"].pid and key[1] != active["entry"].get("pid_created"):
                errors.append({"pid": process.pid, "error": "process_identity_mismatch"})
                continue
            cpu = process.cpu_times()
            active["cpu_by_pid"][key] = max(active["cpu_by_pid"].get(key, 0), cpu.user + cpu.system)
            rss += process.memory_info().rss
            affinities[str(process.pid)] = sorted(process.cpu_affinity())
            threads += process.num_threads()
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
        except (psutil.Error, AttributeError, NotImplementedError, OSError) as exc:
            errors.append({"pid": process.pid, "error": type(exc).__name__})
    active["peak_rss"] = max(active["peak_rss"], rss)
    active["max_threads"] = max(active["max_threads"], threads)
    active["affinities"] = affinities
    active["affinity_sample_identities"] = identities
    active["affinity_sample_errors"] = errors
    return rss


def _read_launch_proof(active):
    """Return a validated proof, or None only when not yet published."""
    try:
        proof = read_json(active["directory"] / "launch.json")
    except FileNotFoundError:
        return None
    except (ValueError, OSError) as exc:
        raise AffinityProtocolError("invalid launch proof: " + str(exc)) from exc
    entry = active["entry"]
    if not isinstance(proof, dict):
        raise AffinityProtocolError("launch proof must be an object")
    created = proof.get("process_created")
    if (type(proof.get("pid")) is not int or proof["pid"] != active["proc"].pid
            or proof["pid"] != entry.get("pid")
            or isinstance(created, bool) or not isinstance(created, (int, float))
            or not math.isfinite(created) or created <= 0 or created != entry.get("pid_created")):
        raise AffinityProtocolError("launch proof process identity mismatch")
    if (proof.get("cpu_affinity") != [entry["core"]]
            or not all(type(core) is int for core in proof["cpu_affinity"])
            or proof.get("affinity_before_worker_import") is not True):
        raise AffinityProtocolError("launch proof pinning mismatch")
    environment = proof.get("thread_environment")
    if not isinstance(environment, dict) or any(environment.get(key) != "1" for key in THREAD_ENV):
        raise AffinityProtocolError("launch proof thread environment mismatch")
    command = entry["command"]
    if proof.get("command") != command[command.index("--") + 1:]:
        raise AffinityProtocolError("launch proof command mismatch")
    previous = active.get("verified_launch_proof")
    if previous is not None and proof != previous:
        raise AffinityProtocolError("launch proof changed after verification")
    active["verified_launch_proof"] = proof
    return proof


def _sample_with_affinity_guard(active):
    """A proof published during a sample cannot authenticate that sample."""
    guard = active["entry"].setdefault("affinity_guard", {
        "protocol": AFFINITY_GUARD_PROTOCOL, "prelaunch_samples": 0, "postlaunch_samples": 0})
    observation = {"observed_utc": utc(), "sample_began_wall_s": time.perf_counter() - active["began"],
                   "expected_pid": active["proc"].pid,
                   "expected_process_created": active["entry"].get("pid_created"),
                   "expected_cpu_affinity": [active["entry"]["core"]],
                   "launch_ready_before_sample": False}
    guard["last_observation"] = observation
    try:
        # This ordering is the barrier: never use a later exists() check to
        # judge an observation that may have preceded launcher pinning.
        proof = _read_launch_proof(active)
        observation["launch_ready_before_sample"] = proof is not None
        if proof is not None:
            observation["launch_proof_json_sha256"] = digest(proof)
        rss = _sample_process(active)
        observation.update(observed_affinities=copy.deepcopy(active["affinities"]),
                           observed_process_created=copy.deepcopy(active.get("affinity_sample_identities", {})),
                           observation_errors=copy.deepcopy(active.get("affinity_sample_errors", [])),
                           sample_ended_wall_s=time.perf_counter() - active["began"])
        guard["postlaunch_samples" if proof is not None else "prelaunch_samples"] += 1
        if proof is not None:
            if observation["observation_errors"]:
                raise AffinityProtocolError("post-launch affinity observation unavailable")
            if not active["affinities"] and active["proc"].poll() is None:
                raise AffinityProtocolError("no live post-launch affinity observation")
            if any(affinity != [active["entry"]["core"]] for affinity in active["affinities"].values()):
                observation["failure"] = "AFFINITY_DRIFT"
                guard["failure_observation"] = copy.deepcopy(observation)
                return rss, "AFFINITY_DRIFT"
        return rss, None
    except (AffinityProtocolError, AttributeError, NotImplementedError) as exc:
        observation["error"] = str(exc)
        observation["failure"] = "AFFINITY_PROTOCOL_ERROR"
        guard["failure_observation"] = copy.deepcopy(observation)
        raise AffinityProtocolError(str(exc)) from exc


def _stop_process(active):
    proc = active["proc"]
    try:
        descendants = psutil.Process(proc.pid).children(recursive=True)
    except psutil.Error:
        descendants = []
    if os.name != "nt":
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    else:
        for process in descendants:
            try:
                process.terminate()
            except psutil.Error:
                pass
        if proc.poll() is None:
            proc.terminate()
    try:
        proc.wait(timeout=.5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    if os.name != "nt":
        # A descendant may outlive its parent or be born between enumeration
        # and SIGTERM. The fresh per-trial session owns this whole group.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for process in descendants:
        try:
            process.kill()
        except psutil.Error:
            pass


def _phase_update(active, now):
    phase = _read_phase(active["directory"] / "phase.json")
    if phase is not None and phase["phase"] != active["phase"]:
        name = phase["phase"]
        if PHASES[name] < PHASES[active["phase"]]:
            raise IntegrityError("worker phase regressed")
        active["phase"] = name
        began = now
        timestamp = phase.get("solve_started_utc") if name == "SOLVING" else phase.get("updated_utc")
        if timestamp:
            parsed = dt.datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                raise IntegrityError("worker phase timestamp lacks timezone")
            age = (dt.datetime.now(dt.timezone.utc) - parsed).total_seconds()
            if age < -1:
                raise IntegrityError("worker phase timestamp is in the future")
            began = max(active["began"], now - max(0, age))
        active["phase_began"] = began
        active["phase_history"].append({"phase": name, "observed_wall_s": now - active["began"],
                                        "worker_timestamp": timestamp})


def _complete(active, failure=None):
    now = time.perf_counter()
    if failure:
        _stop_process(active)
    code = active["proc"].wait()
    for stream in active["streams"]:
        stream.close()
    record = active["entry"]
    status = "COMPLETED"
    if failure:
        status = ("WATCHDOG" if failure.endswith("_WATCHDOG") or failure == "MEMORY_LIMIT"
                  else "RESOURCE_PRESSURE" if failure == "RESOURCE_PRESSURE"
                  else "INTERRUPTED" if failure in ("INTERRUPTED", "SESSION_LIMIT")
                  else "FAILED_CORRECTNESS")
    elif code:
        status, failure = "CRASH", "NONZERO_WORKER_EXIT"
    else:
        result_path = active["directory"] / "worker.json"
        if not result_path.exists():
            status, failure = "CRASH", "MISSING_WORKER_RESULT"
        else:
            result = read_json(result_path)
            if result.get("execution_status") in ("FAILED_CORRECTNESS", "PROVENANCE_MISMATCH"):
                status, failure = "FAILED_CORRECTNESS", result["execution_status"]
            elif (_read_phase(active["directory"] / "phase.json") or {}).get("phase") != "FINISHED":
                status, failure = "CRASH", "INCOMPLETE_PHASE_HANDSHAKE"
        try:
            launch_proof = _read_launch_proof(active)
        except AffinityProtocolError as exc:
            record["protocol_error"] = str(exc)
            launch_proof = None
        if launch_proof is None:
            status, failure = "FAILED_CORRECTNESS", "AFFINITY_LAUNCH_PROOF_MISSING"
    record.update(execution_status=status, termination_reason=failure or "WORKER_FINISHED",
                  exit_code=code, completed_utc=utc(), worker_wall_s=now - active["began"],
                  peak_rss_bytes=active["peak_rss"], process_tree_cpu_s=sum(active["cpu_by_pid"].values()),
                  peak_observed_process_tree_threads=active["max_threads"],
                  resource_measurement="sampled_20ms_process_tree; exited_between_samples_may_be_missed",
                  phase_history=active["phase_history"], success=False, independent_valid=None,
                  validation_status="PENDING" if status == "COMPLETED" else "NOT_RUN")
    atomic_json(active["directory"] / "execution.json", record)
    return status


def run_campaign(manifest_path, registry, output, *, max_workers=4, reservations=None,
                 resume=False, session_elapsed_s=0, soft_stop_s=28800, force_local=False):
    """Collect raw attempts. Input plan is immutable; resume never retries starts.

    reservations maps backend to reserved bytes, or {'peak_rss_bytes': number}.
    Missing peaks reserve 8 GiB conservatively; first serial smoke needs no peaks.
    force_local permits non-Linux correctness/smoke only; it does not change data.
    """
    if not isinstance(max_workers, int) or isinstance(max_workers, bool) or not 1 <= max_workers <= 4:
        raise IntegrityError("max_workers must be an integer in [1,4]")
    if not math.isfinite(session_elapsed_s) or session_elapsed_s < 0 or not math.isfinite(soft_stop_s) or soft_stop_s < 0:
        raise IntegrityError("invalid session time")
    plan, registry, artifacts = _freeze_inputs(manifest_path, registry)
    phase_limits=phase_timeout_policy(plan)
    runtime = detect_runtime()
    if not runtime["platform"].startswith("linux") and not force_local:
        raise IntegrityError("measured campaign requires Linux; use force_local only for local smoke/tests")
    if force_local and plan.get("profile") in ("main", "full", "benchmark"):
        raise IntegrityError("main benchmark is not allowed with force_local")
    reservations = reservations or {}
    reserve = {}
    for backend in registry:
        value = reservations.get(backend, TRIAL_RSS_LIMIT)
        if isinstance(value, dict):
            value = reservation_bytes(value["peak_rss_bytes"])
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= TRIAL_RSS_LIMIT:
            raise IntegrityError("invalid reservation")
        reserve[backend] = math.ceil(value)
    count = min(max_workers, runtime["cpu_capacity"])
    policy = {"protocol": PROTOCOL, "requested_workers": max_workers, "worker_count": count,
              "reservations_bytes": reserve, "admission_fraction": .8, "pressure_fraction": .9,
              "pressure_duration_s": PRESSURE_S, "trial_rss_limit_bytes": TRIAL_RSS_LIMIT,
              **phase_limits, "soft_stop_s": soft_stop_s,
              "thread_environment": {key: "1" for key in THREAD_ENV}, "force_local": force_local}
    identity = {"plan_sha256": digest(plan), "registry_sha256": digest(registry),
                "artifact_sha256": artifacts, "policy_sha256": digest(policy),
                "runtime_sha256": digest(campaign_runtime_identity(plan, runtime))}
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = _lock(output)
    try:
        return _run_locked(plan, registry, output, resume, runtime, policy, identity,
                           session_elapsed_s, soft_stop_s)
    finally:
        lock.unlink(missing_ok=True)


def _run_locked(plan, registry, output, resume, runtime, policy, identity,
                session_elapsed_s, soft_stop_s):
    effective_scheduler_sha256 = file_hash(Path(__file__))
    target, journal = output / "manifest.json", output / "start_journal.jsonl"
    if target.exists():
        if not resume:
            raise IntegrityError("output already exists; resume is required")
        manifest = read_json(target)
        if manifest.get("identity") != identity:
            raise IntegrityError("resume provenance/policy/runtime hash mismatch")
    else:
        if resume:
            raise IntegrityError("resume manifest is missing")
        manifest = copy.deepcopy(plan)
        manifest.update(identity=identity, scheduler_policy=policy, runtime=runtime,
                        concurrency_timeline=[], sessions=[], status="PLANNED")
        atomic_json(output / "runtime.json", runtime)
        atomic_json(output / "registry.json", registry)
        atomic_json(target, manifest)
    entries = {entry["trial_id"]: entry for entry in manifest["entries"]}
    started = set()
    if journal.exists():
        for line in journal.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            trial_id = event["trial_id"]
            if trial_id not in entries or trial_id in started:
                raise IntegrityError("duplicate or unknown journal trial")
            if event.get("identity_sha256") != digest(identity):
                raise IntegrityError("start journal provenance mismatch")
            started.add(trial_id)
            if entries[trial_id]["execution_status"] in ("RUNNING", "START_REQUESTED", "PLANNED"):
                record = entries[trial_id]
                if record.get("pid") and _process_alive(record["pid"], record.get("pid_created")):
                    raise IntegrityError("previous trial still alive; refuse duplicate collection")
                launch_path = output / trial_id / "launch.json"
                if launch_path.exists():
                    launch = read_json(launch_path)
                    # The launcher may have started before the controller had
                    # time to save its PID. On Linux exec retains this PID.
                    if launch.get("process_created") and _process_alive(launch["pid"], launch["process_created"]):
                        raise IntegrityError("orphan trial launcher still alive; refuse resume")
                record.update(execution_status="INTERRUPTED", termination_reason="PREVIOUS_CONTROLLER_INTERRUPTED",
                              success=False, independent_valid=None)
                atomic_json(output / trial_id / "execution.json", record)
    groups = collections.defaultdict(list)
    for entry in manifest["entries"]:
        if entry["execution_status"] == "PLANNED" and entry["trial_id"] not in started:
            groups[entry["block_order"]].append(entry)
    pending = collections.deque(sorted(groups))
    for group in groups.values():
        group.sort(key=lambda value: value["planned_order"])
    lanes = [{"lane": i, "core": runtime["eligible_cpus"][i], "block": None, "queue": [], "active": None}
             for i in range(policy["worker_count"])]
    began = time.perf_counter()
    session = {"session_index": len(manifest["sessions"]), "started_utc": utc(),
               "observed_runtime": runtime,
               "runtime_sha256": digest(runtime_signature(runtime)),
               "timing_stratum": digest({"runtime": runtime_signature(runtime),
                                          "workers": policy["worker_count"],
                                          "threads": policy["thread_environment"]}),
               "elapsed_before_session_s": session_elapsed_s, "timing_mode":
               "concurrent_trials" if policy["worker_count"] > 1 else "serial_trials"}
    manifest["sessions"].append(session)
    manifest["status"] = "RUNNING"
    atomic_json(target, manifest)
    pressure_since, stop_reason = None, None
    last_disk_check = -math.inf

    def checkpoint_concurrency():
        active = [lane["active"] for lane in lanes if lane["active"]]
        ids = sorted(a["entry"]["trial_id"] for a in active)
        event = {"session_index": session["session_index"], "wall_s": time.perf_counter() - began,
                 "utc": utc(), "concurrency": len(ids), "trial_ids": ids}
        manifest["concurrency_timeline"].append(event)
        for item in active:
            item["entry"]["concurrency_events"].append(event)
            item["entry"]["co_runners"] = sorted(set(item["entry"]["co_runners"]).union(
                trial_id for trial_id in ids if trial_id != item["entry"]["trial_id"]))
            item["entry"]["max_concurrency"] = max(item["entry"]["max_concurrency"], len(ids))
        atomic_json(target, manifest)

    try:
        while pending or any(lane["queue"] or lane["active"] for lane in lanes):
            now = time.perf_counter()
            elapsed = session_elapsed_s + now - began
            if plan.get("protocol_id") == "a3t_old_safe_sem_kaggle_v4" and now-last_disk_check >= 1.:
                last_disk_check = now
                if shutil.disk_usage(output).free < 2*GIB:
                    manifest["resource_pressure_reason"] = "DISK_RESERVE_BELOW_2_GIB"
                    stop_reason = "RESOURCE_PRESSURE"
            ratio = memory_pressure(runtime)
            if ratio > .9:
                pressure_since = now if pressure_since is None else pressure_since
                if now - pressure_since >= PRESSURE_S:
                    stop_reason = "RESOURCE_PRESSURE"
            else:
                pressure_since = None
            if elapsed >= 32400:
                stop_reason = "SESSION_LIMIT"
            for lane in lanes:
                active = lane["active"]
                if active is None:
                    continue
                failure = stop_reason
                try:
                    _phase_update(active, now)
                    rss, affinity_failure = _sample_with_affinity_guard(active)
                    failure = affinity_failure or failure
                    phase = active["phase"]
                    limit = (policy["setup_watchdog_s"] if phase == "SETUP" else
                             active["entry"]["request"]["planning_budget_s"] + policy["solve_grace_s"]
                             if phase == "SOLVING" else policy["serialization_watchdog_s"])
                    if rss > TRIAL_RSS_LIMIT:
                        failure = "MEMORY_LIMIT"
                    elif now - active["phase_began"] > limit:
                        failure = phase + "_WATCHDOG"
                except AffinityProtocolError as exc:
                    active["entry"]["protocol_error"] = str(exc)
                    failure = "AFFINITY_PROTOCOL_ERROR"
                except (ValueError, OSError, IntegrityError) as exc:
                    active["entry"]["protocol_error"] = str(exc)
                    failure = "PHASE_PROTOCOL_ERROR"
                if failure or active["proc"].poll() is not None:
                    try:
                        status = _complete(active, failure)
                    except (ValueError, OSError, IntegrityError) as exc:
                        active["entry"]["protocol_error"] = str(exc)
                        status = _complete(active, "MALFORMED_WORKER_RESULT")
                    lane["active"] = None
                    if status == "FAILED_CORRECTNESS":
                        stop_reason = "FAILED_CORRECTNESS"
                    checkpoint_concurrency()
            if stop_reason:
                for lane in lanes:
                    if lane["active"]:
                        _complete(lane["active"], stop_reason)
                        lane["active"] = None
                checkpoint_concurrency()
                break
            # Finish already assigned blocks after soft stop; do not open new ones.
            for lane in lanes:
                if lane["active"]:
                    continue
                if not lane["queue"]:
                    lane["block"] = None
                    if pending and elapsed < soft_stop_s:
                        lane["block"] = pending.popleft()
                        lane["queue"] = groups[lane["block"]]
                if not lane["queue"] or ratio > .9:
                    continue
                entry = lane["queue"][0]
                reserved = sum(policy["reservations_bytes"][other["active"]["entry"]["backend"]]
                               for other in lanes if other["active"])
                needed = policy["reservations_bytes"][entry["backend"]]
                if reserved + needed > .8 * runtime["memory_limit_bytes"]:
                    if not any(other["active"] for other in lanes) and needed > .8 * runtime["memory_limit_bytes"]:
                        stop_reason = "INSUFFICIENT_RAM_RESERVATION"
                    continue
                directory = output / entry["trial_id"]
                directory.mkdir(exist_ok=False)
                request = copy.deepcopy(entry["request"])
                request["cloud_phase_handshake"] = True
                atomic_json(directory / "request.json", request)
                argv = registry[entry["backend"]]["argv"] + [str(directory / "request.json"), str(directory)]
                command = [sys.executable, str(LAUNCHER_PATH),
                           "--core", str(lane["core"]), "--record", str(directory / "launch.json"), "--"] + argv
                started.add(entry["trial_id"])
                entry.update(execution_status="START_REQUESTED", lane=lane["lane"], core=lane["core"],
                             session_index=session["session_index"], started_utc=utc(),
                             runtime_sha256=session["runtime_sha256"], timing_stratum=session["timing_stratum"],
                             queue_wait_s=time.perf_counter() - began, command=command,
                             reservation_bytes=needed, co_runners=[], concurrency_events=[], max_concurrency=0,
                             timing_mode=session["timing_mode"], success=False, independent_valid=None)
                _append_journal(journal, {"trial_id": entry["trial_id"], "utc": entry["started_utc"],
                                         "identity_sha256": digest(identity), "request_sha256": digest(request)})
                atomic_json(target, manifest)
                streams = [(directory / name).open("wb") for name in ("stdout.log", "stderr.log")]
                env = os.environ.copy()
                env.update({key: "1" for key in THREAD_ENV})
                env.update(OMP_DYNAMIC="FALSE", MKL_DYNAMIC="FALSE")
                launch_time = time.perf_counter()
                try:
                    process = subprocess.Popen(command, cwd=registry[entry["backend"]].get("cwd"),
                                               env=env, stdout=streams[0], stderr=streams[1],
                                               start_new_session=os.name != "nt")
                except OSError as exc:
                    for stream in streams:
                        stream.close()
                    entry.update(execution_status="CRASH", termination_reason="SPAWN_FAILED", error=str(exc))
                    atomic_json(directory / "execution.json", entry)
                    lane["queue"].pop(0)
                    atomic_json(target, manifest)
                    continue
                try:
                    created = psutil.Process(process.pid).create_time()
                except psutil.Error:
                    created = None
                entry.update(execution_status="RUNNING", pid=process.pid, pid_created=created)
                entry["affinity_guard"] = {
                    "protocol": AFFINITY_GUARD_PROTOCOL, "prelaunch_samples": 0, "postlaunch_samples": 0,
                    "scheduler_sha256": effective_scheduler_sha256,
                    "launcher_sha256": identity["artifact_sha256"][str(Path(LAUNCHER_PATH).resolve())]}
                lane["active"] = {"entry": entry, "directory": directory, "proc": process, "streams": streams,
                                  "began": launch_time, "phase": "SETUP", "phase_began": launch_time,
                                  "cpu_by_pid": {}, "peak_rss": 0, "max_threads": 0, "affinities": {},
                                  "phase_history": [{"phase": "SETUP", "observed_wall_s": 0}]}
                lane["queue"].pop(0)
                checkpoint_concurrency()
            if stop_reason:
                break
            if elapsed >= soft_stop_s and not any(lane["active"] or lane["queue"] for lane in lanes):
                break
            time.sleep(POLL_S)
    except KeyboardInterrupt:
        stop_reason = "INTERRUPTED"
    except BaseException:
        stop_reason = "CONTROLLER_ERROR"
        raise
    finally:
        for lane in lanes:
            if lane["active"]:
                _complete(lane["active"], stop_reason or "INTERRUPTED")
                lane["active"] = None
        checkpoint_concurrency()
        session.update(completed_utc=utc(), wall_s=time.perf_counter() - began,
                       stop_reason=stop_reason or "SOFT_STOP" if pending else stop_reason or "FINISHED")
        remaining = sum(e["execution_status"] == "PLANNED" for e in manifest["entries"])
        manifest.update(status="FAILED_CORRECTNESS" if stop_reason in ("FAILED_CORRECTNESS", "CONTROLLER_ERROR")
                        else "PARTIAL" if remaining or stop_reason else "COLLECTED_VALIDATION_PENDING",
                        counts=dict(collections.Counter(e["execution_status"] for e in manifest["entries"])))
        atomic_json(target, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("registry", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--reservations", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-local", action="store_true")
    parser.add_argument("--session-elapsed-s", type=float, default=0)
    parser.add_argument("--soft-stop-s", type=float, default=28800)
    args = parser.parse_args()
    result = run_campaign(args.manifest, read_json(args.registry), args.output,
                          max_workers=args.max_workers,
                          reservations=read_json(args.reservations) if args.reservations else None,
                          resume=args.resume, force_local=args.force_local,
                          session_elapsed_s=args.session_elapsed_s, soft_stop_s=args.soft_stop_s)
    print(json.dumps({"status": result["status"], "counts": result["counts"]}))
    raise SystemExit(0 if result["status"] == "COLLECTED_VALIDATION_PENDING" else 2)


if __name__ == "__main__":
    main()
