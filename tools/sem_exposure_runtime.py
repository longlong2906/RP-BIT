"""Local runtime admission and executable identity for SEM-exposure sessions.

No limits, affinity, global environment, cache or external resources are changed.
The pilot controller uses these checks before dispatch; legacy smoke admission
remains unchanged. An interpreter launch path must preserve virtualenv symlinks.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import shutil

from corridor_v4_execution import utc_now

VERSION = "sem_exposure_runtime_admission_v1"


def executable_path(value):
    """Absolute lexical path, deliberately not realpath/Path.resolve()."""
    return Path(os.path.abspath(os.path.expanduser(os.fspath(value))))


def runtime_admission(workers, reservation_bytes, *, required_disk_bytes=0, disk_path=None,
                      cgroup_root=Path("/sys/fs/cgroup"), membership=Path("/proc/self/cgroup")):
    """Require whole CPU capacity, safe RAM headroom and requested free disk.

    Cgroup ancestor quotas and cpusets constrain available CPU capacity. RAM
    uses the existing independently tested clean-inactive-file admission, and
    reserves at most 80% of the minimum live host/cgroup headroom. No floor-to-
    one conversion lets a fractional CPU quota admit one complete worker.
    """
    import psutil
    from native_paper_v3_memory import admission_snapshot
    from terrain_compare_scheduler import detect_runtime

    if type(workers) is not int or workers not in (1, 4):
        raise ValueError("workers must be one or four")
    if type(reservation_bytes) is not int or reservation_bytes <= 0:
        raise ValueError("positive per-worker memory reservation required")
    if type(required_disk_bytes) is not int or required_disk_bytes < 0:
        raise ValueError("nonnegative disk reservation required")
    if required_disk_bytes and disk_path is None:
        raise ValueError("disk_path required for disk admission")
    result = dict(policy_version=VERSION, requested_workers=workers, observed_utc=utc_now(),
                  reserved_bytes_per_worker=reservation_bytes,
                  required_memory_bytes=workers*reservation_bytes,
                  required_disk_bytes=required_disk_bytes, retained_admission_fraction=.8,
                  admitted=False, reason=None, cores=[])
    try:
        runtime = detect_runtime(cgroup_root, membership)
        quota = runtime.get("cpu_quota")
        if quota is not None and (not isinstance(quota, (int, float)) or not math.isfinite(quota) or quota <= 0):
            raise ValueError("invalid runtime CPU quota")
        capacity = min(len(runtime["eligible_cpus"]), math.floor(quota) if quota is not None else len(runtime["eligible_cpus"]))
        result.update(runtime=runtime, eligible_cpus=runtime["eligible_cpus"],
                      cpu_quota=quota, effective_cpu_capacity=capacity)
        if capacity < workers:
            result["reason"] = "INSUFFICIENT_CPU_QUOTA_OR_AFFINITY"
            return result
        def host_sample():
            memory = psutil.virtual_memory()
            return dict(total=int(memory.total), available=int(memory.available))
        memory = admission_snapshot(runtime, host_sample)
        headroom = int(memory["absolute_headroom_bytes"])
        result.update(memory_evidence=memory, available_memory_bytes=headroom,
                      total_memory_bytes=int(runtime["memory_limit_bytes"]))
        if workers*reservation_bytes > headroom*.8:
            result["reason"] = "INSUFFICIENT_AVAILABLE_RAM"
            return result
        if disk_path is not None:
            path = executable_path(disk_path)
            while not path.exists() and path != path.parent:
                path = path.parent
            disk = shutil.disk_usage(path)
            result.update(disk_probe_path=str(path), available_disk_bytes=int(disk.free),
                          total_disk_bytes=int(disk.total))
            if required_disk_bytes > disk.free:
                result["reason"] = "INSUFFICIENT_FREE_DISK"
                return result
        result.update(admitted=True, cores=runtime["eligible_cpus"][:workers])
    except (OSError, ValueError, RuntimeError, KeyError, ZeroDivisionError) as error:
        result.update(reason="UNREADABLE_OR_INVALID_RUNTIME_LIMITS", error=str(error))
    return result
