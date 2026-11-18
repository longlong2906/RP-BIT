"""Pin a trial before importing its backend; no numerical libraries imported here."""
from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import sys
import subprocess

import psutil

THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
              "BLIS_NUM_THREADS", "OMP_THREAD_LIMIT")


def pin_cpu(core: int) -> list[int]:
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {core})
        actual = sorted(os.sched_getaffinity(0))
    elif os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.SetProcessAffinityMask.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
        if not kernel.SetProcessAffinityMask(kernel.GetCurrentProcess(), 1 << core):
            raise ctypes.WinError(ctypes.get_last_error())
        process_mask, system_mask = ctypes.c_size_t(), ctypes.c_size_t()
        kernel.GetProcessAffinityMask.argtypes = (
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t))
        if not kernel.GetProcessAffinityMask(kernel.GetCurrentProcess(),
                                            ctypes.byref(process_mask), ctypes.byref(system_mask)):
            raise ctypes.WinError(ctypes.get_last_error())
        actual = [i for i in range(ctypes.sizeof(process_mask) * 8)
                  if process_mask.value & (1 << i)]
    else:
        raise RuntimeError("CPU affinity is unavailable on this platform")
    if actual != [core]:
        raise RuntimeError(f"Affinity mismatch: expected {[core]}, observed {actual}")
    return actual


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=int, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("worker command is required")
    for key in THREAD_ENV:
        os.environ[key] = "1"
    os.environ["OMP_DYNAMIC"] = "FALSE"
    os.environ["MKL_DYNAMIC"] = "FALSE"
    # The SGP adapter sets torch intra/inter-op threads explicitly after import.
    os.environ["TERRAIN_COMPARE_TORCH_THREADS"] = "1"
    actual = pin_cpu(args.core)
    record = {"pid": os.getpid(), "process_created": psutil.Process().create_time(), "cpu_affinity": actual,
              "thread_environment": {k: os.environ[k] for k in THREAD_ENV},
              "affinity_before_worker_import": True, "command": command}
    temp = args.record.with_suffix(".tmp")
    temp.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    temp.replace(args.record)
    if os.name == "nt":
        # Windows has no POSIX exec replacement. A waiting wrapper also keeps
        # the child in the controller's process tree; affinity is inherited.
        raise SystemExit(subprocess.call(command, env=os.environ))
    os.execvpe(command[0], command, os.environ)


if __name__ == "__main__":
    main()
