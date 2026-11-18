"""ctypes wrapper of the C++ SEM-RRT* port (library sem_rrtstar_c).

`SEMNativePlannerCpp` is a drop-in replacement for `sem_native_planner.SEMNativePlanner`:
same solve() signature, counters, best_path, best_cost and last_components. The native
raster is still the Python/scipy `SEMNativeModel` (preprocessing); only the search runs in
C++. With the same raster, seed and work budget it reproduces the Python search bit for bit.
"""
from __future__ import annotations

import ctypes as C
import math
import os
from pathlib import Path
import time

import numpy as np

from sem_native_planner import _components, _point

COUNTERS = ("iterations", "sample_proposals", "sampler_failures", "tree_insertions", "edge_evaluations",
            "hard_rejections", "native_rejections", "parent_candidates", "rewires", "descendant_cost_updates",
            "start_connection_checks", "incumbents", "late_candidates")
STATUS = {0: "EXACT_SOLUTION", 1: "TIMEOUT", 2: "INVALID_HARD_ENDPOINTS", 3: "INVALID_NATIVE_ENDPOINTS"}
CALLBACK = C.CFUNCTYPE(C.c_int, C.c_double, C.POINTER(C.c_double), C.POINTER(C.c_double), C.c_size_t,
                       C.c_double, C.POINTER(C.c_double), C.c_void_p)


class SEMCppError(RuntimeError):
    pass


def library_beside(bridge):
    """The port is built next to the terrain bridge (same build directory and runtime)."""
    bridge = Path(bridge)
    return bridge.with_name(bridge.name.replace("terrain_compare_c", "sem_rrtstar_c"))


def load_library(library, search_paths=()):
    directories = []
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        for folder in (Path(library).resolve().parent, *map(Path, search_paths)):
            if folder.is_dir():
                directories.append(os.add_dll_directory(str(folder.resolve())))
    lib = C.CDLL(str(Path(library).resolve()))
    d, p, u64 = C.c_double, C.POINTER(C.c_double), C.c_uint64
    for name, args, result in (
            ("sem_error", [], C.c_char_p), ("sem_version", [], C.c_char_p),
            ("sem_create", [C.c_char_p, p, C.c_int, C.c_int, d, C.POINTER(C.c_ubyte), p], C.c_void_p),
            ("sem_destroy", [C.c_void_p], None),
            ("sem_solve", [C.c_void_p, d, d, d, d, u64, d, u64, d, CALLBACK, C.c_void_p], C.c_int),
            ("sem_counters", [C.c_void_p, C.POINTER(u64)], C.c_int),
            ("sem_set_exhaustive_check", [C.c_void_p, C.c_int], C.c_int),
            ("sem_set_convergence_stop", [C.c_void_p, d, u64], C.c_int),
            ("sem_convergence", [C.c_void_p, p], C.c_int),
            ("sem_model_edge", [C.c_void_p, d, d, d, d, p], C.c_int),
            ("sem_model_valid_state", [C.c_void_p, d, d], C.c_int),
            ("sem_debug_random", [u64, C.c_size_t, p], C.c_int),
            ("sem_debug_uniform", [u64, d, d, C.c_size_t, p], C.c_int),
            ("sem_debug_hypot", [d, d], d), ("sem_debug_dist", [d, d, d, d], d),
            ("sem_debug_fsum", [p, C.c_size_t], d), ("sem_debug_turn", [d] * 6, d),
            ("sem_debug_libm", [d, d, d, p], C.c_int)):
        function = getattr(lib, name)
        function.argtypes, function.restype = args, result
    return lib, directories


class SEMNativeCppEngine:
    """C++ terrain plus a copy of the native raster; build it during preprocessing."""

    def __init__(self, library, map_path, model, search_paths=()):
        self.library_path = str(Path(library).resolve())
        self.lib, self._directories = load_library(library, search_paths)
        valid = np.ascontiguousarray(model.valid, dtype=np.uint8)
        risk = np.ascontiguousarray(model.risk, dtype=np.float64)
        if valid.shape != (model.ny, model.nx) or risk.shape != valid.shape:
            raise ValueError("native raster shape differs from the model grid")
        bounds = (C.c_double * 4)(*map(float, model.bounds))
        self.handle = self.lib.sem_create(os.fsencode(str(map_path)), bounds, int(model.nx), int(model.ny),
                                          float(model.resolution), valid.ctypes.data_as(C.POINTER(C.c_ubyte)),
                                          risk.ctypes.data_as(C.POINTER(C.c_double)))
        if not self.handle:
            raise SEMCppError(self.error())
        self.version = self.lib.sem_version().decode()

    def error(self):
        value = self.lib.sem_error()
        return value.decode("utf-8", errors="replace") if value else "C ABI error"

    def close(self):
        if getattr(self, "handle", None):
            self.lib.sem_destroy(self.handle)
            self.handle = None
        for directory in getattr(self, "_directories", []):
            directory.close()
        self._directories = []

    def model_edge(self, a, b):
        out = (C.c_double * 2)()
        value = self.lib.sem_model_edge(self.handle, *_point(a), *_point(b), out)
        if value < 0:
            raise SEMCppError(self.error())
        return (out[0], out[1]) if value else None

    def valid_state(self, q):
        value = self.lib.sem_model_valid_state(self.handle, *_point(q))
        if value < 0:
            raise SEMCppError(self.error())
        return bool(value)

    def set_exhaustive_check(self, enabled):
        if self.lib.sem_set_exhaustive_check(self.handle, int(bool(enabled))) != 1:
            raise SEMCppError(self.error())


def check_convergence_stop(stop):
    """None (off) or {"relative_eps": 0 < float < 1, "window_iterations": int > 0}."""
    if stop is None:
        return
    eps, window = (stop.get("relative_eps"), stop.get("window_iterations")) if isinstance(stop, dict) else (None, None)
    if (not isinstance(stop, dict) or set(stop) != {"relative_eps", "window_iterations"} or type(eps) is not float
            or not math.isfinite(eps) or not 0 < eps < 1 or type(window) is not int or not 0 < window < 2**63):
        raise ValueError("unregistered SEM convergence stop")


class SEMNativePlannerCpp:
    """Same interface and outputs as SEMNativePlanner; the search runs in C++.

    `convergence_stop` (optional, C++ only) stops the search once the published cost has not dropped
    by relative_eps over window_iterations iterations; `convergence` then reports the outcome."""

    def __init__(self, engine, convergence_stop=None):
        check_convergence_stop(convergence_stop)
        self.engine = engine
        self.convergence_stop = convergence_stop
        self.best_path, self.best_cost, self.last_components = None, math.inf, None
        self.counters = dict.fromkeys(COUNTERS, 0)
        self.nodes = 0
        self.convergence = None

    def solve(self, start, goal, seed, budget, callback, work_iterations=0, *, algorithm_started_at=None):
        start, goal = _point(start), _point(goal)
        if not math.isfinite(budget) or budget <= 0 or type(work_iterations) is not int or work_iterations < 0:
            raise ValueError("positive finite budget and nonnegative work_iterations required")
        if type(seed) is not int or not 0 <= seed < 2**64:
            raise ValueError("seed must be uint64")
        now = time.perf_counter()
        if algorithm_started_at is not None and (not math.isfinite(algorithm_started_at) or algorithm_started_at > now):
            raise ValueError("invalid algorithm clock origin")
        began = now if algorithm_started_at is None else algorithm_started_at
        self.best_path, self.best_cost, self.last_components = None, math.inf, None
        self.convergence = None
        errors = []

        def received(_, xy, heights, count, cost, totals, __):
            try:
                # Report time on the caller's clock, as SEMNativePlanner does. The C++ clock (which enforces
                # the budget) starts after the ctypes call and lags it, so its reading could precede a time
                # the caller stamped after the previous incumbent.
                elapsed = time.perf_counter() - began
                path = [(xy[2 * i], xy[2 * i + 1]) for i in range(count)]
                raw = [[x, y, heights[i]] for i, (x, y) in enumerate(path)]
                self.best_path, self.best_cost = tuple(path), cost
                self.last_components = _components((totals[0], totals[1], totals[2]))
                callback(elapsed, [list(point) for point in path], raw, cost)
                return 0
            except BaseException as error:  # never unwind through C++; re-raised below
                errors.append(error)
                return 1

        c_callback = CALLBACK(received)
        stop = self.convergence_stop or dict(relative_eps=0., window_iterations=0)
        if self.engine.lib.sem_set_convergence_stop(self.engine.handle, stop["relative_eps"], stop["window_iterations"]) != 1:
            raise SEMCppError(self.engine.error())
        already = time.perf_counter() - began
        code = self.engine.lib.sem_solve(self.engine.handle, *start, *goal, seed, float(budget), work_iterations,
                                         already, c_callback, None)
        out = (C.c_uint64 * 14)()
        if self.engine.lib.sem_counters(self.engine.handle, out) != 1:
            raise SEMCppError(self.engine.error())
        self.counters = {name: int(value) for name, value in zip(COUNTERS, out)}
        self.nodes = int(out[13])
        if self.convergence_stop:
            reference = (C.c_double * 2)()
            converged = self.engine.lib.sem_convergence(self.engine.handle, reference)
            if converged not in (0, 1):
                raise SEMCppError(self.engine.error())
            self.convergence = dict(converged=bool(converged),
                                    reference_cost=reference[0] if math.isfinite(reference[0]) else None,
                                    reference_iteration=int(reference[1]))
        if errors:
            raise errors[0]
        if code == -1:
            raise SEMCppError(self.engine.error())
        if code not in STATUS:
            raise SEMCppError(f"unexpected C++ SEM status {code}")
        return STATUS[code]

    def close(self):
        self.engine.close()
