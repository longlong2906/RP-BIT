"""ctypes boundary to the canonical C++ terrain checker and PUTN adapter.

This is a transport layer, not an independent verifier. All native handles are
owned by the caller process; no map/model is reused between benchmark trials.
"""
from __future__ import annotations

import ctypes as C
import math
import os
from pathlib import Path


class BridgeError(RuntimeError):
    pass


class TerrainBridge:
    CALLBACK = C.CFUNCTYPE(None, C.c_double, C.POINTER(C.c_double),
                          C.POINTER(C.c_double), C.c_size_t, C.c_double, C.c_void_p)

    def __init__(self, library, map_file, library_search_paths=(), *, risk_enabled=True):
        self.library_path = str(Path(library).resolve())
        self._dll_dirs = []
        if os.name == "nt" and hasattr(os, "add_dll_directory"):
            for directory in (str(Path(library).resolve().parent), *library_search_paths):
                self._dll_dirs.append(os.add_dll_directory(str(Path(directory).resolve())))
        self.lib = C.CDLL(self.library_path)
        self._declare()
        if type(risk_enabled) is not bool:
            raise ValueError("risk_enabled must be bool")
        self.risk_enabled = risk_enabled
        self.handle = self.lib.tc_create(os.fsencode(map_file), int(risk_enabled))
        if not self.handle:
            raise BridgeError(self.error())

    def _declare(self):
        lib = self.lib
        d = C.c_double
        p = C.POINTER(d)
        signatures = {
            "tc_create": ([C.c_char_p, C.c_int], C.c_void_p),
            "tc_destroy": ([C.c_void_p], None),
            "tc_error": ([], C.c_char_p),
            "tc_valid_motion": ([C.c_void_p, d, d, d, d], C.c_int),
            "tc_edge_cost": ([C.c_void_p, d, d, d, d, p], C.c_int),
            "tc_height": ([C.c_void_p, d, d, p], C.c_int),
            "tc_bounds": ([C.c_void_p, p], C.c_int),
            "tc_timings": ([C.c_void_p, p], C.c_int),
        }
        for name, (args, result) in signatures.items():
            fn = getattr(lib, name)
            fn.argtypes, fn.restype = args, result

    def error(self):
        value = self.lib.tc_error()
        return value.decode("utf-8", errors="replace") if value else "C ABI error"

    def close(self):
        if getattr(self, "handle", None):
            self.lib.tc_destroy(self.handle)
            self.handle = None
        for directory in self._dll_dirs:
            directory.close()
        self._dll_dirs = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    @staticmethod
    def _point(q):
        if len(q) != 2 or not all(math.isfinite(float(x)) for x in q):
            raise ValueError("state must contain two finite XY coordinates")
        return tuple(float(x) for x in q)

    def valid_motion(self, a, b):
        value = self.lib.tc_valid_motion(self.handle, *self._point(a), *self._point(b))
        if value < 0:
            raise BridgeError(self.error())
        return bool(value)

    def edge_cost(self, a, b):
        values = (C.c_double * 5)()
        result = self.lib.tc_edge_cost(self.handle, *self._point(a), *self._point(b), values)
        if result < 0:
            raise BridgeError(self.error())
        if not result:
            return None
        out = tuple(values)
        if not all(math.isfinite(x) and x >= 0 for x in out):
            raise BridgeError("invalid canonical edge-cost tuple")
        return out

    def height(self, q):
        out = C.c_double()
        if self.lib.tc_height(self.handle, *self._point(q), C.byref(out)) != 1:
            raise BridgeError("height unavailable: " + self.error())
        return out.value

    @property
    def bounds(self):
        out = (C.c_double * 4)()
        if self.lib.tc_bounds(self.handle, out) != 1:
            raise BridgeError(self.error())
        return tuple(out)

    @property
    def timings(self):
        out = (C.c_double * 3)()
        if self.lib.tc_timings(self.handle, out) != 1:
            raise BridgeError(self.error())
        return dict(zip(("load_s", "risk_preprocessing_s", "collision_index_s"), out))

    def evaluate_path(self, path):
        path = [self._point(p) for p in path]
        if not path or not self.valid_motion(path[0], path[0]):
            raise BridgeError("invalid empty path/start")
        sums = [0.] * 5
        xy = 0.
        for a, b in zip(path, path[1:]):
            if not self.valid_motion(a, b):
                raise BridgeError("adapter supplied hard-invalid incumbent")
            edge = self.edge_cost(a, b)
            if edge is None:
                raise BridgeError("missing edge objective for valid incumbent")
            sums = [x + y for x, y in zip(sums, edge)]
            xy += math.dist(a, b)
        return dict(zip(("cost_terrain_m", "risk_slope_integral_m",
                         "risk_clearance_integral_m", "risk_integral_m",
                         "evaluation_objective_value"), sums), length_xy_m=xy)


class PutnBridge:
    """Separate model construction and solve so cold start stays measurable."""

    def __init__(self, terrain, *, paper=False, points=None):
        self.terrain = terrain
        lib, d = terrain.lib, C.c_double
        lib.tc_putn_create.argtypes, lib.tc_putn_create.restype = [C.c_void_p], C.c_void_p
        lib.tc_putn_destroy.argtypes, lib.tc_putn_destroy.restype = [C.c_void_p], None
        lib.tc_putn_solve.argtypes = [C.c_void_p, d, d, d, d, C.c_uint64,
                                    d, C.c_int, C.c_uint64, terrain.CALLBACK, C.c_void_p]
        lib.tc_putn_solve.restype = C.c_int
        lib.tc_putn_stats.argtypes = [C.c_void_p, C.POINTER(d)]
        lib.tc_putn_stats.restype = C.c_int
        self.paper = bool(paper)
        self.metadata = {"version": "putn_paper_first_static_v2" if paper else "putn_pf_rrtstar_static_v1",
                         "paperpolicy": bool(paper)}
        if paper:
            import numpy as np
            cloud = np.ascontiguousarray(points, dtype=np.float64)
            if cloud.ndim != 2 or cloud.shape[1] != 3 or not len(cloud) or not np.isfinite(cloud).all():
                raise ValueError("paper PUTN needs finite N by 3 master cloud")
            lib.tc_putn_create_paper.argtypes = [C.c_void_p, C.POINTER(d), C.c_size_t]
            lib.tc_putn_create_paper.restype = C.c_void_p
            lib.tc_putn_fit_plane.argtypes = [C.c_void_p, d, d, C.POINTER(d)]
            lib.tc_putn_fit_plane.restype = C.c_int
            lib.tc_putn_paper_stats.argtypes = [C.c_void_p, C.POINTER(d)]
            lib.tc_putn_paper_stats.restype = C.c_int
            lib.tc_putn_snapshot_features.argtypes = [C.c_void_p, C.POINTER(d), C.c_size_t]
            lib.tc_putn_snapshot_features.restype = C.c_int
            self.handle = lib.tc_putn_create_paper(terrain.handle, cloud.ctypes.data_as(C.POINTER(d)), len(cloud))
            self.metadata.update(master_cloud_observations=len(cloud), voxel_resolution_m=.1,
                fit_cube_side_m=1., com_height_m=.7, step_size_m=.4, neighbor_radius_m=1.,
                goal_threshold_m=1., goal_bias=.15, native_cost="paper_eq1_com_traversability_minus2",
                native_cost_penalty=.1, tau_singular_policy="reject_tau_ge_1",
                tau_effective_coefficients=dict(slope_deg=.1, flatness_fourth_moment=1000., sparsity=1.),
                flatness_denominator="N", slope_policy="acos_upward_normal_paper_typo_correction",
                sparsity_covariance="population_2x2_vacant_xy_voxel_indices", sparsity_trace_threshold=1/.1152,
                vacancy_ratio_min=.02, vacancy_ratio_max=.05, nearest_metric="euclidean_xy",
                informed_sampling="uniform_xy_rejection_by_sum_com_distance_lt_native_best",
                informed_bound_roundoff_guard="1e-12*(1+native_best)",
                projection="canonical_exact_H_support_xy", body_samples=21, body_grid=[5, 5])
        else:
            if points is not None:
                raise ValueError("master cloud is only supported by paper policy")
            self.handle = lib.tc_putn_create(terrain.handle)
        if not self.handle:
            raise BridgeError(terrain.error())

    def close(self):
        if self.handle:
            self.terrain.lib.tc_putn_destroy(self.handle)
            self.handle = None

    def solve(self, start, goal, seed, budget_s, own_cost, fixed_work, callback):
        errors = []
        def received(elapsed, xy, xyz, count, cost, _):
            try:
                callback(float(elapsed), [[xy[2*i], xy[2*i+1]] for i in range(count)],
                         [[xyz[3*i], xyz[3*i+1], xyz[3*i+2]] for i in range(count)],
                         float(cost))
            except Exception as exc:
                errors.append(exc)
        c_callback = self.terrain.CALLBACK(received)
        value = self.terrain.lib.tc_putn_solve(self.handle, *start, *goal,
            seed, budget_s, int(own_cost), fixed_work, c_callback, None)
        if errors:
            raise errors[0]
        if value == -1:
            raise BridgeError(self.terrain.error())
        return value

    @property
    def statistics(self):
        out = (C.c_double * 12)()
        if self.terrain.lib.tc_putn_stats(self.handle, out) != 1:
            raise BridgeError(self.terrain.error())
        return list(out)

    @property
    def paper_statistics(self):
        if not self.paper:
            return {}
        out = (C.c_double * 2)()
        if self.terrain.lib.tc_putn_paper_stats(self.handle, out) != 1:
            raise BridgeError(self.terrain.error())
        return dict(informed_proposals=int(out[0]), informed_rejections=int(out[1]))

    def path_features(self, path):
        """Serialization-only evidence; calls fitPlane and thus increments its counters."""
        if not self.paper:
            raise BridgeError("path features require paper policy")
        result = []
        for q in path:
            q = self.terrain._point(q)
            out = (C.c_double * 12)()
            if self.terrain.lib.tc_putn_fit_plane(self.handle, *q, out) != 1:
                raise BridgeError(self.terrain.error())
            if not out[11]:
                raise BridgeError("paper path vertex lacks a traversable native plane")
            result.append(dict(support_xy=list(q), center_of_mass=list(out[:3]), normal=list(out[3:6]),
                traversability=out[6], flatness=out[7], slope_deg=out[8], sparsity=out[9], point_count=int(out[10])))
        return result
