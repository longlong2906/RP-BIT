"""Paper-based static SEM-RRT* reconstruction, not an official implementation.

SEM equations 1--5 and 7--9 are evaluated under explicitly recorded numerical
conventions. The search keeps XY states, a goal-rooted tree and a context-dependent
turn cost. This single-parent tree does not imply RRT* asymptotic optimality for
the three-point objective. Common hard geometry is supplied by TerrainBridge.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

RESOLUTION = .1
WEIGHTS = (.48, .16, .36)
TURN_THRESHOLD = math.pi / 6
RISK_THRESHOLD = 40.
MODEL_VERSION = "sem_native_paper_static_v2"


def _point(value):
    if len(value) != 2 or not all(math.isfinite(float(v)) for v in value):
        raise ValueError("expected two finite XY coordinates")
    return tuple(float(v) for v in value)


def _components(values):
    length, turn, risk = map(float, values)
    return {"length_grid": length, "turn_excess_rad": turn,
            "risk_integral_grid": risk,
            "weighted_length": WEIGHTS[0] * length,
            "weighted_turn": WEIGHTS[1] * turn,
            "weighted_risk": WEIGHTS[2] * risk,
            "total": math.fsum(w * v for w, v in zip(WEIGHTS, values))}


def _turn(a, b, c):
    """Turn at b in a -> b -> c; duplicate endpoints contribute zero."""
    ux, uy, vx, vy = b[0]-a[0], b[1]-a[1], c[0]-b[0], c[1]-b[1]
    norm = math.hypot(ux, uy) * math.hypot(vx, vy)
    if norm == 0:
        return 0.
    return max(0., math.acos(max(-1., min(1., (ux*vx+uy*vy)/norm))) - TURN_THRESHOLD)


class SEMNativeModel:
    """Native SEM field used by SEM only; never a shared benchmark evaluator."""
    def __init__(self, points, bounds, output_dir=None):
        from scipy import ndimage
        began = time.perf_counter()
        self.bounds = tuple(float(v) for v in bounds)
        if len(self.bounds) != 4 or not all(map(math.isfinite, self.bounds)):
            raise ValueError("invalid bounds")
        xmin, ymin, xmax, ymax = self.bounds
        if xmax <= xmin or ymax <= ymin:
            raise ValueError("empty bounds")
        self.resolution = RESOLUTION
        self.nx = int(math.ceil((xmax-xmin)/RESOLUTION))
        self.ny = int(math.ceil((ymax-ymin)/RESOLUTION))
        cloud = np.asarray(points, dtype=np.float64)
        if cloud.ndim != 2 or cloud.shape[1] != 3 or len(cloud) == 0 or not np.isfinite(cloud).all():
            raise ValueError("points must be a nonempty finite Nx3 array")
        if np.any(cloud[:, :2] < (xmin, ymin)) or np.any(cloud[:, :2] > (xmax, ymax)):
            raise ValueError("cloud outside bounds")
        # The upper boundary belongs to the last cell. Interior cells are half-open.
        ij = np.floor((cloud[:, :2] - (xmin, ymin))/RESOLUTION).astype(np.int64)
        ij[:, 0] = np.minimum(ij[:, 0], self.nx-1)
        ij[:, 1] = np.minimum(ij[:, 1], self.ny-1)
        ids = ij[:, 1]*self.nx + ij[:, 0]
        size = self.nx*self.ny
        top = np.full(size, -np.inf)
        np.maximum.at(top, ids, cloud[:, 2])
        kept = cloud[:, 2] > top[ids] - RESOLUTION
        valid_ids, z = ids[kept], cloud[kept, 2]
        count = np.bincount(valid_ids, minlength=size)
        total = np.bincount(valid_ids, weights=z, minlength=size)
        observed = count > 0
        mean = np.divide(total, count, out=np.zeros(size), where=observed)
        squared = np.bincount(valid_ids, weights=(z-mean[valid_ids])**2, minlength=size)
        variance = np.divide(squared, count, out=np.zeros(size), where=observed)
        shape = (self.ny, self.nx)
        self.count = count.reshape(shape)
        self.surface = np.where(observed, top, np.nan).reshape(shape)
        self.mean = np.where(observed, mean, np.nan).reshape(shape)
        self.variance = np.where(observed, variance, np.nan).reshape(shape)
        self.observed = observed.reshape(shape)
        elevation = mean.reshape(shape)
        d1 = np.array([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])/4.
        d2 = d1.T
        d3 = np.array([[-2., -1., 0.], [-1., 0., 1.], [0., 1., 2.]])/3.
        d4 = np.rot90(d3)
        derivatives = [ndimage.correlate(elevation, d, mode="nearest") for d in (d1, d2, d3, d4)]
        s1 = np.arctan(np.hypot(*derivatives[:2])/(2*RESOLUTION))
        s2 = np.arctan(np.hypot(*derivatives[2:])/(2*math.sqrt(2)*RESOLUTION))
        slope = np.degrees(.5*s1 + .5*s2)
        valid = ndimage.minimum_filter(self.observed, size=3, mode="nearest")
        smooth = ndimage.gaussian_filter(slope, sigma=5/3, radius=5, mode="nearest")
        valid = ndimage.minimum_filter(valid, size=11, mode="nearest")
        slope_risk = ndimage.maximum_filter(smooth, size=11, mode="nearest")
        valid = ndimage.minimum_filter(valid, size=11, mode="nearest")
        self.valid = valid & self.observed
        self.slope_raw_deg = np.where(ndimage.minimum_filter(self.observed, size=3, mode="nearest"), slope, np.nan)
        self.slope_risk_deg = np.where(self.valid, slope_risk, np.nan)
        self.risk = np.where(self.valid, 200*self.variance + .75*slope_risk, np.nan)
        self.metadata = {
            "version": MODEL_VERSION, "official_implementation": False,
            "scope": "paper_based_static_reconstruction_with_common_hard_geometry",
            "resolution_m": RESOLUTION, "bounds": list(self.bounds), "shape_yx": list(shape),
            "input_points": len(cloud), "retained_points": int(kept.sum()),
            "observed_cells": int(self.observed.sum()), "valid_risk_cells": int(self.valid.sum()),
            "binning": "floor_relative_to_bounds; upper_boundary_last_cell; rows_increase_y",
            "surface_filter": "z > cell_max_z - resolution", "variance": "population",
            "empty_cells": "forbidden; invalid_support_propagates_through_all_filter_stencils",
            "slope_kernels": [d.tolist() for d in (d1, d2, d3, d4)],
            "slope_units": "degrees", "slope_weights": [.5, .5],
            "gaussian_radius_cells": 5, "gaussian_sigma_cells": 5/3,
            "maximum_filter_radius_cells": 5, "filter_boundary": "nearest",
            "risk_weights": {"variance": 200., "slope": .75}, "risk_threshold": RISK_THRESHOLD,
            "objective_weights": list(WEIGHTS), "turn_threshold_rad": TURN_THRESHOLD,
            "length_units": "XY_grid_units", "turn_cost": "max(0,turn_rad-pi/6)",
            "edge_risk": "Eq9: continuous_grid_distance/(max_abs_continuous_grid_delta+1)*sum_Bresenham_FR",
            "bresenham": "inclusive; lexicographically_canonical_cell_endpoints; symmetric",
            "native_feasibility": "every_Bresenham_cell_has_data_and_FR<=40",
            "rolling_optimization": False, "construction_wall_s": time.perf_counter()-began,
        }
        if output_dir is not None:
            self.save(output_dir)

    def save(self, output_dir):
        """Persist the actual native raster, including explicit validity masks."""
        folder = Path(output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder/"raster.npz"
        with path.open("xb") as stream:
            np.savez_compressed(stream, surface=self.surface, count=self.count, mean=self.mean,
                variance=self.variance, observed=self.observed, valid=self.valid,
                slope_raw_deg=self.slope_raw_deg, slope_risk_deg=self.slope_risk_deg,
                risk=self.risk, bounds=np.asarray(self.bounds), resolution=np.asarray(self.resolution))
        self.metadata["raster_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.metadata["raster_file"] = path.name
        self.metadata["raster_artifacts"] = {path.name: self.metadata["raster_sha256"]}
        with (folder/"raster.json").open("x", encoding="utf-8") as stream:
            json.dump(self.metadata, stream, indent=2, allow_nan=False)
            stream.write("\n")
        return path

    @classmethod
    def from_artifacts(cls, directory):
        """Load the measured native raster; never regenerate or fit its field."""
        folder = Path(directory)
        metadata = json.loads((folder/"raster.json").read_text(encoding="utf-8"))
        if metadata.get("version") != MODEL_VERSION or metadata.get("raster_file") != "raster.npz":
            raise ValueError("unsupported SEM raster artifact")
        path = folder/"raster.npz"
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        if checksum != metadata.get("raster_sha256") or metadata.get("raster_artifacts") != {"raster.npz": checksum}:
            raise ValueError("SEM raster hash mismatch")
        result = cls.__new__(cls)
        with np.load(path, allow_pickle=False) as archive:
            result.bounds = tuple(float(v) for v in archive["bounds"])
            result.resolution = float(archive["resolution"])
            for name in ("surface", "count", "mean", "variance", "observed", "valid",
                         "slope_raw_deg", "slope_risk_deg", "risk"):
                setattr(result, name, archive[name].copy())
        if result.risk.ndim != 2 or result.resolution != RESOLUTION:
            raise ValueError("invalid native raster geometry")
        result.ny, result.nx = result.risk.shape
        if (list(result.bounds) != metadata["bounds"] or list(result.risk.shape) != metadata["shape_yx"]
                or any(getattr(result, name).shape != result.risk.shape for name in
                       ("surface", "count", "mean", "variance", "observed", "valid", "slope_raw_deg", "slope_risk_deg"))):
            raise ValueError("native raster shape/metadata mismatch")
        if (result.valid.dtype != np.bool_ or result.observed.dtype != np.bool_
                or np.any(result.valid & ~result.observed)
                or np.any(~np.isfinite(result.risk[result.valid])) or np.any(result.risk[result.valid] < 0)):
            raise ValueError("invalid native raster values")
        result.metadata = metadata
        return result

    def index(self, point):
        x, y = _point(point)
        xmin, ymin, xmax, ymax = self.bounds
        if not xmin <= x <= xmax or not ymin <= y <= ymax:
            return None
        return min(self.nx-1, math.floor((x-xmin)/self.resolution)), min(self.ny-1, math.floor((y-ymin)/self.resolution))

    def valid_state(self, point):
        ij = self.index(point)
        return bool(ij is not None and self.valid[ij[1], ij[0]] and self.risk[ij[1], ij[0]] <= RISK_THRESHOLD)

    def _cells(self, a, b):
        first, last = self.index(a), self.index(b)
        if first is None or last is None:
            return None
        if last < first:
            first, last = last, first
        x, y = first
        ex, ey = last
        dx, dy = abs(ex-x), -abs(ey-y)
        sx, sy = (1 if x < ex else -1), (1 if y < ey else -1)
        error, cells = dx+dy, []
        while True:
            cells.append((x, y))
            if (x, y) == (ex, ey):
                return cells
            twice = 2*error
            if twice >= dy:
                error += dy
                x += sx
            if twice <= dx:
                error += dx
                y += sy

    def edge(self, a, b):
        """Return unweighted (length, risk), or None for native-invalid edge."""
        a, b = _point(a), _point(b)
        cells = self._cells(a, b)
        if cells is None:
            return None
        values = []
        for x, y in cells:
            if not self.valid[y, x] or not math.isfinite(self.risk[y, x]) or self.risk[y, x] > RISK_THRESHOLD:
                return None
            values.append(float(self.risk[y, x]))
        dx, dy = abs(b[0]-a[0])/self.resolution, abs(b[1]-a[1])/self.resolution
        length = math.hypot(dx, dy)
        return length, length/(max(dx, dy)+1)*math.fsum(values)

    def valid_motion(self, a, b):
        return self.edge(a, b) is not None


def accumulate_native_path_cost(path_xy, raster):
    """Independent cost arithmetic audit, reading the locked native raster.

    This deliberately does not reconstruct SEM from another planner's output or
    verify the native field construction. Raster/model checks belong to model
    tests. Its major-axis Bresenham and atan2 turn implementations are separate
    from the planner's error-step traversal and acos local-cost update.
    """
    close = None
    if isinstance(raster, (str, Path)):
        close = np.load(raster, allow_pickle=False)
        risk, valid, bounds, res = close["risk"], close["valid"], tuple(close["bounds"]), float(close["resolution"])
    else:
        risk, valid, bounds, res = raster.risk, raster.valid, raster.bounds, raster.resolution
    try:
        points = [_point(p) for p in path_xy]
        if not points:
            raise ValueError("empty native path")
        ny, nx = risk.shape
        def cell(p):
            if not bounds[0] <= p[0] <= bounds[2] or not bounds[1] <= p[1] <= bounds[3]:
                raise ValueError("native path outside raster")
            return min(nx-1, int((p[0]-bounds[0])/res)), min(ny-1, int((p[1]-bounds[1])/res))
        length_parts, risk_parts, angle_parts = [], [], []
        for a, b in zip(points, points[1:]):
            x0, y0 = cell(a)
            x1, y1 = cell(b)
            if (x1, y1) < (x0, y0):
                x0, y0, x1, y1 = x1, y1, x0, y0
            dx, dy = x1-x0, y1-y0
            # Canonical x is nondecreasing; ties in the minor axis round upward.
            if abs(dx) >= abs(dy):
                cells = [(x0+k, y0+(1 if dy >= 0 else -1)*((2*k*abs(dy)+abs(dx))//(2*abs(dx))))
                         for k in range(abs(dx)+1)] if dx else [(x0, y0)]
            else:
                cells = [(x0+(1 if dx >= 0 else -1)*((2*k*abs(dx)+abs(dy))//(2*abs(dy))),
                          y0+k*(1 if dy >= 0 else -1)) for k in range(abs(dy)+1)]
            samples = []
            for i, j in cells:
                if not valid[j, i] or not np.isfinite(risk[j, i]) or risk[j, i] > RISK_THRESHOLD:
                    raise ValueError("native path crosses forbidden risk cell")
                samples.append(float(risk[j, i]))
            delta = ((b[0]-a[0])/res, (b[1]-a[1])/res)
            distance = math.sqrt(delta[0]**2+delta[1]**2)
            length_parts.append(distance)
            risk_parts.append(distance/(max(map(abs, delta))+1)*math.fsum(samples))
        if len(points) == 1:
            i, j = cell(points[0])
            if not valid[j, i] or not np.isfinite(risk[j, i]) or risk[j, i] > RISK_THRESHOLD:
                raise ValueError("native singleton is invalid")
        for a, b, c in zip(points, points[1:], points[2:]):
            u, v = (b[0]-a[0], b[1]-a[1]), (c[0]-b[0], c[1]-b[1])
            if math.hypot(*u)*math.hypot(*v) == 0:
                angle_parts.append(0.)
            else:
                angle_parts.append(max(0., math.atan2(abs(u[0]*v[1]-u[1]*v[0]), u[0]*v[0]+u[1]*v[1])-math.pi/6))
        values = (math.fsum(length_parts), math.fsum(angle_parts), math.fsum(risk_parts))
        # Keep this arithmetic separate from _components so the audit is useful.
        return {"length_grid": values[0], "turn_excess_rad": values[1], "risk_integral_grid": values[2],
                "weighted_length": .48*values[0], "weighted_turn": .16*values[1],
                "weighted_risk": .36*values[2], "total": .48*values[0]+.16*values[1]+.36*values[2],
                "verification_scope": "native_raster_path_arithmetic_not_raster_construction"}
    finally:
        if close is not None:
            close.close()


@dataclass
class _Node:
    point: tuple
    parent: int | None
    children: list = field(default_factory=list)
    edge_base: tuple = (0., 0.)
    local: tuple = (0., 0., 0.)
    totals: tuple = (0., 0., 0.)
    cost: float = 0.


class SEMNativePlanner:
    STEP_M = 1.
    NEIGHBOR_M = 3.
    START_CONNECTION_M = 3.
    def __init__(self, terrain, model):
        self.terrain, self.model = terrain, model
        if not np.allclose(terrain.bounds, model.bounds, rtol=0, atol=1e-9):
            raise ValueError("terrain and SEM bounds differ")
        self.nodes = []
        self.best_path = None
        self.best_cost = math.inf
        self.last_components = None
        self.counters = {}
        self._reset_counters()

    def _reset_counters(self):
        self.counters = dict.fromkeys(("iterations", "sample_proposals", "sampler_failures", "tree_insertions",
            "edge_evaluations", "hard_rejections", "native_rejections", "parent_candidates", "rewires",
            "descendant_cost_updates", "start_connection_checks", "incumbents", "late_candidates"), 0)

    def _edge(self, a, b):
        self.counters["edge_evaluations"] += 1
        if not self.terrain.valid_motion(a, b):
            self.counters["hard_rejections"] += 1
            return None
        result = self.model.edge(a, b)
        if result is None:
            self.counters["native_rejections"] += 1
        return result

    def _local(self, q, parent, base):
        node = self.nodes[parent]
        turn = 0. if node.parent is None else _turn(q, node.point, self.nodes[node.parent].point)
        return base[0], turn, base[1]

    def _candidate(self, q, parent, base):
        local = self._local(q, parent, base)
        totals = tuple(a+b for a, b in zip(local, self.nodes[parent].totals))
        return math.fsum(w*v for w, v in zip(WEIGHTS, totals)), local, totals

    def _is_ancestor(self, ancestor, child):
        seen = set()
        while child is not None:
            if child in seen:
                raise RuntimeError("cycle in SEM tree")
            if child == ancestor:
                return True
            seen.add(child)
            child = self.nodes[child].parent
        return False

    def _insert(self, point, parent, base):
        cost, local, totals = self._candidate(point, parent, base)
        index = len(self.nodes)
        self.nodes.append(_Node(tuple(point), parent, edge_base=base, local=local, totals=totals, cost=cost))
        self.nodes[parent].children.append(index)
        self.counters["tree_insertions"] += 1
        return index

    def _reparent(self, index, parent, base):
        if self.nodes[index].parent is None or self._is_ancestor(index, parent):
            raise ValueError("rewire would change root or create a cycle")
        previous = self.nodes[index].parent
        self.nodes[previous].children.remove(index)
        self.nodes[parent].children.append(index)
        self.nodes[index].parent = parent
        self.nodes[index].edge_base = base
        queue = [index]
        for current in queue:
            node = self.nodes[current]
            node.cost, node.local, node.totals = self._candidate(node.point, node.parent, node.edge_base)
            queue.extend(node.children)
            if current != index:
                self.counters["descendant_cost_updates"] += 1
        self.counters["rewires"] += 1

    def _path(self, start, connection):
        path = [tuple(start)]
        seen = set()
        while connection is not None:
            if connection in seen:
                raise RuntimeError("cycle in solution path")
            seen.add(connection)
            node = self.nodes[connection]
            if node.point != path[-1]:
                path.append(node.point)
            connection = node.parent
        return tuple(path)

    def _connection_components(self, start, index, base):
        node = self.nodes[index]
        if tuple(start) == node.point:
            return node.cost, node.totals
        cost, _, totals = self._candidate(start, index, base)
        return cost, totals

    def _publish(self, start, connections, began, deadline, callback):
        selected = None
        for index, base in connections.items():
            cost, totals = self._connection_components(start, index, base)
            if cost < self.best_cost and (selected is None or (cost, index) < selected[:2]):
                selected = (cost, index, totals)
        if selected is None:
            return
        cost, index, totals = selected
        path = self._path(start, index)
        raw = [[*point, self.terrain.height(point)] for point in path]
        now = time.perf_counter()
        if now >= deadline:
            self.counters["late_candidates"] += 1
            return
        self.best_path = path  # Tuples own coordinates independently of future rewiring.
        self.best_cost = cost
        self.last_components = _components(totals)
        self.counters["incumbents"] += 1
        callback(now-began, [list(point) for point in path], raw, cost)

    def _sample(self, rng, start, goal):
        xmin, ymin, xmax, ymax = self.model.bounds
        direct = math.dist(start, goal)
        for _ in range(100):
            self.counters["sample_proposals"] += 1
            if math.isfinite(self.best_cost):
                # Native length is XY/res. All other native terms are nonnegative.
                bound = self.model.resolution*self.best_cost/WEIGHTS[0]
                if bound < direct-1e-10:
                    raise RuntimeError("incumbent violates informed lower bound")
                major = max(bound, direct)/2
                minor = math.sqrt(max(0., (bound-direct)*(bound+direct)))/2
                radius, theta = math.sqrt(rng.random()), 2*math.pi*rng.random()
                x, y = major*radius*math.cos(theta), minor*radius*math.sin(theta)
                angle = math.atan2(goal[1]-start[1], goal[0]-start[0])
                q = ((start[0]+goal[0])/2 + x*math.cos(angle)-y*math.sin(angle),
                     (start[1]+goal[1])/2 + x*math.sin(angle)+y*math.cos(angle))
            else:
                q = (float(rng.uniform(xmin, xmax)), float(rng.uniform(ymin, ymax)))
            if xmin <= q[0] <= xmax and ymin <= q[1] <= ymax:
                return q
        self.counters["sampler_failures"] += 1
        return None

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
        deadline = began+budget
        self._reset_counters()
        self.nodes = [_Node(goal, None)]
        self.best_path, self.best_cost, self.last_components = None, math.inf, None
        if not all(self.terrain.valid_motion(p, p) for p in (start, goal)):
            return "INVALID_HARD_ENDPOINTS"
        if not all(self.model.valid_state(p) for p in (start, goal)):
            return "INVALID_NATIVE_ENDPOINTS"
        rng, connections = np.random.default_rng(seed), {}
        def connect(index):
            q = self.nodes[index].point
            if math.dist(start, q) <= self.START_CONNECTION_M:
                self.counters["start_connection_checks"] += 1
                base = self._edge(start, q)
                if base is not None:
                    connections[index] = base
        connect(0)
        self._publish(start, connections, began, deadline, callback)
        if start == goal:
            return "EXACT_SOLUTION" if self.best_path is not None else "TIMEOUT"
        while time.perf_counter() < deadline and (not work_iterations or self.counters["iterations"] < work_iterations):
            self.counters["iterations"] += 1
            sample = self._sample(rng, start, goal)
            if sample is None:
                continue
            distances = [math.dist(n.point, sample) for n in self.nodes]
            nearest = int(np.argmin(distances))
            distance = distances[nearest]
            if distance == 0:
                continue
            a = self.nodes[nearest].point
            fraction = min(1., self.STEP_M/distance)
            q = (a[0]+fraction*(sample[0]-a[0]), a[1]+fraction*(sample[1]-a[1]))
            if not self.model.valid_state(q):
                self.counters["native_rejections"] += 1
                continue
            neighbors = [i for i, n in enumerate(self.nodes) if math.dist(n.point, q) <= self.NEIGHBOR_M]
            candidates, edges = [], {}
            for index in neighbors:
                if time.perf_counter() >= deadline:
                    break
                self.counters["parent_candidates"] += 1
                base = self._edge(q, self.nodes[index].point)
                edges[index] = base
                if base is not None:
                    candidates.append((self._candidate(q, index, base)[0], index))
            if not candidates or time.perf_counter() >= deadline:
                continue
            _, parent = min(candidates)
            new = self._insert(q, parent, edges[parent])
            connect(new)
            for index in neighbors:
                if time.perf_counter() >= deadline:
                    break
                base = edges.get(index)
                if base is None or self.nodes[index].parent is None or self._is_ancestor(index, new):
                    continue
                cost = self._candidate(self.nodes[index].point, new, base)[0]
                if cost < self.nodes[index].cost:
                    # Complete the mutation and all dependent costs atomically with
                    # respect to termination, even if this slightly crosses deadline.
                    self._reparent(index, new, base)
            self._publish(start, connections, began, deadline, callback)
        return "EXACT_SOLUTION" if self.best_path is not None else "TIMEOUT"
