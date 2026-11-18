"""SGP-RRT*-static-adapted, derived from the pinned BSD-3-Clause upstream.

Source: abeleinin/gp-navigation@597bfb7dafac686248deae8a3b6897108dc2bd9c.
The retained ExactGP + InducingPointKernel model is real GPyTorch. ROS, sensor
coordinates, frontier navigation and controller code are not emulated. See
patches/sgp_static_adaptation.md for the explicit semantic changes.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import time

import numpy as np

UPSTREAM_COMMIT = "597bfb7dafac686248deae8a3b6897108dc2bd9c"
MODEL_VERSION = "sgp_static_physical_thresholds_v1"
PREDICTION_BLOCK_SIZE = 1024


class NativeModelError(RuntimeError):
    pass


def raw_ground_cloud(filename):
    """Read only valid raw H nodes; no C++ feature raster or verifier is used."""
    tokens = iter(Path(filename).read_text(encoding="ascii").split())
    def expect(word):
        if next(tokens) != word:
            raise ValueError("map token mismatch: " + word)
    expect("T25D"); expect("1"); expect("units"); expect("m")
    expect("diagonal"); expect("SW_NE"); expect("grid")
    nx, ny = int(next(tokens)), int(next(tokens))
    x0, y0, dx, dy = (float(next(tokens)) for _ in range(4))
    if nx < 2 or ny < 2 or dx <= 0 or dy <= 0:
        raise ValueError("invalid raw map grid")
    expect("model")
    model = [float(next(tokens)) for _ in range(3)]
    expect("heights")
    points = []
    missing = 0
    for j in range(ny):
        for i in range(nx):
            value = next(tokens)
            if value == "nodata":
                missing += 1
                continue
            h = float(value)
            if not math.isfinite(h):
                raise ValueError("nonfinite raw elevation")
            points.append([x0 + i*dx, y0 + j*dy, h])
    expect("mask")
    for _ in range((nx-1)*(ny-1)):
        if int(next(tokens)) not in (0, 1):
            raise ValueError("invalid raw cell mask")
    expect("rectangles")
    rectangles = [[float(next(tokens)) for _ in range(4)] for _ in range(int(next(tokens)))]
    expect("end")
    if list(tokens):
        raise ValueError("trailing map content")
    if not points:
        raise NativeModelError("no finite elevation observations")
    return np.asarray(points, dtype=np.float64), {
        "bounds": [x0, y0, x0+(nx-1)*dx, y0+(ny-1)*dy],
        "raw_spacing_xy": [dx, dy], "raw_node_count": nx*ny,
        "omitted_nodata_nodes": missing, "explicit_obstacles": rectangles,
        "obstacle_height_m": 2., "model": model,
    }


def global_grid(bounds, resolution=.2):
    xmin, ymin, xmax, ymax = bounds
    counts = [int(round((xmax-xmin)/resolution)), int(round((ymax-ymin)/resolution))]
    if any(n < 2 for n in counts) or not all(math.isclose(v/resolution, n, rel_tol=0, abs_tol=1e-8)
            for v, n in zip((xmax-xmin, ymax-ymin), counts)):
        raise NativeModelError("map extent must be divisible by the fixed 0.2m SGP grid")
    # Canonical row j increases Y; unlike upstream's local, transposed indexing.
    x = xmin + np.arange(counts[0]+1, dtype=np.float64)*resolution
    y = ymin + np.arange(counts[1]+1, dtype=np.float64)*resolution
    xx, yy = np.meshgrid(x, y, indexing="xy")
    return np.column_stack((xx.ravel(), yy.ravel())).astype(np.float32), (len(y), len(x))


def feature_fields(mean, gradient, variance, bounds, resolution=.2):
    """Physical-threshold features; local relief is not a discontinuous step."""
    mean = np.asarray(mean, dtype=np.float64)
    variance = np.asarray(variance, dtype=np.float64)
    gradient = np.asarray(gradient, dtype=np.float64)
    if mean.ndim != 2 or variance.shape != mean.shape or gradient.shape != (*mean.shape, 2):
        raise NativeModelError("GP prediction shape/correspondence mismatch")
    if not all(np.isfinite(a).all() for a in (mean, gradient, variance)) or (variance < 0).any():
        raise NativeModelError("nonfinite GP prediction/invalid uncertainty")
    slope = np.arctan(np.hypot(gradient[:, :, 0], gradient[:, :, 1]))
    tilt = np.empty_like(mean)
    relief = np.empty_like(mean)
    # A 1m-wide centered physical window: include node centers within +/-0.5m.
    half = int(math.floor(.5/resolution + 1e-12))
    for j in range(mean.shape[0]):
        for i in range(mean.shape[1]):
            j0, j1 = max(0, j-half), min(mean.shape[0], j+half+1)
            i0, i1 = max(0, i-half), min(mean.shape[1], i+half+1)
            dx, dy = np.meshgrid((np.arange(i0, i1)-i)*resolution,
                                 (np.arange(j0, j1)-j)*resolution)
            local = mean[j0:j1, i0:i1]
            design = np.column_stack((dx.ravel(), dy.ravel(), np.ones(local.size)))
            coeff, _, rank, _ = np.linalg.lstsq(design, local.ravel(), rcond=None)
            if rank < 3:
                raise NativeModelError("plane-fit neighborhood rank deficient")
            tilt[j, i] = math.atan(math.hypot(*coeff[:2]))
            relief[j, i] = np.max(np.abs(local-mean[j, i]))
    penalty = np.clip(.2*slope/.5236 + .5*tilt/.5236 + .3*relief/.30, 0., 1.)
    uncertainty_limit = float(variance.mean())*1.4
    uncertain = variance >= uncertainty_limit
    penalty[uncertain] = 1.
    return {"slope_rad": slope, "plane_tilt_rad": tilt, "local_relief_m": relief,
            "traversability": penalty, "uncertainty_mask": uncertain,
            "uncertainty_limit": uncertainty_limit}


class SGPStaticModel:
    def __init__(self, points, bounds, seed=0):
        import torch
        import gpytorch
        torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        torch.manual_seed(int(seed))
        np.random.seed(int(seed) % (2**32))
        self.bounds = tuple(bounds)
        self.resolution = .2
        grid, shape = global_grid(bounds, self.resolution)
        # Preserve upstream sort, rounding, float32 inputs, kernel and optimizer.
        cloud = np.round(np.asarray(points, dtype=np.float64), 4)
        cloud = cloud[np.argsort(cloud[:, 0])]
        x = torch.tensor(cloud[:, :2], dtype=torch.float32, device="cpu")
        y = torch.tensor(cloud[:, 2], dtype=torch.float32, device="cpu")
        inducing_step = max(1, int(len(cloud)/500))
        inducing = x[list(range(0, len(cloud), inducing_step)), :]

        class SourceSGP(gpytorch.models.ExactGP):
            def __init__(self, tx, ty, likelihood):
                super().__init__(tx, ty, likelihood)
                self.mean_module = gpytorch.means.ConstantMean()
                self.base_covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RQKernel(lengthscale=torch.tensor([.7, .7]),
                                             alpha=torch.tensor([10])))
                self.covar_module = gpytorch.kernels.InducingPointKernel(
                    self.base_covar_module, inducing_points=inducing, likelihood=likelihood)
            def forward(self, inputs):
                return gpytorch.distributions.MultivariateNormal(
                    self.mean_module(inputs), self.covar_module(inputs))

        likelihood = gpytorch.likelihoods.GaussianLikelihood(mean=0)
        model = SourceSGP(x, y, likelihood).cpu()
        model.train(); likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        optimizer.zero_grad()
        loss = -mll(model(x), y).mean()
        if not torch.isfinite(loss):
            raise NativeModelError("nonfinite GP training loss")
        loss.backward(); optimizer.step()
        model.eval(); likelihood.eval()
        means, variances, gradients = [], [], []
        # Block inference changes memory scheduling only; each block queries the
        # same fitted model. Retain autograd for the predicted-mean gradient.
        for offset in range(0, len(grid), PREDICTION_BLOCK_SIZE):
            query = torch.tensor(grid[offset:offset+PREDICTION_BLOCK_SIZE],
                                 dtype=torch.float32, device="cpu", requires_grad=True)
            prediction = likelihood(model(query))
            mean_grad = torch.autograd.grad(prediction.mean, query,
                                            grad_outputs=torch.ones_like(prediction.mean))[0]
            means.append(prediction.mean.detach().numpy())
            variances.append(prediction.variance.detach().numpy())
            gradients.append(mean_grad.detach().numpy())
        self.mean = np.concatenate(means).reshape(shape)
        self.variance = np.concatenate(variances).reshape(shape)
        self.gradient = np.concatenate(gradients).reshape(*shape, 2)
        self.fields = feature_fields(self.mean, self.gradient, self.variance, bounds)
        self.traversability = self.fields["traversability"]
        self.metadata = {"version": MODEL_VERSION, "upstream_commit": UPSTREAM_COMMIT,
            "torch": torch.__version__, "gpytorch": gpytorch.__version__,
            "numpy": np.__version__, "device": "cpu", "training_steps": 1,
            "learning_rate": .01, "training_observations": len(cloud),
            "inducing_points": len(inducing), "inducing_stride": inducing_step,
            "prediction_block_size": PREDICTION_BLOCK_SIZE, "training_loss": float(loss),
            "prediction_grid_shape_yx": list(shape), "resolution_m": .2,
            "weights": {"slope": .2, "plane_tilt": .5, "local_relief": .3},
            "critical_thresholds": {"slope_rad": .5236, "plane_tilt_rad": .5236, "local_relief_m": .30},
            "traversability_limit": .3, "uncertainty_mean_multiplier": 1.4,
            "plane_window_width_m": 1., "native_filter_grid": [3, 3],
            "step_size_m": .5, "neighbor_radius_m": 1., "goal_threshold_m": .5,
            "nearest_metric": "euclidean_xy", "native_cost": "euclidean_xy_length",
            "uncertainty_limit": self.fields["uncertainty_limit"],
            "uncertain_nodes": int(self.fields["uncertainty_mask"].sum()),
            "traversability_rejected_nodes": int((self.traversability > .3).sum()),
            "thread_count": torch.get_num_threads(), "interop_thread_count": torch.get_num_interop_threads(),
            "trained_hyperparameters": {
                "constant_mean": float(model.mean_module.constant.detach()),
                "noise": float(likelihood.noise.detach()),
                "outputscale": float(model.base_covar_module.outputscale.detach()),
                "lengthscale": model.base_covar_module.base_kernel.lengthscale.detach().tolist(),
                "rq_alpha": model.base_covar_module.base_kernel.alpha.detach().tolist()}}

    def valid_state(self, point):
        x, y = point
        xmin, ymin, xmax, ymax = self.bounds
        if not (xmin <= x <= xmax and ymin <= y <= ymax):
            return False
        # Nearest-node assignment, without upstream's two coordinate roundings.
        i = int(math.floor((x-xmin)/self.resolution + .5))
        j = int(math.floor((y-ymin)/self.resolution + .5))
        if i < 1 or j < 1 or i+1 >= self.traversability.shape[1] or j+1 >= self.traversability.shape[0]:
            return False
        return bool(np.all(self.traversability[j-1:j+2, i-1:i+2] <= .3))

    def valid_motion(self, a, b):
        """Check each constant native-grid region touched by the connector.

        The common C++ checker separately checks continuous swept hard safety.
        The grid crossings here refer to nearest-node boundaries at half cells.
        """
        cuts = [0., 1.]
        for axis in (0, 1):
            delta = b[axis]-a[axis]
            if delta == 0:
                continue
            low, high = sorted((a[axis], b[axis]))
            origin = self.bounds[axis]
            first = int(math.ceil((low-origin)/self.resolution-.5))
            last = int(math.floor((high-origin)/self.resolution-.5))
            for k in range(first, last+1):
                t = (origin+(k+.5)*self.resolution-a[axis])/delta
                if 0. < t < 1.:
                    cuts.append(t)
        cuts = sorted(set(cuts))
        times = cuts + [(l+r)*.5 for l, r in zip(cuts, cuts[1:])]
        return all(self.valid_state((a[0]+t*(b[0]-a[0]), a[1]+t*(b[1]-a[1]))) for t in times)


@dataclass
class Node:
    point: tuple
    parent: int | None
    edge_cost: float = 0.


class SGPStaticPlanner:
    """Source-derived nearest/steer/radius-1 RRT* with safe terminal edges."""
    def __init__(self, terrain, native_model, own_cost=False):
        self.terrain, self.model, self.own_cost = terrain, native_model, own_cost
        self.nodes = []
        self.counters = {name: 0 for name in ("iterations", "samples", "edge_evaluations",
            "common_checker_rejections", "native_filter_rejections", "rewires",
            "parent_candidates", "goal_connector_checks", "incumbents")}

    def edge(self, a, b):
        self.counters["edge_evaluations"] += 1
        if not self.terrain.valid_motion(a, b):
            self.counters["common_checker_rejections"] += 1
            return None
        if not self.model.valid_motion(a, b):
            self.counters["native_filter_rejections"] += 1
            return None
        if self.own_cost:
            return math.dist(a, b)
        result = self.terrain.edge_cost(a, b)
        if result is None:
            raise NativeModelError("valid edge has no common objective")
        return result[4]

    def cost(self, index):
        value, visited = 0., set()
        while self.nodes[index].parent is not None:
            if index in visited:
                raise RuntimeError("parent cycle")
            visited.add(index)
            value += self.nodes[index].edge_cost
            index = self.nodes[index].parent
        return value

    def is_ancestor(self, ancestor, index):
        while index is not None:
            if ancestor == index:
                return True
            index = self.nodes[index].parent
        return False

    def path(self, index, goal):
        result = [list(goal)]
        while index is not None:
            if tuple(result[-1]) != self.nodes[index].point:
                result.append(list(self.nodes[index].point))
            index = self.nodes[index].parent
        return result[::-1]

    def solve(self, start, goal, seed, budget_s, callback, fixed_work=0):
        started = time.perf_counter()
        start, goal = tuple(start), tuple(goal)
        self.nodes = [Node(start, None)]
        self.counters = dict.fromkeys(self.counters, 0)
        if not (self.terrain.valid_motion(start, start) and self.terrain.valid_motion(goal, goal)):
            return "INVALID_HARD_ENDPOINTS"
        if not (self.model.valid_state(start) and self.model.valid_state(goal)):
            return "INVALID_NATIVE_ENDPOINTS"
        rng = np.random.RandomState(int(seed) % (2**32))
        if start == goal:
            callback(time.perf_counter()-started, [list(start)], [[*start, self.terrain.height(start)]], 0.)
            return "EXACT_SOLUTION"
        candidates = {}
        best = math.inf
        xmin, ymin, xmax, ymax = self.terrain.bounds
        while (self.counters["iterations"] < fixed_work if fixed_work else time.perf_counter()-started < budget_s):
            self.counters["iterations"] += 1
            self.counters["samples"] += 1
            sample = (float(rng.uniform(xmin, xmax)), float(rng.uniform(ymin, ymax)))
            distances = [math.dist(n.point, sample) for n in self.nodes]
            nearest = int(np.argmin(distances))
            distance = distances[nearest]
            if distance == 0.:
                continue
            a = self.nodes[nearest].point
            ratio = min(.5, distance)/distance
            q = (a[0]+ratio*(sample[0]-a[0]), a[1]+ratio*(sample[1]-a[1]))
            initial_edge = self.edge(a, q)
            if initial_edge is None:
                continue
            neighbors = [i for i, node in enumerate(self.nodes) if math.dist(node.point, q) <= 1.]
            parent, parent_edge = nearest, initial_edge
            parent_cost = self.cost(nearest)+initial_edge
            checked = {nearest: initial_edge}
            for i in neighbors:
                self.counters["parent_candidates"] += 1
                edge = checked.get(i)
                if edge is None:
                    edge = self.edge(self.nodes[i].point, q)
                    checked[i] = edge
                if edge is not None and self.cost(i)+edge < parent_cost:
                    parent, parent_edge, parent_cost = i, edge, self.cost(i)+edge
            new_index = len(self.nodes)
            self.nodes.append(Node(q, parent, parent_edge))
            for i in neighbors:
                if i == 0 or self.is_ancestor(i, new_index):
                    continue
                edge = checked[i]
                if edge is not None and self.cost(new_index)+edge < self.cost(i):
                    self.nodes[i].parent, self.nodes[i].edge_cost = new_index, edge
                    self.counters["rewires"] += 1
            if math.dist(q, goal) <= .5:
                self.counters["goal_connector_checks"] += 1
                goal_edge = self.edge(q, goal)
                if goal_edge is not None:
                    candidates[new_index] = goal_edge
            if candidates:
                index, value = min(((i, self.cost(i)+e) for i, e in candidates.items()), key=lambda x: x[1])
                if value < best:
                    best = value
                    path = self.path(index, goal)
                    raw = [[*p, self.terrain.height(p)] for p in path]
                    self.counters["incumbents"] += 1
                    callback(time.perf_counter()-started, path, raw, value)
        return "EXACT_SOLUTION" if math.isfinite(best) else "TIMEOUT"
