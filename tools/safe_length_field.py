"""Planner SEM input for constrained surface-length planning.

This builder deliberately does not import the independent posthoc evaluator.
It uses indexed maxima/bincount and SciPy correlations; the evaluator uses a
separate sort/reduce implementation. Missing observations remain unsupported
through the derivative, Gaussian and maximum-filter stencils.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np

VERSION = "planner_sem_safe_length_v1"
FORMAT = "SAFE_SEM_FIELD_V1"
RESOLUTION = 0.1


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@dataclass(frozen=True)
class PlannerSEMField:
    bounds: tuple
    resolution: float
    support: np.ndarray
    risk: np.ndarray
    mean: np.ndarray
    variance: np.ndarray
    count: np.ndarray
    observed: np.ndarray
    input_points: int

    @property
    def valid(self):
        return self.support

    def write(self, path, *, cloud_path=None, map_path=None):
        """Publish one immutable field and its provenance; refuse replacement."""
        path = Path(path)
        metadata_path = path.with_suffix(path.suffix + ".meta.json")
        if path.exists() or metadata_path.exists():
            raise FileExistsError("refusing to replace an immutable planner field: " + str(path))
        ny, nx = self.support.shape
        header = [FORMAT, f"{nx} {ny} " + " ".join(format(v, ".17g") for v in
                  (self.bounds[0], self.bounds[1], self.resolution, self.bounds[2], self.bounds[3]))]
        # A numeric zero is an encoding placeholder only when support is zero.
        # C++ must consult support before reading risk, never interpret it as safe.
        rows = [f"{int(ok)} {format(float(risk) if ok else 0., '.17g')}"
                for ok, risk in zip(self.support.flat, self.risk.flat)]
        data = ("\n".join(header + rows) + "\n").encode("ascii")
        metadata = dict(version=VERSION, format=FORMAT, bounds=list(self.bounds),
            resolution_m=self.resolution, shape_yx=[ny, nx], input_points=self.input_points,
            supported_cells=int(self.support.sum()), observed_cells=int(self.observed.sum()),
            cloud_sha256=file_hash(cloud_path) if cloud_path is not None else None,
            map_sha256=file_hash(map_path) if map_path is not None else None,
            builder_source_sha256=file_hash(__file__),
            field_sha256=hashlib.sha256(data).hexdigest(),
            variance="population", surface_filter="z > cell_max_z - 0.1",
            slope_units="degrees", slope_weights=[.5, .5], risk_weights=[200., .75],
            gaussian_radius_cells=5, gaussian_sigma_cells=5/3, maximum_radius_cells=5,
            support_stencils=[3, 11, 11], padding="nearest",
            missing_policy="forbidden_through_all_stencils",
            unsupported_risk_encoding="zero_placeholder_requires_support_zero",
            binning="floor_relative_to_bounds; upper_boundary_last_cell; rows_increase_y",
            planner_field=True, independent_validator=False)
        _atomic_write(path, data)
        _atomic_write(metadata_path, (json.dumps(metadata, indent=2, sort_keys=True,
                                               allow_nan=False) + "\n").encode("utf-8"))
        return metadata


def build_field(points, bounds):
    """Construct the fixed v3 SEM model without consulting any oracle output."""
    from scipy import ndimage

    bounds = tuple(float(v) for v in bounds)
    if len(bounds) != 4 or not all(map(math.isfinite, bounds)):
        raise ValueError("bounds must contain four finite coordinates")
    x0, y0, xmax, ymax = bounds
    if xmax <= x0 or ymax <= y0:
        raise ValueError("empty SEM extent")
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError("cloud must be nonempty finite Nx3")
    if np.any(points[:, :2] < bounds[:2]) or np.any(points[:, :2] > bounds[2:]):
        raise ValueError("cloud outside SEM bounds")
    nx, ny = math.ceil((xmax-x0)/RESOLUTION), math.ceil((ymax-y0)/RESOLUTION)
    ij = np.floor((points[:, :2] - (x0, y0))/RESOLUTION).astype(np.int64)
    ij = np.minimum(ij, (nx-1, ny-1))
    ids = ij[:, 1]*nx + ij[:, 0]
    size = nx*ny
    top = np.full(size, -np.inf)
    np.maximum.at(top, ids, points[:, 2])
    keep = points[:, 2] > top[ids]-RESOLUTION
    keys, z = ids[keep], points[keep, 2]
    count = np.bincount(keys, minlength=size)
    observed = count > 0
    mean = np.divide(np.bincount(keys, weights=z, minlength=size), count,
                     out=np.zeros(size), where=observed)
    variance = np.divide(np.bincount(keys, weights=(z-mean[keys])**2, minlength=size), count,
                         out=np.zeros(size), where=observed)
    shape = (ny, nx)
    mean, variance, count, observed = (v.reshape(shape) for v in (mean, variance, count, observed))
    kx = np.array([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])/4.
    kd = np.array([[-2., -1., 0.], [-1., 0., 1.], [0., 1., 2.]])/3.
    derivatives = [ndimage.correlate(mean, kernel, mode="nearest")
                   for kernel in (kx, kx.T, kd, np.rot90(kd))]
    slope = np.degrees(.5*np.arctan(np.hypot(*derivatives[:2])/(2*RESOLUTION)) +
                       .5*np.arctan(np.hypot(*derivatives[2:])/(2*math.sqrt(2)*RESOLUTION)))
    support = ndimage.minimum_filter(observed, size=3, mode="nearest")
    support = ndimage.minimum_filter(support, size=11, mode="nearest")
    support = ndimage.minimum_filter(support, size=11, mode="nearest") & observed
    smoothed = ndimage.gaussian_filter(slope, sigma=5/3, radius=5, mode="nearest")
    filtered = ndimage.maximum_filter(smoothed, size=11, mode="nearest")
    risk = np.where(support, 200*variance+.75*filtered, np.nan)
    mean, variance = np.where(observed, mean, np.nan), np.where(observed, variance, np.nan)
    for array in (support, risk, mean, variance, count, observed):
        array.setflags(write=False)
    return PlannerSEMField(bounds, RESOLUTION, support, risk, mean, variance, count, observed, len(points))


def build_field_file(cloud_path, map_path, bounds, output_path):
    before = (file_hash(cloud_path), file_hash(map_path))
    result = build_field(np.load(cloud_path, allow_pickle=False), bounds)
    if before != (file_hash(cloud_path), file_hash(map_path)):
        raise ValueError("input changed while building planner SEM field")
    metadata = result.write(output_path, cloud_path=cloud_path, map_path=map_path)
    if before != (metadata["cloud_sha256"], metadata["map_sha256"]):
        raise ValueError("input changed before publishing planner SEM field")
    return metadata
