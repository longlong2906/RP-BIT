"""Independent posthoc SEM field and closed-cell path scoring for schema 12.

This module never imports the SEM planner. The field uses the documented SEM
v2 numerical conventions; the metric traversal is a closed-cell supercover,
not the planner's Bresenham feasibility/cost routine. Every distinct touched
cell counts once per path, including cells touched only at a grid boundary.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

VERSION = "independent_sem_posthoc_closed_cells_v1"
RESOLUTION = .1
RISK_THRESHOLD = 40.


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _correlate3(array, kernel):
    padded = np.pad(array, 1, mode="edge")
    result = np.zeros_like(array, dtype=np.float64)
    for j in range(3):
        for i in range(3):
            if kernel[j, i]:
                result += kernel[j, i] * padded[j:j+array.shape[0], i:i+array.shape[1]]
    return result


class SharedSEMField:
    """A blinded evaluator built from shared input data, once per immutable map."""

    def __init__(self, points, bounds):
        from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter

        self.bounds = tuple(map(float, bounds))
        if len(self.bounds) != 4 or not all(map(math.isfinite, self.bounds)):
            raise ValueError("invalid evaluator bounds")
        xmin, ymin, xmax, ymax = self.bounds
        if xmax <= xmin or ymax <= ymin:
            raise ValueError("empty evaluator extent")
        self.resolution = RESOLUTION
        nx, ny = (math.ceil((xmax-xmin)/RESOLUTION), math.ceil((ymax-ymin)/RESOLUTION))
        cloud = np.asarray(points, dtype=np.float64)
        if cloud.ndim != 2 or cloud.shape[1] != 3 or not len(cloud) or not np.isfinite(cloud).all():
            raise ValueError("evaluator cloud must be nonempty finite Nx3")
        if np.any(cloud[:, :2] < (xmin, ymin)) or np.any(cloud[:, :2] > (xmax, ymax)):
            raise ValueError("evaluator cloud outside map bounds")
        ix = np.minimum(nx-1, np.floor((cloud[:, 0]-xmin)/RESOLUTION).astype(np.int64))
        iy = np.minimum(ny-1, np.floor((cloud[:, 1]-ymin)/RESOLUTION).astype(np.int64))
        keys = iy*nx+ix
        # Sort/reduce each statistical group, independent of the planner's
        # indexed maximum + bincount implementation.
        order = np.argsort(keys, kind="stable")
        keys, z = keys[order], cloud[order, 2]
        starts = np.r_[0, np.flatnonzero(np.diff(keys))+1]
        labels = keys[starts]
        upper = np.maximum.reduceat(z, starts)
        end = np.r_[starts[1:], len(z)]
        top_at_point = np.repeat(upper, end-starts)
        selected = z > top_at_point-RESOLUTION
        selected_keys, selected_z = keys[selected], z[selected]
        starts2 = np.r_[0, np.flatnonzero(np.diff(selected_keys))+1]
        ends2 = np.r_[starts2[1:], len(selected_z)]
        counts = ends2-starts2
        group_mean = np.add.reduceat(selected_z, starts2)/counts
        centered = selected_z-np.repeat(group_mean, counts)
        group_variance = np.add.reduceat(centered*centered, starts2)/counts
        shape, size = (ny, nx), nx*ny
        observed = np.zeros(size, dtype=bool)
        observed[labels] = True
        mean, variance, count = np.zeros(size), np.zeros(size), np.zeros(size, dtype=np.int64)
        mean[labels], variance[labels], count[labels] = group_mean, group_variance, counts
        mean, variance, count, observed = (a.reshape(shape) for a in (mean, variance, count, observed))
        kernels = (
            np.array([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])/4.,
            np.array([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])/4.,
            np.array([[-2., -1., 0.], [-1., 0., 1.], [0., 1., 2.]])/3.,
            np.array([[0., 1., 2.], [-1., 0., 1.], [-2., -1., 0.]])/3.,
        )
        differences = [_correlate3(mean, kernel) for kernel in kernels]
        slope = np.degrees(.5*np.arctan(np.hypot(differences[0], differences[1])/(2*RESOLUTION))
                           + .5*np.arctan(np.hypot(differences[2], differences[3])/(2*math.sqrt(2)*RESOLUTION)))
        stencil_valid = minimum_filter(observed, size=3, mode="nearest")
        stencil_valid = minimum_filter(stencil_valid, size=11, mode="nearest")
        stencil_valid = minimum_filter(stencil_valid, size=11, mode="nearest")
        filtered = maximum_filter(gaussian_filter(slope, sigma=5/3, radius=5, mode="nearest"),
                                  size=11, mode="nearest")
        self.mean = np.where(observed, mean, np.nan)
        self.variance = np.where(observed, variance, np.nan)
        self.count, self.observed = count, observed
        self.valid = stencil_valid & observed
        self.slope_raw_deg = np.where(minimum_filter(observed, size=3, mode="nearest"), slope, np.nan)
        self.slope_risk_deg = np.where(self.valid, filtered, np.nan)
        self.risk = np.where(self.valid, 200*variance+.75*filtered, np.nan)
        self.metadata = dict(version=VERSION, resolution_m=RESOLUTION, bounds=list(self.bounds),
            shape_yx=[ny, nx], input_points=len(cloud), variance="population", slope_units="degree",
            slope_weights=[.5, .5], risk_weights=[200., .75], risk_threshold=RISK_THRESHOLD,
            gaussian_radius_cells=5, gaussian_sigma_cells=5/3, maximum_radius_cells=5,
            padding="nearest", missing_policy="forbidden_through_all_stencils",
            binning="floor; upper map boundary belongs to last cell",
            metric_rasterization="all_distinct_closed_cells_touched_by_centerline",
            boundary_roundoff="16_ulp_in_grid_coordinates; conservative_boundary_ties",
            independent_of_planner=True)
        self._freeze()

    def _freeze(self):
        for name in ("mean", "variance", "count", "observed", "valid", "slope_raw_deg", "slope_risk_deg", "risk"):
            getattr(self, name).setflags(write=False)

    def save(self, directory, identity):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory/"field.npz"
        temporary = path.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **{name:getattr(self, name) for name in
                ("mean", "variance", "count", "observed", "valid", "slope_raw_deg", "slope_risk_deg", "risk")})
        temporary.replace(path)
        metadata = dict(self.metadata, identity=identity, field_sha256=file_hash(path))
        temp_meta = directory/"field.json.tmp"
        temp_meta.write_text(json.dumps(metadata, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        temp_meta.replace(directory/"field.json")
        self.metadata = metadata

    @classmethod
    def load(cls, directory, identity):
        directory = Path(directory)
        metadata = json.loads((directory/"field.json").read_text(encoding="utf-8"))
        if metadata.get("version") != VERSION or metadata.get("identity") != identity:
            raise ValueError("stale shared SEM evaluator identity")
        if file_hash(directory/"field.npz") != metadata.get("field_sha256"):
            raise ValueError("shared SEM field hash mismatch")
        result = cls.__new__(cls)
        result.metadata = metadata
        result.bounds, result.resolution = tuple(metadata["bounds"]), RESOLUTION
        with np.load(directory/"field.npz", allow_pickle=False) as archive:
            for name in ("mean", "variance", "count", "observed", "valid", "slope_raw_deg", "slope_risk_deg", "risk"):
                value = archive[name].copy()
                if list(value.shape) != metadata["shape_yx"]:
                    raise ValueError("shared SEM field shape mismatch")
                setattr(result, name, value)
        if result.valid.dtype != np.bool_ or np.any(~np.isfinite(result.risk[result.valid])):
            raise ValueError("invalid shared SEM field values")
        result._freeze()
        return result

    @staticmethod
    def _coordinate_cells(value, size):
        nearest = round(value)
        tolerance = 16*math.ulp(max(1., abs(value)))
        choices = (nearest-1, nearest) if abs(value-nearest) <= tolerance else (math.floor(value),)
        return tuple(i for i in choices if 0 <= i < size)

    def path_cells(self, path):
        """Closed-cell supercover using all grid-crossing parameter intervals."""
        points = np.asarray(path, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or not len(points) or not np.isfinite(points).all():
            raise ValueError("path must be nonempty finite Nx2")
        if np.any(points < self.bounds[:2]) or np.any(points > self.bounds[2:]):
            raise ValueError("path outside shared evaluator")
        points = (points-np.asarray(self.bounds[:2]))/self.resolution
        ny, nx = self.risk.shape
        cells = set()

        def add(point):
            for i in self._coordinate_cells(float(point[0]), nx):
                for j in self._coordinate_cells(float(point[1]), ny):
                    cells.add((i, j))

        for a, b in (list(zip(points, points[1:])) or [(points[0], points[0])]):
            delta = b-a
            cuts = [0., 1.]
            for axis in (0, 1):
                if delta[axis] != 0:
                    lo, hi = sorted((float(a[axis]), float(b[axis])))
                    for k in range(math.ceil(lo), math.floor(hi)+1):
                        t = float((k-a[axis])/delta[axis])
                        if 0 < t < 1:
                            cuts.append(t)
            cuts = sorted(set(cuts))
            for t in cuts:
                add(a+t*delta)
            for lo, hi in zip(cuts, cuts[1:]):
                add(a+((lo+hi)/2)*delta)
        return sorted(cells)

    def score_path(self, path):
        try:
            cells = self.path_cells(path)
        except ValueError as error:
            return dict(valid=False, pri=None, risk_max=None, cell_count=0, invalid_cell_count=None,
                        reasons=[str(error)])
        values = [float(self.risk[j, i]) for i, j in cells]
        invalid = sum(not self.valid[j, i] or not math.isfinite(value)
                      for (i, j), value in zip(cells, values))
        maximum = max(values) if values and not invalid else None
        reasons = (["missing_sem_support"] if invalid else [])
        if maximum is not None and maximum > RISK_THRESHOLD:
            reasons.append("sem_risk_above_40")
        return dict(valid=bool(cells and not reasons), pri=math.fsum(values)/len(values) if values and not invalid else None,
                    risk_max=maximum, cell_count=len(cells), invalid_cell_count=invalid, reasons=reasons)


def ensure_shared_field(directory, cloud_path, cloud_hash, bounds, map_hash):
    """Read/write only derived evaluator files, bound to raw hashes and source."""
    if file_hash(cloud_path) != cloud_hash:
        raise ValueError("shared evaluator cloud hash mismatch")
    identity = dict(cloud_sha256=cloud_hash, map_sha256=map_hash, bounds=list(bounds),
                    evaluator_source_sha256=file_hash(__file__), version=VERSION)
    directory = Path(directory)
    if (directory/"field.json").exists():
        return SharedSEMField.load(directory, identity)
    field = SharedSEMField(np.load(cloud_path, allow_pickle=False), bounds)
    field.save(directory, identity)
    return field
