"""Deterministic synthetic observations of the canonical, unchanged terrain.

This is representation conversion, not measured sensor data. The 5 cm master
cloud contains visible ground, 2 m obstacle tops and vertical boundary surfaces.
NoData cells are never filled. Native planners may reduce this common input.
"""
from pathlib import Path
import math
import numpy as np

VERSION = "canonical_surface_cloud_005_v1"
SPACING_M = .05
OBSTACLE_HEIGHT_M = 2.


def _read(filename):
    words = iter(Path(filename).read_text(encoding="ascii").split())
    def expect(word):
        if next(words) != word:
            raise ValueError("map token mismatch: " + word)
    try:
        for word in ("T25D", "1", "units", "m", "diagonal", "SW_NE", "grid"):
            expect(word)
        nx, ny = int(next(words)), int(next(words))
        x0, y0, dx, dy = (float(next(words)) for _ in range(4))
        if nx < 2 or ny < 2 or not np.isfinite([x0, y0, dx, dy]).all() or min(dx, dy) <= 0:
            raise ValueError("invalid grid")
        expect("model")
        model = [float(next(words)) for _ in range(3)]
        if not np.isfinite(model).all():
            raise ValueError("nonfinite model")
        expect("heights")
        values = [next(words) for _ in range(nx*ny)]
        h = np.array([np.nan if v == "nodata" else float(v) for v in values]).reshape(ny, nx)
        if np.isinf(h).any() or any(v != "nodata" and not math.isfinite(float(v)) for v in values):
            raise ValueError("nonfinite height must use nodata")
        expect("mask")
        if any(int(next(words)) not in (0, 1) for _ in range((nx-1)*(ny-1))):
            raise ValueError("invalid mask")
        expect("rectangles")
        count = int(next(words))
        if count < 0:
            raise ValueError("negative rectangle count")
        rectangles = np.array([[float(next(words)) for _ in range(4)] for _ in range(count)]).reshape(-1, 4)
        if not np.isfinite(rectangles).all() or any(r[0] >= r[2] or r[1] >= r[3] for r in rectangles):
            raise ValueError("invalid rectangle")
        expect("end")
        if list(words):
            raise ValueError("trailing map tokens")
    except StopIteration as exc:
        raise ValueError("truncated map") from exc
    return h, (x0, y0, dx, dy), rectangles


def _height(h, grid, xy):
    x0, y0, dx, dy = grid
    uv = (np.asarray(xy, dtype=np.float64)-[x0, y0])/[dx, dy]
    inside = ((uv >= 0).all(axis=1) & (uv <= [h.shape[1]-1, h.shape[0]-1]).all(axis=1))
    ij = np.floor(np.clip(uv, [0, 0], [h.shape[1]-2, h.shape[0]-2])).astype(np.int64)
    i, j = ij.T
    u, v = (uv-ij).T
    sw, se, nw, ne = h[j, i], h[j, i+1], h[j+1, i], h[j+1, i+1]
    # Whole-cell NoData policy, including the triangle not containing the point.
    present = inside & np.isfinite(np.column_stack((sw, se, nw, ne))).all(axis=1)
    z = np.where(u >= v, sw+(se-sw)*u+(ne-se)*v,
                 sw+(ne-nw)*u+(nw-sw)*v)
    return np.where(present, z, np.nan)


def generate_cloud(map_path):
    """Return stable float64 N×3 observations; callers own saving/hash policy."""
    h, grid, rectangles = _read(map_path)
    x0, y0, dx, dy = grid
    xmax, ymax = x0+(h.shape[1]-1)*dx, y0+(h.shape[0]-1)*dy
    x = np.arange(x0+SPACING_M/2, xmax, SPACING_M, dtype=np.float64)
    y = np.arange(y0+SPACING_M/2, ymax, SPACING_M, dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    xy = np.column_stack((xx.ravel(), yy.ravel()))
    z = _height(h, grid, xy)
    covered = np.zeros(len(xy), dtype=bool)
    for r in rectangles:
        covered |= ((xy >= r[:2]) & (xy <= r[2:])).all(axis=1)
    chunks = [np.column_stack((xy, z+covered*OBSTACLE_HEIGHT_M))[np.isfinite(z)]]
    for r in rectangles:
        a, b = np.maximum(r[:2], [x0, y0]), np.minimum(r[2:], [xmax, ymax])
        if (b <= a).any():
            continue
        # Include boundary corners; the interior spacing is never larger than .05 m.
        xs = np.linspace(a[0], b[0], int(math.ceil((b[0]-a[0])/SPACING_M))+1)
        ys = np.linspace(a[1], b[1], int(math.ceil((b[1]-a[1])/SPACING_M))+1)
        boundary = np.vstack((np.column_stack((xs, np.full(len(xs), a[1]))),
                              np.column_stack((xs, np.full(len(xs), b[1]))),
                              np.column_stack((np.full(len(ys), a[0]), ys)),
                              np.column_stack((np.full(len(ys), b[0]), ys))))
        base = _height(h, grid, boundary)
        keep = np.isfinite(base)
        # Internal faces of overlapping solid rectangles are not observations.
        for other in rectangles:
            keep &= ~((boundary > other[:2]) & (boundary < other[2:])).all(axis=1)
        boundary, base = boundary[keep], base[keep]
        heights = np.linspace(0, OBSTACLE_HEIGHT_M, round(OBSTACLE_HEIGHT_M/SPACING_M)+1)
        chunks.append(np.column_stack((np.repeat(boundary, len(heights), axis=0),
                                        (base[:, None]+heights).ravel())))
    points = np.concatenate(chunks)
    if not len(points):
        raise ValueError("master cloud contains no observations")
    return np.unique(points, axis=0)


def voxel_centroids(points, resolution=.2):
    """3D voxel reduction on a fixed world origin; never merge different layers."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError("points must be nonempty finite N by 3")
    if not math.isfinite(resolution) or resolution <= 0:
        raise ValueError("positive finite voxel resolution required")
    if np.max(np.abs(points/resolution)) >= np.iinfo(np.int64).max:
        raise ValueError("voxel index overflow")
    order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
    points = points[order]
    keys = np.floor(points/resolution).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    sums = np.zeros((len(counts), 3), dtype=np.float64)
    np.add.at(sums, inverse, points)
    means = sums/counts[:, None]
    return means[np.lexsort((means[:, 2], means[:, 1], means[:, 0]))]
