"""Canonical on-disk map format. Geometry validation lives in independent_verifier.py."""
import hashlib
import json
import math
from pathlib import Path


def write_map(path, heights, *, origin=(0., 0.), spacing=(1., 1.), radius=.3,
              margin=.1, slope=35., mask=None, rectangles=(), metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ny, nx = len(heights), len(heights[0])
    if nx < 2 or ny < 2 or any(len(row) != nx for row in heights):
        raise ValueError("invalid height dimensions")
    if mask is None:
        mask = [[0] * (nx - 1) for _ in range(ny - 1)]
    def num(value):
        if value is None or math.isnan(value):
            return "nodata"
        if not math.isfinite(value):
            raise ValueError("infinite data")
        return format(value, ".17g")
    lines = ["T25D 1", "units m", "diagonal SW_NE",
             "grid " + " ".join(map(num, [nx, ny, *origin, *spacing])),
             "model " + " ".join(map(num, [radius, margin, slope])), "heights"]
    lines += [" ".join(map(num, row)) for row in heights]
    lines += ["mask"] + [" ".join(map(str, row)) for row in mask]
    lines += [f"rectangles {len(rectangles)}"]
    lines += [" ".join(map(num, rect)) for rect in rectangles] + ["end"]
    data = ("\n".join(lines) + "\n").encode("ascii")
    path.write_bytes(data)
    info = dict(metadata or {}, schema_version=1, data_file=path.name,
                data_sha256=hashlib.sha256(data).hexdigest(), origin_xy=list(origin),
                spacing_xy=list(spacing), node_count_xy=[nx, ny],
                extent_m=[(nx-1)*spacing[0], (ny-1)*spacing[1]], height_units="m",
                row_axis="increasing_y", column_axis="increasing_x", diagonal="SW_NE",
                nodata_encoding="nodata", nodata_policy="whole_cell_if_any_node_missing",
                slope_policy="whole_cell_if_either_triangle_exceeds_limit",
                roughness_enabled=False, robot_radius_m=radius, safety_margin_m=margin,
                max_slope_deg=slope)
    path.with_suffix(".json").write_text(json.dumps(info, indent=2, allow_nan=False), encoding="utf-8")
    return info
