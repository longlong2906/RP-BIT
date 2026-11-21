"""A5 planner SEM field exchange: the registered field in a binary container.

The cells are the ones safe_length_field.build_field computes; only the container differs from the
text exchange SAFE_SEM_FIELD_V1, whose 160,000 formatted rows cost more to write and to parse than the
field costs to build. The cloud is read once and hashed from the bytes that were parsed.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import struct

import numpy as np

import safe_length_field
from safe_length_field import _atomic_write, build_field, file_hash

FORMAT = "SAFE_SEM_FIELD_B1"
MAGIC = b"SEMFLDB1"
FILE_NAME = "planner_sem_field.bin"


def encode(bounds, resolution, support, risk):
    """Little-endian: magic, int32 nx ny, float64 x0 y0 resolution xmax ymax, uint8 support, float64 risk.

    Rows increase y. As in the text exchange, an unsupported cell carries the placeholder risk 0."""
    ny, nx = support.shape
    return (MAGIC + struct.pack("<2i5d", nx, ny, bounds[0], bounds[1], resolution, bounds[2], bounds[3])
            + np.ascontiguousarray(support, dtype=np.uint8).tobytes()
            + np.where(support, risk, 0.).astype("<f8").tobytes())


def build_field_file(cloud_path, map_path, bounds, output_path):
    """Drop-in for safe_length_field.build_field_file: same field, same metadata keys the request reads."""
    path = Path(output_path)
    metadata_path = path.with_suffix(path.suffix + ".meta.json")
    if path.exists() or metadata_path.exists():
        raise FileExistsError("refusing to replace an immutable planner field: " + str(path))
    cloud = Path(cloud_path).read_bytes()
    field = build_field(np.load(io.BytesIO(cloud), allow_pickle=False), bounds)
    data = encode(field.bounds, field.resolution, field.support, field.risk)
    ny, nx = field.support.shape
    metadata = dict(version=safe_length_field.VERSION, format=FORMAT, same_cells_as=safe_length_field.FORMAT,
                    bounds=list(field.bounds), resolution_m=field.resolution, shape_yx=[ny, nx],
                    input_points=field.input_points, supported_cells=int(field.support.sum()),
                    observed_cells=int(field.observed.sum()), cloud_sha256=hashlib.sha256(cloud).hexdigest(),
                    map_sha256=file_hash(map_path), builder_source_sha256=file_hash(safe_length_field.__file__),
                    writer_source_sha256=file_hash(__file__), field_sha256=hashlib.sha256(data).hexdigest(),
                    layout="little_endian: magic[8] int32[nx,ny] float64[x0,y0,resolution,xmax,ymax] "
                           "uint8 support[ny*nx] float64 risk[ny*nx]; rows_increase_y",
                    unsupported_risk_encoding="zero_placeholder_requires_support_zero",
                    planner_field=True, independent_validator=False)
    _atomic_write(path, data)
    _atomic_write(metadata_path, (json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))
    return metadata
