"""Opt-in, bounded edge reuse for schema-11 posthoc geometry validation.

Every miss executes the existing independent continuous Oracle. Neither planner
geometry nor native costs are consulted. validate_path and its endpoint, cost,
surface, footprint, NoData and slope checks are inherited without modification.
The parsed map is immutable for the lifetime of this per-map instance.
"""
from __future__ import annotations

from collections import OrderedDict
import copy
import hashlib
from pathlib import Path
import struct
import sys

import numpy as np

from independent_verifier import Oracle as IndependentOracle


VERSION = "native_paper_independent_edge_memo_v1"


def _exact_config(value):
    if value is None:
        return ("none",)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, (int, np.integer)):
        return ("int", int(value))
    if isinstance(value, (float, np.floating)):
        return ("float", type(value).__name__, struct.pack(">d", float(value)))
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return ("dict", tuple((k, _exact_config(v)) for k, v in sorted(value.items())))
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, tuple(_exact_config(v) for v in value))
    raise TypeError("unsupported cache-key input")


def _endpoint(value):
    array = np.asarray(value)
    if array.shape != (2,) or array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise TypeError("uncacheable endpoint")
    # Preserve dtype, every endpoint bit (including signed zero/adjacent ULPs),
    # and order. No rounding, tolerance matching or reverse-edge canonicalization.
    return array.dtype.str, array.tobytes(order="C")


def _stored_size(value, seen=None):
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        size += sum(_stored_size(k, seen)+_stored_size(v, seen) for k, v in value.items())
    elif isinstance(value, (tuple, list)):
        size += sum(_stored_size(v, seen) for v in value)
    return size


class NativePaperOracle(IndependentOracle):
    """Independent Oracle with a per-instance LRU of exact edge computations.

    Limits apply across all memoized operations together. Cached mutable values
    are never returned directly. Public map fields and NumPy buffers are frozen;
    callers must construct a new instance for another map or geometry model.
    """

    _MAP_FIELDS = frozenset(("nx", "ny", "x0", "y0", "dx", "dy", "radius", "margin", "slope",
                             "heights", "mask", "rectangles", "bounds", "triangles", "cell_triangle_ids",
                             "forbidden", "vertices", "coefficients", "triangle_slopes", "rect_array",
                             "map_sha256"))

    def __init__(self, filename, *, max_entries=8192, max_cache_bytes=64*1024**2):
        for name, value in (("max_entries", max_entries), ("max_cache_bytes", max_cache_bytes)):
            if type(value) is not int or value < 0:
                raise ValueError(name+" must be a nonnegative integer")
        path = Path(filename)
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        super().__init__(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != before:
            raise ValueError("map changed during independent oracle construction")
        self.map_sha256 = before
        for name in self._MAP_FIELDS:
            value = getattr(self, name, None)
            if isinstance(value, np.ndarray):
                value.setflags(write=False)
        for vertices, coefficients in self.triangles:
            vertices.setflags(write=False)
            coefficients.setflags(write=False)
        self.triangles = tuple(self.triangles)
        self.forbidden = tuple(self.forbidden)
        self.rectangles = tuple(self.rectangles)
        self._map_frozen = True
        self.max_entries, self.max_cache_bytes = max_entries, max_cache_bytes
        self._edge_cache = OrderedDict()
        self._cache_bytes = 0
        self._counts = dict(hits=0, misses=0, bypasses=0, evictions=0)
        self._operation_counts = {}

    def __setattr__(self, name, value):
        if getattr(self, "_map_frozen", False) and name in self._MAP_FIELDS:
            raise AttributeError("independent oracle map is immutable; construct a new instance")
        super().__setattr__(name, value)

    def _memo(self, operation, a, b, options, compute):
        counts = self._operation_counts.setdefault(operation, dict(hits=0, misses=0, bypasses=0))
        try:
            key = (operation, _endpoint(a), _endpoint(b), _exact_config(options))
        except (TypeError, ValueError, OverflowError):
            self._counts["bypasses"] += 1
            counts["bypasses"] += 1
            return compute()
        if key in self._edge_cache:
            self._counts["hits"] += 1
            counts["hits"] += 1
            value, size = self._edge_cache.pop(key)
            self._edge_cache[key] = value, size
            return copy.deepcopy(value)
        self._counts["misses"] += 1
        counts["misses"] += 1
        value = compute()  # Exceptions retain their original type/message and are not cached.
        if self.max_entries and self.max_cache_bytes:
            stored = copy.deepcopy(value)
            size = _stored_size(key)+_stored_size(stored)+256  # conservative LRU bookkeeping allowance
            if size <= self.max_cache_bytes:
                while self._edge_cache and (len(self._edge_cache) >= self.max_entries or
                                            self._cache_bytes+size > self.max_cache_bytes):
                    _, (_, removed_size) = self._edge_cache.popitem(last=False)
                    self._cache_bytes -= removed_size
                    self._counts["evictions"] += 1
                self._edge_cache[key] = stored, size
                self._cache_bytes += size
        # Protect frozen map arrays even on a first computation (interval
        # coefficients are views into the original independent Oracle).
        return copy.deepcopy(value)

    def valid_edge(self, a, b):
        return self._memo("valid_edge", a, b, None,
                          lambda: super(NativePaperOracle, self).valid_edge(a, b))

    def intervals(self, a, b, *, use_index=True):
        return self._memo("intervals", a, b, dict(use_index=use_index),
                          lambda: super(NativePaperOracle, self).intervals(a, b, use_index=use_index))

    def surface(self, a, b):
        return self._memo("surface", a, b, None,
                          lambda: super(NativePaperOracle, self).surface(a, b))

    def _swept_slope(self, a, b):
        return self._memo("swept_slope", a, b, None,
                          lambda: super(NativePaperOracle, self)._swept_slope(a, b))

    def traversability_edge(self, a, b, config):
        return self._memo("traversability_edge", a, b, config,
                          lambda: super(NativePaperOracle, self).traversability_edge(a, b, config))

    def cache_info(self):
        return dict(version=VERSION, map_sha256=self.map_sha256, **self._counts,
                    entries=len(self._edge_cache), accounted_bytes=self._cache_bytes,
                    max_entries=self.max_entries, max_cache_bytes=self.max_cache_bytes,
                    operations=copy.deepcopy(self._operation_counts))

    def clear_edge_cache(self):
        self._edge_cache.clear()
        self._cache_bytes = 0
