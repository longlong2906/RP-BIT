"""Bit-identity gate of the C++ SEM-RRT* port on the platform that runs the campaign.

For every map of an existing (non-held-out) phase directory and every planner seed, the Python
reconstruction and the C++ port run the same fixed number of iterations on the same native raster.
They must agree exactly: status, the 13 counters and every incumbent (path, terrain heights, native
cost and its three components). Wall times are recorded, never compared. Each run must complete the
whole iteration budget and publish at least one incumbent, so the gate cannot pass vacuously.
One process on one pinned core, one thread; exit 0 only when every run is identical.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import platform
import sys
import time

GATE = "sem_cpp_bit_identity_v1"


def solve(planner, info, seed, work):
    incumbents = []

    def callback(elapsed, path, raw, cost):
        incumbents.append(dict(cost=cost, path=path, raw=raw, components=dict(planner.last_components)))
    began = time.perf_counter()
    status = planner.solve(info["start"], info["goal"], seed, 3600., callback, work)
    return dict(status=status, counters=dict(planner.counters), incumbents=incumbents, wall_s=time.perf_counter() - began)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-dir", required=True, type=Path, help="phase directory whose manifest lists the maps")
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--bridge", required=True, type=Path)
    parser.add_argument("--search-paths", nargs="*", default=[])
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--work", type=int, required=True)
    parser.add_argument("--core", type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.work <= 0 or args.phase_dir.name.startswith("heldout"):
        raise SystemExit("positive work required; held-out maps are never used here")
    import psutil
    if args.core is not None:
        psutil.Process().cpu_affinity([args.core])
    from corridor_v4_execution import atomic_json, file_hash, read, set_thread_limits, utc_now
    set_thread_limits()
    import numpy as np
    from sem_native_cpp import SEMNativeCppEngine, SEMNativePlannerCpp
    from sem_native_planner import SEMNativeModel, SEMNativePlanner
    from terrain_compare_bridge import TerrainBridge

    manifest = args.phase_dir / "manifest.json"
    maps, rows = read(manifest)["maps"], []
    for name in sorted(maps):
        info = maps[name]
        bridge = TerrainBridge(args.bridge, info["map_path"], args.search_paths, risk_enabled=False)
        model = SEMNativeModel(np.load(info["cloud_path"], allow_pickle=False), bridge.bounds)
        engine = SEMNativeCppEngine(args.library, info["map_path"], model, args.search_paths)
        try:
            for seed in args.seeds:
                python = solve(SEMNativePlanner(bridge, model), info, seed, args.work)
                cpp = solve(SEMNativePlannerCpp(engine), info, seed, args.work)
                identical = all(python[key] == cpp[key] for key in ("status", "counters", "incumbents"))
                complete = python["counters"]["iterations"] == args.work and bool(python["incumbents"])
                final = python["incumbents"][-1]["cost"] if python["incumbents"] else math.inf
                rows.append(dict(map_id=name, seed=seed, identical=identical, complete=complete, status=python["status"],
                                 iterations=python["counters"]["iterations"], incumbents=len(python["incumbents"]),
                                 final_native_cost=final, python_s=python["wall_s"], cpp_s=cpp["wall_s"]))
                print(f"{name} s{seed} identical={identical} complete={complete} python={python['wall_s']:.2f}s "
                      f"cpp={cpp['wall_s']:.3f}s cost={final:.6f}", flush=True)
        finally:
            engine.close()
            bridge.close()
    passed = bool(rows) and all(row["identical"] and row["complete"] for row in rows)
    atomic_json(args.output, dict(gate=GATE, passed=passed, runs=len(rows), identical=sum(r["identical"] for r in rows),
                                  complete=sum(r["complete"] for r in rows), work_iterations=args.work, seeds=args.seeds,
                                  phase_manifest_sha256=file_hash(manifest), library_sha256=file_hash(args.library),
                                  bridge_sha256=file_hash(args.bridge), platform=platform.platform(), python=sys.version,
                                  numpy=np.__version__, utc=utc_now(), rows=rows))
    print("identical", sum(r["identical"] for r in rows), "/", len(rows), "passed", passed)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
