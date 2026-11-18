"""C ABI smoke uses analytic expectations, never the C++ cost as oracle."""
import argparse
import ctypes as C
import math
import os
from pathlib import Path
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    args = parser.parse_args()
    dll_dirs = []
    if os.name == "nt":
        for directory in [args.library.resolve().parent, Path("C:/mingw64/bin")]:
            if directory.exists():
                dll_dirs.append(os.add_dll_directory(str(directory)))
    lib = C.CDLL(str(args.library.resolve()))
    ptr, d = C.c_void_p, C.c_double
    lib.tc_create.argtypes, lib.tc_create.restype = [C.c_char_p, C.c_int], ptr
    lib.tc_destroy.argtypes = [ptr]
    lib.tc_error.restype = C.c_char_p
    lib.tc_valid_motion.argtypes = [ptr, d, d, d, d]
    lib.tc_edge_cost.argtypes = [ptr, d, d, d, d, C.POINTER(d)]
    lib.tc_height.argtypes = [ptr, d, d, C.POINTER(d)]
    lib.tc_putn_create.argtypes, lib.tc_putn_create.restype = [ptr], ptr
    lib.tc_putn_destroy.argtypes = [ptr]
    lib.tc_putn_stats.argtypes = [ptr, C.POINTER(d)]
    callback_type = C.CFUNCTYPE(None, d, C.POINTER(d), C.POINTER(d), C.c_size_t, d, ptr)
    lib.tc_putn_solve.argtypes = [ptr, d, d, d, d, C.c_uint64, d, C.c_int, C.c_uint64, callback_type, ptr]
    with tempfile.TemporaryDirectory(prefix="terrain_capi_") as tmp:
        path = Path(tmp) / "flat.t25d"
        n = 41
        text = "T25D 1\nunits m\ndiagonal SW_NE\ngrid 41 41 0 0 .2 .2\nmodel .3 .1 35\nheights\n"
        text += " ".join(["0"] * (n*n)) + "\nmask\n"
        text += " ".join(["0"] * ((n-1)**2)) + "\nrectangles 0\nend\n"
        path.write_text(text)
        terrain = lib.tc_create(os.fsencode(path), 1)
        assert terrain, lib.tc_error()
        try:
            cost = (d * 5)()
            assert lib.tc_valid_motion(terrain, 1, 1, 7, 7) == 1
            assert lib.tc_valid_motion(terrain, .4, 1, 7, 7) == 0
            assert lib.tc_edge_cost(terrain, 1, 1, 7, 7, cost) == 1
            assert math.isclose(cost[0], math.sqrt(72), rel_tol=1e-12)
            assert list(cost)[1:4] == [0, 0, 0]
            assert cost[4] == cost[0]
            assert lib.tc_valid_motion(None, 1, 1, 2, 2) == -1
            assert b"null" in lib.tc_error()
            planner = lib.tc_putn_create(terrain)
            assert planner, lib.tc_error()
            try:
                setup_stats = (d * 12)()
                assert lib.tc_putn_stats(planner, setup_stats) == 1
                assert setup_stats[8] >= 0 and setup_stats[11] >= 0
                assert setup_stats[8] + setup_stats[11] > 0
                snapshots = []
                @callback_type
                def callback(elapsed, xy, xyz, count, objective, _):
                    snapshots.append((list(xy[:2*count]), list(xyz[:3*count]), objective))
                code = lib.tc_putn_solve(planner, 2.137, 2.249, 2.737, 2.249, 11, .1, 0, 1, callback, None)
                assert code == 1, (code, lib.tc_error())
                xy, xyz, cost_value = snapshots[0]
                assert xy == [2.137, 2.249, 2.737, 2.249]
                assert math.isclose(cost_value, .6, abs_tol=1e-12)
                assert xyz[2] == .7 and xyz[5] == .7
                solved_stats = (d * 12)()
                assert lib.tc_putn_stats(planner, solved_stats) == 1
                assert solved_stats[8] == setup_stats[8] and solved_stats[11] == setup_stats[11]
            finally:
                lib.tc_putn_destroy(planner)
        finally:
            lib.tc_destroy(terrain)
    print("PASS C ABI loader, analytic edge tuple, hard validity, exception boundary, PUTN callback")


if __name__ == "__main__":
    main()
