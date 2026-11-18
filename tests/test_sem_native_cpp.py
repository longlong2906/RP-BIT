"""C++ SEM-RRT* port: bit-for-bit equivalence with the Python reconstruction, and the worker path.

Every comparison is exact (==), never a tolerance: same numpy random stream, same CPython math,
same model edges, and full solves with identical counters, incumbent costs, paths and heights.
"""
import argparse
import copy
import ctypes as C
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from native_paper_oracle import NativePaperOracle
from native_paper_v3_protocol import request_for as native_request, default_config as native_config
from native_paper_v3_sem_evaluator import SharedSEMField
from sem_native_cpp import SEMNativeCppEngine, SEMNativePlannerCpp, load_library
from sem_native_planner import SEMNativeModel, SEMNativePlanner, _turn
from safe_length_field import file_hash
from terrain_compare_bridge import TerrainBridge
from terrain_io import write_map

LIBRARY = BRIDGE = None
SEARCH = [str(p) for p in (ROOT / ".deps/install/lib", Path("C:/mingw64/bin")) if p.is_dir()]


def height(x, y):
    return .25 * math.sin(.6 * x) + .2 * math.cos(.5 * y) + .15 * math.exp(-((x - 6.) ** 2 + (y - 4.) ** 2) / 2.)


def fixture(directory, kind):
    """Labeled synthetic inputs: 'rolling' (12 m, obstacle, noisy patch, cloud hole) or 'plane' (6 m)."""
    directory = Path(directory)
    if kind == "plane":
        terrain = directory / "plane.t25d"
        write_map(terrain, [[.1 * (i * .2) + .05 * (j * .2) for i in range(31)] for j in range(31)], spacing=(.2, .2),
                  rectangles=[(2.5, 2., 3.5, 4.)], metadata=dict(test_fixture=True, purpose="sem_cpp_equivalence"))
        x, y = np.meshgrid(np.arange(.025, 6., .05), np.arange(.025, 6., .05))
        points = np.column_stack((x.ravel(), y.ravel(), (.1 * x + .05 * y).ravel()))
        return terrain, points, (0., 0., 6., 6.), (.8, 3.), (5.2, 3.)
    terrain = directory / "rolling.t25d"
    write_map(terrain, [[height(i * .2, j * .2) for i in range(61)] for j in range(61)], spacing=(.2, .2),
              rectangles=[(5., 4.5, 6., 9.)], metadata=dict(test_fixture=True, purpose="sem_cpp_equivalence"))
    x, y = np.meshgrid(np.arange(.025, 12., .05), np.arange(.025, 12., .05))
    z = np.vectorize(height)(x, y)
    noisy = (x > 8.) & (x < 10.) & (y > 2.) & (y < 5.)
    z = z + noisy * np.random.default_rng(29).uniform(-.08, .08, z.shape)
    keep = ~((x > 2.5) & (x < 3.5) & (y > 8.) & (y < 9.))
    points = np.column_stack((x[keep], y[keep], z[keep]))
    return terrain, points, (0., 0., 12., 12.), (1., 6.), (11., 6.5)


def run_planner(planner, start, goal, seed, work):
    incumbents = []

    def callback(elapsed, path, raw, cost):
        incumbents.append(dict(cost=cost, path=path, raw=raw, components=copy.deepcopy(planner.last_components)))
    status = planner.solve(start, goal, seed, 3600., callback, work)
    return status, dict(planner.counters), incumbents


class PrimitiveTests(unittest.TestCase):
    def setUp(self):
        self.lib, self.dirs = load_library(LIBRARY, SEARCH)

    def tearDown(self):
        for directory in self.dirs:
            directory.close()

    def test_numpy_random_stream(self):
        for seed in (0, 1, 7, 11, 771, 416001, 616001, 2**32 - 1, 2**32, 2**63, 2**64 - 1):
            out = (C.c_double * 3000)()
            self.assertEqual(self.lib.sem_debug_random(seed, 3000, out), 1)
            np.testing.assert_array_equal(np.frombuffer(out, dtype=np.float64), np.random.default_rng(seed).random(3000))
            rng = np.random.default_rng(seed)
            expected = [rng.uniform(.0, 40.) for _ in range(400)]
            out = (C.c_double * 400)()
            self.lib.sem_debug_uniform(seed, 0., 40., 400, out)
            self.assertEqual(list(out), expected, seed)

    def test_cpython_math(self):
        rng = np.random.default_rng(5)
        for k in range(100000):
            a = rng.uniform(-50, 50, 4) * 10.0 ** int(rng.integers(-8, 4))
            if k % 5 == 0:
                a = np.round(a, 1)
            if k % 7 == 0:
                a[1] = a[3]
            x, y = float(a[0] - a[2]), float(a[1] - a[3])
            self.assertEqual(self.lib.sem_debug_hypot(x, y), math.hypot(x, y))
            self.assertEqual(self.lib.sem_debug_dist(*map(float, a)), math.dist((a[0], a[1]), (a[2], a[3])))
        for k in range(30000):
            n = int(rng.integers(0, 12))
            values = (rng.uniform(-1, 1, n) * 10.0 ** rng.integers(-20, 20, n)).tolist()
            if n > 3 and k % 3 == 0:
                values[1] = -values[0]
            self.assertEqual(self.lib.sem_debug_fsum((C.c_double * max(n, 1))(*values), n), math.fsum(values))

    def test_libm_matches_cpython(self):
        """The informed sampler's cos/sin/atan2 and the turn acos must be CPython's libm, bit for bit."""
        rng = np.random.default_rng(17)
        out = (C.c_double * 4)()
        for k in range(200000):
            x = float(rng.uniform(0, 2 * math.pi)) if k % 2 else float(rng.uniform(-60, 60))
            y, z = float(rng.uniform(-60, 60)), float(rng.uniform(-1, 1))
            self.assertEqual(self.lib.sem_debug_libm(x, y, z, out), 1)
            self.assertEqual(tuple(out), (math.cos(x), math.sin(x), math.atan2(y, x), math.acos(z)), (x, y, z))

    def test_turn(self):
        rng = np.random.default_rng(13)
        for k in range(50000):
            p = rng.uniform(0, 12, 6)
            if k % 9 == 0:
                p[2:4] = p[0:2]  # repeated vertex
            if k % 13 == 0:
                p[4:6] = 2 * p[2:4] - p[0:2]  # collinear continuation
            if k % 17 == 0:
                p[4:6] = p[0:2]  # full reversal
            a, b, c = (tuple(map(float, p[i:i + 2])) for i in (0, 2, 4))
            self.assertEqual(self.lib.sem_debug_turn(*a, *b, *c), _turn(a, b, c))


class ModelAndSolveTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(prefix="sem_cpp_")
        self.inputs = {}
        for kind in ("rolling", "plane"):
            terrain, points, bounds, start, goal = fixture(self.folder.name, kind)
            bridge = TerrainBridge(BRIDGE, terrain, SEARCH, risk_enabled=False)
            model = SEMNativeModel(points, bridge.bounds)
            self.inputs[kind] = dict(terrain=terrain, bridge=bridge, model=model, start=start, goal=goal,
                                     engine=SEMNativeCppEngine(LIBRARY, terrain, model, SEARCH))

    def tearDown(self):
        for item in self.inputs.values():
            item["engine"].close()
            item["bridge"].close()
        self.folder.cleanup()

    def test_model_edges_and_states(self):
        rng = np.random.default_rng(3)
        for kind, item in self.inputs.items():
            model, engine, high = item["model"], item["engine"], model_extent(item["model"])
            if kind == "rolling":  # the noisy patch and the cloud hole must create forbidden cells
                self.assertGreater(int((~model.valid).sum()), 100)
            for k in range(15000):
                p = rng.uniform(-.5, high + .5, 4)
                if k % 3 == 0:
                    p = np.round(p, 1)  # cell boundaries
                if k % 20 == 0:
                    p[2:] = p[:2]
                a, b = tuple(map(float, p[:2])), tuple(map(float, p[2:]))
                self.assertEqual(engine.model_edge(a, b), model.edge(a, b), (kind, a, b))
                self.assertEqual(engine.valid_state(a), model.valid_state(a), (kind, a))

    def test_full_solves_are_identical(self):
        cases = [("rolling", 771, 1500, True), ("rolling", 416001, 1500, False), ("plane", 616001, 1200, False),
                 ("rolling", 0, 40, False)]
        for kind, seed, work, exhaustive in cases:
            with self.subTest(kind=kind, seed=seed, work=work):
                item = self.inputs[kind]
                item["engine"].set_exhaustive_check(exhaustive)
                python = run_planner(SEMNativePlanner(item["bridge"], item["model"]), item["start"], item["goal"], seed, work)
                cpp_planner = SEMNativePlannerCpp(item["engine"])
                cpp = run_planner(cpp_planner, item["start"], item["goal"], seed, work)
                self.assertEqual(python[0], cpp[0])
                self.assertEqual(python[1], cpp[1])
                self.assertEqual(len(python[2]), len(cpp[2]))
                self.assertEqual(python[2], cpp[2])
                if work >= 1000:
                    self.assertEqual(python[0], "EXACT_SOLUTION")
                    self.assertGreater(python[1]["rewires"], 0)
                    self.assertGreater(python[1]["hard_rejections"] + python[1]["native_rejections"], 0)
        item = self.inputs["rolling"]
        item["engine"].set_exhaustive_check(False)

    def test_convergence_stop_is_an_exact_prefix(self):
        """The stop reads only the published cost: a stopped search equals the same search without the stop
        cut at the same iteration (C++ and Python), its reference is the cost after reference_iteration
        iterations, and no incumbent inside the window improved on it by relative_eps or more."""
        item = self.inputs["rolling"]
        rule = dict(relative_eps=1e-3, window_iterations=250)
        planner = SEMNativePlannerCpp(item["engine"], rule)
        stopped = run_planner(planner, item["start"], item["goal"], 771, 20000)
        outcome, iterations = planner.convergence, stopped[1]["iterations"]
        self.assertTrue(outcome["converged"])
        self.assertLess(iterations, 20000)
        self.assertEqual(iterations, outcome["reference_iteration"] + rule["window_iterations"])
        self.assertGreater(len(stopped[2]), 1)
        cut = run_planner(SEMNativePlannerCpp(item["engine"]), item["start"], item["goal"], 771, iterations)
        self.assertEqual(stopped, cut)
        python = run_planner(SEMNativePlanner(item["bridge"], item["model"]), item["start"], item["goal"], 771, iterations)
        self.assertEqual(python, cut)
        at_reference = run_planner(SEMNativePlannerCpp(item["engine"]), item["start"], item["goal"], 771,
                                   outcome["reference_iteration"])
        self.assertEqual(at_reference[2][-1]["cost"], outcome["reference_cost"])
        self.assertGreaterEqual(stopped[2][-1]["cost"], outcome["reference_cost"] * (1 - rule["relative_eps"]))
        # Off (the default) for planners built without it, on the same engine.
        plain = SEMNativePlannerCpp(item["engine"])
        self.assertEqual(run_planner(plain, item["start"], item["goal"], 771, iterations + 50)[1]["iterations"], iterations + 50)
        self.assertIsNone(plain.convergence)
        for bad in (dict(rule, relative_eps=0.), dict(rule, relative_eps=1.), dict(rule, window_iterations=0),
                    dict(rule, window_iterations=1.5), dict(relative_eps=1e-3)):
            with self.assertRaisesRegex(ValueError, "unregistered SEM convergence stop"):
                SEMNativePlannerCpp(item["engine"], bad)

    def test_invalid_endpoints_and_budget(self):
        item = self.inputs["rolling"]
        for start in ((5.5, 6.), (3., 8.5)):  # inside the obstacle; inside the cloud hole
            python = run_planner(SEMNativePlanner(item["bridge"], item["model"]), start, item["goal"], 1, 10)
            cpp = run_planner(SEMNativePlannerCpp(item["engine"]), start, item["goal"], 1, 10)
            self.assertEqual(python[:2], cpp[:2])
            self.assertIn(cpp[0], ("INVALID_HARD_ENDPOINTS", "INVALID_NATIVE_ENDPOINTS"))
        planner = SEMNativePlannerCpp(item["engine"])
        with self.assertRaises(ValueError):
            planner.solve(item["start"], item["goal"], 1, 0., lambda *a: None)
        with self.assertRaises(ValueError):
            planner.solve(item["start"], item["goal"], -1, 1., lambda *a: None)
        # A short wall-clock budget (no work limit) stops the C++ search on time.
        planner.solve(item["start"], item["goal"], 5, .2, lambda *a: None, 0)
        self.assertGreater(planner.counters["iterations"], 0)

    def test_incumbent_times_are_on_the_caller_clock(self):
        """Elapsed is read from the caller's perf_counter since the algorithm origin, as in SEMNativePlanner.
        The worker stamps each incumbent on that clock after copying it and rejects a next incumbent that
        reports an earlier time; a separate C++ clock lags it by the call overhead. A clock that jumps 1 ms
        per reading makes any such lag visible."""
        item, real, readings = self.inputs["plane"], time.perf_counter, [0]

        def jumping():
            readings[0] += 1
            return real() + 1e-3 * readings[0]
        with mock.patch.object(time, "perf_counter", jumping):
            origin, seen = time.perf_counter(), []

            def callback(elapsed, path, raw, cost):
                seen.append((elapsed, time.perf_counter() - origin))
            SEMNativePlannerCpp(item["engine"]).solve(item["start"], item["goal"], 616001, 60., callback, 300,
                                                      algorithm_started_at=origin)
        self.assertGreater(len(seen), 1)
        for (_, returned), (elapsed, _) in zip(seen, seen[1:]):
            self.assertGreaterEqual(elapsed, returned)

    def test_callback_error_aborts(self):
        item = self.inputs["rolling"]
        planner = SEMNativePlannerCpp(item["engine"])

        def failing(*_):
            raise RuntimeError("synthetic callback failure")
        with self.assertRaisesRegex(RuntimeError, "synthetic callback failure"):
            planner.solve(item["start"], item["goal"], 771, 60., failing, 1500)
        self.assertLess(planner.counters["iterations"], 1500)


def model_extent(model):
    return model.bounds[2]


class WorkerTests(unittest.TestCase):
    def test_worker_engines_agree_and_pass_strict_validation(self):
        import native_paper_worker
        from pri_portfolio_validation import validate_trial
        with tempfile.TemporaryDirectory(prefix="sem_cpp_worker_") as folder:
            folder = Path(folder)
            terrain, points, bounds, start, goal = fixture(folder, "rolling")
            cloud = folder / "cloud.npy"
            np.save(cloud, points, allow_pickle=False)
            info = dict(map_path=str(terrain), map_sha256=file_hash(terrain), cloud_path=str(cloud),
                        cloud_sha256=file_hash(cloud), bounds=list(bounds))
            outputs = {}
            for engine in ("python", "cpp"):
                request = native_request("sem", dict(start=list(start), goal=list(goal)), terrain, cloud,
                                         info["cloud_sha256"], 771, 120., native_config())
                request.update(map_sha256=info["map_sha256"], bridge_library=str(Path(BRIDGE).resolve()),
                               library_search_paths=[str(Path(BRIDGE).resolve().parent), *SEARCH],
                               variant_id="sem_native", algorithm_budget_policy="pri_portfolio_v1", work_iterations=900)
                if engine == "cpp":
                    request.update(sem_engine="cpp", sem_library=str(Path(LIBRARY).resolve()),
                                   sem_library_sha256=file_hash(LIBRARY))
                out = folder / engine
                out.mkdir()
                (out / "request.json").write_text(json.dumps(request), encoding="utf-8")
                self.assertEqual(native_paper_worker.run(out / "request.json", out), 0)
                worker = json.loads((out / "worker.json").read_text(encoding="utf-8"))
                self.assertEqual(worker["status"], "EXACT_SOLUTION", worker.get("error"))
                execution = dict(status="COMPLETED", launched=True, field_build_s=0.)
                row = validate_trial(out, request, info, execution, oracle=NativePaperOracle(info["map_path"]),
                                     field=SharedSEMField(points, bounds))
                self.assertEqual(row["correctness_errors"], [], engine)
                outputs[engine] = worker
            python, cpp = outputs["python"], outputs["cpp"]
            self.assertEqual(cpp["provenance"]["sem_engine"], "cpp")
            self.assertNotIn("sem_engine", python["provenance"])
            self.assertEqual(python["counters"], cpp["counters"])
            keys = ("path_xy", "raw_path_xyz", "native_objective_value", "native_components")
            self.assertEqual([{k: s[k] for k in keys} for s in python["snapshots"]],
                             [{k: s[k] for k in keys} for s in cpp["snapshots"]])

    def test_request_validation(self):
        import native_paper_worker
        base = dict(backend="sem", track="native", planner_id="sem_native_paper_v2",
                    native_objective_id="sem_length_turn_risk_paper_v2", start=[1., 1.], goal=[2., 2.],
                    planner_seed=1, planning_budget_s=1., map_path="m", cloud_path="c", cloud_sha256="h",
                    bridge_library="b", representation_config={})
        putn = dict(backend="putn", planner_id="putn_native_paper_v2", native_objective_id="putn_com_traversability_paper_v2")
        for change in (dict(sem_engine="rust"), dict(sem_engine="cpp"), dict(putn, sem_engine="cpp", sem_library="x")):
            with self.assertRaisesRegex(ValueError, "unregistered SEM search engine"):
                native_paper_worker.validate_request(dict(base, **change))
        # The convergence stop needs the C++ engine and a valid rule (checked before any file is read).
        stop = dict(relative_eps=1e-3, window_iterations=4000)
        cpp = dict(sem_engine="cpp", sem_library=str(Path(LIBRARY).resolve()))
        for change in (dict(sem_convergence_stop=stop), dict(cpp, sem_convergence_stop=None),
                       dict(cpp, sem_convergence_stop=dict(stop, relative_eps=2.)),
                       dict(cpp, sem_convergence_stop=dict(stop, window_iterations=-1))):
            with self.assertRaisesRegex(ValueError, "unregistered SEM convergence stop"):
                native_paper_worker.validate_request(dict(base, **change))

    def test_worker_records_the_convergence_stop(self):
        import native_paper_worker
        with tempfile.TemporaryDirectory(prefix="sem_cpp_stop_") as folder:
            folder = Path(folder)
            terrain, points, bounds, start, goal = fixture(folder, "rolling")
            cloud = folder / "cloud.npy"
            np.save(cloud, points, allow_pickle=False)
            stop = dict(relative_eps=1e-3, window_iterations=250)
            request = native_request("sem", dict(start=list(start), goal=list(goal)), terrain, cloud, file_hash(cloud),
                                     771, 120., native_config())
            request.update(map_sha256=file_hash(terrain), bridge_library=str(Path(BRIDGE).resolve()),
                           library_search_paths=[str(Path(BRIDGE).resolve().parent), *SEARCH], variant_id="sem_native",
                           algorithm_budget_policy="pri_portfolio_v1", work_iterations=20000, sem_engine="cpp",
                           sem_library=str(Path(LIBRARY).resolve()), sem_library_sha256=file_hash(LIBRARY),
                           sem_convergence_stop=stop)
            (folder / "request.json").write_text(json.dumps(request), encoding="utf-8")
            self.assertEqual(native_paper_worker.run(folder / "request.json", folder), 0)
            worker = json.loads((folder / "worker.json").read_text(encoding="utf-8"))
            self.assertEqual((worker["status"], worker["termination_reason"]), ("EXACT_SOLUTION", "CONVERGED_NO_IMPROVEMENT"))
            outcome = worker["sem_convergence"]
            self.assertEqual(outcome["rule"], stop)
            self.assertEqual(worker["counters"]["iterations"], outcome["reference_iteration"] + 250)
            last = worker["snapshots"][-1]["native_objective_value"]
            self.assertTrue(outcome["reference_cost"] * (1 - 1e-3) <= last <= outcome["reference_cost"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--bridge", required=True)
    args, rest = parser.parse_known_args()
    LIBRARY, BRIDGE = str(Path(args.library).resolve()), str(Path(args.bridge).resolve())
    unittest.main(argv=[sys.argv[0], *rest])
