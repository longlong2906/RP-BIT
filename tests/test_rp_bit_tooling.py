"""A4 / RP-BIT* protocol, runner guard and analysis tests; no planner is launched."""
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import analyze_rp_bit as analysis
import rp_bit_protocol as protocol
import run_rp_bit as runner
from corridor_v4_execution import digest, file_hash as digest_file

HELD_OUT_SEEDS = set(range(61001, 61011))
STOP = dict(relative_eps=1e-3, window_iterations=4000)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.config = protocol.load_config()
        self.info = dict(map_path=str(ROOT / "tests/fixtures/map.t25d"), cloud_path=str(ROOT / "tests/fixtures/cloud.npy"),
                         cloud_sha256="cloud", map_sha256="map", start=[2., 10.], goal=[18., 10.])
        self.field = dict(field_sha256="field", cloud_sha256="cloud", builder_source_sha256="builder")
        self.arguments = dict(field_path=ROOT / "tests/fixtures/planner_field.txt", field_metadata=self.field,
                              bridge=ROOT / "tests/fixtures/libbridge.dll")

    def test_schedule_is_paired_and_rotated(self):
        entries = protocol.inventory(self.config, "smoke")
        methods = self.config["phases"]["smoke"]["methods"]
        self.assertEqual(len(entries), 7 * len(methods))
        self.assertEqual(len({entry["trial_id"] for entry in entries}), len(entries))
        for block in range(7):
            rows = [entry for entry in entries if entry["block_index"] == block]
            self.assertEqual(len({(row["map_id"], row["planner_seed"]) for row in rows}), 1)
            offset = block % len(methods)
            self.assertEqual([row["method"] for row in rows], methods[offset:] + methods[:offset])
        dev = protocol.inventory(self.config, "dev_sem")
        self.assertEqual(len(dev), 75)
        self.assertEqual({entry["map_seed"] for entry in dev}, {41001, 41002, 41003})
        self.assertEqual({entry["family"] for entry in dev}, set(protocol.FAMILIES))

    def test_held_out_seeds_only_in_held_out_phases(self):
        for name, phase in self.config["phases"].items():
            if HELD_OUT_SEEDS & set(phase["map_seeds"]):
                self.assertTrue(name.startswith("heldout"), name)
                self.assertEqual(phase["maps"], "generate")
            elif phase["maps"] == "dev":
                self.assertLessEqual(set(phase["map_seeds"]), {41001, 41002, 41003})

    def test_requests(self):
        by_method = {entry["method"]: entry for entry in protocol.inventory(self.config, "smoke")}
        native = protocol.request_for(by_method[protocol.NATIVE], self.info, self.config, **self.arguments)
        self.assertEqual((native["variant_id"], native["planning_budget_s"], native["planner_seed"]), (protocol.NATIVE, 2., 11))
        self.assertNotIn("rp_bit", native)
        self.assertNotIn("sem_exposure", native)
        for name, entry in by_method.items():
            method = self.config["methods"][name]
            if method["kind"] == "sem_native":
                continue
            request = protocol.request_for(entry, self.info, self.config, **self.arguments)
            self.assertEqual(request["objective_type"], "surface_length_sem_exposure")
            self.assertEqual(request["sem_exposure"], {"lambda": method["lambda"], "risk_scale": 40.})
            self.assertEqual((request["planner_id"], request["variant_id"], request["planner_seed"]), ("bitstar_a3", name, 11))
            self.assertEqual((request["samples_per_batch"], request["batch_after_first_solution"]), (100, 200))
            self.assertEqual((request["safe_length"]["risk_cap"], request["safe_length"]["slope_cap_deg"]), (40., 35.))
            self.assertNotIn("work_iterations", request)
            if method["kind"] == "rp_bit":
                self.assertEqual(request["rp_bit"], {k: v for k, v in method.items() if k not in ("kind", "lambda")})
                self.assertIn("refine", request["rp_bit"])
            else:
                self.assertNotIn("rp_bit", request)

    def test_cpp_sem_engine_request(self):
        """engine=cpp only adds the C++ port (next to the bridge, hash-pinned) to the SEM request."""
        entry = next(e for e in protocol.inventory(self.config, "smoke") if e["method"] == protocol.NATIVE)
        default = protocol.request_for(entry, self.info, self.config, **self.arguments)
        self.assertNotIn("sem_engine", default)
        config = copy.deepcopy(self.config)
        config["methods"][protocol.NATIVE]["engine"] = "cpp"
        with tempfile.TemporaryDirectory() as folder:
            bridge, library = Path(folder) / "libterrain_compare_c.dll", Path(folder) / "libsem_rrtstar_c.dll"
            bridge.write_bytes(b"bridge")
            library.write_bytes(b"port")
            request = protocol.request_for(entry, self.info, config, **dict(self.arguments, bridge=bridge))
            self.assertEqual((request["sem_engine"], Path(request["sem_library"]).name), ("cpp", library.name))
            self.assertEqual(request["sem_library_sha256"], digest_file(library))
            plain = protocol.request_for(entry, self.info, self.config, **dict(self.arguments, bridge=bridge))
            self.assertEqual({k: v for k, v in request.items() if not k.startswith("sem_")}, plain)
            self.assertNotIn("sem_convergence_stop", request)
            # The convergence stop only adds its rule to the C++ SEM request.
            config["methods"][protocol.NATIVE]["convergence_stop"] = dict(STOP)
            stopped = protocol.request_for(entry, self.info, config, **dict(self.arguments, bridge=bridge))
            self.assertEqual(stopped, dict(request, sem_convergence_stop=STOP))

    def test_invalid_configs(self):
        for mutate in (lambda c: c.update(protocol_id="other"),
                       lambda c: c["methods"].update(extra=dict(kind="unknown")),
                       lambda c: c["methods"].update(extra=dict(kind="sem_native")),
                       lambda c: c["phases"]["smoke"].update(methods=["unregistered"]),
                       lambda c: c["phases"]["smoke"].update(maps="elsewhere"),
                       lambda c: c["methods"]["sem_native"].update(engine="rust"),
                       lambda c: c["methods"]["a4_rpbit_l30"].update(engine="cpp"),
                       # The convergence stop exists only on the C++ SEM engine, with a valid rule.
                       lambda c: c["methods"]["sem_native"].update(convergence_stop=STOP),
                       lambda c: c["methods"]["sem_native"].update(engine="cpp", convergence_stop=dict(STOP, relative_eps=0.)),
                       lambda c: c["methods"]["sem_native"].update(engine="cpp", convergence_stop=dict(STOP, window_iterations=4e3)),
                       lambda c: c["methods"]["sem_native"].update(engine="cpp", convergence_stop=dict(STOP, extra=1)),
                       lambda c: c["methods"]["a4_rpbit_l30"].update(convergence_stop=STOP)):
            config = copy.deepcopy(self.config)
            mutate(config)
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "config.json"
                path.write_text(json.dumps(config), encoding="utf-8")
                with self.assertRaises(ValueError):
                    protocol.load_config(path)

    def test_phase_hash_ignores_other_phases_and_guards_held_out(self):
        frozen = digest(runner.phase_config(self.config, "smoke"))
        config = copy.deepcopy(self.config)
        config["methods"]["later"] = dict(kind="rp_bit", refine=True, **{"lambda": 24.})
        config["phases"]["heldout"] = dict(maps="generate", map_seeds=[61001], planner_seeds=[1],
                                           planning_budget_s=20., methods=["sem_native", "later"])
        self.assertEqual(digest(runner.phase_config(config, "smoke")), frozen)
        changed = copy.deepcopy(self.config)
        changed["phases"]["smoke"]["planning_budget_s"] = 3.
        self.assertNotEqual(digest(runner.phase_config(changed, "smoke")), frozen)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "held-out requires"):
                runner.run(Path(folder) / "out", "heldout", "missing.exe", "missing.dll", config_path=path)
            with self.assertRaisesRegex(SystemExit, "unregistered phase"):
                runner.run(Path(folder) / "out", "nowhere", "missing.exe", "missing.dll", config_path=path)
            self.assertFalse((Path(folder) / "out").exists())

    def test_stop_dispatch_deadline_leaves_trials_pending(self):
        """A past soft deadline starts no trial, skips validation and keeps the phase resumable."""
        original_maps, original_preflight = runner._maps, runner._memory_preflight
        runner._maps = lambda config, phase, directory: {f"{family}_{seed}": dict(map_id=f"{family}_{seed}")
                                                         for family, seed in protocol.maps_for(config, phase)}
        runner._memory_preflight = lambda config: None
        try:
            with tempfile.TemporaryDirectory() as folder:
                worker, bridge = Path(folder) / "worker.bin", Path(folder) / "bridge.bin"
                worker.write_bytes(b"w")
                bridge.write_bytes(b"b")
                out = Path(folder) / "out"
                self.assertEqual(runner.run(out, "smoke", worker, bridge, stop_dispatch_epoch=0.), 3)
                manifest = json.loads((out / "smoke" / "manifest.json").read_text(encoding="utf-8"))
                session = manifest["sessions"][-1]
                self.assertEqual((session["collection_status"], session["unstarted"]), ("DEADLINE", 28))
                self.assertFalse((out / "smoke" / "trials").exists())
                self.assertFalse((out / "smoke" / "summary.json").exists())
                with self.assertRaisesRegex(SystemExit, "phase already exists"):
                    runner.run(out, "smoke", worker, bridge, stop_dispatch_epoch=0.)
                self.assertEqual(runner.run(out, "smoke", worker, bridge, resume=True, stop_dispatch_epoch=0.), 3)
        finally:
            runner._maps, runner._memory_preflight = original_maps, original_preflight


class KaggleConfigTests(unittest.TestCase):
    """The Kaggle held-out config is the locked config plus the recorded deviations, nothing else."""

    def setUp(self):
        self.locked, self.kaggle = protocol.load_config(), protocol.load_config(protocol.KAGGLE_CONFIG)
        self.info = dict(map_path=str(ROOT / "tests/fixtures/map.t25d"), cloud_path=str(ROOT / "tests/fixtures/cloud.npy"),
                         cloud_sha256="cloud", map_sha256="map", start=[2., 10.], goal=[18., 10.])
        self.arguments = dict(field_path=ROOT / "tests/fixtures/planner_field.txt",
                              field_metadata=dict(field_sha256="field", cloud_sha256="cloud", builder_source_sha256="builder"),
                              bridge=ROOT / "tests/fixtures/libbridge.dll")

    def test_only_registered_deviations(self):
        locked, kaggle = self.locked, self.kaggle
        for key in ("protocol_id", "schema_version", "safety", "analysis", "methods"):
            self.assertEqual(kaggle[key], locked[key], key)
        self.assertEqual(kaggle["execution"], dict(locked["execution"], cores=[0, 1, 2, 3]))
        self.assertEqual(set(kaggle["phases"]), set(locked["phases"]) - {"heldout"} | {"heldout_kaggle", "kaggle_gate"})
        for name in set(locked["phases"]) - {"heldout"}:
            self.assertEqual(kaggle["phases"][name], locked["phases"][name], name)
        held = dict(locked["phases"]["heldout"], sem_work_iterations=7425, sem_budget_cap_s=180.)
        self.assertEqual(kaggle["phases"]["heldout_kaggle"], held)
        self.assertEqual(kaggle["phases"]["kaggle_gate"], dict(held, map_seeds=[51001], planner_seeds=[516001]))
        before, after = protocol.inventory(locked, "heldout"), protocol.inventory(kaggle, "heldout_kaggle")
        self.assertEqual(len(after), 1500)
        self.assertEqual(after, [dict(entry, phase="heldout_kaggle") for entry in before])
        for name, phase in kaggle["phases"].items():
            if HELD_OUT_SEEDS & set(phase["map_seeds"]):
                self.assertTrue(name.startswith("heldout"), name)

    def test_fixed_work_sem_request_and_unchanged_others(self):
        old = {entry["method"]: entry for entry in protocol.inventory(self.locked, "heldout")[:6]}
        new = {entry["method"]: entry for entry in protocol.inventory(self.kaggle, "heldout_kaggle")[:6]}
        for name in new:
            before = protocol.request_for(old[name], self.info, self.locked, **self.arguments)
            after = protocol.request_for(new[name], self.info, self.kaggle, **self.arguments)
            if name == protocol.NATIVE:
                self.assertEqual((before["work_iterations"], before["planning_budget_s"]), (0, 20.))
                self.assertEqual((after["work_iterations"], after["planning_budget_s"]), (7425, 180.))
                after.update(work_iterations=0, planning_budget_s=20.)
            self.assertEqual(after, before, name)
        smoke = next(e for e in protocol.inventory(self.kaggle, "smoke_final") if e["method"] == protocol.NATIVE)
        request = protocol.request_for(smoke, self.info, self.kaggle, **self.arguments)
        self.assertEqual((request["work_iterations"], request["planning_budget_s"]), (0, 2.))

    def test_invalid_sem_work_budget(self):
        for change in (dict(sem_work_iterations=7425), dict(sem_budget_cap_s=180.),
                       dict(sem_work_iterations=0, sem_budget_cap_s=180.), dict(sem_work_iterations=7425., sem_budget_cap_s=180.),
                       dict(sem_work_iterations=7425, sem_budget_cap_s=10.), dict(sem_work_iterations=7425, sem_budget_cap_s=180)):
            config = copy.deepcopy(self.locked)
            config["phases"]["heldout"].update(change)
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "config.json"
                path.write_text(json.dumps(config), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "invalid SEM work budget"):
                    protocol.load_config(path)

    def test_kaggle_held_out_still_needs_the_flag(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(SystemExit, "held-out requires"):
                runner.run(Path(folder) / "out", "heldout_kaggle", "missing", "missing", config_path=protocol.KAGGLE_CONFIG)
            self.assertFalse((Path(folder) / "out").exists())


class FairtimeConfigTests(unittest.TestCase):
    """Same-time Kaggle run: the locked primary unchanged, SEM-RRT* on the C++ engine, fresh seeds."""

    def setUp(self):
        self.locked, self.fair = protocol.load_config(), protocol.load_config(protocol.FAIRTIME_CONFIG)
        self.info = dict(map_path=str(ROOT / "tests/fixtures/map.t25d"), cloud_path=str(ROOT / "tests/fixtures/cloud.npy"),
                         cloud_sha256="cloud", map_sha256="map", start=[2., 10.], goal=[18., 10.])
        self.arguments = dict(field_path=ROOT / "tests/fixtures/planner_field.txt",
                              field_metadata=dict(field_sha256="field", cloud_sha256="cloud", builder_source_sha256="builder"))

    def test_only_the_engine_cores_and_seeds_change(self):
        locked, fair = self.locked, self.fair
        for key in ("protocol_id", "schema_version", "safety", "analysis"):
            self.assertEqual(fair[key], locked[key], key)
        self.assertEqual(fair["execution"], dict(locked["execution"], cores=[0]))
        primary = locked["phases"]["dev_a4_balance"]["methods"][0]
        self.assertEqual(primary, "a4_rpbit_l33_m20")
        self.assertEqual(fair["methods"], {protocol.NATIVE: dict(kind="sem_native", engine="cpp"),
                                           primary: locked["methods"][primary]})
        held = fair["phases"]["heldout_fairtime"]
        self.assertEqual((held["primary_method"], held["planning_budget_s"], held["methods"]), (primary, 20., [protocol.NATIVE, primary]))
        self.assertNotIn("sem_work_iterations", held)
        self.assertEqual(len(protocol.inventory(fair, "heldout_fairtime")), 500)
        fresh = set(held["map_seeds"]) | set(held["planner_seeds"])
        self.assertEqual(fresh, set(range(71001, 71011)) | set(range(716001, 716006)))
        for config in (locked, protocol.load_config(protocol.KAGGLE_CONFIG), fair):
            for name, phase in config["phases"].items():
                if name != "heldout_fairtime":
                    self.assertFalse(fresh & (set(phase["map_seeds"]) | set(phase["planner_seeds"])), name)

    def test_requests(self):
        with tempfile.TemporaryDirectory() as folder:
            bridge, library = Path(folder) / "libterrain_compare_c.so", Path(folder) / "libsem_rrtstar_c.so"
            bridge.write_bytes(b"bridge")
            library.write_bytes(b"port")
            entries = {e["method"]: e for e in protocol.inventory(self.fair, "heldout_fairtime")[:2]}
            sem = protocol.request_for(entries[protocol.NATIVE], self.info, self.fair, bridge=bridge, **self.arguments)
            self.assertEqual((sem["planning_budget_s"], sem["work_iterations"], sem["sem_engine"]), (20., 0, "cpp"))
            self.assertEqual((Path(sem["sem_library"]).name, sem["sem_library_sha256"]), (library.name, digest_file(library)))
            # The primary A4 request is the one the locked config builds for the same slot.
            a4 = entries["a4_rpbit_l33_m20"]
            self.assertEqual(protocol.request_for(a4, self.info, self.fair, bridge=bridge, **self.arguments),
                             protocol.request_for(a4, self.info, self.locked, phase="heldout", bridge=bridge, **self.arguments))

    def test_held_out_needs_the_flag(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(SystemExit, "held-out requires"):
                runner.run(Path(folder) / "out", "heldout_fairtime", "missing", "missing", config_path=protocol.FAIRTIME_CONFIG)
            self.assertFalse((Path(folder) / "out").exists())


class SelfstopConfigTests(unittest.TestCase):
    """Self-stop Kaggle run: the fairtime design, SEM-RRT* also stopping on convergence by the A4 rule, fresh seeds."""

    def test_only_the_sem_stop_and_seeds_change(self):
        fair, stop = protocol.load_config(protocol.FAIRTIME_CONFIG), protocol.load_config(protocol.SELFSTOP_CONFIG)
        for key in ("protocol_id", "schema_version", "safety", "analysis", "execution"):
            self.assertEqual(stop[key], fair[key], key)
        self.assertEqual(stop["methods"], dict(fair["methods"], **{protocol.NATIVE: dict(
            fair["methods"][protocol.NATIVE], convergence_stop=STOP)}))
        # A4 rule: 20 batches of 200 samples after the first solution = 4 000 samples; same relative epsilon.
        locked = protocol.load_config()["methods"]["a4_rpbit_l33_m20"]
        self.assertEqual(STOP["window_iterations"], locked["stop_batches"] * 200)
        renamed = dict(smoke_fairtime="smoke_selfstop", gate_fairtime="gate_selfstop", heldout_fairtime="heldout_selfstop")
        self.assertEqual(set(stop["phases"]), set(renamed.values()))
        for old, new in renamed.items():
            changed = {k for k in stop["phases"][new] if stop["phases"][new][k] != fair["phases"][old][k]}
            self.assertEqual(changed, {"map_seeds", "planner_seeds"} if new == "heldout_selfstop" else set(), new)
        held = stop["phases"]["heldout_selfstop"]
        fresh = set(held["map_seeds"]) | set(held["planner_seeds"])
        self.assertEqual(fresh, set(range(81001, 81011)) | set(range(816001, 816006)))
        self.assertEqual(len(protocol.inventory(stop, "heldout_selfstop")), 500)
        dev = ROOT / "configs/a4_rpbitstar_v1_selfstop_dev.json"
        for config in (protocol.load_config(), protocol.load_config(protocol.KAGGLE_CONFIG), fair, stop, protocol.load_config(dev)):
            for name, phase in config["phases"].items():
                if name != "heldout_selfstop":
                    self.assertFalse(fresh & (set(phase["map_seeds"]) | set(phase["planner_seeds"])), name)

    def test_sem_request_carries_the_rule(self):
        stop = protocol.load_config(protocol.SELFSTOP_CONFIG)
        info = dict(map_path=str(ROOT / "tests/fixtures/map.t25d"), cloud_path=str(ROOT / "tests/fixtures/cloud.npy"),
                    cloud_sha256="cloud", map_sha256="map", start=[2., 10.], goal=[18., 10.])
        with tempfile.TemporaryDirectory() as folder:
            bridge, library = Path(folder) / "libterrain_compare_c.so", Path(folder) / "libsem_rrtstar_c.so"
            bridge.write_bytes(b"bridge")
            library.write_bytes(b"port")
            entry = next(e for e in protocol.inventory(stop, "heldout_selfstop") if e["method"] == protocol.NATIVE)
            sem = protocol.request_for(entry, info, stop, bridge=bridge)
            self.assertEqual((sem["planning_budget_s"], sem["work_iterations"], sem["sem_engine"], sem["sem_convergence_stop"]),
                             (20., 0, "cpp", STOP))

    def test_held_out_needs_the_flag(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(SystemExit, "held-out requires"):
                runner.run(Path(folder) / "out", "heldout_selfstop", "missing", "missing", config_path=protocol.SELFSTOP_CONFIG)
            self.assertFalse((Path(folder) / "out").exists())


class ValidationEngineTests(unittest.TestCase):
    """A SEM trial must have run on the registered engine (and the exact C++ library)."""

    def verdict(self, engine, provenance):
        from rp_bit_validation import validate_trial
        config = dict(methods={protocol.NATIVE: dict(kind="sem_native", **({"engine": engine} if engine else {}))},
                      analysis=dict(sem_equivalent_lambda=30., checkpoints_s=[1.]))
        entry = dict(trial_id="m__s1__sem_native", method=protocol.NATIVE)
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            for name, body in (("execution.json", dict(status="COMPLETED")),
                               ("request.json", dict(planning_budget_s=20., sem_library_sha256="lib")),
                               ("worker.json", dict(snapshots=[], planner_status="EXACT_SOLUTION", provenance=provenance))):
                (directory / name).write_text(json.dumps(body), encoding="utf-8")
            return validate_trial(directory, entry, {}, config, oracle=None, field=None)["correctness_errors"]

    def test_engine_mismatch_is_a_correctness_error(self):
        self.assertEqual(self.verdict(None, {}), [])
        self.assertEqual(self.verdict("cpp", dict(sem_engine="cpp", sem_library_sha256="lib")), [])
        self.assertEqual(self.verdict("cpp", {}), ["sem_engine_mismatch"])
        self.assertEqual(self.verdict("cpp", dict(sem_engine="cpp", sem_library_sha256="other")), ["sem_engine_mismatch"])
        self.assertEqual(self.verdict(None, dict(sem_engine="cpp")), ["sem_engine_mismatch"])


class ValidationConvergenceTests(unittest.TestCase):
    """A SEM trial must have run the registered convergence stop, and a converged stop must obey it."""

    def verdict(self, rule, requested, worker_update):
        import rp_bit_validation as validation
        method = dict(kind="sem_native", engine="cpp", **({"convergence_stop": rule} if rule else {}))
        config = dict(methods={protocol.NATIVE: method}, analysis=dict(sem_equivalent_lambda=30., checkpoints_s=[1.]))
        entry = dict(trial_id="m__s1__sem_native", method=protocol.NATIVE)
        request = dict(planning_budget_s=20., sem_library_sha256="lib", start=[0., 0.], goal=[1., 1.],
                       **({"sem_convergence_stop": requested} if requested else {}))
        worker = dict(planner_status="EXACT_SOLUTION", provenance=dict(sem_engine="cpp", sem_library_sha256="lib"),
                      snapshots=[dict(snapshot_id="a", elapsed_s=.2, native_objective_value=100.),
                                 dict(snapshot_id="b", elapsed_s=.4, native_objective_value=99.95)],
                      counters=dict(iterations=5000), termination_reason="CONVERGED_NO_IMPROVEMENT",
                      sem_convergence=dict(converged=True, reference_cost=100., reference_iteration=1000, rule=STOP))
        worker.update(worker_update)
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            for name, body in (("execution.json", dict(status="COMPLETED")), ("request.json", request), ("worker.json", worker)):
                (directory / name).write_text(json.dumps(body), encoding="utf-8")
            with mock.patch.object(validation, "score_path", lambda *args: dict(safe=True)), \
                    mock.patch.object(validation, "_csv_path", lambda path: []):
                return validation.validate_trial(directory, entry, {}, config, oracle=None, field=None)["correctness_errors"]

    def test_rule_and_stop_point(self):
        self.assertEqual(self.verdict(STOP, STOP, {}), [])
        # Stopped by the time cap instead: nothing to check beyond the rule.
        self.assertEqual(self.verdict(STOP, STOP, dict(termination_reason="TIME_BUDGET", counters=dict(iterations=7),
            sem_convergence=dict(converged=False, reference_cost=100., reference_iteration=3, rule=STOP))), [])
        self.assertEqual(self.verdict(None, None, dict(sem_convergence=None)), [])
        mismatch = ["sem_convergence_stop_mismatch"]
        self.assertEqual(self.verdict(STOP, STOP, dict(sem_convergence=None)), mismatch)
        self.assertEqual(self.verdict(STOP, None, {}), mismatch)
        self.assertEqual(self.verdict(None, None, {}), mismatch)
        self.assertEqual(self.verdict(STOP, dict(STOP, window_iterations=10), {}), mismatch)
        inconsistent = ["sem_convergence_inconsistent"]
        self.assertEqual(self.verdict(STOP, STOP, dict(counters=dict(iterations=4999))), inconsistent)
        self.assertEqual(self.verdict(STOP, STOP, dict(termination_reason="TIME_BUDGET")), inconsistent)
        # An improvement of 0.1% or more inside the window should have reset the reference.
        self.assertEqual(self.verdict(STOP, STOP, dict(snapshots=[dict(snapshot_id="a", elapsed_s=.2,
            native_objective_value=99.8)])), inconsistent)


def legacy_inventory(config, phase):
    """The schedule before derived methods existed: every method rotates."""
    settings, entries, block = config["phases"][phase], [], 0
    methods = settings["methods"]
    for family, map_seed in protocol.maps_for(config, phase):
        for planner_seed in settings["planner_seeds"]:
            offset = block % len(methods)
            for method in methods[offset:] + methods[:offset]:
                entries.append(dict(trial_id=f"{family}_{map_seed}__s{planner_seed}__{method}", map_id=f"{family}_{map_seed}",
                                    family=family, map_seed=map_seed, planner_seed=planner_seed, query_id="q0",
                                    method=method, phase=phase, block_index=block))
            block += 1
    return entries


class FactorialConfigTests(unittest.TestCase):
    """Refine attribution: {SEM-RRT*, plain BIT*, A4} x rpRefine {off, on}; sem_refine replays its slot's SEM trial."""

    def setUp(self):
        self.config = protocol.load_config(protocol.FACTORIAL_DEV_CONFIG)

    def test_arms_are_the_registered_ones(self):
        methods, locked = self.config["methods"], protocol.load_config()["methods"]
        for name in ("a4_rpbit_l33_m20", "a4_rpbit_l33_m20_norefine", "a4_rpbit_l33_m20_nopotential"):
            self.assertEqual(methods[name], locked[name], name)
        self.assertEqual(methods["a4_bit_plain_l33_m20"], dict(locked["a4_rpbit_l33_m20_nopotential"], refine=False))
        self.assertEqual(methods[protocol.NATIVE], protocol.load_config(protocol.SELFSTOP_CONFIG)["methods"][protocol.NATIVE])
        # The SEM post-processing is A4's own refinement at A4's lambda.
        self.assertEqual(methods["sem_refine"]["lambda"], locked["a4_rpbit_l33_m20"]["lambda"])
        self.assertEqual(self.config["analysis"]["decision"]["contrast"], ["a4_rpbit_l33_m20", "sem_refine"])
        dev = protocol.load_config(ROOT / "configs/a4_rpbitstar_v1_selfstop_dev.json")
        strip = lambda phase: {k: v for k, v in phase.items() if k != "methods"}
        self.assertEqual(strip(self.config["phases"]["dev_factorial"]), strip(dev["phases"]["dev_selfstop"]))
        self.assertEqual(self.config["execution"], dev["execution"])

    def test_derived_trial_follows_its_source(self):
        entries = protocol.inventory(self.config, "dev_factorial")
        self.assertEqual(len(entries), 450)
        base = [m for m in self.config["phases"]["dev_factorial"]["methods"] if m != "sem_refine"]
        for block in range(75):
            order = [entry["method"] for entry in entries if entry["block_index"] == block]
            offset = block % len(base)
            self.assertEqual([m for m in order if m != "sem_refine"], base[offset:] + base[:offset])
            self.assertEqual(order[order.index("sem_refine") - 1], protocol.NATIVE)

    def test_schedules_without_derived_methods_are_unchanged(self):
        for path in sorted((ROOT / "configs").glob("a4_rpbitstar_v1*.json")):
            config = protocol.load_config(path)
            for phase, settings in config["phases"].items():
                if not any(config["methods"][m]["kind"] == protocol.DERIVED for m in settings["methods"]):
                    self.assertEqual(protocol.inventory(config, phase), legacy_inventory(config, phase), (path.name, phase))

    def test_invalid_derived_configs(self):
        for mutate in (lambda c: c["methods"]["sem_refine"].update(extra=1),
                       lambda c: c["methods"]["sem_refine"].update(source="a4_rpbit_l33_m20"),
                       lambda c: c["methods"]["sem_refine"].pop("source"),
                       lambda c: c["methods"]["sem_refine"].update(refine_at_s=[1, 2.]),
                       lambda c: c["methods"]["sem_refine"].update(refine_at_s=[2., 1.]),
                       lambda c: c["methods"]["sem_refine"].update(refine_at_s=[0., 1.]),
                       lambda c: c["methods"]["sem_refine"].update({"lambda": 33}),
                       lambda c: c["methods"]["sem_refine"].update(engine="cpp"),
                       lambda c: c["phases"]["dev_factorial"]["methods"].remove(protocol.NATIVE),
                       lambda c: c["analysis"]["contrasts"].append(["a4_rpbit_l33_m20", "unknown"]),
                       lambda c: c["analysis"]["contrasts"].append(["a4_rpbit_l33_m20", "a4_rpbit_l33_m20"]),
                       lambda c: c["analysis"]["contrasts"].append(["a4_rpbit_l33_m20", "sem_refine"]),
                       lambda c: c["analysis"]["decision"].update(contrast=["sem_refine", "sem_native"]),
                       lambda c: c["analysis"]["decision"].update(rule="other"),
                       lambda c: c["analysis"].pop("contrasts")):
            config = copy.deepcopy(self.config)
            mutate(config)
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "config.json"
                path.write_text(json.dumps(config), encoding="utf-8")
                with self.assertRaises(ValueError):
                    protocol.load_config(path)

    def test_replay_echo_tolerates_one_ulp(self):
        # Boost.JSON 1.82 parsed 0.24111619999894174 as ...177 in the local smoke; structure must still match exactly.
        from rp_bit_validation import _same
        self.assertTrue(_same(dict(a=0.24111619999894174, b=[1., "x"]), dict(a=0.24111619999894177, b=[1., "x"])))
        self.assertFalse(_same(dict(a=.2411), dict(a=.2412)))
        self.assertFalse(_same(dict(a=1.), dict(a=1., b=2.)))
        self.assertFalse(_same(["x"], ["y"]))

    def test_replay_request_binds_the_finished_source(self):
        info = dict(map_path=str(ROOT / "tests/fixtures/map.t25d"), cloud_path=str(ROOT / "tests/fixtures/cloud.npy"),
                    cloud_sha256="cloud", map_sha256="map", start=[2., 10.], goal=[18., 10.])
        field = dict(field_sha256="field", cloud_sha256="cloud", builder_source_sha256="builder")
        entry = next(e for e in protocol.inventory(self.config, "smoke_factorial") if e["method"] == "sem_refine")
        with tempfile.TemporaryDirectory() as folder:
            trial, source = Path(folder) / entry["trial_id"], Path(folder) / entry["trial_id"].replace("sem_refine", "sem_native")
            trial.mkdir()
            source.mkdir()
            path = [[2., 10.], [10.000000000000002, 10.1], [18., 10.]]
            (source / "execution.json").write_text(json.dumps(dict(status="COMPLETED")), encoding="utf-8")
            (source / "worker.json").write_text(json.dumps(dict(planner_status="EXACT_SOLUTION",
                termination_reason="CONVERGED_NO_IMPROVEMENT", phase_timings=dict(solve_s=1.5, total_before_solve_s=.3),
                snapshots=[dict(snapshot_id="incumbent_000000", elapsed_s=.2, path_xy=path, native_objective_value=9.)])),
                encoding="utf-8")
            build = lambda where: protocol.request_for(entry, info, self.config, field_path=where / "planner_sem_field.txt",
                                                       field_metadata=field, bridge=ROOT / "tests/fixtures/libbridge.dll")
            request = build(trial)
            self.assertEqual(set(request), {"planner_id", "variant_id", "planner_seed", "objective_type", "sem_exposure", "map_path",
                                            "start", "goal", "planning_budget_s", "collision_index", "cloud_phase_handshake",
                                            "safe_length", "rp_refine_replay"})
            self.assertEqual((request["planner_id"], request["variant_id"], request["sem_exposure"]["lambda"]),
                             ("rp_refine_replay", "sem_refine", 33.))
            spec = request["rp_refine_replay"]
            self.assertEqual((spec["source_trial_id"], spec["source_solve_s"], spec["source_preprocessing_s"]), (source.name, 1.5, .3))
            self.assertEqual(spec["source_worker_sha256"], digest_file(source / "worker.json"))
            self.assertEqual(spec["refine_at_s"], self.config["methods"]["sem_refine"]["refine_at_s"])
            # Bit patterns carry the exact coordinates.
            decoded = [[protocol.coordinate_from_bits(x), protocol.coordinate_from_bits(y)] for x, y in spec["incumbents"][0]["path_bits"]]
            self.assertEqual(decoded, path)
            with self.assertRaisesRegex(ValueError, "own trial directory"):
                build(Path(folder) / "elsewhere")
            (source / "execution.json").write_text(json.dumps(dict(status="CRASH")), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "did not complete"):
                build(trial)
            shutil.rmtree(source)
            with self.assertRaises(OSError):
                build(trial)


def factorial_phase(directory, a4_delta, a4_time):
    """Labeled parser fixture: SEM, SEM + refine and A4 on 2 families x 2 maps x 2 seeds, one registered contrast."""
    methods = {"sem_native": dict(kind="sem_native"),
               "sem_refine": {"kind": "sem_refine", "source": "sem_native", "lambda": 33., "refine_at_s": [1.]},
               "a4": {"kind": "rp_bit", "lambda": 33., "refine": True}}
    settings = dict(checkpoints_s=[1., 20.], sem_equivalent_lambda=30., noninferiority_margin=.01, bootstrap_replicates=2000,
                    bootstrap_seed=7, contrasts=[["a4", "sem_refine"]],
                    decision=dict(contrast=["a4", "sem_refine"], rule="refine_attribution_v1"))
    shifts = dict(sem_native=(0., 5.), sem_refine=(-1., 5.2), a4=(-1. + a4_delta, a4_time))
    replay = dict(overhead_s=.2, refinements=2, refine_published=1, refine_not_improving=1, refine_rejected=0, late=0,
                  raw_rejected=0, raw_published=3)
    entries, rows, index = [], [], 0
    for family in ("hill", "ridge"):
        for map_seed in (1, 2):
            for seed in (11, 12):
                base, index = 10. + index, index + 1
                for method, (shift, seconds) in shifts.items():
                    trial, cost = f"{family}_{map_seed}__s{seed}__{method}", 2 * base + shift
                    entries.append(dict(trial_id=trial, map_id=f"{family}_{map_seed}", family=family, map_seed=map_seed,
                                        planner_seed=seed, query_id="q0", method=method, phase="synthetic", block_index=index))
                    point = dict(snapshot_id="s", elapsed_s=.5, safe=True, cost_terrain_m=base, pri=2., j_sem=cost, sharp_turn_count=0)
                    rows.append(dict(trial_id=trial, success=True, execution_status="COMPLETED", planner_status="EXACT_SOLUTION",
                                     correctness_errors=[], final_cost_terrain_m=base, final_pri=2., final_j_sem=cost,
                                     final_sharp_turn_count=0, solve_time_s=seconds + .01 * index,
                                     checkpoints={"1": point, "20": point}, reported=[[.5, cost]], events_j_sem=[[.5, cost]],
                                     **(dict(replay=replay) if method == "sem_refine" else {})))
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text(json.dumps(dict(protocol_id=protocol.PROTOCOL_ID, phase="synthetic",
        synthetic_fixture=True, config=dict(analysis=settings, methods=methods, phases=dict(synthetic=dict(primary_method="a4"))),
        config_sha256="fixture", binary_sha256="fixture", entries=entries)), encoding="utf-8")
    (directory / "summary.json").write_text(json.dumps(dict(synthetic_fixture=True, rows=rows)), encoding="utf-8")


def synthetic_phase(directory, *, sem_failure=True, sem_work=None):
    """Labeled parser fixture: two families x two maps x two seeds, SEM and one A4 method."""
    methods = {"sem_native": dict(kind="sem_native"), "a4": {"kind": "rp_bit", "lambda": 30., "refine": True}}
    settings = dict(checkpoints_s=[1., 20.], sem_equivalent_lambda=30., noninferiority_margin=.01,
                    bootstrap_replicates=2000, bootstrap_seed=7)
    entries, rows, index = [], [], 0
    for family in ("hill", "ridge"):
        for map_seed in (1, 2):
            for seed in (11, 12):
                base, index = 10. + index, index + 1
                for method in methods:
                    native = method == "sem_native"
                    trial = f"{family}_{map_seed}__s{seed}__{method}"
                    entries.append(dict(trial_id=trial, map_id=f"{family}_{map_seed}", family=family, map_seed=map_seed,
                                        planner_seed=seed, query_id="q0", method=method, phase="synthetic", block_index=index))
                    length, pri, cost = (base, 2., 2 * base) if native else (base - .1, 1.9, 2 * base - 1.)
                    failed = sem_failure and native and index == 1
                    point = lambda j, safe=True: dict(snapshot_id="s", elapsed_s=.5, safe=safe, cost_terrain_m=length + 1.,
                                                      pri=pri + .5, j_sem=j, sharp_turn_count=2 if native else 0)
                    rows.append(dict(trial_id=trial, success=not failed, execution_status="COMPLETED",
                        planner_status="EXACT_SOLUTION", correctness_errors=[],
                        final_cost_terrain_m=length, final_pri=pri, final_j_sem=cost,
                        final_sharp_turn_count=1 if native else 0, solve_time_s=20. if native else 2.,
                        final_unsafe_reasons=["missing_sem_support"] if failed else None,
                        checkpoints={"1": point(cost + (6. if native else .5)), "20": point(cost, not failed)},
                        reported=[[.2, cost + 5.], [.8, cost + .9], [1.5, cost]]))
                    if sem_work and native:
                        rows[-1]["iterations"] = sem_work - (index == 1)
    phase = dict(primary_method="a4")
    if sem_work:
        phase.update(sem_work_iterations=sem_work, sem_budget_cap_s=60.)
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text(json.dumps(dict(protocol_id=protocol.PROTOCOL_ID, phase="synthetic",
        synthetic_fixture=True, config=dict(analysis=settings, methods=methods,
                                            phases=dict(synthetic=phase)), config_sha256="fixture",
        binary_sha256="fixture", entries=entries)), encoding="utf-8")
    (directory / "summary.json").write_text(json.dumps(dict(synthetic_fixture=True, rows=rows)), encoding="utf-8")


class AnalysisTests(unittest.TestCase):
    def test_report_on_labeled_fixture(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder) / "synthetic"
            synthetic_phase(directory)
            report = analysis.build_report([directory])
            self.assertTrue(report["complete"])
            table = {(row["scope"], row["method"]): row for row in report["tables"]}
            sem, a4 = table["all", "sem_native"], table["all", "a4"]
            # A failed attempt stays in the PSR denominator and leaves the quality means.
            self.assertEqual((sem["success"], sem["attempted"], a4["success"], a4["attempted"]), (7, 8, 8, 8))
            self.assertAlmostEqual(sem["psr"], 7 / 8)
            self.assertAlmostEqual(sem["apl"], sum(range(11, 18)) / 7)
            self.assertAlmostEqual(sem["sti"], 7 / sum(range(11, 18)))
            self.assertEqual((a4["sti"], a4["time"]), (0., 2.))
            pooled = next(row for row in report["paired_comparisons"] if row["scope"] == "all")
            self.assertEqual((pooled["safe_pairs"], pooled["planned_pairs"], pooled["safe_maps"]), (7, 8, 4))
            metrics = pooled["metrics"]
            self.assertAlmostEqual(metrics["apl"]["mean_difference"], -.1)
            self.assertAlmostEqual(metrics["apl"]["relative_pooled"], -.7 / sum(range(11, 18)))
            self.assertAlmostEqual(metrics["j_sem"]["mean_difference"], -1.)
            self.assertAlmostEqual(metrics["time"]["mean_difference"], -18.)
            self.assertEqual((metrics["pri"]["wins"], metrics["pri"]["losses"]), (7, 0))
            low, high = metrics["apl"]["relative_bootstrap"]["ci95"]
            self.assertTrue(low <= metrics["apl"]["relative_pooled"] <= high < 0.)
            self.assertAlmostEqual(metrics["j_sem"]["difference_bootstrap"]["upper95_one_sided"], -1.)
            self.assertLess(pooled["sti_difference"]["bootstrap"]["upper95_one_sided"], 0.)
            family = next(row for row in report["paired_comparisons"] if row["scope"] == "ridge")
            self.assertNotIn("difference_bootstrap", family["metrics"]["apl"])
            # Anytime rows use the checkpoint paths, never the final ones.
            early = {row["method"]: row for row in report["anytime"] if row["checkpoint_s"] == 1.}
            self.assertEqual((early["sem_native"]["success"], early["a4"]["safe_pairs"]), (8, 8))
            self.assertAlmostEqual(early["a4"]["difference"]["j_sem"], -6.5)
            # Equal-time differences carry the same hierarchical bootstrap bound as the final comparison.
            self.assertAlmostEqual(early["a4"]["difference_upper95"]["j_sem"], -6.5)
            self.assertIsNone(early["a4"]["time"])
            late = {row["method"]: row for row in report["anytime"] if row["checkpoint_s"] == 20.}
            self.assertEqual((late["sem_native"]["success"], late["a4"]["safe_pairs"]), (7, 7))
            # A4 traces: J_sem - .1 is first reached at 0.8 s (cost + .9 <= SEM final = cost + 1).
            quality = {row["tolerance"]: row for row in report["time_to_quality"]["a4"]}
            self.assertEqual((quality[0.]["pairs"], quality[0.]["reached"], quality[0.]["median_s"]), (7, 7, .8))
            checks = report["criteria"][0]
            self.assertFalse([key for key, value in checks.items() if key != "method" and value is not True])
            self.assertEqual([row["trial_id"] for row in report["failures"]], ["hill_1__s11__sem_native"])
            text = analysis.markdown(report, "SYNTHETIC TEST FIXTURE")
            self.assertIn("SYNTHETIC TEST FIXTURE", text)
            self.assertIn("| all | a4 | 8/8 |", text)
            self.assertIn("| a4 **(cấu hình chính đã đăng ký)** | đạt |", text)
            self.assertEqual(analysis.main(["--input", str(directory), "--output", str(Path(folder) / "report"),
                                            "--title", "SYNTHETIC TEST FIXTURE"]), 0)
            self.assertTrue((Path(folder) / "report" / "REPORT.md").is_file())
            self.assertIsNone(report["sem_work"])
            self.assertNotIn("cố định theo số vòng lặp", text)
            self.assertIn("ΔJ₃₀ [cận trên]", text)
            self.assertIn("bản tái dựng Python đã khóa", text)
            report["methods"]["sem_native"]["engine"] = "cpp"
            self.assertIn("SEM-RRT* chạy bằng bản C++", analysis.markdown(report, "SYNTHETIC TEST FIXTURE"))

    def test_contrasts_and_refine_attribution_decision(self):
        # A: J lower; B: J equal (non-inferior) and faster; C: neither (slower, or J worse by more than the margin).
        for delta, seconds, outcome in ((-.5, 2., "A"), (0., 2., "B"), (0., 8., "C"), (.5, 2., "C")):
            with tempfile.TemporaryDirectory() as folder:
                directory = Path(folder) / "synthetic"
                factorial_phase(directory, delta, seconds)
                report = analysis.build_report([directory])
                self.assertEqual(report["decision"]["outcome"], outcome, (delta, seconds))
                text = analysis.markdown(report, "SYNTHETIC TEST FIXTURE")
                self.assertIn(f"**{outcome} —", text)
        contrast = report["contrasts"][0]
        pooled = contrast["comparisons"][0]
        self.assertEqual((contrast["method"], contrast["baseline"], pooled["scope"], pooled["sign"]),
                         ("a4", "sem_refine", "all", "method_minus_baseline"))
        self.assertAlmostEqual(pooled["metrics"]["j_sem"]["mean_difference"], .5)
        self.assertIn("mean_baseline", pooled["metrics"]["j_sem"])
        self.assertEqual([row["checkpoint_s"] for row in contrast["anytime"]], [1., 20.])
        self.assertAlmostEqual(contrast["anytime"][0]["difference"]["j_sem"], .5)
        self.assertFalse(contrast["criteria"]["j_sem_mean_lower"])
        self.assertEqual(contrast["time_to_quality"][0]["pairs"], 8)
        self.assertEqual(report["replay"]["sem_refine"]["trials"], 8)
        self.assertEqual(report["replay"]["sem_refine"]["refinements"], 16)
        self.assertIn("## 9. Replay SEM-RRT* + rpRefine", text)
        # Every arm is still compared with SEM by default, including the replay.
        self.assertEqual({c["method"] for c in report["paired_comparisons"]}, {"sem_refine", "a4"})
        # Reports of phases without contrasts gain nothing.
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder) / "synthetic"
            synthetic_phase(directory)
            plain = analysis.build_report([directory])
            self.assertFalse({"contrasts", "decision", "replay"} & set(plain))
            self.assertNotIn("## 7.", analysis.markdown(plain, "SYNTHETIC TEST FIXTURE"))

    def test_fixed_work_sem_summary(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder) / "synthetic"
            synthetic_phase(directory, sem_work=100)
            report = analysis.build_report([directory])
            work = report["sem_work"]
            self.assertEqual((work["budgets"], work["planned"], work["verdicts"], work["full_work"]), ([(100, 60.)], 8, 8, 7))
            self.assertEqual((work["iterations_min"], work["iterations_max"], work["solve_time_mean_s"]), (99, 100, 20.))
            self.assertIn("Chạy đủ số vòng lặp: **7/8**", analysis.markdown(report, "SYNTHETIC TEST FIXTURE"))
            self.assertEqual(report["sem_iterations"], dict(engine="python", runs=8, min=99, median=100, max=100))

    def test_rejects_duplicates_and_reports_missing_verdicts(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder) / "synthetic"
            synthetic_phase(directory, sem_failure=False)
            with self.assertRaisesRegex(ValueError, "duplicate slot"):
                analysis.build_report([directory, directory])
            summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
            summary["rows"] = [row for row in summary["rows"] if not row["trial_id"].startswith("ridge_2__s12")]
            (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            report = analysis.build_report([directory])
            self.assertFalse(report["complete"])
            # A strict verdict overrides the fast verdict of the same trial; other trials keep the fast one.
            strict = copy.deepcopy(summary["rows"][0])
            strict.update(success=False, final_unsafe_reasons=["strict"])
            (directory / "summary_full.json").write_text(json.dumps(dict(rows=[strict])), encoding="utf-8")
            merged = analysis.build_report([directory])
            self.assertEqual([row["trial_id"] for row in merged["failures"]], [strict["trial_id"]])
            self.assertEqual(sum(row["attempted"] for row in merged["trials"]), 14)
            table = {(row["scope"], row["method"]): row for row in report["tables"]}
            self.assertEqual((table["all", "a4"]["planned"], table["all", "a4"]["attempted"]), (8, 7))
            pooled = next(row for row in report["paired_comparisons"] if row["scope"] == "all")
            self.assertEqual(pooled["safe_pairs"], 7)


if __name__ == "__main__":
    unittest.main()
