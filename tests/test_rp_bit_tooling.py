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


PUTN_ARMS = ("putn_slope35", "putn_source")


class PutnConfigTests(unittest.TestCase):
    """PUTN dev preview: two self-stopping PUTN arms beside the self-stop SEM-RRT* and the locked A4, 5 s."""

    def setUp(self):
        self.config = protocol.load_config(protocol.PUTN_DEV_CONFIG)

    def test_arms_and_budget(self):
        selfstop = protocol.load_config(ROOT / "configs/a4_rpbitstar_v1_selfstop_dev.json")
        methods = self.config["methods"]
        for name in (protocol.NATIVE, "a4_rpbit_l33_m20"):
            self.assertEqual(methods[name], selfstop["methods"][name], name)
        # Same convergence rule as SEM-RRT*; the declared cap is the protocol slope cap; the source arm has none.
        self.assertEqual(methods["putn_slope35"], dict(kind=protocol.PUTN, tau_slope_cap_deg=35., convergence_stop=STOP))
        self.assertEqual(methods["putn_source"], dict(kind=protocol.PUTN, convergence_stop=STOP))
        self.assertEqual(methods["putn_slope35"]["tau_slope_cap_deg"], self.config["safety"]["slope_cap_deg"])
        dev = self.config["phases"]["dev_putn"]
        self.assertEqual((dev["maps"], dev["map_seeds"], dev["planning_budget_s"]), ("dev", [41001, 41002, 41003], 5.))
        self.assertTrue(all(t <= dev["planning_budget_s"] for t in self.config["analysis"]["checkpoints_s"]))
        self.assertEqual(len(protocol.inventory(self.config, "dev_putn")), 300)
        self.assertFalse(any(name.startswith("heldout") for name in self.config["phases"]))

    def test_putn_requests(self):
        info = dict(map_path=str(ROOT / "tests/fixtures/map.t25d"), cloud_path=str(ROOT / "tests/fixtures/cloud.npy"),
                    cloud_sha256="cloud", map_sha256="map", start=[2., 10.], goal=[18., 10.])
        with tempfile.TemporaryDirectory() as folder:
            bridge = Path(folder) / "libterrain_compare_c.dll"
            bridge.write_bytes(b"bridge")
            entries = protocol.inventory(self.config, "dev_putn")
            for name in PUTN_ARMS:
                request = protocol.request_for(next(e for e in entries if e["method"] == name), info, self.config,
                                               bridge=bridge)
                self.assertEqual((request["backend"], request["track"], request["planner_id"], request["objective_type"]),
                                 ("putn", "native", "putn_native_paper_v2", "native"))
                self.assertEqual((request["planning_budget_s"], request["work_iterations"], request["variant_id"]),
                                 (5., 0, name))
                self.assertEqual((request["map_sha256"], request["bridge_library"]), ("map", str(bridge.resolve())))
                self.assertEqual(request["putn_convergence_stop"], STOP)
                self.assertEqual(request.get("putn_tau_slope_cap_deg"), 35. if name == "putn_slope35" else None)
                for key in ("algorithm_budget_policy", "sem_convergence_stop", "sem_engine"):
                    self.assertNotIn(key, request)

    def test_invalid_putn_methods(self):
        for change in (dict(engine="cpp"), {"lambda": 30.}, dict(tau_slope_cap_deg=35), dict(tau_slope_cap_deg=0.),
                       dict(tau_slope_cap_deg=90.5), dict(convergence_stop=None),
                       dict(convergence_stop=dict(STOP, relative_eps=0.))):
            config = copy.deepcopy(self.config)
            config["methods"]["putn_slope35"].update(change)
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "config.json"
                path.write_text(json.dumps(config), encoding="utf-8")
                with self.assertRaises(ValueError, msg=str(change)):
                    protocol.load_config(path)

    def test_runner_sends_putn_to_the_native_worker(self):
        seen = {}

        def collect(task, *, request_builder, is_native, field_exchange):
            seen.update({name: is_native(dict(method=name)) for name in task["config"]["methods"]})
            # No arm of this config is A5: the planner field stays in the text exchange.
            self.assertTrue(all(field_exchange(dict(method=name)) is None for name in task["config"]["methods"]))
        with mock.patch("run_sem_exposure._collect_trial", collect):
            runner._collect(dict(config=self.config))
        self.assertEqual(seen, {protocol.NATIVE: True, "a4_rpbit_l33_m20": False, "putn_slope35": True, "putn_source": True})


class PutnKaggleConfigTests(unittest.TestCase):
    """PUTN held-out on Kaggle: the dev methods unchanged, 10 s cap, one timed lane, fresh seeds, validation on 4 cores."""

    def setUp(self):
        self.config = protocol.load_config(protocol.PUTN_KAGGLE_CONFIG)

    def test_only_budget_execution_and_phases_change(self):
        dev = protocol.load_config(protocol.PUTN_DEV_CONFIG)
        for key in ("protocol_id", "schema_version", "safety", "methods"):
            self.assertEqual(self.config[key], dev[key], key)
        self.assertEqual(self.config["analysis"], dict(dev["analysis"], checkpoints_s=[.5, 1., 2., 5., 10.]))
        selfstop = protocol.load_config(protocol.SELFSTOP_CONFIG)
        self.assertEqual(self.config["execution"], dict(selfstop["execution"], validation_cores=[0, 1, 2, 3]))
        phases = self.config["phases"]
        self.assertEqual(set(phases), {"smoke_putn", "gate_putn", "heldout_putn"})
        self.assertEqual(phases["smoke_putn"], dev["phases"]["smoke_putn"])
        for name in ("gate_putn", "heldout_putn"):
            self.assertEqual((phases[name]["maps"], phases[name]["planning_budget_s"], phases[name]["methods"]),
                             ("generate", 10., list(dev["methods"])), name)
        self.assertEqual((phases["gate_putn"]["map_seeds"], phases["gate_putn"]["planner_seeds"]), ([51001], [516001]))
        held = phases["heldout_putn"]
        fresh = set(held["map_seeds"]) | set(held["planner_seeds"])
        self.assertEqual(fresh, set(range(101001, 101011)) | set(range(1016001, 1016006)))
        self.assertEqual(len(protocol.inventory(self.config, "heldout_putn")), 1000)
        for path in sorted((ROOT / "configs").glob("a4_rpbitstar_v1*.json")):
            config = protocol.load_config(path)
            for name, phase in config["phases"].items():
                if (path, name) != (protocol.PUTN_KAGGLE_CONFIG, "heldout_putn"):
                    self.assertFalse(fresh & (set(phase["map_seeds"]) | set(phase["planner_seeds"])), (path.name, name))

    def test_invalid_validation_cores(self):
        for value in ([], [0, 0], [-1], [1.], "0", [True]):
            config = copy.deepcopy(self.config)
            config["execution"]["validation_cores"] = value
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "config.json"
                path.write_text(json.dumps(config), encoding="utf-8")
                with self.assertRaises(ValueError, msg=repr(value)):
                    protocol.load_config(path)

    def test_validation_uses_the_validation_cores(self):
        seen = []

        def lanes(queues, target, control, *, cores):
            seen.append((len(queues), cores))
            return dict(status="COMPLETE")
        for execution, expected in ((dict(cores=[0], validation_cores=[0, 1, 2, 3]), [0, 1, 2, 3]), (dict(cores=[2]), [2])):
            config = dict(execution=dict(execution, reserve_bytes_per_worker=1))
            with tempfile.TemporaryDirectory() as folder:
                directory = Path(folder)
                (directory / "trials/t").mkdir(parents=True)
                (directory / "trials/t/raw_inventory.json").write_text("{}", encoding="utf-8")
                manifest = dict(phase="p", config_sha256="c", maps=dict(m={}),
                                entries=[dict(trial_id="t", map_id="m", method="a4_rpbit_l33_m20")])
                (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                with mock.patch.object(runner, "run_spawn_lanes", lanes):
                    runner.validate(directory, config, False)
            self.assertEqual(seen.pop(), (len(expected), expected))

    def test_held_out_needs_the_flag(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(SystemExit, "held-out requires"):
                runner.run(Path(folder) / "out", "heldout_putn", "missing", "missing", config_path=protocol.PUTN_KAGGLE_CONFIG)
            self.assertFalse((Path(folder) / "out").exists())


class ValidationPutnTests(unittest.TestCase):
    """A PUTN trial must have run the registered convergence stop and tau slope cap, and obey the stop."""

    def verdict(self, method_update, request_update, worker_update):
        import rp_bit_validation as validation
        method = {k: v for k, v in dict(dict(kind=protocol.PUTN, tau_slope_cap_deg=35., convergence_stop=STOP),
                                        **method_update).items() if v is not None}
        config = dict(methods={"putn_slope35": method}, analysis=dict(sem_equivalent_lambda=30., checkpoints_s=[1.]))
        entry = dict(trial_id="m__s1__putn_slope35", method="putn_slope35")
        request = {k: v for k, v in dict(dict(planning_budget_s=5., start=[0., 0.], goal=[1., 1.],
                   putn_convergence_stop=STOP, putn_tau_slope_cap_deg=35.), **request_update).items() if v is not None}
        worker = dict(planner_status="EXACT_SOLUTION", native_model=dict(tau_slope_cap_deg=35.),
                      snapshots=[dict(snapshot_id="a", elapsed_s=.2, native_objective_value=100.),
                                 dict(snapshot_id="b", elapsed_s=.4, native_objective_value=99.95)],
                      counters=dict(iterations=5000, putn_statistics=[0.] * 12), termination_reason="CONVERGED_NO_IMPROVEMENT",
                      putn_convergence=dict(converged=True, reference_cost=100., reference_iteration=1000, rule=STOP))
        worker.update(worker_update)
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            for name, body in (("execution.json", dict(status="COMPLETED")), ("request.json", request), ("worker.json", worker)):
                (directory / name).write_text(json.dumps(body), encoding="utf-8")
            with mock.patch.object(validation, "score_path", lambda *args: dict(safe=True)), \
                    mock.patch.object(validation, "_csv_path", lambda path: []):
                return validation.validate_trial(directory, entry, {}, config, oracle=None, field=None)["correctness_errors"]

    def test_rule_cap_and_stop_point(self):
        self.assertEqual(self.verdict({}, {}, {}), [])
        source = dict(tau_slope_cap_deg=None), dict(putn_tau_slope_cap_deg=None)
        self.assertEqual(self.verdict(*source, dict(native_model={})), [])
        # Stopped by the time cap instead (or no solution): nothing to check beyond the rule.
        self.assertEqual(self.verdict({}, {}, dict(termination_reason="TIME_BUDGET", counters=dict(iterations=7),
            putn_convergence=dict(converged=False, reference_cost=None, reference_iteration=0, rule=STOP))), [])
        cap = ["putn_tau_slope_cap_mismatch"]
        self.assertEqual(self.verdict({}, dict(putn_tau_slope_cap_deg=None), {}), cap)
        self.assertEqual(self.verdict({}, {}, dict(native_model={})), cap)
        self.assertEqual(self.verdict(*source, {}), cap)
        mismatch = ["putn_convergence_stop_mismatch"]
        self.assertEqual(self.verdict({}, {}, dict(putn_convergence=None)), mismatch)
        self.assertEqual(self.verdict({}, dict(putn_convergence_stop=None), {}), mismatch)
        self.assertEqual(self.verdict({}, dict(putn_convergence_stop=dict(STOP, window_iterations=10)), {}), mismatch)
        self.assertEqual(self.verdict(dict(convergence_stop=None), dict(putn_convergence_stop=None), {}), mismatch)
        # A SEM outcome is not a PUTN outcome.
        self.assertEqual(self.verdict({}, {}, dict(putn_convergence=None, sem_convergence=dict(
            converged=True, reference_cost=100., reference_iteration=1000, rule=STOP))), mismatch)
        inconsistent = ["putn_convergence_inconsistent"]
        self.assertEqual(self.verdict({}, {}, dict(counters=dict(iterations=4999))), inconsistent)
        self.assertEqual(self.verdict({}, {}, dict(termination_reason="TIME_BUDGET")), inconsistent)
        self.assertEqual(self.verdict({}, {}, dict(snapshots=[dict(snapshot_id="a", elapsed_s=.2,
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


A5_ARMS = ("a5_rpbit_l33_w4000", "a5_rpbit_l33_w1000", "a5_rpbit_l33_w4000_noseed", "a5_rpbit_l33_w4000_eager",
           "a5_rpbit_l33_w4000_batchonly")


class A5ConfigTests(unittest.TestCase):
    """A5 dev preview: the self-stop SEM-RRT*, the locked A4 (M = 20 and M = 5) and five A5 arms, on seen maps only."""

    def setUp(self):
        self.config = protocol.load_config(protocol.A5_DEV_CONFIG)

    def test_arms_and_phases(self):
        selfstop = protocol.load_config(ROOT / "configs/a4_rpbitstar_v1_selfstop_dev.json")
        locked, methods = protocol.load_config(), self.config["methods"]
        self.assertEqual(methods[protocol.NATIVE], selfstop["methods"][protocol.NATIVE])
        for name in ("a4_rpbit_l33_m20", "a4_rpbit_l33"):
            self.assertEqual(methods[name], locked["methods"][name], name)
        # The primary arm keeps the registered stop (the SEM window, in samples) with every mechanism on.
        primary = methods["a5_rpbit_l33_w4000"]
        self.assertEqual(primary["window_samples"], methods[protocol.NATIVE]["convergence_stop"]["window_iterations"])
        self.assertEqual(primary["window_samples"], 200 * methods["a4_rpbit_l33_m20"]["stop_batches"])
        self.assertTrue(all(primary[key] for key in protocol.A5_SWITCHES))
        self.assertEqual(methods["a5_rpbit_l33_w1000"], dict(primary, window_samples=1000))
        # Ablation ladder: each arm drops one mechanism of the arm before it.
        self.assertEqual(methods["a5_rpbit_l33_w4000_noseed"], dict(primary, seed=False))
        self.assertEqual(methods["a5_rpbit_l33_w4000_eager"], dict(primary, seed=False, lazy_refine=False))
        self.assertEqual(methods["a5_rpbit_l33_w4000_batchonly"], dict(primary, seed=False, lazy_refine=False, fast_validity=False))
        self.assertEqual({name for name, method in methods.items() if method["kind"] == protocol.A5}, set(A5_ARMS))
        self.assertEqual(set(self.config["phases"]), {"smoke_a5", "probe_a5", "dev_a5"})
        for name, phase in self.config["phases"].items():
            self.assertEqual((phase["methods"], phase["primary_method"]), (list(methods), "a5_rpbit_l33_w4000"), name)
        dev, probe = self.config["phases"]["dev_a5"], self.config["phases"]["probe_a5"]
        self.assertEqual((dev["maps"], dev["map_seeds"], dev["planning_budget_s"]), ("dev", [41001, 41002, 41003], 20.))
        self.assertEqual((probe["maps"], probe["map_seeds"], probe["planner_seeds"]), ("dev", dev["map_seeds"], dev["planner_seeds"][:1]))
        self.assertEqual((len(protocol.inventory(self.config, "dev_a5")), len(protocol.inventory(self.config, "probe_a5"))), (600, 120))
        self.assertTrue(all(pair[0] in A5_ARMS for pair in self.config["analysis"]["contrasts"]))

    def test_requests(self):
        info = dict(map_path=str(ROOT / "tests/fixtures/map.t25d"), cloud_path=str(ROOT / "tests/fixtures/cloud.npy"),
                    cloud_sha256="cloud", map_sha256="map", start=[2., 10.], goal=[18., 10.])
        arguments = dict(field_path=ROOT / "tests/fixtures/planner_field.txt", bridge=ROOT / "tests/fixtures/libbridge.dll",
                         field_metadata=dict(field_sha256="field", cloud_sha256="cloud", builder_source_sha256="builder"))
        entries = {entry["method"]: entry for entry in protocol.inventory(self.config, "probe_a5")[:8]}
        a4 = protocol.request_for(entries["a4_rpbit_l33_m20"], info, self.config, **arguments)
        self.assertNotIn("rp_bit_a5", a4)
        strip = lambda r: {k: v for k, v in r.items() if k not in ("rp_bit", "rp_bit_a5", "batch_after_first_solution", "variant_id")}
        for name in A5_ARMS:
            method = self.config["methods"][name]
            request = protocol.request_for(entries[name], info, self.config, **arguments)
            self.assertEqual(request["rp_bit_a5"], {k: v for k, v in method.items() if k not in ("kind", "lambda")})
            self.assertEqual((request["samples_per_batch"], request["batch_after_first_solution"]), (100, method["batch_min"]))
            # Apart from its own block, the batch size after the first solution and its name, it is the A4 request.
            self.assertEqual(strip(request), strip(a4))
            self.assertEqual(request["variant_id"], name)
            self.assertNotIn("rp_bit", request)

    def test_invalid_a5_methods(self):
        changes = [dict(refine=False), dict(refine=False, seed=False), dict(refine=False, lazy_refine=False), dict(window_samples=-1),
                   dict(window_samples=4000.), dict(batch_min=0), dict(batch_growth=1), dict(extra=1), {"lambda": 33},
                   dict(stop_batches=20), dict(engine="cpp"), dict(convergence_stop=STOP)]
        changes += [{key: None} for key in ("window_samples", "batch_min", *protocol.A5_SWITCHES)]
        for change in changes:
            config = copy.deepcopy(self.config)
            method = config["methods"]["a5_rpbit_l33_w4000"]
            for key, value in change.items():
                if value is None:
                    del method[key]
                else:
                    method[key] = value
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "config.json"
                path.write_text(json.dumps(config), encoding="utf-8")
                with self.assertRaises(ValueError, msg=str(change)):
                    protocol.load_config(path)

    def test_runner_sends_a5_to_the_cpp_worker(self):
        import rp_bit_a5_field
        seen, exchange = {}, {}

        def collect(task, *, request_builder, is_native, field_exchange):
            for name in task["config"]["methods"]:
                seen[name], exchange[name] = is_native(dict(method=name)), field_exchange(dict(method=name))
        with mock.patch("run_sem_exposure._collect_trial", collect):
            runner._collect(dict(config=self.config))
        self.assertEqual(seen, {name: name == protocol.NATIVE for name in self.config["methods"]})
        # Only A5 receives the binary container; SEM-RRT* has no field file and A4 keeps the text exchange.
        binary = (rp_bit_a5_field.FILE_NAME, rp_bit_a5_field.build_field_file)
        self.assertEqual(exchange, {name: binary if name in A5_ARMS else None for name in self.config["methods"]})

    def test_binary_field_holds_the_cells_of_the_text_exchange(self):
        import hashlib
        import struct
        import numpy as np
        import rp_bit_a5_field
        import safe_length_field
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            # Labeled synthetic cloud: a rough plane with a hole, so some cells are unsupported.
            x, y = (v.ravel() for v in np.meshgrid(np.arange(.025, 6., .05), np.arange(.025, 6., .05)))
            z = .1 * x + .05 * y + .02 * np.random.default_rng(5).standard_normal(len(x))
            keep = ~((x > 2.) & (x < 2.6) & (y > 3.) & (y < 3.5))
            cloud, terrain, bounds = folder / "cloud.npy", folder / "map.t25d", (0., 0., 6., 6.)
            np.save(cloud, np.column_stack((x, y, z))[keep], allow_pickle=False)
            terrain.write_text("test fixture: only hashed\n", encoding="ascii")
            text = safe_length_field.build_field_file(cloud, terrain, bounds, folder / "planner_sem_field.txt")
            target = folder / rp_bit_a5_field.FILE_NAME
            binary = rp_bit_a5_field.build_field_file(cloud, terrain, bounds, target)
            for key in ("version", "cloud_sha256", "map_sha256", "builder_source_sha256", "supported_cells",
                        "observed_cells", "input_points", "shape_yx"):
                self.assertEqual(binary[key], text[key], key)
            data = target.read_bytes()
            self.assertEqual((binary["field_sha256"], data[:8]), (hashlib.sha256(data).hexdigest(), rp_bit_a5_field.MAGIC))
            lines = (folder / "planner_sem_field.txt").read_text(encoding="ascii").split("\n")
            nx, ny, *grid = struct.unpack_from("<2i5d", data, 8)
            self.assertEqual([nx, ny, *grid], [float(value) for value in lines[1].split()])
            self.assertEqual(len(data), 56 + 9 * nx * ny)
            cells = np.array([[float(value) for value in line.split()] for line in lines[2:-1]])
            support = np.frombuffer(data, np.uint8, nx * ny, 56)
            risk = np.frombuffer(data, "<f8", nx * ny, 56 + nx * ny)
            self.assertTrue(np.array_equal(support, cells[:, 0]) and 0 < support.sum() < support.size)
            # Bit for bit, not merely equal as numbers.
            self.assertTrue(np.array_equal(risk.view(np.uint64), np.ascontiguousarray(cells[:, 1]).view(np.uint64)))
            self.assertTrue(len(set(risk.tolist())) > 100)
            with self.assertRaises(FileExistsError):
                rp_bit_a5_field.build_field_file(cloud, terrain, bounds, target)

    def test_scorer_restates_the_batch_schedule(self):
        from rp_bit_validation import a5_next_batch

        def sizes(options):
            out, stale, drawn = [], 0, options["batch_min"]
            while True:
                out.append(drawn)
                following = a5_next_batch(stale, drawn, options)
                stale += drawn
                if stale >= options["window_samples"]:
                    return out
                drawn = following
        primary = self.config["methods"]["a5_rpbit_l33_w4000"]
        self.assertEqual(sizes(primary), [500, 500, 1000, 2000])
        self.assertEqual(sizes(self.config["methods"]["a5_rpbit_l33_w1000"]), [500, 500])
        self.assertEqual(sizes(dict(primary, batch_growth=False, batch_min=200)), [200] * 20)


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
