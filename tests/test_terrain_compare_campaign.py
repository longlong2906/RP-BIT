"""Campaign gate/provenance regressions using explicitly synthetic fixtures.

These tests never launch a measured planner or generate benchmark observations.
"""
import copy
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import terrain_compare_campaign as campaign
import terrain_compare_analysis as analysis
from experiment_common import digest, dump
from terrain_compare_protocol import sha256


class CampaignGatesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="synthetic_compare_campaign_", dir=ROOT/"tests/fixtures")
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def patch(self, target, **kw):
        p = mock.patch(target, **kw)
        result = p.start()
        self.addCleanup(p.stop)
        return result

    def prepared(self, simulate_venv=False):
        workspace, build = self.root/"workspace", self.root/"build"
        build.mkdir()
        binary, library = campaign.binary_paths(build)
        binary.write_bytes(b"synthetic executable bytes -- never executed")
        library.write_bytes(b"synthetic shared library -- never loaded")
        sgp_python = self.root/"python-sgp-test.exe"
        sgp_python.write_bytes(b"synthetic interpreter -- never executed")
        self.controller_python = Path(sys.executable).absolute()
        self.sgp_python = sgp_python.absolute()
        base_sgp = None
        if simulate_venv:
            # Model Linux venv/bin/python -> base Python resolution without
            # needing Windows symlink privileges in the local correctness run.
            controller = self.root/"controller-venv"/"bin"/"python"
            controller.parent.mkdir(parents=True)
            controller.write_bytes(b"synthetic controller interpreter")
            base_controller = self.root/"base-controller-python"
            base_controller.write_bytes(controller.read_bytes())
            base_sgp = self.root/"base-sgp-python"
            base_sgp.write_bytes(sgp_python.read_bytes())
            redirects = {controller: base_controller, sgp_python: base_sgp}
            original_resolve = Path.resolve
            def resolve_symlink(path, *args, **kwargs):
                return redirects.get(path, original_resolve(path, *args, **kwargs))
            self.patch("pathlib.Path.resolve", new=resolve_symlink)
            self.patch("terrain_compare_campaign.sys.executable", new=str(controller))
            self.controller_python = controller.absolute()
        def environment(path):
            return {"python": "3.10.19" if Path(path) in (sgp_python, base_sgp) else "3.12.7", "packages": [["synthetic", "1.0"]]}
        self.env = self.patch("terrain_compare_campaign.environment_identity", side_effect=environment)
        self.source = self.patch("terrain_compare_campaign.source_identity", return_value={"source_sha256": "synthetic-frozen-source"})
        self.native_library = self.root/"synthetic-native-runtime.dll"
        self.native_library.write_bytes(b"synthetic linked-library bytes -- never loaded")
        self.native_linkage_marker = "synthetic_original_resolution"
        def capture_native(paths):
            return {"binary_hashes": {str(self.native_library): sha256(self.native_library)},
                    "linkage": {"scope": "synthetic_test_fixture", "roots": list(paths),
                                "resolution_marker": self.native_linkage_marker}}
        self.native = self.patch("terrain_compare_campaign.capture_native_dependencies", side_effect=capture_native)
        def generate(config, name, directory):
            directory.mkdir(parents=True)
            data = directory/"synthetic.t25d"
            data.write_text("synthetic test fixture, not a map for execution")
            dump(data.with_suffix(".json"), {"synthetic_test_fixture": True})
            value = {"synthetic_test_fixture": True, "map_path": str(data), "profile": name}
            dump(directory/"manifest.json", value)
            return value
        def plan(config, name, dataset, directory, identity):
            return {"synthetic_test_fixture": True, "entries": [{"backend": "sgp",
                "request": {"map_path": str(directory/"synthetic.t25d"), "planner_seed": 11}}]}
        self.patch("terrain_compare_campaign.generate_dataset", side_effect=generate)
        self.patch("terrain_compare_campaign.plan_trials", side_effect=plan)
        campaign.prepare(workspace, build, sgp_python)
        prepared = campaign.check_frozen(workspace)
        dump(workspace/"correctness.json", {"status": "PASS", "identity_sha256": digest(prepared["identity"])})
        return workspace, build, prepared

    def test_prepare_preserves_venv_interpreter_paths_in_registry_and_freeze(self):
        workspace, _, prepared = self.prepared(simulate_venv=True)
        registry = json.loads((workspace/"registry.json").read_text())
        self.assertEqual(registry["sgp"]["argv"][0], str(self.sgp_python))
        self.assertEqual(registry["putn"]["argv"][0], str(self.controller_python))
        self.assertEqual(prepared["sgp_python"], str(self.sgp_python))
        self.assertEqual(prepared["controller_python"], str(self.controller_python))
        for call in self.env.call_args_list:
            self.assertIn(Path(call.args[0]), (self.sgp_python, self.controller_python))
        for call in self.native.call_args_list:
            self.assertIn(str(self.sgp_python), call.args[0])
            self.assertIn(str(self.controller_python), call.args[0])

    def test_native_library_content_and_resolution_drift_rejected(self):
        workspace, _, prepared = self.prepared()
        self.assertIn(str(self.native_library), prepared["identity"]["binary_hashes"])
        original = self.native_library.read_bytes()
        self.native_library.write_bytes(original+b"changed")
        with self.assertRaisesRegex(ValueError, "binary/interpreter hash mismatch"):
            campaign.check_frozen(workspace)
        self.native_library.write_bytes(original)
        self.native_linkage_marker = "synthetic_changed_loader_resolution"
        with self.assertRaisesRegex(ValueError, "native dependency resolution changed"):
            campaign.check_frozen(workspace)

    def test_prepared_freezes_sources_binaries_environment_and_registry(self):
        workspace, build, _ = self.prepared()
        self.source.return_value = {"source_sha256": "changed"}
        with self.assertRaisesRegex(ValueError, "source/hash mismatch"):
            campaign.check_frozen(workspace)
        self.source.return_value = {"source_sha256": "synthetic-frozen-source"}
        binary, _ = campaign.binary_paths(build)
        old = binary.read_bytes(); binary.write_bytes(old+b"changed")
        with self.assertRaisesRegex(ValueError, "binary/interpreter hash mismatch"):
            campaign.check_frozen(workspace)
        binary.write_bytes(old)
        original = self.env.side_effect
        self.env.side_effect = lambda _: {"python": "3.10.19", "packages": [["synthetic", "2.0"]]}
        with self.assertRaisesRegex(ValueError, "environment identity changed"):
            campaign.check_frozen(workspace)
        self.env.side_effect = original
        registry = workspace/"registry.json"
        registry.write_text(registry.read_text()+" ")
        with self.assertRaisesRegex(ValueError, "registry changed"):
            campaign.check_frozen(workspace)

    def test_prepared_freezes_plan_and_raw_dataset_artifacts(self):
        workspace, _, _ = self.prepared()
        for path in (workspace/"plans/smoke.json", workspace/"data/smoke/manifest.json",
                     workspace/"data/smoke/synthetic.t25d", workspace/"data/smoke/synthetic.json"):
            original = path.read_bytes()
            path.write_bytes(original+b" ")
            with self.subTest(path=path.name):
                with self.assertRaises(ValueError):
                    campaign.check_frozen(workspace)
            path.write_bytes(original)

    def test_serial_uses_one_worker_and_no_unmeasured_parallel_reservations(self):
        workspace, _, _ = self.prepared()
        run = self.patch("terrain_compare_campaign.run_campaign", return_value={"status": "test"})
        campaign.collect(workspace, "smoke-serial", force_local=True)
        self.assertEqual(run.call_args.kwargs["max_workers"], 1)
        self.assertEqual(set(run.call_args.kwargs["reservations"].values()), {8*campaign.GIB})

    def test_parallel_requires_verified_serial_readiness(self):
        workspace, _, _ = self.prepared()
        run = self.patch("terrain_compare_campaign.run_campaign")
        readiness = self.patch("terrain_compare_campaign.require_validated", side_effect=ValueError("synthetic invalid smoke"))
        with self.assertRaisesRegex(ValueError, "invalid smoke"):
            campaign.collect(workspace, "smoke-parallel", force_local=True)
        readiness.assert_called_once_with(workspace, "smoke-serial")
        run.assert_not_called()

    def test_main_requires_explicit_flag_linux_and_both_smokes(self):
        workspace, _, _ = self.prepared()
        run = self.patch("terrain_compare_campaign.run_campaign", return_value={"status": "test"})
        readiness = self.patch("terrain_compare_campaign.require_validated", return_value={"entries": []})
        self.patch("terrain_compare_campaign.collect_reservations", return_value={b: 512*1024**2 for b in ("a3", "sem", "putn", "sgp")})
        for platform, force_local, allow_main in (("linux", False, False), ("win32", False, True), ("linux", True, True)):
            with mock.patch.object(campaign.sys, "platform", platform):
                with self.subTest(platform=platform, force_local=force_local, allow_main=allow_main):
                    with self.assertRaisesRegex(ValueError, "main requires"):
                        campaign.collect(workspace, "main", force_local=force_local, allow_main=allow_main)
        run.assert_not_called()
        with mock.patch.object(campaign.sys, "platform", "linux"):
            campaign.collect(workspace, "main", allow_main=True)
        self.assertEqual([call.args[1] for call in readiness.call_args_list], ["smoke-serial", "smoke-parallel"])
        self.assertEqual(run.call_args.kwargs["max_workers"], 4)

    def test_changed_correctness_identity_cannot_collect(self):
        workspace, _, _ = self.prepared()
        dump(workspace/"correctness.json", {"status": "PASS", "identity_sha256": "stale"})
        run = self.patch("terrain_compare_campaign.run_campaign")
        with self.assertRaisesRegex(ValueError, "correctness not current"):
            campaign.collect(workspace, "smoke-serial")
        run.assert_not_called()

    def correctness_receipt(self, failure=None, skipped=False):
        def execute(command, log, name, **kwargs):
            result = {"argv": list(map(str, command)), "exit_code": 7 if name==failure else 0}
            dump(log/(name+".json"), result)
            (log/(name+".stdout.log")).write_text("synthetic stdout: "+name, encoding="utf-8")
            (log/(name+".stderr.log")).write_text(
                "synthetic stderr: "+name+("\nOK (skipped=1)" if skipped and name=="correctness_sgp" else ""),
                encoding="utf-8")
            return result
        return self.patch("terrain_compare_campaign.receipt", side_effect=execute)

    def test_correctness_exposes_failed_nested_command_and_retains_failed_gate(self):
        workspace, _, _ = self.prepared()
        for index, name in enumerate(("correctness_ctest", "correctness_sgp", "correctness_fixed_work")):
            with self.subTest(command=name):
                # Synthetic receipts are overwritten only within this temporary
                # test fixture; the real command runner rejects duplicate receipts.
                execute = self.correctness_receipt(failure=name)
                output = io.StringIO()
                with contextlib.redirect_stderr(output):
                    with self.assertRaisesRegex(RuntimeError, "correctness gate failed"):
                        campaign.correctness(workspace)
                text = output.getvalue()
                self.assertIn("nonzero test exit", text)
                self.assertIn("exit_code=7", text)
                self.assertIn("argv=", text)
                for suffix in (".json", ".stdout.log", ".stderr.log"):
                    self.assertIn(str(workspace/"logs"/(name+suffix)), text)
                self.assertIn("synthetic stdout: "+name, text)
                self.assertIn("synthetic stderr: "+name, text)
                self.assertEqual(execute.call_count, index+1)
                gate = json.loads((workspace/"correctness.json").read_text())
                self.assertEqual(gate["status"], "FAILED")
                self.assertEqual(gate["commands"][-1]["exit_code"], 7)

    def test_correctness_exposes_required_sgp_coverage_skipped_with_zero_exit(self):
        workspace, _, _ = self.prepared()
        execute = self.correctness_receipt(skipped=True)
        output = io.StringIO()
        with contextlib.redirect_stderr(output):
            with self.assertRaisesRegex(RuntimeError, "correctness gate failed"):
                campaign.correctness(workspace)
        self.assertEqual(execute.call_count, 3)
        self.assertIn("SGP tests skipped", output.getvalue())
        self.assertIn("OK (skipped=1)", output.getvalue())
        self.assertIn("exit_code=0", output.getvalue())
        self.assertEqual(json.loads((workspace/"correctness.json").read_text())["status"], "FAILED")

    def test_correctness_success_stays_silent_and_keeps_all_commands(self):
        workspace, _, _ = self.prepared()
        execute = self.correctness_receipt()
        output = io.StringIO()
        with contextlib.redirect_stderr(output):
            self.assertEqual(campaign.correctness(workspace), {"status": "PASS"})
        self.assertEqual(execute.call_count, 3)
        self.assertEqual(output.getvalue(), "")
        self.assertEqual(json.loads((workspace/"correctness.json").read_text())["status"], "PASS")

    def test_correctness_children_do_not_inherit_notebook_backend(self):
        workspace, _, _ = self.prepared()
        execute = self.correctness_receipt()
        inherited = "module://matplotlib_inline.backend_inline"
        with mock.patch.dict(campaign.os.environ, {"MPLBACKEND": inherited}):
            campaign.correctness(workspace)
            self.assertEqual(campaign.os.environ["MPLBACKEND"], inherited)
        self.assertEqual(execute.call_count, 3)
        self.assertTrue(all(call.kwargs["env"]["MPLBACKEND"] == "Agg"
                            for call in execute.call_args_list))

    def test_correctness_diagnostic_is_bounded_and_missing_log_does_not_mask_failure(self):
        log = self.root/"diagnostics"
        log.mkdir()
        (log/"synthetic.stdout.log").write_bytes(b"omitted-prefix"+b"x"*1000+b"tail-stdout")
        (log/"synthetic.stderr.log").write_bytes(b"omitted-prefix"+b"y"*1000+b"\xfftail-stderr")
        output = io.StringIO()
        with contextlib.redirect_stderr(output):
            campaign.show_correctness_failure(log, "synthetic", {"exit_code": 7}, "test fixture", tail_bytes=32)
        text = output.getvalue()
        self.assertNotIn("omitted-prefix", text)
        self.assertIn("tail-stdout", text)
        self.assertIn("tail-stderr", text)
        self.assertNotIn("x"*33, text)
        (log/"synthetic.stderr.log").unlink()
        output = io.StringIO()
        with contextlib.redirect_stderr(output):
            campaign.show_correctness_failure(log, "synthetic", {"exit_code": 7}, "test fixture")
        self.assertIn("Cannot read correctness diagnostic", output.getvalue())

    def readiness_fixture(self):
        workspace = self.root/"readiness"
        directory = workspace/"campaigns/smoke-serial"
        directory.mkdir(parents=True)
        entries = [{"trial_id": f"synthetic_{i}", "execution_status": "COMPLETED"} for i in range(42)]
        manifest = {"synthetic_test_fixture": True, "status": "COLLECTED_VALIDATION_PENDING", "entries": entries}
        dump(directory/"manifest.json", manifest)
        for entry in entries:
            folder = directory/entry["trial_id"]
            folder.mkdir()
            dump(folder/"worker.json", {"synthetic_test_fixture": True, "status": "TIMEOUT"})
        summary = {"manifest_sha256": sha256(directory/"manifest.json"), "raw_hashes": analysis.raw_hashes(directory),
                   "correctness_failures": 0, "rows": [{"native_status": "TIMEOUT"} for _ in entries]}
        dump(directory/"validation_summary.json", summary)
        return workspace, directory, manifest, summary

    def test_readiness_accepts_timeouts_but_rejects_incomplete_or_model_error(self):
        workspace, directory, manifest, summary = self.readiness_fixture()
        campaign.require_validated(workspace, "smoke-serial")
        manifest["entries"][0]["execution_status"] = "CRASH"
        dump(directory/"manifest.json", manifest)
        summary["manifest_sha256"] = sha256(directory/"manifest.json")
        dump(directory/"validation_summary.json", summary)
        with self.assertRaisesRegex(ValueError, "42 smoke"):
            campaign.require_validated(workspace, "smoke-serial")
        manifest["entries"][0]["execution_status"] = "COMPLETED"
        dump(directory/"manifest.json", manifest)
        summary["manifest_sha256"] = sha256(directory/"manifest.json")
        summary["rows"][0]["native_status"] = "MODEL_ERROR"
        dump(directory/"validation_summary.json", summary)
        with self.assertRaisesRegex(ValueError, "model error"):
            campaign.require_validated(workspace, "smoke-serial")

    def test_readiness_rejects_changed_raw_worker_after_validation(self):
        workspace, directory, _, _ = self.readiness_fixture()
        raw = directory/"synthetic_0/worker.json"
        raw.write_text(raw.read_text()+" ")
        with self.assertRaises(ValueError):
            campaign.require_validated(workspace, "smoke-serial")

    def test_analysis_latest_incumbent_keeps_own_cost_common_risk_tradeoff(self):
        events = [{"elapsed_s": .2, "independent_valid": True, "planner_objective_value": 10.,
                   "metrics": {"objective_value": 12., "cost_terrain_m": 10., "risk_integral_m": 2.}},
                  {"elapsed_s": .7, "independent_valid": True, "planner_objective_value": 9.,
                   "metrics": {"objective_value": 14., "cost_terrain_m": 9., "risk_integral_m": 5.}}]
        self.assertIsNone(analysis.at_checkpoint(events, .1))
        self.assertIs(analysis.at_checkpoint(events, .6), events[0])
        self.assertIs(analysis.at_checkpoint(events, 2.), events[1])
        self.assertEqual(analysis.at_checkpoint(events, 2.)["metrics"]["risk_integral_m"], 5.)


if __name__ == "__main__":
    unittest.main()
