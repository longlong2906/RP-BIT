"""Synthetic packaging/stage fixtures only; no planner or benchmark execution."""
from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tools"))
import kaggle_native_paper_v4_r2_bundle as bundle


class R2BundleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="r2_bundle_synthetic_", dir=ROOT/"tests/fixtures")
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.sources = self.base/"synthetic_sources"
        self.sources.mkdir()
        self.files = {}
        names = dict(bundle.PATCH_NAMES, **{"test_"+k: v for k, v in bundle.TEST_NAMES.items()})
        for key, name in names.items():
            path = self.sources/name
            path.write_text("# SYNTHETIC UNIT FIXTURE: no planner\nvalue = "+repr(key)+"\n", encoding="utf-8")
            self.files[key] = path
        r1_manifest = dict(source_input_sha256=bundle.SOURCE_SHA256,
            runner=dict(path=bundle.PATCH_NAMES["r1_runner"], sha256=bundle.sha(self.files["r1_runner"])),
            analyzer=dict(path=bundle.PATCH_NAMES["analyzer"], sha256=bundle.sha(self.files["analyzer"])))
        self.files["r1_manifest"].write_bytes(bundle.encoded_json(r1_manifest))
        self.identity = dict(bundle.default_manifest(), original_scheduler_sha256="a"*64)

    def build(self, name="delivery", **kwargs):
        return bundle.build(self.base/name, resume_manifest=self.identity, source_files=self.files, **kwargs)

    def book(self, result):
        return json.loads(Path(result["notebook"]).read_text(encoding="utf-8"))

    def test_delivery_binds_every_byte_and_preserves_inputs(self):
        before = {k: bundle.sha(v) for k, v in self.files.items()}
        result = self.build()
        archive = Path(result["archive"])
        self.assertEqual(bundle.sha(archive), result["sha256"])
        self.assertEqual(base64.b64decode(archive.with_suffix(".b64").read_bytes(), validate=True), archive.read_bytes())
        with zipfile.ZipFile(archive) as packed:
            self.assertIsNone(packed.testzip())
            manifest = json.loads(packed.read("bundle.json"))
            self.assertEqual(manifest["kind"], bundle.KIND)
            self.assertEqual(set(packed.namelist()), {"bundle.json", *(r["path"] for r in manifest["files"])})
            for record in manifest["files"]:
                content = packed.read(record["path"])
                self.assertEqual(len(content), record["size_bytes"])
                self.assertEqual(hashlib.sha256(content).hexdigest(), record["sha256"])
            recovery = json.loads(packed.read("recovery_r2.json"))
            self.assertEqual(hashlib.sha256(packed.read("recovery_r2.json")).hexdigest(), result["manifest_sha256"])
            for key, filename in bundle.PATCH_NAMES.items():
                self.assertEqual(recovery[key], dict(path=filename, sha256=bundle.sha(self.files[key])))
            for key, filename in bundle.TEST_NAMES.items():
                self.assertEqual(recovery["test_"+key], dict(path=filename, sha256=bundle.sha(self.files["test_"+key])))
            self.assertEqual(packed.read("r1_recovery.json"), self.files["r1_manifest"].read_bytes())
        self.assertEqual(before, {k: bundle.sha(v) for k, v in self.files.items()})

    def test_all_notebook_cells_compile_and_defaults_are_locked(self):
        result = self.build()
        book = self.book(result)
        self.assertEqual(book["nbformat"], 4)
        self.assertEqual(len({c["id"] for c in book["cells"]}), len(book["cells"]))
        for index, cell in enumerate(book["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"synthetic_cell_{index}", "exec")
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])
        config = {}
        exec("".join(book["cells"][1]["source"]), config)
        self.assertTrue(config["ALLOW_MAIN"])
        self.assertFalse(config["PREPARE_ONLY"])
        self.assertEqual(config["SOURCE_CHECKPOINT_SHA256"], bundle.CHECKPOINT_SHA256)
        self.assertEqual(config["RECOVERY_MANIFEST_SHA256"], result["manifest_sha256"])
        self.assertEqual(base64.b64decode(config["EMBEDDED_PATCH_BASE64"]), Path(result["archive"]).read_bytes())
        self.assertFalse(config["PATCH_ROOT"].is_relative_to(config["ROOT"]))
        restore = "".join(book["cells"][3]["source"])
        self.assertIn("verify_checkpoint_header(value, actual, recovery_manifest, sidecar_hash)", restore)
        self.assertIn("expected_index_sha256=zip_index_sha", restore)
        self.assertIn("initial_'+label+'_ids_sha256", restore)
        self.assertLess(restore.index("matching_indexes"), restore.index("restore_chunk_input"))
        self.assertIn("if len(matching_indexes) > 1:", restore)
        self.assertLess(restore.index("verify_physical_journals(RESUME_INDEX.parents[2]"), restore.index("extract_verified(source_archive"))
        self.assertLess(restore.index("verify_archive_journals(input_archive"), restore.index("zip_input.prepare_zip_input"))

    def test_embedded_patch_extracts_and_existing_mismatch_fails(self):
        result = self.build()
        book = self.book(result)
        context = {}
        exec("".join(book["cells"][1]["source"]), context)
        context.update(BASE=self.base/"work/native_paper_v4", ROOT=self.base/"work/native_paper_v4/repo",
                       PATCH_ROOT=self.base/"work/external_r2")
        source = "".join(book["cells"][2]["source"])
        with patch.object(sys, "platform", "linux"), patch("shutil.disk_usage", return_value=type("Disk", (), {"free": 20*1024**3})()):
            exec(source, context)
            exec(source, context)
            target = context["PATCH_ROOT"]/bundle.PATCH_NAMES["scheduler"]
            target.write_text("unrelated bytes", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Existing source differs"):
                exec(source, context)
            self.assertEqual(target.read_text(), "unrelated bytes")

    def stage(self, book, *, prepare_only=False, allow_main=True, fail_gate=None, omit_receipt=None):
        fake = ModuleType("kaggle_native_paper_v4")
        calls, retained = [], []
        fake.isolated_runtimes = lambda *args, **kwargs: ({"controller": "synthetic-python"}, {})
        def command(name, argv, *args, **kwargs):
            calls.append((name, argv))
            if name == fail_gate:
                raise RuntimeError("synthetic gate failure")
            if name != omit_receipt:
                logs = self.base/"campaign/logs/bootstrap"
                logs.mkdir(parents=True, exist_ok=True)
                number = len(list(logs.glob(name+"_*.json")))
                (logs/(name+f"_{number:04d}.json")).write_bytes(bundle.encoded_json(
                    dict(argv=[str(x) for x in argv], exit_code=0, termination=None, synthetic_fixture=True)))
        fake.command = command
        fake.retain_checkpoint = lambda *args, **kwargs: retained.append(kwargs)
        def write(path, value):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(bundle.encoded_json(value))
        fake.write = write
        env = dict(ALLOW_MAIN=allow_main, PREPARE_ONLY=prepare_only, WORKSPACE=self.base/"campaign",
                   ROOT=self.base/"frozen", PATCH_ROOT=self.base/"external", SESSION_STARTED=1.,
                   recovery_manifest_path=self.base/"external/recovery_r2.json",
                   RECOVERY_MANIFEST_SHA256="b"*64, source_archive=self.base/"source.zip")
        import os
        env["os"] = os
        env["checksum"] = bundle.sha
        with patch.dict(sys.modules, {"kaggle_native_paper_v4": fake}):
            if fail_gate:
                with self.assertRaisesRegex(RuntimeError, "synthetic gate failure"):
                    exec("".join(book["cells"][4]["source"]), env)
            elif omit_receipt:
                with self.assertRaisesRegex(ValueError, "one fresh regression receipt"):
                    exec("".join(book["cells"][4]["source"]), env)
            else:
                exec("".join(book["cells"][4]["source"]), env)
        return calls, retained

    def test_prepare_only_cannot_launch_main_and_keeps_checkpoint(self):
        calls, retained = self.stage(self.book(self.build()), prepare_only=True)
        self.assertEqual([c[0] for c in calls], ["r2_affinity_regression", "r2_preservation_regression", "recovery_r2"])
        self.assertIn("--prepare-only", calls[-1][1])
        self.assertNotIn("--allow-main", calls[-1][1])
        self.assertEqual(len(retained), 1)
        self.assertIsNone(retained[0]["stage_error"])

    def test_main_requires_both_gates_and_failure_retains_checkpoint(self):
        book = self.book(self.build())
        calls, retained = self.stage(book)
        self.assertIn("--allow-main", calls[-1][1])
        self.assertNotIn("--prepare-only", calls[-1][1])
        for gate, count in [("r2_affinity_regression", 1), ("r2_preservation_regression", 2)]:
            calls, retained = self.stage(book, fail_gate=gate)
            self.assertEqual(len(calls), count)
            self.assertNotIn("recovery_r2", [c[0] for c in calls])
            self.assertIsInstance(retained[0]["stage_error"], RuntimeError)

    def test_gate_receipts_are_fresh_and_bound_to_current_manifest(self):
        book = self.book(self.build())
        calls, retained = self.stage(book)
        proof_path = self.base/"campaign/recovery/v4_affinity_recovery_v2/regression_gates.json"
        first = json.loads(proof_path.read_text())
        self.assertEqual(first["manifest_sha256"], "b"*64)
        self.assertEqual(set(first["receipts"]), {"r2_affinity_regression", "r2_preservation_regression"})
        for receipt in first["receipts"].values():
            path = self.base/"campaign"/receipt["path"]
            self.assertEqual(bundle.sha(path), receipt["sha256"])
        self.stage(book)
        second = json.loads(proof_path.read_text())
        self.assertNotEqual(first["receipts"], second["receipts"])
        calls, retained = self.stage(book, omit_receipt="r2_affinity_regression")
        self.assertEqual([name for name, _ in calls], ["r2_affinity_regression"])
        self.assertIsInstance(retained[0]["stage_error"], ValueError)

    def test_wrong_checkpoint_r1_bytes_and_existing_outputs_are_refused(self):
        wrong = copy.deepcopy(self.identity)
        wrong["source_checkpoint_index_sha256"] = "f"*64
        with self.assertRaisesRegex(ValueError, "frozen identity differs"):
            bundle.build(self.base/"bad", resume_manifest=wrong, source_files=self.files)
        result = self.build()
        before = Path(result["archive"]).read_bytes()
        with self.assertRaisesRegex(ValueError, "refusing overwrite"):
            self.build()
        self.assertEqual(Path(result["archive"]).read_bytes(), before)
        self.files["r1_runner"].write_text("# changed R1\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "immutable manifest"):
            self.build("changed_r1")

    def test_fallback_chunks_must_bind_latest_checkpoint(self):
        manifest = self.base/"native_paper_v4_resume_chunks.json"
        manifest.write_bytes(bundle.encoded_json(dict(source_input_sha256=bundle.SOURCE_SHA256,
            checkpoint_index_sha256="f"*64, synthetic_fixture=True)))
        with self.assertRaisesRegex(ValueError, "different checkpoint"):
            self.build(chunk_manifest=manifest)
        manifest.write_bytes(bundle.encoded_json(dict(source_input_sha256=bundle.SOURCE_SHA256,
            checkpoint_index_sha256=bundle.CHECKPOINT_SHA256, synthetic_fixture=True)))
        result = self.build(chunk_manifest=manifest)
        context = {}
        exec("".join(self.book(result)["cells"][1]["source"]), context)
        self.assertEqual(context["CHUNK_MANIFEST_SHA256"], bundle.sha(manifest))

    def test_descendant_requires_exact_base_chain_and_original_index_proof(self):
        manifest = self.identity
        value = dict(source_input_sha256=bundle.SOURCE_SHA256, protocol_id="a3t_old_safe_sem_kaggle_v4",
            schema_version=13, payloads=copy.deepcopy(manifest["base_payloads"]),
            inventory_sha256=manifest["source_checkpoint_inventory_sha256"], inventory={})
        self.assertFalse(bundle.verify_checkpoint_header(value, bundle.CHECKPOINT_SHA256, manifest))
        name = "campaign/recovery/v4_affinity_recovery_v2/baseline_checkpoint_index.json"
        value["payloads"].append(dict(path="synthetic_extension.payload", sha256="c"*64, size_bytes=10))
        value["inventory"][name] = dict(path=name, sha256=bundle.CHECKPOINT_SHA256, size_bytes=10)
        self.assertTrue(bundle.verify_checkpoint_header(value, "d"*64, manifest, "d"*64))
        for mutation in ("source", "payload", "baseline", "sidecar"):
            changed = copy.deepcopy(value)
            sidecar = "d"*64
            if mutation == "source": changed["source_input_sha256"] = "f"*64
            if mutation == "payload": changed["payloads"][0]["sha256"] = "f"*64
            if mutation == "baseline": changed["inventory"][name]["sha256"] = "f"*64
            if mutation == "sidecar": sidecar = "f"*64
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                bundle.verify_checkpoint_header(changed, "d"*64, manifest, sidecar)

    def test_physical_journals_newer_than_checkpoint_fail_before_restore(self):
        base = self.base/"synthetic_input"
        name = "campaign/collection/main-w007/start_journal.jsonl"
        path = base/name
        path.parent.mkdir(parents=True)
        original = b'{"trial_id":"synthetic_old"}\n'
        path.write_bytes(original)
        index = dict(inventory={name: dict(path=name, size_bytes=len(original), sha256=hashlib.sha256(original).hexdigest())})
        self.assertEqual(bundle.verify_physical_journals(base, index), 1)
        path.write_bytes(original+b'{"trial_id":"synthetic_newer_unindexed"}\n')
        with self.assertRaisesRegex(ValueError, "newer physical start journal"):
            bundle.verify_physical_journals(base, index)
        path.write_bytes(original.replace(b"old", b"BAD"))
        with self.assertRaisesRegex(ValueError, "newer physical start journal"):
            bundle.verify_physical_journals(base, index)
        path.write_bytes(original)
        with self.assertRaisesRegex(ValueError, "Unindexed"):
            bundle.verify_physical_journals(base, dict(inventory={}))
        self.assertEqual(bundle.verify_physical_journals(self.base/"slim_input", index), 0)

    def test_raw_output_zip_checks_outer_journals_before_selective_import(self):
        name = "campaign/collection/main-w007/start_journal.jsonl"
        original = b'{"trial_id":"synthetic_old"}\n'
        index = dict(inventory={name: dict(path=name, size_bytes=len(original), sha256=hashlib.sha256(original).hexdigest())})
        for label, payload in (("matching", original), ("newer", original+b'{}\n'), ("tampered", original.replace(b"old", b"BAD"))):
            output = io.BytesIO()
            with zipfile.ZipFile(output, "w") as packed:
                packed.writestr("native_paper_v4/"+name, payload)
            with zipfile.ZipFile(io.BytesIO(output.getvalue())) as packed:
                args = (packed, "native_paper_v4/campaign/exports/checkpoint_index.json", index)
                if label == "matching":
                    self.assertEqual(bundle.verify_archive_journals(*args), 1)
                else:
                    with self.subTest(label=label), self.assertRaisesRegex(ValueError, "start journal"):
                        bundle.verify_archive_journals(*args)


if __name__ == "__main__":
    unittest.main()
