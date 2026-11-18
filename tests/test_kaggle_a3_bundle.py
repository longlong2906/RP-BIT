"""Synthetic archive fixtures; no planner execution and no uploaded resources."""
import ast
import base64
import copy
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import kaggle_a3_bundle as bundle


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="kaggle_bundle_fixture_", dir=ROOT / "tests/fixtures")
        self.root = Path(self.temp.name)
        self.file = self.root / "input.txt"
        self.file.write_text("synthetic archive test only", encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def make_archive(self):
        path = self.root / "fixture.zip"
        bundle.archive([("folder/input.txt", self.file)], path, "SYNTHETIC_TEST_FIXTURE")
        return path

    def test_roundtrip_inventory_and_no_overwrite(self):
        path = self.make_archive()
        manifest = bundle.verify(path, bundle.sha256(path))
        self.assertEqual(manifest["kind"], "SYNTHETIC_TEST_FIXTURE")
        target = self.root / "out"
        bundle.extract(path, target)
        restored = target / "folder/input.txt"
        self.assertEqual(restored.read_bytes(), self.file.read_bytes())
        with self.assertRaises(ValueError): bundle.extract(path, target)
        bundle.extract(path, target, allow_identical=True)
        restored.write_text("modified", encoding="utf-8")
        with self.assertRaises(ValueError): bundle.extract(path, target, allow_identical=True)
        self.assertEqual(restored.read_text(), "modified")
        with self.assertRaises(ValueError): self.make_archive()

    def test_archive_checksum_and_member_tamper(self):
        path = self.make_archive()
        with self.assertRaises(ValueError): bundle.verify(path, "0"*64)
        original = bundle.verify(path)
        tampered = self.root / "tampered.zip"
        with zipfile.ZipFile(tampered, "w") as z:
            z.writestr("bundle.json", json.dumps(original))
            z.writestr("folder/input.txt", b"X"*self.file.stat().st_size)
        with self.assertRaises(ValueError): bundle.verify(tampered)
        self.assertFalse((self.root / "unsafe").exists())

    def test_reject_traversal_absolute_duplicate_and_symlink_members(self):
        for name in ("../escape", "/absolute", "C:/absolute", "folder\\escape", "a/./b", "a//b", "."):
            with self.subTest(name=name), self.assertRaises(ValueError): bundle.safe_name(name)
        path = self.make_archive()
        duplicate = self.root / "duplicate.zip"
        with zipfile.ZipFile(path) as src, zipfile.ZipFile(duplicate, "w") as dst:
            for info in src.infolist(): dst.writestr(info, src.read(info.filename))
            dst.writestr("FOLDER/INPUT.TXT", "duplicate via case")
        with self.assertRaises(ValueError): bundle.verify(duplicate)
        symlink = self.root / "link.zip"
        with zipfile.ZipFile(symlink, "w") as z:
            info = zipfile.ZipInfo("link")
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            z.writestr(info, "../outside")
        with self.assertRaises(ValueError): bundle.verify(symlink)

    def test_unregistered_members_fail_before_destination_creation(self):
        path = self.make_archive()
        with zipfile.ZipFile(path, "a") as z: z.writestr("extra", "unregistered")
        with self.assertRaises(ValueError): bundle.extract(path, self.root / "no_write")
        self.assertFalse((self.root / "no_write").exists())

    def test_notebook_is_clean_executable_source_with_pinned_input(self):
        book = bundle.notebook("a"*64)
        self.assertEqual(book["nbformat"], 4)
        self.assertEqual(len({c["id"] for c in book["cells"]}), len(book["cells"]))
        source = "\n".join("".join(c["source"]) for c in book["cells"] if c["cell_type"] == "code")
        ast.parse(source)
        self.assertIn("a"*64, source)
        self.assertIn("--allow-a3-pilot", source)
        self.assertIn("NOTEBOOK_STARTED_MONOTONIC", source)
        self.assertNotIn("terrain_benchmark.exe", source)
        self.assertNotIn("file_digest", source)
        self.assertIn("a3_kaggle_upload.zip", "".join(book["cells"][0]["source"]))
        for cell in book["cells"]:
            if cell["cell_type"] == "code":
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])

    def test_upload_autoexpansion_roundtrip_and_bounded_streaming(self):
        # ZIP_STORED member crosses a decoder chunk boundary; all bytes are synthetic.
        large = self.root/"synthetic.gz"
        with large.open("wb") as handle:
            handle.write(bytes(range(256)) * (3 * 1024 * 1024 // 256))
            handle.write(b"synthetic suffix")
        source = self.root/"a3_kaggle_input.zip"
        bundle.archive([("synthetic.gz", large)], source, "SYNTHETIC_TEST_FIXTURE")
        expected_hash, expected_size = bundle.sha256(source), source.stat().st_size
        outer = self.root/"a3_kaggle_upload.zip"
        metadata = bundle.upload_transport(source, outer)
        self.assertEqual(metadata["kind"], "A3_KAGGLE_BASE64_TRANSPORT")
        self.assertEqual(metadata["metadata"]["decoded_sha256"], expected_hash)
        mounted = self.root/"mounted"/"private-a3-dataset"
        # Mimic Kaggle expanding uploaded ZIPs: b64 remains an ordinary file.
        with zipfile.ZipFile(outer) as uploaded:
            self.assertEqual(set(uploaded.namelist()), {"a3_kaggle_input.b64", "bundle.json"})
            uploaded.extractall(mounted)
        restored = bundle.select_input_archive(self.root/"mounted", self.root/"scratch", expected_hash, expected_size)
        self.assertEqual(restored.read_bytes(), source.read_bytes())
        self.assertEqual(bundle.sha256(restored), expected_hash)
        bundle.extract(restored, self.root/"restored-source", expected_hash)
        self.assertEqual((self.root/"restored-source/synthetic.gz").read_bytes(), large.read_bytes())
        self.assertFalse(list(self.root.glob("a3_transport_*")))

    def test_transport_tamper_never_publishes_a_source_archive(self):
        source = self.make_archive()
        payload = self.root/"payload.b64"
        payload.write_bytes(base64.b64encode(source.read_bytes()))
        modified = bytearray(payload.read_bytes())
        modified[8] = ord("A") if modified[8] != ord("A") else ord("B")
        payload.write_bytes(modified)
        target = self.root/"scratch"/"a3_kaggle_input.zip"
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            bundle.decode_transport(payload, target, bundle.sha256(source), source.stat().st_size)
        self.assertFalse(target.exists())
        self.assertTrue(target.with_name(target.name+".partial").exists())
        with self.assertRaisesRegex(ValueError, "destination already exists"):
            bundle.decode_transport(payload, target, bundle.sha256(source), source.stat().st_size)

    def test_transport_strict_bounds_padding_and_base64_syntax(self):
        payload = self.root/"payload.b64"
        payload.write_bytes(base64.b64encode(b"abcdef"))
        with self.assertRaisesRegex(ValueError, "size bound"):
            bundle.decode_transport(payload, self.root/"size-bound.zip", "a"*64, 3)
        self.assertFalse((self.root/"size-bound.zip.partial").exists())
        for index, content in enumerate((b"YWJj\n", b"YWJj$A==", b"YQ==Yg==")):
            payload.write_bytes(content)
            with self.subTest(content=content), self.assertRaises(ValueError):
                bundle.decode_transport(payload, self.root/f"invalid-{index}.zip", "a"*64, 100)

    def test_prefer_b64_and_raw_zip_fallback_ignore_extracted_zip_directories(self):
        source = self.make_archive()
        mounted = self.root/"mounted"
        mounted.mkdir()
        # Uploading a raw ZIP may leave a directory bearing its old filename.
        (mounted/"a3_kaggle_input.zip").mkdir()
        with self.assertRaisesRegex(ValueError, "Add Input"):
            bundle.select_input_archive(mounted, self.root/"scratch", bundle.sha256(source), source.stat().st_size)
        nested = mounted/"opaque"
        nested.mkdir()
        raw = nested/"a3_kaggle_input.zip"
        raw.write_bytes(source.read_bytes())
        self.assertEqual(bundle.select_input_archive(mounted, self.root/"scratch", bundle.sha256(source), source.stat().st_size), raw)
        (mounted/"a3_kaggle_input.b64").write_bytes(base64.b64encode(source.read_bytes()))
        # A stale raw ZIP does not defeat selection of the preferred opaque payload.
        raw.write_bytes(b"wrong raw ZIP")
        restored = bundle.select_input_archive(mounted, self.root/"scratch", bundle.sha256(source), source.stat().st_size)
        self.assertEqual(restored.read_bytes(), source.read_bytes())
        (nested/"a3_kaggle_input.b64").write_bytes(base64.b64encode(source.read_bytes()))
        with self.assertRaisesRegex(ValueError, "multiple base64"):
            bundle.select_input_archive(mounted, self.root/"scratch2", bundle.sha256(source), source.stat().st_size)

    def test_notebook_embedded_decoder_runs_without_importing_bundle_source(self):
        source = self.make_archive()
        expected = bundle.sha256(source)
        book = bundle.notebook(expected, source.stat().st_size)
        code = "\n".join("".join(cell["source"]) for cell in book["cells"] if cell["cell_type"] == "code")
        tree = ast.parse(code, feature_version=(3, 10))
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        namespace = {}
        exec(compile(ast.Module(body=functions, type_ignores=[]), "notebook_decoder", "exec"), namespace)
        mounted = self.root/"mounted"
        mounted.mkdir()
        (mounted/"a3_kaggle_input.b64").write_bytes(base64.b64encode(source.read_bytes()))
        restored = namespace["select_input_archive"](mounted, self.root/"scratch", expected, source.stat().st_size)
        self.assertEqual(bundle.sha256(restored), expected)
        assignment = next(node for node in tree.body if isinstance(node, ast.Assign) and
                          any(isinstance(target, ast.Name) and target.id == "archive" for target in node.targets))
        execution = next(node for node in ast.walk(tree) if isinstance(node, ast.Call) and
                         isinstance(node.func, ast.Attribute) and node.func.attr == "run")
        self.assertLess(assignment.lineno, execution.lineno)
        self.assertNotIn("kaggle_a3_bundle", code)

    def test_package_keeps_transport_and_original_audit_zip(self):
        delivery = self.root/"delivery"
        docs = self.root/"docs"
        docs.mkdir()
        (docs/"KAGGLE_A3_PILOT.md").write_text("synthetic handoff instructions", encoding="utf-8")
        with patch.object(bundle, "source_entries", return_value=[("synthetic.txt", self.file)]):
            files = bundle.package(delivery, self.root)
        names = {record["name"] for record in files}
        self.assertTrue({"a3_kaggle_upload.zip", "a3_kaggle_upload.zip.sha256", "a3_kaggle_input.zip",
                         "a3_pilot_kaggle.ipynb"}.issubset(names))
        self.assertFalse(list(delivery.glob("*.b64")))
        self.assertFalse(any(path.is_dir() for path in delivery.iterdir()))
        book = json.loads((delivery/"a3_pilot_kaggle.ipynb").read_text(encoding="utf-8"))
        code = "".join(book["cells"][2]["source"])
        self.assertIn(bundle.sha256(delivery/"a3_kaggle_input.zip"), code)
        self.assertIn(str((delivery/"a3_kaggle_input.zip").stat().st_size), code)

    def test_notebook_sets_headless_backend_before_child_and_preserves_parent(self):
        # Execute the launch assignments embedded in the generated notebook.
        # A fresh child and its grandchild must inherit Agg even when Jupyter
        # supplies an inline backend unavailable in the isolated Python.
        book = bundle.notebook("a" * 64)
        source = "\n".join("".join(cell["source"]) for cell in book["cells"] if cell["cell_type"] == "code")
        tree = ast.parse(source)
        launch = [node for node in tree.body if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in {"run_environment", "completed"}
            for target in node.targets)]
        self.assertEqual(len(launch), 2)
        evidence = self.root / "child_backend.json"
        child = ("import os,json,subprocess,sys; "
                 "grandchild=subprocess.check_output([sys.executable,'-c',"
                 "'import os; print(os.environ.get(\"MPLBACKEND\"))'],text=True).strip(); "
                 "json.dump({'child':os.environ.get('MPLBACKEND'),'grandchild':grandchild},"
                 "open(sys.argv[1],'w'))")
        namespace = {"os": os, "subprocess": subprocess, "workspace": self.root,
                     "cmd": [sys.executable, "-c", child, str(evidence)]}
        inherited = "module://matplotlib_inline.backend_inline"
        with patch.dict(os.environ, {"MPLBACKEND": inherited}):
            exec(compile(ast.Module(body=launch, type_ignores=[]), "notebook_launch", "exec"), namespace)
            self.assertEqual(os.environ["MPLBACKEND"], inherited)
        self.assertEqual(namespace["completed"].returncode, 0)
        self.assertEqual(json.loads(evidence.read_text()), {"child": "Agg", "grandchild": "Agg"})


if __name__ == "__main__": unittest.main()
