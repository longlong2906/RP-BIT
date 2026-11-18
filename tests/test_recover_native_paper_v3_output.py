"""Labelled archive fixtures for recovery; never planner observations."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import recover_native_paper_v3_output as recovery
import kaggle_a3_bundle as bundle


class RecoveryConverterTests(unittest.TestCase):
    def setUp(self):
        fixtures = ROOT / "tests/fixtures/native_paper_v3_recovery"
        fixtures.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="synthetic_converter_", dir=fixtures)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source.zip"
        source_file = self.root / "synthetic_source.txt"
        source_file.write_bytes(b"SYNTHETIC_RECOVERY_FIXTURE_not_a_result")
        source_manifest = bundle.archive([
            ("tools/kaggle_a3_bundle.py", Path(bundle.__file__)),
            ("configs/fixture.txt", source_file)], self.source, recovery.SOURCE_KIND)
        self.rows = [[item["path"], item["sha256"]] for item in source_manifest["files"]]
        self.worker = recovery.ROOT + "/build-native-paper-v3/terrain_benchmark"
        python = recovery.ROOT + "/.venv-native-paper-v3/bin/python"
        target_python = recovery.ROOT + "/.managed-python/cpython/bin/python3.12"
        ompl = recovery.ROOT + "/.deps/install/lib/libompl.so.1.7.0"
        bridge = recovery.ROOT + "/build-native-paper-v3/libterrain_compare_c.so"
        hashes = {name: hashlib.sha256(content).hexdigest() for name, content in
                  [(self.worker, b"synthetic worker"), (python, b"synthetic python"),
                   (target_python, b"synthetic python"), (ompl, b"synthetic ompl"),
                   (bridge, b"synthetic bridge")]}
        registry = b'{"synthetic_fixture":true}'
        plan = b'{"synthetic_fixture":true}'
        self.prepared = dict(schema_version=12, protocol_id=recovery.PROTOCOL, source_root=recovery.ROOT,
            controller_python=python, sem_python=python,
            registry_sha256=hashlib.sha256(registry).hexdigest(), frozen_inputs={"plans/main.json":hashlib.sha256(plan).hexdigest()},
            identity=dict(source_files=self.rows, source_sha256=bundle.digest(self.rows), binary_hashes=hashes,
                native_linkage=dict(roots={python:dict(requested_root=python, resolved_target=target_python),
                    self.worker:dict(libraries={"libompl.so.18":ompl})})))
        prepared_bytes = json.dumps(self.prepared).encode()
        self.policy = recovery._Policy(source_sha256=bundle.sha256(self.source), source_identity=bundle.digest(self.rows),
            prepared_sha256=hashlib.sha256(prepared_bytes).hexdigest(), source_count=2, native_count=5,
            trial_count=1, synthetic_fixture=True)
        p = recovery.PREFIX
        self.files = {p+"source_input.zip":self.source.read_bytes(),
            p+"campaign/prepared.json":prepared_bytes, p+"campaign/registry.json":registry,
            p+"campaign/plans/main.json":plan,
            p+"campaign/campaigns/main/manifest.json":json.dumps(dict(status="COLLECTED_VALIDATION_PENDING",
                entries=[dict(trial_id="synthetic_a3", backend="a3", execution_status="COMPLETED")])).encode(),
            p+"campaign/exports/incomplete.zip.partial":b"omit failed export",
            p+"campaign/tmp/fixed_work_fixture.json":b"omit temporary correctness copy"}
        with zipfile.ZipFile(self.source) as source_zip:
            for item in source_manifest["files"]:
                self.files[p+"repo/"+item["path"]] = source_zip.read(item["path"])
        for name, content in [(self.worker,b"synthetic worker"), (python,b"synthetic python"),
                (target_python,b"synthetic python"), (ompl,b"synthetic ompl"),
                (bridge,b"synthetic bridge"), (str(PurePosixPath(ompl).with_name("libompl.so.18")),b"synthetic ompl")]:
            # Posix path rules apply even when these tests execute on Windows.
            self.files[p+recovery.archive_path(name)] = content
        trial = p+"campaign/campaigns/main/synthetic_a3/"
        for name in ["worker.json", "request.json", "execution.json", "progress.jsonl", "launch.json", "phase.json"]:
            self.files[trial+name] = b'{"synthetic_fixture":true}\n'
        self.input = self.root / "output.zip"
        self.output = self.root / "recovered.payload"
        self.write_input()

    def write_input(self):
        with zipfile.ZipFile(self.input, "w") as archive:
            for name, data in self.files.items():
                info = zipfile.ZipInfo(name)
                info.create_system = 3
                info.external_attr = (stat.S_IFREG | (0o755 if name.endswith(("/python", "/python3.12", "/terrain_benchmark")) else 0o644)) << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, data)

    def execute(self, **kwargs):
        return recovery.repack(self.input, self.output, expected_input_sha256=bundle.sha256(self.input),
            source_archive=self.source, _policy=self.policy, **kwargs)

    def test_stream_roundtrip_omits_only_non_scientific_copies(self):
        original = bundle.sha256(self.input)
        report = self.execute()
        self.assertEqual(bundle.sha256(self.input), original)
        self.assertEqual(report["trial_executions"], 0)
        manifest = bundle.verify(self.output)
        self.assertEqual(manifest["kind"], recovery.KIND)
        self.assertEqual(manifest["path_case_policy"], bundle.LINUX_CASE_POLICY)
        self.assertFalse(any("/exports/" in r["path"] or "/tmp/" in r["path"] for r in manifest["files"]))
        self.assertEqual(len(manifest["metadata"]["internal_runtime_symlinks"]), 2)
        self.assertTrue(manifest["metadata"]["contains_exact_runtime"])
        self.assertTrue(manifest["metadata"]["recovery_origin"]["synthetic_fixture"])
        with zipfile.ZipFile(self.output) as archive:
            for item in manifest["files"]:
                self.assertEqual(archive.read(item["path"]), self.files[recovery.PREFIX+item["path"]])

    def test_independent_inventory_is_checked(self):
        inventory = self.root / "science.jsonl"
        name = recovery.PREFIX+"campaign/campaigns/main/synthetic_a3/worker.json"
        with zipfile.ZipFile(self.input) as archive:
            info = archive.getinfo(name)
            inventory.write_text(json.dumps(dict(member=name, bytes=info.file_size,
                crc32=f"{info.CRC:08x}", sha256=hashlib.sha256(self.files[name]).hexdigest()))+"\n")
        report = self.execute(scientific_inventory=inventory)
        self.assertEqual(report["independently_audited_entries"], 1)

    def test_rejects_bad_inventory_hash(self):
        inventory = self.root / "science.jsonl"
        name = recovery.PREFIX+"campaign/campaigns/main/synthetic_a3/worker.json"
        with zipfile.ZipFile(self.input) as archive:
            info = archive.getinfo(name)
            inventory.write_text(json.dumps(dict(member=name, bytes=info.file_size,
                crc32=f"{info.CRC:08x}", sha256="0"*64))+"\n")
        with self.assertRaisesRegex(ValueError,"checksum mismatch"):
            self.execute(scientific_inventory=inventory)

    def test_refuses_existing_output_and_wrong_input_hash(self):
        self.output.write_bytes(b"do not overwrite")
        with self.assertRaisesRegex(ValueError,"exists"):self.execute()
        self.assertEqual(self.output.read_bytes(),b"do not overwrite")
        with self.assertRaisesRegex(ValueError,"SHA-256"):
            recovery.repack(self.input,self.root/"other.payload",expected_input_sha256="0"*64,
                source_archive=self.source,_policy=self.policy)

    def test_missing_snapshot_sidecar_fails_before_publish(self):
        del self.files[recovery.PREFIX+"campaign/campaigns/main/synthetic_a3/progress.jsonl"]
        self.write_input()
        with self.assertRaisesRegex(ValueError,"missing completed trial"):
            self.execute()
        self.assertFalse(self.output.exists())

    def test_unsafe_and_duplicate_zip_names_refused(self):
        self.files["../escape.txt"] = b"forbidden"
        self.write_input()
        with self.assertRaisesRegex(ValueError,"unsafe"):
            self.execute()
        self.assertFalse(self.output.exists())

    def test_source_and_runtime_tamper_refused(self):
        name = recovery.PREFIX+"repo/configs/fixture.txt"
        self.files[name] = b"changed frozen source"
        self.write_input()
        with self.assertRaisesRegex(ValueError,"source checksum mismatch"):
            self.execute()
        self.assertFalse(self.output.exists())

    def test_incomplete_campaign_is_never_retried(self):
        name = recovery.PREFIX+"campaign/campaigns/main/manifest.json"
        value = json.loads(self.files[name]); value["entries"][0]["execution_status"] = "PLANNED"
        self.files[name] = json.dumps(value).encode(); self.write_input()
        with self.assertRaisesRegex(ValueError,"already be completed"):
            self.execute()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":unittest.main()
